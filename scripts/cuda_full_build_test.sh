#!/usr/bin/env bash
# Full CUDA release-build test on HF Jobs.
#
# This is intentionally slower than scripts/cuda_dev.py. The dev path builds
# maxsim.cu through torch.utils.cpp_extension for quick iteration; this script
# exercises the same kernel-builder packaging and torch.library registration
# path that users get from a published kernels artifact.
set -euo pipefail

SRC=/kernels/maxsim
WORK=/tmp/maxsim-full-build-test

# HF Jobs truncates `hf jobs logs` to ~3000 lines; the full Nix build prints
# tens of thousands. Tee everything from here on to a file inside the mounted
# bucket so we can `uvx hf buckets cat` the complete log from local. Stamp the
# file with the job id when available so retries don't clobber each other.
LOG_TAG="${HF_JOB_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
LOG_DIR=/kernels/maxsim-build-logs
LOG_FILE="$LOG_DIR/cuda_full_build_test-$LOG_TAG.log"
mkdir -p "$LOG_DIR"
echo "[cuda_full_build_test] full log: $LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[cuda_full_build_test] copying source $SRC -> $WORK"
rm -rf "$WORK"
mkdir -p "$WORK"
cp -r "$SRC"/* "$WORK"/
cd "$WORK"
rm -rf build result result-*

# kernel-builder namespaces the extension by git revision. Bucket-synced
# sources have no .git directory, so create a throwaway repo for the build.
echo "[cuda_full_build_test] init throwaway git repo for kernel-builder hashing"
if ! command -v git >/dev/null 2>&1; then
  nix-env -iA nixpkgs.git
fi
git init -q
git -c user.email=ci@local -c user.name=ci add -A
git -c user.email=ci@local -c user.name=ci commit -q -m "full build test"

# Match scripts/cuda_release.sh: enable flakes, use the HF cache, and turn off
# Nix's seccomp syscall filter for the containerized HF Jobs environment.
# Keep Nix sandboxing enabled by default: current kernel-builder requires it
# on Linux.
NIX_SANDBOX="${NIX_SANDBOX:-true}"
mkdir -p /etc/nix
cat > /etc/nix/nix.conf <<EOF
experimental-features = nix-command flakes
trusted-users = root
substituters = https://cache.nixos.org https://huggingface.cachix.org
trusted-public-keys = cache.nixos.org-1:6NCHdD59X431o0gWypbMrAURkbJ16ZPMQFGspcDShjY= huggingface.cachix.org-1:ynTPbLS0W8ofXd9fDjk1KvoFky9K2jhxe6r4nXAkc/o=
sandbox = $NIX_SANDBOX
filter-syscalls = false
EOF

echo "[cuda_full_build_test] restarting nix-daemon"
pkill -HUP nix-daemon 2>/dev/null || true

echo "[cuda_full_build_test] environment"
nvidia-smi || true

echo "[cuda_full_build_test] kernel-builder build-and-copy"
nix run github:huggingface/kernels#kernel-builder -- build-and-copy -L

echo "[cuda_full_build_test] built variants"
find ./build -maxdepth 2 -type d | sort

echo "[cuda_full_build_test] pytest against packaged build"
cat > /tmp/maxsim-run-packaged-pytest.sh <<'EOS'
#!/usr/bin/env bash
set -eu
set -o pipefail

CUDA_COMPAT_PATHS=""
if CUDA_COMPAT_OUT="$(nix build --no-link --print-out-paths github:NixOS/nixpkgs/nixos-unstable#cudaPackages.cuda_compat 2>/dev/null)"; then
  CUDA_COMPAT_PATHS="$CUDA_COMPAT_OUT"
fi

DRIVER_LIB_PATHS="$(
  find $CUDA_COMPAT_PATHS /usr/local/nvidia/lib /usr/local/nvidia/lib64 /usr/lib /run \
    -type f -name "libcuda.so*" \
    -printf "%h\n" 2>/dev/null | sort -u | paste -sd: -
)"
CUDART_LIB_PATHS="$(
  find /nix/store \
    -type f -name "libcudart.so*" ! -path "*stubs*" \
    -printf "%h\n" 2>/dev/null | sort -u | paste -sd: -
)"
CUDA_LIB_PATHS="${DRIVER_LIB_PATHS}${DRIVER_LIB_PATHS:+:}${CUDART_LIB_PATHS}"
if [ -n "${CUDA_LIB_PATHS#:}" ]; then
  export LD_LIBRARY_PATH="${CUDA_LIB_PATHS#:}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

python - <<'PY'
import os
import torch

print(f"[cuda_full_build_test] torch={torch.__version__} cuda={torch.version.cuda}")
cuda_available = torch.cuda.is_available()
print(f"[cuda_full_build_test] torch.cuda.is_available={cuda_available}")
print(f"[cuda_full_build_test] MAXSIM_KERNEL_VARIANT={os.environ.get('MAXSIM_KERNEL_VARIANT')}")
print(f"[cuda_full_build_test] LD_LIBRARY_PATH entries={len(os.environ.get('LD_LIBRARY_PATH', '').split(':')) if os.environ.get('LD_LIBRARY_PATH') else 0}")
if not cuda_available:
    raise SystemExit("CUDA is unavailable in the packaged-build test shell")
PY

MAXSIM_REQUIRE_BUILT_VARIANT=1 python -m pytest tests/ -x -v
EOS
chmod +x /tmp/maxsim-run-packaged-pytest.sh
nix develop .#test -c bash /tmp/maxsim-run-packaged-pytest.sh
