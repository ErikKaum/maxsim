#!/usr/bin/env bash
# Build all CUDA variants of the maxsim kernel inside an HF Job.
#
# Designed to run in `nixos/nix:latest` with the `erikkaum/kernels` bucket
# mounted at /kernels. Outputs the per-variant build directories under
# /kernels/maxsim-build so the local side can `hf buckets sync` them
# back into ./build before running `kernel-builder upload`.
set -eu

SRC=/kernels/maxsim
DST=/kernels/maxsim-build
WORK=/tmp/maxsim

echo "[cuda_release] copying source $SRC -> $WORK"
mkdir -p "$WORK"
cp -r "$SRC"/* "$WORK"/
cd "$WORK"
rm -rf build result result-*

# kernel-builder refuses to build outside a git repo (uses the commit
# hash to namespace the produced .abi3.so). The bucket-synced source
# tree has no .git, so initialise a throwaway one here. The hash from
# this commit becomes the build's unique tag.
echo "[cuda_release] init throwaway git repo for kernel-builder hashing"
nix-env -iA nixpkgs.git
git init -q
git -c user.email=ci@local -c user.name=ci add -A
git -c user.email=ci@local -c user.name=ci commit -q -m "release build"

# Enable flakes / new nix command (the nixos/nix image keeps them gated by
# default). Then bring in cachix + configure the HuggingFace binary cache
# so we don't recompile pytorch / cuda toolkits from scratch. Also turn
# off the sandbox + seccomp syscall filter -- needed when running this
# script under Docker + x86 emulation on M-series (QEMU can't translate
# the BPF program nix loads). Harmless on a native x86_64 host.
mkdir -p /etc/nix
cat > /etc/nix/nix.conf <<EOF
experimental-features = nix-command flakes
trusted-users = root
substituters = https://cache.nixos.org https://huggingface.cachix.org
trusted-public-keys = cache.nixos.org-1:6NCHdD59X431o0gWypbMrAURkbJ16ZPMQFGspcDShjY= huggingface.cachix.org-1:ynTPbLS0W8ofXd9fDjk1KvoFky9K2jhxe6r4nXAkc/o=
sandbox = false
filter-syscalls = false
EOF

# The nixos/nix image starts nix-daemon at boot with the original conf;
# poke it so our overrides take effect.
echo "[cuda_release] restarting nix-daemon"
pkill -HUP nix-daemon 2>/dev/null || true

echo "[cuda_release] kernel-builder build-and-copy (this is the slow step)"
nix run github:huggingface/kernels#kernel-builder -- build-and-copy -L

echo "[cuda_release] writing artifacts to $DST"
rm -rf "$DST"
mkdir -p "$DST"
cp -r ./build/* "$DST"/

echo "[cuda_release] done. variants:"
find "$DST" -maxdepth 2 -type d | sort
