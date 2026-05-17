# Convenience runner for the maxsim Metal kernel.
# Install `just` once: https://github.com/casey/just  (or: brew install just)
# Run `just` (no args) to see this list.

repo_id := "erikkaum/maxsim"

# List available recipes.
default:
    @just --list

# Build all torch variants into ./build/.
build:
    kernel-builder build-and-copy -L

# Run the full pytest suite inside the kernel-builder test shell.
test *args:
    nix develop .#test -c python -m pytest tests/ {{args}}

# Run the smoke-test example against the locally built kernel.
example:
    nix develop .#test -c python example.py

# Benchmark vs naive baseline on the published {{repo_id}} (run `just upload` first).
benchmark *args:
    nix develop .#test -c kernels benchmark {{repo_id}} {{args}}

# Benchmark vs naive baseline using the locally built kernel in ./build.
bench-local *args:
    nix develop .#test -c python benchmarks/run_local.py {{args}}

# Open an interactive test shell (torch + pytest + kernel on PYTHONPATH).
shell:
    kernel-builder testshell

# Open an interactive dev shell (for incremental C++/MSL rebuilds).
devshell:
    kernel-builder devshell

# Print the kernel variants this repo can build.
variants:
    kernel-builder list-variants

# Build and push to {{repo_id}} (uses build.toml's repo-id + version).
upload:
    kernel-builder build-and-upload

# Verify Apple's Metal toolchain is reachable (else see the README).
metal-check:
    @xcrun metal --version

# Remove local build artefacts.
clean:
    rm -rf build result result-* .pytest_cache tests/__pycache__ \
        torch-ext/maxsim/__pycache__ benchmarks/__pycache__

# ---------------------------------------------------------------------------
# CUDA dev loop (HF Jobs).
#
# Sources are synced to the `erikkaum/kernels` HF bucket under a `maxsim/`
# prefix. The job mounts the whole bucket at /kernels and runs the dev
# script (see scripts/cuda_dev.py) inside pytorch/pytorch:*-cuda-devel.
# ---------------------------------------------------------------------------

cuda_bucket := "hf://buckets/erikkaum/kernels"
cuda_prefix := "maxsim"
cuda_image  := "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel"

# Push local sources to the bucket (rsync-style, only changed files).
cuda-sync:
    # NB: patterns without trailing slash so the matcher catches both real
    # dirs and symlinks (the `build` symlink → `result/` would otherwise be
    # followed, shipping Mac Metal binaries to a Linux GPU box).
    uvx hf buckets sync . {{cuda_bucket}}/{{cuda_prefix}} \
        --delete \
        --exclude '.git' --exclude '.git/*' \
        --exclude 'build' --exclude 'build/*' \
        --exclude 'result' --exclude 'result/*' \
        --exclude 'result-*' \
        --exclude '__pycache__' --exclude '*/__pycache__/*' \
        --exclude '.pytest_cache' --exclude '.pytest_cache/*' \
        --exclude '.mypy_cache' --exclude '.mypy_cache/*' \
        --exclude '.ruff_cache' --exclude '.ruff_cache/*' \
        --exclude '.venv' --exclude '.venv-*' --exclude '.venv/*' \
        --exclude '.direnv' --exclude '.direnv/*' \
        --exclude '.claude' --exclude '.claude/*' \
        --exclude '.hypothesis' --exclude '.hypothesis/*' \
        --exclude '*.log'

# Smoke-test the CUDA dev loop on HF Jobs.
# Usage: just cuda-dev               (a10g-small / sm_86)
#        just cuda-dev l4x1          (Lovelace / sm_89)
#        just cuda-dev a100-large    (sm_80)
cuda-dev flavor="a10g-small": cuda-sync
    uvx hf jobs run \
        --flavor {{flavor}} \
        --timeout 30m \
        --secrets HF_TOKEN \
        -v {{cuda_bucket}}:/kernels \
        {{cuda_image}} \
        python /kernels/{{cuda_prefix}}/scripts/cuda_dev.py

# Benchmark vs naive baseline on HF Jobs.
cuda-bench flavor="a10g-small": cuda-sync
    uvx hf jobs run \
        --flavor {{flavor}} \
        --timeout 30m \
        --secrets HF_TOKEN \
        -v {{cuda_bucket}}:/kernels \
        {{cuda_image}} \
        python /kernels/{{cuda_prefix}}/scripts/cuda_bench.py

# Build all CUDA release variants on HF Jobs (via Nix + kernel-builder).
# Writes per-variant build/ trees into the bucket at maxsim-build/ so the
# local side can `just cuda-pull` them back.
cuda-release flavor="cpu-upgrade": cuda-sync
    uvx hf jobs run \
        --flavor {{flavor}} \
        --timeout 3h \
        --secrets HF_TOKEN \
        -v {{cuda_bucket}}:/kernels \
        nixos/nix:latest \
        sh /kernels/{{cuda_prefix}}/scripts/cuda_release.sh

# Pull CUDA release variants from the bucket into the local ./build/ tree.
# Combine with `just build` (which writes Metal variants into ./build/) to
# get a unified tree ready for `kernel-builder upload`.
cuda-pull:
    uvx hf buckets sync {{cuda_bucket}}/maxsim-build ./build

# Local fallback: build CUDA variants via Docker + linux/amd64 emulation.
# Slow (QEMU emulation of x86_64 on M-series) but doesn't depend on HF
# Jobs queue capacity. Output lands directly in ./build-cuda/ for merging.
cuda-release-local:
    mkdir -p ./build-cuda
    docker rm -f maxsim-cuda-build 2>/dev/null || true
    docker run \
        --platform linux/amd64 \
        --name maxsim-cuda-build \
        --security-opt seccomp=unconfined \
        --privileged \
        -v $(pwd):/kernels/maxsim:ro \
        -v $(pwd)/build-cuda:/kernels/maxsim-build \
        nixos/nix:latest \
        sh /kernels/maxsim/scripts/cuda_release.sh

# Inspect the last local cuda-release container's logs and clean up.
cuda-release-local-logs:
    docker logs maxsim-cuda-build 2>&1 | tail -80
    docker rm maxsim-cuda-build 2>/dev/null || true

# Merge ./build-cuda/ (local Docker fallback output) into ./build/.
cuda-merge-local:
    cp -r ./build-cuda/* ./build/
