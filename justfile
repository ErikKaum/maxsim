# Convenience runner for maxsim development, tests, releases, and benchmarks.
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
    nix develop .#test -c python -m pytest {{args}}

# Build local release variants, then test the packaged kernel path.
test-full *args: build
    nix develop .#test -c env MAXSIM_REQUIRE_BUILT_VARIANT=1 python -m pytest {{args}}

# Run an example from ./examples/ by its numeric prefix.
# E.g. `just example 01`, `just example 03`. List with `ls examples/`.
example NUM:
    nix develop .#test -c python examples/{{NUM}}_*.py

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
# CUDA workflows (HF Jobs).
#
# Sources are synced to the `erikkaum/kernels` HF bucket under a `maxsim/`
# prefix. The job mounts the whole bucket at /kernels and runs the dev
# script (see scripts/cuda_dev.py) inside pytorch/pytorch:*-cuda-devel.
# ---------------------------------------------------------------------------

cuda_bucket := "hf://buckets/erikkaum/kernels"
cuda_prefix := "maxsim"
cuda_image  := "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel"
cuda_full_test_image := "pytorch/pytorch:2.12.0-cuda12.6-cudnn9-devel"

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

# Build the CUDA dev bridge in the job, then run pytest in one shot.
# This uses tests/conftest.py's torch.utils.cpp_extension path, not packaged
# kernel-builder variants.
# Usage: just cuda-dev               (a10g-small / sm_86)
#        just cuda-dev l4x1          (Lovelace / sm_89)
#        just cuda-dev a100-large    (sm_80)
cuda-test-dev flavor="a10g-small": cuda-sync
    uvx hf jobs run \
        --flavor {{flavor}} \
        --timeout 30m \
        --secrets HF_TOKEN \
        -v {{cuda_bucket}}:/kernels \
        {{cuda_image}} \
        python /kernels/{{cuda_prefix}}/scripts/cuda_test_dev.py

# Test previously full-built CUDA artifacts from /kernels/maxsim-build in a
# PyTorch CUDA image. Run after `just cuda-release`.
cuda-test-packaged flavor="a100-large" image=cuda_full_test_image: cuda-sync
    uvx hf jobs run \
        --flavor {{flavor}} \
        --timeout 45m \
        --secrets HF_TOKEN \
        -v {{cuda_bucket}}:/kernels \
        {{image}} \
        python /kernels/{{cuda_prefix}}/scripts/cuda_test_packaged.py

# ---------------------------------------------------------------------------
# Benchmarks.
#
# Two systems:
#   * benchmarks/benchmark.py -- the `kernels benchmark` CLI format (per-kernel
#     timing table); `just benchmark` (Hub) and `just bench-local` (./build).
#   * scripts/cuda_bench_matrix.py -- the README-number generator; writes JSON
#     under bench_results/v2/, and `just bench-render` rewrites the README.
# ---------------------------------------------------------------------------

# Run the kernels-CLI benchmark suite against the published Hub kernel.
benchmark *args:
    nix develop .#test -c kernels benchmark {{repo_id}} {{args}}

# Run benchmarks/benchmark.py against the locally built ./build kernel.
bench-local *args:
    nix develop .#test -c python benchmarks/run_local.py {{args}}

# Render README.md benchmark tables from bench_results/v2/*.json.
bench-render:
    python3 scripts/render_readme.py

# Run the benchmark matrix on a CUDA GPU through HF Jobs.
cuda-bench-matrix flavor="a100-large" repeats="1" iters="30" warmup="5": cuda-sync
    #!/usr/bin/env bash
    set -euo pipefail
    COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
    if [ -n "$(git status --porcelain 2>/dev/null)" ]; then
      DIRTY=true
    else
      DIRTY=false
    fi
    uvx hf jobs run \
        --flavor {{flavor}} \
        --timeout 60m \
        --secrets HF_TOKEN \
        -e MAXSIM_BENCH_REPEATS={{repeats}} \
        -e MAXSIM_BENCH_ITERS={{iters}} \
        -e MAXSIM_BENCH_WARMUP={{warmup}} \
        -e MAXSIM_BENCH_COMMIT="$COMMIT" \
        -e MAXSIM_BENCH_GIT_DIRTY="$DIRTY" \
        -v {{cuda_bucket}}:/kernels \
        {{cuda_image}} \
        python /kernels/{{cuda_prefix}}/scripts/bench_matrix.py

# Run the benchmark matrix on Apple Silicon through the Metal/MPS build.
bench-matrix-metal repeats="1" iters="30" warmup="5":
    #!/usr/bin/env bash
    set -euo pipefail
    COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
    if [ -n "$(git status --porcelain 2>/dev/null)" ]; then
      DIRTY=true
    else
      DIRTY=false
    fi
    nix develop .#test -c env \
        MAXSIM_BENCH_METAL=true \
        MAXSIM_BENCH_REPO=. \
        MAXSIM_BENCH_REPEATS={{repeats}} \
        MAXSIM_BENCH_ITERS={{iters}} \
        MAXSIM_BENCH_WARMUP={{warmup}} \
        MAXSIM_BENCH_COMMIT="$COMMIT" \
        MAXSIM_BENCH_GIT_DIRTY="$DIRTY" \
        python scripts/bench_matrix.py

# ---------------------------------------------------------------------------
# CUDA release artifacts.
# ---------------------------------------------------------------------------

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

# Local Docker build for CUDA variants via linux/amd64 emulation.
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
        -e NIX_SANDBOX=false \
        -v $(pwd):/kernels/maxsim:ro \
        -v $(pwd)/build-cuda:/kernels/maxsim-build \
        nixos/nix:latest \
        sh /kernels/maxsim/scripts/cuda_release.sh

# Inspect the last local cuda-release container's logs and clean up.
cuda-release-local-logs:
    docker logs maxsim-cuda-build 2>&1 | tail -80
    docker rm maxsim-cuda-build 2>/dev/null || true

# Merge ./build-cuda/ (local Docker output) into ./build/.
cuda-merge-local:
    cp -r ./build-cuda/* ./build/
