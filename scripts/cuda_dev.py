# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch",
#     "ninja",
#     "numpy",
#     "pytest",
# ]
# ///
"""CUDA dev driver for the maxsim kernel on HF Jobs.

Triggered locally via ``just cuda-dev [flavor]`` (default ``a10g-small``).
The flow is:

  1. ``hf buckets sync`` mirrors the local source tree into
     ``hf://buckets/erikkaum/kernels/maxsim`` (rsync-style; only changed
     files after the first push).
  2. ``hf jobs run`` spins up a ``pytorch/pytorch:*-cuda-devel`` container
     on the requested flavor with the bucket mounted at ``/kernels``, then
     runs ``python /kernels/maxsim/scripts/cuda_dev.py`` (this file).

Inside the job we:

  a. Confirm the source tree is mounted and torch sees a CUDA device.
  b. Make sure ``ninja`` is installed for fast cpp_extension builds.
  c. Run the full pytest suite. ``tests/conftest.py`` detects the absence
     of a kernel-builder variant dir, builds the CUDA extension via
     ``torch.utils.cpp_extension.load()``, and bridges ``import maxsim``
     to it. Stages 2+ do not need to touch this script.
"""

from __future__ import annotations

import os
import subprocess
import sys


def _ensure(pkg: str) -> None:
    try:
        __import__(pkg)
        return
    except ImportError:
        pass
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet", pkg]
    )


def _print_env() -> None:
    import torch

    print(f"python    = {sys.version.split()[0]}")
    print(f"torch     = {torch.__version__}")
    print(f"cuda      = {torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"gpu       = {torch.cuda.get_device_name()}")
        print(f"sm        = {torch.cuda.get_device_capability()}")


def main() -> int:
    repo = "/kernels/maxsim"
    if not os.path.isdir(repo):
        print(f"ERROR: expected source tree at {repo}", file=sys.stderr)
        return 1
    os.chdir(repo)
    print(f"cwd = {repo}")

    _print_env()
    _ensure("ninja")
    _ensure("pytest")

    import torch

    if not torch.cuda.is_available():
        print("ERROR: torch.cuda.is_available() is False", file=sys.stderr)
        return 1

    # Hand off to pytest. Tests run against the conftest-loaded CUDA
    # extension; no separate build step here.
    return subprocess.call(
        [sys.executable, "-m", "pytest", "tests/", "-x", "-v"],
        cwd=repo,
    )


if __name__ == "__main__":
    sys.exit(main())
