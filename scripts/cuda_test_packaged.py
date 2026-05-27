# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch",
#     "numpy",
#     "pytest",
# ]
# ///
"""Run pytest against CUDA artifacts built by kernel-builder.

This is the execution half of the full-build CUDA test flow. Build artifacts
first with ``just cuda-release`` or another kernel-builder full build that
writes per-variant directories to ``/kernels/maxsim-build``. Then run this
script inside a CUDA PyTorch image so the tests exercise packaged variants
instead of the fast torch.utils.cpp_extension dev bridge.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def _ensure(pkg: str) -> None:
    try:
        __import__(pkg)
        return
    except ImportError:
        pass
    cmd = [sys.executable, "-m", "pip", "install", "--quiet", pkg]
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError:
        subprocess.check_call([*cmd, "--break-system-packages"])


def _copy_source(src: Path, dst: Path) -> None:
    ignore = shutil.ignore_patterns(
        ".git",
        "build",
        "result",
        "result-*",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".venv",
        ".venv-*",
        ".direnv",
    )
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=ignore)


def _print_variants(build_dir: Path) -> None:
    variants = sorted(p.name for p in build_dir.iterdir() if p.is_dir())
    print("[cuda_packaged_test] variants:")
    for variant in variants:
        print(f"  {variant}")


def main() -> int:
    source = Path(os.environ.get("MAXSIM_SOURCE_DIR", "/kernels/maxsim"))
    artifacts = Path(os.environ.get("MAXSIM_BUILD_DIR", "/kernels/maxsim-build"))
    work = Path(os.environ.get("MAXSIM_TEST_WORKDIR", "/tmp/maxsim-packaged-test"))

    if not source.is_dir():
        print(f"ERROR: expected source tree at {source}", file=sys.stderr)
        return 1
    if not artifacts.is_dir():
        print(
            f"ERROR: expected kernel-builder artifacts at {artifacts}",
            file=sys.stderr,
        )
        return 1

    _copy_source(source, work)
    shutil.copytree(artifacts, work / "build")
    os.chdir(work)

    _ensure("pytest")
    _ensure("numpy")

    import torch

    print(f"[cuda_packaged_test] python={sys.version.split()[0]}")
    print(f"[cuda_packaged_test] torch={torch.__version__} cuda={torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"[cuda_packaged_test] gpu={torch.cuda.get_device_name()}")
        print(f"[cuda_packaged_test] sm={torch.cuda.get_device_capability()}")
    else:
        print("ERROR: torch.cuda.is_available() is False", file=sys.stderr)
        return 1

    _print_variants(work / "build")

    env = dict(os.environ)
    env["MAXSIM_REQUIRE_BUILT_VARIANT"] = "1"
    return subprocess.call(
        [sys.executable, "-m", "pytest", "tests/", "-x", "-v"],
        cwd=work,
        env=env,
    )


if __name__ == "__main__":
    sys.exit(main())
