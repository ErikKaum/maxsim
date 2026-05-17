"""pytest configuration.

Tests in this repo do ``import maxsim``, which resolves to the *built* kernel
inside ``./result/<variant>/`` (a symlink ``./build`` also works) so they can
run without `pip install`-ing the wheel.

This conftest finds the most appropriate variant directory and prepends it to
``sys.path`` before test collection so the tests just work after ``nix build``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch


def _detect_torch_variant_tag() -> str:
    # Tags look like "torch212-metal-aarch64-darwin".
    major_minor = ".".join(torch.__version__.split(".")[:2])  # e.g. "2.12"
    torch_token = "torch" + major_minor.replace(".", "")
    return torch_token


def _find_variant_dir() -> Path | None:
    repo_root = Path(__file__).resolve().parent.parent
    torch_tag = _detect_torch_variant_tag()

    # Probe (1) `build/` (the user's convention for kernel-builder copies) and
    # (2) `result/` (nix build symlink), in that order.
    for build_dir_name in ("build", "result"):
        build_dir = repo_root / build_dir_name
        if not build_dir.exists():
            continue
        # Prefer an exact torch match, otherwise any metal/* variant.
        candidates = sorted(build_dir.glob(f"{torch_tag}-*"))
        if not candidates:
            candidates = sorted(build_dir.glob("torch*-metal-*"))
        if candidates:
            return candidates[-1]
    return None


_variant_dir = _find_variant_dir()
if _variant_dir is not None:
    sys.path.insert(0, str(_variant_dir))
    os.environ.setdefault("MAXSIM_KERNEL_VARIANT", str(_variant_dir))
elif torch.cuda.is_available():
    # CUDA dev mode (HF Jobs): no nix variant in build/result, but we have
    # a GPU. Build the extension via torch.utils.cpp_extension and bridge
    # the existing `import maxsim` wrapper to it. Cached after the first
    # build, so subsequent test runs are fast.
    import types
    from torch.utils.cpp_extension import load

    _repo_root = Path(__file__).resolve().parent.parent
    _major, _minor = torch.cuda.get_device_capability()
    _sm = f"{_major}{_minor}"
    _ext = load(
        name="maxsim_cuda_dev",
        sources=[
            str(_repo_root / "maxsim_cuda" / "maxsim.cu"),
            str(_repo_root / "maxsim_cuda" / "dev_binding.cpp"),
        ],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            f"-gencode=arch=compute_{_sm},code=sm_{_sm}",
        ],
        extra_cflags=["-O3"],
        verbose=False,
    )

    # `from maxsim import score_pairs_packed` resolves via:
    #   1. `torch-ext/maxsim/__init__.py` (added to sys.path below) which does
    #   2. `from ._ops import ops` — we synthesize that module here.
    sys.path.insert(0, str(_repo_root / "torch-ext"))
    _fake_ops = types.ModuleType("maxsim._ops")
    _fake_ops.ops = types.SimpleNamespace(
        maxsim_forward=_ext.maxsim_forward,
        maxsim_padded_forward=_ext.maxsim_padded_forward,
    )
    sys.modules["maxsim._ops"] = _fake_ops
    os.environ.setdefault("MAXSIM_KERNEL_VARIANT", f"cuda-dev-sm_{_sm}")
