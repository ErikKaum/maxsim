# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch",
#     "ninja",
#     "numpy",
# ]
# ///
"""CUDA kernel benchmark on HF Jobs.

Runs the three padded-API workloads from ``benchmarks/benchmark.py``
(SmallRerank / HeavyRerank / LongDocStress) against both the dev-loaded
CUDA kernel and a naive PyTorch baseline, prints a table.

Triggered via:

    just cuda-bench            # default flavor (a10g-small)
    just cuda-bench l4x1
    just cuda-bench a100-large
"""

from __future__ import annotations

import os
import statistics
import subprocess
import sys
import types
from pathlib import Path


def _ensure(pkg: str) -> None:
    try:
        __import__(pkg)
        return
    except ImportError:
        pass
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet", pkg]
    )


def _setup_bridge(repo: Path):
    """Build the dev extension and bridge `import maxsim` to it."""
    import torch
    from torch.utils.cpp_extension import load

    major, minor = torch.cuda.get_device_capability()
    sm = f"{major}{minor}"
    ext = load(
        name="maxsim_cuda_dev",
        sources=[
            str(repo / "maxsim_cuda" / "maxsim.cu"),
            str(repo / "maxsim_cuda" / "dev_binding.cpp"),
        ],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            f"-gencode=arch=compute_{sm},code=sm_{sm}",
        ],
        extra_cflags=["-O3"],
        verbose=False,
    )
    sys.path.insert(0, str(repo / "torch-ext"))
    fake_ops = types.ModuleType("maxsim._ops")
    fake_ops.ops = types.SimpleNamespace(
        maxsim_forward=ext.maxsim_forward,
        maxsim_padded_forward=ext.maxsim_padded_forward,
    )
    sys.modules["maxsim._ops"] = fake_ops


def _time(fn, n_iter: int, n_warmup: int) -> tuple[float, float]:
    """Returns (mean_ms, std_ms) over n_iter timed runs."""
    import torch

    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()

    times_ms: list[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(n_iter):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))
    return statistics.mean(times_ms), (
        statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0
    )


def main() -> int:
    repo = Path("/kernels/maxsim")
    if not repo.is_dir():
        print(f"ERROR: expected source tree at {repo}", file=sys.stderr)
        return 1
    os.chdir(repo)
    _ensure("ninja")

    import torch

    print(f"torch  = {torch.__version__}  cuda={torch.version.cuda}")
    print(f"gpu    = {torch.cuda.get_device_name()}  "
          f"sm={torch.cuda.get_device_capability()}")

    _setup_bridge(repo)

    # The Benchmark base class lives in the `kernels` PyPI package. The HF
    # Jobs image doesn't have it; pip-install on demand.
    _ensure("kernels")

    # Import workloads
    sys.path.insert(0, str(repo / "benchmarks"))
    import benchmark as bm
    import maxsim

    workloads = [bm.SmallRerank, bm.HeavyRerank, bm.LongDocStress]
    n_iter, n_warmup = 50, 5
    print()
    print(f"{'workload':<16} {'kernel(ms)':>11} {'naive(ms)':>11} {'speedup':>8}  match")
    print("-" * 62)

    for cls in workloads:
        # Instantiate the Benchmark with the bare minimum config
        try:
            wl = cls(kernel=maxsim, device="cuda", seed=cls.seed)
        except TypeError:
            wl = cls()
            wl.kernel = maxsim
            wl.device = "cuda"
            wl.seed = getattr(cls, "seed", 0)
        wl.setup()

        # Correctness check
        wl.benchmark_kernel()
        kernel_out = wl.out
        expected = wl.verify_kernel()
        torch.cuda.synchronize()
        match_ok = torch.allclose(
            kernel_out.float(), expected.float(), rtol=2e-2, atol=2e-2
        )

        kernel_ms, _ = _time(wl.benchmark_kernel, n_iter, n_warmup)
        naive_ms, _ = _time(wl.benchmark_naive, n_iter, n_warmup)
        speedup = naive_ms / kernel_ms if kernel_ms > 0 else float("inf")
        print(
            f"{cls.__name__:<16} {kernel_ms:>11.3f} {naive_ms:>11.3f} "
            f"{speedup:>7.2f}x  {'OK' if match_ok else 'FAIL'}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
