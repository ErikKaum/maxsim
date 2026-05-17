"""Run the benchmarks in ``benchmark.py`` against the locally built kernel.

This is a thin wrapper around the helpers in ``kernels.cli.benchmark`` that
sidesteps the Hub download path (``kernels benchmark <repo_id>`` requires the
kernel to be published on the Hub and currently hard-codes the kernel name
when loading from a local path).

Usage::

    just bench-local                    # default iterations / warmup
    just bench-local --iterations 50 --warmup 5
    just bench-local --filter LongDoc   # only matching class names

The Benchmark subclasses themselves live in ``benchmark.py`` and are shared
with the published-kernel runner (``kernels benchmark <repo_id>``).
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import torch

from kernels import get_local_kernel
from kernels.benchmark import Benchmark
from kernels.cli.benchmark import (
    TimingResults,
    _calculate_iqr_and_outliers,
    _print_results_table,
    _synchronize,
    collect_machine_info,
    discover_benchmark_classes,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = REPO_ROOT / "build"
BENCHMARK_SCRIPT = REPO_ROOT / "benchmarks" / "benchmark.py"


def _device_for_backend(backend_name: str) -> str:
    return {
        "rocm": "cuda",
        "metal": "mps",
        "cann": "npu",
    }.get(backend_name, backend_name)


def _run_class(
    cls: type[Benchmark],
    kernel,
    device: str,
    iterations: int,
    warmup: int,
) -> dict[str, TimingResults]:
    results: dict[str, TimingResults] = {}
    methods = [
        name
        for name in dir(cls)
        if name.startswith("benchmark_") and callable(getattr(cls, name))
    ]
    if not methods:
        raise RuntimeError(f"No benchmark_* methods found on {cls.__name__}")

    print(f"  Running {cls.__name__} on {device}", file=sys.stderr)

    for method_name in methods:
        workload = method_name.removeprefix("benchmark_")

        instance = cls()
        instance.kernel = kernel
        instance.device = device

        if instance.seed is not None:
            torch.manual_seed(instance.seed)
            random.seed(instance.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(instance.seed)

        setup_fn = getattr(instance, f"setup_{workload}", None) or instance.setup
        setup_fn()

        bench_fn = getattr(instance, method_name)
        verify_fn = getattr(instance, f"verify_{workload}", None)

        verified: bool | None = None
        ref_mean_ms: float | None = None
        if verify_fn is not None:
            bench_fn()
            _synchronize()

            for _ in range(warmup):
                verify_fn()
                _synchronize()

            start = time.perf_counter()
            verify_result = verify_fn()
            _synchronize()
            ref_mean_ms = round((time.perf_counter() - start) * 1000, 4)

            verified = torch.allclose(instance.out, verify_result, atol=1e-2)
            if not verified:
                raise RuntimeError(f"Verification failed for {cls.__name__}.{workload}")

        for _ in range(warmup):
            bench_fn()
            _synchronize()

        times_ms: list[float] = []
        for _ in range(iterations):
            start = time.perf_counter()
            bench_fn()
            _synchronize()
            times_ms.append((time.perf_counter() - start) * 1000)

        mean_ms = sum(times_ms) / len(times_ms)
        variance = sum((t - mean_ms) ** 2 for t in times_ms) / len(times_ms)
        std_ms = variance**0.5
        q1, q3, iqr, outliers = _calculate_iqr_and_outliers(times_ms)

        results[workload] = TimingResults(
            mean_ms=round(mean_ms, 4),
            std_ms=round(std_ms, 4),
            min_ms=round(min(times_ms), 4),
            max_ms=round(max(times_ms), 4),
            iterations=iterations,
            q1_ms=round(q1, 4),
            q3_ms=round(q3, 4),
            iqr_ms=round(iqr, 4),
            outliers=outliers,
            verified=verified,
            ref_mean_ms=ref_mean_ms,
        )

    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument(
        "--filter",
        default=None,
        help="Substring; only Benchmark subclasses whose class name contains it run.",
    )
    parser.add_argument(
        "--build-dir",
        default=str(BUILD_DIR),
        help="Path passed to kernels.get_local_kernel (default: ./build).",
    )
    args = parser.parse_args(argv)

    build_dir = Path(args.build_dir)
    if not build_dir.exists():
        print(
            f"Error: build dir '{build_dir}' does not exist. Run `just build` first.",
            file=sys.stderr,
        )
        return 1

    kernel = get_local_kernel(build_dir)

    from kernels.utils import _backend

    backend_name = _backend().name
    device = _device_for_backend(backend_name)

    classes = discover_benchmark_classes(BENCHMARK_SCRIPT, REPO_ROOT)
    if args.filter:
        classes = [c for c in classes if args.filter in c.__name__]
    if not classes:
        print(
            "No Benchmark subclasses to run"
            + (f" (filter='{args.filter}')" if args.filter else ""),
            file=sys.stderr,
        )
        return 1

    machine_info = collect_machine_info()
    cores = f" ({machine_info.gpu_cores} cores)" if machine_info.gpu_cores else ""
    print(file=sys.stderr)
    print(f"  GPU      {machine_info.gpu}{cores}", file=sys.stderr)
    print(f"  CPU      {machine_info.cpu}", file=sys.stderr)
    print(f"  OS       {machine_info.os}", file=sys.stderr)
    print(f"  PyTorch  {machine_info.pytorch_version}", file=sys.stderr)
    print(f"  Kernel   {build_dir}", file=sys.stderr)
    print(file=sys.stderr)

    all_results: dict[str, TimingResults] = {}
    for cls in classes:
        cls_results = _run_class(cls, kernel, device, args.iterations, args.warmup)
        for name, timing in cls_results.items():
            all_results[f"{cls.__name__}.{name}"] = timing

    _print_results_table(all_results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
