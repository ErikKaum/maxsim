# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "kernels",
#     "torch",
# ]
# ///

"""Retained-memory showcase for contrastive MaxSim training.

The naive PyTorch baseline retains the full ``[Nq, Nb, Lq, Ld]`` fp32
similarity tensor for backward. The kernel saves only ``[Nq, Nb, Lq]``
int32 argmax positions. That retained-activation ratio is exactly ``Ld``.

The measured section is backend-aware:

* CUDA exposes reliable peak-memory counters, so we run a real peak/OOM
  sweep there.
* MPS exposes active/driver allocator snapshots, but not a resettable
  per-op peak counter like CUDA. Inspired by MLX's allocator API
  (``get_active_memory`` / ``get_peak_memory`` / ``reset_peak_memory``),
  we report a conservative forward-retention snapshot instead of pretending
  it is an OOM ceiling.

Run with::

    just example 04
    # or: uv run examples/04_memory_showcase.py
"""

from __future__ import annotations

import gc
import platform
import sys
from dataclasses import dataclass

import kernels
import torch


@dataclass
class Workload:
    queries: torch.Tensor
    documents: torch.Tensor
    document_offsets: torch.Tensor
    doc_lengths: list[int]


def make_workload(
    *,
    nq: int,
    nb: int,
    lq: int,
    ld: int,
    dim: int,
    device: torch.device,
) -> Workload:
    queries = torch.randn(nq, lq, dim, device=device, dtype=torch.float16)
    documents = torch.randn(nb * ld, dim, device=device, dtype=torch.float16)
    document_offsets = torch.arange(
        0, (nb + 1) * ld, ld, dtype=torch.int32, device=device
    )
    return Workload(
        queries=queries,
        documents=documents,
        document_offsets=document_offsets,
        doc_lengths=[ld] * nb,
    )


def kernel_step(maxsim, batch: Workload) -> None:
    q = batch.queries.detach().clone().requires_grad_(True)
    d = batch.documents.detach().clone().requires_grad_(True)
    maxsim.score_contrastive_train(q, d, batch.document_offsets).sum().backward()


def naive_step(batch: Workload) -> None:
    _, _, dim = batch.queries.shape
    nb = len(batch.doc_lengths)
    ld_max = max(batch.doc_lengths)

    q = batch.queries.detach().clone().requires_grad_(True)
    d = batch.documents.detach().clone().requires_grad_(True)
    docs_padded = d.new_zeros((nb, ld_max, dim))
    offs = batch.document_offsets.to(torch.int64).cpu().tolist()
    for i in range(nb):
        docs_padded[i, : batch.doc_lengths[i]] = d[offs[i] : offs[i + 1]]

    sim = torch.einsum("qid,njd->qnij", q.float(), docs_padded.float())
    sim.max(dim=-1).values.sum(dim=-1).sum().backward()


def retained_mb(nq: int, nb: int, lq: int, ld: int) -> tuple[float, float]:
    naive = nq * nb * lq * ld * 4
    kernel = nq * nb * lq * 4
    return naive / 1e6, kernel / 1e6


def print_retained_table(lq: int, ld: int) -> None:
    print("=" * 72)
    print(f"Exact retained backward activations (Lq={lq}, Ld={ld})")
    print("=" * 72)
    print(
        f"{'Nq=Nb':>8} | {'naive sim':>13} {'kernel argmax':>15} "
        f"{'ratio':>9}"
    )
    print("-" * 72)
    for n in (8, 16, 32, 64, 128, 256, 512):
        naive, kernel = retained_mb(n, n, lq, ld)
        print(f"{n:>8} | {naive:>10.1f} MB {kernel:>12.2f} MB {naive/kernel:>8.0f}x")
    print()


def cuda_peak_oom_sweep(maxsim, lq: int, ld: int, dim: int) -> None:
    device = torch.device("cuda")
    print("=" * 72)
    print("CUDA measured peak memory and OOM ceiling")
    print("=" * 72)
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"{'Nq=Nb':>8} | {'kernel peak':>13} {'kernel':>8} | {'naive peak':>13} {'naive':>8}")
    print("-" * 72)

    last_kernel_ok = 0
    last_naive_ok = 0
    for n in (8, 16, 32, 64, 128, 256, 384, 512):
        batch = make_workload(nq=n, nb=n, lq=lq, ld=ld, dim=dim, device=device)

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            kernel_step(maxsim, batch)
            torch.cuda.synchronize()
            kernel_peak = torch.cuda.max_memory_allocated() / 1e6
            kernel_ok = True
            last_kernel_ok = n
        except torch.cuda.OutOfMemoryError:
            kernel_peak = float("nan")
            kernel_ok = False

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            naive_step(batch)
            torch.cuda.synchronize()
            naive_peak = torch.cuda.max_memory_allocated() / 1e6
            naive_ok = True
            last_naive_ok = n
        except torch.cuda.OutOfMemoryError:
            naive_peak = float("nan")
            naive_ok = False

        k_peak = f"{kernel_peak:>10.0f} MB" if kernel_ok else f"{'OOM':>13}"
        n_peak = f"{naive_peak:>10.0f} MB" if naive_ok else f"{'OOM':>13}"
        print(
            f"{n:>8} | {k_peak} {'ok' if kernel_ok else 'OOM':>8} | "
            f"{n_peak} {'ok' if naive_ok else 'OOM':>8}"
        )

        if not kernel_ok and not naive_ok:
            break

    print()
    if last_kernel_ok and last_naive_ok:
        print("Largest batch tolerated before OOM:")
        print(f"  kernel: Nq=Nb={last_kernel_ok}")
        print(f"  naive : Nq=Nb={last_naive_ok}")
        print(f"  usable-batch ratio: {last_kernel_ok / last_naive_ok:.1f}x")
    print()


def mps_mem() -> tuple[float, float]:
    return (
        torch.mps.current_allocated_memory() / 1e6,
        torch.mps.driver_allocated_memory() / 1e6,
    )


def mps_forward_retention_snapshot(maxsim, lq: int, ld: int, dim: int) -> None:
    device = torch.device("mps")
    n = 32
    print("=" * 72)
    print("MPS forward-retention snapshot")
    print("=" * 72)
    print(
        "PyTorch MPS exposes allocator snapshots, not a CUDA-style resettable "
        "peak counter. The deltas below are measured after forward, while the "
        "autograd graph is still alive."
    )
    print(
        f"recommended working set: {torch.mps.recommended_max_memory() / 1e9:.1f} GB"
    )
    print()
    print(f"{'path':>8} | {'active delta':>13} {'driver delta':>13}")
    print("-" * 42)

    for name, forward in (
        ("kernel", kernel_forward_only),
        ("naive", naive_forward_only),
    ):
        gc.collect()
        torch.mps.empty_cache()
        torch.mps.synchronize()
        batch = make_workload(nq=n, nb=n, lq=lq, ld=ld, dim=dim, device=device)
        active0, driver0 = mps_mem()

        keepalive = forward(maxsim, batch) if name == "kernel" else forward(batch)
        torch.mps.synchronize()
        active1, driver1 = mps_mem()

        print(
            f"{name:>8} | {active1 - active0:>10.1f} MB "
            f"{driver1 - driver0:>10.1f} MB"
        )
        del keepalive, batch

    print()
    print(
        "Interpret this as a coarse allocator snapshot, not a peak-memory "
        "benchmark. The exact retained-activation table above is the stable "
        "cross-backend memory claim."
    )
    print()


def kernel_forward_only(maxsim, batch: Workload) -> tuple[torch.Tensor, ...]:
    q = batch.queries.detach().clone().requires_grad_(True)
    d = batch.documents.detach().clone().requires_grad_(True)
    scores = maxsim.score_contrastive_train(q, d, batch.document_offsets)
    return q, d, scores


def naive_forward_only(batch: Workload) -> tuple[torch.Tensor, ...]:
    _, _, dim = batch.queries.shape
    nb = len(batch.doc_lengths)
    ld_max = max(batch.doc_lengths)
    q = batch.queries.detach().clone().requires_grad_(True)
    d = batch.documents.detach().clone().requires_grad_(True)
    docs_padded = d.new_zeros((nb, ld_max, dim))
    offs = batch.document_offsets.to(torch.int64).cpu().tolist()
    for i in range(nb):
        docs_padded[i, : batch.doc_lengths[i]] = d[offs[i] : offs[i + 1]]
    sim = torch.einsum("qid,njd->qnij", q.float(), docs_padded.float())
    scores = sim.max(dim=-1).values.sum(dim=-1)
    return q, d, sim, scores


def main() -> None:
    device = detect_device()
    print(f"device = {device}\n")

    maxsim = kernels.get_kernel("erikkaum/maxsim", version=2, trust_remote_code=True)

    lq, ld, dim = 32, 512, 128
    print_retained_table(lq, ld)

    if device.type == "cuda":
        cuda_peak_oom_sweep(maxsim, lq, ld, dim)
    elif device.type == "mps":
        mps_forward_retention_snapshot(maxsim, lq, ld, dim)


def detect_device() -> torch.device:
    if platform.system() == "Darwin" and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    print("ERROR: maxsim needs MPS or CUDA.", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
