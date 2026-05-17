"""Benchmarks for the MaxSim kernel against a naïve PyTorch baseline.

Three workloads are taken straight from plan.md:

* small rerank   : B=32,  candidates=10,  q_len=32, d_len=180,  dim=128
* heavy rerank   : B=32,  candidates=100, q_len=32, d_len=256,  dim=128
* long-doc stress: B=8,   candidates=16,  q_len=64, d_len=1024, dim=128

Run with::

    kernels benchmark <repo_id>
"""

from __future__ import annotations

import torch
from kernels.benchmark import Benchmark


def _naive_maxsim_padded(
    queries: torch.Tensor,        # [B, Lq, D]
    documents: torch.Tensor,      # [B, C, Ld, D]
    query_lengths: torch.Tensor,  # [B]
    doc_lengths: torch.Tensor,    # [B, C]
) -> torch.Tensor:
    """Vectorised but naïve MaxSim: materializes the full [Lq, Ld] matrix.

    This is the natural way to write MaxSim in PyTorch; it is what the
    kernel is meant to beat in both speed and peak memory.
    """
    B, C, Ld, D = documents.shape
    Lq = queries.shape[1]

    sim = torch.einsum("bid,bcjd->bcij", queries.float(), documents.float())

    q_mask = (
        torch.arange(Lq, device=queries.device)[None, :] < query_lengths[:, None]
    )  # [B, Lq]
    d_mask = (
        torch.arange(Ld, device=queries.device)[None, None, :]
        < doc_lengths[:, :, None]
    )  # [B, C, Ld]

    sim = sim.masked_fill(~d_mask[:, :, None, :], float("-inf"))
    per_q_max = sim.max(dim=-1).values  # [B, C, Lq]
    per_q_max = per_q_max.masked_fill(~q_mask[:, None, :], 0.0)
    return per_q_max.sum(dim=-1)  # [B, C]


def _make_workload(
    self: Benchmark, B: int, C: int, Lq: int, Ld: int, D: int
) -> None:
    gen = torch.Generator().manual_seed(self.seed if self.seed is not None else 0)
    self.queries = torch.randn(B, Lq, D, generator=gen, dtype=torch.float16).to(
        self.device
    )
    self.documents = torch.randn(
        B, C, Ld, D, generator=gen, dtype=torch.float16
    ).to(self.device)
    self.query_lengths = torch.full(
        (B,), Lq, dtype=torch.int32, device=self.device
    )
    self.doc_lengths = torch.full(
        (B, C), Ld, dtype=torch.int32, device=self.device
    )


def _run_kernel(self: Benchmark) -> torch.Tensor:
    return self.kernel.score_candidates_padded(
        self.queries, self.documents, self.query_lengths, self.doc_lengths
    )


def _run_naive(self: Benchmark) -> torch.Tensor:
    return _naive_maxsim_padded(
        self.queries, self.documents, self.query_lengths, self.doc_lengths
    )


class SmallRerank(Benchmark):
    """B=32, candidates=10, q_len=32, d_len=180, dim=128 (fp16 inputs)."""

    seed = 1234

    def setup(self) -> None:
        _make_workload(self, B=32, C=10, Lq=32, Ld=180, D=128)

    def benchmark_kernel(self) -> None:
        self.out = _run_kernel(self)

    def benchmark_naive(self) -> None:
        self.out = _run_naive(self)

    def verify_kernel(self) -> torch.Tensor:
        return _run_naive(self)

    def verify_naive(self) -> torch.Tensor:
        return _run_naive(self)


class HeavyRerank(Benchmark):
    """B=32, candidates=100, q_len=32, d_len=256, dim=128 (fp16 inputs)."""

    seed = 1234

    def setup(self) -> None:
        _make_workload(self, B=32, C=100, Lq=32, Ld=256, D=128)

    def benchmark_kernel(self) -> None:
        self.out = _run_kernel(self)

    def benchmark_naive(self) -> None:
        self.out = _run_naive(self)

    def verify_kernel(self) -> torch.Tensor:
        return _run_naive(self)

    def verify_naive(self) -> torch.Tensor:
        return _run_naive(self)


class LongDocStress(Benchmark):
    """B=8, candidates=16, q_len=64, d_len=1024, dim=128 (fp16 inputs)."""

    seed = 1234

    def setup(self) -> None:
        _make_workload(self, B=8, C=16, Lq=64, Ld=1024, D=128)

    def benchmark_kernel(self) -> None:
        self.out = _run_kernel(self)

    def benchmark_naive(self) -> None:
        self.out = _run_naive(self)

    def verify_kernel(self) -> torch.Tensor:
        return _run_naive(self)

    def verify_naive(self) -> torch.Tensor:
        return _run_naive(self)
