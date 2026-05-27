"""Benchmarks for the MaxSim kernel against naive PyTorch baselines.

This is the file the ``kernels`` CLI discovers and runs::

    kernels benchmark erikkaum/maxsim        # against the published kernel
    just bench-local                         # against the local ./build

The workloads mirror ``scripts/cuda_bench_matrix.py`` (the README-number
generator) so the CLI tables and the README matrix describe the same shapes:

* Contrastive training -- the headline ColBERT fine-tuning step (forward +
  backward), at the in-batch shape an actual PyLate batch hits.
* Padded inference     -- the second-stage exact rerank (forward only).
* Packed inference     -- the same rerank shape through the ragged pair API,
  to surface the packing/layout overhead.

Naive baselines are identical to the matrix's: a ``torch.einsum`` that
materialises the full similarity tensor before ``max`` -- what the kernel is
meant to beat. fp16 inputs throughout; the matrix covers bf16 as well.
"""

from __future__ import annotations

import torch
from kernels.benchmark import Benchmark

_DTYPE = torch.float16
_DIM = 128
_SEED = 1234


# ---------------------------------------------------------------------------
# Naive PyTorch baselines (identical to scripts/cuda_bench_matrix.py).
# ---------------------------------------------------------------------------

def _naive_contrastive(q, docs, cu, d_lens):
    """All-pairs MaxSim over packed docs: materialises [Nq, Nb, Lq, Ld]."""
    _Nq, _Lq, dim = q.shape
    Nb = len(d_lens)
    Ld_max = max(d_lens)
    docs_padded = docs.new_zeros((Nb, Ld_max, dim))
    offs = cu.to(torch.int64).cpu().tolist()
    for i, ld_i in enumerate(d_lens):
        docs_padded[i, :ld_i] = docs[offs[i] : offs[i + 1]]
    sim = torch.einsum("qid,njd->qnij", q.float(), docs_padded.float())
    return sim.max(dim=-1).values.sum(dim=-1)


def _naive_padded(q, d, qlen, dlen):
    """Per-query padded MaxSim: materialises [B, C, Lq, Ld]."""
    _B, _C, Ld, _dim = d.shape
    Lq = q.shape[1]
    sim = torch.einsum("bid,bcjd->bcij", q.float(), d.float())
    q_mask = torch.arange(Lq, device=q.device)[None, :] < qlen[:, None]
    d_mask = torch.arange(Ld, device=q.device)[None, None, :] < dlen[:, :, None]
    sim = sim.masked_fill(~d_mask[:, :, None, :], float("-inf"))
    per_q_max = sim.max(dim=-1).values
    per_q_max = per_q_max.masked_fill(~q_mask[:, None, :], 0.0)
    return per_q_max.sum(dim=-1)


# ---------------------------------------------------------------------------
# Workload construction (shared, keyed off each Benchmark's shape attrs).
# ---------------------------------------------------------------------------

def _make_contrastive(self: Benchmark, Nq, Nb, Lq, Ld) -> None:
    gen = torch.Generator().manual_seed(self.seed)
    self.queries = torch.randn(Nq, Lq, _DIM, generator=gen, dtype=_DTYPE).to(self.device)
    self.documents = torch.randn(Nb * Ld, _DIM, generator=gen, dtype=_DTYPE).to(self.device)
    self.document_offsets = torch.arange(
        0, (Nb + 1) * Ld, Ld, dtype=torch.int32, device=self.device
    )
    self.d_lens = [Ld] * Nb


def _make_padded(self: Benchmark, B, C, Lq, Ld) -> None:
    gen = torch.Generator().manual_seed(self.seed)
    self.queries = torch.randn(B, Lq, _DIM, generator=gen, dtype=_DTYPE).to(self.device)
    self.documents = torch.randn(B, C, Ld, _DIM, generator=gen, dtype=_DTYPE).to(self.device)
    self.query_lengths = torch.full((B,), Lq, dtype=torch.int32, device=self.device)
    self.doc_lengths = torch.full((B, C), Ld, dtype=torch.int32, device=self.device)


def _make_packed(self: Benchmark, B, C, Lq, Ld) -> None:
    """Padded tensors plus a flattened CSR pair grid expressing the same work."""
    _make_padded(self, B, C, Lq, Ld)
    q, d = self.queries, self.documents
    self.max_q_len = Lq
    self.batch, self.candidates = B, C
    self.q_flat = q.reshape(B * Lq, _DIM).contiguous()
    self.d_flat = d.reshape(B * C * Ld, _DIM).contiguous()
    self.q_offsets = torch.arange(0, (B + 1) * Lq, Lq, dtype=torch.int32, device=q.device)
    self.d_offsets = torch.arange(0, (B * C + 1) * Ld, Ld, dtype=torch.int32, device=q.device)
    pair_ids = torch.arange(B * C, dtype=torch.int32, device=q.device)
    self.pair_query_ids = pair_ids // C
    self.pair_document_ids = pair_ids


# ---------------------------------------------------------------------------
# Kernel / naive runners.
# ---------------------------------------------------------------------------

def _contrastive_train_kernel(self: Benchmark) -> torch.Tensor:
    q = self.queries.detach().clone().requires_grad_(True)
    d = self.documents.detach().clone().requires_grad_(True)
    scores = self.kernel.score_contrastive_train(q, d, self.document_offsets)
    scores.sum().backward()
    return scores.detach()


def _contrastive_train_naive(self: Benchmark) -> torch.Tensor:
    q = self.queries.detach().clone().requires_grad_(True)
    d = self.documents.detach().clone().requires_grad_(True)
    scores = _naive_contrastive(q, d, self.document_offsets, self.d_lens)
    scores.sum().backward()
    return scores.detach()


def _contrastive_ref(self: Benchmark) -> torch.Tensor:
    return _naive_contrastive(
        self.queries, self.documents, self.document_offsets, self.d_lens
    )


def _padded_kernel(self: Benchmark) -> torch.Tensor:
    return self.kernel.score_candidates_padded(
        self.queries, self.documents, self.query_lengths, self.doc_lengths
    )


def _padded_naive(self: Benchmark) -> torch.Tensor:
    return _naive_padded(
        self.queries, self.documents, self.query_lengths, self.doc_lengths
    )


def _packed_kernel(self: Benchmark) -> torch.Tensor:
    return self.kernel.score_pairs_packed(
        self.q_flat,
        self.q_offsets,
        self.d_flat,
        self.d_offsets,
        self.pair_query_ids,
        self.pair_document_ids,
        max_q_len=self.max_q_len,
    ).view(self.batch, self.candidates)


# ---------------------------------------------------------------------------
# Contrastive training (forward + backward; the headline workload).
# ---------------------------------------------------------------------------

class ContrastiveLateOn(Benchmark):
    """In-batch contrastive training: Nq=Nb=32, Lq=32, Ld=80, dim=128."""

    seed = _SEED

    def setup(self) -> None:
        _make_contrastive(self, Nq=32, Nb=32, Lq=32, Ld=80)

    def benchmark_kernel(self) -> None:
        self.out = _contrastive_train_kernel(self)

    def benchmark_naive(self) -> None:
        self.out = _contrastive_train_naive(self)

    def verify_kernel(self) -> torch.Tensor:
        return _contrastive_ref(self)

    def verify_naive(self) -> torch.Tensor:
        return _contrastive_ref(self)


class ContrastiveLongDocs(Benchmark):
    """Same in-batch shape but long docs (Ld=512) -- stresses retained state."""

    seed = _SEED

    def setup(self) -> None:
        _make_contrastive(self, Nq=32, Nb=32, Lq=32, Ld=512)

    def benchmark_kernel(self) -> None:
        self.out = _contrastive_train_kernel(self)

    def benchmark_naive(self) -> None:
        self.out = _contrastive_train_naive(self)

    def verify_kernel(self) -> torch.Tensor:
        return _contrastive_ref(self)

    def verify_naive(self) -> torch.Tensor:
        return _contrastive_ref(self)


class ContrastiveBigBatch(Benchmark):
    """Doubled in-batch batch size: Nq=Nb=64, Lq=32, Ld=128, dim=128."""

    seed = _SEED

    def setup(self) -> None:
        _make_contrastive(self, Nq=64, Nb=64, Lq=32, Ld=128)

    def benchmark_kernel(self) -> None:
        self.out = _contrastive_train_kernel(self)

    def benchmark_naive(self) -> None:
        self.out = _contrastive_train_naive(self)

    def verify_kernel(self) -> torch.Tensor:
        return _contrastive_ref(self)

    def verify_naive(self) -> torch.Tensor:
        return _contrastive_ref(self)


# ---------------------------------------------------------------------------
# Padded inference (second-stage rerank; forward only).
# ---------------------------------------------------------------------------

class PaddedRerank(Benchmark):
    """Padded rerank at a typical inference shape: B=32, K=50, Ld=180."""

    seed = _SEED

    def setup(self) -> None:
        _make_padded(self, B=32, C=50, Lq=32, Ld=180)

    def benchmark_kernel(self) -> None:
        self.out = _padded_kernel(self)

    def benchmark_naive(self) -> None:
        self.out = _padded_naive(self)

    def verify_kernel(self) -> torch.Tensor:
        return _padded_naive(self)

    def verify_naive(self) -> torch.Tensor:
        return _padded_naive(self)


class PaddedHeavyRerank(Benchmark):
    """Padded rerank at K=100 candidates, Ld=256 -- larger compute envelope."""

    seed = _SEED

    def setup(self) -> None:
        _make_padded(self, B=32, C=100, Lq=32, Ld=256)

    def benchmark_kernel(self) -> None:
        self.out = _padded_kernel(self)

    def benchmark_naive(self) -> None:
        self.out = _padded_naive(self)

    def verify_kernel(self) -> torch.Tensor:
        return _padded_naive(self)

    def verify_naive(self) -> torch.Tensor:
        return _padded_naive(self)


# ---------------------------------------------------------------------------
# Packed inference (same rerank shape via the ragged pair API; forward only).
# ---------------------------------------------------------------------------

class PackedRerank(Benchmark):
    """Rerank shape (B=32, K=50, Ld=180) expressed through the packed pair API."""

    seed = _SEED

    def setup(self) -> None:
        _make_packed(self, B=32, C=50, Lq=32, Ld=180)

    def benchmark_kernel(self) -> None:
        self.out = _packed_kernel(self)

    def benchmark_naive(self) -> None:
        self.out = _padded_naive(self)

    def verify_kernel(self) -> torch.Tensor:
        return _padded_naive(self)

    def verify_naive(self) -> torch.Tensor:
        return _padded_naive(self)


class PackedHeavyRerank(Benchmark):
    """Rerank shape (B=32, K=100, Ld=256) through the packed pair API."""

    seed = _SEED

    def setup(self) -> None:
        _make_packed(self, B=32, C=100, Lq=32, Ld=256)

    def benchmark_kernel(self) -> None:
        self.out = _packed_kernel(self)

    def benchmark_naive(self) -> None:
        self.out = _padded_naive(self)

    def verify_kernel(self) -> torch.Tensor:
        return _padded_naive(self)

    def verify_naive(self) -> torch.Tensor:
        return _padded_naive(self)
