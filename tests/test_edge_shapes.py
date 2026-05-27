"""Edge / boundary shape tests across all three APIs.

The random-shape tests in the other files exercise "typical" inputs.
This file deliberately hits values that sit on or adjacent to the
kernel's internal tile boundaries — places where off-by-ones, wrong
fast/slow-path dispatch, or alignment bugs hide.

For each API we sweep:

  * smallest non-empty (Lq=1, Ld=1, B=1, C=1, etc.)
  * exactly at the WMMA / MMA tile size (16, 32, 64 for CUDA; 8, 16, 32 for Metal)
  * one over the tile (forces a scalar tail)
  * a non-aligned dim that misses the matmul fast path (24, 96)

Forward correctness is checked against the pure-PyTorch reference;
backward (where supported) is checked against torch.autograd through
the reference. Shapes that violate a kernel-side constraint
(e.g. CUDA contrastive needs Lq % 16 == 0) are skipped on that backend
with a clear reason.
"""

from __future__ import annotations

import pytest
import torch

import maxsim
from maxsim._ops import ops
from ._helpers import DEVICE, TOLERANCES


# ---------------------------------------------------------------------------
# Padded API
# ---------------------------------------------------------------------------

# (B, C, Lq, Ld, dim) — each tagged with what boundary it exercises.
PADDED_EDGE_SHAPES = [
    pytest.param(1, 1, 1, 1, 16,     id="smallest-non-empty"),
    pytest.param(1, 1, 1, 16, 16,    id="Lq=1-Ld=16"),
    pytest.param(1, 1, 16, 1, 16,    id="Lq=16-Ld=1"),
    pytest.param(1, 1, 16, 16, 16,   id="exactly-1-WMMA-tile"),
    pytest.param(1, 1, 16, 17, 16,   id="1-tile+1-scalar-tail"),
    pytest.param(1, 1, 16, 32, 16,   id="exactly-2-tiles"),
    pytest.param(1, 1, 16, 33, 16,   id="2-tiles+1-tail"),
    pytest.param(1, 1, 16, 48, 16,   id="exactly-3-tiles"),
    pytest.param(1, 1, 16, 64, 16,   id="exactly-4-tiles-cascade"),
    pytest.param(1, 1, 16, 65, 16,   id="4-tiles+1-tail"),
    pytest.param(2, 2, 8, 24, 24,    id="dim=24-non-aligned-Lq<16"),
    pytest.param(1, 1, 32, 100, 96,  id="dim=96-non-WMMA-aligned"),
    pytest.param(4, 1, 16, 32, 64,   id="C=1-single-candidate"),
    pytest.param(1, 8, 16, 32, 64,   id="B=1-single-batch"),
]


@pytest.mark.parametrize("B, C, Lq, Ld, dim", PADDED_EDGE_SHAPES)
def test_padded_forward_edge_shape(B, C, Lq, Ld, dim) -> None:
    torch.manual_seed(0)
    dtype = torch.float16
    queries = torch.randn(B, Lq, dim, device=DEVICE, dtype=dtype)
    documents = torch.randn(B, C, Ld, dim, device=DEVICE, dtype=dtype)
    qlen = torch.full((B,), Lq, dtype=torch.int32, device=DEVICE)
    dlen = torch.full((B, C), Ld, dtype=torch.int32, device=DEVICE)

    kernel = maxsim.score_candidates_padded(queries, documents, qlen, dlen)
    ref = maxsim.score_candidates_padded_reference(
        queries, documents, qlen, dlen
    )
    torch.testing.assert_close(kernel, ref, **TOLERANCES[dtype])


# Backward requires the WMMA argmax variant (CUDA) or the Metal argmax
# variant. CUDA argmax gates on Lq % 16 == 0; we shrink the sweep to
# WMMA-eligible shapes for the backward subset.
PADDED_BACKWARD_EDGE_SHAPES = [
    pytest.param(1, 1, 16, 16, 16,   id="smallest-WMMA"),
    pytest.param(1, 1, 16, 17, 16,   id="1-tile+1-tail"),
    pytest.param(1, 1, 16, 32, 16,   id="exactly-2-tiles"),
    pytest.param(1, 1, 16, 64, 16,   id="exactly-4-tiles"),
    pytest.param(1, 1, 32, 1, 16,    id="Lq=32-Ld=1"),
    pytest.param(4, 1, 16, 33, 64,   id="C=1-with-tail"),
    pytest.param(1, 8, 32, 50, 64,   id="B=1-many-candidates"),
]


@pytest.mark.parametrize("B, C, Lq, Ld, dim", PADDED_BACKWARD_EDGE_SHAPES)
def test_padded_backward_edge_shape(B, C, Lq, Ld, dim) -> None:
    torch.manual_seed(0)
    dtype = torch.float16
    q = torch.randn(B, Lq, dim, device=DEVICE, dtype=dtype, requires_grad=True)
    d = torch.randn(
        B, C, Ld, dim, device=DEVICE, dtype=dtype, requires_grad=True,
    )
    qlen = torch.full((B,), Lq, dtype=torch.int32, device=DEVICE)
    dlen = torch.full((B, C), Ld, dtype=torch.int32, device=DEVICE)

    scores = maxsim.score_candidates_padded_train(q, d, qlen, dlen)
    weights = torch.randn(B, C, device=DEVICE, dtype=torch.float32)
    (scores * weights).sum().backward()

    # Reference grads via torch.autograd through the pure-PyTorch forward.
    q_ref = q.detach().clone().float().requires_grad_(True)
    d_ref = d.detach().clone().float().requires_grad_(True)
    scores_ref = maxsim.score_candidates_padded_reference(
        q_ref, d_ref, qlen, dlen
    )
    (scores_ref * weights).sum().backward()

    tol = TOLERANCES[dtype]
    torch.testing.assert_close(q.grad.float(), q_ref.grad, **tol)
    torch.testing.assert_close(d.grad.float(), d_ref.grad, **tol)


# ---------------------------------------------------------------------------
# Packed API
# ---------------------------------------------------------------------------

# (num_queries, num_documents, num_pairs, q_len, d_len, dim)
PACKED_EDGE_SHAPES = [
    pytest.param(1, 1, 1, 1, 1, 16,    id="single-q-d-pair-tiny"),
    pytest.param(1, 1, 1, 1, 100, 16,  id="single-q-tok-many-d-toks"),
    pytest.param(1, 1, 1, 100, 1, 16,  id="many-q-toks-single-d-tok"),
    pytest.param(1, 1, 1, 16, 16, 24,  id="dim=24-non-aligned"),
    pytest.param(1, 1, 1, 8, 7, 8,     id="all-under-MMA-tile"),
    pytest.param(2, 2, 4, 16, 16, 32,  id="all-pairs-square"),
    pytest.param(3, 5, 1, 10, 50, 64,  id="single-pair-larger"),
    pytest.param(5, 5, 25, 32, 100, 128, id="many-pairs"),
]


@pytest.mark.parametrize(
    "Nq, Nd, npairs, q_len, d_len, dim", PACKED_EDGE_SHAPES
)
def test_packed_forward_edge_shape(Nq, Nd, npairs, q_len, d_len, dim) -> None:
    torch.manual_seed(0)
    dtype = torch.float16

    # Build packed inputs with uniform per-query / per-doc lengths.
    queries = torch.randn(Nq * q_len, dim, device=DEVICE, dtype=dtype)
    documents = torch.randn(Nd * d_len, dim, device=DEVICE, dtype=dtype)
    qoff = torch.arange(0, (Nq + 1) * q_len, q_len, dtype=torch.int32, device=DEVICE)
    doff = torch.arange(0, (Nd + 1) * d_len, d_len, dtype=torch.int32, device=DEVICE)

    # Build deterministic pair ids (first npairs of the cartesian product).
    pairs = [(i % Nq, i % Nd) for i in range(npairs)]
    qids = torch.tensor([p[0] for p in pairs], dtype=torch.int32, device=DEVICE)
    dids = torch.tensor([p[1] for p in pairs], dtype=torch.int32, device=DEVICE)

    kernel = maxsim.score_pairs_packed(
        queries, qoff, documents, doff, qids, dids, max_q_len=q_len,
    )
    ref = maxsim.score_pairs_packed_reference(
        queries, qoff, documents, doff, qids, dids,
    )
    torch.testing.assert_close(kernel, ref, **TOLERANCES[dtype])


# ---------------------------------------------------------------------------
# Contrastive API
# ---------------------------------------------------------------------------

# CUDA contrastive requires Lq % 16 == 0 AND dim % 16 == 0 (WMMA-only,
# no scalar fallback yet). We keep the sweep WMMA-eligible on both sides
# so the test runs cross-backend; non-aligned-dim coverage is in the
# padded sweep which has a scalar fallback.
CONTRASTIVE_EDGE_SHAPES = [
    pytest.param(1, 1, 16, [16], 16,         id="single-q-single-d"),
    pytest.param(1, 1, 16, [1], 16,          id="Ld=1"),
    pytest.param(1, 1, 16, [17], 16,         id="Ld=17-with-tail"),
    pytest.param(1, 4, 16, [16, 32, 48, 64], 32, id="cascade-boundaries"),
    pytest.param(2, 3, 16, [7, 8, 100], 32,  id="mixed-ragged-Ld"),
    pytest.param(4, 4, 32, [50, 50, 50, 50], 128, id="typical-shape"),
    pytest.param(1, 1, 16, [1], 96,          id="dim=96-larger"),
]


@pytest.mark.parametrize("Nq, Nb, Lq, d_lens, dim", CONTRASTIVE_EDGE_SHAPES)
def test_contrastive_forward_edge_shape(Nq, Nb, Lq, d_lens, dim) -> None:
    assert len(d_lens) == Nb
    torch.manual_seed(0)
    dtype = torch.float16

    queries = torch.randn(Nq, Lq, dim, device=DEVICE, dtype=dtype)
    total_d = sum(d_lens)
    documents = torch.randn(total_d, dim, device=DEVICE, dtype=dtype)
    cu = torch.zeros(Nb + 1, dtype=torch.int32, device=DEVICE)
    cu[1:] = torch.tensor(d_lens, dtype=torch.int32).cumsum(0).to(DEVICE)

    kernel = maxsim.score_contrastive(queries, documents, cu)
    ref = maxsim.score_contrastive_reference(queries, documents, cu)
    torch.testing.assert_close(kernel, ref, **TOLERANCES[dtype])
