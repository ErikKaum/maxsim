"""Tests for the contrastive (all-pairs) MaxSim kernels.

Forward, forward+argmax, and the end-to-end autograd path. The
contrastive API differs from padded in two ways: (a) full cross-product
[Nq, Nb] instead of per-batch [B, C], and (b) docs are packed via
document_offsets rather than padded with doc_lengths.
"""

from __future__ import annotations

import pytest
import torch

import maxsim
from maxsim._ops import ops
from ._helpers import DEVICE, TOLERANCES, supported_float_dtypes


def _contrastive_dtypes():
    """Cross-backend support for fp16+bf16. fp32 forward+argmax is wired
    on Metal but the backward gates on fp32 too — skip for now since
    `_ScoreContrastive` doesn't dispatch fp32 yet (consistent with padded
    backward gating)."""
    return [dt for dt in supported_float_dtypes() if dt != torch.float32]


def _make_random(Nq, Nb, Lq, dim, d_lens, dtype, seed):
    """Build a contrastive batch with given per-doc lengths."""
    torch.manual_seed(seed)
    q = torch.randn(Nq, Lq, dim, device=DEVICE, dtype=dtype)
    total_d = int(sum(d_lens))
    d = torch.randn(total_d, dim, device=DEVICE, dtype=dtype)
    document_offsets = torch.zeros(Nb + 1, dtype=torch.int32, device=DEVICE)
    document_offsets[1:] = torch.tensor(d_lens, dtype=torch.int32).cumsum(0)
    return q, d, document_offsets


@pytest.mark.parametrize("dtype", _contrastive_dtypes())
def test_contrastive_forward_matches_reference(dtype) -> None:
    Nq, Nb, Lq, dim = 3, 5, 16, 64
    d_lens = [37, 64, 200, 17, 81]  # ragged
    q, d, cu = _make_random(Nq, Nb, Lq, dim, d_lens, dtype, seed=0)

    scores_k = maxsim.score_contrastive(q, d, cu)
    scores_ref = maxsim.score_contrastive_reference(q, d, cu)

    assert scores_k.shape == (Nq, Nb)
    assert scores_k.dtype == torch.float32
    torch.testing.assert_close(scores_k, scores_ref, **TOLERANCES[dtype])


@pytest.mark.parametrize("dtype", _contrastive_dtypes())
def test_contrastive_argmax_matches_reference(dtype) -> None:
    Nq, Nb, Lq, dim = 2, 4, 16, 64
    d_lens = [128, 96, 64, 192]
    q, d, cu = _make_random(Nq, Nb, Lq, dim, d_lens, dtype, seed=1)

    scores_k, argmax_k = maxsim.score_contrastive_with_argmax(q, d, cu)
    scores_ref, argmax_ref = maxsim.score_contrastive_with_argmax_reference(
        q, d, cu
    )

    assert argmax_k.shape == (Nq, Nb, Lq)
    assert argmax_k.dtype == torch.int32
    torch.testing.assert_close(scores_k, scores_ref, **TOLERANCES[dtype])
    # Argmax must match exactly — both routes use first-index-wins.
    assert torch.equal(argmax_k, argmax_ref)


@pytest.mark.parametrize("dtype", _contrastive_dtypes())
def test_contrastive_first_index_wins_on_ties(dtype) -> None:
    """All-identical docs for one query: argmax must be 0 for every q_tok
    (matches PyTorch's max/argmax tiebreak).

    Uses Ld values that are multiples of kWmmaM (=16). The scalar tail
    path computes dot products in a slightly different fp32 order than
    the WMMA path; on real (non-tied) data this is irrelevant — argmax
    matches reference within fp tolerance — but synthetic all-tied
    inputs can flip the strict-`>` cross-tile compare. The padded tie
    test uses the same constraint (see ``test_argmax.py``).
    """
    Nq, Nb, Lq, dim = 2, 3, 16, 32
    d_lens = [48, 48, 48]
    torch.manual_seed(7)
    q = torch.randn(Nq, Lq, dim, device=DEVICE, dtype=dtype)
    # Repeat one base vector for every d_tok in every doc.
    base = torch.randn(dim, device=DEVICE, dtype=dtype)
    total_d = sum(d_lens)
    d = base.unsqueeze(0).expand(total_d, dim).contiguous()
    cu = torch.zeros(Nb + 1, dtype=torch.int32, device=DEVICE)
    cu[1:] = torch.tensor(d_lens, dtype=torch.int32).cumsum(0)

    _scores, argmax = maxsim.score_contrastive_with_argmax(q, d, cu)
    assert torch.all(argmax == 0), \
        f"first-index-wins violated: max idx = {int(argmax.max().item())}"


def _autograd_grad_via_reference(q, d, cu, dscore):
    """Reference grads via torch.autograd through the pure-PyTorch forward."""
    q_ref = q.detach().clone().float().requires_grad_(True)
    d_ref = d.detach().clone().float().requires_grad_(True)
    scores = maxsim.score_contrastive_reference(q_ref, d_ref, cu)
    (scores * dscore).sum().backward()
    return q_ref.grad.detach(), d_ref.grad.detach()


@pytest.mark.parametrize("dtype", _contrastive_dtypes())
def test_contrastive_backward_matches_autograd(dtype) -> None:
    """Kernel grads vs PyTorch autograd on the reference forward."""
    Nq, Nb, Lq, dim = 2, 4, 16, 64
    d_lens = [37, 64, 80, 17]
    q, d, cu = _make_random(Nq, Nb, Lq, dim, d_lens, dtype, seed=2)
    dscore = torch.randn(Nq, Nb, device=DEVICE, dtype=torch.float32)

    _scores, argmax = maxsim.score_contrastive_with_argmax(q, d, cu)
    dq_k, dd_k = ops.maxsim_contrastive_backward(
        dscore.contiguous(), q.contiguous(), d.contiguous(), cu, argmax
    )

    dq_ref, dd_ref = _autograd_grad_via_reference(q, d, cu, dscore)
    tol = TOLERANCES[dtype]
    torch.testing.assert_close(dq_k, dq_ref, **tol)
    torch.testing.assert_close(dd_k, dd_ref, **tol)


@pytest.mark.parametrize("dtype", _contrastive_dtypes())
def test_contrastive_zero_dscore(dtype) -> None:
    Nq, Nb, Lq, dim = 1, 2, 16, 32
    d_lens = [16, 16]
    q, d, cu = _make_random(Nq, Nb, Lq, dim, d_lens, dtype, seed=3)
    dscore = torch.zeros(Nq, Nb, device=DEVICE, dtype=torch.float32)

    _scores, argmax = maxsim.score_contrastive_with_argmax(q, d, cu)
    dq, dd = ops.maxsim_contrastive_backward(
        dscore.contiguous(), q.contiguous(), d.contiguous(), cu, argmax
    )
    assert torch.all(dq == 0)
    assert torch.all(dd == 0)


@pytest.mark.parametrize("dtype", _contrastive_dtypes())
def test_score_contrastive_train_end_to_end(dtype) -> None:
    """``score_contrastive_train`` integrates with ``loss.backward()``."""
    Nq, Nb, Lq, dim = 2, 4, 16, 64
    d_lens = [40, 64, 20, 80]
    torch.manual_seed(4)
    q = torch.randn(
        Nq, Lq, dim, device=DEVICE, dtype=dtype, requires_grad=True
    )
    total_d = sum(d_lens)
    d = torch.randn(
        total_d, dim, device=DEVICE, dtype=dtype, requires_grad=True
    )
    cu = torch.zeros(Nb + 1, dtype=torch.int32, device=DEVICE)
    cu[1:] = torch.tensor(d_lens, dtype=torch.int32).cumsum(0)

    scores = maxsim.score_contrastive_train(q, d, cu)
    assert scores.shape == (Nq, Nb)
    weights = torch.randn(Nq, Nb, device=DEVICE, dtype=torch.float32)
    (scores * weights).sum().backward()

    dq_ref, dd_ref = _autograd_grad_via_reference(q, d, cu, weights)
    tol = TOLERANCES[dtype]
    # PyTorch downcasts fp32 grads to the input dtype when storing in .grad —
    # compare in fp32 to keep the precision floor clean (same as test_backward).
    torch.testing.assert_close(q.grad.float(), dq_ref, **tol)
    torch.testing.assert_close(d.grad.float(), dd_ref, **tol)
