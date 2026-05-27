"""Tests for the padded backward kernel.

Validates the kernel-computed (dqueries, ddocuments) against
``torch.autograd.grad`` applied to the pure-PyTorch reference. Both should
agree to fp16/bf16 precision since the kernel uses fp32 accumulation
identical to the reference (the gradient formula has no max/argmax
sensitivity once argmax is fixed; it's just linear routing).
"""

from __future__ import annotations

import pytest
import torch

import maxsim
from maxsim._ops import ops
from ._helpers import DEVICE, TOLERANCES, supported_float_dtypes


def _backward_dtypes():
    """Backward kernel supports fp16/bf16 across backends. fp32 is not yet
    wired up (we have no fp32 argmax forward to feed it)."""
    return [dt for dt in supported_float_dtypes() if dt != torch.float32]


def _autograd_grad_via_reference(q, d, query_lengths, doc_lengths, dscore):
    """Compute reference gradients via PyTorch autograd on the pure-PyTorch
    forward. Inputs are detached + requires_grad=True clones."""
    q_ref = q.detach().clone().float().requires_grad_(True)
    d_ref = d.detach().clone().float().requires_grad_(True)
    scores = maxsim.score_candidates_padded_reference(
        q_ref, d_ref, query_lengths, doc_lengths
    )
    # Apply incoming gradient as a sum-product, matching kernel API.
    (scores * dscore).sum().backward()
    return q_ref.grad.detach(), d_ref.grad.detach()


@pytest.mark.parametrize("dtype", _backward_dtypes())
def test_padded_backward_matches_pytorch_autograd(dtype) -> None:
    """End-to-end: forward kernel produces argmax, backward kernel routes
    gradients via the saved argmax, result matches PyTorch autograd."""
    torch.manual_seed(0)
    B, C, Lq, Ld, D = 2, 4, 16, 32, 64  # 16-aligned for CUDA WMMA gate

    q = torch.randn(B, Lq, D, device=DEVICE, dtype=dtype)
    d = torch.randn(B, C, Ld, D, device=DEVICE, dtype=dtype)
    query_lengths = torch.full((B,), Lq, dtype=torch.int32, device=DEVICE)
    doc_lengths = torch.full((B, C), Ld, dtype=torch.int32, device=DEVICE)
    dscore = torch.randn(B, C, device=DEVICE, dtype=torch.float32)

    # Forward + argmax via kernel.
    _scores_k, argmax = maxsim.score_candidates_padded_with_argmax(
        q, d, query_lengths, doc_lengths
    )

    # Backward via kernel. The op expects flattened layouts matching the
    # forward op's expectations.
    q_flat = q.reshape(B * Lq, D).contiguous()
    d_flat = d.reshape(B * C * Ld, D).contiguous()
    qlen = query_lengths.contiguous()
    dlen = doc_lengths.reshape(B * C).contiguous()
    dscore_flat = dscore.reshape(B * C).contiguous()
    argmax_flat = argmax.reshape(B * C, Lq).contiguous()

    dq_flat, dd_flat = ops.maxsim_padded_backward(
        dscore_flat,
        q_flat,
        d_flat,
        qlen,
        dlen,
        argmax_flat,
        int(Lq),
        int(Ld),
        int(C),
    )
    dq_k = dq_flat.view(B, Lq, D)
    dd_k = dd_flat.view(B, C, Ld, D)

    # Reference via torch.autograd on the pure-PyTorch forward.
    dq_ref, dd_ref = _autograd_grad_via_reference(
        q, d, query_lengths, doc_lengths, dscore
    )

    tol = TOLERANCES[dtype]
    torch.testing.assert_close(dq_k, dq_ref, **tol)
    torch.testing.assert_close(dd_k, dd_ref, **tol)


@pytest.mark.parametrize("dtype", _backward_dtypes())
def test_padded_backward_zero_dscore(dtype) -> None:
    """If dscore is all zeros, both gradient tensors should be all zeros."""
    B, C, Lq, Ld, D = 1, 2, 16, 16, 32
    torch.manual_seed(1)

    q = torch.randn(B, Lq, D, device=DEVICE, dtype=dtype)
    d = torch.randn(B, C, Ld, D, device=DEVICE, dtype=dtype)
    query_lengths = torch.full((B,), Lq, dtype=torch.int32, device=DEVICE)
    doc_lengths = torch.full((B, C), Ld, dtype=torch.int32, device=DEVICE)
    dscore = torch.zeros(B, C, device=DEVICE, dtype=torch.float32)

    _, argmax = maxsim.score_candidates_padded_with_argmax(
        q, d, query_lengths, doc_lengths
    )

    dq, dd = ops.maxsim_padded_backward(
        dscore.reshape(B * C).contiguous(),
        q.reshape(B * Lq, D).contiguous(),
        d.reshape(B * C * Ld, D).contiguous(),
        query_lengths.contiguous(),
        doc_lengths.reshape(B * C).contiguous(),
        argmax.reshape(B * C, Lq).contiguous(),
        int(Lq), int(Ld), int(C),
    )
    assert torch.all(dq == 0)
    assert torch.all(dd == 0)


@pytest.mark.parametrize("dtype", _backward_dtypes())
def test_score_candidates_padded_train_end_to_end(dtype) -> None:
    """The autograd.Function wrapper integrates end-to-end with
    ``loss.backward()`` exactly like a regular PyTorch op."""
    torch.manual_seed(2)
    B, C, Lq, Ld, D = 2, 4, 16, 32, 64

    q = torch.randn(
        B, Lq, D, device=DEVICE, dtype=dtype, requires_grad=True
    )
    d = torch.randn(
        B, C, Ld, D, device=DEVICE, dtype=dtype, requires_grad=True
    )
    query_lengths = torch.full((B,), Lq, dtype=torch.int32, device=DEVICE)
    doc_lengths = torch.full((B, C), Ld, dtype=torch.int32, device=DEVICE)

    scores = maxsim.score_candidates_padded_train(
        q, d, query_lengths, doc_lengths
    )
    # Use a non-trivial loss to exercise the chain rule.
    weights = torch.randn(B, C, device=DEVICE, dtype=torch.float32)
    (scores * weights).sum().backward()

    # Reference grads.
    dq_ref, dd_ref = _autograd_grad_via_reference(
        q, d, query_lengths, doc_lengths, weights
    )

    # PyTorch downcasts fp32 grads from our kernel to match q/d dtype when
    # storing into .grad. Compare in fp32 to keep the precision floor clean.
    tol = TOLERANCES[dtype]
    torch.testing.assert_close(q.grad.float(), dq_ref, **tol)
    torch.testing.assert_close(d.grad.float(), dd_ref, **tol)
