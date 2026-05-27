"""Compatibility tests for real-training-loop scenarios.

Covers three cases that aren't exercised by the per-API correctness tests
but bite users in practice:

  1. ``torch.autocast`` contexts — most training loops wrap forward in one.
  2. ``torch.compile`` — many users wrap models in ``torch.compile``.
  3. Non-contiguous inputs — slicing or transposing upstream produces
     non-contig tensors; our Python API calls ``.contiguous()`` defensively
     but it's worth verifying the path is sound.
"""

from __future__ import annotations

import platform

import pytest
import torch

import maxsim
from ._helpers import DEVICE, TOLERANCES


# ---------------------------------------------------------------------------
# torch.autocast
# ---------------------------------------------------------------------------

def _autocast_dtype() -> torch.dtype:
    # bf16 has better numeric range; preferred where supported.
    if DEVICE.type == "cuda":
        return torch.bfloat16
    # MPS bf16 autocast support is recent; fall back to fp16 for portability.
    return torch.float16


@pytest.mark.parametrize("api", ["padded", "contrastive", "packed"])
def test_works_inside_autocast(api) -> None:
    """All three training surfaces should work when wrapped in a
    ``torch.autocast`` context. We pass inputs at the kernel's dtype
    (autocast policy isn't registered for our custom ops; users hand us
    pre-cast tensors) — the test is that autocast doesn't break us.
    """
    dt = _autocast_dtype()
    torch.manual_seed(0)

    if api == "padded":
        B, C, Lq, Ld, D = 2, 4, 16, 32, 64
        q = torch.randn(B, Lq, D, device=DEVICE, dtype=dt, requires_grad=True)
        d = torch.randn(B, C, Ld, D, device=DEVICE, dtype=dt, requires_grad=True)
        qlen = torch.full((B,), Lq, dtype=torch.int32, device=DEVICE)
        dlen = torch.full((B, C), Ld, dtype=torch.int32, device=DEVICE)

        with torch.autocast(device_type=DEVICE.type, dtype=dt):
            scores = maxsim.score_candidates_padded_train(q, d, qlen, dlen)
            loss = scores.sum()
        loss.backward()
        assert torch.isfinite(loss).item()
        assert q.grad is not None and torch.isfinite(q.grad).all().item()
        assert d.grad is not None and torch.isfinite(d.grad).all().item()

    elif api == "contrastive":
        Nq, Nb, Lq, Ld, D = 4, 4, 16, 32, 64
        q = torch.randn(Nq, Lq, D, device=DEVICE, dtype=dt, requires_grad=True)
        d = torch.randn(
            Nb * Ld, D, device=DEVICE, dtype=dt, requires_grad=True
        )
        cu = torch.arange(0, (Nb + 1) * Ld, Ld, dtype=torch.int32, device=DEVICE)

        with torch.autocast(device_type=DEVICE.type, dtype=dt):
            scores = maxsim.score_contrastive_train(q, d, cu)
            loss = scores.sum()
        loss.backward()
        assert torch.isfinite(loss).item()
        assert q.grad is not None and torch.isfinite(q.grad).all().item()

    else:  # packed
        Nq, Nb, Lq, Ld, D = 3, 4, 12, 24, 64
        q = torch.randn(
            Nq * Lq, D, device=DEVICE, dtype=dt, requires_grad=True
        )
        d = torch.randn(
            Nb * Ld, D, device=DEVICE, dtype=dt, requires_grad=True
        )
        qoff = torch.arange(0, (Nq + 1) * Lq, Lq, dtype=torch.int32, device=DEVICE)
        doff = torch.arange(0, (Nb + 1) * Ld, Ld, dtype=torch.int32, device=DEVICE)
        qids = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.int32, device=DEVICE)
        dids = torch.tensor([0, 1, 2, 3, 0, 1], dtype=torch.int32, device=DEVICE)

        with torch.autocast(device_type=DEVICE.type, dtype=dt):
            scores = maxsim.score_pairs_packed_train(
                q, qoff, d, doff, qids, dids, max_q_len=Lq,
            )
            loss = scores.sum()
        loss.backward()
        assert torch.isfinite(loss).item()
        assert q.grad is not None and torch.isfinite(q.grad).all().item()


# ---------------------------------------------------------------------------
# torch.compile
# ---------------------------------------------------------------------------

def _compile_supported() -> bool:
    """torch.compile on MPS is still experimental as of torch 2.6 / 2.12;
    skip there. CUDA support is well-established."""
    if DEVICE.type != "cuda":
        return False
    return hasattr(torch, "compile")


@pytest.mark.skipif(
    not _compile_supported(),
    reason="torch.compile flaky/unsupported on this backend",
)
@pytest.mark.parametrize("api", ["padded", "contrastive"])
def test_works_under_torch_compile(api) -> None:
    """Wrap a training step in ``torch.compile`` and verify it produces the
    same output as eager. Validates the custom op is dispatcher-visible
    in a way ``torch.compile`` accepts."""
    dt = torch.float16
    torch.manual_seed(1)

    if api == "padded":
        B, C, Lq, Ld, D = 2, 4, 16, 32, 64
        q = torch.randn(B, Lq, D, device=DEVICE, dtype=dt)
        d = torch.randn(B, C, Ld, D, device=DEVICE, dtype=dt)
        qlen = torch.full((B,), Lq, dtype=torch.int32, device=DEVICE)
        dlen = torch.full((B, C), Ld, dtype=torch.int32, device=DEVICE)

        def step(q, d):
            return maxsim.score_candidates_padded_train(
                q, d, qlen, dlen
            ).sum()

        eager = step(q, d)
        compiled = torch.compile(step, fullgraph=False)
        out = compiled(q, d)

    else:  # contrastive
        Nq, Nb, Lq, Ld, D = 4, 4, 16, 32, 64
        q = torch.randn(Nq, Lq, D, device=DEVICE, dtype=dt)
        d = torch.randn(Nb * Ld, D, device=DEVICE, dtype=dt)
        cu = torch.arange(0, (Nb + 1) * Ld, Ld, dtype=torch.int32, device=DEVICE)

        def step(q, d):
            return maxsim.score_contrastive_train(q, d, cu).sum()

        eager = step(q, d)
        compiled = torch.compile(step, fullgraph=False)
        out = compiled(q, d)

    torch.testing.assert_close(out, eager, rtol=1e-3, atol=1e-3)


# ---------------------------------------------------------------------------
# Non-contiguous inputs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("api", ["padded", "contrastive"])
def test_non_contiguous_inputs(api) -> None:
    """Pass deliberately non-contiguous tensors (made via stride-2 slicing)
    and verify the result matches a contiguous-input run. Validates the
    ``.contiguous()`` path in our Python API + C++ host functions."""
    dt = torch.float16
    torch.manual_seed(2)

    if api == "padded":
        # Create stride-2 slice along batch dim so the result is non-contig.
        B, C, Lq, Ld, D = 2, 4, 16, 32, 64
        big_q = torch.randn(2 * B, Lq, D, device=DEVICE, dtype=dt)
        big_d = torch.randn(2 * B, C, Ld, D, device=DEVICE, dtype=dt)
        q_nc = big_q[::2]  # stride 2 along B
        d_nc = big_d[::2]
        assert not q_nc.is_contiguous()
        assert not d_nc.is_contiguous()
        qlen = torch.full((B,), Lq, dtype=torch.int32, device=DEVICE)
        dlen = torch.full((B, C), Ld, dtype=torch.int32, device=DEVICE)

        scores_nc = maxsim.score_candidates_padded(q_nc, d_nc, qlen, dlen)
        scores_c = maxsim.score_candidates_padded(
            q_nc.contiguous(), d_nc.contiguous(), qlen, dlen,
        )
        torch.testing.assert_close(scores_nc, scores_c, **TOLERANCES[dt])

    else:  # contrastive
        Nq, Nb, Lq, Ld, D = 4, 4, 16, 32, 64
        # Build 2× oversize buffers, take every other row → stride-2 non-contig.
        big_q = torch.randn(2 * Nq, Lq, D, device=DEVICE, dtype=dt)
        big_d = torch.randn(2 * Nb * Ld, D, device=DEVICE, dtype=dt)
        q_nc = big_q[::2]
        d_nc = big_d[::2]
        assert not q_nc.is_contiguous()
        assert not d_nc.is_contiguous()
        cu = torch.arange(0, (Nb + 1) * Ld, Ld, dtype=torch.int32, device=DEVICE)

        scores_nc = maxsim.score_contrastive(q_nc, d_nc, cu)
        scores_c = maxsim.score_contrastive(
            q_nc.contiguous(), d_nc.contiguous(), cu,
        )
        torch.testing.assert_close(scores_nc, scores_c, **TOLERANCES[dt])


def test_padded_train_rejects_shape_and_dtype_mismatch() -> None:
    dt = torch.float16
    B, C, Lq, Ld, D = 2, 3, 16, 32, 64
    q = torch.randn(B, Lq, D, device=DEVICE, dtype=dt)
    d = torch.randn(B, C, Ld, D, device=DEVICE, dtype=torch.float32)
    qlen = torch.full((B,), Lq, dtype=torch.int32, device=DEVICE)
    dlen = torch.full((B, C), Ld, dtype=torch.int32, device=DEVICE)

    with pytest.raises(TypeError, match="same dtype"):
        maxsim.score_candidates_padded_train(q, d, qlen, dlen)

    d = d.to(dtype=dt)
    bad_dlen = dlen[:, :2]
    with pytest.raises(ValueError, match="doc_lengths"):
        maxsim.score_candidates_padded_train(q, d, qlen, bad_dlen)


def test_contrastive_train_rejects_shape_and_dtype_mismatch() -> None:
    dt = torch.float16
    Nq, Nb, Lq, Ld, D = 2, 3, 16, 32, 64
    q = torch.randn(Nq, Lq, D, device=DEVICE, dtype=dt)
    d = torch.randn(Nb * Ld, D + 1, device=DEVICE, dtype=dt)
    cu = torch.arange(0, (Nb + 1) * Ld, Ld, dtype=torch.int32, device=DEVICE)

    with pytest.raises(ValueError, match="must match"):
        maxsim.score_contrastive_train(q, d, cu)

    d = torch.randn(Nb * Ld, D, device=DEVICE, dtype=torch.float32)
    with pytest.raises(TypeError, match="share dtype"):
        maxsim.score_contrastive_train(q, d, cu)
