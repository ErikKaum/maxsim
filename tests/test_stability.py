"""Long-running stability test.

Runs ~200 forward+backward iterations and checks: (1) no NaN/inf appears
in loss or gradients, (2) allocated GPU memory doesn't grow unboundedly
(catches per-step leaks), (3) gradients stay finite under repeated atomic
accumulation.

This is the canary for "looks fine on a single step, blows up over a real
training run" failures.

Cross-backend: works on CUDA and MPS via a tiny memory-snapshot helper.
"""

from __future__ import annotations

import math

import pytest
import torch

import maxsim
from ._helpers import DEVICE


def _allocated_mb() -> float:
    """Currently-allocated GPU memory in MB. 0 for CPU."""
    if DEVICE.type == "cuda":
        return torch.cuda.memory_allocated() / (1024 * 1024)
    if DEVICE.type == "mps":
        return torch.mps.current_allocated_memory() / (1024 * 1024)
    return 0.0


def _reset_peak() -> None:
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    # MPS doesn't have a reset hook; we just rely on the snapshot at the end.


def _is_finite(t: torch.Tensor) -> bool:
    return bool(torch.isfinite(t).all().item())


@pytest.mark.parametrize("api", ["padded", "contrastive", "packed"])
def test_no_leak_or_nan_over_many_iterations(api) -> None:
    """200 fwd+bwd iterations of each training surface; check memory stays
    flat and no NaN/inf appears."""
    torch.manual_seed(0)
    dtype = torch.float16
    n_iter = 200
    lr = 0.01

    # Build inputs once and clone-with-grad each iteration (matching how a
    # real training loop reallocates the loss graph).
    if api == "padded":
        B, C, Lq, Ld, D = 2, 4, 16, 32, 64
        q0 = torch.randn(B, Lq, D, device=DEVICE, dtype=dtype)
        d0 = torch.randn(B, C, Ld, D, device=DEVICE, dtype=dtype)
        qlen = torch.full((B,), Lq, dtype=torch.int32, device=DEVICE)
        dlen = torch.full((B, C), Ld, dtype=torch.int32, device=DEVICE)

        def step(q, d):
            scores = maxsim.score_candidates_padded_train(q, d, qlen, dlen)
            return scores.mean()

    elif api == "contrastive":
        Nq, Nb, Lq, Ld, D = 4, 4, 16, 32, 64
        q0 = torch.randn(Nq, Lq, D, device=DEVICE, dtype=dtype)
        d0 = torch.randn(Nb * Ld, D, device=DEVICE, dtype=dtype)
        cu = torch.arange(0, (Nb + 1) * Ld, Ld, dtype=torch.int32, device=DEVICE)

        def step(q, d):
            scores = maxsim.score_contrastive_train(q, d, cu)
            return scores.mean()

    else:  # packed
        Nq, Nb, Lq, Ld, D = 3, 4, 12, 24, 64
        q0 = torch.randn(Nq * Lq, D, device=DEVICE, dtype=dtype)
        d0 = torch.randn(Nb * Ld, D, device=DEVICE, dtype=dtype)
        qoff = torch.arange(0, (Nq + 1) * Lq, Lq, dtype=torch.int32, device=DEVICE)
        doff = torch.arange(0, (Nb + 1) * Ld, Ld, dtype=torch.int32, device=DEVICE)
        # 6 pairs sampled deterministically.
        qids = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.int32, device=DEVICE)
        dids = torch.tensor([0, 1, 2, 3, 0, 1], dtype=torch.int32, device=DEVICE)

        def step(q, d):
            scores = maxsim.score_pairs_packed_train(
                q, qoff, d, doff, qids, dids, max_q_len=Lq,
            )
            return scores.mean()

    _reset_peak()
    # Warmup so allocator settles before we sample baseline memory.
    for _ in range(5):
        q = q0.detach().clone().requires_grad_(True)
        d = d0.detach().clone().requires_grad_(True)
        loss = step(q, d)
        loss.backward()
        with torch.no_grad():
            q.sub_(lr * q.grad.to(q.dtype))
            d.sub_(lr * d.grad.to(d.dtype))
    mem_start = _allocated_mb()

    # Periodically check loss + grads finite. Sample 10 times across the run.
    sample_every = max(1, n_iter // 10)
    last_loss = math.nan

    for i in range(n_iter):
        q = q0.detach().clone().requires_grad_(True)
        d = d0.detach().clone().requires_grad_(True)
        loss = step(q, d)
        loss.backward()
        with torch.no_grad():
            q.sub_(lr * q.grad.to(q.dtype))
            d.sub_(lr * d.grad.to(d.dtype))

        if i % sample_every == 0:
            assert math.isfinite(float(loss.item())), \
                f"non-finite loss at iter {i}: {loss.item()}"
            assert _is_finite(q.grad), f"non-finite q.grad at iter {i}"
            assert _is_finite(d.grad), f"non-finite d.grad at iter {i}"
            last_loss = float(loss.item())

    mem_end = _allocated_mb()

    # Memory should be flat — allow up to 20 MB drift from allocator caching
    # but no unbounded growth.
    leak_mb = mem_end - mem_start
    assert leak_mb < 20.0, (
        f"possible memory leak in {api}: started at {mem_start:.1f} MB, "
        f"ended at {mem_end:.1f} MB after {n_iter} iters (delta {leak_mb:.1f} MB)"
    )
    # Final loss is finite (already checked in loop sampling).
    assert math.isfinite(last_loss)
