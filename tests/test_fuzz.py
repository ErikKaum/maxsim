"""Property-based fuzz tests.

Random valid (shape, dtype) tuples for each API, compared against the
reference. The point is to catch divergence between the kernel's fast
and slow paths on shapes that hand-picked tests don't happen to hit.

Each ``test_*_fuzz`` runs ``N_ITERS`` random configurations. Bumping
``N_ITERS`` increases coverage at the cost of test time; the defaults
keep the full suite under ~30s.
"""

from __future__ import annotations

import random

import pytest
import torch

import maxsim
from ._helpers import DEVICE, TOLERANCES


# Number of random configurations per API. Tuned so the file finishes
# in ~10s on a modern Mac.
N_ITERS = 50


def _fuzz_dtypes():
    # fp16 + bf16 (skip fp32 — random tests already cover it elsewhere
    # and we want training-mode dtypes specifically).
    dts = [torch.float16]
    if DEVICE.type == "cuda":
        dts.append(torch.bfloat16)
    elif DEVICE.type == "mps":
        dts.append(torch.bfloat16)
    return dts


def _random_int(rng: random.Random, lo: int, hi: int, multiple_of: int = 1) -> int:
    """Random int in [lo, hi] rounded up to a multiple of ``multiple_of``."""
    if multiple_of <= 1:
        return rng.randint(lo, hi)
    n = rng.randint(lo, hi)
    return max(multiple_of, ((n + multiple_of - 1) // multiple_of) * multiple_of)


# ---------------------------------------------------------------------------
# Padded fuzz
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", _fuzz_dtypes())
def test_padded_forward_fuzz(dtype) -> None:
    """Random padded shapes, fp16/bf16, forward vs reference."""
    rng = random.Random(123)
    for i in range(N_ITERS):
        B = rng.randint(1, 4)
        C = rng.randint(1, 8)
        Lq = rng.randint(1, 64)
        Ld_max = rng.randint(1, 200)
        dim = rng.choice([16, 24, 32, 64, 96, 128])

        torch.manual_seed(i * 7919 + 1)
        queries = torch.randn(B, Lq, dim, device=DEVICE, dtype=dtype)
        documents = torch.randn(
            B, C, Ld_max, dim, device=DEVICE, dtype=dtype,
        )
        qlen = torch.randint(
            1, Lq + 1, (B,), dtype=torch.int32, device=DEVICE,
        )
        dlen = torch.randint(
            1, Ld_max + 1, (B, C), dtype=torch.int32, device=DEVICE,
        )

        kernel = maxsim.score_candidates_padded(
            queries, documents, qlen, dlen
        )
        ref = maxsim.score_candidates_padded_reference(
            queries, documents, qlen, dlen
        )
        torch.testing.assert_close(
            kernel, ref, **TOLERANCES[dtype],
        ), f"iter {i}: shapes B={B} C={C} Lq={Lq} Ld_max={Ld_max} dim={dim}"


# ---------------------------------------------------------------------------
# Packed fuzz
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", _fuzz_dtypes())
def test_packed_forward_fuzz(dtype) -> None:
    """Random packed shapes — fully ragged."""
    rng = random.Random(7777)
    for i in range(N_ITERS):
        Nq = rng.randint(1, 6)
        Nd = rng.randint(1, 8)
        n_pairs = rng.randint(1, 16)
        dim = rng.choice([16, 24, 32, 64, 96, 128])

        # Per-query / per-doc lengths.
        q_lens = [rng.randint(1, 32) for _ in range(Nq)]
        d_lens = [rng.randint(1, 128) for _ in range(Nd)]
        total_q = sum(q_lens)
        total_d = sum(d_lens)

        torch.manual_seed(i * 4283 + 1)
        queries = torch.randn(total_q, dim, device=DEVICE, dtype=dtype)
        documents = torch.randn(total_d, dim, device=DEVICE, dtype=dtype)

        qoff = torch.zeros(Nq + 1, dtype=torch.int32, device=DEVICE)
        qoff[1:] = torch.tensor(
            q_lens, dtype=torch.int32, device=DEVICE,
        ).cumsum(0)
        doff = torch.zeros(Nd + 1, dtype=torch.int32, device=DEVICE)
        doff[1:] = torch.tensor(
            d_lens, dtype=torch.int32, device=DEVICE,
        ).cumsum(0)

        qids = torch.tensor(
            [rng.randint(0, Nq - 1) for _ in range(n_pairs)],
            dtype=torch.int32, device=DEVICE,
        )
        dids = torch.tensor(
            [rng.randint(0, Nd - 1) for _ in range(n_pairs)],
            dtype=torch.int32, device=DEVICE,
        )

        max_q_len = max(q_lens)
        kernel = maxsim.score_pairs_packed(
            queries, qoff, documents, doff, qids, dids,
            max_q_len=max_q_len,
        )
        ref = maxsim.score_pairs_packed_reference(
            queries, qoff, documents, doff, qids, dids,
        )
        torch.testing.assert_close(
            kernel, ref, **TOLERANCES[dtype],
        ), f"iter {i}: Nq={Nq} Nd={Nd} pairs={n_pairs} dim={dim}"


# ---------------------------------------------------------------------------
# Contrastive fuzz
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", _fuzz_dtypes())
def test_contrastive_forward_fuzz(dtype) -> None:
    """Random contrastive shapes. Lq is constrained to multiples of 16 to
    satisfy the CUDA WMMA gate (so the test stays cross-backend)."""
    rng = random.Random(31337)
    for i in range(N_ITERS):
        Nq = rng.randint(1, 6)
        Nb = rng.randint(1, 8)
        Lq = _random_int(rng, 16, 64, multiple_of=16)
        dim = rng.choice([16, 32, 64, 96, 128])

        d_lens = [rng.randint(1, 200) for _ in range(Nb)]
        total_d = sum(d_lens)

        torch.manual_seed(i * 9013 + 1)
        queries = torch.randn(Nq, Lq, dim, device=DEVICE, dtype=dtype)
        documents = torch.randn(total_d, dim, device=DEVICE, dtype=dtype)
        cu = torch.zeros(Nb + 1, dtype=torch.int32, device=DEVICE)
        cu[1:] = torch.tensor(
            d_lens, dtype=torch.int32, device=DEVICE,
        ).cumsum(0)

        kernel = maxsim.score_contrastive(queries, documents, cu)
        ref = maxsim.score_contrastive_reference(queries, documents, cu)
        torch.testing.assert_close(
            kernel, ref, **TOLERANCES[dtype],
        ), f"iter {i}: Nq={Nq} Nb={Nb} Lq={Lq} dim={dim}"
