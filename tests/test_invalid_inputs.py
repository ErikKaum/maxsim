"""Invalid-input contract tests.

For each API and each "wrong input" scenario a real user might hit,
verify that we raise a clear, actionable error rather than crash with
a cryptic kernel-side message.

We use ``pytest.raises(match=...)`` to assert the error message
contains a useful substring. If a test fails it usually means one of:

  * the wrong-input case wasn't caught at all (kernel crash);
  * it was caught but the message is unhelpful;
  * our Python-side guard caught it correctly (the test passes).

Tests cover both inference and training entry points.
"""

from __future__ import annotations

import pytest
import torch

import maxsim
from ._helpers import DEVICE


# ---------------------------------------------------------------------------
# Padded API
# ---------------------------------------------------------------------------

def _padded_inputs(B=2, C=3, Lq=16, Ld=32, dim=64, dtype=torch.float16):
    q = torch.randn(B, Lq, dim, device=DEVICE, dtype=dtype)
    d = torch.randn(B, C, Ld, dim, device=DEVICE, dtype=dtype)
    qlen = torch.full((B,), Lq, dtype=torch.int32, device=DEVICE)
    dlen = torch.full((B, C), Ld, dtype=torch.int32, device=DEVICE)
    return q, d, qlen, dlen


def test_padded_rejects_wrong_dim_match() -> None:
    q, _d, qlen, dlen = _padded_inputs(dim=64)
    d_wrong = torch.randn(2, 3, 32, 32, device=DEVICE, dtype=q.dtype)
    with pytest.raises((ValueError, TypeError, RuntimeError), match="dim|D|dimension"):
        maxsim.score_candidates_padded(q, d_wrong, qlen, dlen)


def test_padded_rejects_dtype_mismatch() -> None:
    q, d, qlen, dlen = _padded_inputs(dtype=torch.float16)
    d_bf = d.to(torch.bfloat16)
    with pytest.raises((TypeError, RuntimeError), match="dtype"):
        maxsim.score_candidates_padded(q, d_bf, qlen, dlen)


def test_padded_rejects_int_dtype_in_q() -> None:
    q = torch.randint(0, 5, (2, 16, 64), device=DEVICE, dtype=torch.int32)
    _q_real, d, qlen, dlen = _padded_inputs()
    with pytest.raises((TypeError, RuntimeError), match="float|dtype"):
        maxsim.score_candidates_padded(q, d, qlen, dlen)


@pytest.mark.parametrize(
    "entrypoint",
    [
        maxsim.score_candidates_padded,
        maxsim.score_candidates_padded_with_argmax,
        maxsim.score_candidates_padded_train,
    ],
)
@pytest.mark.parametrize(
    ("bad_length", "message"),
    [
        ("query_zero", "query_lengths"),
        ("query_too_long", "query_lengths"),
        ("doc_zero", "doc_lengths"),
        ("doc_too_long", "doc_lengths"),
    ],
)
def test_padded_rejects_real_lengths_out_of_bounds(
    entrypoint, bad_length, message
) -> None:
    q, d, qlen, dlen = _padded_inputs()
    if bad_length == "query_zero":
        qlen[0] = 0
    elif bad_length == "query_too_long":
        qlen[0] = q.shape[1] + 1
    elif bad_length == "doc_zero":
        dlen[0, 0] = 0
    elif bad_length == "doc_too_long":
        dlen[0, 0] = d.shape[2] + 1

    with pytest.raises(ValueError, match=message):
        entrypoint(q, d, qlen, dlen)


# ---------------------------------------------------------------------------
# Packed API
# ---------------------------------------------------------------------------

def test_packed_rejects_dim_mismatch_2d() -> None:
    q = torch.randn(10, 64, device=DEVICE, dtype=torch.float16)
    d = torch.randn(20, 32, device=DEVICE, dtype=torch.float16)
    qoff = torch.tensor([0, 10], dtype=torch.int32, device=DEVICE)
    doff = torch.tensor([0, 20], dtype=torch.int32, device=DEVICE)
    qids = torch.tensor([0], dtype=torch.int32, device=DEVICE)
    dids = torch.tensor([0], dtype=torch.int32, device=DEVICE)
    with pytest.raises((ValueError, TypeError, RuntimeError), match="dim|match"):
        maxsim.score_pairs_packed(q, qoff, d, doff, qids, dids)


def test_packed_rejects_offsets_not_starting_at_zero() -> None:
    q = torch.randn(10, 64, device=DEVICE, dtype=torch.float16)
    d = torch.randn(20, 64, device=DEVICE, dtype=torch.float16)
    qoff = torch.tensor([1, 10], dtype=torch.int32, device=DEVICE)  # bad
    doff = torch.tensor([0, 20], dtype=torch.int32, device=DEVICE)
    qids = torch.tensor([0], dtype=torch.int32, device=DEVICE)
    dids = torch.tensor([0], dtype=torch.int32, device=DEVICE)
    with pytest.raises((ValueError, RuntimeError),
                       match="0|offset|start"):
        maxsim.score_pairs_packed(q, qoff, d, doff, qids, dids)


def test_packed_rejects_offsets_not_ending_at_total() -> None:
    q = torch.randn(10, 64, device=DEVICE, dtype=torch.float16)
    d = torch.randn(20, 64, device=DEVICE, dtype=torch.float16)
    qoff = torch.tensor([0, 5], dtype=torch.int32, device=DEVICE)  # bad: ends at 5, not 10
    doff = torch.tensor([0, 20], dtype=torch.int32, device=DEVICE)
    qids = torch.tensor([0], dtype=torch.int32, device=DEVICE)
    dids = torch.tensor([0], dtype=torch.int32, device=DEVICE)
    with pytest.raises((ValueError, RuntimeError),
                       match="total|equal|offset"):
        maxsim.score_pairs_packed(q, qoff, d, doff, qids, dids)


def test_packed_rejects_negative_pair_id() -> None:
    q = torch.randn(10, 64, device=DEVICE, dtype=torch.float16)
    d = torch.randn(20, 64, device=DEVICE, dtype=torch.float16)
    qoff = torch.tensor([0, 10], dtype=torch.int32, device=DEVICE)
    doff = torch.tensor([0, 20], dtype=torch.int32, device=DEVICE)
    qids = torch.tensor([-1], dtype=torch.int32, device=DEVICE)  # bad
    dids = torch.tensor([0], dtype=torch.int32, device=DEVICE)
    with pytest.raises((ValueError, RuntimeError),
                       match="range|negative|out"):
        maxsim.score_pairs_packed(q, qoff, d, doff, qids, dids)


@pytest.mark.parametrize(
    "entrypoint",
    [
        maxsim.score_pairs_packed,
        maxsim.score_pairs_packed_with_argmax,
        maxsim.score_pairs_packed_train,
    ],
)
def test_packed_rejects_max_q_len_smaller_than_actual(entrypoint) -> None:
    q = torch.randn(13, 64, device=DEVICE, dtype=torch.float16)
    d = torch.randn(20, 64, device=DEVICE, dtype=torch.float16)
    qoff = torch.tensor([0, 5, 13], dtype=torch.int32, device=DEVICE)
    doff = torch.tensor([0, 20], dtype=torch.int32, device=DEVICE)
    qids = torch.tensor([1], dtype=torch.int32, device=DEVICE)
    dids = torch.tensor([0], dtype=torch.int32, device=DEVICE)
    with pytest.raises(ValueError, match="max_q_len"):
        entrypoint(q, qoff, d, doff, qids, dids, max_q_len=5)


# ---------------------------------------------------------------------------
# Contrastive API
# ---------------------------------------------------------------------------

def _contrastive_inputs(Nq=2, Nb=4, Lq=16, Ld=32, dim=64, dtype=torch.float16):
    q = torch.randn(Nq, Lq, dim, device=DEVICE, dtype=dtype)
    total_d = Nb * Ld
    d = torch.randn(total_d, dim, device=DEVICE, dtype=dtype)
    cu = torch.arange(0, (Nb + 1) * Ld, Ld,
                      dtype=torch.int32, device=DEVICE)
    return q, d, cu


def test_contrastive_rejects_wrong_queries_dim() -> None:
    q = torch.randn(2, 16, 64, 4, device=DEVICE, dtype=torch.float16)  # 4-D
    d = torch.randn(64, 64, device=DEVICE, dtype=torch.float16)
    cu = torch.tensor([0, 32, 64], dtype=torch.int32, device=DEVICE)
    with pytest.raises((ValueError, RuntimeError),
                       match="3-D|3D|Nq, Lq"):
        maxsim.score_contrastive(q, d, cu)


def test_contrastive_rejects_documents_wrong_rank() -> None:
    q = torch.randn(2, 16, 64, device=DEVICE, dtype=torch.float16)
    d = torch.randn(64, 64, 4, device=DEVICE, dtype=torch.float16)  # 3-D
    cu = torch.tensor([0, 32, 64], dtype=torch.int32, device=DEVICE)
    with pytest.raises((ValueError, RuntimeError),
                       match="2-D|2D|packed"):
        maxsim.score_contrastive(q, d, cu)


def test_contrastive_rejects_dim_mismatch() -> None:
    q = torch.randn(2, 16, 64, device=DEVICE, dtype=torch.float16)
    d = torch.randn(64, 32, device=DEVICE, dtype=torch.float16)
    cu = torch.tensor([0, 32, 64], dtype=torch.int32, device=DEVICE)
    with pytest.raises((ValueError, RuntimeError), match="dim|D|match"):
        maxsim.score_contrastive(q, d, cu)


def test_contrastive_rejects_dtype_mismatch() -> None:
    q = torch.randn(2, 16, 64, device=DEVICE, dtype=torch.float16)
    d = torch.randn(64, 64, device=DEVICE, dtype=torch.bfloat16)
    cu = torch.tensor([0, 32, 64], dtype=torch.int32, device=DEVICE)
    with pytest.raises((TypeError, RuntimeError), match="dtype"):
        maxsim.score_contrastive(q, d, cu)


def test_contrastive_rejects_document_offsets_not_starting_at_zero() -> None:
    q, d, _cu = _contrastive_inputs()
    bad_cu = torch.tensor([1, 33, 65, 97, 129],
                          dtype=torch.int32, device=DEVICE)
    with pytest.raises((ValueError, RuntimeError),
                       match="0|start|offset"):
        maxsim.score_contrastive(q, d, bad_cu)


def test_contrastive_rejects_document_offsets_not_monotonic() -> None:
    q, d, _cu = _contrastive_inputs(Nb=4)
    # Bad: position 2 < position 1
    bad_cu = torch.tensor([0, 32, 16, 96, 128],
                          dtype=torch.int32, device=DEVICE)
    with pytest.raises((ValueError, RuntimeError),
                       match="monoton|decreas|order|offset"):
        maxsim.score_contrastive(q, d, bad_cu)


@pytest.mark.parametrize(
    "entrypoint",
    [
        maxsim.score_contrastive,
        maxsim.score_contrastive_with_argmax,
        maxsim.score_contrastive_train,
    ],
)
def test_contrastive_rejects_empty_document_segment(entrypoint) -> None:
    q, d, _cu = _contrastive_inputs(Nb=2)
    bad_cu = torch.tensor([0, 0, d.shape[0]], dtype=torch.int32, device=DEVICE)
    with pytest.raises((ValueError, RuntimeError), match="strictly|empty|offset"):
        entrypoint(q, d, bad_cu)


def test_contrastive_rejects_document_offsets_wrong_total() -> None:
    q, d, _cu = _contrastive_inputs()
    # docs has 128 rows but document_offsets[-1] = 100
    bad_cu = torch.tensor([0, 25, 50, 75, 100],
                          dtype=torch.int32, device=DEVICE)
    with pytest.raises((ValueError, RuntimeError),
                       match="total|match|offset"):
        maxsim.score_contrastive(q, d, bad_cu)


def test_contrastive_rejects_zero_query_length() -> None:
    q, d, cu = _contrastive_inputs(Lq=16)
    q_zero = q[:, :0, :]
    with pytest.raises((ValueError, RuntimeError), match="Lq"):
        maxsim.score_contrastive(q_zero, d, cu)


def test_contrastive_rejects_zero_embedding_dim() -> None:
    q, d, cu = _contrastive_inputs(dim=64)
    q_zero = q[:, :, :0]
    d_zero = d[:, :0]
    with pytest.raises((ValueError, RuntimeError), match="dim"):
        maxsim.score_contrastive(q_zero, d_zero, cu)


# ---------------------------------------------------------------------------
# Device mismatch — applies to all APIs uniformly
# ---------------------------------------------------------------------------

def test_padded_rejects_cpu_query_device_input() -> None:
    if DEVICE.type == "cpu":
        pytest.skip("test designed for device != cpu")
    q_cpu = torch.randn(2, 16, 64, dtype=torch.float16)  # CPU
    _q, d, qlen, dlen = _padded_inputs()
    with pytest.raises((RuntimeError, ValueError),
                       match="device|share"):
        maxsim.score_candidates_padded(q_cpu, d, qlen, dlen)
