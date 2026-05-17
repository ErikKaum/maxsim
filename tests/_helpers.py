"""Shared utilities for the maxsim test suite.

Centralises device selection, dtype tolerances, and helpers for constructing
random ragged / padded test inputs so the individual test files stay short.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass
from typing import List, Tuple

import torch


def _pick_device() -> torch.device:
    if platform.system() == "Darwin" and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE = _pick_device()


# Dtype tolerances chosen to absorb fp16/bf16 rounding plus the
# nondeterministic add order inside the atomic accumulator.
TOLERANCES = {
    torch.float32: dict(rtol=1e-5, atol=1e-4),
    torch.float16: dict(rtol=2e-2, atol=2e-2),
    torch.bfloat16: dict(rtol=3e-2, atol=3e-2),
}


def supported_float_dtypes() -> List[torch.dtype]:
    dtypes = [torch.float32, torch.float16]
    if DEVICE.type == "mps":
        # bf16 on Metal requires MSL 3.1 / macOS 14+. Assume modern macOS.
        dtypes.append(torch.bfloat16)
    elif DEVICE.type == "cuda":
        # bf16 native from sm_80 (Ampere) onward, which is our minimum.
        dtypes.append(torch.bfloat16)
    return dtypes


@dataclass
class PackedBatch:
    queries: torch.Tensor          # [total_q_tokens, dim]
    query_offsets: torch.Tensor    # [num_queries + 1] int32
    documents: torch.Tensor        # [total_d_tokens, dim]
    document_offsets: torch.Tensor  # [num_documents + 1] int32
    pair_query_ids: torch.Tensor   # [num_pairs] int32
    pair_document_ids: torch.Tensor  # [num_pairs] int32

    def args(self) -> Tuple[torch.Tensor, ...]:
        return (
            self.queries,
            self.query_offsets,
            self.documents,
            self.document_offsets,
            self.pair_query_ids,
            self.pair_document_ids,
        )


def make_random_packed_batch(
    *,
    num_queries: int,
    num_documents: int,
    num_pairs: int,
    dim: int,
    q_len_range: Tuple[int, int],
    d_len_range: Tuple[int, int],
    dtype: torch.dtype,
    device: torch.device = DEVICE,
    seed: int = 0,
) -> PackedBatch:
    """Build a random packed batch with ragged query/document lengths."""
    gen = torch.Generator(device="cpu").manual_seed(seed)

    q_lens = torch.randint(
        q_len_range[0], q_len_range[1] + 1, (num_queries,), generator=gen
    )
    d_lens = torch.randint(
        d_len_range[0], d_len_range[1] + 1, (num_documents,), generator=gen
    )

    total_q = int(q_lens.sum())
    total_d = int(d_lens.sum())

    queries = torch.randn(total_q, dim, generator=gen).to(device=device, dtype=dtype)
    documents = torch.randn(total_d, dim, generator=gen).to(device=device, dtype=dtype)

    query_offsets = torch.zeros(num_queries + 1, dtype=torch.int32)
    query_offsets[1:] = q_lens.to(torch.int32).cumsum(0)
    document_offsets = torch.zeros(num_documents + 1, dtype=torch.int32)
    document_offsets[1:] = d_lens.to(torch.int32).cumsum(0)

    pair_query_ids = torch.randint(
        0, num_queries, (num_pairs,), generator=gen, dtype=torch.int32
    )
    pair_document_ids = torch.randint(
        0, num_documents, (num_pairs,), generator=gen, dtype=torch.int32
    )

    return PackedBatch(
        queries=queries,
        query_offsets=query_offsets.to(device),
        documents=documents.to(device),
        document_offsets=document_offsets.to(device),
        pair_query_ids=pair_query_ids.to(device),
        pair_document_ids=pair_document_ids.to(device),
    )


def make_random_padded_batch(
    *,
    batch: int,
    candidates: int,
    q_len_max: int,
    d_len_max: int,
    dim: int,
    dtype: torch.dtype,
    device: torch.device = DEVICE,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a random padded batch with random per-row lengths."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    queries = torch.randn(batch, q_len_max, dim, generator=gen).to(
        device=device, dtype=dtype
    )
    documents = torch.randn(batch, candidates, d_len_max, dim, generator=gen).to(
        device=device, dtype=dtype
    )
    query_lengths = torch.randint(
        1, q_len_max + 1, (batch,), generator=gen, dtype=torch.int32
    ).to(device)
    doc_lengths = torch.randint(
        1, d_len_max + 1, (batch, candidates), generator=gen, dtype=torch.int32
    ).to(device)
    return queries, documents, query_lengths, doc_lengths
