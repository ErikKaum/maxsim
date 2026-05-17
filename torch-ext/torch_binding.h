#pragma once

#include <torch/torch.h>

// Packed ragged MaxSim forward.
//
//   queries           : [total_q_tokens, dim] (fp32 / fp16 / bf16)
//   query_offsets     : [num_queries + 1]    (int32 / int64)
//   documents         : [total_d_tokens, dim] (same dtype as queries)
//   document_offsets  : [num_documents + 1]  (int32 / int64)
//   pair_query_ids    : [num_pairs]          (int32 / int64)
//   pair_document_ids : [num_pairs]          (int32 / int64)
//   max_q_len         : int64. If >= 0, this is the maximum query segment
//                       length (queries.shape[0] of the largest query) and
//                       the kernel SKIPS the CPU-side validation of offsets
//                       and pair ids. If < 0, the kernel pulls the offsets
//                       back to CPU to compute the max and validate (slower;
//                       only useful for first-time / debug paths).
//
// Returns a [num_pairs] fp32 tensor with per-pair MaxSim scores.
torch::Tensor maxsim_forward(torch::Tensor queries,
                             torch::Tensor query_offsets,
                             torch::Tensor documents,
                             torch::Tensor document_offsets,
                             torch::Tensor pair_query_ids,
                             torch::Tensor pair_document_ids,
                             int64_t max_q_len);

// Padded MaxSim forward.
//
// Operates directly on padded `[B, Lq_max, D]` / `[B, C, Ld_max, D]` tensors
// (typically just reshapes of the user's data), so it avoids the
// pack/gather/cumsum/sync overhead the packed entry point pays. Pair ordering
// is the canonical row-major (b, c) -> (q_id=b, d_id=b*C+c).
//
//   queries        : [B * Lq_max, D]        (fp32 / fp16 / bf16)
//   query_lengths  : [B] int32 (on the same device as `queries`)
//   documents      : [B * C * Ld_max, D]    (same dtype as `queries`)
//   doc_lengths    : [B * C] int32          (on the same device)
//   Lq_max         : padded query length stride (rows of `queries` per b)
//   Ld_max         : padded document length stride (rows of `documents` per
//                    (b, c))
//   num_candidates : C
//
// Returns a [B * C] fp32 tensor of scores on the same device.
torch::Tensor maxsim_padded_forward(torch::Tensor queries,
                                    torch::Tensor query_lengths,
                                    torch::Tensor documents,
                                    torch::Tensor doc_lengths,
                                    int64_t Lq_max,
                                    int64_t Ld_max,
                                    int64_t num_candidates);
