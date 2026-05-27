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
// is fixed row-major: (b, c) -> (q_id=b, d_id=b*C+c).
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

// Forward + per-q-tok argmax position. Returns:
//   scores  : [B*C]                  fp32
//   argmax  : [B*C, Lq_max]          int32
//
// argmax[pair, i] is the document-token index in [0, Ld) that won
// max_j <q_i, d_j>. PyTorch's first-index-wins tiebreak applies; slots
// where i >= query_lengths[pair / C] are filled with 0.
//
// This is the forward saved-for-backward op: the int32 argmax buffer is
// the small dense thing we keep instead of the full [Lq, Ld] similarity
// matrix (95-205x memory reduction at typical batch sizes).
std::tuple<torch::Tensor, torch::Tensor> maxsim_padded_forward_with_argmax(
    torch::Tensor queries,
    torch::Tensor query_lengths,
    torch::Tensor documents,
    torch::Tensor doc_lengths,
    int64_t Lq_max,
    int64_t Ld_max,
    int64_t num_candidates);

// Backward pass for the padded form. Given the argmax positions saved by
// `maxsim_padded_forward_with_argmax`, routes the incoming `dscore` gradient
// back to query and document tensors:
//
//   g = dscore[pair]
//   j = argmax[pair, q]
//   dqueries[q_id, q] += g * documents[d_id, j]   (atomic, queries shared across C)
//   ddocuments[d_id, j] += g * queries[q_id, q]   (atomic, multiple q's may share j)
//
// Returns (dqueries, ddocuments) — both fp32 regardless of forward input
// dtype. Downstream can cast if needed. fp32 accumulation avoids the bf16
// atomicAdd hardware gap (only on Hopper as of CUDA 13).
std::tuple<torch::Tensor, torch::Tensor> maxsim_padded_backward(
    torch::Tensor dscore,
    torch::Tensor queries,
    torch::Tensor documents,
    torch::Tensor query_lengths,
    torch::Tensor doc_lengths,
    torch::Tensor argmax,
    int64_t Lq_max,
    int64_t Ld_max,
    int64_t num_candidates);

// Packed forward + per-q-tok argmax (training mode). Same indexing as
// `maxsim_forward` but additionally returns argmax positions per q_tok:
//   scores : [num_pairs]            fp32
//   argmax : [num_pairs, max_q_len] int32  (slots beyond this pair's Lq = 0)
std::tuple<torch::Tensor, torch::Tensor> maxsim_packed_forward_with_argmax(
    torch::Tensor queries,
    torch::Tensor query_offsets,
    torch::Tensor documents,
    torch::Tensor document_offsets,
    torch::Tensor pair_query_ids,
    torch::Tensor pair_document_ids,
    int64_t max_q_len);

// Packed backward. Routes `dscore` to (dqueries, ddocuments) via the
// argmax positions saved by the forward. fp32 grad outputs.
std::tuple<torch::Tensor, torch::Tensor> maxsim_packed_backward(
    torch::Tensor dscore,
    torch::Tensor queries,
    torch::Tensor query_offsets,
    torch::Tensor documents,
    torch::Tensor document_offsets,
    torch::Tensor pair_query_ids,
    torch::Tensor pair_document_ids,
    torch::Tensor argmax,
    int64_t max_q_len);

// Contrastive MaxSim forward.
//
// Full cross-product: every query in `queries` is scored against every doc
// in `documents`. Used for in-batch contrastive training (ColBERT-style).
//
//   queries     : [Nq, Lq, D]              (fp16 / bf16)
//   documents   : [total_d_tokens, D]      packed, same dtype as queries
//   document_offsets  : [Nb + 1] int32           CSR-style doc offsets
//
// Returns a [Nq, Nb] fp32 tensor.
torch::Tensor maxsim_contrastive_forward(torch::Tensor queries,
                                         torch::Tensor documents,
                                         torch::Tensor document_offsets);

// Contrastive forward + argmax. Returns:
//   scores : [Nq, Nb]      fp32
//   argmax : [Nq, Nb, Lq]  int32   per-q-tok winning doc-token index
//
// Same shape/dtype contract as `maxsim_padded_forward_with_argmax`. The
// argmax buffer is the small thing we save for backward; the full
// per-pair similarity matrix is never materialised.
std::tuple<torch::Tensor, torch::Tensor> maxsim_contrastive_forward_with_argmax(
    torch::Tensor queries,
    torch::Tensor documents,
    torch::Tensor document_offsets);

// Contrastive backward. Given the argmax positions from the forward and
// the incoming `dscore` gradient, routes:
//
//   g = dscore[qi, di]
//   j = argmax[qi, di, i]
//   dq[qi, i]            += g * d[d_offset(di) + j]    (atomic, q shared across Nb)
//   dd[d_offset(di) + j] += g * q[qi, i]               (atomic, d shared across Nq)
//
// Returns (dqueries, ddocuments). Both fp32 regardless of forward dtype
// (same rationale as the padded backward — sidesteps the bf16 atomicAdd
// hardware gap).
std::tuple<torch::Tensor, torch::Tensor> maxsim_contrastive_backward(
    torch::Tensor dscore,
    torch::Tensor queries,
    torch::Tensor documents,
    torch::Tensor document_offsets,
    torch::Tensor argmax);
