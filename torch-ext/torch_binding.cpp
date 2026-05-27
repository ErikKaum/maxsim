#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def(
      "maxsim_forward("
      "Tensor queries, "
      "Tensor query_offsets, "
      "Tensor documents, "
      "Tensor document_offsets, "
      "Tensor pair_query_ids, "
      "Tensor pair_document_ids, "
      "int max_q_len"
      ") -> Tensor");
  ops.def(
      "maxsim_padded_forward("
      "Tensor queries, "
      "Tensor query_lengths, "
      "Tensor documents, "
      "Tensor doc_lengths, "
      "int Lq_max, "
      "int Ld_max, "
      "int num_candidates"
      ") -> Tensor");
  ops.def(
      "maxsim_padded_forward_with_argmax("
      "Tensor queries, "
      "Tensor query_lengths, "
      "Tensor documents, "
      "Tensor doc_lengths, "
      "int Lq_max, "
      "int Ld_max, "
      "int num_candidates"
      ") -> (Tensor, Tensor)");
  ops.def(
      "maxsim_padded_backward("
      "Tensor dscore, "
      "Tensor queries, "
      "Tensor documents, "
      "Tensor query_lengths, "
      "Tensor doc_lengths, "
      "Tensor argmax, "
      "int Lq_max, "
      "int Ld_max, "
      "int num_candidates"
      ") -> (Tensor, Tensor)");
  ops.def(
      "maxsim_packed_forward_with_argmax("
      "Tensor queries, "
      "Tensor query_offsets, "
      "Tensor documents, "
      "Tensor document_offsets, "
      "Tensor pair_query_ids, "
      "Tensor pair_document_ids, "
      "int max_q_len"
      ") -> (Tensor, Tensor)");
  ops.def(
      "maxsim_packed_backward("
      "Tensor dscore, "
      "Tensor queries, "
      "Tensor query_offsets, "
      "Tensor documents, "
      "Tensor document_offsets, "
      "Tensor pair_query_ids, "
      "Tensor pair_document_ids, "
      "Tensor argmax, "
      "int max_q_len"
      ") -> (Tensor, Tensor)");
  ops.def(
      "maxsim_contrastive_forward("
      "Tensor queries, "
      "Tensor documents, "
      "Tensor document_offsets"
      ") -> Tensor");
  ops.def(
      "maxsim_contrastive_forward_with_argmax("
      "Tensor queries, "
      "Tensor documents, "
      "Tensor document_offsets"
      ") -> (Tensor, Tensor)");
  ops.def(
      "maxsim_contrastive_backward("
      "Tensor dscore, "
      "Tensor queries, "
      "Tensor documents, "
      "Tensor document_offsets, "
      "Tensor argmax"
      ") -> (Tensor, Tensor)");
#if defined(CUDA_KERNEL) || defined(ROCM_KERNEL)
  ops.impl("maxsim_forward", torch::kCUDA, &maxsim_forward);
  ops.impl("maxsim_padded_forward", torch::kCUDA, &maxsim_padded_forward);
  ops.impl("maxsim_padded_forward_with_argmax",
           torch::kCUDA, &maxsim_padded_forward_with_argmax);
  ops.impl("maxsim_padded_backward",
           torch::kCUDA, &maxsim_padded_backward);
  ops.impl("maxsim_packed_forward_with_argmax",
           torch::kCUDA, &maxsim_packed_forward_with_argmax);
  ops.impl("maxsim_packed_backward",
           torch::kCUDA, &maxsim_packed_backward);
  ops.impl("maxsim_contrastive_forward",
           torch::kCUDA, &maxsim_contrastive_forward);
  ops.impl("maxsim_contrastive_forward_with_argmax",
           torch::kCUDA, &maxsim_contrastive_forward_with_argmax);
  ops.impl("maxsim_contrastive_backward",
           torch::kCUDA, &maxsim_contrastive_backward);
#elif defined(METAL_KERNEL)
  ops.impl("maxsim_forward", torch::kMPS, &maxsim_forward);
  ops.impl("maxsim_padded_forward", torch::kMPS, &maxsim_padded_forward);
  ops.impl("maxsim_padded_forward_with_argmax",
           torch::kMPS, &maxsim_padded_forward_with_argmax);
  ops.impl("maxsim_padded_backward",
           torch::kMPS, &maxsim_padded_backward);
  ops.impl("maxsim_packed_forward_with_argmax",
           torch::kMPS, &maxsim_packed_forward_with_argmax);
  ops.impl("maxsim_packed_backward",
           torch::kMPS, &maxsim_packed_backward);
  ops.impl("maxsim_contrastive_forward",
           torch::kMPS, &maxsim_contrastive_forward);
  ops.impl("maxsim_contrastive_forward_with_argmax",
           torch::kMPS, &maxsim_contrastive_forward_with_argmax);
  ops.impl("maxsim_contrastive_backward",
           torch::kMPS, &maxsim_contrastive_backward);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
