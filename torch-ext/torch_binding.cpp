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
#if defined(CUDA_KERNEL) || defined(ROCM_KERNEL)
  ops.impl("maxsim_forward", torch::kCUDA, &maxsim_forward);
  ops.impl("maxsim_padded_forward", torch::kCUDA, &maxsim_padded_forward);
#elif defined(METAL_KERNEL)
  ops.impl("maxsim_forward", torch::kMPS, &maxsim_forward);
  ops.impl("maxsim_padded_forward", torch::kMPS, &maxsim_padded_forward);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
