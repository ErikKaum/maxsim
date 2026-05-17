// pybind11 binding used ONLY by `scripts/cuda_dev.py` via
// `torch.utils.cpp_extension.load()`. The production build path goes
// through `torch-ext/torch_binding.{h,cpp}` + kernel-builder's
// `registration.h`, which we don't have at dev time. Keeping the dev
// binding minimal and decoupled means we can iterate on `maxsim.cu`
// without touching kernel-builder.

#include <torch/extension.h>

torch::Tensor maxsim_forward(torch::Tensor queries,
                             torch::Tensor query_offsets,
                             torch::Tensor documents,
                             torch::Tensor document_offsets,
                             torch::Tensor pair_query_ids,
                             torch::Tensor pair_document_ids,
                             int64_t max_q_len);

torch::Tensor maxsim_padded_forward(torch::Tensor queries,
                                    torch::Tensor query_lengths,
                                    torch::Tensor documents,
                                    torch::Tensor doc_lengths,
                                    int64_t Lq_max,
                                    int64_t Ld_max,
                                    int64_t num_candidates);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("maxsim_forward", &maxsim_forward,
        "MaxSim packed forward (CUDA, dev)");
  m.def("maxsim_padded_forward", &maxsim_padded_forward,
        "MaxSim padded forward (CUDA, dev)");
}
