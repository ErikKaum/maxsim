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

std::tuple<torch::Tensor, torch::Tensor> maxsim_padded_forward_with_argmax(
    torch::Tensor queries,
    torch::Tensor query_lengths,
    torch::Tensor documents,
    torch::Tensor doc_lengths,
    int64_t Lq_max,
    int64_t Ld_max,
    int64_t num_candidates);

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

std::tuple<torch::Tensor, torch::Tensor> maxsim_packed_forward_with_argmax(
    torch::Tensor queries,
    torch::Tensor query_offsets,
    torch::Tensor documents,
    torch::Tensor document_offsets,
    torch::Tensor pair_query_ids,
    torch::Tensor pair_document_ids,
    int64_t max_q_len);

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

torch::Tensor maxsim_contrastive_forward(torch::Tensor queries,
                                         torch::Tensor documents,
                                         torch::Tensor document_offsets);

std::tuple<torch::Tensor, torch::Tensor> maxsim_contrastive_forward_with_argmax(
    torch::Tensor queries,
    torch::Tensor documents,
    torch::Tensor document_offsets);

std::tuple<torch::Tensor, torch::Tensor> maxsim_contrastive_backward(
    torch::Tensor dscore,
    torch::Tensor queries,
    torch::Tensor documents,
    torch::Tensor document_offsets,
    torch::Tensor argmax);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("maxsim_forward", &maxsim_forward,
        "MaxSim packed forward (CUDA, dev)");
  m.def("maxsim_padded_forward", &maxsim_padded_forward,
        "MaxSim padded forward (CUDA, dev)");
  m.def("maxsim_padded_forward_with_argmax",
        &maxsim_padded_forward_with_argmax,
        "MaxSim padded forward + argmax (CUDA, dev)");
  m.def("maxsim_padded_backward", &maxsim_padded_backward,
        "MaxSim padded backward (CUDA, dev)");
  m.def("maxsim_packed_forward_with_argmax",
        &maxsim_packed_forward_with_argmax,
        "MaxSim packed forward + argmax (CUDA, dev)");
  m.def("maxsim_packed_backward", &maxsim_packed_backward,
        "MaxSim packed backward (CUDA, dev)");
  m.def("maxsim_contrastive_forward", &maxsim_contrastive_forward,
        "MaxSim contrastive forward (CUDA, dev)");
  m.def("maxsim_contrastive_forward_with_argmax",
        &maxsim_contrastive_forward_with_argmax,
        "MaxSim contrastive forward + argmax (CUDA, dev)");
  m.def("maxsim_contrastive_backward", &maxsim_contrastive_backward,
        "MaxSim contrastive backward (CUDA, dev)");
}
