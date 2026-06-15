// SM120 wrapper for the portable sparse_topk_select indexer (plain CUDA + cub).
// Bypasses the tvm-ffi build layer so it loads via torch.utils.cpp_extension.
// SPDX-License-Identifier: MIT
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include "include/sparse_topk_select.cuh"

using flashinfer::sparse_topk::SparseTopKSelect;
using flashinfer::sparse_topk::SparseTopKWorkspaceSize;

// max_score: [num_qo_heads, max_k_tiles, total_qo_len] fp32 (contiguous)
// returns  : [total_qo_len, num_qo_heads, 16] int32 (block ids, asc; -1 = pad)
torch::Tensor topk_select(torch::Tensor max_score, int64_t num_valid_pages,
                          int64_t force_begin, int64_t force_end) {
    TORCH_CHECK(max_score.is_cuda(), "max_score must be CUDA");
    TORCH_CHECK(max_score.dtype() == torch::kFloat32, "max_score must be fp32");
    TORCH_CHECK(max_score.dim() == 3, "max_score must be [H, K_tiles, Q]");
    max_score = max_score.contiguous();

    const int num_qo_heads = max_score.size(0);
    const int max_k_tiles  = max_score.size(1);
    const int total_qo_len = max_score.size(2);
    const int topk = 16;

    auto out = torch::empty({total_qo_len, num_qo_heads, topk},
                            torch::dtype(torch::kInt32).device(max_score.device()));
    const size_t ws = SparseTopKWorkspaceSize(total_qo_len, num_qo_heads, max_k_tiles);
    auto wsbuf = torch::empty({(int64_t)ws},
                              torch::dtype(torch::kInt32).device(max_score.device()));

    auto stream = at::cuda::getCurrentCUDAStream();
    cudaError_t st = SparseTopKSelect(
        max_score.data_ptr<float>(), out.data_ptr<int32_t>(), wsbuf.data_ptr<int32_t>(),
        (uint32_t)total_qo_len, (uint32_t)num_qo_heads, (uint32_t)max_k_tiles,
        (uint32_t)num_valid_pages, (uint32_t)force_begin, (uint32_t)force_end, stream);
    TORCH_CHECK(st == cudaSuccess, "SparseTopKSelect failed: ", cudaGetErrorString(st));
    return out;
}

// VARLEN: per-query num_valid. num_valid is an int32 CUDA tensor of length
// total_qo_len (one entry per query token, indexed by the query axis). Each
// query's OOB clamp and local-block forcing use ITS OWN num_valid, so the op is
// set-exact vs the Triton per-query topk on a MIXED-seq-length decode/prefill
// batch. The caller MUST pre-fill score slots beyond each query's causal range
// with -inf so non-forced out-of-range blocks never out-score real ones.
// max_score: [num_qo_heads, max_k_tiles, total_qo_len] fp32 (contiguous)
// num_valid: [total_qo_len] int32 CUDA
// returns  : [total_qo_len, num_qo_heads, 16] int32 (block ids, asc; -1 = pad)
torch::Tensor topk_select_varlen(torch::Tensor max_score, torch::Tensor num_valid,
                                 int64_t force_begin, int64_t force_end) {
    TORCH_CHECK(max_score.is_cuda(), "max_score must be CUDA");
    TORCH_CHECK(max_score.dtype() == torch::kFloat32, "max_score must be fp32");
    TORCH_CHECK(max_score.dim() == 3, "max_score must be [H, K_tiles, Q]");
    TORCH_CHECK(num_valid.is_cuda() && num_valid.dtype() == torch::kInt32,
                "num_valid must be int32 CUDA");
    max_score = max_score.contiguous();
    num_valid = num_valid.contiguous();

    const int num_qo_heads = max_score.size(0);
    const int max_k_tiles  = max_score.size(1);
    const int total_qo_len = max_score.size(2);
    const int topk = 16;
    TORCH_CHECK(num_valid.numel() == total_qo_len,
                "num_valid length must equal total_qo_len (Q)");

    auto out = torch::empty({total_qo_len, num_qo_heads, topk},
                            torch::dtype(torch::kInt32).device(max_score.device()));
    const size_t ws = SparseTopKWorkspaceSize(total_qo_len, num_qo_heads, max_k_tiles);
    auto wsbuf = torch::empty({(int64_t)ws},
                              torch::dtype(torch::kInt32).device(max_score.device()));

    auto stream = at::cuda::getCurrentCUDAStream();
    // scalar num_valid_pages passed as max_k_tiles (disabled); per-query nv wins.
    cudaError_t st = SparseTopKSelect(
        max_score.data_ptr<float>(), out.data_ptr<int32_t>(), wsbuf.data_ptr<int32_t>(),
        (uint32_t)total_qo_len, (uint32_t)num_qo_heads, (uint32_t)max_k_tiles,
        (uint32_t)max_k_tiles, (uint32_t)force_begin, (uint32_t)force_end, stream,
        num_valid.data_ptr<int32_t>());
    TORCH_CHECK(st == cudaSuccess, "SparseTopKSelect(varlen) failed: ",
                cudaGetErrorString(st));
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("topk_select", &topk_select,
          "SM120 sparse top-16 KV block selector (FlashInfer indexerTopK)");
    m.def("topk_select_varlen", &topk_select_varlen,
          "SM120 sparse top-16 KV block selector, per-query num_valid (varlen)");
}
