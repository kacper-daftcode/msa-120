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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("topk_select", &topk_select,
          "SM120 sparse top-16 KV block selector (FlashInfer indexerTopK)");
}
