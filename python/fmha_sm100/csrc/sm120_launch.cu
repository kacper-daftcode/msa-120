// SM120 FlashAttention unified launcher — torch extension binding
// Supports BF16 and FP8 (E4M3) paths with automatic dispatch.
// Compiles with: nvcc -gencode arch=compute_120f,code=sm_120f
// SPDX-License-Identifier: MIT

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

// Forward declarations
extern "C" __global__ void sm120_fmha_fwd_bf16(
    const __nv_bfloat16 *Q, const __nv_bfloat16 *K, const __nv_bfloat16 *V,
    __nv_bfloat16 *O, float *LSE,
    int seq_len_q, int seq_len_k, int num_heads_q, int num_heads_kv, float scale);

extern "C" __global__ void sm120_fmha_fwd_fp8(
    const __nv_fp8_e4m3 *Q, const __nv_fp8_e4m3 *K, const __nv_fp8_e4m3 *V,
    __nv_bfloat16 *O, float *LSE,
    int seq_len_q, int seq_len_k, int num_heads_q, int num_heads_kv, float scale);

// BF16 forward
std::vector<torch::Tensor> sm120_fmha_forward_bf16(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, float softmax_scale)
{
    TORCH_CHECK(q.is_cuda(), "q must be CUDA tensor");
    TORCH_CHECK(q.dtype() == torch::kBFloat16, "q must be BF16");
    TORCH_CHECK(q.dim() == 3, "q must be [total_q, Hq, D]");
    TORCH_CHECK(q.size(2) == 128, "head_dim must be 128");

    const int seq_q = q.size(0);
    const int num_heads_q = q.size(1);
    const int seq_k = k.size(0);
    const int num_heads_kv = k.size(1);

    auto o = torch::zeros_like(q);
    auto lse = torch::zeros({seq_q, num_heads_q},
                            torch::dtype(torch::kFloat32).device(q.device()));

    const int BLK_M = 64;
    const int num_m_blocks = (seq_q + BLK_M - 1) / BLK_M;
    dim3 grid(num_m_blocks, num_heads_q);
    dim3 block(128);

    const int smem_bytes = 64 * 128 * 2 + 64 * 128 * 2 * 2 + 64 * 128 * 2 * 2;

    cudaFuncSetAttribute(sm120_fmha_fwd_bf16,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);

    auto stream = at::cuda::getCurrentCUDAStream();
    sm120_fmha_fwd_bf16<<<grid, block, smem_bytes, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(k.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(o.data_ptr()),
        lse.data_ptr<float>(),
        seq_q, seq_k, num_heads_q, num_heads_kv, softmax_scale);

    return {o, lse};
}

// FP8 forward (Q/K/V as FP8, output as BF16)
std::vector<torch::Tensor> sm120_fmha_forward_fp8(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, float softmax_scale)
{
    TORCH_CHECK(q.is_cuda(), "q must be CUDA tensor");
    TORCH_CHECK(q.dtype() == torch::kFloat8_e4m3fn, "q must be float8_e4m3fn");
    TORCH_CHECK(q.dim() == 3, "q must be [total_q, Hq, D]");
    TORCH_CHECK(q.size(2) == 128, "head_dim must be 128");

    const int seq_q = q.size(0);
    const int num_heads_q = q.size(1);
    const int seq_k = k.size(0);
    const int num_heads_kv = k.size(1);

    auto o = torch::zeros({seq_q, num_heads_q, 128},
                          torch::dtype(torch::kBFloat16).device(q.device()));
    auto lse = torch::zeros({seq_q, num_heads_q},
                            torch::dtype(torch::kFloat32).device(q.device()));

    const int BLK_M = 64;
    const int num_m_blocks = (seq_q + BLK_M - 1) / BLK_M;
    dim3 grid(num_m_blocks, num_heads_q);
    dim3 block(128);

    // FP8 SMEM: Q(8KB) + K(8KB) + V(8KB) = 24KB
    const int smem_bytes = 64 * 128 * 1 + 64 * 128 * 1 * 2 + 64 * 128 * 1 * 2;

    cudaFuncSetAttribute(sm120_fmha_fwd_fp8,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);

    auto stream = at::cuda::getCurrentCUDAStream();
    sm120_fmha_fwd_fp8<<<grid, block, smem_bytes, stream>>>(
        reinterpret_cast<const __nv_fp8_e4m3*>(q.data_ptr()),
        reinterpret_cast<const __nv_fp8_e4m3*>(k.data_ptr()),
        reinterpret_cast<const __nv_fp8_e4m3*>(v.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(o.data_ptr()),
        lse.data_ptr<float>(),
        seq_q, seq_k, num_heads_q, num_heads_kv, softmax_scale);

    return {o, lse};
}

// Auto-dispatch based on dtype
std::vector<torch::Tensor> sm120_fmha_forward(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, float softmax_scale)
{
    if (q.dtype() == torch::kBFloat16) {
        return sm120_fmha_forward_bf16(q, k, v, softmax_scale);
    } else if (q.dtype() == torch::kFloat8_e4m3fn) {
        return sm120_fmha_forward_fp8(q, k, v, softmax_scale);
    } else {
        TORCH_CHECK(false, "SM120 FMHA supports BF16 and FP8_E4M3 only, got ", q.dtype());
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &sm120_fmha_forward, "SM120 FlashAttention forward (auto BF16/FP8)");
    m.def("forward_bf16", &sm120_fmha_forward_bf16, "SM120 FlashAttention forward (BF16)");
    m.def("forward_fp8", &sm120_fmha_forward_fp8, "SM120 FlashAttention forward (FP8 E4M3)");
}
