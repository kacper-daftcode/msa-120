// SM120 MSA "max-pool block-score" kernel.
//
// Computes, for the top-k block selector (sparse_topk_select), the per-block
// MAX of q.k dot products:
//
//   max_score[h, blk, qi] = MAX over kv in
//       [blk*block_size, min((blk+1)*block_size, seq_k))
//     of dot(q[qi, h, :], k[kv, hkv, :]) * softmax_scale
//
// Inputs:
//   q     : [seq_q, Hq,  128] bf16
//   k     : [seq_k, Hkv, 128] bf16
//   scale : float (softmax_scale)
//   block_size : KV block size (64 or 128)
//
// Output:
//   max_score : float32 [Hq, num_kv_blocks, seq_q]
//               num_kv_blocks = ceil(seq_k / block_size)
//
// This [H, K_tiles, Q] layout is exactly what sparse_topk_select consumes.
//
// GQA: Hq >= Hkv, hkv = hq / (Hq / Hkv).
//
// Design (correctness-first v1):
//   - One CTA per (h, qi) row.  HEAD_DIM (128) threads per CTA.
//   - Each thread t loads q[qi, h, t] once into a register.
//   - Loop over all kv in [0, seq_k): each thread computes the partial product
//     q[t]*k[kv, hkv, t], a block reduction (warp shuffle + smem) yields the
//     full 128-d dot, scaled.  Thread 0 max-accumulates it into the running
//     max for the kv-block that kv belongs to.
//   - At the end the per-block running maxes are written out.
//
// SPDX-License-Identifier: MIT

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>

#include <cfloat>
#include <cstdint>
#include <vector>

namespace {

constexpr int HEAD_DIM = 128;
constexpr int WARP_SIZE = 32;
constexpr int NUM_WARPS = HEAD_DIM / WARP_SIZE;  // 4

// One CTA per (h, qi).  blockDim = HEAD_DIM (128) threads.
// Grid: (seq_q, Hq).
__global__ void block_max_score_kernel(
    const __nv_bfloat16* __restrict__ q,   // [seq_q, Hq, 128]
    const __nv_bfloat16* __restrict__ k,   // [seq_k, Hkv, 128]
    float* __restrict__ out,               // [Hq, num_kv_blocks, seq_q]
    int seq_q, int seq_k, int Hq, int Hkv,
    int block_size, int num_kv_blocks,
    float scale) {
  const int qi = blockIdx.x;
  const int h = blockIdx.y;
  const int tid = threadIdx.x;        // 0..127, the head-dim index
  const int warp_id = tid / WARP_SIZE;
  const int lane = tid % WARP_SIZE;

  const int group = Hq / Hkv;
  const int hkv = h / group;

  // Load this thread's q element once.
  const float q_val =
      __bfloat162float(q[(static_cast<size_t>(qi) * Hq + h) * HEAD_DIM + tid]);

  __shared__ float warp_partial[NUM_WARPS];
  // Running max per kv-block, held by thread 0.
  // num_kv_blocks can be large; we store running maxes in shared memory.
  extern __shared__ float block_max[];  // size = num_kv_blocks floats
  for (int b = tid; b < num_kv_blocks; b += blockDim.x) {
    block_max[b] = -FLT_MAX;
  }
  __syncthreads();

  for (int kv = 0; kv < seq_k; ++kv) {
    const float k_val =
        __bfloat162float(k[(static_cast<size_t>(kv) * Hkv + hkv) * HEAD_DIM + tid]);
    float prod = q_val * k_val;

    // Warp reduction.
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
      prod += __shfl_down_sync(0xffffffffu, prod, offset);
    }
    if (lane == 0) warp_partial[warp_id] = prod;
    __syncthreads();

    if (tid == 0) {
      float dot = 0.0f;
#pragma unroll
      for (int w = 0; w < NUM_WARPS; ++w) dot += warp_partial[w];
      dot *= scale;
      const int blk = kv / block_size;
      if (dot > block_max[blk]) block_max[blk] = dot;
    }
    __syncthreads();
  }

  // Write out: out[h, blk, qi].
  for (int b = tid; b < num_kv_blocks; b += blockDim.x) {
    out[(static_cast<size_t>(h) * num_kv_blocks + b) * seq_q + qi] = block_max[b];
  }
}

}  // namespace

torch::Tensor block_max_score(torch::Tensor q, torch::Tensor k, double scale,
                              int64_t block_size) {
  TORCH_CHECK(q.is_cuda(), "q must be CUDA");
  TORCH_CHECK(k.is_cuda(), "k must be CUDA");
  TORCH_CHECK(q.dtype() == torch::kBFloat16, "q must be bf16");
  TORCH_CHECK(k.dtype() == torch::kBFloat16, "k must be bf16");
  TORCH_CHECK(q.dim() == 3 && k.dim() == 3, "q,k must be 3-D");
  TORCH_CHECK(q.size(2) == HEAD_DIM, "head_dim must be 128");
  TORCH_CHECK(k.size(2) == HEAD_DIM, "head_dim must be 128");
  TORCH_CHECK(block_size == 64 || block_size == 128, "block_size must be 64 or 128");

  q = q.contiguous();
  k = k.contiguous();

  const int seq_q = q.size(0);
  const int Hq = q.size(1);
  const int seq_k = k.size(0);
  const int Hkv = k.size(1);
  TORCH_CHECK(Hq % Hkv == 0, "Hq must be divisible by Hkv (GQA)");

  const int bs = static_cast<int>(block_size);
  const int num_kv_blocks = (seq_k + bs - 1) / bs;

  auto out = torch::empty({Hq, num_kv_blocks, seq_q},
                          torch::dtype(torch::kFloat32).device(q.device()));

  if (seq_q == 0 || num_kv_blocks == 0) return out;

  dim3 grid(seq_q, Hq);
  dim3 block(HEAD_DIM);
  const size_t smem = static_cast<size_t>(num_kv_blocks) * sizeof(float);

  auto stream = at::cuda::getCurrentCUDAStream();
  block_max_score_kernel<<<grid, block, smem, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(k.data_ptr()),
      out.data_ptr<float>(), seq_q, seq_k, Hq, Hkv, bs, num_kv_blocks,
      static_cast<float>(scale));

  C10_CUDA_CHECK(cudaGetLastError());
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("block_max_score", &block_max_score,
        "SM120 MSA max-pool block-score (bf16 q.k -> [Hq, nblk, seq_q] fp32)",
        pybind11::arg("q"), pybind11::arg("k"), pybind11::arg("scale"),
        pybind11::arg("block_size"));
}
