// SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
// SPDX-License-Identifier: MIT
//
// SM120 (RTX PRO 6000, compute_120f) CUDA kernel reproducing the MiniMax-M3
// learned "lightning indexer" block-score output -- see
// python/fmha_sm100/indexer_ref.py (verified torch reference) and
// docs/M3_INDEXER_SPEC.md.
//
// Pipeline (correctness-first, no HMMA):
//   1. Project hidden states X[N,hidden] with q_proj[H*d,hidden] / k_proj[d,hidden].
//   2. Gemma RMSNorm per head over d (gain = 1 + w), computed in fp32.
//   3. Partial NeoX (split-half) RoPE on the first rotary_dim channels.
//   4. Score S[r,i,j] = scale * (q_idx[i,r] . k_idx[j,0]) with single shared key head.
//   5. Causal block max-pool over block_size key tiles -> max_score[H, nblk, N].
//
// Two kernels:
//   project_norm_rope_kernel: X -> q_idx[N,H,d], k_idx[N,d]   (one block per row*head)
//   block_score_kernel:       q_idx,k_idx -> max_score[H,nblk,N] (one block per head*query)

#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

#include <cmath>

namespace {

constexpr int kWarpSize = 32;

__device__ __forceinline__ float warp_reduce_max(float v) {
#pragma unroll
  for (int off = kWarpSize / 2; off > 0; off >>= 1)
    v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, off));
  return v;
}

// One CUDA block computes one (row n, head h) projected/normed/rope'd vector of
// length head_dim. blockDim.x == head_dim (e.g. 128); each thread owns one
// output channel c. For q: head index h in [0,H). For k: H==1 (single head).
//
// Inputs are bf16 (or fp32) hidden states X[N, hidden] and weight W[H*d, hidden].
template <typename scalar_t>
__global__ void project_norm_rope_kernel(
    const scalar_t* __restrict__ X,     // [N, hidden]
    const scalar_t* __restrict__ W,     // [H*d, hidden]  (row-major, row = h*d + c)
    const scalar_t* __restrict__ norm,  // [d]   Gemma RMSNorm gain
    const int64_t* __restrict__ positions,  // [N]
    float* __restrict__ out,            // [N, H, d]  fp32
    int N, int hidden, int H, int d,
    int rotary_dim, float theta, float eps,
    bool apply_rope) {
  const int row_head = blockIdx.x;       // n * H + h
  const int n = row_head / H;
  const int h = row_head % H;
  const int c = threadIdx.x;             // channel, 0..d-1
  if (n >= N || c >= d) return;

  const scalar_t* xrow = X + static_cast<int64_t>(n) * hidden;
  const scalar_t* wrow = W + (static_cast<int64_t>(h) * d + c) * hidden;

  // Dot product over hidden dimension: out_c = sum_k X[n,k] * W[h*d+c, k].
  // Each thread does its own channel's full reduction (hidden up to 6144).
  float acc = 0.f;
  for (int k = 0; k < hidden; ++k) {
    acc += static_cast<float>(xrow[k]) * static_cast<float>(wrow[k]);
  }

  // Gemma RMSNorm over d: need mean of squares across all d channels.
  // Reduce sum of squares across the block via a shared-memory tree.
  extern __shared__ float smem[];  // size d (d is a power of two, e.g. 128)
  smem[c] = acc * acc;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (c < stride) smem[c] += smem[c + stride];
    __syncthreads();
  }
  float ssum = smem[0];
  __syncthreads();

  float mean_sq = ssum / static_cast<float>(d);
  float inv_rms = rsqrtf(mean_sq + eps);
  float normed = acc * inv_rms * (1.0f + static_cast<float>(norm[c]));

  // Store all normed channels into shared so RoPE can pair-rotate (NeoX split-half).
  smem[c] = normed;
  __syncthreads();

  float result;

  if (apply_rope && c < rotary_dim) {
    const int half = rotary_dim / 2;
    long pos = positions[n];
    // inv_freq index: NeoX pairs channel c (c<half) with c+half.
    int fi = (c < half) ? c : (c - half);
    float ang = static_cast<float>(pos) *
                (1.0f / powf(theta, static_cast<float>(fi) / static_cast<float>(half)));
    float cosv = cosf(ang);
    float sinv = sinf(ang);
    float x1 = smem[fi];          // channel fi          (first half)
    float x2 = smem[fi + half];   // channel fi + half   (second half)
    if (c < half) {
      result = x1 * cosv - x2 * sinv;
    } else {
      result = x2 * cosv + x1 * sinv;
    }
  } else {
    result = normed;
  }

  out[(static_cast<int64_t>(n) * H + h) * d + c] = result;
}

// Score + causal block max-pool.
// One CUDA block per (head h, query i). Threads cooperatively scan key rows j.
// max_score[h, b, i] = max_{j in block b, j<=i} scale * (q_idx[i,h] . k_idx[j]).
__global__ void block_score_kernel(
    const float* __restrict__ q_idx,    // [N, H, d]
    const float* __restrict__ k_idx,    // [N, 1, d] == [N, d]
    const int64_t* __restrict__ positions,  // [N]
    float* __restrict__ max_score,      // [H, nblk, N]
    int N, int H, int d, int block_size, int nblk,
    float scale, bool causal) {
  const int h = blockIdx.x;
  const int i = blockIdx.y;
  if (h >= H || i >= N) return;

  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;
  const int lane = tid % kWarpSize;
  const int warp = tid / kWarpSize;
  const int nwarps = nthreads / kWarpSize;

  // Load q_idx[i,h,:] into shared memory for reuse across all j.
  extern __shared__ float qsh[];  // size d  (+ nwarps scratch after)
  for (int c = tid; c < d; c += nthreads) {
    qsh[c] = q_idx[(static_cast<int64_t>(i) * H + h) * d + c];
  }
  __syncthreads();

  long pos_i = causal ? positions[i] : 0;

  // Per-block running max in registers; init -inf.
  // We iterate over key rows j; thread maps via tid striding over j, each thread
  // computes the full dot product q.k for its j (d up to 128), then we reduce
  // the max within each kv-block.
  // To keep it simple and correct, loop blocks then loop j in block with thread
  // striding; reduce max via shared.
  float* wred = qsh + d;  // nwarps scratch

  for (int b = 0; b < nblk; ++b) {
    int jstart = b * block_size;
    int jend = jstart + block_size;
    if (jend > N) jend = N;

    float local_max = -INFINITY;
    for (int j = jstart + tid; j < jend; j += nthreads) {
      bool keep = true;
      if (causal) {
        long pos_j = positions[j];
        keep = (pos_j <= pos_i);
      }
      if (keep) {
        const float* krow = k_idx + static_cast<int64_t>(j) * d;
        float dot = 0.f;
#pragma unroll 4
        for (int c = 0; c < d; ++c) dot += qsh[c] * krow[c];
        dot *= scale;
        local_max = fmaxf(local_max, dot);
      }
    }
    // reduce max across block
    float wmax = warp_reduce_max(local_max);
    if (lane == 0) wred[warp] = wmax;
    __syncthreads();
    if (tid == 0) {
      float bmax = -INFINITY;
      for (int w = 0; w < nwarps; ++w) bmax = fmaxf(bmax, wred[w]);
      max_score[(static_cast<int64_t>(h) * nblk + b) * N + i] = bmax;
    }
    __syncthreads();
  }
}

template <typename scalar_t>
void launch_project(const torch::Tensor& X, const torch::Tensor& W,
                    const torch::Tensor& norm, const torch::Tensor& positions,
                    torch::Tensor& out, int N, int hidden, int H, int d,
                    int rotary_dim, float theta, float eps, bool apply_rope,
                    cudaStream_t stream) {
  dim3 grid(N * H);
  dim3 block(d);
  size_t shmem = d * sizeof(float);
  project_norm_rope_kernel<scalar_t><<<grid, block, shmem, stream>>>(
      X.data_ptr<scalar_t>(), W.data_ptr<scalar_t>(), norm.data_ptr<scalar_t>(),
      positions.data_ptr<int64_t>(), out.data_ptr<float>(),
      N, hidden, H, d, rotary_dim, theta, eps, apply_rope);
}

}  // namespace

// Full pipeline: hidden states -> max_score[H, nblk, N].
// q, k are the same hidden states [N, hidden] (single shared key head).
torch::Tensor sm120_indexer_block_scores(
    torch::Tensor q,          // [N, hidden]  hidden states feeding indexer
    torch::Tensor k,          // [N, hidden]
    torch::Tensor q_proj,     // [H*d, hidden]
    torch::Tensor k_proj,     // [d, hidden]
    torch::Tensor q_norm,     // [d]
    torch::Tensor k_norm,     // [d]
    torch::Tensor positions,  // [N] int64
    int64_t block_size,
    int64_t n_heads,
    int64_t head_dim,
    double scale,
    int64_t rotary_dim,
    double rope_theta,
    double eps,
    bool causal,
    bool apply_rope) {
  TORCH_CHECK(q.is_cuda(), "q must be CUDA");
  TORCH_CHECK(q.dim() == 2, "q must be [N, hidden]");
  const int N = q.size(0);
  const int hidden = q.size(1);
  const int H = n_heads;
  const int d = head_dim;
  TORCH_CHECK(q_proj.size(0) == H * d && q_proj.size(1) == hidden, "q_proj shape");
  TORCH_CHECK(k_proj.size(0) == d && k_proj.size(1) == hidden, "k_proj shape");
  TORCH_CHECK(positions.dtype() == torch::kInt64, "positions must be int64");
  TORCH_CHECK(d <= 1024, "head_dim must be <= 1024 for this kernel");

  auto stream = at::cuda::getCurrentCUDAStream();

  auto fopt = torch::TensorOptions().dtype(torch::kFloat32).device(q.device());
  auto q_idx = torch::empty({N, H, d}, fopt);   // [N,H,d]
  auto k_idx = torch::empty({N, 1, d}, fopt);   // [N,1,d]

  // Make weights/inputs contiguous and same dtype.
  q = q.contiguous();
  k = k.contiguous();
  q_proj = q_proj.contiguous();
  k_proj = k_proj.contiguous();
  q_norm = q_norm.contiguous();
  k_norm = k_norm.contiguous();
  positions = positions.contiguous();

  AT_DISPATCH_SWITCH(
      q.scalar_type(), "sm120_indexer_project",
      AT_DISPATCH_CASE(torch::kBFloat16, [&] {
        launch_project<scalar_t>(q, q_proj, q_norm, positions, q_idx, N, hidden,
                                 H, d, rotary_dim, (float)rope_theta, (float)eps,
                                 apply_rope, stream);
        launch_project<scalar_t>(k, k_proj, k_norm, positions, k_idx, N, hidden,
                                 1, d, rotary_dim, (float)rope_theta, (float)eps,
                                 apply_rope, stream);
      })
      AT_DISPATCH_CASE(torch::kFloat, [&] {
        launch_project<scalar_t>(q, q_proj, q_norm, positions, q_idx, N, hidden,
                                 H, d, rotary_dim, (float)rope_theta, (float)eps,
                                 apply_rope, stream);
        launch_project<scalar_t>(k, k_proj, k_norm, positions, k_idx, N, hidden,
                                 1, d, rotary_dim, (float)rope_theta, (float)eps,
                                 apply_rope, stream);
      })
      AT_DISPATCH_CASE(torch::kHalf, [&] {
        launch_project<scalar_t>(q, q_proj, q_norm, positions, q_idx, N, hidden,
                                 H, d, rotary_dim, (float)rope_theta, (float)eps,
                                 apply_rope, stream);
        launch_project<scalar_t>(k, k_proj, k_norm, positions, k_idx, N, hidden,
                                 1, d, rotary_dim, (float)rope_theta, (float)eps,
                                 apply_rope, stream);
      }));

  const int nblk = (N + block_size - 1) / block_size;
  auto max_score = torch::full({H, nblk, N}, -std::numeric_limits<float>::infinity(), fopt);

  const int score_threads = 128;
  const int nwarps = score_threads / kWarpSize;
  dim3 sgrid(H, N);
  dim3 sblock(score_threads);
  size_t sshmem = (d + nwarps) * sizeof(float);
  block_score_kernel<<<sgrid, sblock, sshmem, stream>>>(
      q_idx.data_ptr<float>(), k_idx.data_ptr<float>(),
      positions.data_ptr<int64_t>(), max_score.data_ptr<float>(),
      N, H, d, block_size, nblk, (float)scale, causal);

  C10_CUDA_CHECK(cudaGetLastError());
  return max_score;  // [H, nblk, N]
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("block_scores", &sm120_indexer_block_scores,
        "M3 indexer block scores (SM120), full pipeline from hidden states");
}
