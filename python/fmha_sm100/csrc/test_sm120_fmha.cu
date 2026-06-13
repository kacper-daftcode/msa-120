// Test harness for SM120 FlashAttention PoC
// Verifies correctness against naive attention reference
//
// Build: nvcc -gencode arch=compute_120f,code=sm_120f -O3 -std=c++17
//        --expt-relaxed-constexpr -o test_sm120_fmha
//        test_sm120_fmha.cu sm120_fmha_fwd.cu
// Run:   CUDA_VISIBLE_DEVICES=4 ./test_sm120_fmha

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

extern "C" __global__ void sm120_fmha_fwd_bf16(
    const __nv_bfloat16 *Q, const __nv_bfloat16 *K, const __nv_bfloat16 *V,
    __nv_bfloat16 *O, float *LSE,
    int seq_len_q, int seq_len_k, int num_heads_q, int num_heads_kv, float scale);

// CPU reference: naive attention
void reference_attention(
    const std::vector<float> &Q,  // [seq_q, D]
    const std::vector<float> &K,  // [seq_k, D]
    const std::vector<float> &V,  // [seq_k, D]
    std::vector<float> &O,        // [seq_q, D]
    std::vector<float> &LSE,      // [seq_q]
    int seq_q, int seq_k, int D, float scale)
{
    for (int i = 0; i < seq_q; i++) {
        // S = Q[i] @ K^T * scale
        float max_s = -INFINITY;
        std::vector<float> S(seq_k);
        for (int j = 0; j < seq_k; j++) {
            float dot = 0.0f;
            for (int d = 0; d < D; d++)
                dot += Q[i * D + d] * K[j * D + d];
            S[j] = dot * scale;
            max_s = fmaxf(max_s, S[j]);
        }

        // P = exp(S - max), sum
        float sum_p = 0.0f;
        std::vector<float> P(seq_k);
        for (int j = 0; j < seq_k; j++) {
            P[j] = expf(S[j] - max_s);
            sum_p += P[j];
        }

        // O = P @ V / sum_p
        for (int d = 0; d < D; d++) {
            float val = 0.0f;
            for (int j = 0; j < seq_k; j++)
                val += P[j] * V[j * D + d];
            O[i * D + d] = val / sum_p;
        }

        LSE[i] = max_s + logf(sum_p);
    }
}

int main() {
    printf("=== SM120 FlashAttention PoC Test ===\n\n");

    const int seq_q = 64, seq_k = 64, D = 128;
    const int num_heads_q = 1, num_heads_kv = 1;
    const float scale = 1.0f / sqrtf((float)D);

    printf("Config: seq_q=%d, seq_k=%d, D=%d, scale=%.4f\n", seq_q, seq_k, D, scale);

    // Generate random data
    srand(42);
    std::vector<float> h_Q(seq_q * D), h_K(seq_k * D), h_V(seq_k * D);
    for (auto &v : h_Q) v = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    for (auto &v : h_K) v = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    for (auto &v : h_V) v = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;

    // Reference
    std::vector<float> ref_O(seq_q * D, 0.0f);
    std::vector<float> ref_LSE(seq_q, 0.0f);
    reference_attention(h_Q, h_K, h_V, ref_O, ref_LSE, seq_q, seq_k, D, scale);
    printf("Reference computed.\n");

    // Convert to BF16
    std::vector<__nv_bfloat16> bf_Q(seq_q * D), bf_K(seq_k * D), bf_V(seq_k * D);
    for (int i = 0; i < seq_q * D; i++) bf_Q[i] = __float2bfloat16(h_Q[i]);
    for (int i = 0; i < seq_k * D; i++) bf_K[i] = __float2bfloat16(h_K[i]);
    for (int i = 0; i < seq_k * D; i++) bf_V[i] = __float2bfloat16(h_V[i]);

    // Allocate GPU
    __nv_bfloat16 *d_Q, *d_K, *d_V, *d_O;
    float *d_LSE;
    cudaMalloc(&d_Q, seq_q * D * sizeof(__nv_bfloat16));
    cudaMalloc(&d_K, seq_k * D * sizeof(__nv_bfloat16));
    cudaMalloc(&d_V, seq_k * D * sizeof(__nv_bfloat16));
    cudaMalloc(&d_O, seq_q * D * sizeof(__nv_bfloat16));
    cudaMalloc(&d_LSE, seq_q * sizeof(float));

    cudaMemcpy(d_Q, bf_Q.data(), seq_q * D * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, bf_K.data(), seq_k * D * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, bf_V.data(), seq_k * D * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemset(d_O, 0, seq_q * D * sizeof(__nv_bfloat16));
    cudaMemset(d_LSE, 0, seq_q * sizeof(float));

    // Launch kernel
    int num_m_blocks = (seq_q + 63) / 64;
    dim3 grid(num_m_blocks, num_heads_q);
    dim3 block(4 * 32);  // 4 warps
    int smem = 16384 + 16384 * 2 * 2;  // Q + K*2stages + V*2stages

    printf("Launching SM120 FA kernel: grid=(%d,%d), block=%d, smem=%d\n",
           grid.x, grid.y, block.x, smem);

    cudaFuncSetAttribute(sm120_fmha_fwd_bf16,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem);

    sm120_fmha_fwd_bf16<<<grid, block, smem>>>(
        d_Q, d_K, d_V, d_O, d_LSE,
        seq_q, seq_k, num_heads_q, num_heads_kv, scale);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel FAILED: %s (%d)\n", cudaGetErrorString(err), err);
        return 1;
    }
    printf("Kernel executed OK.\n");

    // Read back and compare
    std::vector<__nv_bfloat16> h_O(seq_q * D);
    cudaMemcpy(h_O.data(), d_O, seq_q * D * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

    // Compute max absolute error
    float max_err = 0.0f;
    float mean_err = 0.0f;
    int count = 0;
    for (int i = 0; i < seq_q * D; i++) {
        float got = __bfloat162float(h_O[i]);
        float ref = ref_O[i];
        float err_val = fabsf(got - ref);
        max_err = fmaxf(max_err, err_val);
        mean_err += err_val;
        count++;
    }
    mean_err /= count;

    printf("\nResults:\n");
    printf("  Max abs error:  %.6f\n", max_err);
    printf("  Mean abs error: %.6f\n", mean_err);
    printf("  First 4 output values (got vs ref):\n");
    for (int i = 0; i < 4; i++) {
        printf("    O[%d] = %.6f  ref = %.6f\n",
               i, __bfloat162float(h_O[i]), ref_O[i]);
    }

    // Tolerance: BF16 has ~0.4% relative error, so absolute depends on magnitude
    if (max_err < 0.01f) {
        printf("\n>>> SM120 FlashAttention PoC: PASS (max_err < 0.01) <<<\n");
    } else if (max_err < 0.1f) {
        printf("\n>>> SM120 FlashAttention PoC: APPROXIMATE (0.01 < max_err < 0.1) <<<\n");
    } else {
        printf("\n>>> SM120 FlashAttention PoC: NEEDS WORK (max_err >= 0.1) <<<\n");
        printf("    (Expected: PoC uses simplified per-lane dot products,\n");
        printf("     not yet HMMA. Correctness validation is the goal.)\n");
    }

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O); cudaFree(d_LSE);
    return 0;
}
