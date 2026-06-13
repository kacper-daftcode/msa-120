// Definitive SM120 HMMA m16n8k16 fragment layout discovery.
// Sets each A-register to a unique value, B to identity-like pattern,
// and reads output to determine the exact mapping.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

__device__ __forceinline__ void hmma(
    float &d0, float &d1, float &d2, float &d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1, float c0, float c1, float c2, float c3) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
        : "=f"(d0),"=f"(d1),"=f"(d2),"=f"(d3)
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),
          "f"(c0),"f"(c1),"f"(c2),"f"(c3));
}

__device__ uint32_t pk(float lo, float hi) {
    uint32_t r;
    reinterpret_cast<__nv_bfloat16*>(&r)[0] = __float2bfloat16(lo);
    reinterpret_cast<__nv_bfloat16*>(&r)[1] = __float2bfloat16(hi);
    return r;
}

// Test: set ONE A-register to non-zero (specific value encoding which reg),
// all others to zero. B=all-ones. See which D outputs are non-zero.
__global__ void probe_a_reg(float *out, int which_reg) {
    int lid = threadIdx.x % 32;
    int sub = lid % 4;

    // Only thread sub=0 provides non-zero A (to isolate k-positions)
    uint32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    if (sub == 0) {
        float tag = 1.0f;
        uint32_t val = pk(tag, tag);
        if (which_reg == 0) a0 = val;
        if (which_reg == 1) a1 = val;
        if (which_reg == 2) a2 = val;
        if (which_reg == 3) a3 = val;
    }

    // B: identity-like, only sub=0 non-zero (one k-position = B[0, grp])
    uint32_t b0 = (sub == 0) ? pk(1.0f, 0.0f) : 0;
    uint32_t b1 = 0;

    float d0=0, d1=0, d2=0, d3=0;
    hmma(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, 0, 0, 0, 0);

    // Each lane writes its 4 D values
    out[lid*4+0] = d0;
    out[lid*4+1] = d1;
    out[lid*4+2] = d2;
    out[lid*4+3] = d3;
}

// Test: ALL threads provide A, with unique values per (sub, reg)
// B = all-ones. Check how k-positions distribute.
__global__ void probe_k_distribution(float *out) {
    int lid = threadIdx.x % 32;
    int grp = lid / 4, sub = lid % 4;

    // A: each sub provides A=1.0 in one register, zero in others
    // This tells us which sub's values contribute to D
    uint32_t a0 = (sub == 0) ? pk(1.0f, 1.0f) : 0;
    uint32_t a1 = (sub == 1) ? pk(1.0f, 1.0f) : 0;
    uint32_t a2 = (sub == 2) ? pk(1.0f, 1.0f) : 0;
    uint32_t a3 = (sub == 3) ? pk(1.0f, 1.0f) : 0;

    // B: all ones
    uint32_t b0 = pk(1.0f, 1.0f);
    uint32_t b1 = pk(1.0f, 1.0f);

    float d0=0, d1=0, d2=0, d3=0;
    hmma(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, 0, 0, 0, 0);

    out[lid*4+0] = d0;
    out[lid*4+1] = d1;
    out[lid*4+2] = d2;
    out[lid*4+3] = d3;
}

int main() {
    float *d_out; cudaMalloc(&d_out, 512);
    float h[128];

    printf("=== SM120 HMMA m16n8k16 Fragment Layout Discovery ===\n\n");

    // Probe each A register separately
    for (int reg = 0; reg < 4; reg++) {
        cudaMemset(d_out, 0, 512);
        probe_a_reg<<<1, 32>>>(d_out, reg);
        cudaDeviceSynchronize();
        cudaMemcpy(h, d_out, 512, cudaMemcpyDeviceToHost);

        printf("A reg %d (a%d) non-zero at sub=0:\n", reg, reg);
        printf("  lane0 (grp=0,sub=0): D=[%.1f, %.1f, %.1f, %.1f]\n",
               h[0], h[1], h[2], h[3]);
        printf("  lane1 (grp=0,sub=1): D=[%.1f, %.1f, %.1f, %.1f]\n",
               h[4], h[5], h[6], h[7]);
        printf("  lane4 (grp=1,sub=0): D=[%.1f, %.1f, %.1f, %.1f]\n",
               h[16], h[17], h[18], h[19]);
    }

    printf("\n--- K-distribution test (each sub provides one A-reg) ---\n");
    cudaMemset(d_out, 0, 512);
    probe_k_distribution<<<1, 32>>>(d_out);
    cudaDeviceSynchronize();
    cudaMemcpy(h, d_out, 512, cudaMemcpyDeviceToHost);

    printf("Expected: D = sum of ALL A×B = should be 8.0 if all 4 regs contribute equally\n");
    for (int s = 0; s < 4; s++) {
        int lane = s;  // grp=0
        printf("  grp=0 sub=%d: D=[%.1f, %.1f, %.1f, %.1f]\n",
               s, h[lane*4], h[lane*4+1], h[lane*4+2], h[lane*4+3]);
    }

    // Key diagnostic: per-register contribution to D[0] vs D[2]
    printf("\n--- Per-register contribution to D[0] and D[2] at lane 0 ---\n");
    for (int reg = 0; reg < 4; reg++) {
        cudaMemset(d_out, 0, 512);
        probe_a_reg<<<1, 32>>>(d_out, reg);
        cudaDeviceSynchronize();
        cudaMemcpy(h, d_out, 512, cudaMemcpyDeviceToHost);
        printf("  a%d: D[0]=%.2f D[2]=%.2f\n", reg, h[0], h[2]);
    }

    cudaFree(d_out);
    return 0;
}
