// Test: does a2=0 affect D[0]/D[1] on SM120?
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

__global__ void test(float *out) {
    int sub = threadIdx.x % 4;

    uint32_t a_nz = pk(1.0f, 1.0f);  // non-zero A
    uint32_t b_nz = pk(0.5f, 0.5f);
    float d0,d1,d2,d3;

    // Test A: both a0 and a2 non-zero (baseline)
    d0=d1=d2=d3=0;
    hmma(d0,d1,d2,d3, a_nz,a_nz,a_nz,a_nz, b_nz,b_nz, 0,0,0,0);
    out[0] = d0; out[1] = d2;

    // Test B: a0 non-zero, a2=0 → does D[0] change?
    d0=d1=d2=d3=0;
    hmma(d0,d1,d2,d3, a_nz,a_nz,0,0, b_nz,b_nz, 0,0,0,0);
    out[2] = d0; out[3] = d2;

    // Test C: a0=0, a2 non-zero → does D[2] work?
    d0=d1=d2=d3=0;
    hmma(d0,d1,d2,d3, 0,0,a_nz,a_nz, b_nz,b_nz, 0,0,0,0);
    out[4] = d0; out[5] = d2;
}

int main() {
    float *d; cudaMalloc(&d, 256);
    test<<<1,32>>>(d);
    cudaDeviceSynchronize();
    float h[6]; cudaMemcpy(h, d, 24, cudaMemcpyDeviceToHost);

    printf("Test A (both nz): D[0]=%.4f D[2]=%.4f\n", h[0], h[1]);
    printf("Test B (a2=0):    D[0]=%.4f D[2]=%.4f  (D[0] should match A)\n", h[2], h[3]);
    printf("Test C (a0=0):    D[0]=%.4f D[2]=%.4f  (D[2] should match A)\n", h[4], h[5]);

    if (h[0] == h[2]) printf("\na2=0 does NOT affect D[0] ✓\n");
    else printf("\na2=0 DOES affect D[0]! diff=%.6f ✗\n", h[0]-h[2]);

    if (h[1] == h[5]) printf("a0=0 does NOT affect D[2] ✓\n");
    else printf("a0=0 DOES affect D[2]! diff=%.6f ✗\n", h[1]-h[5]);

    cudaFree(d);
}
