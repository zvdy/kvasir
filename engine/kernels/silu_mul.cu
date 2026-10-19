#include "common.cuh"

extern "C" __global__ void silu_mul(const __half* __restrict__ gu, __half* __restrict__ out,
                                    int inter, long long n) {
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x; idx < n;
         idx += (long long)gridDim.x * blockDim.x) {
        const long long t = idx / inter, i = idx % inter;
        const float g = __half2float(gu[t * 2 * inter + i]);
        const float u = __half2float(gu[t * 2 * inter + inter + i]);
        out[idx] = __float2half(g / (1.f + __expf(-g)) * u);
    }
}
