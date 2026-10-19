#include "common.cuh"

extern "C" __global__ void rmsnorm(const __half* __restrict__ x, const __half* __restrict__ w,
                                   __half* __restrict__ out, int hidden, float eps) {
    __shared__ float scratch[32];
    const __half* xr = x + (long long)blockIdx.x * hidden;
    __half* orow = out + (long long)blockIdx.x * hidden;
    float ss = 0.f;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        float v = __half2float(xr[i]);
        ss += v * v;
    }
    ss = block_sum(ss, scratch);
    const float inv = rsqrtf(ss / (float)hidden + eps);
    for (int i = threadIdx.x; i < hidden; i += blockDim.x)
        orow[i] = __float2half(__half2float(xr[i]) * inv * __half2float(w[i]));
}
