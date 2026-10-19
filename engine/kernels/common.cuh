#pragma once
#include <cuda_fp16.h>

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
__device__ __forceinline__ float block_sum(float v, float* scratch) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int nwarps = (blockDim.x + 31) >> 5;
    v = warp_sum(v);
    __syncthreads();  // protect scratch reuse across consecutive reductions
    if (lane == 0) scratch[warp] = v;
    __syncthreads();
    float r = (lane < nwarps) ? scratch[lane] : 0.f;
    return warp_sum(r);
}
__device__ __forceinline__ float block_max(float v, float* scratch) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int nwarps = (blockDim.x + 31) >> 5;
    v = warp_max(v);
    __syncthreads();
    if (lane == 0) scratch[warp] = v;
    __syncthreads();
    float r = (lane < nwarps) ? scratch[lane] : -INFINITY;
    return warp_max(r);
}
