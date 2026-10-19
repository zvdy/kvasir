#include <cuda_bf16.h>
#include "common.cuh"

extern "C" __global__ void add_inplace(__half* __restrict__ x, const __half* __restrict__ y,
                                       long long n) {
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (long long)gridDim.x * blockDim.x)
        x[i] = __float2half(__half2float(x[i]) + __half2float(y[i]));
}

extern "C" __global__ void bf16_to_f16(const __nv_bfloat16* __restrict__ in,
                                       __half* __restrict__ out, long long n,
                                       unsigned int* __restrict__ bad) {
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (long long)gridDim.x * blockDim.x) {
        const float v = __bfloat162float(in[i]);
        if (!(fabsf(v) <= 65504.f)) atomicAdd(bad, 1u);
        out[i] = __float2half(v);
    }
}

extern "C" __global__ void kv_append(const __half* __restrict__ qkv, __half* __restrict__ kc,
                                     __half* __restrict__ vc, int pos0, int n_kv, int head_dim,
                                     int max_seq, int q_dim, int row_stride) {
    const int t = blockIdx.x;
    const int kv_dim = n_kv * head_dim;
    const __half* row = qkv + (long long)t * row_stride;
    for (int i = threadIdx.x; i < kv_dim; i += blockDim.x) {
        const int h = i / head_dim, d = i % head_dim;
        const long long dst = ((long long)h * max_seq + pos0 + t) * head_dim + d;
        kc[dst] = row[q_dim + i];
        vc[dst] = row[q_dim + kv_dim + i];
    }
}
