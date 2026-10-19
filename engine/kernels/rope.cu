#include "common.cuh"

extern "C" __global__ void rope_inplace(__half* __restrict__ qkv, const float* __restrict__ cos_t,
                                        const float* __restrict__ sin_t, int pos0, int n_q,
                                        int n_kv, int head_dim, int row_stride) {
    const int half_d = head_dim / 2;
    const int t = blockIdx.x;
    const float* c = cos_t + (long long)(pos0 + t) * half_d;
    const float* s = sin_t + (long long)(pos0 + t) * half_d;
    __half* row = qkv + (long long)t * row_stride;
    const int heads = n_q + n_kv;  // q heads then k heads are contiguous at the row start
    for (int i = threadIdx.x; i < heads * half_d; i += blockDim.x) {
        const int h = i / half_d, j = i % half_d;
        __half* v = row + h * head_dim;
        const float x0 = __half2float(v[j]), x1 = __half2float(v[j + half_d]);
        v[j] = __float2half(x0 * c[j] - x1 * s[j]);
        v[j + half_d] = __float2half(x1 * c[j] + x0 * s[j]);
    }
}
