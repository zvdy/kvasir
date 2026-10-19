#include "common.cuh"

// dynamic smem: (pos0 + T + head_dim + 128) floats
#define ATT_THREADS 128
extern "C" __global__ void attention(const __half* __restrict__ qkv, const __half* __restrict__ kc,
                                     const __half* __restrict__ vc, __half* __restrict__ out,
                                     int pos0, int n_q, int n_kv, int head_dim, int max_seq,
                                     int row_stride, float scale) {
    extern __shared__ float smem[];
    const int h = blockIdx.x, t = blockIdx.y;
    const int kv_len = pos0 + t + 1;
    const int kvh = h / (n_q / n_kv);
    const int max_len = pos0 + gridDim.y;
    float* scores = smem;                 // [max_len]
    float* qs = smem + max_len;           // [head_dim]
    float* part = qs + head_dim;          // [ATT_THREADS]
    __shared__ float scratch[32];

    const __half* q = qkv + (long long)t * row_stride + h * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) qs[d] = __half2float(q[d]);
    __syncthreads();

    const __half* kbase = kc + (long long)kvh * max_seq * head_dim;
    const __half* vbase = vc + (long long)kvh * max_seq * head_dim;

    float lmax = -INFINITY;
    for (int j = threadIdx.x; j < kv_len; j += blockDim.x) {
        const __half* k = kbase + (long long)j * head_dim;
        float acc = 0.f;
        for (int d = 0; d < head_dim; ++d) acc += qs[d] * __half2float(k[d]);
        acc *= scale;
        scores[j] = acc;
        lmax = fmaxf(lmax, acc);
    }
    const float m = block_max(lmax, scratch);

    float lsum = 0.f;
    for (int j = threadIdx.x; j < kv_len; j += blockDim.x) {
        const float p = __expf(scores[j] - m);
        scores[j] = p;
        lsum += p;
    }
    const float denom = block_sum(lsum, scratch);  // also syncs: scores visible to all

    const int slices = ATT_THREADS / head_dim;
    const int d = threadIdx.x % head_dim, s = threadIdx.x / head_dim;
    float acc = 0.f;
    for (int j = s; j < kv_len; j += slices)
        acc += scores[j] * __half2float(vbase[(long long)j * head_dim + d]);
    part[threadIdx.x] = acc;
    __syncthreads();
    if (s == 0) {
        float r = 0.f;
        for (int i = 0; i < slices; ++i) r += part[i * head_dim + d];
        out[(long long)t * n_q * head_dim + h * head_dim + d] = __float2half(r / denom);
    }
}
