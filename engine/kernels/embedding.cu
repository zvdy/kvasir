#include "common.cuh"

extern "C" __global__ void embedding_gather(const int* __restrict__ ids,
                                            const __half* __restrict__ table,
                                            __half* __restrict__ out, int hidden) {
    const long long row = ids[blockIdx.x];
    const __half* src = table + row * hidden;
    __half* dst = out + (long long)blockIdx.x * hidden;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) dst[i] = src[i];
}
