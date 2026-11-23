#include <stdint.h>

#include "common.cuh"

// Decode GEMV y = W·x, W [out, in] row-major FP16, FP32 accumulate; launch shapes and epilogues: DESIGN.md §3.
#define GEMV_UNROLL 8
#define GEMV_ROWS 2       // = GEMV_ROWS in gemm.rs
#define GEMV_SILU_ROWS 1  // gate/up row pairs per warp, = GEMV_SILU_ROWS in gemm.rs

__device__ __forceinline__ uint4 ld_stream(const uint4* p) {
    uint4 r;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
                 : "l"(p));
    return r;
}

__device__ __forceinline__ float dot8(const uint4& w, const uint4& x, float acc) {
    const __half2* wh = reinterpret_cast<const __half2*>(&w);
    const __half2* xh = reinterpret_cast<const __half2*>(&x);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 a = __half22float2(wh[i]), b = __half22float2(xh[i]);
        acc = fmaf(a.x, b.x, acc);
        acc = fmaf(a.y, b.y, acc);
    }
    return acc;
}

__device__ __forceinline__ float sumsq8(const uint4& x, float ss) {
    const __half2* h = reinterpret_cast<const __half2*>(&x);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 f = __half22float2(h[i]);
        ss += f.x * f.x + f.y * f.y;
    }
    return ss;
}

// Same arithmetic as the rmsnorm kernel: half(x · inv · w).
__device__ __forceinline__ uint4 norm8(const uint4& x, const uint4& w, float inv) {
    uint4 r;
    const __half2* xh = reinterpret_cast<const __half2*>(&x);
    const __half2* wh = reinterpret_cast<const __half2*>(&w);
    __half2* rh = reinterpret_cast<__half2*>(&r);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 a = __half22float2(xh[i]), b = __half22float2(wh[i]);
        rh[i] = __halves2half2(__float2half(a.x * inv * b.x), __float2half(a.y * inv * b.y));
    }
    return r;
}

// Lane l reads 16-byte chunks l, l+32, ... of NR rows; with `nw`, x is RMS-normalised on the fly (DESIGN.md §3).
template <int NR>
__device__ __forceinline__ void gemv_rows(const __half* const (&rows)[NR], const __half* __restrict__ x,
                                          const __half* __restrict__ nw, float eps, int nc, float (&acc)[NR]) {
    const int lane = threadIdx.x & 31;
    const uint4* xv = reinterpret_cast<const uint4*>(x);
    const uint4* nv = reinterpret_cast<const uint4*>(nw);
    const uint4* wv[NR];
#pragma unroll
    for (int r = 0; r < NR; ++r) {
        wv[r] = reinterpret_cast<const uint4*>(rows[r]);
        acc[r] = 0.f;
    }
    float inv = 1.f;
    for (int base = 0; base < nc; base += 32 * GEMV_UNROLL) {
        uint4 w[GEMV_UNROLL][NR], xx[GEMV_UNROLL];
#pragma unroll
        for (int u = 0; u < GEMV_UNROLL; ++u) {
            const int c = base + 32 * u + lane;
#pragma unroll
            for (int r = 0; r < NR; ++r) w[u][r] = c < nc ? ld_stream(wv[r] + c) : make_uint4(0, 0, 0, 0);
        }
#pragma unroll
        for (int u = 0; u < GEMV_UNROLL; ++u) {
            const int c = base + 32 * u + lane;
            xx[u] = c < nc ? __ldg(xv + c) : make_uint4(0, 0, 0, 0);
        }
        if (nw != nullptr) {
            uint4 nn[GEMV_UNROLL];
#pragma unroll
            for (int u = 0; u < GEMV_UNROLL; ++u) {
                const int c = base + 32 * u + lane;
                nn[u] = c < nc ? __ldg(nv + c) : make_uint4(0, 0, 0, 0);
            }
            if (base == 0) {
                // One pass covers the whole row when nc <= 32·GEMV_UNROLL (in <= 2048); otherwise re-read x.
                float ss = 0.f;
                if (nc <= 32 * GEMV_UNROLL) {
#pragma unroll
                    for (int u = 0; u < GEMV_UNROLL; ++u) ss = sumsq8(xx[u], ss);
                } else {
                    for (int c = lane; c < nc; c += 32) ss = sumsq8(__ldg(xv + c), ss);
                }
                inv = rsqrtf(warp_sum(ss) / (float)(8 * nc) + eps);
            }
#pragma unroll
            for (int u = 0; u < GEMV_UNROLL; ++u) xx[u] = norm8(xx[u], nn[u], inv);
        }
#pragma unroll
        for (int u = 0; u < GEMV_UNROLL; ++u)
#pragma unroll
            for (int r = 0; r < NR; ++r) acc[r] = dot8(w[u][r], xx[u], acc[r]);
    }
#pragma unroll
    for (int r = 0; r < NR; ++r) acc[r] = warp_sum(acc[r]);
}

__device__ __forceinline__ int gemv_warp() { return blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5); }

// MODE 0: FP16 out, 1: FP32 out, 2: y = half(y + half(W·x)). Tail rows of the last warp are read, not stored.
template <int R, int MODE>
__device__ __forceinline__ void gemv_impl(const __half* __restrict__ w, const __half* __restrict__ x, void* y, int in,
                                          int out, const __half* __restrict__ nw, float eps) {
    const int row0 = gemv_warp() * R;
    if (row0 >= out) return;
    const __half* rows[R];
#pragma unroll
    for (int r = 0; r < R; ++r) rows[r] = w + (long long)min(row0 + r, out - 1) * in;
    float acc[R];
    gemv_rows<R>(rows, x, nw, eps, in / 8, acc);
    const int lane = threadIdx.x & 31;
#pragma unroll
    for (int r = 0; r < R; ++r) {
        if (lane == r && row0 + r < out) {
            if (MODE == 0) {
                static_cast<__half*>(y)[row0 + r] = __float2half(acc[r]);
            } else if (MODE == 1) {
                static_cast<float*>(y)[row0 + r] = acc[r];
            } else {
                __half* yh = static_cast<__half*>(y);
                yh[row0 + r] = __float2half(__half2float(yh[row0 + r]) + __half2float(__float2half(acc[r])));
            }
        }
    }
}

// W = [gate | up] rows; act = half(silu(half(g)) · half(u)), the roundings of gate_up + silu_mul.
template <int R>
__device__ __forceinline__ void gemv_silu_mul_impl(const __half* __restrict__ w, const __half* __restrict__ x,
                                                   __half* __restrict__ act, int in, int inter,
                                                   const __half* __restrict__ nw, float eps) {
    const int i0 = gemv_warp() * R;
    if (i0 >= inter) return;
    const __half* rows[2 * R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const long long i = min(i0 + r, inter - 1);
        rows[r] = w + i * in;
        rows[R + r] = w + (i + inter) * in;
    }
    float acc[2 * R];
    gemv_rows<2 * R>(rows, x, nw, eps, in / 8, acc);
    const int lane = threadIdx.x & 31;
#pragma unroll
    for (int r = 0; r < R; ++r) {
        if (lane == r && i0 + r < inter) {
            const float g = __half2float(__float2half(acc[r])), u = __half2float(__float2half(acc[R + r]));
            act[i0 + r] = __float2half(g / (1.f + __expf(-g)) * u);
        }
    }
}

// `nw` (RMSNorm weight, [in]) may be null: x is then used as is.
extern "C" __global__ void gemv_f16(const __half* __restrict__ w, const __half* __restrict__ x, __half* y, int in,
                                    int out, const __half* __restrict__ nw, float eps) {
    gemv_impl<GEMV_ROWS, 0>(w, x, y, in, out, nw, eps);
}

extern "C" __global__ void gemv_f32(const __half* __restrict__ w, const __half* __restrict__ x, float* y, int in,
                                    int out, const __half* __restrict__ nw, float eps) {
    gemv_impl<GEMV_ROWS, 1>(w, x, y, in, out, nw, eps);
}

extern "C" __global__ void gemv_add(const __half* __restrict__ w, const __half* __restrict__ x, __half* y, int in,
                                    int out, const __half* __restrict__ nw, float eps) {
    gemv_impl<GEMV_ROWS, 2>(w, x, y, in, out, nw, eps);
}

extern "C" __global__ void gemv_silu_mul(const __half* __restrict__ w, const __half* __restrict__ x,
                                         __half* __restrict__ act, int in, int inter, const __half* __restrict__ nw,
                                         float eps) {
    gemv_silu_mul_impl<GEMV_SILU_ROWS>(w, x, act, in, inter, nw, eps);
}
