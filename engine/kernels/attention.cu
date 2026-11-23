#include <stdint.h>

#include "common.cuh"

// Algorithm and launch shapes: DESIGN.md §3. Scores are kept in the exp2 domain (q pre-scaled by log2 e / sqrt(d)).
#define ATT_THREADS 128
#define DEC_UNROLL 4
#define DEC_MAX_SPLITS 128  // = ATT_DEC_MAX_SPLITS in kernels.rs

__device__ __forceinline__ void unpack8(const uint4& r, float* f) {
    const __half2* h = reinterpret_cast<const __half2*>(&r);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 v = __half22float2(h[i]);
        f[2 * i] = v.x;
        f[2 * i + 1] = v.y;
    }
}

__device__ __forceinline__ float rescale(float m_old, float m_new) {
    return m_old == -INFINITY ? 0.f : exp2f(m_old - m_new);
}

// Decode (T = 1): block = (kv head, context split). LPK lanes cover one 16-byte slice each of a key row, so a warp
// reads KPW whole rows per load; every key/value row is used for all G query heads of the group.
template <int HD, int G>
__device__ __forceinline__ void attention_decode_impl(const __half* __restrict__ qkv, const __half* __restrict__ kc,
                                                      const __half* __restrict__ vc, __half* __restrict__ out,
                                                      float* __restrict__ part, int kv_len, int n_q, int max_seq,
                                                      int chunk, float scale_log2) {
    constexpr int LPK = HD / 8;
    constexpr int KPW = 32 / LPK;
    constexpr int NW = ATT_THREADS / 32;
    constexpr int KPB = NW * KPW;
    __shared__ float sm_m[NW][G], sm_l[NW][G];
    __shared__ float sm_acc[NW][G][HD];

    const int kvh = blockIdx.x, split = blockIdx.y;
    part += n_q;  // scratch = [n_q counter words | partials]
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int sub = lane / LPK, d0 = (lane % LPK) * 8;

    float q[G][8];
#pragma unroll
    for (int g = 0; g < G; ++g) {
        unpack8(*reinterpret_cast<const uint4*>(qkv + (kvh * G + g) * HD + d0), q[g]);
#pragma unroll
        for (int i = 0; i < 8; ++i) q[g][i] *= scale_log2;
    }
    const __half* kb = kc + (long long)kvh * max_seq * HD + d0;
    const __half* vb = vc + (long long)kvh * max_seq * HD + d0;

    float m[G], l[G], acc[G][8];
#pragma unroll
    for (int g = 0; g < G; ++g) {
        m[g] = -INFINITY;
        l[g] = 0.f;
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[g][i] = 0.f;
    }

    const int j0 = split * chunk, j1 = min(j0 + chunk, kv_len);
    // Warp-uniform trip count (the score reduction shuffles need every lane); next round's rows are prefetched.
    auto load = [&](int base, uint4* kr, uint4* vr) {
#pragma unroll
        for (int u = 0; u < DEC_UNROLL; ++u) {
            const int j = base + u * KPB;
            if (j < j1) {
                kr[u] = __ldcs(reinterpret_cast<const uint4*>(kb + (long long)j * HD));
                vr[u] = __ldcs(reinterpret_cast<const uint4*>(vb + (long long)j * HD));
            } else {
                kr[u] = vr[u] = make_uint4(0, 0, 0, 0);
            }
        }
    };
    uint4 kn[DEC_UNROLL], vn[DEC_UNROLL];
    load(j0 + warp * KPW + sub, kn, vn);
    for (int wbase = j0 + warp * KPW; wbase < j1; wbase += DEC_UNROLL * KPB) {
        const int base = wbase + sub;
        uint4 kr[DEC_UNROLL], vr[DEC_UNROLL];
#pragma unroll
        for (int u = 0; u < DEC_UNROLL; ++u) {
            kr[u] = kn[u];
            vr[u] = vn[u];
        }
        load(base + DEC_UNROLL * KPB, kn, vn);
        float s[DEC_UNROLL][G];
#pragma unroll
        for (int u = 0; u < DEC_UNROLL; ++u) {
            float kf[8];
            unpack8(kr[u], kf);
#pragma unroll
            for (int g = 0; g < G; ++g) {
                float a = 0.f;
#pragma unroll
                for (int i = 0; i < 8; ++i) a = fmaf(q[g][i], kf[i], a);
#pragma unroll
                for (int o = LPK / 2; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
                s[u][g] = (base + u * KPB < j1) ? a : -INFINITY;
            }
        }
#pragma unroll
        for (int g = 0; g < G; ++g) {
            float mx = s[0][g];
#pragma unroll
            for (int u = 1; u < DEC_UNROLL; ++u) mx = fmaxf(mx, s[u][g]);
            const float mn = fmaxf(m[g], mx);
            const float c = rescale(m[g], mn);
            m[g] = mn;
            l[g] *= c;
#pragma unroll
            for (int i = 0; i < 8; ++i) acc[g][i] *= c;
        }
#pragma unroll
        for (int u = 0; u < DEC_UNROLL; ++u) {
            float vf[8];
            unpack8(vr[u], vf);
#pragma unroll
            for (int g = 0; g < G; ++g) {
                const float p = s[u][g] == -INFINITY ? 0.f : exp2f(s[u][g] - m[g]);
                l[g] += p;
#pragma unroll
                for (int i = 0; i < 8; ++i) acc[g][i] = fmaf(p, vf[i], acc[g][i]);
            }
        }
    }

#pragma unroll
    for (int o = LPK; o < 32; o <<= 1) {
#pragma unroll
        for (int g = 0; g < G; ++g) {
            const float mo = __shfl_xor_sync(0xffffffffu, m[g], o);
            const float lo = __shfl_xor_sync(0xffffffffu, l[g], o);
            const float mn = fmaxf(m[g], mo);
            const float a = rescale(m[g], mn), b = rescale(mo, mn);
            m[g] = mn;
            l[g] = l[g] * a + lo * b;
#pragma unroll
            for (int i = 0; i < 8; ++i) acc[g][i] = acc[g][i] * a + __shfl_xor_sync(0xffffffffu, acc[g][i], o) * b;
        }
    }
    if (lane < LPK) {
#pragma unroll
        for (int g = 0; g < G; ++g) {
#pragma unroll
            for (int i = 0; i < 8; ++i) sm_acc[warp][g][d0 + i] = acc[g][i];
            if (lane == 0) {
                sm_m[warp][g] = m[g];
                sm_l[warp][g] = l[g];
            }
        }
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < G * HD; idx += ATT_THREADS) {
        const int g = idx / HD, d = idx % HD;
        float mx = -INFINITY;
#pragma unroll
        for (int w = 0; w < NW; ++w) mx = fmaxf(mx, sm_m[w][g]);
        float L = 0.f, A = 0.f;
#pragma unroll
        for (int w = 0; w < NW; ++w) {
            const float c = rescale(sm_m[w][g], mx);
            L += sm_l[w][g] * c;
            A += sm_acc[w][g][d] * c;
        }
        const int h = kvh * G + g;
        if (gridDim.y == 1) {
            out[h * HD + d] = __float2half(A / L);
        } else {
            float* p = part + ((long long)split * n_q + h) * (HD + 2);
            p[2 + d] = A;
            if (d == 0) {
                p[0] = mx;
                p[1] = L;
            }
        }
    }
    if (gridDim.y == 1) return;

    // Last-block-reduces: the final split of this kv head merges all partials and re-arms the counter.
    __shared__ bool is_last;
    unsigned* counter = reinterpret_cast<unsigned*>(part - n_q) + kvh;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        is_last = atomicAdd(counter, 1u) == gridDim.y - 1;
        if (is_last) *counter = 0;
    }
    __syncthreads();
    if (!is_last) return;
    __threadfence();
    __shared__ float sm_c[G][DEC_MAX_SPLITS], sm_ls[G][DEC_MAX_SPLITS], sm_inv[G];
    const int S = gridDim.y;
    const long long stride = (long long)n_q * (HD + 2);
    const float* pk = part + kvh * G * (HD + 2);
    for (int i = threadIdx.x; i < G * S; i += ATT_THREADS) {
        const float* p = pk + (i % S) * stride + (i / S) * (HD + 2);
        sm_c[i / S][i % S] = __ldcg(p);
        sm_ls[i / S][i % S] = __ldcg(p + 1);
    }
    __syncthreads();
    if (threadIdx.x < G) {
        const int g = threadIdx.x;
        float mx = -INFINITY;
        for (int s = 0; s < S; ++s) mx = fmaxf(mx, sm_c[g][s]);
        float L = 0.f;
        for (int s = 0; s < S; ++s) {
            const float c = exp2f(sm_c[g][s] - mx);
            sm_c[g][s] = c;
            L += sm_ls[g][s] * c;
        }
        sm_inv[g] = 1.f / L;
    }
    __syncthreads();
    for (int idx = threadIdx.x; idx < G * HD; idx += ATT_THREADS) {
        const int g = idx / HD, d = idx % HD;
        const float* p = pk + g * (HD + 2) + 2 + d;
        float A = 0.f;
#pragma unroll 8
        for (int s = 0; s < S; ++s) A = fmaf(__ldcg(p + s * stride), sm_c[g][s], A);
        out[(kvh * G + g) * HD + d] = __float2half(A * sm_inv[g]);
    }
}

__device__ __forceinline__ void cp_async16(void* dst, const void* src, bool ok) {
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(dst));
    const int n = ok ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(src), "r"(n));
}

__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t pack_half2(float lo, float hi, float& sum) {
    const __half2 h = __floats2half2_rn(lo, hi);
    sum += __low2float(h) + __high2float(h);
    return *reinterpret_cast<const uint32_t*>(&h);
}

// Prefill (T > 1, causal): FA2-style. Block = (query tile, kv head); each warp owns 16 rows = (16 / G tokens) x G heads
// as mma A fragments; K/V tiles are double-buffered in smem with cp.async and shared by the 4 warps.
template <int HD, int G>
__device__ __forceinline__ void attention_prefill_impl(const __half* __restrict__ qkv, const __half* __restrict__ kc,
                                                       const __half* __restrict__ vc, __half* __restrict__ out, int T,
                                                       int pos0, int n_q, int max_seq, int row_stride,
                                                       float scale_log2) {
    constexpr int KT = HD == 64 ? 64 : 32;
    constexpr int PAD = HD + 8;  // 16-byte row skew: conflict-free fragment loads and ldmatrix
    constexpr int TPW = 16 / G;
    constexpr int QT = (ATT_THREADS / 32) * TPW;
    __shared__ __align__(16) __half ks[2][KT][PAD];
    __shared__ __align__(16) __half vs[2][KT][PAD];

    const int kvh = blockIdx.y;
    const int q0 = (gridDim.x - 1 - blockIdx.x) * QT;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int gq = lane >> 2, tq = lane & 3;
    const int tok0 = q0 + warp * TPW + gq / G, tok1 = q0 + warp * TPW + (gq + 8) / G;
    const int h0 = kvh * G + gq % G, h1 = kvh * G + (gq + 8) % G;

    uint32_t qa[HD / 16][4];
    {
        const __half* r0 = qkv + (long long)min(tok0, T - 1) * row_stride + h0 * HD + 2 * tq;
        const __half* r1 = qkv + (long long)min(tok1, T - 1) * row_stride + h1 * HD + 2 * tq;
#pragma unroll
        for (int kk = 0; kk < HD / 16; ++kk) {
            qa[kk][0] = *reinterpret_cast<const uint32_t*>(r0 + 16 * kk);
            qa[kk][1] = *reinterpret_cast<const uint32_t*>(r1 + 16 * kk);
            qa[kk][2] = *reinterpret_cast<const uint32_t*>(r0 + 16 * kk + 8);
            qa[kk][3] = *reinterpret_cast<const uint32_t*>(r1 + 16 * kk + 8);
        }
    }
    float o[HD / 8][4];
#pragma unroll
    for (int n = 0; n < HD / 8; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
    const int lim0 = pos0 + tok0, lim1 = pos0 + tok1;

    const int kv_end = pos0 + min(q0 + QT, T);
    const int n_tiles = (kv_end + KT - 1) / KT;
    const __half* kb = kc + (long long)kvh * max_seq * HD;
    const __half* vb = vc + (long long)kvh * max_seq * HD;
    auto load_tile = [&](int tile, int buf) {
        constexpr int CH = HD / 8;
        for (int c = threadIdx.x; c < KT * CH; c += ATT_THREADS) {
            const int r = c / CH, cc = (c % CH) * 8;
            const int j = tile * KT + r;
            const bool ok = j < kv_end;
            const long long off = (long long)(ok ? j : 0) * HD + cc;
            cp_async16(&ks[buf][r][cc], kb + off, ok);
            cp_async16(&vs[buf][r][cc], vb + off, ok);
        }
        asm volatile("cp.async.commit_group;\n" ::);
    };

    load_tile(0, 0);
    for (int it = 0; it < n_tiles; ++it) {
        const int buf = it & 1;
        if (it + 1 < n_tiles) {
            load_tile(it + 1, buf ^ 1);
            asm volatile("cp.async.wait_group 1;\n" ::);
        } else {
            asm volatile("cp.async.wait_group 0;\n" ::);
        }
        __syncthreads();

        float s[KT / 8][4];
#pragma unroll
        for (int nt = 0; nt < KT / 8; ++nt) {
            s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.f;
#pragma unroll
            for (int kk = 0; kk < HD / 16; ++kk) {
                const __half* kr = &ks[buf][nt * 8 + gq][16 * kk + 2 * tq];
                mma16816(s[nt], qa[kk], *reinterpret_cast<const uint32_t*>(kr),
                         *reinterpret_cast<const uint32_t*>(kr + 8));
            }
        }

        float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
        for (int nt = 0; nt < KT / 8; ++nt) {
            const int j = it * KT + nt * 8 + 2 * tq;
            s[nt][0] = j <= lim0 ? s[nt][0] * scale_log2 : -INFINITY;
            s[nt][1] = j + 1 <= lim0 ? s[nt][1] * scale_log2 : -INFINITY;
            s[nt][2] = j <= lim1 ? s[nt][2] * scale_log2 : -INFINITY;
            s[nt][3] = j + 1 <= lim1 ? s[nt][3] * scale_log2 : -INFINITY;
            mx0 = fmaxf(mx0, fmaxf(s[nt][0], s[nt][1]));
            mx1 = fmaxf(mx1, fmaxf(s[nt][2], s[nt][3]));
        }
#pragma unroll
        for (int off = 1; off < 4; off <<= 1) {
            mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, off));
            mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, off));
        }
        // Key 0 is in tile 0 and visible to every row, so m is finite from the first tile on.
        const float mn0 = fmaxf(m0, mx0), mn1 = fmaxf(m1, mx1);
        const float c0 = rescale(m0, mn0), c1 = rescale(m1, mn1);
        m0 = mn0;
        m1 = mn1;
        l0 *= c0;
        l1 *= c1;
#pragma unroll
        for (int n = 0; n < HD / 8; ++n) {
            o[n][0] *= c0;
            o[n][1] *= c0;
            o[n][2] *= c1;
            o[n][3] *= c1;
        }

        uint32_t pa[KT / 16][4];
#pragma unroll
        for (int nt = 0; nt < KT / 8; ++nt) {
            const int kk = nt >> 1, hi = (nt & 1) * 2;
            pa[kk][hi] = pack_half2(exp2f(s[nt][0] - m0), exp2f(s[nt][1] - m0), l0);
            pa[kk][hi + 1] = pack_half2(exp2f(s[nt][2] - m1), exp2f(s[nt][3] - m1), l1);
        }
#pragma unroll
        for (int kk = 0; kk < KT / 16; ++kk) {
#pragma unroll
            for (int np = 0; np < HD / 16; ++np) {
                const int mat = lane >> 3;
                const __half* vr = &vs[buf][kk * 16 + (mat & 1) * 8 + (lane & 7)][np * 16 + (mat >> 1) * 8];
                const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(vr));
                uint32_t b0, b1, b2, b3;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(b0), "=r"(b1), "=r"(b2), "=r"(b3)
                             : "r"(addr));
                mma16816(o[2 * np], pa[kk], b0, b1);
                mma16816(o[2 * np + 1], pa[kk], b2, b3);
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int off = 1; off < 4; off <<= 1) {
        l0 += __shfl_xor_sync(0xffffffffu, l0, off);
        l1 += __shfl_xor_sync(0xffffffffu, l1, off);
    }
    const float inv0 = 1.f / l0, inv1 = 1.f / l1;
#pragma unroll
    for (int n = 0; n < HD / 8; ++n) {
        if (tok0 < T)
            *reinterpret_cast<__half2*>(out + ((long long)tok0 * n_q + h0) * HD + n * 8 + 2 * tq) =
                __floats2half2_rn(o[n][0] * inv0, o[n][1] * inv0);
        if (tok1 < T)
            *reinterpret_cast<__half2*>(out + ((long long)tok1 * n_q + h1) * HD + n * 8 + 2 * tq) =
                __floats2half2_rn(o[n][2] * inv1, o[n][3] * inv1);
    }
}

#define ATT_INSTANTIATE(HD, G)                                                                                      \
    extern "C" __global__ void __launch_bounds__(ATT_THREADS) attention_decode_##HD##_##G(                           \
        const __half* qkv, const __half* kc, const __half* vc, __half* out, float* part, int kv_len, int n_q,       \
        int max_seq, int chunk, float scale_log2) {                                                                 \
        attention_decode_impl<HD, G>(qkv, kc, vc, out, part, kv_len, n_q, max_seq, chunk, scale_log2);              \
    }                                                                                                               \
    extern "C" __global__ void __launch_bounds__(ATT_THREADS) attention_prefill_##HD##_##G(                          \
        const __half* qkv, const __half* kc, const __half* vc, __half* out, int T, int pos0, int n_q, int max_seq, \
        int row_stride, float scale_log2) {                                                                         \
        attention_prefill_impl<HD, G>(qkv, kc, vc, out, T, pos0, n_q, max_seq, row_stride, scale_log2);           \
    }

ATT_INSTANTIATE(64, 1)
ATT_INSTANTIATE(64, 2)
ATT_INSTANTIATE(64, 4)
ATT_INSTANTIATE(64, 8)
ATT_INSTANTIATE(128, 1)
ATT_INSTANTIATE(128, 2)
ATT_INSTANTIATE(128, 4)
ATT_INSTANTIATE(128, 8)
