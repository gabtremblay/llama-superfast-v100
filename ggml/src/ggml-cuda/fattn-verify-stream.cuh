#pragma once
// The 1..8-row verify's / draft step's attention over an F16 cache at depth, streamed on Volta (sm_70).
//
// Second design. One block (8 warps, one per SM) per (KV head, slice of the sequence), blockIdx.x = kvh + n_kv_heads*split.
// All NR = 48 query rows of the group (8 tokens x G = 6 heads) are resident; Q is staged once in shared memory as f16, pre-scaled
// by scale*log2(e). Keys are streamed in tiles of TK = 64. The two phases run on separate warp groups, one tile apart:
//   warps 0..3 (QK group): K tile (64 keys x 256 dims) loaded ROW-COALESCED (a warp instruction reads one 512 B key row) into
//     registers one tile ahead, stored to shared memory (8-half row pad), read back as m8n8k4 A fragments (LDS.128 per lane),
//     S = K Q^T for 32 keys x 24 rows per warp, then the online softmax; P = exp2(S - m) (f16) and the per-row rescale go through
//     shared memory (P double-buffered).
//   warps 4..7 (PV group): O^T += V^T P^T for tile j-1 while the QK group does tile j. V is read from global memory straight into
//     the A fragments, 16 B per lane (a lane's 8 dims are two M blocks of 4), one tile ahead in registers; warp w owns 64 dims.
//   One __syncthreads per tile, plus a 128-thread barrier in the QK group (the max exchange, which also frees the K tile).
// Key labels in the QK fragments are permuted so that the LDS.128 of K is bank-conflict free; P is written in natural key order.
// The online softmax is exact-exp2 in fp32 per row; row sums are kept per lane (rescaled with the row's factor) and reduced once
// at the end. Each block writes an unnormalised fp32 partial plus (max in natural-log units, sum) per row, combined by
// flash_attn_combine_results; with one split it writes the normalised result directly.
// STUB = true is the ceiling probe: the same data path, both mma phases, the split and the combine, with the softmax replaced by
// P = S/1024 (no max, no exp, no rescale, no sums).
// KVQ8 = true reads a Q8_0 cache instead. Each lane loads the same 8 dims of a K or V row as on F16 (the row-coalesced K
// load, the V fragment load), from its Q8_0 block: the block's d and 8 int8 codes, as five 2-byte loads (a 34-byte block is
// only 2-byte aligned). The raw words wait in the same registers the F16 data would (kreg, vreg) and are widened to F16 where
// those are consumed (the K tile's store to shared memory, the V fragment), so the loads stay one tile ahead and the mma
// sequence is unchanged. The widening is exact up to one rounding of q*d to F16, as the dequantization to an F16 copy.

#include <cuda_fp16.h>
#include <cstdint>

#ifndef FVS_NACC
#define FVS_NACC 2
#endif

namespace fvs {
constexpr int D   = 256;
constexpr int NT  = 6;          // query n-tiles of 8 rows
constexpr int NR  = NT*8;       // 48 query rows
constexpr int TK  = 64;         // keys per tile
constexpr int NW  = 8;
constexpr int NTH = NW*32;
constexpr int QST = D  + 8;     // Q and K row stride in halves (8-half pad: conflict-free fragment reads)
constexpr int KST = D  + 8;
constexpr int PST = TK + 8;     // P row stride in halves
constexpr int MST = TK + 8;     // mask row stride in halves

struct smem_t {
    half  Q[NR*QST];
    half  K[TK*KST];
    half  P[2][NR*PST];
    half  M[2][8*MST];
    float redmax[2][NR];
    float redsum[2][NR];
    float alpha[2][NR];
    int   resc[2][2];        // per tile buffer and row group: does any row of the group need its O rescaled (a factor != 1)
    float m[NR];
};

static __device__ __forceinline__ void mma_rc(float (&c)[8], const uint32_t a0, const uint32_t a1, const uint32_t b0, const uint32_t b1) {
    asm("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]), "+f"(c[5]), "+f"(c[6]), "+f"(c[7])
        : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}
static __device__ __forceinline__ void mma_cc(float (&c)[8], const uint32_t a0, const uint32_t a1, const uint32_t b0, const uint32_t b1) {
    asm("mma.sync.aligned.m8n8k4.col.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]), "+f"(c[5]), "+f"(c[6]), "+f"(c[7])
        : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

static __device__ __forceinline__ float ex2(const float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

// 8 dims of a row: on F16 one 16-byte load at p; on Q8_0 p is the Q8_0 block and off (0, 8, 16, 24) the first of the 8 codes,
// returned raw as {codes 0-3, codes 4-7, d, 0}. W (Q8_0 only): the codes as three 4-byte loads of the aligned 12-byte window
// around them and two funnel shifts (they start 2 or 0 bytes past a 4-byte boundary: blocks are 34 bytes, rows 16-aligned),
// instead of four 2-byte loads; the same code on every lane, no branch
template <bool Q8, bool W = false>
static __device__ __forceinline__ uint4 ld_row8(const char * p, const int off) {
    if constexpr (Q8) {
        const uint32_t d = (uint32_t) __ldg((const unsigned short *) p);
        if constexpr (W) {
            const uintptr_t s = (uintptr_t)(p + 2 + off);
            const uint32_t * w = (const uint32_t *)(s & ~(uintptr_t) 3);
            const uint32_t sh = 8*(uint32_t)(s & 3);
            // aligned codes need two words: the third load re-reads the second, so no load leaves the codes' own words
            // (the last block of the last row ends the buffer on an aligned 8 codes)
            const uint32_t w0 = __ldg(w + 0), w1 = __ldg(w + 1), w2 = __ldg(w + 1 + (sh != 0));
            return make_uint4(__funnelshift_r(w0, w1, sh), __funnelshift_r(w1, w2, sh), d, 0);
        } else {
            const unsigned short * qs = (const unsigned short *)(p + 2 + off);
            const uint32_t lo = (uint32_t) __ldg(qs + 0) | ((uint32_t) __ldg(qs + 1) << 16);
            const uint32_t hi = (uint32_t) __ldg(qs + 2) | ((uint32_t) __ldg(qs + 3) << 16);
            return make_uint4(lo, hi, d, 0);
        }
    } else {
        (void) off;
        return __ldg((const uint4 *) p);
    }
}

// what ld_row8 returned, as 8 F16 values. Q8_0: each int8 code q is biased to an unsigned byte and placed under the F16
// exponent of 1024 (0x64xx = 1024 + q + 128, exact), 1152 is taken off (exact: q itself) and the product with d is the one
// rounding
template <bool Q8>
static __device__ __forceinline__ uint4 to_f16x8(const uint4 r) {
    if constexpr (Q8) {
        const half2  d    = __half2half2(__ushort_as_half((unsigned short) r.z));
        const half2  bias = __half2half2(__ushort_as_half((unsigned short) 0x6480)); // 1152
        const uint32_t a  = r.x ^ 0x80808080u;
        const uint32_t b  = r.y ^ 0x80808080u;
        uint32_t w[4] = { __byte_perm(a, 0x64646464u, 0x5140), __byte_perm(a, 0x64646464u, 0x7362),
                          __byte_perm(b, 0x64646464u, 0x5140), __byte_perm(b, 0x64646464u, 0x7362) };
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const half2 h = __hmul2(__hsub2(*(const half2 *) &w[i], bias), d);
            w[i] = *(const uint32_t *) &h;
        }
        return make_uint4(w[0], w[1], w[2], w[3]);
    } else {
        return r;
    }
}

// G: GQA ratio (query heads per KV head); rows r < n_tok*G are real: token r/G, head kvh*G + r%G.
// 9 to 16 tokens run as nhalf = 2 blocks per (KV head, split), blockIdx.x = half + nhalf*(kvh + n_kv_heads*split): block
// half takes tokens 8*half .. 8*half + 7 (its own 8-row tile, exactly as an 8-token call of the same split computes them), and the
// two blocks of a pair stream the same K and V tiles side by side, so the second read can come from L2. n_tok_all counts all tokens, n_tok the block's.
// BlockIdx.y is the KV stream (llama.cpp's non-unified cache: one stream per sequence, on ne[3] of Q, K, V and the mask); a block
// reads only its own stream's Q, K, V and mask, at the same split as a call on that stream alone, and writes its tokens at stream*n_tok_all.
// VAR & 31 = KP: of the 16 key rows per lane of a K tile, the PV group loads and stores the last KP (0 = the earlier kernel, all in the QK group)
template <int G, bool STUB, int VAR = 0, bool KVQ8 = false>
__global__ void __launch_bounds__(NTH, 1) verify_stream(
        const float * __restrict__ Q, const char * __restrict__ K, const char * __restrict__ V, const char * __restrict__ mask,
        float * __restrict__ dst, float * __restrict__ parts, float2 * __restrict__ meta,
        const int n_tok_all, const int n_head, const int n_kv_heads,
        const int64_t nbq1, const int64_t nbq2, const int64_t nbk1, const int64_t nbk2, const int64_t nbv1, const int64_t nbv2,
        const int64_t nbm1, const int n_kv, const int32_t * __restrict__ kv_count, const int nsplit, const float qscale, const int nhalf,
        const int64_t nbq3, const int64_t nbk3, const int64_t nbv3, const int64_t nbm3) {
#if __CUDA_ARCH__ == 700
    static_assert(G >= NR/8 && G <= 8, "the mask tile holds 8 tokens");
    constexpr float LOG2E = 1.4426950408889634f;
    constexpr float LN2   = 0.6931471805599453f;
    extern __shared__ __align__(16) unsigned char smem_raw[];
    smem_t & s = *reinterpret_cast<smem_t *>(smem_raw);
    // a decode graph over the whole cache passes the used key count as data; the tiles are divided over the used keys only
    const int n_kv_used = kv_count ? *kv_count : n_kv;

    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int w    = tid >> 5;
    const int l4   = lane & 3;
    const int hi   = lane >> 4;
    const int qp   = (lane >> 2) & 3;
    const bool isQK = w < 4;
    constexpr int KP = VAR & 31, KQ = 16 - KP;
    constexpr bool Q8W = (VAR & 32) != 0;   // VAR & 32: the Q8_0 codes as aligned-window loads (ld_row8)

    // this block's 8-token half; Q, the mask and the outputs from its first token on
    const int hf    = blockIdx.x % nhalf;
    const int bxk   = blockIdx.x / nhalf;
    const int tok0  = 8*hf;
    const int n_tok = min(8, n_tok_all - tok0);
    // this block's stream; its tokens are stream*n_tok_all + tok in the output, parts and meta ([stream][token][head])
    const int strm  = blockIdx.y;
    const int64_t otok0 = (int64_t) strm*n_tok_all + tok0;
    Q    = (const float *) ((const char *) Q + tok0*nbq1 + strm*nbq3);
    K    += strm*nbk3;
    V    += strm*nbv3;
    mask += tok0*nbm1 + strm*nbm3;
    if (nsplit > 1) {
        parts += otok0*n_head*nsplit*D;
        meta  += otok0*n_head*nsplit;
    } else {
        dst   += otok0*n_head*D;
    }
    const int kvh   = bxk % n_kv_heads;
    const int split = bxk / n_kv_heads;
    const int ntiles = n_kv_used / TK;
    const int it0 = (int)(((int64_t) split     *ntiles)/nsplit);
    const int it1 = (int)(((int64_t)(split + 1)*ntiles)/nsplit);
    const int nt  = it1 - it0;
    const int nrows = n_tok*G;

    // Q -> f16, scaled; padded rows are zero
    for (int idx = tid; idx < NR*(D/4); idx += NTH) {
        const int r = idx / (D/4), c4 = idx % (D/4);
        const int tok = r / G, hl = r % G;
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (r < nrows) {
            v = *(const float4 *)((const char *) Q + tok*nbq1 + (kvh*G + hl)*nbq2 + 16*c4);
        }
        half2 h0 = __floats2half2_rn(v.x*qscale, v.y*qscale);
        half2 h1 = __floats2half2_rn(v.z*qscale, v.w*qscale);
        *(uint2 *) &s.Q[r*QST + 4*c4] = make_uint2(*(uint32_t *) &h0, *(uint32_t *) &h1);
    }
    // mask for the first tile (64 threads: 8 tokens x 8 chunks of 8 keys)
    if (tid < 64 && nt > 0) {
        const int tok = tid >> 3, c = tid & 7;
        uint4 mv = make_uint4(0, 0, 0, 0);
        if (tok < n_tok) mv = *(const uint4 *)(mask + tok*nbm1 + (int64_t)(it0*TK + 8*c)*2);
        *(uint4 *) &s.M[0][tok*MST + 8*c] = mv;
    }
    if (tid < NR) {
        s.m[tid] = -INFINITY;
        s.alpha[0][tid] = 1.0f;
        s.alpha[1][tid] = 1.0f;
        if (tid < 4) s.resc[tid >> 1][tid & 1] = 0;
    }

    const char * Kbase = K + kvh*nbk2;
    const int ka = (w >> 1) & 1;   // key group of the QK phase (32 keys)
    const int rg = w & 1;          // row group of the QK phase (24 rows)
    const int keyA = 32*ka + l4 + 4*(qp & 1) + 8*hi + 16*(qp >> 1);   // this thread's key in the QK A fragment
    // a lane's 8 dims of a row: dims 8*lane.. of K, dims vd.. of V; on Q8_0 the byte offset of their block and the first code
    const int vd   = 64*(w & 3) + 16*qp + 8*hi;
    const int kofs = KVQ8 ? 34*(lane >> 2) : 16*lane;
    const int kq   = 8*(lane & 3);
    const int vq   = vd & 31;
    const char * Vbase = V + kvh*nbv2 + (int64_t) l4*nbv1 + (KVQ8 ? 34*(vd >> 5) : 2*vd);

    // The two groups have separate loops (their register state is disjoint); both cross the same barriers nt + 2 times.
    if (isQK) {
        uint4 kreg[KQ > 0 ? KQ : 1];
        float mrun[3][4], lrun[3][4];
#pragma unroll
        for (int t = 0; t < 3; ++t)
#pragma unroll
            for (int j = 0; j < 4; ++j) { mrun[t][j] = -INFINITY; lrun[t][j] = 0.0f; }
        if (KQ > 0 && nt > 0) {
            {
                const char * kp = Kbase + (int64_t)(it0*TK + w)*nbk1 + kofs;
#pragma unroll
                for (int rr = 0; rr < KQ; ++rr, kp += 4*nbk1) kreg[rr] = ld_row8<KVQ8, Q8W>(kp, kq);
            }
#pragma unroll
            for (int rr = 0; rr < KQ; ++rr) *(uint4 *) &s.K[(4*rr + w)*KST + 8*lane] = to_f16x8<KVQ8>(kreg[rr]);
            if (nt > 1) {
                const char * kp = Kbase + (int64_t)((it0 + 1)*TK + w)*nbk1 + kofs;
#pragma unroll
                for (int rr = 0; rr < KQ; ++rr, kp += 4*nbk1) kreg[rr] = ld_row8<KVQ8, Q8W>(kp, kq);
            }
        }
        __syncthreads();

        for (int j = 0; j <= nt; ++j) {
            if (j < nt) {
                const int b = j & 1;
                // mask of the next tile: loaded now, stored after the QK phase
                uint4 mnext = make_uint4(0, 0, 0, 0);
                const bool has_next = j + 1 < nt;
                if (tid < 64 && has_next && (tid >> 3) < n_tok) {
                    mnext = *(const uint4 *)(mask + (tid >> 3)*nbm1 + ((int64_t)(it0 + j + 1)*TK + 8*(tid & 7))*2);
                }

                // ---- S = K Q^T for 32 keys x 24 rows, two accumulator sets over the dims
                float S[FVS_NACC][3][8];
#pragma unroll
                for (int a = 0; a < FVS_NACC; ++a)
#pragma unroll
                    for (int t = 0; t < 3; ++t)
#pragma unroll
                        for (int i = 0; i < 8; ++i) S[a][t][i] = 0.0f;
                {
                    const half * Krow = s.K + keyA*KST;
                    const half * Qrow = s.Q + (24*rg + l4 + 4*hi)*QST;
#ifndef FVS_SKIP_QK
#pragma unroll
                    for (int c = 0; c < D/8; ++c) {
                        const uint4 kf = *(const uint4 *)(Krow + 8*c);
#pragma unroll
                        for (int t = 0; t < 3; ++t) {
                            const uint4 qf = *(const uint4 *)(Qrow + 8*t*QST + 8*c);
                            mma_rc(S[c % FVS_NACC][t], kf.x, kf.y, qf.x, qf.y);
                            mma_rc(S[c % FVS_NACC][t], kf.z, kf.w, qf.z, qf.w);
                        }
                    }
#endif
                }
                if constexpr (KP > 0) { if (j + 1 < nt) asm volatile("bar.arrive 2, 256;" ::: "memory"); }   // this warp is done reading the K tile
                // mask, in log2 units; keys and rows of this lane's accumulators
#pragma unroll
                for (int t = 0; t < 3; ++t)
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        const int qrow = 24*rg + 8*t + (i & 4) + (lane & 2) + (i & 1);
                        const int key  = 32*ka + (lane & 1) + (i & 2) + 4*(qp & 1) + 8*hi + 16*(qp >> 1);
                        {
#pragma unroll
                            for (int a = 1; a < FVS_NACC; ++a) S[0][t][i] += S[a][t][i];
                        }
                        S[0][t][i] += LOG2E*__half2float(s.M[b][(qrow / G)*MST + key]);
                    }

                if constexpr (!STUB) {
                    float mx[3][4];
#pragma unroll
                    for (int t = 0; t < 3; ++t)
#pragma unroll
                        for (int jj = 0; jj < 4; ++jj) {
                            const int i = (jj & 1) + 2*(jj & 2); // 0, 1, 4, 5
                            float v = fmaxf(S[0][t][i], S[0][t][i | 2]);
                            v = fmaxf(v, __shfl_xor_sync(0xffffffff, v,  1));
                            v = fmaxf(v, __shfl_xor_sync(0xffffffff, v,  4));
                            v = fmaxf(v, __shfl_xor_sync(0xffffffff, v,  8));
                            v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, 16));
                            mx[t][jj] = v;
                        }
                    if ((lane & 0x1D) == 0) {
#pragma unroll
                        for (int t = 0; t < 3; ++t)
#pragma unroll
                            for (int jj = 0; jj < 4; ++jj) {
                                const int i = (jj & 1) + 2*(jj & 2);
                                s.redmax[ka][24*rg + 8*t + (i & 4) + (lane & 2) + (i & 1)] = mx[t][jj];
                            }
                    }
                    asm volatile("bar.sync 1, 128;" ::: "memory");   // the max exchange; every QK warp is also done reading the K tile
                    // P = exp2(S - m); the row's rescale factor; per-lane row sums
                    bool moved = false;
#pragma unroll
                    for (int t = 0; t < 3; ++t)
#pragma unroll
                        for (int jj = 0; jj < 4; ++jj) {
                            const int i = (jj & 1) + 2*(jj & 2);
                            const int qrow = 24*rg + 8*t + (i & 4) + (lane & 2) + (i & 1);
                            const int key  = 32*ka + (lane & 1) + (i & 2) + 4*(qp & 1) + 8*hi + 16*(qp >> 1);
                            const float mt = fmaxf(mx[t][jj], s.redmax[ka ^ 1][qrow]);
                            const float mo = mrun[t][jj];
                            const float mn = fmaxf(mo, mt);
                            const float al = mn == -INFINITY ? 1.0f : ex2(mo - mn);
                            mrun[t][jj] = mn;
                            moved |= al != 1.0f;
                            if (ka == 0 && (lane & 0x1D) == 0) s.alpha[b][qrow] = al;
                            const float ms = mn == -INFINITY ? 0.0f : mn;
                            const float p0 = ex2(S[0][t][i]     - ms);
                            const float p1 = ex2(S[0][t][i | 2] - ms);
                            s.P[b][qrow*PST + key]     = __float2half(p0);
                            s.P[b][qrow*PST + key + 2] = __float2half(p1);
                            lrun[t][jj] = lrun[t][jj]*al + (p0 + p1);
                        }
                    const bool any_moved = __any_sync(0xffffffff, moved);
                    if (ka == 0 && lane == 0) s.resc[b][rg] = any_moved;
                } else {
                    asm volatile("bar.sync 1, 128;" ::: "memory");
#pragma unroll
                    for (int t = 0; t < 3; ++t)
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            const int qrow = 24*rg + 8*t + (i & 4) + (lane & 2) + (i & 1);
                            const int key  = 32*ka + (lane & 1) + (i & 2) + 4*(qp & 1) + 8*hi + 16*(qp >> 1);
                            s.P[b][qrow*PST + key] = __float2half(S[0][t][i]*(1.0f/1024.0f));
                        }
                }

                // the next K tile (loaded one iteration ago) into shared memory, then the one after it into registers
                if (has_next) {
#pragma unroll
                    for (int rr = 0; rr < KQ; ++rr) *(uint4 *) &s.K[(4*rr + w)*KST + 8*lane] = to_f16x8<KVQ8>(kreg[rr]);
                    if (tid < 64) *(uint4 *) &s.M[b ^ 1][(tid >> 3)*MST + 8*(tid & 7)] = mnext;
                    if (KQ > 0 && j + 2 < nt) {
                        const char * kp = Kbase + (int64_t)((it0 + j + 2)*TK + w)*nbk1 + kofs;
#pragma unroll
                        for (int rr = 0; rr < KQ; ++rr, kp += 4*nbk1) kreg[rr] = ld_row8<KVQ8, Q8W>(kp, kq);
                    }
                }
            }
            __syncthreads();
        }

        // row sums: per-lane partials -> lanes -> the two key groups
        float lsum[3][4];
#pragma unroll
        for (int t = 0; t < 3; ++t)
#pragma unroll
            for (int jj = 0; jj < 4; ++jj) {
                float v = lrun[t][jj];
                v += __shfl_xor_sync(0xffffffff, v,  1);
                v += __shfl_xor_sync(0xffffffff, v,  4);
                v += __shfl_xor_sync(0xffffffff, v,  8);
                v += __shfl_xor_sync(0xffffffff, v, 16);
                lsum[t][jj] = v;
            }
        if ((lane & 0x1D) == 0) {
#pragma unroll
            for (int t = 0; t < 3; ++t)
#pragma unroll
                for (int jj = 0; jj < 4; ++jj) {
                    const int i = (jj & 1) + 2*(jj & 2);
                    const int qrow = 24*rg + 8*t + (i & 4) + (lane & 2) + (i & 1);
                    s.redsum[ka][qrow] = STUB ? 0.5f : lsum[t][jj];
                    if (ka == 0) s.m[qrow] = STUB ? 0.0f : mrun[t][jj];
                }
        }
        __syncthreads();
        if (nsplit > 1 && tid < nrows) {
            const int tok = tid / G, head = kvh*G + tid % G;
            const float mm = s.m[tid];
            meta[((int64_t) tok*n_head + head)*nsplit + split] = make_float2(mm == -INFINITY ? -INFINITY : mm*LN2, s.redsum[0][tid] + s.redsum[1][tid]);
        }
    } else {
        float O[2][NT][8];
        uint4 vreg[16];
        uint4 kreg[KP > 0 ? KP : 1];
#pragma unroll
        for (int u = 0; u < 2; ++u)
#pragma unroll
            for (int t = 0; t < NT; ++t)
#pragma unroll
                for (int i = 0; i < 8; ++i) O[u][t][i] = 0.0f;
        if (nt > 0) {
            const char * vp = Vbase + (int64_t)(it0*TK)*nbv1;
#pragma unroll
            for (int q = 0; q < 16; ++q, vp += 4*nbv1) vreg[q] = ld_row8<KVQ8, Q8W>(vp, vq);
        }
        if constexpr (KP > 0) {   // the PV group owns the last KP rows of the K tile's load and its store to shared memory
            const int wk = w & 3;
            if (nt > 0) {
                {
                    const char * kp = Kbase + (int64_t)(it0*TK + wk + 4*KQ)*nbk1 + kofs;
#pragma unroll
                    for (int rr = 0; rr < KP; ++rr, kp += 4*nbk1) kreg[rr] = ld_row8<KVQ8, Q8W>(kp, kq);
                }
#pragma unroll
                for (int rr = 0; rr < KP; ++rr) *(uint4 *) &s.K[(4*(KQ + rr) + wk)*KST + 8*lane] = to_f16x8<KVQ8>(kreg[rr]);
                if (nt > 1) {
                    const char * kp = Kbase + (int64_t)((it0 + 1)*TK + wk + 4*KQ)*nbk1 + kofs;
#pragma unroll
                    for (int rr = 0; rr < KP; ++rr, kp += 4*nbk1) kreg[rr] = ld_row8<KVQ8, Q8W>(kp, kq);
                }
            }
        }
        __syncthreads();

        for (int j = 0; j <= nt; ++j) {
            if (j >= 1) {
                const int b = (j - 1) & 1;
                const half * Prow = s.P[b] + (l4 + 4*hi)*PST;
                // rescale O by the tile's factors, when any is not 1
                if (s.resc[b][0] | s.resc[b][1]) {
#pragma unroll
                for (int t = 0; t < NT; ++t)
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const float2 al = *(const float2 *) &s.alpha[b][8*t + 4*h + (lane & 2)];
#pragma unroll
                        for (int u = 0; u < 2; ++u) {
                            O[u][t][4*h + 0] *= al.x; O[u][t][4*h + 1] *= al.y;
                            O[u][t][4*h + 2] *= al.x; O[u][t][4*h + 3] *= al.y;
                        }
                    }
                }
                // O^T += V^T P^T for 64 dims x 48 rows over the tile's 64 keys; V of the next tile is requested as this one is consumed
                const bool has_next = j < nt;
                const char * vp = Vbase + (int64_t)((it0 + j)*TK)*nbv1;
#ifndef FVS_SKIP_PV
#pragma unroll
                for (int c = 0; c < TK/8; ++c) {
                    const uint4 v0 = to_f16x8<KVQ8>(vreg[2*c]), v1 = to_f16x8<KVQ8>(vreg[2*c + 1]);
                    if (has_next) {
                        vreg[2*c]     = ld_row8<KVQ8, Q8W>(vp, vq);
                        vreg[2*c + 1] = ld_row8<KVQ8, Q8W>(vp + 4*nbv1, vq);
                    }
                    vp += 8*nbv1;
#pragma unroll
                    for (int t = 0; t < NT; ++t) {
                        const uint4 pf = *(const uint4 *)(Prow + 8*t*PST + 8*c);
                        mma_cc(O[0][t], v0.x, v0.y, pf.x, pf.y);
                        mma_cc(O[1][t], v0.z, v0.w, pf.x, pf.y);
                        mma_cc(O[0][t], v1.x, v1.y, pf.z, pf.w);
                        mma_cc(O[1][t], v1.z, v1.w, pf.z, pf.w);
                    }
                }
#endif
            }
            if constexpr (KP > 0) {
                if (j + 1 < nt) {
                    const int wk = w & 3;
                    asm volatile("bar.sync 2, 256;" ::: "memory");   // every QK warp is done reading the K tile
#pragma unroll
                    for (int rr = 0; rr < KP; ++rr) *(uint4 *) &s.K[(4*(KQ + rr) + wk)*KST + 8*lane] = to_f16x8<KVQ8>(kreg[rr]);
                    if (j + 2 < nt) {
                        const char * kp = Kbase + (int64_t)((it0 + j + 2)*TK + wk + 4*KQ)*nbk1 + kofs;
#pragma unroll
                        for (int rr = 0; rr < KP; ++rr, kp += 4*nbk1) kreg[rr] = ld_row8<KVQ8, Q8W>(kp, kq);
                    }
                }
            }
            __syncthreads();
        }
        __syncthreads();   // pairs with the QK group's row-sum barrier

#pragma unroll
        for (int u = 0; u < 2; ++u)
#pragma unroll
            for (int t = 0; t < NT; ++t)
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int qrow = 8*t + (i & 4) + (lane & 2) + (i & 1);
                    const int dim  = 64*(w & 3) + 16*qp + 8*hi + 4*u + (lane & 1) + (i & 2);
                    if (qrow < nrows) {
                        const int tok = qrow / G, head = kvh*G + qrow % G;
                        if (nsplit == 1) {
                            dst[((int64_t) tok*n_head + head)*D + dim] = O[u][t][i] / (s.redsum[0][qrow] + s.redsum[1][qrow]);
                        } else {
                            parts[(((int64_t) tok*n_head + head)*nsplit + split)*D + dim] = O[u][t][i];
                        }
                    }
                }
    }
#else
    (void) Q; (void) K; (void) V; (void) mask; (void) dst; (void) parts; (void) meta; (void) n_tok_all; (void) n_head; (void) n_kv_heads;
    (void) nbq1; (void) nbq2; (void) nbk1; (void) nbk2; (void) nbv1; (void) nbv2; (void) nbm1; (void) n_kv; (void) kv_count; (void) nsplit; (void) qscale; (void) nhalf;
    (void) nbq3; (void) nbk3; (void) nbv3; (void) nbm3;
#endif // __CUDA_ARCH__ == 700
}
} // namespace fvs
