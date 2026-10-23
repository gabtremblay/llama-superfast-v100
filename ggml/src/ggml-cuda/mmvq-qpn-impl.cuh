#pragma once

// The kernels and launch templates of mmvq-qpn.cu (the design notes are at the top of that file). Each instantiated launch
// lives in its own file under template-instances/ so that ptxas compiles them in parallel: as one file, this was the
// build's longest serial step (about 10 minutes).

#include "mmvq-qpn.cuh"
#include "qpn-prep.cuh"
#include "dequantize.cuh"
#include "mma.cuh"
#include "allreduce-p2p.cuh"

#include <cfloat>
#include <unordered_set>
#include <cstdlib>

using namespace ggml_cuda_mma;

#define QPN_MAX_WARPS  8   // warps per block (split-K x row tiles)
#define QPN_MIN_BLOCKS 2   // so at most 128 registers: 16 warps per SM
#define QPN_MAX_TILES  16384 // row tiles a product split across blocks may have (the counters)
#define QPN_CHUNK      512 // bytes per chunk: 32 lanes x 16
#define QPN_MAGIC      0x64006400u
#define QPN_FB_N       32  // the fp32 fallback's columns (qpn_fallback), every pass width

template <ggml_type type> struct qpn_t;
template <> struct qpn_t<GGML_TYPE_Q4_K> {
    typedef block_q4_K block;
    static constexpr int  NCH  = 9;
    static constexpr int  TAIL = 0;
    static constexpr bool mins = true;
    static constexpr int  bit  = GGML_CUDA_QPN_Q4_K;
};
template <> struct qpn_t<GGML_TYPE_Q5_K> {
    typedef block_q5_K block;
    static constexpr int  NCH  = 11;
    static constexpr int  TAIL = 0;
    static constexpr bool mins = true;
    static constexpr int  bit  = GGML_CUDA_QPN_Q5_K;
};
template <> struct qpn_t<GGML_TYPE_Q6_K> {
    typedef block_q6_K block;
    static constexpr int  NCH  = 13;
    static constexpr int  TAIL = 64;
    static constexpr bool mins = false;
    static constexpr int  bit  = GGML_CUDA_QPN_Q6_K;
};
template <> struct qpn_t<GGML_TYPE_IQ4_XS> {
    typedef block_iq4_xs block;
    static constexpr int  NCH  = 8;
    static constexpr int  TAIL = 256;
    static constexpr bool mins = false;
    static constexpr int  bit  = GGML_CUDA_QPN_IQ4_XS;
};
// IQ4_NL: the 8 blocks of a 256-column span of one row
struct qpn_iq4_nl_sb { block_iq4_nl b[QK_K/QK4_NL]; };
template <> struct qpn_t<GGML_TYPE_IQ4_NL> {
    typedef qpn_iq4_nl_sb block;
    static constexpr int  NCH  = 9;
    static constexpr int  TAIL = 0;
    static constexpr bool mins = false;
    static constexpr int  bit  = GGML_CUDA_QPN_IQ4_NL;
};
template <> struct qpn_t<GGML_TYPE_Q3_K> {
    typedef block_q3_K block;
    static constexpr int  NCH  = 6;
    static constexpr int  TAIL = 448;
    static constexpr bool mins = false;
    static constexpr int  bit  = GGML_CUDA_QPN_Q3_K;
};
// Q8_0: the 8 blocks of a 256-column span of one row
struct qpn_q8_0_sb { block_q8_0 b[QK_K/QK8_0]; };
template <> struct qpn_t<GGML_TYPE_Q8_0> {
    typedef qpn_q8_0_sb block;
    static constexpr int  NCH  = 17;
    static constexpr int  TAIL = 0;
    static constexpr bool mins = false;
    static constexpr int  bit  = GGML_CUDA_QPN_Q8_0;
};
template <ggml_type type> static constexpr __host__ __device__ int qpn_tsb() { return 32*(int) sizeof(typename qpn_t<type>::block); }
static_assert(qpn_tsb<GGML_TYPE_Q4_K>()   ==  9*QPN_CHUNK,       "Q4_K layout");
static_assert(qpn_tsb<GGML_TYPE_Q5_K>()   == 11*QPN_CHUNK,       "Q5_K layout");
static_assert(qpn_tsb<GGML_TYPE_Q6_K>()   == 13*QPN_CHUNK + 64,  "Q6_K layout");
static_assert(qpn_tsb<GGML_TYPE_IQ4_XS>() ==  8*QPN_CHUNK + 256, "IQ4_XS layout");
static_assert(qpn_tsb<GGML_TYPE_IQ4_NL>() ==  9*QPN_CHUNK,       "IQ4_NL layout");
static_assert(qpn_tsb<GGML_TYPE_Q3_K>()   ==  6*QPN_CHUNK + 448, "Q3_K layout");
static_assert(qpn_tsb<GGML_TYPE_Q8_0>()   == 17*QPN_CHUNK,       "Q8_0 layout");
template <ggml_type type> static constexpr __host__ __device__ bool qpn_iq4() { return type == GGML_TYPE_IQ4_XS || type == GGML_TYPE_IQ4_NL; }
// the types which run only the streaming loop (qpn_stream_loop), at every width
template <ggml_type type> static constexpr __host__ __device__ bool qpn_new() { return type == GGML_TYPE_Q3_K || type == GGML_TYPE_Q8_0; }

// ------------------------------------------------------------------------------------------------------------------
// layout: where each code and high bit goes (used by the repack, the inverse and the fp32 fallback)

// nibble of a code word that holds slot s (column k0 + s)
static __host__ __device__ __forceinline__ int qpn_nib(const int s) { return (s & 1) ? 4 + s/2 : s/2; }
// Q5_K: the bit of sub-block j's high-bit word that holds the fifth bit of word q, slot s
static __host__ __device__ __forceinline__ int qpn_hb5(const int q, const int s) { return (8 + 16*(s & 1) + 4*(s/2) + q) & 31; }
// Q6_K: the lower of the two bits (in high-bit word 2c + q/2) that hold bits 4-5 of chunk c's word q, slot s
static __host__ __device__ __forceinline__ int qpn_hb6(const int q, const int s) { return (8 + 16*(s & 1) + 4*(s/2) + 2*(q & 1)) & 31; }
// Q6_K: position of row r's d in the tail (rows r and r + 2 adjacent, r & 2 == 0)
static __host__ __device__ __forceinline__ int qpn_dpos(const int r) { return (r & ~3) | ((r & 1) << 1) | ((r >> 1) & 1); }

// code of column k (0..255) in a GGUF block
static __device__ __forceinline__ int qpn_code(const block_q4_K * b, const int k) {
    return (b->qs[32*(k/64) + k % 32] >> (4*((k % 64)/32))) & 0xF;
}
static __device__ __forceinline__ int qpn_code(const block_q5_K * b, const int k) {
    return ((b->qs[32*(k/64) + k % 32] >> (4*((k % 64)/32))) & 0xF) | (((b->qh[k % 32] >> (k/32)) & 1) << 4);
}
static __device__ __forceinline__ int qpn_code(const block_q6_K * b, const int k) {
    const int ip = k/128, g = (k % 128)/32, l = k % 32;
    const int lo = (b->ql[64*ip + l + 32*(g & 1)] >> (4*(g/2))) & 0xF;
    const int hi = (b->qh[32*ip + l] >> (2*g)) & 3;
    return lo | (hi << 4);
}
static __device__ __forceinline__ void qpn_set_code(block_q4_K * b, const int k, const int q) {
    uint8_t & v = b->qs[32*(k/64) + k % 32];
    v |= q << (4*((k % 64)/32));
}
static __device__ __forceinline__ void qpn_set_code(block_q5_K * b, const int k, const int q) {
    uint8_t & v = b->qs[32*(k/64) + k % 32];
    v |= (q & 0xF) << (4*((k % 64)/32));
    b->qh[k % 32] |= (q >> 4) << (k/32);
}
static __device__ __forceinline__ void qpn_set_code(block_q6_K * b, const int k, const int q) {
    const int ip = k/128, g = (k % 128)/32, l = k % 32;
    b->ql[64*ip + l + 32*(g & 1)] |= (q & 0xF) << (4*(g/2));
    b->qh[32*ip + l]              |= (q >> 4) << (2*g);
}
// IQ4_XS, IQ4_NL: in each 32-column block, byte l of qs holds column l (low nibble) and column l + 16 (high nibble)
static __device__ __forceinline__ int qpn_code(const block_iq4_xs * b, const int k) {
    return (b->qs[16*(k/32) + k % 16] >> (4*((k % 32)/16))) & 0xF;
}
static __device__ __forceinline__ int qpn_code(const qpn_iq4_nl_sb * b, const int k) {
    return (b->b[k/32].qs[k % 16] >> (4*((k % 32)/16))) & 0xF;
}
static __device__ __forceinline__ void qpn_set_code(block_iq4_xs * b, const int k, const int q) {
    b->qs[16*(k/32) + k % 16] |= q << (4*((k % 32)/16));
}
static __device__ __forceinline__ void qpn_set_code(qpn_iq4_nl_sb * b, const int k, const int q) {
    b->b[k/32].qs[k % 16] |= q << (4*((k % 32)/16));
}
// Q3_K: the 3-bit code c = q2 | h << 2 of column k, the value being d*(s - 32)*(c - 4)
static __device__ __forceinline__ int qpn_code(const block_q3_K * b, const int k) {
    const int q2 = (b->qs[32*(k/128) + k % 32] >> (2*((k % 128)/32))) & 3;
    const int h  = (b->hmask[k % 32] >> (k/32)) & 1;
    return q2 | (h << 2);
}
static __device__ __forceinline__ void qpn_set_code(block_q3_K * b, const int k, const int c) {
    b->qs[32*(k/128) + k % 32] |= (c & 3) << (2*((k % 128)/32));
    b->hmask[k % 32]           |= (c >> 2) << (k/32);
}
// Q3_K: the bit of low-bit word k/16 that holds bit 0 of column k's code (bit 1 is the next one, mod 32), and the bit of
// high-bit word k/32 that holds its high bit
static __host__ __device__ __forceinline__ int qpn_q3_lpos(const int k) {
    const int s = k % 8, e = 4*((k/8) & 1) + s/2;
    return (2*e + ((k/16) & 1) + 16*(s & 1)) & 31;
}
static __host__ __device__ __forceinline__ int qpn_q3_hpos(const int k) { return (qpn_q3_lpos(k) + 2) & 31; }
// Q3_K: the 16 six-bit sub-block scales from the 12 GGUF scale bytes a0..a2, one per byte (sub-block j: byte j % 4 of
// word j/4), as dequantize_row_q3_K unpacks them
static __device__ __forceinline__ void qpn_q3_scales(const uint32_t a0, const uint32_t a1, const uint32_t a2, uint32_t * s) {
    s[0] = ( a0       & 0x0F0F0F0Fu) | (( a2       & 0x03030303u) << 4);
    s[1] = ( a1       & 0x0F0F0F0Fu) | (((a2 >> 2) & 0x03030303u) << 4);
    s[2] = ((a0 >> 4) & 0x0F0F0F0Fu) | (((a2 >> 4) & 0x03030303u) << 4);
    s[3] = ((a1 >> 4) & 0x0F0F0F0Fu) | (((a2 >> 6) & 0x03030303u) << 4);
}

// one row's GGUF block -> its 16 bytes of every chunk (w[ch*4 + i]) and its tail entry tl (Q6_K: d in tl.x; IQ4_XS:
// the 8-byte header in tl.x, tl.y; Q3_K: the 12 scale bytes in tl.x..tl.z and d in tl.w)
template <ggml_type type>
static __device__ void qpn_pack_row(const typename qpn_t<type>::block * b, uint32_t * w, uint4 & tl) {
    constexpr int NCH = qpn_t<type>::NCH;
    for (int i = 0; i < NCH*4; ++i) {
        w[i] = 0;
    }
    tl = make_uint4(0, 0, 0, 0);
    if constexpr (type == GGML_TYPE_Q3_K) {
        for (int k = 0; k < 256; ++k) {
            const int c = qpn_code(b, k), p = qpn_q3_lpos(k);
            w[k/16]      |= (uint32_t) (c & 1)        << p;
            w[k/16]      |= (uint32_t) ((c >> 1) & 1) << ((p + 1) & 31);
            w[16 + k/32] |= (uint32_t) (c >> 2)       << qpn_q3_hpos(k);
        }
        memcpy(&tl, b->scales, 12);
        memcpy(&tl.w, &b->d, 2);
    } else if constexpr (type == GGML_TYPE_Q8_0) {
        for (int k = 0; k < 256; ++k) {
            w[(k/16)*4 + (k % 16)/4] |= (uint32_t) (uint8_t) (b->b[k/32].qs[k % 32] + 128) << (8*(k % 4));
        }
        for (int j = 0; j < QK_K/QK8_0; ++j) {
            memcpy((char *) (w + 16*4) + 2*j, &b->b[j].d, 2);
        }
    } else if constexpr (qpn_iq4<type>()) {
        for (int k = 0; k < 256; ++k) {
            w[(k/32)*4 + (k % 32)/8] |= (uint32_t) qpn_code(b, k) << (4*(k % 8));
        }
        if constexpr (type == GGML_TYPE_IQ4_XS) {
            memcpy(&tl, b, 8); // d, scales_h, scales_l
        } else {
            for (int j = 0; j < QK_K/QK4_NL; ++j) {
                memcpy((char *) (w + 8*4) + 2*j, &b->b[j].d, 2);
            }
        }
    } else if constexpr (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) {
        memcpy(w, b, 16); // d, dmin, scales
        constexpr int ch0 = type == GGML_TYPE_Q4_K ? 1 : 3;
        for (int k = 0; k < 256; ++k) {
            const int j = k/32, q = (k % 32)/8, s = k % 8, c = qpn_code(b, k);
            w[(ch0 + j)*4 + q] |= (uint32_t) (c & 0xF) << (4*qpn_nib(s));
            if constexpr (type == GGML_TYPE_Q5_K) {
                w[4 + j] |= (uint32_t) (c >> 4) << qpn_hb5(q, s);
            }
        }
    } else {
        for (int k = 0; k < 256; ++k) {
            const int ch = k/32, q = (k % 32)/8, s = k % 8, c = qpn_code(b, k);
            w[ch*4 + q]               |= (uint32_t) (c & 0xF) << (4*qpn_nib(s));
            w[8*4 + 2*ch + q/2]       |= (uint32_t) (c >> 4)  << qpn_hb6(q, s);
        }
        memcpy(w + 12*4, b->scales, 16);
        memcpy(&tl.x, &b->d, 2);
    }
}

// the inverse: one row's chunk words and tail entry -> its GGUF block
template <ggml_type type>
static __device__ void qpn_unpack_row(const uint32_t * w, const uint4 tl, typename qpn_t<type>::block * b) {
    memset(b, 0, sizeof(*b));
    if constexpr (type == GGML_TYPE_Q3_K) {
        for (int k = 0; k < 256; ++k) {
            const int p = qpn_q3_lpos(k);
            const int c = ((w[k/16] >> p) & 1) | (((w[k/16] >> ((p + 1) & 31)) & 1) << 1) | (((w[16 + k/32] >> qpn_q3_hpos(k)) & 1) << 2);
            qpn_set_code(b, k, c);
        }
        memcpy(b->scales, &tl, 12);
        memcpy(&b->d, &tl.w, 2);
    } else if constexpr (type == GGML_TYPE_Q8_0) {
        for (int k = 0; k < 256; ++k) {
            b->b[k/32].qs[k % 32] = (int8_t) (((w[(k/16)*4 + (k % 16)/4] >> (8*(k % 4))) & 0xFF) - 128);
        }
        for (int j = 0; j < QK_K/QK8_0; ++j) {
            memcpy(&b->b[j].d, (const char *) (w + 16*4) + 2*j, 2);
        }
    } else if constexpr (qpn_iq4<type>()) {
        for (int k = 0; k < 256; ++k) {
            qpn_set_code(b, k, (w[(k/32)*4 + (k % 32)/8] >> (4*(k % 8))) & 0xF);
        }
        if constexpr (type == GGML_TYPE_IQ4_XS) {
            memcpy(b, &tl, 8);
        } else {
            for (int j = 0; j < QK_K/QK4_NL; ++j) {
                memcpy(&b->b[j].d, (const char *) (w + 8*4) + 2*j, 2);
            }
        }
    } else if constexpr (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) {
        memcpy(b, w, 16);
        constexpr int ch0 = type == GGML_TYPE_Q4_K ? 1 : 3;
        for (int k = 0; k < 256; ++k) {
            const int j = k/32, q = (k % 32)/8, s = k % 8;
            int c = (w[(ch0 + j)*4 + q] >> (4*qpn_nib(s))) & 0xF;
            if constexpr (type == GGML_TYPE_Q5_K) {
                c |= ((w[4 + j] >> qpn_hb5(q, s)) & 1) << 4;
            }
            qpn_set_code(b, k, c);
        }
    } else {
        for (int k = 0; k < 256; ++k) {
            const int ch = k/32, q = (k % 32)/8, s = k % 8;
            const int c = ((w[ch*4 + q] >> (4*qpn_nib(s))) & 0xF) | (((w[8*4 + 2*ch + q/2] >> qpn_hb6(q, s)) & 3) << 4);
            qpn_set_code(b, k, c);
        }
        memcpy(b->scales, w + 12*4, 16);
        memcpy(&b->d, &tl.x, 2);
    }
}

// where a tile-superblock's tail entry of row r lives (Q6_K: its d, 2 bytes; IQ4_XS: its header, 8 bytes; Q3_K: its 12
// scale bytes at 12r, then its d at 384 + 2*qpn_dpos(r))
template <ggml_type type>
static __device__ __forceinline__ void qpn_store_tail(char * tsb, const int r, const uint4 tl) {
    if constexpr (type == GGML_TYPE_Q6_K) {
        ((uint16_t *) (tsb + qpn_t<type>::NCH*QPN_CHUNK))[qpn_dpos(r)] = (uint16_t) tl.x;
    } else if constexpr (type == GGML_TYPE_IQ4_XS) {
        ((uint2 *) (tsb + qpn_t<type>::NCH*QPN_CHUNK))[r] = make_uint2(tl.x, tl.y);
    } else if constexpr (type == GGML_TYPE_Q3_K) {
        uint32_t * s = (uint32_t *) (tsb + qpn_t<type>::NCH*QPN_CHUNK) + 3*r;
        s[0] = tl.x; s[1] = tl.y; s[2] = tl.z;
        ((uint16_t *) (tsb + qpn_t<type>::NCH*QPN_CHUNK + 384))[qpn_dpos(r)] = (uint16_t) tl.w;
    }
}
template <ggml_type type>
static __device__ __forceinline__ uint4 qpn_load_tail(const char * tsb, const int r) {
    if constexpr (type == GGML_TYPE_Q6_K) {
        return make_uint4(((const uint16_t *) (tsb + qpn_t<type>::NCH*QPN_CHUNK))[qpn_dpos(r)], 0, 0, 0);
    } else if constexpr (type == GGML_TYPE_IQ4_XS) {
        const uint2 h = ((const uint2 *) (tsb + qpn_t<type>::NCH*QPN_CHUNK))[r];
        return make_uint4(h.x, h.y, 0, 0);
    } else if constexpr (type == GGML_TYPE_Q3_K) {
        const uint32_t * s = (const uint32_t *) (tsb + qpn_t<type>::NCH*QPN_CHUNK) + 3*r;
        return make_uint4(s[0], s[1], s[2], ((const uint16_t *) (tsb + qpn_t<type>::NCH*QPN_CHUNK + 384))[qpn_dpos(r)]);
    }
    return make_uint4(0, 0, 0, 0);
}

// ------------------------------------------------------------------------------------------------------------------
// repack (load time) and the fp16 dequantization of a repacked tensor (prefill)

// one block of 32 threads per tile-superblock; src holds the GGUF rows of tiles [0, ntiles) of this pass
template <ggml_type type>
static __global__ void qpn_repack_kernel(const char * __restrict__ src, char * __restrict__ dst, const int nsb) {
    typedef typename qpn_t<type>::block block;
    constexpr int NCH = qpn_t<type>::NCH, TSB = qpn_tsb<type>(), BS = sizeof(block);
    __shared__ block blk[32];
    const int lane = threadIdx.x;
    const int tile = blockIdx.x / nsb, sb = blockIdx.x % nsb;

    // stage the 32 rows' blocks of this superblock (2-byte units, Q6_K blocks are only 2-byte aligned)
    uint16_t * s16 = (uint16_t *) blk;
    for (int i = lane; i < 32*BS/2; i += 32) {
        const int r = i / (BS/2), o = i % (BS/2);
        s16[i] = ((const uint16_t *) (src + ((int64_t) (tile*32 + r)*nsb + sb)*BS))[o];
    }
    __syncwarp();

    uint32_t w[NCH*4];
    uint4    tl;
    qpn_pack_row<type>(&blk[lane], w, tl);

    char * out = dst + ((int64_t) tile*nsb + sb)*TSB;
#pragma unroll
    for (int ch = 0; ch < NCH; ++ch) {
        *(uint4 *) (out + ch*QPN_CHUNK + lane*16) = make_uint4(w[ch*4 + 0], w[ch*4 + 1], w[ch*4 + 2], w[ch*4 + 3]);
    }
    qpn_store_tail<type>(out, lane, tl);
}

// row lane's chunk words of one tile-superblock
template <ggml_type type>
static __device__ __forceinline__ void qpn_load_row(const char * tsb, const int lane, uint32_t * w, uint4 & tl) {
    constexpr int NCH = qpn_t<type>::NCH;
#pragma unroll
    for (int ch = 0; ch < NCH; ++ch) {
        const uint4 v = *(const uint4 *) (tsb + ch*QPN_CHUNK + lane*16);
        w[ch*4 + 0] = v.x; w[ch*4 + 1] = v.y; w[ch*4 + 2] = v.z; w[ch*4 + 3] = v.w;
    }
    tl = qpn_load_tail<type>(tsb, lane);
}

// the 32 values of sub-block s (columns 32s..32s+31) of one GGUF block, each computed with the arithmetic of ggml's
// dequantize_* (dequantize.cuh) for that value: the same float operations in the same order, so the same fp16 after the cast
template <ggml_type type>
static __device__ __forceinline__ void qpn_sub32_to_f32(const typename qpn_t<type>::block & b, const int s, float * v) {
    if constexpr (type == GGML_TYPE_Q4_K) {
        const float dall = __low2half(b.dm);
        const float dmin = __high2half(b.dm);
        uint8_t sc, m;
        get_scale_min_k4(s, b.scales, sc, m);
        const float d = dall * sc; const float mm = dmin * m;
        const uint8_t * q = b.qs + 32*(s/2);
#pragma unroll
        for (int l = 0; l < 32; ++l) {
            v[l] = (s & 1) ? d * (q[l] >> 4) - mm : d * (q[l] & 0xF) - mm;
        }
    } else if constexpr (type == GGML_TYPE_Q5_K) {
        const float dall = __low2half(b.dm);
        const float dmin = __high2half(b.dm);
        uint8_t sc, m;
        get_scale_min_k4(s, b.scales, sc, m);
        const float d = dall * sc; const float mm = dmin * m;
        const uint8_t * ql = b.qs + 32*(s/2);
        const uint8_t   hm = 1 << s;
#pragma unroll
        for (int l = 0; l < 32; ++l) {
            v[l] = (s & 1) ? d * ((ql[l] >> 4) + (b.qh[l] & hm ? 16 : 0)) - mm
                           : d * ((ql[l] & 0xF) + (b.qh[l] & hm ? 16 : 0)) - mm;
        }
    } else if constexpr (type == GGML_TYPE_Q6_K) {
        const float d  = b.d;
        const int   ip = s/4, q4 = s%4;
        const uint8_t * ql = b.ql + 64*ip + 32*(q4 & 1);
        const uint8_t * qh = b.qh + 32*ip;
#pragma unroll
        for (int l = 0; l < 32; ++l) {
            const int8_t * sc = b.scales + 8*ip + l/16 + 2*q4;
            const int nib = q4 >= 2 ? (ql[l] >> 4) : (ql[l] & 0xF);
            v[l] = d * sc[0] * ((int8_t)(nib | (((qh[l] >> (2*q4)) & 3) << 4)) - 32);
        }
    } else if constexpr (type == GGML_TYPE_IQ4_XS) {
        const float d = (float)b.d * ((((b.scales_l[s/2] >> 4*(s%2)) & 0xf) | (((b.scales_h >> 2*s) & 3) << 4)) - 32);
        const uint8_t * q4 = b.qs + 16*s;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            v[j +  0] = d * kvalues_iq4nl[q4[j] & 0xf];
            v[j + 16] = d * kvalues_iq4nl[q4[j] >>  4];
        }
    }
}

template <ggml_type type>
static constexpr bool qpn_has_fast_expand() {
    return type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K || type == GGML_TYPE_Q6_K || type == GGML_TYPE_IQ4_XS;
}

// the codes of columns 32j..32j+31 of one row (one chunk word set), column l = 8q + s in word q at nibble pos(s)
template <ggml_type type>
static __device__ __forceinline__ int qpn_code4(const uint32_t * c4, const int l) {
    const int q = l/8, s = l % 8;
    return (c4[q] >> (4*(qpn_iq4<type>() ? s : qpn_nib(s)))) & 0xF;
}

// the prefill expansion of a repacked weight (ggml_cuda_qpn_to_fp16), straight from the repacked layout: thread (r, j) reads
// row r's chunk word of sub-block j (columns 32j..32j+31) and that row's scales, computes the 32 values with the arithmetic of
// ggml's dequantize_* (dequantize.cuh; the same float operations in the same order, as qpn_sub32_to_f32), and writes them as
// four 16-byte stores, so every row goes out as whole 512-byte runs. Rebuilding the GGUF blocks first (one warp, 256 serial
// steps per row, the other warps waiting) ran at a third of the memory bandwidth: 0.8 ms per 27B FFN weight
template <ggml_type type>
static __global__ void __launch_bounds__(256) qpn_to_fp16_fast_kernel(const char * __restrict__ W, half * __restrict__ y, const int nsb, const int64_t K) {
    constexpr int TSB = qpn_tsb<type>();
    const int tile = blockIdx.x / nsb, sb = blockIdx.x % nsb;
    const int r = threadIdx.x / 8, j = threadIdx.x % 8;
    const char * tsb = W + ((int64_t) tile*nsb + sb)*TSB;
    float v[32];
    if constexpr (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) {
        constexpr int ch0 = type == GGML_TYPE_Q4_K ? 1 : 3;
        const uint4 hd = *(const uint4 *) (tsb + r*16); // d, dmin, scales
        const uint4 cw = *(const uint4 *) (tsb + (ch0 + j)*QPN_CHUNK + r*16);
        const uint32_t c4[4] = {cw.x, cw.y, cw.z, cw.w};
        half2 dm;
        memcpy(&dm, &hd.x, 4);
        const uint32_t sw[3] = {hd.y, hd.z, hd.w};
        const float dall = __low2half(dm);
        const float dmin = __high2half(dm);
        uint8_t sc, m;
        get_scale_min_k4(j, (const uint8_t *) sw, sc, m);
        const float d = dall * sc; const float mm = dmin * m;
        if constexpr (type == GGML_TYPE_Q5_K) {
            const uint32_t hw = *(const uint32_t *) (tsb + (1 + j/4)*QPN_CHUNK + r*16 + 4*(j % 4)); // w[4 + j]
#pragma unroll
            for (int l = 0; l < 32; ++l) {
                v[l] = d * (qpn_code4<type>(c4, l) + ((hw >> qpn_hb5(l/8, l % 8)) & 1 ? 16 : 0)) - mm;
            }
        } else {
#pragma unroll
            for (int l = 0; l < 32; ++l) {
                v[l] = d * qpn_code4<type>(c4, l) - mm;
            }
        }
    } else if constexpr (type == GGML_TYPE_Q6_K) {
        const uint4 cw = *(const uint4 *) (tsb + j*QPN_CHUNK + r*16);
        const uint32_t c4[4] = {cw.x, cw.y, cw.z, cw.w};
        const uint2 hw = *(const uint2 *) (tsb + (8 + j/2)*QPN_CHUNK + r*16 + 4*((2*j) % 4)); // w[32 + 2j], w[33 + 2j]
        const uint4 sv = *(const uint4 *) (tsb + 12*QPN_CHUNK + r*16);                         // the 16 scales
        const int8_t * scales = (const int8_t *) &sv;
        const uint16_t dh = (uint16_t) qpn_load_tail<type>(tsb, r).x;
        const float d = __ushort_as_half(dh);
        const int ip = j/4, q4 = j % 4;
#pragma unroll
        for (int l = 0; l < 32; ++l) {
            const int q = l/8, s = l % 8;
            const uint32_t h = q < 2 ? hw.x : hw.y;
            const int c = qpn_code4<type>(c4, l) | (((h >> qpn_hb6(q, s)) & 3) << 4);
            const int8_t * sc = scales + 8*ip + l/16 + 2*q4;
            v[l] = d * sc[0] * ((int8_t) c - 32);
        }
    } else if constexpr (type == GGML_TYPE_IQ4_XS) {
        const uint4 cw = *(const uint4 *) (tsb + j*QPN_CHUNK + r*16);
        const uint32_t c4[4] = {cw.x, cw.y, cw.z, cw.w};
        const uint4 tl = qpn_load_tail<type>(tsb, r);
        block_iq4_xs b;
        memcpy(&b, &tl, 8); // d, scales_h, scales_l
        const float d = (float)b.d * ((((b.scales_l[j/2] >> 4*(j%2)) & 0xf) | (((b.scales_h >> 2*j) & 3) << 4)) - 32);
#pragma unroll
        for (int l = 0; l < 32; ++l) {
            v[l] = d * kvalues_iq4nl[qpn_code4<type>(c4, l)];
        }
    }
    uint4 * dst = (uint4 *) (y + (int64_t) (tile*32 + r)*K + sb*QK_K + 32*j);
#pragma unroll
    for (int c = 0; c < 4; ++c) {
        half2 h[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            h[k] = make_half2(ggml_cuda_cast<half>(v[8*c + 2*k]), ggml_cuda_cast<half>(v[8*c + 2*k + 1]));
        }
        dst[c] = *(const uint4 *) h;
    }
}

// Rebuilds the GGUF blocks of one tile-superblock in shared memory and dequantizes them with ggml's own functions
// (dequantize.cuh), so the values are those ggml's to_fp16 gives for the original tensor.
template <ggml_type type>
static __global__ void qpn_to_fp16_kernel(const char * __restrict__ W, half * __restrict__ y, const int nsb, const int64_t K) {
    typedef typename qpn_t<type>::block block;
    constexpr int NCH = qpn_t<type>::NCH, TSB = qpn_tsb<type>();
    __shared__ block blk[32];
    const int tile = blockIdx.x / nsb, sb = blockIdx.x % nsb;
    if (threadIdx.x < 32) {
        uint32_t w[NCH*4];
        uint4    tl;
        qpn_load_row<type>(W + ((int64_t) tile*nsb + sb)*TSB, threadIdx.x, w, tl);
        qpn_unpack_row<type>(w, tl, &blk[threadIdx.x]);
    }
    __syncthreads();
    for (int i = 0; i < 32; ++i) {
        half * yi = y + (int64_t) (tile*32 + i)*K + sb*QK_K;
        if constexpr (type == GGML_TYPE_Q4_K) {
            dequantize_q4_K(blk, i, yi, threadIdx.x); // 32 threads
        } else if constexpr (type == GGML_TYPE_Q5_K) {
            dequantize_q5_K(blk, i, yi, threadIdx.x); // 64 threads
        } else if constexpr (type == GGML_TYPE_Q6_K) {
            dequantize_q6_K(blk, i, yi, threadIdx.x); // 64 threads
        } else if constexpr (type == GGML_TYPE_IQ4_XS) {
            dequantize_iq4_xs(blk, i, yi, threadIdx.x); // 32 threads
        } else if constexpr (type == GGML_TYPE_Q3_K) {
            dequantize_q3_K(blk, i, yi, threadIdx.x); // 64 threads
        } else if constexpr (type == GGML_TYPE_Q8_0) {
            // 32 threads, 8 columns each; the arithmetic of dequantize_block_q8_0_f16 (convert.cu), ggml's Q8_0 -> fp16
            const block_q8_0 & bq = blk[i].b[threadIdx.x/4];
            const half2 d2 = __half2half2(bq.d);
#pragma unroll
            for (int l = 0; l < 8; l += 2) {
                const int c = 8*(threadIdx.x % 4) + l;
                ((half2 *) yi)[(8*threadIdx.x + l)/2] = __hmul2(make_half2(bq.qs[c], bq.qs[c + 1]), d2);
            }
        } else {
            dequantize_iq4_nl(blk, i, yi, threadIdx.x); // 32 threads; block i*8 .. i*8 + 7
        }
    }
}

// ------------------------------------------------------------------------------------------------------------------
// the product

static __device__ __forceinline__ half2 qpn_u2h(const uint32_t u) { return *(const half2 *) &u; }

// weights: read once, keep them out of L1 (the activations live there)
static __device__ __forceinline__ uint4 qpn_ldg_w(const char * p) {
    uint4 r;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];"
        : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
    return r;
}
static __device__ __forceinline__ uint2 qpn_ldg_w2(const char * p) {
    uint2 r;
    asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0, %1}, [%2];" : "=r"(r.x), "=r"(r.y) : "l"(p));
    return r;
}
// the 128-byte line holding p into L2, for a later load (no registers)
static __device__ __forceinline__ void qpn_prefetch_l2(const char * p) {
    asm volatile("prefetch.global.L2 [%0];" :: "l"(p));
}
static __device__ __forceinline__ uint4 qpn_ldg_x(const half * p) {
    return __ldg((const uint4 *) p);
}
// half2 (b, b) of byte i of w, exact: 0x64 above the byte is 1024 + b
static __device__ __forceinline__ half2 qpn_byte_h2(const uint32_t w, const int i) {
    return __hsub2(qpn_u2h(__byte_perm(w, 0x64, (i) | (4 << 4) | ((i) << 8) | (4 << 12))), qpn_u2h(QPN_MAGIC));
}
// half2 (byte i, byte i + 1) of w, exact
static __device__ __forceinline__ half2 qpn_bytes_h2(const uint32_t w, const int i) {
    return __hsub2(qpn_u2h(__byte_perm(w, 0x64, (i) | (4 << 4) | ((i + 1) << 8) | (4 << 12))), qpn_u2h(QPN_MAGIC));
}

// the A fragment of one code word: sc*q (Q4_K, Q5_K) or sc*(q - 32) (Q6_K) for its 8 columns, from s16 = sc/16 and
// z = -64*sc or -96*sc; hw holds the high bits of this word already rotated by its word offset
// mg: QPN_MAGIC. The streaming loop passes it from a register the compiler cannot fold into an immediate, so that
// "x & mask | magic" is one LOP3 (two register operands, one immediate) and a half2's two ORs are two LOP3 instead of three; same values
template <ggml_type type>
static __device__ __forceinline__ void qpn_decode(tile<32, 4, half2> & A, const uint32_t w, const uint32_t hw, const half2 s16, const half2 z,
        const uint32_t mg = QPN_MAGIC) {
    uint32_t g[4];
    if constexpr (type == GGML_TYPE_Q4_K) {
        g[0] = g[1] = g[2] = g[3] = mg;
    } else {
        constexpr uint32_t hm = type == GGML_TYPE_Q5_K ? 0x01000100u : 0x03000300u;
        g[0] = ( hw                          & hm) | mg;
        g[1] = (__funnelshift_r(hw, hw,  4)  & hm) | mg;
        g[2] = (__funnelshift_r(hw, hw,  8)  & hm) | mg;
        g[3] = (__funnelshift_r(hw, hw, 12)  & hm) | mg;
    }
    A.x[0] = __hfma2(qpn_u2h(((w << 4) & 0x00F000F0u) | g[0]), s16, z);
    A.x[1] = __hfma2(qpn_u2h(( w       & 0x00F000F0u) | g[1]), s16, z);
    A.x[2] = __hfma2(qpn_u2h(((w >> 4) & 0x00F000F0u) | g[2]), s16, z);
    A.x[3] = __hfma2(qpn_u2h(((w >> 8) & 0x00F000F0u) | g[3]), s16, z);
}

// PRMT as the hardware does it: selector bit 3 fills the byte with the selected byte's sign (__byte_perm clears it)
static __device__ __forceinline__ uint32_t qpn_prmt(const uint32_t a, const uint32_t b, const uint32_t c) {
    uint32_t r;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    return r;
}

// kvalues_iq4nl + 128 as bytes: codes 0-7 (all below 128) and 8-15
#define QPN_IQ4_T0 0x3F2D1801u
#define QPN_IQ4_T1 0x766A5D4Fu
#define QPN_IQ4_T2 0xA6998D81u
#define QPN_IQ4_T3 0xF1D9C5B5u

// the 4 codes of s's low 16 bits -> their kv + 128 as bytes, in order. sm = s without bit 3 of each code, sel = the
// codes' bit 3 at bit 2 over 0x3210 (the low half of the codebook where it is 0, from lo; else from hi)
static __device__ __forceinline__ uint32_t qpn_iq4_lookup(const uint32_t s, const uint32_t sm, const uint32_t sel) {
    const uint32_t lo = qpn_prmt(QPN_IQ4_T0, QPN_IQ4_T1, s); // a code with bit 3 set gets a sign fill here, never taken
    const uint32_t hi = qpn_prmt(QPN_IQ4_T2, QPN_IQ4_T3, sm);
    return qpn_prmt(lo, hi, sel);
}

// the A fragment of one IQ4 code word (columns k0..k0+7 in nibbles 0..7): 1024 + 128 + kv as fp16, exactly, then
// IQ4_XS: kv*s by one HFMA2 with S = s and Z = -1152*s; IQ4_NL: kv*d by HSUB2 and HMUL2 with S = d
template <ggml_type type>
static __device__ __forceinline__ void qpn_decode_iq4(tile<32, 4, half2> & A, const uint32_t w, const half2 S, const half2 Z) {
    const uint32_t wm  = w & 0x77777777u;
    const uint32_t sel = ((w >> 1) & 0x44444444u) | 0x32103210u;
    const uint32_t ra  = qpn_iq4_lookup(w,       wm,       sel);
    const uint32_t rb  = qpn_iq4_lookup(w >> 16, wm >> 16, sel >> 16);
    const uint32_t x[4] = {
        __byte_perm(ra, 0x64646464u, 0x4140), __byte_perm(ra, 0x64646464u, 0x4342),
        __byte_perm(rb, 0x64646464u, 0x4140), __byte_perm(rb, 0x64646464u, 0x4342),
    };
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        if constexpr (type == GGML_TYPE_IQ4_XS) {
            A.x[i] = __hfma2(qpn_u2h(x[i]), S, Z);
        } else {
            A.x[i] = __hmul2(__hsub2(qpn_u2h(x[i]), make_half2(1152.0f, 1152.0f)), S);
        }
    }
}

// Partial fp32 dot products, over this warp's superblocks, for the columns whose token held a non-finite value: the
// dequantized weights (the same expressions as dequantize.cuh) times the fp32 activations. lane = row; fb[c] for bad c.
template <ggml_type type>
static __device__ __noinline__ void qpn_fallback(
        const char * __restrict__ Wt, const int sb0, const int sb1, const int S, const unsigned badmask, const int T,
        const float * __restrict__ x, const int64_t stride_col_x, float * fb) {
    typedef typename qpn_t<type>::block block;
    constexpr int NCH = qpn_t<type>::NCH, TSB = qpn_tsb<type>();
    const int lane = threadIdx.x;
    for (int c = 0; c < QPN_FB_N; ++c) { // up to QPN_FB_N columns
        fb[c] = 0.0f;
    }
    for (int sb = sb0; sb < sb1; sb += S) {
        uint32_t w[NCH*4];
        uint4    tl;
        block b;
        qpn_load_row<type>(Wt + (int64_t) sb*TSB, lane, w, tl);
        qpn_unpack_row<type>(w, tl, &b);
        float v[QK_K];
        if constexpr (type == GGML_TYPE_IQ4_XS) {
            for (int j = 0; j < 8; ++j) {
                const float d = (float) b.d * ((((b.scales_l[j/2] >> 4*(j%2)) & 0xf) | (((b.scales_h >> 2*j) & 3) << 4)) - 32);
                for (int l = 0; l < 32; ++l) {
                    v[32*j + l] = d * kvalues_iq4nl[qpn_code(&b, 32*j + l)];
                }
            }
        } else if constexpr (type == GGML_TYPE_IQ4_NL) {
            for (int k = 0; k < QK_K; ++k) {
                v[k] = (float) b.b[k/32].d * kvalues_iq4nl[qpn_code(&b, k)];
            }
        } else if constexpr (type == GGML_TYPE_Q3_K) { // dequantize_row_q3_K: (d*(s - 32))*(c - 4)
            uint32_t a[3], s[4];
            memcpy(a, b.scales, 12);
            qpn_q3_scales(a[0], a[1], a[2], s);
            const float d = b.d;
            for (int k = 0; k < QK_K; ++k) {
                const int sc = (s[k/64] >> (8*((k/16) % 4))) & 0xFF;
                v[k] = (d * (sc - 32)) * (qpn_code(&b, k) - 4);
            }
        } else if constexpr (type == GGML_TYPE_Q8_0) { // dequantize_row_q8_0: q*d
            for (int k = 0; k < QK_K; ++k) {
                v[k] = b.b[k/32].qs[k % 32] * (float) b.b[k/32].d;
            }
        } else if constexpr (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) {
            const float dall = __low2half(b.dm), dmin = __high2half(b.dm);
            for (int j = 0; j < 8; ++j) {
                uint8_t sc, m;
                get_scale_min_k4(j, b.scales, sc, m);
                const float d1 = dall * sc, m1 = dmin * m;
                for (int l = 0; l < 32; ++l) {
                    v[32*j + l] = d1 * qpn_code(&b, 32*j + l) - m1;
                }
            }
        } else {
            const float d = b.d;
            for (int k = 0; k < QK_K; ++k) {
                v[k] = d * b.scales[k/16] * ((int8_t) qpn_code(&b, k) - 32);
            }
        }
        for (int c = 0; c < T; ++c) {
            if ((badmask >> c) & 1) {
                const float * xc = x + c*stride_col_x + (int64_t) sb*QK_K;
                float acc = 0.0f;
                for (int k = 0; k < QK_K; ++k) {
                    acc += v[k] * xc[k];
                }
                fb[c] += acc;
            }
        }
    }
}

// C elements of a lane that can hold a real token at T tokens: one 8-token B tile (4 or 8), two at T = 16 (9 to 16
// tokens, the second tile's at 8 .. 15), or four at T = 32 (17 to 32 tokens, tile b's at 8b .. 8b + 7)
template <int T> static constexpr __host__ __device__ int qpn_nl() { return T > 16 ? 32 : T > 8 ? 16 : T > 4 ? 8 : 4; }
// the output column of C element l: c0 + (l & 5) within its 8-token tile, + 8 per tile (l >> 3)
static constexpr __host__ __device__ int qpn_col(const int l) { return (l & 5) + (l & ~7); }
// the column scale of C element l in cs[] (NL/2 entries: two per 4 columns)
static constexpr __host__ __device__ int qpn_csi(const int l) { return (l & 1) + ((l & 4) >> 1) + 4*(l >> 3); }


// One superblock of one row tile: buf holds this lane's 16 bytes of every chunk (dp: Q6_K's d of rows r0, r0 + 2 in
// dp.x; IQ4_XS: the lane's row's header),
// xk this superblock's activation fragments of token nB (fragment of column group g at xk[g*T]; shared memory if
// STAGE). Adds the superblock's contribution to the lane's C elements D and marks tokens whose slice is not finite.
// A register ring: as soon as a chunk of buf (or dp) has been read for the last time, the same chunk of
// the next superblock (at Wn; if more) is requested into it, so each load has about one superblock of lead with no
// second buffer. The arithmetic is unchanged, in the same order. Without RING the caller holds a second
// buffer and loads the whole next superblock itself, as before.
#define QPN_NEXT(c) do { if (RING && more) { buf[c] = qpn_ldg_w(Wn + (c)*QPN_CHUNK); } } while (0)
template <ggml_type type> static constexpr __host__ __device__ int qpn_ring_min_t() {
    // measured (microbenchmark on the real tensors, card 1, SM clock 1380 MHz): Q6_K's spills go (23-32
    // LDL in the loop -> 1-6) and every shape is 22-40% faster at T = 3, 4, 8; Q5_K is 9-25% faster at T = 8 but up
    // to 15% slower at T = 3, 4 (ffn down, GDN qkv, GDN out); Q4_K and IQ4_XS are 0-6% slower
    return type == GGML_TYPE_Q6_K ? 1 : type == GGML_TYPE_Q5_K ? 5 : 99;
}
template <ggml_type type, int T, int NL, bool STAGE, bool RING>
static __device__ __forceinline__ void qpn_superblock(
        uint4 * buf, uint2 & dp, const char * Wn, const bool more, const int dofs, const int sb, const uint4 * xk,
        const half * __restrict__ xs, const float * __restrict__ xsc, const int nB, const bool hasB, const int r0,
        const int c0, float * D, unsigned & bad) {
#if defined(VOLTA_MMA_AVAILABLE)
    typedef tile<32, 4, half2>                               tile_A;
    typedef tile< 8, 4, half2, DATA_LAYOUT_I_MAJOR_MIRRORED> tile_B;
    typedef tile<32, 8, float>                               tile_C;

    // this superblock's column scales (0: the token's slice is not finite)
    float cs[NL/2];
#pragma unroll
    for (int i = 0; i < NL/2; ++i) {
        const int col = c0 + (i & 1) + 4*(i/2);
        cs[i] = col < T ? xsc[sb*T + col] : 1.0f;
        bad |= (col < T && cs[i] == 0.0f) ? 1u << col : 0u;
    }

    tile_C Ds[2];

    if constexpr (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) {
        constexpr int ch0 = type == GGML_TYPE_Q4_K ? 1 : 3;
        const uint4 hdr = buf[0];
        uint4 qh[2] = {make_uint4(0, 0, 0, 0), make_uint4(0, 0, 0, 0)};
        if constexpr (type == GGML_TYPE_Q5_K) {
            qh[0] = buf[1];
            qh[1] = buf[2];
        }
        // 6-bit scales and mins, a byte each (get_scale_min_k4)
        const uint32_t scl[2] = { hdr.y & 0x3F3F3F3Fu, (hdr.w & 0x0F0F0F0Fu) | ((hdr.y >> 2) & 0x30303030u) };
        const uint32_t mnl[2] = { hdr.z & 0x3F3F3F3Fu, ((hdr.w >> 4) & 0x0F0F0F0Fu) | ((hdr.z >> 2) & 0x30303030u) };
        // RING: the rows' d and dmin now, so the header chunk can be requested for the next superblock
        float2 dm0, dm2;
        if constexpr (RING) {
            dm0 = __half22float2(qpn_u2h(__shfl_sync(0xFFFFFFFF, hdr.x, r0)));
            dm2 = __half22float2(qpn_u2h(__shfl_sync(0xFFFFFFFF, hdr.x, r0 + 2)));
        }
        QPN_NEXT(0);

#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const uint4 cw = buf[ch0 + j];
            const half2 sc  = qpn_byte_h2(scl[j/4], j % 4);
            const half2 s16 = __hmul2(sc, make_half2(0.0625f, 0.0625f));
            const half2 z   = __hmul2(sc, make_half2(-64.0f, -64.0f));
            uint32_t H = 0;
            if constexpr (type == GGML_TYPE_Q5_K) {
                const uint4 hq = qh[j/4];
                H = (j % 4) == 0 ? hq.x : (j % 4) == 1 ? hq.y : (j % 4) == 2 ? hq.z : hq.w;
            }
            const uint32_t cwv[4] = {cw.x, cw.y, cw.z, cw.w};
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                tile_A A;
                qpn_decode<type>(A, cwv[q], __funnelshift_r(H, H, q), s16, z);
                tile_B B; // a lane of a padding token (nB >= T) feeds only output columns that are never written
                *(uint4 *) B.x = STAGE ? xk[(4*j + q)*T] : __ldg(xk + (4*j + q)*T);
                mma(Ds[q & 1], A, B);
            }
            QPN_NEXT(ch0 + j);
            if constexpr (type == GGML_TYPE_Q5_K) {
                if (j % 4 == 3) {
                    QPN_NEXT(1 + j/4);
                }
            }
        }

        // mins: sum_j m_j * (per-32 sums of the activations)/32 through one more mma
        tile_A Am;
        Am.x[0] = qpn_bytes_h2(mnl[0], 0);
        Am.x[1] = qpn_bytes_h2(mnl[0], 2);
        Am.x[2] = qpn_bytes_h2(mnl[1], 0);
        Am.x[3] = qpn_bytes_h2(mnl[1], 2);
        tile_B Bm;
        *(uint4 *) Bm.x = __ldg((const uint4 *) xs + ((int64_t) sb*T + nB));
        tile_C M;
        mma(M, Am, Bm);

        if constexpr (!RING) {
            dm0 = __half22float2(qpn_u2h(__shfl_sync(0xFFFFFFFF, hdr.x, r0)));
            dm2 = __half22float2(qpn_u2h(__shfl_sync(0xFFFFFFFF, hdr.x, r0 + 2)));
        }
#pragma unroll
        for (int l = 0; l < NL; ++l) {
            const float2 dm = (l & 2) ? dm2 : dm0;
            const float  c  = cs[(l & 1) + ((l & 4) >> 1)];
            D[l] += c * (dm.x * (Ds[0].x[l] + Ds[1].x[l]) - 32.0f*dm.y * M.x[l]);
        }
    } else if constexpr (type == GGML_TYPE_Q6_K) {
        const uint4 scw = buf[12];
        // int8 scales, offset to unsigned so the 0x64 byte trick applies: 1024 + sc + 128
        const uint32_t scu[4] = {scw.x ^ 0x80808080u, scw.y ^ 0x80808080u, scw.z ^ 0x80808080u, scw.w ^ 0x80808080u};
        uint4 hq[4];
#pragma unroll
        for (int h = 0; h < 4; ++h) {
            hq[h] = buf[8 + h];
        }
        float2 d02;
        if constexpr (RING) {
            d02 = __half22float2(qpn_u2h(dp.x));
        }
        QPN_NEXT(12);
        if (RING && more) {
            dp.x = __ldg((const uint32_t *) Wn + dofs);
        }
#pragma unroll
        for (int c = 0; c < 8; ++c) {
            const uint4 cw = buf[c];
            const uint32_t cwv[4] = {cw.x, cw.y, cw.z, cw.w};
            const uint4    hc     = hq[c/2];
            const uint32_t hcv[2] = {(c & 1) ? hc.z : hc.x, (c & 1) ? hc.w : hc.y};
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int   sbi = 2*c + q/2; // 16-column sub-block
                const half2 sc  = __hsub2(qpn_u2h(__byte_perm(scu[sbi/4], 0x64, (sbi % 4) | (4 << 4) | ((sbi % 4) << 8) | (4 << 12))),
                                          make_half2(1152.0f, 1152.0f));
                const half2 s16 = __hmul2(sc, make_half2(0.0625f, 0.0625f));
                const half2 z   = __hmul2(sc, make_half2(-96.0f, -96.0f));
                const uint32_t H = hcv[q/2];
                tile_A A;
                qpn_decode<type>(A, cwv[q], __funnelshift_r(H, H, 2*(q & 1)), s16, z);
                tile_B B;
                *(uint4 *) B.x = STAGE ? xk[(4*c + q)*T] : __ldg(xk + (4*c + q)*T);
                mma(Ds[q & 1], A, B);
            }
            QPN_NEXT(c);
            if (c & 1) {
                QPN_NEXT(8 + c/2);
            }
        }
        if constexpr (!RING) {
            d02 = __half22float2(qpn_u2h(dp.x));
        }
#pragma unroll
        for (int l = 0; l < NL; ++l) {
            const float d = (l & 2) ? d02.y : d02.x;
            const float c = cs[(l & 1) + ((l & 4) >> 1)];
            D[l] += c * d * (Ds[0].x[l] + Ds[1].x[l]);
        }
    } else { // IQ4_XS, IQ4_NL
        // IQ4_XS: the 8 six-bit sub-block scales ls_j of this lane's row as bytes, even j in ls02, odd j in ls13
        // (dp.x = d | scales_h << 16, dp.y = scales_l); IQ4_NL: its 8 fp16 block scales
        uint32_t ls02 = 0, ls13 = 0;
        uint4    dnl  = make_uint4(0, 0, 0, 0);
        if constexpr (type == GGML_TYPE_IQ4_XS) {
            const uint32_t h = __byte_perm(dp.x, 0, 0x3322); // scales_h bytes 0, 0, 1, 1
            ls02 = ( dp.y       & 0x0F0F0F0Fu) | ((h << 4) & 0x00300030u) | ( h       & 0x30003000u);
            ls13 = ((dp.y >> 4) & 0x0F0F0F0Fu) | ((h << 2) & 0x00300030u) | ((h >> 2) & 0x30003000u);
        } else {
            dnl = buf[8];
        }
        float d0 = 1.0f, d2 = 1.0f;
        if constexpr (type == GGML_TYPE_IQ4_XS && RING) {
            d0 = __low2float(qpn_u2h(__shfl_sync(0xFFFFFFFF, dp.x, r0)));
            d2 = __low2float(qpn_u2h(__shfl_sync(0xFFFFFFFF, dp.x, r0 + 2)));
            if (more) {
                dp = qpn_ldg_w2((const char *) ((const uint32_t *) Wn + dofs));
            }
        } else if constexpr (type == GGML_TYPE_IQ4_NL) {
            QPN_NEXT(8);
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const uint4 cw = buf[j];
            half2 S, Z;
            if constexpr (type == GGML_TYPE_IQ4_XS) {
                const int b = j/2;
                S = __hsub2(qpn_u2h(__byte_perm(j & 1 ? ls13 : ls02, 0x64, b | (4 << 4) | (b << 8) | (4 << 12))),
                            make_half2(1056.0f, 1056.0f)); // ls_j - 32, exact
                Z = __hmul2(S, make_half2(-1152.0f, -1152.0f));
            } else {
                const uint32_t dw = (j/2) == 0 ? dnl.x : (j/2) == 1 ? dnl.y : (j/2) == 2 ? dnl.z : dnl.w;
                S = qpn_u2h(__byte_perm(dw, 0, (j & 1) ? 0x3232 : 0x1010));
                Z = S;
            }
            const uint32_t cwv[4] = {cw.x, cw.y, cw.z, cw.w};
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                tile_A A;
                qpn_decode_iq4<type>(A, cwv[q], S, Z);
                tile_B B;
                *(uint4 *) B.x = STAGE ? xk[(4*j + q)*T] : __ldg(xk + (4*j + q)*T);
                mma(Ds[q & 1], A, B);
            }
            QPN_NEXT(j);
        }
        if constexpr (type == GGML_TYPE_IQ4_XS && !RING) {
            d0 = __low2float(qpn_u2h(__shfl_sync(0xFFFFFFFF, dp.x, r0)));
            d2 = __low2float(qpn_u2h(__shfl_sync(0xFFFFFFFF, dp.x, r0 + 2)));
        }
#pragma unroll
        for (int l = 0; l < NL; ++l) {
            const float c = cs[(l & 1) + ((l & 4) >> 1)];
            if constexpr (type == GGML_TYPE_IQ4_XS) {
                D[l] += c * ((l & 2) ? d2 : d0) * (Ds[0].x[l] + Ds[1].x[l]);
            } else {
                D[l] += c * (Ds[0].x[l] + Ds[1].x[l]);
            }
        }
    }
#else
    GGML_UNUSED_VARS(buf, dp, Wn, more, dofs, sb, xk, xs, xsc, nB, hasB, r0, c0, D, bad);
    NO_DEVICE_CODE;
#endif // defined(VOLTA_MMA_AVAILABLE)
}

// grid = (row tile groups, P), blockDim = (32, R*S). Block (x, p) takes row tiles x*R .. x*R + R - 1 over superblocks
// [nsb*p/P, nsb*(p + 1)/P); its warp y takes tile x*R + y/S and every S-th superblock of that range from y%S on.
// STAGE: the block's activations are copied to shared memory first. The S warps of a tile add up in shared memory;
// with P > 1 the P blocks of a tile leave their partial sums in ws and the last one to finish (a per-stream, per-tile
// counter) adds them in the order p = 0 .. P-1, so the result does not depend on which block finishes last.
// One register buffer: each chunk of the next superblock is requested once the current one has been read
// (inside qpn_superblock, chunk by chunk).
// Streaming: the superblock loop with only what a superblock needs
// throughout (the header or scale chunk, the tail entry) and a lookahead of two code chunks and one high-bit chunk in
// registers (about 24 weight registers instead of 88), so the kernel fits 64-80 registers and 24-32 warps per SM; latency
// is hidden with warps, not with register buffers. The arithmetic is qpn_superblock's, expression for expression and in
// the same order, so at the same split the outputs are bit-identical.
//   Q4_K, Q5_K: header chunk 0, Q5_K's high-bit chunks 1-2 (one per 4 code chunks), codes from chunk 1 or 3
//   Q6_K      : scale chunk 12, d in the tail, high-bit chunks 8-11 (one per 2 code chunks), codes from chunk 0
//   IQ4_XS    : the row's 8-byte header in the tail, codes from chunk 0
//   IQ4_NL    : the d_j chunk 8, codes from chunk 0
//   Q3_K      : the scales and d in the tail, high-bit chunks 4-5 (one per 4 units), low-bit chunks from 0 (one per
//               2 units; so the lookahead of two code chunks is 4 units)
//   Q8_0      : the d_j chunk 16, codes from chunk 0 (two per unit)
// A unit j is the superblock's 32 columns 32j..32j+31 (4 fragments of 8); a type has CPU2 code chunks per 2 units.
template <ggml_type type> struct qpn_stream_t {
    static constexpr int CH0 = type == GGML_TYPE_Q4_K ? 1 : type == GGML_TYPE_Q5_K ? 3 : 0;          // first code chunk
    static constexpr int HDR = type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K ? 0 :
                               type == GGML_TYPE_Q6_K ? 12 : type == GGML_TYPE_IQ4_NL ? 8 :
                               type == GGML_TYPE_Q8_0 ? 16 : -1;                                        // held chunk
    static constexpr int NHB = type == GGML_TYPE_Q5_K || type == GGML_TYPE_Q3_K ? 2 : type == GGML_TYPE_Q6_K ? 4 : 0; // high-bit chunks
    static constexpr int HB0 = type == GGML_TYPE_Q5_K ? 1 : type == GGML_TYPE_Q3_K ? 4 : 8;
    static constexpr int JPH = type == GGML_TYPE_Q5_K || type == GGML_TYPE_Q3_K ? 4 : 2;              // units per high-bit chunk
    static constexpr bool TAIL = type == GGML_TYPE_Q6_K || type == GGML_TYPE_IQ4_XS || type == GGML_TYPE_Q3_K;
    static constexpr int CPU2 = type == GGML_TYPE_Q3_K ? 1 : type == GGML_TYPE_Q8_0 ? 4 : 2;          // code chunks per 2 units
};

// what the lanes of padding tokens (T <= token < 8) read instead of a real token's fragments:
// zeros, so they no longer depend on real rows; their outputs are never written, and a real token's C column depends
// only on its own B column, so the outputs are unchanged. It is the same one load per lane (a warp-wide LDG.128 either way).
static __device__ uint4 qpn_zero_x[32*32]; // 32 tokens (fragment stride t <= 32)
template <ggml_type type, int T, int NL, bool STAGE, int LA = 2, bool FULL = false>
static __device__ __forceinline__ void qpn_stream_loop(
        const char * Wp, const int64_t step, int sb, const int sbB, const int S, const bool active, const uint4 * xsm, const int sbA, const half * __restrict__ xh,
        const half * __restrict__ xs, const float * __restrict__ xsc, const int nB_, const int r0, const int c0, const int dofs, float * D, unsigned & bad,
        const int t_, const int ts_) {
#if defined(VOLTA_MMA_AVAILABLE)
    // t is the block's token count, ts the token stride of the prepared input (t, or a paired block's whole pass, 9 to 16)
    typedef tile<32, 4, half2>                               tile_A;
    typedef tile< 8, 4, half2, DATA_LAYOUT_I_MAJOR_MIRRORED> tile_B;
    typedef tile<32, 8, float>                               tile_C;
    typedef qpn_stream_t<type> st;
    constexpr int CH0 = st::CH0, HDR = st::HDR, NHB = st::NHB, HB0 = st::HB0, JPH = st::JPH;
    // FULL, the pass holds exactly T tokens (t = ts = T, every lane's token real), so every activation address is a constant
    // offset from one base (the second tile's at +8 fragments' tokens) and needs no arithmetic of its own
    static_assert(!FULL || !STAGE, "full: unstaged");
    const int t  = FULL ? T : t_, ts = FULL ? T : ts_;
    const int nB = FULL ? tile_B::get_i(0) : nB_;
    // LA, the lookahead in units: unit j requests code chunk j + LA, and the next superblock's first LA code chunks (and
    // the high-bit chunks its first LA units read) are requested in this superblock's last LA units. LA = 2 is the default;
    // LA = 4 holds twice the weight bytes in flight per warp (only load timing moves: the arithmetic is unchanged)
    static_assert(LA == 2 || (LA == 4 && st::CPU2 == 2), "LA = 4: the types with two code chunks per two units");
    constexpr int HBH = LA == 2 || NHB == 0 ? 1 : (LA + JPH - 1)/JPH; // high-bit chunks requested with the next superblock's head
    // at T = 16 (9 to 16 tokens) two 8-token B tiles share each decoded A fragment, so one weight read serves both, and
    // four at T = 32 (17 to 32 tokens); tile b's C elements are D[8*b .. 8*b + 7], summed in the same order as a pass of 8
    // (each column bit-identical to it)
    constexpr int NBT = T > 16 ? 4 : T > 8 ? 2 : 1;
    static_assert(NBT == 1 || (!STAGE && NL == 8*NBT), "16 or 32 tokens: unstaged, 8 C elements a lane per tile");
    // QPN_MAGIC for qpn_decode, as a register: qpn_zero_x is never written, so the OR leaves the value unchanged
    constexpr bool MGR = type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K || type == GGML_TYPE_Q6_K;
    const uint32_t mg = MGR ? (QPN_MAGIC | __ldg(&qpn_zero_x[0].x)) : QPN_MAGIC;

    uint4 hdrn = make_uint4(0, 0, 0, 0), hbn = make_uint4(0, 0, 0, 0), c0n = make_uint4(0, 0, 0, 0), c1n = make_uint4(0, 0, 0, 0);
    uint4 cxn[LA > 2 ? LA - 2 : 1], hbxn[HBH > 1 ? HBH - 1 : 1]; // LA = 4: the next superblock's code chunks 2, 3 and high-bit chunks 1 ..
    uint2 tln  = make_uint2(0, 0);
    uint4 q3n  = make_uint4(0, 0, 0, 0); // Q3_K: the row's 12 scale bytes and the d of rows r0, r0 + 2
    // this superblock's held chunk, first high-bit chunk, tail entry and first two code chunks, from Wb
    auto head = [&](const char * Wb) {
        if constexpr (HDR >= 0) {
            hdrn = qpn_ldg_w(Wb + HDR*QPN_CHUNK);
        }
        if constexpr (NHB > 0) {
            hbn = qpn_ldg_w(Wb + HB0*QPN_CHUNK);
        }
        if constexpr (type == GGML_TYPE_Q6_K) {
            tln.x = __ldg((const uint32_t *) Wb + dofs);
        } else if constexpr (type == GGML_TYPE_IQ4_XS) {
            tln = qpn_ldg_w2((const char *) ((const uint32_t *) Wb + dofs));
        } else if constexpr (type == GGML_TYPE_Q3_K) {
            // dofs: the u32 index of this lane's scales from Wb; its d pair follows the 32 rows' scales like Q6_K's
            const uint32_t * sp = (const uint32_t *) Wb + dofs;
            q3n.x = __ldg(sp); q3n.y = __ldg(sp + 1); q3n.z = __ldg(sp + 2);
            q3n.w = __ldg(sp - 3*(threadIdx.x % 32) + 96 + (r0 >> 2)*2 + (r0 & 1));
        }
    };
    if (active && sb < sbB) {
        head(Wp);
        c0n = qpn_ldg_w(Wp + CH0*QPN_CHUNK);
        c1n = qpn_ldg_w(Wp + (CH0 + 1)*QPN_CHUNK);
        if constexpr (LA > 2) {
#pragma unroll
            for (int m = 2; m < LA; ++m) {
                cxn[m - 2] = qpn_ldg_w(Wp + (CH0 + m)*QPN_CHUNK);
            }
#pragma unroll
            for (int h = 1; h < HBH; ++h) {
                hbxn[h - 1] = qpn_ldg_w(Wp + (HB0 + h)*QPN_CHUNK);
            }
        }
    }
    for (; active && sb < sbB; sb += S) {
        const bool more = sb + S < sbB;
        const char * Wn = Wp + step;
        const uint4 hdr = hdrn;
        const uint2 tl  = tln;
        const uint4 q3  = q3n;
        uint4 hb[NHB > 0 ? NHB : 1];
        hb[0] = hbn;
        uint4 cc[4*st::CPU2];
        cc[0] = c0n;
        cc[1] = c1n;
        if constexpr (LA > 2) {
#pragma unroll
            for (int m = 2; m < LA; ++m) {
                cc[m] = cxn[m - 2];
            }
#pragma unroll
            for (int h = 1; h < HBH; ++h) {
                hb[h] = hbxn[h - 1];
            }
        }
        // padding-token lanes read qpn_zero_x (staged: the block's shared copy of a real token, as before)
        const int nBr = tile_B::get_i(0);
        const uint4 * xk = STAGE ? xsm + ((sb - sbA)*32*t + nB) : (FULL || nBr < t) ? (const uint4 *) xh + ((int64_t) sb*32*ts + nB) : qpn_zero_x + nBr;
        const uint4 * xm = (const uint4 *) xs + ((int64_t) sb*ts + nB);
        // tile b's tokens 8b .. min(8b + 7, t - 1) (lanes of padding tokens read qpn_zero_x; their sums a real token's)
        const uint4 * xkb[NBT];
        const uint4 * xmb[NBT];
        xkb[0] = xk;
        xmb[0] = xm;
#pragma unroll
        for (int b = 1; b < NBT; ++b) {
            xkb[b] = FULL ? xk + 8*b : nBr + 8*b < t ? (const uint4 *) xh + ((int64_t) sb*32*ts + nBr + 8*b) : qpn_zero_x + nBr;
            xmb[b] = FULL ? xm + 8*b : (const uint4 *) xs + ((int64_t) sb*ts + min(nBr + 8*b, t - 1));
        }
// one decoded fragment A (fragment f of the superblock) into accumulator acc of every B tile
#define QPN_MMA2(A, f, acc) do {                                     \
            tile_B B;                                                \
            *(uint4 *) B.x = STAGE ? xk[(f)*t] : __ldg(xk + (f)*ts); \
            mma(Ds[0][acc], A, B);                                   \
            _Pragma("unroll")                                        \
            for (int b_ = 1; b_ < NBT; ++b_) {                       \
                tile_B B1;                                           \
                *(uint4 *) B1.x = __ldg(xkb[b_] + (f)*ts);           \
                mma(Ds[b_][acc], A, B1);                             \
            }                                                        \
        } while (0)

        float cs[NL/2];
        if constexpr (NL == 4) {
#pragma unroll
        for (int i = 0; i < NL/2; ++i) {
            const int col = c0 + (i & 1) + 4*(i/2);
            cs[i] = col < t ? xsc[sb*ts + col] : 1.0f;
            bad |= (col < t && cs[i] == 0.0f) ? 1u << col : 0u;
        }
        }
        tile_C Ds[NBT][2];

        // per-superblock scales: Q4_K/Q5_K 6-bit scales and mins; Q6_K int8 scales + 0x80; IQ4_XS 6-bit ls_j
        uint32_t scl[2] = {0, 0}, mnl[2] = {0, 0}, scu[4] = {0, 0, 0, 0}, ls02 = 0, ls13 = 0;
        if constexpr (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) {
            scl[0] = hdr.y & 0x3F3F3F3Fu;
            scl[1] = (hdr.w & 0x0F0F0F0Fu) | ((hdr.y >> 2) & 0x30303030u);
            mnl[0] = hdr.z & 0x3F3F3F3Fu;
            mnl[1] = ((hdr.w >> 4) & 0x0F0F0F0Fu) | ((hdr.z >> 2) & 0x30303030u);
        } else if constexpr (type == GGML_TYPE_Q6_K) {
            scu[0] = hdr.x ^ 0x80808080u; scu[1] = hdr.y ^ 0x80808080u; scu[2] = hdr.z ^ 0x80808080u; scu[3] = hdr.w ^ 0x80808080u;
        } else if constexpr (type == GGML_TYPE_IQ4_XS) {
            const uint32_t h = __byte_perm(tl.x, 0, 0x3322); // scales_h bytes 0, 0, 1, 1
            ls02 = ( tl.y       & 0x0F0F0F0Fu) | ((h << 4) & 0x00300030u) | ( h       & 0x30003000u);
            ls13 = ((tl.y >> 4) & 0x0F0F0F0Fu) | ((h << 2) & 0x00300030u) | ((h >> 2) & 0x30003000u);
        } else if constexpr (type == GGML_TYPE_Q3_K) {
            qpn_q3_scales(q3.x, q3.y, q3.z, scu); // six-bit s_j, sub-block j at byte j % 4 of scu[j/4]
        }

#pragma unroll
        for (int j = 0; j < 8; ++j) {
            if constexpr (st::CPU2 == 2) {
                if (j + LA < 8) {
                    cc[j + LA] = qpn_ldg_w(Wp + (CH0 + j + LA)*QPN_CHUNK);
                }
            } else if constexpr (st::CPU2 == 4) { // Q8_0: the next unit's two code chunks
                if (j + 1 < 8) {
                    cc[2*j + 2] = qpn_ldg_w(Wp + (CH0 + 2*j + 2)*QPN_CHUNK);
                    cc[2*j + 3] = qpn_ldg_w(Wp + (CH0 + 2*j + 3)*QPN_CHUNK);
                }
            } else { // Q3_K: the low-bit chunk of units j + 4, j + 5
                if (j % 2 == 0 && j/2 + 2 < 4) {
                    cc[j/2 + 2] = qpn_ldg_w(Wp + (CH0 + j/2 + 2)*QPN_CHUNK);
                }
            }
            if constexpr (NHB > 0) {
                if ((j + LA) % JPH == 0 && (j + LA)/JPH < NHB && (j + LA)/JPH >= HBH) {
                    hb[(j + LA)/JPH] = qpn_ldg_w(Wp + (HB0 + (j + LA)/JPH)*QPN_CHUNK);
                }
            }
            if constexpr (LA == 2) {
                if (j == 6 && more) {
                    head(Wn);
                }
                if (j == 7 && more) {
                    c0n = qpn_ldg_w(Wn + CH0*QPN_CHUNK);
                    c1n = qpn_ldg_w(Wn + (CH0 + 1)*QPN_CHUNK);
                }
            } else { // the next superblock's chunk m in unit 8 - LA + m, its head with chunk 0
                if (j == 8 - LA && more) {
                    head(Wn);
                    c0n = qpn_ldg_w(Wn + CH0*QPN_CHUNK);
                }
                if (j == 9 - LA && more) {
                    c1n = qpn_ldg_w(Wn + (CH0 + 1)*QPN_CHUNK);
                }
#pragma unroll
                for (int m = 2; m < LA; ++m) {
                    if (j == 8 - LA + m && more) {
                        cxn[m - 2] = qpn_ldg_w(Wn + (CH0 + m)*QPN_CHUNK);
                    }
                }
#pragma unroll
                for (int h = 1; h < HBH; ++h) {
                    if (j == 8 - LA + h*JPH && more) {
                        hbxn[h - 1] = qpn_ldg_w(Wn + (HB0 + h)*QPN_CHUNK);
                    }
                }
            }
            const uint4 cw = cc[st::CPU2 == 2 ? j : st::CPU2 == 4 ? 2*j : j/2];
            const uint32_t cwv[4] = {cw.x, cw.y, cw.z, cw.w};
            if constexpr (type == GGML_TYPE_Q3_K) {
                // low-bit words 2j, 2j + 1 (sub-blocks 2j, 2j + 1; parity p = 0, 1), high-bit word j
                const uint4    hq = hb[j/4];
                const uint32_t H  = (j % 4) == 0 ? hq.x : (j % 4) == 1 ? hq.y : (j % 4) == 2 ? hq.z : hq.w;
#pragma unroll
                for (int p = 0; p < 2; ++p) {
                    const uint32_t L  = cwv[2*(j % 2) + p];
                    const int      sb = 2*j + p;
                    const half2 v = qpn_u2h(__byte_perm(scu[sb/4], 0x64, (sb % 4) | (4 << 4) | ((sb % 4) << 8) | (4 << 12))); // 1024 + s
                    const half2 S2 = __hfma2(v, make_half2(0.25f, 0.25f), make_half2(-264.0f, -264.0f)); // (s - 32)/4, exact
                    const half2 Z  = __hmul2(S2, make_half2(-1040.0f, -1040.0f));                        // -260*(s - 32), exact
#pragma unroll
                    for (int f = 0; f < 2; ++f) {
                        tile_A A;
#pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            const int e = 4*f + i;
                            const uint32_t K = __funnelshift_l(0x00030003u, 0x00030003u, 2*e + p); // this pair's low bits
                            const uint32_t C = (L & K) | (H & ~K);
                            const uint32_t x = (__funnelshift_r(C, C, (2*e + p + 30) & 31) & 0x001C001Cu) | QPN_MAGIC; // 1024 + 4c
                            A.x[i] = __hfma2(qpn_u2h(x), S2, Z);
                        }
                        const int q = 2*p + f;
                        QPN_MMA2(A, 4*j + q, q & 1);
                    }
                }
            } else if constexpr (type == GGML_TYPE_Q8_0) {
                // chunks 2j, 2j + 1: fragments 4j .. 4j + 3, two words each; block j's d_j
                const uint32_t dw = (j/2) == 0 ? hdr.x : (j/2) == 1 ? hdr.y : (j/2) == 2 ? hdr.z : hdr.w;
                const half2    S2 = qpn_u2h(__byte_perm(dw, 0, (j & 1) ? 0x3232 : 0x1010));
                const uint4    cw1 = cc[2*j + 1];
                const uint32_t w8[8] = {cw.x, cw.y, cw.z, cw.w, cw1.x, cw1.y, cw1.z, cw1.w};
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    tile_A A;
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const uint32_t x = __byte_perm(w8[2*q + i/2], 0x64646464u, (i & 1) ? 0x4342 : 0x4140); // 1024 + 128 + q
                        A.x[i] = __hmul2(__hsub2(qpn_u2h(x), make_half2(1152.0f, 1152.0f)), S2);
                    }
                    QPN_MMA2(A, 4*j + q, q & 1);
                }
            } else if constexpr (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) {
                const half2 sc  = qpn_byte_h2(scl[j/4], j % 4);
                const half2 s16 = __hmul2(sc, make_half2(0.0625f, 0.0625f));
                const half2 z   = __hmul2(sc, make_half2(-64.0f, -64.0f));
                uint32_t H = 0;
                if constexpr (type == GGML_TYPE_Q5_K) {
                    const uint4 hq = hb[j/4];
                    H = (j % 4) == 0 ? hq.x : (j % 4) == 1 ? hq.y : (j % 4) == 2 ? hq.z : hq.w;
                }
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    tile_A A;
                    qpn_decode<type>(A, cwv[q], __funnelshift_r(H, H, q), s16, z, mg);
                    QPN_MMA2(A, 4*j + q, q & 1);
                }
            } else if constexpr (type == GGML_TYPE_Q6_K) {
                const uint4    hc     = hb[j/2];
                const uint32_t hcv[2] = {(j & 1) ? hc.z : hc.x, (j & 1) ? hc.w : hc.y};
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const int   sbi = 2*j + q/2; // 16-column sub-block
                    const half2 sc  = __hsub2(qpn_u2h(__byte_perm(scu[sbi/4], 0x64, (sbi % 4) | (4 << 4) | ((sbi % 4) << 8) | (4 << 12))),
                                              make_half2(1152.0f, 1152.0f));
                    const half2 s16 = __hmul2(sc, make_half2(0.0625f, 0.0625f));
                    const half2 z   = __hmul2(sc, make_half2(-96.0f, -96.0f));
                    const uint32_t H = hcv[q/2];
                    tile_A A;
                    qpn_decode<type>(A, cwv[q], __funnelshift_r(H, H, 2*(q & 1)), s16, z, mg);
                    QPN_MMA2(A, 4*j + q, q & 1);
                }
            } else {
                half2 S2, Z;
                if constexpr (type == GGML_TYPE_IQ4_XS) {
                    const int b = j/2;
                    S2 = __hsub2(qpn_u2h(__byte_perm(j & 1 ? ls13 : ls02, 0x64, b | (4 << 4) | (b << 8) | (4 << 12))),
                                 make_half2(1056.0f, 1056.0f)); // ls_j - 32, exact
                    Z  = __hmul2(S2, make_half2(-1152.0f, -1152.0f));
                } else {
                    const uint32_t dw = (j/2) == 0 ? hdr.x : (j/2) == 1 ? hdr.y : (j/2) == 2 ? hdr.z : hdr.w;
                    S2 = qpn_u2h(__byte_perm(dw, 0, (j & 1) ? 0x3232 : 0x1010));
                    Z  = S2;
                }
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    tile_A A;
                    qpn_decode_iq4<type>(A, cwv[q], S2, Z);
                    QPN_MMA2(A, 4*j + q, q & 1);
                }
            }
        }

        // 5 to 8 tokens: the range scales are read after the products, so they hold no registers through them
        // (at 80 registers this takes the loop's spills from 2-17 LDL to 0 for Q4_K, IQ4_XS and IQ4_NL, 7 for Q5_K and Q6_K)
        if constexpr (NL >= 8) {
#pragma unroll
            for (int i = 0; i < NL/2; ++i) {
                const int col = c0 + (i & 1) + 4*(i/2);
                cs[i] = col < t ? xsc[sb*ts + col] : 1.0f;
                bad |= (col < t && cs[i] == 0.0f) ? 1u << col : 0u;
            }
        }
        if constexpr (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) {
            tile_A Am;
            Am.x[0] = qpn_bytes_h2(mnl[0], 0);
            Am.x[1] = qpn_bytes_h2(mnl[0], 2);
            Am.x[2] = qpn_bytes_h2(mnl[1], 0);
            Am.x[3] = qpn_bytes_h2(mnl[1], 2);
            tile_C M[NBT];
#pragma unroll
            for (int b = 0; b < NBT; ++b) {
                tile_B Bm;
                *(uint4 *) Bm.x = __ldg(xmb[b]);
                mma(M[b], Am, Bm);
            }
            const float2 dm0 = __half22float2(qpn_u2h(__shfl_sync(0xFFFFFFFF, hdr.x, r0)));
            const float2 dm2 = __half22float2(qpn_u2h(__shfl_sync(0xFFFFFFFF, hdr.x, r0 + 2)));
#pragma unroll
            for (int l = 0; l < NL; ++l) {
                const float2 dm = (l & 2) ? dm2 : dm0;
                const float  c  = cs[qpn_csi(l)];
                D[l] += c * (dm.x * (Ds[l >> 3][0].x[l & 7] + Ds[l >> 3][1].x[l & 7]) - 32.0f*dm.y * M[l >> 3].x[l & 7]);
            }
        } else if constexpr (type == GGML_TYPE_Q6_K || type == GGML_TYPE_Q3_K) {
            const float2 d02 = __half22float2(qpn_u2h(type == GGML_TYPE_Q6_K ? tl.x : q3.w));
#pragma unroll
            for (int l = 0; l < NL; ++l) {
                const float d = (l & 2) ? d02.y : d02.x;
                const float c = cs[qpn_csi(l)];
                D[l] += c * d * (Ds[l >> 3][0].x[l & 7] + Ds[l >> 3][1].x[l & 7]);
            }
        } else if constexpr (type == GGML_TYPE_IQ4_XS) {
            const float d0 = __low2float(qpn_u2h(__shfl_sync(0xFFFFFFFF, tl.x, r0)));
            const float d2 = __low2float(qpn_u2h(__shfl_sync(0xFFFFFFFF, tl.x, r0 + 2)));
#pragma unroll
            for (int l = 0; l < NL; ++l) {
                const float c = cs[qpn_csi(l)];
                D[l] += c * ((l & 2) ? d2 : d0) * (Ds[l >> 3][0].x[l & 7] + Ds[l >> 3][1].x[l & 7]);
            }
        } else {
#pragma unroll
            for (int l = 0; l < NL; ++l) {
                const float c = cs[qpn_csi(l)];
                D[l] += c * (Ds[l >> 3][0].x[l & 7] + Ds[l >> 3][1].x[l & 7]);
            }
        }
        Wp = Wn;
    }
#undef QPN_MMA2
#else
    GGML_UNUSED_VARS(Wp, step, sb, sbB, S, active, xsm, sbA, xh, xs, xsc, nB_, r0, c0, dofs, D, bad, t_, ts_);
    NO_DEVICE_CODE;
#endif // defined(VOLTA_MMA_AVAILABLE)
}

// MODE 0: the whole-superblock loop at QPN_MIN_BLOCKS blocks per SM; MODE 2, 3 or 4: qpn_stream_loop, with the launch bounds of
// MODE blocks of 8 warps per SM. The streaming kernels at T = 8 take
// any of 5 to 8 tokens, nt, at run time; every other kernel takes exactly T.
// MODE 5, the streaming loop with a lookahead of 4 units (LA = 4) for the products whose grid leaves registers free (160
// blocks of 4 warps, about 9 warps per SM): blocks of at most 4 warps, 3 per SM, so at most 168 registers (one wave of 216 blocks)
// QPN_FULL added to a streaming mode (2, 3 or 5) at T = 8 or 16: the kernel takes exactly T tokens, so the token count and
// the input's stride are constants and every activation address is a constant offset from one base (qpn_stream_loop's FULL)
#define QPN_FULL 100
static constexpr __host__ __device__ bool qpn_mode_full(const int mode) { return mode >= QPN_FULL; }
static constexpr __host__ __device__ int  qpn_mode_m(const int mode) { return mode % QPN_FULL; }
// MODE 6 (two row tiles a warp) takes MODE 5's bounds: blocks of at most 4 warps, 3 per SM, so up to 168 registers (at 2 blocks of
// 8 warps, 128 registers, its loop spilled: 51 LDL and 38 STL a superblock)
static constexpr __host__ __device__ int qpn_min_blocks(const int mode) { return mode == 0 ? QPN_MIN_BLOCKS : qpn_mode_m(mode) == 5 || qpn_mode_m(mode) == 6 ? 3 : qpn_mode_m(mode); }
static constexpr __host__ __device__ int qpn_max_threads(const int mode) { return qpn_mode_m(mode) == 5 || qpn_mode_m(mode) == 6 ? 4*WARP_SIZE : QPN_MAX_WARPS*WARP_SIZE; }

// the two one-tile members' outputs, strides and stored rows (CLIP), and the member of the calling block, found at the store
// from a fresh read of blockIdx.x (as the fallback reads blockIdx.y), so no choice made before the loop is live across it
struct qpn_clip { float * dst0; int64_t stride_col_dst0; int nrow0; float * dst1; int64_t stride_col_dst1; int nrow1; int nb0; };
static __device__ __forceinline__ void qpn_clip_member(const qpn_clip & c, float * & dst, int64_t & stride_col_dst, int & nrow) {
    unsigned bx;
    asm volatile("mov.u32 %0, %%ctaid.x;" : "=r"(bx));
    const bool second = (int) bx >= c.nb0;
    dst            = second ? c.dst1 : c.dst0;
    stride_col_dst = second ? c.stride_col_dst1 : c.stride_col_dst0;
    nrow           = second ? c.nrow1 : c.nrow0;
}

// One block's work on one product (a segment): grid = (row tile groups, P), blockDim = (32, R*S); bx is the block's index
// among the product's row tile groups. The caller has zeroed *sbad and staged the activations (STAGE), then synchronized.
// t: the product's token count. It is T (a constant) except in the streaming kernels instantiated at T = 8, which take 5 to 8
// tokens at run time: their activations are laid out for t tokens, and lanes of tokens t .. 7 read qpn_zero_x.
// CLIP (qpn_mul4ab_kernel only): the block is one of two one-tile members (blocks [0, nb0) the first, the rest the second),
// each storing only the first nrow rows of its tile into its own output; dst and stride_col_dst are unused and the store takes the member's
// from clip (qpn_clip_member). Every other instantiation stores all 32 rows to dst, with no test.
template <ggml_type type, int T, bool STAGE, int MODE, bool PUSH = false, bool CLIP = false>
static __device__ __forceinline__ void qpn_mul_block(const int bx,
        const char * __restrict__ W, const int nsb, const int ntiles, const int S, const int P,
        const half * __restrict__ xh, const half * __restrict__ xs, const float * __restrict__ xsc,
        const float * __restrict__ x, const int64_t stride_col_x, float * __restrict__ dst, const int64_t stride_col_dst,
        float * __restrict__ ws, unsigned * __restrict__ cnt, const uint4 * qpn_xsm, float * red, unsigned & sbad, const int t, const int ts,
        const ggml_cuda_p2p_push push = {}, const qpn_clip clip = {}) {
#if defined(VOLTA_MMA_AVAILABLE)
    typedef tile< 8, 4, half2, DATA_LAYOUT_I_MAJOR_MIRRORED> tile_B;
    constexpr int NCH = qpn_t<type>::NCH, TSB = qpn_tsb<type>();
    constexpr int NL  = qpn_nl<T>(); // C elements of a lane that can hold a real token

    const int lane = threadIdx.x, wy = threadIdx.y;
    const int R    = blockDim.y / S;
    const int tile = bx*R + wy/S, sp = wy % S;
    const int p    = blockIdx.y;
    const int sbA  = nsb*p/P, sbB = nsb*(p + 1)/P;
    const bool active = tile < ntiles;
    const int  nB   = min(tile_B::get_i(0), t - 1); // token of this lane's B fragment (a real one for lanes of padding)
    const bool hasB = true;
    const int  r0   = lane & ~2;          // rows of this lane's C: r0, r0 + 2 (tile_C::get_i)
    const int  c0   = lane & 2;           // its columns: c0 + qpn_col(l) (tile_C::get_j; l >> 3 is the B tile)

    float    D[NL] = {0.0f};
    unsigned bad   = 0;

    const char * Wt = W + (int64_t) tile*nsb*TSB;
    const char * Wp = Wt + (int64_t) (sbA + sp)*TSB + lane*16;
    const int64_t step = (int64_t) S*TSB;
    // u32 index from Wp of Q6_K's d(r0), d(r0 + 2), of the IQ4_XS header of row lane, or of the Q3_K scales of row lane
    const int dofs = type == GGML_TYPE_Q6_K ? NCH*QPN_CHUNK/4 - lane*4 + (r0 >> 2)*2 + (r0 & 1) :
                     type == GGML_TYPE_Q3_K ? NCH*QPN_CHUNK/4 - lane : NCH*QPN_CHUNK/4 - lane*2;

    if constexpr (MODE != 0 || qpn_new<type>()) { // qpn_new types have only the streaming loop
        qpn_stream_loop<type, T, NL, STAGE, qpn_mode_m(MODE) == 5 && qpn_stream_t<type>::CPU2 == 2 ? 4 : 2, qpn_mode_full(MODE)>(Wp, step, sbA + sp, sbB, S, active, qpn_xsm, sbA, xh, xs, xsc, nB, r0, c0, dofs, D, bad, t, ts);
    } else {
    uint4 buf[NCH];
    uint2 dp = make_uint2(0, 0);
    int sb = sbA + sp;
    if (active && sb < sbB) {
#pragma unroll
        for (int ch = 0; ch < NCH; ++ch) {
            buf[ch] = qpn_ldg_w(Wp + ch*QPN_CHUNK);
        }
        if constexpr (type == GGML_TYPE_Q6_K) {
            dp.x = __ldg((const uint32_t *) Wp + dofs);
        } else if constexpr (type == GGML_TYPE_IQ4_XS) {
            dp = qpn_ldg_w2((const char *) ((const uint32_t *) Wp + dofs));
        }
    }
    constexpr bool RING = T >= qpn_ring_min_t<type>();
    for (; active && sb < sbB; sb += S) {
        const bool more = sb + S < sbB;
        const uint4 * xk = STAGE ? qpn_xsm + ((sb - sbA)*32*T + nB) : (const uint4 *) xh + ((int64_t) sb*32*T + nB);
        if constexpr (RING) {
            Wp += step;
            qpn_superblock<type, T, NL, STAGE, true>(buf, dp, Wp, more, dofs, sb, xk, xs, xsc, nB, hasB, r0, c0, D, bad);
        } else {
            uint4 cur[NCH];
            uint2 dcur = dp;
#pragma unroll
            for (int ch = 0; ch < NCH; ++ch) {
                cur[ch] = buf[ch];
            }
            Wp += step;
            if (more) {
#pragma unroll
                for (int ch = 0; ch < NCH; ++ch) {
                    buf[ch] = qpn_ldg_w(Wp + ch*QPN_CHUNK);
                }
                if constexpr (type == GGML_TYPE_Q6_K) {
                    dp.x = __ldg((const uint32_t *) Wp + dofs);
                } else if constexpr (type == GGML_TYPE_IQ4_XS) {
                    dp = qpn_ldg_w2((const char *) ((const uint32_t *) Wp + dofs));
                }
            }
            qpn_superblock<type, T, NL, STAGE, false>(cur, dcur, Wp, false, dofs, sb, xk, xs, xsc, nB, hasB, r0, c0, D, bad);
        }
    }

    } // MODE

    // the S warps of each tile
    if (bad) {
        atomicOr(&sbad, bad);
    }
    if (S > 1) {
#pragma unroll
        for (int l = 0; l < NL; ++l) {
            red[(wy*NL + l)*WARP_SIZE + lane] = D[l];
        }
    }
    __syncthreads();
    const unsigned badmask = sbad;
    if (sp == 0) {
        for (int i = 1; i < S; ++i) {
#pragma unroll
            for (int l = 0; l < NL; ++l) {
                D[l] += red[((wy + i)*NL + l)*WARP_SIZE + lane];
            }
        }
    }
    // The block's result goes out now, to dst or (P > 1) to its slice of ws, and the non-finite fallback below only
    // overwrites the columns it recomputes, so D is not live across the fallback's call (a D live across that
    // __noinline__ call gets a stack home for the whole kernel, which put it in local memory in every superblock).
    float * wsp = P > 1 ? ws + ((int64_t) tile*P + p)*NL*WARP_SIZE : nullptr;
    if (active && sp == 0) {
        float * dstm = dst;
        int64_t sdm  = stride_col_dst;
        int     nrow = 32;
        if constexpr (CLIP) {
            qpn_clip_member(clip, dstm, sdm, nrow);
        }
#pragma unroll
        for (int l = 0; l < NL; ++l) {
            const int row = tile*32 + r0 + (l & 2);
            const int col = c0 + qpn_col(l);
            if (P > 1) {
                wsp[l*WARP_SIZE + lane] = D[l];
            } else if (col < t && (!CLIP || r0 + (l & 2) < nrow)) {
                dstm[col*sdm + row] = D[l];
            }
        }
    }
    if (badmask) { // block-uniform: those columns again, in fp32 from the dequantized weights
        float fb[QPN_FB_N];
        if (active) {
            // the split's bounds again, from a fresh read of blockIdx.y, so the loop's copies are not live across the call
            unsigned py;
            asm volatile("mov.u32 %0, %%ctaid.y;" : "=r"(py));
            qpn_fallback<type>(Wt, nsb*(int) py/P + sp, nsb*((int) py + 1)/P, S, badmask, t, x, stride_col_x, fb);
        } else {
            for (int c = 0; c < QPN_FB_N; ++c) {
                fb[c] = 0.0f;
            }
        }
        for (int c = 0; c < t; ++c) { // one bad column at a time through red
            if (!((badmask >> c) & 1)) {
                continue;
            }
            __syncthreads();
            red[wy*WARP_SIZE + lane] = fb[c];
            __syncthreads();
            if (active && sp == 0) {
                float * dstm = dst;
                int64_t sdm  = stride_col_dst;
                int     nrow = 32;
                if constexpr (CLIP) {
                    qpn_clip_member(clip, dstm, sdm, nrow);
                }
#pragma unroll
                for (int l = 0; l < NL; ++l) {
                    if (c0 + qpn_col(l) == c) {
                        float v = 0.0f;
                        for (int i = 0; i < S; ++i) {
                            v += red[(wy + i)*WARP_SIZE + r0 + (l & 2)];
                        }
                        if (P > 1) {
                            wsp[l*WARP_SIZE + lane] = v;
                        } else if (!CLIP || r0 + (l & 2) < nrow) {
                            dstm[c*sdm + tile*32 + r0 + (l & 2)] = v;
                        }
                    }
                }
            }
        }
    }
    // with --split-mode tensor, the finished tile goes to the peer's landing slot of the ALLREDUCE that reduces dst
    if constexpr (PUSH) {
        if (active && sp == 0 && P == 1) {
            ggml_cuda_p2p_push_tile(push, dst, stride_col_dst, (int64_t) tile*32, t);
        }
    }
    if (!active || sp != 0 || P == 1 || CLIP) { // CLIP: P = 1
        return;
    }

    // P > 1: the last of the tile's P blocks adds their partial sums in the order p = 0 .. P-1
    __threadfence();
    unsigned * cp = cnt + tile;
    unsigned prev = 0;
    if (lane == 0) {
        prev = atomicAdd(cp, 1u);
    }
    prev = __shfl_sync(0xFFFFFFFF, prev, 0);
    if (prev != (unsigned) P - 1) {
        return;
    }
    __threadfence();
    const float * ws0 = ws + (int64_t) tile*P*NL*WARP_SIZE;
#pragma unroll
    for (int l = 0; l < NL; ++l) {
        float v = 0.0f;
        for (int i = 0; i < P; ++i) {
            v += __ldcg(ws0 + (i*NL + l)*WARP_SIZE + lane);
        }
        const int row = tile*32 + r0 + (l & 2);
        const int col = c0 + qpn_col(l);
        if (col < t) {
            dst[col*stride_col_dst + row] = v;
        }
    }
    if constexpr (PUSH) {
        ggml_cuda_p2p_push_tile(push, dst, stride_col_dst, (int64_t) tile*32, t);
    }
    if (lane == 0) {
        *cp = 0; // ready for the next product on this stream
    }
#else
    GGML_UNUSED_VARS(bx, W, nsb, ntiles, S, P, xh, xs, xsc, x, stride_col_x, dst, stride_col_dst, ws, cnt, qpn_xsm, red, sbad, t, push, clip);
    NO_DEVICE_CODE;
#endif // defined(VOLTA_MMA_AVAILABLE)
}

// [TAG_QPN_NAT2] MODE 6 (prototype; Q6_K, 1 to 4 tokens, unstaged, P = 1): each warp takes two row tiles over the same superblocks,
// so each activation fragment is loaded once for both tiles' mma (ncu, the Q6_K head at 4 tokens: 52% of the streaming loop's stall
// samples were mma waiting on those loads, the L1 hit rate 41%). Per tile, the arithmetic is qpn_stream_loop's at LA = 2,
// expression for expression and in the same order, and the S warps' partial sums are added in the same order as qpn_mul_block's,
// so the outputs are bit-identical to MODE 3 at the same S (LLAMA_QPN_CHECK=3,2R checks it). Blocks of S*R warps take 2R tiles.
template <ggml_type type, int T>
static __device__ __forceinline__ void qpn_mul_block_nat2(const int bx, const char * __restrict__ W, const int nsb, const int ntiles, const int S,
        const half * __restrict__ xh, const float * __restrict__ xsc, const float * __restrict__ x, const int64_t stride_col_x,
        float * __restrict__ dst, const int64_t stride_col_dst, float * red, unsigned & sbad, const int t) {
#if defined(VOLTA_MMA_AVAILABLE)
    static_assert(type == GGML_TYPE_Q6_K && T <= 4, "MODE 6: the Q6_K prototype at 1 to 4 tokens");
    typedef tile<32, 4, half2>                               tile_A;
    typedef tile< 8, 4, half2, DATA_LAYOUT_I_MAJOR_MIRRORED> tile_B;
    typedef tile<32, 8, float>                               tile_C;
    constexpr int NCH = qpn_t<type>::NCH, TSB = qpn_tsb<type>();
    constexpr int NL  = qpn_nl<T>();
    constexpr int NA  = 2;                              // row tiles a warp
    constexpr int HDR = 12, NHB = 4, HB0 = 8, JPH = 2;  // qpn_stream_t<Q6_K>
    constexpr int LA  = 2;

    const int lane = threadIdx.x, wy = threadIdx.y;
    const int R    = blockDim.y / S;
    const int grp  = bx*R + wy/S, sp = wy % S;
    const int tile0 = grp*NA;
    const int ts   = t;
    const int sbA  = 0, sbB = nsb;
    const int nB   = min(tile_B::get_i(0), t - 1);
    const int r0   = lane & ~2, c0 = lane & 2;
    const int dofs = NCH*QPN_CHUNK/4 - lane*4 + (r0 >> 2)*2 + (r0 & 1);
    bool act[NA];
    const char * Wp[NA];
#pragma unroll
    for (int a = 0; a < NA; ++a) {
        act[a] = tile0 + a < ntiles;
        Wp[a]  = W + (int64_t) (tile0 + a)*nsb*TSB + (int64_t) (sbA + sp)*TSB + lane*16;
    }
    const bool active = act[0]; // tile0 + 1 exists whenever tile0 does but for an odd ntiles' last group
    const int64_t step = (int64_t) S*TSB;
    const uint32_t mg = QPN_MAGIC | __ldg(&qpn_zero_x[0].x);

    float    D[NA][NL] = {};
    unsigned bad = 0;
    uint4 hdrn[NA], hbn[NA], c0n[NA], c1n[NA];
    uint2 tln[NA];
#pragma unroll
    for (int a = 0; a < NA; ++a) {
        hdrn[a] = hbn[a] = c0n[a] = c1n[a] = make_uint4(0, 0, 0, 0);
        tln[a]  = make_uint2(0, 0);
    }
    auto head = [&](const int a, const char * Wb) {
        hdrn[a]  = qpn_ldg_w(Wb + HDR*QPN_CHUNK);
        hbn[a]   = qpn_ldg_w(Wb + HB0*QPN_CHUNK);
        tln[a].x = __ldg((const uint32_t *) Wb + dofs);
    };
    int sb = sbA + sp;
    if (active && sb < sbB) {
#pragma unroll
        for (int a = 0; a < NA; ++a) {
            if (act[a]) {
                head(a, Wp[a]);
                c0n[a] = qpn_ldg_w(Wp[a]);
                c1n[a] = qpn_ldg_w(Wp[a] + QPN_CHUNK);
            }
        }
    }
    for (; active && sb < sbB; sb += S) {
        const bool more = sb + S < sbB;
        uint4 hdr[NA], hb[NA][NHB], cc[NA][8];
        uint2 tl[NA];
        uint32_t scu[NA][4];
#pragma unroll
        for (int a = 0; a < NA; ++a) {
            hdr[a] = hdrn[a]; tl[a] = tln[a]; hb[a][0] = hbn[a]; cc[a][0] = c0n[a]; cc[a][1] = c1n[a];
            scu[a][0] = hdr[a].x ^ 0x80808080u; scu[a][1] = hdr[a].y ^ 0x80808080u;
            scu[a][2] = hdr[a].z ^ 0x80808080u; scu[a][3] = hdr[a].w ^ 0x80808080u;
        }
        const int nBr = tile_B::get_i(0);
        const uint4 * xk = nBr < t ? (const uint4 *) xh + ((int64_t) sb*32*ts + nB) : qpn_zero_x + nBr;

        float cs[NL/2];
#pragma unroll
        for (int i = 0; i < NL/2; ++i) {
            const int col = c0 + (i & 1) + 4*(i/2);
            cs[i] = col < t ? xsc[sb*ts + col] : 1.0f;
            bad |= (col < t && cs[i] == 0.0f) ? 1u << col : 0u;
        }
        tile_C Ds[NA][2];

#pragma unroll
        for (int j = 0; j < 8; ++j) {
#pragma unroll
            for (int a = 0; a < NA; ++a) {
                if (j + LA < 8 && act[a]) {
                    cc[a][j + LA] = qpn_ldg_w(Wp[a] + (j + LA)*QPN_CHUNK);
                }
                if ((j + LA) % JPH == 0 && (j + LA)/JPH < NHB && (j + LA)/JPH >= 1 && act[a]) {
                    hb[a][(j + LA)/JPH] = qpn_ldg_w(Wp[a] + (HB0 + (j + LA)/JPH)*QPN_CHUNK);
                }
                if (j == 6 && more && act[a]) {
                    head(a, Wp[a] + step);
                }
                if (j == 7 && more && act[a]) {
                    c0n[a] = qpn_ldg_w(Wp[a] + step);
                    c1n[a] = qpn_ldg_w(Wp[a] + step + QPN_CHUNK);
                }
            }
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                tile_A A[NA];
#pragma unroll
                for (int a = 0; a < NA; ++a) {
                    const uint4    cw     = cc[a][j];
                    const uint32_t cwv[4] = {cw.x, cw.y, cw.z, cw.w};
                    const uint4    hc     = hb[a][j/2];
                    const uint32_t hcv[2] = {(j & 1) ? hc.z : hc.x, (j & 1) ? hc.w : hc.y};
                    const int   sbi = 2*j + q/2;
                    const half2 sc  = __hsub2(qpn_u2h(__byte_perm(scu[a][sbi/4], 0x64, (sbi % 4) | (4 << 4) | ((sbi % 4) << 8) | (4 << 12))),
                                              make_half2(1152.0f, 1152.0f));
                    const half2 s16 = __hmul2(sc, make_half2(0.0625f, 0.0625f));
                    const half2 z   = __hmul2(sc, make_half2(-96.0f, -96.0f));
                    const uint32_t H = hcv[q/2];
                    qpn_decode<type>(A[a], cwv[q], __funnelshift_r(H, H, 2*(q & 1)), s16, z, mg);
                }
                tile_B B;
                *(uint4 *) B.x = __ldg(xk + (4*j + q)*ts);
#pragma unroll
                for (int a = 0; a < NA; ++a) {
                    mma(Ds[a][q & 1], A[a], B);
                }
            }
        }
#pragma unroll
        for (int a = 0; a < NA; ++a) {
            const float2 d02 = __half22float2(qpn_u2h(tl[a].x));
#pragma unroll
            for (int l = 0; l < NL; ++l) {
                const float d = (l & 2) ? d02.y : d02.x;
                const float c = cs[qpn_csi(l)];
                D[a][l] += c * d * (Ds[a][0].x[l & 7] + Ds[a][1].x[l & 7]);
            }
            Wp[a] += step;
        }
    }

    // per tile, qpn_mul_block's epilogue at P = 1: the S warps' sums in the order sp = 0 .. S-1, then the store
    if (bad) {
        atomicOr(&sbad, bad);
    }
#pragma unroll
    for (int a = 0; a < NA; ++a) {
        if (a > 0) {
            __syncthreads();
        }
        if (S > 1) {
#pragma unroll
            for (int l = 0; l < NL; ++l) {
                red[(wy*NL + l)*WARP_SIZE + lane] = D[a][l];
            }
        }
        __syncthreads();
        if (sp == 0) {
            for (int i = 1; i < S; ++i) {
#pragma unroll
                for (int l = 0; l < NL; ++l) {
                    D[a][l] += red[((wy + i)*NL + l)*WARP_SIZE + lane];
                }
            }
        }
        if (act[a] && sp == 0) {
#pragma unroll
            for (int l = 0; l < NL; ++l) {
                const int row = (tile0 + a)*32 + r0 + (l & 2);
                const int col = c0 + qpn_col(l);
                if (col < t) {
                    dst[col*stride_col_dst + row] = D[a][l];
                }
            }
        }
    }
    const unsigned badmask = sbad;
    if (badmask) { // block-uniform: those columns again in fp32 from the dequantized weights, tile by tile
#pragma unroll 1
        for (int a = 0; a < NA; ++a) {
            float fb[QPN_FB_N];
            if (act[a]) {
                qpn_fallback<type>(W + (int64_t) (tile0 + a)*nsb*TSB, sbA + sp, sbB, S, badmask, t, x, stride_col_x, fb);
            } else {
                for (int c = 0; c < QPN_FB_N; ++c) {
                    fb[c] = 0.0f;
                }
            }
            for (int c = 0; c < t; ++c) {
                if (!((badmask >> c) & 1)) {
                    continue;
                }
                __syncthreads();
                red[wy*WARP_SIZE + lane] = fb[c];
                __syncthreads();
                if (act[a] && sp == 0) {
#pragma unroll
                    for (int l = 0; l < NL; ++l) {
                        if (c0 + qpn_col(l) == c) {
                            float v = 0.0f;
                            for (int i = 0; i < S; ++i) {
                                v += red[(wy + i)*WARP_SIZE + r0 + (l & 2)];
                            }
                            dst[c*stride_col_dst + (tile0 + a)*32 + r0 + (l & 2)] = v;
                        }
                    }
                }
            }
        }
    }
#else
    GGML_UNUSED_VARS(bx, W, nsb, ntiles, S, xh, xsc, x, stride_col_x, dst, stride_col_dst, red, sbad, t);
    NO_DEVICE_CODE;
#endif // defined(VOLTA_MMA_AVAILABLE)
}

// the block's shared state and staged activations (the same input for every segment of a launch)
template <bool STAGE>
static __device__ __forceinline__ void qpn_block_start(const half * __restrict__ xh, const int nsb, const int P, uint4 * qpn_xsm, unsigned & sbad, const int t) {
    const int lane = threadIdx.x, wy = threadIdx.y;
    if (threadIdx.x == 0 && wy == 0) {
        sbad = 0;
    }
    if constexpr (STAGE) {
        const int p   = blockIdx.y;
        const int sbA = nsb*p/P, sbB = nsb*(p + 1)/P;
        const uint4 * src = (const uint4 *) xh + (int64_t) sbA*32*t;
        const int n = (sbB - sbA)*32*t;
        for (int i = wy*WARP_SIZE + lane; i < n; i += blockDim.y*WARP_SIZE) {
            qpn_xsm[i] = __ldg(src + i);
        }
    }
    __syncthreads();
}

template <int T, int MODE> static constexpr __host__ __device__ bool qpn_wide() { return (T == 8 || T == 16 || T == 32) && MODE != 0; }
// PAIR: the paired 8-token blocks are their own instantiation, so every other kernel keeps one token count
// (t, which is also the stride): with the pairing a run-time branch, t and ts were two values in every 8-token kernel (Q5_K down,
// 8 tokens, 55.6 -> 61.0 us with the pairing)
template <ggml_type type, int T, bool STAGE, int MODE = 0, bool PUSH = false, bool PAIR = false>
static __global__ void __launch_bounds__(qpn_max_threads(MODE), qpn_min_blocks(MODE))
qpn_mul_kernel(const char * __restrict__ W, const int nsb, const int ntiles, const int S, const int P,
               const half * __restrict__ xh, const half * __restrict__ xs, const float * __restrict__ xsc,
               const float * __restrict__ x, const int64_t stride_col_x, float * __restrict__ dst, const int64_t stride_col_dst,
               float * __restrict__ ws, unsigned * __restrict__ cnt, const int nt, const int nh, ggml_cuda_p2p_push push) {
    constexpr int NL = qpn_nl<T>();
    int t = qpn_wide<T, MODE>() ? nt : T, ts = t, bx = blockIdx.x;
    // nh = 2 runs a pass of 9 to 16 tokens (nt, the prepared input's token stride) as paired 8-token blocks of this kernel at
    // T = 8: block x takes tokens 8*(x % 2) .. of tile group x/2, exactly as an 8-token call computes them, and the two blocks of a pair
    // stream the same weights side by side (the second read can come from L2)
    static_assert(!PAIR || (qpn_wide<T, MODE>() && T == 8 && !STAGE), "paired blocks: the unstaged 8-token kernel");
    if constexpr (PAIR) {
        const int hf = blockIdx.x % nh;
        bx  = blockIdx.x / nh;
        t   = min(8, nt - 8*hf);
        xh  += 8*hf*8;          // uint4 (8 halves) per token in [sb][fragment][token]
        xs  += xs ? 8*hf*8 : 0; // uint4 per token in [sb][token]
        xsc += 8*hf;            // float per token in [sb][token]
        x   += 8*hf*stride_col_x;
        dst += 8*hf*stride_col_dst;
        if constexpr (PUSH) {
            ggml_cuda_p2p_push_advance(push, 8*hf*stride_col_dst); // paired blocks advance in wire elements
        }
        if (P > 1) {
            ws  += (int64_t) hf*ntiles*P*NL*WARP_SIZE;
            cnt += hf*ntiles;
        }
    }
    extern __shared__ uint4 qpn_xsm[];
    __shared__ float    red[QPN_MAX_WARPS*NL*WARP_SIZE];
    __shared__ unsigned sbad;
    qpn_block_start<STAGE>(xh, nsb, P, qpn_xsm, sbad, t);
    if constexpr (qpn_mode_m(MODE) == 6) {
        static_assert(!STAGE && !PUSH && !PAIR, "MODE 6: unstaged single passes");
        qpn_mul_block_nat2<type, T>(bx, W, nsb, ntiles, S, xh, xsc, x, stride_col_x, dst, stride_col_dst, red, sbad, t);
    } else {
        qpn_mul_block<type, T, STAGE, MODE, PUSH>(bx, W, nsb, ntiles, S, P, xh, xs, xsc, x, stride_col_x, dst, stride_col_dst, ws, cnt,
            qpn_xsm, red, sbad, t, ts, push);
    }
}

// Two products of the same input in one launch: blocks [0, nb0) take product 0,
// the rest product 1, each with its own weights, type, rows and output, and the same K, S, R and P = 1. Each product's
// arithmetic and order are those of its own launch, so the outputs are bit-identical to two launches.
struct qpn_seg { const char * W; int ntiles; float * dst; int64_t stride_col_dst; };
template <ggml_type type0, ggml_type type1, int T, bool STAGE, int MODE>
static __global__ void __launch_bounds__(qpn_max_threads(MODE), qpn_min_blocks(MODE))
qpn_mul2_kernel(const qpn_seg g0, const qpn_seg g1, const int nb0, const int nsb, const int S,
               const half * __restrict__ xh, const half * __restrict__ xs, const float * __restrict__ xsc,
               const float * __restrict__ x, const int64_t stride_col_x, const int nt) {
    constexpr int NL = qpn_nl<T>();
    const int t = qpn_wide<T, MODE>() ? nt : T;
    extern __shared__ uint4 qpn_xsm[];
    __shared__ float    red[QPN_MAX_WARPS*NL*WARP_SIZE];
    __shared__ unsigned sbad;
    qpn_block_start<STAGE>(xh, nsb, 1, qpn_xsm, sbad, t);
    if ((int) blockIdx.x < nb0) {
        qpn_mul_block<type0, T, STAGE, MODE>(blockIdx.x, g0.W, nsb, g0.ntiles, S, 1, xh, xs, xsc, x, stride_col_x, g0.dst, g0.stride_col_dst,
            nullptr, nullptr, qpn_xsm, red, sbad, t, t);
    } else {
        qpn_mul_block<type1, T, STAGE, MODE>(blockIdx.x - nb0, g1.W, nsb, g1.ntiles, S, 1, xh, xs, xsc, x, stride_col_x, g1.dst, g1.stride_col_dst,
            nullptr, nullptr, qpn_xsm, red, sbad, t, t);
    }
}

// Three products of the same input in one launch (the attention input, q + gate, k and v, whose 1024-row k and v
// alone are 32 row tiles that cannot fill 72 SMs): blocks [0, nb0) take product 0, [nb0, nb0 + nb1) product 1, the rest product 2,
// each as in its own launch at P = 1, unstaged: q + gate at its own split (S, R), k and v at split-K S12 with blockDim.y/S12 tiles a block.
template <ggml_type type0, ggml_type type1, ggml_type type2, int T, int MODE>
static __global__ void __launch_bounds__(QPN_MAX_WARPS*WARP_SIZE, qpn_min_blocks(MODE))
qpn_mul3_kernel(const qpn_seg g0, const qpn_seg g1, const qpn_seg g2, const int nb0, const int nb1, const int nb2, const int nsb, const int S,
               const int S12, const half * __restrict__ xh, const half * __restrict__ xs, const float * __restrict__ xsc,
               const float * __restrict__ x, const int64_t stride_col_x, const int nt) {
    constexpr int NL = qpn_nl<T>();
    const int t = qpn_wide<T, MODE>() ? nt : T;
    __shared__ float    red[QPN_MAX_WARPS*NL*WARP_SIZE];
    __shared__ unsigned sbad;
    qpn_block_start<false>(xh, nsb, 1, nullptr, sbad, t);
    // products 1 and 2 (k, v) take the first blocks, with their own split-K S12 (the block's rows are then blockDim.y/S12 tiles)
    const int bx = blockIdx.x;
    if (bx < nb1) {
        qpn_mul_block<type1, T, false, MODE>(bx, g1.W, nsb, g1.ntiles, S12, 1, xh, xs, xsc, x, stride_col_x, g1.dst, g1.stride_col_dst,
            nullptr, nullptr, nullptr, red, sbad, t, t);
    } else if (bx < nb1 + nb2) {
        qpn_mul_block<type2, T, false, MODE>(bx - nb1, g2.W, nsb, g2.ntiles, S12, 1, xh, xs, xsc, x, stride_col_x, g2.dst, g2.stride_col_dst,
            nullptr, nullptr, nullptr, red, sbad, t, t);
    } else {
        qpn_mul_block<type0, T, false, MODE>(bx - nb1 - nb2, g0.W, nsb, g0.ntiles, S, 1, xh, xs, xsc, x, stride_col_x, g0.dst, g0.stride_col_dst,
            nullptr, nullptr, nullptr, red, sbad, t, t);
    }
    GGML_UNUSED(nb0);
}

// The GDN layer's three products of one input in one launch at 5 to 8 tokens: the merged alpha/beta weight ab (Q8_0, 96 rows =
// 3 row tiles) first, then in-proj (product 0) and z (product 1) exactly as qpn_mul2_kernel runs them. ab's blocks are the pair's shape
// (S warps a tile, R tiles a block), unstaged, streaming loop at T = 8 with the token count at run time (or at T = 16, 9 to 16 tokens).
template <ggml_type type0, ggml_type type1, int T, int MODE>
static __global__ void __launch_bounds__(QPN_MAX_WARPS*WARP_SIZE, qpn_min_blocks(MODE))
qpn_mul2ab_kernel(const qpn_seg gab, const qpn_seg g0, const qpn_seg g1, const int nbab, const int nb0, const int nsb, const int S,
               const half * __restrict__ xh, const half * __restrict__ xs, const float * __restrict__ xsc,
               const float * __restrict__ x, const int64_t stride_col_x, const int nt) {
    static_assert(T == 8 || T == 16, "the wide kernels");
    constexpr int NL = qpn_nl<T>();
    const int t = nt;
    __shared__ float    red[QPN_MAX_WARPS*NL*WARP_SIZE];
    __shared__ unsigned sbad;
    qpn_block_start<false>(xh, nsb, 1, nullptr, sbad, t);
    const int bx = blockIdx.x;
    if (bx < nbab) {
        qpn_mul_block<GGML_TYPE_Q8_0, T, false, MODE>(bx, gab.W, nsb, gab.ntiles, S, 1, xh, xs, xsc, x, stride_col_x, gab.dst, gab.stride_col_dst,
            nullptr, nullptr, nullptr, red, sbad, t, t);
    } else if (bx < nbab + nb0) {
        qpn_mul_block<type0, T, false, MODE>(bx - nbab, g0.W, nsb, g0.ntiles, S, 1, xh, xs, xsc, x, stride_col_x, g0.dst, g0.stride_col_dst,
            nullptr, nullptr, nullptr, red, sbad, t, t);
    } else {
        qpn_mul_block<type1, T, false, MODE>(bx - nbab - nb0, g1.W, nsb, g1.ntiles, S, 1, xh, xs, xsc, x, stride_col_x, g1.dst, g1.stride_col_dst,
            nullptr, nullptr, nullptr, red, sbad, t, t);
    }
}

// Under --split-mode tensor, the GDN layer's in-proj and z with beta and alpha, each a device's 24-row slice as a padded one-tile
// private copy (Q8_0): beta's nbb blocks, alpha's nba blocks, then in-proj (product 0) and z (product 1) as qpn_mul2ab_kernel runs them. A
// member stores the first nrowb (nrowa) rows of its tile. Only the weights are chosen before the loop; the store finds its member again from a
// fresh read of blockIdx.x (qpn_clip), so the member's output, stride and row count stay kernel parameters and are not live across the loop.
template <ggml_type type0, ggml_type type1, int T, int MODE>
static __global__ void __launch_bounds__(QPN_MAX_WARPS*WARP_SIZE, qpn_min_blocks(MODE))
qpn_mul4ab_kernel(const qpn_seg gb, const qpn_seg ga, const int nrowb, const int nrowa, const qpn_seg g0, const qpn_seg g1, const int nbb,
               const int nba, const int nb0, const int nsb, const int S, const half * __restrict__ xh, const half * __restrict__ xs,
               const float * __restrict__ xsc, const float * __restrict__ x, const int64_t stride_col_x, const int nt) {
    static_assert(T == 8 || T == 16, "the wide kernels");
    constexpr int NL = qpn_nl<T>();
    const int t = nt;
    __shared__ float    red[QPN_MAX_WARPS*NL*WARP_SIZE];
    __shared__ unsigned sbad;
    qpn_block_start<false>(xh, nsb, 1, nullptr, sbad, t);
    const int bx  = blockIdx.x;
    const int nbq = nbb + nba;
    if (bx < nbq) {
        const bool second = bx >= nbb;
        const qpn_clip clip = { gb.dst, gb.stride_col_dst, nrowb, ga.dst, ga.stride_col_dst, nrowa, nbb };
        qpn_mul_block<GGML_TYPE_Q8_0, T, false, MODE, false, true>(second ? bx - nbb : bx, second ? ga.W : gb.W, nsb, second ? ga.ntiles : gb.ntiles,
            S, 1, xh, xs, xsc, x, stride_col_x, nullptr, 0, nullptr, nullptr, nullptr, red, sbad, t, t, {}, clip);
    } else if (bx < nbq + nb0) {
        qpn_mul_block<type0, T, false, MODE>(bx - nbq, g0.W, nsb, g0.ntiles, S, 1, xh, xs, xsc, x, stride_col_x, g0.dst, g0.stride_col_dst,
            nullptr, nullptr, nullptr, red, sbad, t, t);
    } else {
        qpn_mul_block<type1, T, false, MODE>(bx - nbq - nb0, g1.W, nsb, g1.ntiles, S, 1, xh, xs, xsc, x, stride_col_x, g1.dst, g1.stride_col_dst,
            nullptr, nullptr, nullptr, red, sbad, t, t);
    }
}

struct qpn_args {
    const char * W; int nsb, ntiles, S, P;
    half * xh, * xs; float * xsc;
    const float * x; int64_t stride_col_x; float * dst; int64_t stride_col_dst;
    float * ws; unsigned * cnt;
    int nh = 1; // 2 = paired 8-token blocks (qpn_mul_kernel)
    ggml_cuda_p2p_push push; // peer == nullptr, no push
};

// true if it launched the pushing kernel (a.push set, and one of the TP verify's kernels: 5 to 16 tokens, unstaged; only those
// are instantiated with the push, so every other kernel is the one-card kernel unchanged)
template <ggml_type type, int T, bool STAGE, int MODE, bool PAIR = false>
static bool ggml_cuda_qpn_launch_m(const int device, const dim3 grid, const dim3 block, const size_t smem, cudaStream_t stream, const qpn_args & a,
        const int nt) {
    if constexpr (STAGE) { // staged activations prefer all of the shared memory carveout; set once per device and kernel
        static bool done[GGML_CUDA_MAX_DEVICES] = { false };
        if (!done[device]) {
            CUDA_CHECK(cudaFuncSetAttribute(qpn_mul_kernel<type, T, true, MODE>, cudaFuncAttributePreferredSharedMemoryCarveout,
                (int) cudaSharedmemCarveoutMaxShared));
            done[device] = true;
        }
    }
    if constexpr (!STAGE && qpn_mode_m(MODE) != 4 && (T == 8 || (T == 16 && (qpn_mode_m(MODE) == 2 || qpn_mode_m(MODE) == 5)))) {
        if (a.push.peer != nullptr) {
            qpn_mul_kernel<type, T, false, MODE, true, PAIR><<<grid, block, 0, stream>>>(a.W, a.nsb, a.ntiles, a.S, a.P, a.xh, a.xs, a.xsc, a.x,
                a.stride_col_x, a.dst, a.stride_col_dst, a.ws, a.cnt, nt, a.nh, a.push);
            return true;
        }
    }
    qpn_mul_kernel<type, T, STAGE, MODE, false, PAIR><<<grid, block, STAGE ? smem : 0, stream>>>(a.W, a.nsb, a.ntiles, a.S, a.P, a.xh, a.xs, a.xsc, a.x,
        a.stride_col_x, a.dst, a.stride_col_dst, a.ws, a.cnt, nt, a.nh, a.push);
    return false;
}

// LLAMA_QPN_FULL (default on; 0 = off): a pass of exactly 8 or 16 tokens on a streaming kernel runs its QPN_FULL variant
static inline bool ggml_cuda_qpn_full_on() {
    static const bool on = [] { const char * e = getenv("LLAMA_QPN_FULL"); return e == nullptr || atoi(e) != 0; }();
    return on;
}

// [TAG_QPN_RING] MODE 7 (prototype; Q6_K, 1 to 4 tokens, P = 1): a producer/consumer weight ring. ncu on the streaming loop: DRAM
// at 77% of peak, issue slots 23% busy, 78% of stall samples on L1TEX loads, so the loads in flight do not cover HBM's latency.
// Persistent blocks (one per SM) loop over the row tiles; each has QPN_RING_S consumer warps (split-K over one tile, as S warps of
// MODE 3) and QPN_RING_NP producer warps. A producer serves QPN_RING_S/QPN_RING_NP consumers: it loads each one's next
// tile-superblock (TSB contiguous bytes) into registers for all of them, then waits for a free slot of that consumer's
// QPN_RING_NS-deep ring in shared memory, stores it and publishes a fill count. A consumer runs the streaming loop's Q6_K arithmetic
// at LA = 2 expression for expression, with the weight chunks read from its ring slot (activations through __ldg, which stay in
// the L1 beside the 64 KB carveout), and the S consumers' sums are added in the order sp = 0 .. S-1, so the outputs are
// bit-identical to MODE 3 at S = 4 (LLAMA_QPN_CHECK=3,1 checks it). The counters are monotonic per consumer, in shared memory.
// A block takes QPN_RING_R tiles at a time (S*R consumers): at R = 1 (4 consumers, one a scheduler) nothing hid an instruction's
// latency and the head ran at 1795 us against MODE 3's 1559. A slot is one tile-superblock; at NS = 1 the producer's registers
// hold the next one, so a consumer's ring is still two deep.
// One producer a consumer: with a producer serving two consumers in turn, its next loads waited until both had freed their slot,
// so each round exposed DRAM's latency (ncu, staged at 4 tokens: 17% of stall samples were consumers polling their fill count)
#define QPN_RING_S  4
#define QPN_RING_R  2
#define QPN_RING_NP 8
#define QPN_RING_NS 1
static __device__ __forceinline__ void qpn_ring_bar() { // the consumers only (named barrier 1)
    asm volatile("bar.sync 1, %0;" :: "r"(QPN_RING_S*QPN_RING_R*WARP_SIZE) : "memory");
}

// STG: the activations staged in shared memory beside the ring (when they fit: K = 5120 at 4 tokens is 40 KB), so a consumer's
// mma reads its B fragment at shared-memory latency (ncu on the ring with B through __ldg: 16% of stall samples were mma waiting
// on those loads, L1 hit rate 29%, a consumer ~0.12 instructions a cycle). Lanes of padding tokens read a real token's staged
// fragment, as MODE 3 staged does (a real token's C column depends only on its own B column)
template <ggml_type type, int T, bool STG>
static __global__ void __launch_bounds__((QPN_RING_S*QPN_RING_R + QPN_RING_NP)*WARP_SIZE, 1)
qpn_ring_kernel(const char * __restrict__ W, const int nsb, const int ntiles, const half * __restrict__ xh, const float * __restrict__ xsc,
                const float * __restrict__ x, const int64_t stride_col_x, float * __restrict__ dst, const int64_t stride_col_dst) {
#if defined(VOLTA_MMA_AVAILABLE)
    static_assert(type == GGML_TYPE_Q6_K && T <= 4, "MODE 7: the Q6_K prototype at 1 to 4 tokens");
    typedef tile<32, 4, half2>                               tile_A;
    typedef tile< 8, 4, half2, DATA_LAYOUT_I_MAJOR_MIRRORED> tile_B;
    typedef tile<32, 8, float>                               tile_C;
    constexpr int S = QPN_RING_S, R = QPN_RING_R, NC = S*R, NP = QPN_RING_NP, NS = QPN_RING_NS, CPP = NC/NP; // consumers a producer
    static_assert(NC % NP == 0, "each producer serves the same number of consumers");
    constexpr int NCH = qpn_t<type>::NCH, TSB = qpn_tsb<type>(), TSB16 = TSB/16;  // Q6_K: 13 chunks + 64 bytes (lanes 0-3)
    constexpr int NL  = qpn_nl<T>();
    constexpr int t = T, ts = T;
    static_assert(TSB == NCH*QPN_CHUNK + 64, "Q6_K's tile-superblock: 13 chunks and the 32 rows' d");

    extern __shared__ uint4 qpn_ring[];          // [NC][NS][TSB16], then (STG) the activations [nsb][32 fragments][t]
    __shared__ float    red[(S - 1)*R*NL*WARP_SIZE]; // the partial sums of warps sp = 1 .. S-1 of each tile (warp 0 adds them)
    __shared__ int      filled[NC], consumed[NC];
    __shared__ unsigned sbad[2];
    static_assert((S - 1)*R*NL >= NC, "the fallback's column exchange (one value a lane per consumer) fits in red");
    uint4 * xsm = qpn_ring + NC*NS*TSB16;

    const int lane = threadIdx.x, wy = threadIdx.y;
    if (wy == 0 && lane < NC) {
        filled[lane] = consumed[lane] = 0;
    }
    if (wy == 0 && lane < 2) {
        sbad[lane] = 0;
    }
    if constexpr (STG) {
        const int n16 = nsb*32*t;
        for (int i = wy*WARP_SIZE + lane; i < n16; i += blockDim.y*WARP_SIZE) {
            xsm[i] = __ldg((const uint4 *) xh + i);
        }
    }
    __syncthreads();

    if (wy >= NC) { // ---- producer p: consumers p, p + NP, ... (consumer c: tile c / S of the group, superblocks c % S + i*S)
        const int p = wy - NC;
        int seq[CPP] = {};
        for (int tg = blockIdx.x; tg*R < ntiles; tg += gridDim.x) {
            for (int i = 0; S*i < nsb; ++i) {
                uint4 v[CPP][NCH + 1];
#pragma unroll
                for (int k = 0; k < CPP; ++k) {
                    const int c = p + k*NP, tile = tg*R + c / S, sb = c % S + i*S;
                    if (tile < ntiles && sb < nsb) {
                        const char * src = W + ((int64_t) tile*nsb + sb)*TSB + lane*16;
#pragma unroll
                        for (int ch = 0; ch < NCH; ++ch) {
                            v[k][ch] = qpn_ldg_w(src + ch*QPN_CHUNK);
                        }
                        v[k][NCH] = lane < 4 ? qpn_ldg_w(src + NCH*QPN_CHUNK) : make_uint4(0, 0, 0, 0);
                    }
                }
#pragma unroll
                for (int k = 0; k < CPP; ++k) {
                    const int c = p + k*NP, tile = tg*R + c / S, sb = c % S + i*S;
                    if (tile >= ntiles || sb >= nsb) {
                        continue;
                    }
                    const int n = seq[k]++;
                    while (n - *(volatile int *) &consumed[c] >= NS) { }
                    uint4 * slot = qpn_ring + (c*NS + n % NS)*TSB16;
#pragma unroll
                    for (int ch = 0; ch < NCH; ++ch) {
                        slot[ch*WARP_SIZE + lane] = v[k][ch];
                    }
                    if (lane < 4) {
                        slot[NCH*WARP_SIZE + lane] = v[k][NCH];
                    }
                    __syncwarp();
                    __threadfence_block();
                    if (lane == 0) {
                        *(volatile int *) &filled[c] = n + 1;
                    }
                }
            }
        }
        return;
    }

    // ---- consumer wy: warp sp of the group's tile wy / S
    const int sp = wy % S;
    const int nB = min(tile_B::get_i(0), t - 1);
    const int r0 = lane & ~2, c0 = lane & 2;
    const int dw = NCH*QPN_CHUNK/4 + (r0 >> 2)*2 + (r0 & 1); // u32 index in the slot of Q6_K's d(r0), d(r0 + 2)
    const uint32_t mg = QPN_MAGIC | __ldg(&qpn_zero_x[0].x);
    int n = 0, par = 0;
    for (int tg = blockIdx.x; tg*R < ntiles; tg += gridDim.x, par ^= 1) {
        const int  tile = tg*R + wy / S;
        const bool act  = tile < ntiles;
        float    D[NL] = {0.0f};
        unsigned bad   = 0;
        for (int sb = sp; act && sb < nsb; sb += S, ++n) {
            while (*(volatile int *) &filled[wy] <= n) { }
            __threadfence_block();
            const uint4 * sl = qpn_ring + (wy*NS + n % NS)*TSB16;
            const uint4 hdr = sl[12*WARP_SIZE + lane];
            const uint2 tl  = make_uint2(((const uint32_t *) sl)[dw], 0);
            uint32_t scu[4];
            scu[0] = hdr.x ^ 0x80808080u; scu[1] = hdr.y ^ 0x80808080u; scu[2] = hdr.z ^ 0x80808080u; scu[3] = hdr.w ^ 0x80808080u;
            const int nBr = tile_B::get_i(0);
            const uint4 * xk = STG ? xsm + (sb*32*t + nB) : nBr < t ? (const uint4 *) xh + ((int64_t) sb*32*ts + nB) : qpn_zero_x + nBr;
            float cs[NL/2];
#pragma unroll
            for (int i = 0; i < NL/2; ++i) {
                const int col = c0 + (i & 1) + 4*(i/2);
                cs[i] = col < t ? xsc[sb*ts + col] : 1.0f;
                bad |= (col < t && cs[i] == 0.0f) ? 1u << col : 0u;
            }
            tile_C Ds[2];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const uint4    cw     = sl[j*WARP_SIZE + lane];
                const uint32_t cwv[4] = {cw.x, cw.y, cw.z, cw.w};
                const uint4    hc     = sl[(8 + j/2)*WARP_SIZE + lane];
                const uint32_t hcv[2] = {(j & 1) ? hc.z : hc.x, (j & 1) ? hc.w : hc.y};
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const int   sbi = 2*j + q/2;
                    const half2 sc  = __hsub2(qpn_u2h(__byte_perm(scu[sbi/4], 0x64, (sbi % 4) | (4 << 4) | ((sbi % 4) << 8) | (4 << 12))),
                                              make_half2(1152.0f, 1152.0f));
                    const half2 s16 = __hmul2(sc, make_half2(0.0625f, 0.0625f));
                    const half2 z   = __hmul2(sc, make_half2(-96.0f, -96.0f));
                    const uint32_t H = hcv[q/2];
                    tile_A A;
                    qpn_decode<type>(A, cwv[q], __funnelshift_r(H, H, 2*(q & 1)), s16, z, mg);
                    tile_B B;
                    *(uint4 *) B.x = STG ? xk[(4*j + q)*t] : __ldg(xk + (4*j + q)*ts);
                    mma(Ds[q & 1], A, B);
                }
            }
            __syncwarp();
            __threadfence_block();
            if (lane == 0) {
                *(volatile int *) &consumed[wy] = n + 1;
            }
            const float2 d02 = __half22float2(qpn_u2h(tl.x));
#pragma unroll
            for (int l = 0; l < NL; ++l) {
                const float d = (l & 2) ? d02.y : d02.x;
                const float c = cs[qpn_csi(l)];
                D[l] += c * d * (Ds[0].x[l & 7] + Ds[1].x[l & 7]);
            }
        }

        // the tile's S partial sums in the order sp = 0 .. S-1 (qpn_mul_block's at P = 1), then the store
        if (bad) {
            atomicOr(&sbad[par], bad);
        }
        const int rb = (wy / S)*(S - 1) - 1; // red's slot of warp sp of this tile: rb + sp (sp >= 1)
        if (S > 1 && sp > 0) {
#pragma unroll
            for (int l = 0; l < NL; ++l) {
                red[((rb + sp)*NL + l)*WARP_SIZE + lane] = D[l];
            }
        }
        qpn_ring_bar();
        const unsigned badmask = sbad[par];
        if (sp == 0 && act) {
            for (int i = 1; i < S; ++i) {
#pragma unroll
                for (int l = 0; l < NL; ++l) {
                    D[l] += red[((rb + i)*NL + l)*WARP_SIZE + lane];
                }
            }
#pragma unroll
            for (int l = 0; l < NL; ++l) {
                const int row = tile*32 + r0 + (l & 2);
                const int col = c0 + qpn_col(l);
                if (col < t) {
                    dst[col*stride_col_dst + row] = D[l];
                }
            }
        }
        if (wy == 0 && lane == 0) {
            sbad[par ^ 1] = 0; // the next group's
        }
        qpn_ring_bar();
        if (badmask) { // block-uniform: those columns again in fp32 from the dequantized weights
            float fb[QPN_FB_N];
            if (act) {
                qpn_fallback<type>(W + (int64_t) tile*nsb*TSB, sp, nsb, S, badmask, t, x, stride_col_x, fb);
            } else {
                for (int c = 0; c < QPN_FB_N; ++c) {
                    fb[c] = 0.0f;
                }
            }
            for (int c = 0; c < t; ++c) {
                if (!((badmask >> c) & 1)) {
                    continue;
                }
                red[wy*WARP_SIZE + lane] = fb[c];
                qpn_ring_bar();
                if (sp == 0 && act) {
#pragma unroll
                    for (int l = 0; l < NL; ++l) {
                        if (c0 + qpn_col(l) == c) {
                            float v = 0.0f;
                            for (int i = 0; i < S; ++i) {
                                v += red[(wy + i)*WARP_SIZE + r0 + (l & 2)];
                            }
                            dst[c*stride_col_dst + tile*32 + r0 + (l & 2)] = v;
                        }
                    }
                }
                qpn_ring_bar();
            }
        }
    }
#else
    GGML_UNUSED_VARS(W, nsb, ntiles, xh, xsc, x, stride_col_x, dst, stride_col_dst);
    NO_DEVICE_CODE;
#endif // defined(VOLTA_MMA_AVAILABLE)
}

// the ring's dynamic shared memory: QPN_RING_S consumers x QPN_RING_NS slots of one tile-superblock
template <ggml_type type> static constexpr size_t qpn_ring_smem() { return (size_t) QPN_RING_S*QPN_RING_R*QPN_RING_NS*qpn_tsb<type>(); }

// the activations staged when ring, activations and the static state (about 3.2 KB) fit in the SM's 96 KB
static constexpr size_t QPN_RING_SMEM_MAX = 96*1024 - 3*1024 - 256;

template <ggml_type type, int T, bool STG>
static void ggml_cuda_qpn_ring_launch_s(const int device, const dim3 grid, const dim3 block, const size_t smem, cudaStream_t stream, const qpn_args & a) {
    static bool done[GGML_CUDA_MAX_DEVICES] = { false };
    if (!done[device]) {
        CUDA_CHECK(cudaFuncSetAttribute(qpn_ring_kernel<type, T, STG>, cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int) (STG ? QPN_RING_SMEM_MAX : qpn_ring_smem<type>())));
        CUDA_CHECK(cudaFuncSetAttribute(qpn_ring_kernel<type, T, STG>, cudaFuncAttributePreferredSharedMemoryCarveout,
            STG ? (int) cudaSharedmemCarveoutMaxShared : 64*100/96));
        done[device] = true;
    }
    qpn_ring_kernel<type, T, STG><<<grid, block, smem, stream>>>(a.W, a.nsb, a.ntiles, a.xh, a.xsc, a.x, a.stride_col_x,
        a.dst, a.stride_col_dst);
}

template <ggml_type type, int T>
static bool ggml_cuda_qpn_ring_launch(const int device, const dim3 grid, const dim3 block, cudaStream_t stream, const qpn_args & a) {
    const size_t act = (size_t) a.nsb*32*T*sizeof(uint4);
    if (qpn_ring_smem<type>() + act <= QPN_RING_SMEM_MAX) {
        ggml_cuda_qpn_ring_launch_s<type, T, true>(device, grid, block, qpn_ring_smem<type>() + act, stream, a);
    } else {
        ggml_cuda_qpn_ring_launch_s<type, T, false>(device, grid, block, qpn_ring_smem<type>(), stream, a);
    }
    return false;
}

// mode 0: the whole-superblock loop at QPN_MIN_BLOCKS blocks per SM, one kernel per T; 2, 3, 4: qpn_stream_loop at that many
// blocks of 8 warps per SM, one kernel per T at 1 to 4 tokens and
// one kernel at T = 8 for 5 to 8 tokens; at exactly 8 tokens, unstaged, its QPN_FULL variant
template <ggml_type type, int T>
static bool ggml_cuda_qpn_launch_t(const bool stage, const int mode, const dim3 grid, const dim3 block, const size_t smem, cudaStream_t stream, const qpn_args & a) {
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    constexpr int TK = T <= 4 ? T : 8;
    if constexpr (T == 8) {
        if (!stage && ggml_cuda_qpn_full_on()) {
            switch (mode) {
                case 2: return ggml_cuda_qpn_launch_m<type, 8, false, 2 + QPN_FULL>(device, grid, block, smem, stream, a, T);
                case 3: return ggml_cuda_qpn_launch_m<type, 8, false, 3 + QPN_FULL>(device, grid, block, smem, stream, a, T);
                case 5: return ggml_cuda_qpn_launch_m<type, 8, false, 5 + QPN_FULL>(device, grid, block, smem, stream, a, T);
                default: break;
            }
        }
    }
    switch (mode) {
        case 2: return stage ? ggml_cuda_qpn_launch_m<type, TK, true, 2>(device, grid, block, smem, stream, a, T) : ggml_cuda_qpn_launch_m<type, TK, false, 2>(device, grid, block, smem, stream, a, T);
        case 3: return stage ? ggml_cuda_qpn_launch_m<type, TK, true, 3>(device, grid, block, smem, stream, a, T) : ggml_cuda_qpn_launch_m<type, TK, false, 3>(device, grid, block, smem, stream, a, T);
        case 4: return stage ? ggml_cuda_qpn_launch_m<type, TK, true, 4>(device, grid, block, smem, stream, a, T) : ggml_cuda_qpn_launch_m<type, TK, false, 4>(device, grid, block, smem, stream, a, T);
        case 5: if constexpr (TK == 8) { GGML_ASSERT(!stage); return ggml_cuda_qpn_launch_m<type, 8, false, 5>(device, grid, block, smem, stream, a, T); } break;
        case 6: // [TAG_QPN_NAT2] two row tiles a warp: the Q6_K prototype at 1 to 4 tokens, unstaged, P = 1
            if constexpr (type == GGML_TYPE_Q6_K && T <= 4) {
                GGML_ASSERT(!stage && a.P == 1 && a.nh == 1);
                return ggml_cuda_qpn_launch_m<type, T, false, 6>(device, grid, block, smem, stream, a, T);
            }
            GGML_ABORT("MODE 6 is the Q6_K prototype at 1 to 4 tokens");
        case 7: // [TAG_QPN_RING] the producer/consumer weight ring: the Q6_K prototype at 1 to 4 tokens, P = 1
            if constexpr (type == GGML_TYPE_Q6_K && T <= 4) {
                GGML_ASSERT(!stage && a.P == 1 && a.nh == 1 && a.S == QPN_RING_S);
                return ggml_cuda_qpn_ring_launch<type, T>(device, grid, block, stream, a);
            }
            GGML_ABORT("MODE 7 is the Q6_K prototype at 1 to 4 tokens");
        default: break;
    }
    return stage ? ggml_cuda_qpn_launch_m<type, T, true, 0>(device, grid, block, smem, stream, a, T) : ggml_cuda_qpn_launch_m<type, T, false, 0>(device, grid, block, smem, stream, a, T);
}

template <ggml_type type>
bool ggml_cuda_qpn_launch(const int T, const bool stage, const int mode, const dim3 grid, const dim3 block, const size_t smem, cudaStream_t stream, const qpn_args & a) {
    switch (T) {
        case 1: return ggml_cuda_qpn_launch_t<type, 1>(stage, mode, grid, block, smem, stream, a);
        case 2: return ggml_cuda_qpn_launch_t<type, 2>(stage, mode, grid, block, smem, stream, a);
        case 3: return ggml_cuda_qpn_launch_t<type, 3>(stage, mode, grid, block, smem, stream, a);
        case 4: return ggml_cuda_qpn_launch_t<type, 4>(stage, mode, grid, block, smem, stream, a);
        case 5: return ggml_cuda_qpn_launch_t<type, 5>(stage, mode, grid, block, smem, stream, a);
        case 6: return ggml_cuda_qpn_launch_t<type, 6>(stage, mode, grid, block, smem, stream, a);
        case 7: return ggml_cuda_qpn_launch_t<type, 7>(stage, mode, grid, block, smem, stream, a);
        case 8: return ggml_cuda_qpn_launch_t<type, 8>(stage, mode, grid, block, smem, stream, a);
        default: {
            int device;
            CUDA_CHECK(cudaGetDevice(&device));
            // [TAG_QPN_W32] 17 to 32 tokens, one kernel at T = 32 (four B tiles, the token count at run time), unstaged, one block
            // of 8 warps per SM (MODE 1: the four tiles' accumulators take more than the 128 registers of two blocks)
            if (T > 16) {
                GGML_ASSERT(T <= 32 && !stage && a.nh == 1 && mode == 1);
                return T == 32 && ggml_cuda_qpn_full_on() ? ggml_cuda_qpn_launch_m<type, 32, false, 1 + QPN_FULL>(device, grid, block, smem, stream, a, T)
                                                          : ggml_cuda_qpn_launch_m<type, 32, false, 1>(device, grid, block, smem, stream, a, T);
            }
            // 9 to 16 tokens, one kernel at T = 16 (two B tiles, the token count at run time), unstaged, 2 blocks of 8 warps per SM;
            // or (a.nh = 2) paired blocks of the kernel at T = 8, unstaged, at 3 blocks of 8 warps per SM
            GGML_ASSERT(T >= 9 && T <= 16 && !stage && (a.nh == 1 ? mode == 2 || mode == 5 : mode == 3));
            const bool full = T == 16 && ggml_cuda_qpn_full_on();
            if (a.nh == 1 && mode == 5) {
                return full ? ggml_cuda_qpn_launch_m<type, 16, false, 5 + QPN_FULL>(device, grid, block, smem, stream, a, T)
                            : ggml_cuda_qpn_launch_m<type, 16, false, 5>(device, grid, block, smem, stream, a, T);
            }
            if (a.nh == 1) {
                return full ? ggml_cuda_qpn_launch_m<type, 16, false, 2 + QPN_FULL>(device, grid, block, smem, stream, a, T)
                            : ggml_cuda_qpn_launch_m<type, 16, false, 2>(device, grid, block, smem, stream, a, T);
            }
            return ggml_cuda_qpn_launch_m<type, 8, false, 3, true>(device, grid, block, smem, stream, a, T);
        }
    }
}

static constexpr int qpn_type_idx(const ggml_type t) {
    return t == GGML_TYPE_Q4_K ? 0 : t == GGML_TYPE_Q5_K ? 1 : t == GGML_TYPE_Q6_K ? 2 : t == GGML_TYPE_IQ4_XS ? 3 : t == GGML_TYPE_IQ4_NL ? 4 :
           t == GGML_TYPE_Q3_K ? 5 : -1; // Q3_K (the target's Q3_K ffn gate/up pair with IQ4_XS); Q8_0 is in no configured pair
}

template <ggml_type t0, ggml_type t1, int T, bool STAGE, int MODE = 3>
static void ggml_cuda_qpn2_launch_k(const dim3 grid, const dim3 block, const size_t smem, cudaStream_t stream, const qpn_seg & g0, const qpn_seg & g1,
        const int nb0, const int nsb, const int S, const half * xh, const half * xs, const float * xsc, const float * x, const int64_t sx, const int nt) {
    if constexpr (STAGE) {
        static bool done[GGML_CUDA_MAX_DEVICES] = { false };
        int device;
        CUDA_CHECK(cudaGetDevice(&device));
        if (!done[device]) {
            CUDA_CHECK(cudaFuncSetAttribute(qpn_mul2_kernel<t0, t1, T, true, MODE>, cudaFuncAttributePreferredSharedMemoryCarveout,
                (int) cudaSharedmemCarveoutMaxShared));
            done[device] = true;
        }
    }
    qpn_mul2_kernel<t0, t1, T, STAGE, MODE><<<grid, block, STAGE ? smem : 0, stream>>>(g0, g1, nb0, nsb, S, xh, xs, xsc, x, sx, nt);
}

template <ggml_type t0, ggml_type t1>
void ggml_cuda_qpn2_launch_tt(const int T, const int stage, const dim3 grid, const dim3 block, const size_t smem, cudaStream_t stream,
        const qpn_seg & g0, const qpn_seg & g1, const int nb0, const int nsb, const int S, const half * xh, const half * xs, const float * xsc,
        const float * x, const int64_t sx) {
    if constexpr (qpn_type_idx(t0) <= qpn_type_idx(t1)) {
        if (T == 1 || T == 2) {
            // the draft's Q6_K q + k or v (ggml_cuda_qpn_draft_pair_ok), unstaged; a pair of slices of one type
            // under --split-mode tensor (the ffn gate and up of the MTP draft step), at its split, staged or not
            if constexpr (t0 == t1) {
                if (T == 1) {
                    stage ? ggml_cuda_qpn2_launch_k<t0, t1, 1, true >(grid, block, smem, stream, g0, g1, nb0, nsb, S, xh, xs, xsc, x, sx, T)
                          : ggml_cuda_qpn2_launch_k<t0, t1, 1, false>(grid, block, smem, stream, g0, g1, nb0, nsb, S, xh, xs, xsc, x, sx, T);
                } else {
                    stage ? ggml_cuda_qpn2_launch_k<t0, t1, 2, true >(grid, block, smem, stream, g0, g1, nb0, nsb, S, xh, xs, xsc, x, sx, T)
                          : ggml_cuda_qpn2_launch_k<t0, t1, 2, false>(grid, block, smem, stream, g0, g1, nb0, nsb, S, xh, xs, xsc, x, sx, T);
                }
            } else {
                GGML_ABORT("fatal error");
            }
        } else if (T == 3) {
            stage ? ggml_cuda_qpn2_launch_k<t0, t1, 3, true >(grid, block, smem, stream, g0, g1, nb0, nsb, S, xh, xs, xsc, x, sx, T)
                  : ggml_cuda_qpn2_launch_k<t0, t1, 3, false>(grid, block, smem, stream, g0, g1, nb0, nsb, S, xh, xs, xsc, x, sx, T);
        } else if (T == 4) {
            stage ? ggml_cuda_qpn2_launch_k<t0, t1, 4, true >(grid, block, smem, stream, g0, g1, nb0, nsb, S, xh, xs, xsc, x, sx, T)
                  : ggml_cuda_qpn2_launch_k<t0, t1, 4, false>(grid, block, smem, stream, g0, g1, nb0, nsb, S, xh, xs, xsc, x, sx, T);
        } else if (T <= 8) { // 5 to 8 tokens, one kernel at T = 8, unstaged, at 2 or 3 blocks of 8 warps per SM (stage holds it)
            if (T == 8 && ggml_cuda_qpn_full_on()) {
                stage == 2 ? ggml_cuda_qpn2_launch_k<t0, t1, 8, false, 2 + QPN_FULL>(grid, block, smem, stream, g0, g1, nb0, nsb, S, xh, xs, xsc, x, sx, T)
                           : ggml_cuda_qpn2_launch_k<t0, t1, 8, false, 3 + QPN_FULL>(grid, block, smem, stream, g0, g1, nb0, nsb, S, xh, xs, xsc, x, sx, T);
            } else {
                stage == 2 ? ggml_cuda_qpn2_launch_k<t0, t1, 8, false, 2>(grid, block, smem, stream, g0, g1, nb0, nsb, S, xh, xs, xsc, x, sx, T)
                           : ggml_cuda_qpn2_launch_k<t0, t1, 8, false, 3>(grid, block, smem, stream, g0, g1, nb0, nsb, S, xh, xs, xsc, x, sx, T);
            }
        } else { // 9 to 16 tokens, one kernel at T = 16 (two B tiles), unstaged, at 2 blocks of 8 warps per SM
            if (T == 16 && ggml_cuda_qpn_full_on()) {
                ggml_cuda_qpn2_launch_k<t0, t1, 16, false, 2 + QPN_FULL>(grid, block, smem, stream, g0, g1, nb0, nsb, S, xh, xs, xsc, x, sx, T);
            } else {
                ggml_cuda_qpn2_launch_k<t0, t1, 16, false, 2>(grid, block, smem, stream, g0, g1, nb0, nsb, S, xh, xs, xsc, x, sx, T);
            }
        }
    } else {
        GGML_ABORT("fatal error"); // the caller orders the pair by type
    }
}

// the instantiated types: product 0 (q + gate) Q4_K, Q5_K or IQ4_XS; products 1 and 2 (k, v) Q4_K, Q5_K, Q6_K or Q8_0,
// in the order of qpn3_idx
static constexpr bool qpn3_t0_ok(const ggml_type t) { return t == GGML_TYPE_Q4_K || t == GGML_TYPE_Q5_K || t == GGML_TYPE_IQ4_XS; }
static constexpr int  qpn3_idx(const ggml_type t) {
    return t == GGML_TYPE_Q4_K ? 0 : t == GGML_TYPE_Q5_K ? 1 : t == GGML_TYPE_Q6_K ? 2 : t == GGML_TYPE_Q8_0 ? 3 : -1;
}
static constexpr bool qpn3_t1_ok(const ggml_type t) { return qpn3_idx(t) >= 0; }

template <ggml_type t0, ggml_type t1, ggml_type t2>
static void ggml_cuda_qpn3_launch_ttt(const int T, const dim3 grid, const dim3 block, cudaStream_t stream, const qpn_seg & g0, const qpn_seg & g1,
        const qpn_seg & g2, const int nb0, const int nb1, const int nb2, const int nsb, const int S, const int S12, const half * xh, const half * xs, const float * xsc,
        const float * x, const int64_t sx) {
    if constexpr (qpn3_idx(t1) <= qpn3_idx(t2)) {
        if (T == 4) {
            qpn_mul3_kernel<t0, t1, t2, 4, 3><<<grid, block, 0, stream>>>(g0, g1, g2, nb0, nb1, nb2, nsb, S, S12, xh, xs, xsc, x, sx, T);
        } else if (T <= 8) {
            if (T == 8 && ggml_cuda_qpn_full_on()) {
                qpn_mul3_kernel<t0, t1, t2, 8, 3 + QPN_FULL><<<grid, block, 0, stream>>>(g0, g1, g2, nb0, nb1, nb2, nsb, S, S12, xh, xs, xsc, x, sx, T);
            } else {
                qpn_mul3_kernel<t0, t1, t2, 8, 3><<<grid, block, 0, stream>>>(g0, g1, g2, nb0, nb1, nb2, nsb, S, S12, xh, xs, xsc, x, sx, T);
            }
        } else { // 9 to 16 tokens, two B tiles, at 2 blocks of 8 warps per SM
            if (T == 16 && ggml_cuda_qpn_full_on()) {
                qpn_mul3_kernel<t0, t1, t2, 16, 2 + QPN_FULL><<<grid, block, 0, stream>>>(g0, g1, g2, nb0, nb1, nb2, nsb, S, S12, xh, xs, xsc, x, sx, T);
            } else {
                qpn_mul3_kernel<t0, t1, t2, 16, 2><<<grid, block, 0, stream>>>(g0, g1, g2, nb0, nb1, nb2, nsb, S, S12, xh, xs, xsc, x, sx, T);
            }
        }
    } else {
        GGML_ABORT("fatal error"); // the caller orders k and v by type
    }
}

template <ggml_type t0, ggml_type t1>
static void ggml_cuda_qpn3_launch_tt(const ggml_type t2, const int T, const dim3 grid, const dim3 block, cudaStream_t stream, const qpn_seg & g0,
        const qpn_seg & g1, const qpn_seg & g2, const int nb0, const int nb1, const int nb2, const int nsb, const int S, const int S12, const half * xh, const half * xs,
        const float * xsc, const float * x, const int64_t sx) {
    switch (t2) {
        case GGML_TYPE_Q4_K: ggml_cuda_qpn3_launch_ttt<t0, t1, GGML_TYPE_Q4_K>(T, grid, block, stream, g0, g1, g2, nb0, nb1, nb2, nsb, S, S12, xh, xs, xsc, x, sx); break;
        case GGML_TYPE_Q5_K: ggml_cuda_qpn3_launch_ttt<t0, t1, GGML_TYPE_Q5_K>(T, grid, block, stream, g0, g1, g2, nb0, nb1, nb2, nsb, S, S12, xh, xs, xsc, x, sx); break;
        case GGML_TYPE_Q6_K: ggml_cuda_qpn3_launch_ttt<t0, t1, GGML_TYPE_Q6_K>(T, grid, block, stream, g0, g1, g2, nb0, nb1, nb2, nsb, S, S12, xh, xs, xsc, x, sx); break;
        case GGML_TYPE_Q8_0: ggml_cuda_qpn3_launch_ttt<t0, t1, GGML_TYPE_Q8_0>(T, grid, block, stream, g0, g1, g2, nb0, nb1, nb2, nsb, S, S12, xh, xs, xsc, x, sx); break;
        default: GGML_ABORT("fatal error");
    }
}

template <ggml_type t0>
void ggml_cuda_qpn3_launch_t(const ggml_type t1, const ggml_type t2, const int T, const dim3 grid, const dim3 block, cudaStream_t stream,
        const qpn_seg & g0, const qpn_seg & g1, const qpn_seg & g2, const int nb0, const int nb1, const int nb2, const int nsb, const int S, const int S12, const half * xh,
        const half * xs, const float * xsc, const float * x, const int64_t sx) {
    switch (t1) {
        case GGML_TYPE_Q4_K: ggml_cuda_qpn3_launch_tt<t0, GGML_TYPE_Q4_K>(t2, T, grid, block, stream, g0, g1, g2, nb0, nb1, nb2, nsb, S, S12, xh, xs, xsc, x, sx); break;
        case GGML_TYPE_Q5_K: ggml_cuda_qpn3_launch_tt<t0, GGML_TYPE_Q5_K>(t2, T, grid, block, stream, g0, g1, g2, nb0, nb1, nb2, nsb, S, S12, xh, xs, xsc, x, sx); break;
        case GGML_TYPE_Q6_K: ggml_cuda_qpn3_launch_tt<t0, GGML_TYPE_Q6_K>(t2, T, grid, block, stream, g0, g1, g2, nb0, nb1, nb2, nsb, S, S12, xh, xs, xsc, x, sx); break;
        case GGML_TYPE_Q8_0: ggml_cuda_qpn3_launch_tt<t0, GGML_TYPE_Q8_0>(t2, T, grid, block, stream, g0, g1, g2, nb0, nb1, nb2, nsb, S, S12, xh, xs, xsc, x, sx); break;
        default: GGML_ABORT("fatal error");
    }
}

// the GDN layer's in-proj + z + merged alpha/beta as one launch (ggml_cuda_qpn_gdn3_ok), 5 to 8 tokens (to 16), the pair's split and mode
template <ggml_type t0, ggml_type t1>
static void ggml_cuda_qpn2ab_launch_tt(const dim3 grid, const dim3 block, cudaStream_t stream, const qpn_seg & gab, const qpn_seg & g0, const qpn_seg & g1,
        const int nbab, const int nb0, const int nsb, const int S, const half * xh, const half * xs, const float * xsc, const float * x,
        const int64_t sx, const int nt) {
    if constexpr (qpn_type_idx(t0) <= qpn_type_idx(t1)) {
        const bool full = (nt == 8 || nt == 16) && ggml_cuda_qpn_full_on();
        if (nt <= 8) {
            full ? qpn_mul2ab_kernel<t0, t1, 8, 2 + QPN_FULL><<<grid, block, 0, stream>>>(gab, g0, g1, nbab, nb0, nsb, S, xh, xs, xsc, x, sx, nt)
                 : qpn_mul2ab_kernel<t0, t1, 8, 2><<<grid, block, 0, stream>>>(gab, g0, g1, nbab, nb0, nsb, S, xh, xs, xsc, x, sx, nt);
        } else { // 9 to 16 tokens, two B tiles
            full ? qpn_mul2ab_kernel<t0, t1, 16, 2 + QPN_FULL><<<grid, block, 0, stream>>>(gab, g0, g1, nbab, nb0, nsb, S, xh, xs, xsc, x, sx, nt)
                 : qpn_mul2ab_kernel<t0, t1, 16, 2><<<grid, block, 0, stream>>>(gab, g0, g1, nbab, nb0, nsb, S, xh, xs, xsc, x, sx, nt);
        }
    } else {
        GGML_ABORT("fatal error"); // the caller orders the pair by type
    }
}

template <ggml_type t0>
void ggml_cuda_qpn2ab_launch_t(const ggml_type t1, const dim3 grid, const dim3 block, cudaStream_t stream, const qpn_seg & gab, const qpn_seg & g0,
        const qpn_seg & g1, const int nbab, const int nb0, const int nsb, const int S, const half * xh, const half * xs, const float * xsc,
        const float * x, const int64_t sx, const int nt) {
    switch (t1) {
        case GGML_TYPE_Q4_K:   ggml_cuda_qpn2ab_launch_tt<t0, GGML_TYPE_Q4_K  >(grid, block, stream, gab, g0, g1, nbab, nb0, nsb, S, xh, xs, xsc, x, sx, nt); break;
        case GGML_TYPE_Q5_K:   ggml_cuda_qpn2ab_launch_tt<t0, GGML_TYPE_Q5_K  >(grid, block, stream, gab, g0, g1, nbab, nb0, nsb, S, xh, xs, xsc, x, sx, nt); break;
        case GGML_TYPE_Q6_K:   ggml_cuda_qpn2ab_launch_tt<t0, GGML_TYPE_Q6_K  >(grid, block, stream, gab, g0, g1, nbab, nb0, nsb, S, xh, xs, xsc, x, sx, nt); break;
        case GGML_TYPE_IQ4_XS: ggml_cuda_qpn2ab_launch_tt<t0, GGML_TYPE_IQ4_XS>(grid, block, stream, gab, g0, g1, nbab, nb0, nsb, S, xh, xs, xsc, x, sx, nt); break;
        case GGML_TYPE_IQ4_NL: ggml_cuda_qpn2ab_launch_tt<t0, GGML_TYPE_IQ4_NL>(grid, block, stream, gab, g0, g1, nbab, nb0, nsb, S, xh, xs, xsc, x, sx, nt); break;
        default: GGML_ABORT("fatal error");
    }
}

// in-proj, z, beta and alpha under --split-mode tensor as one launch (ggml_cuda_qpn_gdn4_ok), 5 to 16 tokens, the pair's split and mode
template <ggml_type t0, ggml_type t1>
static void ggml_cuda_qpn4ab_launch_tt(const dim3 grid, const dim3 block, cudaStream_t stream, const qpn_seg & gb, const qpn_seg & ga, const int nrowb,
        const int nrowa, const qpn_seg & g0, const qpn_seg & g1, const int nbb, const int nba, const int nb0, const int nsb, const int S, const half * xh,
        const half * xs, const float * xsc, const float * x, const int64_t sx, const int nt) {
    if constexpr (qpn_type_idx(t0) <= qpn_type_idx(t1)) {
        const bool full = (nt == 8 || nt == 16) && ggml_cuda_qpn_full_on();
        if (nt <= 8) {
            full ? qpn_mul4ab_kernel<t0, t1, 8, 2 + QPN_FULL><<<grid, block, 0, stream>>>(gb, ga, nrowb, nrowa, g0, g1, nbb, nba, nb0, nsb, S, xh, xs, xsc, x, sx, nt)
                 : qpn_mul4ab_kernel<t0, t1, 8, 2><<<grid, block, 0, stream>>>(gb, ga, nrowb, nrowa, g0, g1, nbb, nba, nb0, nsb, S, xh, xs, xsc, x, sx, nt);
        } else { // 9 to 16 tokens, two B tiles
            full ? qpn_mul4ab_kernel<t0, t1, 16, 2 + QPN_FULL><<<grid, block, 0, stream>>>(gb, ga, nrowb, nrowa, g0, g1, nbb, nba, nb0, nsb, S, xh, xs, xsc, x, sx, nt)
                 : qpn_mul4ab_kernel<t0, t1, 16, 2><<<grid, block, 0, stream>>>(gb, ga, nrowb, nrowa, g0, g1, nbb, nba, nb0, nsb, S, xh, xs, xsc, x, sx, nt);
        }
    } else {
        GGML_ABORT("fatal error"); // the caller orders the pair by type
    }
}

template <ggml_type t0>
void ggml_cuda_qpn4ab_launch_t(const ggml_type t1, const dim3 grid, const dim3 block, cudaStream_t stream, const qpn_seg & gb, const qpn_seg & ga,
        const int nrowb, const int nrowa, const qpn_seg & g0, const qpn_seg & g1, const int nbb, const int nba, const int nb0, const int nsb, const int S,
        const half * xh, const half * xs, const float * xsc, const float * x, const int64_t sx, const int nt) {
    switch (t1) {
        case GGML_TYPE_Q4_K:   ggml_cuda_qpn4ab_launch_tt<t0, GGML_TYPE_Q4_K  >(grid, block, stream, gb, ga, nrowb, nrowa, g0, g1, nbb, nba, nb0, nsb, S, xh, xs, xsc, x, sx, nt); break;
        case GGML_TYPE_Q5_K:   ggml_cuda_qpn4ab_launch_tt<t0, GGML_TYPE_Q5_K  >(grid, block, stream, gb, ga, nrowb, nrowa, g0, g1, nbb, nba, nb0, nsb, S, xh, xs, xsc, x, sx, nt); break;
        case GGML_TYPE_Q6_K:   ggml_cuda_qpn4ab_launch_tt<t0, GGML_TYPE_Q6_K  >(grid, block, stream, gb, ga, nrowb, nrowa, g0, g1, nbb, nba, nb0, nsb, S, xh, xs, xsc, x, sx, nt); break;
        case GGML_TYPE_IQ4_XS: ggml_cuda_qpn4ab_launch_tt<t0, GGML_TYPE_IQ4_XS>(grid, block, stream, gb, ga, nrowb, nrowa, g0, g1, nbb, nba, nb0, nsb, S, xh, xs, xsc, x, sx, nt); break;
        case GGML_TYPE_IQ4_NL: ggml_cuda_qpn4ab_launch_tt<t0, GGML_TYPE_IQ4_NL>(grid, block, stream, gb, ga, nrowb, nrowa, g0, g1, nbb, nba, nb0, nsb, S, xh, xs, xsc, x, sx, nt); break;
        default: GGML_ABORT("fatal error");
    }
}


// ------------------------------------------------------------------------------------------------------------------
// the entry points, each defined in template-instances/ (generate_qpn_cu_files.py) so that the kernels compile in parallel

#define DECL_QPN_CASE(type)                                                                                                                           \
    template bool ggml_cuda_qpn_launch<type>(const int T, const bool stage, const int mode, const dim3 grid, const dim3 block, const size_t smem, cudaStream_t stream, const qpn_args & a)
#define DECL_QPN2_CASE(t0, t1)                                                                                                                        \
    template void ggml_cuda_qpn2_launch_tt<t0, t1>(const int T, const int stage, const dim3 grid, const dim3 block, const size_t smem, cudaStream_t stream, const qpn_seg & g0, const qpn_seg & g1, const int nb0, const int nsb, const int S, const half * xh, const half * xs, const float * xsc, const float * x, const int64_t sx)
#define DECL_QPN3_CASE(t0)                                                                                                                            \
    template void ggml_cuda_qpn3_launch_t<t0>(const ggml_type t1, const ggml_type t2, const int T, const dim3 grid, const dim3 block, cudaStream_t stream, const qpn_seg & g0, const qpn_seg & g1, const qpn_seg & g2, const int nb0, const int nb1, const int nb2, const int nsb, const int S, const int S12, const half * xh, const half * xs, const float * xsc, const float * x, const int64_t sx)
#define DECL_QPN2AB_CASE(t0)                                                                                                                          \
    template void ggml_cuda_qpn2ab_launch_t<t0>(const ggml_type t1, const dim3 grid, const dim3 block, cudaStream_t stream, const qpn_seg & gab, const qpn_seg & g0, const qpn_seg & g1, const int nbab, const int nb0, const int nsb, const int S, const half * xh, const half * xs, const float * xsc, const float * x, const int64_t sx, const int nt)
#define DECL_QPN4AB_CASE(t0)                                                                                                                          \
    template void ggml_cuda_qpn4ab_launch_t<t0>(const ggml_type t1, const dim3 grid, const dim3 block, cudaStream_t stream, const qpn_seg & gb, const qpn_seg & ga, const int nrowb, const int nrowa, const qpn_seg & g0, const qpn_seg & g1, const int nbb, const int nba, const int nb0, const int nsb, const int S, const half * xh, const half * xs, const float * xsc, const float * x, const int64_t sx, const int nt)

extern DECL_QPN_CASE(GGML_TYPE_Q4_K);
extern DECL_QPN_CASE(GGML_TYPE_Q5_K);
extern DECL_QPN_CASE(GGML_TYPE_Q6_K);
extern DECL_QPN_CASE(GGML_TYPE_IQ4_XS);
extern DECL_QPN_CASE(GGML_TYPE_IQ4_NL);
extern DECL_QPN_CASE(GGML_TYPE_Q3_K);
extern DECL_QPN_CASE(GGML_TYPE_Q8_0);

extern DECL_QPN2_CASE(GGML_TYPE_Q4_K, GGML_TYPE_Q4_K);
extern DECL_QPN2_CASE(GGML_TYPE_Q4_K, GGML_TYPE_Q5_K);
extern DECL_QPN2_CASE(GGML_TYPE_Q4_K, GGML_TYPE_Q6_K);
extern DECL_QPN2_CASE(GGML_TYPE_Q4_K, GGML_TYPE_IQ4_XS);
extern DECL_QPN2_CASE(GGML_TYPE_Q4_K, GGML_TYPE_IQ4_NL);
extern DECL_QPN2_CASE(GGML_TYPE_Q4_K, GGML_TYPE_Q3_K);
extern DECL_QPN2_CASE(GGML_TYPE_Q5_K, GGML_TYPE_Q4_K);
extern DECL_QPN2_CASE(GGML_TYPE_Q5_K, GGML_TYPE_Q5_K);
extern DECL_QPN2_CASE(GGML_TYPE_Q5_K, GGML_TYPE_Q6_K);
extern DECL_QPN2_CASE(GGML_TYPE_Q5_K, GGML_TYPE_IQ4_XS);
extern DECL_QPN2_CASE(GGML_TYPE_Q5_K, GGML_TYPE_IQ4_NL);
extern DECL_QPN2_CASE(GGML_TYPE_Q5_K, GGML_TYPE_Q3_K);
extern DECL_QPN2_CASE(GGML_TYPE_Q6_K, GGML_TYPE_Q4_K);
extern DECL_QPN2_CASE(GGML_TYPE_Q6_K, GGML_TYPE_Q5_K);
extern DECL_QPN2_CASE(GGML_TYPE_Q6_K, GGML_TYPE_Q6_K);
extern DECL_QPN2_CASE(GGML_TYPE_Q6_K, GGML_TYPE_IQ4_XS);
extern DECL_QPN2_CASE(GGML_TYPE_Q6_K, GGML_TYPE_IQ4_NL);
extern DECL_QPN2_CASE(GGML_TYPE_Q6_K, GGML_TYPE_Q3_K);
extern DECL_QPN2_CASE(GGML_TYPE_IQ4_XS, GGML_TYPE_Q4_K);
extern DECL_QPN2_CASE(GGML_TYPE_IQ4_XS, GGML_TYPE_Q5_K);
extern DECL_QPN2_CASE(GGML_TYPE_IQ4_XS, GGML_TYPE_Q6_K);
extern DECL_QPN2_CASE(GGML_TYPE_IQ4_XS, GGML_TYPE_IQ4_XS);
extern DECL_QPN2_CASE(GGML_TYPE_IQ4_XS, GGML_TYPE_IQ4_NL);
extern DECL_QPN2_CASE(GGML_TYPE_IQ4_XS, GGML_TYPE_Q3_K);
extern DECL_QPN2_CASE(GGML_TYPE_IQ4_NL, GGML_TYPE_Q4_K);
extern DECL_QPN2_CASE(GGML_TYPE_IQ4_NL, GGML_TYPE_Q5_K);
extern DECL_QPN2_CASE(GGML_TYPE_IQ4_NL, GGML_TYPE_Q6_K);
extern DECL_QPN2_CASE(GGML_TYPE_IQ4_NL, GGML_TYPE_IQ4_XS);
extern DECL_QPN2_CASE(GGML_TYPE_IQ4_NL, GGML_TYPE_IQ4_NL);
extern DECL_QPN2_CASE(GGML_TYPE_IQ4_NL, GGML_TYPE_Q3_K);
extern DECL_QPN2_CASE(GGML_TYPE_Q3_K, GGML_TYPE_Q4_K);
extern DECL_QPN2_CASE(GGML_TYPE_Q3_K, GGML_TYPE_Q5_K);
extern DECL_QPN2_CASE(GGML_TYPE_Q3_K, GGML_TYPE_Q6_K);
extern DECL_QPN2_CASE(GGML_TYPE_Q3_K, GGML_TYPE_IQ4_XS);
extern DECL_QPN2_CASE(GGML_TYPE_Q3_K, GGML_TYPE_IQ4_NL);
extern DECL_QPN2_CASE(GGML_TYPE_Q3_K, GGML_TYPE_Q3_K);

extern DECL_QPN3_CASE(GGML_TYPE_Q4_K);
extern DECL_QPN3_CASE(GGML_TYPE_Q5_K);
extern DECL_QPN3_CASE(GGML_TYPE_IQ4_XS);

extern DECL_QPN2AB_CASE(GGML_TYPE_Q4_K);
extern DECL_QPN2AB_CASE(GGML_TYPE_Q5_K);
extern DECL_QPN2AB_CASE(GGML_TYPE_Q6_K);
extern DECL_QPN2AB_CASE(GGML_TYPE_IQ4_XS);
extern DECL_QPN2AB_CASE(GGML_TYPE_IQ4_NL);
extern DECL_QPN4AB_CASE(GGML_TYPE_Q4_K);
extern DECL_QPN4AB_CASE(GGML_TYPE_Q5_K);
extern DECL_QPN4AB_CASE(GGML_TYPE_Q6_K);
extern DECL_QPN4AB_CASE(GGML_TYPE_IQ4_XS);
extern DECL_QPN4AB_CASE(GGML_TYPE_IQ4_NL);
