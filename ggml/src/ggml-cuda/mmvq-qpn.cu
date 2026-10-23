// Volta tensor-core products for the dense K-quants (Q4_K, Q5_K, Q6_K) at 1 to 8 tokens, on weights repacked into
// mma fragment order once at load (toggle LLAMA_MMVQ_QPN).
//
// The design follows NInfer's QPN kernels (github.com/... ninfer-v100, Apache-2.0: ops/linear/nvfp4/nvfp4_volta_qpn_gemm.cuh,
// which credits the "v100-skinny" project): tokens are the 8-row M side of mma.sync.m8n8k4 (zero-padded), each quadpair
// owns 8 output rows, a warp owns 32, the inner dimension is split across the warps of a block (split-K) with a
// shared-memory reduction only at the end, and the weights are read as one contiguous 16-byte chunk per lane
// (512 bytes per warp) with no shared-memory staging, because they were put in fragment order at load. No NInfer code
// is copied: its kernels decode NVFP4/FP8, these decode K-quant codes.
//
// Layout. A tensor of N rows and K = 256*nsb columns becomes N/32 row tiles; tile t is a contiguous run of nsb
// tile-superblocks (32 rows x 256 columns, 32*sizeof(block) bytes, so the size is unchanged and the repack is in place).
// A tile-superblock is a list of 512-byte chunks; in each chunk lane l's 16 bytes belong to row l:
//   Q4_K (9 chunks)  : [d, dmin, scales[12]] verbatim, then one chunk per 32-column sub-block j: 32 4-bit codes
//   Q5_K (11 chunks) : the same header, 2 chunks of high-bit words (word j: sub-block j's 32 fifth bits), 8 code chunks
//   Q6_K (13 chunks + 64 bytes): 8 code chunks (low 4 bits), 4 chunks of high-bit words (2 bits per code, one word per
//                      16 codes), the 16 int8 scales verbatim, then the 32 rows' d, stored so rows r and r + 2 are adjacent
// A code word holds 8 codes of consecutive columns k0..k0+7: slot s at nibble (s even ? s/2 : 4 + s/2). Masking
// (w << 4), w, (w >> 4), (w >> 8) with 0x00F000F0 and or-ing 0x6400 puts slots (2x, 2x+1) into a half2 as exactly
// 1024 + 16*q; the high bits for pair x sit in the high-bit word so that one rotate and one mask put them at bits 8
// (and 9) of each half. One HFMA2 then gives sc*q exactly for Q4_K/Q5_K (sc/16 and -64*sc) and sc*(q - 32) with a
// single rounding for Q6_K (sc/16 and -96*sc). The superblock scale d, Q4_K/Q5_K's mins (dmin * sum_j m_j * sum_j x,
// from per-32 activation sums through one more mma) and the activations' power-of-two range scale are applied to the
// fp32 accumulators once per superblock.
//
// IQ4_XS and IQ4_NL hold 4-bit indices into the 16-entry int8 codebook kvalues_iq4nl. Their blocks (136
// and 18 bytes) are not 16-byte aligned, so they too are only read repacked; a tile-superblock keeps its size:
//   IQ4_XS (8 chunks + 256 bytes): one chunk per sub-block j, then each row's 8-byte GGUF header (d, scales_h,
//                      scales_l) verbatim, in row order
//   IQ4_NL (9 chunks): the same 8 code chunks, then one chunk of each row's 8 fp16 block scales d_j
// Here a code word holds the codes of columns k0..k0+7 in nibbles 0..7. PRMT looks up 4 codes at a time: its low 3 bits
// pick one of 8 bytes of kv + 128 from each codebook half, its bit 3 which half (a third PRMT), and a fourth puts two
// bytes under 0x64, exactly 1024 + 128 + kv. One HFMA2 then gives kv*(ls - 32) with a single rounding for IQ4_XS
// (ls - 32 and -1152*(ls - 32), both exact; d in fp32 per superblock), and one HSUB2 and one HMUL2 give kv*d_j for
// IQ4_NL, rounded once exactly as ggml's fp16 dequantization rounds it.
//
// Q3_K and Q8_0, also in place and the same size:
//   Q3_K (6 chunks + 448 bytes): 4 chunks of the codes' low 2 bits, 2 chunks of their high bits (hmask), then each row's
//                      12 scale bytes verbatim (row r at 12r, so 4-byte aligned), then the 32 rows' d as Q6_K stores them
//                      (rows r and r + 2 adjacent). Low-bit word w (16 columns 16w..16w+15, one 16-column sub-block)
//                      holds pair e = 4f + i (fragment f of the word, columns 2i, 2i+1 of it) at bits 2e + p + {0,1} and
//                      16 + 2e + p + {0,1} (mod 32), p = w & 1; high-bit word v (32 columns, low-bit words 2v, 2v+1) holds
//                      that pair's high bits at 2e + p + 2 and 16 + 2e + p + 2 (mod 32). One LOP3 selects the pair's 4 low
//                      bits from w and its 2 high bits from v, one rotate puts the 3-bit codes c = q2 | h << 2 at bits 2-4
//                      of each half, and under 0x64 they are 1024 + 4c. One HFMA2 with S = (s - 32)/4 and Z = -260*(s - 32)
//                      gives (c - 4)*(s - 32) exactly (an integer of at most 128 in magnitude), and d is applied in fp32
//                      per superblock, so a weight is d*(s - 32)*(c - 4) with no rounding, as ggml's dequantize_row_q3_K
//   Q8_0 (17 chunks)  : 16 code chunks (chunk c: columns 16c..16c+15 of the row, each code stored as q + 128), then one chunk
//                      of each row's 8 fp16 block scales d_j (IQ4_NL's). PRMT puts two codes under 0x64, exactly
//                      1024 + 128 + q; one HSUB2 gives q and one HMUL2 by d_j rounds q*d_j once, as ggml's fp16
//                      dequantization (dequantize_block_q8_0_f16) does
//
// Activations are fp16, prepared once per input (qpn-prep.cuh): each 256-column slice of a token is scaled by a power
// of two so its max |x| is below 2^15 (exact; undone in fp32). A slice holding a non-finite value marks its token, and
// that token's column is then computed in fp32 from the dequantized weights (qpn_fallback), as the generic path would.
//
// Nothing else may read a repacked tensor: it carries GGML_TENSOR_FLAG_BACKEND_LAYOUT, ggml_cuda_mul_mat sends it here
// (or, above GGML_CUDA_QPN_MAX_TOKENS tokens, through ggml_cuda_qpn_to_fp16 to cuBLAS, as a quantized weight already
// goes on this GPU), fusion never takes it, and ggml_cuda_compute_forward aborts if any other op reads it.

#include "mmvq-qpn-impl.cuh"

#include <mutex>
#include <string>

// LLAMA_QPN_BENCH=N: the first direct (uncaptured) launch of each distinct key is repeated N times back to back and timed
static int qpn_bench_reps() {
    static const int n = [] { const char * e = getenv("LLAMA_QPN_BENCH"); return e == nullptr ? 0 : atoi(e); }();
    return n;
}

// LLAMA_QPN_LOG=1: each distinct launch (kind, tokens, grid, block, mode, and each weight's type, rows x columns, bytes) once on
// stderr, to match a profile's kernels to the products they read. Returns the launch's key (empty unless the log or the bench is on)
static std::string qpn_log_launch(const char * kind, std::initializer_list<const ggml_tensor *> w, const int T, const dim3 grid, const dim3 block,
        const int mode, const int stage) {
    static const bool on = [] { const char * e = getenv("LLAMA_QPN_LOG"); return e != nullptr && atoi(e) != 0; }();
    if (!on && qpn_bench_reps() <= 0) {
        return {};
    }
    char buf[512];
    int  n = snprintf(buf, sizeof(buf), "%s T=%d grid=%ux%u block=%ux%u mode=%d stage=%d", kind, T, grid.x, grid.y, block.x, block.y, mode, stage);
    for (const ggml_tensor * t : w) {
        n += snprintf(buf + n, sizeof(buf) - n, " %s:%lldx%lld:%zu", ggml_type_name(t->type), (long long) t->ne[1], (long long) t->ne[0], ggml_nbytes(t));
    }
    static std::mutex                      mu;
    static std::unordered_set<std::string> seen;
    std::lock_guard<std::mutex> lock(mu);
    if (on && seen.insert(buf).second) {
        fprintf(stderr, "qpn-launch %s\n", buf);
    }
    return buf;
}

// LLAMA_QPN_BENCH: times launch() repeated back to back, once per key, outside a capture (the weights are read cold: each
// routed tensor but the small k/v is larger than the L2)
template <typename F>
static void qpn_bench(const std::string & key, const size_t bytes, cudaStream_t stream, F && launch) {
    const int reps = qpn_bench_reps();
    if (reps <= 0 || key.empty()) {
        return;
    }
    cudaStreamCaptureStatus cs;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &cs));
    if (cs != cudaStreamCaptureStatusNone) {
        return;
    }
    static std::mutex                      mu;
    static std::unordered_set<std::string> seen;
    {
        std::lock_guard<std::mutex> lock(mu);
        if (!seen.insert(key).second) {
            return;
        }
    }
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    launch(); // warm the launch path
    CUDA_CHECK(cudaEventRecord(e0, stream));
    for (int i = 0; i < reps; ++i) {
        launch();
    }
    CUDA_CHECK(cudaEventRecord(e1, stream));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    const double us = 1e3*ms/reps;
    fprintf(stderr, "qpn-bench %s : %.1f us %.0f GB/s\n", key.c_str(), us, bytes/us/1e3);
    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));
}

// One warp per (256-column slice, token); the arithmetic and the layout are qpn-prep.cuh's.
static __global__ void qpn_prep_kernel(const float * __restrict__ x, const int64_t stride_col_x, const int nsb, const int T,
        half * __restrict__ xh, half * __restrict__ xs, float * __restrict__ xsc) {
    const int lane = threadIdx.x % WARP_SIZE;
    const int g    = blockIdx.x*(blockDim.x/WARP_SIZE) + threadIdx.x/WARP_SIZE;
    if (g >= nsb*T) {
        return;
    }
    const int sb = g / T, t = g % T;
    const float4 * xp = (const float4 *) (x + t*stride_col_x + (int64_t) sb*QK_K + lane*8);
    const float4 a = xp[0], b = xp[1];
    const float v[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
    qpn_prep_slice_warp(v, sb, t, T, xh, xs, xsc);
}

// ------------------------------------------------------------------------------------------------------------------
// routing and launch

static int ggml_cuda_qpn_mask() {
    static const int mask = [] {
        const char * e = getenv("LLAMA_MMVQ_QPN");
        return e == nullptr ? GGML_CUDA_QPN_ALL : atoi(e);
    }();
    return mask;
}

// LLAMA_MMVQ_QPN_FORCE: 1 = every enabled type, any other nonzero value = a bitmask of the types to force (as
// LLAMA_MMVQ_QPN's), e.g. 24 = IQ4_XS and IQ4_NL, the others keeping their routes
static bool ggml_cuda_qpn_force(const int bit) {
    static const int force = [] {
        const char * e = getenv("LLAMA_MMVQ_QPN_FORCE");
        const int v = e == nullptr ? 0 : atoi(e);
        return v == 1 ? GGML_CUDA_QPN_ALL : v;
    }();
    return (force & bit) != 0;
}

static int ggml_cuda_qpn_type_bit(const ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_K: return GGML_CUDA_QPN_Q4_K;
        case GGML_TYPE_Q5_K: return GGML_CUDA_QPN_Q5_K;
        case GGML_TYPE_Q6_K:   return GGML_CUDA_QPN_Q6_K;
        case GGML_TYPE_IQ4_XS: return GGML_CUDA_QPN_IQ4_XS;
        case GGML_TYPE_IQ4_NL: return GGML_CUDA_QPN_IQ4_NL;
        case GGML_TYPE_Q3_K:   return GGML_CUDA_QPN_Q3_K;
        case GGML_TYPE_Q8_0:   return GGML_CUDA_QPN_Q8_0;
        default:               return 0;
    }
}

// How a product is split (see qpn_mul_kernel): S warps per row tile, R row tiles per block, P blocks per row tile
// along K, stage: the block's activations in shared memory. Per shape (N rows, K), type-independent: the best of
// a sweep at 3 and 4 tokens, and the nearest swept shape for the two it did not sweep.
// Any other shape takes { 4, 2, 1, 1 }, or { 1, 4, 1, 0 } at 65536 rows and more (the output head).
struct qpn_cfg       { int S, R, P, stage, mode = 0; }; // mode: 0, or 2/3 = qpn_stream_loop at that many blocks per SM
// the (N, K) that keys a repacked weight's launch configuration (the split tables below), its pairs and its triple: its own,
// or for one device's slice of a weight under --split-mode tensor the whole weight's, recorded when the per-slice repack ran. Every
// table here is keyed by the model's shapes, so a slice (8704 x 5120 of the 17408-row ffn gate, 5120 x 8704 of the down projection, ...)
// otherwise missed them all: it ran the default split ({4, 2}, too few blocks for the card) and never formed a pair or a triple. Keyed
// by the whole weight, a slice of rows runs the one-card split (the same S, R, P and mode, so each output row sums in the one-card
// order), a slice along K runs the one-card split over its half of K, and the planner groups slices as it groups the whole weights.
// Written at load only (empty on one card), read by each device's compute thread afterwards.
static std::unordered_map<const void *, std::pair<int64_t, int64_t>> qpn_slice_keys;

// the slices a draft model's route repacked (kind 1 of ggml_backend_cuda_qpn_repack_slice), written at load only
static std::unordered_set<const void *> qpn_draft_slices;

void ggml_cuda_qpn_mark_draft_slice(const ggml_tensor * slice) {
    qpn_draft_slices.insert(slice->data);
}

void ggml_cuda_qpn_set_slice_key(const ggml_tensor * slice, const int64_t n, const int64_t k) {
    qpn_slice_keys[slice->data] = { n, k };
}

static inline int64_t qpn_key_n(const ggml_tensor * t) {
    if (!qpn_slice_keys.empty()) {
        const auto it = qpn_slice_keys.find(t->data);
        if (it != qpn_slice_keys.end()) {
            return it->second.first;
        }
    }
    return t->ne[1];
}

// a device's slice of a weight under --split-mode tensor (its key recorded)
static inline bool qpn_is_slice(const ggml_tensor * t) {
    return !qpn_slice_keys.empty() && qpn_slice_keys.count(t->data) != 0;
}
static inline bool qpn_is_draft_slice(const ggml_tensor * t) {
    return !qpn_draft_slices.empty() && qpn_draft_slices.count(t->data) != 0;
}

static inline int64_t qpn_key_k(const ggml_tensor * t) {
    if (!qpn_slice_keys.empty()) {
        const auto it = qpn_slice_keys.find(t->data);
        if (it != qpn_slice_keys.end()) {
            return it->second.second;
        }
    }
    return t->ne[0];
}

struct qpn_shape_cfg { int64_t n, k; qpn_cfg cfg; };
static const qpn_shape_cfg qpn_cfgs[] = {
    { 10240,  5120, { 2, 2, 1, 0 } }, // GDN qkv
    { 12288,  5120, { 2, 2, 1, 0 } }, // attention q + gate (not swept; as 10240)
    { 17408,  5120, { 2, 4, 1, 0 } }, // ffn gate, ffn up
    {  5120, 17408, { 4, 1, 1, 0 } }, // ffn down
    {  6144,  5120, { 4, 1, 1, 0 } }, // GDN z (attn_gate)
    {  5120,  6144, { 4, 1, 1, 0 } }, // GDN out (ssm_out), attention out
    {  1024,  5120, { 4, 1, 1, 0 } }, // attention k, v (not swept; as the other small grids)
};

// Routed shapes, repacked at load by default: each (type, N, K) whose products, prep included, took less time per
// round in the graph on the repacked weights than on dp4a by more than the drift of that capture (the
// Qwen3.8-27B, n-max 3, 4-token verifies, node-level capture of both arms; SM clock 1380 MHz throughout, untouched
// kernels within 2%). Per round, dp4a -> QPN. Not routed: the 1024-row k/v (+64% to +125%) and every other Q6_K
// shape (+2% to +18%; four of them are routed below). Anything else keeps the GGUF layout; LLAMA_MMVQ_QPN_FORCE=1
// repacks every eligible shape.
struct qpn_route { ggml_type type; int64_t n, k; };
static const qpn_route qpn_routes[] = {
    { GGML_TYPE_Q5_K,  17408,  5120 }, // ffn gate, up      6.32 -> 4.50 ms
    { GGML_TYPE_Q5_K,   5120, 17408 }, // ffn down          3.91 -> 3.33
    { GGML_TYPE_Q5_K,   5120,  6144 }, // GDN out, attn out 1.94 -> 1.83
    { GGML_TYPE_Q5_K,   6144,  5120 }, // GDN z             1.79 -> 1.67
    { GGML_TYPE_Q5_K,  10240,  5120 }, // GDN qkv           1.67 -> 1.35
    { GGML_TYPE_Q5_K,  12288,  5120 }, // attn q + gate     0.75 -> 0.56
    { GGML_TYPE_Q4_K,  17408,  5120 }, // ffn gate, up      2.29 -> 1.44
    { GGML_TYPE_Q4_K,  10240,  5120 }, // GDN qkv           1.76 -> 1.34
    { GGML_TYPE_Q4_K,   6144,  5120 }, // GDN z             0.71 -> 0.59
    { GGML_TYPE_Q4_K,   5120, 17408 }, // ffn down          0.54 -> 0.45
    { GGML_TYPE_Q4_K,  12288,  5120 }, // attn q + gate     0.54 -> 0.37
    { GGML_TYPE_Q4_K,   5120,  6144 }, // GDN out           0.05 -> 0.04
    { GGML_TYPE_Q6_K, 248320,  5120 }, // output head       1.86 -> 1.72
    // the same rule (n-max 3, rejection sampling, the 98,304-token draft vocabulary; LLAMA_MMVQ_QPN=7 against
    // LLAMA_MMVQ_QPN_FORCE=24, SM clock 1380 MHz in both, untouched items within 3.5%). Not routed: IQ4_XS ffn down (+7.7%)
    // and attn out (+9.1%), and every IQ4_NL shape (gate/up -2.8%, inside the drift; down +9.0%; GDN qkv +15.2%)
    { GGML_TYPE_IQ4_XS, 17408, 5120 }, // ffn gate, up      4.43 -> 3.44
    { GGML_TYPE_IQ4_XS, 12288, 5120 }, // attn q + gate     0.149 -> 0.117
    { GGML_TYPE_IQ4_XS,  6144, 5120 }, // GDN z             0.149 -> 0.138
    { GGML_TYPE_IQ4_XS, 10240, 5120 }, // GDN qkv           0.131 -> 0.123
};

// the same rule after the register ring and the shared prep (n-max 3, rejection sampling, the 98,304-token
// draft vocabulary; defaults against LLAMA_MMVQ_QPN_FORCE=1, SM clock recorded). LLAMA_QPN_ROUTES_0086=0 leaves them
// out. Drift 3.25% (k_bin_bcast); per round, dp4a -> QPN with prep. Not routed: Q6_K 5120x6144 (-2.8%,
// inside the drift), IQ4_XS ffn down (-0.4%) and attn out (+7.7%), IQ4_NL ffn down (+8.9%) and GDN qkv (+15.9%), the 1024-row k/v
// (+42% to +87%).
static const qpn_route qpn_routes_0086[] = {
    { GGML_TYPE_Q6_K,  17408,  5120 }, // ffn gate, up      0.834 -> 0.610 ms
    { GGML_TYPE_Q6_K,   5120, 17408 }, // ffn down          0.660 -> 0.561
    { GGML_TYPE_Q6_K,  10240,  5120 }, // GDN qkv           0.091 -> 0.078
    { GGML_TYPE_Q6_K,   6144,  5120 }, // GDN z             0.062 -> 0.059
    { GGML_TYPE_IQ4_NL, 17408, 5120 }, // ffn gate, up      0.265 -> 0.255
};

// the same rule on the finished kernel (n-max 3, rejection sampling, the 98,304-token draft vocabulary; defaults
// against LLAMA_MMVQ_QPN_FORCE=1, node-level, card 0, drift 1.67% (flash_attn_tile); per round, dp4a -> QPN with prep).
// LLAMA_QPN_ROUTES_0091=0 leaves them out. Not routed: IQ4_XS ffn down (+0.5%) and the 1024-row k/v (+62% to +92%).
static const qpn_route qpn_routes_0091[] = {
    { GGML_TYPE_Q6_K,   5120,  6144 }, // GDN out, attn out 1.236 -> 0.982 ms
    { GGML_TYPE_IQ4_NL, 5120, 17408 }, // ffn down          0.179 -> 0.170
    { GGML_TYPE_IQ4_XS, 5120,  6144 }, // attn out          0.042 -> 0.041
    { GGML_TYPE_IQ4_NL, 10240, 5120 }, // GDN qkv           0.056 -> 0.055 (then grouped with its z)
};

// the same rule over the widths in use, for the new types' keys: node-level captures at
// n-max 3 and 7 (4- and 8-token verifies) on poetry, CSV and the code prompt (17.7K deep), each width's pair in one session;
// arm O = LLAMA_QPN_DRAFT=0 LLAMA_QPN_ROUTES_0096=0, arm F = LLAMA_MMVQ_QPN_FORCE=96 LLAMA_QPN_DRAFT_FORCE=1;
// card 0; drift 4.0-5.1% at width 4, 5.7-6.4% at width 8 (flash_attn_tile, or a 0.13 ms IQ4_XS item). Per round,
// dp4a -> QPN, width 4 / width 8, the same on all three prompts to 1%.
// LLAMA_QPN_ROUTES_0096=0 leaves them out. Not routed: Q8_0 1024x5120 (attn v, and one k: +20% / +78%; 32 row tiles, the
// three-product grouping is its route).
static const qpn_route qpn_routes_0096[] = {
    { GGML_TYPE_Q3_K, 17408,  5120 }, // ffn gate, up (3)  0.506 -> 0.179 / 0.944 -> 0.281 ms
    { GGML_TYPE_Q8_0,  5120,  6144 }, // GDN out (1)       0.050 -> 0.046 / 0.070 -> 0.053
};

static int ggml_cuda_qpn_wide_mask(); // LLAMA_QPN_WIDE, below

// LLAMA_QPN_DRAFT_KV (default OFF: no gain measured, -1 us a step; 1 = on): the draft's attention k and v (Q6_K, 1,024 x 5,120) are repacked (qpn_routes_draft) and, at one
// token, run with q as one launch (q + k pair, and the triple with v)
static bool ggml_cuda_qpn_draft_kv_on() {
    static const bool on = [] { const char * e = getenv("LLAMA_QPN_DRAFT_KV"); return e != nullptr && atoi(e) != 0; }();
    return on;
}

// LLAMA_QPN_GDN_GROUP (default on; 0 = off): qwen35.cpp puts the GDN layer's in-proj and z (and ab) side by side in the graph, and the
// sibling planner runs in-proj + z as one qpn_mul2 launch at 3 to 8 tokens (outputs bit-identical to two launches).
// LLAMA_QPN_GDN_AB (default ON, 0 = off: not bit-identical, KL 0.000021; -0.6 ms per verify graph at 16K and 64K): the merged alpha/beta weight ab (Q8_0, 96 x 5,120) is
// repacked and runs in the same launch at 5 to 8 tokens (ggml_cuda_mul_mat_qpn2ab), on the tensor cores instead of dp4a
static bool ggml_cuda_qpn_gdn_group_on() {
    static const bool on = [] { const char * e = getenv("LLAMA_QPN_GDN_GROUP"); return e == nullptr || atoi(e) != 0; }();
    return on;
}
static bool ggml_cuda_qpn_gdn_ab_on() {
    static const bool on = [] { const char * e = getenv("LLAMA_QPN_GDN_AB"); return e == nullptr || atoi(e) != 0; }() && ggml_cuda_qpn_gdn_group_on();
    return on;
}

// the same rule over the widths in use: a key routes if it is no slower at width 4 within the capture's drift and faster
// at width 8, on poetry, CSV and code (n-max 3 and 7, T=0, generated tokens 100-220, node-level, card 0; defaults against
// LLAMA_MMVQ_QPN_FORCE=1 in one session, drift at most 4.75% (flash_attn_tile); per round, dp4a -> QPN with prep).
// LLAMA_QPN_ROUTES_0095=0 leaves them out. Every earlier route still wins against dp4a at both widths (LLAMA_MMVQ_QPN=0).
static const qpn_route qpn_routes_0095[] = {
    { GGML_TYPE_IQ4_XS, 5120, 17408 }, // ffn down          1.342 -> 1.300 ms at width 4, 2.667 -> 1.478 at width 8 (poetry; CSV, code within 0.3%)
    // the 1024-row attention k, v: only in one launch with their q + gate (ggml_cuda_mul_mat_qpn3, or the pair when v is Q8_0), so
    // only with LLAMA_QPN_GROUP and LLAMA_QPN_WIDE bit 8 on. The attention input's products together (q + gate, k, v, prep):
    // 1.418 -> 1.280 ms at width 4, 1.749 -> 1.465 at width 8 (poetry; CSV, code within 0.4%)
    { GGML_TYPE_Q6_K,   1024,  5120 },
    { GGML_TYPE_Q5_K,   1024,  5120 },
    { GGML_TYPE_Q4_K,   1024,  5120 },
};

static const qpn_route * ggml_cuda_qpn_find_route(const ggml_type type, const int64_t n, const int64_t k) {
    for (const qpn_route & r : qpn_routes) {
        if (r.type == type && r.n == n && r.k == k) {
            return &r;
        }
    }
    static const bool routes_0086 = [] { const char * e = getenv("LLAMA_QPN_ROUTES_0086"); return e == nullptr || atoi(e) != 0; }();
    if (routes_0086) {
        for (const qpn_route & r : qpn_routes_0086) {
            if (r.type == type && r.n == n && r.k == k) {
                return &r;
            }
        }
    }
    static const bool routes_0091 = [] { const char * e = getenv("LLAMA_QPN_ROUTES_0091"); return e == nullptr || atoi(e) != 0; }();
    if (routes_0091) {
        for (const qpn_route & r : qpn_routes_0091) {
            if (r.type == type && r.n == n && r.k == k) {
                return &r;
            }
        }
    }
    static const bool routes_0096 = [] { const char * e = getenv("LLAMA_QPN_ROUTES_0096"); return e == nullptr || atoi(e) != 0; }();
    if (routes_0096) {
        for (const qpn_route & r : qpn_routes_0096) {
            if (r.type == type && r.n == n && r.k == k) {
                return &r;
            }
        }
    }
    static const bool routes_0095 = [] { const char * e = getenv("LLAMA_QPN_ROUTES_0095"); return e == nullptr || atoi(e) != 0; }();
    static const bool grouped3    = [] { const char * e = getenv("LLAMA_QPN_GROUP"); return e == nullptr || atoi(e) != 0; }() &&
                                    (ggml_cuda_qpn_wide_mask() & 8);
    if (routes_0095) {
        for (const qpn_route & r : qpn_routes_0095) {
            if (r.type == type && r.n == n && r.k == k && (n != 1024 || grouped3)) {
                return &r;
            }
        }
    }
    // the merged GDN alpha/beta weight, whose only reader is the 8-row verify's grouped launch (5 to 8 tokens) or, alone, the
    // QPN kernel at any other width (build_layer_attn_linear reads it through ab at every token count once it is repacked)
    if (type == GGML_TYPE_Q8_0 && n == 96 && k == 5120 && ggml_cuda_qpn_gdn_ab_on()) {
        static const qpn_route r_ab = { GGML_TYPE_Q8_0, 96, 5120 };
        return &r_ab;
    }
    return nullptr;
}

// Per type, at 1 to 4 tokens. LLAMA_QPN_STREAM: bitmask, default 15.
// 1 = ffn gate/up (Q4_K/Q5_K), streamed with
// staged activations at the same S (outputs bit-identical); 2 = GDN qkv (Q4_K/Q5_K) at S = 4 instead of 2 (a different split-K
// summation order); 4 = qpn_stream_loop, 80 registers, for every
// other key at 1 to 4 tokens, at the key's split (outputs bit-identical); 8 = the splits of qpn_cfgs_0091 (new
// summation orders), which then replace bits 1 and 2. 0 = the original whole-superblock kernel and splits; 5 = every
// type on the streaming loop at those splits, bit-identical to it. Microbenchmark on the
// real tensors, card 1, SM 1380 MHz, T = 4, us: Q5_K gate/up 77.4 -> 73.1, Q4_K 63.8 -> 59.0; GDN qkv
// Q5_K 52.7 -> 48.1, Q4_K 47.0 -> 39.2.
struct qpn_type_cfg { ggml_type type; int64_t n, k; int bit; qpn_cfg cfg; };
static const qpn_type_cfg qpn_type_cfgs[] = {
    { GGML_TYPE_Q5_K, 17408, 5120, 1, { 2, 4, 1, 1, 3 } },
    { GGML_TYPE_Q4_K, 17408, 5120, 1, { 2, 4, 1, 1, 2 } },
    { GGML_TYPE_Q5_K, 10240, 5120, 2, { 4, 1, 1, 0, 3 } },
    { GGML_TYPE_Q4_K, 10240, 5120, 2, { 4, 1, 1, 0, 3 } },
};

static int ggml_cuda_qpn_stream_mask() {
    static const int mask = [] { const char * e = getenv("LLAMA_QPN_STREAM"); return e == nullptr ? 15 : atoi(e); }();
    return mask;
}

static const qpn_cfg * ggml_cuda_qpn_find_type_cfg(const ggml_type type, const int64_t n, const int64_t k) {
    const int mask = ggml_cuda_qpn_stream_mask();
    for (const qpn_type_cfg & c : qpn_type_cfgs) {
        if (c.type == type && c.n == n && c.k == k && (mask & c.bit)) {
            return &c.cfg;
        }
    }
    return nullptr;
}

// LLAMA_QPN_STREAM bit 4 (default on): at 1 to 4 tokens, every type on qpn_stream_loop (80 registers, 24 warps per SM)
// with the split that gives it the warps: microbenchmark on the real tensors at 4 tokens, card 1, SM clock recorded,
// us against the whole-superblock kernel. type GGML_TYPE_COUNT = any type.
struct qpn_cfg_0091 { ggml_type type; int64_t n, k; qpn_cfg cfg; };
static const qpn_cfg_0091 qpn_cfgs_0091[] = {
    // the Q3_K and Q8_0 types, the best of a sweep at 1 and 4 tokens (card 0, SM clock recorded), us against dp4a:
    { GGML_TYPE_Q3_K,  98304,  5120, { 2, 4, 1, 1, 3 } }, // the draft's head subset: 1 token 564.1 -> 329.4, 4 tokens 856.0 -> 353.0
    { GGML_TYPE_Q8_0,   1024,  5120, { 8, 1, 4, 0, 3 } }, // attn v, k: 1 token 7.4 -> 12.6, 4 tokens 10.9 -> 13.2 (32 row tiles)
    { GGML_TYPE_Q8_0,   5120,  6144, { 8, 1, 1, 0, 3 } }, // GDN out: 1 token 38.8 -> 40.7, 4 tokens 44.1 -> 38.4
    { GGML_TYPE_COUNT, 17408,  5120, { 2, 4, 1, 1, 3 } }, // ffn gate/up, staged (it still wins by 1-4%: Q5_K 73.8 against 73.3 at S = 3, IQ4_NL 63.7 against 66.1)
    { GGML_TYPE_COUNT, 10240,  5120, { 4, 1, 1, 0, 3 } }, // GDN qkv: Q6_K 62.4 -> 52.7, IQ4_XS 47.0 -> 40.5, Q5_K 54.8 -> 47.5
    { GGML_TYPE_COUNT, 12288,  5120, { 4, 2, 1, 0, 3 } }, // attn q + gate: Q4_K 59.8 -> 44.2, Q5_K 59.3 -> 54.7, IQ4_XS 55.5 -> 47.3
    { GGML_TYPE_COUNT,  5120, 17408, { 4, 1, 2, 0, 3 } }, // ffn down: Q5_K 88.4 -> 82.2, Q6_K 100.5 -> 94.0, IQ4_XS 76.8 -> 70.8, IQ4_NL 81.3 -> 71.8
    { GGML_TYPE_COUNT,  6144,  5120, { 8, 1, 1, 0, 3 } }, // GDN z: Q5_K 34.9 -> 32.1, Q4_K 28.5 -> 25.9, Q6_K 38.3 -> 34.6
    { GGML_TYPE_Q6_K,   5120,  6144, { 8, 1, 1, 0, 3 } }, // GDN out, attn out: Q6_K 39.8 -> 35.3
    { GGML_TYPE_COUNT,  5120,  6144, { 4, 1, 2, 0, 3 } }, //                    Q5_K 36.0 -> 33.4, IQ4_XS 31.2 -> 29.2
    { GGML_TYPE_COUNT, 248320, 5120, { 4, 2, 1, 0, 3 } }, // the output head: Q6_K 1220.7 -> 1209.0 (streamed at A's 1,4: 1273.9)
};

static const qpn_cfg * ggml_cuda_qpn_find_cfg_0091(const ggml_type type, const int64_t n, const int64_t k) {
    for (const qpn_cfg_0091 & c : qpn_cfgs_0091) {
        if ((c.type == type || c.type == GGML_TYPE_COUNT) && c.n == n && c.k == k) {
            return &c.cfg;
        }
    }
    return nullptr;
}

// LLAMA_QPN_WIDE bit 2 (default on): at 5 to 8 tokens, the splits of the streaming kernel at T = 8 per shape and type
// (type GGML_TYPE_COUNT = any type), from a sweep of S, R, P, staging and blocks per SM (mode) on the real tensors at 8 tokens, checked at
// 5, card 1, SM clock recorded; us at 8 tokens against the default (the streaming kernel at 24 warps per SM at the original
// split). New summation orders.
static const qpn_cfg_0091 qpn_cfgs_0095[] = {
    { GGML_TYPE_Q3_K,   17408,  5120, { 2, 4, 1, 0, 0 } }, // ffn gate/up, Q3_K: the whole-superblock kernel, 68.6 (streamed 74.6 at 16 warps per SM, 82.4 at 24)
    { GGML_TYPE_COUNT,  17408,  5120, { 2, 4, 1, 0, 2 } }, // ffn gate/up, 16 warps per SM: Q5_K 83.6 -> 75.2, Q6_K 94.6 -> 86.8, Q4_K 70.6 -> 62.0, IQ4_NL 74.5 -> 70.2, IQ4_XS 70.8 -> 67.6
    { GGML_TYPE_COUNT,  10240,  5120, { 4, 1, 1, 0, 3 } }, // GDN qkv: Q5_K 67.5 -> 55.3, Q6_K 70.0 -> 60.6, Q4_K 59.6 -> 41.7, IQ4_NL 63.3 -> 46.2, IQ4_XS 57.9 -> 47.8
    { GGML_TYPE_IQ4_XS, 12288,  5120, { 2, 2, 1, 0, 0 } }, // attn q + gate, IQ4_XS: the whole-superblock kernel, 51.7 (streamed at best 54.4; at 5 tokens 51.4 against 54.1)
    { GGML_TYPE_COUNT,  12288,  5120, { 4, 2, 1, 0, 3 } }, //                Q5_K 69.8 -> 61.9, Q4_K 61.2 -> 47.2
    { GGML_TYPE_IQ4_XS,  5120, 17408, { 4, 1, 1, 0, 0 } }, // ffn down, IQ4_XS: the whole-superblock kernel, 80.9 (streamed at best 85.4; at 5 tokens 80.1 against 84.1)
    { GGML_TYPE_IQ4_NL,  5120, 17408, { 4, 1, 2, 0, 3 } }, //           IQ4_NL 106.3 -> 84.0
    { GGML_TYPE_Q4_K,    5120, 17408, { 8, 1, 1, 0, 3 } }, //           Q4_K 99.3 -> 74.4 (16 warps per SM: 78.1)
    { GGML_TYPE_COUNT,   5120, 17408, { 4, 1, 1, 0, 2 } }, //           Q5_K 112.4 -> 94.9, Q6_K 120.6 -> 97.7
    { GGML_TYPE_COUNT,   6144,  5120, { 4, 1, 1, 0, 2 } }, // GDN z: Q5_K 37.3 -> 32.0, Q6_K 40.0 -> 33.6, Q4_K 32.6 -> 26.7, IQ4_XS 31.8 -> 29.6
    { GGML_TYPE_Q8_0,    5120,  6144, { 4, 1, 1, 0, 0 } }, // GDN out, Q8_0: the whole-superblock kernel, 38.6 (streamed at best 39.1)
    { GGML_TYPE_IQ4_XS,  5120,  6144, { 4, 1, 1, 0, 0 } }, // GDN out, attn out, IQ4_XS: the whole-superblock kernel, 32.9 (streamed at best 33.9; at 5 tokens 32.7 against 33.8)
    { GGML_TYPE_COUNT,   5120,  6144, { 4, 1, 1, 0, 2 } }, //                    Q5_K 42.7 -> 36.3, Q6_K 44.3 -> 36.4, Q4_K 37.4 -> 30.2
    { GGML_TYPE_COUNT, 248320,  5120, { 4, 2, 1, 0, 2 } }, // the output head: Q6_K 1375.8 -> 1268.7 (at 5 tokens 1361.5 -> 1216.5)
};

static const qpn_cfg * ggml_cuda_qpn_find_cfg_0095(const ggml_type type, const int64_t n, const int64_t k) {
    for (const qpn_cfg_0091 & c : qpn_cfgs_0095) {
        if ((c.type == type || c.type == GGML_TYPE_COUNT) && c.n == n && c.k == k) {
            return &c.cfg;
        }
    }
    return nullptr;
}

// the DFlash2 draft's shapes at 5 to 8 tokens (it runs them at 8), taken from the nearest shape of qpn_cfgs_0095 (no
// sweep): attn q 4096x5120 as GDN z 6144x5120, attn out 5120x4096 as 5120x6144, fc 5120x25600 as the Q4_K ffn down, and the grids of
// 40 row tiles or fewer (the conv projections 1280x5120, k/v 1024x5120, selector_hidden 256x5120) split along K as the Q8_0 1024-row k/v
// at 1-4 tokens; the Q6_K head subset as the output head. Only consulted when qpn_cfgs_0095 has no entry.
static const qpn_cfg_0091 qpn_cfgs_dflash[] = {
    { GGML_TYPE_COUNT,  4096,  5120, { 4, 1, 1, 0, 2 } },
    { GGML_TYPE_COUNT,  5120,  4096, { 4, 1, 1, 0, 2 } },
    { GGML_TYPE_COUNT,  5120, 25600, { 8, 1, 1, 0, 3 } },
    { GGML_TYPE_COUNT,  1280,  5120, { 8, 1, 4, 0, 3 } },
    { GGML_TYPE_COUNT,  1024,  5120, { 8, 1, 4, 0, 3 } },
    { GGML_TYPE_COUNT,   256,  5120, { 8, 1, 4, 0, 3 } },
    { GGML_TYPE_Q6_K,  98304,  5120, { 4, 2, 1, 0, 2 } },
};

static const qpn_cfg * ggml_cuda_qpn_find_cfg_dflash(const ggml_type type, const int64_t n, const int64_t k) {
    for (const qpn_cfg_0091 & c : qpn_cfgs_dflash) {
        if ((c.type == type || c.type == GGML_TYPE_COUNT) && c.n == n && c.k == k) {
            return &c.cfg;
        }
    }
    return nullptr;
}

static const qpn_cfg * ggml_cuda_qpn_find_cfg(const int64_t n, const int64_t k) {
    for (const qpn_shape_cfg & c : qpn_cfgs) {
        if (c.n == n && c.k == k) {
            return &c.cfg;
        }
    }
    return nullptr;
}

// Wide verify. LLAMA_QPN_WIDE: bitmask, default 15. 2 = the split table
// qpn_cfgs_0095 at 5 to 8 tokens (a new summation order); 4 = pairs at 5 to 8 tokens; 8 = the attention input's q + gate,
// k and v as one launch (ggml_cuda_mul_mat_qpn3) at 4 to 8 tokens. 1 = products of 5 to
// 8 tokens on qpn_stream_loop (the kernel at T = 8 with the token count at run time), at their split (outputs bit-identical
// to the whole-superblock kernel at the same split), unstaged, at QPN_WIDE_MODE blocks of 8 warps per SM. 0 = the
// whole-superblock kernel at 5 to 8 tokens.
#define QPN_WIDE_MODE 3
static int ggml_cuda_qpn_wide_mask() {
    static const int mask = [] { const char * e = getenv("LLAMA_QPN_WIDE"); return e == nullptr ? 15 : atoi(e); }();
    return mask;
}

// the block's staged activations must fit next to the static shared memory in 48 KiB
static size_t ggml_cuda_qpn_stage_bytes(const int nsb, const int P, const int T) {
    return (size_t) ((nsb + P - 1)/P)*32*T*sizeof(uint4);
}
static constexpr size_t QPN_STAGE_MAX = 40*1024;
static constexpr size_t QPN_STAGE_MAX_WIDE = 36*1024; // at 5 to 8 tokens the reduction buffer is 8 KiB, not 4

static qpn_cfg ggml_cuda_qpn_config(const ggml_type type, const int64_t n, const int64_t k, const int T) {
    const int nsb = (int) (k/QK_K);
    qpn_cfg c = { 4, 2, 1, 1 };
    const qpn_cfg * c91 = T <= 4 && (ggml_cuda_qpn_stream_mask() & 8) ? ggml_cuda_qpn_find_cfg_0091(type, n, k) : nullptr;
    const qpn_cfg * c95 = T > 4 && (ggml_cuda_qpn_wide_mask() & 2) ? ggml_cuda_qpn_find_cfg_0095(type, n, k) : nullptr;
    if (c95 == nullptr && T > 4 && (ggml_cuda_qpn_wide_mask() & 2)) {
        c95 = ggml_cuda_qpn_find_cfg_dflash(type, n, k);
    }
    if (c95) {
        c = *c95;
    } else if (c91) {
        c = *c91;
    } else if (const qpn_cfg * tc = T <= 4 ? ggml_cuda_qpn_find_type_cfg(type, n, k) : nullptr) {
        c = *tc;
    } else if (const qpn_cfg * sc = ggml_cuda_qpn_find_cfg(n, k)) {
        c = *sc;
    } else if (n >= 65536) {
        c = { 1, 4, 1, 0 };
    }
    // [TAG_QPN_SMALL] LLAMA_QPN_SMALL_CFG (default off; 1 = { 8, 1, 4, 0 }, or "S,R,P,stage"): at 1 to 4 tokens, the products of at
    // most 1,280 rows (attention k/v, the GDN alpha/beta, the draft's conv projections) split along K too. As their own row tiles
    // alone (2 to 40 blocks) they run latency-bound: alone on this card, the 96-row ab 16.5 -> 8.6 us, the 1024-row k/v 23-27 -> 11-15.
    // A new summation order; in a 1-slot decode at ~41% acceptance a verify step went 39.8 -> 39.4 ms, but the greedy path moved
    // (1219/2960 -> 1201/3016 accepted) and the run was no faster, so it stays off
    if (T <= 4 && n <= 1280) {
        static const qpn_cfg small = [] {
            qpn_cfg s = { 0, 1, 1, 0 };
            if (const char * e = getenv("LLAMA_QPN_SMALL_CFG")) {
                if (strchr(e, ',') != nullptr) {
                    sscanf(e, "%d,%d,%d,%d", &s.S, &s.R, &s.P, &s.stage);
                } else if (atoi(e) != 0) {
                    s = { 8, 1, 4, 0 };
                }
            }
            return s;
        }();
        if (small.S > 0) {
            c = small;
        }
    }
    const char * ecfg = getenv("LLAMA_MMVQ_QPN_CFG");
    if (ecfg) { // "S,R,P,stage[,mode]", for the sweep
        c.mode = 0;
        sscanf(ecfg, "%d,%d,%d,%d,%d", &c.S, &c.R, &c.P, &c.stage, &c.mode);
    }
    if (T <= 4 && c.mode == 0 && (ggml_cuda_qpn_stream_mask() & 4) && ecfg == nullptr) {
        c.mode = 3; // the streaming loop for every other key, at its split (bit-identical)
    }
    if (T > 4 && ecfg == nullptr) { // 5 to 8 tokens streamed, unstaged, at the key's split (the table's split and mode)
        if (!(ggml_cuda_qpn_wide_mask() & 1)) {
            c.mode = 0;
        } else if (c95 == nullptr) {
            c.mode  = QPN_WIDE_MODE;
            c.stage = 0;
        }
    }
    if (c.mode != 2 && c.mode != 3 && c.mode != 4 && c.mode != 5) {
        c.mode = 0;
    }
    c.P = std::max(1, std::min(c.P, nsb));
    c.S = std::max(1, std::min(c.S, QPN_MAX_WARPS));
    c.R = std::max(1, std::min(c.R, QPN_MAX_WARPS / c.S));
    if (ggml_cuda_qpn_stage_bytes(nsb, c.P, T) > (T > 4 ? QPN_STAGE_MAX_WIDE : QPN_STAGE_MAX)) {
        c.stage = 0;
    }
    if (T <= 4 && c.mode == 2 && !c.stage) {
        c.mode = 3;
    }
    return c;
}

// the single products (ggml_cuda_mul_mat_qpn) whose pass of 9 to 16 tokens runs as paired 8-token blocks instead of the
// kernel at T = 16 (LLAMA_QPN_PAIR16: unset = this table, 0 = none, 1 = every unstaged key). Paired blocks decode every weight twice, so
// they win only where the kernel at T = 16 (128 registers, 16 warps per SM) leaves a second wave or idle warps: the checker on the real
// tensors, card 0, 1380 MHz, us at 16 tokens, T = 16 -> paired, two runs each. Not routed: Q5_K, Q6_K, IQ4_XS
// ffn down (154 -> 168, 166 -> 189, 137 -> 161), IQ4_NL ffn down (174 -> 173), Q5_K, Q6_K, IQ4_XS 5120x6144 (61 -> 69, 64 -> 73, 54 -> 65)
struct qpn_pair16_key { ggml_type type; int64_t n, k; };
static const qpn_pair16_key qpn_pair16_keys[] = {
    { GGML_TYPE_Q4_K, 5120, 17408 }, // ffn down (5 layers)  172.6, 167.7 -> 161.1, 160.9
    { GGML_TYPE_Q8_0, 5120,  6144 }, // GDN out (1)          71.5, 71.5 -> 61.1, 61.1
    { GGML_TYPE_Q4_K, 5120,  6144 }, // GDN out (1)          58.3, 58.3 -> 55.8, 55.7
};
static bool ggml_cuda_qpn_pair16(const ggml_type type, const int64_t n, const int64_t k) {
    static const int mode = [] { const char * e = getenv("LLAMA_QPN_PAIR16"); return e == nullptr ? -1 : atoi(e); }();
    if (mode >= 0) {
        return mode != 0;
    }
    for (const qpn_pair16_key & e : qpn_pair16_keys) {
        if ((e.type == type || e.type == GGML_TYPE_COUNT) && e.n == n && e.k == k) {
            return true;
        }
    }
    return false;
}

// which single products run MODE 5 (qpn_stream_loop with a lookahead of 4 units, at most 168 registers, blocks of at most 4
// warps): the ffn down and the GDN / attention out projections, whose grids are 160 blocks of 4 warps (about 9 warps per SM, so the
// registers cost no occupancy) and which stalled on their loads. The checker (p6dcheck, random weights of the real types, card 1, 1380 MHz,
// two runs each, us per product, one card's shape / the --split-mode tensor slice): at 16 tokens every type gains (Q5_K down 155.1 ->
// 124.3 / 85.0 -> 69.0, Q6_K 167.3 -> 133.8 / 91.2 -> 73.6, IQ4_XS 138.6 -> 123.2 / 75.8 -> 67.5, IQ4_NL 175.6 -> 144.9 / 106.5 -> 87.3,
// Q5_K out 61.7 -> 50.6 / 35.5 -> 30.1, Q6_K out 66.6 -> 54.5 / 38.5 -> 32.9); at 8 tokens Q4_K, Q5_K and Q6_K gain (Q5_K down 102.1 ->
// 90.7 / 55.6 -> 50.3, Q6_K down 102.0 -> 97.1 / 57.5 -> 55.2, Q5_K out 40.1 -> 37.1 / 23.5 -> 22.6, Q4_K out 33.8 -> 31.9 / 20.1 -> 19.8)
// and IQ4_XS and IQ4_NL lose (85.2 -> 89.5 / 48.2 -> 50.3, 92.3 -> 100.0 / 52.0 -> 58.5). LLAMA_QPN_DEEP: unset = this rule, 0 = never,
// 1 = every eligible product.
// and IQ4_XS at exactly 8 tokens, with QPN_FULL (random weights of the type, card 1, the slice of ffn down 5120 x 8704:
// 46.2 us on today's whole-superblock kernel, 42.1 deep and full; IQ4_NL stays streamed at LA 2: 47.3 against 54.0 deep)
static bool ggml_cuda_qpn_deep(const ggml_type type, const int T) {
    static const int mode = [] { const char * e = getenv("LLAMA_QPN_DEEP"); return e == nullptr ? -1 : atoi(e); }();
    const bool eligible = type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K || type == GGML_TYPE_Q6_K || type == GGML_TYPE_IQ4_XS ||
        type == GGML_TYPE_IQ4_NL; // two code chunks per two units (qpn_stream_t::CPU2 == 2)
    if (!eligible || mode == 0) {
        return false;
    }
    if (mode > 0 || T > 8) {
        return true;
    }
    return type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K || type == GGML_TYPE_Q6_K || (type == GGML_TYPE_IQ4_XS && T == 8 && ggml_cuda_qpn_full_on());
}

// a pass of 9 to 16 tokens takes the 8-token key's split (S, R, P), so each column's sums are in the order of its pass
// of 8 (bit-identical to it), on qpn_stream_loop at T = 16, unstaged, at 2 blocks of 8 warps per SM (128 registers: two tiles'
// accumulators)
static qpn_cfg ggml_cuda_qpn_config16(const ggml_type type, const int64_t n, const int64_t k) {
    qpn_cfg c = ggml_cuda_qpn_config(type, n, k, 8);
    c.mode  = 2;
    c.stage = 0;
    return c;
}

// t's type is enabled and t is a plain 2-D weight the kernels can take
bool ggml_cuda_qpn_takes(const ggml_tensor * t, const int cc) {
    const int bit = ggml_cuda_qpn_type_bit(t->type);
    if (!bit || !(ggml_cuda_qpn_mask() & bit) || !volta_mma_available(cc)) {
        return false;
    }
    return !(t->view_src != nullptr || t->ne[2] != 1 || t->ne[3] != 1 || !ggml_is_contiguous(t) ||
            t->ne[0] % QK_K != 0 || t->ne[1] % 32 != 0 || (t->flags & GGML_TENSOR_FLAG_BACKEND_LAYOUT));
}

bool ggml_cuda_qpn_eligible(const ggml_tensor * t, const int cc) {
    if (!ggml_cuda_qpn_takes(t, cc)) {
        return false;
    }
    return ggml_cuda_qpn_force(ggml_cuda_qpn_type_bit(t->type)) || ggml_cuda_qpn_find_route(t->type, t->ne[1], t->ne[0]) != nullptr;
}

// a draft model's weights (its MTP layer, and its LM head subset, which llama_model::set_head_subset makes as a
// tensor of its own) have their own routes, because the draft runs them at 1 token per draft step and at the verify width only
// in the catch-up. LLAMA_QPN_DRAFT=0 repacks none of them; LLAMA_QPN_DRAFT_FORCE: 1 = every enabled type, any other nonzero
// value = a bitmask of types (as LLAMA_MMVQ_QPN_FORCE's), for study. The routes: the same rule over the draft steps (1 token)
// and the catch-up (the verify width), per round at n-max 3 / 7, from the captures of qpn_routes_0096 (drift of the draft graphs'
// untouched kernels at most 2.4%), us per round on poetry, dp4a -> QPN; CSV and code within 1-3 points of it.
// Not routed: eh_proj 5120x10240 (+21% / +7%), attn k, v 1024x5120 (+126% to +137%).
static const qpn_route qpn_routes_draft[] = {
    { GGML_TYPE_Q3_K, 98304,  5120 }, // the LM head over the 98,304-token draft vocabulary  1706 -> 1030 / 3983 -> 2423
    { GGML_TYPE_Q4_K, 17408,  5120 }, // ffn gate, up (dp4a: one fused kernel)               664 ->  510 / 1459 -> 1054
    { GGML_TYPE_Q6_K,  5120,  6144 }, // attn out                                            203 ->  156 /  435 ->  321
    { GGML_TYPE_Q4_K,  5120, 17408 }, // ffn down                                            311 ->  297 /  719 ->  616
    { GGML_TYPE_Q6_K, 12288,  5120 }, // attn q + gate                                       305 ->  302 /  643 ->  613
};

bool ggml_cuda_qpn_eligible_draft(const ggml_tensor * t, const int cc) {
    static const bool on    = [] { const char * e = getenv("LLAMA_QPN_DRAFT"); return e == nullptr || atoi(e) != 0; }();
    static const int  force = [] { const char * e = getenv("LLAMA_QPN_DRAFT_FORCE"); const int v = e == nullptr ? 0 : atoi(e);
                                   return v == 1 ? GGML_CUDA_QPN_ALL : v; }();
    if (!on || !ggml_cuda_qpn_takes(t, cc)) {
        return false;
    }
    if (force & ggml_cuda_qpn_type_bit(t->type)) {
        return true;
    }
    for (const qpn_route & r : qpn_routes_draft) {
        if (r.type == t->type && r.n == t->ne[1] && r.k == t->ne[0]) {
            return true;
        }
    }
    // k and v alone lose to dp4a (+126% to +137%, above); in the launch with q they win (ggml_cuda_qpn_draft3_ok)
    return t->type == GGML_TYPE_Q6_K && t->ne[1] == 1024 && t->ne[0] == 5120 && ggml_cuda_qpn_draft_kv_on();
}

// a DFlash2 draft's weights (its five layers' attn q, k, v, out, ffn gate, up, down, the two dynamic-conv projections,
// fc, selector_hidden, and its LM head subset) have their own routes, because the draft runs all of them at 8 rows (the block, and the
// injection of the verify's 8 rows) and none of its attention shapes is a target shape. Before it, the trunk went through the target's
// table, so only the shapes that equal a target key (ffn gate/up, down, and the 1024-row k/v) took QPN. LLAMA_QPN_DFLASH=0 repacks
// none of them; LLAMA_QPN_DFLASH_FORCE: 1 = every enabled type, any other nonzero value = a bitmask of types, for study. The routes:
// in-graph timing at 8 rows, node-level captures of the draft pass and the injection (poetry at T=0, generated tokens 100-220, card 1,
// the same text in every arm). Per shape, dp4a -> QPN with prep, us per round (the
// five layers): q 326 -> 176, k + v 467 -> 389 (draft pass + injection), out 346 -> 159, gate + up 2295 -> 672, down 1214 -> 487, fc 355
// -> 118, the head 1702 -> 448; the conv projections 337 -> 333 and selector_hidden 18 -> 28 are even or worse alone, and the two Q6_K
// v 44 -> 67. But the products interact in the graph: with only the winning shapes routed the Q4_K k took 28 us a layer against
// 17 with every weight on QPN, and the whole draft pass is fastest with every weight routed: 3.70 ms against 4.14
// (before the Q6_K down was added) and 7.82 on dp4a. So every weight of this draft's layers, fc and selector_hidden is routed.
static const qpn_route qpn_routes_dflash[] = {
    { GGML_TYPE_Q4_K, 17408,  5120 }, // ffn gate, up
    { GGML_TYPE_Q4_K,  5120, 17408 }, // ffn down
    { GGML_TYPE_Q6_K,  5120, 17408 }, // ffn down (Q4_K_M's Q6_K half)
    { GGML_TYPE_Q4_K,  4096,  5120 }, // attn q
    { GGML_TYPE_Q4_K,  1024,  5120 }, // attn k, v (draft pass + injection)
    { GGML_TYPE_Q6_K,  1024,  5120 }, // attn v (Q4_K_M's Q6_K half)
    { GGML_TYPE_Q4_K,  5120,  4096 }, // attn out
    { GGML_TYPE_Q4_K,  1280,  5120 }, // the dynamic-conv projections (attn, ffn)
    { GGML_TYPE_Q4_K,  5120, 25600 }, // fc (the injection)
    { GGML_TYPE_Q4_K,   256,  5120 }, // selector_hidden
    { GGML_TYPE_Q3_K, 98304,  5120 }, // the LM head subset over the 98,304-token draft vocabulary, from the MTP GGUF's head
    { GGML_TYPE_Q6_K, 98304,  5120 }, // ... or from the target's head (not captured)
};

bool ggml_cuda_qpn_eligible_dflash(const ggml_tensor * t, const int cc) {
    static const bool on    = [] { const char * e = getenv("LLAMA_QPN_DFLASH"); return e == nullptr || atoi(e) != 0; }();
    static const int  force = [] { const char * e = getenv("LLAMA_QPN_DFLASH_FORCE"); const int v = e == nullptr ? 0 : atoi(e);
                                   return v == 1 ? GGML_CUDA_QPN_ALL : v; }();
    if (!on || !ggml_cuda_qpn_takes(t, cc)) {
        return false;
    }
    if (force & ggml_cuda_qpn_type_bit(t->type)) {
        return true;
    }
    for (const qpn_route & r : qpn_routes_dflash) {
        if (r.type == t->type && r.n == t->ne[1] && r.k == t->ne[0]) {
            return true;
        }
    }
    return false;
}

// per device: the counters of products split across blocks, one row per stream (kernels on one stream run in order,
// and the last block of a tile resets its counter); allocated at load, zeroed once
static unsigned * qpn_counters[GGML_CUDA_MAX_DEVICES] = { nullptr };

template <ggml_type type>
static void ggml_cuda_qpn_repack_type(ggml_tensor * t, char * tmp, const size_t tmp_size, cudaStream_t stream) {
    constexpr int TSB = qpn_tsb<type>();
    const int     nsb       = (int) (t->ne[0]/QK_K);
    const int64_t ntiles    = t->ne[1]/32;
    const size_t  tile_size = (size_t) nsb*TSB;
    const int64_t per_pass  = std::max<int64_t>(1, (int64_t) (tmp_size / tile_size));
    char * data = (char *) t->data;
    for (int64_t t0 = 0; t0 < ntiles; t0 += per_pass) {
        const int64_t nt = std::min(per_pass, ntiles - t0);
        CUDA_CHECK(cudaMemcpyAsync(tmp, data + t0*tile_size, nt*tile_size, cudaMemcpyDeviceToDevice, stream));
        qpn_repack_kernel<type><<<(unsigned) (nt*nsb), 32, 0, stream>>>(tmp, data + t0*tile_size, nsb);
        CUDA_CHECK(cudaGetLastError());
    }
}

bool ggml_cuda_qpn_repack(ggml_tensor * t, const int device) {
    ggml_cuda_set_device(device);
    if (qpn_counters[device] == nullptr) {
        const size_t sz = (size_t) GGML_CUDA_MAX_STREAMS*QPN_MAX_TILES*sizeof(unsigned);
        CUDA_CHECK(cudaMalloc(&qpn_counters[device], sz));
        CUDA_CHECK(cudaMemset(qpn_counters[device], 0, sz));
    }
    const size_t nbytes = ggml_nbytes(t);
    const size_t tile_size = (size_t) (t->ne[0]/QK_K)*32*ggml_type_size(t->type);
    // one scratch of at most 64 MiB (whole row tiles), freed before returning
    const size_t tmp_size = std::min(nbytes, std::max(tile_size, (size_t) 64*1024*1024 / tile_size * tile_size));
    char * tmp = nullptr;
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMalloc(&tmp, tmp_size));
    cudaStream_t stream = 0;
    switch (t->type) {
        case GGML_TYPE_Q4_K: ggml_cuda_qpn_repack_type<GGML_TYPE_Q4_K>(t, tmp, tmp_size, stream); break;
        case GGML_TYPE_Q5_K: ggml_cuda_qpn_repack_type<GGML_TYPE_Q5_K>(t, tmp, tmp_size, stream); break;
        case GGML_TYPE_Q6_K: ggml_cuda_qpn_repack_type<GGML_TYPE_Q6_K>(t, tmp, tmp_size, stream); break;
        case GGML_TYPE_IQ4_XS: ggml_cuda_qpn_repack_type<GGML_TYPE_IQ4_XS>(t, tmp, tmp_size, stream); break;
        case GGML_TYPE_IQ4_NL: ggml_cuda_qpn_repack_type<GGML_TYPE_IQ4_NL>(t, tmp, tmp_size, stream); break;
        case GGML_TYPE_Q3_K:   ggml_cuda_qpn_repack_type<GGML_TYPE_Q3_K>(t, tmp, tmp_size, stream); break;
        case GGML_TYPE_Q8_0:   ggml_cuda_qpn_repack_type<GGML_TYPE_Q8_0>(t, tmp, tmp_size, stream); break;
        default: GGML_ABORT("fatal error");
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(tmp));
    t->flags |= GGML_TENSOR_FLAG_BACKEND_LAYOUT;
    return true;
}

// the inverse of the repack: the GGUF bytes of tiles [0, ntiles) of this pass, from the repacked W
template <ggml_type type>
static __global__ void qpn_unrepack_kernel(const char * __restrict__ W, char * __restrict__ dst, const int nsb) {
    typedef typename qpn_t<type>::block block;
    constexpr int NCH = qpn_t<type>::NCH, TSB = qpn_tsb<type>(), BS = sizeof(block);
    __shared__ block blk[32];
    const int lane = threadIdx.x;
    const int tile = blockIdx.x / nsb, sb = blockIdx.x % nsb;
    uint32_t w[NCH*4];
    uint4    tl;
    qpn_load_row<type>(W + ((int64_t) tile*nsb + sb)*TSB, lane, w, tl);
    qpn_unpack_row<type>(w, tl, &blk[lane]);
    __syncwarp();
    const uint16_t * s16 = (const uint16_t *) blk;
    for (int i = lane; i < 32*BS/2; i += 32) {
        const int r = i / (BS/2), o = i % (BS/2);
        ((uint16_t *) (dst + ((int64_t) (tile*32 + r)*nsb + sb)*BS))[o] = s16[i];
    }
}

// word i of row r's chunk words in a tile-superblock (w[i] of qpn_load_row)
static __device__ __forceinline__ uint32_t qpn_word(const char * tsb, const int r, const int i) {
    return *(const uint32_t *) (tsb + (i/4)*QPN_CHUNK + r*16 + 4*(i % 4));
}
static __device__ __forceinline__ uint4 qpn_chunk_row(const char * tsb, const int r, const int ch) {
    return *(const uint4 *) (tsb + ch*QPN_CHUNK + r*16);
}
// 32 bytes as 8 words into p, 2-byte aligned (Q6_K blocks) or 4-byte aligned
template <int A>
static __device__ __forceinline__ void qpn_st32(unsigned char * p, const uint32_t * q) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        if constexpr (A >= 4) {
            ((uint32_t *) p)[i] = q[i];
        } else {
            ((uint16_t *) p)[2*i] = (uint16_t) q[i]; ((uint16_t *) p)[2*i + 1] = (uint16_t) (q[i] >> 16);
        }
    }
}
template <ggml_type type>
static constexpr __host__ __device__ bool qpn_has_fast_unrepack() {
    return type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K || type == GGML_TYPE_Q6_K || type == GGML_TYPE_IQ4_XS;
}

// the same bytes as qpn_unrepack_kernel, for the device-side unrepack of the MMQ path. Q4_K, Q5_K, Q6_K, IQ4_XS: four
// threads per row build its GGUF block in shared memory straight from the repacked words, each a disjoint 32-byte run of
// codes (plus the header, scales, d), the high-bit bytes Q5_K and Q6_K share between sub-blocks ORed across the row's lanes;
// other types: one warp rebuilds the 32 blocks as qpn_unrepack_kernel does. Then 128 threads copy them out a block-aligned
// word at a time (16 bytes where the block size allows), rows of consecutive threads landing on consecutive addresses
template <ggml_type type>
static __global__ void __launch_bounds__(128) qpn_unrepack_fast_kernel(const char * __restrict__ W, char * __restrict__ dst, const int nsb) {
    typedef typename qpn_t<type>::block block;
    constexpr int NCH = qpn_t<type>::NCH, TSB = qpn_tsb<type>(), BS = sizeof(block);
    constexpr int U   = BS % 16 == 0 ? 16 : BS % 8 == 0 ? 8 : BS % 4 == 0 ? 4 : 2;
    constexpr int NU  = BS / U;
    __shared__ __align__(16) unsigned char raw[32*BS];
    block * blk = (block *) raw;
    const int tile = blockIdx.x / nsb, sb = blockIdx.x % nsb;
    const char * tsb = W + ((int64_t) tile*nsb + sb)*TSB;
    if constexpr (qpn_has_fast_unrepack<type>()) {
        const int r = threadIdx.x / 4, t = threadIdx.x % 4;
        unsigned char * b = raw + r*BS;
        uint32_t q[8] = {0}, h[8] = {0};
        if constexpr (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) {
            // qs[32t + l]: column l of sub-block 2t (low nibble) and of 2t + 1 (high)
            constexpr int ch0 = type == GGML_TYPE_Q4_K ? 1 : 3, qs0 = type == GGML_TYPE_Q4_K ? 16 : 48;
            const uint4 c0 = qpn_chunk_row(tsb, r, ch0 + 2*t), c1 = qpn_chunk_row(tsb, r, ch0 + 2*t + 1);
            const uint32_t lo[4] = {c0.x, c0.y, c0.z, c0.w}, hi[4] = {c1.x, c1.y, c1.z, c1.w};
#pragma unroll
            for (int l = 0; l < 32; ++l) {
                q[l/4] |= (uint32_t) (qpn_code4<type>(lo, l) | qpn_code4<type>(hi, l) << 4) << (8*(l % 4));
            }
            qpn_st32<4>(b + qs0 + 32*t, q);
            if (t == 0) {
                *(uint4 *) b = qpn_chunk_row(tsb, r, 0); // d, dmin, scales
            }
            if constexpr (type == GGML_TYPE_Q5_K) {
                // qh[l] bit j: the fifth bit of column l of sub-block j (high-bit word w[4 + j])
                const uint32_t w0 = qpn_word(tsb, r, 4 + 2*t), w1 = qpn_word(tsb, r, 4 + 2*t + 1);
#pragma unroll
                for (int l = 0; l < 32; ++l) {
                    const int p = qpn_hb5(l/8, l % 8);
                    h[l/4] |= (uint32_t) ((((w0 >> p) & 1) << (2*t)) | (((w1 >> p) & 1) << (2*t + 1))) << (8*(l % 4));
                }
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    h[i] |= __shfl_xor_sync(0xffffffff, h[i], 1);
                    h[i] |= __shfl_xor_sync(0xffffffff, h[i], 2);
                }
                if (t == 1) {
                    qpn_st32<4>(b + 16, h);
                }
            }
        } else if constexpr (type == GGML_TYPE_Q6_K) {
            // ql[64ip + 32hh + l]: column l of sub-block 4ip + hh (low nibble) and of 4ip + hh + 2 (high); qh[32ip + l] bits 2g, 2g + 1:
            // bits 4-5 of column l of sub-block 4ip + g (high-bit words w[32 + 2j + q/2])
            const int ip = t/2, hh = t % 2, j0 = 4*ip + hh, j1 = j0 + 2;
            const uint4 c0 = qpn_chunk_row(tsb, r, j0), c1 = qpn_chunk_row(tsb, r, j1);
            const uint32_t lo[4] = {c0.x, c0.y, c0.z, c0.w}, hi[4] = {c1.x, c1.y, c1.z, c1.w};
            const uint32_t a0 = qpn_word(tsb, r, 32 + 2*j0), a1 = qpn_word(tsb, r, 33 + 2*j0);
            const uint32_t b0 = qpn_word(tsb, r, 32 + 2*j1), b1 = qpn_word(tsb, r, 33 + 2*j1);
#pragma unroll
            for (int l = 0; l < 32; ++l) {
                const int qq = l/8, s = l % 8, p = qpn_hb6(qq, s);
                q[l/4] |= (uint32_t) (qpn_code4<type>(lo, l) | qpn_code4<type>(hi, l) << 4) << (8*(l % 4));
                const uint32_t ha = ((qq < 2 ? a0 : a1) >> p) & 3, hb = ((qq < 2 ? b0 : b1) >> p) & 3;
                h[l/4] |= (uint32_t) ((ha << (2*hh)) | (hb << (2*hh + 4))) << (8*(l % 4));
            }
            qpn_st32<2>(b + 64*ip + 32*hh, q);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                h[i] |= __shfl_xor_sync(0xffffffff, h[i], 1);
            }
            if (hh == 0) {
                qpn_st32<2>(b + 128 + 32*ip, h);
            }
            if (t == 0) {
                const uint4 sv = qpn_chunk_row(tsb, r, 12);
                const uint32_t sw[4] = {sv.x, sv.y, sv.z, sv.w};
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    ((uint16_t *) (b + 192))[2*i] = (uint16_t) sw[i]; ((uint16_t *) (b + 192))[2*i + 1] = (uint16_t) (sw[i] >> 16);
                }
                *(uint16_t *) (b + 208) = (uint16_t) qpn_load_tail<type>(tsb, r).x;
            }
        } else {
            // IQ4_XS: qs[16j + m]: column m of sub-block j (low nibble) and column m + 16 (high), j = 2t, 2t + 1
#pragma unroll
            for (int jj = 0; jj < 2; ++jj) {
                const uint4 c = qpn_chunk_row(tsb, r, 2*t + jj);
                const uint32_t cw[4] = {c.x, c.y, c.z, c.w};
#pragma unroll
                for (int m = 0; m < 16; ++m) {
                    q[4*jj + m/4] |= (uint32_t) (qpn_code4<type>(cw, m) | qpn_code4<type>(cw, m + 16) << 4) << (8*(m % 4));
                }
            }
            qpn_st32<4>(b + 8 + 32*t, q);
            if (t == 0) {
                const uint4 tl = qpn_load_tail<type>(tsb, r);
                ((uint32_t *) b)[0] = tl.x; ((uint32_t *) b)[1] = tl.y; // d, scales_h, scales_l
            }
        }
    } else if (threadIdx.x < 32) {
        uint32_t w[NCH*4];
        uint4    tl;
        qpn_load_row<type>(tsb, threadIdx.x, w, tl);
        qpn_unpack_row<type>(w, tl, &blk[threadIdx.x]);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < 32*NU; i += blockDim.x) {
        const int r = i / NU, o = i % NU;
        char       * d = dst + ((int64_t) (tile*32 + r)*nsb + sb)*BS + o*U;
        const char * s = (const char *) &blk[r] + o*U;
        if constexpr (U == 16) {
            *(uint4 *) d = *(const uint4 *) s;
        } else if constexpr (U == 8) {
            *(uint2 *) d = *(const uint2 *) s;
        } else if constexpr (U == 4) {
            *(uint32_t *) d = *(const uint32_t *) s;
        } else {
            *(uint16_t *) d = *(const uint16_t *) s;
        }
    }
}

// LLAMA_QPN_EXPAND_CHECK=1: every expansion / device unrepack also runs the earlier kernel into a second buffer the caller
// provides and compares the two on the device, printing the first differing element of a call (nothing on agreement)
static __global__ void qpn_check_equal_kernel(const uint16_t * __restrict__ a, const uint16_t * __restrict__ b, const int64_t n, const int kind) {
    for (int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x*blockDim.x) {
        if (a[i] != b[i]) {
            printf("qpn %s check: element %lld differs (new 0x%04x, old 0x%04x)\n", kind ? "unrepack" : "expand",
                   (long long) i, (unsigned) a[i], (unsigned) b[i]);
            return;
        }
    }
}

bool ggml_cuda_qpn_check_enabled() {
    static const bool on = [] {
        const char * e = getenv("LLAMA_QPN_EXPAND_CHECK");
        const bool v = e && atoi(e) != 0;
        if (v) {
            GGML_LOG_WARN("%s: LLAMA_QPN_EXPAND_CHECK: every QPN expansion / unrepack is checked against the earlier kernel\n", __func__);
        }
        return v;
    }();
    return on;
}

bool ggml_cuda_qpn_unpack(const ggml_tensor * t, void * host_dst, const int device) {
    if (!ggml_cuda_qpn_is_repacked(t) || t->view_src != nullptr || ggml_cuda_qpn_type_bit(t->type) == 0) {
        return false;
    }
    ggml_cuda_set_device(device);
    const int     nsb       = (int) (t->ne[0]/QK_K);
    const int64_t ntiles    = t->ne[1]/32;
    const size_t  tile_size = (size_t) nsb*32*ggml_type_size(t->type);
    const size_t  tmp_size  = std::min(ggml_nbytes(t), std::max(tile_size, (size_t) 64*1024*1024 / tile_size * tile_size));
    const int64_t per_pass  = (int64_t) (tmp_size / tile_size);
    char * tmp = nullptr;
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMalloc(&tmp, tmp_size));
    cudaStream_t stream = 0;
    const char * data = (const char *) t->data;
    for (int64_t t0 = 0; t0 < ntiles; t0 += per_pass) {
        const int64_t  nt   = std::min(per_pass, ntiles - t0);
        const unsigned nblk = (unsigned) (nt*nsb);
        switch (t->type) {
            case GGML_TYPE_Q4_K: qpn_unrepack_kernel<GGML_TYPE_Q4_K><<<nblk, 32, 0, stream>>>(data + t0*tile_size, tmp, nsb); break;
            case GGML_TYPE_Q5_K: qpn_unrepack_kernel<GGML_TYPE_Q5_K><<<nblk, 32, 0, stream>>>(data + t0*tile_size, tmp, nsb); break;
            case GGML_TYPE_Q6_K: qpn_unrepack_kernel<GGML_TYPE_Q6_K><<<nblk, 32, 0, stream>>>(data + t0*tile_size, tmp, nsb); break;
            case GGML_TYPE_IQ4_XS: qpn_unrepack_kernel<GGML_TYPE_IQ4_XS><<<nblk, 32, 0, stream>>>(data + t0*tile_size, tmp, nsb); break;
            case GGML_TYPE_IQ4_NL: qpn_unrepack_kernel<GGML_TYPE_IQ4_NL><<<nblk, 32, 0, stream>>>(data + t0*tile_size, tmp, nsb); break;
            case GGML_TYPE_Q3_K:   qpn_unrepack_kernel<GGML_TYPE_Q3_K><<<nblk, 32, 0, stream>>>(data + t0*tile_size, tmp, nsb); break;
            case GGML_TYPE_Q8_0:   qpn_unrepack_kernel<GGML_TYPE_Q8_0><<<nblk, 32, 0, stream>>>(data + t0*tile_size, tmp, nsb); break;
            default: GGML_ABORT("fatal error");
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync((char *) host_dst + t0*tile_size, tmp, nt*tile_size, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    CUDA_CHECK(cudaFree(tmp));
    return true;
}

template <ggml_type type>
static void qpn_unpack_device_type(const char * W, char * d, const int nsb, const unsigned nblk, cudaStream_t stream, char * check) {
    qpn_unrepack_fast_kernel<type><<<nblk, 128, 0, stream>>>(W, d, nsb);
    if (check) {
        qpn_unrepack_kernel<type><<<nblk, 32, 0, stream>>>(W, check, nsb);
        const int64_t n = (int64_t) nblk*32*sizeof(typename qpn_t<type>::block)/2;
        qpn_check_equal_kernel<<<1024, 256, 0, stream>>>((const uint16_t *) d, (const uint16_t *) check, n, 1);
    }
}

bool ggml_cuda_qpn_unpack_device(const ggml_tensor * t, void * dst, cudaStream_t stream, void * check_dst) {
    if (!ggml_cuda_qpn_is_repacked(t) || t->view_src != nullptr) {
        return false;
    }
    const int      nsb  = (int) (t->ne[0]/QK_K);
    const unsigned nblk = (unsigned) ((t->ne[1]/32)*nsb);
    const char *   W    = (const char *) t->data;
    char *         d    = (char *) dst;
    char *         c    = (char *) check_dst;
    switch (t->type) {
        case GGML_TYPE_Q4_K:   qpn_unpack_device_type<GGML_TYPE_Q4_K>  (W, d, nsb, nblk, stream, c); break;
        case GGML_TYPE_Q5_K:   qpn_unpack_device_type<GGML_TYPE_Q5_K>  (W, d, nsb, nblk, stream, c); break;
        case GGML_TYPE_Q6_K:   qpn_unpack_device_type<GGML_TYPE_Q6_K>  (W, d, nsb, nblk, stream, c); break;
        case GGML_TYPE_IQ4_XS: qpn_unpack_device_type<GGML_TYPE_IQ4_XS>(W, d, nsb, nblk, stream, c); break;
        case GGML_TYPE_IQ4_NL: qpn_unpack_device_type<GGML_TYPE_IQ4_NL>(W, d, nsb, nblk, stream, c); break;
        case GGML_TYPE_Q3_K:   qpn_unpack_device_type<GGML_TYPE_Q3_K>  (W, d, nsb, nblk, stream, c); break;
        case GGML_TYPE_Q8_0:   qpn_unpack_device_type<GGML_TYPE_Q8_0>  (W, d, nsb, nblk, stream, c); break;
        default: return false;
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

template <ggml_type type>
static void qpn_to_fp16_fast_type(const char * W, half * dst, const int nsb, const unsigned nblk, const int64_t K, cudaStream_t stream, half * check) {
    qpn_to_fp16_fast_kernel<type><<<nblk, 256, 0, stream>>>(W, dst, nsb, K);
    if (check) {
        constexpr int nth = type == GGML_TYPE_Q4_K || type == GGML_TYPE_IQ4_XS ? 32 : 64; // the earlier kernel's thread count
        qpn_to_fp16_kernel<type><<<nblk, nth, 0, stream>>>(W, check, nsb, K);
        qpn_check_equal_kernel<<<1024, 256, 0, stream>>>((const uint16_t *) dst, (const uint16_t *) check, (int64_t) nblk*32*QK_K, 0);
    }
}

void ggml_cuda_qpn_to_fp16(const ggml_tensor * src0, half * dst, cudaStream_t stream, half * check_dst) {
    const int     nsb    = (int) (src0->ne[0]/QK_K);
    const int64_t ntiles = src0->ne[1]/32;
    const char *  W      = (const char *) src0->data;
    const unsigned nblk  = (unsigned) (ntiles*nsb);
    // the four types of a K-quant model's large weights take the coalesced kernel (LLAMA_QPN_FAST_EXPAND=0: the earlier one)
    static const bool fast = [] { const char * e = getenv("LLAMA_QPN_FAST_EXPAND"); return e == nullptr || atoi(e) != 0; }();
    if (fast) {
        switch (src0->type) {
            case GGML_TYPE_Q4_K:   qpn_to_fp16_fast_type<GGML_TYPE_Q4_K>  (W, dst, nsb, nblk, src0->ne[0], stream, check_dst); CUDA_CHECK(cudaGetLastError()); return;
            case GGML_TYPE_Q5_K:   qpn_to_fp16_fast_type<GGML_TYPE_Q5_K>  (W, dst, nsb, nblk, src0->ne[0], stream, check_dst); CUDA_CHECK(cudaGetLastError()); return;
            case GGML_TYPE_Q6_K:   qpn_to_fp16_fast_type<GGML_TYPE_Q6_K>  (W, dst, nsb, nblk, src0->ne[0], stream, check_dst); CUDA_CHECK(cudaGetLastError()); return;
            case GGML_TYPE_IQ4_XS: qpn_to_fp16_fast_type<GGML_TYPE_IQ4_XS>(W, dst, nsb, nblk, src0->ne[0], stream, check_dst); CUDA_CHECK(cudaGetLastError()); return;
            default: break;
        }
    }
    switch (src0->type) {
        case GGML_TYPE_Q4_K: qpn_to_fp16_kernel<GGML_TYPE_Q4_K><<<nblk, 32, 0, stream>>>(W, dst, nsb, src0->ne[0]); break;
        case GGML_TYPE_Q5_K: qpn_to_fp16_kernel<GGML_TYPE_Q5_K><<<nblk, 64, 0, stream>>>(W, dst, nsb, src0->ne[0]); break;
        case GGML_TYPE_Q6_K: qpn_to_fp16_kernel<GGML_TYPE_Q6_K><<<nblk, 64, 0, stream>>>(W, dst, nsb, src0->ne[0]); break;
        case GGML_TYPE_IQ4_XS: qpn_to_fp16_kernel<GGML_TYPE_IQ4_XS><<<nblk, 32, 0, stream>>>(W, dst, nsb, src0->ne[0]); break;
        case GGML_TYPE_IQ4_NL: qpn_to_fp16_kernel<GGML_TYPE_IQ4_NL><<<nblk, 32, 0, stream>>>(W, dst, nsb, src0->ne[0]); break;
        case GGML_TYPE_Q3_K:   qpn_to_fp16_kernel<GGML_TYPE_Q3_K><<<nblk, 64, 0, stream>>>(W, dst, nsb, src0->ne[0]); break;
        case GGML_TYPE_Q8_0:   qpn_to_fp16_kernel<GGML_TYPE_Q8_0><<<nblk, 32, 0, stream>>>(W, dst, nsb, src0->ne[0]); break;
        default: GGML_ABORT("fatal error");
    }
    CUDA_CHECK(cudaGetLastError());
}


// the prepared input of T tokens: xh (fp16 fragments), xs (per-32 sums, used by Q4_K and Q5_K), xsc (range scales)
static void ggml_cuda_qpn_prep_sizes(const int64_t K, const int T, size_t & sz_xh, size_t & sz_xs, size_t & sz_xsc) {
    const int nsb = (int) (K/QK_K);
    sz_xh  = GGML_PAD((size_t) T*K*sizeof(half), 256);
    sz_xs  = GGML_PAD((size_t) nsb*T*8*sizeof(half), 256);
    sz_xsc = GGML_PAD((size_t) nsb*T*sizeof(float), 256);
}

size_t ggml_cuda_qpn_prep_bytes(const int64_t K, const int T) {
    size_t a, b, c;
    ggml_cuda_qpn_prep_sizes(K, T, a, b, c);
    return a + b + c;
}

void ggml_cuda_qpn_prep_ptrs(char * base, const int64_t K, const int T, half ** xh, half ** xs, float ** xsc) {
    size_t sz_xh, sz_xs, sz_xsc;
    ggml_cuda_qpn_prep_sizes(K, T, sz_xh, sz_xs, sz_xsc);
    *xh  = (half  *) base;
    *xs  = (half  *) (base + sz_xh);
    *xsc = (float *) (base + sz_xh + sz_xs);
}

void ggml_cuda_qpn_prep(const float * x, const int64_t stride_col_x, const int64_t K, const int T, char * base, cudaStream_t stream) {
    half * xh; half * xs; float * xsc;
    ggml_cuda_qpn_prep_ptrs(base, K, T, &xh, &xs, &xsc);
    const int nsb = (int) (K/QK_K), nwarps = nsb*T;
    qpn_prep_kernel<<<(nwarps + 7)/8, 8*WARP_SIZE, 0, stream>>>(x, stride_col_x, nsb, T, xh, xs, xsc);
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_mul_mat_qpn(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_qpn_is_repacked(src0) && src0->view_src == nullptr);
    GGML_ASSERT(src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    // a 3-D input whose rows are one run is ne[1]*ne[2]*ne[3] columns (ggml_cuda_qpn_flat_cols)
    GGML_ASSERT(ggml_cuda_qpn_flat_cols(src1) && ggml_cuda_qpn_flat_cols(dst) && src1->ne[0] == src0->ne[0] && src1->nb[0] == sizeof(float));
    GGML_ASSERT(dst->nb[0] == sizeof(float) && dst->ne[0] == src0->ne[1] && ggml_nrows(dst) == ggml_nrows(src1));
    GGML_ASSERT(ggml_nrows(src1) <= GGML_CUDA_QPN_MAX_TOKENS);

    const int64_t K      = src0->ne[0];
    const int     nsb    = (int) (K/QK_K);
    const int64_t N      = src0->ne[1];
    const int     ntiles = (int) (N/32);
    const int     ncols  = (int) ggml_nrows(src1);
    const bool    mins   = src0->type == GGML_TYPE_Q4_K || src0->type == GGML_TYPE_Q5_K;
    cudaStream_t  stream = ctx.stream();

    const int64_t stride_col_x   = src1->nb[1]/sizeof(float);
    const int64_t stride_col_dst = dst->nb[1]/sizeof(float);

    // with --split-mode tensor, dst is the partial of the ALLREDUCE node the graph loop names, so each finished tile goes to
    // the peer as the product writes it, and that node only polls and sums (ggml_cuda_p2p_ar_push_target says whether it can)
    ggml_cuda_p2p_push push;
    // (one pass only, so the call pushes all of its partial or none of it)
    const bool pushing = ctx.p2p_push_ar != nullptr && ncols <= ggml_cuda_qpn_pass_width() && ctx.p2p_push_ar->data == dst->data && ggml_is_contiguous(dst) &&
        stride_col_dst == N && ggml_nelements(ctx.p2p_push_ar) == ggml_nelements(dst) && (uintptr_t) dst->data % 16 == 0 &&
        ggml_cuda_p2p_ar_push_target(ctx.p2p_push_ar, &push);

    // passes of 16 tokens (LLAMA_QPN_W16=0: of 8, as before), so 9 to 16 tokens read the weights once; of 32 with
    // LLAMA_QPN_W32=1 [TAG_QPN_W32]
    const int pw = ggml_cuda_qpn_single_width();
    for (int col0 = 0; col0 < ncols; ) {
        const int T = std::min(pw, ncols - col0);
        // by the weight's key (a slice under the split takes the whole weight's split); P stays within the slice's K.
        // a pass of 9 to 16 tokens on a routed key runs as paired 8-token blocks (the 8-token kernel and split, unstaged);
        // routed by the same key, so a slice under the split pairs as its whole weight does on one card
        const int64_t kn = qpn_key_n(src0), kk = qpn_key_k(src0);
        qpn_cfg c = T > 8 ? ggml_cuda_qpn_config16(src0->type, kn, kk) : ggml_cuda_qpn_config(src0->type, kn, kk, T);
        int nh = 1;
        // not at exactly 16 tokens, where the kernel at T = 16 runs full (card 1, the slices, us: Q4_K down 85.2 paired ->
        // 73.0, Q4_K GDN out 28.2 -> 21.8)
        if (T > 8 && T <= 16 && ggml_cuda_qpn_pair16(src0->type, kn, kk) && !(T == 16 && ggml_cuda_qpn_full_on())) {
            const qpn_cfg c8 = ggml_cuda_qpn_config(src0->type, kn, kk, 8);
            if (!c8.stage && 2*ntiles <= QPN_MAX_TILES) {
                c      = c8;
                c.mode = 3;
                nh     = 2;
            }
        }
        // the lookahead of 4 units (MODE 5) for the single products of blocks of at most 4 warps (the split is the key's, so the
        // outputs are bit-identical)
        if (T >= 5 && T <= 16 && nh == 1 && !c.stage && c.S*c.R <= 4 && ggml_cuda_qpn_deep(src0->type, T)) {
            c.mode = 5;
        }
        // [TAG_QPN_W32] 17 to 32 tokens: the 16-token split, unstaged, one block per SM
        if (T > 16) {
            c.stage = false;
            c.mode  = 1;
        }
        c.P = std::max(1, std::min(c.P, nsb));
        const dim3 grid(nh*((ntiles + c.R - 1)/c.R), c.P), block(WARP_SIZE, c.S*c.R);
        const size_t smem = c.stage ? ggml_cuda_qpn_stage_bytes(nsb, c.P, T) : 0;
        const std::string key = qpn_log_launch("mul", { src0 }, T, grid, block, c.mode, c.stage);
        const int NL = T > 16 ? 32 : T > 8 ? 16 : T > 4 ? 8 : 4;

        size_t sz_xh, sz_xs, sz_xsc;
        ggml_cuda_qpn_prep_sizes(K, T, sz_xh, sz_xs, sz_xsc);
        const size_t sz_ws  = c.P > 1 ? GGML_PAD((size_t) ntiles*c.P*NL*WARP_SIZE*sizeof(float), 256) : 0;
        // a single pass may share its prepared input with the other products reading src1 (it then
        // always holds the per-32 sums, which only the K-quants with mins read)
        bool   ready  = false;
        char * shared = ncols <= pw ? ggml_cuda_qpn_share_buffer(src0, src1, sz_xh + sz_xs + sz_xsc, stream, &ready) : nullptr;
        const size_t sz_scratch = (shared ? 0 : sz_xh + sz_xs + sz_xsc) + sz_ws;
        ggml_cuda_pool_alloc<char> scratch(ctx.pool());
        if (sz_scratch > 0) {
            scratch.alloc(sz_scratch);
        }
        char * prep = shared ? shared : scratch.get();
        qpn_args a;
        a.xh  = (half  *) prep;
        a.xs  = mins ? (half *) (prep + sz_xh) : nullptr;
        a.xsc = (float *) (prep + sz_xh + sz_xs);
        a.ws  = c.P > 1 ? (float *) (scratch.get() + (shared ? 0 : sz_xh + sz_xs + sz_xsc)) : nullptr;
        a.cnt = nullptr;
        if (c.P > 1) {
            GGML_ASSERT(qpn_counters[ctx.device] != nullptr && ntiles <= QPN_MAX_TILES && ctx.curr_stream_no < GGML_CUDA_MAX_STREAMS);
            a.cnt = qpn_counters[ctx.device] + (size_t) ctx.curr_stream_no*QPN_MAX_TILES;
        }
        a.W = (const char *) src0->data; a.nsb = nsb; a.ntiles = ntiles; a.S = c.S; a.P = c.P;
        a.x = (const float *) src1->data + col0*stride_col_x; a.stride_col_x = stride_col_x;
        a.dst = (float *) dst->data + col0*stride_col_dst;    a.stride_col_dst = stride_col_dst;
        a.nh  = nh;
        if (pushing) {
            a.push = push;
            ggml_cuda_p2p_push_advance(a.push, col0*stride_col_dst);
        }

        if (!ready) {
            const int nwarps = nsb*T;
            qpn_prep_kernel<<<(nwarps + 7)/8, 8*WARP_SIZE, 0, stream>>>(a.x, stride_col_x, nsb, T, a.xh,
                shared ? (half *) (prep + sz_xh) : a.xs, a.xsc);
            CUDA_CHECK(cudaGetLastError());
        }
        // a check (off by default): LLAMA_QPN_FALLBACK_ALL marks every slice non-finite, so every column goes
        // through the fp32 fallback (qpn_fallback), whose dequantization is then compared with ggml's by the caller
        static const bool fallback_all = getenv("LLAMA_QPN_FALLBACK_ALL") != nullptr;
        if (fallback_all) {
            CUDA_CHECK(cudaMemsetAsync(a.xsc, 0, sz_xsc, stream));
        }

        auto launch = [&]() {
            switch (src0->type) {
                case GGML_TYPE_Q4_K: return ggml_cuda_qpn_launch<GGML_TYPE_Q4_K>(T, c.stage, c.mode, grid, block, smem, stream, a);
                case GGML_TYPE_Q5_K: return ggml_cuda_qpn_launch<GGML_TYPE_Q5_K>(T, c.stage, c.mode, grid, block, smem, stream, a);
                case GGML_TYPE_Q6_K: return ggml_cuda_qpn_launch<GGML_TYPE_Q6_K>(T, c.stage, c.mode, grid, block, smem, stream, a);
                case GGML_TYPE_IQ4_XS: return ggml_cuda_qpn_launch<GGML_TYPE_IQ4_XS>(T, c.stage, c.mode, grid, block, smem, stream, a);
                case GGML_TYPE_IQ4_NL: return ggml_cuda_qpn_launch<GGML_TYPE_IQ4_NL>(T, c.stage, c.mode, grid, block, smem, stream, a);
                case GGML_TYPE_Q3_K:   return ggml_cuda_qpn_launch<GGML_TYPE_Q3_K>(T, c.stage, c.mode, grid, block, smem, stream, a);
                case GGML_TYPE_Q8_0:   return ggml_cuda_qpn_launch<GGML_TYPE_Q8_0>(T, c.stage, c.mode, grid, block, smem, stream, a);
                default: GGML_ABORT("fatal error");
            }
        };
        const bool pushed = launch();
        if (!pushing) {
            qpn_bench(key, ggml_nbytes(src0), stream, launch);
        }
        if (pushed) {
            ctx.p2p_pushed = ctx.p2p_push_ar;
        }
        CUDA_CHECK(cudaGetLastError());
        col0 += T;
    }
}

// ------------------------------------------------------------------------------------------------------------------
// two products of one input in one launch (ggml-cuda.cu plans the pairs)


// the pair's split, shared by both products (one block shape): S, R and staging; P = 1. Per (N0 + N1, K), the best of a
// sweep at 4 tokens; LLAMA_QPN_GROUP_CFG="S,R,stage" overrides it, for the sweep (at 5 to 8 tokens "S,R,mode").
struct qpn_group_shape_cfg { int64_t n0, n1, k; int S, R, stage; };
static const qpn_group_shape_cfg qpn_group_cfgs[] = {
    { 12288,  1024, 5120, 4, 2, 0 }, // attn q + gate with k, when v is not repacked (q alone: 4,2, unstaged)
    { 10240,  6144, 5120, 3, 2, 0 }, // GDN qkv + z: Q5_K 81.7 -> 79.7 us (staged 2,4: 77.0), Q4_K 65.9 -> 61.6 (staged 63.4); no staging
    { 17408, 17408, 5120, 2, 4, 1 }, // ffn up + gate, staged as each alone (so bit-identical to two launches): Q5_K + Q4_K 137.4 -> 132.2
};                                   //   (unstaged at 3,2: 134.3, so staging still wins)

// at 5 to 8 tokens, the split of each product alone at those widths (qpn_cfgs_0095), so a pair is bit-identical to
// two launches; stage = 0 always; mode = blocks of 8 warps per SM (2: 128 registers, 3: 80), as the products alone
static const qpn_group_shape_cfg qpn_group_cfgs_0095[] = {
    { 10240,  6144, 5120, 4, 1, 2 }, // GDN qkv + z (alone: 4,1 at mode 3 and 4,1 at mode 2)
    { 17408, 17408, 5120, 2, 4, 2 }, // ffn up + gate (alone: 2,4 at mode 2)
    { 12288,  1024, 5120, 4, 2, 3 }, // attn q + gate with k, when v is not repacked (q alone: 4,2 at mode 3)
};

static bool ggml_cuda_qpn_group_config(const int64_t n0, const int64_t n1, const int64_t k, const int T, int & S, int & R, int & stage) {
    bool found = false;
    const qpn_group_shape_cfg * tab = T > 4 ? qpn_group_cfgs_0095 : qpn_group_cfgs;
    const size_t ntab = T > 4 ? sizeof(qpn_group_cfgs_0095)/sizeof(qpn_group_cfgs_0095[0]) : sizeof(qpn_group_cfgs)/sizeof(qpn_group_cfgs[0]);
    for (size_t i = 0; i < ntab; ++i) {
        const qpn_group_shape_cfg & c = tab[i];
        if (((c.n0 == n0 && c.n1 == n1) || (c.n0 == n1 && c.n1 == n0)) && c.k == k) {
            S = c.S; R = c.R; stage = c.stage; // at T > 4: stage holds the mode
            found = true;
        }
    }
    if (const char * e = getenv("LLAMA_QPN_GROUP_CFG")) {
        sscanf(e, "%d,%d,%d", &S, &R, &stage);
        found = true;
    }
    if (!found) {
        return false;
    }
    S = std::max(1, std::min(S, QPN_MAX_WARPS));
    R = std::max(1, std::min(R, QPN_MAX_WARPS / S));
    return true;
}

// the draft step's (one token) q + gate with k, or with v: Q6_K, 12,288 and 1,024 rows
static bool ggml_cuda_qpn_draft_pair_ok(const ggml_tensor * a, const ggml_tensor * b, const ggml_tensor * src1) {
    const ggml_tensor * q = qpn_key_n(a) > qpn_key_n(b) ? a : b, * kv = qpn_key_n(a) > qpn_key_n(b) ? b : a;
    return src1->ne[1] == 1 && ggml_cuda_qpn_draft_kv_on() && (ggml_cuda_qpn_wide_mask() & 8) && q->type == GGML_TYPE_Q6_K && kv->type == GGML_TYPE_Q6_K &&
        qpn_key_n(q) == 12288 && qpn_key_n(kv) == 1024;
}

bool ggml_cuda_qpn_group_ok(const ggml_tensor * a, const ggml_tensor * b, const ggml_tensor * src1) {
    static const bool enabled = [] { const char * e = getenv("LLAMA_QPN_GROUP"); return e == nullptr || atoi(e) != 0; }();
    const int T = (int) src1->ne[1];
    int S, R, stage;
    // the GDN pair (in-proj + z) only at 5 to 8 tokens, where both products' own split is the pair's (S = 4, R = 1): at 3 and 4 tokens the
    // pair's split (3, 2) is not either product's own (4, 1 and 8, 1), so it would not be bit-identical to two launches
    if (T < 5 && ((qpn_key_n(a) == 10240 && qpn_key_n(b) == 6144) || (qpn_key_n(a) == 6144 && qpn_key_n(b) == 10240))) {
        return false;
    }
    // LLAMA_QPN_PAIR_SLICE12 (default on): under --split-mode tensor the ffn gate and up slices (68 blocks each, fewer than the
    // SMs) pair at 1 and 2 tokens too (the MTP draft step), at qpn_group_cfgs' split, which is each alone's (so bit-identical to two launches)
    static const bool slice12 = [] { const char * e = getenv("LLAMA_QPN_PAIR_SLICE12"); return e == nullptr || atoi(e) != 0; }();
    // (a draft's slices only, of one type: the pair kernel is instantiated at T 1 and 2 for pairs of one type, and the target's decodes
    // of 1 or 2 tokens stay on two launches, as before)
    const bool pair12 = slice12 && T <= 2 && a->type == b->type && qpn_is_draft_slice(a) && qpn_is_draft_slice(b) && qpn_key_n(a) == 17408 &&
        qpn_key_n(b) == 17408 && qpn_key_k(a) == 5120;
    // and 9 to 16 tokens (one pass of two B tiles), at the pair's 8-token split
    return enabled && (T == 3 || T == 4 || pair12 || (T >= 5 && T <= ggml_cuda_qpn_pass_width() && (ggml_cuda_qpn_wide_mask() & 5) == 5) || ggml_cuda_qpn_draft_pair_ok(a, b, src1)) && ggml_cuda_qpn_is_repacked(a) && ggml_cuda_qpn_is_repacked(b) &&
        a->view_src == nullptr && b->view_src == nullptr && qpn_type_idx(a->type) >= 0 && qpn_type_idx(b->type) >= 0 &&
        a->ne[0] == b->ne[0] && a->ne[0] == src1->ne[0] && qpn_key_k(a) == qpn_key_k(b) &&
        ggml_cuda_qpn_group_config(qpn_key_n(a), qpn_key_n(b), qpn_key_k(a), T, S, R, stage);
}


template <ggml_type t0>
static void ggml_cuda_qpn2_launch_t(const ggml_type t1, const int T, const int stage, const dim3 grid, const dim3 block, const size_t smem,
        cudaStream_t stream, const qpn_seg & g0, const qpn_seg & g1, const int nb0, const int nsb, const int S, const half * xh,
        const half * xs, const float * xsc, const float * x, const int64_t sx) {
    switch (t1) {
        case GGML_TYPE_Q4_K:   ggml_cuda_qpn2_launch_tt<t0, GGML_TYPE_Q4_K  >(T, stage, grid, block, smem, stream, g0, g1, nb0, nsb, S, xh, xs, xsc, x, sx); break;
        case GGML_TYPE_Q5_K:   ggml_cuda_qpn2_launch_tt<t0, GGML_TYPE_Q5_K  >(T, stage, grid, block, smem, stream, g0, g1, nb0, nsb, S, xh, xs, xsc, x, sx); break;
        case GGML_TYPE_Q6_K:   ggml_cuda_qpn2_launch_tt<t0, GGML_TYPE_Q6_K  >(T, stage, grid, block, smem, stream, g0, g1, nb0, nsb, S, xh, xs, xsc, x, sx); break;
        case GGML_TYPE_IQ4_XS: ggml_cuda_qpn2_launch_tt<t0, GGML_TYPE_IQ4_XS>(T, stage, grid, block, smem, stream, g0, g1, nb0, nsb, S, xh, xs, xsc, x, sx); break;
        case GGML_TYPE_IQ4_NL: ggml_cuda_qpn2_launch_tt<t0, GGML_TYPE_IQ4_NL>(T, stage, grid, block, smem, stream, g0, g1, nb0, nsb, S, xh, xs, xsc, x, sx); break;
        case GGML_TYPE_Q3_K:   ggml_cuda_qpn2_launch_tt<t0, GGML_TYPE_Q3_K  >(T, stage, grid, block, smem, stream, g0, g1, nb0, nsb, S, xh, xs, xsc, x, sx); break;
        default: GGML_ABORT("fatal error");
    }
}

void ggml_cuda_mul_mat_qpn2(ggml_backend_cuda_context & ctx, const ggml_tensor * const src0s[2], const ggml_tensor * src1, ggml_tensor * const dsts[2]) {
    GGML_ASSERT(ggml_cuda_qpn_group_ok(src0s[0], src0s[1], src1));
    // the lower type index first (the instantiated order); the outputs follow their weights
    const int  i0  = qpn_type_idx(src0s[0]->type) <= qpn_type_idx(src0s[1]->type) ? 0 : 1;
    const ggml_tensor * w0 = src0s[i0], * w1 = src0s[1 - i0];
    ggml_tensor       * d0 = dsts[i0],  * d1 = dsts[1 - i0];
    for (const ggml_tensor * d : { (const ggml_tensor *) d0, (const ggml_tensor *) d1 }) {
        GGML_ASSERT(d->type == GGML_TYPE_F32 && d->nb[0] == sizeof(float) && d->ne[1] == src1->ne[1]);
    }
    GGML_ASSERT(src1->type == GGML_TYPE_F32 && src1->nb[0] == sizeof(float) && src1->ne[2] == 1 && src1->ne[3] == 1);

    const int64_t K   = w0->ne[0];
    const int     nsb = (int) (K/QK_K);
    const int     T   = (int) src1->ne[1];
    const bool    mins = (w0->type == GGML_TYPE_Q4_K || w0->type == GGML_TYPE_Q5_K ||
                                                         w1->type == GGML_TYPE_Q4_K || w1->type == GGML_TYPE_Q5_K);
    cudaStream_t  stream = ctx.stream();
    int S, R, stage; // at T > 4, stage holds the mode (2 or 3); nothing is staged
    ggml_cuda_qpn_group_config(qpn_key_n(w0), qpn_key_n(w1), qpn_key_k(w0), T, S, R, stage); // by the weights' keys
    if (T <= 4 && ggml_cuda_qpn_stage_bytes(nsb, 1, T) > QPN_STAGE_MAX) {
        stage = 0;
    }
    const int nt0 = (int) (w0->ne[1]/32), nt1 = (int) (w1->ne[1]/32);
    const int nb0 = (nt0 + R - 1)/R, nb1 = (nt1 + R - 1)/R;
    const dim3 grid(nb0 + nb1, 1), block(WARP_SIZE, S*R);
    const size_t smem = T <= 4 && stage ? ggml_cuda_qpn_stage_bytes(nsb, 1, T) : 0;
    const std::string key = qpn_log_launch("mul2", { w0, w1 }, T, grid, block, stage, -1);

    size_t sz_xh, sz_xs, sz_xsc;
    ggml_cuda_qpn_prep_sizes(K, T, sz_xh, sz_xs, sz_xsc);
    bool   ready  = false;
    char * shared = ggml_cuda_qpn_share_buffer(w0, src1, sz_xh + sz_xs + sz_xsc, stream, &ready);
    ggml_cuda_pool_alloc<char> scratch(ctx.pool());
    if (!shared) {
        scratch.alloc(sz_xh + sz_xs + sz_xsc);
    }
    char * prep = shared ? shared : scratch.get();
    half  * xh  = (half *) prep;
    half  * xs  = (half *) (prep + sz_xh);
    float * xsc = (float *) (prep + sz_xh + sz_xs);
    const float * x = (const float *) src1->data;
    const int64_t sx = src1->nb[1]/sizeof(float);
    if (!ready) {
        const int nwarps = nsb*T;
        qpn_prep_kernel<<<(nwarps + 7)/8, 8*WARP_SIZE, 0, stream>>>(x, sx, nsb, T, xh, (shared || mins) ? xs : nullptr, xsc);
        CUDA_CHECK(cudaGetLastError());
    }
    const qpn_seg g0 = { (const char *) w0->data, nt0, (float *) d0->data, (int64_t) (d0->nb[1]/sizeof(float)) };
    const qpn_seg g1 = { (const char *) w1->data, nt1, (float *) d1->data, (int64_t) (d1->nb[1]/sizeof(float)) };
    const half * xsp = mins ? xs : nullptr;
    auto launch = [&]() {
        switch (w0->type) {
            case GGML_TYPE_Q4_K:   ggml_cuda_qpn2_launch_t<GGML_TYPE_Q4_K  >(w1->type, T, stage, grid, block, smem, stream, g0, g1, nb0, nsb, S, xh, xsp, xsc, x, sx); break;
            case GGML_TYPE_Q5_K:   ggml_cuda_qpn2_launch_t<GGML_TYPE_Q5_K  >(w1->type, T, stage, grid, block, smem, stream, g0, g1, nb0, nsb, S, xh, xsp, xsc, x, sx); break;
            case GGML_TYPE_Q6_K:   ggml_cuda_qpn2_launch_t<GGML_TYPE_Q6_K  >(w1->type, T, stage, grid, block, smem, stream, g0, g1, nb0, nsb, S, xh, xsp, xsc, x, sx); break;
            case GGML_TYPE_IQ4_XS: ggml_cuda_qpn2_launch_t<GGML_TYPE_IQ4_XS>(w1->type, T, stage, grid, block, smem, stream, g0, g1, nb0, nsb, S, xh, xsp, xsc, x, sx); break;
            case GGML_TYPE_IQ4_NL: ggml_cuda_qpn2_launch_t<GGML_TYPE_IQ4_NL>(w1->type, T, stage, grid, block, smem, stream, g0, g1, nb0, nsb, S, xh, xsp, xsc, x, sx); break;
            case GGML_TYPE_Q3_K:   ggml_cuda_qpn2_launch_t<GGML_TYPE_Q3_K  >(w1->type, T, stage, grid, block, smem, stream, g0, g1, nb0, nsb, S, xh, xsp, xsc, x, sx); break;
            default: GGML_ABORT("fatal error");
        }
    };
    launch();
    CUDA_CHECK(cudaGetLastError());
    qpn_bench(key, ggml_nbytes(w0) + ggml_nbytes(w1), stream, launch);
}

// ------------------------------------------------------------------------------------------------------------------
// three products of one input in one launch (the attention input q + gate, k, v; ggml-cuda.cu plans them)


// the triple's split: q's own (S, R: qpn_cfgs_0091 at 4 tokens, qpn_cfgs_0095 at 5 to 8), so q's outputs are bit-identical to its own
// launch (IQ4_XS's at 5 to 8 tokens excepted, whose own launch is the whole-superblock kernel at S = 2); k and v at split-K S12 in
// blocks of the same 8 warps; mode = blocks of 8 warps per SM
struct qpn_group3_cfg { int64_t n0, n1, n2, k; int S, R, S12, mode; };
static const qpn_group3_cfg qpn_group3_cfgs[] = {
    { 12288, 1024, 1024, 5120, 4, 2, 8, 3 }, // k, v at split-K 8, one tile a block, first: per layer, against q alone with k and v on dp4a,
};                                           // 79.5 -> 68.4 us at 4 tokens, 99.2 -> 83.5 at 8 (S12 = 4: 81.9, 89.4; 2: 76.4, 85.7)

// the draft step's (one token) q + gate, k, v, all Q6_K: q's split (4, 2), k and v at split-K 8 (qpn_group3_cfgs)
static bool ggml_cuda_qpn_draft3_ok(const ggml_tensor * a, const ggml_tensor * b, const ggml_tensor * c, const ggml_tensor * src1) {
    static const bool enabled = [] { const char * e = getenv("LLAMA_QPN_GROUP"); return e == nullptr || atoi(e) != 0; }();
    if (!enabled || src1->ne[1] != 1 || !ggml_cuda_qpn_draft_kv_on() || !(ggml_cuda_qpn_wide_mask() & 8) ||
            a->type != GGML_TYPE_Q6_K || b->type != GGML_TYPE_Q6_K || c->type != GGML_TYPE_Q6_K ||
            qpn_key_n(a) != 12288 || qpn_key_n(b) != 1024 || qpn_key_n(c) != 1024) {
        return false;
    }
    for (const ggml_tensor * w : { a, b, c }) {
        if (!ggml_cuda_qpn_is_repacked(w) || w->view_src != nullptr || w->ne[0] != src1->ne[0] || qpn_key_k(w) != 5120) {
            return false;
        }
    }
    return true;
}

// the GDN layer's in-proj (a), z (b) and merged alpha/beta (c, Q8_0, 96 rows), 5 to 8 tokens, on the pair's configured shape
// (5 to 16, one pass of two B tiles at 9 to 16)
static constexpr bool qpn_gdn_type_ok(const ggml_type t) {
    return t == GGML_TYPE_Q4_K || t == GGML_TYPE_Q5_K || t == GGML_TYPE_Q6_K || t == GGML_TYPE_IQ4_XS || t == GGML_TYPE_IQ4_NL;
}
static bool ggml_cuda_qpn_gdn3_ok(const ggml_tensor * a, const ggml_tensor * b, const ggml_tensor * c, const ggml_tensor * src1) {
    static const bool enabled = [] { const char * e = getenv("LLAMA_QPN_GROUP"); return e == nullptr || atoi(e) != 0; }();
    const int T = (int) src1->ne[1];
    int S, R, stage;
    if (!enabled || !ggml_cuda_qpn_gdn_ab_on() || T < 5 || T > ggml_cuda_qpn_pass_width() || (ggml_cuda_qpn_wide_mask() & 5) != 5 ||
            c->type != GGML_TYPE_Q8_0 || c->ne[1] != 96 || a->ne[1] != 10240 || b->ne[1] != 6144 ||
            !qpn_gdn_type_ok(a->type) || !qpn_gdn_type_ok(b->type)) {
        return false;
    }
    for (const ggml_tensor * w : { a, b, c }) {
        if (!ggml_cuda_qpn_is_repacked(w) || w->view_src != nullptr || w->ne[0] != src1->ne[0]) {
            return false;
        }
    }
    return ggml_cuda_qpn_group_config(a->ne[1], b->ne[1], a->ne[0], T, S, R, stage) && stage == 2; // the kernel is instantiated at mode 2
}

// LLAMA_QPN_GDN_AB (default on; 0 = none): under --split-mode tensor alpha and beta stay upstream's two Q8_0 weights, 48 x 5,120, each
// split into a 24-row slice a device. A slice keeps its GGUF layout for every reader it had (the dp4a products, the prompt path's fp16
// expansions); at load it also gets a private copy, padded to one 32-row tile with rows 24 to 31 zero and repacked, which only the GDN
// layer's in-proj + z launch reads (ggml_cuda_mul_mat_qpn4ab). The slice's buffer holds exactly its 24 rows, so the padded tile cannot be in
// place. Copies live in 2 MiB device chunks for the process's life (as qpn_counters); written at load only.
// (the entry points ggml-cuda.cu calls, declared there and here)
bool ggml_cuda_qpn_ab_split_on();
bool ggml_cuda_qpn_ab_slice_copy(const ggml_tensor * slice, int device, int64_t n, int64_t k);
bool ggml_cuda_qpn_has_ab_copy(const ggml_tensor * t);
bool ggml_cuda_qpn_gdn4_ok(const ggml_tensor * a, const ggml_tensor * b, const ggml_tensor * c, const ggml_tensor * d, const ggml_tensor * src1);
void ggml_cuda_mul_mat_qpn4ab(ggml_backend_cuda_context & ctx, const ggml_tensor * const src0s[4], const ggml_tensor * src1, ggml_tensor * const dsts[4]);

struct qpn_ab_copy { const char * W; int nrow; };
static std::unordered_map<const void *, qpn_ab_copy> qpn_ab_copies;

bool ggml_cuda_qpn_ab_split_on() {
    return ggml_cuda_qpn_gdn_ab_on();
}

bool ggml_cuda_qpn_ab_slice_copy(const ggml_tensor * slice, const int device, const int64_t n, const int64_t k) {
    if (!ggml_cuda_qpn_gdn_ab_on() || slice->type != GGML_TYPE_Q8_0 || slice->view_src != nullptr || !ggml_is_contiguous(slice) ||
            slice->ne[1] < 1 || slice->ne[1] >= 32 || slice->ne[2] != 1 || slice->ne[3] != 1 || slice->ne[0] % QK_K != 0) {
        return false;
    }
    ggml_tensor pad = *slice; // the padded tile, as the repack sees it
    pad.ne[1]    = 32;
    pad.nb[2]    = pad.nb[3] = 32*pad.nb[1];
    pad.view_src = nullptr;
    pad.flags    = 0;
    if (!ggml_cuda_qpn_takes(&pad, ggml_cuda_info().devices[device].cc)) {
        return false;
    }
    static char * chunk[GGML_CUDA_MAX_DEVICES] = { nullptr };
    static size_t used[GGML_CUDA_MAX_DEVICES]  = { 0 };
    const size_t chunk_size = 2*1024*1024, size = ggml_nbytes(&pad);
    GGML_ASSERT(size <= chunk_size);
    ggml_cuda_set_device(device);
    if (chunk[device] == nullptr || used[device] + size > chunk_size) {
        CUDA_CHECK(cudaMalloc((void **) &chunk[device], chunk_size));
        used[device] = 0;
    }
    pad.data = chunk[device] + used[device];
    used[device] += size;
    CUDA_CHECK(cudaMemset(pad.data, 0, size)); // Q8_0 zero blocks (d = 0): rows 24 to 31 give 0
    CUDA_CHECK(cudaMemcpy(pad.data, slice->data, ggml_nbytes(slice), cudaMemcpyDeviceToDevice));
    ggml_cuda_qpn_repack(&pad, device);
    qpn_ab_copies[slice->data] = { (const char *) pad.data, (int) slice->ne[1] };
    ggml_cuda_qpn_set_slice_key(slice, n, k); // the record of the whole weight's key, read here only (the slice itself is not repacked)
    return true;
}

bool ggml_cuda_qpn_has_ab_copy(const ggml_tensor * t) {
    return !qpn_ab_copies.empty() && t != nullptr && qpn_ab_copies.count(t->data) != 0;
}

// in-proj (a, key 10,240) and z (b, key 6,144) as the one-card triple takes them, with two Q8_0 slices that have a private copy and the
// whole-weight key 48 x 5,120 (c, d: beta and alpha), one input of 5 to 16 tokens
bool ggml_cuda_qpn_gdn4_ok(const ggml_tensor * a, const ggml_tensor * b, const ggml_tensor * c, const ggml_tensor * d, const ggml_tensor * src1) {
    static const bool enabled = [] { const char * e = getenv("LLAMA_QPN_GROUP"); return e == nullptr || atoi(e) != 0; }();
    const int T = (int) src1->ne[1];
    int S, R, stage;
    if (!enabled || !ggml_cuda_qpn_gdn_ab_on() || qpn_ab_copies.empty() || T < 5 || T > ggml_cuda_qpn_pass_width() || (ggml_cuda_qpn_wide_mask() & 5) != 5 ||
            src1->type != GGML_TYPE_F32 || src1->ne[2] != 1 || src1->ne[3] != 1 ||
            qpn_key_n(a) != 10240 || qpn_key_n(b) != 6144 || !qpn_gdn_type_ok(a->type) || !qpn_gdn_type_ok(b->type)) {
        return false;
    }
    for (const ggml_tensor * w : { a, b }) {
        if (!ggml_cuda_qpn_is_repacked(w) || w->view_src != nullptr || w->ne[0] != src1->ne[0] || qpn_key_k(w) != 5120) {
            return false;
        }
    }
    for (const ggml_tensor * w : { c, d }) {
        if (!ggml_cuda_qpn_has_ab_copy(w) || ggml_cuda_qpn_is_repacked(w) || w->view_src != nullptr || w->type != GGML_TYPE_Q8_0 ||
                w->ne[0] != src1->ne[0] || qpn_key_n(w) != 48 || qpn_key_k(w) != 5120) {
            return false;
        }
    }
    return ggml_cuda_qpn_group_config(qpn_key_n(a), qpn_key_n(b), qpn_key_k(a), T, S, R, stage) && stage == 2; // the kernel is instantiated at mode 2
}

bool ggml_cuda_qpn_group3_ok(const ggml_tensor * a, const ggml_tensor * b, const ggml_tensor * c, const ggml_tensor * src1) {
    static const bool enabled = [] { const char * e = getenv("LLAMA_QPN_GROUP"); return e == nullptr || atoi(e) != 0; }();
    const int T = (int) src1->ne[1];
    if (ggml_cuda_qpn_gdn3_ok(a, b, c, src1) || ggml_cuda_qpn_draft3_ok(a, b, c, src1)) {
        return true;
    }
    // and 9 to 16 tokens, at the triple's 8-token split
    if (!enabled || !(ggml_cuda_qpn_wide_mask() & 8) || !(T == 4 || (T >= 5 && T <= ggml_cuda_qpn_pass_width() && (ggml_cuda_qpn_wide_mask() & 1)))) {
        return false;
    }
    for (const ggml_tensor * w : { a, b, c }) {
        if (!ggml_cuda_qpn_is_repacked(w) || w->view_src != nullptr || w->ne[0] != src1->ne[0]) {
            return false;
        }
    }
    if (!qpn3_t0_ok(a->type) || !qpn3_t1_ok(b->type) || !qpn3_t1_ok(c->type)) {
        return false;
    }
    if (qpn_key_k(a) != qpn_key_k(b) || qpn_key_k(a) != qpn_key_k(c)) {
        return false;
    }
    for (const qpn_group3_cfg & g : qpn_group3_cfgs) { // by the weights' keys
        if (g.n0 == qpn_key_n(a) && g.n1 == qpn_key_n(b) && g.n2 == qpn_key_n(c) && g.k == qpn_key_k(a)) {
            return true;
        }
    }
    return false;
}


static void ggml_cuda_mul_mat_qpn2ab(ggml_backend_cuda_context & ctx, const ggml_tensor * const src0s[3], const ggml_tensor * src1, ggml_tensor * const dsts[3]);

void ggml_cuda_mul_mat_qpn3(ggml_backend_cuda_context & ctx, const ggml_tensor * const src0s[3], const ggml_tensor * src1, ggml_tensor * const dsts[3]) {
    GGML_ASSERT(ggml_cuda_qpn_group3_ok(src0s[0], src0s[1], src0s[2], src1));
    if (ggml_cuda_qpn_gdn3_ok(src0s[0], src0s[1], src0s[2], src1)) {
        ggml_cuda_mul_mat_qpn2ab(ctx, src0s, src1, dsts);
        return;
    }
    // products 1 and 2 in type order (the instantiated order); the outputs follow their weights
    const int i1 = qpn3_idx(src0s[1]->type) <= qpn3_idx(src0s[2]->type) ? 1 : 2;
    const ggml_tensor * w[3] = { src0s[0], src0s[i1], src0s[3 - i1] };
    ggml_tensor       * d[3] = { dsts[0],  dsts[i1],  dsts[3 - i1] };
    for (const ggml_tensor * t : d) {
        GGML_ASSERT(t->type == GGML_TYPE_F32 && t->nb[0] == sizeof(float) && t->ne[1] == src1->ne[1]);
    }
    GGML_ASSERT(src1->type == GGML_TYPE_F32 && src1->nb[0] == sizeof(float) && src1->ne[2] == 1 && src1->ne[3] == 1);

    const int64_t K   = w[0]->ne[0];
    const int     nsb = (int) (K/QK_K);
    const int     T   = (int) src1->ne[1];
    bool mins = false;
    for (const ggml_tensor * t : w) {
        mins = mins || t->type == GGML_TYPE_Q4_K || t->type == GGML_TYPE_Q5_K;
    }
    cudaStream_t stream = ctx.stream();
    int S = 4, R = 2, S12 = 4;
    for (const qpn_group3_cfg & g : qpn_group3_cfgs) { // by the weights' keys
        if (g.n0 == qpn_key_n(w[0]) && g.n1 == qpn_key_n(w[1]) && g.n2 == qpn_key_n(w[2]) && g.k == qpn_key_k(w[0])) {
            S = g.S; R = g.R; S12 = g.S12;
        }
    }
    if (const char * e = getenv("LLAMA_QPN_GROUP3_CFG")) { // "S,R,S12", for the sweep
        sscanf(e, "%d,%d,%d", &S, &R, &S12);
    }
    const int R12 = std::max(1, S*R/S12);
    const int nt[3] = { (int) (w[0]->ne[1]/32), (int) (w[1]->ne[1]/32), (int) (w[2]->ne[1]/32) };
    const int nb[3] = { (nt[0] + R - 1)/R, (nt[1] + R12 - 1)/R12, (nt[2] + R12 - 1)/R12 };
    const dim3 grid(nb[0] + nb[1] + nb[2], 1), block(WARP_SIZE, S*R);
    qpn_log_launch("mul3", { w[0], w[1], w[2] }, T, grid, block, -1, -1);

    size_t sz_xh, sz_xs, sz_xsc;
    ggml_cuda_qpn_prep_sizes(K, T, sz_xh, sz_xs, sz_xsc);
    bool   ready  = false;
    char * shared = ggml_cuda_qpn_share_buffer(w[0], src1, sz_xh + sz_xs + sz_xsc, stream, &ready);
    ggml_cuda_pool_alloc<char> scratch(ctx.pool());
    if (!shared) {
        scratch.alloc(sz_xh + sz_xs + sz_xsc);
    }
    char * prep = shared ? shared : scratch.get();
    half  * xh  = (half *) prep;
    half  * xs  = (half *) (prep + sz_xh);
    float * xsc = (float *) (prep + sz_xh + sz_xs);
    const float * x = (const float *) src1->data;
    const int64_t sx = src1->nb[1]/sizeof(float);
    if (!ready) {
        const int nwarps = nsb*T;
        qpn_prep_kernel<<<(nwarps + 7)/8, 8*WARP_SIZE, 0, stream>>>(x, sx, nsb, T, xh, (shared || mins) ? xs : nullptr, xsc);
        CUDA_CHECK(cudaGetLastError());
    }
    qpn_seg g[3];
    for (int i = 0; i < 3; ++i) {
        g[i] = { (const char *) w[i]->data, nt[i], (float *) d[i]->data, (int64_t) (d[i]->nb[1]/sizeof(float)) };
    }
    const half * xsp = mins ? xs : nullptr;
    switch (w[0]->type) {
        case GGML_TYPE_Q6_K: // the draft step's Q6_K triple (ggml_cuda_qpn_draft3_ok), one token
            qpn_mul3_kernel<GGML_TYPE_Q6_K, GGML_TYPE_Q6_K, GGML_TYPE_Q6_K, 1, 3><<<grid, block, 0, stream>>>(g[0], g[1], g[2], nb[0], nb[1], nb[2], nsb, S, S12, xh, xsp, xsc, x, sx, T);
            break;
        case GGML_TYPE_Q4_K:   ggml_cuda_qpn3_launch_t<GGML_TYPE_Q4_K  >(w[1]->type, w[2]->type, T, grid, block, stream, g[0], g[1], g[2], nb[0], nb[1], nb[2], nsb, S, S12, xh, xsp, xsc, x, sx); break;
        case GGML_TYPE_Q5_K:   ggml_cuda_qpn3_launch_t<GGML_TYPE_Q5_K  >(w[1]->type, w[2]->type, T, grid, block, stream, g[0], g[1], g[2], nb[0], nb[1], nb[2], nsb, S, S12, xh, xsp, xsc, x, sx); break;
        case GGML_TYPE_IQ4_XS: ggml_cuda_qpn3_launch_t<GGML_TYPE_IQ4_XS>(w[1]->type, w[2]->type, T, grid, block, stream, g[0], g[1], g[2], nb[0], nb[1], nb[2], nsb, S, S12, xh, xsp, xsc, x, sx); break;
        default: GGML_ABORT("fatal error");
    }
    CUDA_CHECK(cudaGetLastError());
}


static void ggml_cuda_mul_mat_qpn2ab(ggml_backend_cuda_context & ctx, const ggml_tensor * const src0s[3], const ggml_tensor * src1, ggml_tensor * const dsts[3]) {
    // src0s = { in-proj, z, ab }; the pair in type order (the instantiated order), the outputs following their weights
    const int  i0  = qpn_type_idx(src0s[0]->type) <= qpn_type_idx(src0s[1]->type) ? 0 : 1;
    const ggml_tensor * w0 = src0s[i0], * w1 = src0s[1 - i0], * wab = src0s[2];
    ggml_tensor       * d0 = dsts[i0],  * d1 = dsts[1 - i0],  * dab = dsts[2];
    for (const ggml_tensor * d : { (const ggml_tensor *) d0, (const ggml_tensor *) d1, (const ggml_tensor *) dab }) {
        GGML_ASSERT(d->type == GGML_TYPE_F32 && d->nb[0] == sizeof(float) && d->ne[1] == src1->ne[1]);
    }
    GGML_ASSERT(src1->type == GGML_TYPE_F32 && src1->nb[0] == sizeof(float) && src1->ne[2] == 1 && src1->ne[3] == 1);

    const int64_t K   = w0->ne[0];
    const int     nsb = (int) (K/QK_K);
    const int     T   = (int) src1->ne[1];
    const bool    mins = (w0->type == GGML_TYPE_Q4_K || w0->type == GGML_TYPE_Q5_K || w1->type == GGML_TYPE_Q4_K || w1->type == GGML_TYPE_Q5_K);
    cudaStream_t  stream = ctx.stream();
    int S, R, stage;
    ggml_cuda_qpn_group_config(w0->ne[1], w1->ne[1], K, T, S, R, stage);
    const int nt0 = (int) (w0->ne[1]/32), nt1 = (int) (w1->ne[1]/32), ntab = (int) (wab->ne[1]/32);
    const int nb0 = (nt0 + R - 1)/R, nb1 = (nt1 + R - 1)/R, nbab = (ntab + R - 1)/R;
    const dim3 grid(nbab + nb0 + nb1, 1), block(WARP_SIZE, S*R);
    qpn_log_launch("mul2ab", { w0, w1, wab }, T, grid, block, stage, -1);

    size_t sz_xh, sz_xs, sz_xsc;
    ggml_cuda_qpn_prep_sizes(K, T, sz_xh, sz_xs, sz_xsc);
    bool   ready  = false;
    char * shared = ggml_cuda_qpn_share_buffer(w0, src1, sz_xh + sz_xs + sz_xsc, stream, &ready);
    ggml_cuda_pool_alloc<char> scratch(ctx.pool());
    if (!shared) {
        scratch.alloc(sz_xh + sz_xs + sz_xsc);
    }
    char * prep = shared ? shared : scratch.get();
    half  * xh  = (half *) prep;
    half  * xs  = (half *) (prep + sz_xh);
    float * xsc = (float *) (prep + sz_xh + sz_xs);
    const float * x = (const float *) src1->data;
    const int64_t sx = src1->nb[1]/sizeof(float);
    if (!ready) {
        const int nwarps = nsb*T;
        qpn_prep_kernel<<<(nwarps + 7)/8, 8*WARP_SIZE, 0, stream>>>(x, sx, nsb, T, xh, (shared || mins) ? xs : nullptr, xsc);
        CUDA_CHECK(cudaGetLastError());
    }
    const qpn_seg g0  = { (const char *) w0->data,  nt0, (float *) d0->data,  (int64_t) (d0->nb[1]/sizeof(float)) };
    const qpn_seg g1  = { (const char *) w1->data,  nt1, (float *) d1->data,  (int64_t) (d1->nb[1]/sizeof(float)) };
    const qpn_seg gab = { (const char *) wab->data, ntab, (float *) dab->data, (int64_t) (dab->nb[1]/sizeof(float)) };
    const half * xsp = mins ? xs : nullptr;
    switch (w0->type) {
        case GGML_TYPE_Q4_K:   ggml_cuda_qpn2ab_launch_t<GGML_TYPE_Q4_K  >(w1->type, grid, block, stream, gab, g0, g1, nbab, nb0, nsb, S, xh, xsp, xsc, x, sx, T); break;
        case GGML_TYPE_Q5_K:   ggml_cuda_qpn2ab_launch_t<GGML_TYPE_Q5_K  >(w1->type, grid, block, stream, gab, g0, g1, nbab, nb0, nsb, S, xh, xsp, xsc, x, sx, T); break;
        case GGML_TYPE_Q6_K:   ggml_cuda_qpn2ab_launch_t<GGML_TYPE_Q6_K  >(w1->type, grid, block, stream, gab, g0, g1, nbab, nb0, nsb, S, xh, xsp, xsc, x, sx, T); break;
        case GGML_TYPE_IQ4_XS: ggml_cuda_qpn2ab_launch_t<GGML_TYPE_IQ4_XS>(w1->type, grid, block, stream, gab, g0, g1, nbab, nb0, nsb, S, xh, xsp, xsc, x, sx, T); break;
        case GGML_TYPE_IQ4_NL: ggml_cuda_qpn2ab_launch_t<GGML_TYPE_IQ4_NL>(w1->type, grid, block, stream, gab, g0, g1, nbab, nb0, nsb, S, xh, xsp, xsc, x, sx, T); break;
        default: GGML_ABORT("fatal error");
    }
    CUDA_CHECK(cudaGetLastError());
}


// src0s = { in-proj, z, beta, alpha }: beta's and alpha's padded repacked private copies as the two one-tile Q8_0 members, each storing its
// slice's rows into its own output; in-proj and z as ggml_cuda_mul_mat_qpn2ab runs them, at the split of their whole weights' keys
void ggml_cuda_mul_mat_qpn4ab(ggml_backend_cuda_context & ctx, const ggml_tensor * const src0s[4], const ggml_tensor * src1, ggml_tensor * const dsts[4]) {
    GGML_ASSERT(ggml_cuda_qpn_gdn4_ok(src0s[0], src0s[1], src0s[2], src0s[3], src1));
    // the pair in type order (the instantiated order), the outputs following their weights
    const int  i0  = qpn_type_idx(src0s[0]->type) <= qpn_type_idx(src0s[1]->type) ? 0 : 1;
    const ggml_tensor * w0 = src0s[i0], * w1 = src0s[1 - i0];
    ggml_tensor       * d0 = dsts[i0],  * d1 = dsts[1 - i0];
    for (const ggml_tensor * d : { (const ggml_tensor *) d0, (const ggml_tensor *) d1 }) {
        GGML_ASSERT(d->type == GGML_TYPE_F32 && d->nb[0] == sizeof(float) && d->ne[1] == src1->ne[1]);
    }
    GGML_ASSERT(src1->type == GGML_TYPE_F32 && src1->nb[0] == sizeof(float) && src1->ne[2] == 1 && src1->ne[3] == 1);

    const int64_t K   = w0->ne[0];
    const int     nsb = (int) (K/QK_K);
    const int     T   = (int) src1->ne[1];
    const bool    mins = (w0->type == GGML_TYPE_Q4_K || w0->type == GGML_TYPE_Q5_K || w1->type == GGML_TYPE_Q4_K || w1->type == GGML_TYPE_Q5_K);
    cudaStream_t  stream = ctx.stream();
    int S, R, stage;
    ggml_cuda_qpn_group_config(qpn_key_n(w0), qpn_key_n(w1), qpn_key_k(w0), T, S, R, stage); // by the whole weights' keys
    const int nt0 = (int) (w0->ne[1]/32), nt1 = (int) (w1->ne[1]/32);
    const int nb0 = (nt0 + R - 1)/R, nb1 = (nt1 + R - 1)/R, nbm = (1 + R - 1)/R; // nbm: a one-tile member's blocks
    const dim3 grid(2*nbm + nb0 + nb1, 1), block(WARP_SIZE, S*R);
    qpn_log_launch("mul4ab", { w0, w1 }, T, grid, block, stage, -1);

    size_t sz_xh, sz_xs, sz_xsc;
    ggml_cuda_qpn_prep_sizes(K, T, sz_xh, sz_xs, sz_xsc);
    bool   ready  = false;
    char * shared = ggml_cuda_qpn_share_buffer(w0, src1, sz_xh + sz_xs + sz_xsc, stream, &ready);
    ggml_cuda_pool_alloc<char> scratch(ctx.pool());
    if (!shared) {
        scratch.alloc(sz_xh + sz_xs + sz_xsc);
    }
    char * prep = shared ? shared : scratch.get();
    half  * xh  = (half *) prep;
    half  * xs  = (half *) (prep + sz_xh);
    float * xsc = (float *) (prep + sz_xh + sz_xs);
    const float * x = (const float *) src1->data;
    const int64_t sx = src1->nb[1]/sizeof(float);
    if (!ready) {
        const int nwarps = nsb*T;
        qpn_prep_kernel<<<(nwarps + 7)/8, 8*WARP_SIZE, 0, stream>>>(x, sx, nsb, T, xh, (shared || mins) ? xs : nullptr, xsc);
        CUDA_CHECK(cudaGetLastError());
    }
    const qpn_seg g0 = { (const char *) w0->data, nt0, (float *) d0->data, (int64_t) (d0->nb[1]/sizeof(float)) };
    const qpn_seg g1 = { (const char *) w1->data, nt1, (float *) d1->data, (int64_t) (d1->nb[1]/sizeof(float)) };
    qpn_seg gm[2];
    int     nrow[2];
    for (int i = 0; i < 2; ++i) { // beta, alpha
        const ggml_tensor * w = src0s[2 + i];
        ggml_tensor       * d = dsts[2 + i];
        GGML_ASSERT(d->type == GGML_TYPE_F32 && d->nb[0] == sizeof(float) && d->ne[0] == w->ne[1] && d->ne[1] == src1->ne[1]);
        const qpn_ab_copy & c = qpn_ab_copies.at(w->data);
        gm[i]   = { c.W, 1, (float *) d->data, (int64_t) (d->nb[1]/sizeof(float)) };
        nrow[i] = c.nrow;
    }
    const half * xsp = mins ? xs : nullptr;
    switch (w0->type) {
        case GGML_TYPE_Q4_K:   ggml_cuda_qpn4ab_launch_t<GGML_TYPE_Q4_K  >(w1->type, grid, block, stream, gm[0], gm[1], nrow[0], nrow[1], g0, g1, nbm, nbm, nb0, nsb, S, xh, xsp, xsc, x, sx, T); break;
        case GGML_TYPE_Q5_K:   ggml_cuda_qpn4ab_launch_t<GGML_TYPE_Q5_K  >(w1->type, grid, block, stream, gm[0], gm[1], nrow[0], nrow[1], g0, g1, nbm, nbm, nb0, nsb, S, xh, xsp, xsc, x, sx, T); break;
        case GGML_TYPE_Q6_K:   ggml_cuda_qpn4ab_launch_t<GGML_TYPE_Q6_K  >(w1->type, grid, block, stream, gm[0], gm[1], nrow[0], nrow[1], g0, g1, nbm, nbm, nb0, nsb, S, xh, xsp, xsc, x, sx, T); break;
        case GGML_TYPE_IQ4_XS: ggml_cuda_qpn4ab_launch_t<GGML_TYPE_IQ4_XS>(w1->type, grid, block, stream, gm[0], gm[1], nrow[0], nrow[1], g0, g1, nbm, nbm, nb0, nsb, S, xh, xsp, xsc, x, sx, T); break;
        case GGML_TYPE_IQ4_NL: ggml_cuda_qpn4ab_launch_t<GGML_TYPE_IQ4_NL>(w1->type, grid, block, stream, gm[0], gm[1], nrow[0], nrow[1], g0, g1, nbm, nbm, nb0, nsb, S, xh, xsp, xsc, x, sx, T); break;
        default: GGML_ABORT("fatal error");
    }
    CUDA_CHECK(cudaGetLastError());
}
