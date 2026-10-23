#pragma once

#include "llama-batch.h"
#include "llama-graph.h"
#include "llama-memory.h"

#include <functional>
#include <map>
#include <set>
#include <vector>

//
// llama_memory_recurrent
//

// TODO: extract the cache state used for graph computation into llama_memory_recurrent_context_i
//       see the implementation of llama_kv_cache_context_i for an example how to do it
class llama_memory_recurrent : public llama_memory_i {
public:
    llama_memory_recurrent(
            const llama_model & model,
                    ggml_type   type_r,
                    ggml_type   type_s,
                         bool   offload,
                     uint32_t   mem_size,
                     uint32_t   n_seq_max,
                     uint32_t   n_rs_seq,
        const layer_filter_cb & filter);

    ~llama_memory_recurrent() = default;

    //
    // llama_memory_i
    //

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    bool prepare(const std::vector<llama_ubatch> & ubatches);

    // find a contiguous slot of memory cells and emplace the ubatch there
    bool find_slot(const llama_ubatch & ubatch);

    bool get_can_shift() const override;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) override;

    uint32_t head = 0; // the location where the batch will be placed in the cache (see find_slot())
    uint32_t size = 0; // total number of cells, shared across all sequences
    uint32_t used = 0; // used cells (i.e. at least one seq_id)

    // number of recurrent-state snapshots per seq for rollback; tensors are widened to (1 + n_rs_seq) groups
    uint32_t n_rs_seq = 0;

    // [TAG_RS_POOL] LLAMA_RS_POOL=N: the rollback snapshots of all sequences share N rows after the live rows (rows 0 .. size-1 are
    // the cells' live states as before), instead of (1 + n_rs_seq) rows a cell. Each sequence holds rs_depth[seq] <= n_rs_seq
    // snapshots at rows size + rs_snapbase[seq] + 0 .. depth-1 (snapshot j at + j - 1), the sequences packed in seq order, so
    // sequences of equal depth in consecutive cells have their slot-j rows at a uniform stride of depth rows (the fused graph
    // write). set_rs_depth re-packs the bases, which only moves rows that hold no valid snapshot (they are consumed within a
    // round), never a live row. 0: the uniform layout
    uint32_t rs_pool = 0;
    std::vector<uint32_t> rs_depth;    // per seq
    std::vector<uint32_t> rs_snapbase; // per seq, rows from the pool's start
    std::vector<uint8_t>  rs_lock;     // per seq: a prompt job holds its rows (tu_capture), no re-packing past it

    // every sequence's wanted depth at once (the plan: see set_rs_depths), or one sequence's; the depth granted is at most
    // n_rs_seq and what the pool's free rows hold, and a prompt job's sequence keeps its run until the job ends
    void     set_rs_depths(const std::vector<uint32_t> & want);
    uint32_t set_rs_depth(llama_seq_id seq_id, uint32_t depth);
    // a rollback snapshot still to be read goes to the live row now (the caller synchronized the device)
    void     rs_apply_pending(llama_seq_id seq_id);
    // the rollback depth of a sequence (n_rs_seq in the uniform layout)
    uint32_t get_rs_depth(llama_seq_id seq_id) const;
    // the row of snapshot idx of sequence seq whose live state is in cell (idx 0: that live row)
    uint32_t rs_row(uint32_t idx, llama_seq_id seq, uint32_t cell) const;
    uint32_t n_seq_max_rs() const { return n_seq_max; }

    // per-seq rollback index
    std::vector<uint32_t> rs_idx;

    // (LLAMA_TU_DIAG) per seq: the rollback index that the first s_copy after the last reset consumed, -1 if none
    std::vector<int32_t> rs_idx_seen;

    // a hash of what a prompt job's sequence holds in this memory (its tail cell and pending rollback index), which
    // must not change between the groups of one of its ubatches
    uint64_t tu_seq_hash(llama_seq_id seq_id) const;

    void set_rs_idx(llama_seq_id seq_id, uint32_t idx);

    // [TAG_CKPT_TAIL_SNAP] until rs_view_end, the state writes of seq_id read its state n_back tokens back: a rollback snapshot of
    // the last ubatch (and the position its metadata says), the live state unchanged. False when there is none: no snapshots, a
    // rollback pending, or a last ubatch of at most n_back tokens of the sequence
    bool rs_view_back(llama_seq_id seq_id, int32_t n_back);
    void rs_view_end (llama_seq_id seq_id);
    std::vector<llama_pos> rs_view_pos; // the live positions under a view (-1: none)

    // [TAG_SPEC_PIPELINE] a second group of (1 + n_rs_seq) snapshot rows per cell (0 when off, LLAMA_SPEC_PIPELINE),
    // holding a copy of the previous decode's snapshots: a pipelined decode overwrites them while the decode before
    // it is not yet verified, and seq_rm can then roll back into that decode from the copy
    uint32_t n_rs_bak = 0;

    struct rs_bak_info {
        bool      valid = false;
        int32_t   cell  = -1;
        llama_pos end   = -1; // the cell's last position when the copy was taken
        uint32_t  n     = 0;  // the number of valid snapshots in the copy
    };

    std::vector<rs_bak_info> rs_bak; // per seq

    // copy the seq's snapshot rows to the backup rows, with copy(src, dst) on the device that holds them
    void rs_backup(llama_seq_id seq_id, const std::function<void(ggml_tensor *, ggml_tensor *)> & copy);

    // computed before each graph build
    uint32_t n = 0;

    // first zero-ed state
    int32_t rs_z = -1;

    // TODO: optimize for recurrent state needs
    struct mem_cell {
        llama_pos pos  = -1;
        int32_t   src  = -1; // used to know where states should be copied from
        int32_t   src0 = -1; // like src, but only used when setting the inputs (allowing to copy once)
        int32_t   tail = -1;

        uint32_t  n_last = 0; // [TAG_SPEC_PIPELINE] the number of tokens of the last ubatch placed in this cell

        std::set<llama_seq_id> seq_id;

        bool has_seq_id(const llama_seq_id & id) const {
            return seq_id.find(id) != seq_id.end();
        }

        bool is_empty() const {
            return seq_id.empty();
        }

        bool is_same_seq(const mem_cell & other) const {
            return seq_id == other.seq_id;
        }
    };

    std::vector<mem_cell> cells;

    // per layer
    std::vector<ggml_tensor *> r_l;
    std::vector<ggml_tensor *> s_l;
    // a second conv history that must stay replicated across devices, so it cannot share the r row
    std::vector<ggml_tensor *> p_l;

private:
    //const llama_model & model;
    const llama_hparams & hparams;

    const uint32_t n_seq_max = 1;

    // ggml contexts for the KV cache along with the allocated backend buffers:
    std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_ptr>> ctxs_bufs;

    ggml_context_ptr bak_view_ctx; // [TAG_SPEC_PIPELINE] views for rs_backup

    size_t total_size() const;

    size_t size_r_bytes() const;
    size_t size_s_bytes() const;
    size_t size_p_bytes() const;

    void state_write_meta(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges, llama_seq_id seq_id = -1) const;
    void state_write_data(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges) const;

    bool state_read_meta(llama_io_read_i & io, uint32_t cell_count, llama_seq_id dest_seq_id = -1);
    bool state_read_data(llama_io_read_i & io, uint32_t cell_count);
};

class llama_memory_recurrent_context : public llama_memory_context_i {
public:
    // used for errors
    llama_memory_recurrent_context(llama_memory_status status);

    // used to create a full-cache or update context
    llama_memory_recurrent_context(
            llama_memory_recurrent * mem);

    // used to create a batch processing context from a batch
    llama_memory_recurrent_context(
            llama_memory_recurrent * mem,
            std::vector<llama_ubatch> ubatches);

    virtual ~llama_memory_recurrent_context();

    //
    // llama_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    llama_memory_status  get_status() const override;
    const llama_ubatch & get_ubatch() const override;

    //
    // llama_memory_recurrent_context specific API
    //

    uint32_t get_n_rs() const;
    uint32_t get_head() const;
    int32_t  get_rs_z() const;
    uint32_t get_size() const;

    ggml_tensor * get_r_l(int32_t il) const;
    ggml_tensor * get_s_l(int32_t il) const;
    ggml_tensor * get_p_l(int32_t il) const;

    int32_t s_copy(int i) const;

    // [TAG_RS_POOL] the rollback slots this ubatch's graph writes (1 + the smallest depth of its sequences; 1 + n_rs_seq in the
    // uniform layout), and where: rs_snap_geom gives the row of sequence 0's slot 1 and the stride in rows between the
    // sequences' slot-1 rows when the slots form one strided view (true), rs_snap_row the row of ubatch state i's slot j
    uint32_t get_rs_pool() const;
    uint32_t get_n_snap() const;
    bool     rs_snap_geom(uint32_t & row0, uint32_t & seq_rows) const;
    int32_t  rs_snap_row(int i, int j) const;

    // commit once, restore only: after apply(), the bindings that this ubatch's graphs read from the cache (head, n,
    // rs_z and the copy sources, whose read consumes the rollback index) are captured once; from then on the getters and s_copy
    // return the capture and no longer read the live cache. rs_idx_out: the rollback index consumed for seq_id (-1: none)
    void tu_capture(llama_seq_id seq_id, int32_t * rs_idx_out);
    void tu_release();

private:
    struct tu_capture_t {
        bool                 on   = false;
        uint32_t             head = 0;
        uint32_t             n    = 0;
        int32_t              rs_z = -1;
        llama_seq_id         seq  = -1; // the job's sequence, whose pool rows stay locked until tu_release
        std::vector<int32_t> s_copy;
    } cap;

    const llama_memory_status status;

    llama_memory_recurrent * mem;

    size_t i_next = 0;

    std::vector<llama_ubatch> ubatches;

    //
    // data needed for building the compute graph for the current ubatch:
    // TODO: extract all the state like `head` and `n` here
    //

    const bool is_full = false;
};
