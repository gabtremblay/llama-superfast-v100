# llama-superfast-v100-q8-inscora

A fork of [bitexploder/llama-superfast-v100](https://github.com/bitexploder/llama-superfast-v100), the llama.cpp fork
tuned for the Tesla V100 (Volta, `sm_70`), created for **one V100 card**, a single user, and two concurrent slots on a
unified KV cache, to make agentic work palatable while maximizing context size. It adds:

- an 8-bit (`q8_0`) KV cache on upstream's fast Volta attention kernels, so one 32 GB card holds about two thirds more
  context at the same speed;
- a unified KV cache (`--kv-unified`) that runs at the split cache's speed, so two slots share one 245,760-token pool;
- faster prefill on upstream's repacked weights;
- prompt work (DFlash2 features, checkpoints, masks) that stays on the GPU instead of crossing the host link;
- a smaller CUDA graph cache, which freed ~1.2 GB of VRAM that only grew under real traffic.

The model is Qwen3.8-27B (`UD-Q4_K_XL`) with the DFlash2 draft on a V100 PCIe 32 GB. The two slots serve one user: the
main thinking thread, and the side requests an agent harness sends in parallel with it. Our card sits behind a PCIe Gen3 x4
eGPU link (~3.3 GB/s), so anything that crosses the host link is on the critical path; the gains from keeping work on
the GPU are largest on a link like that.

Everything upstream (the tensor-core matvec on repacked K-quants, MTP, the two-user scheduler, two-card tensor
parallel) is unchanged and documented further down, from [Upstream: llama-superfast-v100](#upstream-llama-superfast-v100).

Part of this work was sponsored by [Inscora](https://inscora.com). If you are an MSP or MSSP and want to offer cyber
insurance services to your clients, have a look at [inscora.com](https://inscora.com).

## Results

One V100 PCIe 32 GB, Qwen3.8-27B `UD-Q4_K_XL`, DFlash2 draft, `q8_0` KV, two slots.

**Against upstream's routes** (the same build with this fork's prefill and GPU-residency changes switched off):

| | Upstream routes + q8 KV | This fork |
|---|---|---|
| Cold prefill, 34k-token prompt | 832 tok/s | 892 tok/s |
| Cold prefill, 5.6k-token prompt | 760 tok/s | 851 tok/s |
| Time to first token, new ~125-token chat turn (median) | 828 ms | 680 ms, 430 ms with `--cache-ram 0` |
| Decode, 400 tokens, three prompts | 88 / 90 / 104 tok/s | 88 / 91 / 106 tok/s |
| Context | 2 × 98,304 | one 245,760-token pool (`--kv-unified`), or 2 × 122,880 split |

**BetterBench**, an agentic serving benchmark (combined decode over both slots, inter-token latency, time to first
token, throughput at 8 concurrent requests, prefill of a 64k prompt). Compare rows within a table: the two tables ran
in different sessions and under different load.

On a fresh server:

| Build | Combined decode | Update p99 | Low 1% ITL | TTFT p50 | Aggregate @8 | Prefill @64k |
|---|---|---|---|---|---|---|
| vLLM on the same card, for reference | 49.0 tok/s | 74.1 ms | — | 256 ms | 138.0 tok/s | 772 tok/s |
| MTP, 2 × 118,784 (`5c15ec5`) | 91.1 tok/s | 49.5 ms | 19.6 tok/s | 862 ms | 106.8 tok/s | 755 tok/s |
| DFlash2, 2 × 98,304 (`5c15ec5`) | 105.7 tok/s | 46.8 ms | 20.6 tok/s | 874 ms | 85.6 tok/s | 771 tok/s |
| DFlash2, 2 × 106,496 (`64107c5`) | 109.5 tok/s | 48.9 ms | 19.5 tok/s | 324 ms | 92.6 tok/s | 836 tok/s |

With a 47k-token conversation held in one slot, the common case for an agent server, and a unified cache:

| Build | Combined decode | TTFT p50 | Aggregate @8 | Prefill @64k |
|---|---|---|---|---|
| Unified 212,992, no window (`64107c5`) | 93.8 tok/s | 372 ms | 83.7 tok/s | 531 tok/s |
| + per-batch window (`9a20120`) | 96.1 tok/s | 371 ms | 90.4 tok/s | 683 tok/s |
| + smaller graph cache, 245,760 pool (`0b1a768`) | 96.2 tok/s | 372 ms | 87.6 tok/s | 684 tok/s |
| + prompts placed after their own cells (`263398e`) | 98.7 tok/s | 316 ms | 88.2 tok/s | 837 tok/s |
| **+ decode placed after its own cells (`dd1665f`, current)** | **108.6 tok/s** | **316 ms** | **94.3 tok/s** | 719 tok/s |

The current build peaks at 31,980 MiB of the card's 32,494 over a full run (update p99 45.2 ms, low 1% ITL 21.4 tok/s).
The last step trades some 64k prefill for decode; see [section 6](#6-cell-placement-in-a-unified-cache-263398e-dd1665f).

## Build

Requirements:

- A Tesla V100 (`sm_70`). The changes here are Volta-specific; other GPUs are untested.
- CUDA 12.x (12.8 tested; CUDA 13 dropped Volta), gcc/g++ 14, CMake, Ninja. ccache is optional and recommended.
- The NVIDIA container toolkit, for the Docker route.

**Native:**

```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70-real -DGGML_NATIVE=OFF -DGGML_CUDA_CUB_3DOT2=ON \
  -DGGML_CUDA_FA_ALL_QUANTS=OFF -DGGML_CCACHE=ON \
  -DCMAKE_C_COMPILER=gcc-14 -DCMAKE_CXX_COMPILER=g++-14 -DCMAKE_CUDA_HOST_COMPILER=g++-14 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc
cmake --build build --target llama-server -j 16
```

- `GGML_CUDA_FA_ALL_QUANTS=OFF` still builds the `f16` and `q8_0` KV pairs this setup reads.
- `GGML_CUDA_CUB_3DOT2=ON` fetches CCCL 3.2 at configure time.
- A cold build takes about 15 minutes on 16 threads. The CUTLASS attention units need several GB of RAM per `nvcc`;
  lower `-j` on a machine with less than ~32 GB.
- Add `--target test-backend-ops` (with `-DLLAMA_BUILD_TESTS=ON`) for the kernel tests.

**Docker:**

```bash
DOCKER_BUILDKIT=1 docker build -f .devops/cuda-v100.Dockerfile -t llama-superfast-v100 .devops
```

The image clones this repository at `--build-arg LLAMA_REF=` (default `main`; pin a commit for a reproducible image).
Other build args: `BUILD_JOBS` (default 8), `CUDA_VERSION` (12.8.1). Its compiler cache persists across builds, so
rebuilding after a small change takes a few minutes.

## Models

```bash
hf download unsloth/Qwen3.8-27B-GGUF Qwen3.8-27B-UD-Q4_K_XL.gguf MTP/mtp-Qwen3.8-27B-Q4_0.gguf --local-dir models
hf download z-lab/Qwen3.8-27B-DFlash2-GGUF Qwen3.8-27B-DFlash2-Q4_K_M.gguf --local-dir models
```

Put the files flat in `./models` (move `MTP/mtp-Qwen3.8-27B-Q4_0.gguf` up). DFlash2 needs the MTP file too: its output
head reads that file's rows. The draft vocabulary ranking, `models/draft-vocab-qwen3.8-27b.txt`, ships in this
repository.

## Run

One card, two slots, DFlash2, one 245,760-token pool:

```bash
LLAMA_TU=0 LLAMA_DFLASH2_HEAD_FILE=models/mtp-Qwen3.8-27B-Q4_0.gguf LLAMA_SPEC_DRAFT_VOCAB_FILE=models/draft-vocab-qwen3.8-27b.txt build/bin/llama-server -m models/Qwen3.8-27B-UD-Q4_K_XL.gguf -md models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf --spec-type draft-dflash --spec-draft-n-max 7 --spec-draft-ubatch 512 -c 245760 --parallel 2 --kv-unified -fa on -ngl 99 -ctk q8_0 -ctv q8_0 -ctkd f16 -ctvd f16 -t 12 -b 2048 -ub 2048 --cache-ram 0 --jinja --metrics --temp 1.0 --top-p 0.95 --top-k 20 --host 127.0.0.1 --port 8080
```

The same with the Docker image (models in `$PWD/models`):

```bash
docker run -d --name llama-superfast --init --gpus all -p 8080:8080 -v "$PWD/models:/models:ro" -e LLAMA_TU=0 -e LLAMA_DFLASH2_HEAD_FILE=/models/mtp-Qwen3.8-27B-Q4_0.gguf llama-superfast-v100 --host 0.0.0.0 --port 8080 -m /models/Qwen3.8-27B-UD-Q4_K_XL.gguf -md /models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf --spec-type draft-dflash --spec-draft-n-max 7 --spec-draft-ubatch 512 -c 245760 --parallel 2 --kv-unified -fa on -ngl 99 -ctk q8_0 -ctv q8_0 -ctkd f16 -ctvd f16 -t 12 -b 2048 -ub 2048 --cache-ram 0 --jinja --metrics --temp 1.0 --top-p 0.95 --top-k 20
```

What the settings do:

- `LLAMA_TU=0` is required with DFlash2: the two-user scheduler rejects its long prompts.
- `LLAMA_DFLASH2_HEAD_FILE` points DFlash2's output head at the MTP file's rows.
- `LLAMA_SPEC_DRAFT_VOCAB_FILE` (set by the Docker image) restricts the draft to the top 98,304 ids of the file's
  131,072-id ranking. Without it, decode is ~5–10% slower.
- `--kv-unified` gives both slots one 245,760-token pool: either slot can take most of it (one took a 157k-token prompt
  while the other answered side requests). `--kv-unified-per-slot N` caps each slot inside the pool;
  `--no-kv-unified` gives each slot a fixed half (2 × 122,880) at the same speed.
- `-b 2048` equal to `-ub` keeps the DFlash2 features on the GPU during prefill (section 3).
- `--spec-draft-ubatch 512` sizes the draft's compute buffer for 512-token batches, which frees 1.7 GB.
- `--cache-ram 0` turns off the host prompt cache. Saving a slot's state to host memory costs ~220 ms over a slow link
  each time a slot switches conversations; with one user and pinned slots (`id_slot` in the request) the cache brings
  little. On a fast link, try it on.
- 245,760 tokens peaks at ~31.9 GB. VRAM keeps growing for the first minutes of real traffic, so size a different
  context from a long mixed run, not a fill test (section 5).

Check the startup log for `two-user scheduling (LLAMA_TU): off`, the draft model loading, and
`n_slots = 2, n_ctx_slot = 245760`. Draft acceptance on this target is ~0.5; near zero means the draft did not load.

For MTP instead of DFlash2: `-md models/mtp-Qwen3.8-27B-Q4_0.gguf --spec-type draft-mtp`, without
`LLAMA_DFLASH2_HEAD_FILE`, `LLAMA_TU=0` or `--spec-draft-ubatch`. It decodes slower here but leaves the two-user
scheduler on.

## What changed

### 1. Q8_0 KV cache on the Volta attention paths (`bcf1877`, `5c15ec5`)

Upstream's fast paths, the CUTLASS 6-head prefill attention (`fattn-gqa6`) and the streaming decode/verify kernel
(`fattn-verify-stream`), took only an f16 KV cache. With `-ctk q8_0 -ctv q8_0`, every step fell to the generic kernels
over a temporary f16 copy of the cache.

- Both paths now read `q8_0` directly. The prefill kernel widens key blocks to f16 in shared memory; the verify kernel
  has a `KVQ8` variant with 8-byte row loads and aligned windows.
- Allocations no longer reserve the f16 copy on those routes.
- `test-backend-ops` has `q8_0` cases for both kernels, including the 27B's shapes.
- Switches: `LLAMA_FA_GQA6_Q8=0` and `LLAMA_FATTN_VERIFY_STREAM_Q8=0` fall back to the generic kernels;
  `LLAMA_FATTN_VERIFY_STREAM_Q8_KP` and `LLAMA_FATTN_VERIFY_STREAM_Q8_W` tune the verify kernel.

With sections 3 and 5, this took the server from two slots of 73,728 tokens (f16 cache, MTP) to one 245,760-token
pool (`q8_0`, DFlash2).

### 2. Faster prefill on repacked weights (`044ef3b`)

- Long prompts expand each weight to fp16 for cuBLAS. The expansion now decodes straight from the repacked layout with
  16-byte stores, instead of one warp rebuilding the GGUF blocks while the rest of the block waits. It matches ggml's
  dequantizers bit for bit.
- Products of 129 to 191 tokens go to MMQ on a device-side unrepack instead of fp16 + cuBLAS. The Volta MMQ crossover is
  192 tokens (64 tokens: 0.5 vs 1.25 ms per 27B FFN product; 256 tokens: 1.72 vs 1.50 ms).
- The tensor-core matvec takes up to 128 tokens by default.
- Switches: `LLAMA_QPN_MAX_TOKENS` (default 128, at most 256); `GGML_CUDA_MMQ_MAX_BATCH` overrides the crossover;
  `LLAMA_QPN_MMQ=0` and `LLAMA_QPN_FAST_EXPAND=0` restore the earlier routes; `LLAMA_QPN_EXPAND_CHECK=1` runs the
  earlier kernels alongside and reports any difference.

### 3. Prompt work kept on the GPU (`64107c5`)

- **DFlash2 features stay on the device.** The draft reads the target's layer inputs. For any decode that is one
  ubatch they now stay on the GPU, and the draft copies only the rows its 2,048-token window can see, device to
  device. Before, every prefill ubatch sent all of them to the host and back (about 2.3 GB for a 24k prompt). Verify
  steps with two slots take this path too. Switches: `LLAMA_DFLASH2_FEAT_DEV=0`, `LLAMA_DFLASH2_FEAT_DEV_ROWS`.
- **Checkpoints copy in the background.** A prompt checkpoint (the 27B's recurrent state, ~150 MiB) goes into a GPU
  staging buffer, then to the host on a second stream, so the next compute does not wait for the link. The draft's
  small state is read first. Switch: `LLAMA_CKPT_STAGE=0`.
- **No duplicate checkpoints.** A checkpoint at the position of an existing one is kept, not copied again; before, every
  new conversation re-saved the shared system prompt's checkpoint. Switch: `LLAMA_CKPT_KEEP_SAME=0`.
- **Cheaper prompt-cache saves.** A save reuses evicted entries' buffers, and moves checkpoints instead of copying them
  when the next prompt won't need them.
- **Two-user scheduler.** A lone prompt is no longer cut into small slices when no other slot is waiting
  (`LLAMA_TU_IDLE_JOB_MIN` sets the minimum slice).
- **Smaller draft compute buffer** with `--spec-draft-ubatch 512`: 1.7 GB freed at no measured speed cost.

### 4. Unified KV at full speed (`146f3c4`, `9a20120`)

- **A per-batch window.** With `--kv-unified`, every batch attended over the other slot's cells too, masked out but
  still paid for. A batch of one sequence now reads only the cells from its first to its last (aligned to 256): the
  K/V views and the mask start there, and graph reuse keys on the window. Batches holding both slots skip it (their
  window would span everything, and the scan cost ~4% of two-slot decode). The window is on for the qwen35 line only,
  whose attention uses nothing else. Switches: `LLAMA_KV_WINDOW=0`, `LLAMA_KV_WINDOW_MIN_TOKENS`.
- **Both slots' masks on the GPU.** The device mask builder handled only sequence 0 of a unified cache, so slot 1's
  masks were built on the host and sent over the link every batch. It now keeps one cell mirror per sequence and
  builds mixed batches run by run.
- With the other slot holding 66k tokens: prefill of a 24k prompt 549 → 874 tok/s (split cache: 918), slot 0 decode
  75 / 74 / 96 → 90 / 90 / 110 tok/s (split cache: 88 / 91 / 106). Repeated 40k and 64k prompts alternating between
  loaded slots read at the split cache's speed (858 vs 858, 776 vs 779 tok/s).
- Verified: device masks match the host's bit for bit (`LLAMA_KQ_MASK_DEVICE_CHECK=1`, 711 masks under two-slot load),
  and temperature-0 outputs are identical with the window on and off.

### 5. A smaller CUDA graph cache (`0b1a768`)

- Under real traffic VRAM grew ~2 GB over the first ~25 minutes, which no fill test showed. `GGML_CUDA_MEM_TRACE`
  traced it to the CUDA graph cache: each instantiated graph of the 27B holds 8 MiB, decode graphs are keyed by cache
  length in steps of 256, and the cache kept up to 512 graphs for 600 s.
- The defaults are now 48 graphs for 120 s (`GGML_CUDA_GRAPH_MAX`, `GGML_CUDA_GRAPH_TTL_S`). A conversation moves to the
  next step every ~256 tokens, so old steps are rarely met again. Decode and TTFT are unchanged.
- Growth over a BetterBench run is now +784 MiB (temporary-buffer pools plus the capped cache), which made room for
  32k more tokens of context.
- Diagnostics: `GGML_CUDA_MEM_TRACE=1` logs every drop of free device memory of 8 MiB or more and where it happened
  (`=2` also per node); `GGML_CUDA_POOL_LOG=1` logs each pool growth; `GGML_CUDA_GRAPH_STATS=1` logs the cached graph
  count at the server's debug level.

### 6. Cell placement in a unified cache (`263398e`, `dd1665f`)

The window of section 4 is only as tight as a sequence's cells are contiguous. llama.cpp's head search fills free
cells from a moving head, token by token, so in a shared pool a conversation's cells scattered into holes left by the
other slot or by its own earlier prompts, and every later window spanned them.

- A ubatch of one sequence in a unified cache now goes right after that sequence's last cell when those cells are free.
  Otherwise it goes to the start of the largest free run that holds it, so a conversation stays one run of cells.
  The head search is the fallback.
- `263398e` does this for prompts: with a 47k-token conversation resident, TTFT 372 → 316 ms and 64k prefill
  684 → 837 tok/s. `dd1665f` does it for decode too, which had restarted the search at cell 0 and decoded into small
  holes beside the resident conversation: combined decode 98.7 → 108.6 tok/s, aggregate 88.2 → 94.3 tok/s.
- Known trade-off: a new conversation placed right after the other slot's cells can block that slot's next prompt
  chunk, which then jumps past it, and 64k prefill drops to 719 tok/s. Placing new conversations in the middle of the
  free run recovers it but costs two-slot decode (125 → 110 tok/s), so the current build keeps the tight placement.
- Switches: `LLAMA_KV_PLACE_MIN_TOKENS` (default 1) sets the smallest ubatch placed this way; `LLAMA_KV_WINDOW_LOG=1`
  logs placements and windows.

### 7. Build time (`716cd08`)

`mmvq-qpn.cu`, upstream's tensor-core matvec, instantiates ~1,000 kernels in one file, which took 14 minutes on one core
and set the length of every build. Its kernels now live in `mmvq-qpn-impl.cuh`, with each launch entry point
instantiated in its own file under `ggml/src/ggml-cuda/template-instances/` (`generate_qpn_cu_files.py`), like upstream
llama.cpp's MMQ. The longest of those files takes ~90 s. Every kernel's PTX is unchanged.

## Known limits

- Tested only on V100 PCIe 32 GB with Qwen3.8-27B. The window and placement changes are on for the qwen35 model line
  only.
- DFlash2 requires `LLAMA_TU=0`, so a long prompt in one slot shares batches with the other slot's decode, and the
  decoding slot slows down while it reads.

## Credits and license

MIT, like [llama.cpp](https://github.com/ggml-org/llama.cpp) and [ggml](https://github.com/ggml-org/ggml). Built on
[bitexploder/llama-superfast-v100](https://github.com/bitexploder/llama-superfast-v100); its credits for the models,
the draft vocabulary and ported kernels are in [`README-27B.md`](README-27B.md). DFlash2 is
[z-lab/dflash](https://github.com/z-lab/dflash). Part of this work was sponsored by [Inscora](https://inscora.com).

---

# Upstream: llama-superfast-v100

The rest of this page is upstream's README: its headline figures, hardware notes and serving modes for one and two
cards.

## Llama Superfast

There are many llama forks, but this one is mine :)

We have made llama ultra superfast for v100. We specialize in two models: Qwen 27B 3.8 and Qwen Flash Next 3.8.

The operational models are pretty simple: 27B can run one user on two cards or one user on one card. It's fast in either mode. Flash Next can run with one user on two cards. Two users on two cards for flash next is still experimental. I don't recommend it yet.

### Headline figures: Qwen3.8-27B

Tesla V100 32 GB PCIe cards, MTP speculative decoding, temperature 1.0. Decode speeds are the server's steady rate on real coding-agent work. Prefill means a cold prompt with nothing cached.

- **One user, one card (131,072-token context).**
  - **Decode:** about 100–113 tok/s on long agentic coding sessions, and about 125 tok/s on short coding tasks.
  - **Prefill:** 951 tok/s for a 16K prompt and 770 tok/s for a 64K prompt.
- **One user, two cards (tensor parallel).**
  - **Decode:** 152–156 tok/s on the same agentic work, 1.35× one card.
  - **Prefill:** averages 1,383 tok/s reading a prompt up to 64K, 969 tok/s up to 128K and 686 tok/s up to 240K. On agents' real requests it reads 1.7× as fast as one card.
- **Two users, two cards (131,072 tokens each).**
  - **Decode:** about 200 tok/s combined while both users are decoding (about 100 each), and 150–160 tok/s for one user while the other is idle.
  - **Prefill:** each user's prompts read at 1,020–1,080 tok/s of its own GPU time, or 800–830 tok/s by the wall clock with the other user active.
  - **Throughput:** over a whole agentic session, the pair does about 1.2× the work of one user on the same two cards.

How these were measured, figures by context depth and the 27B's build and run details are in [`README-27B.md`](README-27B.md).

Build and run instructions are at the bottom of this page, under [llama-superfast-v100](#llama-superfast-v100).

## llama-superfast-v100

A llama.cpp fork tuned for the Tesla V100 (Volta, `sm_70`): Qwen3.8-27B on one or two cards and Qwen3.8-Flash-Next on two, with MTP speculative decoding. Build steps, model files, measurements and credits are in [`README-27B.md`](README-27B.md).

### Hardware and software

- One or two Tesla V100 32 GB PCIe (no NVLink needed). Other GPUs are untested.
- NVIDIA driver 580, CUDA 12.8, gcc/g++ 14, Ubuntu 26.04.
- **Two cards in tensor parallel (`GGML_CUDA_P2P=1`) need the IOMMU in passthrough mode.** The cards write to each other's memory directly over PCIe. On our AMD board with the IOMMU translating, the first peer copy raised hundreds of `AMD-Vi IO_PAGE_FAULT`s and stopped the second card (Xid 62), and the server's output was garbage from the first token. Boot with `iommu=pt`: add it to `GRUB_CMDLINE_LINUX_DEFAULT` in `/etc/default/grub`, run `sudo update-grub`, reboot, and check that `grep -o iommu=pt /proc/cmdline` prints it. Intel boards with VT-d on are untested and may need the same. Layer split (the Flash-Next line below) does not set `GGML_CUDA_P2P`.

### Build

```
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc -DCMAKE_C_COMPILER=gcc-14 -DCMAKE_CXX_COMPILER=g++-14 -DCMAKE_CUDA_HOST_COMPILER=g++-14 -DGGML_NATIVE=OFF -DGGML_CUDA_CUB_3DOT2=ON
cmake --build build --target llama-server -j 12
```

### Models

Put them flat in `./models`.

- **Qwen3.8-27B:** `Qwen3.8-27B-UD-Q4_K_XL.gguf` and the MTP head `mtp-Qwen3.8-27B-Q4_0.gguf` from `unsloth/Qwen3.8-27B-GGUF` (download lines in `README-27B.md`). The draft vocabulary ships in `models/`.
- **Qwen3.8-Flash-Next:** ISTA-DASLab's `Qwen3.8-Flash-Next-GSQ-RCO-GGUF` (`IQ3_XXS`, two shards) and the shared MTP head `mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`.

### Run options

Our default serving modes on 32 GB V100s. All commands run from the repository root, with the models in `./models` (`$MODELS`). Sampling: temperature 1.0, top-p 0.95, top-k 20, thinking on.

**1. Qwen3.8-27B, one card** (131,072 tokens)

```
scripts/serve-27b.sh mtp
```

That is one user. For two users on the card, add `--parallel 2` to the script's line in `README-27B.md`; they share the 131,072 tokens (65,536 each). Two users on one card is untested.

**2. Qwen3.8-27B, two cards, two users** (tensor parallel, 131,072 tokens each)

```
CUDA_VISIBLE_DEVICES=0,1 GGML_CUDA_P2P=1 LLAMA_SPEC_DRAFT_VOCAB_FILE=models/draft-vocab-qwen3.8-27b.txt build/bin/llama-server -m $MODELS/Qwen3.8-27B-UD-Q4_K_XL.gguf -md $MODELS/mtp-Qwen3.8-27B-Q4_0.gguf --spec-type draft-mtp --spec-draft-n-max 7 -c 262144 -fa on -ngl 99 -ctk f16 -ctv f16 -ctkd f16 -ctvd f16 -t 12 -b 4096 -ub 2048 --jinja --parallel 2 --split-mode tensor --temp 1.0 --top-p 0.95 --top-k 20
```

For one user, use `--parallel 1 -c 131072`. Two-user scheduling is on by default; `LLAMA_TU=0` turns it off.

**3. Qwen3.8-Flash-Next, two cards, two users** (layer split, 131,072 tokens each)

```
CUDA_VISIBLE_DEVICES=0,1 GGML_SCHED_COPIES=2 build/bin/llama-server -m $MODELS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf -md $MODELS/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf --spec-type draft-mtp --spec-draft-n-max 2 -c 262144 -fa on -ngl 99 -ncmoe 0 --tensor-split 55,45 -ctk bf16 -ctv bf16 -ctkd f16 -ctvd f16 -t 12 -b 4096 -ub 1024 --spec-draft-ubatch 512 -lm mmap --lazy-mode on --jinja --parallel 2 --no-cache-idle-slots --temp 1.0 --top-p 0.95 --top-k 20
```

This leaves only about 240 MB free on the second card; `-b 6144 -ub 1536` does not fit with two users. For one user with the whole 262,144 tokens, use `--parallel 1 -b 6144 -ub 1536`, which reads prompts faster.

**Switches** (environment, all optional)

- `LLAMA_SPEC_REJECTION=0`: exact-match draft acceptance instead of rejection sampling.
- `LLAMA_SPEC_DRAFT_VOCAB=-1`: draft over the full vocabulary.
- `LLAMA_TU=0`: no two-user scheduling.
- `LLAMA_QSA_ATTN_MMA=1`: Flash-Next's alternative tensor-core prompt attention (slower here, off by default).
- `LLAMA_KQ_MASK_DEVICE=0`: build the attention mask on the host.

Add `--host`, `--port` and your sampling settings as usual. Ours: temperature 1.0, top-p 0.95, top-k 20, thinking on.
