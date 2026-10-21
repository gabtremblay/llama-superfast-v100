# llama-superfast-v100-q8-inscora

Inscora's fork of [bitexploder/llama-superfast-v100](https://github.com/bitexploder/llama-superfast-v100). It adds an
8-bit (`q8_0`) KV cache that keeps the upstream fork's attention speed, so one 32 GB V100 holds about two thirds more
context. It also makes prompt processing faster and keeps it on the GPU.

We serve Qwen3.8-27B (`UD-Q4_K_XL`) on **one** Tesla V100 PCIe 32 GB behind an eGPU dock (PCIe Gen3 x4, about 3.3 GB/s).
The draft model is DFlash2. The server runs two slots: one user, so an agent harness can send a side request in parallel
with the main thinking thread. Every change below targets that setup. On a slow host link, anything that crosses the
link is on the critical path, so the work stays on the GPU wherever possible.

Everything upstream (the tensor-core matvec on repacked K-quants, MTP, the two-user scheduler, tensor-parallel two-card
serving) is unchanged. Upstream's description follows the [Llama Superfast](#llama-superfast) heading below.

## What we changed

All of it is on the `q8-attention` branch, in four commits on top of upstream `403f06f`.

**1. Q8_0 KV cache on the Volta attention paths** (`bcf1877`, `5c15ec5`)

- Upstream's fast paths (the CUTLASS 6-head prefill attention `fattn-gqa6` and the streaming decode/verify kernel
  `fattn-verify-stream`) took only an f16 KV cache. `-ctk q8_0 -ctv q8_0` sent every step through the generic kernels
  over a temporary f16 copy of the cache.
- Both paths now read `q8_0` directly:
  - The prefill kernel widens key blocks to f16 in shared memory.
  - The verify kernel has a `KVQ8` variant with 8-byte row loads and aligned windows.
  - Allocations no longer reserve the f16 copy for those routes.
- Context: our server went from two slots of 73,728 tokens (f16 cache, MTP draft) to two slots of 122,880 (`q8_0`,
  DFlash2 draft, with the VRAM change at the end of section 3). 2 × 131,072 also passes our fill test (both slots full
  at once, then generation), with about 870 MiB to spare.
- Tests: `q8_0` cases for both kernels in `test-backend-ops`, including the shapes the 27B uses.
- Switches: `LLAMA_FA_GQA6_Q8=0` and `LLAMA_FATTN_VERIFY_STREAM_Q8=0` fall back to the generic kernels.
  `LLAMA_FATTN_VERIFY_STREAM_Q8_KP` and `LLAMA_FATTN_VERIFY_STREAM_Q8_W` tune the verify kernel.

**2. Faster prefill on repacked weights** (`044ef3b`)

- Long prompts expand each weight to fp16 for cuBLAS. The expansion now decodes straight from the repacked layout with
  16-byte stores, instead of one warp rebuilding the GGUF blocks while the rest of the block waits. It matches ggml's
  dequantizers bit for bit.
- Products of 129 to 191 tokens go to MMQ on a device-side unrepack instead of fp16 + cuBLAS. The Volta MMQ crossover is
  192 tokens (measured: 64 tokens 0.5 ms vs 1.25 ms per 27B FFN product; 256 tokens 1.72 vs 1.50 ms).
- The tensor-core kernel takes up to 128 tokens by default.
- Switches:
  - `LLAMA_QPN_MAX_TOKENS` (default 128, at most 256).
  - `GGML_CUDA_MMQ_MAX_BATCH` overrides the crossover.
  - `LLAMA_QPN_MMQ=0` and `LLAMA_QPN_FAST_EXPAND=0` restore the earlier routes.
  - `LLAMA_QPN_EXPAND_CHECK=1` runs the earlier kernels alongside and reports any difference.

**3. Prompt work kept on the GPU** (`64107c5`)

- **DFlash2 features stay on the device.** The draft reads the target's layer inputs. For any decode that is one ubatch,
  they now stay on the GPU, and the draft copies only the rows its 2,048-token window can see, device to device. Before,
  every prefill ubatch shipped all of them to the host and back (about 2.3 GB for a 24k prompt). Run with `-b` equal to
  `-ub` so prefill qualifies. Verify steps with two slots now take this path too.
  - Switches: `LLAMA_DFLASH2_FEAT_DEV=0` turns it off; `LLAMA_DFLASH2_FEAT_DEV_ROWS` caps the rows.
- **Checkpoints copy in the background.** A prompt checkpoint (the 27B's recurrent state, ~150 MiB) is first copied into
  a GPU staging buffer. It then goes to the host on a second CUDA stream, so the next compute no longer waits for the
  link. The draft's small state is read first, before the big copy occupies the link.
  - Switch: `LLAMA_CKPT_STAGE=0` turns staging off.
- **No duplicate checkpoints.** A checkpoint at the same position as an existing one is kept, not copied again. Before,
  every new conversation re-saved the shared system prompt's checkpoint.
  - Switch: `LLAMA_CKPT_KEEP_SAME=0` restores the old behavior.
- **Cheaper prompt-cache saves.** A save reuses the evicted entries' buffers. It moves checkpoints instead of copying
  them when the next prompt won't need them.
- **Two-user scheduler.** A lone prompt is no longer cut into small slices when no other slot is waiting.
  - Switch: `LLAMA_TU_IDLE_JOB_MIN` sets the minimum slice.
- **Smaller draft compute buffer.** With `--spec-draft-ubatch 512`, the DFlash2 draft's compute buffer stops being
  sized for 2,048-token batches. That frees 1.7 GB of VRAM at no measured speed cost.

## Measured (one V100 32 GB, eGPU, Qwen3.8-27B UD-Q4_K_XL, DFlash2, q8_0 KV, two slots)

| | upstream route + q8 KV | this fork |
|---|---|---|
| Cold prefill, 34k-token prompt | 832 tok/s | 892 tok/s |
| Cold prefill, 5.6k-token prompt | 760 tok/s | 851 tok/s |
| Time to first token, new ~125-token chat turn (median) | 828 ms | 680 ms, 430 ms with `--cache-ram 0` |
| Decode, 400 tokens, three prompts | 88 / 90 / 104 tok/s | 88 / 91 / 106 tok/s |
| Context | 2 × 98,304 | 2 × 122,880 |

The "upstream route" column is the same build with this fork's prefill and GPU-residency changes switched off. BetterBench
results for the full configuration are in progress.

`--cache-ram 0` turns off the host prompt cache. Saving a slot's state to host memory costs about 220 ms over this link
each time a slot switches conversations. With one user and pinned slots (`id_slot` in the request), the cache brings
little.

## Run (one card, two slots, DFlash2)

```
LLAMA_TU=0 LLAMA_DFLASH2_HEAD_FILE=$MODELS/mtp-Qwen3.8-27B-Q4_0.gguf \
build/bin/llama-server -m $MODELS/Qwen3.8-27B-UD-Q4_K_XL.gguf \
  -md $MODELS/Qwen3.8-27B-DFlash2-Q4_K_M.gguf --spec-type draft-dflash --spec-draft-n-max 7 --spec-draft-ubatch 512 \
  -c 245760 --parallel 2 --no-kv-unified -fa on -ngl 99 -ctk q8_0 -ctv q8_0 -ctkd f16 -ctvd f16 \
  -t 12 -b 2048 -ub 2048 --cache-ram 0 --jinja --temp 1.0 --top-p 0.95 --top-k 20
```

- `LLAMA_TU=0` is required with DFlash2: the two-user scheduler rejects its long prompts.
- `LLAMA_DFLASH2_HEAD_FILE` points DFlash2's output head at the MTP file's rows.
- `-b 2048` (equal to `-ub`) keeps the DFlash2 features on the GPU during prefill.

---

# Llama Superfast

There are many llama forks, but this one is mine :)

We have made llama ultra superfast for v100. We specialize in two models: Qwen 27B 3.8 and Qwen Flash Next 3.8.

The operational models are pretty simple: 27B can run one user on two cards or one user on one card. It's fast in either mode. Flash Next can run with one user on two cards. Two users on two cards for flash next is still experimental. I don't recommend it yet.

## Headline figures: Qwen3.8-27B

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

# llama.cpp

![llama](https://raw.githubusercontent.com/ggml-org/llama.brand/refs/heads/master/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp?filter=v*&color=brightgreen)](https://github.com/ggml-org/llama.cpp/releases?q=tag:v0)
[![Nightly](https://img.shields.io/github/v/release/ggml-org/llama.cpp?label=nightly&filter=b*&color=orange)](https://github.com/ggml-org/llama.cpp/releases?q=b)
[![Server](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/server.yml?label=Server)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/docker.yml?label=Docker)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/winget.yml?label=Winget)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/issues?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Ajhen0409%20OR%20author%3Abartowski1182%20OR%20author%3Anikwen%20OR%20author%3Ahipudding%20OR%20author%3Aravi9%20OR%20author%3AServeurpersoCom%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3Amarty1885%20OR%20author%3A0cc4m%20OR%20author%3ATitaniumtown%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3Aallozaur%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Awine99%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Alhez%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming)%20sort%3Aupdated-desc) / [dev stats](https://github.com/ggml-org/llama.cpp-dev) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

## Quick start

A few options to get `llama.cpp` installed on your machine:

- Visit https://llama.app and follow the instructions
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed:

```sh
# Download and run a model directly from Hugging Face
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# Launch OpenAI-compatible API server
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF
```

<table align="center">
    <tr>
        <td align="center" width=50%>
            <img width="1310" height="888" alt="VLM session with `llama cli`" src="https://github.com/user-attachments/assets/88726b48-1713-48aa-a525-95a02e78afc4" />
            <i>VLM session with <b>llama cli</b></i>
        </td>
        <td align="center">
            <img width="1392" height="958" alt="Built-in web UI against `llama serve` running Qwen 3.6" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>Built-in web UI against <b>llama serve</b></i>
        </td>
    </tr>
<table>

## Description

The main goal of `llama.cpp` is to enable LLM (and VLM) inference with minimal setup and state-of-the-art performance on
a wide range of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- RVV, ZVFH, ZFH, ZICBOP and ZIHINTPAUSE support for RISC-V architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads GPUs via MUSA)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

The `llama.cpp` project is build on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Supported backends

| Backend | Target devices |
| --- | --- |
| [BLAS](docs/build.md#blas-build) | All |
| [BLIS](docs/backend/BLIS.md) | All |
| [CANN](docs/build.md#cann) | Ascend NPU |
| [CUDA](docs/build.md#cuda) | Nvidia GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [Hexagon](docs/backend/snapdragon/README.md) | Snapdragon |
| [IBM zDNN](docs/backend/zDNN.md) | IBM Z & LinuxONE |
| [MUSA](docs/build.md#musa) | Moore Threads GPU |
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [OpenCL](docs/backend/OPENCL.md) | Adreno GPU |
| [OpenVINO [In Progress]](docs/backend/OPENVINO.md) | Intel CPUs, GPUs, and NPUs |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | All |
| [SYCL](docs/backend/SYCL.md) | Intel GPU |
| [VirtGPU](docs/backend/VirtGPU.md) | VirtGPU APIR |
| [Vulkan](docs/build.md#vulkan) | GPU |
| [WebGPU](docs/build.md#webgpu) | All |
| [ZenDNN](docs/build.md#zendnn) | AMD CPU |

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [XCFramework](docs/xcframework.md)
- [Completions](docs/completions.md)
- [Models](docs/models.md)
- [Release process](docs/release.md)

## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information

## Acknowledgements

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain

---

# llama-superfast-v100



A llama.cpp fork tuned for the Tesla V100 (Volta, `sm_70`): Qwen3.8-27B on one or two cards and Qwen3.8-Flash-Next on two, with MTP speculative decoding. Build steps, model files, measurements and credits are in [`README-27B.md`](README-27B.md).

## Hardware and software

- One or two Tesla V100 32 GB PCIe (no NVLink needed). Other GPUs are untested.
- NVIDIA driver 580, CUDA 12.8, gcc/g++ 14, Ubuntu 26.04.
- **Two cards in tensor parallel (`GGML_CUDA_P2P=1`) need the IOMMU in passthrough mode.** The cards write to each other's memory directly over PCIe. On our AMD board with the IOMMU translating, the first peer copy raised hundreds of `AMD-Vi IO_PAGE_FAULT`s and stopped the second card (Xid 62), and the server's output was garbage from the first token. Boot with `iommu=pt`: add it to `GRUB_CMDLINE_LINUX_DEFAULT` in `/etc/default/grub`, run `sudo update-grub`, reboot, and check that `grep -o iommu=pt /proc/cmdline` prints it. Intel boards with VT-d on are untested and may need the same. Layer split (the Flash-Next line below) does not set `GGML_CUDA_P2P`.

## Build

```
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc -DCMAKE_C_COMPILER=gcc-14 -DCMAKE_CXX_COMPILER=g++-14 -DCMAKE_CUDA_HOST_COMPILER=g++-14 -DGGML_NATIVE=OFF -DGGML_CUDA_CUB_3DOT2=ON
cmake --build build --target llama-server -j 12
```

## Models

Put them flat in `./models`.

- **Qwen3.8-27B:** `Qwen3.8-27B-UD-Q4_K_XL.gguf` and the MTP head `mtp-Qwen3.8-27B-Q4_0.gguf` from `unsloth/Qwen3.8-27B-GGUF` (download lines in `README-27B.md`). The draft vocabulary ships in `models/`.
- **Qwen3.8-Flash-Next:** ISTA-DASLab's `Qwen3.8-Flash-Next-GSQ-RCO-GGUF` (`IQ3_XXS`, two shards) and the shared MTP head `mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`.

## Run options

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
