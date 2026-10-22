# syntax=docker/dockerfile:1.7
###############################################################################
# llama-superfast-v100-q8-inscora for the Tesla V100 (sm_70), CUDA 12.
#
#   DOCKER_BUILDKIT=1 docker build -f .devops/cuda-v100.Dockerfile -t llama-superfast-v100 .devops
#
# The source is cloned from LLAMA_REPO at LLAMA_REF (a branch, tag or commit), so
# the build context can be any small directory. Pin LLAMA_REF to a commit for a
# reproducible image.
#
# CUDA 12, not 13: CUDA 13 dropped Volta. 12.8 is what the fork is built and
# measured with.
###############################################################################

##### stage 1: build #########################################################
ARG CUDA_VERSION=12.8.1
ARG UBUNTU_VERSION=24.04
ARG BASE_IMAGE=nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}
ARG RUNTIME_IMAGE=nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}
FROM ${BASE_IMAGE} AS build

ARG LLAMA_REPO=https://github.com/gabtremblay/llama-superfast-v100-q8-inscora.git
ARG LLAMA_REF=main

# 70-real = sm_70 SASS only: no PTX, no JIT. Never "native": no GPU is visible in a docker build.
ARG CUDA_ARCH=70-real

# OFF still builds the f16 and q8_0 KV pairs the fast attention paths read.
ARG FA_ALL_QUANTS=OFF

ARG GGML_NATIVE=OFF

# Prebuilt web UI (fails soft to "no UI").
ARG PREBUILT_UI=ON

ARG BUILD_TESTS=OFF

# Parallel compile jobs. The CUTLASS attention units take several GB of RAM per nvcc.
ARG BUILD_JOBS=8

ARG GCC_VERSION=14

ENV DEBIAN_FRONTEND=noninteractive

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
      git cmake ninja-build ccache pkg-config python3 \
      build-essential gcc-${GCC_VERSION} g++-${GCC_VERSION} \
      libgomp1 libssl-dev ca-certificates curl

ENV CC=gcc-${GCC_VERSION} \
    CXX=g++-${GCC_VERSION} \
    CUDAHOSTCXX=g++-${GCC_VERSION}

WORKDIR /src
# Full clone: the build number (which the prebuilt UI lookup keys on) comes from `git rev-list --count HEAD`.
RUN git clone "${LLAMA_REPO}" . \
 && git checkout --detach "${LLAMA_REF}" \
 && git submodule update --init --recursive \
 && git log -1 --format='%H %ad %s' --date=short

# --allow-shlib-undefined: libggml-cuda.so uses the CUDA driver API (VMM), and libcuda.so.1 only exists at
# runtime, injected by the NVIDIA container toolkit.
# CCACHE_DIR is exported for the whole RUN so the build itself uses the cache mount, which persists across image
# builds: a rebuild after a small change recompiles only what changed.
RUN --mount=type=cache,target=/ccache \
    export CCACHE_DIR=/ccache CCACHE_MAXSIZE=10G \
 && cmake -S . -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_NATIVE="${GGML_NATIVE}" \
        -DGGML_CCACHE=ON \
        -DGGML_OPENMP=ON \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
        -DGGML_CUDA_CUB_3DOT2=ON \
        -DGGML_CUDA_FA=ON \
        -DGGML_CUDA_FA_ALL_QUANTS="${FA_ALL_QUANTS}" \
        -DGGML_CUDA_GRAPHS=ON \
        -DGGML_CUDA_FORCE_MMQ=OFF \
        -DGGML_CUDA_FORCE_CUBLAS=OFF \
        -DGGML_CUDA_NO_VMM=OFF \
        -DGGML_CUDA_NCCL=OFF \
        -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined \
        -DLLAMA_OPENSSL=ON \
        -DLLAMA_BUILD_COMMON=ON \
        -DLLAMA_BUILD_TOOLS=ON \
        -DLLAMA_BUILD_SERVER=ON \
        -DLLAMA_BUILD_EXAMPLES=OFF \
        -DLLAMA_BUILD_TESTS="${BUILD_TESTS}" \
        -DLLAMA_TESTS_INSTALL="${BUILD_TESTS}" \
        -DLLAMA_BUILD_UI=OFF \
        -DLLAMA_USE_PREBUILT_UI="${PREBUILT_UI}" \
 && cmake --build build --config Release -j"${BUILD_JOBS}" \
 && cmake --install build --prefix /opt/llama \
 && ccache -s

RUN mkdir -p /opt/llama/lib && cp -a build/bin/*.so /opt/llama/lib/ 2>/dev/null || true

# The draft vocabulary rankings (token ids only, no text). The engine's compiled-in default path points at
# /src/models, which does not exist in the runtime stage, so the runtime ENV below names the file.
RUN install -d /opt/llama/share/llama-superfast \
 && install -m 0644 models/draft-vocab-*.txt /opt/llama/share/llama-superfast/ \
 && wc -l /opt/llama/share/llama-superfast/draft-vocab-*.txt

##### stage 2: runtime #######################################################
ARG RUNTIME_IMAGE
FROM ${RUNTIME_IMAGE} AS runtime

ENV DEBIAN_FRONTEND=noninteractive
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
      libgomp1 libssl3t64 ca-certificates curl

COPY --from=build /opt/llama /opt/llama

RUN echo /opt/llama/lib > /etc/ld.so.conf.d/000-llama.conf && ldconfig

# LLAMA_SPEC_DRAFT_VOCAB_FILE: the 27B's token ranking; the draft scores its first 98,304 ids (the engine's
# default). Override per run with `docker run -e`: LLAMA_SPEC_DRAFT_VOCAB=-1 drafts over the full vocabulary.
ENV PATH=/opt/llama/bin:$PATH \
    LD_LIBRARY_PATH=/opt/llama/lib:${LD_LIBRARY_PATH} \
    LLAMA_CACHE=/tmp/llama-cache \
    LLAMA_SPEC_DRAFT_VOCAB_FILE=/opt/llama/share/llama-superfast/draft-vocab-qwen3.8-27b.txt \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /models
EXPOSE 8080
ENTRYPOINT ["llama-server"]
