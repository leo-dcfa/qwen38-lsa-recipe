# syntax=docker/dockerfile:1
# Qwen3.8-27B + D-LSA on one GB10: the authors SGLang fork, installed with uv
# into a venv. Not built from the fork Dockerfile, whose documented build is
# amd64 only and asks for about 100 GB of disk.
# Versions come from pins.env. Use ./build.sh, which passes them in.

# CUDA 13.0.3: the CUDA that torch 2.13.0 ships with. "devel" adds nvcc, which
# FlashInfer uses to compile kernels when the server starts. Pinned by digest.
FROM nvidia/cuda:13.0.3-devel-ubuntu24.04@sha256:b7ae301dea2c162444795462ce17a05f6a516e5a75944b57af5b88540a1a2266

# uv, copied from its official image rather than piped from an install script.
COPY --from=ghcr.io/astral-sh/uv:0.13.4@sha256:9f130421e930bfbdc563c00740466603f8723158e90297d07957c2294640262b /uv /usr/local/bin/uv

# git to fetch the code, curl for the map. A compiler for cuda-tile, which ships
# only as source. procps because sparkrun checks on the server with ps.
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      build-essential ca-certificates curl git procps \
 && rm -rf /var/lib/apt/lists/*

ARG SGLANG_REPO
ARG SGLANG_COMMIT

# Fetch exactly one commit. git checks the content against the hash. Then .git
# goes, so the version falls back to 0.0.0.dev0 (the pyproject allows it).
RUN : "${SGLANG_COMMIT:?pass --build-arg SGLANG_COMMIT from pins.env}" \
 && git init -q /src/sglang \
 && cd /src/sglang \
 && git fetch -q --depth 1 "$SGLANG_REPO" "$SGLANG_COMMIT" \
 && git checkout -q FETCH_HEAD \
 && test "$(git rev-parse HEAD)" = "$SGLANG_COMMIT" \
 && rm -rf .git

ARG PYTHON_VERSION
ARG EXCLUDE_NEWER

# The fork, without its [all] extra: that adds image and video generation.
# The cutoff settles the dependencies the fork leaves open.
# No Rust extensions: local serving does not use them.
ENV UV_PYTHON_INSTALL_DIR=/opt/python VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH
RUN : "${PYTHON_VERSION:?pass --build-arg PYTHON_VERSION from pins.env}" \
      "${EXCLUDE_NEWER:?pass --build-arg EXCLUDE_NEWER from pins.env}" \
 && uv venv -q --python "$PYTHON_VERSION" /opt/venv \
 && UV_EXCLUDE_NEWER="$EXCLUDE_NEWER" SGLANG_BUILD_RUST_EXTS=none \
    MAX_JOBS=4 CMAKE_BUILD_PARALLEL_LEVEL=4 \
    uv pip install --no-cache /src/sglang/python

ARG LSA_REPO
ARG LSA_COMMIT

# The LSA plugin. SGLang finds it through its entry point. --no-deps as in the
# authors README: its runtime dependencies are already here at the same pins,
# and nothing from its optional harness is wanted in a serving image.
RUN : "${LSA_COMMIT:?pass --build-arg LSA_COMMIT from pins.env}" \
 && git init -q /src/lsa \
 && cd /src/lsa \
 && git fetch -q --depth 1 "$LSA_REPO" "$LSA_COMMIT" \
 && git checkout -q FETCH_HEAD \
 && test "$(git rev-parse HEAD)" = "$LSA_COMMIT" \
 && rm -rf .git \
 && UV_EXCLUDE_NEWER="$EXCLUDE_NEWER" uv pip install --no-cache --no-deps /src/lsa
