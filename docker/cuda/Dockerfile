# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Native torch-2.9 ABI CUDA base image for PyTorch-Plugin-FL CI.
#
# This rebuilds, from source, the same stack the vendor 2.10 image shipped as
# a black box -- but pinned to torch 2.9.0 so the C++ FlagGems extensions
# (liboperators.so / libtriton_jit.so) and FlagCX are compiled against a
# single coherent 2.9 ABI, retiring the fix_2.9_watch hack (CPU venv +
# extracted libtorch_cuda + LD_LIBRARY_PATH stripping). torch_fl runs in
# "active-torch-supplies-libtorch_cuda" mode (FLAGOS_SKIP_CUDA_ASSETS=1 at
# build, FLAGOS_DISABLE_CUDA_ASSETS=1 at runtime), so the full GPU torch in
# this image provides libtorch_cuda directly.
#
# Bumping to 2.11 / 2.12: change the ARG defaults below (and, if the CUDA
# major version moves, BASE_IMAGE + TORCH_CUDA_TAG + TORCH_INDEX_URL +
# CUDA_VERSION together). No CI-script changes required.

ARG BASE_IMAGE=nvidia/cuda:13.0.2-devel-ubuntu22.04
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130
ARG PYPI_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ARG TORCH_VERSION=2.9.0
ARG TORCH_CUDA_TAG=cu130
ARG CUDA_VERSION=13.0
ARG FLAGGEMS_REF=v5.3.0
ARG FLAGCX_REF=v0.13.0

FROM ${BASE_IMAGE}

ARG TORCH_INDEX_URL
ARG PYPI_INDEX_URL
ARG TORCH_VERSION
ARG TORCH_CUDA_TAG
ARG CUDA_VERSION
ARG FLAGGEMS_REF
ARG FLAGCX_REF

ENV CUDA_HOME=/usr/local/cuda
ENV CUDA_PATH=/usr/local/cuda
ENV VIRTUAL_ENV=/opt/venv
# NCCL root for the FlagCX NVIDIA adaptor. Pinned to python3.10 because the
# base image is ubuntu22.04; change this (or derive dynamically) if the base
# moves to a different Python.
ENV CCL_HOME=/opt/venv/lib/python3.10/site-packages/nvidia/nccl
ENV CMAKE_BUILD_PARALLEL_LEVEL=16
ENV MAX_JOBS=16
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV PATH=/opt/venv/bin:${PATH}
ENV FLAGCX_PATH=/opt/FlagCX

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential ca-certificates ccache curl git \
      libibverbs-dev libnuma-dev librdmacm-dev pkg-config \
      python3 python3-dev python3-pip python3-venv \
    && git config --global http.version HTTP/1.1 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv \
 && pip install --no-cache-dir --upgrade \
      --index-url "${PYPI_INDEX_URL}" \
      pip setuptools wheel cmake ninja \
      scikit-build-core pybind11

# Install the full GPU torch wheel. Do NOT pin triton or the nvidia-* runtime
# libs here: torch's own wheel metadata pulls the version-matched triton and
# CUDA runtime libraries it was built against, which is the only correct source
# of truth. torchvision/torchaudio are omitted (the test suite does not import
# them); re-add with matching +cu130 wheels if a future test needs them.
RUN pip install --no-cache-dir \
      --index-url "${TORCH_INDEX_URL}" \
      "torch==${TORCH_VERSION}+${TORCH_CUDA_TAG}"

# Build FlagGems C++ extensions against this image's torch, producing 2.9-ABI
# liboperators.so / libtriton_jit.so. CMAKE_PREFIX_PATH must point at the
# active torch's cmake dir so scikit-build-core resolves torch targets.
RUN CMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')" \
    CMAKE_ARGS="-DFLAGGEMS_BUILD_C_EXTENSIONS=ON -DFLAGGEMS_BACKEND=CUDA" \
    pip install --no-cache-dir \
      --index-url "${PYPI_INDEX_URL}" \
      "git+https://github.com/flagos-ai/FlagGems.git@${FLAGGEMS_REF}"

# Build FlagCX with the NVIDIA adaptor against the active torch/CUDA env.
RUN git clone --depth 1 --branch "${FLAGCX_REF}" \
      https://github.com/flagos-ai/FlagCX.git /opt/FlagCX \
 && git -C /opt/FlagCX submodule update --init --recursive \
 && FLAGCX_ADAPTOR=nvidia \
    pip install --no-cache-dir \
      --no-build-isolation \
      --index-url "${PYPI_INDEX_URL}" \
      /opt/FlagCX

RUN pip install --no-cache-dir \
      --index-url "${PYPI_INDEX_URL}" \
      build pytest transformers

COPY docker/cuda/verify.py /tmp/verify.py
RUN VERIFY_TORCH_VERSION="${TORCH_VERSION}" \
    VERIFY_CUDA_VERSION="${CUDA_VERSION}" \
    python /tmp/verify.py && rm /tmp/verify.py

WORKDIR /workspace
