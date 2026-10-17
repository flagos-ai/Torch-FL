# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

# Inspected on the BI-V150 host on 2026-10-16. Unlike runtime 2.3.0, this
# development stack supplies IX/mapping_cublas.h, ixblasStatus_t and CUDA_R_16BF,
# which the current native Iluvatar backend compiles against.
# Tag: flagos-dev/vllm-plugin-fl:v0.24.0-iluvatar-ci
ARG BASE_IMAGE=harbor.baai.ac.cn/flagos-dev/vllm-plugin-fl@sha256:e4272f02f80461edeb130fbce7e20ac213e6c0c69bfbc2fc06fc2a2fe62f3f22
FROM ${BASE_IMAGE}

ARG PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple
ARG CPU_TORCH_INDEX_URL=https://mirror.sjtu.edu.cn/pytorch-wheels/cpu

# Preserve only the native build/runtime assets outside Python. The official
# CPU frontend does not inherit any of the base's Python packages.
RUN mkdir -p /opt/torch-fl-iluvatar-libtorch/lib /opt/torch-fl-ci \
 && cp -a /usr/local/lib/python3.12/site-packages/torch/include \
          /usr/local/lib/python3.12/site-packages/torch/share \
          /opt/torch-fl-iluvatar-libtorch/ \
 && cp -a /usr/local/lib/python3.12/site-packages/torch/lib/*.so* \
          /opt/torch-fl-iluvatar-libtorch/lib/ \
 && cp /usr/local/lib/python3.12/site-packages/torch/version.py \
       /opt/torch-fl-iluvatar-libtorch/lib/vendor_version.py \
 && /usr/local/bin/python3.12 -c 'import importlib.metadata as m; print(m.version("torch"))' \
       > /opt/torch-fl-ci/vendor-torch-version \
 && python3.12 -m venv /opt/torch-fl-iluvatar-venv
ENV VIRTUAL_ENV=/opt/torch-fl-iluvatar-venv \
    PATH=/opt/torch-fl-iluvatar-venv/bin:${PATH} \
    PYTHONPATH= \
    PYTHONNOUSERSITE=1 \
    COREX_PATH=/usr/local/corex \
    FLAGOS_VENDOR_TORCH_LIB=/opt/torch-fl-iluvatar-libtorch/lib \
    TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
    FLAGOS_ACCELERATOR=iluvatar \
    FLAGOS_BUILD_VENDOR=1 \
    FLAGOS_BUILD_FLAGGEMS=1 \
    FLAGOS_BUILD_BOXING=0 \
    FLAGOS_BUILD_FLAGGEMS_CPP=0 \
    FLAGOS_BUILD_TILEOPS=0 \
    FLAGOS_SKIP_CUDA_ASSETS=1 \
    FLAGOS_DISABLE_CUDA_ASSETS=1 \
    PIP_CONSTRAINT=/opt/torch-fl-ci/constraints.txt \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY .github/version-pins.env /opt/torch-fl-ci/version-pins.env
COPY docker/iluvatar/verify.py /opt/torch-fl-ci/verify.py
COPY docker/iluvatar/copy_flagtree.py /opt/torch-fl-ci/copy_flagtree.py

RUN set -eu; \
    . /opt/torch-fl-ci/version-pins.env; \
    printf 'torch==%s+cpu\n' "$CPU_TORCH_VERSION_DEFAULT" \
        > /opt/torch-fl-ci/constraints.txt; \
    printf 'numpy==1.26.4\nflag_gems==%s\nflagtree==%s\n' \
        "$FLAGGEMS_VERSION_DEFAULT" "$FLAGTREE_VERSION_iluvatar" \
        >> /opt/torch-fl-ci/constraints.txt; \
    python -m pip install --no-cache-dir --index-url "$CPU_TORCH_INDEX_URL" \
        --extra-index-url "$PIP_INDEX_URL" "torch==$CPU_TORCH_VERSION_DEFAULT+cpu"; \
    python -m pip install --no-cache-dir --index-url "$PIP_INDEX_URL" \
        setuptools wheel build 'cmake==3.31.10' ninja pybind11 patchelf \
        pytest pytest-timeout 'numpy==1.26.4' \
        'transformers>=4.51,<5' safetensors sentencepiece tiktoken protobuf \
        'packaging>=26.0' 'PyYAML==6.0.1' 'sqlalchemy==2.0.48'; \
    /usr/local/bin/python3.12 /opt/torch-fl-ci/copy_flagtree.py "$FLAGTREE_VERSION_iluvatar"; \
    python -m pip install --no-cache-dir --ignore-installed --no-deps \
        --index-url https://resource.flagos.net/repository/flagos-pypi-iluvatar/simple \
        "flag_gems==$FLAGGEMS_VERSION_DEFAULT"

# The native backend uses CXX/C only, so no vendor nvcc/IX CMake probe is needed.
# This check also compiles the SDK types used by the native kernels.
RUN python /opt/torch-fl-ci/verify.py \
 && python -m pip check \
 && python -m pip freeze --all > /opt/torch-fl-ci/installed.txt

# Build the revision under test in CI; never bake torch_fl into this image.
WORKDIR /workspace

LABEL org.opencontainers.image.title="Torch-FL Iluvatar isolated CPU frontend" \
      org.opencontainers.image.description="Official CPU torch venv and separate CoreX native assets" \
      org.opencontainers.image.source="https://github.com/flagos-ai/PyTorch-Plugin-FL" \
      org.opencontainers.image.version="manual-20261016-iluvatar-cpu-dev" \
      org.opencontainers.image.base.name="harbor.baai.ac.cn/flagos-dev/vllm-plugin-fl@sha256:e4272f02f80461edeb130fbce7e20ac213e6c0c69bfbc2fc06fc2a2fe62f3f22" \
      io.flagos.sdk.version="4.5.0" \
      io.flagos.validation="isolated-cpu-frontend-sdk" \
      io.flagos.gpu-validation="requires-installed-wheel"
