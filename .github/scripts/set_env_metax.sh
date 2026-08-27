#!/usr/bin/env bash
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

set -euo pipefail

case "${CI_STAGE:-}" in
  build|integration) ;;
  *)
    echo "::error::CI_STAGE must be either 'build' or 'integration'"
    exit 1
    ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export PATH="/opt/venv/bin:/opt/maca/tools/cu-bridge/bin:/opt/maca/mxgpu_llvm/bin:/opt/maca/bin:$PATH"
export VIRTUAL_ENV=/opt/venv
export PYTHONNOUSERSITE=1

export ACCELERATOR=metax
export METAX_PATH=/opt/maca
export MACA_PATH=/opt/maca
export MACA_HOME=/opt/maca

export FLAGOS_METAX_BOXING=1
export FLAGOS_METAX_CUDART_SHIM=1
export FLAGOS_DISABLE_CUDA_ASSETS=1
export FLAGOS_USE_FLAGGEMS=0
export FLAGGEMS_KERNEL=0
export FLAGGEMS_PYTHON=0
export FLAGOS_WHEEL_LOCAL=metax3.8.0
export FLAGOS_MACA_TORCH_LIB=/opt/vendor-libtorch/lib

export LD_LIBRARY_PATH="/opt/maca/lib:/opt/maca/tools/cu-bridge/lib:/opt/maca/mxgpu_llvm/lib:/opt/maca/mxshmem/lib:/opt/maca/ompi/lib:/opt/maca/ucx/lib:/opt/mxdriver/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LIBRARY_PATH="/opt/maca/lib:/opt/maca/tools/cu-bridge/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export CPATH="/opt/maca/tools/cu-bridge/include:/opt/maca/include:/opt/maca/include/mcr${CPATH:+:$CPATH}"

for path in \
  /opt/venv/bin/python \
  /opt/maca/tools/cu-bridge/bin/cucc \
  /opt/vendor-libtorch/lib/libc10.so \
  /opt/vendor-libtorch/lib/libtorch_cpu.so \
  /opt/vendor-libtorch/lib/libtorch.so \
  /opt/vendor-libtorch/lib/libtorch_global_deps.so \
  /opt/vendor-libtorch/lib/libtorch_python.so \
  /opt/vendor-libtorch/lib/libc10_cuda.so \
  /opt/vendor-libtorch/lib/libtorch_cuda.so \
  /opt/vendor-libtorch/lib/libtorch_cuda_linalg.so; do
  if [[ ! -e "$path" ]]; then
    echo "::error::Required MetaX image asset is missing: $path"
    exit 1
  fi
done

# Device probes only run during integration tests. Build-only runs do not need
# the device, and manifests have a more complete device-availability check.
if [[ "$CI_STAGE" == "integration" && ! -c /dev/mxcd ]]; then
  echo "::error::MetaX device node /dev/mxcd is unavailable"
  exit 1
fi

python - <<'PY'
from pathlib import Path

import torch

torch_path = Path(torch.__file__).resolve()
assert torch.__version__.startswith("2.10.0+cpu"), torch.__version__
assert str(torch_path).startswith("/opt/venv/"), torch_path
assert "/opt/conda/" not in str(torch_path), torch_path

print(f"Build Python: {Path(__import__('sys').executable).resolve()}")
print(f"Build PyTorch: {torch.__version__}")
print(f"Build torch path: {torch_path}")
PY

# Expose the vendor Triton (triton-metax) to the CPU torch venv. torch.compile
# needs it: the active torch is the CPU wheel, which ships no Triton, so
# inductor raises TritonMissing without this. The vendor package lives in the
# image's MetaX torch install, which we otherwise deliberately do not use --
# only libtorch is consumed, from /opt/vendor-libtorch.
#
# Linked rather than copied: the metax backend carries ~2.4GB of device
# libraries and a cp -a of that is pure CI wall time. set_env_cuda.sh copies
# because it also relocates FlagGems/FlagCX; here only Triton is needed.
if [[ "$CI_STAGE" == "integration" ]]; then
  VENV_SITE="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
  VENDOR_TRITON=""
  for candidate in /opt/conda/lib/python3.*/site-packages \
                   /opt/vendor-torch/lib/python3.*/site-packages \
                   /usr/lib/python3.*/site-packages \
                   /usr/local/lib/python3.*/site-packages; do
    if [[ -d "$candidate/triton" ]]; then
      VENDOR_TRITON="$candidate/triton"
      break
    fi
  done

  # A missing vendor Triton is a warning, not a setup failure. Exiting here would
  # take the other nine baseline groups down over a torch.compile gap and leave
  # the run with no evidence; instead compile-tests fails on its own and that
  # failure is the record for the platform owners.
  if [[ -n "$VENDOR_TRITON" ]]; then
    if [[ ! -e "$VENV_SITE/triton" ]]; then
      ln -s "$VENDOR_TRITON" "$VENV_SITE/triton"
    fi
    for metadata in "$(dirname "$VENDOR_TRITON")"/triton-*.dist-info; do
      [[ -e "$metadata" ]] || continue
      [[ -e "$VENV_SITE/$(basename "$metadata")" ]] || ln -s "$metadata" "$VENV_SITE/"
    done
  fi

  # The model stack the inference and training groups load. Installed whether or
  # not the venv came prebuilt: a venv baked without it makes those groups fail as
  # an environment problem that reads as a MACA problem.
  #
  # transformers is pinned to [4.51, 5) for Qwen3 model_type support, below the
  # 5.x TokenizersBackend regression; sentencepiece/tiktoken/protobuf drive the
  # slow-to-fast tokenizer conversion when the model dir has no tokenizer.json;
  # numpy stays on 1.x so the +cpu torch C extensions keep importing. triton is
  # excluded on purpose: it must stay the vendor triton-metax linked above.
  python -m pip install \
    pytest "transformers>=4.51,<5" "numpy<2" safetensors sentencepiece tiktoken protobuf

  # Verify triton-metax is available (searched /opt/conda, /opt/vendor-torch,
  # /usr and /usr/local site-packages). Stock PyPI triton is not a substitute.
  python .github/scripts/check_integration_deps.py \
    --require pytest transformers safetensors triton
fi

if [[ -n "${GITHUB_PATH:-}" ]]; then
  printf '%s\n' \
    /opt/venv/bin \
    /opt/maca/tools/cu-bridge/bin \
    /opt/maca/mxgpu_llvm/bin \
    /opt/maca/bin >> "$GITHUB_PATH"
fi

if [[ -n "${GITHUB_ENV:-}" ]]; then
  printf '%s=%s\n' PATH "$PATH" >> "$GITHUB_ENV"
  for name in \
    VIRTUAL_ENV PYTHONNOUSERSITE ACCELERATOR METAX_PATH MACA_PATH MACA_HOME \
    FLAGOS_METAX_BOXING FLAGOS_METAX_CUDART_SHIM \
    FLAGOS_DISABLE_CUDA_ASSETS FLAGOS_USE_FLAGGEMS \
    FLAGGEMS_KERNEL FLAGGEMS_PYTHON FLAGOS_WHEEL_LOCAL \
    FLAGOS_MACA_TORCH_LIB LD_LIBRARY_PATH LIBRARY_PATH CPATH; do
    printf '%s=%s\n' "$name" "${!name}" >> "$GITHUB_ENV"
  done
fi

cd "$REPO_ROOT"

if [[ "$CI_STAGE" == "build" || "$CI_STAGE" == "integration" ]]; then
  # setuptools collects package_data before build_ext on the first wheel build.
  # Prebuilding makes torch_fl/lib/*.so available when the common workflow
  # packages the local wheel for either stage. The following python -m build
  # is incremental.
  python setup.py build_ext --inplace
fi

# Populate the ignored package-data directory after the optional native
# prebuild. This also normalizes the native RPATHs before wheel/editable install.
bash scripts/bundle_maca_libtorch.sh
