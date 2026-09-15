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
export FLAGGEMS_PYTHON=1
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

if [[ ! -c /dev/mxcd ]]; then
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

# Expose the vendor Triton (triton-metax) and FlagGems to the CPU torch venv.
# torch.compile needs Triton: the active torch is the CPU wheel, which ships no
# Triton, so inductor raises TritonMissing without this. FlagGems is required
# because backends_metax.conf routes 451 ops to the Python FlagGems path by
# default (FLAGGEMS_PYTHON=1 above compiles the dispatcher slot).
#
# The vendor packages live in the image's MetaX torch install, which we
# otherwise deliberately do not use -- only libtorch is consumed, from
# /opt/vendor-libtorch.
#
# Linked rather than copied: the metax backend carries ~2.4GB of device
# libraries and a cp -a of that is pure CI wall time. set_env_cuda.sh copies
# because it also relocates FlagCX; here only Triton and FlagGems are needed.
if [[ "$CI_STAGE" == "integration" ]]; then
  VENV_SITE="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"

  # Install FlagGems' runtime dependencies into the venv. flag_gems imports
  # packaging, yaml (PyYAML), sqlalchemy and numpy at or shortly after import.
  # Use --no-deps to protect the torch ABI (same rationale as test-dependencies).
  # numpy<2 because numpy 2.x breaks the stock +cpu torch C extensions at import
  # (documented in set_env_musa.sh:176-178).
  python -m pip install --no-deps 'packaging>=20.0' 'PyYAML>=5.0' 'sqlalchemy>=1.4' 'numpy>=1.20,<2.0'

  VENDOR_SITE=""
  for candidate in /opt/conda/lib/python3.*/site-packages \
                   /opt/vendor-torch/lib/python3.*/site-packages \
                   /usr/lib/python3.*/site-packages \
                   /usr/local/lib/python3.*/site-packages; do
    if [[ -d "$candidate/triton" ]]; then
      VENDOR_SITE="$candidate"
      break
    fi
  done

  if [[ -z "$VENDOR_SITE" ]]; then
    echo "::error::Vendor Triton (triton-metax) was not found in the image;" \
         "torch.compile tests cannot run. Searched /opt/conda, /opt/vendor-torch," \
         "/usr and /usr/local site-packages."
    exit 1
  fi

  # Link triton
  if [[ ! -e "$VENV_SITE/triton" ]]; then
    ln -s "$VENDOR_SITE/triton" "$VENV_SITE/triton"
  fi
  for metadata in "$VENDOR_SITE"/triton-*.dist-info; do
    [[ -e "$metadata" ]] || continue
    [[ -e "$VENV_SITE/$(basename "$metadata")" ]] || ln -s "$metadata" "$VENV_SITE/"
  done

  # Discover flag_gems via interpreter query, not directory probe. FlagGems is
  # normally an editable install, so there is no site-packages/flag_gems directory
  # to test -- only a .pth file and a finder module pointing at a source tree.
  VENDOR_FLAGGEMS_ROOT=""
  for candidate_python in /opt/conda/bin/python3 /opt/conda/bin/python \
                          /opt/vendor-torch/bin/python3 /opt/vendor-torch/bin/python \
                          /usr/bin/python3 /usr/local/bin/python3; do
    [[ -x "$candidate_python" ]] || continue
    VENDOR_FLAGGEMS_ROOT="$("$candidate_python" - <<'PY'
import importlib.util
from pathlib import Path
spec = importlib.util.find_spec("flag_gems")
if spec is None or not spec.submodule_search_locations:
    print("")
else:
    root = Path(next(iter(spec.submodule_search_locations))).resolve()
    print(root if (root / "__init__.py").is_file() else "")
PY
)"
    [[ -n "$VENDOR_FLAGGEMS_ROOT" ]] && break
  done

  if [[ -z "$VENDOR_FLAGGEMS_ROOT" ]]; then
    echo "FlagGems not found in vendor interpreters. Installing from source..."
    # FlagGems is not available on PyPI. Install from GitHub.
    # Use a pinned ref for reproducibility (matching the baseline from docs/reference/operator-support.md).
    python -m pip install --no-deps git+https://github.com/FlagOpen/FlagGems.git@7fb49bad

    # After installation, resolve the package location in the venv itself
    VENDOR_FLAGGEMS_ROOT="$(python - <<'PY'
import importlib.util
from pathlib import Path
spec = importlib.util.find_spec("flag_gems")
if spec is None or not spec.submodule_search_locations:
    print("")
else:
    root = Path(next(iter(spec.submodule_search_locations))).resolve()
    print(root if (root / "__init__.py").is_file() else "")
PY
)"

    if [[ -z "$VENDOR_FLAGGEMS_ROOT" ]]; then
      echo "::error::Failed to install FlagGems from source. The MetaX backend" \
           "requires FlagGems because backends_metax.conf routes 451 ops to the" \
           "Python FlagGems path."
      exit 1
    fi
  fi

  # Link the resolved flag_gems root (if from vendor) or use directly (if pip installed)
  if [[ ! -e "$VENV_SITE/flag_gems" && "$VENDOR_FLAGGEMS_ROOT" != "$VENV_SITE"* ]]; then
    ln -s "$VENDOR_FLAGGEMS_ROOT" "$VENV_SITE/flag_gems"
  fi
  # Link dist-info metadata if it exists alongside the package (for non-editable installs)
  VENDOR_FLAGGEMS_PARENT="$(dirname "$VENDOR_FLAGGEMS_ROOT")"
  for metadata in "$VENDOR_FLAGGEMS_PARENT"/flag_gems-*.dist-info; do
    [[ -e "$metadata" ]] || continue
    [[ -e "$VENV_SITE/$(basename "$metadata")" ]] || ln -s "$metadata" "$VENV_SITE/"
  done

  # Confirm the vendor packages actually import against the CPU torch wheel,
  # rather than discovering failures at test time.
  python - <<'PY'
import triton
import flag_gems

print(f"Vendor Triton: {triton.__version__} ({triton.__file__})")
print(f"FlagGems: {flag_gems.__version__} ({flag_gems.__file__})")
PY
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
bash scripts/vendor/bundle_maca_libtorch.sh
