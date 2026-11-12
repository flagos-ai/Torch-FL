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

# PPU (T-Head Jianwu ZW810E) environment bootstrap for CI.
#
# PPU is a CUDA-ABI boxing backend with its own FLAGOS_ACCELERATOR value: the toolchain
# is CUDA (PPU torch is a local USE_CUDA=1 build whose libtorch_cpu.so provides
# ~2092 undefined symbols of libtorch_fl.so), but the vendor is PPU, so the
# build bundles its libtorch into lib_thead/. The stock CPU wheel's core libs must
# be selected from the PPU build at import time. This script:
#   1. Validates PPU_SDK and the vendor torch assets.
#   2. Builds an isolated venv with stock CPU torch 2.10.0 (link target).
#   3. Reuses matching wheels by RECORD; installs missing published wheels.
#   4. Exports FLAGOS_ACCELERATOR=ppu + PPU_SDK + the two CUDA-assets kill switches.
#   5. Preserves the existing native build; only boxing bundles vendor libtorch.
#
# Nothing here reads a host bind mount, so the environment is fully reproducible
# from the image plus this script's indexes.
#
# Core selection itself is automatic at `import torch_fl` time, gated on
# lib_thead/libtorch_cuda.so existing; the FLAGOS_ACCELERATOR value is what selects
# backends_thead.conf. No mode env var is involved.

# SDK startup has already run in the image. Child shells must not restore its
# vendor Python/compiler paths after CI activates the isolated environment.
export BASH_ENV=/dev/null
export PYTHONNOUSERSITE=1
export PYTHONPATH=""
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
CPU_TORCH_VERSION="${TORCH_FL_CPU_TORCH_VERSION:-$CPU_TORCH_VERSION_DEFAULT}"
# Use the official indexes by default: the PPU runner pod's HTTP proxy returns
# 500 on HTTPS CONNECT to *.tuna.tsinghua.edu.cn, so the Tsinghua PyPI and
# pytorch-wheels mirrors are unreachable from the pod (the proxy whitelists
# pypi.org and download.pytorch.org, which CI was green on before). The PPU
# image's own pip.conf points at an internal mirror that 503s, so the explicit
# index-url here bypasses it. Both stay overridable via env: flip to the
# Tsinghua mirrors once the pod proxy allows them.
#
# Setting this is not enough on its own, and that is why these lanes still saw
# Tsinghua. install_cpu_torch probes a *second* index -- the shared
# CPU_TORCH_MIRROR_INDEX_URL_DEFAULT -- and prefers it when it measures faster.
# That policy is unchanged; the default it points at is now mirror.nju.edu.cn,
# because mirror.sjtu.edu.cn redirected into the one host this comment set out
# to avoid, which is how a filelock 403 failed 6 of 14 consecutive runs across
# the thead and thead-boxing lanes. The fallback in install_cpu_torch still
# returns to this pin when the mirror does not serve the wheel, and the mirror
# stays enabled, because it is also what keeps the 146-190 MB CPU-torch wheel
# from taking 17-21 minutes.
CPU_TORCH_INDEX_URL="${TORCH_FL_CPU_TORCH_INDEX_URL:-$CPU_TORCH_INDEX_URL_DEFAULT}"
PIP_INDEX_URL="${TORCH_FL_PIP_INDEX_URL:-$PIP_INDEX_URL_DEFAULT}"

# FlagTree is the FlagOS Triton distribution, and its `ppu` variant *is* the
# `triton` package: the wheel ships triton/ (with triton/FLAGTREE_BACKEND =
# ppu), not a triton plugin, so it replaces the vendor triton rather than
# sitting beside it. cp312 only, hosted on the FlagOS index. The runner's HTTP
# proxy does not allowlist that index (it answers CONNECT with 500), so the
# install below reaches it directly -- see prefer_direct_route. Install it as:
#   python3.12 -m pip install flagtree===0.7.0+ppu3.6 \
#     --index-url=https://resource.flagos.net/repository/flagos-pypi-hosted/simple
FLAGTREE_VERSION="${TORCH_FL_FLAGTREE_VERSION:-$FLAGTREE_VERSION_thead}"
FLAGTREE_INDEX_URL="${TORCH_FL_FLAGTREE_INDEX_URL:-$FLAGTREE_INDEX_URL_DEFAULT}"

# The T-Head index publishes a Python wheel for the pinned FlagGems release.
FLAGGEMS_VERSION="${TORCH_FL_FLAGGEMS_VERSION:-$FLAGGEMS_VERSION_DEFAULT}"
FLAGGEMS_INDEX_URL="${TORCH_FL_FLAGGEMS_INDEX_URL:-$FLAGOS_WHEEL_ROOT_DEFAULT/flagos-pypi-thead/simple}"

# The runtime SDK lives under /usr/local/PPU_SDK. Keep the legacy hyphenated
# location as a local-development fallback. A preset PPU_SDK wins; scan candidates
# and pick the first with CUDA_SDK/lib64/libcudart.so. Mirrors
# set_env_ascend.sh's CANN toolkit scan so layout changes do not break the env.
_ppu_candidates=(
  "${PPU_SDK:-}"
  /usr/local/PPU_SDK
  /usr/local/PPU-SDK
  /opt/PPU-SDK
)
PPU_SDK=""
for _cand in "${_ppu_candidates[@]}"; do
  [[ -z "$_cand" ]] && continue
  if [[ -d "$_cand" && -e "$_cand/CUDA_SDK/lib64/libcudart.so" ]]; then
    PPU_SDK="$_cand"
    break
  fi
done
if [[ -z "$PPU_SDK" ]]; then
  echo "::error::PPU SDK not found; none of the candidates had CUDA_SDK/lib64/libcudart.so."
  echo "::error::Tried: ${_ppu_candidates[*]}. Set PPU_SDK to the SDK root."
  exit 1
fi

select_vendor_python() {
  local candidate="${TORCH_FL_VENDOR_PYTHON:-}"
  if [[ -n "$candidate" && "$candidate" != */* ]]; then
    candidate="$(command -v "$candidate" 2>/dev/null || true)"
  fi
  if [[ -z "$candidate" ]]; then
    candidate="$(command -v python 2>/dev/null || true)"
  fi
  if [[ -z "$candidate" || ! -x "$candidate" ]]; then
    echo "::error::Unable to find the vendor Python interpreter" >&2
    exit 1
  fi
  if ! "$candidate" -I -c "import torch" >/dev/null 2>&1; then
    echo "::error::Vendor Python cannot import torch: $candidate" >&2
    exit 1
  fi
  printf '%s' "$candidate"
}

VENDOR_PYTHON="$(select_vendor_python)"
VENDOR_INFO="$("$VENDOR_PYTHON" -I - <<'PY'
import json
from pathlib import Path
import site
import sys

import torch

print(json.dumps({
    "version": torch.__version__,
    "base_version": torch.__version__.split("+", 1)[0],
    "cuda": torch.version.cuda,
    "root": str(Path(torch.__file__).resolve().parent),
    "site": site.getsitepackages()[0],
    "python": sys.version.split()[0],
}, sort_keys=True))
PY
)"

readarray -t VENDOR_FIELDS < <(
  VENDOR_INFO="$VENDOR_INFO" "$VENDOR_PYTHON" - <<'PY'
import json
import os

info = json.loads(os.environ["VENDOR_INFO"])
for key in ("version", "base_version", "cuda", "root", "site", "python"):
    print(info.get(key) or "")
PY
)
VENDOR_TORCH_VERSION="${VENDOR_FIELDS[0]}"
VENDOR_TORCH_BASE_VERSION="${VENDOR_FIELDS[1]}"
VENDOR_CUDA_VERSION="${VENDOR_FIELDS[2]}"
VENDOR_TORCH_ROOT="${VENDOR_FIELDS[3]}"
VENDOR_SITE="${VENDOR_FIELDS[4]}"
VENDOR_PYTHON_VERSION="${VENDOR_FIELDS[5]}"
VENDOR_TORCH_LIB="$VENDOR_TORCH_ROOT/lib"

VENDOR_NVIDIA_LIBS=""
for nvidia_lib in "$VENDOR_SITE"/nvidia/*/lib; do
  [[ -d "$nvidia_lib" ]] || continue
  VENDOR_NVIDIA_LIBS="${VENDOR_NVIDIA_LIBS:+$VENDOR_NVIDIA_LIBS:}$nvidia_lib"
done

if [[ "$VENDOR_TORCH_BASE_VERSION" != "$CPU_TORCH_VERSION" ]]; then
  echo "::error::Vendor torch is $VENDOR_TORCH_VERSION; expected $CPU_TORCH_VERSION"
  exit 1
fi
if [[ "$VENDOR_CUDA_VERSION" != "13.0" ]]; then
  echo "::error::Vendor torch CUDA runtime is $VENDOR_CUDA_VERSION; expected 13.0"
  exit 1
fi

echo "Vendor Python: $VENDOR_PYTHON ($VENDOR_PYTHON_VERSION)"
echo "Vendor PyTorch: $VENDOR_TORCH_VERSION"
echo "Vendor torch root: $VENDOR_TORCH_ROOT"

# PPU build_ext runs with FLAGOS_BUILD_FLAGGEMS_CPP=OFF (setup.py cuda-branch default), so
# the C++ FlagGems dispatch (which needs liboperators.so + FlagGemsConfig.cmake)
# is never linked and find_package(FlagGems) is skipped. FLAGOS_BUILD_FLAGGEMS=ON
# compiles the Python-path kernels without importing flag_gems at build time, so
# the cuda-style FlagGems C++ asset discovery is omitted here. The runtime
# flag_gems import for the FlagGems test step comes from the venv install further
# down, not from the container filesystem.

# PPU core libs are a local USE_CUDA=1 build, not an upstream wheel. They carry
# the undefined symbols libtorch_fl.so needs, so they must be bundled and later
# selected via a private torch facade at import time.
for path in \
  "$VENDOR_TORCH_LIB/libc10.so" \
  "$VENDOR_TORCH_LIB/libtorch_cpu.so" \
  "$VENDOR_TORCH_LIB/libtorch.so" \
  "$VENDOR_TORCH_LIB/libtorch_global_deps.so" \
  "$VENDOR_TORCH_LIB/libtorch_python.so" \
  "$VENDOR_TORCH_LIB/libc10_cuda.so" \
  "$VENDOR_TORCH_LIB/libtorch_cuda.so"; do
  if [[ ! -e "$path" ]]; then
    echo "::error::Required PPU torch asset is missing: $path"
    exit 1
  fi
done

# Device nodes: three classes on the runner -- /dev/alixpu (base),
# /dev/alixpu_ctl (control), /dev/alixpu_ppu0-15 (compute). The CI container
# mounts all of them via container_options --device=...; probe alixpu_ppu0 as
# the "devices are visible" canary.
if [[ ! -c /dev/alixpu_ppu0 ]]; then
  echo "::error::PPU device node /dev/alixpu_ppu0 is unavailable"
  exit 1
fi

# PPU torch ships its own libtorch_cuda.so, so neither the build-time copy into
# torch_fl/lib nor the runtime preload should run. Both kill switches are needed:
#   FLAGOS_SKIP_CUDA_ASSETS  -> setup.py skips the libtorch_cuda.so copy + cu12 deps
#   FLAGOS_DISABLE_CUDA_ASSETS -> torch_fl skips the runtime ctypes preload
export FLAGOS_SKIP_CUDA_ASSETS=1
export FLAGOS_DISABLE_CUDA_ASSETS=1

# PPU runs both builds of the wheel, and BOXING_LANE says which one this job is.
#
# The boxing lane selects the day-0 conf, the one whose second route is the
# boxing kernel that wheel carries: backends_thead.conf leaves the 1445 ops this
# build registers without a FlagGems kernel on `none`, which on a boxing wheel
# is a raise rather than a cpu_fallback. Exported here, before the build,
# because it also travels to the integration runner through GITHUB_ENV and is
# what the tests' `routed_backend()` reads. The vendor lane selects nothing: the
# build record names the platform and the runtime picks backends_thead.conf.
if [[ "${BOXING_LANE:-}" == "1" ]]; then
  export FLAGOS_BACKEND_CONFIG="${FLAGOS_BACKEND_CONFIG:-$REPO_ROOT/torch_fl/configs/backends_thead_boxing.conf}"
  if [[ ! -f "$FLAGOS_BACKEND_CONFIG" ]]; then
    echo "::error::the PPU boxing lane needs the day-0 conf $FLAGOS_BACKEND_CONFIG"
    exit 1
  fi
fi

VENV_ROOT="${TORCH_FL_VENV_ROOT:-${RUNNER_TEMP:-$REPO_ROOT/.ci}/torch-fl-ppu-${CI_STAGE}${BOXING_LANE:+-boxing}}"
"$VENDOR_PYTHON" -I - "$VENV_ROOT" "$REPO_ROOT" <<'PYCODE'
from pathlib import Path
import sys

target, repo = (Path(p).resolve() for p in sys.argv[1:])
for prefix in (sys.prefix, sys.base_prefix):
    source = Path(prefix).resolve()
    if source.is_relative_to(target) or target.is_relative_to(source):
        raise SystemExit("CI venv must be separate from the runtime Python prefixes")
if repo.is_relative_to(target):
    raise SystemExit("CI venv must not contain the source checkout")
assert sys.version_info[:2] == (3, 12), "PPU FlagTree requires Python 3.12"
PYCODE
if ! "$VENDOR_PYTHON" -m venv --clear "$VENV_ROOT"; then
  if ! command -v uv >/dev/null 2>&1; then
    echo "::error::Python venv creation failed and the runtime has no uv fallback" >&2
    exit 1
  fi
  uv venv --clear --seed --python "$VENDOR_PYTHON" "$VENV_ROOT"
fi
VENV_PYTHON="$VENV_ROOT/bin/python"
if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "::error::Isolated Python was not created at $VENV_ROOT"
  exit 1
fi

"$VENV_PYTHON" - "$VENV_ROOT" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
assert Path(sys.prefix).resolve() == root
assert sys.prefix != sys.base_prefix
settings = {}
for line in (root / "pyvenv.cfg").read_text().splitlines():
    key, separator, value = line.partition("=")
    if separator:
        settings[key.strip().lower()] = value.strip().lower()
assert settings.get("include-system-site-packages") == "false"
PY

# The pod's proxied egress to files.pythonhosted.org has been observed at
# ~90 kB/s, well below what pip's 15s default read timeout tolerates for a
# single wheel (e.g. the ~30 MB cmake wheel stalled mid-download and tripped
# ReadTimeoutError). Raise it for every venv pip install below so a slow but
# still-progressing download is not killed early; this is independent of the
# index-url choice and applies regardless of which index ends up serving it.
export PIP_DEFAULT_TIMEOUT=120

# Selectively copy RECORD-owned wheels. Never expose /flagos or /opt/flagtree
# on the target import path, and never copy vendor Torch or editable finders.
"$VENDOR_PYTHON" "$REPO_ROOT/.github/scripts/seed_runtime.py" \
  --target-python "$VENV_PYTHON" \
  --torch-version "${CPU_TORCH_VERSION%+cpu}" \
  --flagtree-version "$FLAGTREE_VERSION" \
  --flaggems-version "$FLAGGEMS_VERSION" \
  --requirements-file "$REPO_ROOT/.github/requirements-thead.txt"

ppu_runtime_requirement() {
  if ! "$VENV_PYTHON" "$REPO_ROOT/.github/scripts/seed_runtime.py" \
      --check-installed "$1"; then
    pip_retry --index-url "$PIP_INDEX_URL" "$1"
  fi
  "$VENV_PYTHON" "$REPO_ROOT/.github/scripts/seed_runtime.py" \
    --check-installed "$1"
}

while IFS= read -r requirement || [[ -n "$requirement" ]]; do
  [[ -z "$requirement" || "$requirement" == \#* ]] && continue
  ppu_runtime_requirement "$requirement"
done < "$REPO_ROOT/.github/requirements-thead.txt"
if ! "$VENV_PYTHON" - "$CPU_TORCH_VERSION" <<'PYCODE'
import sys
import torch

assert torch.__version__ == sys.argv[1].removesuffix("+cpu") + "+cpu"
assert torch.version.cuda is None
PYCODE
then
  install_cpu_torch
fi

ppu_runtime_flagtree_is_usable() {
  "$VENV_PYTHON" - "$FLAGTREE_VERSION" <<'PYCODE'
from importlib import metadata, util
from pathlib import Path
import sys

try:
    dist = metadata.distribution("flagtree")
    root = Path(dist.locate_file("triton")).resolve()
    spec = util.find_spec("triton")
    assert dist.version == sys.argv[1]
    assert root.is_relative_to(Path(sys.prefix).resolve())
    assert spec is not None and spec.origin
    assert Path(spec.origin).resolve().parent == root
    assert (root / "backends/ppu").is_dir()
except (metadata.PackageNotFoundError, AssertionError):
    raise SystemExit(1)
PYCODE
}

# Retain the existing integration-only compiler, operator and model packages.
if [[ "$CI_STAGE" == "integration" ]]; then
  for requirement in pytest 'transformers>=4.51,<5' sentencepiece tiktoken protobuf; do
    ppu_runtime_requirement "$requirement"
  done
  if ppu_runtime_flagtree_is_usable; then
    echo "Using prepared FlagTree $FLAGTREE_VERSION"
  else
    "$VENV_PYTHON" -m pip uninstall -y flagtree triton triton_kernels
    install_flagtree
  fi
  install_flag_gems
fi

CPU_TORCH_ROOT="$("$VENV_PYTHON" - <<'PY'
from pathlib import Path
import torch

print(Path(torch.__file__).resolve().parent)
assert torch.__version__.split("+", 1)[0] == "2.10.0", torch.__version__
assert torch.version.cuda is None, torch.version.cuda
PY
)"

export VIRTUAL_ENV="$VENV_ROOT"
export PATH="$VENV_ROOT/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONPATH=""
export FLAGOS_ACCELERATOR=ppu
# The boxing wheel: it stages the PPU torch wheel's libtorch into lib_thead/ and
# the lane measures it against the day-0 conf that exists for exactly this
# wheel (backends_thead_boxing.conf, selected below).
#
# This lane built the FlagGems-primary wheel for a while, and that wheel cannot
# run the gen_vendor_confs.py confs at all: `torch_fl` selects `backends_thead.conf`
# whenever the build record names the platform, and that conf names `flaggems`
# for 592 ops while a build without FLAGOS_BUILD_FLAGGEMS=1 registers no FlagGems
# wrapper -- measured on the PPU runner as `add.Tensor: routed to 'none' ... but
# the op is registered on PrivateUse1`. The default wheel's wheel-local build
# directory read *its own* conf, which is why it looked fine there; the
# installed-tree measurement is what failed.
#
# `boxing` is the route a boxing wheel serves without FlagGems, and the conf for
# it is the day-0 variant AGENTS.md describes: FlagGems where this build has it,
# boxing everywhere else. That is the combination the boxing lane measures; the
# vendor lane leaves FLAGOS_BUILD_BOXING unset and builds the default wheel the
# sentence above describes.
if [[ "${BOXING_LANE:-}" == "1" ]]; then
  export FLAGOS_BUILD_BOXING=1
fi
export CUDA_HOME="$PPU_SDK/CUDA_SDK"
export CUDA_PATH="$CUDA_HOME"
export PPU_SDK="$PPU_SDK"
export FLAGOS_VENDOR_TORCH_LIB="$VENDOR_TORCH_LIB"
export FLAGOS_VENDOR_TORCH_VERSION=""
if [[ "${BOXING_LANE:-}" == 1 ]]; then
  export FLAGOS_VENDOR_TORCH_VERSION="$VENDOR_TORCH_VERSION"
fi
export FLAGOS_SDK_VERSION="${PPU_VERSION:-}"
FLAGOS_SDK_VERSION="${FLAGOS_SDK_VERSION#v}"
if [[ "$FLAGOS_SDK_VERSION" != 2.1.0 ]]; then
  echo "::error::Expected runtime PPU_VERSION=v2.1.0, got ${PPU_VERSION:-unset}" >&2
  exit 1
fi
# FLAGOS_WHEEL_LOCAL is not set here: the wheel's local segment is the vendor
# name, taken from the wheel_local column of cmake/flagos_platforms.json
# (currently "ppu2.0.0"). The manifest records the actual SDK separately;
# changing the shared release naming policy is outside this image migration.
# Setting an override here would make the
# artifact name depend on the build script instead of the platform table.
# PPU runtime ships the Python FlagGems wheel without liboperators.so /
# FlagGemsConfig.cmake, so the C++ kFlagOs dispatch (FLAGOS_BUILD_FLAGGEMS_CPP) must be off.
# setup.py's `ppu` branch already forces it OFF; exporting it here keeps the
# environment consistent for the pre-build assertions and any manual cmake run.
# The Python-path kernels (FLAGOS_BUILD_FLAGGEMS) stay at the default ON -- they
# compile without importing flag_gems, and the FlagGems runtime test step needs
# them.
export FLAGOS_BUILD_FLAGGEMS_CPP=0
export FLAGCX_PATH="${FLAGCX_PATH:-/opt/FlagCX}"

CLEAN_CMAKE_PREFIX_PATH="$(strip_vendor_paths "${CMAKE_PREFIX_PATH:-}")"
CLEAN_LIBRARY_PATH="$(strip_vendor_paths "${LIBRARY_PATH:-}")"
CLEAN_LD_LIBRARY_PATH="$(strip_vendor_paths "${LD_LIBRARY_PATH:-}")"
export CMAKE_PREFIX_PATH="$CPU_TORCH_ROOT/share/cmake${CLEAN_CMAKE_PREFIX_PATH:+:$CLEAN_CMAKE_PREFIX_PATH}"
export CPATH="$CUDA_HOME/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$CUDA_HOME/lib64:$PPU_SDK/lib:$PPU_SDK/lib64${CLEAN_LIBRARY_PATH:+:$CLEAN_LIBRARY_PATH}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$PPU_SDK/lib:$PPU_SDK/lib64${VENDOR_NVIDIA_LIBS:+:$VENDOR_NVIDIA_LIBS}:$CPU_TORCH_ROOT/lib${CLEAN_LD_LIBRARY_PATH:+:$CLEAN_LD_LIBRARY_PATH}"

"$VENV_PYTHON" -m pip check
"$VENV_PYTHON" "$REPO_ROOT/.github/scripts/seed_runtime.py" --report-environment

cd "$REPO_ROOT"
if [[ "$CI_STAGE" == "build" || "$CI_STAGE" == "integration" ]]; then
  # Prebuild so package_data sees libtorch_fl.so and the bundled PPU assets
  # before the common workflow invokes python -m build.
  build_flagos_inplace
fi

# Bundle PPU core+CUDA+MKL .so into torch_fl/lib_thead/ and rewrite the plugin
# RPATH. setup.py does not auto-invoke this; mirrors the metax pattern. The
# bundle is idempotent and must run after build_ext so libtorch_fl.so exists.
#
# Boxing lane only. `torch_fl` recognises a PPU wheel by `lib_thead/libtorch_cuda.so`
# (torch_fl/__init__.py:_is_ppu_build), and the presence of that dir is what makes
# import select the PPU core and reconcile `torch.version.cuda`. The vendor lane
# is the default FlagGems-primary wheel, which ships neither (setup.py honours
# the PPU row's `vendor_torch_gate: FLAGOS_BUILD_BOXING`); bundling here made
# that lane a PPU-core wheel and its environment check read cuda "13.0" on a
# wheel advertised as stock +cpu torch (run 40168685639).
if [[ "${BOXING_LANE:-}" == "1" ]]; then
  bash scripts/vendor/bundle_thead_libtorch.sh
fi

if command -v ppu-smi >/dev/null 2>&1; then
  ppu-smi
elif command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "::error::Neither ppu-smi nor nvidia-smi is available in the PPU runtime" >&2
  exit 1
fi

python - <<'PY'
import os
from pathlib import Path
import sys
import torch

torch_path = Path(torch.__file__).resolve()
assert sys.executable.startswith("/"), sys.executable
assert torch.__version__.split("+", 1)[0] == "2.10.0", torch.__version__
assert torch.version.cuda is None, torch.version.cuda
assert "/opt/conda/" not in str(torch_path), torch_path
if os.environ.get("FLAGOS_BUILD_BOXING") == "1":
    assert Path("torch_fl/lib_thead/libtorch_cuda.so").is_file()
print(f"Isolated Python: {sys.executable}")
print(f"CPU PyTorch: {torch.__version__}")
print(f"CPU torch path: {torch_path}")
print(f"PPU bundle: {Path('torch_fl/lib_thead').resolve()}")
PY

# Prove the installed FlagGems stack is importable and wired to PPU, so a broken
# install fails here instead of turning into a wall of identical failures in the
# FlagGems runtime step. flag_gems queries torch.cuda at import time, so torch_fl
# must be imported first: that is what swaps the stock CPU core libs for the PPU
# build bundled above. The check therefore has to run after
# bundle_thead_libtorch.sh, and only in the integration stage, which is where
# flagtree and flag_gems are installed.
#
# The three assertions are the ones that actually went wrong while bringing this
# up: the installed `triton` must be FlagTree's (its registry exposes the `ppu`
# backend, the vendor build exposes ['amd', 'nvidia']), flag_gems must have picked
# the `thead` vendor from PPU_SDK rather than falling back to nvidia, and it must
# resolve to the venv rather than the image's editable-install path.
if [[ "$CI_STAGE" == "integration" ]]; then
  python - <<'PY'
import hashlib
import importlib.util
import os
from pathlib import Path
import sys

lib = Path(importlib.util.find_spec("torch").submodule_search_locations[0]) / "lib"
before = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
          for path in lib.iterdir() if path.is_file()}
import torch_fl  # noqa: F401  (selects the bundled PPU libtorch core)
import flag_gems
import triton
import triton.backends

venv = sys.prefix
assert str(Path(triton.__file__).resolve()).startswith(venv), triton.__file__
assert "ppu" in triton.backends.backends, list(triton.backends.backends)
assert str(Path(flag_gems.__file__).resolve()).startswith(venv), flag_gems.__file__

vendor = flag_gems.runtime.backend.device_finder.DeviceDetector()._get_vendor_from_env()
assert vendor == "thead", f"flag_gems selected vendor {vendor!r}; PPU_SDK={os.environ.get('PPU_SDK')!r}"

after = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
         for path in lib.iterdir() if path.is_file()}
assert after == before, "PPU import changed the installed torch/lib"

# Which build this lane made, and the routing half of the same contract. A
# boxing wheel registers every op on PrivateUse1, and a `none` route there
# raises instead of falling back, so its conf has to agree with the
# registration for the ops a model actually calls -- the day-0 conf is the one
# that does, where the default one leaves 1445 of them on `none` (measured on
# the PPU runner). The vendor lane selects the default conf and stages nothing.
import torch_fl._env as _env

kernels = _env.build_kernels()
conf = torch_fl.backend_config_path()
if os.environ.get("BOXING_LANE") == "1":
    assert "boxing" in kernels, kernels
    assert Path("torch_fl/lib_thead/libtorch_cuda.so").is_file(), "no PPU bundle"
    assert conf.endswith("backends_thead_boxing.conf"), conf
else:
    assert "boxing" not in kernels, kernels
    assert conf.endswith("backends_thead.conf"), conf
assert "flaggems" in kernels, kernels
print(f"triton (FlagTree): {triton.__version__} -> {triton.__file__}")
print(f"triton backends: {sorted(triton.backends.backends)}")
print(f"flag_gems: {flag_gems.__version__} -> {flag_gems.__file__}")
print(f"flag_gems vendor: {vendor}")
PY
fi

if [[ -n "${GITHUB_PATH:-}" ]]; then
  printf '%s\n' "$VENV_ROOT/bin" >> "$GITHUB_PATH"
fi
if [[ -n "${GITHUB_ENV:-}" ]]; then

  # The boxing lane assigns these; the vendor lane leaves them unset, and
  # export_ci_env reads every name under `set -u`. Assign the empty default here
  # so both lanes persist an explicit value.
  : "${FLAGOS_BUILD_BOXING:=}"
  : "${FLAGOS_BACKEND_CONFIG:=}"
  export_ci_env PATH VIRTUAL_ENV BASH_ENV TORCH_DEVICE_BACKEND_AUTOLOAD FLAGOS_SDK_VERSION FLAGOS_VENDOR_TORCH_VERSION PYTHONNOUSERSITE PYTHONPATH FLAGOS_ACCELERATOR FLAGOS_BUILD_BOXING FLAGOS_BACKEND_CONFIG CUDA_HOME CUDA_PATH PPU_SDK FLAGOS_VENDOR_TORCH_LIB FLAGOS_SKIP_CUDA_ASSETS FLAGOS_DISABLE_CUDA_ASSETS FLAGOS_BUILD_FLAGGEMS_CPP FLAGCX_PATH CMAKE_PREFIX_PATH CPATH LIBRARY_PATH LD_LIBRARY_PATH
fi
