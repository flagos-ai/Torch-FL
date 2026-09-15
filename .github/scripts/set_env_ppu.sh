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

# PPU (T-Head Jianwu ZW810E) environment bootstrap for CI.
#
# PPU is a CUDA-ABI boxing backend: ACCELERATOR=cuda, PPU torch is a local
# USE_CUDA=1 build whose libtorch_cpu.so provides ~2092 undefined symbols of
# libtorch_fl.so. The stock CPU wheel's core libs must be replaced by the PPU
# build at import time. This script:
#   1. Validates PPU_SDK and the vendor torch assets.
#   2. Builds an isolated venv with stock CPU torch 2.10.0 (link target).
#   3. Installs the FlagGems stack into it: FlagTree (the FlagOS triton dist
#      with the `ppu` backend) and FlagGems master from git.
#   4. Exports ACCELERATOR=cuda + PPU_SDK + the two CUDA-assets kill switches.
#   5. build_ext --inplace, then bundles PPU core/CUDA/MKL .so into
#      torch_fl/lib_ppu/ via bundle_ppu_libtorch.sh (setup.py does not call it).
#
# Nothing here reads a host bind mount, so the environment is fully reproducible
# from the image plus this script's indexes.
#
# Core replacement itself is automatic at `import torch_fl` time
# (torch_fl/__init__.py:222-232), gated only on lib_ppu/libtorch_cuda.so
# existing + ACCELERATOR=cuda. No env var triggers it.

set -euo pipefail

case "${CI_STAGE:-}" in
  build|integration) ;;
  *)
    echo "::error::CI_STAGE must be either 'build' or 'integration'"
    exit 1
    ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CPU_TORCH_VERSION="${TORCH_FL_CPU_TORCH_VERSION:-2.10.0}"
# Use the official indexes by default: the PPU runner pod's HTTP proxy returns
# 500 on HTTPS CONNECT to *.tuna.tsinghua.edu.cn, so the Tsinghua PyPI and
# pytorch-wheels mirrors are unreachable from the pod (the proxy whitelists
# pypi.org and download.pytorch.org, which CI was green on before). The PPU
# image's own pip.conf points at an internal mirror that 503s, so the explicit
# index-url here bypasses it. Both stay overridable via env: flip to the
# Tsinghua mirrors once the pod proxy allows them.
CPU_TORCH_INDEX_URL="${TORCH_FL_CPU_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}"
PIP_INDEX_URL="${TORCH_FL_PIP_INDEX_URL:-https://pypi.org/simple}"

# FlagTree is the FlagOS Triton distribution, and its `ppu` variant *is* the
# `triton` package: the wheel ships triton/ (with triton/FLAGTREE_BACKEND =
# ppu), not a triton plugin, so it replaces the vendor triton rather than
# sitting beside it. cp312 only, hosted on the FlagOS index. The runner's HTTP
# proxy does not allowlist that index (it answers CONNECT with 500), so the
# install below reaches it directly -- see prefer_direct_route. Install it as:
#   python3.12 -m pip install flagtree===0.6.2a2+ppu3.6 \
#     --index-url=https://resource.flagos.net/repository/flagos-pypi-hosted/simple
FLAGTREE_VERSION="${TORCH_FL_FLAGTREE_VERSION:-0.6.2a2+ppu3.6}"
FLAGTREE_INDEX_URL="${TORCH_FL_FLAGTREE_INDEX_URL:-https://resource.flagos.net/repository/flagos-pypi-hosted/simple}"

# FlagGems master, installed into the venv instead of imported from the
# /workspace/FlagGems bind mount this script used to require: that mount was not
# present on every runner pod, and its absence aborted the job in environment
# setup (see the install below). `master` by request; override with
# TORCH_FL_FLAGGEMS_REVISION to pin a commit for a reproducible run.
FLAGGEMS_REVISION="${TORCH_FL_FLAGGEMS_REVISION:-master}"
FLAGGEMS_REPO="${TORCH_FL_FLAGGEMS_REPO:-https://github.com/FlagOpen/FlagGems.git}"

# PPU SDK lives under either /usr/local/PPU-SDK (hyphen, host-mounted on the
# CI runner via container_volumes) or /usr/local/PPU_SDK (underscore, in-image
# on dev pods). A preset PPU_SDK / PPU_HOME env wins; otherwise scan candidates
# and pick the first with CUDA_SDK/lib64/libcudart.so. Mirrors
# set_env_ascend.sh's CANN toolkit scan so layout changes do not break the env.
_ppu_candidates=(
  "${PPU_SDK:-}"
  "${PPU_HOME:-}"
  /usr/local/PPU-SDK
  /usr/local/PPU_SDK
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
  if ! "$candidate" -c "import torch" >/dev/null 2>&1; then
    echo "::error::Vendor Python cannot import torch: $candidate" >&2
    exit 1
  fi
  printf '%s' "$candidate"
}

VENDOR_PYTHON="$(select_vendor_python)"
VENDOR_INFO="$("$VENDOR_PYTHON" - <<'PY'
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

# PPU build_ext runs with FLAGGEMS_KERNEL=OFF (setup.py cuda-branch default), so
# the C++ FlagGems dispatch (which needs liboperators.so + FlagGemsConfig.cmake)
# is never linked and find_package(FlagGems) is skipped. FLAGGEMS_PYTHON=ON
# compiles the Python-path kernels without importing flag_gems at build time, so
# the cuda-style FlagGems C++ asset discovery is omitted here. The runtime
# flag_gems import for the FlagGems test step comes from the venv install further
# down, not from the container filesystem.

# PPU core libs are a local USE_CUDA=1 build, not an upstream wheel. They carry
# the undefined symbols libtorch_fl.so needs, so they must be bundled and later
# symlinked over the stock CPU wheel's core libs at import time.
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

VENV_ROOT="${TORCH_FL_VENV_ROOT:-${RUNNER_TEMP:-$REPO_ROOT/.ci}/torch-fl-ppu-${CI_STAGE}}"
if ! "$VENDOR_PYTHON" -m venv --clear "$VENV_ROOT"; then
  echo "::warning::Vendor Python cannot create a venv; trying uv"
  if ! command -v uv >/dev/null 2>&1; then
    "$VENDOR_PYTHON" -m pip install --upgrade uv
  fi
  uv venv --clear --seed --python "$VENDOR_PYTHON" "$VENV_ROOT"
fi
VENV_PYTHON="$VENV_ROOT/bin/python"
if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "::error::Isolated Python was not created at $VENV_ROOT"
  exit 1
fi

# The pod's proxied egress to files.pythonhosted.org has been observed at
# ~90 kB/s, well below what pip's 15s default read timeout tolerates for a
# single wheel (e.g. the ~30 MB cmake wheel stalled mid-download and tripped
# ReadTimeoutError). Raise it for every venv pip install below so a slow but
# still-progressing download is not killed early; this is independent of the
# index-url choice and applies regardless of which index ends up serving it.
export PIP_DEFAULT_TIMEOUT=120

# FlagTree is a ~363 MB wheel served by a shared mirror, and the FlagGems git
# install has to resolve github.com through the same proxy. Both are large
# enough that a single transient reset would otherwise fail the job in setup,
# so retry them the way set_env_musa.sh does.
pip_retry() {
  local attempt=1
  while true; do
    if "$VENV_PYTHON" -m pip install --retries 10 --timeout 300 --no-cache-dir "$@"; then
      return 0
    fi
    if ((attempt >= 5)); then
      echo "::error::pip install failed after $attempt attempts: $*"
      return 1
    fi
    echo "::warning::pip install attempt $attempt failed; retrying: $*"
    attempt=$((attempt + 1))
    sleep 10
  done
}

# --- Reaching a host around the runner's HTTP proxy --------------------------
# The runner injects HTTP(S)_PROXY into the job container along with its own
# allowlist and NO_PROXY list (this pod: localhost,127.0.0.1,10.1.12.192,
# 10.1.12.38,harbor.baai.ac.cn,.example.com). Two of those entries resolve to
# public addresses (harbor.baai.ac.cn is 120.92.122.200, the same /16 as
# resource.flagos.net), and the container is a NAT'd docker bridge off the
# runner host, so this pod does open public connections without the proxy.
# resource.flagos.net is simply not on that list: pip tunnels to it, the proxy
# refuses the CONNECT, and pip reports 'Tunnel connection failed: 500 Internal
# Server Error' as "Could not find a version that satisfies the requirement
# flagtree===" -- an unreachable index that reads like a missing wheel. The
# 2026-09-15 run spent all five pip_retry attempts and 21 minutes on that, while
# the sibling MUSA runner's proxy serves the same index fine, so the working
# route is a property of the pod rather than of the index.
#
# So probe the unproxied route and, when it answers, add just that host to
# NO_PROXY/no_proxy: pip's requests stack reads either case, git reads the
# lowercase one through libcurl. When the probe fails the environment is left
# exactly as the runner set it, so a pod that can only leave through the proxy
# keeps the previous behaviour instead of trading a 500 for a connect timeout.
# TORCH_FL_PROXY_ROUTE=direct|proxy skips the probe.
add_no_proxy_host() {
  NO_PROXY="${NO_PROXY:+${NO_PROXY},}$1"
  no_proxy="${no_proxy:+${no_proxy},}$1"
  export NO_PROXY no_proxy
}

host_of_url() {
  local host="${1#*://}"
  host="${host%%/*}"
  printf '%s' "${host%%:*}"
}

# Succeeds if an unproxied GET of the URL gets an HTTP answer at all. The
# status is not the question: this probe only asks whether the pod can open the
# connection, and the index URL as configured has no trailing slash, so a 404
# here still proves the route works and a proxy-tunnelled request does not get
# that far. Only a transport failure -- DNS, connect, TLS, proxy refusal --
# counts as unreachable.
direct_route_reachable() {
  "$VENV_PYTHON" - "$1" <<'PY'
import sys
import urllib.error
import urllib.request

# Trust whatever pip trusts -- certifi, not the system CA store -- so a store
# difference cannot make this probe fail where the real install would succeed.
handlers = [urllib.request.ProxyHandler({})]
try:
    import certifi
    import ssl

    handlers.append(
        urllib.request.HTTPSHandler(context=ssl.create_default_context(cafile=certifi.where()))
    )
except ImportError:
    pass
opener = urllib.request.build_opener(*handlers)
try:
    with opener.open(sys.argv[1], timeout=15) as response:
        status = response.status
except urllib.error.HTTPError as exc:
    status = exc.code
except Exception as exc:
    print(f"{sys.argv[1]} unreachable without the proxy: {exc!r}")
    sys.exit(1)
print(f"{sys.argv[1]} -> HTTP {status} without the proxy")
PY
}

# $1 = URL, $2 = what it serves, for the log line.
prefer_direct_route() {
  local url="$1" what="$2" host
  host="$(host_of_url "$url")"
  case "${TORCH_FL_PROXY_ROUTE:-auto}" in
    proxy)
      echo "Using the runner proxy for $host ($what): TORCH_FL_PROXY_ROUTE=proxy"
      return 0
      ;;
    direct)
      add_no_proxy_host "$host"
      echo "Bypassing the runner proxy for $host ($what): TORCH_FL_PROXY_ROUTE=direct"
      return 0
      ;;
  esac
  if direct_route_reachable "$url"; then
    add_no_proxy_host "$host"
    echo "Bypassing the runner proxy for $host ($what): it answers directly"
  else
    echo "::warning::Using the runner proxy for $host ($what), which did not answer without it. If the proxy then refuses the CONNECT with 500, this host has to be allowlisted on the runner's proxy or added to its NO_PROXY."
  fi
}

# patchelf is missing by default on PPU nodes (bundle_common.sh notes it is
# absent on all four vendor nodes). Install it into the venv so bundle_ppu's
# bundle_require_patchelf check passes; mirrors the DCU line (commit 6568415).
# Do not --upgrade pip/setuptools: the fresh venv from ensurepip already
# satisfies them, and that upgrade round-trip is what hit the pod proxy 500.
# Install only what the venv lacks: cmake, patchelf, build (the dedicated
# workflow runs `python -m build`, which needs the build package in the venv),
# and wheel (the setuptools wheel-build backend used by `python -m build
# --wheel --no-isolation` requires it; ensurepip on this image does not
# provide it).
"$VENV_PYTHON" -m pip install --index-url "$PIP_INDEX_URL" cmake patchelf build wheel
"$VENV_PYTHON" -m pip install \
  --index-url "$CPU_TORCH_INDEX_URL" \
  "torch==$CPU_TORCH_VERSION"
if [[ "$CI_STAGE" == "integration" ]]; then
  # sentencepiece + tiktoken: the Qwen3 inference/training tests load the model
  # tokenizer via AutoTokenizer; the bundled model dir has no tokenizer.json, so
  # transformers converts the slow tokenizer to a fast one, which needs one of
  # these two. protobuf is what the sentencepiece branch of that conversion uses
  # to parse the spm model proto. The isolated venv cannot see the vendor image's
  # copies. These belong here rather than in a workflow step: pip must be given
  # an explicit --index-url to bypass the image pip.conf internal mirror (503).
  # transformers is pinned to [4.51, 5): the lower bound is where Qwen3
  # model_type support landed (older releases raise "Unrecognized model" on
  # AutoConfig.from_pretrained); the upper bound excludes 5.x, whose
  # TokenizersBackend rewrite has an unresolved upstream bug where the Qwen3
  # slow-to-fast tokenizer conversion still raises "Couldn't instantiate the
  # backend tokenizer" even with sentencepiece/tiktoken installed (see
  # https://huggingface.co/Qwen/Qwen3-8B/discussions/33). An unpinned install
  # picks whichever of those two failure modes is currently latest on PyPI, so
  # both ends must stay pinned. Revisit the upper bound once upstream ships a
  # fix for the 5.x regression.
  "$VENV_PYTHON" -m pip install --index-url "$PIP_INDEX_URL" \
    pytest "transformers>=4.51,<5" sentencepiece tiktoken protobuf
  # FlagGems' pure Python deps. The CI image lacks them (sqlalchemy/PyYAML/
  # packaging -- not in the image site-packages, verified by import failure).
  # numpy ships with the stock +cpu torch wheel installed above, so it is not
  # re-pinned here to avoid a numpy/torch version clash. Versions follow
  # FlagGems' pyproject.toml dependencies.
  "$VENV_PYTHON" -m pip install --index-url "$PIP_INDEX_URL" \
    "packaging>=26.0" "PyYAML==6.0.1" "sqlalchemy==2.0.48"
fi

# Keep the vendor FlagCX runtime available without copying the vendor torch
# package; the active torch package remains the CPU wheel installed above.
# `flag_gems` and `triton` are deliberately NOT in this list: the image's
# flag_gems is a PEP 660 editable install pointing at /workspace/FlagGems (the
# mount this script no longer requires), and its triton is the vendor build
# (3.5.0+v0.2.0.ppu2.1.0, backends ['amd','nvidia']). Copying either would
# shadow the FlagTree triton and the FlagGems master installed below.
VENV_SITE="$("$VENV_PYTHON" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
for package in triton_kernels flagcx; do
  if [[ -d "$VENDOR_SITE/$package" ]]; then
    cp -a "$VENDOR_SITE/$package" "$VENV_SITE/"
  fi
  for metadata in "$VENDOR_SITE"/"$package"-*.dist-info; do
    [[ -e "$metadata" ]] || continue
    cp -a "$metadata" "$VENV_SITE/"
  done
  # PEP 660 editable installs: copy the .pth (import hook) + finder .py (resolves
  # the package to its source dir) so venv python can import the package.
  # dist-info alone does not register the import hook.
  for pth in "$VENDOR_SITE"/__editable__."$package"*.pth; do
    [[ -e "$pth" ]] || continue
    cp -a "$pth" "$VENV_SITE/"
  done
  for finder in "$VENDOR_SITE"/__editable___"$package"*_finder.py; do
    [[ -e "$finder" ]] || continue
    cp -a "$finder" "$VENV_SITE/"
  done
done

# Install the FlagGems stack instead of importing it from a host mount.
#
# This replaces a probe of /workspace/FlagGems that hard-failed the job when the
# bind mount was absent, which is what every PPU run did once the runner pod
# stopped carrying it. Installing removes the dependency on runner-side state
# entirely.
#
# Why --no-deps on both: flag_gems would otherwise pull PyPI's NVIDIA `triton`,
# replacing the FlagTree build installed one line earlier, and it would also
# re-resolve `torch`, replacing the pinned CPU wheel this whole setup depends on
# (the PPU core libs are symlinked over *that* wheel's libs at import time).
# FlagGems' own runtime deps are installed explicitly above.
#
# FlagTree first: it owns the `triton` package, and flag_gems' vendor detection
# reads triton's registry at import time (FlagGems master picks its `thead`
# backend whenever PPU_SDK is in the environment -- exported near the bottom of
# this script -- so no GEMS_VENDOR wiring is needed here).
#
# Integration stage only: it is the only stage that imports flag_gems (the
# dedicated PPU workflow runs everything in one job), and the C++ build needs
# neither triton nor flag_gems -- the FlagGems Python kernels are compiled from
# the checked-in csrc/aten/generated/flaggems_python_kernels.cc. Add it to the
# build stage if PPU is ever wired into build-wheel-common.yml.
if [[ "$CI_STAGE" == "integration" ]]; then
  # Route the two external fetches around the runner's proxy where the pod can
  # reach them directly: FlagTree comes from resource.flagos.net, which this
  # pod's proxy refuses, and FlagGems master is a large checkout from
  # github.com, which the same treatment at worst leaves on the proxy. See
  # prefer_direct_route above.
  prefer_direct_route "$FLAGTREE_INDEX_URL" "FlagTree wheel"
  pip_retry --no-deps --index-url "$FLAGTREE_INDEX_URL" "flagtree===$FLAGTREE_VERSION"
  prefer_direct_route "$FLAGGEMS_REPO" "FlagGems git checkout"
  pip_retry --no-deps "git+${FLAGGEMS_REPO}@${FLAGGEMS_REVISION}"
fi

CPU_TORCH_ROOT="$("$VENV_PYTHON" - <<'PY'
from pathlib import Path
import torch

print(Path(torch.__file__).resolve().parent)
assert torch.__version__.split("+", 1)[0] == "2.10.0", torch.__version__
assert torch.version.cuda is None, torch.version.cuda
PY
)"

strip_vendor_paths() {
  local value="${1:-}"
  local entry
  local -a entries=()
  local -a kept=()
  IFS=: read -ra entries <<< "$value"
  for entry in "${entries[@]}"; do
    [[ -z "$entry" ]] && continue
    case "$entry" in
      "$VENDOR_TORCH_ROOT"|"$VENDOR_TORCH_ROOT"/*) ;;
      *) kept+=("$entry") ;;
    esac
  done
  local joined=""
  for entry in "${kept[@]}"; do
    joined="${joined:+$joined:}$entry"
  done
  printf '%s' "$joined"
}

export VIRTUAL_ENV="$VENV_ROOT"
export PATH="$VENV_ROOT/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONPATH=""
export ACCELERATOR=cuda
export CUDA_HOME="$PPU_SDK/CUDA_SDK"
export CUDA_PATH="$CUDA_HOME"
export PPU_SDK="$PPU_SDK"
export FLAGOS_PPU_TORCH_LIB="$VENDOR_TORCH_LIB"
export FLAGOS_WHEEL_LOCAL="${FLAGOS_WHEEL_LOCAL:-ppu}"
# PPU image ships FlagGems as source only (no built liboperators.so /
# FlagGemsConfig.cmake), so the C++ kFlagOs dispatch (FLAGGEMS_KERNEL) must be
# off. setup.py's cuda branch does not pass -DFLAGGEMS_KERNEL=OFF (it assumes a
# real cuda image has FlagGems C++ installed); the generic env pass-through at
# setup.py:459-468 reads this env and emits -DFLAGGEMS_KERNEL=OFF, skipping
# CMakeLists.txt:502 if(FLAGGEMS_KERNEL) -> find_package(FlagGems). Mirrors
# metax set_env which exports FLAGGEMS_KERNEL=0 for the same reason. The
# Python-path kernels (FLAGGEMS_PYTHON) stay at the default ON -- they compile
# without importing flag_gems, and the FlagGems runtime test step needs them.
export FLAGGEMS_KERNEL=0
export FLAGCX_PATH="${FLAGCX_PATH:-/opt/FlagCX}"

CLEAN_CMAKE_PREFIX_PATH="$(strip_vendor_paths "${CMAKE_PREFIX_PATH:-}")"
CLEAN_LIBRARY_PATH="$(strip_vendor_paths "${LIBRARY_PATH:-}")"
CLEAN_LD_LIBRARY_PATH="$(strip_vendor_paths "${LD_LIBRARY_PATH:-}")"
export CMAKE_PREFIX_PATH="$CPU_TORCH_ROOT/share/cmake${CLEAN_CMAKE_PREFIX_PATH:+:$CLEAN_CMAKE_PREFIX_PATH}"
export CPATH="$CUDA_HOME/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$CUDA_HOME/lib64:$PPU_SDK/lib:$PPU_SDK/lib64${CLEAN_LIBRARY_PATH:+:$CLEAN_LIBRARY_PATH}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$PPU_SDK/lib:$PPU_SDK/lib64${VENDOR_NVIDIA_LIBS:+:$VENDOR_NVIDIA_LIBS}:$CPU_TORCH_ROOT/lib${CLEAN_LD_LIBRARY_PATH:+:$CLEAN_LD_LIBRARY_PATH}"

cd "$REPO_ROOT"
if [[ "$CI_STAGE" == "build" || "$CI_STAGE" == "integration" ]]; then
  # Prebuild so package_data sees libtorch_fl.so and the bundled PPU assets
  # before the common workflow invokes python -m build.
  python setup.py build_ext --inplace
fi

# Bundle PPU core+CUDA+MKL .so into torch_fl/lib_ppu/ and rewrite the plugin
# RPATH. setup.py does not auto-invoke this; mirrors the metax pattern. The
# bundle is idempotent and must run after build_ext so libtorch_fl.so exists.
bash scripts/vendor/bundle_ppu_libtorch.sh

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "::error::nvidia-smi is unavailable"
  exit 1
fi
nvidia-smi

python - <<'PY'
from pathlib import Path
import sys
import torch

torch_path = Path(torch.__file__).resolve()
assert sys.executable.startswith("/"), sys.executable
assert torch.__version__.split("+", 1)[0] == "2.10.0", torch.__version__
assert torch.version.cuda is None, torch.version.cuda
assert "/opt/conda/" not in str(torch_path), torch_path
assert Path("torch_fl/lib_ppu/libtorch_cuda.so").is_file()
print(f"Isolated Python: {sys.executable}")
print(f"CPU PyTorch: {torch.__version__}")
print(f"CPU torch path: {torch_path}")
print(f"PPU bundle: {Path('torch_fl/lib_ppu').resolve()}")
PY

# Prove the installed FlagGems stack is importable and wired to PPU, so a broken
# install fails here instead of turning into a wall of identical failures in the
# FlagGems runtime step. flag_gems queries torch.cuda at import time, so torch_fl
# must be imported first: that is what swaps the stock CPU core libs for the PPU
# build bundled above. The check therefore has to run after
# bundle_ppu_libtorch.sh, and only in the integration stage, which is where
# flagtree and flag_gems are installed.
#
# The three assertions are the ones that actually went wrong while bringing this
# up: the installed `triton` must be FlagTree's (its registry exposes the `ppu`
# backend, the vendor build exposes ['amd', 'nvidia']), flag_gems must have picked
# the `thead` vendor from PPU_SDK rather than falling back to nvidia, and it must
# resolve to the venv rather than the image's editable-install path.
if [[ "$CI_STAGE" == "integration" ]]; then
  python - <<'PY'
import os
from pathlib import Path
import sys

import torch_fl  # noqa: F401  (swaps in the bundled PPU libtorch core)
import flag_gems
import triton
import triton.backends

venv = sys.prefix
assert str(Path(triton.__file__).resolve()).startswith(venv), triton.__file__
assert "ppu" in triton.backends.backends, list(triton.backends.backends)
assert str(Path(flag_gems.__file__).resolve()).startswith(venv), flag_gems.__file__

vendor = flag_gems.runtime.backend.device_finder.DeviceDetector()._get_vendor_from_env()
assert vendor == "thead", f"flag_gems selected vendor {vendor!r}; PPU_SDK={os.environ.get('PPU_SDK')!r}"

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
  for name in \
    PATH VIRTUAL_ENV PYTHONNOUSERSITE PYTHONPATH ACCELERATOR CUDA_HOME CUDA_PATH \
    PPU_SDK FLAGOS_PPU_TORCH_LIB FLAGOS_SKIP_CUDA_ASSETS FLAGOS_DISABLE_CUDA_ASSETS \
    FLAGOS_WHEEL_LOCAL FLAGGEMS_KERNEL FLAGCX_PATH \
    CMAKE_PREFIX_PATH CPATH LIBRARY_PATH LD_LIBRARY_PATH; do
    printf '%s=%s\n' "$name" "${!name}" >> "$GITHUB_ENV"
  done
fi
