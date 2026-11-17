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

"""Experimental source-preserving native hosts for ordinary FlagGems eager calls.

Compilation runs in a separate process. Only metadata and scalar specialization
constants cross that boundary; user tensors, streams and output buffers do not.
"""

import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import warnings
from typing import Union

import torch

_mode = "collect"
_root = None
_seen = set()
_collected = set()
_lock = threading.RLock()
_local = threading.local()
_libraries = []


def enable(cache_dir: Union[str, os.PathLike], *, mode: str = "compile") -> None:
    """Enable native lookup and cold compilation (or metadata-only collection).

    This initial implementation admits inference calls only. A guard miss still
    executes the original FlagGems function, then attempts an isolated compile.
    Unsupported capture is recorded, without changing the numerical provider.
    """
    global _root, _mode
    if mode not in {"compile", "collect", "load"}:
        raise ValueError(mode)
    from torch_fl import _build_config
    from torch_fl._platform import detect_platform

    if detect_platform() != "cuda" or "flaggems" not in _build_config.KERNELS:
        raise RuntimeError("host compiler requires the NVIDIA FlagGems build")
    if torch.__version__.split("+")[0] != "2.10.0":
        raise RuntimeError("host compiler currently requires PyTorch 2.10.0")
    if "boxing" in _build_config.KERNELS or torch.version.cuda is not None:
        raise RuntimeError(
            "host compiler validation requires the CPU-Torch vendor build"
        )
    with _lock:
        if not _libraries:
            libraries = [
                ctypes.CDLL("libcuda.so.1", mode=ctypes.RTLD_GLOBAL),
                ctypes.CDLL(
                    str(Path(__file__).parents[1] / "lib/libflagos.so"),
                    mode=ctypes.RTLD_GLOBAL,
                ),
            ]
            _libraries.extend(libraries)
        _root = Path(cache_dir).resolve()
        _root.mkdir(parents=True, exist_ok=True)
        _mode = mode
        for manifest in sorted(_root.glob("*/artifact.json")):
            try:
                _register(json.loads(manifest.read_text()), manifest)
            except (OSError, ValueError) as error:
                warnings.warn(
                    f"Cannot read host artifact {manifest}: {error}", RuntimeWarning
                )
        torch.ops.flagos.host_program_enable(True)


def disable() -> None:
    """Disable lookup; resident modules remain accounted for until process exit."""
    torch.ops.flagos.host_program_enable(False)


def _register(artifact, manifest):
    try:
        _register_checked(artifact, manifest)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        (manifest.parent / "admission_failure.txt").write_text(
            f"Artifact admission failed; original FlagGems route retained: {error}\n"
        )


def _register_checked(artifact, manifest):
    path = Path(artifact["library"])
    source = Path(artifact.get("source", str(manifest.parent / "host.cpp")))
    cpp = source.read_text()
    calls = re.findall(r"launchKernel\((\w+),", cpp)
    binary = path.read_bytes()
    unexpected = [
        name
        for name in calls
        if not any(
            name == original or name.startswith(original + "_")
            for original in artifact["kernels"]
        )
    ]
    # Old experimental cache entries predate the producer's provenance gate.
    # Validate again at admission, including the actual shared-library bytes.
    forbidden = any(
        marker in binary
        for marker in (
            b"triton_poi_fused_",
            b"triton_red_fused_",
            b"triton_per_fused_",
            b"cpp_fused_",
        )
    )
    checksum = artifact.get("library_sha256")
    if (
        unexpected
        or (artifact["kernels"] and not calls)
        or forbidden
        or "aoti_torch_copy_" in cpp
        or (checksum and hashlib.sha256(binary).hexdigest() != checksum)
    ):
        (manifest.parent / "admission_failure.txt").write_text(
            "Artifact provenance or integrity check failed; original FlagGems route retained.\n"
        )
        return
    torch.ops.flagos.host_program_register(
        artifact["key"], str(path), artifact["outputs"], path.stat().st_size
    )


def _encode(value):
    if isinstance(value, torch.Tensor):
        return {
            "tensor": True,
            "shape": list(value.shape),
            "stride": list(value.stride()),
            "offset": value.storage_offset(),
            "dtype": str(value.dtype).removeprefix("torch."),
            "device": str(value.device),
            "conj": value.is_conj(),
            "neg": value.is_neg(),
        }
    if isinstance(value, torch.dtype):
        return {"dtype": str(value).removeprefix("torch.")}
    if isinstance(value, (list, tuple)):
        return {"sequence": type(value).__name__, "items": [_encode(v) for v in value]}
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"unsupported host argument: {type(value)}")


def _observe(key, func, args, kwargs, outputs):
    if _mode == "load" or getattr(_local, "busy", False):
        return
    _local.busy = True
    try:
        spec = {
            "key": key,
            "module": func.__module__,
            "name": func.__name__,
            "args": _encode(args),
            "kwargs": {k: _encode(v) for k, v in kwargs.items()},
            "outputs": outputs,
        }
        encoded = json.dumps(spec, sort_keys=True)
        digest = hashlib.sha256(encoded.encode()).hexdigest()[:24]
        with _lock:
            # Collection must not consume the later compilation attempt, and
            # selecting a different cache must write its own request file.
            if _mode == "load":
                return
            seen = _seen if _mode == "compile" else _collected
            if len(seen) >= 512:
                return
            directory = _root / digest
            if directory in seen:
                return
            seen.add(directory)
            directory.mkdir(exist_ok=True)
            request = directory / "request.json"
            request.write_text(encoded)
            if _mode == "compile":
                _compile(request)
                manifest = directory / "artifact.json"
                if manifest.exists():
                    _register(json.loads(manifest.read_text()), manifest)
    except Exception as error:
        # Observation follows a successful numerical call. A compiler or
        # admission failure must not turn that result into an operator failure.
        warnings.warn(f"Host compilation bypassed: {error}", RuntimeWarning)
    finally:
        _local.busy = False


def _compile(request):
    env = dict(os.environ)
    env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    env["TORCHINDUCTOR_CACHE_DIR"] = str(request.parent / "inductor")
    env["TORCH_EXTENSIONS_DIR"] = str(request.parent / "extensions")
    env["MAX_JOBS"] = "4"
    env["CPLUS_INCLUDE_PATH"] = os.pathsep.join(
        [
            str(Path(__file__).parent / "include"),
            "/usr/local/cuda/include",
            env.get("CPLUS_INCLUDE_PATH", ""),
        ]
    )
    with (request.parent / "compile.log").open("w") as log:
        return subprocess.run(
            [sys.executable, "-m", "torch_fl.host_compiler.worker", str(request)],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        ).returncode
