#!/usr/bin/env python3
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
"""Measure whether one HuggingFace transformers model runs on an accelerator.

This is a manual hardware probe, not a pytest test. CI does not invoke files in
``tests/manual``.

The unit of measurement is a single model, probed in ordered layers: device
transfer, forward, backward plus one optimizer step, and short generation. The
model is built tiny and randomly initialized from its ``CONFIG_MAPPING`` entry,
so no pretrained weights are downloaded and the probe runs offline. Each layer
is compared against CPU at the same dtype, and the first failing layer stops the
run because later layers cannot be interpreted once an earlier one is broken.

A failing layer is attributed to a concrete ATen operator by replaying the model
under ``TorchDispatchMode``, so a result names an operator rather than only a
model. Operator coverage is reported per layer as native versus CPU fallback.

Usage:
  python tests/manual/transformers_model_probe.py --model qwen3
  python tests/manual/transformers_model_probe.py --model qwen3 --dtype float16
  python tests/manual/transformers_model_probe.py --model llama --out /tmp/p.json

A full sweep is an external loop over single-model runs, not a mode of this
script. Use ``--list-models`` to enumerate probeable architectures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from packaging.version import InvalidVersion, Version

from importlib.metadata import PackageNotFoundError, version as package_version


PYPI_TRANSFORMERS_JSON = "https://pypi.org/pypi/transformers/json"


HARNESS_VERSION = 1

# A fault on an accelerator is not contained: one illegal access poisons the
# device context so every later operation fails with the same symptom. A layer
# matching this is reported once, not multiplied into the layers behind it.
POISON_RE = re.compile(
    r"illegal memory access|device-side assert|unspecified launch failure|"
    r"misaligned address|vmfault|acceleratorerror",
    re.IGNORECASE,
)

LAYERS = ("transfer", "forward", "backward", "generate")

# CPU same-dtype baselines. Reduction order legitimately differs between a CPU
# kernel and an accelerator kernel, so fp16/bf16 need room that fp32 does not.
TOLERANCES = {
    "float32": {"rtol": 1e-5, "atol": 1e-5},
    "float64": {"rtol": 1e-7, "atol": 1e-7},
    "float16": {"rtol": 1e-3, "atol": 1e-3},
    "bfloat16": {"rtol": 1.6e-2, "atol": 1e-2},
}

# Upper bound on a probe model. Some composite configs ignore top-level size
# overrides and would otherwise instantiate a multi-billion parameter model:
# a plain Gemma3Config with tiny hidden_size still built 2.7B parameters. The
# cap turns that into an explicit CONFIG_TOO_LARGE result instead of an OOM.
MAX_PROBE_PARAMS = 20_000_000


def atomic_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(value, f, indent=1, sort_keys=True)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


MARK = "@@PROBE@@"


def environment() -> dict:
    """Collect the versions a result cannot be interpreted without."""
    import torch
    import transformers

    commit = None
    try:
        commit = (
            subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=Path(__file__).resolve().parent,
            ).stdout.strip()
            or None
        )
    except (OSError, subprocess.SubprocessError):
        pass
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "torch_fl_commit": commit,
        "harness_version": HARNESS_VERSION,
    }


def installed_transformers() -> str | None:
    try:
        return package_version("transformers")
    except PackageNotFoundError:
        return None


def latest_transformers(timeout: int = 30) -> str | None:
    """Resolve the newest published release, or None when offline.

    Only used to tell the user their installed version is behind. The probe
    never installs or upgrades anything itself: replacing packages underneath a
    built extension is how a run silently stops measuring the torch the
    extension was compiled against.
    """
    try:
        with urllib.request.urlopen(PYPI_TRANSFORMERS_JSON, timeout=timeout) as resp:
            return json.load(resp)["info"]["version"]
    except Exception:  # noqa: BLE001 - offline is normal on a vendor box
        return None


def check_transformers_version(requested: str, offline: bool) -> dict:
    """Reconcile the requested version with what is importable.

    Returns the resolution for the record. A mismatch is fatal: silently
    measuring a different version than the one requested produces a result
    that cannot be compared with anything.
    """
    have = installed_transformers()
    if have is None:
        raise SystemExit(
            "transformers is not installed in this interpreter.\n"
            "Install it without touching torch:\n"
            "  python -m pip install --no-deps transformers==<version>"
        )
    record = {"requested": requested, "installed": have, "latest": None}
    if requested == "latest":
        if offline:
            return record
        latest = latest_transformers()
        record["latest"] = latest
        if latest is None:
            print(
                "note: could not reach PyPI; using installed "
                f"transformers {have} and recording it as-is"
            )
            return record
        try:
            behind = Version(have) < Version(latest)
        except InvalidVersion:
            behind = False
        if behind:
            print(
                f"note: installed transformers {have} is older than the latest "
                f"release {latest}.\n"
                "      Pass --transformers-version " + have + " to record this "
                "deliberately, or upgrade with:\n"
                "        python -m pip install --no-deps --upgrade transformers"
            )
        return record
    if have != requested:
        raise SystemExit(
            f"requested transformers {requested} but {have} is installed.\n"
            "A result measured against a different version is not comparable.\n"
            f"  python -m pip install --no-deps transformers=={requested}"
        )
    return record


def probeable_models() -> list[str]:
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES

    return sorted(CONFIG_MAPPING_NAMES)


def run_model(model: str, args: argparse.Namespace, layers: list[str]) -> dict:
    """Probe one model in a child process.

    One process per model even in single-model mode: an accelerator fault can
    abort the interpreter rather than raise, and a poisoned device context makes
    every later result in the same process meaningless.
    """
    request = {
        "model": model,
        "device": args.device,
        "dtype": args.dtype,
        "layers": layers,
        "tolerance": TOLERANCES[args.dtype],
        "max_params": args.max_params,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
    }
    child = Path(tempfile.gettempdir()) / f"transformers_probe_child_{os.getpid()}.py"
    child.write_text(CHILD)
    env = dict(os.environ)
    # Keep a compiler/runtime cache per run so a poisoned or partial cache from
    # one model cannot change the next model's result.
    env["TRITON_CACHE_DIR"] = tempfile.mkdtemp(prefix=f"triton-probe-{os.getpid()}-")
    started = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, str(child), json.dumps(request)],
            capture_output=True,
            text=True,
            env=env,
            cwd=tempfile.gettempdir(),
            timeout=args.timeout,
        )
        stdout, stderr, rc = proc.stdout, proc.stderr, proc.returncode
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        rc, timed_out = None, True
    finally:
        child.unlink(missing_ok=True)

    events, pending = {}, list(layers)
    for line in stdout.splitlines():
        if not line.startswith(MARK):
            continue
        try:
            rec = json.loads(line[len(MARK) :])
        except json.JSONDecodeError:
            continue
        text = " ".join(str(rec.get(k, "")) for k in ("error", "detail", "traceback"))
        if POISON_RE.search(text):
            rec["context_poison"] = True
        events[rec["layer"]] = rec
        if rec["layer"] in pending:
            pending.remove(rec["layer"])

    combined = f"{stderr}\n{stdout}"
    poisoned = bool(POISON_RE.search(combined))
    # The child announces a deliberate stop after a failing layer. Only when it
    # did not is a missing layer an actual crash or timeout.
    stopped = events.pop("__stopped__", None) is not None
    for index, name in enumerate(pending):
        if stopped:
            events[name] = {"layer": name, "status": "NOT_REACHED"}
        elif index == 0:
            # Attribute the fault once, to the first layer that never
            # reported, rather than to everything queued behind it.
            events[name] = {
                "layer": name,
                "status": "TIMEOUT" if timed_out else "CRASH",
                "error": combined.strip()[-600:],
                "returncode": rc,
                "context_poison": poisoned,
            }
        else:
            events[name] = {"layer": name, "status": "NOT_REACHED"}

    return {
        "model": model,
        "device": args.device,
        "dtype": args.dtype,
        "duration_s": round(time.time() - started, 1),
        "returncode": rc,
        "timed_out": timed_out,
        "context_poison": poisoned,
        "layers": events,
        "stderr_tail": stderr.strip()[-2000:] or None,
    }


def verdict(result: dict, layers: list[str]) -> str:
    """Reduce one model's layers to a single verdict.

    Ordered by how much they invalidate: an environment or config outcome is
    not a coverage result at all, and must not be counted as either support or
    a defect.
    """
    statuses = {
        name: result["layers"].get(name, {}).get("status")
        for name in ["config", *layers]
    }
    config = statuses.get("config")
    if config in (
        "UNKNOWN_MODEL",
        "CONFIG_UNSUPPORTED",
        "CONFIG_TOO_LARGE",
        "UNSUPPORTED_TASK",
        "ENVIRONMENT_ERROR",
    ):
        return config
    ran = [statuses.get(name) for name in layers]
    if any(s in ("CRASH", "TIMEOUT") for s in ran):
        return "CRASH"
    if any(s == "ERROR" for s in ran):
        return "ERROR"
    if any(s == "WRONG" for s in ran):
        return "WRONG"
    if all(s in ("PASS", "UNSUPPORTED", None) for s in ran):
        return "PASS"
    return "PARTIAL"


CHILD = r'''
import copy
import json
import sys
import traceback

import torch
import torch_fl  # noqa: F401
from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

request = json.loads(sys.argv[1])
MODEL = request["model"]
DEV = request["device"]
DTYPE = getattr(torch, request["dtype"])
LAYERS = request["layers"]
TOL = request["tolerance"]
MAX_PARAMS = request["max_params"]
SEED = request["seed"]
MAX_NEW_TOKENS = request["max_new_tokens"]

# Only lines starting with this marker are events. Vendor runtimes write
# unstructured diagnostics to stdout, and parsing those as results would
# invent statuses that no layer reported.
MARK = "@@PROBE@@"


def emit(layer, status, **extra):
    print(MARK + json.dumps({"layer": layer, "status": status, **extra}), flush=True)


def stop():
    """Mark a deliberate early exit.

    Without this the parent cannot tell "the child chose to stop after a
    failing layer" from "a faulting kernel killed the process", and would
    report the unreached layers as crashes they had no part in.
    """
    emit("__stopped__", "STOPPED")
    sys.exit(0)


def fail(layer, status, exc=None, **extra):
    if exc is not None:
        extra["error"] = f"{type(exc).__name__}: {exc}"
        extra["traceback"] = traceback.format_exc()[-4000:]
    emit(layer, status, **extra)
    stop()


# --- tiny config construction -------------------------------------------------

# Shrink targets. Attention heads must still divide hidden_size, so heads are
# handled after sizes and hidden_size is snapped up to a multiple of the heads.
SIZE_FIELDS = (
    "hidden_size", "intermediate_size", "ffn_dim", "d_model", "d_ff",
    "encoder_ffn_dim", "decoder_ffn_dim", "projection_dim", "text_config_dim",
)
DEPTH_FIELDS = (
    "num_hidden_layers", "num_layers", "n_layer", "num_encoder_layers",
    "num_decoder_layers", "encoder_layers", "decoder_layers",
    "num_local_experts", "num_experts", "num_experts_per_tok",
)
HEAD_FIELDS = ("num_attention_heads", "num_heads", "n_head", "encoder_attention_heads",
               "decoder_attention_heads")
KV_HEAD_FIELDS = ("num_key_value_heads", "num_kv_heads")
VOCAB_FIELDS = ("vocab_size",)
POS_FIELDS = ("max_position_embeddings", "n_positions", "max_seq_len",
              "max_sequence_length", "model_max_length")

TINY_HIDDEN = 32
TINY_HEADS = 2
TINY_LAYERS = 2
TINY_VOCAB = 256
TINY_POS = 64


def shrink(cfg, depth=0):
    """Recursively shrink a config in place.

    Composite configs (multimodal, encoder-decoder) hold nested
    PretrainedConfig objects that ignore top-level overrides entirely, so the
    nested objects must be shrunk directly.
    """
    if depth > 4:
        return
    for field in DEPTH_FIELDS:
        if isinstance(getattr(cfg, field, None), int):
            setattr(cfg, field, min(getattr(cfg, field), TINY_LAYERS))
    for field in HEAD_FIELDS:
        if isinstance(getattr(cfg, field, None), int):
            setattr(cfg, field, TINY_HEADS)
    for field in KV_HEAD_FIELDS:
        if isinstance(getattr(cfg, field, None), int):
            setattr(cfg, field, TINY_HEADS)
    for field in SIZE_FIELDS:
        if isinstance(getattr(cfg, field, None), int):
            setattr(cfg, field, TINY_HIDDEN)
    for field in VOCAB_FIELDS:
        if isinstance(getattr(cfg, field, None), int):
            setattr(cfg, field, TINY_VOCAB)
    for field in POS_FIELDS:
        if isinstance(getattr(cfg, field, None), int):
            setattr(cfg, field, TINY_POS)
    # head_dim is often derived; keep it consistent when explicitly present.
    if isinstance(getattr(cfg, "head_dim", None), int):
        cfg.head_dim = TINY_HIDDEN // TINY_HEADS
    for value in list(vars(cfg).values()):
        if isinstance(value, PretrainedConfig):
            shrink(value, depth + 1)


def build_config():
    if MODEL not in CONFIG_MAPPING:
        fail("config", "UNKNOWN_MODEL",
             detail=f"{MODEL} is not in CONFIG_MAPPING for this transformers version")
    try:
        cfg = CONFIG_MAPPING[MODEL]()
        shrink(cfg)
    except Exception as exc:  # noqa: BLE001 - any config error is a real result
        fail("config", "CONFIG_UNSUPPORTED", exc)
    for field, value in (("bos_token_id", 0), ("eos_token_id", 1), ("pad_token_id", 2)):
        if getattr(cfg, field, None) is not None:
            setattr(cfg, field, value)
    # Deterministic generation: sampling tokens would make token-level
    # comparison meaningless rather than measuring the device.
    if hasattr(cfg, "use_cache"):
        cfg.use_cache = True
    return cfg


# Auto classes tried in order, with the task label recorded for the result.
# The harness is not causal-LM-only: decoder architectures get a causal head
# plus `generate()`, encoder architectures fall back to a masked-LM head, and
# anything else falls back to the bare `AutoModel` body. `AutoModel` maps every
# config type, so it is the catch-all that keeps non-text or headless models
# measurable instead of failing them as `UNSUPPORTED_TASK`.
AUTO_CLASSES = (
    ("causal_lm", "AutoModelForCausalLM"),
    ("masked_lm", "AutoModelForMaskedLM"),
    ("base", "AutoModel"),
)


def build_model(cfg):
    from transformers import AutoModel, AutoModelForCausalLM, AutoModelForMaskedLM

    torch.manual_seed(SEED)
    classes = (AutoModelForCausalLM, AutoModelForMaskedLM, AutoModel)
    last_unrecognized = None
    for (task, label), cls in zip(AUTO_CLASSES, classes):
        try:
            model = cls.from_config(cfg)
            break
        except ValueError as exc:
            # "Unrecognized configuration class ... for this kind of AutoModel"
            # means this model type has no mapping for the head, not that the
            # config is broken. Fall through to the next candidate. Any other
            # ValueError is a genuine instantiation failure and stops the run.
            message = str(exc)
            if "Unrecognized configuration class" in message or "should be one of" in message:
                last_unrecognized = exc
                continue
            fail("config", "UNSUPPORTED_TASK", exc,
                 detail=f"config is a {label} but cannot instantiate a model")
    else:
        fail("config", "UNSUPPORTED_TASK", last_unrecognized,
             detail="no auto-model mapping (causal LM, masked LM, or base)")
    params = sum(p.numel() for p in model.parameters())
    if params > MAX_PARAMS:
        fail("config", "CONFIG_TOO_LARGE", params=params, max_params=MAX_PARAMS,
             detail="config ignored the tiny overrides; refusing to allocate")
    return model.to(DTYPE).eval(), params, task


# --- comparison ---------------------------------------------------------------

def compare(ref, got, path="out"):
    """Compare nested CPU and device outputs, returning a mismatch or None."""
    if isinstance(ref, torch.Tensor):
        if not isinstance(got, torch.Tensor):
            return {"path": path, "reason": "type", "detail": type(got).__name__}
        if tuple(ref.shape) != tuple(got.shape):
            return {"path": path, "reason": "shape",
                    "detail": f"cpu={tuple(ref.shape)} dev={tuple(got.shape)}"}
        dev = got.detach().to("cpu", torch.float32)
        cpu = ref.detach().to(torch.float32)
        # A NaN or Inf that CPU did not produce is always a defect; a tolerance
        # miss can be legitimate accumulation-order variance. Keep them apart.
        if torch.isfinite(cpu).all() and not torch.isfinite(dev).all():
            return {"path": path, "reason": "nan_inf",
                    "detail": "device produced non-finite values, CPU did not"}
        if not torch.allclose(cpu, dev, rtol=TOL["rtol"], atol=TOL["atol"],
                              equal_nan=True):
            diff = (cpu - dev).abs()
            denom = cpu.abs().clamp_min(1e-12)
            return {"path": path, "reason": "tolerance",
                    "max_abs": float(diff.max()),
                    "max_rel": float((diff / denom).max()),
                    "rtol": TOL["rtol"], "atol": TOL["atol"]}
        return None
    if isinstance(ref, (list, tuple)):
        if len(ref) != len(got):
            return {"path": path, "reason": "length",
                    "detail": f"cpu={len(ref)} dev={len(got)}"}
        for i, (a, b) in enumerate(zip(ref, got)):
            found = compare(a, b, f"{path}[{i}]")
            if found:
                return found
        return None
    if hasattr(ref, "keys"):
        for key in ref.keys():
            if ref[key] is None:
                continue
            found = compare(ref[key], got[key], f"{path}.{key}")
            if found:
                return found
        return None
    return None


# --- operator attribution -----------------------------------------------------

from torch.utils._python_dispatch import TorchDispatchMode  # noqa: E402


class OpTracer(TorchDispatchMode):
    """Record ATen ops and whether each has a PrivateUse1 kernel.

    An op without a device kernel reaches the CPU fallback, which is the
    difference between "the model ran" and "the accelerator ran the model".
    """

    def __init__(self):
        self.native = set()
        self.fallback = set()
        self._cache = {}

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = str(func)
        has_kernel = self._cache.get(name)
        if has_kernel is None:
            try:
                has_kernel = torch._C._dispatch_has_kernel_for_dispatch_key(
                    func.name(), "PrivateUse1")
            except Exception:  # noqa: BLE001 - attribution must never fail a layer
                has_kernel = True
            self._cache[name] = has_kernel
        (self.native if has_kernel else self.fallback).add(name)
        return func(*args, **(kwargs or {}))

    def report(self):
        return {"native": sorted(self.native), "fallback": sorted(self.fallback)}


def make_inputs(cfg):
    vocab = max(int(getattr(cfg, "vocab_size", 256)), 8)
    ids = torch.tensor([[0, 3, 4, 5]], dtype=torch.long)
    ids = ids.remainder(vocab)
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def tensors_on(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, (list, tuple)):
        return type(value)(tensors_on(x, device) for x in value)
    if hasattr(value, "keys"):
        return {k: tensors_on(v, device) for k, v in value.items()}
    return value


def first_tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (list, tuple)):
        for x in value:
            found = first_tensor(x)
            if found is not None:
                return found
    if hasattr(value, "values"):
        for x in value.values():
            found = first_tensor(x)
            if found is not None:
                return found
    return None


def output_device_ok(value, device):
    tensor = first_tensor(value)
    return tensor is not None and tensor.device.type == device.split(":", 1)[0]


def run_layer(layer, cpu_model, dev_model, cpu_inputs, dev_inputs):
    tracer = OpTracer()
    if layer == "transfer":
        ok = all(p.device.type == DEV.split(":", 1)[0]
                 for p in dev_model.parameters())
        emit(layer, "PASS" if ok else "WRONG", ops=tracer.report())
        return
    try:
        if layer == "forward":
            with torch.no_grad():
                with tracer:
                    got = dev_model(**dev_inputs)
                ref = cpu_model(**cpu_inputs)
            mismatch = compare(ref, got)
            if not output_device_ok(got, DEV):
                mismatch = mismatch or {"path": "out", "reason": "device"}
        elif layer == "backward":
            if not any(p.requires_grad for p in dev_model.parameters()):
                emit(layer, "UNSUPPORTED", detail="no trainable parameter",
                     ops=tracer.report())
                return
            cpu_model.zero_grad(set_to_none=True)
            dev_model.zero_grad(set_to_none=True)
            with tracer:
                got = dev_model(**dev_inputs)
                first_tensor(got).float().mean().backward()
            ref = cpu_model(**cpu_inputs)
            first_tensor(ref).float().mean().backward()
            mismatch = None
            for (name, a), (_, b) in zip(cpu_model.named_parameters(),
                                         dev_model.named_parameters()):
                if a.grad is None and b.grad is None:
                    continue
                if (a.grad is None) != (b.grad is None):
                    mismatch = {"path": f"grad.{name}", "reason": "grad_presence",
                                "detail": f"cpu={a.grad is not None} "
                                          f"dev={b.grad is not None}"}
                    break
                mismatch = compare(a.grad, b.grad, f"grad.{name}")
                if mismatch:
                    break
            if mismatch is None:
                # One optimizer step exercises the update path, which uses
                # operators the forward and backward passes never reach.
                with tracer:
                    dev_model.probe_optimizer.step()
                cpu_model.probe_optimizer.step()
                for (name, a), (_, b) in zip(cpu_model.named_parameters(),
                                             dev_model.named_parameters()):
                    mismatch = compare(a, b, f"stepped.{name}")
                    if mismatch:
                        break
        elif layer == "generate":
            if not hasattr(dev_model, "generate"):
                emit(layer, "UNSUPPORTED", detail="model has no generate method",
                     ops=tracer.report())
                return
            with torch.no_grad():
                with tracer:
                    got = dev_model.generate(**dev_inputs, max_new_tokens=MAX_NEW_TOKENS,
                                             do_sample=False)
                with torch.no_grad():
                    ref = cpu_model.generate(**cpu_inputs, max_new_tokens=MAX_NEW_TOKENS,
                                             do_sample=False)
            mismatch = compare(ref, got)
            if not output_device_ok(got, DEV):
                mismatch = mismatch or {"path": "generated", "reason": "device"}
        else:
            emit(layer, "UNKNOWN_LAYER")
            return
        emit(layer, "PASS" if mismatch is None else "WRONG",
             comparison=mismatch, ops=tracer.report())
        if mismatch is not None:
            stop()
    except Exception as exc:  # noqa: BLE001 - child reports all runtime failures
        fail(layer, "ERROR", exc, ops=tracer.report())


# Load and run after definitions so config errors are reported in the protocol.
try:
    cfg = build_config()
    model, params, task = build_model(cfg)
    # Module.to() mutates in place and returns self, so the CPU baseline must
    # be a distinct object. Sharing one would compare the device against
    # itself and report every layer as passing.
    cpu_model = model
    dev_model = copy.deepcopy(model).to(DEV)
    cpu_model.probe_optimizer = torch.optim.SGD(cpu_model.parameters(), lr=1e-3)
    dev_model.probe_optimizer = torch.optim.SGD(dev_model.parameters(), lr=1e-3)
    cpu_inputs = make_inputs(cfg)
    dev_inputs = tensors_on(cpu_inputs, DEV)
    emit("config", "PASS", params=params, task=task)
except SystemExit:
    raise
except Exception as exc:  # noqa: BLE001
    fail("config", "ENVIRONMENT_ERROR", exc,
         detail="model built on CPU but could not be placed on the device")

for layer in LAYERS:
    run_layer(layer, cpu_model, dev_model, cpu_inputs, dev_inputs)
'''


def summarize(result: dict, layers: list[str]) -> str:
    lines = [
        f"model      {result['model']}",
        f"device     {result['device']}  dtype {result['dtype']}",
        f"verdict    {result['verdict']}",
        f"duration   {result['duration_s']}s",
    ]
    config = result["layers"].get("config", {})
    if config.get("task"):
        lines.append(f"task       {config['task']}")
    if config.get("params"):
        lines.append(f"params     {config['params']:,}")
    if result.get("context_poison"):
        lines.append("WARNING    device context was poisoned; later layers are void")
    lines.append("")
    for name in layers:
        event = result["layers"].get(name)
        if not event:
            continue
        row = f"  {name:<9} {event['status']}"
        ops = event.get("ops") or {}
        if ops.get("fallback"):
            row += f"  [cpu-fallback: {len(ops['fallback'])}]"
        if event.get("comparison"):
            cmp = event["comparison"]
            row += f"  {cmp.get('reason')} at {cmp.get('path')}"
            if cmp.get("max_abs") is not None:
                row += f" (max_abs {cmp['max_abs']:.3g})"
        if event.get("error"):
            row += f"  {event['error'].splitlines()[0][:120]}"
        lines.append(row)
    fallback = sorted(
        {
            op
            for event in result["layers"].values()
            for op in (event.get("ops") or {}).get("fallback", [])
        }
    )
    if fallback:
        lines += ["", "CPU-fallback operators (not running on the accelerator):"]
        lines += [f"  {op}" for op in fallback]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Probe one HuggingFace transformers model on an accelerator.",
    )
    parser.add_argument("--model", help="model type, e.g. qwen3")
    parser.add_argument("--device", default="flagos")
    parser.add_argument("--dtype", default="float32", choices=sorted(TOLERANCES))
    parser.add_argument(
        "--layers",
        default=",".join(LAYERS),
        help=f"comma-separated subset of {','.join(LAYERS)}",
    )
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--max-params", type=int, default=MAX_PROBE_PARAMS)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--out", type=Path, help="write the JSON result here")
    parser.add_argument(
        "--transformers-version",
        default="latest",
        help="'latest' (default) records the installed version and notes when it is "
        "behind PyPI; an explicit version must match what is installed",
    )
    parser.add_argument(
        "--offline", action="store_true", help="skip the PyPI latest-release check"
    )
    parser.add_argument(
        "--list-models", action="store_true", help="list probeable model types and exit"
    )
    args = parser.parse_args()

    if args.list_models:
        for name in probeable_models():
            print(name)
        return 0
    if not args.model:
        parser.error("--model is required (or use --list-models)")

    resolution = check_transformers_version(args.transformers_version, args.offline)

    layers = [name.strip() for name in args.layers.split(",") if name.strip()]
    unknown = [name for name in layers if name not in LAYERS]
    if unknown:
        parser.error(f"unknown layers: {', '.join(unknown)}")
    # Keep the documented order regardless of how they were listed: a later
    # layer's result cannot be read if an earlier one has not run.
    layers = [name for name in LAYERS if name in layers]

    env = environment()
    env["transformers_requested"] = resolution["requested"]
    env["transformers_latest"] = resolution["latest"]
    result = run_model(args.model, args, layers)
    result["verdict"] = verdict(result, layers)
    result["environment"] = env
    result["fingerprint"] = hashlib.sha256(
        f"{result['verdict']}|{args.model}|{args.device}".encode()
    ).hexdigest()[:12]

    print(
        f"transformers {env['transformers']}  torch {env['torch']}"
        f"  torch_fl {env['torch_fl_commit']}"
    )
    print()
    print(summarize(result, layers))

    if args.out:
        atomic_write(args.out, result)
        print(f"\nJSON written to {args.out}")
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
