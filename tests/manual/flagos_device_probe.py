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

"""Live hardware probe: Distributed backend auto-selection and flex attention on ``flagos``.

Two questions, one script, every platform:

1. **Backend auto-selection.** ``register_flagos_backend`` registers the backend
   under ``"flagos"`` but keys ``Backend.default_device_backend_map`` with the
   literal ``"privateuseone"``.  At runtime the device type is the registered
   name (``torch._C._get_privateuse1_backend_name()``), so that entry is
   unreachable and ``get_default_backend_for_device`` raises.  The probe
   measures the current behaviour, then *emulates* the proposed fix in-process
   (``default_device_backend_map.setdefault(<device type>, "flagos")``) and
   re-measures, including a single-rank ``init_process_group(device_id=...)``
   end-to-end -- so the fix is validated on hardware before it lands.

2. **Flex attention.** ``torch.nn.attention.flex_attention`` validates its inputs
   against a hard-coded device set upstream; ``torch_fl.compat.flex_attention``
   relaxes that gate for flagos.  The probe checks the gate is patched, then runs
   the eager path against a CPU reference (causal BlockMask, ``score_mod``,
   GQA + backward), and optionally the fused ``torch.compile`` path.

Every section runs in its own subprocess, so a crash or hang in one -- a vendor
compiler in the fused path is the realistic case -- is reported as that
section's result and cannot take the rest of the probe down.  Run it on the
live platform, inside the platform's container, with the platform's torch_fl
environment; results are JSON lines on stdout after ``@@PROBE@@`` markers.

Usage::

    python tests/manual/flagos_device_probe.py                    # env, dist, flex_eager
    python tests/manual/flagos_device_probe.py --with-compile     # + the fused section
    python tests/manual/flagos_device_probe.py --section dist     # one raw section

``FLAGOS_PROBE_DEVICE`` selects the device index (default 0).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback

MARKER = "@@PROBE@@"
BASE_SECTIONS = ("env", "dist", "flex_eager")
ALL_SECTIONS = BASE_SECTIONS + ("flex_compile",)

#: Wall-clock budget per section, seconds.  ``flex_compile`` compiles a Triton
#: template on whatever the platform ships; that is the slow one.
TIMEOUTS = {"env": 300, "dist": 300, "flex_eager": 600, "flex_compile": 1800}


def _probe_device():
    import torch

    device_type = torch._C._get_privateuse1_backend_name()
    index = int(os.environ.get("FLAGOS_PROBE_DEVICE", "0"))
    return torch.device(f"{device_type}:{index}")


def _import_plugins(res: dict) -> None:
    """Import torch_fl before torch (it preloads the boxing library), guarded."""
    try:
        import torch_fl

        res["torch_fl_import"] = {"ok": True, "file": torch_fl.__file__}
    except Exception as exc:  # noqa: BLE001 - the probe reports, it does not fix
        res["torch_fl_import"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def section_env(res: dict) -> None:
    """Versions, device type and availability, triton, git head of the checkout."""
    _import_plugins(res)
    import torch

    res["python"] = sys.version.split()[0]
    res["torch"] = torch.__version__
    try:
        res["accelerator"] = torch._C._get_accelerator().type
    except Exception as exc:  # noqa: BLE001
        res["accelerator_error"] = f"{type(exc).__name__}: {exc}"

    device_type = torch._C._get_privateuse1_backend_name()
    res["privateuse1_name"] = device_type
    try:
        tensor = torch.zeros(1, device=f"{device_type}:0")
        res["flagos_tensor_ok"] = True
        res["flagos_value"] = float(tensor.item())
    except Exception as exc:  # noqa: BLE001
        res["flagos_tensor_ok"] = False
        res["flagos_tensor_error"] = f"{type(exc).__name__}: {exc}"
    for name, fn in (
        ("device_count_accelerator", lambda: torch.accelerator.device_count()),
        ("device_count_cuda", lambda: torch.cuda.device_count()),
        ("cuda_is_available", lambda: torch.cuda.is_available()),
    ):
        try:
            res[name] = fn()
        except Exception as exc:  # noqa: BLE001
            res[name] = f"{type(exc).__name__}: {exc}"
    try:
        import triton

        res["triton"] = triton.__version__
    except Exception as exc:  # noqa: BLE001
        res["triton"] = f"{type(exc).__name__}: {exc}"

    for var in (
        "FLAGOS_ACCELERATOR",
        "FLAGOS_DIST_REDIRECT_GLOO",
        "FLAGOS_DISABLE_FLEX_ATTENTION_COMPAT",
        "TORCH_DEVICE_BACKEND_AUTOLOAD",
        "CUDA_VISIBLE_DEVICES",
    ):
        res[f"env_{var}"] = os.environ.get(var)

    # Best-effort: which commit of the repo is the editable install serving?
    info = res.get("torch_fl_import", {})
    if info.get("ok"):
        repo = os.path.dirname(os.path.dirname(os.path.abspath(info["file"])))

        def _git_head() -> str:
            proc = subprocess.run(
                ["git", "-C", repo, "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            return proc.stdout.strip()

        try:
            res["torch_fl_git_head"] = _git_head() or None
        except Exception:  # noqa: BLE001 - not a source checkout, or no git
            res["torch_fl_git_head"] = None


def _try_dist_init(res: dict, label: str, init) -> None:
    """Run one init_process_group variant, capture outcome + traceback, clean up."""
    import torch.distributed as dist

    entry: dict = {}
    try:
        init()
        entry["ok"] = True
    except Exception as exc:  # noqa: BLE001
        entry["ok"] = False
        entry["error"] = f"{type(exc).__name__}: {exc}"
        entry["traceback"] = traceback.format_exc()[-2500:]
    finally:
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception:  # noqa: BLE001
            pass
    res[label] = entry


def section_dist(res: dict) -> None:
    """Backend auto-selection: reproduce, emulate the fix, init end-to-end."""
    _import_plugins(res)
    import inspect

    import torch
    import torch.distributed as dist

    device_type = torch._C._get_privateuse1_backend_name()
    device = _probe_device()
    res["device_type"] = device_type
    res["backend_list_has_flagos"] = "flagos" in dist.Backend.backend_list
    res["map_snapshot"] = {
        key: value
        for key, value in dist.Backend.default_device_backend_map.items()
        if "flagos" in value or "private" in key
    }
    try:
        res["register_backend_source"] = inspect.getsource(
            dist.Backend.register_backend
        )
    except Exception as exc:  # noqa: BLE001
        res["register_backend_source_error"] = f"{type(exc).__name__}: {exc}"

    # --- Repro: the map entry the plugin added is keyed by the wrong name. ---
    for label, spec in (
        ("runtime_type", f"{device_type}:0"),
        ("legacy_spelling", "privateuseone:0"),
    ):
        try:
            backend = dist.get_default_backend_for_device(torch.device(spec))
            res[f"get_default_{label}"] = {"ok": True, "backend": backend}
        except Exception as exc:  # noqa: BLE001
            res[f"get_default_{label}"] = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }

    # --- Unfixed auto-selection: the idiomatic call without a backend string. ---
    def _hash_store():
        return dist.HashStore()

    _try_dist_init(
        res,
        "init_auto_device_id_unfixed",
        lambda: dist.init_process_group(
            rank=0, world_size=1, store=_hash_store(), device_id=device
        ),
    )

    # --- Control: backend named explicitly (the handoff measured this works). ---
    _try_dist_init(
        res,
        "init_explicit_backend_no_device_id",
        lambda: dist.init_process_group(
            backend="flagos", rank=0, world_size=1, store=_hash_store()
        ),
    )
    # This is the sharp edge the fix walks users into (CUDAGuardImpl traceback).
    _try_dist_init(
        res,
        "init_explicit_backend_with_device_id",
        lambda: dist.init_process_group(
            backend="flagos",
            rank=0,
            world_size=1,
            store=_hash_store(),
            device_id=device,
        ),
    )

    # --- The proposed fix, emulated in-process: key the map by the runtime name. ---
    dist.Backend.default_device_backend_map.setdefault(device_type, "flagos")
    res["map_after_fix"] = {
        key: value
        for key, value in dist.Backend.default_device_backend_map.items()
        if "flagos" in value or "private" in key
    }
    try:
        res["get_default_after_fix"] = dist.get_default_backend_for_device(device)
    except Exception as exc:  # noqa: BLE001
        res["get_default_after_fix"] = f"{type(exc).__name__}: {exc}"

    _try_dist_init(
        res,
        "init_auto_device_id_fixed",
        lambda: dist.init_process_group(
            rank=0, world_size=1, store=_hash_store(), device_id=device
        ),
    )
    _try_dist_init(
        res,
        "init_auto_no_device_id_fixed",
        lambda: dist.init_process_group(rank=0, world_size=1, store=_hash_store()),
    )


def _causal(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


def _score_shift(score, b, h, q_idx, kv_idx):
    return score + 0.1 * (b - h)


def section_flex_eager(res: dict) -> None:
    """Eager flex attention on flagos against a CPU reference."""
    _import_plugins(res)
    import torch
    import torch.nn.attention.flex_attention as fa
    from torch.nn.attention.flex_attention import create_block_mask

    validate = getattr(fa, "_validate_device", None)
    res["has_validate_device"] = validate is not None
    res["validate_device_patched"] = bool(
        getattr(validate, "_flagos_flex_attention_compat_patched", False)
    )

    device = _probe_device()
    batch, heads, seq, dim = 2, 4, 256, 64
    torch.manual_seed(0)
    q = torch.randn(batch, heads, seq, dim, device=device, dtype=torch.float32)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    started = time.perf_counter()
    block_mask = create_block_mask(_causal, batch, heads, seq, seq, device=device)
    res["block_mask_seconds"] = round(time.perf_counter() - started, 2)

    out = fa.flex_attention(q, k, v, block_mask=block_mask)
    res["shape"] = list(out.shape)
    res["finite"] = bool(torch.isfinite(out).all())

    # CPU reference on identical inputs.
    cpu_mask = create_block_mask(_causal, batch, heads, seq, seq, device="cpu")
    ref = fa.flex_attention(q.cpu(), k.cpu(), v.cpu(), block_mask=cpu_mask)
    res["max_abs_diff_vs_cpu"] = float((out.cpu().double() - ref.double()).abs().max())
    res["ref_finite"] = bool(torch.isfinite(ref).all())

    # score_mod on the same inputs.
    out_score_mod = fa.flex_attention(
        q, k, v, score_mod=_score_shift, block_mask=block_mask
    )
    res["score_mod_finite"] = bool(torch.isfinite(out_score_mod).all())
    ref_score_mod = fa.flex_attention(
        q.cpu(), k.cpu(), v.cpu(), score_mod=_score_shift, block_mask=cpu_mask
    )
    res["score_mod_max_abs_diff_vs_cpu"] = float(
        (out_score_mod.cpu().double() - ref_score_mod.double()).abs().max()
    )

    # GQA + backward (no mask: the op + autograd graph is what this validates).
    kv_heads = heads // 2
    qg = torch.randn(batch, heads, seq, dim, device=device, requires_grad=True)
    kg = torch.randn(batch, kv_heads, seq, dim, device=device, requires_grad=True)
    vg = torch.randn(batch, kv_heads, seq, dim, device=device, requires_grad=True)
    out_gqa = fa.flex_attention(qg, kg, vg, enable_gqa=True)
    out_gqa.sum().backward()
    res["gqa_shape"] = list(out_gqa.shape)
    res["gqa_finite"] = bool(torch.isfinite(out_gqa).all())
    res["gqa_grads_finite"] = {
        "query": bool(torch.isfinite(qg.grad).all()),
        "key": bool(torch.isfinite(kg.grad).all()),
        "value": bool(torch.isfinite(vg.grad).all()),
    }

    # bf16 eager, the dtype real frameworks run at.
    qb = q.to(torch.bfloat16)
    kb = k.to(torch.bfloat16)
    vb = v.to(torch.bfloat16)
    out_bf16 = fa.flex_attention(qb, kb, vb, block_mask=block_mask)
    res["bf16_finite"] = bool(torch.isfinite(out_bf16).all())
    res["bf16_max_abs_diff_vs_cpu_fp32"] = float(
        (out_bf16.float().cpu().double() - ref.double()).abs().max()
    )


def section_flex_compile(res: dict) -> None:
    """Fused (torch.compile -> Triton template) flex on flagos, bf16."""
    _import_plugins(res)
    import torch
    import torch.nn.attention.flex_attention as fa
    from torch.nn.attention.flex_attention import create_block_mask

    res["torch_compile_available"] = hasattr(torch, "compile")
    device = _probe_device()
    batch, heads, seq, dim = 2, 4, 256, 64
    torch.manual_seed(0)
    q = torch.randn(batch, heads, seq, dim, device=device, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    block_mask = create_block_mask(_causal, batch, heads, seq, seq, device=device)

    started = time.perf_counter()
    compiled = torch.compile(fa.flex_attention)
    out = compiled(q, k, v, block_mask=block_mask)
    res["first_call_seconds"] = round(time.perf_counter() - started, 2)
    res["finite"] = bool(torch.isfinite(out).all())
    res["shape"] = list(out.shape)

    eager = fa.flex_attention(q, k, v, block_mask=block_mask)
    res["max_abs_diff_vs_eager_bf16"] = float((out.float() - eager.float()).abs().max())


SECTIONS = {
    "env": section_env,
    "dist": section_dist,
    "flex_eager": section_flex_eager,
    "flex_compile": section_flex_compile,
}


def _child(section_name: str) -> None:
    res: dict = {}
    error = None
    try:
        SECTIONS[section_name](res)
    except BaseException as exc:  # noqa: BLE001 - the driver reports what happened
        error = f"{type(exc).__name__}: {exc}"
        res.setdefault("traceback", traceback.format_exc()[-2500:])
    wrapper = {
        "section": section_name,
        "ok": error is None,
        "error": error,
        "result": res,
    }
    print(MARKER + " " + json.dumps(wrapper, default=str), flush=True)


def _run_child(section_name: str) -> dict:
    cmd = [sys.executable, os.path.abspath(__file__), "--section", section_name]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=TIMEOUTS[section_name]
        )
    except subprocess.TimeoutExpired as exc:
        tail = exc.stdout or b""
        if isinstance(tail, bytes):
            tail = tail.decode(errors="replace")
        return {
            "section": section_name,
            "ok": False,
            "error": f"timeout after {TIMEOUTS[section_name]}s",
            "result": {"partial_stdout_tail": tail[-2000:]},
        }
    for line in proc.stdout.splitlines():
        if line.startswith(MARKER):
            return json.loads(line[len(MARKER) :])
    return {
        "section": section_name,
        "ok": False,
        "error": f"no marker in output; exit code {proc.returncode}",
        "result": {
            "stdout_tail": proc.stdout[-2000:],
            "stderr_tail": proc.stderr[-2000:],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--section", choices=ALL_SECTIONS, help="run one section raw")
    parser.add_argument(
        "--with-compile", action="store_true", help="also run the fused section"
    )
    args = parser.parse_args()

    if args.section:
        _child(args.section)
        return 0

    sections = list(BASE_SECTIONS) + (["flex_compile"] if args.with_compile else [])
    results = {}
    for name in sections:
        print(f"--- section {name} ---", flush=True)
        results[name] = _run_child(name)
        print(json.dumps(results[name], indent=2, default=str)[:6000], flush=True)
    print(
        "@@PROBE_DONE@@ " + json.dumps({"sections": results}, default=str), flush=True
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
