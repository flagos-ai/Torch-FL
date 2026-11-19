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

"""Full official Qwen3 prefill/decode, with a separate vendor reference.

Eager execution is the default. --model-host compiles only the model's host
graph through ATen. Numerical kernels come from FlagGems or audited, measured
vendor precision exceptions. Neither mode uses
CUDA graphs, layer truncation, a replacement model or output pooling. Run each
backend in its own process on an otherwise idle GPU.
"""

# Device libraries must precede Torch import on the CPU-Torch Hygon build.
# ruff: noqa: E402

import argparse
from collections import Counter
from contextlib import nullcontext
import json
import importlib.util
import os
from pathlib import Path
import statistics
import sys
import time

_backend_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
_backend_parser.add_argument("--device")
if (_backend_parser.parse_known_args()[0].device or "").startswith("flagos"):
    import torch_fl  # noqa: F401

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from logits_validation import (
    bf16_vector_accuracy,
    bf16_vector_policy,
    LogitsValidationFailed,
    file_sha256,
    math_settings,
    summarize_accuracy,
    summarize_bf16_accuracy,
    tensor_accuracy,
    validate_reference,
    validate_tolerances,
)


def compare(actual, expected):
    a, b = actual.float(), expected.float()
    return {
        "max_abs": (a - b).abs().max().item(),
        "relative_l2": ((a - b).norm() / b.norm()).item(),
        "cosine": torch.nn.functional.cosine_similarity(a, b).mean().item(),
        "top1_equal": bool(torch.equal(a.argmax(-1), b.argmax(-1))),
        "bitwise_equal": bool(
            torch.equal(
                actual.contiguous().view(torch.uint8),
                expected.contiguous().view(torch.uint8),
            )
        ),
        "allclose_rtol_003_atol_01": bool(torch.allclose(a, b, rtol=0.03, atol=0.1)),
    }


def vendor_audit():
    from torch_fl import _build_config

    if not (
        getattr(_build_config, "NATIVE_HOSTS", False)
        or getattr(_build_config, "NATIVE_VENDOR", False)
    ):
        return None
    from torch_fl.native_host import vendor

    return vendor.audit()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--device", required=True, choices=["cuda:0", "npu:0", "flagos:0"]
    )
    parser.add_argument(
        "--mode", choices=["plain", "collect", "compile", "load"], default="plain"
    )
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--hosts", type=Path, help="Prebuilt FlagGems source hosts")
    parser.add_argument(
        "--model-host", action="store_true", help="Use the generic FlagGems model host"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--profile-kernels", type=Path, help="Export a separate kernel-provenance trace"
    )
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--logits-atol", type=float, default=1e-3)
    parser.add_argument("--logits-rtol", type=float, default=1e-3)
    parser.add_argument(
        "--logits-policy",
        choices=["elementwise", "bf16-v2"],
        default="elementwise",
        help="Select strict elementwise or fixed per-token BF16 vector acceptance",
    )
    parser.add_argument(
        "--check-logits",
        action="store_true",
        help="Fail after reporting if the selected logit acceptance policy fails",
    )
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument(
        "--main-cpu", type=int, help="Pin the timing thread after runtime workers start"
    )
    parser.add_argument(
        "--skip-sampling",
        action="store_true",
        help="Omit the separate autoregressive demonstration",
    )
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--snapshot-layers", action="store_true")
    parser.add_argument(
        "--prompt", default="Explain in one sentence why the sky is blue."
    )
    args = parser.parse_args()
    if args.steps < 1 or args.rounds < 0:
        parser.error("steps must be positive and rounds must be nonnegative")
    if args.main_cpu is not None and (
        not hasattr(os, "sched_setaffinity")
        or args.main_cpu not in os.sched_getaffinity(0)
    ):
        parser.error("--main-cpu must name a CPU in this process's allowed affinity")
    if args.check_logits and args.reference is None:
        parser.error("--check-logits requires a vendor --reference")
    if args.logits_policy == "bf16-v2":
        if args.dtype != "bfloat16":
            parser.error("bf16-v2 requires the vendor bfloat16 workload")
        if (args.logits_atol, args.logits_rtol) != (1e-3, 1e-3):
            parser.error("bf16-v2 has fixed limits; atol/rtol apply to elementwise")
    try:
        validate_tolerances(args.logits_atol, args.logits_rtol)
    except ValueError as error:
        parser.error(str(error))
    if args.model_host and (args.device != "flagos:0" or args.mode != "plain"):
        parser.error("--model-host requires --device flagos:0 --mode plain")
    if args.profile_kernels and args.device != "flagos:0":
        parser.error("--profile-kernels audits FlagGems launches on --device flagos:0")
    hosts = None
    native_scope = nullcontext()
    report = {
        "torch": str(torch.__version__),
        "device": args.device,
        "mode": args.mode,
        "dtype": args.dtype,
        "execution": "flaggems-model-host" if args.model_host else "eager",
        "steps": args.steps,
        "rounds": args.rounds,
        "routing_environment": {
            name: value
            for name, value in sorted(os.environ.items())
            if name.startswith("FLAGOS_OP_")
            or name
            in {"FLAGOS_FORCE_BACKEND", "FLAGOS_BACKEND_CONFIG", "FLAGOS_ALIAS_CUDA"}
        },
        "logits_policy": (
            bf16_vector_policy()
            if args.logits_policy == "bf16-v2"
            else {
                "name": "elementwise",
                "atol": args.logits_atol,
                "rtol": args.logits_rtol,
            }
        ),
    }
    if args.device.startswith("flagos"):
        if args.hosts:
            sys.path.insert(0, str(args.hosts.resolve().parent))
            loader_path = (
                Path(__file__).resolve().parents[2] / "torch_fl/native_host/load.py"
            )
            spec = importlib.util.spec_from_file_location("host_loader", loader_path)
            loader = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(loader)
            loader.install_launcher()
        import_started = time.perf_counter()
        import torch_fl
        from torch_fl import _build_config, host_compiler

        assert torch.version.cuda is None and "boxing" not in _build_config.KERNELS
        synchronize = torch_fl.flagos.synchronize
        report["kernels"] = list(_build_config.KERNELS)
        report["config"] = torch_fl.backend_config_path()
        if getattr(_build_config, "NATIVE_HOSTS", False):
            report["torch_fl_import_ms"] = (time.perf_counter() - import_started) * 1000
            hosts = torch_fl.native_host._hosts
            native_scope = torch_fl.native_host.execution_scope(0)
            report["native_execution_scope"] = {
                "vendor_stream_binding": _build_config.ACCELERATOR == "hygon"
                and getattr(_build_config, "NATIVE_VENDOR", False),
                "graph_capture": False,
                "cached_tensors": False,
            }
        if args.hosts:
            started = time.perf_counter()
            hosts = loader.Hosts(args.hosts)
            hosts.activate()
            report["source_host_load_ms"] = (time.perf_counter() - started) * 1000
    elif args.device == "npu:0":
        import torch_npu

        synchronize = torch.npu.synchronize
        report["torch_npu"] = str(torch_npu.__version__)
        report["npu_math_settings"] = {
            "matmul_allow_hf32": torch.npu.matmul.allow_hf32,
            "conv_allow_hf32": torch.npu.conv.allow_hf32,
            "allow_internal_format": getattr(
                torch.npu.config, "allow_internal_format", None
            ),
        }
        if report["npu_math_settings"]["matmul_allow_hf32"]:
            raise RuntimeError("paired native-host acceptance requires HF32 disabled")
    else:
        synchronize = torch.cuda.synchronize

    report["math_settings"] = math_settings()

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model,
            local_files_only=True,
            torch_dtype=getattr(torch, args.dtype),
            attn_implementation="eager",
        )
        .eval()
        .to(args.device)
    )
    if args.model_host:
        model = torch_fl.native_host.compile(model)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(args.device)
    report.update(
        layers=len(model.model.layers),
        parameters=sum(p.numel() for p in model.parameters()),
        prompt_tokens=ids.shape[-1],
    )
    reference = (
        torch.load(args.reference, weights_only=True) if args.reference else None
    )
    checkpoint_files = sorted(args.model.glob("*.safetensors"))
    if args.check_logits and not checkpoint_files:
        raise ValueError("logit acceptance requires local safetensors checkpoint files")
    contract = {
        "version": 1,
        "checkpoint_sha256": {
            path.name: file_sha256(path) for path in checkpoint_files
        },
        "config_sha256": file_sha256(args.model / "config.json"),
        "dtype": args.dtype,
        "attention": "eager",
        "math_settings": report["math_settings"],
    }
    report["contract"] = contract
    if reference is not None:
        validate_reference(
            reference,
            args.steps,
            contract=contract if args.check_logits else None,
            prompt_ids=ids,
        )
        report["reference_sha256"] = file_sha256(args.reference)
    if args.mode != "plain":
        if args.device != "flagos:0" or args.cache is None:
            raise ValueError(
                "host compiler modes require --device flagos:0 and --cache"
            )
        host_compiler.enable(args.cache, mode=args.mode)

    layers = {}
    hooks = []
    if args.snapshot_layers:
        report["parameter_layouts"] = {}
        for name, parameter in model.named_parameters():
            if (
                name.startswith("model.layers.0.")
                or name == "model.embed_tokens.weight"
            ):
                layout = {
                    "shape": list(parameter.shape),
                    "stride": list(parameter.stride()),
                    "dtype": str(parameter.dtype),
                }
                if args.device == "npu:0":
                    get_format = getattr(torch_npu, "get_npu_format", None)
                    layout["npu_format"] = (
                        int(get_format(parameter)) if get_format is not None else None
                    )
                report["parameter_layouts"][name] = layout

        def save(name):
            def hook(module, inputs, output):
                if isinstance(output, tuple):
                    output = output[0]
                layers[name] = output.detach().cpu()

            return hook

        for name, module in model.named_modules():
            if name.startswith("model.layers.0.") or name in {
                "model.embed_tokens",
                "model.rotary_emb",
                "model.norm",
            }:
                hooks.append(module.register_forward_hook(save(name)))

    with torch.no_grad(), native_scope:
        synchronize()
        started = time.perf_counter()
        out = model(ids, use_cache=True)
        synchronize()
        report["cold_prefill_ms"] = (time.perf_counter() - started) * 1000
        for hook in hooks:
            hook.remove()
        prefill_logits = out.logits.cpu()
        logits = [prefill_logits[:, -1, :]]
        predictions, inputs = [], []
        for step in range(args.steps):
            predicted = out.logits[:, -1, :].argmax(-1, keepdim=True)
            predictions.append(predicted.cpu())
            token = (
                reference["inputs"][step].to(args.device) if reference else predicted
            )
            inputs.append(token.cpu())
            out = model(token, past_key_values=out.past_key_values, use_cache=True)
            synchronize()
            logits.append(out.logits[:, -1, :].cpu())
            if args.progress_every and (step + 1) % args.progress_every == 0:
                print(
                    json.dumps({"decode_step": step + 1, "steps": args.steps}),
                    flush=True,
                )
        report["teacher_forced_predictions"] = [int(t.item()) for t in predictions]
        if reference:
            report["reference_comparison"] = [
                compare(a, b) for a, b in zip(logits, reference["logits"])
            ]

            def accuracy_check(actual, expected):
                if args.logits_policy == "bf16-v2":
                    return bf16_vector_accuracy(actual, expected)
                return tensor_accuracy(
                    actual, expected, atol=args.logits_atol, rtol=args.logits_rtol
                )

            accuracy = [
                accuracy_check(a, b) for a, b in zip(logits, reference["logits"])
            ]
            if "prefill_logits" in reference:
                accuracy[0] = accuracy_check(
                    prefill_logits, reference["prefill_logits"]
                )
            report["logits_scope"] = {
                "prefill": (
                    "all_positions"
                    if "prefill_logits" in reference
                    else "last_position_legacy_reference"
                ),
                "decode": "every_requested_step",
            }
            report["logits_comparison"] = accuracy
            report["logits_validation"] = (
                summarize_bf16_accuracy(accuracy, args.steps + 1)
                if args.logits_policy == "bf16-v2"
                else summarize_accuracy(
                    accuracy,
                    args.steps + 1,
                    atol=args.logits_atol,
                    rtol=args.logits_rtol,
                )
            )
            if layers and reference.get("layers"):
                report["layer_comparison"] = {
                    name: compare(value, reference["layers"][name])
                    for name, value in layers.items()
                }

        # A separate, genuinely autoregressive sample; teacher-forced predictions
        # above must never be concatenated and reported as generated text.
        generated = []
        if not args.skip_sampling:
            out = model(ids, use_cache=True)
            for _ in range(args.steps):
                token = out.logits[:, -1, :].argmax(-1, keepdim=True)
                generated.append(int(token.item()))
                out = model(token, past_key_values=out.past_key_values, use_cache=True)
            report["generated_tokens"] = generated
            report["generated_text"] = tokenizer.decode(
                generated, skip_special_tokens=True
            )

        timings = {
            "traversal_complete_ms": [],
            "prefill_submit_ms": [],
            "prefill_complete_ms": [],
            "decode_submit_ms": [],
            "decode_complete_ms": [],
        }
        if hasattr(os, "sched_getaffinity"):
            report["cpu_affinity_before_timing"] = sorted(os.sched_getaffinity(0))
        if args.main_cpu is not None:
            os.sched_setaffinity(0, {args.main_cpu})
        if hasattr(os, "sched_getaffinity"):
            report["measurement_main_cpu_affinity"] = sorted(os.sched_getaffinity(0))
        device_inputs = [token.to(args.device) for token in inputs]
        for repetition in range(args.rounds + 1 if args.rounds else 0):
            synchronize()
            traversal_started = time.perf_counter()
            t0 = time.perf_counter()
            out = model(ids, use_cache=True)
            t1 = time.perf_counter()
            synchronize()
            t2 = time.perf_counter()
            if repetition:
                timings["prefill_submit_ms"].append((t1 - t0) * 1000)
                timings["prefill_complete_ms"].append((t2 - t0) * 1000)
            for token in device_inputs:
                synchronize()
                t0 = time.perf_counter()
                out = model(token, past_key_values=out.past_key_values, use_cache=True)
                t1 = time.perf_counter()
                synchronize()
                t2 = time.perf_counter()
                if repetition:
                    timings["decode_submit_ms"].append((t1 - t0) * 1000)
                    timings["decode_complete_ms"].append((t2 - t0) * 1000)
            synchronize()
            if repetition:
                timings["traversal_complete_ms"].append(
                    (time.perf_counter() - traversal_started) * 1000
                )
        report["timings"] = timings
        report["medians"] = {
            name: statistics.median(values)
            for name, values in timings.items()
            if values
        }
        if args.rounds:
            # Preserve completed timing evidence if a later provenance audit fails.
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.with_suffix(".timings.json").write_text(
                json.dumps(
                    {
                        "acceptance": "pending remaining accuracy and provenance audits",
                        "contract": contract,
                        "math_settings": report["math_settings"],
                        "timings": timings,
                        "medians": report["medians"],
                        "logits_validation": report.get("logits_validation"),
                    },
                    indent=2,
                )
            )
        calls = Counter()

        def profile(frame, event, arg):
            path = frame.f_code.co_filename
            if event == "call" and ("/flag_gems/" in path or "/triton/" in path):
                calls[path + ":" + frame.f_code.co_name] += 1

        sys.setprofile(profile)
        try:
            out = model(ids, use_cache=True)
            out = model(
                device_inputs[0], past_key_values=out.past_key_values, use_cache=True
            )
            synchronize()
        finally:
            sys.setprofile(None)
        report["warm_python_calls"] = dict(calls)
        if args.profile_kernels:
            if hosts is None:
                raise RuntimeError("kernel provenance requires the native-host wheel")
            from torch.profiler import ProfilerActivity, profile as torch_profile

            with torch_profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1]
            ) as profiler:
                out = model(ids, use_cache=True)
                out = model(
                    device_inputs[0],
                    past_key_values=out.past_key_values,
                    use_cache=True,
                )
                synchronize()
            args.profile_kernels.parent.mkdir(parents=True, exist_ok=True)
            profiler.export_chrome_trace(str(args.profile_kernels))
            trace = json.loads(args.profile_kernels.read_text())
            events = trace["traceEvents"]
            from kernel_provenance import audit_kernels

            admitted = hosts.audit()["launched_kernels"]
            report["gpu_kernel_audit"] = audit_kernels(events, admitted, vendor_audit())
        if hosts is not None:
            report["kernel_sources"] = hosts.provenance
            report["native_audit"] = hosts.audit()
            report["vendor_precision_audit"] = vendor_audit()
            if hasattr(hosts, "typed"):
                report["native_source_stats"] = torch.ops.flagos.native_source_stats()
                report["typed_variants"] = len(hosts.typed["compiled"])
                report["typed_fallbacks"] = len(hosts.typed["fallback"])
                if hasattr(torch.ops.flagos, "native_source_typed_stats"):
                    report["typed_entry_stats"] = (
                        torch.ops.flagos.native_source_typed_stats()
                    )
        if args.device == "flagos:0":
            report["host_program_stats"] = torch.ops.flagos.host_program_stats()
        if args.model_host:
            report["model_host_graphs"] = model.forward._flagos_host_backend.graphs
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report["math_settings_after"] = math_settings()
    producer = {
        "backend": (
            "vendor"
            if args.device in {"cuda:0", "npu:0"}
            and "torch_fl" not in sys.modules
            and "flag_gems" not in sys.modules
            else "torch_fl"
        ),
        "torch": str(torch.__version__),
        "math_settings_after": report["math_settings_after"],
    }
    report["producer"] = producer
    torch.save(
        {
            "logits": logits,
            "prefill_logits": prefill_logits,
            "inputs": inputs,
            "layers": layers,
            "contract": contract,
            "prompt_ids": ids.cpu(),
            "producer": producer,
        },
        args.output.with_suffix(".pt"),
    )
    args.output.write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in {"timings", "warm_python_calls"}
            },
            indent=2,
        ),
        flush=True,
    )
    if args.check_logits and (
        not report["logits_validation"]["passed"]
        or report["math_settings_after"] != report["math_settings"]
    ):
        raise LogitsValidationFailed(
            f"logit acceptance failed; full accuracy and timing report: {args.output}"
        )


if __name__ == "__main__":
    main()
