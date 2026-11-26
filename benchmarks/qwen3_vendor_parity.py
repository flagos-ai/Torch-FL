# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Measure eager Qwen3-0.6B training and generation on one accelerator.

Run each backend in its own Python process on the same otherwise idle device.
Measure a full AdamW step or a full greedy generation, including device
synchronization. Load the checkpoint and transfer inputs before measuring.
"""

import argparse
from contextlib import nullcontext
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
from statistics import median
import sys
from time import perf_counter


SPEED_TARGETS = {"train": 1.00, "infer": 0.95}
TRANSFORMERS_VERSION = "5.19.0"
LEARNING_RATE = 1e-4
LOSS_RELATIVE_LIMIT = 0.02
TRAINING_SAMPLE_RELATIVE_L2_LIMIT = 0.10
PROMPT = (
    "Explain how a language model predicts the next word. Include a short example "
    "and explain how the answer changes when the preceding sentence changes."
)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def checkpoint_identity(model_path):
    files = [
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    return {name: digest(model_path / name) for name in files}


def measure(step, synchronize, origins):
    synchronize()
    before = origins() if origins is not None else None
    start = perf_counter()
    result = step()
    synchronize()
    elapsed = perf_counter() - start
    enqueues = None
    if before is not None:
        enqueues = {
            origin: counter_delta(before[origin], counts)
            for origin, counts in origins().items()
        }
    return elapsed, result, enqueues


def parameter_sample(model, *, gradient=False):
    sample = {}
    for name, parameter in model.named_parameters():
        value = parameter.grad if gradient else parameter
        if value is None:
            raise RuntimeError(f"Missing gradient for {name}")
        flat = value.detach().reshape(-1)
        sample[name] = flat[:: max(1, flat.numel() // 128)][:128].cpu().float().tolist()
    return sample


def train(torch, model, device, synchronize, args, origins, *, foreach):
    model.train()
    rng = torch.Generator().manual_seed(42)
    inputs = [
        torch.randint(
            0,
            model.config.vocab_size,
            (args.batch_size, args.prompt_tokens),
            generator=rng,
        ).to(device)
        for _ in range(args.warmup + args.rounds)
    ]
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, foreach=foreach)
    losses, timings, submissions, samples = [], [], [], {}

    for i, ids in enumerate(inputs):

        def step():
            optimizer.zero_grad(set_to_none=True)
            loss = model(input_ids=ids, labels=ids, use_cache=False).loss
            loss.backward()
            optimizer.step()
            return loss.detach()

        check_step = i == 0 or i == len(inputs) - 1
        before = parameter_sample(model) if check_step else None
        elapsed, loss, enqueues = measure(step, synchronize, origins)
        loss = float(loss.cpu())
        if not math.isfinite(loss):
            raise RuntimeError(f"Nonfinite training loss in step {i + 1}")
        if check_step:
            after = parameter_sample(model)
            samples[str(i)] = {
                "gradients": parameter_sample(model, gradient=True),
                "updates": {
                    name: [new - old for new, old in zip(values, before[name])]
                    for name, values in after.items()
                },
            }
        losses.append(loss)
        timings.append(elapsed)
        if enqueues is not None:
            submissions.append(enqueues)
        print(f"step {i + 1}: {elapsed:.4f}s, loss {loss:.6f}", flush=True)

    return timings, {"losses": losses, "samples": samples}, submissions


def infer(
    torch, model, tokenizer, generation_config, device, synchronize, args, origins
):
    model.eval()
    prompt = tokenizer(PROMPT, add_special_tokens=False)["input_ids"]
    repeats = (args.prompt_tokens + len(prompt) - 1) // len(prompt)
    ids = torch.tensor((prompt * repeats)[: args.prompt_tokens])[None, :]
    mask = torch.ones_like(ids).expand(args.batch_size, -1).contiguous().to(device)
    ids = ids.expand(args.batch_size, -1).contiguous().to(device)
    # Older Transformers versions merge model defaults into an explicit
    # config. Replace the default so the checkpoint cannot enable sampling.
    model.generation_config = generation_config(
        do_sample=False,
        min_new_tokens=args.new_tokens,
        max_new_tokens=args.new_tokens,
        use_cache=True,
        eos_token_id=None,
        pad_token_id=tokenizer.pad_token_id,
        cache_implementation="dynamic",
        disable_compile=True,
    )
    outputs, timings, submissions = [], [], []

    with torch.inference_mode():
        for i in range(args.warmup + args.rounds):
            elapsed, tokens, enqueues = measure(
                lambda: model.generate(input_ids=ids, attention_mask=mask),
                synchronize,
                origins,
            )
            generated = tokens[:, args.prompt_tokens :].cpu()
            if generated.shape != (args.batch_size, args.new_tokens):
                raise RuntimeError(
                    f"Unexpected generated token shape {generated.shape}"
                )
            outputs.append(generated.tolist())
            timings.append(elapsed)
            if enqueues is not None:
                submissions.append(enqueues)
            print(f"generation {i + 1}: {elapsed:.4f}s", flush=True)

        observed = []

        def record_logits(_head, _inputs, logits):
            observed.append(logits.detach().cpu())

        # Generation normally projects only the last prefill position. Collect
        # raw logits in a separate run with the same mask and cache, before the
        # generation code converts the last position to float32 for sampling.
        with model.lm_head.register_forward_hook(record_logits):
            traced = model.generate(
                input_ids=ids, attention_mask=mask, logits_to_keep=0
            )
        trace_tokens = traced[:, args.prompt_tokens :].cpu().tolist()
        if (
            any(tokens != trace_tokens for tokens in outputs)
            or len(observed) != args.new_tokens
        ):
            raise RuntimeError("Logit collection changed the generated token sequence")
        logits = {
            "prefill": observed[0].contiguous(),
            "generation": torch.stack([value[:, -1] for value in observed], dim=1),
        }
    from safetensors.torch import save_file

    vocabulary = model.config.vocab_size
    if logits["prefill"].shape != (args.batch_size, args.prompt_tokens, vocabulary):
        raise RuntimeError("Expected logits for every prefill position")
    if logits["generation"].shape != (args.batch_size, args.new_tokens, vocabulary):
        raise RuntimeError("Expected logits for every generation step")
    path = logits_path(args.output)
    save_file(logits, str(path))
    return timings, {"tokens": outputs, "logits_sha256": digest(path)}, submissions


def logits_path(report_path):
    return report_path.with_suffix(".logits.safetensors")


def compare_logits(actual_report, expected_report, output, reference):
    import torch
    from host_compiler.logits_validation import (
        bf16_vector_accuracy,
        summarize_bf16_accuracy,
        tensor_accuracy,
    )
    from safetensors.torch import load_file

    actual_path, expected_path = logits_path(output), logits_path(reference)
    if (
        digest(actual_path) != actual_report["logits_sha256"]
        or digest(expected_path) != expected_report["logits_sha256"]
    ):
        raise ValueError("Logits do not match their recorded digest")
    actual, expected = load_file(str(actual_path)), load_file(str(expected_path))
    phases = ("prefill", "generation")
    bf16 = expected["prefill"].dtype == torch.bfloat16
    check = bf16_vector_accuracy if bf16 else tensor_accuracy
    reports = {phase: check(actual[phase], expected[phase]) for phase in phases}
    if bf16:
        summary = summarize_bf16_accuracy(list(reports.values()), len(phases))
        for phase, group in (("prefill", "prefill"), ("generation", "decode")):
            metrics = summary["phases"][group]
            reports[phase].update(
                phase_relative_l2=metrics["relative_l2"], passed=metrics["passed"]
            )
    return reports


def compare_samples(actual, expected):
    if not expected or actual.keys() != expected.keys():
        raise ValueError("Candidate and reference training parameter sets differ")
    violations = []
    max_error = 0.0
    for name, values in expected.items():
        candidate = actual[name]
        if len(candidate) != len(values):
            raise ValueError(f"Training sample size differs for {name}")
        if not all(math.isfinite(v) for v in (*values, *candidate)):
            violations.append(name)
            continue
        delta = math.sqrt(sum((a - b) ** 2 for a, b in zip(candidate, values)))
        norm = math.sqrt(sum(v**2 for v in values))
        relative_error = delta / max(norm, 1e-8)
        max_error = max(max_error, relative_error)
        if relative_error > TRAINING_SAMPLE_RELATIVE_L2_LIMIT:
            violations.append(name)
    return {"max_relative_l2": max_error, "violations": violations}


def compare(report, vendor, output, reference):
    if vendor["role"] != "vendor" or vendor["device"].startswith("flagos"):
        raise ValueError("Reference must come from a separate vendor Torch process")
    for field in ("checkpoint", "workload", "transformers", "safetensors", "host"):
        if report[field] != vendor[field]:
            raise ValueError(f"Reference differs in {field}")
    ratio = report["tokens_per_second"] / vendor["tokens_per_second"]
    mode = report["workload"]["mode"]
    speed_target = SPEED_TARGETS[mode]
    actual, expected = report["correctness"], vendor["correctness"]
    if mode == "infer":
        agreement = {
            "tokens_equal": actual["tokens"] == expected["tokens"],
            "logits": compare_logits(actual, expected, output, reference),
        }
        valid = agreement["tokens_equal"] and all(
            result["passed"] for result in agreement["logits"].values()
        )
    else:
        if len(actual["losses"]) != len(expected["losses"]):
            raise ValueError("Candidate and reference training step counts differ")
        losses = list(zip(actual["losses"], expected["losses"]))
        max_loss_error = max(abs(a - b) / max(abs(b), 1e-6) for a, b in losses)
        if actual["samples"].keys() != expected["samples"].keys():
            raise ValueError("Candidate and reference sampled different training steps")
        if not expected["samples"] or any(
            not any(
                value for values in samples["gradients"].values() for value in values
            )
            for samples in expected["samples"].values()
        ):
            raise ValueError("Vendor training sampled only zero gradients")
        checks = {
            step: {
                kind: compare_samples(actual["samples"][step][kind], values)
                for kind, values in samples.items()
            }
            for step, samples in expected["samples"].items()
        }
        agreement = {
            "max_relative_loss_error": max_loss_error,
            "samples": checks,
        }
        valid = max_loss_error <= LOSS_RELATIVE_LIMIT and all(
            not metrics["violations"]
            for step in checks.values()
            for metrics in step.values()
        )
    native = report["native_host"]
    apple = report.get("candidate_vendor") == "apple"
    submission_passed = (
        apple
        if native is None
        else (
            not native["python_enqueues"]
            and native["warmup_submission_passed"]
            and native["timed_submission_passed"]
        )
    )
    artifacts = native.get("device_artifacts") if native is not None else None
    precompiled_passed = apple or bool(
        artifacts and artifacts["required"] and artifacts["loaded_specializations"]
    )
    tuning_passed = apple or bool(artifacts and artifacts.get("runtime_tuning_blocked"))
    speed_passed = ratio >= speed_target
    return {
        "throughput_ratio": ratio,
        "throughput_target": speed_target,
        "throughput_required": not apple,
        "throughput_passed": speed_passed,
        "correctness": agreement,
        "correctness_passed": valid,
        "native_submission_passed": submission_passed,
        "precompiled_device_kernels_passed": precompiled_passed,
        "runtime_tuning_passed": tuning_passed,
        "passed": valid
        and submission_passed
        and precompiled_passed
        and tuning_passed
        and (apple or speed_passed),
    }


def native_snapshot(torch, native_host):
    audit = native_host.audit()
    from torch_fl.native_host import vendor

    vendor_audit = vendor.audit()
    return {
        "backend": audit["backend"]["vendor"],
        "compiler_distribution": audit["backend"]["compiler_distribution"],
        "kernels": audit["enqueued_kernels"],
        "device_artifacts": audit.get("device_artifacts"),
        "pointwise_sources": audit.get("pointwise_sources", {}),
        "python_enqueues": native_host.enqueue_origins()["python"],
        "typed_calls": torch.ops.flagos.native_source_typed_stats()[1],
        "vendor_calls": {} if vendor_audit is None else vendor_audit["hits"],
    }


def counter_delta(before, after):
    return {
        name: count - before.get(name, 0)
        for name, count in after.items()
        if count > before.get(name, 0)
    }


def native_evidence(before, after, submissions, warmup, rounds):
    launches = counter_delta(before["kernels"], after["kernels"])
    if not launches:
        raise RuntimeError(
            "The workload did not launch any FlagGems native-host kernels"
        )
    compiler = after["compiler_distribution"]
    installed = importlib.metadata.version(compiler["name"])
    if installed != compiler["version"]:
        raise RuntimeError(
            f"Native Host requires {compiler['name']} {compiler['version']}. "
            f"The installed version is {installed}."
        )
    warmup_enqueues = submissions[:warmup]
    timed_enqueues = submissions[warmup:]
    return {
        "backend": after["backend"],
        "flaggems": importlib.metadata.version("flag-gems"),
        "compiler_distribution": compiler,
        **({"flagtree": installed} if compiler["name"] == "flagtree" else {}),
        "flaggems_kernel_launches": launches,
        "device_artifacts": after.get("device_artifacts"),
        "pointwise_sources": after.get("pointwise_sources", {}),
        "typed_calls": after["typed_calls"] - before["typed_calls"],
        "vendor_calls": counter_delta(before["vendor_calls"], after["vendor_calls"]),
        "python_enqueues": after["python_enqueues"],
        "warmup_enqueues": warmup_enqueues,
        "timed_enqueues": timed_enqueues,
        "warmup_submission_passed": len(warmup_enqueues) == warmup
        and all(
            counts["native"] and not counts["python"] for counts in warmup_enqueues
        ),
        "timed_submission_passed": len(timed_enqueues) == rounds
        and all(counts["native"] and not counts["python"] for counts in timed_enqueues),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--device", required=True)
    parser.add_argument("--vendor-module", action="append", default=[])
    parser.add_argument("--mode", required=True, choices=("train", "infer"))
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--new-tokens", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="use PyTorch deterministic algorithms",
    )
    parser.add_argument(
        "--diagnose-device-kernels",
        action="store_true",
        help="allow runtime device specialization for diagnostics without an acceptance pass",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()

    for name in ("batch_size", "prompt_tokens", "new_tokens", "warmup", "rounds"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.mode == "train" and args.prompt_tokens < 2:
        parser.error("Training needs at least two input tokens")
    candidate = args.device.split(":")[0] == "flagos"
    if (candidate and args.vendor_module) or (not candidate and args.reference):
        parser.error("Vendor modules belong to the baseline and references to Torch-FL")

    transformers_version = importlib.metadata.version("transformers")
    if transformers_version != TRANSFORMERS_VERSION:
        parser.error(
            f"Use Transformers {TRANSFORMERS_VERSION} on every accelerator. "
            f"Found {transformers_version}."
        )
    reference = json.loads(args.reference.read_text()) if args.reference else None
    if reference is not None and (
        reference["workload"].get("deterministic", False) != args.deterministic
    ):
        parser.error("Use the same deterministic setting as the vendor report.")
    if args.mode == "train" and reference is not None:
        if type(reference["workload"].get("optimizer_foreach")) is not bool:
            parser.error(
                "Regenerate the vendor report to record its AdamW foreach mode."
            )

    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    # Initialize PyTorch before loading an optional vendor extension.
    import torch

    native_host = None
    candidate_vendor = None
    if candidate:
        if args.diagnose_device_kernels:
            os.environ.pop("FLAGOS_NATIVE_HOST_REQUIRE_DEVICE_ARTIFACTS", None)
        else:
            os.environ["FLAGOS_NATIVE_HOST_REQUIRE_DEVICE_ARTIFACTS"] = "1"
        import torch_fl
        from torch_fl import _build_config

        candidate_vendor = _build_config.ACCELERATOR
        if candidate_vendor != "apple":
            if not _build_config.NATIVE_HOSTS or set(_build_config.KERNELS) & {
                "boxing",
                "flaggems_cpp",
                "tileops",
            }:
                raise RuntimeError(
                    "Install the FlagGems Native Host wheel for this accelerator"
                )
            from torch_fl import native_host

            if args.diagnose_device_kernels:
                native_host.collect_device_build_outputs()
    vendor_modules = {}
    for name in args.vendor_module:
        module = importlib.import_module(name)
        try:
            version = importlib.metadata.version(name.split(".")[0])
        except importlib.metadata.PackageNotFoundError:
            version = getattr(module, "__version__", None)
        vendor_modules[name] = version

    import transformers

    if transformers.__version__ != transformers_version:
        parser.error(
            f"Installed Transformers {transformers_version}, but imported "
            f"{transformers.__version__}."
        )

    if not candidate and {"torch_fl", "flag_gems"} & sys.modules.keys():
        raise RuntimeError("Vendor Torch baseline loaded Torch-FL or FlagGems")
    device = torch.device(args.device)
    if device.type in ("cpu", "meta"):
        parser.error("Select a real accelerator device")
    accelerator = torch_fl.flagos if candidate else getattr(torch, device.type)
    if hasattr(accelerator, "set_device"):
        accelerator.set_device(device.index or 0)
    synchronize = accelerator.synchronize
    if native_host is not None:
        torch.ops.flagos.strict_host_routes(True)
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    if args.deterministic:
        torch.use_deterministic_algorithms(True)
    torch.manual_seed(42)
    checkpoint = checkpoint_identity(args.model)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=getattr(torch, args.dtype),
        attn_implementation="eager",
    ).to(device)
    if (
        model.config.model_type != "qwen3"
        or model.num_parameters() != 596_049_920
        or any(p.device.type != device.type for p in model.parameters())
    ):
        raise ValueError(
            "Expected the full Qwen3-0.6B checkpoint on the requested device"
        )
    print(f"{args.mode}: {device}, torch {torch.__version__}, {args.dtype}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    native_before = native_snapshot(torch, native_host) if native_host else None
    origins = getattr(native_host, "enqueue_origins", None)
    scope = (
        native_host.execution_scope(device.index or 0)
        if native_host is not None
        else nullcontext()
    )
    foreach = None
    if args.mode == "train":
        if reference is not None:
            foreach = reference["workload"]["optimizer_foreach"]
        elif candidate_vendor == "apple":
            foreach = False
        else:
            from torch.optim.optimizer import _default_to_fused_or_foreach

            _, foreach = _default_to_fused_or_foreach(
                list(model.parameters()), differentiable=False
            )

    with scope:
        if args.mode == "train":
            times, correctness, submissions = train(
                torch, model, device, synchronize, args, origins, foreach=foreach
            )
            tokens_per_round = args.batch_size * (args.prompt_tokens - 1)
        else:
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                args.model, local_files_only=True
            )
            times, correctness, submissions = infer(
                torch,
                model,
                tokenizer,
                transformers.GenerationConfig,
                device,
                synchronize,
                args,
                origins,
            )
            tokens_per_round = args.batch_size * args.new_tokens

    timed = times[args.warmup :]
    report = {
        "role": "torch_fl" if candidate else "vendor",
        "device": str(device),
        "host": os.environ.get("QWEN3_BENCH_HOST", os.uname().nodename),
        "torch": str(torch.__version__),
        "torch_distribution": importlib.metadata.version("torch"),
        "torch_revision": torch.version.git_version,
        "vendor_modules": vendor_modules,
        "torch_fl": importlib.metadata.version("torch-fl") if candidate else None,
        "candidate_vendor": candidate_vendor,
        "native_host": (
            native_evidence(
                native_before,
                native_snapshot(torch, native_host),
                submissions,
                args.warmup,
                args.rounds,
            )
            if native_host is not None
            else None
        ),
        "transformers": transformers_version,
        "safetensors": importlib.metadata.version("safetensors"),
        "checkpoint": checkpoint,
        "workload": {
            "mode": args.mode,
            "execution": "eager",
            "dtype": args.dtype,
            "attention": "eager",
            "batch_size": args.batch_size,
            "prompt_tokens": args.prompt_tokens,
            "new_tokens": args.new_tokens if args.mode == "infer" else None,
            "cache": "dynamic" if args.mode == "infer" else None,
            "warmup": args.warmup,
            "rounds": args.rounds,
            **({"deterministic": True} if args.deterministic else {}),
            **({"optimizer_foreach": foreach} if args.mode == "train" else {}),
        },
        "warmup_seconds": times[: args.warmup],
        "seconds": timed,
        "tokens_per_second": tokens_per_round / median(timed),
        "correctness": correctness,
    }
    if reference is not None:
        report["acceptance"] = compare(report, reference, args.output, args.reference)
    args.output.write_text(json.dumps(report, allow_nan=False) + "\n")
    print(f"{report['tokens_per_second']:.2f} tokens/s", flush=True)
    if args.reference:
        acceptance = report["acceptance"]
        print(
            json.dumps({k: v for k, v in acceptance.items() if k != "correctness"}),
            flush=True,
        )
        return 0 if acceptance["passed"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
