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

"""Full official Qwen3 eager prefill/decode, with a separate vendor reference.

No torch.compile, CUDA graphs, layer truncation, replacement model or output
pooling is used. Run each backend in its own process on an otherwise idle GPU.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", required=True, choices=["cuda:0", "flagos:0"])
    parser.add_argument(
        "--mode", choices=["plain", "collect", "compile", "load"], default="plain"
    )
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--snapshot-layers", action="store_true")
    args = parser.parse_args()
    report = {
        "torch": torch.__version__,
        "device": args.device,
        "mode": args.mode,
        "dtype": args.dtype,
    }
    if args.device.startswith("flagos"):
        import torch_fl
        from torch_fl import _build_config, host_compiler

        assert torch.version.cuda is None and "boxing" not in _build_config.KERNELS
        synchronize = torch_fl.flagos.synchronize
        report["kernels"] = list(_build_config.KERNELS)
        report["config"] = torch_fl.backend_config_path()
    else:
        synchronize = torch.cuda.synchronize

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model,
            local_files_only=True,
            dtype=getattr(torch, args.dtype),
            attn_implementation="eager",
        )
        .eval()
        .to(args.device)
    )
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Explain in one sentence why the sky is blue."}],
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
    if args.mode != "plain":
        if args.device != "flagos:0" or args.cache is None:
            raise ValueError(
                "host compiler modes require --device flagos:0 and --cache"
            )
        host_compiler.enable(args.cache, mode=args.mode)

    layers = {}
    hooks = []
    if args.snapshot_layers:

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

    with torch.no_grad():
        synchronize()
        started = time.perf_counter()
        out = model(ids, use_cache=True)
        synchronize()
        report["cold_prefill_ms"] = (time.perf_counter() - started) * 1000
        for hook in hooks:
            hook.remove()
        logits = [out.logits[:, -1, :].cpu()]
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
        report["teacher_forced_predictions"] = [int(t.item()) for t in predictions]
        if reference:
            report["reference_comparison"] = [
                compare(a, b) for a, b in zip(logits, reference["logits"])
            ]
            if layers and reference.get("layers"):
                report["layer_comparison"] = {
                    name: compare(value, reference["layers"][name])
                    for name, value in layers.items()
                }

        # A separate, genuinely autoregressive sample; teacher-forced predictions
        # above must never be concatenated and reported as generated text.
        out = model(ids, use_cache=True)
        generated = []
        for _ in range(args.steps):
            token = out.logits[:, -1, :].argmax(-1, keepdim=True)
            generated.append(int(token.item()))
            out = model(token, past_key_values=out.past_key_values, use_cache=True)
        report["generated_tokens"] = generated
        report["generated_text"] = tokenizer.decode(generated, skip_special_tokens=True)

        timings = {
            "prefill_submit_ms": [],
            "prefill_complete_ms": [],
            "decode_submit_ms": [],
            "decode_complete_ms": [],
        }
        device_inputs = [token.to(args.device) for token in inputs]
        for repetition in range(args.rounds + 1):
            synchronize()
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
        report["timings"] = timings
        report["medians"] = {
            name: statistics.median(values) for name, values in timings.items()
        }
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
        if args.device == "flagos:0":
            report["host_program_stats"] = torch.ops.flagos.host_program_stats()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"logits": logits, "inputs": inputs, "layers": layers},
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


if __name__ == "__main__":
    main()
