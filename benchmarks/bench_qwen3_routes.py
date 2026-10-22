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

"""Qwen3 three-route performance benchmark and profiler driver.

Runs the same greedy-decode workload on one route per process (routing is
fixed at import time), driven three times by the operator:

  vendor   : system python, --device cuda   (vendor fork torch, baseline)
  boxing   : /opt/fl-envs/boxing python, --device flagos
  flaggems : same venv + FLAGOS_FORCE_BACKEND=flaggems, --device flagos

Modes:
  bench   : 3 prompts x (1 warmup + --iters measured rounds); per-step timing.
  profile : single medium prompt, 1 prefill + --profile-steps decode steps
            under torch.profiler; chrome trace + top-op tables.

Metrics:
  ttft_ms     prefill latency for the prompt (one full forward)
  fwd_ms      mean single-forward latency during decode
  decode_tps  decode steps / decode wall time
  e2e_tps     new tokens / total round wall time
  total_s     round wall time (prompt -> last token)
"""

import argparse
import json
import os
import time

import torch

PROMPTS = {
    "short_zh": "介绍一下海光 DCU。",
    "medium_zh": "请用三句话说明为什么大语言模型推理分为 prefill 和 decode 两个阶段，以及它们各自的瓶颈是什么。",
    "medium_en": "Explain in three sentences why matrix multiplication dominates the runtime of transformer inference.",
}


def sync():
    if torch.cuda.is_available() and torch.cuda.current_device() == 0:
        torch.cuda.synchronize()
    else:
        torch.flagos.synchronize()


def load_model(model_path, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def greedy_round(model, ids, max_new_tokens):
    """One greedy round: prefill + decode, returning per-step timings."""
    sync()
    t0 = time.perf_counter()
    out = model(ids, use_cache=True)
    sync()
    ttft = time.perf_counter() - t0

    past = out.past_key_values
    next_id = out.logits[:, -1, :].argmax(-1, keepdim=True)
    gen_ids = [next_id.item()]
    step_times = []

    for _ in range(max_new_tokens - 1):
        sync()
        ts = time.perf_counter()
        out = model(next_id, past_key_values=past, use_cache=True)
        sync()
        step_times.append(time.perf_counter() - ts)
        past = out.past_key_values
        next_id = out.logits[:, -1, :].argmax(-1, keepdim=True)
        tok = next_id.item()
        gen_ids.append(tok)
        if tok == model.generation_config.eos_token_id:
            break

    total = ttft + sum(step_times)
    return {
        "ttft_ms": ttft * 1e3,
        "fwd_ms": (sum(step_times) / max(len(step_times), 1)) * 1e3,
        "decode_tps": len(step_times) / max(sum(step_times), 1e-9),
        "e2e_tps": len(gen_ids) / total,
        "total_s": total,
        "new_tokens": len(gen_ids),
    }, gen_ids


def run_bench(model, tokenizer, device, iters, max_new_tokens):
    results = {}
    for name, text in PROMPTS.items():
        ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
        # warmup
        greedy_round(model, ids, max_new_tokens)
        rounds = []
        for _ in range(iters):
            metrics, _ = greedy_round(model, ids, max_new_tokens)
            rounds.append(metrics)
        # mean over rounds (new_tokens stays an int)
        results[name] = {k: sum(r[k] for r in rounds) / iters for k in rounds[0]}
        results[name]["new_tokens"] = rounds[0]["new_tokens"]
        results[name]["prompt_tokens"] = ids.shape[1]
    return results


def run_profile(model, tokenizer, device, profile_steps, trace_dir, cpu_only=False):
    from torch.profiler import ProfilerActivity, profile

    if cpu_only:
        activities = [ProfilerActivity.CPU]
    elif device.startswith("flagos"):
        activities = [ProfilerActivity.CPU, ProfilerActivity.PrivateUse1]
    else:
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

    ids = tokenizer(PROMPTS["medium_zh"], return_tensors="pt").input_ids.to(device)
    # warmup outside the profiled window
    greedy_round(model, ids, profile_steps)

    with torch.no_grad():
        with profile(activities=activities) as prof:
            sync()
            out = model(ids, use_cache=True)
            past = out.past_key_values
            next_id = out.logits[:, -1, :].argmax(-1, keepdim=True)
            for _ in range(profile_steps - 1):
                out = model(next_id, past_key_values=past, use_cache=True)
                past = out.past_key_values
                next_id = out.logits[:, -1, :].argmax(-1, keepdim=True)
            sync()

    os.makedirs(trace_dir, exist_ok=True)
    trace_path = os.path.join(trace_dir, f"trace_{device.replace(':', '_')}.json")
    prof.export_chrome_trace(trace_path)

    dev_key = "self_device_time_total"
    lines = {
        "top_device": prof.key_averages().table(
            sort_by=dev_key, row_limit=30, max_name_column_width=60
        ),
        "top_cpu": prof.key_averages().table(
            sort_by="self_cpu_time_total", row_limit=30, max_name_column_width=60
        ),
    }
    return {"trace": trace_path, **lines}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/models/Qwen3-0.6B")
    ap.add_argument("--device", required=True, choices=["cuda", "flagos"])
    ap.add_argument("--mode", default="bench", choices=["bench", "profile"])
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--profile-steps", type=int, default=32)
    ap.add_argument(
        "--cpu-only",
        action="store_true",
        help="profile CPU ops only (skip device tracer)",
    )
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--trace-dir", default=None)
    args = ap.parse_args()

    if args.device == "flagos":
        import torch_fl  # noqa: F401  -- registers the flagos device

    torch.manual_seed(0)
    route = os.environ.get("ROUTE_LABEL", args.device)
    device = f"{args.device}:0"
    model, tokenizer = load_model(args.model, device)

    if args.mode == "bench":
        results = run_bench(model, tokenizer, device, args.iters, args.max_new_tokens)
        print(f"\n=== route={route} device={device} ===")
        header = f"{'prompt':<12}{'ttft_ms':>10}{'fwd_ms':>9}{'decode_tps':>11}{'e2e_tps':>9}{'total_s':>8}{'new_tok':>8}"
        print(header)
        for name, m in results.items():
            print(
                f"{name:<12}{m['ttft_ms']:>10.1f}{m['fwd_ms']:>9.2f}"
                f"{m['decode_tps']:>11.1f}{m['e2e_tps']:>9.1f}"
                f"{m['total_s']:>8.2f}{m['new_tokens']:>8d}"
            )
        if args.out_json:
            with open(args.out_json, "w") as f:
                json.dump(
                    {"route": route, "device": device, "results": results}, f, indent=2
                )
    else:
        out = run_profile(
            model,
            tokenizer,
            device,
            args.profile_steps,
            args.trace_dir or "/tmp/qwen3_traces",
            cpu_only=args.cpu_only,
        )
        print(f"trace: {out['trace']}")
        print("\n=== top ops by self device time ===")
        print(out["top_device"])
        print("\n=== top ops by self CPU time ===")
        print(out["top_cpu"])


if __name__ == "__main__":
    main()
