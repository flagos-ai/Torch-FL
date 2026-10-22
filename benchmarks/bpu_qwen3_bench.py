#!/usr/bin/env python3
"""End-to-end Qwen3 generation on the BPU, against the vendor `llm` demo.

Drives the vendor's own two-graph .hbm (`prefill` + `decode`) through
torch_fl's `infer.Package`, so the comparison isolates the runtime: same
artifact, same four BPU cores, same weights. What differs is only who submits
the work.

    python benchmarks/bpu_qwen3_bench.py \
        --hbm ~/llm_sdk/.../Qwen3-0.6B_language_chunk_512_cache_4096_w8_nash-p_corenum_4_4.hbm \
        --tokenizer ~/llm_sdk/.../configs/Qwen3_config

The vendor number to beat comes from its own demo, which prints it directly:

    cd oellm_runtime/examples/llm_demo && ./llm -c qwen3_0.6b_config.json
    [Performance] prefill TPS: 5626 tokens/s  decode TPS: 84.8 tokens/s

The KV cache is a sliding window over one allocation per layer, which is what
the vendor runtime does and the only layout that produces correct text -- see
`KVWindow` for why appending into the cache does not.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from torch_fl.accelerator.bpu.infer import KVWindow, Package  # noqa: E402

# Additive mask: 0 attends, -65504 (f16 min) blocks.
BLOCK = np.float16(-65504.0)
OPEN = np.float16(0.0)


class Qwen3BPU:
    """Prefill/decode driver over one multi-model .hbm.

    Both graphs share a single `KVWindow`, so a prefill step needs no
    publishing into the decode graph -- there is only one cache.
    """

    def __init__(self, hbm: str, cores=(0, 1, 2, 3), max_tokens: int = 1024):
        self.pkg = Package(hbm)
        for need in ("prefill", "decode"):
            if need not in self.pkg.model_names:
                raise SystemExit(
                    f"{hbm} has models {self.pkg.model_names}, expected a "
                    f"'{need}' graph -- this is not an LLM artifact"
                )
        self.decode = self.pkg.model("decode", cores)
        self.prefill = self.pkg.model("prefill", cores)

        self.layers = sum(
            1 for n in self.decode.input_names if n.endswith("_cache_key")
        )
        self.chunk = self.prefill.inputs["input_ids"].shape[1]
        self.context = self.decode.inputs["attention_mask"].shape[-1]
        # A prefill pass writes `chunk` slots past the window, so the room
        # beyond it has to cover the widest single step, not just the tokens.
        self.kv = KVWindow(
            (self.decode, self.prefill),
            self.layers,
            self.context,
            max_tokens + self.chunk,
        )

    def reset(self) -> None:
        self.kv.reset()

    def run_prefill(self, ids: list[int]) -> int:
        """Consume `ids` in chunk-sized passes; returns the first sampled token."""
        tok = 0
        m = self.prefill
        for start in range(0, len(ids), self.chunk):
            piece = ids[start : start + self.chunk]
            n = len(piece)
            self.kv.bind(m, self.chunk)
            m.inputs["input_ids"][:] = 0
            m.inputs["input_ids"][0, :n] = piece
            m.inputs["position_ids"][0, :] = np.arange(
                self.kv.pos, self.kv.pos + self.chunk, dtype=np.int32
            )
            mask = m.inputs["attention_mask"]
            mask[:] = BLOCK
            for r in range(n):
                lo, hi = self.kv.mask_range(self.chunk, r)
                mask[0, r, lo:hi] = OPEN
            m.flush_inputs(["input_ids", "position_ids", "attention_mask"])
            m.infer()
            m.invalidate_outputs()
            tok = int(np.argmax(m.outputs["logits"][0, n - 1]))
            # Advance by the real tokens, not the padded chunk: the device
            # wrote `chunk` slots but only the first `n` mean anything, and
            # sliding by `n` leaves the rest beyond the next window's edge.
            self.kv.advance(n)
        return tok

    def step(self, tok: int) -> int:
        """One decode step at the current position."""
        m = self.decode
        self.kv.bind(m, 1)
        m.inputs["input_ids"][:] = tok
        m.inputs["position_ids"][:] = self.kv.pos
        mask = m.inputs["attention_mask"]
        mask[:] = BLOCK
        lo, hi = self.kv.mask_range(1)
        mask[0, 0, lo:hi] = OPEN
        m.flush_inputs(["input_ids", "position_ids", "attention_mask"])
        m.infer()
        m.invalidate_outputs()
        self.kv.advance(1)
        return int(np.argmax(m.outputs["logits"][0, 0]))

    def free(self) -> None:
        self.kv.free()
        self.decode.free()
        self.prefill.free()
        self.pkg.release()


def load_tokenizer(path: Path):
    from tokenizers import Tokenizer

    tk = Tokenizer.from_file(str(path / "tokenizer.json"))
    cfg = json.loads((path / "generation_config.json").read_text())
    eos = cfg.get("eos_token_id")
    eos = set(eos) if isinstance(eos, list) else {eos} if eos is not None else set()
    return tk, eos


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hbm", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--prompt", default="用一句话介绍杭州西湖。")
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--cores", default="0,1,2,3")
    args = ap.parse_args()

    cores = tuple(int(c) for c in args.cores.split(","))
    tk, eos_ids = load_tokenizer(Path(args.tokenizer).expanduser())

    t0 = time.perf_counter()
    model = Qwen3BPU(
        str(Path(args.hbm).expanduser()), cores, max_tokens=args.max_new + 64
    )
    load_s = time.perf_counter() - t0
    print(
        f"loaded in {load_s:.2f}s: {model.pkg.model_names}, "
        f"{model.layers} layers, chunk {model.chunk}, context {model.context}"
    )

    # Qwen3 chat template, thinking disabled to keep the run short.
    text = (
        f"<|im_start|>user\n{args.prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    ids = tk.encode(text, add_special_tokens=False).ids
    print(f"prompt: {len(ids)} tokens")

    t0 = time.perf_counter()
    tok = model.run_prefill(ids)
    prefill_s = time.perf_counter() - t0

    out: list[int] = []
    t0 = time.perf_counter()
    for _ in range(args.max_new):
        if tok in eos_ids:
            break
        out.append(tok)
        tok = model.step(tok)
    decode_s = time.perf_counter() - t0

    print("\n--- generated ---")
    print(tk.decode(out))
    print("\n--- performance ---")
    print(
        f"prefill: {len(ids)} tok in {prefill_s * 1000:.1f} ms "
        f"-> {len(ids) / prefill_s:.0f} tok/s"
    )
    if out:
        print(
            f"decode : {len(out)} tok in {decode_s * 1000:.1f} ms "
            f"-> {len(out) / decode_s:.1f} tok/s "
            f"({decode_s / len(out) * 1000:.2f} ms/token)"
        )
    print(
        "vendor : prefill 5626 tok/s, decode 84.8 tok/s "
        "(oellm_runtime llm demo, same .hbm)"
    )
    model.free()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
