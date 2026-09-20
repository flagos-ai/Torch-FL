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

"""What a measured call costs in arithmetic, and how much of a chip's peak it reaches.

``bench.py`` reports how long a call took. That answers "is this chip fast at this
model", which is not the same question as "is this chip working": a call that
takes 10 s at 2% of the card's arithmetic peak and one that takes 10 s at 60% are
the same latency and completely different results. This file supplies the second
number.

Three readings, and what each one is honest about:

**FLOPs per image.** An analytic model of the 2.1 transformer, not a profile. Its
inputs are the model's own ``config.json`` and the two sequence lengths the model
actually saw (:class:`ShapeHook`), so a reader can redo the arithmetic by hand
against the checkpoint's config and check it. It counts multiply-accumulates on
weights and attention scores and ignores elementwise work, normalisations and the
small modulation MLP: at the shape this model runs, the matmuls are the
arithmetic.

**MFU.** The modelled FLOPs divided by the wall time they happened in, as a
fraction of a peak the caller supplies. Scoped to the denoise loop, because the
loop is what the model covers -- see below.

**MBU.** A short probe of memory-bound elementwise ops at this workload's
activation shape, in GB/s moved and as a fraction of a caller-supplied bandwidth
peak. It exists because this workload is a mix: the attention and the MLPs are
arithmetic-bound and belong to the MFU reading, while the ~40-operator elementwise
cohort between them is bound by how fast the card streams activations, and no
FLOPs number can see that. A chip can be at 25% MFU and 3% MBU at the same time,
and which one it is tells you where the remaining time went.

What is deliberately not modelled, and why the scope is stated wherever a ratio is
reported:

- **The text encoder and the VAE.** Both are one call per image at completely
  different shapes -- a 17.5 GB language model over 26 tokens, and a causal 3-D
  convolutional decoder -- and neither is where step count goes. Modelling them
  would be two more shape-specific derivations to keep correct for a single-digit
  share of the wall time. The ``phases`` section of the JSON already gives their
  time; the ratio here is a ratio for the loop.
- **Elementwise work, norms, and the modulation MLP.** Not counted, which is the
  usual convention for a FLOPs figure quoted as an MFU.

A caller who supplies no peak gets the measurements and no ratio: FLOPs per image
and GB/s per probe op are real either way, while ``mfu`` and ``mbu`` stay
``None``. That is deliberate -- a ratio against a guessed peak is worse than no
ratio at all, because it reads like a result.
"""

import common

#: The shape the bandwidth probe streams, in elements: ``(1, 4096, 4096)`` bf16 is
#: 32 MiB per tensor, which is the size of the activations the denoise loop
#: actually moves (the joint sequence's hidden states are ``(1, 4122, 4096)``).
#: Fixed rather than derived, because the probe is a reference point for the
#: cohort and not a model of it: a chip whose pointwise throughput differs at
#: another size is a question for a shape sweep, not for this number.
PROBE_SHAPE = (1, 4096, 4096)

#: Seconds the timer aims for per probe op. The ops are tens of microseconds, so
#: this buys thousands of samples; the whole probe is a couple of seconds and it
#: is not on the measured path.
PROBE_SECONDS = 0.25

#: ``(name, statement, passes)`` per probe op. ``passes`` is how many times the
#: tensor's bytes move: a unary op reads and writes (2), a binary op reads two
#: operands and writes one (3), an in-place copy reads and writes (2). The byte
#: count is the definition of the op's traffic, not a guess about the kernel.
PROBE_OPS = (
    ("copy_", "dst.copy_(src)", 2),
    ("mul", "src * other", 3),
    ("add", "src + other", 3),
    ("rsqrt", "torch.rsqrt(src)", 2),
    ("silu", "torch.nn.functional.silu(src)", 2),
)


class ShapeHook:
    """The two sequence lengths a run used, read off the model rather than assumed.

    A transformer block sees a joint sequence of text tokens followed by image
    tokens, and the split matters to the FLOPs: ``txt_in`` is two matmuls over the
    text tokens where ``img_in`` is one over the image tokens, and the attention
    term scales with their sum. The text length comes from the prompt and the
    tokeniser, so it cannot be derived from ``--height``/``--width``; this reads
    both lengths off the first forward instead of guessing either.

    Installed for the warmup calls only and removed before anything is timed, so
    no hook is on the measured path. The lengths do not depend on the batch, so
    one reading covers every batch of a sweep.
    """

    def __init__(self, transformer):
        self._transformer = transformer
        self.text_tokens = None
        self.target_tokens = None
        self._handle = None

    def __enter__(self):
        # with_kwargs, because diffusers calls the transformer with keyword
        # arguments: a plain pre-hook is handed the positional tuple only, which
        # is empty here, and would silently record nothing.
        self._handle = self._transformer.register_forward_pre_hook(
            self._capture, with_kwargs=True
        )
        return self

    def _capture(self, _module, args, kwargs=None):
        # Some diffusers paths pass these positionally and some by keyword; take
        # whichever holds them, and never raise -- a hook that breaks a benchmark
        # run in order to report a FLOPs figure has the priorities backwards.
        values = dict(kwargs or {})
        for name, value in zip(("hidden_states", "encoder_hidden_states"), args):
            values.setdefault(name, value)
        hidden = values.get("hidden_states")
        text = values.get("encoder_hidden_states")
        if hidden is None or text is None:
            return
        self.target_tokens = int(hidden.shape[1])
        self.text_tokens = int(text.shape[1])

    @property
    def joint_tokens(self):
        """Text plus image tokens: the sequence the blocks attend over.

        The sum, not the number of image slots: this flow exercises no condition
        image (``img_mask`` carries no True slot -- see the README, §4), so every
        position is one token and the joint length is exactly ``text + target``.
        Checkable against the run: ``infer.py`` prints the joint sequence width,
        which is 4122 for the canonical prompt against 26 + 4096 here.
        """
        if self.text_tokens is None or self.target_tokens is None:
            return None
        return self.text_tokens + self.target_tokens

    def __exit__(self, *_exception):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        return False


def transformer_flops(config, *, queries, keys, target_tokens, text_tokens, batch=1):
    """FLOPs one forward of the 2.1 transformer costs, from its config and shapes.

    ``queries`` is how many joint-sequence positions this call computes, and
    ``keys`` how many it attends over. The two are equal in a prefill -- every
    query recomputes the whole sequence -- and differ in a cached decode step,
    where only the target image's tokens are recomputed but the attention still
    reads the cached prefix; see :func:`image_flops`.

    ``target_tokens`` and ``text_tokens`` are the shapes of the two input
    projections, which are per-forward constants rather than ``queries``.

    The structure this walks is the one in
    ``diffusers/models/transformers/transformer_qwenimage21.py``: a single-stream
    MMDiT whose blocks are pre-norm -- scale-only modulation, attention, gated
    residual, SwiGLU, gated residual -- with no biases anywhere. One term per
    matmul, and one for the attention's two matmuls (scores and weighted sum).
    """
    heads = config["num_attention_heads"]
    head_dim = config["attention_head_dim"]
    dim = heads * head_dim
    # QwenImage21SwiGLUFeedForward is built with hidden_size=dim and
    # mlp_hidden_size=dim * mlp_ratio.
    mlp_hidden = dim * config["mlp_ratio"]
    layers = config["num_layers"]

    def macs(rows, inner, columns):
        """A matmul's multiply-accumulates, in FLOPs: 2 x rows x inner x columns."""
        return 2.0 * batch * rows * inner * columns

    # Blocks first: they are the three orders of magnitude, and keeping them in
    # one dict makes the per-layer total checkable against a parameter count.
    per_layer = {
        # to_q, to_k, to_v are all applied to this call's own hidden states, so a
        # cached decode step -- which recomputes only the target tokens -- scales
        # them with queries, not with keys.
        "attention projections": 3.0 * macs(queries, dim, dim),
        # Scores and the weighted sum: two matmuls over heads x queries x keys x
        # head_dim. The prefill's per-segment masks make the text segment's own
        # block causal, and its triangle is computed densely by the kernel, so it
        # is counted densely here; against the 4096-query target block that is a
        # fraction of a per cent and it is the one approximation in this model.
        "attention": 2.0 * macs(heads * queries, head_dim, keys),
        "attention output": macs(queries, dim, dim),
        # proj and gate into the SwiGLU's hidden width, then out of it.
        "feed-forward": 2.0 * macs(queries, dim, mlp_hidden)
        + macs(queries, mlp_hidden, dim),
    }
    terms = {name: value * layers for name, value in per_layer.items()}
    # img_in: Linear(in_channels -> dim) over the packed latents.
    terms["img_in"] = macs(target_tokens, config["in_channels"], dim)
    # txt_in: QwenImage21TextProjection, whose in_layer and out_layer are both
    # dim-wide here because context_in_dim happens to equal dim in this
    # checkpoint -- read from the config rather than assumed equal.
    terms["txt_in"] = macs(text_tokens, config["context_in_dim"], dim) + macs(
        text_tokens, dim, dim
    )
    # proj_out: Linear(dim -> out_channels) over every query.
    terms["proj_out"] = macs(queries, dim, config["out_channels"])

    return {
        "total": sum(terms.values()),
        "per_layer": sum(per_layer.values()),
        "terms": terms,
    }


def image_flops(config, *, text_tokens, target_tokens, steps, batch=1, kv_cache=True):
    """The transformer's FLOPs for one batch's worth of image, prefill and decode split.

    Two shapes of call, and with the prefix KV cache on -- which is the default,
    see the README §4.1 -- the pipeline makes one of the first and the rest of the
    second:

    - **prefill**, the first step: every query recomputes the joint sequence and
      attends over it, so ``queries == keys == text + target``;
    - **decode**, every step after it: only the target image's tokens are
      recomputed, and the attention reads the whole joint sequence out of the
      cache, so ``queries == target`` and ``keys == text + target``.

    With ``--no-kv-cache`` every step is a prefill. The two differ by the query
    count in every projection and in the attention's score matmul, so the split is
    carried rather than averaged: 1 prefill at 4122 queries plus 39 decodes at
    4096 is not 40 of either.

    Both totals are reported, and the difference between them is not cosmetic. A
    call's FLOPs are what the loop's wall time corresponds to, so they are what an
    MFU divides; a per-image figure is the one that compares across batches. At
    batch 4 the two differ by 4x, and a table row labelled "per image" showing a
    per-call number would read as a chip doing four times the arithmetic.
    """
    joint = text_tokens + target_tokens
    prefill = transformer_flops(
        config,
        queries=joint,
        keys=joint,
        target_tokens=target_tokens,
        text_tokens=text_tokens,
        batch=batch,
    )
    if not kv_cache:
        per_call = prefill["total"] * steps
        return {
            "prefill_calls": steps,
            "decode_calls": 0,
            "flops_prefill": prefill["total"],
            "flops_decode": None,
            "flops_per_call": per_call,
            "flops_per_image": per_call / batch,
            "terms": prefill["terms"],
        }

    decode = transformer_flops(
        config,
        queries=target_tokens,
        keys=joint,
        target_tokens=target_tokens,
        text_tokens=text_tokens,
        batch=batch,
    )
    per_call = prefill["total"] + decode["total"] * (steps - 1)
    return {
        "prefill_calls": 1,
        "decode_calls": steps - 1,
        "flops_prefill": prefill["total"],
        "flops_decode": decode["total"],
        "flops_per_call": per_call,
        "flops_per_image": per_call / batch,
        "terms": prefill["terms"],
    }


def achievement(flops, seconds, peak_tflops=None, batch=1):
    """A batch's modelled FLOPs and the wall time they happened in, as TFLOP/s and MFU.

    ``flops`` and ``seconds`` are one call's, not one image's. ``batch`` is carried
    only so the per-image time can be reported beside them; it deliberately does
    not enter the ratio, and the arithmetic says why:

        achieved = flops / seconds = (batch * flops_per_image) / (batch * t_image)
                 = flops_per_image / t_image

    The ratio is a per-image quantity already, which is why MFU is unchanged from
    batch 1 to batch 4 in a table whose FLOPs-per-image row is likewise unchanged.
    ``seconds_per_image`` is the same reduction of the only input that varies, so
    a reader can see that the loop's per-image time -- not the FLOPs -- is what
    moves a result, and can compare that time directly between two runs whose
    batches differ.

    ``seconds`` is also the phase those FLOPs belong to, not the whole call:
    dividing the loop's FLOPs by the whole call's time would produce a ratio that
    is low for the honest reason that two other components ran, which is not what
    a utilisation figure is supposed to say. The per-image time is scoped the same
    way: it is the denoise loop's, not the pipeline's.
    """
    tflops = None if not flops or not seconds else flops / seconds / 1e12
    return {
        "flops_per_call": flops,
        "seconds_per_call": seconds,
        "seconds_per_image": None if seconds is None else seconds / batch,
        "tflops": tflops,
        "peak_tflops": peak_tflops,
        "mfu": None if not tflops or not peak_tflops else tflops / peak_tflops,
    }


def bandwidth_probe(torch, device, device_kind, peak_gbs=None, shape=PROBE_SHAPE):
    """Stream a few elementwise ops at this workload's activation shape.

    Returns ``{"shape", "dtype", "tensor_mib", "peak_gbs", "ops": [...]}``, one
    entry per op carrying its GB/s moved and, when a peak was supplied, its MBU.

    An op whose call raises is recorded with the backend's message rather than
    taking the probe down: which elementwise ops a chip can run is a routing
    question this file has no business deciding, and a probe that fails on one op
    still reports the others.
    """
    import torch.utils.benchmark as benchmark

    src = torch.randn(*shape, dtype=torch.bfloat16, device=device)
    other = torch.randn(*shape, dtype=torch.bfloat16, device=device)
    dst = torch.empty_like(src)
    scope = {"src": src, "other": other, "dst": dst}
    tensor_bytes = src.numel() * src.element_size()

    ops = []
    for name, statement, passes in PROBE_OPS:
        try:
            common.sync(torch, device_kind)
            measurement = benchmark.Timer(
                stmt=statement, globals=scope, num_threads=1
            ).blocked_autorange(min_run_time=PROBE_SECONDS)
            common.sync(torch, device_kind)
        except Exception as error:  # noqa: BLE001 - the message is the result
            ops.append({"op": name, "error": f"{type(error).__name__}: {error}"})
            continue
        seconds = measurement.mean
        gbs = tensor_bytes * passes / seconds / 1e9
        ops.append(
            {
                "op": name,
                "passes": passes,
                "gb_per_s": gbs,
                "mbu": None if not peak_gbs else gbs / peak_gbs,
                "seconds": seconds,
            }
        )

    # Held only for the probe, and given back before anything else runs: 96 MiB of
    # activations is worth releasing on a card where the allocator is the binding
    # constraint (see the README, §1.2).
    del src, other, dst, scope
    return {
        "shape": list(shape),
        "dtype": "bfloat16",
        "tensor_mib": tensor_bytes / (1024**2),
        "peak_gbs": peak_gbs,
        "ops": ops,
    }


def probe_summary(bandwidth):
    """The probe's cohort as one reading, or ``None`` when it measured nothing.

    The **median** op, not the best one. Every op in the probe moves the same
    bytes at the same traffic, so a spread across the cohort is a difference in
    how each operator is launched and routed -- and a headline that took the best
    would hide exactly the slow operator that is worth knowing about, while also
    flipping between ``mul`` and ``add`` from run to run on noise that is inside
    the timer's own error. The range is reported beside the median for the same
    reason, and every op's own number stays in the JSON.
    """
    measured = [op for op in (bandwidth or {}).get("ops", []) if op.get("gb_per_s")]
    if not measured:
        return None
    ordered = sorted(measured, key=lambda op: op["gb_per_s"])
    middle = ordered[len(ordered) // 2]
    return {
        "op": middle["op"],
        "median_gb_per_s": middle["gb_per_s"],
        "mbu": middle.get("mbu"),
        "min_gb_per_s": ordered[0]["gb_per_s"],
        "max_gb_per_s": ordered[-1]["gb_per_s"],
        "measured_ops": len(ordered),
        "peak_gbs": (bandwidth or {}).get("peak_gbs"),
    }


def percent(value, places=1):
    """A fraction as a percentage string; ``n/a`` when it is not a fraction."""
    return "n/a" if value is None else f"{value * 100:.{places}f}%"


def probe_spread(summary):
    """``"2650-2900 GB/s over 5 ops"`` for the probe's cohort, or ``None``."""
    if summary is None:
        return None
    return (
        f"{summary['min_gb_per_s']:.0f}-{summary['max_gb_per_s']:.0f} GB/s "
        f"over {summary['measured_ops']} ops"
    )


def human_flops(flops):
    """A FLOPs count as a short string, for the run's own log lines."""
    if flops is None:
        return "n/a"
    for scale, suffix in ((1e15, "PFLOP"), (1e12, "TFLOP"), (1e9, "GFLOP")):
        if flops >= scale:
            return f"{flops / scale:.2f} {suffix}"
    return f"{flops:.3e} FLOP"
