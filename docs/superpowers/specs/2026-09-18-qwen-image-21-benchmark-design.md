# Qwen-Image-2.1 performance measurement — design document

Date: 2026-09-18
Status: Approved for implementation planning
Branch: `test/qwen-image-21-manual-flow` (PR flagos-ai/Torch-FL#342)

## 1. Goal

Give the Qwen-Image-2.1 manual test flow a fixed prompt and a performance
measurement whose numbers can be quoted.

PR #342 added the flow (`tests/manual/qwen_image_21/`) to bring a new chip up on
a diffusion model. It is good at what it was built for — localising a failure to
one stage — and it is not a benchmark. `prompts.py` says so itself: *"Nothing
here is a quality benchmark. These prompts exist to make a failure
attributable."* Nothing in the flow produces a number that a hardware vendor or
a release note can cite, and two things actively prevent it:

- there is no fixed prompt. `infer.py` carries a private string literal,
  `prompts.py` carries a table, and they are two independent sources for the same
  text.
- the timing that exists is single-shot with no warmup. `sweep.py` records one
  `time.time()` delta per prompt with a cold Triton cache on the first one, and
  reports their arithmetic mean. The README's own §4.3 measures that cold cache
  taking the same 40-step run from 23.8 s to 71.5 s, so that mean is dominated by
  a compile artifact and is not reproducible.

The deliverable is a benchmark entry point that measures the pipeline the way
the industry measures diffusion models, and the record of what its numbers mean
and do not mean.

## 2. Non-goals

- **Qwen-Image-2512.** The older flow gets no benchmark mode in this change.
  Reusing `bench.py` for it would mean moving the phase timer to a neutral
  location and re-deriving its placement assumptions; that is a separate change.
  `qwen_image_2512/README.md` states its own position — "Timing is recorded, not
  optimised. Nothing here is a performance claim" — and it stays as written.
- **Performance optimisation.** Nothing here makes the backend faster. The
  measurement is the deliverable, and a slow result is a valid result.
- **A submission-grade MLPerf harness.** MLPerf's text_to_image benchmark is
  5000 COCO captions, a 600 s minimum duration per scenario, p90 latency tiles
  and a CLIP-score accuracy gate. Reproducing that is a different project with a
  dataset dependency and a multi-hour runtime. This design borrows MLPerf's
  *metric definitions* (latency, throughput) and the diffusers harness's
  *measurement protocol*, and says so where it differs.
- **Multi-chip numbers.** Only CUDA-compatible hardware is measured here. The
  scripts stay backend-generic so a vendor can produce the other columns, and
  `--table` renders them together, but no other vendor's numbers are claimed.
- **CI.** `tests/manual/` is not executed by any CI workflow, and the model is
  33 GB. Unchanged.

## 3. What "the industry standard" means here, concretely

Two sources are used, and each is used for what it actually specifies.

**HF diffusers' own benchmark harness** —
`benchmarks/benchmarking_utils.py` in the diffusers tree (present locally at
`/nfs/lvyufeng/Qwen-Image-2.1-ref-qwen-image-2.1/benchmarks/`). It defines:

```python
NUM_WARMUP_ROUNDS = 5
time_s = benchmark.Timer(stmt="f(*args, **kwargs)", globals=..., num_threads=1).blocked_autorange().mean
mem_gb = torch.cuda.max_memory_allocated() / 1024**3
```

The protocol is adopted: warmup rounds that are discarded, a
`torch.utils.benchmark.Timer` that self-sizes its repeat count, `num_threads=1`,
and peak allocated memory as the memory figure. The model-level benchmark of
that harness (`benchmarking_sdxl.py`) benchmarks one model component at a time
(an `UNet2DConditionModel` forward), not a pipeline; this flow measures the
pipeline, which is what a chip bring-up has to report.

**MLPerf Inference, `text_to_image` (Stable Diffusion XL).** Its scenarios
supply the metric *definitions*: SingleStream is a latency measurement
(per-sample wall time), Offline is a throughput measurement (samples/s), and the
prompt cohort is a fixed caption set. This design reports the same two
quantities: `latency_s_per_image` (SingleStream analogue) and
`throughput_images_per_s` (Offline analogue, batch > 1). It does not reproduce
MLPerf's duration or percentile requirements.

**Where this design deliberately differs, and why:**

| Choice | diffusers harness | Here | Why |
| --- | --- | --- | --- |
| `min_run_time` | 0.2 s (default) | 60 s | diffusers times ~50 ms component forwards; this times a ~26 s pipeline, where 0.2 s yields one sample |
| warmup rounds | 5 | 2 | 5 rounds at 26 s is 2.2 minutes of pure warmup; 2 still covers Triton JIT and autotune convergence, and the value is recorded and overridable |
| sync around the timed region | relies on the `.cpu()` at the end | explicit `synchronize()` on both sides | for a forward with no CPU-visible output the harness measures launch time, not device time; the 2512 `memprobe.py` docstring records how CUDA runtime-API timing under-measures this workload |

## 4. The fixed prompt

`prompts.py` becomes the single source. One entry is designated canonical:

```python
CANONICAL_ID = "01"

def canonical():
    """The prompt every performance run uses: {id, prompt, sha256}."""

def sha256(text):
    """sha256 of the exact UTF-8 prompt bytes, lowercase hex."""
```

Prompt `01` is the only prompt the 2.1 authors published anywhere — the example
in `QwenImage21Pipeline`'s own docstring — and it is already the flow's de facto
default. It is 11 words, which is the same order as an MLPerf COCO caption
(10–15 words), so the text-encoder share of the latency is representative rather
than inflated or deflated.

Three changes make the fixing structural rather than a matter of discipline:

1. `infer.py` deletes its `DEFAULT_PROMPT` string literal and takes the default
   from `prompts.canonical()`. Two sources for one string is the drift this
   removes.
2. `bench.py` has **no `--prompt` flag at all**. Substituting a prompt is a
   capability of `infer.py` and `sweep.py`, which exist for other purposes; the
   benchmark entry point cannot be pointed at a different prompt, so two runs of
   it are always the same measurement.
3. `prompts.load()` attaches a `sha256` to every entry, and every artifact —
   `bench.py`'s JSON, `sweep.py`'s manifest — carries `prompt_id` and
   `prompt_sha256`. A number is therefore self-describing: it names the exact
   bytes that produced it.

`--prompts-file` on `sweep.py` is unaffected. Testing a different cohort is a
legitimate use of the sweep; it is not a benchmark.

## 5. Metric definitions

All are computed over the measured (post-warmup) samples. `latency` and each
entry of `phases` carry the same set of statistics — `mean`, `median`, `min`,
`max`, `std` — because the spread is what says whether the mean is worth
quoting, and a single-shot number cannot say that.

`latency` is the statistic of one whole `pipe(...)` call at the run's batch; the
per-image figure is that divided by the batch size. Every sample therefore keeps
the batch it was taken at in `config.batch`, and a latency taken at batch 4 is
never compared against one taken at batch 1 without that division.

| Metric | Definition | Industry analogue |
| --- | --- | --- |
| `latency_s_per_image` | Wall time of one complete `pipe(...)` call at the run's `--batch`, divided by the batch size: text encoder + denoise loop + VAE decode + postprocess to PIL. Excludes model load, placement and PNG write. At the default batch of 1 this is the whole call. | MLPerf SingleStream latency; diffusers `time_plain_s` || `phases["text encoder"]` | The text encoder forward, timed by a forward hook. One per call. | component breakdown |
| `phases["denoise loop"]` | First-to-last `callback_on_step_end` timestamp. | — |
| `phases["loop per step"]` | `denoise loop` ÷ the number of steps in it, the markup-free per-step cost. Steps-normalised so two chips at different step counts still compare. | diffusers s/iteration |
| `phases["vae decode"]` | The wrapped `pipe.vae.decode`. | — |
| `throughput_images_per_s` | `batch / latency_batch` when `--batch N` (N > 1) runs. `null` with the original error text when it does not. | MLPerf Offline samples/s |
| `memory.peak_gib` | The peak allocated watermark, reset before each measured call and read after it. Falls back to `memory_reserved` on a backend without the peak API, and names which it used. | diffusers `mem_plain_GB` |
| `determinism.identical` | Whether the first and last measured calls produced byte-identical PNGs. Every image of a call is hashed from the PIL object in memory, and `image_sha256` lists the last call's hashes in batch order. | prerequisite for the paired comparison |
| `vs baseline` (in `--table`) | Latency ratio against a named baseline column; throughput ratio as its reciprocal. | vendor speedup ratio |

`throughput` is reported only when it is real. At batch 1 the flag is not
specified and no synthetic QPS is derived from the latency — a single-card
number relabelled as a rate measures nothing the latency did not already say. On
a 40 GB card the 33 GB of weights mean batch > 1 will most likely not fit; the
row then records `null` and the allocator's error verbatim, which is itself a
useful result.

## 6. Measurement protocol

```text
outside the timed region   from_pretrained, split/placement, latents
                           construction, the PNG write
inside the timed region    one full pipe(...) call
```

- **Warmup**: `--warmup 2` discarded calls, default 2, actual count recorded in
  the JSON. This is the only Triton JIT and autotune coverage a benchmark run
  gets, and on a cold cache the README's measurement says the first call can be
  3x the warm one — so it is not optional.
- **Timing**: `torch.utils.benchmark.Timer(stmt=..., globals=..., num_threads=1)`
  with `.blocked_autorange(min_run_time=args.min_run_time)`, default
  `--min-run-time 60`. The timer self-sizes the repeat count; in practice that is
  2–3 calls at 26 s each. The timer's mean/median/min/max and its `number` field
  are all recorded.
- **Synchronisation**: the timed closure calls `synchronize()` on the backend's
  own device module on entry and exit. `torch.utils.benchmark.Timer` synchronises
  through `torch.accelerator.synchronize()`, which the torch-fl accelerator
  surface participates in, but the explicit pair makes the device attribution a
  property of this file rather than of a dependency's internals, and it is what
  the existing phase timer already does.
- **Excluded from the number**: weight loading and placement. They are a
  property of the disk and of `from_pretrained`, not of the chip.
- **Dispatch logging off by default.** `run.sh` exports `FLAGOS_LOG_DISPATCH=1`,
  and the README's §7 records that it writes one unbuffered line per operator
  call — 12% of the denoise loop on the 2512 workload. `run.sh bench` therefore
  defaults it to `0`; the census is still produced by `infer` and `sweep`, which
  keep the logging on because their job is attribution, not timing. Whatever the
  run actually used is recorded in the JSON, so a number taken with logging on is
  identifiable rather than silently inflated.
- **Cold cache is an explicit flag.** `bench.py` records `TRITON_CACHE_DIR` but
  never clears it. `--cold-cache` points it at a fresh directory, which is how
  "first run on a new box" is measured. Any cross-chip comparison requires both
  sides warm, and the recorded cache state is what makes that checkable.

## 7. Components

Each unit has one job and a narrow interface.

### `common.PhaseTimer`

The phase timer currently lives inside `infer.py` as a closure that accumulates
into one dict across a whole run. A benchmark needs one independent sample per
measured call, so it moves to `common.py` as a small class and `infer.py`
consumes it:

```python
class PhaseTimer:
    def __init__(self, torch, pipe, device_kind)  # installs the text-encoder
                                                  # hook, the transformer hook
                                                  # that brackets the loop, and
                                                  # wraps pipe.vae.decode
    def reset(self)                               # begin a new sample
    def sample(self) -> dict                      # this sample's four phases
    def on_step_end(self, pipe, index, ts, kwargs)  # for callback_on_step_end
    def close(self)                               # remove hooks, restore decode
```

The loop's two edges are the part the original implementation got wrong, and
both are now explicit. `callback_on_step_end` fires *after* each step, so N
callbacks bound only N−1 intervals and the loop total omitted the first step —
2.5% of it at 40 steps, and enough that the phase rows visibly failed to add up
to the end-to-end latency. A pre-forward hook on the transformer records where
the loop begins, so the span covers all N steps and `loop per step` divides by N.
The callback also synchronises before stamping: nothing in the pipeline blocks
the host between steps, so an unsynchronised stamp is a launch time rather than a
completion time. Measured on A100 by running the same loop twice in one process,
the two agree to 0.15% (20.88 s against 20.91 s) — but only because this card's
allocator is nearly full and blocks the host on every step. That is a property of
the chip's memory pressure, not of the measurement, which is why the synchronise
is what makes the number device-attributed rather than incidentally so.

`infer.py` calls `reset()` once and `sample()` once, so its printed phase table
has the same shape as before.

It also gains a small `sync()` helper, because three call sites now need the
same "synchronise this backend or fail with a message naming it" behaviour.

### `bench.py`

The only thing in the flow that produces a quotable number, and the only caller
of `PhaseTimer`'s per-sample path. Responsibilities: parse the CLI, warm up,
time, sample phases and memory around each call, and write one JSON. It does not
render, compare or aggregate — that is `--table`.

```bash
run.sh bench --device flagos                                 # -> <OUT_DIR>/flagos/bench.json
run.sh bench --device cuda --batch 4                         # the throughput column
run.sh bench --device flagos --load-latents latents.pt       # paired run, same noise
run.sh bench --device flagos --cold-cache                    # fresh TRITON_CACHE_DIR
run.sh table a.json b.json --baseline a
```

Flags: `--out` (required), `--batch 1`, `--steps 40`, `--height/--width 1024`,
`--seed 42`, `--warmup 2`, `--min-run-time 60`, `--true-cfg-scale 1.0`,
`--negative-prompt`, `--use-kv-cache/--no-kv-cache`, `--load-latents`,
`--cold-cache`, `--image <path>` (writes the last measured PNG, for the
determinism check), plus the placement flags from `common`. No `--prompt`.

`--batch` is passed as `num_images_per_prompt`, so all N images come from one
canonical prompt and two images are the same content at two seeds — the prompt
invariant holds under batching. Each call's returned PIL image is hashed in
memory for the determinism check; `--image` additionally writes the last one to
disk, which is what a human looks at.

### Schema

One JSON, versioned, per run. It is the entire interface between `bench.py` and
everything that reads a result.

```json
{
  "schema": 1,
  "kind": "qwen-image-21-bench",
  "revision": "<git rev-parse HEAD>",
  "prompt": {"id": "01", "sha256": "...", "text": "..."},
  "config": {
    "height": 1024, "width": 1024, "steps": 40, "batch": 1,
    "dtype": "bfloat16", "seed": 42, "true_cfg_scale": 1.0,
    "negative_prompt": " ", "use_kv_cache": true,
    "placement": {"encoder": "...", "transformer": ["..."], "vae": "..."}
  },
  "protocol": {
    "warmup_rounds": 2, "min_run_time_s": 60, "runs": 3, "num_threads": 1,
    "timer": "torch.utils.benchmark.Timer.blocked_autorange",
    "dispatch_log": false, "fallback_log": true,
    "triton_cache_dir": "...", "cold_cache": false
  },
  "latency":  {"per_call_s": {"mean_s": 0, "median_s": 0, "min_s": 0, "max_s": 0, "std_s": 0},
               "per_image_s": {"mean_s": 0, "median_s": 0, "min_s": 0, "max_s": 0, "std_s": 0}},
  "phases": {
    "text encoder":  {"mean_s": 0, "median_s": 0, "min_s": 0, "max_s": 0, "std_s": 0},
    "denoise loop":  {"mean_s": 0, "median_s": 0, "min_s": 0, "max_s": 0, "std_s": 0},
    "loop per step": {"mean_ms": 0, "median_ms": 0, "min_ms": 0, "max_ms": 0, "std_ms": 0},
    "vae decode":    {"mean_s": 0, "median_s": 0, "min_s": 0, "max_s": 0, "std_s": 0}
  },
  "throughput": {"images_per_s": null, "error": null},
  "memory": {"peak_gib": 0, "source": "max_memory_allocated"},
  "determinism": {"identical": true, "image_sha256": ["..."]},
  "versions": {"torch": "...", "torch_fl": "...", "diffusers": "...", "triton": "..."},
  "notes": []
}
```

`notes` is where a run records what it could not do — a batch that OOM'd, a
vendor surface without a peak counter, a cold cache it was told to use.

### `--table`

Reads N JSON files and renders one markdown table: metric rows against one
column per file, with `--baseline NAME` appending a `vs baseline` column
(latency ratio = row ÷ baseline, throughput ratio = its reciprocal). It needs no
torch and no device, so it runs on any interpreter and the JSONs travel between
machines.

There is no default baseline: a ratio without a named counterpart is not a
measurement. Two conditions are called out below the table rather than folded
into a cell — a row whose `determinism.identical` is false, and a row whose
`throughput.error` is non-null. The reference table in the README's §7.1 is
generated by this command rather than hand-copied.

## 8. Edits to existing files

| File | Change |
| --- | --- |
| `prompts.py` | `CANONICAL_ID`, `canonical()`, `sha256()`; `load()` entries carry `sha256` |
| `common.py` | `PhaseTimer` class and `sync()` helper, moved from `infer.py` |
| `infer.py` | `DEFAULT_PROMPT` literal deleted, default taken from `prompts.canonical()`; local `phase_timers` deleted in favour of `common.PhaseTimer`; phase-table output unchanged |
| `bench.py` | new |
| `sweep.py` | manifest carries `prompt_id` / `prompt_sha256` per image and a top-level `prompts_sha256`; the trailing mean is replaced by a total, since the mean of eight heterogeneous prompts is not a metric |
| `run.sh` | `bench` and `table` modes; the `FLAGOS_LOG_DISPATCH` export moves below mode parsing so `bench` defaults it to 0; `bench` defaults `--out` to `$DEVICE_DIR/bench.json`; `summarise` prints the JSON's metrics for `bench` instead of counting dispatch lines |
| `README.md` | new "Performance measurement" section (prompt invariant, metric definitions, protocol, what is inside the timed region, schema, `--table`, and the four traps: dispatch logging, cold cache, the KV cache changing the image, single-card batch); §5 states that the sweep's `seconds` is not a benchmark number; §7.1's reference table regenerated by `run.sh table` |

No change to `numerics.py`, `compare.py`, `side_by_side.py`, or anything under
`qwen_image_2512/`.

## 9. Verification

`ruff` is pinned to the version CI pins (0.15.12) and both `ruff check` and
`ruff format --check` are run before the PR is updated.

1. **No regression in attribution.** `infer.py --stage full`'s phase table is
   diffed against the pre-change output; it must be identical. The refactor moves
   code, it does not change what is measured.
2. **Two backends, one table.** On the A100: `run.sh bench --device cuda` and
   `run.sh bench --device flagos`, both with a warm Triton cache, then
   `run.sh table` over the two JSONs. Both JSONs and the rendered table go into
   the PR.
3. **Determinism.** `bench.py --image` twice at the same seed, and the
   `determinism.identical` field within a single run, must both be true.
4. **Throughput failure path.** `--batch 4` on a 40 GB card is expected to OOM.
   That exercises the `null` + recorded-error path, which is the point. A real
   throughput figure needs a 64 GB card and, if one is unavailable, the PR says
   the column was not obtained rather than estimating it.
5. **Cold cache.** `--cold-cache` is run once to confirm the flag reaches
   Triton and that the recorded cache state differs from the warm run's.
6. **§7.1 regenerated** from the new measurement.

## 10. Review round: placement and attribution

A review of the implementation found four defects, all of which are fixed here
and all of which are recorded because they share a shape: **something the flow
reported that was not what the flow did.**

1. **`resolve_placement` placed the encoder off the transformer's card when
   `--devices` named several.** The default had been rewritten to "everything on
   one card, the first of `--devices`", but the encoder still took `chosen[-1]`
   while the transformer took `chosen[0]`, so `--devices flagos:0 flagos:1` put
   the transformer on card 0 and the encoder and VAE on card 1. The module
   docstring and the two flag help texts also disagreed with each other about
   what the transformer's default was. The rule is now one statement in one
   place: the transformer gets every card of `--devices` (so naming several is
   the request for a split, which is the only reason to name several), and the
   encoder and VAE get the first of them.

2. **`PhaseTimer` reported a loop one step short and attributed the rest to the
   host.** Covered in §7; the loop is now bracketed from the transformer forward
   that begins it, and the step callback synchronises before stamping.

3. **`bench.py` and `sweep.py` recorded a VAE placement they had not applied.**
   Both moved the VAE to the encoder's card and then wrote the *requested*
   `--vae-device` into the record; `sweep.py` also discarded the value and
   accepted the flag without honouring it. Placement is now applied in one place,
   `common.place_components`, which returns what it did and refuses a full run
   whose `--vae-device` names a card the pipeline cannot decode on. `sweep.py`'s
   manifest gains the placement it never had, and its per-prompt memory reading
   covers every card the placement uses rather than the transformer's first.

4. **`numerics.py` could report a broken view as agreement.** The aliasing
   comparison only looked at the `aliases` flag when both sides had one, so a
   candidate whose view op raised produced an `error` entry, matched nothing and
   exited 0 printing `all agree`. A missing or errored candidate entry is now a
   failure; a reference-side error is still skipped. Separately, the `where` and
   `masked_fill` inputs were a `randn` cast to `bool`, which is 100% True, so
   neither op's false branch was ever reached — the mask is now genuinely
   both-valued.

## 11. Open items

None. The two decisions left open during design were settled: `--warmup`
defaults to 2, and Qwen-Image-2512 is not covered by this change.

## 12. Addendum, 2026-09-20: the curve, the percentiles, the ratios

Added after the first measured pair on a chip this design had not seen. What
prompted it: the H100 pair reported 1.68x end-to-end and nothing in the record
could say *why* — which side was bound, or whether either was. Everything below is
additive; every invocation that worked before produces the same record it did.

**`--batch` takes a list.** One value behaves exactly as before. Several measure a
throughput curve, one record per value (`bench-b1.json`, `bench-b2.json`, ...),
which `--table` renders as one column each. The reason is §5's own premise: a
throughput at one batch is a point, and the batch at which `latency_s_per_image`
stops falling is the difference between a card doing arithmetic and a card waiting
on the host. §7.1.1 of the README; measured in §7.3, where cuda is flat from batch
1 and flagos falls 20% by batch 2.

**`latency.per_image_s` gains `p90_s` and `p99_s`.** This revises the non-goal in
§2 that said MLPerf's percentile requirements are not reproduced: the percentiles
are now reported, linearly interpolated, with the sample count beside them and a
table caveat whenever the count is too low for p99 to be anything but the largest
call measured. The rest of that non-goal stands unchanged — no CLIP-score gate, no
5000-caption cohort, no 600 s minimum duration.

**A `compute` section, and `cost.py` to derive it.** `--peak-tflops` and
`--peak-bandwidth-gbs` are new flags, with no defaults (a ratio against a guessed
peak reads like a result). With them, the record carries MFU for the denoise loop
against an analytic FLOPs model of the transformer, and MBU for a five-op
elementwise probe at the workload's activation shape. The model's inputs are the
checkpoint's `config.json` and the two sequence lengths read off a hook during
warmup — nothing is instrumented on a measured call — and its total was checked
against a hook-based count of a real call to 0.02%. The text encoder and the VAE
are measured but not modelled, so every ratio is scoped to the loop.

The section carries both per-call and per-image forms of the two inputs to that
ratio, because they answer different questions. `flops_per_image` is constant
across a batch sweep by construction — every matmul is `2 x batch x rows x inner x
columns` with nothing coupling one sample to another — so its constancy is the
model's own consistency check, and it is the figure that compares across step
counts and resolutions. `loop.seconds_per_image` is not constant, and it is the
only input that varies: `MFU = flops_per_image / (seconds_per_image x peak)`, so
within one table the ratio is the per-image loop time in different units. Both are
reported so a reader can see that rather than discover it, and the loop time
doubles as the discriminating reading in the curve.

**What is still not covered**, and is now the honest open list:

- Goodput and the Server scenario: no concurrency, no arrival process, no SLO.
- Energy per image (MLPerf Power). NVML would supply it; nothing here reads it.
- A CLIP-score accuracy gate, so a fast-but-wrong result still cannot be detected
  by this flow. §6.1's paired PSNR remains the only quality check, and it measures
  backend agreement rather than image quality.
- Elementwise FLOPs, which is why MBU is a probe rather than a derivation.
- Qwen-Image-2512, unchanged from §2.

