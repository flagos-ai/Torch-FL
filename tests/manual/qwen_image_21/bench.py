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

"""Qwen-Image-2.1, measured: one JSON of numbers a chip's result can be quoted from.

``infer.py`` and ``sweep.py`` exist to say *whether* a chip runs the pipeline and
*where* it fails. This file exists to say how fast it is, and it is the only
thing in this directory that produces a number worth quoting.

What makes the number quotable is not the arithmetic, it is the protocol:

**The prompt is fixed.** There is no ``--prompt`` flag. The prompt comes from
``prompts.canonical()`` -- the one the 2.1 authors published -- so two runs of
this file are always the same measurement rather than two measurements of two
things. Use ``infer.py`` to reach a surface the canonical prompt does not.

**The measurement is repeated and the spread is reported.** ``infer.py`` reports
one sample, which is the right shape for localising a failure and the wrong shape
for a benchmark: one number cannot say whether a second would have agreed.
Latency and every phase carry mean, median, min, max and std over the measured
calls, and the number of calls is recorded next to them.

**Slow work is warmed up first.** The FlagGems path JIT-compiles Triton kernels;
the README's §4.3 measures a cold cache taking the same 40-step run from 23.8 s
to 71.5 s. ``--warmup`` calls are run and discarded before any measurement, and
the count used is recorded. ``--cold-cache`` is the explicit way to measure the
other thing.

**Dispatch logging is off by default.** ``run.sh`` puts ``dispatch`` in
``FLAGOS_LOG`` because ``infer`` and ``sweep`` are censuses, and the README
records that it costs 12% of the denoise loop. ``run.sh bench`` leaves it out,
and what the environment actually held is written into the JSON, so a number
taken with it on is identifiable rather than silently inflated.

Where the protocol comes from, and where it departs from its sources:

  - warmup-then-discard, ``torch.utils.benchmark.Timer`` with ``num_threads=1``,
    and peak allocated memory as the memory figure are the diffusers harness'
    (``benchmarks/benchmarking_utils.py``), which is the reference measurement
    for this model family.
  - ``latency_s_per_image`` and ``throughput_images_per_s`` are MLPerf
    Inference's ``text_to_image`` metrics -- its SingleStream and Offline
    scenarios respectively.
  - Departures, both deliberate: ``min_run_time`` defaults to 60 s rather than
    diffusers' 0.2 s, because 0.2 s yields a single sample of a 26 s pipeline;
    and warmup defaults to 2 rather than 5, because five rounds of a 26 s
    pipeline is 2.2 minutes spent before any measurement at all. Both are flags,
    and the value used is recorded.

Run:
    python tests/manual/qwen_image_21/bench.py --device cuda   --out cuda.json
    python tests/manual/qwen_image_21/bench.py --device flagos --out flagos.json
    python tests/manual/qwen_image_21/bench.py --device flagos --batch 4 --out b4.json

Then, on any interpreter, with no device and no torch:
    python tests/manual/qwen_image_21/bench.py --table cuda.json flagos.json \\
        --baseline cuda

Protocol, metric definitions and the readings to record:
    tests/manual/qwen_image_21/README.md, "Performance measurement".
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import common
import prompts

#: Phases whose JSON value is in seconds but whose table row reads better in
#: milliseconds, because it is three orders of magnitude smaller than the others.
MILLISECOND_PHASES = ("loop per step",)


class Unrunnable(Exception):
    """A call the backend could not execute at all.

    Reported rather than raised through, because on a 40 GB card the 33 GB of
    weights leave no room for a larger batch and the allocator's own message is
    the useful part of that result. The caller writes a record carrying it
    instead of a number.
    """


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Measure Qwen-Image-2.1 on one backend into a JSON that --table "
            "renders. The prompt is fixed; see the module docstring."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    common.add_placement_args(parser)
    parser.add_argument(
        "--table",
        nargs="+",
        default=None,
        metavar="JSON",
        help=(
            "render these bench JSONs as one markdown table and exit. Needs no "
            "torch and no device, so it runs anywhere the files can be read"
        ),
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help=(
            "with --table: the label to measure ratios against. The label is the "
            "JSON's file stem. There is deliberately no default -- a ratio "
            "without a named counterpart is not a measurement"
        ),
    )
    parser.add_argument(
        "--model",
        default=common.model_ref(),
        help=f"a directory or a hub id; default: ${common.MODEL_ENV} or its fallback",
    )
    parser.add_argument(
        "--out", default=None, help="where the JSON lands; required to measure"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help=(
            "images per call, passed as num_images_per_prompt -- all of them from "
            "the fixed prompt, so the batch is the only thing that varies. Above 1 "
            "a throughput is reported; if the batch does not fit, the JSON records "
            "the allocator's message instead of a number"
        ),
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help=(
            "discarded calls before measurement; at least 1 is always run, because "
            "a cold Triton cache makes the first call unrepresentative"
        ),
    )
    parser.add_argument(
        "--min-run-time",
        type=float,
        default=60.0,
        help="seconds the timer aims for; it sizes the number of calls from this",
    )
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--true-cfg-scale", type=float, default=1.0)
    parser.add_argument("--negative-prompt", default=prompts.NEGATIVE_PROMPT)
    parser.add_argument(
        "--no-kv-cache",
        action="store_true",
        help=(
            "disable the prefix KV cache. The pipeline's help is explicit that "
            "toggling it gives an equally valid but visibly different sample, so "
            "fix it per comparison rather than mixing"
        ),
    )
    parser.add_argument(
        "--load-latents",
        default=None,
        help=(
            "inject initial latents from this .pt. This is the only injection 2.1 "
            "supports, and it is what makes a paired comparison across two "
            "backends meaningful -- they share no RNG stream"
        ),
    )
    parser.add_argument(
        "--cold-cache",
        action="store_true",
        help=(
            "point TRITON_CACHE_DIR at a fresh directory, to measure what a box "
            "that has never run this pays. Never done implicitly"
        ),
    )
    parser.add_argument(
        "--image", default=None, help="write the last measured call's PNG here"
    )
    args = parser.parse_args(argv)
    if args.table is None and args.out is None:
        parser.error("--out is required to measure (or use --table)")
    if args.batch < 1:
        parser.error("--batch must be at least 1")
    if not args.table and args.warmup < 0:
        parser.error("--warmup cannot be negative")
    return args


def revision():
    """The commit this run measured, or ``None`` outside a checkout."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def cold_cache_dir():
    """A fresh Triton cache directory, named so two concurrent runs do not share."""
    import os
    import tempfile
    import time

    path = Path(tempfile.gettempdir()) / f"triton-cold-{os.getpid()}-{int(time.time())}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def statistics(samples):
    """``{mean, median, min, max, std, runs}`` over the measured calls.

    Every metric gets this shape rather than a bare number, because the spread is
    what says whether the mean is worth quoting. A single-sample run -- which
    ``--min-run-time`` can still produce on a slow chip -- gets a std of 0.0, and
    the ``runs`` field is what exposes it: a spread of one sample is not a spread.
    """
    values = list(samples)
    count = len(values)
    mean = sum(values) / count
    ordered = sorted(values)
    middle = count // 2
    median = (
        ordered[middle] if count % 2 else (ordered[middle - 1] + ordered[middle]) / 2.0
    )
    variance = sum((value - mean) ** 2 for value in values) / count
    return {
        "mean_s": mean,
        "median_s": median,
        "min_s": ordered[0],
        "max_s": ordered[-1],
        "std_s": variance**0.5,
        "runs": count,
    }


def phase_statistics(samples):
    """Per-phase statistics over the measured calls, in seconds.

    A phase absent from every sample is absent from the result rather than zero:
    a run that never reached the VAE decode must not report a 0.000 s decode,
    which reads as an impossibly fast one.

    Seconds for every phase, including the one the table renders in
    milliseconds: the unit a measurement is stored in should not depend on how
    it is displayed.
    """
    phases = {}
    for name in common.PhaseTimer.KEYS:
        values = [sample[name] for sample in samples if name in sample]
        if not values:
            continue
        phases[name] = statistics(values)
    return phases


def image_hash(image):
    """sha256 of a PIL image's bytes, for the determinism check.

    Hashed from the in-memory image rather than from a file on disk, so the check
    does not depend on ``--image`` being given or on the PNG encoder being
    deterministic across writes.
    """
    return hashlib.sha256(image.tobytes()).hexdigest()


def versions(torch, device_kind):
    """The versions a result is only comparable within.

    ``torch_fl`` is read out of ``sys.modules`` rather than imported. Importing
    it into a process running the vendor's own torch is not a harmless lookup:
    ``torch_fl``'s import preloads the CUDA assets bundled for the chip under
    test, and loading them into a process that has already initialised a
    different CUDA runtime aborts the process with a duplicate-allocator-config
    error. ``common.import_torch`` has already imported it whenever the run is on
    ``flagos``, so this reports it exactly when it applies and stays out of the
    way when it does not.
    """
    import sys

    result = {"torch": torch.__version__}
    for name in ("diffusers", "triton"):
        try:
            result[name] = getattr(__import__(name), "__version__", "unknown")
        except ImportError:
            result[name] = None
    module = sys.modules.get("torch_fl")
    result["torch_fl"] = getattr(module, "__version__", None) if module else None
    return result


def environment():
    """The environment settings that change what a timing means.

    Read from the environment rather than assumed, because the number is
    comparable only within them: the dispatch diagnostic writes a line per
    operator call and costs real time per call, so a run that had it on is a
    different measurement from one that did not. ``--cold-cache`` has already set
    ``TRITON_CACHE_DIR`` by the time this is read, so the record shows the
    directory that was used.

    ``FLAGOS_LOG`` is one comma-separated list, and ``dispatch_log`` is the
    parsed answer to the question the table actually asks. ``log`` keeps the raw
    string so the record is readable without knowing the list's spelling.
    """
    import os

    raw = os.environ.get("FLAGOS_LOG") or ""
    items = [item.strip() for item in raw.split(",") if item.strip()]
    return {
        "log": raw or None,
        "log_items": items,
        "dispatch_log": "dispatch" in items,
        "triton_cache_dir": os.environ.get("TRITON_CACHE_DIR") or None,
        "torch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF") or None,
    }


class Caller:
    """One ``pipe(...)`` call, and the bookkeeping a benchmark needs around it.

    Three things have to line up per call -- the phase sample, the peak memory and
    the image hashes -- and they are recorded in one place so an off-by-one in one
    of them is not possible. ``record`` is called once per call, from the same
    frame that made it.
    """

    def __init__(self, torch, pipe, args, prompt, latents, device, timer):
        self.torch = torch
        self.pipe = pipe
        self.args = args
        self.prompt = prompt
        self.latents = latents
        self.device = device
        self.timer = timer
        self.device_kind = args.device
        self.samples = []
        self.peaks = []
        self.hashes = []
        self.calls = 0
        self.images = None

    def __call__(self):
        """Make the call and record it. Raises ``Unrunnable`` if it cannot run."""
        self.timer.reset()
        # Reset before, read after: without this the reading is the run's
        # high-water mark, which the warmup or the model load may have set, and
        # it would say nothing about the calls being measured.
        common.reset_peak_memory(self.torch, self.device_kind, self.device)
        latents = None if self.latents is None else self.latents.clone()
        generator = self.torch.Generator(device=self.device).manual_seed(self.args.seed)
        with self.torch.no_grad():
            try:
                result = self.pipe(
                    prompt=self.prompt,
                    negative_prompt=(
                        self.args.negative_prompt
                        if self.args.true_cfg_scale > 1.0
                        else None
                    ),
                    true_cfg_scale=self.args.true_cfg_scale,
                    num_images_per_prompt=self.args.batch,
                    num_inference_steps=self.args.steps,
                    height=self.args.height,
                    width=self.args.width,
                    latents=latents,
                    generator=generator,
                    output_type="pil",
                    use_kv_cache=not self.args.no_kv_cache,
                    callback_on_step_end=self.timer.on_step_end,
                )
            except Exception as error:  # noqa: BLE001 - re-raised as Unrunnable
                raise Unrunnable(f"{type(error).__name__}: {error}") from error

        self.calls += 1
        self.samples.append(self.timer.sample())
        self.peaks.append(common.peak_memory(self.torch, self.device_kind, self.device))
        self.hashes.append([image_hash(image) for image in result.images])
        self.images = result.images

    def measured(self, count):
        """The last ``count`` calls, which are the ones the timer timed.

        ``blocked_autorange`` runs the statement while estimating the block size,
        before it starts the loop it measures, and its reported count covers only
        that loop. The estimate's calls are therefore always at the front, and
        keeping the last ``count`` records is what lines the phases, the memory
        readings and the hashes up with the reported times. Indexing by the count
        rather than by "a few extra calls" keeps this correct whatever the
        estimate does.
        """
        return (
            self.samples[-count:],
            self.peaks[-count:],
            self.hashes[-count:],
        )


def measured_calls(measurement):
    """How many times the timed statement ran.

    ``Measurement.number_per_run`` is the block size and ``raw_times`` holds one
    entry per block, so the count is their product -- not ``len(raw_times)``,
    which is the number of *blocks* and is 1 whenever a single call already
    exceeds the duration the block-size estimate aims for, which is the case for
    any real diffusion pipeline.
    """
    return measurement.number_per_run * len(measurement.raw_times)


def measure(torch, pipe, args, prompt, device):
    """Warm up, then measure. Returns the record's measurement sections.

    Latents are rebuilt per call from the template and a fresh generator is made
    each time: a call that consumed them would make every repetition a
    differently-conditioned run, and the spread would be measuring that instead
    of the chip.
    """
    import torch.utils.benchmark as benchmark

    notes = []
    discarded = max(args.warmup, 1)
    if discarded != args.warmup:
        notes.append(
            "--warmup 0 raised to 1: a benchmark with no discarded call measures "
            "the first call's lazy initialisation and compile"
        )
    if args.load_latents is None:
        notes.append(
            f"latents sampled from seed {args.seed}, with a fresh generator per "
            "call, so this backend's repetitions are identical. Two backends do "
            "not share an RNG stream, so pass --load-latents to put both on the "
            "same noise before comparing them."
        )

    timer = common.PhaseTimer(torch, pipe, args.device)
    template = None
    if args.load_latents is not None:
        template = torch.load(args.load_latents, map_location="cpu")
        print(f"  loaded latents <- {args.load_latents} {tuple(template.shape)}")

    caller = Caller(torch, pipe, args, prompt, template, device, timer)

    print(f"warming up: {discarded} call(s), discarded")
    for index in range(discarded):
        try:
            caller()
        except Unrunnable as error:
            timer.close()
            raise Unrunnable(str(error)) from error
        print(f"  warmup {index + 1}/{discarded} done")

    def timed_call():
        common.sync(torch, args.device)
        try:
            caller()
        finally:
            common.sync(torch, args.device)

    print(f"measuring: {args.min_run_time:g}s minimum, at least two calls")
    measurement = benchmark.Timer(
        stmt="timed_call()", globals={"timed_call": timed_call}, num_threads=1
    ).blocked_autorange(min_run_time=args.min_run_time)

    timer.close()

    runs = measured_calls(measurement)
    samples, peaks, hashes = caller.measured(runs)
    # One source for the latency: the timer's own per-call times. Its mean,
    # median, min and max are computed from exactly these, so computing them here
    # from the same list keeps one arithmetic in play rather than two. Note that
    # ``times`` has one entry per *block* of calls, not per call; see the note
    # below when the block size is not 1.
    per_call = statistics(measurement.times)
    latency = {
        "per_call_s": per_call,
        "per_image_s": {
            key: (value / args.batch if key != "runs" else value)
            for key, value in per_call.items()
        },
    }
    if measurement.number_per_run != 1:
        notes.append(
            f"the timer measured in blocks of {measurement.number_per_run} calls, "
            f"so {runs} calls gave {len(measurement.times)} latency samples: the "
            "latency spread is over blocks, while the phases are per call. Quote "
            "`latency.per_image_s.runs` as the latency sample count."
        )

    memory = memory_summary(peaks)
    determinism = {
        "identical": bool(hashes) and hashes[0] == hashes[-1],
        "image_sha256": hashes[-1] if hashes else None,
    }
    if len(hashes) != len(samples):
        notes.append(
            f"{len(samples)} phase samples against {len(hashes)} image hashes; "
            "the determinism check is over the images it did record"
        )
    return {
        "latency": latency,
        "phases": phase_statistics(samples),
        "memory": memory,
        "determinism": determinism,
        "images": caller.images,
        "notes": notes,
        "runs": runs,
    }


def memory_summary(peaks):
    """Peak memory over the calls, and which counter it came from.

    The source is recorded because the two counters are not the same
    measurement: a peak allocated and a reserved total can differ by the
    allocator's whole pool, and a table that silently mixes them is worse than
    one that says which it used.
    """
    values = [value for value, _ in peaks if value is not None]
    sources = {source for _, source in peaks if source is not None}
    if not values:
        return {"peak_gib": None, "source": None}
    return {
        "peak_gib": max(values),
        "source": sources.pop() if len(sources) == 1 else "mixed",
    }


def record_for(args, prompt, torch=None, placement=None, measured=None, error=None):
    """The whole JSON, measured or not.

    A run whose batch did not fit still writes a record: the failure is a result
    and a table that lists the chip with ``n/a`` says more than one that omits it.
    """
    measured = measured or {}
    return {
        "schema": 1,
        "kind": "qwen-image-21-bench",
        "revision": revision(),
        "prompt": {
            "id": prompt["id"],
            "sha256": prompt["sha256"],
            "text": prompt["prompt"],
        },
        "config": {
            "height": args.height,
            "width": args.width,
            "steps": args.steps,
            "batch": args.batch,
            "dtype": "bfloat16",
            "seed": args.seed,
            "true_cfg_scale": args.true_cfg_scale,
            "negative_prompt": args.negative_prompt,
            "use_kv_cache": not args.no_kv_cache,
            "latents": "loaded" if args.load_latents else "sampled",
            "placement": placement,
        },
        "protocol": {
            "warmup_rounds": max(args.warmup, 1),
            "min_run_time_s": args.min_run_time,
            "runs": measured.get("runs", 0),
            "num_threads": 1,
            "timer": "torch.utils.benchmark.Timer.blocked_autorange",
            "cold_cache": args.cold_cache,
        },
        "latency": measured.get("latency"),
        "phases": measured.get("phases", {}),
        "throughput": throughput_for(args, measured.get("latency"), error),
        "memory": measured.get("memory", {"peak_gib": None, "source": None}),
        "determinism": measured.get(
            "determinism", {"identical": None, "image_sha256": None}
        ),
        "environment": environment(),
        "versions": versions(torch, args.device) if torch is not None else None,
        # The failure's own text lives in throughput.error; repeating it here
        # would print an allocator dump twice in the table's caveats.
        "notes": measured.get("notes", [])
        + (
            [
                "the requested batch did not run, so this record carries no "
                "latency or phase measurement; `throughput.error` has the "
                "backend's message"
            ]
            if error
            else []
        ),
    }


def throughput_for(args, latency, error):
    """``{images_per_s, error}`` -- measured, absent, or refused with the reason.

    Reported only when it is real. At batch 1 nothing is derived from the latency:
    a batch-1 rate is the latency restated, and restating it as a rate would
    invite a comparison against a real throughput number. Above batch 1 the rate
    is the batch divided by the per-call median, which is the quantity MLPerf's
    Offline scenario reports.
    """
    if error:
        return {"images_per_s": None, "error": error}
    if args.batch < 2 or latency is None:
        return {"images_per_s": None, "error": None}
    return {
        "images_per_s": args.batch / latency["per_call_s"]["median_s"],
        "error": None,
    }


def run(argv=None):
    args = parse_args(argv)

    if args.cold_cache:
        cache = cold_cache_dir()
        import os

        os.environ["TRITON_CACHE_DIR"] = str(cache)
        print(f"cold cache: TRITON_CACHE_DIR={cache}")

    torch = common.import_torch(args.device)
    diffusers = common.import_diffusers()

    prompt = prompts.canonical()
    encoder, transformer, vae = common.resolve_placement(torch, args)
    device = torch.device(transformer[0])

    print(f"device: {args.device}   torch: {torch.__version__}")
    print(
        f"prompt: {prompt['id']}  sha256 {prompt['sha256'][:16]}...  {prompt['prompt']}"
    )
    print(
        f"{args.width}x{args.height}, {args.steps} steps, batch {args.batch}, "
        f"true_cfg_scale={args.true_cfg_scale}, seed={args.seed}"
    )
    print("loading the pipeline (outside the measurement)")

    pipe = diffusers.QwenImage21Pipeline.from_pretrained(
        args.model, dtype=torch.bfloat16
    )
    # What place_components returns, not what was asked for: the two differ
    # whenever --vae-device or --devices cannot be honoured, and a record that
    # describes the request rather than the run is worse than no record.
    _, placement = common.place_components(
        torch, pipe, encoder, transformer, vae, args.blocks_per_device
    )

    try:
        measured = measure(torch, pipe, args, prompt["prompt"], device)
        error = None
    except Unrunnable as failure:
        measured = {}
        error = str(failure)
        print()
        print(f"the requested batch could not run: {error}")

    record = record_for(args, prompt, torch, placement, measured, error)
    Path(args.out).write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print()
    print(f"-> {args.out}")
    if error is None:
        images = measured["images"]
        if args.image and images:
            images[0].save(args.image)
            print(f"last call's first image -> {args.image}")
        latency = measured["latency"]
        print(
            f"  latency    {latency['per_image_s']['median_s']:.2f} s/image "
            f"(median over {measured['runs']} calls)"
        )
        throughput = record["throughput"]["images_per_s"]
        if throughput is not None:
            print(f"  throughput {throughput:.4f} images/s at batch {args.batch}")
        peak = record["memory"]["peak_gib"]
        print(
            f"  memory     {'n/a' if peak is None else f'{peak:.2f}'} GiB "
            f"({record['memory']['source']})"
        )
        print(
            "  determinism "
            + ("two calls agreed" if record["determinism"]["identical"] else "FAILED")
        )
        # Seconds for every phase; print_phases owns the one row that is shown
        # in milliseconds.
        common.print_phases(
            {name: stats["median_s"] for name, stats in measured["phases"].items()}
        )
        if args.batch < 2:
            print()
            print(
                "batch 1: no throughput is reported. Pass --batch N for one; "
                "see the README."
            )
        common.report_memory(torch, args.device)
        return 0

    common.report_memory(torch, args.device)
    return 1


def load_records(paths):
    """Read the bench JSONs, refusing anything that is not one.

    A file that is missing, unreadable or not JSON fails with the path rather
    than a traceback: this is the one entry point that runs on a machine that
    did not produce the files, so the error is the user interface.
    """
    records = []
    for path in paths:
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except OSError as error:
            raise SystemExit(f"{path}: {error.strerror}") from error
        except json.JSONDecodeError as error:
            raise SystemExit(f"{path}: not JSON ({error})") from error
        if data.get("kind") != "qwen-image-21-bench":
            raise SystemExit(
                f"{path}: not a qwen-image-21 bench JSON (kind={data.get('kind')!r})"
            )
        records.append((Path(path).stem, data))
    return records


def _cell(records, getter, fmt="{:.2f}"):
    cells = []
    for _, data in records:
        value = getter(data)
        cells.append("n/a" if value is None else fmt.format(value))
    return cells


def _row(name, records, getter, fmt="{:.2f}"):
    return f"| {name} | " + " | ".join(_cell(records, getter, fmt)) + " |"


def _phase(data, name, key):
    phase = (data.get("phases") or {}).get(name)
    return None if phase is None else phase.get(key)


def render_table(paths, baseline):
    """One markdown table, one column per JSON. Needs no torch and no device.

    Rows are the readings a chip's result is quoted from. A value a run could not
    obtain renders as ``n/a`` rather than being dropped, so "not measured" is
    distinguishable from "not shown", and the reasons are listed below the table
    instead of being folded into a cell.
    """
    records = load_records(paths)
    labels = [label for label, _ in records]
    if baseline is not None and baseline not in labels:
        raise SystemExit(f"--baseline {baseline!r} is none of {labels}")

    lines = [
        "| metric | " + " | ".join(labels) + " |",
        "| --- | " + " | ".join("---" for _ in labels) + " |",
        "| prompt | "
        + " | ".join(
            f"{d['prompt']['id']} / `{d['prompt']['sha256'][:8]}`" for _, d in records
        )
        + " |",
        "| config | "
        + " | ".join(
            f"{d['config']['width']}x{d['config']['height']}, "
            f"{d['config']['steps']} steps, batch {d['config']['batch']}, "
            f"kv={d['config']['use_kv_cache']}"
            for _, d in records
        )
        + " |",
        _row(
            "latency, median s/image",
            records,
            lambda d: _dig(d, "latency", "per_image_s", "median_s"),
        ),
        _row(
            "latency, mean s/image",
            records,
            lambda d: _dig(d, "latency", "per_image_s", "mean_s"),
        ),
        _row(
            "latency, min s/image",
            records,
            lambda d: _dig(d, "latency", "per_image_s", "min_s"),
        ),
        _row(
            "latency std, s",
            records,
            lambda d: _dig(d, "latency", "per_image_s", "std_s"),
            "{:.3f}",
        ),
        _row("measured calls", records, lambda d: d["protocol"]["runs"], "{:d}"),
    ]
    for phase in common.PhaseTimer.KEYS:
        if phase in MILLISECOND_PHASES:
            lines.append(
                _row(
                    f"{phase}, median ms",
                    records,
                    lambda d, p=phase: _scaled(_phase(d, p, "median_s"), 1e3),
                    "{:.1f}",
                )
            )
        else:
            lines.append(
                _row(
                    f"{phase}, median s",
                    records,
                    lambda d, p=phase: _phase(d, p, "median_s"),
                )
            )
    lines += [
        _row(
            "throughput, images/s",
            records,
            lambda d: d["throughput"]["images_per_s"],
            "{:.4f}",
        ),
        _row("peak GiB", records, lambda d: d["memory"]["peak_gib"]),
        "| memory counter | "
        + " | ".join(str(d["memory"]["source"] or "n/a") for _, d in records)
        + " |",
        "| determinism | "
        + " | ".join(
            {True: "identical", False: "**differs**"}.get(
                d["determinism"]["identical"], "n/a"
            )
            for _, d in records
        )
        + " |",
        "| `FLAGOS_LOG` | "
        + " | ".join(str(_dig(d, "environment", "log") or "unset") for _, d in records)
        + " |",
        "| triton cache | "
        + " | ".join(
            str(_dig(d, "environment", "triton_cache_dir") or "default")
            for _, d in records
        )
        + " |",
    ]

    if baseline is not None:
        base = dict(records)[baseline]
        cells = []
        for _, data in records:
            mine = _dig(data, "latency", "per_image_s", "median_s")
            theirs = _dig(base, "latency", "per_image_s", "median_s")
            cells.append("n/a" if not mine or not theirs else f"{mine / theirs:.2f}x")
        lines.append(f"| vs {baseline}, latency | " + " | ".join(cells) + " |")

        cells = []
        for _, data in records:
            mine = data["throughput"]["images_per_s"]
            theirs = base["throughput"]["images_per_s"]
            cells.append("n/a" if not mine or not theirs else f"{mine / theirs:.2f}x")
        lines.append(f"| vs {baseline}, throughput | " + " | ".join(cells) + " |")

    caveats = []
    for label, data in records:
        identical = data["determinism"]["identical"]
        if identical is False:
            caveats.append(
                f"- **{label}: repeated calls did not produce identical images.** "
                "The latency is still a measurement; a paired comparison against "
                "another backend is not, because the two are not sampling the same "
                "trajectory."
            )
        if data["throughput"]["error"]:
            caveats.append(f"- {label}: no throughput -- {data['throughput']['error']}")
        if data["latency"] is None:
            caveats.append(f"- {label}: nothing was measured; see its `notes` field")
        elif (_dig(data, "latency", "per_image_s", "runs") or 0) < 2:
            caveats.append(
                f"- {label}: {_dig(data, 'latency', 'per_image_s', 'runs')} latency "
                "sample(s), so the spread is not a spread. Raise --min-run-time."
            )
        if data["protocol"]["cold_cache"]:
            caveats.append(
                f"- {label}: measured with a **cold Triton cache**, which is not "
                "comparable with a warm run."
            )
        if _dig(data, "environment", "dispatch_log"):
            caveats.append(
                f"- {label}: `FLAGOS_LOG={_dig(data, 'environment', 'log')}` includes "
                "`dispatch`, which writes a line per operator call and costs real "
                "time. Do not quote this number against a run with it off."
            )
        for note in data.get("notes", []):
            caveats.append(f"- {label}: {note}")

    text = "\n".join(lines) + "\n"
    if caveats:
        text += "\n" + "\n".join(caveats) + "\n"
    print(text)
    return 0


def _dig(data, *keys):
    """A nested field, or ``None`` -- a missing section is a missing reading, not
    an error, since a run that could not measure anything has no latency."""
    current = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _scaled(value, factor):
    """``value * factor``, or ``None`` -- so a missing reading survives the
    unit conversion instead of becoming a zero that looks like a measurement."""
    return None if value is None else value * factor


def main(argv=None):
    args = parse_args(argv)
    if args.table:
        return render_table(args.table, args.baseline)
    return run(argv)


if __name__ == "__main__":
    sys.exit(main())
