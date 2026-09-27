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

"""The Ascend validation profile: assert the whole Qwen-Image contract (issue #327).

The rest of this directory runs the model and prints readings. This file is the
part that *enforces* them. It reads only what a finished run already wrote -- the
four ``run.sh infer`` logs, the ``run.sh sweep`` log and the sweep's ``manifest.json``
plus its PNGs -- and checks them against a frozen baseline, so a later Ascend run
is judged rather than eyeballed.

The contract is the one the 12-prompt Showcase established, and each check below
exists because something in that chain broke once:

``configuration``
    The shipped FlagGems-first conf is the one in force, nothing repinned the
    route table globally, and no per-operator env override was used. The Ascend
    work of #323--#326 was routing work; a run that quietly set
    ``FLAGOS_FORCE_BACKEND`` would be measuring a different tree.
``fallback``
    Zero ``cpu_fallback`` records and zero triton compile failures. These are what
    a *missing* route looks like, and they are silent in the only sense that
    matters here: the run finishes and the images look plausible.
``stages``
    All four stages ran and exited 0.
``placement``
    The encoder/VAE card and the transformer split are the ones the baseline
    recorded, on stock configuration -- the multi-device placement that previously
    needed context hooks.
``routing``
    The distinct operator set per effective backend equals the baseline's. This
    is the check that catches a route moving without anyone noticing, and it is
    where the operator fixes of this change are visible: ``gelu`` and
    ``sum.dim_IntList`` are expected in the ``ascend`` list and absent from the
    ``flagos_python`` one.

    Only what the cohort dispatches is in scope, so ``gelu_backward`` is not
    here: these four stages are inference, and a forward-only graph never calls
    it. Its route is asserted where it is observable, against
    ``torch_fl/configs/backends_ascend.conf``, by
    ``tests/unit/test_qwen_image_2512_ascend_profile.py``.
``cohort``
    Twelve model-card prompts, at the frozen dimensions/steps/seed/CFG scale and
    placement, each with a PNG of the expected size whose pixel statistics are
    finite and within tolerance of the baseline's.

A baseline is data about one accepted run, not a claim of exact reproducibility:
two Ascend runs of the same prompt do not produce bit-identical pixels (the RNG
stream is the backend's), and the box is shared, so timings move. Correctness
checks are exact; the per-case statistics and timings are reported as deltas and
gated on a stated tolerance or not at all.

Usage::

    python tests/manual/qwen_image_2512/ascend_profile.py \\
        --stage-log text-encoder=$OUT/text-encoder.log \\
        --stage-log transformer-step=$OUT/transformer-step.log \\
        --stage-log vae=$OUT/vae.log \\
        --stage-log full=$OUT/full.log \\
        --sweep-log $OUT/sweep.log \\
        --sweep-dir $OUT/flagos

Every path above is a ``run.sh`` *stdout* capture -- ``run.sh ... > log 2>&1``
rather than the ``$OUT_DIR/infer-*.log`` it writes for itself. The inner log is
everything the runner needs, but ``run.sh`` prints the readings and the exit
status around it rather than into it, so the inner log alone has neither and
every ``fallback`` and ``stages`` check would report a missing value.

``--record FILE`` writes the accepted baseline this run measured instead of
comparing against it; that is how ``ascend_baseline.json`` was produced.

See ``tests/manual/qwen_image_2512/README.md``, section "The Ascend validation
profile", for the procedure that produces the inputs and the readings to expect.
"""

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_BASELINE = HERE / "ascend_baseline.json"

STAGES = ("text-encoder", "transformer-step", "vae", "full")

# Frozen cohort settings. Restated here rather than imported from sweep.py: the
# profile's job is to notice if that module's constants move, which it cannot do
# if it reads them from the same place.
COHORT = {
    "model": "Qwen/Qwen-Image-2512",
    "width": 1664,
    "height": 928,
    "num_inference_steps": 50,
    "true_cfg_scale": 4.0,
    "seed": 42,
    "prompts": 12,
}

# The conf this profile is written for, named by its path under the repository.
EXPECTED_CONF = "torch_fl/configs/backends_ascend.conf"

# A summary line run.sh prints, e.g. ``cpu_fallback ops   : 0``. The key class
# admits capitals because one of the readings it prints is ``distinct ATen ops``.
_READING = re.compile(r"^(?P<key>[A-Za-z_ ]+?)\s*:\s*(?P<value>-?\d+)\s*$", re.M)

# ``[flagos dispatch] op -> backend``. The backend token is the last field, which
# is what run.sh's own census keys on.
_DISPATCH = re.compile(r"flagos dispatch\] (\S+) -> ([a-z_]+)")


class Report:
    """Collects checks and prints them, failing the process on any failure."""

    def __init__(self):
        self.failures = []
        self.checks = 0

    def check(self, name, ok, detail=""):
        self.checks += 1
        if not ok:
            self.failures.append(name)
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}" + (f" -- {detail}" if detail else ""))
        return ok

    def equal(self, name, got, want):
        return self.check(name, got == want, f"got {got!r}, expected {want!r}")

    def section(self, title):
        print()
        print(f"---- {title} ----")


def read_log(path):
    """A run.sh log as text, with a fatal error if it was never written."""
    path = Path(path)
    if not path.is_file():
        raise SystemExit(f"no such log: {path}")
    return path.read_text(encoding="utf-8", errors="replace")


def readings(text):
    """The ``key : value`` block run.sh's summarise() prints."""
    found = {}
    for match in _READING.finditer(text):
        found[match.group("key").strip()] = int(match.group("value"))
    return found


def dispatch_map(text):
    """``{backend: sorted distinct operator names}`` from the dispatch records.

    Read from the dispatch lines themselves rather than from run.sh's printed
    census, so the profile does not depend on that shell formatting surviving.
    """
    by_backend = {}
    for op, backend in _DISPATCH.findall(text):
        by_backend.setdefault(backend, set()).add(op)
    return {backend: sorted(ops) for backend, ops in by_backend.items()}


def placement(text):
    """The placement lines the runners print, as a dict.

    Both ``infer.py`` and ``sweep.py`` print the same three facts; the sweep
    label is ``text_encoder`` and the staged runner's is the same, so one parser
    covers both.
    """
    found = {}
    encoder = re.search(r"^\s*text_encoder\s*->\s*(\S+)", text, re.M)
    vae = re.search(r"^\s*vae\s*->\s*(\S+)", text, re.M)
    blocks = re.search(r"^\s*transformer: blocks (.*)$", text, re.M)
    if encoder:
        found["encoder"] = encoder.group(1)
    if vae:
        found["vae"] = vae.group(1)
    if blocks:
        # ``0..29 -> flagos:0, 30..59 -> flagos:1``
        ranges = re.findall(r"(\d+)\.\.(\d+) -> (\S+?)(?:,|$)", blocks.group(1))
        found["blocks"] = [
            [int(start), int(end), device.strip()] for start, end, device in ranges
        ]
    return found


def image_stats(path):
    """Finite pixel statistics for a PNG, or None if it is unreadable.

    ``Image.convert("RGB")`` first: the sweep writes RGB, but a profile that
    silently aggregated an RGBA or palette image would report statistics for a
    different pixel layout than the baseline recorded.
    """
    import numpy as np
    from PIL import Image

    with Image.open(path) as handle:
        size = handle.size
        array = np.asarray(handle.convert("RGB"), dtype=np.float64) / 255.0
    if not np.isfinite(array).all():
        return {"size": list(size), "finite": False}
    return {
        "size": list(size),
        "finite": True,
        "mean": round(float(array.mean()), 4),
        "std": round(float(array.std()), 4),
        "min": round(float(array.min()), 4),
        "max": round(float(array.max()), 4),
    }


def check_configuration(report, texts):
    """The conf in force, and the absence of any global or per-op repin."""
    report.section("configuration")

    loaded = [
        match.group(1)
        for text in texts
        for match in re.finditer(r"\[flagos\] loading backend config from (\S+)", text)
    ]
    report.check(
        "the shipped Ascend conf is the one loaded",
        bool(loaded) and all(path.endswith(EXPECTED_CONF) for path in loaded),
        f"loaded {sorted(set(loaded)) or 'nothing'}",
    )
    if loaded:
        report.check(
            "it is loaded from this worktree",
            all(Path(path).is_relative_to(REPO_ROOT) for path in loaded),
            f"worktree {REPO_ROOT}",
        )

    forced = [t for t in texts if "FLAGOS_FORCE_BACKEND=" in t]
    report.check(
        "no global backend override (the retired ALL_USE_VENDOR)",
        not forced,
        f"{len(forced)} log(s) carry a FLAGOS_FORCE_BACKEND repin line",
    )

    overridden = sorted(
        {
            match.group(0)
            for text in texts
            for match in re.finditer(r"\[flagos\] env override: .*", text)
        }
    )
    report.check(
        "no per-operator env override",
        not overridden,
        f"{len(overridden)} override line(s): {overridden[:3]}",
    )


def check_fallback(report, texts):
    """Zero CPU fallback and zero triton compile failures, per log."""
    report.section("fallback")
    for name, text in texts.items():
        found = readings(text)
        report.equal(f"{name}: cpu_fallback ops", found.get("cpu_fallback ops"), 0)
        report.equal(f"{name}: libentry failures", found.get("libentry failures"), 0)


def check_stages(report, stage_logs):
    """Every stage ran, named itself, and exited 0."""
    report.section("stages")
    for stage in STAGES:
        text = stage_logs.get(stage)
        if text is None:
            report.check(f"{stage}: log supplied", False)
            continue
        header = re.search(rf"^stage: {re.escape(stage)}\b", text, re.M)
        report.check(f"{stage}: ran", header is not None)
        exit_status = re.search(r"^exit status\s*:\s*(\d+)\s*$", text, re.M)
        report.equal(
            f"{stage}: exit status",
            int(exit_status.group(1)) if exit_status else None,
            0,
        )


def check_placement(report, observed, expected):
    """The multi-device placement, on stock configuration."""
    report.section("placement")
    report.equal("encoder card", observed.get("encoder"), expected.get("encoder"))
    report.equal("VAE card", observed.get("vae"), expected.get("vae"))
    report.equal(
        "transformer split (block ranges -> card)",
        observed.get("blocks"),
        expected.get("blocks"),
    )
    # On the observed placement, not the baseline's: the baseline was recorded
    # from a run that split, so checking it there would only re-assert the
    # baseline against itself.
    report.check(
        "the transformer is actually split across more than one card",
        len(observed.get("blocks") or []) > 1,
        f"{len(observed.get('blocks') or [])} shard(s)",
    )


def check_routing(report, observed, expected):
    """Per-backend distinct operator sets, against the baseline."""
    report.section("routing")
    for backend in sorted(set(observed) | set(expected)):
        got = observed.get(backend, [])
        want = expected.get(backend, [])
        if backend == "flagos_python":
            report.check(
                "FlagGems-first is active (flagos_python dispatched to)",
                bool(got),
                f"{len(got)} operator(s)",
            )
        report.equal(
            f"{backend}: distinct operators ({len(want)} in the baseline)",
            got,
            want,
        )


def check_cohort(report, sweep_dir, baseline):
    """Twelve prompts, frozen settings, expected sizes, finite statistics."""
    report.section("cohort")
    manifest_path = Path(sweep_dir) / "manifest.json"
    if not manifest_path.is_file():
        report.check("manifest.json present", False, str(manifest_path))
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    for key, want in COHORT.items():
        if key == "prompts":
            continue
        report.equal(f"manifest: {key}", manifest.get(key), want)

    images = manifest.get("images", [])
    report.equal("prompt count", len(images), COHORT["prompts"])

    want_placement = baseline.get("placement", {})
    got_placement = {
        "encoder": manifest.get("placement", {}).get("encoder"),
        "vae": manifest.get("placement", {}).get("vae"),
    }
    report.equal(
        "manifest: encoder card",
        got_placement["encoder"],
        want_placement.get("encoder"),
    )
    report.equal("manifest: VAE card", got_placement["vae"], want_placement.get("vae"))
    report.equal(
        "manifest: transformer cards",
        manifest.get("placement", {}).get("transformer"),
        want_placement.get("transformer"),
    )

    expected_images = {entry["id"]: entry for entry in baseline.get("images", [])}
    missing = [entry["id"] for entry in images if entry["id"] not in expected_images]
    report.check(
        "every prompt id is one the baseline recorded",
        not missing,
        f"unexpected ids: {missing}",
    )

    for entry in images:
        prompt_id = entry["id"]
        want = expected_images.get(prompt_id, {})
        path = Path(sweep_dir) / entry["file"]
        if not path.is_file():
            report.check(f"[{prompt_id}] PNG written", False, str(path))
            continue

        stats = image_stats(path)
        report.check(
            f"[{prompt_id}] statistics are finite", stats.get("finite") is True
        )
        report.equal(
            f"[{prompt_id}] image size",
            stats.get("size"),
            [COHORT["width"], COHORT["height"]],
        )

        # Mean/std move run to run because the RNG stream is the backend's own,
        # so they are compared with the tolerance the baseline states rather
        # than exactly. The tolerance is wide enough to absorb a different
        # sample and tight enough to catch a decode that produced noise.
        for field, tolerance in baseline.get("stats_tolerance", {}).items():
            got_value = stats.get(field)
            want_value = want.get(field)
            if got_value is None or want_value is None:
                continue
            report.check(
                f"[{prompt_id}] {field} within {tolerance}",
                abs(got_value - want_value) <= tolerance,
                f"got {got_value}, baseline {want_value}",
            )

        got_seconds = entry.get("seconds")
        want_seconds = want.get("seconds")
        if got_seconds is not None and want_seconds is not None:
            print(
                f"  [info] [{prompt_id}] {got_seconds}s "
                f"({'slower' if got_seconds > want_seconds else 'faster'} than the "
                f"baseline's {want_seconds}s)"
            )


def record_baseline(args, stage_logs, sweep_log):
    """Write the accepted baseline this run measured."""
    manifest = json.loads(
        (Path(args.sweep_dir) / "manifest.json").read_text(encoding="utf-8")
    )
    images = []
    for entry in manifest["images"]:
        stats = image_stats(Path(args.sweep_dir) / entry["file"])
        images.append(
            {
                "id": entry["id"],
                "prompt": entry["prompt"],
                "file": entry["file"],
                "seconds": entry["seconds"],
                "memory": entry["memory"],
                "mean": stats.get("mean"),
                "std": stats.get("std"),
                "min": stats.get("min"),
                "max": stats.get("max"),
            }
        )

    baseline = {
        "description": (
            "Ascend validation profile baseline for Qwen-Image-2512 (issue #327). "
            "Produced by ascend_profile.py --record."
        ),
        "hardware": args.hardware,
        "torch": manifest.get("torch"),
        "stages": {stage: readings(text) for stage, text in sorted(stage_logs.items())},
        "placement": {
            **placement(stage_logs.get("full", "") or sweep_log),
            # The log's block ranges are the shard map; the manifest also names
            # the transformer's cards as a list, which is what check_cohort
            # compares against a later run.
            "transformer": manifest.get("placement", {}).get("transformer"),
            "manifest": manifest.get("placement"),
        },
        "routing": dispatch_map("\n".join(list(stage_logs.values()) + [sweep_log])),
        # See check_cohort: the RNG stream is the backend's own, so a re-run of
        # the same prompt is a different sample. These bounds are what a correct
        # decode of this prompt looks like, not an exact reproduction target.
        "stats_tolerance": {"mean": 0.05, "std": 0.05},
        "images": images,
    }
    out = Path(args.record)
    out.write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")
    print(f"recorded {out} ({len(images)} images)")
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Assert the Ascend Qwen-Image-2512 validation contract",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--stage-log",
        action="append",
        default=[],
        metavar="STAGE=PATH",
        help="one run.sh infer log per stage; give one flag per stage",
    )
    parser.add_argument("--sweep-log", required=False, help="the run.sh sweep log")
    parser.add_argument("--sweep-dir", required=False, help="where the sweep wrote")
    parser.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    parser.add_argument(
        "--record",
        default=None,
        metavar="PATH",
        help="write a baseline from this run instead of comparing against one",
    )
    parser.add_argument(
        "--hardware",
        default="Ascend 910",
        help="the hardware string recorded in a baseline",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    stage_logs = {}
    for item in args.stage_log:
        stage, _, path = item.partition("=")
        stage_logs[stage] = read_log(path)

    if args.record:
        if not args.sweep_dir:
            raise SystemExit("--record needs --sweep-dir")
        sweep_log = read_log(args.sweep_log) if args.sweep_log else ""
        return record_baseline(args, stage_logs, sweep_log)

    if not args.sweep_dir or not args.sweep_log:
        raise SystemExit("--sweep-dir and --sweep-log are required to compare")
    sweep_log = read_log(args.sweep_log)
    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))

    report = Report()
    print(f"log set    : {len(stage_logs)} stage log(s) + 1 sweep log")
    print(f"sweep      : {args.sweep_dir}")
    print(f"baseline   : {args.baseline}")

    all_texts = list(stage_logs.values()) + [sweep_log]
    check_configuration(report, all_texts)
    check_fallback(report, {**stage_logs, "sweep": sweep_log})
    check_stages(report, stage_logs)
    check_placement(
        report,
        placement(stage_logs.get("full", "") or sweep_log),
        baseline.get("placement", {}),
    )
    check_routing(
        report, dispatch_map("\n".join(all_texts)), baseline.get("routing", {})
    )
    check_cohort(report, args.sweep_dir, baseline)

    print()
    print(f"{report.checks - len(report.failures)}/{report.checks} checks passed")
    if report.failures:
        print("failed checks:")
        for name in report.failures:
            print(f"  - {name}")
        return 1
    print("the Ascend validation contract holds")
    return 0


if __name__ == "__main__":
    sys.exit(main())
