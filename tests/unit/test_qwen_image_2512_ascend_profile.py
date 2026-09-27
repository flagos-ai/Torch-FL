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

"""Unit coverage for the Qwen-Image-2512 Ascend validation profile.

``ascend_profile.py`` is the artifact issue #327 asks for, and a validator that
cannot fail is not a validator. Everything here runs on synthetic inputs -- a
hand-built run.sh log, a manifest and small PNGs -- so the checks are exercised
on a box with no accelerator, and each test breaks exactly one clause of the
contract and asserts the profile says so.

The one thing this cannot cover is whether the real run satisfies the contract;
that is what the hardware run and its recorded baseline are for.
"""

import importlib.util
import json
import re
from pathlib import Path

import pytest

# The cohort half of the profile reads PNGs, so these fixtures need the same two
# packages the flow itself needs -- and the profile reaches for them lazily, so
# it is only the fixtures here that depend on them at import time.
#
# They are imported defensively rather than with a module-level ``importorskip``
# because the two are not equivalent: an ``importorskip`` that fires during
# collection leaves the module with nothing to collect, and pytest reports that
# as exit 5, which the per-file unit runner scores as a failure rather than as a
# skip. A marker keeps the tests collectible, so the box sees "skipped" and the
# file sees exit 0. Neither package is a dependency of this one, and Pillow in
# particular is not in every image that runs the unit suite.
try:
    import numpy  # noqa: F401
except ImportError:  # pragma: no cover - only reached where the stack is absent
    numpy = None
try:
    from PIL import Image
except ImportError:  # pragma: no cover - only reached where the stack is absent
    Image = None

# Everything that builds a run needs both, since ``make_run`` writes the cohort's
# PNGs. ``test_the_shipped_baseline_describes_the_cohort`` reads only JSON and
# the conf, so it is deliberately left unmarked and runs on any box.
needs_the_png_stack = pytest.mark.skipif(
    numpy is None or Image is None,
    reason="the profile's PNG half needs numpy and Pillow",
)

FLOW = Path(__file__).parents[1] / "manual" / "qwen_image_2512"
REPO_ROOT = FLOW.parents[2]

SPEC = importlib.util.spec_from_file_location(
    "qwen_image_2512_ascend_profile_under_test", FLOW / "ascend_profile.py"
)
PROFILE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROFILE)

STAGES = ("text-encoder", "transformer-step", "vae", "full")

# Two shards, so the placement check's "more than one card" clause is satisfiable
# and the sweep's manifest placement can disagree with it independently.
PLACEMENT = {
    "encoder": "flagos:2",
    "vae": "flagos:2",
    "transformer": ["flagos:0", "flagos:1"],
    "blocks": [[0, 29, "flagos:0"], [30, 59, "flagos:1"]],
}

# The operators the profile is written to see in the Ascend list. Kept to the
# three this change moved plus one that was already native, so a test that
# breaks the routing check is testing the check and not the fixture's size.
ASCEND_OPS = ["convolution", "gelu", "gelu_backward", "sum.dim_IntList"]
FLAGGEMS_OPS = ["add.Tensor", "mul.Tensor", "pow.Tensor_Scalar"]

WIDTH, HEIGHT = 1664, 928


def run_sh_log(stage=None, exit_status=0, extra=""):
    """A log shaped like the one ``run.sh`` writes for an infer or sweep run."""
    header = ""
    if stage is not None:
        header += f"stage: {stage}   device: flagos   torch: 2.10.0+cpu\n"
    else:
        header += (
            "device: flagos   torch: 2.10.0+cpu\n"
            "12 model-card prompts, 1664x928, 50 steps, \n"
        )

    placement_lines = (
        f"  text_encoder -> {PLACEMENT['encoder']}\n"
        f"  vae          -> {PLACEMENT['vae']} (the pipeline's execution device)\n"
        "  transformer: blocks 0..29 -> flagos:0, 30..59 -> flagos:1\n"
    )

    dispatch = "".join(
        f"[flagos dispatch] {op} -> ascend\n" for op in ASCEND_OPS
    ) + "".join(f"[flagos dispatch] {op} -> flagos_python\n" for op in FLAGGEMS_OPS)

    records = len(ASCEND_OPS) + len(FLAGGEMS_OPS)
    census = (
        "\n---- readings ----\n"
        f"cpu_fallback ops   : 0\n"
        f"libentry failures  : 0\n"
        f"dispatch records   : {records}\n"
        f"distinct ATen ops  : {records}\n"
        f"exit status        : {exit_status}\n"
    )
    return (
        f"[flagos] loading backend config from {REPO_ROOT}/torch_fl/configs/backends_ascend.conf\n"
        f"  loaded in 5.9s\n" + header + placement_lines + dispatch + extra + census
    )


def write_png(path, colour):
    """A real PNG at the profile's expected size, filled with one colour.

    Truncated to the cohort dimensions by ``resize`` rather than by writing a
    1664x928 array: the statistics only have to be *measurable* and finite, and a
    single-colour image keeps the expected mean exactly predictable.
    """
    image = Image.new("RGB", (16, 16), colour).resize((WIDTH, HEIGHT))
    image.save(path)


def make_run(tmp_path, sweep=True, stage_logs=True, manifest_overrides=None, pngs=12):
    """A complete, passing run on disk; the tests then break one thing each."""
    run_dir = tmp_path / "run"
    sweep_dir = run_dir / "flagos"
    sweep_dir.mkdir(parents=True)

    args = []

    stage_logs_out = {}
    if stage_logs:
        for stage in STAGES:
            path = run_dir / f"{stage}.log"
            path.write_text(run_sh_log(stage=stage), encoding="utf-8")
            args += ["--stage-log", f"{stage}={path}"]
            stage_logs_out[stage] = path

    images = []
    if sweep:
        for index in range(pngs):
            prompt_id = f"{index + 1:02d}"
            file = f"{prompt_id}.png"
            write_png(sweep_dir / file, (20 + index, 40, 60))
            images.append(
                {
                    "id": prompt_id,
                    "prompt": f"prompt {prompt_id}",
                    "file": file,
                    "seconds": 600.0 + index,
                    "memory": {"reserved_gib": 26.0, "allocated_gib": 18.0},
                }
            )

    manifest = {
        "device": "flagos",
        "torch": "2.10.0+cpu",
        "model": "Qwen/Qwen-Image-2512",
        "negative_prompt": "low quality",
        "width": WIDTH,
        "height": HEIGHT,
        "num_inference_steps": 50,
        "true_cfg_scale": 4.0,
        "seed": 42,
        "placement": {
            "encoder": PLACEMENT["encoder"],
            "vae": PLACEMENT["vae"],
            "transformer": PLACEMENT["transformer"],
            "blocks_per_device": [30, 30],
        },
        "images": images,
    }
    manifest.update(manifest_overrides or {})
    (sweep_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    sweep_log = run_dir / "sweep.log"
    sweep_log.write_text(run_sh_log(), encoding="utf-8")
    args += ["--sweep-log", str(sweep_log), "--sweep-dir", str(sweep_dir)]

    return args, run_dir, sweep_dir, stage_logs_out


def build_baseline(tmp_path, run_dir, sweep_dir, stage_logs):
    """Record a baseline from a synthetic run, the way ``--record`` does."""
    baseline = tmp_path / "baseline.json"
    argv = ["--record", str(baseline), "--sweep-dir", str(sweep_dir)]
    argv += ["--sweep-log", str(run_dir / "sweep.log")]
    for stage, path in stage_logs.items():
        argv += ["--stage-log", f"{stage}={path}"]
    assert PROFILE.main(argv) == 0
    return baseline


def run_profile(args, baseline, capsys):
    code = PROFILE.main(args + ["--baseline", str(baseline)])
    return code, capsys.readouterr().out


@needs_the_png_stack
def test_a_complete_run_satisfies_the_contract(tmp_path, capsys):
    args, run_dir, sweep_dir, stage_logs = make_run(tmp_path)
    baseline = build_baseline(tmp_path, run_dir, sweep_dir, stage_logs)

    code, out = run_profile(args, baseline, capsys)

    assert code == 0, out
    assert "the Ascend validation contract holds" in out
    assert "[FAIL]" not in out
    # The three checks that carry the issue's own wording, named explicitly so a
    # rename that quietly drops one is a failure rather than a shorter report.
    assert "FlagGems-first is active (flagos_python dispatched to)" in out
    assert "no global backend override (the retired ALL_USE_VENDOR)" in out
    assert "the transformer is actually split across more than one card" in out


@needs_the_png_stack
def test_cpu_fallback_fails_the_contract(tmp_path, capsys):
    args, run_dir, sweep_dir, stage_logs = make_run(tmp_path)
    baseline = build_baseline(tmp_path, run_dir, sweep_dir, stage_logs)

    # One op fell off the accelerator entirely, which is what the census counts.
    path = stage_logs["full"]
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "cpu_fallback ops   : 0", "cpu_fallback ops   : 3"
        ),
        encoding="utf-8",
    )

    code, out = run_profile(args, baseline, capsys)

    assert code == 1
    assert "[FAIL] full: cpu_fallback ops" in out


@needs_the_png_stack
def test_a_global_backend_override_fails_the_contract(tmp_path, capsys):
    args, run_dir, sweep_dir, stage_logs = make_run(tmp_path)
    baseline = build_baseline(tmp_path, run_dir, sweep_dir, stage_logs)

    # FLAGOS_FORCE_BACKEND (the retired ALL_USE_VENDOR) repins the whole table.
    path = stage_logs["full"]
    path.write_text(
        path.read_text(encoding="utf-8")
        + "[flagos] FLAGOS_FORCE_BACKEND=vendor: 151 ops repinned, 8 left as configured\n",
        encoding="utf-8",
    )

    code, out = run_profile(args, baseline, capsys)

    assert code == 1
    assert "no global backend override" in out
    assert "[FAIL] no global backend override" in out


@needs_the_png_stack
def test_a_per_operator_override_fails_the_contract(tmp_path, capsys):
    args, run_dir, sweep_dir, stage_logs = make_run(tmp_path)
    baseline = build_baseline(tmp_path, run_dir, sweep_dir, stage_logs)

    path = stage_logs["full"]
    path.write_text(
        path.read_text(encoding="utf-8")
        + "[flagos] env override: gelu -> flagos_python\n",
        encoding="utf-8",
    )

    code, out = run_profile(args, baseline, capsys)

    assert code == 1
    assert "[FAIL] no per-operator env override" in out


@needs_the_png_stack
def test_an_operator_leaving_the_ascend_route_fails_the_contract(tmp_path, capsys):
    args, run_dir, sweep_dir, stage_logs = make_run(tmp_path)
    baseline = build_baseline(tmp_path, run_dir, sweep_dir, stage_logs)

    # The regression this profile exists to catch: gelu slips back onto FlagGems,
    # where its tanh kernel cannot compile on this backend. Replaced in every log,
    # because the routing check reads the union of them -- moving it in one stage
    # alone would leave the op on both routes, which is a different defect.
    for path in list(stage_logs.values()) + [run_dir / "sweep.log"]:
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                "[flagos dispatch] gelu -> ascend",
                "[flagos dispatch] gelu -> flagos_python",
            ),
            encoding="utf-8",
        )

    code, out = run_profile(args, baseline, capsys)

    assert code == 1
    assert "[FAIL] ascend: distinct operators" in out
    assert "[FAIL] flagos_python: distinct operators" in out


@needs_the_png_stack
def test_a_different_placement_fails_the_contract(tmp_path, capsys):
    args, run_dir, sweep_dir, stage_logs = make_run(tmp_path)
    baseline = build_baseline(tmp_path, run_dir, sweep_dir, stage_logs)

    # Everything on one card: the split the profile is required to exercise is
    # gone, and the old context hooks were what used to be needed to get it back.
    for path in stage_logs.values():
        path.write_text(
            path.read_text(encoding="utf-8")
            .replace(
                "  transformer: blocks 0..29 -> flagos:0, 30..59 -> flagos:1",
                "  transformer: all 60 blocks -> flagos:0 (no split)",
            )
            .replace("  text_encoder -> flagos:2", "  text_encoder -> flagos:0")
            .replace("  vae          -> flagos:2", "  vae          -> flagos:0"),
            encoding="utf-8",
        )

    code, out = run_profile(args, baseline, capsys)

    assert code == 1
    assert "[FAIL] encoder card" in out
    assert "[FAIL] transformer split (block ranges -> card)" in out
    assert "[FAIL] the transformer is actually split across more than one card" in out


@needs_the_png_stack
def test_an_incomplete_cohort_fails_the_contract(tmp_path, capsys):
    args, run_dir, sweep_dir, stage_logs = make_run(tmp_path, pngs=11)
    # A baseline recorded from a complete cohort, so the shortfall is the run's.
    _, complete_run, complete_dir, complete_logs = make_run(tmp_path / "complete")
    baseline = build_baseline(tmp_path, complete_run, complete_dir, complete_logs)

    code, out = run_profile(args, baseline, capsys)

    assert code == 1
    assert "[FAIL] prompt count" in out


@needs_the_png_stack
def test_a_wrong_image_size_fails_the_contract(tmp_path, capsys):
    args, run_dir, sweep_dir, stage_logs = make_run(tmp_path)
    baseline = build_baseline(tmp_path, run_dir, sweep_dir, stage_logs)

    Image.new("RGB", (1024, 1024), (10, 10, 10)).save(sweep_dir / "01.png")

    code, out = run_profile(args, baseline, capsys)

    assert code == 1
    assert "[FAIL] [01] image size" in out


@needs_the_png_stack
def test_a_reshuffled_manifest_setting_fails_the_contract(tmp_path, capsys):
    args, run_dir, sweep_dir, stage_logs = make_run(tmp_path)
    baseline = build_baseline(tmp_path, run_dir, sweep_dir, stage_logs)

    # The cohort is frozen: a sweeter 30-step run is a different measurement and
    # must not be compared against a 50-step baseline silently.
    manifest = json.loads((sweep_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["num_inference_steps"] = 30
    (sweep_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    code, out = run_profile(args, baseline, capsys)

    assert code == 1
    assert "[FAIL] manifest: num_inference_steps" in out


@needs_the_png_stack
def test_a_failed_stage_fails_the_contract(tmp_path, capsys):
    args, run_dir, sweep_dir, stage_logs = make_run(tmp_path)
    baseline = build_baseline(tmp_path, run_dir, sweep_dir, stage_logs)

    path = stage_logs["vae"]
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "exit status        : 0", "exit status        : 1"
        ),
        encoding="utf-8",
    )

    code, out = run_profile(args, baseline, capsys)

    assert code == 1
    assert "[FAIL] vae: exit status" in out


@needs_the_png_stack
def test_a_missing_stage_log_fails_the_contract(tmp_path, capsys):
    args, run_dir, sweep_dir, stage_logs = make_run(tmp_path)
    baseline = build_baseline(tmp_path, run_dir, sweep_dir, stage_logs)

    stage_logs["full"].unlink()

    with pytest.raises(SystemExit, match="no such log"):
        PROFILE.main(args + ["--baseline", str(baseline)])


def test_the_shipped_baseline_describes_the_cohort(tmp_path):
    """The committed baseline is data the profile reads, so it has to be real."""
    path = FLOW / "ascend_baseline.json"
    if not path.is_file():
        pytest.skip("no baseline recorded yet")
    baseline = json.loads(path.read_text(encoding="utf-8"))

    assert len(baseline["images"]) == PROFILE.COHORT["prompts"]
    assert baseline["placement"]["encoder"]
    assert len(baseline["placement"]["blocks"]) > 1
    assert "ascend" in baseline["routing"]
    # The routes this change moved, as the cohort actually dispatches them, have
    # to be in the recorded set -- otherwise the baseline was recorded against a
    # tree that does not contain the fix.
    for op in ("gelu", "sum.dim_IntList"):
        assert op in baseline["routing"]["ascend"], op
        assert op not in baseline["routing"].get("flagos_python", []), op

    # ``gelu_backward`` cannot be in that set: the cohort is inference, and a
    # forward-only graph never calls it. Its route is still part of this change,
    # so it is asserted against the conf the run loaded, which is where a later
    # reroute of it would appear.
    conf = (REPO_ROOT / PROFILE.EXPECTED_CONF).read_text(encoding="utf-8")
    for op in ("gelu", "gelu_backward", "sum.dim_IntList"):
        route = re.search(rf"^{re.escape(op)} = (\S+)", conf, re.M)
        assert route and route.group(1) == "ascend", (op, route and route.group(1))
