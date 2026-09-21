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

"""Unit coverage for the device-free parts of the Qwen-Image-2.1 flow."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


FLOW = Path(__file__).parents[1] / "manual" / "qwen_image_21"
SPEC = importlib.util.spec_from_file_location(
    "qwen_image_21_common_under_test", FLOW / "common.py"
)
COMMON = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMMON)

sys.path.insert(0, str(FLOW))
try:
    BENCH_SPEC = importlib.util.spec_from_file_location(
        "qwen_image_21_bench_under_test", FLOW / "bench.py"
    )
    BENCH = importlib.util.module_from_spec(BENCH_SPEC)
    BENCH_SPEC.loader.exec_module(BENCH)
finally:
    sys.path.remove(str(FLOW))


def placement_args(**overrides):
    values = {
        "device": "flagos",
        "devices": None,
        "encoder_device": None,
        "transformer_devices": None,
        "vae_device": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_multiple_named_devices_split_the_transformer(monkeypatch):
    monkeypatch.setattr(COMMON, "device_count", lambda _torch, _kind: 4)
    args = placement_args(devices=["flagos:1", "flagos:3"])

    assert COMMON.resolve_placement(None, args) == (
        "flagos:1",
        ["flagos:1", "flagos:3"],
        "flagos:1",
    )


def test_colocated_rejects_a_full_pipeline_vae_split():
    with pytest.raises(SystemExit, match="is not the encoder's card"):
        COMMON.colocated("flagos:0", "flagos:1")


class Handle:
    def remove(self):
        pass


class Module:
    def __init__(self):
        self.pre_hooks = []
        self.post_hooks = []

    def register_forward_pre_hook(self, hook):
        self.pre_hooks.append(hook)
        return Handle()

    def register_forward_hook(self, hook):
        self.post_hooks.append(hook)
        return Handle()


class VAE(Module):
    def decode(self):
        return "decoded"


class Pipe:
    def __init__(self):
        self.text_encoder = Module()
        self.transformer = Module()
        self.vae = VAE()


def test_phase_timer_includes_first_and_last_denoise_steps(monkeypatch):
    syncs = []
    clock = iter([10.0, 16.0, 17.0, 19.0])
    monkeypatch.setattr(COMMON, "sync", lambda _torch, _kind: syncs.append(True))
    monkeypatch.setattr(COMMON.time, "perf_counter", lambda: next(clock))
    pipe = Pipe()
    original_decode = pipe.vae.decode
    timer = COMMON.PhaseTimer(None, pipe, "flagos")

    timer.reset()
    pipe.transformer.pre_hooks[0](pipe.transformer, ())
    for index in range(3):
        timer.on_step_end(pipe, index, None, {})
    assert pipe.vae.decode() == "decoded"

    phases = timer.sample()
    assert phases["denoise loop"] == pytest.approx(6.0)
    assert phases["loop per step"] == pytest.approx(2.0)
    assert phases["vae decode"] == pytest.approx(2.0)
    assert len(syncs) == 3

    timer.close()
    assert pipe.vae.decode == original_decode


def test_benchmark_supplements_a_single_autorange_call():
    class Caller:
        def __init__(self):
            self.records = ["primary"]

        def measured(self, count):
            values = self.records[-count:]
            return values, values, values

    def measurement(seconds):
        return SimpleNamespace(
            number_per_run=1,
            raw_times=[seconds],
            times=[seconds],
        )

    caller = Caller()
    primary = measurement(61.0)

    def supplement(missing):
        assert missing == 1
        caller.records.append("supplement")
        return measurement(62.0)

    batches = BENCH.ensure_minimum_measured_calls(
        caller, primary, supplement, minimum=2
    )

    assert [batch["runs"] for batch in batches] == [1, 1]
    assert [sample for batch in batches for sample in batch["samples"]] == [
        "primary",
        "supplement",
    ]


def test_benchmark_cli_applies_runtime_environment_before_torch_import(
    monkeypatch, tmp_path
):
    monkeypatch.delenv("TORCH_DEVICE_BACKEND_AUTOLOAD", raising=False)
    monkeypatch.delenv("TRITON_CACHE_DIR", raising=False)
    monkeypatch.delenv("FLAGGEMS_CACHE_DIR", raising=False)
    cache = tmp_path / "cache"
    args = SimpleNamespace(disable_backend_autoload=True, cache_dir=str(cache))

    BENCH.apply_runtime_environment(args)

    assert os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] == "0"
    assert os.environ["TRITON_CACHE_DIR"] == str(cache)
    assert os.environ["FLAGGEMS_CACHE_DIR"] == str(cache)
    assert cache.is_dir()


class FakeMask:
    def __init__(self, all_valid):
        self.all_valid = all_valid

    def detach(self):
        return self

    def to(self, _device):
        return self

    def bool(self):
        return self

    def all(self):
        return self

    def item(self):
        return self.all_valid


@pytest.mark.parametrize("all_valid, expected_is_none", [(True, True), (False, False)])
def test_prompt_mask_elision_preserves_masks_with_padding(all_valid, expected_is_none):
    mask = FakeMask(all_valid)

    class PromptPipe:
        def encode_prompt(self):
            return "embeds", mask, "image-mask"

    pipe = PromptPipe()
    COMMON.enable_all_valid_prompt_mask_elision(pipe)
    COMMON.enable_all_valid_prompt_mask_elision(pipe)

    embeds, returned_mask, image_mask = pipe.encode_prompt()
    assert embeds == "embeds"
    assert (returned_mask is None) is expected_is_none
    assert returned_mask is None or returned_mask is mask
    assert image_mask == "image-mask"


def test_runner_appends_the_census_summary_and_exit_status_to_its_log(tmp_path):
    fake_python = tmp_path / "fake-python"
    fake_python.write_text(
        "#!/bin/sh\n"
        "echo '[flagos dispatch] add.Tensor -> flagos_python'\n"
        "echo '[flagos dispatch] mm -> cuda'\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    output = tmp_path / "run"
    log = output / "infer.log"
    env = os.environ.copy()
    env["PYTHON"] = str(fake_python)

    completed = subprocess.run(
        [
            "bash",
            str(FLOW / "run.sh"),
            "infer",
            "--device",
            "flagos",
            "--run-dir",
            str(output),
            "--log",
            str(log),
        ],
        cwd=FLOW.parents[2],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    saved = log.read_text(encoding="utf-8")
    assert "---- readings ----" in saved
    assert "dispatch records   : 2" in saved
    assert "1 flagos_python" in saved
    assert "1 cuda" in saved
    assert "exit status        : 0" in saved
    assert "dispatch records   : 2" in completed.stdout
