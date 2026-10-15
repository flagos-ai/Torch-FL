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

"""What each platform still does not exercise, in one writable list.

Every entry in `.github/integration-manifest.yml` runs on every platform, so a
group that one platform cannot exercise would read as full coverage: the job
runs the command, the command collects the file, and the file skips itself
quietly. This module is the answer -- the skips that remain, per platform,
recorded in one place and printed by the loader on every run, ahead of the
hardware job.

Two halves, and the split is deliberate. The *enforcement* stays in the test or
the conftest that knows the reason, because that is where the platform can
actually be observed (an import that raises, a device that is absent, a
capability row that is false). The *record* is here, because a skip buried in a
test file is exactly the invisible difference this layer exists to remove.
`tests/unit/test_platform_gates.py` holds this file to the manifest and to the
platform list, so a gate naming a step or a platform that does not exist fails
in a unit test rather than printing a line nobody reads.

Nothing here may import torch, pytest or anything else outside the standard
library: `.github/scripts/compose_integration_manifest.py` loads this file by
path in the loader job on `ubuntu-latest`, where no vendor wheel and no
`tests/` package is installed.

`platforms` is the set of platforms the gate *applies to* -- the ones where the
step's target does not run. A gate that applied to every platform would be a
step that belongs in no manifest at all, and one that applied to none would be
no gate.
"""

from typing import NamedTuple


ALL_PLATFORMS: tuple[str, ...] = (
    "ascend",
    "cuda",
    "dcu",
    "gcu",
    "kunlunxin",
    "metax",
    "musa",
    "ppu",
)


class Gate(NamedTuple):
    """One recorded per-platform skip.

    `step` is a manifest id, `target` is the file or the cases the skip covers
    (free text, for the loader's line), `reason` is the measured reason it is
    still there and what would remove it, and `platforms` is where it applies.
    """

    step: str
    target: str
    reason: str
    platforms: tuple[str, ...]


_EVERY_PLATFORM_BUT_MUSA = tuple(p for p in ALL_PLATFORMS if p != "musa")
_EVERY_PLATFORM_BUT_DCU = tuple(p for p in ALL_PLATFORMS if p != "dcu")


_GATES: tuple[Gate, ...] = (
    Gate(
        step="rng",
        target=(
            "ops/test_rng_dispatch.py::TestRngSeedSource, "
            "::TestRngMultiDevice, ::TestRngDistribution, ::TestRngDropout"
        ),
        reason=(
            "MUSA's vendor RNG library murand is broken on this runner: "
            "murandGenerateUniform/Normal fail with LAUNCH_FAILURE and randperm "
            "returns garbage (err 700), so the four classes that draw through "
            "it are gated inside the file. The reproducible-draw class still "
            "runs there. Removing the gate is what fixes it -- "
            "docs/vendors/musa/installation.md records the same suite passing "
            "on an S5000 on 2026-08-17, so this is a driver/toolkit/image "
            "regression, not a torch_fl defect"
        ),
        platforms=_EVERY_PLATFORM_BUT_MUSA,
    ),
    Gate(
        step="vendor_ops",
        target="ops/test_flaggems_device_name.py",
        reason=(
            "Kunlunxin's flagtree XPU stack cannot import flag_gems at all: the "
            "XPU TLE backend raises NotImplementedError for `tle.device_mesh`, "
            "which pytest.importorskip does not catch, so the file skips itself "
            "there. Removing the gate means a flagtree XPU build that imports "
            "flag_gems"
        ),
        platforms=("kunlunxin",),
    ),
    Gate(
        step="musa_dispatch",
        target="ops/test_musa_dispatch.py",
        reason=(
            "the file's cases carry @pytest.mark.musa, which the ops conftest "
            "skips on every other platform: `-m musa` selects nothing off MUSA "
            "by design, because MUSA compiles no CUDA boxing kernels and the "
            "per-op `-> cuda` assertions cannot hold there either. The step is "
            "kept universal so the MUSA dispatch coverage is one shared entry "
            "rather than a MUSA-only variant"
        ),
        platforms=_EVERY_PLATFORM_BUT_MUSA,
    ),
    Gate(
        step="profiler_parity",
        target="test_profiler_parity.py",
        reason=(
            "the comparison is against tests/data/profiler_cuda_baseline.json, "
            "so it is only meaningful where the tracer emits stock-cuda-shaped "
            "records: MUSA's MUPTI reports no gpu_memset category and no "
            "occupancy-derived kernel args (2026-08-29: 5 passed, 2 failed) and "
            "PPU's vendor kernels produce no gpu_memset record and no "
            "'::'-qualified kernel name (3 passed, 1 xpassed, 3 failed). "
            "Ascend, GCU and Kunlunxin gate it through the capability table in "
            "tests/integration/profiler_support.py instead. The structural "
            "contract those platforms do owe is the profiler_contract step"
        ),
        platforms=("ascend", "gcu", "kunlunxin", "metax", "musa", "ppu"),
    ),
    Gate(
        step="torch_compile",
        target="test_compile.py",
        reason=(
            "torch.compile needs a vendor Triton stack, and this MUSA image "
            "ships none: set_env_musa.sh deliberately does not link "
            "flagtree-0.5.0+mthreads3.1, and stock PyPI triton is not a "
            "substitute -- it carries no mthreads backend and cannot compile a "
            "MUSA kernel. The file gates itself on the active Triton lacking "
            "that backend rather than on the platform name, so provisioning the "
            "vendor flagtree removes the gate. The suite is green on an MTT "
            "S5000 with that runtime on PYTHONPATH (2026-08-29: 25 passed, 19 "
            "skipped)"
        ),
        platforms=("musa",),
    ),
    Gate(
        step="soft_lowp_matrix",
        target="ops/test_soft_lowp_gate_dispatch.py",
        reason=(
            "the software low-precision matmul path is compiled for DCU and "
            "MetaX only, and the ops conftest skips the `soft_lowp` marker "
            "elsewhere -- so the command is identical on every platform and the "
            "selection is the capability. Removing the gate means compiling "
            "csrc/aten/backends/soft_lowp for the other six platforms"
        ),
        platforms=("ascend", "cuda", "gcu", "kunlunxin", "musa", "ppu"),
    ),
    Gate(
        step="native_comm_bridge",
        target="test_comm_native_bridge.py",
        reason=(
            "the native RCCL bridge is a DCU artifact: the file carries "
            "`pytestmark = pytest.mark.dcu` and skips itself everywhere else, "
            "because the groups it asserts are the ones the DCU image stages. "
            "The step is kept universal so a platform that grows an equivalent "
            "bridge has one shared entry to extend rather than a private one"
        ),
        platforms=_EVERY_PLATFORM_BUT_DCU,
    ),
)


def all_gates() -> tuple[Gate, ...]:
    """Every recorded gate, in the order the manifest states the steps."""
    return _GATES


def gates_for(platform: str) -> tuple[Gate, ...]:
    """The gates that apply to one platform, in manifest order."""
    return tuple(gate for gate in _GATES if platform in gate.platforms)


GATED_STEPS: frozenset[str] = frozenset(gate.step for gate in _GATES)
