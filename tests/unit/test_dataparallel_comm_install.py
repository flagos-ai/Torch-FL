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

"""Unit coverage for how torch_fl publishes DataParallel's comm layer.

``torch._C`` carries seven comm functions, and stock torch registers all seven
from its CUDA build alone (``torch/csrc/cuda/python_comm.cpp``). A ``+cpu``
wheel therefore has none of them -- which is the torch every platform pipeline
of this project installs, and the reason the first push of this change was red
on CUDA, Ascend and GCU at once: ``InitDataParallelComm()`` read the seven
unconditionally and raised ``AttributeError: module 'torch._C' has no attribute
'_broadcast_coalesced'`` out of ``import torch_fl``.

So there are two configurations to hold, and the difference is decided inside
``InitDataParallelComm()``:

  * a stock implementation exists -- it is replaced, and the replacement
    delegates to it for input that is not flagos;
  * there is none -- the seven are published, and this layer is the only
    implementation of those names, so it serves every call.

Measured in a subprocess, not in this one: the install runs once, at
``import torch_fl``, and is idempotent, so by the time a test body could delete
the seven the decision has already been made and re-running would prove
nothing. Each script takes the seven off ``torch._C`` before the install reads
them, which is both the ``+cpu`` case on a machine that has a CUDA build of torch
and the native case on one that does not.

Where that deletion lands is not free. It has to be after torch is imported --
there is no ``torch._C`` to take anything off before that -- and before
``torch_fl`` looks at it, and on DCU torch_fl has to be imported *first*: its
preload is what puts the DTK device libraries in the process, and importing
torch ahead of it leaves every later device operation dying with "Cannot
initialize CUDA without ATen_cuda library" (the constraint ``_phase_preload``
documents). Importing torch in the script to do the deleting is what took the
roundtrip below down on the DCU pipeline, in run 36261615856. So the deletion
runs from a meta path finder instead, on ``torch._C`` itself: that is the one
point that is both after torch is born and before torch_fl reads it, whichever
order the two are imported in.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The seven names ``torch_fl/csrc/dataparallel_comm.cc`` publishes. torch_npu
#: reaches the same set from its own ``initCommMethods()``.
COMM_NAMES = (
    "_broadcast_coalesced",
    "_broadcast",
    "_broadcast_out",
    "_scatter",
    "_scatter_out",
    "_gather",
    "_gather_out",
)

#: Cold import of torch and torch_fl is ~8 s on the PPU dev host; the budget is
#: for a cold filesystem cache, not for the work this does.
TIMEOUT_SECONDS = 300

#: Hides the stock comm layer without importing torch to do it. The deletion has
#: to land after ``torch._C`` exists and before ``InitDataParallelComm()`` reads
#: it, and on DCU the two imports either side of that window have an order that
#: cannot be broken to make room for it. A meta path finder is the one point
#: that satisfies both, because it runs on ``torch._C`` itself: the imports stay
#: in whatever order the platform needs, and the seven are gone by the time the
#: install looks.
_PREAMBLE = f"""
import importlib.abc
import importlib.util
import sys

NAMES = {list(COMM_NAMES)!r}


class _StripStockCommLayer(importlib.abc.MetaPathFinder):
    # Deletes the seven the moment torch._C is created.
    def find_spec(self, fullname, path=None, target=None):
        if fullname != "torch._C":
            return None
        # One shot: the lookup below walks sys.meta_path, and this finder must
        # not be on it when that happens.
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(fullname)
        if spec is None or spec.loader is None:
            return None
        stock = spec.loader

        class _Loader:
            def create_module(self, spec):
                return stock.create_module(spec)

            def exec_module(self, module):
                stock.exec_module(module)
                for name in NAMES:
                    if hasattr(module, name):
                        delattr(module, name)

        spec.loader = _Loader()
        return spec


sys.meta_path.insert(0, _StripStockCommLayer())

# torch_fl first. Its preload is what loads the device libraries, and importing
# torch ahead of it is what DCU's import gate forbids -- see the module
# docstring. The hook above is what lets that order hold while the seven are
# still absent.
import torch_fl
import torch
"""


def _run(body):
    """Run ``body`` in a fresh interpreter and return (stdout, stderr)."""
    script = _PREAMBLE + textwrap.dedent(body)
    env = dict(os.environ)
    # The repository's torch_fl, not whatever an editable install points at.
    # Subprocesses are the one place that is not already decided by sys.path.
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        timeout=TIMEOUT_SECONDS,
        check=False,
    )
    assert completed.returncode == 0, (
        f"exit {completed.returncode}\n--- stdout ---\n{completed.stdout}"
        f"\n--- stderr ---\n{completed.stderr}"
    )
    return completed.stdout, completed.stderr


def _report(stdout, key):
    """The single ``key:value`` line the script printed."""
    lines = [line for line in stdout.splitlines() if line.startswith(f"{key}:")]
    assert len(lines) == 1, f"expected one {key}: line, got {stdout!r}"
    return lines[0][len(key) + 1 :]


def test_import_publishes_all_seven_when_torch_has_no_comm_layer():
    """``import torch_fl`` must leave all seven names on torch._C.

    The failure this pins is the import itself: reading a name that is not
    there, before reaching any tensor, is what took down the CUDA, Ascend and
    GCU pipelines.
    """
    stdout, _ = _run(
        """
        missing = [name for name in NAMES if not hasattr(torch._C, name)]
        print("MISSING:" + ",".join(missing))
        print("DEVICE_COUNT:" + str(torch.flagos.device_count()))
        """
    )
    assert _report(stdout, "MISSING") == ""
    # The import got all the way through the ecosystem phase, not just past the
    # comm install.
    assert int(_report(stdout, "DEVICE_COUNT")) >= 1


def test_the_published_layer_serves_calls_with_no_stock_implementation():
    """With nothing to delegate to, the layer has to do the work itself.

    Includes a CPU *source* with no scatter scope set: on a torch that has the
    stock functions that call is delegated to them, but when there is no stock
    implementation this is the same path DataParallel's CPU-input forward takes,
    and it has to produce flagos chunks rather than raise.
    """
    stdout, stderr = _run(
        """
        from torch.nn.parallel import comm

        if torch.flagos.device_count() < 2:
            print("ROUNDTRIP:skipped")
        else:
            x = torch.arange(16, dtype=torch.float32).reshape(4, 4)
            chunks = comm.scatter(x, [0, 1], None, 0)
            back = comm.gather(list(chunks), 0, 0)
            ok = (
                [str(c.device) for c in chunks] == ["flagos:0", "flagos:1"]
                and str(back.device) == "flagos:0"
                and torch.equal(back.cpu(), x)
            )
            print("ROUNDTRIP:" + ("ok" if ok else str([str(c.device) for c in chunks])))
        """
    )
    assert _report(stdout, "ROUNDTRIP") in ("ok", "skipped"), stderr
