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

"""Diagnostic: which MPSGraph sequence segfaults, isolated per candidate.

The step-3 segfault is localized to the `runAsyncWithMTLCommandQueue:` call on
the *second* MPSGraph executable this process builds, with every argument the
backend passes proven valid by the stage trace, and the faulting frames
entirely inside MetalPerformanceShadersGraph. That leaves the trigger -- what
about the state when the second executable runs -- and the trigger is not
observable from the crashing process itself, because it dies.

So each candidate sequence runs in its own subprocess: a candidate that
segfaults kills only its own child, the parent records the exit status, and one
CI run answers which of the sequences is the one that dies. The candidates
separate the three things that change between the passing `(1,)(1,)` mm and
the dying `(32,64,)` mm -- the operand shape, the fact that it is a second
executable, and an intervening allocation churn.

Diagnostic only -- deleted once it has answered.
"""

import os
import subprocess
import sys
import textwrap

import pytest
import torch_fl  # noqa: F401

pytestmark = pytest.mark.multi_device


def _run(body: str) -> subprocess.CompletedProcess:
    """Run `body` in a fresh interpreter, with the trace disabled.

    A clean environment is deliberate: the trace itself writes to stderr on
    every op, and a candidate has to be judged on whether it dies, not on
    whether its log is readable.
    """
    script = textwrap.dedent(
        """
        import torch, torch_fl
        torch_fl.flagos.init()
        dev = torch.device("flagos", 0)
        """
    ) + textwrap.dedent(body)
    env = dict(os.environ)
    for name in ("FLAGOS_MPS_STAGE_TRACE", "FLAGOS_MPS_CRASH_TRACE"):
        env.pop(name, None)
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )


def _report(name: str, result: subprocess.CompletedProcess) -> None:
    print(
        f"[probe] {name}: returncode={result.returncode}\n"
        f"[probe] {name}: stdout={result.stdout!r}\n"
        f"[probe] {name}: stderr tail={result.stderr[-600:]!r}",
        flush=True,
    )


@pytest.fixture(autouse=True)
def _setup():
    if not torch_fl.flagos.is_available():
        pytest.skip("No flagos device available")


def test_probe_single_mm():
    """One mm, one executable: the shape the passing call has."""
    result = _run(
        """
        a = torch.randn(32, 64, device=dev)
        b = torch.randn(64, 16, device=dev)
        c = torch.mm(a, b)
        assert c.shape == (32, 16)
        print("ok", flush=True)
        """
    )
    _report("single_mm", result)
    assert result.returncode == 0, result.stderr


def test_probe_two_shapes():
    """A tiny mm, then a differently shaped one: two executables."""
    result = _run(
        """
        a = torch.randn(1, 1, device=dev)
        b = torch.randn(1, 1, device=dev)
        c = torch.mm(a, b)
        assert c.shape == (1, 1)
        a2 = torch.randn(32, 64, device=dev)
        b2 = torch.randn(64, 16, device=dev)
        c2 = torch.mm(a2, b2)
        assert c2.shape == (32, 16)
        print("ok", flush=True)
        """
    )
    _report("two_shapes", result)
    assert result.returncode == 0, result.stderr


def test_probe_second_device_like():
    """The CI shape: one (1,)(1,) mm first, then the (32,64) mm."""
    result = _run(
        """
        # The (1,)(1,) mm the collect phase leaves behind.
        a = torch.randn(1, 1, device=dev)
        b = torch.randn(1, 1, device=dev)
        torch.mm(a, b)
        del a, b
        # The allocation churn the allocator tests run between them.
        for size in (128, 1024 * 1024, 1024, 4096):
            x = torch.randn(size, device=dev)
            del x
        a2 = torch.randn(32, 64, device=dev)
        b2 = torch.randn(64, 16, device=dev)
        c2 = torch.mm(a2, b2)
        assert c2.shape == (32, 16)
        print("ok", flush=True)
        """
    )
    _report("second_device_like", result)
    assert result.returncode == 0, result.stderr
