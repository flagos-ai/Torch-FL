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

"""The `cuda` alias must install on a build whose CUDA probe torch_fl rewrote.

`_alias_cuda_to_flagos` returns early when a real CUDA device is present, because
on the CUDA and boxing backends `cuda` already names genuine hardware. The
question it has to answer is therefore "does *torch* report a CUDA device", and
that is not what `torch.cuda.is_available()` answers once torch_fl is imported:
`torch_fl.compile.inductor_backend._patch_native_cuda_probe` repoints that
attribute at the flagos device count so Inductor's CUDA-shaped FakeTensor probe
finds the accelerator on a build without a CUDA runtime.

Which import of `torch_fl.compile` installs that redirect before the guard reads
it depends on the platform: an earlier phase pulls it in on MUSA and the boxing
backends, while on kunlunxin it arrives with the optional-integration step of
`_phase_ecosystem` -- after `_alias_cuda_to_flagos`. On MUSA the redirect was in
place first, which made the guard answer "yes, there is CUDA", the alias never
installed, and every hardcoded `"cuda"` in the ecosystem kept dying in
`torch.cuda._lazy_init` -- including the `torch.cuda.current_device()` call
dynamo's `cuda_extra_check` makes while building compilation metrics, so
`torch.compile(model, fullgraph=True)` could not compile at all. Both orders now
record torch's own probe in that slot, so the guard reads the same answer either
way.

The guard now reads `_real_cuda_is_available`, which answers from the copy
`_patch_native_cuda_probe` saved before overwriting the attribute. The first
test here pins that structure; the rest pin the behaviour, each in a fresh
interpreter because the alias is decided once, at import.
"""

from __future__ import annotations

import ast
import inspect
import json
import os
import subprocess
import sys
import textwrap

import pytest
import torch

torch_fl = pytest.importorskip("torch_fl")


def test_the_guard_asks_the_helper_not_the_probe_it_wrote():
    """Structural: the guard must not read the attribute it is deciding about."""
    function = ast.parse(inspect.getsource(torch_fl._alias_cuda_to_flagos)).body[0]
    guard_tests = [
        ast.unparse(statement.test)
        for statement in function.body
        if isinstance(statement, ast.If)
    ]
    assert "_real_cuda_is_available()" in guard_tests, guard_tests
    assert "torch.cuda.is_available()" not in guard_tests, guard_tests


def test_the_helper_reports_what_torch_said_before_the_redirect(monkeypatch):
    """After the redirect, only the saved copy still answers for torch.

    `torch.cuda.is_available` is left saying `True` throughout: the helper must
    ignore it and report the saved original, whichever way that original went.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    monkeypatch.setattr(
        torch.cuda,
        "_flagos_original_is_available",
        lambda: False,
        raising=False,
    )
    assert torch_fl._real_cuda_is_available() is False

    monkeypatch.setattr(
        torch.cuda,
        "_flagos_original_is_available",
        lambda: True,
        raising=False,
    )
    assert torch_fl._real_cuda_is_available() is True


def test_the_helper_falls_back_when_nothing_redirected(monkeypatch):
    """Without the patch there is no saved copy, and torch's own answer stands."""
    monkeypatch.delattr(torch.cuda, "_flagos_original_is_available", raising=False)
    calls = []

    def is_available():
        calls.append(1)
        return True

    monkeypatch.setattr(torch.cuda, "is_available", is_available)

    assert torch_fl._real_cuda_is_available() is True
    assert calls == [1]


def test_vendor_device_aliases_cover_every_native_vendor():
    """A vendor's own device name must map to flagos on that vendor's build.

    torch_fl never imports the vendor package (torch_musa / torch_txda), so the
    privateuse1 backend is registered as "flagos" and the vendor's name is not a
    valid device type here. That is not cosmetic: FlagTree's TsingMicro Triton
    driver resolves its active device as `torch.device("txda", ...)` while
    Triton's hint manager is imported, and FlagGems' tsingmicro descriptor names
    the device "txda" too, so an unmapped name makes `import flag_gems` raise.
    """
    assert torch_fl._VENDOR_DEVICE_ALIASES["musa"] == ("cuda", "musa")
    assert torch_fl._VENDOR_DEVICE_ALIASES["tsingmicro"] == ("cuda", "txda")
    # Every accelerator with a vendor-private device name is listed; the rest
    # (nvidia/ascend/...) only ever see "cuda" or a name torch already knows.
    for accelerator, aliases in torch_fl._VENDOR_DEVICE_ALIASES.items():
        assert aliases[0] == "cuda", accelerator


_CHILD = textwrap.dedent(
    """
    import json
    import os

    # torch_fl before torch, the same order tests/unit/conftest.py uses and the
    # only order a vendor-core build supports: torch_fl has to select its forked
    # libtorch core before torch starts. Reaching ``import torch`` first maps the
    # upstream libtorch, and PPU's own ``_preload_global`` then aborts the
    # process while static init registers the Conjugate dispatch fallback a
    # second time (run 38723076856, test_cuda_alias_guard).
    import torch_fl
    import torch

    def device_type(dev):
        try:
            return torch.device(dev).type
        except Exception as exc:  # noqa: BLE001
            return f"{type(exc).__name__}: {exc}"

    def probe(fn):
        try:
            return fn().device.type
        except Exception as exc:  # noqa: BLE001
            return f"{type(exc).__name__}: {exc}"

    state = {
        "autoload": os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD"),
        "alias_env": os.environ.get("FLAGOS_ALIAS_CUDA"),
        # Whether this build rewrote torch's CUDA probe at all.
        "redirected": hasattr(torch.cuda, "_flagos_original_is_available"),
        "cuda_is_available": bool(torch.cuda.is_available()),
        "real_cuda": torch_fl._real_cuda_is_available(),
        "alias_active": torch_fl._cuda_alias_active,
        "device_is_wrapped": torch.device is not torch._C.device,
        "synchronize_is_flagos": torch.cuda.synchronize is torch.flagos.synchronize,
        "current_device_is_flagos": (
            torch.cuda.current_device is torch.flagos.current_device
        ),
        "device_cuda": device_type("cuda"),
        "device_cuda_1": device_type("cuda:1"),
        # The vendor's own name for the row this build takes, spelled out here
        # rather than read from _VENDOR_DEVICE_ALIASES: the test has to state
        # what "txda" *should* resolve to, not ask the code under test.
        "vendor_device": device_type(
            {"musa": "musa", "tsingmicro": "txda"}.get(
                torch_fl._build_accelerator(), "cuda"
            )
        ),
        "randn_cuda": probe(lambda: torch.randn(2, device="cuda")),
        "zeros_cuda": probe(lambda: torch.zeros(2).cuda()),
    }
    print(json.dumps(state))
    """
)


def _child_state(alias_env: str) -> dict:
    """Import torch_fl and torch in a fresh interpreter, and report the result.

    `TORCH_DEVICE_BACKEND_AUTOLOAD` is cleared for the same reason
    `tests/manual/transformers_hf_tests.py` clears it for its children: torch_fl
    then runs from an explicit `import torch_fl` rather than from inside
    `import torch`, which is the order the ecosystem reaches it in -- a
    transformers script imports torch_fl last, and MUSA never autoloads at all.

    The child still imports torch_fl *before* torch, because that is what the
    explicit-import path has to look like on a vendor-core build: PPU, DCU and
    MetaX relink their forked libtorch core from `torch_fl/__init__.py`, and an
    upstream libtorch mapped first cannot be replaced afterwards. The alias has
    to install on that path too, which is what these tests assert.
    """
    env = {
        **os.environ,
        "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
        "FLAGOS_ALIAS_CUDA": alias_env,
    }
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("alias_env", ["1", ""])
def test_the_alias_installs_when_the_probe_was_redirected(alias_env):
    """The regression: on MUSA this reached `return` and nothing past it ran."""
    state = _child_state(alias_env)
    if not state["redirected"]:
        pytest.skip("build leaves torch.cuda.is_available alone; nothing to confuse")

    if state["real_cuda"]:
        # Genuine CUDA, so the alias must keep out of the way. Reachable only on
        # a host that both has CUDA and runs a native-accelerator build.
        assert state["alias_active"] is False
        return

    assert state["alias_active"] is True
    assert state["device_is_wrapped"] is True
    # The ecosystem probe still finds the accelerator ...
    assert state["cuda_is_available"] is True
    # ... and the calls it makes on the strength of that answer are flagos's.
    assert state["synchronize_is_flagos"] is True, state
    assert state["current_device_is_flagos"] is True, state
    assert state["device_cuda"] == "flagos", state
    assert state["device_cuda_1"] == "flagos", state
    assert state["randn_cuda"] == "flagos", state
    assert state["zeros_cuda"] == "flagos", state


def test_the_alias_stays_off_when_it_is_opted_out_of():
    state = _child_state("0")
    if not state["redirected"]:
        pytest.skip("build leaves torch.cuda.is_available alone; nothing to confuse")
    assert state["alias_active"] is False, state
    assert state["device_is_wrapped"] is False, state
    assert state["device_cuda"] == "cuda", state


@pytest.mark.parametrize("alias_env", ["1", ""])
def test_the_vendor_device_name_is_aliased_on_a_native_vendor_build(alias_env):
    """A vendor's own device name resolves to flagos, without needing the
    CUDA-probe redirect the test above gates on.

    Asserted from the alias table rather than from a fixed name because this
    suite runs on every platform: only a build with a vendor-private device
    name (musa, tsingmicro) has an alias to check, and on any other build the
    probe is the only alias and this passes trivially.
    """
    state = _child_state(alias_env)
    if not state["alias_active"]:
        # Genuine CUDA, or the alias is opted out: nothing is remapped, so the
        # vendor name is either real or invalid and not this test's business.
        # `alias_active` and not the post-import `real_cuda`: the guard decides
        # during import, and torch_fl repoints `torch.cuda.is_available` at the
        # flagos device count later in the same import, so reading it here
        # answers a question the guard never asked.
        pytest.skip("alias not installed on this host")
    assert state["device_cuda"] == "flagos", state
    if state["vendor_device"] == "cuda":
        pytest.skip("build has no vendor-private device name to alias")
    assert state["vendor_device"] == "flagos", state
