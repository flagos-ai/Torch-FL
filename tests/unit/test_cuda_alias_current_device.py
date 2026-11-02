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

"""An index-less `cuda` target must mean the current device, not device 0.

`Tensor.cuda()` with no argument lands on `torch.cuda.current_device()` on CUDA,
and `nn.Module.to("cuda")` places parameters on the current device. The alias
pinned both to index 0, so with `world_size == 1` nothing looked wrong and every
rank of a multi-rank job built its tensors on `flagos:0` -- NCCL then rejected
the duplicate device at the first collective (issue #561).

Two sites, reached by different spellings:

* `_tensor_cuda` opened with `idx = 0`, so a bare `Tensor.cuda()` never asked
  what the current device was.
* `_remap`'s `torch.device` branch rewrote an index-less device object to index
  0. The `str` branch immediately above it already left the index to torch,
  which is what made this asymmetric and easy to miss: `torch.zeros(2,
  device="cuda")` was correct while `nn.Module.to("cuda")` was not, because
  `_parse_to` turns its argument into a device object first.

The fix is to pass `None` through in both, letting torch's own device guard
resolve it -- so these tests assert on the *target* each site produces rather
than on a device index, which needs no accelerator and no multi-rank job. The
two functions are closures inside `_alias_cuda_to_flagos` and are extracted by
AST, in the style of tests/unit/test_kunlunxin_triton_stream.py; the structural
check in tests/unit/test_cuda_alias_guard.py parses the same function.

Run: pytest tests/unit/test_cuda_alias_current_device.py -v
"""

from __future__ import annotations

import ast
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_SOURCE = Path(__file__).resolve().parents[2] / "torch_fl" / "__init__.py"
_OUTER = "_alias_cuda_to_flagos"
_UNDER_TEST = ("_remap", "_tensor_cuda")

_FLAGOS = "flagos"


def _load_closures():
    """Compile `_alias_cuda_to_flagos`'s two device-target closures standalone.

    They are nested functions, so `inspect.getsource` on the module attribute
    would hand back the whole 200-line installer. Pulling just these two out
    keeps the test to the device-target logic and avoids running the installer,
    which pushes a permanent torch-function mode and replaces `torch.device`
    process-wide.
    """
    tree = ast.parse(_SOURCE.read_text())
    outer = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == _OUTER
    )
    wanted = [
        node
        for node in ast.walk(outer)
        if isinstance(node, ast.FunctionDef) and node.name in _UNDER_TEST
    ]
    assert {node.name for node in wanted} == set(_UNDER_TEST), (
        f"{_SOURCE}:{_OUTER} no longer defines {_UNDER_TEST}"
    )
    module = types.ModuleType("cuda_alias_closures_under_test")
    module.__dict__.update(
        torch=torch,
        # The names the closures capture from the installer's scope.
        _orig_device=torch.device,
        _flagos_type=_FLAGOS,
        _VENDOR_DEVICE_ALIASES={"mthreads": ("cuda", "musa")},
        _build_accelerator=lambda: "cuda",
    )
    exec(
        compile(ast.Module(body=wanted, type_ignores=[]), str(_SOURCE), "exec"),
        module.__dict__,
    )
    return module


@pytest.fixture(scope="module")
def closures():
    if torch._C._get_privateuse1_backend_name() != _FLAGOS:
        torch.utils.rename_privateuse1_backend(_FLAGOS)
    return _load_closures()


class _ToRecorder:
    """Stands in for a tensor, reporting the target `.cuda()` asked `.to()` for.

    The question is which device string the alias builds, which is answerable
    without a device -- an accelerator would only let the test re-measure what
    torch does with the string afterwards.
    """

    def to(self, target, non_blocking=False):
        return target


def test_a_bare_tensor_cuda_does_not_name_an_index(closures):
    """The defect: `idx = 0` meant every rank placed its tensors on flagos:0."""
    assert closures._tensor_cuda(_ToRecorder()) == _FLAGOS


@pytest.mark.parametrize(
    "device",
    ["cuda", torch.device("cuda")],
    ids=["str", "device"],
)
def test_an_index_less_cuda_argument_does_not_name_an_index(closures, device):
    """Both spellings of "cuda with no index" have to behave the same way.

    The str branch was already correct and the device branch was not, which is
    why `torch.zeros(2, device="cuda")` worked while `nn.Module.to("cuda")` did
    not.
    """
    assert closures._tensor_cuda(_ToRecorder(), device) == _FLAGOS


@pytest.mark.parametrize(
    "device, expected",
    [
        ("cuda:3", f"{_FLAGOS}:3"),
        (torch.device("cuda", 3), f"{_FLAGOS}:3"),
        # index 0 asked for explicitly still means index 0
        ("cuda:0", f"{_FLAGOS}:0"),
        (torch.device("cuda", 0), f"{_FLAGOS}:0"),
        # `Tensor.cuda(1)`, the int form torch.cuda accepts
        (5, f"{_FLAGOS}:5"),
        (0, f"{_FLAGOS}:0"),
    ],
)
def test_an_explicit_index_is_kept(closures, device, expected):
    """The fix must not lose an index that was named."""
    assert closures._tensor_cuda(_ToRecorder(), device) == expected


class TestRemap:
    """`_remap`, which `nn.Module.to("cuda")` reaches through `_parse_to`."""

    def test_an_index_less_device_object_stays_index_less(self, closures):
        """The defect: this returned `device(type='flagos', index=0)`."""
        remapped = closures._remap(torch.device("cuda"))
        assert remapped.type == _FLAGOS
        assert remapped.index is None, (
            "an index-less target means the current device; pinning 0 here is "
            "what put every rank on flagos:0 (issue #561)"
        )

    def test_an_index_less_string_stays_index_less(self, closures):
        """The branch that was already right, pinned so the two cannot diverge."""
        assert closures._remap("cuda") == _FLAGOS

    @pytest.mark.parametrize("index", [0, 2, 7])
    def test_an_explicit_index_is_kept(self, closures, index):
        assert closures._remap(torch.device("cuda", index)).index == index
        assert closures._remap(f"cuda:{index}") == f"{_FLAGOS}:{index}"

    @pytest.mark.parametrize("device", ["cpu", "meta"])
    def test_a_non_cuda_target_is_untouched(self, closures, device):
        assert closures._remap(device) == device
        assert closures._remap(torch.device(device)) == torch.device(device)

    def test_a_vendor_alias_resolves_index_less_too(self, closures):
        """`musa` is the mthreads row's own spelling; same rule applies."""
        closures._build_accelerator = lambda: "mthreads"
        try:
            assert closures._remap("musa") == _FLAGOS
            assert closures._remap("musa:4") == f"{_FLAGOS}:4"
        finally:
            closures._build_accelerator = lambda: "cuda"


def test_neither_site_hardcodes_index_zero():
    """Structural, so the next edit to either site cannot quietly restore it.

    Both regressions were literally `else 0` / `idx = 0`. A behavioural test
    covers the current spelling; this covers the shape, because the two sites
    are far apart in the installer and the next person to touch one may not read
    the other.
    """
    tree = ast.parse(_SOURCE.read_text())
    outer = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == _OUTER
    )
    for name in _UNDER_TEST:
        function = next(
            node
            for node in ast.walk(outer)
            if isinstance(node, ast.FunctionDef) and node.name == name
        )
        # An index fallback reads a device index and substitutes a constant when
        # it is None; `IfExp`/`or` with a literal 0 is how both sites spelled it.
        for node in ast.walk(function):
            if isinstance(node, ast.IfExp):
                assert not (
                    isinstance(node.orelse, ast.Constant) and node.orelse.value == 0
                ), f"{name} defaults a device index to 0: {ast.unparse(node)}"
