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

"""Execute the generated FlagGems SDPA probe's host reference on CPU.

The hardware probe imports the accelerator stack and moves every operand to
``flagos:0``. Its argument handling is shared with the host reference that runs
on every platform, so exercise that generated text here instead of checking
only the table values it consumes.
"""

import ast
import sys
from types import SimpleNamespace

import pytest
import torch
from cpu_device_init import INTEGRATION, load_on_cpu


FLAGGEMS = INTEGRATION / "flaggems"
for directory in (FLAGGEMS, INTEGRATION):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))


with load_on_cpu("flaggems/sdpa_caps.py") as sdpa_caps:
    COMMON_BY_TAG = sys.modules["shared_sdpa_caps"].COMMON_BY_TAG
    COMMON_TAGS = sys.modules["shared_sdpa_caps"].COMMON_TAGS
    SDPA_CASES = sdpa_caps.SDPA_CASES
    probe_source = sdpa_caps.probe_source


def _host_reference(monkeypatch):
    """Run the generated helpers and ``host_ref`` without selecting a device."""
    source = probe_source(())
    lines = source.splitlines()
    tree = ast.parse(source)
    helper_end = next(
        node.lineno
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_common"
    )
    device = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "DEVICE"
            for target in node.targets
        )
    )
    generated = [
        line
        for index, line in enumerate(lines[: helper_end - 1], start=1)
        if index not in range(device.lineno, device.end_lineno + 1)
    ]
    interpreter = (
        "from shared_sdpa_caps import COMMON_CASES\n"
        "COMMON_BY_TAG = {case.tag: case for case in COMMON_CASES}\n"
        + "\n".join(generated)
        + "\nDEVICE = 'cpu'\n"
    )
    namespace = {
        "__name__": "generated_flagcicd_probe_host",
        "__file__": str(FLAGGEMS / "sdpa_caps.py"),
        "torch": torch,
    }
    flag_gems = SimpleNamespace(
        scaled_dot_product_attention=torch.nn.functional.scaled_dot_product_attention
    )
    monkeypatch.setitem(sys.modules, "flag_gems", flag_gems)
    with load_on_cpu("shared_sdpa_caps.py") as shared_sdpa_caps:
        monkeypatch.setitem(sys.modules, "shared_sdpa_caps", shared_sdpa_caps)
        exec(
            compile(interpreter, "<generated FlagGems SDPA probe host>", "exec"),
            namespace,
        )
    return namespace


def _probe_case(tag, namespace):
    """Keep the generated keyword expression while shrinking its mask."""
    case = next(case for case in SDPA_CASES if case.tag == tag)
    query = torch.randn(1, 2, 2, 8 if case.build == "square" else 3)
    length = 3 if case.build == "nonsquare" else 2
    key = torch.randn(1, 2, length, query.shape[-1])
    kwargs = eval(
        case.kwargs.replace("1024", "2").replace("1040", "3"),
        {**namespace, "torch": torch},
    )
    return query, key, key, kwargs


def test_generated_host_reference_accepts_boolean_keywords_and_tensor_masks(
    monkeypatch,
):
    namespace = _host_reference(monkeypatch)
    reference = namespace["host_ref"]

    operands = COMMON_BY_TAG["causal"].setup()
    actual = reference(*operands, {"is_causal": True})
    torch.testing.assert_close(actual, COMMON_BY_TAG["causal"].reference(operands))

    for tag in ("mask_full_true", "mask_4d_bool", "mask_float_zeros", "mask_row_f32"):
        query, key, value, kwargs = _probe_case(tag, namespace)
        reference(query, key, value, kwargs)

    for extra in (
        {"is_causal": False},
        {"is_causal": True, "scale": 0.5},
        {"enable_gqa": False},
    ):
        operands = COMMON_BY_TAG["causal"].setup()
        reference(*operands, extra)

    for tag in ("causal_with_allow_mask", "causal_with_neg_inf"):
        common = COMMON_BY_TAG[tag]
        operands = common.setup()
        reference(operands[0], operands[1], operands[2], {"attn_mask": common.mask()})


@pytest.mark.parametrize("tag", COMMON_TAGS)
def test_generated_host_reference_holds_every_common_case(monkeypatch, tag):
    namespace = _host_reference(monkeypatch)
    common = namespace["COMMON_BY_TAG"][tag]
    reference = namespace["host_ref"]
    kwargs = {}
    if common.causal:
        kwargs["is_causal"] = True
    elif common.mask is not None:
        kwargs["attn_mask"] = common.mask()
    actual = reference(*common.setup(), kwargs)
    if common.math:
        assert torch.isfinite(actual).all()
        return
    expected = common.reference(common.setup())
    torch.testing.assert_close(actual, expected)
