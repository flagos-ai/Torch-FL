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

"""Unit coverage for the ``index_put_`` route installed on DCU.

``_flagos_index_put_roundtrip`` is the fallback under
``_patch_flaggems_index_put``: whenever the FlagGems kernel raises -- an index
form it does not accept, an ``accumulate`` it mishandles, a flag_gems that is
absent -- the op is still served by this function rather than by an error. That
makes it the correctness-critical half of the patch and the half that no
on-hardware benchmark would notice, so it is checked here against
``Tensor.__setitem__``, which is how the model spells the op and which does not
dispatch through the kernel under test.

These run on any host: the function is device-agnostic under the hood (it copies
to the CPU and scatters there), so a CPU tensor exercises it end to end.

The on-hardware measurements are in ``docs/reference/operator-support.md``.
"""

import pytest
import torch

from torch_fl import _build_accelerator, flagos


def _roundtrip(x, indices, values, accumulate=False):
    return flagos._flagos_index_put_roundtrip(x, indices, values, accumulate)


class TestFallbackMatchesSetitem:
    """The fallback has to agree with the spelling the model uses."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_int_index_on_a_dim(self, dtype):
        x = torch.randn(64, dtype=dtype)
        idx = torch.arange(16, dtype=torch.int64)
        values = torch.randn(16, dtype=dtype)

        ref = x.clone()
        ref[idx] = values

        got = _roundtrip(x.clone(), [idx], values)
        assert torch.equal(got, ref)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_bool_mask(self, dtype):
        x = torch.randn(64, dtype=dtype)
        mask = torch.zeros(64, dtype=torch.bool)
        mask[:16] = True
        values = torch.randn(16, dtype=dtype)

        ref = x.clone()
        ref[mask] = values

        got = _roundtrip(x.clone(), [mask], values)
        assert torch.equal(got, ref)

    def test_none_entry_is_a_full_slice(self):
        # ``x[:, idx] = v`` -- the form Tensor.index_put_ rejects, and the reason
        # the fallback goes through the aten op instead of the method.
        x = torch.randn(4, 64)
        idx = torch.arange(16, dtype=torch.int64)
        values = torch.randn(4, 16)

        ref = x.clone()
        ref[:, idx] = values

        got = _roundtrip(x.clone(), [None, idx], values)
        assert torch.equal(got, ref)

    def test_trailing_none_entries_are_padded(self):
        x = torch.randn(4, 8, 16)
        idx = torch.arange(3, dtype=torch.int64)
        values = torch.randn(3, 8, 16)

        ref = x.clone()
        ref[idx] = values

        got = _roundtrip(x.clone(), [idx], values)
        assert torch.equal(got, ref)

    def test_broadcast_values(self):
        # ``joint_hidden_states[:, image_pad_mask] = hidden_states`` in
        # Qwen-Image-2.1: a bool mask on the middle axis of a rank-3 tensor.
        x = torch.randn(2, 64, 32)
        mask = torch.zeros(64, dtype=torch.bool)
        mask[:16] = True
        values = torch.randn(2, 16, 32)

        ref = x.clone()
        ref[:, mask] = values

        got = _roundtrip(x.clone(), [None, mask, None], values)
        assert torch.equal(got, ref)

    def test_scalar_values(self):
        x = torch.randn(64)
        idx = torch.arange(8, dtype=torch.int64)

        ref = x.clone()
        ref[idx] = 3.0

        got = _roundtrip(x.clone(), [idx], torch.tensor(3.0))
        assert torch.equal(got, ref)

    def test_accumulate_adds_instead_of_replacing(self):
        x = torch.randn(64)
        idx = torch.arange(16, dtype=torch.int64)
        values = torch.randn(16)

        ref = x.clone()
        ref.index_put_([idx], values, True)

        got = _roundtrip(x.clone(), [idx], values, accumulate=True)
        assert torch.equal(got, ref)

    def test_unselected_elements_are_untouched(self):
        x = torch.randn(64)
        idx = torch.arange(16, dtype=torch.int64)
        values = torch.randn(16)

        got = _roundtrip(x.clone(), [idx], values)
        assert torch.equal(got[16:], x[16:])


class TestFallbackReturnsSelf:
    """The op is in-place: the caller keeps the same tensor object."""

    def test_returns_the_input(self):
        x = torch.randn(64)
        got = _roundtrip(x, [torch.arange(8, dtype=torch.int64)], torch.randn(8))
        assert got is x


class TestGate:
    """The patch is scoped to the build and conf it was measured on."""

    def test_does_not_register_off_dcu(self):
        if _build_accelerator() == "dcu":
            pytest.skip("DCU build -- this is the configuration the gate opens on")
        before = len(flagos._PATCH_LIBS)
        flagos._patch_flaggems_index_put()
        assert len(flagos._PATCH_LIBS) == before

    def test_registers_on_dcu(self):
        if _build_accelerator() != "dcu":
            pytest.skip("not a DCU build")
        flag_gems = pytest.importorskip("flag_gems")
        if getattr(flag_gems, "index_put_", None) is None:
            pytest.skip("this flag_gems has no index_put_")
        # The function's own second gate, out here rather than inside the call,
        # so a skip says which of the two conditions was not met.
        conf_routes = flagos.__dict__.get("_conf_routes_to_flaggems")
        if conf_routes is None:
            from torch_fl import _conf_routes_to_flaggems as conf_routes
        if not conf_routes():
            pytest.skip("the selected conf routes nothing to FlagGems")
        # Registering needs no device: torch.library only touches the dispatcher.
        before = len(flagos._PATCH_LIBS)
        flagos._patch_flaggems_index_put()
        assert len(flagos._PATCH_LIBS) == before + 1
