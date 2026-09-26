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

"""FlagGems device identity on the CUDA-compatible paths.

FlagGems decides per op between running its own Triton kernel and handing the
call back to ATen by comparing the device type of its inputs against the device
string its vendor backend declares::

    device = _select_device(a, b)
    if device.type != _DEVICE_NAME:
        return torch.ops.aten.mul.Tensor.redispatch(_FALLBACK_KEYSET, a, b)

Its nvidia and hygon descriptors both name that device ``cuda``, while torch_fl
registers the accelerator as ``flagos``. The two never compare equal, so a
module that makes that comparison took the fallback branch for every flagos
input rather than only for the ones it means to exclude.

The consequence differs by vendor, because the guarded modules do not all fall
back. On nvidia the module redispatches to its own aten reference path, which is
what made CUDA CI fail: the ``Tensor`` overload cannot take the Python scalar a
wrapped number is now handed over as, so ``mask * 1.3333333333333333`` raised
"Expected a value of type 'Tensor' for argument 'other' but instead found type
'float'". On hygon the guarded modules raise instead --
``ValueError: i0: input tensor must be on cuda device`` -- so the route is dead
rather than slow, and seven of ``backends_dcu.conf``'s fourteen guarded
``flaggems`` routes raised one on the survey's profiles.

The invariant these tests pin is the one the fix establishes: on a box whose
FlagGems resolved to a vendor the alignment covers, FlagGems' device name equals
the name torch_fl registered, its entry points accept a flagos tensor, and the
routes that guard on that name execute rather than raise. It extends to the
name's other use as well -- FlagGems hands the same global to
``torch.cuda.get_device_properties``, which has to keep accepting it, or the
routes that merely size a grid with it raise instead of running. Every test
carries ``main_ops`` because CI's CUDA and DCU operator jobs select on it, and
the alignment applies whenever FlagGems is importable -- the routing does not
wait for ``FLAGOS_USE_FLAGGEMS`` -- so no backend marker is used. Vendors the
alignment does not cover skip internally, which keeps the file selectable in the
other platforms' runs.

Usage:
    pytest tests/integration/ops/test_flaggems_device_name.py -v
"""

import importlib

import pytest

import torch
import torch_fl  # noqa: F401

DEVICE = "flagos:0"
SEED = 1234


def _aligned_device_detector():
    """FlagGems' DeviceDetector, or a skip when no aligned vendor is in play."""
    flag_gems = pytest.importorskip("flag_gems")
    try:
        from flag_gems.runtime.backend.device_finder import DeviceDetector
    except ImportError:
        from flag_gems.runtime.backend.device import DeviceDetector

    try:
        from torch_fl.accelerator.cuda._cuda_compat import _ALIGNED_VENDORS
    except ImportError:  # pragma: no cover - non CUDA-compatible platform
        pytest.skip("torch_fl has no CUDA-compatible accelerator on this platform")

    detector = DeviceDetector()
    if detector.vendor_name not in _ALIGNED_VENDORS:
        pytest.skip(
            f"FlagGems resolved the {detector.vendor_name} vendor, which "
            f"{_ALIGNED_VENDORS} does not cover"
        )
    return flag_gems, detector


def _registered_device_name():
    """The device name torch_fl registered with PyTorch, after device init."""
    torch_fl.flagos.init()
    return torch._C._get_privateuse1_backend_name()


def _flag_gems_mul():
    """FlagGems' ``mul`` entry point.

    ``flag_gems.ops`` re-exports each op *callable* under its submodule's own
    name, so this is normally the function; fall back to the module attribute if
    a FlagGems build stops shadowing the module.
    """
    entry = importlib.import_module("flag_gems.ops.mul")
    return entry if callable(entry) else entry.mul


def _assert_matches_cpu(produce, reference):
    """Run ``produce`` on the accelerator and compare against ``reference``."""
    torch.manual_seed(SEED)
    result = produce()
    torch.flagos.synchronize()
    torch.testing.assert_close(result.to("cpu"), reference, rtol=1e-4, atol=1e-4)


class TestFlaggemsDeviceName:
    """The name FlagGems uses must be the one torch_fl registered."""

    @pytest.mark.anyplatform
    @pytest.mark.main_ops
    def test_detector_name_matches_registered_backend(self):
        flag_gems, detector = _aligned_device_detector()
        registered = _registered_device_name()
        assert detector.name == registered
        assert flag_gems.device == registered

    @pytest.mark.anyplatform
    @pytest.mark.main_ops
    def test_op_module_copies_are_realigned(self):
        """The modules that capture the name at import time must be fixed too.

        ``DeviceDetector`` is a singleton, so correcting it covers every later
        reader, but the op modules do ``device = device.name`` at module scope
        (``ops/mul.py`` uses the ``_DEVICE_NAME`` spelling) and importing any part
        of ``flag_gems.runtime`` eagerly imports them. Those already hold the old
        literal by the time the alignment can run.
        """
        _aligned_device_detector()
        registered = _registered_device_name()

        # Imported by path: flag_gems.ops re-exports each op *callable* under the
        # submodule's own name, shadowing the module.
        cumsum_mod = importlib.import_module("flag_gems.ops.cumsum")
        mul_mod = importlib.import_module("flag_gems.ops.mul")

        assert cumsum_mod.device == registered
        assert mul_mod._DEVICE_NAME == registered

    @pytest.mark.anyplatform
    @pytest.mark.main_ops
    def test_flag_gems_mul_accepts_a_python_scalar(self):
        """The guard's fallback cannot take the scalar a wrapped number becomes.

        Called on the entry point rather than through ``x * 1.333`` because the
        routed call only reaches the guard with a Python scalar on a build that
        carries ``TensorToPython``'s wrapped-number handling; this spelling
        reproduces the CUDA CI failure on every build. Before the alignment it
        raised ``RuntimeError: aten::mul() Expected a value of type 'Tensor' for
        argument 'other' but instead found type 'float'.``
        """
        _aligned_device_detector()
        torch.manual_seed(SEED)
        value = torch.randn((64, 64), device=DEVICE)
        reference = value.to("cpu") * 1.3333333333333333
        result = _flag_gems_mul()(value, 1.3333333333333333)
        assert torch.allclose(result.to("cpu"), reference)

    @pytest.mark.anyplatform
    @pytest.mark.main_ops
    def test_python_float_multiply_matches_cpu(self):
        """The same case on the routed path, as a user writes it.

        Python-float multiplication always dispatches to ``aten.mul.Tensor``
        with a Python ``float`` as ``other`` -- never to ``mul.Scalar`` -- so it
        is the routed overload, and the one the CUDA CI failure came in on.
        """
        _aligned_device_detector()
        torch.manual_seed(SEED)
        value = torch.randn((64, 64), device=DEVICE)
        reference = value.to("cpu") * 1.3333333333333333
        assert torch.allclose((value * 1.3333333333333333).to("cpu"), reference)


class TestFlaggemsGuardedRoutes:
    """Routes whose FlagGems kernel refuses an operand it cannot recognise.

    Each route below resolves to a module that compares an operand's device type
    against the name FlagGems gave the device and raises when they differ, so the
    kernel's own guard, not a fallback, is what a call meets. Seven of the
    fourteen such routes in ``backends_dcu.conf`` raised on a flagos tensor
    before the alignment -- the four ``reflection_pad`` routes and
    ``_embedding_bag_dense_backward`` are the other five, and are exercised here
    with arguments ATen's schema accepts, which the survey's profiles do not
    build. The ``eq`` pair carries the comparison on a path that a two-operand
    call does not reach and passed either way.
    """

    @pytest.mark.anyplatform
    @pytest.mark.main_ops
    @pytest.mark.parametrize(
        "build",
        [
            pytest.param(lambda t: torch.ops.aten.i0(t), id="i0"),
            pytest.param(
                lambda t: torch.ops.aten.i0.out(t, out=torch.empty_like(t)),
                id="i0.out",
            ),
            pytest.param(lambda t: torch.ops.aten.special_i0e(t), id="special_i0e"),
            pytest.param(lambda t: torch.ops.aten.special_i1(t), id="special_i1"),
            pytest.param(
                lambda t: torch.ops.aten.special_scaled_modified_bessel_k1(t),
                id="special_scaled_modified_bessel_k1",
            ),
            pytest.param(
                lambda t: torch.ops.aten.special_scaled_modified_bessel_k1.out(
                    t, out=torch.empty_like(t)
                ),
                id="special_scaled_modified_bessel_k1.out",
            ),
        ],
    )
    def test_guarded_pointwise_routes_execute(self, build):
        """Positive inputs, because one of these ops is singular below zero.

        ``special_scaled_modified_bessel_k1`` is undefined at ``x <= 0``, where
        ATen itself answers nan, so a comparison against the CPU result only
        carries information on the positive domain. The domain is harmless for
        the rest.
        """
        _aligned_device_detector()
        torch.manual_seed(SEED)
        value = torch.rand((64,), device=DEVICE) * 20.0 + 0.5
        _assert_matches_cpu(lambda: build(value), build(value.to("cpu")))

    @pytest.mark.anyplatform
    @pytest.mark.main_ops
    def test_guarded_soft_margin_loss_route_executes(self):
        _aligned_device_detector()
        torch.manual_seed(SEED)
        value = torch.rand((64,), device=DEVICE) * 20.0 + 0.5
        other = torch.rand((64,), device=DEVICE) * 20.0 + 0.5
        _assert_matches_cpu(
            lambda: torch.ops.aten.soft_margin_loss(value, other),
            torch.ops.aten.soft_margin_loss(value.to("cpu"), other.to("cpu")),
        )

    @pytest.mark.anyplatform
    @pytest.mark.main_ops
    @pytest.mark.parametrize(
        "op,shape,padding",
        [
            pytest.param(
                torch.ops.aten.reflection_pad2d, (2, 3, 8, 8), [1, 1, 1, 1], id="2d"
            ),
            pytest.param(
                torch.ops.aten.reflection_pad3d,
                (2, 3, 4, 8, 8),
                [1, 1, 1, 1, 1, 1],
                id="3d",
            ),
        ],
    )
    def test_guarded_reflection_pad_routes_execute(self, op, shape, padding):
        """The padding arity has to be the one ATen's schema asks for.

        The overload survey derives one padding element from the tensor's rank
        and ATen rejects that on arity before the operator body runs, which is
        why these routes carry no verdict there. Passing a well-formed list is
        what reaches the device guard underneath.
        """
        _aligned_device_detector()
        torch.manual_seed(SEED)
        value = torch.randn(shape, device=DEVICE)
        _assert_matches_cpu(lambda: op(value, padding), op(value.to("cpu"), padding))

    @pytest.mark.anyplatform
    @pytest.mark.main_ops
    @pytest.mark.parametrize(
        "op,shape,padding",
        [
            pytest.param(
                torch.ops.aten.reflection_pad2d,
                (2, 3, 8, 8),
                [1, 1, 1, 1],
                id="2d.out",
            ),
            pytest.param(
                torch.ops.aten.reflection_pad3d,
                (2, 3, 4, 8, 8),
                [1, 1, 1, 1, 1, 1],
                id="3d.out",
            ),
        ],
    )
    def test_guarded_reflection_pad_out_routes_execute(self, op, shape, padding):
        """The ``.out`` spelling, which is a route of its own in the conf.

        Reflection padding widens each of the padded dims by two and pads the
        trailing dims only, one pair of elements per dim.
        """
        _aligned_device_detector()
        torch.manual_seed(SEED)
        value = torch.randn(shape, device=DEVICE)
        padded_dims = len(padding) // 2
        out = torch.empty(
            tuple(shape[:-padded_dims]) + tuple(s + 2 for s in shape[-padded_dims:]),
            device=DEVICE,
        )
        _assert_matches_cpu(
            lambda: op.out(value, padding, out=out),
            op.out(value.to("cpu"), padding, out=out.to("cpu")),
        )

    @pytest.mark.anyplatform
    @pytest.mark.main_ops
    def test_guarded_eq_routes_execute(self):
        """The ``eq`` pair carries the comparison where a two-operand call reaches it."""
        _aligned_device_detector()
        torch.manual_seed(SEED)
        value = torch.rand((64,), device=DEVICE)
        other = torch.rand((64,), device=DEVICE)
        _assert_matches_cpu(
            lambda: torch.ops.aten.eq.Tensor(value, other),
            torch.ops.aten.eq.Tensor(value.to("cpu"), other.to("cpu")),
        )
        _assert_matches_cpu(
            lambda: torch.ops.aten.eq.Scalar(value, 0.5),
            torch.ops.aten.eq.Scalar(value.to("cpu"), 0.5),
        )

    @pytest.mark.anyplatform
    @pytest.mark.main_ops
    def test_guarded_embedding_bag_route_executes(self):
        """``_embedding_bag_dense_backward`` with the argument convention ATen uses.

        ``grad`` is the bag-output gradient, ``(num_bags, D)``; ``bag_size`` and
        ``maximum_indices`` are per bag while ``indices`` and ``offset2bag`` are
        per sample. ``per_sample_weights`` is nullable but carries no default in
        the schema, so it has to be passed explicitly.
        """
        _aligned_device_detector()
        torch.manual_seed(SEED)
        num_weights, dim, num_bags, num_samples = 16, 4, 3, 8
        grad = torch.randn(num_bags, dim, device=DEVICE)
        indices = torch.randint(
            0, num_weights, (num_samples,), device=DEVICE, dtype=torch.int64
        )
        offset2bag = torch.zeros(num_samples, device=DEVICE, dtype=torch.int64)
        bag_size = torch.full(
            (num_bags,), num_samples, device=DEVICE, dtype=torch.int64
        )
        maximum_indices = torch.zeros(num_bags, device=DEVICE, dtype=torch.int64)

        def call(args):
            return torch.ops.aten._embedding_bag_dense_backward(*args)

        host = tuple(
            a.to("cpu") for a in (grad, indices, offset2bag, bag_size, maximum_indices)
        )
        _assert_matches_cpu(
            lambda: call(
                (
                    grad,
                    indices,
                    offset2bag,
                    bag_size,
                    maximum_indices,
                    num_weights,
                    False,
                    0,
                    None,
                )
            ),
            call((*host, num_weights, False, 0, None)),
        )

    @pytest.mark.anyplatform
    @pytest.mark.main_ops
    def test_is_cuda_is_false_on_a_flagos_tensor(self):
        """A second guard class the alignment cannot reach, and must not claim to.

        Some FlagGems modules assert ``x.is_cuda`` instead of comparing against
        the device string. ``is_cuda`` is a property of the tensor, not of the
        name FlagGems gave the device, so it stays false on a PrivateUse1 tensor
        whatever the alignment does; ``special_modified_bessel_k0`` is the DCU
        route that reaches one. Pinned here so the boundary between the two
        classes is recorded rather than assumed -- this test states the tensor
        property, not the op's verdict, so it does not turn an upstream fix into
        a failure.
        """
        _aligned_device_detector()
        torch.manual_seed(SEED)
        value = torch.randn((64,), device=DEVICE)
        assert not value.is_cuda


class TestFlaggemsDeviceSpecifiers:
    """The realignment also reaches a global used as a device *specifier*.

    A module global holding the descriptor's name is not always compared against
    ``tensor.device.type``. ``flag_gems.ops.cumsum`` does both: one branch
    compares the name, another hands the same global to ``get_device_properties``
    to size a grid, and ``torch.cuda`` accepts no name but its own. Rewriting the
    name without touching that lookup therefore traded a dead route for a raising
    one: ``multinomial`` with ``replacement=True`` reaches the grid sizing through
    ``normed_cumsum``, and it raised ``ValueError: Expected a cuda device, but
    got: flagos`` until the lookup was wrapped.

    ``masked_select`` and ``masked_scatter`` are the same lookup with a different
    argument -- they pass ``mask.device``, a flagos *tensor* device, above the
    4096-element single-pass cutoff -- and they raised on DCU before the
    realignment as well, so covering them is a fix the wrapper brings rather than
    a regression it avoids.
    """

    @pytest.mark.anyplatform
    @pytest.mark.main_ops
    @pytest.mark.parametrize(
        "specifier",
        [
            pytest.param("flagos", id="name"),
            pytest.param("flagos:0", id="name-index"),
            pytest.param(torch.device("flagos"), id="device"),
            pytest.param(torch.device("flagos", 0), id="device-index"),
        ],
    )
    def test_device_properties_accepts_the_registered_name(self, specifier):
        _aligned_device_detector()
        reference = torch.cuda.get_device_properties(0)
        resolved = torch.cuda.get_device_properties(specifier)
        assert resolved.name == reference.name
        assert resolved.multi_processor_count == reference.multi_processor_count

    @pytest.mark.anyplatform
    @pytest.mark.main_ops
    def test_multinomial_with_replacement_runs_on_a_second_device(self):
        """The route the survey cannot build, on the device that broke it.

        ``multinomial`` with ``replacement=True`` calls ``normed_cumsum``, which
        is where the grid sizing lives. Device 0 does not show the difference: a
        specifier of index 0 resolves to the current device whichever way it is
        spelled, so the second device is what pins the mapping.
        """
        _aligned_device_detector()
        if torch_fl.flagos.device_count() < 2:
            pytest.skip("needs at least 2 flagos devices")
        torch.manual_seed(SEED)
        drawn = torch.multinomial(
            torch.ones(10, device="flagos:1"), 5, replacement=True
        )
        assert drawn.device == torch.device("flagos", 1)
        assert drawn.numel() == 5

    @pytest.mark.anyplatform
    @pytest.mark.main_ops
    def test_masked_select_past_the_single_pass_cutoff(self):
        """4097 elements, one past the single-pass kernel's cutoff.

        ``flag_gems.ops.masked_select`` takes a single-pass kernel up to 4096
        elements and a multi-pass one above it, and only the second reads
        ``mask.device``. That is why the operator survey records the route as
        STRICT on DCU: its profiles top out at 1536 elements and never reach the
        lookup that raised.
        """
        _aligned_device_detector()
        torch.manual_seed(SEED)
        value = torch.randn(4097, device=DEVICE)
        mask = value > 0.0
        _assert_matches_cpu(
            lambda: torch.masked_select(value, mask),
            torch.masked_select(value.to("cpu"), mask.to("cpu")),
        )
