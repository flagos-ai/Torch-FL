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

"""Input shapes that decide whether DCU's pinned routes may be unpinned.

``backends_dcu.conf`` routes ``scatter_add_``, ``argmin``, ``mean``,
``mean.dim`` and ``native_batch_norm`` to the CUDA boxing kernel. Each was
moved off the FlagGems Python route after a same-tensor A/B on DCU in which the
boxing arm matched ATen and the FlagGems arm did not:

* ``scatter_add_`` with an ``index`` built by ``view(...).expand(...)`` --
  logically ``(340, 12)``, physically 340 elements, stride ``(1, 0)``. The
  FlagGems 2-D path walks that index as though it were contiguous, so it
  addresses through logical offset 4079 and mismatches all 120 output elements
  by up to 284.0. On another allocation the same overrun crosses an
  inaccessible page and aborts with an HSA VMFault. FlagGems issue #6448; the
  same defect is filed upstream as #6017.
* ``argmin`` over a 20-column reduction. The kernel runs 32 lanes and does not
  mask the tail, so lanes the row does not have take part in the reduction and
  the answer comes back ``20`` -- an index the tensor does not have. FlagGems
  issue #6447.
* ``mean`` and ``mean.dim`` on an empty reduction, where the division by zero
  inside the kernel escapes as a Python ``ZeroDivisionError`` instead of ATen's
  ``nan``. FlagGems issue #6333.
* ``native_batch_norm``, for a different reason: the FlagGems route cannot be
  entered at all rather than answering wrongly. The generated kernel calls
  ``flag_gems.native_batch_norm`` by name at run time, and the pinned cohort
  exports ``batch_norm``, ``batch_norm_backward`` and ``_batch_norm_no_update``
  but not that name, so every call -- training or eval, any shape -- raises
  ``AttributeError`` where ATen returns a tensor. Issue #295, mirroring FlagGems
  #6332.

The first three are invisible to ``flaggems_overload_survey.py``: its profiles
are contiguous, power-of-two and non-empty, so it records all four routes
``STRICT`` while they are unusable. It does not separate the fourth either,
recording ``native_batch_norm`` ``FAILED`` before the pin and after it. The
shapes below are the missing ones.

None of the cases reads the routing table, deliberately. They assert that these
shapes produce ATen's answer on DCU, which is a property of the *platform* and
not of whichever kernel happens to serve it -- so unpinning any of the four
without an upstream fix fails this file rather than silently retiring it, and
the A/B that motivated the pin is reproducible by setting
``FLAGOS_OP_<op>=flaggems`` and watching the same cases fail.

Usage:
    pytest tests/integration/ops/test_dcu_flaggems_pinned_routes.py -v
"""

import pytest

import torch
import torch_fl  # noqa: F401


DEVICE = "flagos:0"

MOMENTUM, EPS = 0.1, 1e-5

# Issue #295's tensor, kept verbatim so the case below is the reported one
# rather than a shape chosen here.
VALUES = (
    (-3.0, 0.5, 8.0, 1.0),
    (-1.0, 2.0, 4.0, -2.0),
    (2.0, -4.0, 0.0, 3.0),
    (7.0, 1.5, -2.0, 6.0),
    (0.0, 9.0, 3.0, -5.0),
    (4.0, -1.0, 6.0, 2.0),
)


class TestScatterAddExpandedIndex:
    """``scatter_add_`` over a stride-zero expanded index.

    ``expand`` produces exactly this layout, so ATen's contract is the
    reference and not a convenience the kernel may narrow. The backing storage
    is the index's own logical size, which makes the overrun read defined
    memory: a stride-ignorant kernel then returns a deterministic wrong answer
    rather than faulting, which is what makes the case assertable at all.
    """

    ROWS, COLUMNS, OUTPUT_ROWS = 340, 12, 10

    def _build(self, contiguous_index: bool):
        rows, columns, output_rows = self.ROWS, self.COLUMNS, self.OUTPUT_ROWS

        output = torch.zeros((output_rows, columns), dtype=torch.float32, device=DEVICE)
        source = torch.ones((rows, columns), dtype=torch.float32, device=DEVICE)

        storage = torch.zeros(rows * columns, dtype=torch.int64, device=DEVICE)
        storage[:rows] = torch.arange(rows, dtype=torch.int64, device=DEVICE)
        storage[:rows] %= output_rows
        if contiguous_index:
            index = storage[:rows].view(rows, 1).repeat(1, columns)
        else:
            index = storage[:rows].view(rows, 1).expand(rows, columns)

        reference_output = torch.zeros((output_rows, columns), dtype=torch.float32)
        reference_source = torch.ones((rows, columns), dtype=torch.float32)
        reference_index = (torch.arange(rows, dtype=torch.int64) % output_rows).view(
            rows, 1
        )
        reference_index = (
            reference_index.repeat(1, columns)
            if contiguous_index
            else reference_index.expand(rows, columns)
        )
        reference = reference_output.scatter_add_(0, reference_index, reference_source)
        return output, source, index, reference

    @pytest.mark.dcu
    def test_expanded_index_matches_cpu(self):
        output, source, index, reference = self._build(contiguous_index=False)
        assert index.shape == (self.ROWS, self.COLUMNS)
        assert index.stride() == (1, 0)

        result = output.scatter_add_(0, index, source)
        torch.flagos.synchronize()
        torch.testing.assert_close(result.cpu(), reference)

    @pytest.mark.dcu
    def test_contiguous_index_matches_cpu(self):
        """The control: the same shape with the index materialized.

        A stride-ignorant 2-D path is already correct here, so a case that only
        ever ran the expanded index would not distinguish a kernel taught about
        strides from one that merely got slower.
        """
        output, source, index, reference = self._build(contiguous_index=True)
        assert index.stride() == (self.COLUMNS, 1)

        result = output.scatter_add_(0, index, source)
        torch.flagos.synchronize()
        torch.testing.assert_close(result.cpu(), reference)


class TestArgminReductionTail:
    """``argmin`` over a reduction width that is not a power of two."""

    ROWS, COLUMNS, ROUNDING = 32, 20, 32

    def _build(self):
        """The measured tensor: a ``(32, 20)`` view over 32-element rows.

        The backing storage is ``ROWS * ROUNDING`` long while the view strides
        by ``COLUMNS``, so the view covers offsets 0..639 of a 1024-element
        tensor and the lanes a 32-wide kernel reads past a row's last column
        are the next row's first columns -- defined values this constructor
        chose, not whatever the allocator put after the tensor.
        """
        rows, columns, rounding = self.ROWS, self.COLUMNS, self.ROUNDING

        storage = torch.full((rows * rounding,), 1000.0, dtype=torch.float32)
        values = storage.as_strided((rows, columns), (columns, 1))
        values[:, columns - 1] = 1.0
        values[1:, :12] = -1000.0
        return values

    @pytest.mark.dcu
    def test_tail_lanes_do_not_reach_the_result(self):
        values = self._build()
        reference = torch.argmin(values, dim=1)
        result = torch.argmin(values.to(DEVICE), dim=1)
        torch.flagos.synchronize()
        actual = result.cpu()

        # Row 0 is the one that detects the defect: its own minimum is 1.0 at
        # column 19, while the twelve lanes past it -- row 1's first columns --
        # hold -1000.0, so a kernel that reduces all 32 lanes answers 20. Rows
        # 1..31 already contain -1000.0 at lane 0, so the same over-read costs
        # them nothing and the assertion has to be on the whole vector.
        assert reference.tolist() == [self.COLUMNS - 1] + [0] * (self.ROWS - 1)
        assert (actual < self.COLUMNS).all(), (
            f"argmin returned an index outside [0, {self.COLUMNS - 1}]: "
            f"{actual.tolist()}"
        )
        assert actual.tolist() == reference.tolist()

    @pytest.mark.dcu
    def test_power_of_two_width_is_unaffected(self):
        """The control: a width the lane count divides exactly.

        It pins that the case above is about the tail rather than about
        ``argmin`` on the device in general.
        """
        values = torch.arange(self.ROUNDING, 0, -1, dtype=torch.float32).repeat(8, 1)
        values[:, self.ROUNDING - 1] = -1.0
        assert torch.argmin(values, dim=1).tolist() == [self.ROUNDING - 1] * 8

        result = torch.argmin(values.to(DEVICE), dim=1)
        torch.flagos.synchronize()
        assert result.cpu().tolist() == [self.ROUNDING - 1] * 8


class TestEmptyMeanIsNan:
    """Empty reductions, where ATen answers ``nan``."""

    @pytest.mark.dcu
    def test_scalar_mean_of_an_empty_tensor(self):
        empty = torch.empty((0,), device=DEVICE)
        torch.flagos.synchronize()
        assert torch.isnan(torch.mean(empty).cpu())

    @pytest.mark.dcu
    def test_dim_mean_with_an_empty_reduced_dim(self):
        empty = torch.empty((2, 0), device=DEVICE)
        result = torch.mean(empty, dim=1)
        torch.flagos.synchronize()
        assert result.shape == (2,)
        assert torch.isnan(result.cpu()).all()


class TestNativeBatchNormTrainingMode:
    """``aten.native_batch_norm`` in training mode, the case issue #295 names.

    The report's claims are that the failure is not restricted to
    ``affine=False`` and that the input tensor comes back modified, so the
    cases below cover both affine settings, running statistics absent and
    present, and the buffer update the op owes a caller that hands it running
    statistics at all.
    """

    WEIGHT = torch.ones(4) * 0.5
    BIAS = torch.zeros(4) + 0.25
    RUNNING_MEAN = torch.tensor([0.5, -1.0, 2.0, 0.0])
    RUNNING_VAR = torch.tensor([1.5, 0.5, 3.0, 1.0])

    def _build(self, *args):
        """ATen's answer to the call, plus device copies of the same inputs.

        Both arms start from the same CPU tensors, so the reference is ATen's
        answer to the identical call rather than a formula derived here, and
        what the op writes back -- to its input, or to the running statistics
        it updates in place -- lands on copies of its own.  ``cpu_args`` is
        returned after the reference call, so it carries ATen's buffer update;
        ``device_args`` is what the device call consumes.
        """
        cpu_args = [None if t is None else t.clone() for t in args]
        reference = torch.ops.aten.native_batch_norm.default(
            torch.tensor(VALUES), *cpu_args, True, MOMENTUM, EPS
        )
        device_args = [None if t is None else t.to(DEVICE) for t in args]
        return reference, cpu_args, device_args

    def _run(self, device_args):
        """The device call, its output wrappers unpacked by the caller."""
        x = torch.tensor(VALUES, device=DEVICE)
        before = x.cpu().clone()
        result = torch.ops.aten.native_batch_norm.default(
            x, *device_args, True, MOMENTUM, EPS
        )
        torch.flagos.synchronize()
        return result, x, before

    @pytest.mark.dcu
    def test_running_statistics_absent_matches_cpu(self):
        """The reported case: training with no running statistics at all.

        The mutation assertion is not decoration.  It is the second half of
        what was reported, and a kernel that returns the right tensor while
        writing to its input is still the defect -- which is why the check is
        ``torch.equal`` against the bytes the tensor was built with and not a
        tolerance on the output.
        """
        reference, _, device_args = self._build(None, None, None, None)
        result, x, before = self._run(device_args)

        assert torch.equal(x.cpu(), before), (
            "training-mode batch norm modified its input; issue #295 reports the "
            "FlagGems route mutating the tensor it was handed"
        )
        torch.testing.assert_close(result[0].cpu(), reference[0])

    @pytest.mark.dcu
    def test_affine_without_running_statistics_matches_cpu(self):
        """The report's second shape: weight/bias given, statistics not."""
        reference, _, device_args = self._build(self.WEIGHT, self.BIAS, None, None)
        result, _, _ = self._run(device_args)

        torch.testing.assert_close(result[0].cpu(), reference[0])

    @pytest.mark.dcu
    def test_running_statistics_are_updated_in_place(self):
        """Training with running statistics present updates them.

        The returned batch mean and inverse standard deviation are asserted
        with the output because they are the two intermediates the update is
        built from: a route that computes the output some other way and leaves
        the statistics alone would satisfy the first assertion only.
        """
        reference, cpu_args, device_args = self._build(
            self.WEIGHT, self.BIAS, self.RUNNING_MEAN, self.RUNNING_VAR
        )
        result, _, _ = self._run(device_args)

        for actual, expected in zip(result, reference):
            torch.testing.assert_close(actual.cpu(), expected)
        torch.testing.assert_close(device_args[2].cpu(), cpu_args[2])
        torch.testing.assert_close(device_args[3].cpu(), cpu_args[3])


class TestBatchNorm2dMatchesCpu:
    """The module form of the same call, which is how a user reaches it.

    ``nn.BatchNorm2d`` is what a model holds, and its forward arrives at
    ``native_batch_norm`` through the same route, so a route that cannot be
    entered breaks the model and not only the raw op.  Both modes are covered:
    training is the reported one, and eval is the pairing that keeps the claim
    about the route rather than about one mode of it.
    """

    @pytest.mark.dcu
    @pytest.mark.parametrize("training", [True, False], ids=["train", "eval"])
    def test_module_matches_cpu(self, training):
        torch.manual_seed(0)
        reference_module = torch.nn.BatchNorm2d(4)
        source = torch.randn(2, 4, 8, 8)
        reference_module.train(training)
        with torch.no_grad():
            reference = reference_module(source.clone())

        module = torch.nn.BatchNorm2d(4)
        module.load_state_dict(reference_module.state_dict())
        module.to(DEVICE)
        module.train(training)

        x = source.to(DEVICE)
        before = x.cpu().clone()
        output = module(x)
        torch.flagos.synchronize()

        assert torch.equal(x.cpu(), before)
        torch.testing.assert_close(output.cpu(), reference)


class TestTheRestOfTheFamilyIsNotPinned:
    """The sibling routes the same pin deliberately left on FlagGems.

    Pinning ``native_batch_norm`` to CUDA boxing says nothing about the rest of
    the family, and the family cannot be pinned with it: the CUDA boxing kernel
    for ``_batch_norm_no_update`` refuses the device tensor it is handed
    (``Expected tensor to have CPU Backend``) and only FlagGems answers there,
    while ``native_batch_norm_backward`` is correct on both routes.  A file
    that asserted no more than "batch norm works on DCU" would stay green under
    a pin that widened to the whole family, which is why these two are cases
    rather than a note.
    """

    WEIGHT = torch.ones(4) * 0.5
    BIAS = torch.zeros(4) + 0.25
    RUNNING_MEAN = torch.tensor([0.5, -1.0, 2.0, 0.0])
    RUNNING_VAR = torch.tensor([1.5, 0.5, 3.0, 1.0])

    @pytest.mark.dcu
    def test_batch_norm_no_update_matches_cpu(self):
        reference = torch.ops.aten._batch_norm_no_update.default(
            torch.tensor(VALUES),
            self.WEIGHT.clone(),
            self.BIAS.clone(),
            self.RUNNING_MEAN.clone(),
            self.RUNNING_VAR.clone(),
            MOMENTUM,
            EPS,
        )
        result = torch.ops.aten._batch_norm_no_update.default(
            torch.tensor(VALUES, device=DEVICE),
            self.WEIGHT.to(DEVICE),
            self.BIAS.to(DEVICE),
            self.RUNNING_MEAN.to(DEVICE),
            self.RUNNING_VAR.to(DEVICE),
            MOMENTUM,
            EPS,
        )
        torch.flagos.synchronize()

        torch.testing.assert_close(result[0].cpu(), reference[0])

    @pytest.mark.dcu
    def test_native_batch_norm_backward_matches_cpu(self):
        torch.manual_seed(0)
        x = torch.tensor(VALUES)
        grad = torch.randn_like(x)
        mean = x.mean(dim=0)
        invstd = (self.RUNNING_VAR + EPS).rsqrt()

        reference = torch.ops.aten.native_batch_norm_backward.default(
            grad,
            x,
            self.WEIGHT.clone(),
            self.RUNNING_MEAN.clone(),
            self.RUNNING_VAR.clone(),
            mean,
            invstd,
            True,
            EPS,
            (True, True, True),
        )
        result = torch.ops.aten.native_batch_norm_backward.default(
            grad.to(DEVICE),
            x.to(DEVICE),
            self.WEIGHT.to(DEVICE),
            self.RUNNING_MEAN.to(DEVICE),
            self.RUNNING_VAR.to(DEVICE),
            mean.to(DEVICE),
            invstd.to(DEVICE),
            True,
            EPS,
            (True, True, True),
        )
        torch.flagos.synchronize()

        for actual, expected in zip(result, reference):
            torch.testing.assert_close(actual.cpu(), expected)
