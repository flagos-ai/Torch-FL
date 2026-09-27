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

"""Input shapes that decide whether DCU's four pinned routes may be unpinned.

``backends_dcu.conf`` routes ``scatter_add_``, ``argmin``, ``mean`` and
``mean.dim`` to the CUDA boxing kernel. Each of the four was moved off the
FlagGems Python route after a same-tensor A/B on DCU in which the boxing arm
matched ATen and the FlagGems arm did not:

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

The three are invisible to ``flaggems_overload_survey.py``: its profiles are
contiguous, power-of-two and non-empty, so it records all four routes
``STRICT`` while they are unusable. The shapes below are the missing ones.

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
