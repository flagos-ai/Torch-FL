# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Compare the host promotion cache with Torch without loading a device backend."""

import ast
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from torch._prims_common import (  # noqa: E402
    ELEMENTWISE_TYPE_PROMOTION_KIND as Kind,
    elementwise_dtypes,
)


@pytest.fixture
def promotion():
    path = Path(__file__).resolve().parents[2] / "torch_fl/compat/flaggems/launch.py"
    function = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "_memoized_type_promotion"
    )
    namespace = {"torch": torch}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"),
        namespace,
    )

    def original(*args, type_promotion):
        return elementwise_dtypes(*args, type_promotion_kind=type_promotion)

    return namespace["_memoized_type_promotion"](original)


def check(promotion, args, kind=Kind.DEFAULT):
    assert promotion(*args, type_promotion=kind) == elementwise_dtypes(
        *args, type_promotion_kind=kind
    )


def test_scalar_tensor_rank_retains_torch_precedence(promotion):
    value = torch.ones(7, dtype=torch.float16)
    for shape in ((), (1,), (), (7,)):
        check(promotion, (value, torch.ones(shape, dtype=torch.float32)))


def test_equal_python_values_keep_their_scalar_types(promotion):
    value = torch.ones(3, dtype=torch.int32)
    for scalar in (True, 1, 1.0, 1.0 + 0j, 1):
        check(promotion, (value, scalar))


def test_default_dtype_changes_invalidate_promotion(promotion):
    value = torch.ones(3, dtype=torch.int32)
    previous = torch.get_default_dtype()
    try:
        for dtype in (torch.float32, torch.float64, torch.float32):
            torch.set_default_dtype(dtype)
            check(promotion, (value,), Kind.INT_TO_FLOAT)
            check(promotion, (value, 1.0))
    finally:
        torch.set_default_dtype(previous)


def test_promotion_kind_and_input_dtype_remain_independent(promotion):
    for dtype in (torch.int32, torch.float32, torch.int32):
        for kind in (Kind.DEFAULT, Kind.INT_TO_FLOAT, Kind.ALWAYS_BOOL):
            check(promotion, (torch.ones(3, dtype=dtype),), kind)
