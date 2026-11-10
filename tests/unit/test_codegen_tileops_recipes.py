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

"""Cross-generation recipe selection in the TileOPs route generator."""

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "codegen" / "codegen_tileops.py"


def _load():
    spec = importlib.util.spec_from_file_location("_tileops_codegen", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


g = _load()


def _op(*, forward_arity: int):
    namespace = {"__init__": lambda self: None}
    if forward_arity == 1:
        namespace["forward"] = lambda self, x: x
    else:
        namespace["forward"] = lambda self, a, b: a + b
    return type("FakeOp", (), namespace)


def test_new_unary_constructor_is_inferred_from_forward_arity():
    cls = _op(forward_arity=1)

    assert g.infer_recipe("ReluFwdOp", "relu", cls, ("inplace",)) == g.UNARY


def test_new_binary_constructor_is_inferred_from_forward_arity():
    cls = _op(forward_arity=2)

    assert g.infer_recipe("AddFwdOp", "add", cls, ("alpha",)) == g.BINARY


def test_new_reduction_is_distinguished_from_softmax():
    cls = _op(forward_arity=1)

    assert g.infer_recipe("SumFwdOp", "sum", cls, ("dim",)) == g.REDUCE
    assert g.infer_recipe("SoftmaxFwdOp", "_softmax", cls, ("dim",)) == g.SOFTMAX


def test_count_nonzero_uses_reduce_despite_same_new_constructor():
    cls = _op(forward_arity=1)

    assert (
        g.infer_recipe("CountNonzeroFwdOp", "count_nonzero", cls, ("dim",)) == g.REDUCE
    )


def test_manual_ops_are_not_inferred_from_a_new_constructor():
    cls = _op(forward_arity=1)

    assert (
        g.infer_recipe(
            "RMSNormFwdOp",
            "_fused_rms_norm",
            cls,
            ("normalized_shape", "eps"),
        )
        is None
    )
