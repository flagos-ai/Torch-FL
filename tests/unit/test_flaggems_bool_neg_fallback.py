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

"""Check that reference dtype errors precede every dispatcher entry point."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _dispatcher_source() -> str:
    return (ROOT / "csrc/aten/dispatcher.h").read_text()


def test_bool_neg_is_rejected_before_backend_resolution():
    source = _dispatcher_source()
    unary = source[source.index("IsBoolAbsOrNeg(") :]
    for name in ("neg", "neg.out", "neg_"):
        assert f'std::strcmp(op_name, "{name}") == 0' in unary
    assert "TORCH_CHECK_NOT_IMPLEMENTED(false" in unary
    assert "Negation, the `-` operator" in unary

    resolve = source[source.index("FnPtr ResolveFn(") : source.index("FnPtr GetFn(")]
    assert "CheckReferenceDtype" not in resolve

    common = (ROOT / "csrc/aten/common.cc").read_text()
    op_dtype = common[common.index("bool FlagGemsRejectsOpDtype(") :]
    op_dtype = op_dtype[: op_dtype.index("\n}\n")]
    assert "kUniversalGaps" not in op_dtype


def test_validation_runs_before_backend_lookup_in_every_entry_point():
    source = _dispatcher_source()
    dispatcher = source[source.index("class Dispatcher") :]
    entries = (
        ("operator()", "GetBackendForOp(op_name_)"),
        ("DispatchBackend(", "GetFn(backend)"),
        ("DispatchAs(", "GetBackendForOp(op_name)"),
    )
    for entry, dispatch in entries:
        start = (
            dispatcher.index("decltype(auto) " + entry)
            if entry == "operator()"
            else dispatcher.index(entry)
        )
        body = dispatcher[start : dispatcher.index("\n  }\n", start)]
        assert "detail::CheckReferenceDtype" in body
        assert body.index("detail::CheckReferenceDtype") < body.index(dispatch)


def test_bool_reductions_keep_the_reference_contract():
    source = _dispatcher_source()
    assert (
        'std::strcmp(op_name, "sum") == 0' in source[source.index("IsFullReduction(") :]
    )
    assert "Expected out tensor to have dtype long long" in source
    assert "Attempting to cast from torch.int64" in source
    assert "invalid dtype for reduction" not in source
