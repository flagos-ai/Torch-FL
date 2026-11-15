# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Registry accounting tests run in the no-Torch CI lane."""

import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = (
    Path(__file__).resolve().parents[2] / "torch_fl/native_host/compiler/coverage.py"
)
SPEC = importlib.util.spec_from_file_location("native_host_coverage", SOURCE)
coverage = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(coverage)


class Type:
    def __init__(self, kind, element=None):
        self._kind, self._element = kind, element

    def kind(self):
        return self._kind

    def getElementType(self):
        return self._element


class Schema:
    def __init__(self, arguments, returns=("TensorType",), overload=""):
        self.arguments = [
            SimpleNamespace(name=name, type=Type(kind)) for name, kind in arguments
        ]
        self.returns = [SimpleNamespace(type=Type(kind)) for kind in returns]
        self.overload_name = overload

    def __str__(self):
        args = ", ".join(f"{a.type.kind()} {a.name}" for a in self.arguments)
        returns = ", ".join(a.type.kind() for a in self.returns)
        return f"test.{self.overload_name}({args}) -> ({returns})"


def tensor_host(value):
    return value


def vendor_host(value):
    if value is None:
        raise ValueError("missing input")
    return value


def scalar_host(value, alpha=1):
    return value * alpha


def unmapped_host(value, extra):
    return value


def default_host(value, shape=(1, 2)):
    return value


def varargs_host(value, *args):
    return value


def kwargs_host(value, **kwargs):
    return value


def keyword_host(value, *, scale=1.0, out=None):
    return value


def discover(function=tensor_host, schema=None, **kwargs):
    schema = schema or Schema([("value", "TensorType")])
    return coverage.discover_variants(
        [("test", function)],
        kwargs.get("original_ops", {function.__name__: function}),
        kwargs.get("names", {function.__name__}),
        kwargs.get("lookup", lambda name: [schema]),
    )


@pytest.mark.parametrize(
    "returns", [(), ("TensorType", "TensorType"), ("ListType",), ("IntType",)]
)
def test_unsupported_returns_remain_in_denominator(returns):
    variants, rows = discover(schema=Schema([("value", "TensorType")], returns))
    assert variants == []
    assert len(rows) == 1
    assert rows[0]["reason"] == "unsupported_return"
    assert rows[0]["operator"] == "aten::test"


@pytest.mark.parametrize("function", [varargs_host, kwargs_host])
def test_variadic_parameters_are_explicit(function):
    variants, rows = discover(function)
    assert variants == []
    assert rows[0]["reason"] == "variadic_parameters"


@pytest.mark.parametrize(
    "function,arguments,reason",
    [
        (unmapped_host, [("value", "TensorType")], "unmapped_parameter"),
        (tensor_host, [("value", "GeneratorType")], "unsupported_parameter_type"),
        (
            default_host,
            [("value", "TensorType"), ("shape", "IntType")],
            "unsupported_default",
        ),
    ],
)
def test_signature_rejections_are_classified(function, arguments, reason):
    variants, rows = discover(function, Schema(arguments))
    assert variants == []
    assert rows[0]["reason"] == reason


def test_missing_schema_host_and_selection_are_distinct():
    cases = [
        ({"lookup": lambda name: []}, "excluded", "missing_schema"),
        ({"original_ops": {}}, "excluded", "missing_host"),
        ({"names": set()}, "not_selected", "not_in_host_bundle"),
    ]
    for kwargs, status, reason in cases:
        variants, rows = discover(**kwargs)
        assert variants == []
        assert (rows[0]["status"], rows[0]["reason"]) == (status, reason)


def test_schema_query_failure_is_a_row():
    def unavailable(name):
        raise RuntimeError("operator not registered")

    _, rows = discover(lookup=unavailable)
    assert rows[0]["reason"] == "schema_lookup_failed"


@pytest.mark.parametrize(
    "method,error,reason",
    [
        ("signature", ValueError, "signature_unavailable"),
        ("getsource", OSError, "source_unavailable"),
    ],
)
def test_uninspectable_callable_keeps_its_record(monkeypatch, method, error, reason):
    def unavailable(function):
        raise error("unavailable")

    monkeypatch.setattr(coverage.inspect, method, unavailable)
    _, rows = discover()
    assert rows[0]["reason"] == reason


def test_schema_aliases_and_duplicate_rows_share_one_compilation():
    variants, rows = coverage.discover_variants(
        [("test", tensor_host), ("alias.Tensor", tensor_host), ("test", tensor_host)],
        {"tensor_host": tensor_host},
        {"tensor_host"},
        lambda name: [
            Schema([("value", "TensorType")]),
            Schema([("value", "TensorType")], overload="Tensor"),
        ],
    )
    assert len(variants) == 1
    assert [r["registry_index"] for r in rows] == [0, 1, 2]
    compiled = [{"name": variants[0][0], "signature": variants[0][2]}]
    report = coverage.coverage_report(rows, compiled, [], cohort={}, complete=True)
    assert report["summary"]["registry_entries"] == 3
    assert report["summary"]["unique_operators"] == 2
    assert report["summary"]["unique_signatures"] == 1
    assert report["summary"]["entries_by_status"] == {"lowered": 3}
    assert not report["native_artifact_built"]


def test_selected_vendor_source_is_identified_without_execution():
    variants, rows = discover(original_ops={"tensor_host": vendor_host})
    assert variants[0][1] is vendor_host
    assert rows[0]["registered_host"].endswith(".tensor_host")
    assert rows[0]["resolved_host"].endswith(".vendor_host")
    assert (
        rows[0]["source_sha256"]
        == hashlib.sha256(inspect.getsource(vendor_host).encode()).hexdigest()
    )


def test_keyword_defaults_and_optional_out_preserve_variant_inputs():
    variants, rows = discover(
        keyword_host, Schema([("value", "TensorType"), ("scale", "FloatType")])
    )
    assert len(variants) == 1
    _, _, signature, public = variants[0]
    assert signature == (
        "def f(value: torch.Tensor, *, scale: float=1.0, "
        "out: Optional[torch.Tensor]=None) -> torch.Tensor: pass"
    )
    assert public[-1] == {
        "name": "out",
        "type": "Optional[torch.Tensor]",
        "keyword": True,
        "has_default": True,
        "default": None,
    }
    assert rows[0]["signatures"] == [signature]


def test_partial_signature_coverage_is_not_full_operator_coverage():
    variants, rows = discover(
        scalar_host, Schema([("value", "TensorType"), ("alpha", "NumberType")])
    )
    assert len(variants) == 3
    results = [{"name": name, "signature": s} for name, _, s, _ in variants]
    report = coverage.coverage_report(
        rows, results[:1], results[1:], cohort={"torch": "test"}, complete=True
    )
    assert report["entries"][0]["status"] == "partial"
    assert report["entries"][0]["signature_outcomes"] == {
        "compiler_fallback": 2,
        "lowered": 1,
    }
    assert report["summary"]["registry_entries"] == 1
    assert report["summary"]["unique_signatures"] == 3
    assert rows[0]["status"] == "pending"
    assert report["cohort"] == {"torch": "test"}


def test_all_fallback_and_pending_have_different_statuses():
    variants, rows = discover()
    results = [{"name": name, "signature": s} for name, _, s, _ in variants]
    pending = coverage.coverage_report(rows, [], [], cohort={})
    failed = coverage.coverage_report(rows, [], results, cohort={}, complete=True)
    assert pending["entries"][0]["status"] == "pending"
    assert pending["summary"]["pending_signatures"] == 1
    assert failed["entries"][0]["status"] == "compiler_fallback"
    assert failed["summary"]["pending_signatures"] == 0


def test_missing_duplicate_and_unknown_results_cannot_claim_completion():
    variants, rows = discover()
    result = {"name": variants[0][0], "signature": variants[0][2]}
    for good, bad, message in [
        ([], [], "missing"),
        ([result], [result], "duplicate"),
        ([{"name": "unknown", "signature": "unknown"}], [], "unexpected"),
    ]:
        with pytest.raises(ValueError, match=message):
            coverage.coverage_report(rows, good, bad, cohort={}, complete=True)
    with pytest.raises(ValueError, match="requires complete"):
        coverage.coverage_report(rows, [result], [], cohort={}, artifact_built=True)


def test_zero_success_report_is_persisted_and_deterministic(tmp_path):
    _, rows = discover(schema=Schema([("value", "TensorType")], ("IntType",)))
    for entries in (rows, []):
        report = coverage.write_coverage(
            tmp_path, entries, [], [], cohort={"torch": "test"}, complete=True
        )
        before = (tmp_path / "coverage.json").read_bytes()
        coverage.write_coverage(
            tmp_path, entries, [], [], cohort={"torch": "test"}, complete=True
        )
        assert (tmp_path / "coverage.json").read_bytes() == before
        assert json.loads(before) == report
        assert report["summary"]["registry_entries"] == len(entries)
        assert report["summary"]["unique_signatures"] == 0
        assert json.loads((tmp_path / "diagnostics.json").read_text()) == {
            "compiled": [],
            "fallback": [],
        }
