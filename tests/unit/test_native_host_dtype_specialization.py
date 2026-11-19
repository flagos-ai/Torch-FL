# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Dtype admission may remove only branches of unchanged guarded inputs."""

import importlib
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "dtype_specialization_compiler"
spec = importlib.util.spec_from_file_location(
    PACKAGE,
    ROOT / "torch_fl/native_host/compiler/__init__.py",
    submodule_search_locations=[str(ROOT / "torch_fl/native_host/compiler")],
)
module = importlib.util.module_from_spec(spec)
sys.modules[PACKAGE] = module
spec.loader.exec_module(module)
frontend = importlib.import_module(PACKAGE + ".frontend")


def guarded_host(value):
    if value.is_complex():
        return value + value
    return value


def rebound_host(value, other):
    value = other
    if value.is_complex():
        return value + value
    return value


def helper(value):
    if value.is_complex():
        return value + value
    return value


def helper_host(value):
    return helper(value)


def source(function, dtypes, signature):
    hosts = SimpleNamespace(
        originals={__name__: sys.modules[__name__]}, resolve=lambda fn: fn
    )
    compiler = frontend.Frontend(hosts, SimpleNamespace())
    generated = compiler.source(function, signature, input_dtypes=dtypes)
    assert not compiler.fallback_functions, compiler.fallback_reasons
    return generated


@pytest.mark.parametrize("dtypes", [(), ("ComplexFloat",), ("Float", "ComplexFloat")])
def test_unguarded_or_complex_inputs_keep_the_runtime_branch(dtypes):
    generated = source(
        guarded_host, dtypes, "def f(value: torch.Tensor) -> torch.Tensor: pass"
    )
    assert "value.is_complex()" in generated


def test_guarded_real_input_avoids_unreachable_complex_helpers():
    generated = source(
        guarded_host,
        ("Float", "BFloat16"),
        "def f(value: torch.Tensor) -> torch.Tensor: pass",
    )
    assert "is_complex" not in generated
    assert "value + value" not in generated
    assert "return value" in generated


def test_rebinding_a_guarded_input_keeps_its_runtime_branch():
    generated = source(
        rebound_host,
        ("Float", "BFloat16"),
        "def f(value: torch.Tensor, other: torch.Tensor) -> torch.Tensor: pass",
    )
    assert "value.is_complex()" in generated


def test_input_constraints_do_not_apply_to_an_unconstrained_helper():
    generated = source(
        helper_host,
        ("Float", "BFloat16"),
        "def f(value: torch.Tensor) -> torch.Tensor: pass",
    )
    assert "value.is_complex()" in generated
