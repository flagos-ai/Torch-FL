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

import copy
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


_MODULE_PATH = (
    Path(__file__).parents[2] / "torch_fl" / "provisioning" / "attestation.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "torch_fl_attestation_test", _MODULE_PATH
)
attestation = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(attestation)

_KEY = "a-test-signing-key-that-is-longer-than-thirty-two-bytes"


class _FakeFlagos:
    def __init__(self, name="Hygon DCU Z100"):
        self.name = name

    @staticmethod
    def is_available():
        return True

    @staticmethod
    def device_count():
        return 1

    def get_device_properties(self, index):
        assert index == 0
        return SimpleNamespace(name=self.name, total_memory=64 * 1024**3)

    @staticmethod
    def synchronize(index):
        assert index == 0


class _FakeTensor:
    device = SimpleNamespace(type="flagos")

    def __add__(self, other):
        return self

    def __mul__(self, other):
        return self

    def sum(self):
        return self

    @staticmethod
    def item():
        return 96.0


class _FakeTorch:
    __version__ = "2.9.1"

    @staticmethod
    def ones(*args, **kwargs):
        return _FakeTensor()

    @staticmethod
    def full(*args, **kwargs):
        return _FakeTensor()


def _identity(accelerator="dcu", *, platform_hint=""):
    return attestation.collect_runtime_identity(
        torch_module=_FakeTorch,
        flagos_module=_FakeFlagos(),
        build_accelerator=accelerator,
        platform_hint=platform_hint,
    )


def _canary(**overrides):
    result = {
        "scope": attestation.AUDIT_SCOPE,
        "logical_device": "flagos:0",
        "result_device_type": "flagos",
        "result": 96.0,
        "cpu_fallback_observed": False,
        "host_execution_observed": False,
        "fallback_log_sha256": "0" * 64,
    }
    result.update(overrides)
    return result


def _build(identity=None, canary=None):
    return attestation.build_attestation(
        runtime_identity=identity or _identity(),
        canary=canary or _canary(),
        provisioner="flagos-lab",
        device_model="Hygon DCU Z100",
        architecture="gfx926",
        device_identifier="DCU-serial-0001",
        driver_version="6.3.1",
        vendor_runtime_version="DTK-25.04",
        torch_fl_revision="a" * 40,
        container_image_digest="sha256:" + "b" * 64,
        signing_key=_KEY,
        key_id="flagos-lab-2026-01",
        collected_at="2026-08-25T00:00:00Z",
    )


def test_dcu_runtime_identity_is_domestic():
    identity = _identity()
    assert identity["vendor"] == "hygon"
    assert identity["accelerator_class"] == "domestic_accelerator"
    assert identity["devices"][0]["logical_device"] == "flagos:0"


def test_runtime_identity_uses_selected_physical_properties_api():
    identity = attestation.collect_runtime_identity(
        torch_module=_FakeTorch,
        flagos_module=_FakeFlagos(name="generic flagos name"),
        build_accelerator="dcu",
        device_properties_getter=lambda index: SimpleNamespace(
            name=f"Hygon DCU {index}", total_memory=32 * 1024**3
        ),
    )
    assert identity["devices"][0]["name"] == "Hygon DCU 0"


def test_cuda_runtime_identity_is_reference_not_domestic():
    identity = _identity("cuda")
    assert identity["vendor"] == "nvidia"
    assert identity["accelerator_class"] == "cuda_reference"
    with pytest.raises(attestation.AttestationError, match="not.*domestic"):
        _build(identity=identity)


def test_ppu_requires_torch_fl_platform_signal():
    assert attestation.classify_accelerator("cuda") == ("nvidia", "cuda_reference")
    assert attestation.classify_accelerator("cuda", platform_hint="ppu") == (
        "thead",
        "domestic_accelerator",
    )


def test_signed_payload_matches_flagquantum_contract():
    payload = _build()
    expected_identity = {
        "schema": attestation.SCHEMA,
        "status": "verified",
        "accelerator_class": "domestic_accelerator",
        "provider": "torch_fl",
        "logical_device": "flagos:0",
        "evidence_source": attestation.EVIDENCE_SOURCE,
    }
    assert {key: payload[key] for key in expected_identity} == expected_identity
    assert set(payload["physical_device"]) == {
        "vendor",
        "model",
        "architecture",
        "device_identifier",
        "logical_device_name",
    }
    assert set(payload["software"]) == {
        "driver_version",
        "vendor_runtime_version",
        "torch_fl_revision",
        "container_image_digest",
    }
    route = payload["route_audit"]
    assert route["provider_internal_route_audited"] is True
    assert route["logical_to_physical_device_verified"] is True
    assert route["cpu_fallback_allowed"] is False
    assert route["cpu_fallback_observed"] is False
    assert route["host_execution_observed"] is False
    assert len(route["physical_device_probe_sha256"]) == 64
    assert attestation.verify_attestation_signature(payload, signing_key=_KEY)


def test_signature_rejects_tampering():
    payload = _build()
    tampered = copy.deepcopy(payload)
    tampered["physical_device"]["model"] = "different device"
    assert not attestation.verify_attestation_signature(tampered, signing_key=_KEY)


def test_fallback_observation_blocks_attestation():
    with pytest.raises(attestation.AttestationError, match="CPU fallback"):
        _build(canary=_canary(cpu_fallback_observed=True))


def test_device_route_canary_captures_native_fallback(monkeypatch):
    class FallbackTorch(_FakeTorch):
        @staticmethod
        def full(*args, **kwargs):
            os.write(2, b"[flagos cpu_fallback] aten::full\n")
            return _FakeTensor()

    monkeypatch.setenv("FLAGOS_LOG_FALLBACK", "1")
    result = attestation.run_device_route_canary(
        torch_module=FallbackTorch, flagos_module=_FakeFlagos()
    )
    assert result["cpu_fallback_observed"] is True


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("torch_fl_revision", "short", "40-character"),
        ("container_image_digest", "latest", "immutable"),
        ("architecture", "unknown", "non-placeholder"),
    ],
)
def test_invalid_immutable_or_placeholder_evidence_is_rejected(field, value, message):
    kwargs = {
        "runtime_identity": _identity(),
        "canary": _canary(),
        "provisioner": "flagos-lab",
        "device_model": "Hygon DCU Z100",
        "architecture": "gfx926",
        "device_identifier": "DCU-serial-0001",
        "driver_version": "6.3.1",
        "vendor_runtime_version": "DTK-25.04",
        "torch_fl_revision": "a" * 40,
        "container_image_digest": "sha256:" + "b" * 64,
        "signing_key": _KEY,
        "key_id": "flagos-lab-2026-01",
    }
    kwargs[field] = value
    with pytest.raises(attestation.AttestationError, match=message):
        attestation.build_attestation(**kwargs)
