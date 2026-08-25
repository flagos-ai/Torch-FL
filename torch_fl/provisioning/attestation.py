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

"""Issue a signed, fail-closed FlagQuantum device-route attestation.

This module deliberately keeps policy in Torch-FL, the component that owns the
``flagos`` logical-to-physical route. Importing it does not import PyTorch, which
also makes the classification and signing policy independently testable.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import hashlib
import hmac
import json
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Iterator, Mapping


SCHEMA = "flagquantum_domestic_single_card_attestation_v1"
EVIDENCE_SOURCE = "provisioner_owned_runtime_probe"
FALLBACK_MARKER = "[flagos cpu_fallback]"
AUDIT_SCOPE = "torch_fl_device_route_canary_v1"

_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_IMAGE_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}")
_PLACEHOLDER_TOKENS = ("replace", "placeholder", "unknown", "todo", "tbd")

_ACCELERATOR_IDENTITIES = {
    "ascend": ("huawei", "domestic_accelerator"),
    "bpu": ("d-robotics", "domestic_accelerator"),
    "dcu": ("hygon", "domestic_accelerator"),
    "gcu": ("enflame", "domestic_accelerator"),
    "metax": ("metax", "domestic_accelerator"),
    "musa": ("moore_threads", "domestic_accelerator"),
    "tsingmicro": ("tsingmicro", "domestic_accelerator"),
}


class AttestationError(RuntimeError):
    """Raised when a runtime cannot produce verified attestation evidence."""


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _non_placeholder(name: str, value: str) -> str:
    normalized = str(value).strip()
    if not normalized or any(
        token in normalized.lower() for token in _PLACEHOLDER_TOKENS
    ):
        raise AttestationError(f"{name} must be a concrete, non-placeholder value")
    return normalized


def classify_accelerator(
    build_accelerator: str, *, platform_hint: str = ""
) -> tuple[str, str]:
    """Return ``(vendor, accelerator_class)`` from Torch-FL-owned signals.

    PPU uses a CUDA-compatible libtorch build, so it must additionally carry
    Torch-FL's PPU platform signal. An arbitrary vendor environment variable is
    intentionally not sufficient to turn a normal CUDA build into a domestic
    accelerator.
    """

    accelerator = str(build_accelerator).strip().lower()
    platform = str(platform_hint).strip().lower()
    if platform == "ppu":
        if accelerator not in ("", "cuda"):
            return "unknown", "unknown"
        return "thead", "domestic_accelerator"
    if accelerator == "cuda":
        return "nvidia", "cuda_reference"
    return _ACCELERATOR_IDENTITIES.get(accelerator, ("unknown", "unknown"))


def collect_runtime_identity(
    *,
    torch_module: Any,
    flagos_module: Any,
    build_accelerator: str,
    platform_hint: str = "",
    device_properties_getter: Any | None = None,
) -> dict[str, Any]:
    """Collect the self-reported Torch-FL runtime identity without running ops."""

    vendor, accelerator_class = classify_accelerator(
        build_accelerator, platform_hint=platform_hint
    )
    count = int(flagos_module.device_count())
    get_properties = device_properties_getter or flagos_module.get_device_properties
    devices = []
    for index in range(count):
        properties = get_properties(index)
        devices.append(
            {
                "index": index,
                "logical_device": f"flagos:{index}",
                "name": str(getattr(properties, "name", "")),
                "total_memory": int(getattr(properties, "total_memory", 0)),
            }
        )
    version = getattr(torch_module, "__version__", "")
    return {
        "provider": "torch_fl",
        "device_type": "flagos",
        "build_accelerator": str(build_accelerator).strip().lower(),
        "platform_hint": str(platform_hint).strip().lower(),
        "vendor": vendor,
        "accelerator_class": accelerator_class,
        "torch_version": str(version),
        "device_count": count,
        "devices": devices,
    }


@contextlib.contextmanager
def _capture_process_stderr() -> Iterator[BinaryIO]:
    """Capture Python and native stderr emitted through file descriptor 2."""

    sys.stderr.flush()
    saved_fd = os.dup(2)
    with tempfile.TemporaryFile(mode="w+b") as stream:
        try:
            os.dup2(stream.fileno(), 2)
            yield stream
        finally:
            try:
                ctypes.CDLL(None).fflush(None)
            except (AttributeError, OSError):
                pass
            sys.stderr.flush()
            os.dup2(saved_fd, 2)
            os.close(saved_fd)


def run_device_route_canary(
    *, torch_module: Any, flagos_module: Any, device_index: int = 0
) -> dict[str, Any]:
    """Run a small operator chain and record whether it left the device route."""

    if os.environ.get("FLAGOS_LOG_FALLBACK") in (None, "", "0"):
        raise AttestationError(
            "FLAGOS_LOG_FALLBACK must be enabled before running the route canary"
        )
    if device_index != 0:
        raise AttestationError(f"{SCHEMA} currently certifies exactly flagos:0")
    if not flagos_module.is_available() or int(flagos_module.device_count()) <= 0:
        raise AttestationError("no flagos device is available")

    logical_device = f"flagos:{device_index}"
    with _capture_process_stderr() as captured:
        left = torch_module.ones(16, device=logical_device)
        right = torch_module.full((16,), 2.0, device=logical_device)
        result = ((left + right) * right).sum()
        flagos_module.synchronize(device_index)
        device_type = str(result.device.type)
        scalar = float(result.item())
        try:
            ctypes.CDLL(None).fflush(None)
        except (AttributeError, OSError):
            pass
        sys.stderr.flush()
        captured.seek(0)
        fallback_log = captured.read().decode("utf-8", errors="replace")

    fallback_observed = FALLBACK_MARKER in fallback_log
    host_observed = device_type not in ("flagos", "privateuseone")
    return {
        "scope": AUDIT_SCOPE,
        "logical_device": logical_device,
        "result_device_type": device_type,
        "result": scalar,
        "cpu_fallback_observed": fallback_observed,
        "host_execution_observed": host_observed,
        "fallback_log_sha256": hashlib.sha256(fallback_log.encode()).hexdigest(),
    }


def build_attestation(
    *,
    runtime_identity: Mapping[str, Any],
    canary: Mapping[str, Any],
    provisioner: str,
    device_model: str,
    architecture: str,
    device_identifier: str,
    driver_version: str,
    vendor_runtime_version: str,
    torch_fl_revision: str,
    container_image_digest: str,
    signing_key: str | bytes,
    key_id: str,
    collected_at: str | None = None,
) -> dict[str, Any]:
    """Validate probe evidence and construct a signed FlagQuantum attestation."""

    if runtime_identity.get("provider") != "torch_fl":
        raise AttestationError("runtime identity is not owned by Torch-FL")
    if runtime_identity.get("accelerator_class") != "domestic_accelerator":
        raise AttestationError(
            "runtime is not a Torch-FL-classified domestic accelerator"
        )
    if int(runtime_identity.get("device_count", 0)) <= 0:
        raise AttestationError("runtime identity contains no flagos devices")
    if canary.get("scope") != AUDIT_SCOPE or canary.get("logical_device") != "flagos:0":
        raise AttestationError(
            "route canary does not cover the required flagos:0 scope"
        )
    if canary.get("cpu_fallback_observed") is not False:
        raise AttestationError("CPU fallback was observed during the route canary")
    if canary.get("host_execution_observed") is not False:
        raise AttestationError("host execution was observed during the route canary")

    devices = runtime_identity.get("devices", [])
    if not devices or devices[0].get("logical_device") != "flagos:0":
        raise AttestationError("runtime identity does not map flagos:0 to device zero")
    logical_name = _non_placeholder(
        "logical device name", devices[0].get("logical_device", "")
    )
    revision = str(torch_fl_revision).strip().lower()
    image_digest = str(container_image_digest).strip().lower()
    if not _COMMIT_RE.fullmatch(revision):
        raise AttestationError(
            "torch_fl_revision must be a full 40-character Git revision"
        )
    if not _IMAGE_DIGEST_RE.fullmatch(image_digest):
        raise AttestationError(
            "container_image_digest must be an immutable sha256 digest"
        )

    physical_probe = {
        "runtime_identity": runtime_identity,
        "route_canary": canary,
        "device_identifier": _non_placeholder("device_identifier", device_identifier),
    }
    probe_digest = hashlib.sha256(_canonical_json(physical_probe)).hexdigest()
    timestamp = collected_at or datetime.now(timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "verified",
        "accelerator_class": "domestic_accelerator",
        "provider": "torch_fl",
        "logical_device": "flagos:0",
        "evidence_source": EVIDENCE_SOURCE,
        "collected_at": _non_placeholder("collected_at", timestamp),
        "provisioner": _non_placeholder("provisioner", provisioner),
        "physical_device": {
            "vendor": _non_placeholder("vendor", runtime_identity.get("vendor", "")),
            "model": _non_placeholder("device_model", device_model),
            "architecture": _non_placeholder("architecture", architecture),
            "device_identifier": _non_placeholder(
                "device_identifier", device_identifier
            ),
            "logical_device_name": logical_name,
        },
        "software": {
            "driver_version": _non_placeholder("driver_version", driver_version),
            "vendor_runtime_version": _non_placeholder(
                "vendor_runtime_version", vendor_runtime_version
            ),
            "torch_fl_revision": revision,
            "container_image_digest": image_digest,
        },
        "route_audit": {
            "provider_internal_route_audited": True,
            "logical_to_physical_device_verified": True,
            "physical_device_probe_sha256": probe_digest,
            "cpu_fallback_allowed": False,
            "cpu_fallback_observed": False,
            "host_execution_observed": False,
            "audit_scope": AUDIT_SCOPE,
        },
    }
    return sign_attestation(payload, signing_key=signing_key, key_id=key_id)


def sign_attestation(
    payload: Mapping[str, Any], *, signing_key: str | bytes, key_id: str
) -> dict[str, Any]:
    """Return a copy of *payload* signed over canonical JSON with HMAC-SHA256."""

    key = signing_key.encode() if isinstance(signing_key, str) else signing_key
    if len(key) < 32:
        raise AttestationError("attestation signing key must contain at least 32 bytes")
    unsigned = dict(payload)
    unsigned.pop("signature", None)
    signature = hmac.new(key, _canonical_json(unsigned), hashlib.sha256).hexdigest()
    signed = dict(unsigned)
    signed["signature"] = {
        "algorithm": "hmac-sha256",
        "key_id": _non_placeholder("key_id", key_id),
        "value": signature,
    }
    return signed


def verify_attestation_signature(
    payload: Mapping[str, Any], *, signing_key: str | bytes
) -> bool:
    """Verify an attestation signature without mutating the supplied payload."""

    signature = payload.get("signature", {})
    if signature.get("algorithm") != "hmac-sha256":
        return False
    value = signature.get("value", "")
    if not isinstance(value, str):
        return False
    unsigned = dict(payload)
    unsigned.pop("signature", None)
    key = signing_key.encode() if isinstance(signing_key, str) else signing_key
    expected = hmac.new(key, _canonical_json(unsigned), hashlib.sha256).hexdigest()
    return hmac.compare_digest(value, expected)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provisioner", required=True)
    parser.add_argument("--device-model", required=True)
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--device-identifier", required=True)
    parser.add_argument("--driver-version", required=True)
    parser.add_argument("--vendor-runtime-version", required=True)
    parser.add_argument("--torch-fl-revision", required=True)
    parser.add_argument("--container-image-digest", required=True)
    parser.add_argument("--key-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--signing-key-env", default="TORCH_FL_ATTESTATION_SIGNING_KEY")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    signing_key = os.environ.get(args.signing_key_env, "")
    if not signing_key:
        raise AttestationError(
            f"signing key environment variable {args.signing_key_env!r} is unset"
        )

    # Import the initialized public package only for the actual runtime probe.
    import torch_fl
    import torch

    identity = torch_fl.runtime_identity()
    canary = run_device_route_canary(torch_module=torch, flagos_module=torch_fl.flagos)
    payload = build_attestation(
        runtime_identity=identity,
        canary=canary,
        provisioner=args.provisioner,
        device_model=args.device_model,
        architecture=args.architecture,
        device_identifier=args.device_identifier,
        driver_version=args.driver_version,
        vendor_runtime_version=args.vendor_runtime_version,
        torch_fl_revision=args.torch_fl_revision,
        container_image_digest=args.container_image_digest,
        signing_key=signing_key,
        key_id=args.key_id,
    )
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
