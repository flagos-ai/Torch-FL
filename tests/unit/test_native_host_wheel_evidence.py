# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[2] / ".github/scripts/record_native_host_wheel.py"
)
_SPEC = importlib.util.spec_from_file_location("record_native_host_wheel", _SCRIPT)
_RECORD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_RECORD)


@pytest.mark.parametrize("matches", [True, False])
def test_record_requires_the_installed_archive_digest(tmp_path, monkeypatch, matches):
    wheel = tmp_path / "torch_fl-2.10.0-py3-none-any.whl"
    with ZipFile(wheel, "w") as archive:
        archive.writestr("torch_fl/_build_config.py", "NATIVE_HOSTS = False\n")
        archive.writestr("torch_fl/compatibility.json", "{}")
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    direct_url = {
        "archive_info": {"hashes": {"sha256": digest if matches else "0" * 64}}
    }
    package = tmp_path / "install/torch_fl"
    monkeypatch.setattr(
        _RECORD,
        "distribution",
        lambda name: SimpleNamespace(
            read_text=lambda filename: json.dumps(direct_url),
            locate_file=lambda name: package,
        ),
    )
    plan = {"source_sha": "source"}
    monkeypatch.setenv("CI_PLAN_CONTEXT", json.dumps(plan))
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    record = evidence / "wheel.json"

    if matches:
        _RECORD.record_wheel(wheel, evidence)
        assert json.loads(record.read_text()) == {
            "wheel": wheel.name,
            "package": str(package.resolve()),
            "sha256": digest,
            "plan": plan,
        }
        assert (evidence / "torch_fl/compatibility.json").read_text() == "{}"
    else:
        with pytest.raises(ValueError, match="does not match the tested wheel"):
            _RECORD.record_wheel(wheel, evidence)
        assert not record.exists()
