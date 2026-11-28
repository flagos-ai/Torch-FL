# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

from contextlib import ExitStack
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tarfile

import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "hygon_native_host_cache", ROOT / ".github/scripts/hygon_native_host_cache.py"
)
CACHE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CACHE)


def test_independent_build_roots_share_the_runner_cache(tmp_path, monkeypatch):
    cache = tmp_path / "runner" / "work" / ".torch-fl-native-host-cache" / "hygon"
    checkout = tmp_path / "runner" / "work" / "Torch-FL" / "Torch-FL"
    monkeypatch.setenv("GITHUB_WORKSPACE", str(checkout))
    monkeypatch.delenv("RUNNER_WORKSPACE", raising=False)
    monkeypatch.delenv("TORCH_FL_CI_NATIVE_HOST_CACHE_DIR", raising=False)
    monkeypatch.setenv("GITHUB_REF", "refs/pull/10/merge")
    monkeypatch.setenv("TORCH_FL_CI_IMAGE", "ci-image@sha256:first")
    monkeypatch.setattr(CACHE, "prepare_ccache", lambda _, work: work / "ccache")
    environments = []
    for name, sha in (("cold", "first"), ("warm", "second")):
        monkeypatch.setenv("GITHUB_SHA", sha)
        source = tmp_path / name / "source"
        (source / ".github").mkdir(parents=True)
        (source / ".github/version-pins.env").write_text("PINNED=1\n")
        tools = tmp_path / name / "tools"
        tools.mkdir()
        with ExitStack() as resources:
            environments.append(CACHE.cache_environment(source, tools, resources))
    first, second = environments
    assert first["CCACHE_BASEDIR"] != second["CCACHE_BASEDIR"]
    assert first["CCACHE_STATSLOG"] != second["CCACHE_STATSLOG"]
    assert first["CCACHE_NAMESPACE"] == second["CCACHE_NAMESPACE"]
    assert first["CCACHE_DIR"] == second["CCACHE_DIR"] == str(cache / "objects")
    assert (
        first["CYTHON_CACHE_DIR"] == second["CYTHON_CACHE_DIR"] == str(cache / "cython")
    )
    monkeypatch.setenv("TORCH_FL_CI_IMAGE", "ci-image@sha256:second")
    with ExitStack() as resources:
        changed = CACHE.cache_environment(source, tools, resources)
    assert changed["CCACHE_NAMESPACE"] != second["CCACHE_NAMESPACE"]


def test_cython_and_compiler_caches_can_fall_back_independently(tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("TORCH_FL_CI_NATIVE_HOST_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("CYTHON_CACHE_DIR", raising=False)
    monkeypatch.delenv("_TORCH_FL_NATIVE_HOST_CCACHE", raising=False)
    source = tmp_path / "source"
    (source / ".github").mkdir(parents=True)
    (source / ".github/version-pins.env").touch()
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()

    def missing_binary(*_args):
        raise OSError("download unavailable")

    monkeypatch.setattr(CACHE, "prepare_ccache", missing_binary)
    with ExitStack() as writer, ExitStack() as reader:
        environment = CACHE.cache_environment(source, first, writer)
        assert "CYTHON_CACHE_DIR" in environment
        assert "_TORCH_FL_NATIVE_HOST_CCACHE" not in environment
        monkeypatch.setattr(CACHE, "prepare_ccache", lambda _, work: work / "ccache")
        environment = CACHE.cache_environment(source, second, reader)
        assert "CYTHON_CACHE_DIR" not in environment
        assert "_TORCH_FL_NATIVE_HOST_CCACHE" in environment


@pytest.mark.parametrize("valid", [True, False])
def test_ccache_archive_is_checked_before_use(tmp_path, monkeypatch, valid):
    monkeypatch.setattr(CACHE.shutil, "which", lambda _: None)
    monkeypatch.setattr(CACHE.platform, "system", lambda: "Linux")
    monkeypatch.setattr(CACHE.platform, "machine", lambda: "x86_64")
    binary = b"pinned ccache executable"
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        member = tarfile.TarInfo(f"{CACHE.CCACHE_RELEASE}/ccache")
        member.size = len(binary)
        archive.addfile(member, io.BytesIO(binary))
    data = stream.getvalue()
    monkeypatch.setattr(CACHE, "CCACHE_SHA256", hashlib.sha256(data).hexdigest())
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    saved = cache_root / CACHE.CCACHE_ARCHIVE
    saved.write_bytes(b"partial archive")
    tools = tmp_path / "tools"
    tools.mkdir()
    calls = []

    def download(command, **_kwargs):
        calls.append(command)
        Path(command[command.index("--output") + 1]).write_bytes(
            data if valid else b"unexpected bytes"
        )

    monkeypatch.setattr(CACHE.subprocess, "run", download)
    if valid:
        assert CACHE.prepare_ccache(cache_root, tools).read_bytes() == binary
        assert CACHE.prepare_ccache(cache_root, tools).read_bytes() == binary
        assert saved.read_bytes() == data
    else:
        with pytest.raises(ValueError, match="SHA256"):
            CACHE.prepare_ccache(cache_root, tools)
        assert not (tools / "ccache").exists()
        assert saved.read_bytes() == b"partial archive"
    assert len(calls) == 1


@pytest.mark.parametrize("cache_available", [True, False])
def test_wheel_build_keeps_its_result_and_records_cache_stats(
    tmp_path, monkeypatch, cache_available
):
    def configure(_source, tools, _resources):
        if not cache_available:
            return {}
        return {
            "_TORCH_FL_NATIVE_HOST_CCACHE": str(tools / "ccache"),
            "CCACHE_STATSLOG": str(tools / "stats"),
        }

    def run(command, *, env, **_kwargs):
        if command[-1] == "--print-log-stats":
            return subprocess.CompletedProcess(command, 0, "direct_cache_hit\t2\n")
        assert command[1:] == ["setup.py", "bdist_wheel", "--dist-dir", str(tmp_path)]
        if cache_available:
            Path(env["CCACHE_STATSLOG"]).touch()
        else:
            assert env == {}
        return subprocess.CompletedProcess(command, 7)

    monkeypatch.setattr(CACHE, "cache_environment", configure)
    monkeypatch.setattr(CACHE.subprocess, "run", run)
    assert CACHE.build(tmp_path) == 7
    report = json.loads((tmp_path / "native-host-cache.json").read_text())
    assert report["ccache_enabled"] is cache_available
    assert report["ccache"] == ({"direct_cache_hit": 2} if cache_available else {})
