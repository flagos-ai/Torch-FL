#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Build the Hygon Native Host wheel with optional runner-local caches."""

from contextlib import ExitStack
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import sysconfig
import tarfile
import tempfile
import time


CCACHE_VERSION = "4.14.3"
CCACHE_RELEASE = f"ccache-{CCACHE_VERSION}-linux-x86_64-musl-static"
CCACHE_ARCHIVE = f"{CCACHE_RELEASE}.tar.gz"
CCACHE_URL = f"https://github.com/ccache/ccache/releases/download/v{CCACHE_VERSION}/{CCACHE_ARCHIVE}"
CCACHE_SHA256 = "e4583fd653b794a480b9915358679a75cdd7a7010dd865eaf4cbdc61e665a3f0"


def prepare_ccache(cache_root, temporary):
    """Use the image's pinned ccache or a verified copy of the upstream release."""
    installed = shutil.which("ccache")
    if installed:
        info = subprocess.run(
            [installed, "--version"], capture_output=True, text=True, timeout=5
        )
        if info.returncode == 0 and info.stdout.startswith(
            f"ccache version {CCACHE_VERSION}\n"
        ):
            return Path(installed)

    if (platform.system(), platform.machine()) != ("Linux", "x86_64"):
        raise ValueError("the pinned ccache binary requires Linux x86_64")
    saved_archive = cache_root / CCACHE_ARCHIVE
    data = saved_archive.read_bytes() if saved_archive.is_file() else b""
    if hashlib.sha256(data).hexdigest() != CCACHE_SHA256:
        with tempfile.TemporaryDirectory(dir=cache_root) as staging:
            archive = Path(staging) / CCACHE_ARCHIVE
            subprocess.run(
                [
                    "curl",
                    "--fail",
                    "--silent",
                    "--show-error",
                    "--location",
                    "--connect-timeout",
                    "5",
                    "--max-time",
                    "20",
                    "--output",
                    str(archive),
                    CCACHE_URL,
                ],
                check=True,
                timeout=25,
            )
            data = archive.read_bytes()
            if hashlib.sha256(data).hexdigest() != CCACHE_SHA256:
                raise ValueError("ccache archive SHA256 differs from its release pin")
            archive.replace(saved_archive)

    binary = temporary / "ccache"
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as bundle:
        with bundle.extractfile(f"{CCACHE_RELEASE}/ccache") as source, binary.open(
            "wb"
        ) as out:
            shutil.copyfileobj(source, out)
    binary.chmod(0o755)
    return binary


def cache_environment(source, temporary, resources):
    checkout = Path(os.environ["GITHUB_WORKSPACE"])
    runner_workspace = Path(os.environ.get("RUNNER_WORKSPACE") or checkout.parent)
    root = Path(
        os.environ.get("TORCH_FL_CI_NATIVE_HOST_CACHE_DIR")
        or runner_workspace.parent / ".torch-fl-native-host-cache" / "hygon"
    ).absolute()
    objects = root / "objects"
    cython = root / "cython"
    environment = os.environ.copy()
    environment.pop("CYTHON_CACHE_DIR", None)
    environment.pop("_TORCH_FL_NATIVE_HOST_CCACHE", None)
    try:
        cython.mkdir(parents=True, exist_ok=True)
        # Cython writes each entry through a fixed temporary filename.
        lock = resources.enter_context((cython / ".lock").open("a"))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (cython / "compiler").mkdir(exist_ok=True)
        with tempfile.TemporaryFile(dir=cython / "compiler"):
            pass
        environment["CYTHON_CACHE_DIR"] = str(cython)
    except OSError as error:
        print(
            f"::warning::Runner-local Cython cache unavailable ({error}).", flush=True
        )

    try:
        objects.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile(dir=objects):
            pass
        binary = prepare_ccache(root, temporary)
        config = temporary / "ccache.conf"
        config.touch()
    except (OSError, ValueError, subprocess.SubprocessError, tarfile.TarError) as error:
        print(f"::warning::Native Host ccache unavailable ({error}).", flush=True)
        return environment

    # Content checks cover compiler binaries, inputs and headers. Keep object
    # lookups separate across CI profiles and runtime cohorts, with one size
    # limit for the runner. The source commit is deliberately absent.
    scope = [
        os.environ.get(name, "")
        for name in (
            "GITHUB_REPOSITORY",
            "GITHUB_EVENT_NAME",
            "GITHUB_REF",
            "TORCH_FL_CI_IMAGE",
        )
    ]
    scope.extend(
        [
            platform.machine(),
            sysconfig.get_config_var("SOABI"),
            hashlib.sha256(
                (source / ".github/version-pins.env").read_bytes()
            ).hexdigest(),
        ]
    )
    namespace = hashlib.sha256(json.dumps(scope).encode()).hexdigest()
    return {
        **environment,
        "_TORCH_FL_NATIVE_HOST_CCACHE": str(binary),
        "CCACHE_BASEDIR": str(source),
        "CCACHE_COMPILERCHECK": "content",
        "CCACHE_CONFIGPATH": str(config),
        "CCACHE_DIR": str(objects),
        "CCACHE_MAXSIZE": "4G",
        "CCACHE_NAMESPACE": namespace,
        "CCACHE_STATSLOG": str(temporary / "ccache.stats"),
    }


def cache_stats(environment):
    if not Path(environment["CCACHE_STATSLOG"]).is_file():
        return {}
    result = subprocess.run(
        [environment["_TORCH_FL_NATIVE_HOST_CCACHE"], "--print-log-stats"],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
        timeout=5,
    )
    return {
        name: int(value)
        for name, value in (line.split() for line in result.stdout.splitlines())
    }


def build(evidence):
    start = time.monotonic()
    stats = {}
    with tempfile.TemporaryDirectory(
        prefix="torch-fl-ccache.", dir=os.environ.get("RUNNER_TEMP")
    ) as temporary, ExitStack() as resources:
        environment = cache_environment(Path.cwd(), Path(temporary), resources)
        build_start = time.monotonic()
        result = subprocess.run(
            [sys.executable, "setup.py", "bdist_wheel", "--dist-dir", str(evidence)],
            env=environment,
        )
        duration = time.monotonic() - build_start
        ccache_enabled = "_TORCH_FL_NATIVE_HOST_CCACHE" in environment
        if ccache_enabled:
            try:
                stats = cache_stats(environment)
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                print(f"::warning::Could not read ccache statistics ({error}).")
    report = {
        "ccache_enabled": ccache_enabled,
        "cython_enabled": bool(environment.get("CYTHON_CACHE_DIR")),
        "prepare_seconds": round(build_start - start, 2),
        "build_seconds": round(duration, 2),
        "ccache": stats,
    }
    message = json.dumps(report)
    print(f"Native Host build cache: {message}")
    (evidence / "native-host-cache.json").write_text(message + "\n")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(build(Path(sys.argv[1])))
