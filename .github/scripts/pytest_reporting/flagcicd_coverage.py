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

"""Collect top-level pytest runs and explicitly selected native-host children."""

from contextlib import contextmanager
import importlib.util
import json
import os
import sys
import uuid
from pathlib import Path

import pytest


NATIVE_HOST_COVERAGE_DIR = "FLAGCICD_NATIVE_HOST_COVERAGE_DIR"


class CoverageSession:
    def __init__(self, collector, marker):
        self.collector = collector
        self.marker = marker
        self.saved = False

    def save(self, exitstatus=None):
        if self.saved:
            return
        self.saved = True
        try:
            self.collector.stop()
            self.collector.save()
            if exitstatus is not None:
                self.marker.with_suffix(".complete").write_text(
                    json.dumps({"exitcode": int(exitstatus)}), encoding="utf-8"
                )
        except Exception as error:
            # Metrics are best effort; preserve pytest's original verdict.
            print(f"::warning::Cannot save Python coverage: {error}", file=sys.stderr)

    @pytest.hookimpl(trylast=True)
    def pytest_sessionfinish(self, session, exitstatus):
        self.save(exitstatus)

    def pytest_unconfigure(self, config):
        self.save()


def start_coverage(directory, package, group):
    marker = directory / f"coverage-{uuid.uuid4().hex}.pending"
    marker.write_text(json.dumps({"group": group}), encoding="utf-8")
    import coverage

    collector = coverage.Coverage(
        data_file=str(marker.with_suffix(".data")),
        config_file=False,
        source=[str(package)],
        branch=False,
    )
    collector.start()
    return CoverageSession(collector, marker)


def native_host_directory(directory, ordinary_package):
    """Use the wheel recorded after the current plan's native-host preflight."""
    runner_temp = Path(os.environ["RUNNER_TEMP"])
    evidence_file = runner_temp / "torch-fl-native-host-evidence/wheel.json"
    evidence = json.loads(evidence_file.read_text(encoding="utf-8"))
    context = json.loads(os.environ.get("CI_PLAN_CONTEXT") or "{}")
    package = Path(evidence["package"]).resolve()
    expected = (runner_temp / "torch-fl-native-host-install/torch_fl").resolve()
    if (
        evidence["plan"] != context
        or package != expected
        or package == ordinary_package
        or not package.is_dir()
    ):
        raise ValueError("Native-host coverage requires this plan's isolated wheel")
    native = directory / "native-host"
    native.mkdir(exist_ok=True)
    manifest_file = native / "manifest.json"
    manifest = {"package_root": str(package), "wheel_sha256": evidence["sha256"]}
    if manifest_file.is_file():
        if json.loads(manifest_file.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Native-host wheel changed between coverage sessions")
    else:
        manifest_file.write_text(json.dumps(manifest), encoding="utf-8")
    return native, package


def warn_native_host(directory, group, error):
    message = f"{group}: {error}"
    print(
        f"::warning::Cannot start native-host Python coverage: {message}",
        file=sys.stderr,
    )
    try:
        directory.mkdir(exist_ok=True)
        with (directory / "collection-warnings.txt").open("a", encoding="utf-8") as log:
            log.write(f"{message}\n")
    except OSError as log_error:
        print(
            f"::warning::Cannot save native-host coverage warning: {log_error}",
            file=sys.stderr,
        )


def imports_package(package):
    spec = importlib.util.find_spec("torch_fl")
    return spec and spec.origin and Path(spec.origin).resolve().parent == package


@contextmanager
def native_host_coverage(package):
    """Collect a common regression's native-host child before its first import."""
    session = None
    directory = Path(os.environ.pop(NATIVE_HOST_COVERAGE_DIR))
    try:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if Path(manifest["package_root"]) != package:
            raise ValueError("Native-host child uses a different wheel")
        session = start_coverage(directory, package, "Common regression")
    except Exception as error:
        warn_native_host(directory, "Common regression", error)
    exitcode = 1
    try:
        yield
        exitcode = 0
    finally:
        if session is not None:
            session.save(exitcode)


@pytest.hookimpl(tryfirst=True)
def pytest_load_initial_conftests(early_config):
    directory = os.environ.pop("FLAGCICD_COVERAGE_DIR", "")
    unit_probe = os.environ.pop("FLAGCICD_NVIDIA_UNIT_PROBE", "") == "1"
    if not directory:
        return
    # Only actual groups/per-file runs inherit activation from their runner.
    # A pytest fixture's own sample subprocess must not add coverage sessions.
    plugins = [
        name
        for name in os.environ.get("PYTEST_PLUGINS", "").split(",")
        if name.strip() not in ("", "flagcicd_coverage")
    ]
    if plugins:
        os.environ["PYTEST_PLUGINS"] = ",".join(plugins)
    else:
        os.environ.pop("PYTEST_PLUGINS", None)
    try:
        directory = Path(directory).resolve()
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        package = Path(manifest["package_root"])
        group_id = os.environ.get("FLAGCICD_GROUP_ID")
        group = os.environ.get("FLAGCICD_TEST_GROUP", "")
        if unit_probe:
            directory /= "unit-probe"
            directory.mkdir(exist_ok=True)
            if not imports_package(package):
                raise ValueError("Unit probe imports a different wheel")
            (directory / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
        elif os.environ.get("REPORT_PLATFORM") == "nvidia" and group_id in {
            "common_regression",
            "eager_ops_regression",
        }:
            try:
                native, native_package = native_host_directory(directory, package)
                if group_id == "eager_ops_regression":
                    if not imports_package(native_package):
                        raise ValueError(
                            "Eager operator regression imports a different wheel"
                        )
                    directory, package = native, native_package
                else:
                    os.environ[NATIVE_HOST_COVERAGE_DIR] = str(native)
            except Exception as error:
                warn_native_host(directory / "native-host", group, error)
                if group_id == "eager_ops_regression":
                    return
        early_config.pluginmanager.register(
            start_coverage(directory, package, group),
            "flagcicd_python_coverage_session",
        )
    except Exception as error:
        print(f"::warning::Cannot start Python coverage: {error}", file=sys.stderr)
