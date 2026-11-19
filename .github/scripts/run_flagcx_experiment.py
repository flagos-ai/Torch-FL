#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Check Ascend FlagCX admission, then optionally validate collectives."""

import argparse
from email.parser import Parser
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import zipfile


def inspect_wheel(wheel, version, checksum, directory):
    """Verify original bytes and report native dependencies for admission."""
    actual = hashlib.sha256(wheel.read_bytes()).hexdigest()
    if actual != checksum:
        raise ValueError(f"FlagCX wheel SHA256 mismatch: {wheel.name}")
    dependencies = {}
    with zipfile.ZipFile(wheel) as archive:
        metadata = [n for n in archive.namelist() if n.endswith(".dist-info/METADATA")]
        if len(metadata) != 1:
            raise ValueError("FlagCX wheel must have exactly one METADATA file")
        info = Parser().parsestr(archive.read(metadata[0]).decode())
        if info.get("Name", "").lower() != "flagcx" or info.get("Version") != version:
            raise ValueError("FlagCX wheel metadata differs from the version pin")
        extensions = [
            n
            for n in archive.namelist()
            if n.startswith("flagcx/_C.") and n.endswith(".so")
        ]
        if len(extensions) != 1:
            raise ValueError("FlagCX wheel must contain one native Python extension")
        for index, name in enumerate(
            n for n in archive.namelist() if n.endswith(".so")
        ):
            target = directory / f"{index}.so"
            target.write_bytes(archive.read(name))
            result = subprocess.run(
                ["readelf", "-d", str(target)],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            dependencies[name] = re.findall(r"\(NEEDED\).*?\[(.*?)\]", result.stdout)
    return {
        "file": wheel.name,
        "version": version,
        "sha256": actual,
        "native_dependencies": dependencies,
    }


def assert_no_vendor_torch_npu():
    """Allow Torch-FL's Python shim, but exclude real vendor modules/libraries."""
    from check_platform_environment import assert_loaded_torch_libraries

    root = sys.modules.get("torch_npu")
    allowed = {}
    if root is not None and getattr(root, "__torch_fl_shim__", False) is True:
        allowed = {"torch_npu": root, "torch_npu._C": getattr(root, "_C", None)}
    unexpected = [
        name
        for name, module in list(sys.modules.items())
        if module is not None
        and (name == "torch_npu" or name.startswith("torch_npu."))
        and (module is not allowed.get(name) or getattr(module, "__file__", None))
    ]
    assert not unexpected, f"Real vendor torch_npu modules imported: {unexpected}"
    assert_loaded_torch_libraries(Path("."), (), forbidden=("libtorch_npu.so",))


def collective_worker():
    """Require Torch-FL's FlagCX tier, never a successful host fallback."""
    from datetime import timedelta
    from importlib import metadata

    import torch_fl
    import torch
    import torch.distributed as dist

    assert_no_vendor_torch_npu()
    import flagcx  # noqa: F401 -- backend registration is part of this probe

    assert_no_vendor_torch_npu()
    assert torch.version.cuda is None, torch.__version__
    assert torch.__version__.split("+")[0] == "2.10.0", torch.__version__
    assert metadata.version("flagcx") == os.environ["FLAGCX_EXPERIMENT_VERSION"]
    local_rank = int(os.environ["LOCAL_RANK"])
    torch_fl.flagos.set_device(local_rank)
    dist.init_process_group("flagos", timeout=timedelta(seconds=120))
    try:
        group = dist.distributed_c10d._get_default_group()
        inner = getattr(group, "_inner", None)
        assert inner is not None, "Torch-FL did not create an inner process group"
        name = inner._get_backend_name()
        assert name.lower() == "flagcx", f"Expected FlagCX, got {name}"
        rank, world = dist.get_rank(), dist.get_world_size()
        assert world == 2, world
        value = torch.full((8,), float(rank + 1), device=f"flagos:{local_rank}")
        assert value.device == torch.device("flagos", local_rank), value.device
        dist.all_reduce(value)
        torch.testing.assert_close(value.cpu(), torch.full((8,), 3.0))
        source = torch.full((8,), float(rank + 1), device=value.device)
        gathered = [torch.empty_like(source) for _ in range(world)]
        dist.all_gather(gathered, source)
        for peer, result in enumerate(gathered):
            torch.testing.assert_close(result.cpu(), torch.full((8,), float(peer + 1)))
        print(
            f"FlagCX experiment rank={rank}: backend={name}; all_reduce/all_gather passed",
            flush=True,
        )
    finally:
        dist.destroy_process_group()


def run_collectives(command, workdir, env, log, timeout=300):
    """Reap this experiment's entire worker group on timeout."""
    with subprocess.Popen(
        command,
        cwd=workdir,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    ) as process:
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            finally:
                # A launcher can exit before a worker that ignored SIGTERM.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            raise


def experiment(args):
    report = {
        "schema_version": "torch-fl.flagcx-experiment/v1",
        "platform": "ascend",
        "required": False,
        "status": "failed",
        "admission_status": "not_evaluated",
        "import_attempted": False,
        "runtime_compatibility": "not_tested",
        "baseline_environment_modified": False,
        "source_ref": os.environ.get("SOURCE_REF", os.environ.get("GITHUB_SHA")),
        "python": sys.executable,
        "collectives_started": False,
        "requested_wheel": args.wheel,
        "requested_version": args.version,
        "requested_sha256": args.sha256,
        "torch_fl_wheel": args.torch_fl_wheel.name,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        if os.environ.get("FLAGOS_ACCELERATOR") != "ascend":
            raise ValueError("The FlagCX experiment is enabled only for Ascend")
        if Path(args.wheel).name != args.wheel:
            raise ValueError("Expected a wheel filename, not a path")
        with tempfile.TemporaryDirectory(prefix="torch-fl-flagcx-") as scratch:
            scratch = Path(scratch)
            wheelhouse = args.wheelhouse
            if wheelhouse is None:
                wheelhouse = scratch / "wheels"
                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "download",
                        "--no-deps",
                        "--only-binary=:all:",
                        "--index-url",
                        args.index_url,
                        "--dest",
                        str(wheelhouse),
                        "flagcx===" + args.version,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=180,
                )
                (args.output / "download.log").write_text(result.stdout + result.stderr)
                result.check_returncode()
            wheel = wheelhouse / args.wheel
            report["wheel"] = inspect_wheel(wheel, args.version, args.sha256, scratch)
            dependencies = report["wheel"]["native_dependencies"]
            if any(
                "libtorch_npu.so" in libraries for libraries in dependencies.values()
            ):
                report.update(
                    status="blocked",
                    admission_status="blocked",
                    reason_type="environment_policy",
                    reason="CI admission policy excludes the required libtorch_npu.so. No import or collective was attempted; runtime compatibility was not tested.",
                )
                return 1
            report["admission_status"] = "passed"
            overlay = scratch / "site-packages"
            # Native-host tests may have installed a different Torch-FL wheel.
            # Pair FlagCX with the ordinary wheel tested by the baseline groups.
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--no-index",
                    "--no-deps",
                    "--no-compile",
                    "--target",
                    str(overlay),
                    str(args.torch_fl_wheel.resolve()),
                    str(wheel),
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            (args.output / "install.log").write_text(result.stdout + result.stderr)
            result.check_returncode()
            # Only the experimental children see this package overlay; the
            # original venv, main environment report and regular tests stay intact.
            env = {
                **os.environ,
                "PYTHONPATH": str(overlay),
                "PYTHONNOUSERSITE": "1",
                "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
                "FLAGCX_TORCH_BACKEND": "flagos",
                "FLAGCX_EXPERIMENT_VERSION": args.version,
            }
            for name in ("PYTEST_PLUGINS", "COVERAGE_PROCESS_START"):
                env.pop(name, None)
            report["collectives_started"] = True
            report["import_attempted"] = True
            report["runtime_compatibility"] = "unverified"
            with (args.output / "collectives.log").open("w") as log:
                code = run_collectives(
                    [
                        sys.executable,
                        "-m",
                        "torch.distributed.run",
                        "--standalone",
                        "--nnodes=1",
                        "--nproc-per-node=2",
                        "--max-restarts=0",
                        str(Path(__file__).resolve()),
                        "--worker",
                    ],
                    args.workdir,
                    env,
                    log,
                )
            report.update(
                status="passed" if code == 0 else "failed", collective_exit_code=code
            )
            if code == 0:
                report["runtime_compatibility"] = "validated"
            return int(code != 0)
    except (
        OSError,
        ValueError,
        subprocess.SubprocessError,
        zipfile.BadZipFile,
    ) as error:
        report["reason"] = str(error)
        return 1
    finally:
        text = json.dumps(report, indent=2) + "\n"
        (args.output / "result.json").write_text(text)
        print(text)
        if report["status"] != "passed":
            print(
                f"::warning::Ascend FlagCX admission/collective check {report['status']}: {report.get('reason', 'see collective log')}"
            )
        if os.environ.get("GITHUB_STEP_SUMMARY"):
            with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a") as summary:
                summary.write(
                    f"\n### Ascend FlagCX admission and optional collectives: {report['status']}\n\n"
                    "This check is not required by baseline CI. A green baseline does not establish FlagCX support.\n\n"
                    + report.get("reason", "See the dedicated experiment artifact.")
                    + f"\n\nRuntime compatibility: {report['runtime_compatibility']}; "
                    + f"import attempted: {report['import_attempted']}; "
                    + f"collective probe started: {report['collectives_started']}.\n"
                )


def main():
    if sys.argv[1:] == ["--worker"]:
        collective_worker()
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheelhouse", type=Path)
    parser.add_argument(
        "--index-url",
        default="https://resource.flagos.net/repository/flagos-pypi-ascend/simple",
    )
    parser.add_argument("--wheel", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--torch-fl-wheel", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return experiment(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
