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

"""Prepare the host paths and the single vendor container for integration."""

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

MODEL_ROOT = Path("/var/lib/torch-fl-ci/models")
MODEL_NAME = Path("Qwen3-0.6B")
INLINE_ENVIRONMENT = {
    "GITHUB_ENV",
    "GITHUB_OUTPUT",
    "GITHUB_PATH",
    "GITHUB_WORKSPACE",
    "HOME",
    "INTEGRATION_RUNTIME",
    "INTEGRATION_PROFILE",
    "INTEGRATION_CONTAINER",
    "INTEGRATION_MODEL",
    "MODEL_PATH",
    "RUNNER_TEMP",
}
REPORT_DIRS = ("test-workdir", "pytest", "coverage", "inventories", "junit")


class WorkspaceError(Exception):
    """A runner or image cannot provide the shared integration workspace."""


def _publish_inline(settings, values):
    """Write inline values by importing the mounted script the same way CI does."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        from publish_container_environment import publish_inline

        publish_inline(settings, values, "host-profile.sh")
    finally:
        del sys.path[0]


def _write_environment(path, settings, *, native):
    values = {
        "GITHUB_ENV": settings["state"] + "/container-environment",
        "GITHUB_OUTPUT": settings["state"] + "/output",
        "GITHUB_PATH": settings["state"] + "/path",
        "GITHUB_WORKSPACE": settings["workspace"],
        "HOME": "/tmp",
        "INTEGRATION_RUNTIME": settings["runtime"],
        "INTEGRATION_PROFILE": settings["state"] + "/profile.sh",
        "INTEGRATION_CONTAINER": settings.get("container", ""),
        "PIP_CACHE_DIR": settings["cache"],
        "INTEGRATION_WORKDIR": settings["reports"]["test-workdir"],
        "FLAGCICD_REPORT_DIR": settings["reports"]["pytest"],
        "FLAGCICD_COVERAGE_DIR": settings["reports"]["coverage"],
        "FLAGCICD_VERDICT_FILE": settings["verdict"],
        "INTEGRATION_INVENTORY_ROOT": settings["reports"]["inventories"],
        "INTEGRATION_PLATFORM": os.environ["INTEGRATION_PLATFORM"],
        "INTEGRATION_IMAGE": settings["digest"],
        "INTEGRATION_MODEL": (
            settings["host_model_path"] if native else settings["model_path"]
        ),
        "MODEL_PATH": settings["model_path"] if native else settings["host_model_path"],
        # Keep setup venvs and caches in mounted runner directories outside the
        # source copy so every container command sees the same installation.
        "RUNNER_TEMP": str(Path(settings["workspace"]).parent),
        "CHECKOUT_REPOSITORY": os.environ["CHECKOUT_REPOSITORY"],
        "CHECKOUT_REF": os.environ["CHECKOUT_REF"],
    }
    Path(path).write_text(
        "".join(
            f"{name}={shlex.quote(value)}\n"
            for name, value in values.items()
            if (value or name == "INTEGRATION_CONTAINER")
        ),
        encoding="utf-8",
    )
    _publish_inline(
        settings,
        {name: values[name] for name in INLINE_ENVIRONMENT},
    )
    shell = Path(settings["state"]) / "shell.sh"
    shell.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'cd "$GITHUB_WORKSPACE"\n'
        'source "$INTEGRATION_PROFILE"\n'
        'exec bash "$@"\n',
        encoding="utf-8",
    )
    shell.chmod(0o755)
    settings["shell"] = str(shell)


def _host_files(state):
    names = ("GITHUB_ENV", "GITHUB_OUTPUT", "GITHUB_PATH")
    live = {}
    for name in names:
        source = os.environ.get(name, "")
        if not source:
            raise WorkspaceError(f"the runner supplied no {name}")
        path = Path(source).resolve()
        live[name] = path
    return live


def _volume(item):
    parts = item.split(":")
    if len(parts) < 2 or not Path(parts[0]).is_absolute():
        raise WorkspaceError(f"container volume has no absolute source: {item}")
    source, destination = Path(parts[0]).resolve(), parts[1]
    if not destination.startswith("/"):
        raise WorkspaceError(f"container volume has no absolute target: {item}")
    return item, source, destination


def _container_volumes(volumes, settings):
    """Mount every directory that the container commands read or write."""
    output = []
    for item in volumes:
        entry, _source, destination = _volume(item)
        if destination == str(MODEL_ROOT) or destination == "/models":
            continue
        if not destination.startswith(str(MODEL_ROOT) + "/"):
            output.append(entry)
    storage = (
        "workspace",
        "cache",
        "state",
        "runtime",
        "verdict",
        "artifacts",
    )
    for name in storage:
        output.append(f"{settings[name]}:{settings[name]}")
    output.extend(f"{path}:{path}" for path in settings["reports"].values())
    output.append(f"{settings['profile_home']}:/github/home")
    output.append(f"{settings['model_root']}:{MODEL_ROOT}:ro")
    return output


def prepare(arguments):
    """Copy the checkout and prepare paths shared by the host and container."""
    checkout = Path(arguments.checkout).resolve()
    runner_temp = Path(arguments.runner_temp).resolve()
    model_root = Path(arguments.model_root) if arguments.model_root else MODEL_ROOT
    model_path = Path(arguments.model_path)
    if not checkout.is_dir():
        raise WorkspaceError(f"the checkout is missing: {checkout}")
    if runner_temp == checkout or runner_temp.is_relative_to(checkout):
        raise WorkspaceError("RUNNER_TEMP must not be inside the checkout")
    if not model_root.is_absolute() or not model_root.is_dir():
        raise WorkspaceError(f"the prepared model root is missing: {model_root}")
    model_root = model_root.resolve()
    if not (model_root / MODEL_NAME).is_dir():
        raise WorkspaceError(f"the prepared checkpoint is missing: {model_root}")
    expected_path = model_root / MODEL_NAME
    if not model_path.is_absolute() or model_path.parent != MODEL_ROOT:
        raise WorkspaceError(f"the model path must be below {MODEL_ROOT}: {model_path}")
    if model_path.name != expected_path.name:
        raise WorkspaceError(f"the model path and checkpoint disagree: {model_path}")
    if arguments.image and not arguments.digest:
        raise WorkspaceError("a configured image has no resolved RepoDigest")
    try:
        volumes = json.loads(arguments.volumes)
    except json.JSONDecodeError as error:
        raise WorkspaceError(f"container volumes are not JSON: {error}") from error
    if not isinstance(volumes, list) or not all(
        isinstance(item, str) and item for item in volumes
    ):
        raise WorkspaceError("container volumes must be a list of non-empty strings")

    settings = {
        "workspace": str(runner_temp / "torch-fl-workspace"),
        "state": str(runner_temp / "torch-fl-preparation"),
        "runtime": str(runner_temp / "torch-fl-runtime"),
        "profile_home": str(runner_temp / "torch-fl-profile-home"),
        "cache": str(runner_temp / "torch-fl-pip"),
        "model_root": str(model_root),
        # Hosted macOS validates the runner path; containers see the bind target.
        "model_path": str(model_path),
        "host_model_path": str(expected_path),
        "artifacts": str(runner_temp / "torch-fl-artifacts"),
        "reports": {
            name: str(runner_temp / f"torch-fl-{name}") for name in REPORT_DIRS
        },
        "verdict": str(runner_temp / "torch-fl-verdict.json"),
        "image": arguments.image,
        "digest": arguments.digest,
        "volumes": volumes,
        "options": arguments.options,
        "platform": os.environ["INTEGRATION_PLATFORM"],
    }
    paths = [
        settings["workspace"],
        settings["state"],
        settings["runtime"],
        settings["profile_home"],
        settings["cache"],
        settings["artifacts"],
        Path(settings["verdict"]),
        *map(Path, settings["reports"].values()),
    ]
    if len(set(paths)) != len(paths):
        raise WorkspaceError("integration storage paths overlap")
    paths = list(map(Path, paths))
    for path in paths:
        if path == checkout or checkout.is_relative_to(path):
            raise WorkspaceError(f"preparation path contains the checkout: {path}")
        if path.exists():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
    Path(settings["state"]).mkdir(parents=True)
    live = _host_files(Path(settings["state"]))
    settings["host_files"] = {name: str(path) for name, path in live.items()}
    settings["inline_environment"] = sorted(INLINE_ENVIRONMENT)
    for path in settings["reports"].values():
        Path(path).mkdir(parents=True)
    Path(settings["artifacts"]).mkdir(parents=True)
    Path(settings["runtime"]).mkdir()
    Path(settings["cache"]).mkdir()
    Path(settings["profile_home"]).mkdir()
    shutil.copytree(
        checkout,
        settings["workspace"],
        symlinks=True,
        ignore=shutil.ignore_patterns(".git", "build", "dist", ".ci"),
    )
    profile = Path(settings["state"]) / "profile.sh"
    # Hosted macOS prepends the current directory to PATH. A native lane must
    # still find its own tools, but a relative entry would make the same
    # command search the mounted workspace for an executable with its name.
    paths = os.environ.get("PATH", "").split(os.pathsep)
    native_path = os.pathsep.join(path for path in paths if Path(path).is_absolute())
    profile.write_text(
        f"export PATH={shlex.quote(native_path)}\n"
        # Source this profile once without BASH_ENV, then pass its final state
        # to setup. Opening a script would execute BASH_ENV too early.
        f"export BASH_ENV={shlex.quote(str(profile))}\n"
        'if [[ -s "$GITHUB_ENV" ]]; then\n'
        "  set -a\n"
        '  . "$GITHUB_ENV"\n'
        "  set +a\n"
        "fi\n"
        'if [[ -s "$GITHUB_PATH" ]]; then\n'
        "  while IFS= read -r entry; do\n"
        '    [[ -n "$entry" ]] && export PATH="$entry:$PATH"\n'
        '  done < "$GITHUB_PATH"\n'
        "fi\n",
        encoding="utf-8",
    )
    environment = Path(settings["state"]) / "container-environment"
    settings["environment"] = str(environment)
    _write_environment(environment, settings, native=not bool(arguments.image))
    settings["environment"] = str(environment)
    config = Path(settings["state"]) / "config.json"
    config.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    Path(settings["verdict"]).write_text('{"outcome":"failure"}\n')
    print(f"Integration workspace: {settings['workspace']}")
    print(f"Integration config: {config}")
    return settings


def configure(settings, container):
    """Persist the address of an already-running container."""
    if settings.get("container") or not container:
        raise WorkspaceError("configuration needs one new container")
    probe = subprocess.run(
        [
            "docker",
            "exec",
            container,
            "--env",
            "HOME=/tmp",
            "bash",
            settings["shell"],
            "-c",
            'test -d "$PIP_CACHE_DIR" && test -w "$GITHUB_WORKSPACE" && '
            'test -d "$INTEGRATION_MODEL" && test -d "$MODEL_PATH"',
        ],
        capture_output=True,
        text=True,
        timeout=45,
    )
    if probe.returncode:
        logs = subprocess.run(
            ["docker", "logs", container], capture_output=True, text=True
        ).stdout
        raise WorkspaceError(
            "the vendor container cannot use the integration mount:\n"
            + probe.stderr
            + logs
        )
    settings["container"] = container
    settings["container_name"] = (
        subprocess.run(
            ["docker", "inspect", "--format", "{{.Name}}", container],
            capture_output=True,
            text=True,
            check=True,
        )
        .stdout.strip()
        .removeprefix("/")
    )
    settings["environment"] = str(Path(settings["state"]) / "container-environment")
    _write_environment(settings["environment"], settings, native=False)
    Path(settings["state"], "config.json").write_text(
        json.dumps(settings, indent=2), encoding="utf-8"
    )
    print(f"Integration container: {container}")
    return settings


def start(settings):
    """Start exactly one new vendor container; never attach to an old one."""
    if "container" in settings:
        raise WorkspaceError("the integration container was already started")
    image, digest = settings["image"], settings["digest"]
    _verify_image(image, digest, require_digest=True)
    name = (
        f"torch-fl-{os.environ['GITHUB_RUN_ID']}-"
        f"{os.environ['GITHUB_RUN_ATTEMPT']}-"
        f"{os.environ['INTEGRATION_PLATFORM']}"
    )
    subprocess.run(
        ["docker", "rm", "--force", name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    volumes = _container_volumes(settings["volumes"], settings)
    image_reference = image if "@" in image else digest
    command = [
        "docker",
        "create",
        "--name",
        name,
        "--workdir",
        settings["workspace"],
        "--env-file",
        settings["environment"],
        "--env",
        "HOME=/tmp",
        *sum((["--volume", value] for value in volumes), []),
        *shlex.split(settings["options"]),
        # This entrypoint works even when the vendor image has no init process
        # or a different default command.
        # A configured tagless digest is already the exact image. Other
        # references are started from their repository digest so a local tag
        # and the backend's metadata lookup are not involved in startup.
        image_reference,
        "tail",
        "-f",
        "/dev/null",
    ]
    # Resolve the image and volumes before startup. A backend with an explicit
    # runtime may fail after creating the container, leaving docker run's name
    # without a URL. Starting from the ID makes that failure name the container
    # and lets the later probe report the container's state and logs.
    resolved = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )
    if resolved.returncode:
        raise WorkspaceError(
            "the vendor container could not be created:\n" + resolved.stderr.strip()
        )
    container_id = resolved.stdout.strip()
    if not container_id:
        raise WorkspaceError("docker container create returned no container ID")
    started = subprocess.run(
        ["docker", "start", container_id],
        capture_output=True,
        text=True,
    )
    if started.returncode:
        logs = subprocess.run(
            ["docker", "logs", container_id], capture_output=True, text=True
        )
        raise WorkspaceError(
            "the vendor container could not start:\n"
            + started.stderr.strip()
            + "\n"
            + logs.stdout
            + logs.stderr
        )
    return container_id


def _verify_image(image, digest, *, require_digest):
    """Match a tagged image or repository against its local digest."""
    if require_digest and not image:
        raise WorkspaceError("starting a container requires image and digest")
    if not image:
        return
    repository, separator, resolved = digest.partition("@sha256:")
    if (
        not separator
        or len(resolved) != 64
        or any(char not in "0123456789abcdef" for char in resolved)
        or repository.rsplit(":", 1)[0] != image.partition("@")[0].rsplit(":", 1)[0]
    ):
        raise WorkspaceError("the image name does not match its resolved digest")


def verify(config):
    """Save a normalized image identity before starting its container."""
    _verify_image(config["image"], config["digest"], require_digest=False)
    Path(config["state"], "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", nargs="?", choices=("prepare", "start", "configure", "verify")
    )
    parser.add_argument("--config")
    parser.add_argument("--container", default="")
    parser.add_argument("--checkout")
    parser.add_argument("--runner-temp")
    parser.add_argument("--model-root")
    parser.add_argument("--model-path", default=str(MODEL_ROOT / MODEL_NAME))
    parser.add_argument("--image", default="")
    parser.add_argument("--digest", default="")
    parser.add_argument("--volumes", default="[]")
    parser.add_argument("--options", default="")
    arguments = parser.parse_args(argv)
    try:
        if arguments.command in ("start", "configure", "verify"):
            settings = json.loads(Path(arguments.config).read_text(encoding="utf-8"))
            if arguments.command == "start":
                print(start(settings))
            elif arguments.command == "verify":
                verify(settings)
            else:
                configure(settings, arguments.container)
        else:
            if not arguments.checkout or not arguments.runner_temp:
                raise WorkspaceError("--checkout and --runner-temp are required")
            prepare(arguments)
    except (KeyError, OSError, TypeError, ValueError, WorkspaceError) as error:
        print(f"::error::{error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
