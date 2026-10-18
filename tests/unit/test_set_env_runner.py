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

"""The set_env entrypoint: one runner, per-platform hooks, thin wrappers.

`.github/scripts/set_env.sh --platform <p>` owns the boilerplate every platform
shares and sources `hooks/set_env_<p>.sh`, which is the platform-specific part.
The old `set_env_<p>.sh` files are thin wrappers around the runner, and the
integration pipeline calls the runner directly. This pins that shape: the runner
dispatches, every platform has a hook and a wrapper, and the pipeline uses the
parameterized entrypoint.

The runner is exercised with `--help`/an unknown platform, which return before
any provisioning runs.
"""

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / ".github" / "scripts"
RUNNER = SCRIPTS / "set_env.sh"
HOOKS = SCRIPTS / "hooks"
WORKFLOWS = REPO_ROOT / ".github" / "workflows"

PLATFORMS = [
    "cuda",
    "ascend",
    "dcu",
    "gcu",
    "kunlunxin",
    "metax",
    "mps",
    "musa",
    "ppu",
]


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(RUNNER), *args],
        capture_output=True,
        text=True,
        env={"CI_STAGE": "build", "PATH": "/usr/bin:/bin"},
    )


def test_runner_rejects_a_missing_platform():
    done = _run()
    assert done.returncode == 2
    assert "--platform is required" in done.stderr


def test_runner_rejects_an_unknown_platform():
    done = _run("--platform", "bogus")
    assert done.returncode == 2
    assert "unknown platform 'bogus'" in done.stderr


def test_runner_rejects_an_unknown_argument():
    done = _run("--nope")
    assert done.returncode == 2
    assert "unknown argument" in done.stderr


def test_every_platform_has_a_hook():
    for platform in PLATFORMS:
        assert (HOOKS / f"set_env_{platform}.sh").is_file(), platform


def test_every_wrapper_routes_to_the_runner():
    for platform in PLATFORMS:
        text = (SCRIPTS / f"set_env_{platform}.sh").read_text(encoding="utf-8")
        assert "set_env.sh" in text and f"--platform {platform}" in text, platform


def test_the_parameterized_workflow_provisions_every_platform():
    """One workflow provisions every platform, from that platform's config.

    There used to be a file per platform under `integration-test-<p>.yml`, each
    repeating `bash .github/scripts/set_env.sh --platform <p>` and hard-coding
    its stage. The pipeline reads setup_script/setup_platform/setup_stage from
    the config now, so a platform that needs the `build` stage (Ascend,
    Kunlunxin) is no longer the only shape the file can express.
    """
    text = (WORKFLOWS / "integration-tests.yml").read_text(encoding="utf-8")
    assert (
        'bash "${{ needs.configuration.outputs.setup_script }}" '
        '--platform "${{ needs.configuration.outputs.setup_platform }}"' in text
    )
    assert "CI_STAGE: ${{ needs.configuration.outputs.setup_stage }}" in text


def test_config_driven_path_uses_the_runner():
    """The configs name the runner and the platform, so no wrapper hop.

    The pipeline reads setup_script/setup_platform from the config and runs
    `bash set_env.sh --platform <p>`; the wrappers stay only for the docs and
    local use.
    """
    for platform in PLATFORMS:
        text = (REPO_ROOT / ".github" / "configs" / f"{platform}.yml").read_text(
            encoding="utf-8"
        )
        assert "setup_script: .github/scripts/set_env.sh" in text, platform
        assert f"setup_platform: {platform}" in text, platform


def test_common_workflows_pass_the_platform():
    """The config-driven wheel build calls the runner with --platform."""
    text = (WORKFLOWS / "build-wheel-common.yml").read_text(encoding="utf-8")
    assert (
        'bash "${{ inputs.setup_script }}" --platform "${{ inputs.setup_platform }}"'
        in text
    )
    assert "      setup_platform:" in text


def test_config_discovery_exports_the_platform():
    """load-platform-config.yml must declare setup_platform as a job output.

    It is read into the config step's GITHUB_OUTPUT, but `needs.<job>.outputs`
    only exposes what the job's `outputs:` block lists; without this the
    pipeline receives an empty platform and the runner exits 2.
    """
    text = (WORKFLOWS / "load-platform-config.yml").read_text(encoding="utf-8")
    assert "setup_platform: ${{ steps.config.outputs.setup_platform }}" in text
    assert 'emit setup_platform "$SETUP_PLATFORM"' in text


def test_runner_applies_the_pip_constraints():
    """The runner exports PIP_CONSTRAINT; the file exists and pins the known drift.

    The constraints file is what fixes the transitive pulls the pin table cannot
    reach, so it must be applied to every install (the runner exports it before
    sourcing the hook) and must still name them.
    """
    runner = RUNNER.read_text(encoding="utf-8")
    assert 'export PIP_CONSTRAINT="${REPO_ROOT}/.github/constraints.txt"' in runner
    constraints = (REPO_ROOT / ".github" / "constraints.txt").read_text(
        encoding="utf-8"
    )
    for pin in ("torch==", "numpy<2", "transformers>="):
        assert pin in constraints, pin


def test_mps_hook_installs_the_wheel_build_dependencies():
    """The MPS hook provisions what `python -m build --no-isolation` reads.

    The job builds the wheel with `--no-isolation`, which resolves the
    `[build-system]` requirements against the venv instead of provisioning them
    in a throwaway env. `pyproject.toml` asks for `setuptools`, `wheel` and
    `cmake>=3.18`; a stock macOS runner image has none of them in the venv, so
    the step failed with `ERROR Unmet dependencies ... found: not installed` and
    the lane never reached a single test.

    `cmake` needs the explicit guard rather than a plain `import cmake`: the repo
    ships its own top-level `cmake/` directory, which Python resolves as a
    namespace package from the repo root, so an import check passes in a venv
    where the real distribution is absent and the build then dies later at
    `running build_ext` with `error: [Errno 2] No such file or directory:
    'cmake'`. That is the same trap the Ascend hook's comment records.
    """
    hook = (HOOKS / "set_env_mps.sh").read_text(encoding="utf-8")
    # The comments above the install name `cmake` and `wheel` too, so this reads
    # the commands rather than the file: the requirement has to appear on a line
    # that installs it, which a comment cannot satisfy.
    installs = [
        line.strip()
        for line in hook.splitlines()
        if line.strip().startswith("pip_retry ")
    ]
    assert installs, "the mps hook installs nothing"

    # `torch` is exempt: the hook installs it through `install_cpu_torch`, which
    # names the version from a variable, so the literal never appears on a line.
    # The three wheel-build deps are what the runner image is missing.
    for requirement in ("setuptools", "cmake", "wheel"):
        assert requirement in _pyproject_build_requirements(), (
            f"{requirement} is no longer a [build-system] requirement; this "
            "test pins the hook to the file's list and needs updating"
        )
        assert any(requirement in line for line in installs), (
            f"{requirement} is a [build-system] requirement but the mps hook "
            f"never installs it; installs found: {installs}"
        )


def test_runner_resolves_repository_shorthand_names():
    runner = RUNNER.read_text(encoding="utf-8")
    assert "${!prefix[@]}" not in runner


def test_runner_persists_the_environment_with_bash_3_2(tmp_path):
    """Saving the environment must not use bash 4's `${!name}` indirection.

    macOS runners invoke the script with /bin/bash 3.2; the machine running
    these tests may have a newer bash on PATH.
    """
    common = SCRIPTS / "lib" / "set_env_common.sh"
    assert "${!name}" not in common.read_text(encoding="utf-8")
    macos_bash = Path("/bin/bash")
    shell = str(macos_bash) if macos_bash.exists() else "bash"
    output = tmp_path / "github-env"
    done = subprocess.run(
        [
            shell,
            "-c",
            f'set -u; source "{common}"; TEST_VALUE=--help; export_ci_env TEST_VALUE',
        ],
        capture_output=True,
        text=True,
        env={
            "GITHUB_ENV": str(output),
            "PATH": "/usr/bin:/bin",
        },
    )
    assert done.returncode == 0, done.stderr
    assert output.read_text() == "TEST_VALUE=--help\n"


def test_mps_hook_selects_the_metal_buffer_wrapper_before_torch():
    """The framework Python carries the buffer wrapper used by MPS."""
    hook = (HOOKS / "set_env_mps.sh").read_text(encoding="utf-8")
    assert "METAL_DEVICE_WRAPPER_TYPE=1" in hook
    assert "MetalTools.framework" in hook
    assert "METAL_DEVICE_WRAPPER_TYPE=0" in hook
    assert hook.index("METAL_DEVICE_WRAPPER_TYPE_WAS_SET=0") < hook.index(
        "import torch"
    )

    workflow = (WORKFLOWS / "integration-tests.yml").read_text(encoding="utf-8")
    assert "METAL_DEVICE_WRAPPER_TYPE: ${{ env.METAL_DEVICE_WRAPPER_TYPE }}" in (
        workflow
    )
    assert "PYTORCH_MPS_HIGH_WATERMARK_RATIO=" in workflow
    assert 'if [[ -n "$TORCH_FL_MPS_HIGH_WATERMARK_RATIO" ]]' in workflow


def test_mps_hook_removes_the_upper_limit_and_probes_a_real_allocation():
    """Setup must test the allocator in the same venv as the integration lane."""
    hook = (HOOKS / "set_env_mps.sh").read_text(encoding="utf-8")
    assert "TORCH_FL_MPS_HIGH_WATERMARK_RATIO=0.0" in hook
    assert "TORCH_FL_MPS_LOW_WATERMARK_RATIO=0.5" in hook
    probe = hook.index('torch.arange(12, dtype=torch.float32, device="mps")')
    assert hook.index("install_cpu_torch") < probe < hook.index("build_flagos_inplace")
    assert 'b.copy_(a.reshape(3, 4))' in hook

    workflow = (WORKFLOWS / "integration-tests.yml").read_text(encoding="utf-8")
    assert (
        'export PYTORCH_MPS_HIGH_WATERMARK_RATIO='
        '"$TORCH_FL_MPS_HIGH_WATERMARK_RATIO"'
    ) in workflow
    assert (
        'export PYTORCH_MPS_LOW_WATERMARK_RATIO='
        '"$TORCH_FL_MPS_LOW_WATERMARK_RATIO"'
    ) in workflow
    assert ':="${TORCH_FL_MPS_' not in workflow


def _pyproject_build_requirements() -> list[str]:
    """The requirement names in `[build-system].requires`, normalised.

    Read the real file rather than restating the list and drifting from it. The
    DCU runner still uses Python 3.10, so use the tomli backport when tomllib is
    unavailable.
    """
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib

    with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
        requires = tomllib.load(handle)["build-system"]["requires"]

    names = []
    for requirement in requires:
        name = requirement
        for separator in ("[", ">", "<", "=", "!", "~", " "):
            name = name.split(separator, 1)[0]
        names.append(name)
    return names
