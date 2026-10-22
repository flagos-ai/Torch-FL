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

Nothing here may need more than the standard library of Python 3.10, because
.github/scripts/run_unit_tests.py runs this whole directory file by file with
whatever interpreter the platform's image provides -- and the Hygon DCU image's
is 3.10.12. Standard-library entries newer than that (`tomllib` is 3.11) are
not available there; see `_pyproject_build_requirements`.
"""

import re
import subprocess
from pathlib import Path

import pytest

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
    "iluvatar",
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
    hard-coding a shell and a platform. The pipeline reads setup_script,
    setup_platform and setup_stage from the config now. It also resolves Bash
    after loading the runner profile: the MPS runner has no `bash` on its
    default PATH, although its profile exports an absolute interpreter path.
    """
    text = (WORKFLOWS / "integration-tests.yml").read_text(encoding="utf-8")
    assert "SETUP_BASH: ${{ env.bash || '/bin/bash' }}" in text
    assert "container_image=$(python3 -c" in text
    assert "container_workspace=$(python3 -c" in text
    assert "env -u BASH_ENV /bin/bash -c" in text
    assert 'source "$INTEGRATION_PROFILE"' in text
    assert (
        'unset CI_WORKSPACE; create_setup_entry "$5" "$3"; '
        'entry=$(cat "$5"); exec env -u BASH_ENV bash -c "$entry" "$6"' in text
    )
    assert 'source "$1" </dev/null; source "$2"' in text
    assert "create_setup_entry" in text
    assert "SETUP_PLATFORM: ${{ needs.configuration.outputs.setup_platform }}" in text
    assert "command -v bash" in text
    assert '[[ "$selected_bash" = /* ]]' in text
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


@pytest.mark.parametrize("platform", ["gcu", "kunlunxin"])
def test_onnx_platforms_install_the_shared_numpy_pin(platform):
    hook = (HOOKS / f"set_env_{platform}.sh").read_text(encoding="utf-8")
    assert "pip_retry --index-url \"$PIP_INDEX_URL_ARG\" 'numpy<2'" in hook


def test_ascend_resolves_flag_gems_without_dropping_the_standard_library():
    hook = (HOOKS / "set_env_ascend.sh").read_text(encoding="utf-8")
    assert "sys.path[:] =" not in hook
    assert 'distribution("flag-gems").locate_file("flag_gems")' in hook
    assert (
        "for site in sorted(sites, reverse=True):\n    sys.path.insert(0, str(site))"
        in hook
    )
    assert "importlib.invalidate_caches()" in hook


def test_ppu_children_load_torch_fl_before_torch():
    hook = (HOOKS / "set_env_ppu.sh").read_text(encoding="utf-8")
    parent = "import torch_fl  # noqa: F401\nimport torch\n\ntorch_path"
    child = (
        "import os, torch_fl; import torch; "
        "assert torch.version.cuda == '13.0', torch.version.cuda; "
        "print(os.path.realpath(torch.__file__))"
    )
    child_check = hook.index('"import os, torch_fl; import torch; "')
    assert parent in hook[:child_check]
    for statement in child.split("; "):
        assert statement.strip() in hook[child_check:]


def test_flag_gems_hooks_install_its_yaml_dependency_before_the_probe():
    for platform in ("metax", "ppu"):
        hook = (HOOKS / f"set_env_{platform}.sh").read_text(encoding="utf-8")
        probe = hook.rindex("import flag_gems")
        assert hook.index("pyyaml==6.0.1") < probe, platform


def test_ppu_checks_runtime_cuda_metadata_after_loading_its_facade():
    """The stock wheel's version.py reports None until torch_fl loads PPU libs."""
    hook = (HOOKS / "set_env_ppu.sh").read_text(encoding="utf-8")
    setup = hook.split("if ! command -v nvidia-smi", 1)[0]
    checks = hook.split("import torch_fl  # noqa: F401", 1)[1]
    assert checks.count('assert torch.version.cuda == "13.0"') == 1
    assert "assert torch.version.cuda is None" not in checks
    stock_check = setup.split('CPU_TORCH_ROOT="$("$VENV_PYTHON"', 1)[1].split(
        "\nPY", 1
    )[0]
    assert "import torch_fl" not in stock_check
    assert "assert torch.version.cuda is None" in stock_check
    assert 'assert torch.version.cuda == "13.0"' not in stock_check


def test_ppu_verification_uses_the_isolated_interpreter_after_the_import_probe():
    hook = (HOOKS / "set_env_ppu.sh").read_text(encoding="utf-8")
    import_probe = (
        "import torch_fl  # noqa: F401  (selects the bundled PPU libtorch core)"
    )
    child_check = "  \"$VENV_PYTHON\" - <<'PY'\nimport os"
    assert hook.index(import_probe) < hook.index(child_check)


def test_mps_hook_selects_the_host_macos_toolchain():
    """The MPS hook fails before downloading torch when the runner is wrong.

    A macOS build checks the host OS, Apple Silicon, and the Metal SDK before it
    installs anything. Those checks are what stop an x86 runner or a missing Xcode
    image from reaching a test suite that skips every MPS case.
    """
    hook = (HOOKS / "set_env_mps.sh").read_text(encoding="utf-8")
    assert '[[ "$(uname -s)" != "Darwin" ]]' in hook
    assert '[[ "$(uname -m)" != "arm64" ]]' in hook
    assert "xcode-select -p" in hook
    assert "xcrun --sdk macosx --show-sdk-path" in hook
    assert "export FLAGOS_ACCELERATOR=mps" in hook
    assert "export FLAGOS_DISABLE_CUDA_ASSETS=1" in hook
    assert 'CPU_TORCH_INDEX_URL="https://pypi.org/simple"' in hook
    assert 'export PYTHONNOUSERSITE=1\nexport PYTHONPATH=""' in hook


def test_mps_hook_installs_the_wheel_build_dependencies():
    """The MPS hook provisions what `python -m build --no-isolation` reads.

    The job builds the wheel with `--no-isolation`, which resolves the
    `[build-system]` requirements against the venv instead of provisioning them
    in a throwaway env. `pyproject.toml` asks for `setuptools`, `wheel` and
    `cmake>=3.18`; the standard macOS runner image does not provide them inside
    the hook's venv.

    `cmake` needs the explicit guard rather than a plain `import cmake`: the repo
    ships its own top-level `cmake/` directory, which Python resolves as a
    namespace package from the repo root, so an import check passes in a venv
    where the real distribution is absent and the build then dies later at
    `running build_ext`.
    """
    hook = (HOOKS / "set_env_mps.sh").read_text(encoding="utf-8")
    installs = [
        line.strip()
        for line in hook.splitlines()
        if line.strip().startswith("pip_retry ")
    ]
    assert installs, "the mps hook installs nothing"

    # `torch` is exempt: the hook installs it through `install_cpu_torch`.
    for requirement in ("setuptools", "cmake", "wheel"):
        assert requirement in _pyproject_build_requirements(), (
            f"{requirement} is no longer a [build-system] requirement; this "
            "test pins the hook to the file's list and needs updating"
        )
        assert any(requirement in line for line in installs), (
            f"{requirement} is a [build-system] requirement but the mps hook "
            f"never installs it; installs found: {installs}"
        )


TEST_ONLY_PACKAGES = {
    "onnx",
    "onnxruntime",
    "jq",
    "transformers",
    "huggingface-hub",
    "hf-xet",
    "tokenizers",
    "safetensors",
    "protobuf",
    "ml_dtypes",
    "coloredlogs",
    "humanfriendly",
    "flatbuffers",
    "regex",
    "requests",
    "urllib3",
    "charset-normalizer",
    "certifi",
    "idna",
    "httpx",
    "httpcore",
    "h11",
    "anyio",
    "diffusers",
    "pillow",
    "pyyaml",
    "typer",
    "shellingham",
    "annotated-doc",
    "rich",
    "markdown-it-py",
    "pygments",
    "mdurl",
    "tqdm",
    "filelock",
    "packaging",
    "importlib-metadata",
    "zipp",
    "click",
    "fsspec",
    "typing-extensions",
    "sympy",
    "sentencepiece",
    "tiktoken",
}


def test_platform_hooks_leave_the_shared_test_cohort_to_the_pipeline():
    """A platform hook must not install a different version of a shared package.

    The pipeline installs the cohort with --no-deps after the hook. pip leaves
    an already-satisfied version alone, so an earlier hook pin such as
    transformers<5 survives the cohort install and fails its version check.
    """
    for platform in PLATFORMS:
        hook = (HOOKS / f"set_env_{platform}.sh").read_text(encoding="utf-8")
        installs = [
            line
            for line in hook.splitlines()
            if re.search(r"\bpip(?:_retry)?\b.*\binstall\b", line)
        ]
        for requirement in TEST_ONLY_PACKAGES:
            if requirement == "pyyaml" and platform in {"metax", "ppu"}:
                continue
            pattern = rf"(?<![\w-]){re.escape(requirement)}(?![\w-])"
            conflicts = [line for line in installs if re.search(pattern, line)]
            assert not conflicts, f"{platform}: {requirement} in {conflicts}"


def test_ppu_setup_installs_flag_gems_sqlalchemy_before_torch_probe():
    hook = (HOOKS / "set_env_ppu.sh").read_text(encoding="utf-8")
    setup = hook.split('CPU_TORCH_ROOT="$("$VENV_PYTHON" - <<\'PY\'', 1)[0]
    integration = setup.split('if [[ "$CI_STAGE" == "integration" ]]; then', 1)[1]

    assert '"sqlalchemy==2.0.48"' in integration
    assert "import flag_gems" not in setup


def _pyproject_build_requirements() -> list[str]:
    """The requirement names in `[build-system].requires`, normalised.

    Parsed without a TOML library. `tomllib` is stdlib only from Python 3.11,
    and .github/scripts/run_unit_tests.py runs this file in its own process on
    whatever interpreter the platform's image provides -- the Hygon DCU image's
    is 3.10.12, where `import tomllib` raises ModuleNotFoundError and took the
    whole DCU unit-test group down. The backport (`tomli`) is not a declared
    test dependency either, so it cannot be the fallback.

    Only one key of one table is needed, so this reads that table and nothing
    else: basic strings with TOML's escapes, single-line literal strings, and
    the `\\` line continuation -- the three forms a dependency specifier is
    written in -- and raises on anything else rather than guessing. It raises a
    plain AssertionError because the file is a contract this test pins: a
    `[build-system]` the parser cannot read means the wrong thing is being
    checked, which is a failure here and not a skip.
    """
    import re

    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    table = re.search(r"^\[build-system\]\s*$", text, re.MULTILINE)
    assert table is not None, "pyproject.toml has no [build-system] table"

    body = text[table.end() :]
    next_table = re.search(r"^\[", body, re.MULTILINE)
    if next_table is not None:
        body = body[: next_table.start()]

    match = re.search(r"^[ \t]*requires[ \t]*=[ \t]*\[", body, re.MULTILINE)
    assert match is not None, "[build-system] has no requires list"

    escapes = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\"}
    items: list[str] = []
    index = match.end()
    while index < len(body):
        char = body[index]
        if char in " \t\r\n,":
            index += 1
        elif char == "]":
            return _requested_names(items)
        elif char in "\"'":
            quote = char
            index += 1
            value = ""
            while True:
                assert index < len(body), f"unterminated string in requires: {value}"
                char = body[index]
                if char == "\\" and quote == '"':
                    following = body[index + 1]
                    if following == "\n":  # line continuation
                        index += 2
                        continue
                    assert following in escapes, f"unsupported escape \\{following}"
                    value += escapes[following]
                    index += 2
                elif char == quote:
                    index += 1
                    break
                elif char == "\n" and quote == "'":
                    raise AssertionError(f"newline in literal string: {value}")
                else:
                    value += char
                    index += 1
            items.append(value)
        else:
            raise AssertionError(
                f"unsupported [build-system].requires syntax at {body[index:][:20]!r}"
            )
    raise AssertionError("[build-system].requires list is not closed")


def _requested_names(requirements: list[str]) -> list[str]:
    names = []
    for requirement in requirements:
        name = requirement
        for separator in ("[", ">", "<", "=", "!", "~", " ", ";"):
            name = name.split(separator, 1)[0]
        names.append(name)
    return names
