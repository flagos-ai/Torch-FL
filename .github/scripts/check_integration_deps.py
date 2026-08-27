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

"""Verify the interpreter that will run the tests can import what they need.

Called from every platform's set_env script, including when it adopted a
prebuilt venv. A venv baked without pytest or transformers turns the functional
groups into an environment failure wearing a platform failure's clothes, and the
resulting log blames the vendor backend for a missing pip install.

All dependencies declared with --require are mandatory: missing means the job
stops in setup with an unambiguous message.

For triton specifically, --triton-backend requests an *advisory* validation:
importability, version, backend registry key, and driver availability are
probed and any gap is reported as a ::warning::, but the check never fails the
job. Triton is only needed by torch.compile (the compile-tests group); a missing
or wrong triton must not abort setup and take the other nine groups down with
no evidence at all. compile-tests fails on its own when triton is unavailable,
and the operator, RNG, AMP, profiler, inference and training groups still
report. Stock PyPI triton can be imported but has 0 active drivers, so the probe
goes past importlib.util.find_spec and inspects the driver registry too.

Usage:
    python .github/scripts/check_integration_deps.py --require pytest transformers safetensors
    python .github/scripts/check_integration_deps.py --require pytest transformers safetensors --triton-backend ascend
"""

import argparse
import importlib.util
import sys


def _advise_triton_backend(backend_name):
    """Probe vendor triton and emit ::warning:: for any gap, never failing.

    A missing or misconfigured triton only affects torch.compile (the
    compile-tests group); the operator, RNG, AMP, profiler, inference and
    training groups do not need it. Hard-failing on triton would abort setup
    and take all ten groups down with zero evidence. Instead this prints the
    triton status (importable? version? active drivers? expected backend key?)
    as warnings so the gap is visible in the log, and lets compile-tests fail
    on its own.

    Stock PyPI triton imports but has 0 active drivers and is not a substitute
    for the vendor build.
    """
    try:
        import triton
    except ImportError as e:
        print(
            f"::warning::triton not importable ({e}); torch.compile will fail "
            f"until triton-{backend_name} is baked into the image. The other "
            f"test groups are unaffected."
        )
        return

    print(f"triton.__file__ = {triton.__file__}")
    print(f"triton.__version__ = {triton.__version__}")

    if not hasattr(triton.runtime, "driver"):
        print(
            f"::warning::triton.runtime.driver not found; cannot validate "
            f"triton-{backend_name} backend. compile-tests may fail."
        )
        return

    drivers = getattr(triton.runtime.driver, "get_active_drivers", lambda: [])()
    if not drivers:
        drivers = getattr(triton.runtime.driver, "get_drivers", lambda: [])()

    print(f"triton active drivers: {drivers if drivers else '(none)'}")

    if not drivers:
        print(
            f"::warning::triton has 0 active drivers; stock PyPI triton is not "
            f"a substitute for triton-{backend_name}. compile-tests will fail; "
            f"the other test groups are unaffected."
        )
        return

    expected_key = backend_name.lower()
    if expected_key not in [d.lower() for d in drivers]:
        print(
            f"::warning::expected triton backend '{expected_key}' not in active "
            f"drivers: {drivers}. compile-tests may fail."
        )
        return

    print(f"triton backend '{expected_key}' validated successfully")


def _missing(names):
    missing = []
    for name in names:
        try:
            found = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            # A package whose parent is absent, or a name that cannot be a spec.
            found = False
        if not found:
            missing.append(name)
    return missing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--require",
        nargs="*",
        default=[],
        metavar="MODULE",
        help="Modules that must be importable (exit non-zero if absent)",
    )
    parser.add_argument(
        "--triton-backend",
        metavar="NAME",
        help="Advisory triton validation (ascend, metax, cuda, etc.): probe "
        "importability, version, backend registry and drivers, emitting "
        "::warning:: on any gap without failing the job",
    )
    args = parser.parse_args()

    print(f"Dependency check interpreter: {sys.executable}")

    all_errors = []

    # Check required modules (mandatory: missing aborts setup).
    required_missing = _missing(args.require)
    present = [n for n in args.require if n not in required_missing]
    if present:
        print(f"Required modules present: {', '.join(present)}")

    if required_missing:
        all_errors.append(
            f"Cannot import required modules: {', '.join(required_missing)}"
        )

    # Advisory triton probe. A missing vendor Triton only breaks torch.compile,
    # not the other nine groups, so this warns rather than fails. See the module
    # docstring for the full rationale.
    if args.triton_backend:
        _advise_triton_backend(args.triton_backend)

    if all_errors:
        print("\n=== Dependency Check Failed ===")
        for error in all_errors:
            print(f"::error::{error}")
        print(
            "\nInstall missing dependencies in the platform setup script or bake them "
            "into the CI image."
        )
        return 1

    print("\n✓ All dependency checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
