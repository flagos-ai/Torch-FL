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

"""Read the integration manifest the way CI's loader job does.

The steps a platform runs are no longer written in that platform's config: they
are one ordered list in ``.github/integration-manifest.yml``, which
``load-platform-config.yml`` composes against the config it just read. A test
that wants to know what a platform runs has to ask the same composer, which is
why this module exists -- a second reader would be a second answer.

The composer itself is loaded by path rather than imported: it lives under
``.github/scripts/``, which is not a package and is not on ``sys.path``. It
imports nothing but the standard library, so loading it is free, and the same
goes for the environment check it hands its data to.
"""

import importlib.util
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO / ".github/configs"
MANIFEST_PATH = REPO / ".github/integration-manifest.yml"
COMPOSER_PATH = REPO / ".github/scripts/compose_integration_manifest.py"
ENVIRONMENT_CHECK_PATH = REPO / ".github/scripts/check_platform_environment.py"


def load_module(path: Path, name: str):
    """Execute one repository script by path and return the module."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, path
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


composer = load_module(COMPOSER_PATH, "compose_integration_manifest")
environment_check = load_module(ENVIRONMENT_CHECK_PATH, "check_platform_environment")


def platform_names() -> list[str]:
    """Every platform that has a config, with the manifest file left out."""
    return composer.known_platforms(CONFIG_DIR)


def manifest() -> dict:
    """The shared step list, as the composer's own input shape."""
    return yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))


def platform_config(platform: str) -> dict:
    return yaml.safe_load((CONFIG_DIR / f"{platform}.yml").read_text(encoding="utf-8"))


def composed_steps(platform: str) -> list[dict]:
    """The groups one platform runs, in the order it runs them."""
    return composer.compose(
        manifest(), platform_config(platform), platform, platform_names()
    )


def step_ids() -> list[str]:
    """Every manifest id, which is also what every platform runs.

    The platform argument this used to take is gone: the manifest states one
    entry per step and the composer runs all of them everywhere, so an id set
    that varied by platform could only be the old `platforms:`-restricted
    variants coming back.
    """
    return list(composer.validate_manifest(manifest())[1])
