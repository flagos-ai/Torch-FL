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
"""Locate an exact-version Transformers source tree for the official tests.

The published wheel does not contain ``tests/``.  The runner therefore uses the
source archive for the same release as the installed wheel.  The archive is
kept intact while validating paths and links because official tests import
repository fixtures outside ``tests/`` as well as shared files inside it.
"""

from __future__ import annotations

import json
import os
import shutil
import tarfile
import tempfile
import urllib.request
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path

PYPI_TRANSFORMERS_JSON = "https://pypi.org/pypi/transformers/json"
ARCHIVE_URL = (
    "https://codeload.github.com/huggingface/transformers/tar.gz/refs/tags/v{version}"
)
DEFAULT_CACHE = Path.home() / ".cache" / "torch_fl" / "hf-tests"
CACHE_FORMAT = 2
CACHE_MARKER = ".torch-fl-hf-source"
VERSION_FILE = Path("src") / "transformers" / "__init__.py"


class SourceError(RuntimeError):
    """A source tree could not be obtained, so no test result is attributable."""


def atomic_write(path: Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(value, f, indent=1, sort_keys=True)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def installed_transformers() -> str | None:
    try:
        return package_version("transformers")
    except PackageNotFoundError:
        return None


def latest_transformers(timeout: int = 30) -> str | None:
    """Resolve the newest published release, or None when offline."""
    try:
        with urllib.request.urlopen(PYPI_TRANSFORMERS_JSON, timeout=timeout) as resp:
            return json.load(resp)["info"]["version"]
    except Exception:  # noqa: BLE001 - offline is normal on a vendor box
        return None


def resolve_version(requested: str, offline: bool) -> dict:
    """Require the requested source and installed wheel versions to match."""
    have = installed_transformers()
    if have is None:
        raise SourceError(
            "transformers is not installed in this interpreter.\n"
            "Install it without touching torch:\n"
            "  python -m pip install --no-deps transformers==<version>"
        )
    record = {
        "requested": requested,
        "installed": have,
        "latest": None,
        "version": have,
    }
    if requested == "latest":
        if not offline:
            record["latest"] = latest_transformers()
        return record
    if have != requested:
        raise SourceError(
            f"requested transformers {requested} but {have} is installed.\n"
            "The official tests must run against the version they ship with.\n"
            f"  python -m pip install --no-deps transformers=={requested}"
        )
    return record


def cache_root(cache_dir: str | os.PathLike[str] | None = None) -> Path:
    """Where source trees are kept, one directory per Transformers version."""
    if cache_dir:
        return Path(cache_dir).expanduser()
    if value := os.environ.get("HF_COVERAGE_CACHE"):
        return Path(value).expanduser()
    return DEFAULT_CACHE


def source_dir(version: str, cache_dir: str | os.PathLike[str] | None = None) -> Path:
    return cache_root(cache_dir) / f"transformers-{version}"


def source_version(path: str | os.PathLike[str]) -> str | None:
    """Read the version declared by a source tree, or None if it is absent."""
    try:
        text = (Path(path) / VERSION_FILE).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("__version__"):
            _, _, value = stripped.partition("=")
            return value.strip().strip("\"'") or None
    return None


def _cache_marker(path: Path) -> dict | None:
    try:
        value = json.loads((path / CACHE_MARKER).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def is_usable(path: str | os.PathLike[str], version: str) -> bool:
    """Only reuse complete archives created by the current cache format."""
    root = Path(path)
    marker = _cache_marker(root)
    return bool(
        marker
        and marker.get("format") == CACHE_FORMAT
        and marker.get("version") == version
        and (root / "tests" / "models").is_dir()
        and source_version(root) == version
    )


def _safe_members(tar: tarfile.TarFile, prefix: str):
    """Yield archive members whose paths and links stay below ``prefix``."""
    root = f"{prefix}/"
    for member in tar.getmembers():
        name = member.name
        if name == prefix:
            relative = "."
        else:
            if not name.startswith(root):
                raise SourceError(
                    f"archive member is outside its top-level directory: {name}"
                )
            relative = name[len(root) :]
            resolved = os.path.normpath(relative)
            if (
                not relative
                or os.path.isabs(relative)
                or resolved == ".."
                or resolved.startswith("../")
            ):
                raise SourceError(f"refusing to extract unsafe archive member: {name}")

        if member.issym():
            link_target = member.linkname
            resolved = os.path.normpath(
                os.path.join(os.path.dirname(relative), link_target)
            )
            if (
                not link_target
                or os.path.isabs(link_target)
                or resolved == ".."
                or resolved.startswith("../")
            ):
                raise SourceError(f"refusing to extract unsafe archive link: {name}")
        elif member.islnk():
            # Tar hard-link targets are archive paths, unlike symlink targets,
            # but accept a prefix-relative form as well for portability.
            link_target = member.linkname
            if link_target.startswith(root):
                link_target = link_target[len(root) :]
            resolved = os.path.normpath(link_target)
            if (
                not link_target
                or os.path.isabs(link_target)
                or resolved == ".."
                or resolved.startswith("../")
            ):
                raise SourceError(f"refusing to extract unsafe archive link: {name}")

        yield member


def _archive_prefix(names: list[str]) -> str:
    prefixes = {name.split("/", 1)[0] for name in names if name}
    if len(prefixes) != 1:
        raise SourceError("archive must contain exactly one top-level directory")
    return prefixes.pop()


def _extract_complete_archive(archive: Path, staging: Path) -> Path:
    with tarfile.open(archive, "r:gz") as tar:
        names = tar.getnames()
        if not names:
            raise SourceError(f"archive {archive} is empty")
        prefix = _archive_prefix(names)
        tar.extractall(staging, members=list(_safe_members(tar, prefix)))
    return staging / prefix


def _write_cache_marker(root: Path, version: str) -> None:
    atomic_write(root / CACHE_MARKER, {"format": CACHE_FORMAT, "version": version})


def _validate_extracted(root: Path, archive: Path) -> str:
    version = source_version(root)
    if not (root / "tests" / "models").is_dir() or version is None:
        raise SourceError(
            f"archive {archive} is missing the official tests or version marker"
        )
    return version


# Keep extraction in a small function so the security-sensitive member filter
# remains separately testable and the cache operation stays easy to audit.
def extract_archive(
    archive: str | os.PathLike[str], dest: str | os.PathLike[str]
) -> Path:
    """Safely extract the complete archive into an atomic versioned cache."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f"{dest.name}.partial-", dir=dest.parent))
    try:
        unpacked = _extract_complete_archive(Path(archive), staging)
        version = _validate_extracted(unpacked, Path(archive))
        _write_cache_marker(unpacked, version)
        if dest.exists():
            shutil.rmtree(dest)
        os.replace(unpacked, dest)
    except tarfile.TarError as exc:
        raise SourceError(f"could not read archive {archive}: {exc}") from exc
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return dest


def download_archive(
    version: str, dest: str | os.PathLike[str], timeout: int = 300
) -> Path:
    url = ARCHIVE_URL.format(version=version)
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f"{dest.name}.", dir=dest.parent)
    os.close(fd)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp, open(
            tmp, "wb"
        ) as out:
            shutil.copyfileobj(resp, out)
        os.replace(tmp, dest)
    except Exception as exc:  # noqa: BLE001 - network and HTTP errors read alike here
        raise SourceError(
            f"could not download the transformers {version} source from {url}: {exc}\n"
            "Provide a prepared tree with --source-dir, or point --cache-dir at one."
        ) from exc
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return dest


def ensure_source(
    version: str,
    cache_dir: str | os.PathLike[str] | None = None,
    offline: bool = False,
    timeout: int = 300,
) -> dict:
    """Return a complete exact-version source tree, downloading when needed."""
    target = source_dir(version, cache_dir)
    if is_usable(target, version):
        return {"path": str(target), "version": version, "downloaded": False}
    if offline:
        raise SourceError(
            f"no complete cached transformers {version} source at {target} and --offline "
            "was requested. Prepare the tree once with network access, or pass --source-dir."
        )
    archive = cache_root(cache_dir) / f"transformers-v{version}.tar.gz"
    download_archive(version, archive, timeout=timeout)
    extract_archive(archive, target)
    if not is_usable(target, version):
        raise SourceError(
            f"downloaded source at {target} declares transformers {source_version(target)}, "
            f"expected {version}"
        )
    return {"path": str(target), "version": version, "downloaded": True}


def use_source(
    version: str,
    source_dir_override: str | os.PathLike[str] | None,
    cache_dir: str | os.PathLike[str] | None,
    offline: bool,
    timeout: int = 300,
) -> dict:
    """Resolve either an explicitly supplied checkout or the versioned cache."""
    if source_dir_override:
        path = Path(source_dir_override).expanduser().resolve()
        found = source_version(path)
        if not (path / "tests" / "models").is_dir():
            raise SourceError(f"{path} has no tests/models directory")
        if found != version:
            raise SourceError(
                f"source tree {path} declares transformers {found}, but {version} is installed."
            )
        return {"path": str(path), "version": version, "downloaded": False}
    return ensure_source(version, cache_dir=cache_dir, offline=offline, timeout=timeout)
