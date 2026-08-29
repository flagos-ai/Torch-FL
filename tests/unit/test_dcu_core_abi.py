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

"""Unit coverage for the DCU decoupled-core ABI guard.

Pure logic over ``nm`` output: no DTK, no GPU. The guard is what stops a DTK
upgrade from silently shipping a wheel whose device libraries import ATen symbols
the official PyTorch core does not define. It rejects every new unaccounted import
while allowing the manifest's known compatibility superset: a plugin built against
official headers imports fewer DTK-private wrappers than one built against DTK's
patched headers.
"""

import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "check_dcu_core_abi.py"

_spec = importlib.util.spec_from_file_location("check_dcu_core_abi", SCRIPT)
abi = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(abi)


def _compile_so(path: Path, source: str) -> None:
    """Build a tiny .so so the guard runs against real nm output, not a fake."""
    cc = shutil.which("cc") or shutil.which("gcc")
    if cc is None:
        pytest.skip("no C compiler available")
    src = path.with_suffix(".c")
    src.write_text(textwrap.dedent(source), encoding="utf-8")
    subprocess.run(
        [cc, "-shared", "-fPIC", "-o", str(path), str(src)],
        check=True,
        capture_output=True,
    )


@pytest.fixture
def abi_tree(tmp_path):
    """A miniature vendor/official/shim layout with one vendor-private symbol.

    ``vendor_only`` stands in for DTK's DTK-only ATen wrapper: exported by the
    vendor core, imported by the device lib, absent from the official core.
    ``shared`` stands for every upstream symbol, which must never be reported.
    """
    vendor = tmp_path / "vendor"
    official = tmp_path / "official"
    vendor.mkdir()
    official.mkdir()

    _compile_so(
        vendor / "libc10.so",
        "int vendor_only(void) { return 1; } int shared(void) { return 2; }",
    )
    _compile_so(vendor / "libtorch_cpu.so", "int vendor_cpu_side(void) { return 3; }")
    _compile_so(official / "libc10.so", "int shared(void) { return 2; }")
    _compile_so(official / "libtorch_cpu.so", "int other(void) { return 4; }")

    device = (
        "extern int vendor_only(void); extern int shared(void);\n"
        "int use(void) { return vendor_only() + shared(); }"
    )
    _compile_so(vendor / "libc10_hip.so", device)
    _compile_so(vendor / "libtorch_hip.so", "int hip_noop(void) { return 0; }")
    _compile_so(tmp_path / "shim.so", "int vendor_only(void) { return 1; }")

    manifest = tmp_path / "manifest.txt"
    manifest.write_text("# comment line\nvendor_only\n\n", encoding="utf-8")
    return {
        "vendor": vendor,
        "official": official,
        "shim": tmp_path / "shim.so",
        "manifest": manifest,
        "root": tmp_path,
    }


def test_gap_is_exactly_the_vendor_private_import(abi_tree):
    gap = abi.vendor_core_gap(abi_tree["vendor"], abi_tree["official"])
    assert gap == {"vendor_only"}


def test_symbols_present_in_official_core_are_not_reported(abi_tree):
    # `shared` is imported by the device lib and exported by the vendor core, but
    # the official core has it too -- it needs no shim entry.
    assert "shared" not in abi.vendor_core_gap(abi_tree["vendor"], abi_tree["official"])


def test_verify_passes_when_manifest_and_shim_agree(abi_tree):
    gap = abi.verify_core_abi(
        abi_tree["vendor"], abi_tree["official"], abi_tree["shim"], abi_tree["manifest"]
    )
    assert gap == {"vendor_only"}


def test_new_unaccounted_import_fails(abi_tree):
    abi_tree["manifest"].write_text("# nothing expected\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="new vendor-core-only imports"):
        abi.verify_core_abi(
            abi_tree["vendor"],
            abi_tree["official"],
            abi_tree["shim"],
            abi_tree["manifest"],
        )


def test_known_manifest_superset_is_allowed(abi_tree):
    """Official-header builds legitimately import only the device-side family.

    The real manifest has 32 symbols: 16 ``native_fuse_*`` entries always imported
    by ``libtorch_hip.so``, plus 16 ``fuse_*`` entries imported only when the plugin
    was compiled against DTK's patched headers. CI compiles against official headers,
    so requiring exact equality rejects that valid 16-symbol shape.
    """
    _compile_so(
        abi_tree["shim"],
        "int vendor_only(void) { return 1; } int plugin_only(void) { return 2; }",
    )
    abi_tree["manifest"].write_text("vendor_only\nplugin_only\n", encoding="utf-8")
    gap = abi.verify_core_abi(
        abi_tree["vendor"],
        abi_tree["official"],
        abi_tree["shim"],
        abi_tree["manifest"],
    )
    assert gap == {"vendor_only"}


def test_shim_missing_a_manifest_symbol_fails(abi_tree):
    _compile_so(abi_tree["shim"], "int unrelated(void) { return 0; }")
    with pytest.raises(RuntimeError, match="does not export every manifest symbol"):
        abi.verify_core_abi(
            abi_tree["vendor"],
            abi_tree["official"],
            abi_tree["shim"],
            abi_tree["manifest"],
        )


def test_plugin_imports_are_folded_into_the_same_gap(abi_tree):
    """libtorch_fl.so's own DTK-private imports must be caught too.

    This is the regression that made `import torch_fl._C` fail outright: the
    plugin compiles against DTK's patched ATen headers and so imports a second
    family of DTK-only wrappers that no device lib references.
    """
    plugin = abi_tree["root"] / "libtorch_fl.so"
    _compile_so(
        plugin,
        "extern int vendor_cpu_side(void);\nint plugin_use(void) { return vendor_cpu_side(); }",
    )

    without = abi.vendor_core_gap(abi_tree["vendor"], abi_tree["official"])
    assert "vendor_cpu_side" not in without

    with_plugin = abi.vendor_core_gap(abi_tree["vendor"], abi_tree["official"], plugin)
    assert with_plugin == {"vendor_only", "vendor_cpu_side"}

    # And the manifest that passed without --plugin must now fail.
    with pytest.raises(RuntimeError, match="vendor_cpu_side"):
        abi.verify_core_abi(
            abi_tree["vendor"],
            abi_tree["official"],
            abi_tree["shim"],
            abi_tree["manifest"],
            plugin,
        )


def test_missing_input_is_reported_not_silently_skipped(abi_tree):
    with pytest.raises(FileNotFoundError):
        abi.vendor_core_gap(abi_tree["root"] / "absent", abi_tree["official"])


def test_manifest_comments_and_blank_lines_are_ignored(tmp_path):
    path = tmp_path / "m.txt"
    path.write_text("# header\n\n  _Z3foo\n\n  # indented comment\n_Z3bar\n", "utf-8")
    assert abi.read_manifest(path) == {"_Z3foo", "_Z3bar"}


def test_cli_reports_success(abi_tree):
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--vendor-lib",
            str(abi_tree["vendor"]),
            "--official-lib",
            str(abi_tree["official"]),
            "--shim",
            str(abi_tree["shim"]),
            "--manifest",
            str(abi_tree["manifest"]),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "1 vendor-private imports are covered" in proc.stdout


def test_cli_fails_with_nonzero_exit(abi_tree):
    abi_tree["manifest"].write_text("", encoding="utf-8")
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--vendor-lib",
            str(abi_tree["vendor"]),
            "--official-lib",
            str(abi_tree["official"]),
            "--shim",
            str(abi_tree["shim"]),
            "--manifest",
            str(abi_tree["manifest"]),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 1
    assert "error:" in proc.stderr


def test_shipped_manifest_covers_both_symbol_families():
    """The checked-in manifest must stay in step with the shim source.

    Both are hand-maintained lists of mangled names; a symbol added to one and
    not the other is only caught at bundle time on a DTK machine otherwise.
    """
    manifest = abi.read_manifest(
        REPO_ROOT / "torch_fl" / "accelerator" / "dcu" / "dtk_core_compat_symbols.txt"
    )
    source = (
        REPO_ROOT / "csrc" / "runtime" / "accelerator" / "dcu" / "dtk_core_compat.cc"
    ).read_text(encoding="utf-8")

    missing = sorted(sym for sym in manifest if sym not in source)
    assert not missing, f"manifest symbols absent from the shim source: {missing}"

    native = {s for s in manifest if "native_fuse" in s}
    public = manifest - native
    assert len(native) == 16, sorted(native)
    assert len(public) == 16, sorted(public)
