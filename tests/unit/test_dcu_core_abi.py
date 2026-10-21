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
import json
from pathlib import Path
import subprocess
import sys


import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "vendor" / "check_dcu_core_abi.py"

_spec = importlib.util.spec_from_file_location("check_dcu_core_abi", SCRIPT)
abi = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(abi)


def _symbol_from_nm_line(line):
    fields = line.split()
    return fields[-1].split("@", 1)[0] if fields else None


def _nm_table(names):
    rows = ["                 T " + name for name in names]
    return chr(10).join(rows) + chr(10)


@pytest.fixture
def abi_tree(tmp_path, monkeypatch):
    """A miniature vendor/official/shim layout with one vendor-private symbol.

    ``vendor_only`` stands in for DTK's DTK-only ATen wrapper: exported by the
    vendor core, imported by the device lib, absent from the official core.
    ``shared`` stands for every upstream symbol, which must never be reported.
    """
    vendor = tmp_path / "vendor"
    official = tmp_path / "official"
    shim = tmp_path / "shim.so"
    plugin = tmp_path / "libtorch_fl.so"
    for directory, names in (
        (vendor, ("libc10.so", "libtorch_cpu.so", "libc10_hip.so", "libtorch_hip.so")),
        (official, ("libc10.so", "libtorch_cpu.so")),
    ):
        directory.mkdir()
        for name in names:
            (directory / name).touch()
    shim.touch()
    plugin.touch()

    tables = {
        vendor / "libc10.so": {
            "defined": {"vendor_only", "shared"},
            "undefined": set(),
        },
        vendor / "libtorch_cpu.so": {
            "defined": {"vendor_cpu_side", "other"},
            "undefined": set(),
        },
        official / "libc10.so": {"defined": {"shared"}, "undefined": set()},
        official / "libtorch_cpu.so": {"defined": {"other"}, "undefined": set()},
        vendor / "libc10_hip.so": {
            "defined": {"use"},
            "undefined": {"vendor_only", "shared"},
        },
        vendor / "libtorch_hip.so": {"defined": {"hip_noop"}, "undefined": set()},
        shim: {"defined": {"vendor_only"}, "undefined": set()},
        plugin: {"defined": set(), "undefined": {"vendor_cpu_side"}},
    }

    def nm(path, *, defined):
        table = tables.get(path)
        if table is None:
            raise RuntimeError(f"unexpected nm input: {path}")
        names = table["defined" if defined else "undefined"]
        return {_symbol_from_nm_line(line) for line in _nm_table(names).splitlines()}

    monkeypatch.setattr(abi, "_nm_symbols", nm)

    manifest = tmp_path / "manifest.txt"
    manifest.write_text("# comment line\nvendor_only\n\n", encoding="utf-8")
    return {
        "vendor": vendor,
        "official": official,
        "shim": shim,
        "plugin": plugin,
        "tables": tables,
        "manifest": manifest,
        "root": tmp_path,
    }


def test_nm_output_without_an_address_keeps_whole_symbols(abi_tree):
    table = abi_tree["tables"][abi_tree["vendor"] / "libc10.so"]
    output = _nm_table(table["defined"])
    assert len(_symbol_from_nm_line(output.splitlines()[0])) > 1


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
    abi_tree["tables"][abi_tree["shim"]]["defined"].add("plugin_only")
    abi_tree["manifest"].write_text("vendor_only\nplugin_only\n", encoding="utf-8")
    gap = abi.verify_core_abi(
        abi_tree["vendor"],
        abi_tree["official"],
        abi_tree["shim"],
        abi_tree["manifest"],
    )
    assert gap == {"vendor_only"}


def test_shim_missing_a_manifest_symbol_fails(abi_tree):
    abi_tree["tables"][abi_tree["shim"]]["defined"] = {"unrelated"}
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
    plugin = abi_tree["plugin"]

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


def _run_cli(abi_tree, real_abi):
    """Run the CLI through a synthetic ``nm`` while using the real parser."""
    tables = {
        str(path): {side: sorted(names) for side, names in table.items()}
        for path, table in abi_tree["tables"].items()
        if path != abi_tree["plugin"]
    }
    probe = (
        "import importlib.util, json, pathlib, sys; "
        "path, tables, *argv = sys.argv[1:]; "
        "spec = importlib.util.spec_from_file_location('abi_probe', path); "
        "module = importlib.util.module_from_spec(spec); "
        "spec.loader.exec_module(module); "
        "text = lambda names: chr(10).join(names) + chr(10); "
        "module._nm_symbols = lambda path, *, defined: "
        "{line.split()[-1] for line in text(json.loads(tables)[str(path)]["
        "'defined' if defined else 'undefined']).splitlines() if line.split()}; "
        "sys.argv = [path, *argv]; "
        "raise SystemExit(module.main())"
    )
    return subprocess.run(
        [
            sys.executable,
            "-P",
            "-c",
            probe,
            str(Path(real_abi.__file__).resolve()),
            json.dumps(tables),
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
        check=False,
    )


def test_cli_reports_success(abi_tree):
    proc = _run_cli(abi_tree, abi)
    assert proc.returncode == 0, proc.stderr
    assert "1 vendor-private imports are covered" in proc.stdout


def test_cli_fails_with_nonzero_exit(abi_tree):
    abi_tree["manifest"].write_text("", encoding="utf-8")
    proc = _run_cli(abi_tree, abi)
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
