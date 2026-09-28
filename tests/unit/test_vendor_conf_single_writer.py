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

"""Guard: scripts/codegen/gen_vendor_confs.py is the only writer of vendor confs.

tests/unit/test_gen_vendor_confs.py pins what that generator produces. This file
pins that nothing else produces it.

codegen_ops.py used to write backends_metax.conf and backends_dcu.conf itself,
from `op_info` plus a local fallback set (`metax_triton_fallback`,
`dcu_triton_fallback`) in the pre-#272 `cuda | flagos_python` vocabulary. Both
writers overwrote the conf unconditionally, so
`FLAGOS_CODEGEN_ALL=1 python scripts/codegen/codegen_ops.py` -- the full
regeneration command documented in scripts/README.md and
docs/development/testing.md -- silently reverted the pins those platforms
record: mm/bmm/mean.dim on metax, and on dcu the 78 ops the conf holds on cuda,
among them slice_backward, silu_backward and the native_batch_norm pin from
#295/#458. It also dropped matmul, matmul_backward and
scaled_dot_product_attention, which the generator's op universe carries on top
of `op_info`. Measured in issue #459: backends_dcu.conf went 2037 -> 2034 ops,
531 routes moved, 121 `# Note:` lines deleted, and `gen_vendor_confs.py --check`
then failed on three confs.

The fix deletes both writers and leaves the confs to their owner, so a full
regeneration ends with every `torch_fl/configs/backends_<platform>.conf`
byte-identical and a printed command for the case where the run really did move
the coverage ceiling. Both halves are pinned below.

The regression is cheap to reintroduce and easy to miss: the confs are
generated, so a reviewer looking at a large regeneration diff has reason to read
the new contents as expected churn. These checks are source-level on purpose --
they cost no build, no torch import and no hardware, and they fail at the line
that would do the writing.

Run: pytest tests/unit/test_vendor_conf_single_writer.py
"""

import ast
import importlib.util
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CODEGEN_OPS = REPO_ROOT / "scripts" / "codegen" / "codegen_ops.py"
GEN_VENDOR_CONFS = REPO_ROOT / "scripts" / "codegen" / "gen_vendor_confs.py"
CONF_DIR = REPO_ROOT / "torch_fl" / "configs"

# The one conf codegen_ops.py may write: it is the CUDA boxing table, one line
# per wrapper that run generates, and it is deliberately not an input to the
# vendor conf generator -- that one enumerates from the generated
# csrc/aten/generated/register.inc instead (see GENERATED_REGISTER_INC there).
OWNED_BY_CODEGEN_OPS = {"backends_cuda.conf"}

# Sets that fed the two deleted writers. Their names coming back means a second
# routing policy is being grown next to the confs again.
LEGACY_FALLBACK_SETS = {"metax_triton_fallback", "dcu_triton_fallback"}


def _load_generator():
    """Import gen_vendor_confs.py by path -- scripts/ is not a package."""
    spec = importlib.util.spec_from_file_location("gen_vendor_confs", GEN_VENDOR_CONFS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


g = _load_generator()


def _tree(path):
    return ast.parse(path.read_text())


def _string_constants(tree):
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }


def _conf_paths_named(tree):
    """Every torch_fl/configs/backends_<platform>.conf the module spells out."""
    found = set()
    for text in _string_constants(tree):
        found.update(re.findall(r"configs/(backends_[\w.]+\.conf)", text))
    return found


def _assigned_names(tree):
    return {
        target.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }


# ---------------------------------------------------------------------------
# One writer
# ---------------------------------------------------------------------------


def test_codegen_ops_names_no_generated_vendor_conf():
    """codegen_ops.py must not reach for a conf it does not own.

    This is the whole regression: it is not that the writers produced a wrong
    conf, it is that there were two writers at all, and the second one answered
    to a policy set that had drifted away from the confs it overwrote.
    """
    named = _conf_paths_named(_tree(CODEGEN_OPS))
    assert named, (
        "the scan found no conf path in codegen_ops.py at all -- it is meant to "
        "find backends_cuda.conf, so this test has stopped policing anything"
    )
    assert named <= OWNED_BY_CODEGEN_OPS, (
        f"codegen_ops.py names {sorted(named - OWNED_BY_CODEGEN_OPS)}; every "
        "torch_fl/configs/backends_<platform>.conf except backends_cuda.conf is "
        "written by scripts/codegen/gen_vendor_confs.py"
    )


def test_the_cuda_conf_is_not_a_generated_vendor_conf():
    """Keep the exemption honest: cuda is not in the generator's platform set."""
    built = g.build_all(CONF_DIR)
    assert "cuda" not in built
    assert set(built) == set(g.BOXING_PLATFORMS) | set(g.VENDORS)


def test_no_local_vendor_fallback_table_is_left_behind():
    """The policy sets that fed the deleted writers have no other reader.

    A conf's routing diagnosis belongs next to the routing -- the gap sets and
    BOXING_GAP_NOTES in gen_vendor_confs.py -- not in a second table here that
    only a second writer would consult.
    """
    leftover = _assigned_names(_tree(CODEGEN_OPS)) & LEGACY_FALLBACK_SETS
    assert not leftover, (
        f"{sorted(leftover)} reappeared in codegen_ops.py; per-platform FlagGems "
        "fallbacks are stated in gen_vendor_confs.py, which renders the confs"
    )


# ---------------------------------------------------------------------------
# The generator is left to run on its own
# ---------------------------------------------------------------------------


def test_codegen_ops_does_not_run_the_generator():
    """The follow-up conf regeneration stays a separate command.

    Running it from here would make the CUDA-path regeneration a superset of the
    vendor-conf generator: one command would rewrite six confs from whatever
    FlagGems coverage the caller's environment happened to discover, inside a
    diff a reviewer reads as CUDA-binding churn. Measured on the delegation
    variant of this fix, in a container whose flag_gems discovers 533 of the 639
    committed ops: backends_metax.conf moved 95 routes and backends_ppu.conf was
    rewritten too, neither of them on the DCU/CUDA path this script owns.

    The chain is already manual one step upstream -- codegen_tileops.py writes
    the TILEOPS_OPS block of backend_coverage.py, codegen_ops.py writes
    FLAGGEMS_PYTHON_OPS beside it, and neither runs gen_vendor_confs.py, which
    reads both. Keeping the last link the same way costs one command and keeps
    the routing diff separable.
    """
    tree = _tree(CODEGEN_OPS)
    blocked_modules = {"subprocess", "runpy"}
    blocked_calls = {"system", "popen", "execv", "execve", "execvp", "spawnv"}
    shelled_out = {
        f"{node.func.value.id}.{node.func.attr}"
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and (
            node.func.value.id in blocked_modules
            or (node.func.value.id == "os" and node.func.attr in blocked_calls)
        )
    }
    assert not shelled_out, (
        f"codegen_ops.py calls {sorted(shelled_out)}; the vendor confs are "
        "regenerated by running scripts/codegen/gen_vendor_confs.py directly"
    )
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        (node.module or "").split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert not imported & blocked_modules, (
        f"codegen_ops.py imports {sorted(imported & blocked_modules)}"
    )
    assert not any("gen_vendor_confs" in name for name in imported), (
        "gen_vendor_confs.py parses sys.argv at import time and codegen_ops.py is "
        "itself imported by the FlagGems register generators, so importing it "
        "would consume the caller's arguments"
    )


def test_codegen_ops_points_at_the_generator():
    """The hint is the whole handoff: a moved ceiling has to say what to run.

    A runnable command, not a mention. Both files discuss each other in prose,
    so the check is for an invocation the caller can paste -- which is also the
    only thing that makes the exit state of a moved ceiling recoverable.
    """
    command = re.compile(r"python3?\s+scripts/codegen/gen_vendor_confs\.py")
    printed = [
        text for text in _string_constants(_tree(CODEGEN_OPS)) if command.search(text)
    ]
    assert printed, (
        "codegen_ops.py no longer prints a runnable command for refreshing the "
        "vendor confs after a run whose FlagGems coverage moved"
    )


def test_codegen_ops_still_publishes_the_flaggems_coverage():
    """backend_coverage.py is the generator's input and codegen_ops' output.

    The two halves of #459 only work together: gen_vendor_confs.py decides the
    routes, but the FlagGems Python surface it routes over is discovered -- and
    published for every platform at once -- by the coverage write in
    codegen_ops.py.
    """
    tree = _tree(CODEGEN_OPS)
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "render_flaggems_coverage" in called, (
        "codegen_ops.py no longer calls render_flaggems_coverage(); "
        "scripts/codegen/backend_coverage.py is what gen_vendor_confs.py "
        "imports FLAGGEMS_PYTHON_OPS from"
    )
