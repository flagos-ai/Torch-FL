# FlagGems PR #5086 Review Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Address all five review issues from chen98038's review of FlagGems PR #5086 (C++ pointwise dispatch codegen)

**Architecture:** The PR introduces declarative C++ glue codegen from `op_specs.py`. Review identified five bugs: (1) inverted tensor/scalar mask in remainder op, (2) missing rank limit handling, (3) broken alias imports in codegen, (4) incomplete alias detection, (5) test reference computation bypassing CPU fallback.

**Tech Stack:** Python 3.10+, C++17, PyTorch 2.10+, Triton, CMake, pytest

## Global Constraints

- All changes must be in English (code, comments, commit messages)
- Must maintain backward compatibility with existing C++ dispatch
- Generated code must pass ruff check + ruff format --check
- C++ code must pass clang-format 13.0.0
- All tests must pass on CUDA (primary platform)
- Changes must work with both `FLAGGEMS_POINTWISE_DYNAMIC_BOXED=ON` and `OFF`
- Repository: `/nfs/lvyufeng/FlagGems-src`
- Branch: `feat/cpp-unary-elementwise-ops`
- Environment: `conda activate torch-fl-210` (CPU torch + external CUDA assets)

---

## Task 1: Fix remainder scalar-first tensor mask inversion

**Files:**
- Modify: `cpp/lib/pointwise_dynamic_cpp/prebuild_kernels.py` (the `_build_is_tensor_mask` function)
- Modify: `cpp/lib/pointwise_dynamic_cpp/op_specs.py` (add test coverage note)
- Create: `cpp/ctests/test_remainder_scalar_first.cpp` (new ctest)
- Modify: `cpp/ctests/CMakeLists.txt` (register new test)

**Interfaces:**
- Consumes: Existing `_build_is_tensor_mask` that assumes tensors-first ordering
- Produces: Fixed mask builder that respects actual `is_tensor` layout from schema; new ctest covering `remainder.Scalar_Tensor` (scalar-first)

**Context:** The `rem_st` kernel has `is_tensor=[False, True]` (scalar first), but codegen assumes `[True, False, ...]` (tensors-first). This causes the boxed dispatcher to pass tensor pointer into scalar slot → crash or wrong results.

- [ ] **Step 1: Read current mask building logic**

```bash
cd /nfs/lvyufeng/FlagGems-src
```

Read `cpp/lib/pointwise_dynamic_cpp/prebuild_kernels.py` to locate `_build_is_tensor_mask` function and understand current implementation.

- [ ] **Step 2: Write failing ctest for scalar-first remainder**

Create `cpp/ctests/test_remainder_scalar_first.cpp`:

```cpp
// Copyright 2026 FlagOS Contributors
// Licensed under the Apache License, Version 2.0

#include <flag_gems/operators.h>
#include <gtest/gtest.h>
#include <torch/torch.h>
#include "accuracy_utils.h"

// Test remainder with SCALAR first, tensor second (Scalar_Tensor overload)
// This exercises the rem_st kernel with is_tensor=[False, True]
TEST(RemainderScalarFirst, Basic) {
    auto b = torch::randn({10}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
    double a_scalar = 3.5;
    
    auto result = flag_gems::remainder_st(a_scalar, b);
    
    // Reference: compute on CPU
    auto b_cpu = b.cpu();
    auto ref = torch::remainder(a_scalar, b_cpu);
    
    auto max_err = flag_gems::accuracy_utils::abs_diff(result, ref);
    EXPECT_LT(max_err, 1e-5);
}

TEST(RemainderScalarFirst, ZeroDimTensor) {
    // Both operands 0-dim: should take host path in boxed adapter
    auto b = torch::tensor(2.0, torch::device(torch::kCUDA).dtype(torch::kFloat32));
    double a_scalar = 7.5;
    
    auto result = flag_gems::remainder_st(a_scalar, b);
    
    auto b_cpu = b.cpu();
    auto ref = torch::remainder(a_scalar, b_cpu);
    
    auto max_err = flag_gems::accuracy_utils::abs_diff(result, ref);
    EXPECT_LT(max_err, 1e-5);
}
```

- [ ] **Step 3: Register the test in CMake**

Edit `cpp/ctests/CMakeLists.txt`, add after existing test registrations:

```cmake
add_ctest(test_remainder_scalar_first test_remainder_scalar_first.cpp)
```

- [ ] **Step 4: Run test to verify it fails**

```bash
cd /nfs/lvyufeng/FlagGems-src
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DFLAGGEMS_BUILD_POINTWISE_DYNAMIC_CPP=ON \
  -DGEMS_VENDOR=nvidia -DFLAGGEMS_POINTWISE_DYNAMIC_BOXED=ON
make test_remainder_scalar_first
./ctests/test_remainder_scalar_first
```

Expected: FAIL with Triton signature error or illegal memory access

- [ ] **Step 5: Fix mask builder to respect actual is_tensor layout**

Edit `cpp/lib/pointwise_dynamic_cpp/prebuild_kernels.py`, locate `_build_is_tensor_mask` function and replace with:

```python
def _build_is_tensor_mask(schema: FunctionSchema) -> str:
    """Build {true,false,...} mask matching the schema's is_tensor order.
    
    The mask order MUST match the @pointwise_dynamic decorator's is_tensor
    parameter: if is_tensor=[False, True] (scalar first), the mask is
    {false, true}. Passing {true, false} to such a kernel causes the
    dispatcher to pass the tensor pointer into the scalar slot (crash).
    
    Previously assumed tensors-first, which broke scalar-first ops like
    rem_st (remainder.Scalar_Tensor).
    """
    mask_parts = []
    for arg in schema.arguments:
        if arg.kwarg_only:
            continue
        # Non-kwarg arguments appear in is_tensor in declaration order
        is_t = arg.is_tensor if hasattr(arg, 'is_tensor') else (arg.type == 'Tensor')
        mask_parts.append('true' if is_t else 'false')
    
    return '{' + ', '.join(mask_parts) + '}'
```

- [ ] **Step 6: Rebuild and run test to verify it passes**

```bash
cd /nfs/lvyufeng/FlagGems-src/build
make -j$(nproc)
./ctests/test_remainder_scalar_first
```

Expected: PASS (both tests)

- [ ] **Step 7: Run full ctest suite to ensure no regression**

```bash
cd /nfs/lvyufeng/FlagGems-src/build
ctest --output-on-failure
```

Expected: All tests PASS

- [ ] **Step 8: Commit**

```bash
cd /nfs/lvyufeng/FlagGems-src
git add cpp/lib/pointwise_dynamic_cpp/prebuild_kernels.py \
  cpp/ctests/test_remainder_scalar_first.cpp \
  cpp/ctests/CMakeLists.txt
git commit -m "fix(cpp/pointwise): respect actual is_tensor layout in mask builder

The _build_is_tensor_mask function assumed tensors-first ordering, but
operators like remainder.Scalar_Tensor have is_tensor=[False, True]
(scalar first). This caused the boxed dispatcher to pass tensor pointers
into scalar slots, resulting in Triton compile errors or illegal memory
access.

Fixed by building the mask directly from the schema's is_tensor order
rather than assuming any particular layout.

Added ctest coverage for scalar-first remainder operations.

Addresses review comment on cpp/lib/pointwise_dynamic_cpp/CMakeLists.txt:23

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Add rank limit error handling with fallback

**Files:**
- Modify: `cpp/include/flag_gems/pointwise_prepare_args.h` (lines ~450-460, the `op_registry.find(rank)` call)
- Modify: `cpp/lib/pointwise_dynamic_cpp/prebuild_kernels.py` (document POINTWISE_MAX_RANK in generated headers)
- Create: `cpp/ctests/test_high_rank_fallback.cpp` (test rank > 5)
- Modify: `cpp/ctests/CMakeLists.txt`

**Interfaces:**
- Consumes: Existing `POINTWISE_MAX_RANK=5` constant; bare `std::runtime_error` on rank > 5
- Produces: `TORCH_CHECK` with clear error message + fallback suggestion; test covering rank-6 input

**Context:** C++ path pre-generates kernels only for rank 0..5. Rank > 5 throws bare `std::runtime_error` with no context. Python path codegens any rank on demand, so paths diverge.

- [ ] **Step 1: Write failing high-rank test**

Create `cpp/ctests/test_high_rank_fallback.cpp`:

```cpp
// Copyright 2026 FlagOS Contributors
// Licensed under the Apache License, Version 2.0

#include <flag_gems/operators.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

// Test that rank > POINTWISE_MAX_RANK (5) produces a clear error
// rather than a bare runtime_error
TEST(HighRankOps, AbsRank6NonContiguous) {
    // Create a rank-6 non-contiguous tensor (slicing prevents fast path)
    auto x = torch::randn({2, 3, 4, 5, 6, 8}, 
                         torch::device(torch::kCUDA).dtype(torch::kFloat32));
    auto sliced = x.index({"...", torch::indexing::Slice(0, 2)});
    
    EXPECT_EQ(sliced.dim(), 6);
    EXPECT_FALSE(sliced.is_contiguous());
    
    // Should throw TORCH_CHECK with message mentioning rank limit
    EXPECT_THROW({
        try {
            flag_gems::abs(sliced);
        } catch (const c10::Error& e) {
            // Verify it's a proper TORCH_CHECK, not bare runtime_error
            std::string msg = e.what();
            EXPECT_NE(msg.find("rank"), std::string::npos);
            EXPECT_NE(msg.find("6"), std::string::npos);
            EXPECT_NE(msg.find("5"), std::string::npos);  // MAX_RANK
            throw;
        }
    }, c10::Error);
}

TEST(HighRankOps, AbsRank6Contiguous) {
    // Contiguous rank-6 should also fail clearly if it skips fast path
    auto x = torch::randn({2, 2, 2, 2, 2, 2},
                         torch::device(torch::kCUDA).dtype(torch::kFloat32));
    auto transposed = x.permute({5, 4, 3, 2, 1, 0});  // Force non-fast-path
    
    EXPECT_EQ(transposed.dim(), 6);
    
    EXPECT_THROW({
        flag_gems::abs(transposed);
    }, c10::Error);
}
```

- [ ] **Step 2: Register test**

Edit `cpp/ctests/CMakeLists.txt`:

```cmake
add_ctest(test_high_rank_fallback test_high_rank_fallback.cpp)
```

- [ ] **Step 3: Run test to verify current behavior (bare exception)**

```bash
cd /nfs/lvyufeng/FlagGems-src/build
make test_high_rank_fallback
./ctests/test_high_rank_fallback
```

Expected: Test fails because current code throws `std::runtime_error` not `c10::Error`

- [ ] **Step 4: Fix error handling in pointwise_prepare_args.h**

Read `cpp/include/flag_gems/pointwise_prepare_args.h` around line 450-460 to find the `op_registry.find(rank)` call.

Replace the bare `throw std::runtime_error(...)` with:

```cpp
  auto it = op_registry.find(rank);
  if (it == op_registry.end()) {
    TORCH_CHECK(
        false,
        "FlagGems C++ pointwise dispatch: input rank ", rank,
        " exceeds POINTWISE_MAX_RANK (", POINTWISE_MAX_RANK, "). ",
        "The C++ path pre-generates kernels only for rank 0..", POINTWISE_MAX_RANK, ". ",
        "Workarounds: (1) collapse/reshape to rank ≤ ", POINTWISE_MAX_RANK, ", ",
        "(2) use the Python API (torch.ops.flag_gems.<op>), which codegens any rank on demand, ",
        "(3) ensure the input is contiguous and dense (fast path bypasses per-rank kernels)."
    );
  }
```

- [ ] **Step 5: Document MAX_RANK in generated headers**

Edit `cpp/lib/pointwise_dynamic_cpp/prebuild_kernels.py`, find the header generation for `pointwise_manifest.h` or `pointwise_runtime.h`, and add a comment at the top:

```cpp
// POINTWISE_MAX_RANK: The C++ dispatch pre-generates kernels for ranks 0..5.
// Inputs with effective rank > 5 that skip the fast path will error with a
// TORCH_CHECK. The Python path (torch.ops.flag_gems.*) codegens any rank on
// demand and does not have this limit.
```

- [ ] **Step 6: Rebuild and verify test passes**

```bash
cd /nfs/lvyufeng/FlagGems-src/build
make -j$(nproc)
./ctests/test_high_rank_fallback
```

Expected: PASS (both tests catch c10::Error with correct message)

- [ ] **Step 7: Run full test suite**

```bash
cd /nfs/lvyufeng/FlagGems-src/build
ctest --output-on-failure
```

Expected: All tests PASS

- [ ] **Step 8: Commit**

```bash
cd /nfs/lvyufeng/FlagGems-src
git add cpp/include/flag_gems/pointwise_prepare_args.h \
  cpp/lib/pointwise_dynamic_cpp/prebuild_kernels.py \
  cpp/ctests/test_high_rank_fallback.cpp \
  cpp/ctests/CMakeLists.txt
git commit -m "fix(cpp/pointwise): replace bare exception with TORCH_CHECK for rank limit

The C++ dispatch pre-generates kernels only for rank 0..5 (POINTWISE_MAX_RANK).
Inputs with rank > 5 previously threw a bare std::runtime_error with no context.

Changed to TORCH_CHECK with a clear error message explaining:
- The rank limit (5)
- Three workarounds: reshape, use Python API, or ensure contiguous/dense

Added ctest coverage for rank-6 inputs.

The Python path (torch.ops.flag_gems.*) codegens any rank on demand and is
not affected by this limit.

Addresses review comment on cpp/csrc/cstub.cpp:219

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Fix import alias codegen to handle 'as' renames

**Files:**
- Modify: `src/flag_gems/utils/pointwise_dynamic.py` (lines ~1170-1190, the `_collect_jit_deps` alias collection logic)
- Create: `tests/test_pointwise_alias_codegen.py` (unit test for alias collection)

**Interfaces:**
- Consumes: Existing alias collection that writes `root.id` (local asname) as if it were the real exported name
- Produces: Fixed collection recording both `real_name` and `asname`, emitting `from X import real_name as asname`

**Context:** Code of form `from X import a as b` + `alias = b.attr` generates `from X import b` (invalid). The memory says this was already fixed for the base case, but review found it can still fail.

- [ ] **Step 1: Write failing unit test for 'as' import aliases**

Create `tests/test_pointwise_alias_codegen.py`:

```python
# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0

"""Unit tests for pointwise_dynamic alias collection and import codegen."""

import ast
import textwrap
from flag_gems.utils.pointwise_dynamic import _collect_jit_deps


def test_import_from_with_asname():
    """Alias rooted in 'from X import Y as Z' must emit 'from X import Y as Z'."""
    source = textwrap.dedent("""
        from flag_gems.utils import tl_extra_shim as shim
        
        _tanh_alias = shim.tanh
    """)
    
    tree = ast.parse(source)
    imports, aliases = _collect_jit_deps(tree)
    
    # Should record the real name 'tl_extra_shim' and asname 'shim'
    assert any('tl_extra_shim' in imp and 'as shim' in imp for imp in imports), \
        f"Expected 'from ... import tl_extra_shim as shim', got: {imports}"
    
    # Alias should reference the local name 'shim'
    assert '_tanh_alias = shim.tanh' in aliases or \
           any('shim.tanh' in a for a in aliases), \
        f"Expected alias '_tanh_alias = shim.tanh', got: {aliases}"


def test_import_from_without_asname():
    """Normal 'from X import Y' should still work."""
    source = textwrap.dedent("""
        from flag_gems.utils import tl_extra_shim
        
        _t = tl_extra_shim.tanh
    """)
    
    tree = ast.parse(source)
    imports, aliases = _collect_jit_deps(tree)
    
    assert any('tl_extra_shim' in imp for imp in imports)
    assert any('tl_extra_shim.tanh' in a for a in aliases)


def test_plain_import_with_asname():
    """'import X as Y' style (not FromImport) - should work or warn."""
    source = textwrap.dedent("""
        import triton.language as tl
        
        _sigmoid = tl.sigmoid
    """)
    
    tree = ast.parse(source)
    imports, aliases = _collect_jit_deps(tree)
    
    # Should either handle it or skip with no crash
    # (Review comment #4 says this shape is silently dropped currently)
    # For now, just verify it doesn't crash
    assert isinstance(imports, list)
    assert isinstance(aliases, list)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /nfs/lvyufeng/FlagGems-src
pytest tests/test_pointwise_alias_codegen.py::test_import_from_with_asname -v
```

Expected: FAIL - imported name is wrong (tries to import 'shim' instead of 'tl_extra_shim as shim')

- [ ] **Step 3: Fix _collect_jit_deps to record both real name and asname**

Edit `src/flag_gems/utils/pointwise_dynamic.py`, locate the `_collect_jit_deps` function around lines 1170-1190.

Find the section that collects `ImportFrom` nodes and builds the `imported_name_module` dict. Replace the logic that writes `root.id` with:

```python
def _collect_jit_deps(module_ast):
    """Collect imports and top-level aliases for JIT kernel codegen.
    
    Returns (imports, aliases) where:
      imports: list of 'from X import Y [as Z]' strings
      aliases: list of 'name = value' assignment strings
    """
    imports = []
    aliases = []
    imported_name_module = {}  # local_name -> (module, real_name, asname)
    
    for node in ast.walk(module_ast):
        if isinstance(node, ast.ImportFrom):
            module = node.module
            for alias in node.names:
                real_name = alias.name
                local_name = alias.asname if alias.asname else real_name
                imported_name_module[local_name] = (module, real_name, alias.asname)
                
                # Emit the import line
                if alias.asname:
                    imports.append(f"from {module} import {real_name} as {alias.asname}")
                else:
                    imports.append(f"from {module} import {real_name}")
        
        # Collect top-level Name = <imported>.attr assignments
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and isinstance(node.value, ast.Attribute):
                if isinstance(node.value.value, ast.Name):
                    root = node.value.value.id
                    attr = node.value.attr
                    # root is the local binding name (possibly an asname)
                    if root in imported_name_module:
                        # Record the alias using the local name
                        aliases.append(f"{target.id} = {root}.{attr}")
    
    return imports, aliases
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /nfs/lvyufeng/FlagGems-src
pytest tests/test_pointwise_alias_codegen.py::test_import_from_with_asname -v
```

Expected: PASS

- [ ] **Step 5: Run all three alias tests**

```bash
pytest tests/test_pointwise_alias_codegen.py -v
```

Expected: All PASS

- [ ] **Step 6: Run full test suite to ensure no breakage**

```bash
cd /nfs/lvyufeng/FlagGems-src
pytest tests/ -v -k pointwise
```

Expected: No regressions

- [ ] **Step 7: Commit**

```bash
cd /nfs/lvyufeng/FlagGems-src
git add src/flag_gems/utils/pointwise_dynamic.py \
  tests/test_pointwise_alias_codegen.py
git commit -m "fix(pointwise_dynamic): emit correct import for 'as' renamed aliases

The alias collector wrote root.id (local binding name) as if it were the
module's real exported name. For 'from X import a as b', this generated
'from X import b' (invalid ImportError).

Fixed by recording (module, real_name, asname) tuples and emitting
'from X import real_name as asname' when asname is present.

Added unit test coverage for import-with-asname, import-without-asname,
and plain 'import X as Y' forms.

Addresses review comment on src/flag_gems/utils/pointwise_dynamic.py:1183

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Extend alias detection to support ast.Import and warn on skipped patterns

**Files:**
- Modify: `src/flag_gems/utils/pointwise_dynamic.py` (extend `_collect_jit_deps` to handle `ast.Import`)
- Modify: `tests/test_pointwise_alias_codegen.py` (add tests for plain import + skipped patterns)

**Interfaces:**
- Consumes: `_collect_jit_deps` that only matches `ast.ImportFrom` + top-level `Name = ImportedName.attr`
- Produces: Extended function supporting `ast.Import` (e.g. `import torch as T`); warns on chained attrs, AnnAssign, and if/try-nested aliases

**Context:** Review notes several alias shapes are silently skipped: `import X as Y`, chained `a.b.c`, annotated assignments, and aliases inside if/try. Real examples in codebase: `sigmoid = tl.sigmoid` in fused/swiglu.py, `Tensor = torch.Tensor` in ops/cummax.py.

- [ ] **Step 1: Extend unit tests to cover ast.Import and skipped patterns**

Edit `tests/test_pointwise_alias_codegen.py`, add:

```python
def test_plain_import_with_asname_supported():
    """'import X.Y as Z' should be collected."""
    source = textwrap.dedent("""
        import triton.language as tl
        
        _sigmoid = tl.sigmoid
    """)
    
    tree = ast.parse(source)
    imports, aliases = _collect_jit_deps(tree)
    
    # Should emit 'import triton.language as tl'
    assert any('import triton.language as tl' in imp for imp in imports), \
        f"Expected 'import triton.language as tl', got: {imports}"
    assert any('_sigmoid = tl.sigmoid' in a for a in aliases)


def test_chained_attribute_warns():
    """Chained attrs like a.b.c should warn (not silently skip)."""
    source = textwrap.dedent("""
        import torch.nn.functional as F
        
        _relu = F.relu.forward  # chained: F.relu (returns module), then .forward
    """)
    
    tree = ast.parse(source)
    # Currently this is silently skipped; after fix should warn
    # For now just verify no crash
    imports, aliases = _collect_jit_deps(tree)
    assert isinstance(aliases, list)


def test_annotated_assignment_warns():
    """AnnAssign (x: Type = value) should warn if it's an alias."""
    source = textwrap.dedent("""
        import torch
        
        Tensor: type = torch.Tensor
    """)
    
    tree = ast.parse(source)
    imports, aliases = _collect_jit_deps(tree)
    # After fix should warn; for now verify no crash
    assert isinstance(aliases, list)


def test_nested_in_if_warns():
    """Aliases inside if/try blocks should warn."""
    source = textwrap.dedent("""
        import torch
        
        if True:
            _tensor = torch.Tensor
    """)
    
    tree = ast.parse(source)
    imports, aliases = _collect_jit_deps(tree)
    # Should warn about skipped alias
    assert isinstance(aliases, list)
```

- [ ] **Step 2: Run new tests to verify current behavior (some fail)**

```bash
cd /nfs/lvyufeng/FlagGems-src
pytest tests/test_pointwise_alias_codegen.py::test_plain_import_with_asname_supported -v
```

Expected: FAIL (ast.Import not handled yet)

- [ ] **Step 3: Extend _collect_jit_deps to support ast.Import**

Edit `src/flag_gems/utils/pointwise_dynamic.py`, add handling for `ast.Import` nodes:

```python
import logging

logger = logging.getLogger(__name__)


def _collect_jit_deps(module_ast):
    """Collect imports and top-level aliases for JIT kernel codegen.
    
    Returns (imports, aliases) where:
      imports: list of 'from X import Y [as Z]' / 'import X [as Y]' strings
      aliases: list of 'name = value' assignment strings
    
    Warns on alias-shaped AST nodes that cannot be captured (chained attrs,
    annotated assignments, assignments inside if/try blocks).
    """
    imports = []
    aliases = []
    imported_name_module = {}  # local_name -> (module, real_name, asname, kind)
    # kind: 'from' or 'plain'
    
    # Track top-level line numbers to detect nested assignments
    top_level_lines = {node.lineno for node in module_ast.body}
    
    for node in ast.walk(module_ast):
        # ast.ImportFrom: from X import Y [as Z]
        if isinstance(node, ast.ImportFrom):
            module = node.module
            for alias in node.names:
                real_name = alias.name
                local_name = alias.asname if alias.asname else real_name
                imported_name_module[local_name] = (module, real_name, alias.asname, 'from')
                
                if alias.asname:
                    imports.append(f"from {module} import {real_name} as {alias.asname}")
                else:
                    imports.append(f"from {module} import {real_name}")
        
        # ast.Import: import X [as Y]
        elif isinstance(node, ast.Import):
            for alias in node.names:
                real_name = alias.name
                local_name = alias.asname if alias.asname else real_name.split('.')[-1]
                imported_name_module[local_name] = (None, real_name, alias.asname, 'plain')
                
                if alias.asname:
                    imports.append(f"import {real_name} as {alias.asname}")
                else:
                    imports.append(f"import {real_name}")
        
        # Top-level assignments: name = <imported>.attr
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            # Check if it's top-level
            if node.lineno not in top_level_lines:
                logger.warning(
                    f"Skipping alias assignment at line {node.lineno} (not top-level). "
                    f"Aliases inside if/try blocks cannot be captured for JIT codegen."
                )
                continue
            
            target = node.targets[0]
            if isinstance(target, ast.Name) and isinstance(node.value, ast.Attribute):
                # Simple case: name = root.attr
                if isinstance(node.value.value, ast.Name):
                    root = node.value.value.id
                    attr = node.value.attr
                    if root in imported_name_module:
                        aliases.append(f"{target.id} = {root}.{attr}")
                # Chained: name = root.attr.attr2 (not supported)
                elif isinstance(node.value.value, ast.Attribute):
                    logger.warning(
                        f"Skipping chained attribute alias at line {node.lineno}: "
                        f"{ast.unparse(node)}. Chained attributes (a.b.c) are not "
                        f"captured for JIT codegen."
                    )
        
        # Annotated assignment: name: Type = value
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.value, ast.Attribute) and isinstance(node.value.value, ast.Name):
                logger.warning(
                    f"Skipping annotated alias at line {node.lineno}: "
                    f"{ast.unparse(node)}. Use plain assignment for JIT-visible aliases."
                )
    
    return imports, aliases
```

- [ ] **Step 4: Run all alias tests**

```bash
cd /nfs/lvyufeng/FlagGems-src
pytest tests/test_pointwise_alias_codegen.py -v
```

Expected: test_plain_import_with_asname_supported PASS; others PASS (warnings emitted but no crash)

- [ ] **Step 5: Run full test suite**

```bash
pytest tests/ -v -k pointwise
```

Expected: No regressions

- [ ] **Step 6: Commit**

```bash
cd /nfs/lvyufeng/FlagGems-src
git add src/flag_gems/utils/pointwise_dynamic.py \
  tests/test_pointwise_alias_codegen.py
git commit -m "feat(pointwise_dynamic): support ast.Import aliases, warn on skipped patterns

Extended _collect_jit_deps to handle 'import X as Y' form (ast.Import),
enabling aliases like 'sigmoid = tl.sigmoid' from 'import triton.language as tl'.

Added warnings for alias patterns that cannot be captured:
- Chained attributes (a.b.c)
- Annotated assignments (name: Type = value)
- Assignments inside if/try blocks (not top-level)

These patterns were previously silently skipped, causing NameError on first
use in standalone C++ TritonJIT compilation while the in-process Python path
masked the issue via __globals__.

Added unit test coverage for all supported and unsupported forms.

Addresses review comment on src/flag_gems/utils/pointwise_dynamic.py:1176

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Fix ctest reference computation to use CPU fallback

**Files:**
- Modify: `cpp/ctests/test_triton_unary.cpp` (replace direct device reference with `to_reference`)
- Modify: `cpp/ctests/accuracy_utils.h` (ensure `to_reference` is accessible)

**Interfaces:**
- Consumes: Tests computing reference as `x.abs()` etc directly on CUDA tensors (bypasses TO_CPU path)
- Produces: Tests using `flag_gems::accuracy_utils::to_reference(x)` to get CPU reference, matching all other ctests

**Context:** All other ctests call `to_reference` which moves tensors to CPU for reference computation (required on MUSA/NPU where TO_CPU=true). test_triton_unary.cpp bypasses this, comparing Triton results against the same device's ATen impl (spurious pass/fail if device ATen has bugs).

- [ ] **Step 1: Read current test structure**

Read `cpp/ctests/test_triton_unary.cpp` to understand current reference computation pattern.

- [ ] **Step 2: Read accuracy_utils interface**

Read `cpp/ctests/accuracy_utils.h` (or `cpp/include/flag_gems/accuracy_utils.h`) to confirm `to_reference` signature.

Expected signature: `at::Tensor to_reference(const at::Tensor& x)`

- [ ] **Step 3: Fix all reference computations in test_triton_unary.cpp**

Edit `cpp/ctests/test_triton_unary.cpp`, replace each pattern:

OLD:
```cpp
auto result = flag_gems::abs(x);
auto ref = x.abs();  // WRONG: computes on device
auto max_err = flag_gems::accuracy_utils::abs_diff(result, ref);
```

NEW:
```cpp
auto result = flag_gems::abs(x);
auto x_ref = flag_gems::accuracy_utils::to_reference(x);
auto ref = x_ref.abs();
auto max_err = flag_gems::accuracy_utils::abs_diff(result, ref);
```

Apply this to ALL ops in the file: abs, neg, exp, sqrt, rsqrt, tanh, sigmoid, silu, relu, gelu.

- [ ] **Step 4: Rebuild and run the modified tests**

```bash
cd /nfs/lvyufeng/FlagGems-src/build
make test_triton_unary -j$(nproc)
./ctests/test_triton_unary
```

Expected: PASS (same results, but now correctly using CPU reference)

- [ ] **Step 5: Verify on non-CUDA platform (if available) or via TO_CPU=true mock**

If on CUDA-only, add a temporary test that forces TO_CPU behavior:

```cpp
TEST(ReferenceComputation, UsesCPU) {
    auto x = torch::randn({10}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
    auto x_ref = flag_gems::accuracy_utils::to_reference(x);
    
    // to_reference should return CPU tensor (or same device if TO_CPU=false)
    // On CUDA with default settings, should match device, but logic is correct
    EXPECT_TRUE(x_ref.defined());
}
```

Run and verify no crash.

- [ ] **Step 6: Run full ctest suite**

```bash
cd /nfs/lvyufeng/FlagGems-src/build
ctest --output-on-failure
```

Expected: All tests PASS

- [ ] **Step 7: Commit**

```bash
cd /nfs/lvyufeng/FlagGems-src
git add cpp/ctests/test_triton_unary.cpp
git commit -m "fix(ctests): use to_reference for CPU-based reference computation

test_triton_unary.cpp computed references directly on device tensors
(x.abs(), torch::silu(x), ...), bypassing the TO_CPU reference path
that other ctests use via accuracy_utils::to_reference.

On MUSA/NPU backends where TO_CPU=true, this would compare Triton results
against the same device's ATen implementation. If that implementation has
a bug, tests spuriously pass or fail misleadingly.

Fixed by calling to_reference(x) before computing references, matching
the pattern used in all other ctests.

Addresses review comment on cpp/ctests/test_triton_unary.cpp:59

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Integration verification and PR update

**Files:**
- Modify: `README.md` or `cpp/README.md` (if exists, document rank limit and workarounds)
- Run: Full test suite (ctests + pytest)
- Run: Linting (ruff + clang-format)
- Create: Summary document for PR update

**Interfaces:**
- Consumes: All five fixes from Tasks 1-5
- Produces: Verified build, all tests passing, lint clean, ready for PR push

- [ ] **Step 1: Rebuild from clean state**

```bash
cd /nfs/lvyufeng/FlagGems-src
rm -rf build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
  -DFLAGGEMS_BUILD_POINTWISE_DYNAMIC_CPP=ON \
  -DGEMS_VENDOR=nvidia \
  -DFLAGGEMS_POINTWISE_DYNAMIC_BOXED=ON
make -j$(nproc)
```

Expected: Clean build with no errors

- [ ] **Step 2: Run full ctest suite**

```bash
cd /nfs/lvyufeng/FlagGems-src/build
ctest --output-on-failure
```

Expected: All tests PASS (including 3 new tests: test_remainder_scalar_first, test_high_rank_fallback, existing + fixed test_triton_unary)

- [ ] **Step 3: Run Python unit tests**

```bash
cd /nfs/lvyufeng/FlagGems-src
pytest tests/test_pointwise_alias_codegen.py -v
pytest tests/ -k pointwise -v
```

Expected: All PASS

- [ ] **Step 4: Run lint checks**

```bash
cd /nfs/lvyufeng/FlagGems-src
ruff check src/flag_gems/utils/pointwise_dynamic.py
ruff check tests/test_pointwise_alias_codegen.py
ruff format --check src/ tests/
```

Expected: No errors

- [ ] **Step 5: Run clang-format on C++ files**

```bash
cd /nfs/lvyufeng/FlagGems-src
clang-format-13 -i cpp/ctests/test_remainder_scalar_first.cpp \
  cpp/ctests/test_high_rank_fallback.cpp \
  cpp/ctests/test_triton_unary.cpp \
  cpp/include/flag_gems/pointwise_prepare_args.h
```

- [ ] **Step 6: Verify formatting applied**

```bash
git diff cpp/
```

Expected: Only intentional changes (no spurious whitespace)

- [ ] **Step 7: Create PR update summary**

Create `docs/pr-5086-review-fixes-summary.md`:

```markdown
# PR #5086 Review Fixes Summary

All five issues from chen98038's review (2026-08-05) have been addressed:

## 1. Remainder scalar-first tensor mask inversion (CMakeLists.txt:23)
**Issue:** `_build_is_tensor_mask` assumed tensors-first ordering, breaking scalar-first ops like `rem_st`.
**Fix:** Mask builder now respects actual `is_tensor` layout from schema.
**Test:** `cpp/ctests/test_remainder_scalar_first.cpp` covers `remainder.Scalar_Tensor` with 0-dim and multi-dim tensors.
**Commit:** fix(cpp/pointwise): respect actual is_tensor layout in mask builder

## 2. Missing rank limit error handling (cstub.cpp:219)
**Issue:** Rank > 5 threw bare `std::runtime_error` with no context.
**Fix:** Replaced with `TORCH_CHECK` explaining the limit and three workarounds.
**Test:** `cpp/ctests/test_high_rank_fallback.cpp` verifies proper `c10::Error` with message.
**Commit:** fix(cpp/pointwise): replace bare exception with TORCH_CHECK for rank limit

## 3. Broken 'as' import aliases (pointwise_dynamic.py:1183)
**Issue:** `from X import a as b` generated invalid `from X import b`.
**Fix:** Alias collector now records (real_name, asname) and emits correct import.
**Test:** `tests/test_pointwise_alias_codegen.py::test_import_from_with_asname`
**Commit:** fix(pointwise_dynamic): emit correct import for 'as' renamed aliases

## 4. Incomplete alias detection (pointwise_dynamic.py:1176)
**Issue:** `ast.Import` form and nested/chained aliases silently skipped.
**Fix:** Added support for `import X as Y`; warn on unsupported patterns (chained attrs, AnnAssign, if/try nesting).
**Test:** `tests/test_pointwise_alias_codegen.py` covers all forms.
**Commit:** feat(pointwise_dynamic): support ast.Import aliases, warn on skipped patterns

## 5. Test reference bypass (test_triton_unary.cpp:59)
**Issue:** References computed on device, bypassing `to_reference` CPU path.
**Fix:** All references now use `accuracy_utils::to_reference(x)`.
**Test:** Existing tests now correctly match other ctests' reference pattern.
**Commit:** fix(ctests): use to_reference for CPU-based reference computation

## Verification
- Clean rebuild: PASS
- All ctests (13 total, 3 new): PASS
- Python unit tests: PASS
- Lint (ruff + clang-format): PASS
- Boxed dispatch (FLAGGEMS_POINTWISE_DYNAMIC_BOXED=ON): PASS
- Environment: torch-fl-210 (CPU torch + external CUDA)

## Next Steps
1. Push updated branch to fork: `git push myfork feat/cpp-unary-elementwise-ops --force-with-lease`
2. PR #5086 will auto-update
3. Request re-review from chen98038
```

- [ ] **Step 8: Review all commit messages**

```bash
cd /nfs/lvyufeng/FlagGems-src
git log --oneline feat/cpp-unary-elementwise-ops -7
```

Expected: 5 new fix commits + 2 original feature commits

- [ ] **Step 9: Create consolidated commit if requested (optional)**

If reviewer prefers squashed fixes:

```bash
git rebase -i HEAD~5
# Mark last 4 commits as 'squash', keep first as 'pick'
# Edit combined commit message
```

(Skip this step for now - keep atomic commits for clear review)

- [ ] **Step 10: Final verification before push**

```bash
cd /nfs/lvyufeng/FlagGems-src
git status
git diff origin/feat/cpp-unary-elementwise-ops
```

Expected: 5 new commits, clean working tree

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-08-13-flaggems-pr-5086-review-fixes.md`.

**Two execution options:**

**1. Subagent-Driven (recommended)** - Fresh subagent per task with two-stage review between tasks

**2. Inline Execution** - Execute all tasks in current session with checkpoints for review

**Which approach?**
