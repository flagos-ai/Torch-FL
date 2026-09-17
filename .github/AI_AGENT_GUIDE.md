---
name: AI Agent Workflow Guide
description: Guidelines for AI agents (Claude Code, Codex) working on torch_fl
---

# AI Agent Guidelines for torch_fl

This document provides specific guidelines for AI coding agents (Claude Code, GitHub Codex, Cursor, etc.) when working on the torch_fl repository.

## Mandatory Requirements

### 1. Language
**ALL GitHub-facing text MUST be in English:**
- PR titles and descriptions
- Issue text and comments
- Commit messages
- Code comments
- Documentation files

This is **non-negotiable** per CLAUDE.md. PRs/issues in other languages will be closed.

### 2. Pre-Submission Checklist

Before creating any PR, you **MUST**:
```bash
# 1. Run linting (exact version from CI)
ruff check
ruff format --check

# 2. Run relevant tests
pytest tests/unit/ -v
pytest tests/integration/ops/ -v -m "anyplatform or cuda"

# 3. Verify the actual fix works
# Run the reproduction case from the issue
```

**Do not submit a PR without pasting actual command output.** Saying "tests pass" is not sufficient.

### 3. Use the AI PR Template

When creating PRs, use `.github/PULL_REQUEST_TEMPLATE/ai_agent_pr.md`:

```bash
gh pr create --template ai_agent_pr.md \
  --title "fix: <description>" \
  --body "$(cat pr_body.md)"
```

## Investigation Before Action

### Root Cause Analysis Required

Before proposing any fix, you must:

1. **Read the relevant code** (don't guess)
   ```bash
   # Example investigation commands
   grep -r "function_name" csrc/
   find . -name "*relevant*" -type f
   ```

2. **Understand the call chain**
   - Where does the error originate?
   - What calls lead to this point?
   - What are the neighboring functions doing?

3. **Check existing patterns**
   - How do similar operators handle this?
   - What does the existing code convention show?
   - Are there utilities you should reuse?

4. **Verify your hypothesis**
   - Add debug output and test your theory
   - Don't just pattern-match to similar-looking issues

### Code Reading Checklist

Before modifying any file:
- [ ] Read the entire file to understand context
- [ ] Check imports and dependencies
- [ ] Find similar functions in the same file
- [ ] Check how other platforms handle this (CUDA vs MetaX vs Ascend)
- [ ] Look for existing utilities you can reuse

## Common Pitfalls to Avoid

### ❌ Don't Do This

1. **Shallow pattern matching**
   ```python
   # BAD: Copying similar code without understanding
   def new_op(...):
       # Just copied from some_other_op without understanding
   ```

2. **Incomplete testing**
   ```bash
   # BAD: Only running one test
   pytest tests/integration/ops/test_new_op.py  # Only this
   ```

3. **Missing edge cases**
   ```python
   # BAD: Only testing the happy path
   def test_op():
       x = torch.randn(4, 4, device='flagos')  # Only one shape
   ```

4. **Vague commit messages**
   ```bash
   # BAD
   git commit -m "fix bug"
   git commit -m "update code"
   ```

### ✅ Do This Instead

1. **Understand before copying**
   ```python
   # GOOD: Read surrounding code, understand the abstraction
   def new_op(...):
       # Uses the same pattern because I understand why:
       # - This pattern handles both contiguous and strided inputs
       # - The boxing layer needs this specific signature
       # - Reference: similar_op() in this same file
   ```

2. **Comprehensive testing**
   ```bash
   # GOOD: Test suite covering all affects
   ruff check && ruff format --check
   pytest tests/unit/ -v
   pytest tests/integration/ops/test_new_op.py -v
   pytest tests/integration/test_qwen3_infer.py -v  # If relevant
   ```

3. **Thorough edge case coverage**
   ```python
   # GOOD: Multiple dtypes, shapes, edge cases
   @pytest.mark.parametrize("shape", [(2,2), (100,100), (1,1000)])
   @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
   def test_op(shape, dtype):
       ...
       
   def test_op_empty_input():
       x = torch.empty(0, device='flagos')
       ...
   ```

4. **Descriptive commits**
   ```bash
   # GOOD
   git commit -m "fix: resolve CUDA stream sync race in profiler callbacks

   The CUPTI callbacks must be registered before first CUDA context.
   Moved registration from lazy init to module import time.
   
   Tested: pytest tests/integration/test_profiler.py - all pass
   
   Fixes #123"
   ```

## Template Usage by Task Type

### Bug Fix
1. **Investigation**:
   - Reproduce the bug locally
   - Read the code to find root cause
   - Verify your understanding with debug output

2. **Fix**:
   - Implement minimal fix
   - Add regression test
   - Verify fix resolves original issue

3. **PR**: Use AI PR template with:
   - Root cause analysis section filled
   - Before/after verification output
   - Edge cases considered

### New Feature
1. **Design**:
   - Check if similar features exist
   - Understand the platform's conventions
   - Consider cross-platform implications

2. **Implementation**:
   - Follow existing patterns
   - Reuse existing utilities
   - Add comprehensive tests

3. **PR**: Use AI PR template with:
   - Design decisions explained
   - Alternative approaches discussed
   - All platforms tested (or justified why not)

### Performance Optimization
1. **Baseline**:
   - Measure before optimization
   - Profile to find bottleneck
   - Document measurement method

2. **Optimize**:
   - Implement optimization
   - Verify correctness unchanged
   - Measure improvement

3. **PR**: Use AI PR template with:
   - Benchmark results (before/after)
   - Why this approach was chosen
   - Correctness verification

## File Modification Guidelines

### Backend Implementation Files

When modifying `csrc/aten/backends/<platform>/*.cc`:

1. **Check all platforms**: If you change CUDA backend, check if MetaX/Ascend need similar changes
2. **Match signatures**: Operator signatures must exactly match PyTorch's
3. **Error handling**: Add proper error messages (not just asserts)
4. **Memory safety**: No memory leaks, use RAII

### Python Integration Files

When modifying `torch_fl/*.py`:

1. **Type hints**: Add type hints for all functions
2. **Docstrings**: Document public APIs
3. **Error messages**: User-facing errors should be clear
4. **Backward compatibility**: Don't break existing code

### Test Files

When modifying `tests/**`:

1. **Pytest marks**: Use appropriate marks (anyplatform, cuda, ascend, etc.)
2. **Fixtures**: Reuse existing fixtures from conftest.py
3. **Assertions**: Use `torch.testing.assert_close()` for numerical comparisons
4. **Cleanup**: Tests should not leave state behind

## Commit Strategy

### One Logical Change Per Commit

```bash
# GOOD: Separate concerns
git commit -m "feat: add torch.linalg.cholesky CUDA backend"
git commit -m "feat: add torch.linalg.cholesky FlagGems backend"
git commit -m "test: add tests for torch.linalg.cholesky"

# BAD: Everything in one commit
git commit -m "add cholesky"  # Contains backend + tests + docs + refactor
```

### Amending Commits

- Only amend unpushed commits
- Don't amend commits others may have based work on
- Use fixup commits if already pushed

## Debugging Tips

### When Tests Fail

1. **Read the full traceback** (don't just look at the last line)
2. **Check test log output** (FLAGOS_LOG=dispatch)
3. **Isolate the failure**: Run just one test at a time
4. **Compare with CPU**: Does the CPU version work?
5. **Check backend config**: Is the right backend being used?

### When Build Fails

1. **Read the full compiler error** (scroll up to find the actual error)
2. **Check environment**: Are all required env vars set?
3. **Clean build**: Try `rm -rf build/` and rebuild
4. **Check dependencies**: Is FlagGems/CUDA/etc. properly installed?

## Verification Before PR

### Minimum Verification

```bash
# 1. Linting
ruff check
ruff format --check

# 2. Unit tests
pytest tests/unit/ -v

# 3. Affected integration tests
pytest tests/integration/ops/test_<your_op>.py -v

# 4. Smoke test
python -c "
import torch
import torch_fl
x = torch.randn(4, 4, device='flagos')
print(torch.<your_op>(x))
"
```

### Comprehensive Verification (for major changes)

```bash
# All integration tests
pytest tests/integration/ -v

# Model tests
pytest tests/integration/test_qwen3_infer.py -v -s

# Platform-specific (if you have hardware)
pytest tests/integration/ops/ -v -m ascend
```

## Questions to Ask Yourself

Before submitting a PR, answer:

1. **Did I understand the root cause?** (Not just symptoms)
2. **Did I check existing patterns?** (Am I reinventing something?)
3. **Did I test edge cases?** (Empty tensors, large sizes, different dtypes)
4. **Did I verify on all affected platforms?** (Or justify why not)
5. **Did I update documentation?** (README, docstrings, CLAUDE.md)
6. **Did I include actual test output?** (Not just "it passes")
7. **Can a human reviewer understand my changes?** (Clear explanations)
8. **Is everything in English?** (PR, commits, comments)

## Anti-Patterns to Recognize

If you find yourself:
- Copying large blocks of code without understanding them
- Making the same fix in 10 places (should be a utility function)
- Writing "TODO" comments (finish it now or file an issue)
- Saying "I think this will work" (test it first)
- Skipping linting because "it's just whitespace"
- Writing commit messages like "fix", "update", "wip"

**Stop and reconsider your approach.**

## Summary

The key principle: **Understand before changing.**

1. Investigate thoroughly
2. Understand the root cause
3. Follow existing patterns
4. Test comprehensively
5. Document clearly
6. Verify everything

Your PR should make a reviewer say "this person clearly understood the codebase" not "this might work but I'm not sure why."
