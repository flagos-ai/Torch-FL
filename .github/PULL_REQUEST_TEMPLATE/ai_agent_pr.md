<!-- 
⚠️ AI-GENERATED PR TEMPLATE ⚠️
This template is for PRs created by AI agents (Claude Code, GitHub Codex, Cursor, etc.).
ALL sections marked REQUIRED must be completed before submission.
PRs missing required sections will be automatically closed.
-->

## AI Agent Information
- **Agent/Tool**: <!-- e.g., Claude Code CLI v1.x, GitHub Codex, Cursor -->
- **Model**: <!-- e.g., Claude Opus 5, GPT-4 Turbo -->
- **Human Reviewer**: @<!-- GitHub username of human who will review -->
- **Session Summary**: <!-- 1-2 sentence summary of the conversation that led to this PR -->

## Summary
<!-- REQUIRED: 2-4 sentence executive summary -->
<!-- What changes were made, why, and what is the impact? -->


## Change Type
<!-- REQUIRED: Check ALL that apply -->
- [ ] Bug Fix
- [ ] New Feature
- [ ] Performance Optimization
- [ ] Refactoring
- [ ] Documentation
- [ ] Testing
- [ ] CI/Infrastructure
- [ ] Breaking Change

## Platforms Affected
<!-- REQUIRED: Check ALL that apply -->
- [ ] CUDA
- [ ] MetaX
- [ ] Ascend
- [ ] PPU
- [ ] Platform-agnostic (all platforms)

## Problem Analysis
<!-- REQUIRED: Root cause analysis -->
### What was broken/missing?


### Why did it happen?


### Investigation process:
<!-- List files examined, commands run, hypotheses tested -->
1. 
2. 
3. 

## Solution Design
<!-- REQUIRED: Technical approach -->
### Implementation approach:


### Key design decisions:
<!-- Why this approach over alternatives? -->


### Code changes by file:
<!-- REQUIRED: List every modified file with brief explanation -->
- `path/to/file.cc` (lines X-Y): <!-- what changed and why -->
- `path/to/file.py` (lines A-B): <!-- what changed and why -->

### Changes by commit:
<!-- REQUIRED: Each commit with meaningful message -->
1. `commit-sha` - `commit-message`: <!-- why this commit -->
2. `commit-sha` - `commit-message`: <!-- why this commit -->

## Verification
<!-- REQUIRED: All sections must be completed -->

### Pre-submission Checklist
- [ ] **Linting passed** (ruff check, ruff format --check)
- [ ] **Type checking passed** (if applicable)
- [ ] **All tests pass** (unit + integration)
- [ ] **Manual testing completed** (include reproduction of original issue)
- [ ] **No debug/temporary code** (no print statements, commented code, TODOs)
- [ ] **Documentation updated** (README, AGENTS.md, docstrings as needed)
- [ ] **Commit messages follow conventions** (type: description format)
- [ ] **All text in English** (required per AGENTS.md)

### Linting Results
<!-- REQUIRED: Paste actual output -->
```bash
$ ruff check
# Output:

$ ruff format --check
# Output:
```

### Test Results
<!-- REQUIRED: Paste actual test output, not just "passed" -->
```bash
# Command:
pytest tests/... -v

# Output (including test count):

```

### Manual Verification
<!-- REQUIRED: Demonstrate the fix works -->
```bash
# Command to reproduce original issue:


# Output BEFORE fix:


# Output AFTER fix:

```

## Performance Impact
<!-- REQUIRED for performance changes, delete otherwise -->
<details>
<summary>Click to expand benchmark results</summary>

**Benchmark setup:**
- Hardware: 
- Workload: 
- Measurement method: 

**Results:**
| Metric | Before | After | Change |
|--------|--------|-------|--------|
|        |        |       |        |

</details>

## Breaking Changes
<!-- REQUIRED if "Breaking Change" checked above -->
**⚠️ This PR contains breaking changes**

### What breaks:
- 

### Migration path:
```python
# Old code:


# New code:

```

### Affected users:
<!-- Who needs to update their code? -->


## Code Quality Verification
<!-- REQUIRED for AI agents -->

### Style Consistency
- [ ] Matched existing code style in modified files
- [ ] Followed naming conventions (checked similar code)
- [ ] Comment density matches surrounding code
- [ ] Used project's existing utilities/helpers (no reinventing)

### Edge Cases Considered
<!-- REQUIRED: List edge cases you tested/handled -->
1. 
2. 
3. 

### Potential Risks
<!-- REQUIRED: What could go wrong? -->
1. 
2. 

### Rollback Plan
<!-- REQUIRED: How to revert if this breaks production? -->


## Related Work
<!-- REQUIRED: Link all related issues/PRs -->
- Fixes #
- Related to #
- Depends on #

## Explicitly Not Included
<!-- REQUIRED: What related work was intentionally excluded? -->
<!-- This proves you understand scope boundaries -->
- 
- 

## Human Review Notes
<!-- REQUIRED: What should the human reviewer focus on? -->
### Areas needing special attention:
1. 
2. 
3. 

### Questions for reviewer:
1. 
2. 

## Additional Context
<!-- Supporting information: benchmarks, screenshots, links -->


---

<!-- 
REJECTION CRITERIA - PRs will be closed if:
❌ Missing required sections
❌ No verification results (must paste actual output)
❌ No root cause analysis
❌ No manual testing proof
❌ Tests not passing
❌ Linting not passing
❌ Not written in English
❌ Contains debug/temporary code
❌ Doesn't follow existing code style
❌ No explanation of edge cases
❌ No human reviewer assigned
-->

---
🤖 Generated with [Claude Code](https://claude.com/claude-code)
