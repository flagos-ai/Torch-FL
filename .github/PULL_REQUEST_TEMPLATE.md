## Summary
<!-- 
REQUIRED: One-paragraph executive summary (2-4 sentences).
What does this PR do, why is it needed, and what is the impact?
-->


## Change Type
<!-- Check ALL that apply -->
- [ ] Bug Fix
- [ ] New Feature
- [ ] Performance Optimization
- [ ] Refactoring
- [ ] Documentation
- [ ] CI/Infrastructure
- [ ] Breaking Change

## Platforms Affected
<!-- Check ALL that apply -->
- [ ] CUDA
- [ ] MetaX
- [ ] Ascend
- [ ] PPU
- [ ] Platform-agnostic

## Detailed Description

### Problem
<!-- What problem does this PR solve? Include issue references if applicable -->


### Solution
<!-- How does this PR solve the problem? Describe the approach and key changes -->


### Changes by Commit
<!-- List each commit with a one-line summary -->
1. `commit-hash-or-title`: Brief description
2. `commit-hash-or-title`: Brief description

## Testing

### Test Coverage
<!-- Check what has been tested -->
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Manual testing completed

### Test Commands
<!-- Provide exact commands to verify the changes -->
```bash
# Unit tests
pytest tests/unit/test_<module>.py -v

# Integration tests
pytest tests/integration/ops/test_<op>.py -v

# Platform-specific
FLAGOS_BACKEND_CONFIG=... pytest ...
```

### Test Results
<!-- Paste test results or summarize (e.g., "125 passed, 0 failed") -->
```
# Test output here
```

## Verification

### Pre-merge Checklist
- [ ] All tests pass (see test results above)
- [ ] Code follows project style (ran linters)
- [ ] Documentation updated (if applicable)
- [ ] AGENTS.md updated (if adding new conventions)
- [ ] Commit messages follow conventions
- [ ] No debug code or temporary changes
- [ ] PR description is in **English** (required per AGENTS.md)

### Linting
<!-- REQUIRED: Proof that linting passes -->
```bash
# Ran: ruff check && ruff format --check
# Result: All passed / <N> issues fixed
```

## Performance Impact
<!-- Delete if not a performance change -->
- **Metric**: <!-- e.g., throughput, latency, memory -->
- **Before**: <!-- baseline measurement -->
- **After**: <!-- new measurement -->
- **Improvement**: <!-- percentage or absolute -->

## Breaking Changes
<!-- Delete if no breaking changes -->
**⚠️ This PR contains breaking changes:**
- [ ] API changes (document old → new)
- [ ] Config changes (document migration path)
- [ ] Build/runtime requirement changes

**Migration guide:**
<!-- Provide step-by-step migration instructions -->


## Explicitly Not Included
<!-- List related work that was intentionally excluded and why -->
- 
- 

## Related Issues/PRs
<!-- Link related issues and PRs -->
Fixes #<!-- issue number -->
Related to #<!-- issue/PR number -->

## Additional Context
<!-- Any other context, screenshots, benchmarks, or references -->


---
🤖 Generated with [Claude Code](https://claude.com/claude-code)
