# Claude Code Integration Guide for torch_fl

This document provides specific instructions for using Claude Code (CLI, desktop app, or web) when working on torch_fl.

## Quick Reference

### Before Any Work
```bash
# Read these in order:
1. cat CLAUDE.md                    # Project conventions
2. cat .github/AI_AGENT_GUIDE.md   # AI-specific guidelines
3. cat CONTRIBUTING.md              # Development workflow
```

### Creating PRs with Claude Code

```bash
# 1. Let Claude investigate and implement
# (Claude will read code, understand patterns, make changes)

# 2. Before submitting, run validation
python scripts/tools/validate_ai_pr.py --pr-body pr.md

# 3. Create PR using AI template
gh pr create \
  --title "fix: resolve CUDA stream sync race condition" \
  --body-file pr.md \
  --label "ai-generated"
```

## Claude Code Skills for torch_fl

### Pre-installed Skills

The repository includes these skills in `.claude/skills/`:
- `cuda-op-integration` - CUDA operator codegen integration
- `pre-pr-checks` - Run CI linting checks
- `setup-torch-fl-docker` - Docker environment setup

### Using Skills

When you ask Claude Code to work on torch_fl, it will automatically use these skills when appropriate. You can also invoke them explicitly:

```bash
# In Claude Code CLI
/pre-pr-checks    # Run linting and basic validation
```

## Common Tasks

### 1. Fixing a Bug

**Your prompt to Claude:**
```
There's a CUDA stream synchronization issue causing random test failures.
Error: "CUDA error: invalid argument at line 123 in file.cu"
Platform: CUDA 12.8, PyTorch 2.11.0
Can you investigate and fix this?
```

**Claude will:**
1. Read relevant code files
2. Search for similar issues in the codebase
3. Understand the CUDA stream model used
4. Propose and implement a fix
5. Run tests to verify
6. Create a PR with the AI template

**What you should verify:**
- Review the root cause analysis
- Check that tests actually pass (don't just trust the claim)
- Verify the fix doesn't break other platforms

### 2. Adding a New Operator

**Your prompt to Claude:**
```
Add support for torch.nn.functional.scaled_dot_product_attention
Target platform: CUDA
Backend: FlagGems Triton kernel (if available), else CUDA
```

**Claude will:**
1. Check if FlagGems already has this operator
2. Examine existing operator implementations
3. Implement following the project's patterns
4. Add tests with multiple shapes/dtypes
5. Update backend config files
6. Create PR with verification results

**What you should verify:**
- Operator works on actual models (not just synthetic tests)
- Performance is acceptable
- All platforms are considered (or justified why not)

### 3. Performance Investigation

**Your prompt to Claude:**
```
The Qwen3 inference is 30% slower than expected on CUDA.
Can you profile and optimize?
```

**Claude will:**
1. Set up profiling (torch.profiler, FLAGOS_LOG=dispatch)
2. Identify bottleneck operators
3. Check backend routing (are FlagGems kernels being used?)
4. Propose optimizations with benchmarks
5. Verify correctness is maintained

**What you should verify:**
- Benchmark methodology is sound
- Improvement is real (not measurement noise)
- No numerical accuracy loss

## Important Constraints

### Language (Critical)
```bash
# ❌ WRONG - Claude generates Chinese text
gh pr create --title "修复CUDA同步问题" --body "..."

# ✅ CORRECT - All English
gh pr create --title "fix: resolve CUDA stream sync race" --body "..."
```

**All GitHub text must be English:**
- PR titles and descriptions
- Commit messages
- Issue text
- Code comments

You should configure Claude Code to always output English for GitHub-facing text:

```
# In your prompt (if working in Chinese):
请用英文创建PR描述，因为项目要求所有GitHub文本必须是英文的。
```

### Testing Requirements

Claude Code should ALWAYS:
1. Run linting before claiming work is done
2. Run tests and paste actual output
3. Manually verify the fix works (not just unit tests)

```bash
# Minimum verification Claude must do:
ruff check && ruff format --check
pytest tests/unit/ -v
pytest tests/integration/ops/test_<relevant>.py -v

# Claude should paste output in PR, not just say "tests pass"
```

### Template Usage

For PRs, Claude Code MUST use:
- `.github/PULL_REQUEST_TEMPLATE/ai_agent_pr.md`

Not:
- `.github/PULL_REQUEST_TEMPLATE.md` (that's for humans)

You can help by specifying:
```
Please create a PR using the AI agent template at
.github/PULL_REQUEST_TEMPLATE/ai_agent_pr.md
```

## Verification Workflow

### 1. Pre-Implementation
- [ ] Claude has read relevant code files
- [ ] Claude understands existing patterns
- [ ] Claude identified root cause (not just symptoms)
- [ ] Approach discussed if non-trivial

### 2. Implementation
- [ ] Changes follow existing code style
- [ ] No debug code left behind
- [ ] Comments match surrounding density
- [ ] Reused existing utilities (didn't reinvent)

### 3. Testing
- [ ] Linting passes (Claude ran ruff)
- [ ] Unit tests pass (Claude pasted output)
- [ ] Integration tests pass (Claude pasted output)
- [ ] Manual verification done (Claude showed before/after)

### 4. PR Creation
- [ ] Used AI PR template
- [ ] All required sections filled
- [ ] Actual test output included (not just claims)
- [ ] Root cause analysis provided
- [ ] Edge cases listed
- [ ] Human reviewer assigned
- [ ] Everything in English

## Integration with Development Tools

### VS Code / JetBrains
If using Claude Code IDE extension:

```json
// .vscode/settings.json or similar
{
  "claude.prePrompt": "Remember: all GitHub text must be English per CLAUDE.md",
  "claude.postCodeReview": "Run: python scripts/tools/validate_ai_pr.py"
}
```

### CLI
If using Claude Code CLI:

```bash
# Add to your shell rc file
alias claude-pr='python scripts/tools/validate_ai_pr.py && gh pr create'

# Use the validation script before every PR
claude code "fix the bug in file.cc"
python scripts/tools/validate_ai_pr.py --pr-body pr.md
```

## Common Pitfalls

### ❌ Pitfall 1: Shallow Investigation
```
User: "Fix the test failure in test_conv.py"
Claude: [Makes a random change without understanding why it fails]
```

**Solution:** Ask Claude to investigate first:
```
User: "The test_conv.py::test_channels_last is failing. Can you:
1. Read the test and understand what it's checking
2. Run it to see the actual error
3. Investigate the root cause
4. Then propose a fix"
```

### ❌ Pitfall 2: No Verification
```
Claude: "I've fixed the issue. The change is in file.cc:123."
[No test output shown]
```

**Solution:** Always ask for evidence:
```
User: "Can you run the tests and show me the output?
Also run ruff to make sure style is correct."
```

### ❌ Pitfall 3: Language Mix
```
Claude: [Creates PR with Chinese commit messages]
```

**Solution:** Remind at start of session:
```
User: "Before we start: remember that all GitHub text (PRs, commits,
issues, code comments) must be in English per CLAUDE.md. This is
non-negotiable."
```

## Advanced Usage

### Custom Prompts for Complex Tasks

**For large refactorings:**
```
I need to refactor the CUDA operator backend to support per-operator
compilation flags. This affects ~50 files.

Can you:
1. First survey the current architecture (read key files)
2. Propose a design (before any changes)
3. Create a step-by-step implementation plan
4. Implement it incrementally (commit per logical step)
5. Verify each step passes tests
6. Create a PR with the full design explained

Follow .github/AI_AGENT_GUIDE.md for all steps.
```

**For cross-platform changes:**
```
Add support for torch.cumsum on all platforms (CUDA, MetaX, Ascend).

For each platform:
1. Check if vendor SDK has a primitive for this
2. Implement following that platform's existing patterns
3. Add platform-specific tests
4. Update the appropriate backend config

Create one commit per platform for easy review.
```

### Working with Memory

Claude Code has session memory. Use it:

```
User: "Remember: in this repo, CUDA boxing kernels are in
csrc/aten/generated/cuda_kernels.cc and are auto-generated.
Never edit that file directly - edit the template instead."

Claude: [Saves to memory]

# Later in session:
User: "Fix the CUDA relu kernel"
Claude: [Remembers to edit template, not generated file]
```

## Debugging Claude Code Issues

If Claude is not following guidelines:

1. **Check it read the guidelines:**
   ```
   User: "Have you read .github/AI_AGENT_GUIDE.md?
   Can you summarize the key requirements?"
   ```

2. **Explicitly reference requirements:**
   ```
   User: "The PR template requires a root cause analysis section.
   Can you fill that out with your investigation findings?"
   ```

3. **Provide examples:**
   ```
   User: "Here's an example of a good commit message for this repo:
   
   fix: resolve CUDA stream sync race in profiler callbacks
   
   The CUPTI callbacks must be registered before first CUDA context.
   Moved registration from lazy init to module import time.
   
   Can you format your commits like this?"
   ```

## Summary Checklist

Before asking Claude to create a PR, verify:

- [ ] Claude has read CLAUDE.md, AI_AGENT_GUIDE.md, CONTRIBUTING.md
- [ ] Claude investigated the issue (not just pattern-matched)
- [ ] Changes follow existing code patterns
- [ ] Tests were run and output was shown
- [ ] Linting passes
- [ ] PR will use AI agent template
- [ ] All text will be in English
- [ ] A human reviewer is assigned

## Getting Help

If you're unsure whether Claude is doing things correctly:

1. Check `.github/AI_AGENT_GUIDE.md` for the official guidelines
2. Look at merged PRs for examples
3. Ask Claude to show its reasoning before making changes
4. Use `python scripts/tools/validate_ai_pr.py` to catch issues early

## Related Documentation

- **CLAUDE.md** - Project conventions (MUST READ)
- **AI_AGENT_GUIDE.md** - Detailed AI guidelines
- **CONTRIBUTING.md** - Development workflow
- **README.md** - Build and usage instructions
