# GitHub Templates and AI Agent Guidelines

This directory contains templates and guidelines for contributing to torch_fl.

## For Human Contributors

### Issue Templates
- `ISSUE_TEMPLATE/bug_report.md` - Report bugs or unexpected behavior
- `ISSUE_TEMPLATE/feature_request.md` - Request new features
- `ISSUE_TEMPLATE/operator_support.md` - Request missing PyTorch operators
- `ISSUE_TEMPLATE/platform_support.md` - Request new hardware platform support

### PR Template
- `PULL_REQUEST_TEMPLATE.md` - Standard template for all human-created PRs

## For AI Agents (Claude Code, Codex, etc.)

### Mandatory Reading
1. **`AI_AGENT_GUIDE.md`** - Complete guidelines for AI agents working on this repo
2. **`PULL_REQUEST_TEMPLATE/ai_agent_pr.md`** - Required PR template for AI-generated PRs
3. **`ISSUE_TEMPLATE/ai_agent_issue.md`** - Required issue template for AI-filed issues

### Quick Start for AI Agents

Before creating any PR:

```bash
# 1. Read the AI agent guide
cat .github/AI_AGENT_GUIDE.md

# 2. Validate your changes
python scripts/tools/validate_ai_pr.py --pr-body pr_description.md

# 3. Run tests and save output
pytest tests/unit/ -v > test_output.txt 2>&1

# 4. Create PR with AI template
gh pr create \
  --title "fix: <description>" \
  --body-file pr_description.md \
  --label "ai-generated"
```

### Key Requirements for AI Agents

✅ **MUST DO:**
- Write everything in **English** (CLAUDE.md requirement)
- Use the AI PR template (`.github/PULL_REQUEST_TEMPLATE/ai_agent_pr.md`)
- Include actual test output (paste command results)
- Provide root cause analysis (not just symptoms)
- Verify linting passes before submission
- Assign a human reviewer
- List edge cases considered
- Explain investigation process

❌ **DO NOT:**
- Submit PRs without running tests
- Use vague commit messages ("fix bug", "update")
- Mix multiple unrelated changes in one commit
- Skip the root cause analysis
- Claim "tests pass" without pasting output
- Use Chinese or other non-English text
- Submit without human reviewer assignment

### Validation Script

Use `scripts/tools/validate_ai_pr.py` to check your PR before submission:

```bash
# Basic validation
python scripts/tools/validate_ai_pr.py

# With PR body file
python scripts/tools/validate_ai_pr.py --pr-body my_pr.md

# Skip linting (if you've already confirmed it passes)
python scripts/tools/validate_ai_pr.py --skip-lint
```

The script checks:
- ✓ Linting (ruff check, ruff format)
- ✓ Git status (uncommitted changes warning)
- ✓ Commit message format
- ✓ PR body required sections
- ✓ Language (English only)
- ✓ Debug code patterns
- ✓ Test availability

## Workflow Comparison

### Human Workflow
1. Create issue (optional for small changes)
2. Fork and create branch
3. Make changes following CONTRIBUTING.md
4. Run tests and linting
5. Create PR with standard template
6. Respond to review feedback

### AI Agent Workflow
1. Investigate thoroughly (read code, understand patterns)
2. Create issue with AI template (required for non-trivial changes)
3. Implement fix/feature
4. **Run validation script**
5. Run tests and **save output**
6. Create PR with AI template (include all outputs)
7. Human reviewer validates approach
8. Address feedback

## Template Selection Guide

| You are | Creating | Use Template |
|---------|----------|--------------|
| Human | Bug report | `bug_report.md` |
| Human | Feature request | `feature_request.md` |
| Human | PR | `PULL_REQUEST_TEMPLATE.md` |
| AI Agent | Any issue | `ai_agent_issue.md` |
| AI Agent | PR | `ai_agent_pr.md` |
| Anyone | Platform support | `platform_support.md` |
| Anyone | Operator request | `operator_support.md` |

## Related Documentation

- **Project conventions**: `/CLAUDE.md` (language requirements, workflows)
- **Contributing guide**: `/CONTRIBUTING.md` (development workflow)
- **README**: `/README.md` (project overview, build instructions)
- **AI agent guide**: `.github/AI_AGENT_GUIDE.md` (detailed AI guidelines)

## Questions?

- General questions: [Open a discussion](https://github.com/flagos-ai/PyTorch-Plugin-FL/discussions)
- Template issues: File an issue with the `meta` label
- Unclear guidelines: Tag maintainers in your PR/issue

## Template Maintenance

These templates are living documents. If you find:
- Missing required sections
- Unclear instructions
- Outdated information
- Better practices

Please open an issue or PR to improve them.
