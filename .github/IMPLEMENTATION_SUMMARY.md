# torch_fl GitHub Templates and Guidelines - Complete Solution

## 📋 Executive Summary

A comprehensive issue/PR template system designed for the torch_fl project, with specialized support for:
- **Human contributors** - Streamlined templates focusing on clarity
- **AI agents** (Claude Code, Codex, etc.) - Strict templates requiring verification and analysis

## 🎯 Key Features

### 1. **Language Enforcement**
- All GitHub text must be in English (PR/issue/commit messages)
- Automated detection of non-English text
- Multiple enforcement layers (docs, templates, validation script)

### 2. **AI Agent Standards**
- Mandatory root cause analysis (not just symptoms)
- Required verification with actual output (no claims without evidence)
- Investigation process documentation
- Edge case identification
- Human reviewer assignment

### 3. **Automated Validation**
- Pre-submission validation script (`scripts/tools/validate_ai_pr.py`)
- Checks linting, commit messages, PR structure, language
- Exit code 0/1 for CI integration

### 4. **Comprehensive Documentation**
- Guidelines for AI agents
- Claude Code specific integration guide
- Quick reference card
- Complete contribution workflow

## 📁 Files Created

### Issue Templates (`.github/ISSUE_TEMPLATE/`)
```
bug_report.md              - Standard bug report
feature_request.md         - Feature/enhancement request
platform_support.md        - New hardware platform request
operator_support.md        - Missing operator request
ai_agent_issue.md          - AI-specific issue (strict requirements)
```

### PR Templates
```
.github/PULL_REQUEST_TEMPLATE.md              - Human PR template
.github/PULL_REQUEST_TEMPLATE/ai_agent_pr.md  - AI PR template
```

### Documentation
```
CLAUDE.md                           - Project conventions (top-level)
CONTRIBUTING.md                     - Complete contribution guide
.github/AI_AGENT_GUIDE.md          - Detailed AI agent guidelines
.github/CLAUDE_CODE_GUIDE.md       - Claude Code integration guide
.github/TEMPLATES.md               - Template directory index
.github/QUICK_REFERENCE.txt        - Quick reference card
.github/IMPLEMENTATION_SUMMARY.md  - This document
```

### Automation
```
scripts/tools/validate_ai_pr.py          - Pre-submission validation script
```

## 🔧 How to Use

### For Human Contributors

1. **Create an issue**:
   - Go to GitHub Issues → New Issue
   - Select appropriate template (bug report, feature request, etc.)
   - Fill in all sections

2. **Create a PR**:
   ```bash
   # Fork and clone
   git clone https://github.com/YOUR_USERNAME/PyTorch-Plugin-FL.git
   cd PyTorch-Plugin-FL
   
   # Create branch
   git checkout -b fix/my-bugfix
   
   # Make changes, then verify
   ruff check && ruff format --check
   pytest tests/unit/ -v
   
   # Create PR (uses default template)
   gh pr create --title "fix: resolve CUDA sync issue" --body "..."
   ```

3. **Follow CONTRIBUTING.md** for detailed workflow

### For AI Agents (Claude Code, etc.)

1. **Before starting any work**:
   ```bash
   # Read guidelines (in order)
   cat CLAUDE.md
   cat .github/AI_AGENT_GUIDE.md
   cat .github/CLAUDE_CODE_GUIDE.md  # if using Claude Code
   ```

2. **Investigate and implement**:
   - Read relevant code to understand patterns
   - Find root cause (not just symptoms)
   - Implement following existing style
   - No debug code (print, TODO, etc.)

3. **Verify changes**:
   ```bash
   # Linting (required)
   ruff check
   ruff format --check
   
   # Tests (required - save output)
   pytest tests/unit/ -v > test_output.txt 2>&1
   pytest tests/integration/ops/test_*.py -v
   ```

4. **Validate before submission**:
   ```bash
   # Run validation script
   python scripts/tools/validate_ai_pr.py --pr-body pr_description.md
   ```

5. **Create PR with AI template**:
   ```bash
   gh pr create \
     --title "fix: <description>" \
     --body-file pr_description.md \
     --label "ai-generated"
   ```

## 🎨 Template Design Principles

### Separation of Concerns
- **Human templates**: Trust-based, streamlined
- **AI templates**: Verification-heavy, comprehensive

### Rejection Criteria
Templates explicitly state what will cause rejection:
- Missing root cause analysis
- No verification output
- Non-English text
- Missing required sections

### Evidence Over Claims
- "Tests pass" ❌ Not sufficient
- Pasted pytest output ✅ Required
- "I investigated" ❌ Not sufficient  
- Investigation process documented ✅ Required

## 🚀 Quick Commands Reference

### For AI Agents
```bash
# Pre-flight checks
python scripts/tools/validate_ai_pr.py --pr-body pr.md

# Linting
ruff check
ruff format --check

# Testing
pytest tests/unit/ -v
pytest tests/integration/ops/ -v -m "anyplatform or cuda"

# Create AI PR
gh pr create --body-file pr.md --label "ai-generated"
```

### For Humans
```bash
# Linting
ruff check && ruff format --check

# Testing
pytest tests/unit/ -v

# Create standard PR
gh pr create --title "feat: add new feature"
```

## 📊 Validation Script Details

`scripts/tools/validate_ai_pr.py` performs:

| Check | What it does |
|-------|-------------|
| **Linting** | Runs `ruff check` and `ruff format --check` |
| **Git status** | Warns about uncommitted changes |
| **Commit format** | Validates `<type>: <description>` format |
| **PR structure** | Checks required sections present |
| **Language** | Detects non-English text (Chinese) |
| **Debug code** | Warns about print/TODO/FIXME |
| **Test availability** | Confirms pytest is available |

**Exit codes:**
- `0` - All checks passed, ready to submit
- `1` - Failures found, fix before submitting

## 🔒 Enforcement Mechanisms

### Layer 1: Documentation
- CLAUDE.md states top-level requirements
- AI_AGENT_GUIDE.md provides detailed workflow
- Templates have rejection criteria

### Layer 2: Templates
- Required sections prevent omissions
- Checklists ensure completeness
- Examples show what good looks like

### Layer 3: Automation
- Validation script catches common issues
- Returns non-zero on failure (CI-ready)
- Can be integrated into pre-commit hooks

### Layer 4: Human Review
- Reviewers can quickly spot template violations
- Actual output visible (can't hallucinate)
- Root cause forces understanding

## 📈 Expected Outcomes

### Quality Improvements
- ✅ Higher quality AI-generated PRs
- ✅ Faster review cycle (all info upfront)
- ✅ Fewer back-and-forth iterations
- ✅ Better documentation of changes

### Consistency
- ✅ Standard format across all PRs
- ✅ English-only GitHub communication
- ✅ Consistent commit message style
- ✅ Predictable testing standards

### Transparency
- ✅ Investigation process visible
- ✅ Test results verifiable
- ✅ Edge cases documented
- ✅ Design decisions explained

## 🔄 Integration with CI/CD

### Current State
- Manual validation via script
- Human reviewer enforcement

### Future Enhancement
```yaml
# .github/workflows/validate-ai-pr.yml
name: Validate AI PR

on:
  pull_request:
    types: [opened, synchronize]

jobs:
  validate:
    if: contains(github.event.pull_request.labels.*.name, 'ai-generated')
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Validate AI PR
        run: |
          python scripts/tools/validate_ai_pr.py \
            --pr-body <(gh pr view ${{ github.event.pull_request.number }} \
            --json body -q .body)
```

## 🛠️ Maintenance

### Regular Updates
- Review templates quarterly
- Update based on contributor feedback
- Add new examples as patterns emerge
- Keep validation script in sync

### Metrics to Track
- AI vs human PR acceptance rate
- Review cycle time
- Common validation failures
- Template usage statistics

## 📚 Documentation Hierarchy

```
CLAUDE.md (top-level conventions)
    ├── CONTRIBUTING.md (detailed workflow)
    │   ├── .github/TEMPLATES.md (template index)
    │   └── .github/QUICK_REFERENCE.txt (quick ref)
    │
    └── AI-specific docs
        ├── .github/AI_AGENT_GUIDE.md (comprehensive guide)
        ├── .github/CLAUDE_CODE_GUIDE.md (Claude Code specific)
        └── scripts/tools/validate_ai_pr.py (automation)
```

## ❓ FAQ

**Q: Do ALL AI PRs need the strict template?**
A: Yes. Even with human supervision, if an AI implemented it, use the AI template. The investigation/verification sections are the key value.

**Q: What if the AI template is too long?**
A: It's intentionally comprehensive. It forces thoroughness. If sections can't be filled, the PR probably isn't ready.

**Q: Can humans use the AI template?**
A: Yes, but the human template is usually sufficient. Use the AI template if you want to provide extra detail.

**Q: How is English enforced?**
A: Three layers:
1. Documentation states requirement
2. Templates remind contributors
3. Validation script detects non-English text

**Q: What happens if someone ignores templates?**
A: Reviewers can request changes. Future: automation can block merges.

## 🎯 Success Criteria

This solution succeeds if:
- ✅ All new PRs use appropriate templates
- ✅ AI PRs include verification evidence
- ✅ No non-English text in GitHub communication
- ✅ Review cycle time decreases
- ✅ PR quality increases (fewer iterations)

## 🔗 Related Resources

- **GitHub**: https://github.com/flagos-ai/PyTorch-Plugin-FL
- **Issues**: Use templates to file bugs/features
- **Discussions**: For questions and general discussion
- **Contributing**: See CONTRIBUTING.md

## 📞 Getting Help

- **Template questions**: File issue with `meta` label
- **Unclear guidelines**: Tag maintainers in PR/issue
- **Tool problems**: Check .github/CLAUDE_CODE_GUIDE.md
- **General questions**: Use GitHub Discussions

## 🎉 Summary

This template system provides a complete solution for managing contributions to torch_fl:

✅ **Clear expectations** - Contributors know exactly what's needed  
✅ **Quality assurance** - Verification requirements catch issues early  
✅ **Language consistency** - English-only enforcement  
✅ **Automation** - Validation script reduces manual work  
✅ **Flexibility** - Different paths for humans vs AI agents  
✅ **Scalability** - Ready for CI/CD integration  

The system raises the bar for AI-generated contributions while keeping the process smooth for human contributors.

---

**Next Steps:**
1. Review this implementation summary
2. Test the validation script
3. Try creating a sample PR with each template
4. Gather feedback from contributors
5. Iterate based on real usage

For questions or suggestions, open an issue or discussion on GitHub.
