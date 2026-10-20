---
name: AI Agent Issue
about: Issue created by Claude Code or other AI coding agents
title: "[AI] "
labels: ai-generated
assignees: ''

---

<!-- 
⚠️ AI-GENERATED ISSUE CHECKLIST ⚠️
This template is for issues created by AI agents (Claude Code, Codex, etc.).
All fields marked REQUIRED must be filled before submission.
-->

## Issue Type
<!-- Select ONE -->
- [ ] Bug Report
- [ ] Feature Request
- [ ] Operator Implementation
- [ ] Platform Support
- [ ] Performance Issue
- [ ] Documentation

## AI Agent Information
- **Agent**: <!-- e.g., Claude Code CLI, GitHub Codex, Cursor -->
- **Model**: <!-- e.g., Claude Opus 5, GPT-4 -->
- **Session Context**: <!-- Brief description of the conversation that led to this issue -->

## Summary
<!-- REQUIRED: One-paragraph summary of the issue -->


## Environment (for bug reports)
<!-- REQUIRED for bug reports, delete if not applicable -->
<details>
<summary>Click to expand environment details</summary>

- **Platform**: <!-- CUDA / MetaX / Ascend / PPU -->
- **Python**: <!-- version -->
- **PyTorch**: <!-- version -->
- **torch_fl**: <!-- version/commit -->
- **Hardware**: <!-- GPU model, driver version -->

**Build config:**
```bash
# Full build command used
```

**Runtime config:**
```bash
# All relevant environment variables
```
</details>

## Reproduction
<!-- REQUIRED: Complete, runnable code that reproduces the issue -->
```python
import torch
import torch_fl

# Minimal reproducer - must be complete and self-contained
# Include all imports, setup, and the exact line that fails
```

## Expected vs Actual Behavior
**Expected:**
<!-- What should happen -->

**Actual:**
<!-- What actually happens, with full error traceback -->
```
# Full error output here
```

## Root Cause Analysis
<!-- REQUIRED for AI agents: Provide analysis of WHY this happens -->
<!-- Do not say "unknown" - investigate before filing -->


## Proposed Solution
<!-- REQUIRED for AI agents: Suggest specific fix or implementation approach -->


## Verification Plan
<!-- REQUIRED: How should the fix be tested? -->
- [ ] Unit test: <!-- describe test case -->
- [ ] Integration test: <!-- describe test case -->
- [ ] Manual verification: <!-- describe steps -->

## Context & Investigation
<!-- REQUIRED: What investigation was done before filing this issue? -->
<!-- Include: files read, commands run, hypotheses tested -->


## Related Code Locations
<!-- REQUIRED: Link to specific files and line numbers -->
- <!-- file.cc:123 - relevant function -->
- <!-- file.py:456 - where error originates -->

## Checklist - AI Agents MUST Complete All
- [ ] I have provided complete environment information
- [ ] I have included a minimal, self-contained reproducer
- [ ] I have included full error output with traceback
- [ ] I have analyzed the root cause (not just symptoms)
- [ ] I have proposed a specific solution with implementation approach
- [ ] I have identified affected code locations with line numbers
- [ ] I have described how to verify the fix
- [ ] I have checked for duplicate issues
- [ ] All text is in **English** (required per AGENTS.md)
- [ ] Code follows project conventions (checked existing code style)

---

<!-- 
REJECTION CRITERIA - Issues will be closed if:
❌ Missing reproducer
❌ No root cause analysis
❌ No proposed solution
❌ Incomplete environment info
❌ Not written in English
❌ No verification plan
-->
