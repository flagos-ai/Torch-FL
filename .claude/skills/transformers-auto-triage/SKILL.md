---
name: transformers-auto-triage
description: Automated triage and issue filing for HuggingFace transformers test failures
---

# Transformers Auto-Triage Skill

**Purpose**: Automatically classify transformers test failures, verify real issues, deduplicate against baseline, generate issue bodies, and batch-file to GitHub.

**When to use**: After running `pytest tests/transformers/` and collecting test results to a JSON file.

**Time**: ~6 minutes for typical run (5min verification + 1min review)

## Prerequisites

1. Test results JSON from pytest-json-report
2. Chip name (e.g., "MUSA", "GCU", "Ascend")
3. Model name (e.g., "qwen3")
4. Transformers version (e.g., "4.47.0")
5. GitHub CLI (`gh`) authenticated

## Workflow

Run these 5 commands in sequence:

### Step 1: Classify failures

```bash
python scripts/transformers_triage.py \
  /path/to/test-results.json \
  --output /tmp/model-classified.json
```

**What it does**: Platform-agnostic crash detection and failure classification (OP_UNSUPPORTED, PRECISION, CRASH, FEATURE_UNSUPPORTED).

### Step 2: Verify in isolation

```bash
python scripts/transformers_verify.py \
  /tmp/model-classified.json \
  --output /tmp/model-verified.json \
  --test-source-dir tests/transformers/models/MODEL_NAME \
  --max-workers 4 \
  --timeout 60
```

**What it does**: Reruns each failed test in isolation to separate real failures from collateral damage. Uses parallel workers.

### Step 3: Deduplicate

```bash
python scripts/transformers_deduplicate.py \
  /tmp/model-verified.json \
  --output /tmp/model-new.json \
  --baseline docs/reference/hf-coverage.md \
  --repo flagos-ai/Torch-FL
```

**What it does**: Removes findings already in baseline or existing GitHub issues. Only new failures remain.

### Step 4: Generate issue bodies

```bash
python scripts/transformers_preview_issues.py \
  /tmp/model-new.json \
  --chip CHIP_NAME \
  --model MODEL_NAME \
  --transformers-version X.Y.Z \
  --issue-bodies-dir /tmp/transformers-issues \
  --preview /tmp/transformers-preview.md
```

**What it does**: Creates individual issue body files and consolidated preview markdown.

### Step 5: Review and approve

```bash
cat /tmp/transformers-preview.md
```

Human reviews the preview. Then either:

**File all issues**:
```bash
python scripts/transformers_file_issues.py \
  /tmp/model-new.json \
  --approve-all \
  --repo flagos-ai/Torch-FL
```

**File specific issues**:
```bash
python scripts/transformers_file_issues.py \
  /tmp/model-new.json \
  --approve fingerprint1 fingerprint2 fingerprint3 \
  --repo flagos-ai/Torch-FL
```

**What it does**: Creates GitHub issues, applies labels, updates baseline with issue numbers.

## Example: Complete run for qwen3 on MUSA

```bash
# Step 1: Classify
python scripts/transformers_triage.py \
  ~/qwen3-test-results.json \
  --output /tmp/qwen3-classified.json

# Step 2: Verify
python scripts/transformers_verify.py \
  /tmp/qwen3-classified.json \
  --output /tmp/qwen3-verified.json \
  --test-source-dir tests/transformers/models/qwen3 \
  --max-workers 4

# Step 3: Deduplicate
python scripts/transformers_deduplicate.py \
  /tmp/qwen3-verified.json \
  --output /tmp/qwen3-new.json \
  --baseline docs/reference/hf-coverage.md \
  --repo flagos-ai/Torch-FL

# Step 4: Generate previews
python scripts/transformers_preview_issues.py \
  /tmp/qwen3-new.json \
  --chip MUSA \
  --model qwen3 \
  --transformers-version 4.47.0 \
  --issue-bodies-dir /tmp/transformers-issues \
  --preview /tmp/transformers-preview.md

# Step 5: Review
cat /tmp/transformers-preview.md

# File all
python scripts/transformers_file_issues.py \
  /tmp/qwen3-new.json \
  --approve-all \
  --repo flagos-ai/Torch-FL
```

## Cross-chip support

All 5 tools are platform-agnostic and work with any chip using `torch.flagos`:
- MUSA
- GCU (Enflame)
- Ascend
- MetaX
- PPU
- Graphcore
- Habana
- Cambricon

No chip-specific keywords or detection logic. Universal crash patterns based on exit codes and generic signals.

## Dry run mode

Test the workflow without filing issues:

```bash
python scripts/transformers_file_issues.py \
  /tmp/model-new.json \
  --approve-all \
  --dry-run
```

## Notes

- **All findings from torch-fl failures are filed** - no manual filtering needed
- Baseline scoping: Only findings with identical fingerprint are considered "known"
- Verification timeout: Adjust `--timeout` based on model complexity (default: 60s)
- Rate limiting: 2s delay between issue creations (configurable with `--delay`)
- Issue labels: Auto-applied based on failure class (P0 for crashes, enhancement for unsupported ops)

## Output artifacts

- `/tmp/model-classified.json` - All failures with classifications
- `/tmp/model-verified.json` - Only real failures (collateral removed)
- `/tmp/model-new.json` - New findings not in baseline
- `/tmp/transformers-issues/issue-{fingerprint}.md` - Individual issue bodies
- `/tmp/transformers-preview.md` - Combined preview for review
- Updated `docs/reference/hf-coverage.md` - Baseline with filed issue numbers

## Time comparison

- **Old manual workflow**: ~2 hours
  - 30min: Manual classification and crash inspection
  - 60min: Rerun tests one by one
  - 20min: Search baseline and GitHub
  - 10min: Write issue bodies
  
- **New automated workflow**: ~6 minutes
  - 5min: Scripts run (mostly parallel verification)
  - 1min: Human review preview and approve

**Savings**: 95% reduction in manual work
