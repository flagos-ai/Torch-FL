# Transformers Auto-Triage Workflow

Automated pipeline for triaging HuggingFace transformers test failures and filing GitHub issues.

## Overview

**Time**: ~6 minutes (5min automation + 1min human review)  
**Replaces**: 2 hours of manual classification, verification, and issue filing

## Quick Start

```bash
# Run complete pipeline for a model
MODEL=qwen3
CHIP=MUSA
TF_VERSION=4.47.0

# 1. Classify failures
python scripts/transformers_triage.py \
  ~/test-results/${MODEL}.json \
  --output /tmp/${MODEL}-classified.json

# 2. Verify in isolation
python scripts/transformers_verify.py \
  /tmp/${MODEL}-classified.json \
  --output /tmp/${MODEL}-verified.json \
  --test-source-dir tests/transformers/models/${MODEL} \
  --max-workers 4

# 3. Deduplicate
python scripts/transformers_deduplicate.py \
  /tmp/${MODEL}-verified.json \
  --output /tmp/${MODEL}-new.json \
  --baseline docs/reference/hf-coverage.md \
  --repo flagos-ai/Torch-FL

# 4. Generate issue previews
python scripts/transformers_preview_issues.py \
  /tmp/${MODEL}-new.json \
  --chip ${CHIP} \
  --model ${MODEL} \
  --transformers-version ${TF_VERSION} \
  --preview /tmp/${MODEL}-preview.md

# 5. Review and file
cat /tmp/${MODEL}-preview.md
python scripts/transformers_file_issues.py \
  /tmp/${MODEL}-new.json \
  --approve-all \
  --repo flagos-ai/Torch-FL
```

## Pipeline Stages

### 1. Classification (`transformers_triage.py`)

**Input**: pytest-json-report output  
**Output**: Classified failures with fingerprints

**What it does**:
- Platform-agnostic crash detection (exit codes, signals, timeouts)
- Classifies into: OP_UNSUPPORTED, PRECISION, CRASH, FEATURE_UNSUPPORTED
- Groups related failures by fingerprint
- No chip-specific keywords required

**Example**:
```bash
python scripts/transformers_triage.py \
  ~/qwen3-test-results.json \
  --output /tmp/qwen3-classified.json
```

**Output summary**:
```
Summary:
  total_tests: 847
  passed: 520
  failed: 327
  OP_UNSUPPORTED: 45
  PRECISION: 12
  CRASH: 3
  FEATURE_UNSUPPORTED: 8
  UNKNOWN: 259
```

### 2. Verification (`transformers_verify.py`)

**Input**: Classified failures  
**Output**: Verified real failures (collateral removed)

**What it does**:
- Reruns each failed test in subprocess isolation
- Parallel execution (default 4 workers)
- Separates real failures from collateral damage
- Configurable timeout per test

**Example**:
```bash
python scripts/transformers_verify.py \
  /tmp/qwen3-classified.json \
  --output /tmp/qwen3-verified.json \
  --test-source-dir tests/transformers/models/qwen3 \
  --max-workers 4 \
  --timeout 60
```

**Output summary**:
```
Verification summary:
  REPRODUCED: 58
  COLLATERAL: 269
  total_findings: 68
  verified_findings: 58
```

### 3. Deduplication (`transformers_deduplicate.py`)

**Input**: Verified failures  
**Output**: New findings not in baseline

**What it does**:
- Checks fingerprints against `docs/reference/hf-coverage.md`
- Searches GitHub issues via `gh` CLI
- Only keeps genuinely new findings
- Fingerprint-based exact matching

**Example**:
```bash
python scripts/transformers_deduplicate.py \
  /tmp/qwen3-verified.json \
  --output /tmp/qwen3-new.json \
  --baseline docs/reference/hf-coverage.md \
  --repo flagos-ai/Torch-FL
```

**Output summary**:
```
Deduplication summary:
  new: 15
  known_baseline: 32
  known_github: 11
  total_new_findings: 15
```

### 4. Preview Generation (`transformers_preview_issues.py`)

**Input**: New findings  
**Output**: Issue bodies and consolidated preview

**What it does**:
- Generates individual issue body files
- Creates consolidated markdown preview
- Follows `.github/ISSUE_TEMPLATE/ai_agent_issue.md` format
- Includes root cause, investigation, and evidence

**Example**:
```bash
python scripts/transformers_preview_issues.py \
  /tmp/qwen3-new.json \
  --chip MUSA \
  --model qwen3 \
  --transformers-version 4.47.0 \
  --issue-bodies-dir /tmp/transformers-issues \
  --preview /tmp/qwen3-preview.md
```

**Output files**:
```
/tmp/transformers-issues/
  ├── issue-a1b2c3d4e5f6.md
  ├── issue-b2c3d4e5f6g7.md
  └── ...
/tmp/qwen3-preview.md
```

### 5. Issue Filing (`transformers_file_issues.py`)

**Input**: New findings + approval  
**Output**: Filed GitHub issues + updated baseline

**What it does**:
- Creates GitHub issues via `gh` CLI
- Applies appropriate labels (P0 for crashes, enhancement for unsupported ops)
- Updates `docs/reference/hf-coverage.md` with issue numbers
- Rate limiting (2s delay between issues)

**Example - file all**:
```bash
python scripts/transformers_file_issues.py \
  /tmp/qwen3-new.json \
  --approve-all \
  --repo flagos-ai/Torch-FL
```

**Example - file specific**:
```bash
python scripts/transformers_file_issues.py \
  /tmp/qwen3-new.json \
  --approve a1b2c3d4e5f6 b2c3d4e5f6g7 c3d4e5f6g7h8 \
  --repo flagos-ai/Torch-FL
```

**Example - dry run**:
```bash
python scripts/transformers_file_issues.py \
  /tmp/qwen3-new.json \
  --approve-all \
  --dry-run
```

**Output summary**:
```
Filing complete
============================================================
Successfully filed: 15
Failed: 0

Filed issues:
  #251: a1b2c3d4e5f6
  #252: b2c3d4e5f6g7
  ...

Updated docs/reference/hf-coverage.md with 15 issue references
```

## Cross-Platform Support

All tools work with any chip using `torch.flagos`:

- **MUSA** (Moore Threads)
- **GCU** (Enflame)
- **Ascend** (Huawei)
- **MetaX** (MetaX)
- **PPU** (Stream Computing)
- **Graphcore** IPU
- **Habana** Gaudi
- **Cambricon** MLU

No chip-specific keywords or detection logic. Universal crash patterns based on:
- Exit codes (segfault, timeout)
- Generic error signals
- Device context poisoning
- Python exception types

## Failure Classifications

### OP_UNSUPPORTED
Operation not implemented on the chip backend.

**Detection**: Error messages containing "not implemented", "unsupported", or operator name extraction.

**Labels**: `ai-generated`, `enhancement`

**Example**: `aten::scaled_dot_product_attention not supported on flagos device`

### PRECISION
Numerical mismatch vs CPU reference.

**Detection**: Assertion errors with tensor comparisons, "torch.testing.assert_close" failures.

**Labels**: `ai-generated`, `bug`

**Example**: `Tensor mismatch: max_diff=0.05, expected < 0.001`

### CRASH
Segfault, SIGABRT, timeout, or device context poisoning.

**Detection**: Exit codes, signal names, timeout markers, context poison flag.

**Labels**: `ai-generated`, `bug`, `P0`

**Example**: `Segmentation fault (core dumped)`

### FEATURE_UNSUPPORTED
PyTorch feature not available on PrivateUse1 devices.

**Detection**: Error messages about registration, dispatch, or device type restrictions.

**Labels**: `ai-generated`, `enhancement`

**Example**: `flex_attention not supported for PrivateUse1 devices`

## Configuration

### Verification Timeout

Adjust per-test timeout based on model complexity:

```bash
# Fast models (< 1s per test)
python scripts/transformers_verify.py ... --timeout 30

# Average models (1-5s per test)
python scripts/transformers_verify.py ... --timeout 60

# Large models (5-30s per test)
python scripts/transformers_verify.py ... --timeout 120
```

### Parallel Workers

Adjust based on available CPU cores and memory:

```bash
# Conservative (low memory)
python scripts/transformers_verify.py ... --max-workers 2

# Balanced (default)
python scripts/transformers_verify.py ... --max-workers 4

# Aggressive (high memory, many cores)
python scripts/transformers_verify.py ... --max-workers 8
```

### Issue Filing Rate Limit

Adjust delay between GitHub API calls:

```bash
# Slower (more conservative)
python scripts/transformers_file_issues.py ... --delay 5.0

# Default
python scripts/transformers_file_issues.py ... --delay 2.0

# Faster (risk rate limiting)
python scripts/transformers_file_issues.py ... --delay 1.0
```

## Baseline Management

### Recording Baseline

After filing issues, the baseline is automatically updated:

```markdown
## qwen3 - MUSA - 2026-09-02

- **Transformers**: 4.47.0
- **PyTorch**: 2.10.0
- **Torch-FL**: 64e60dd
- **Total tests**: 847
- **Passed**: 520
- **Failed**: 327

| Fingerprint | Class | Subject | Issue |
| --- | --- | --- | --- |
| `a1b2c3d4e5f6` | OP_UNSUPPORTED | scaled_dot_product_attention | [#251](https://github.com/flagos-ai/Torch-FL/issues/251) |
| `b2c3d4e5f6g7` | CRASH | device context poisoned | [#252](https://github.com/flagos-ai/Torch-FL/issues/252) |
```

### Baseline Scoping

Only findings with **identical fingerprint** are considered "known". This means:

- Same operation/error pattern
- Same failure mechanism
- Same subject

Different models hitting the same underlying issue share the same fingerprint and are deduplicated.

## Troubleshooting

### No findings classified

**Symptom**: All failures marked as "UNKNOWN"

**Fix**: The test output format may not match expected patterns. Check that:
- Input is valid pytest-json-report format
- Test actually failed (not skipped or passed)
- Error details are present in JSON

### Verification hangs

**Symptom**: `transformers_verify.py` stuck on one test

**Fix**: Increase timeout or kill hanging process:
```bash
# Increase timeout
python scripts/transformers_verify.py ... --timeout 120

# Find and kill hanging test
ps aux | grep pytest
kill -9 <pid>
```

### GitHub API rate limit

**Symptom**: `transformers_file_issues.py` returns 403 errors

**Fix**: Increase delay between requests:
```bash
python scripts/transformers_file_issues.py ... --delay 5.0
```

### Duplicate issues filed

**Symptom**: Same issue filed multiple times

**Fix**: Ensure baseline is up to date before filing. If duplicates exist:
1. Close the duplicate issues on GitHub
2. Update baseline with the kept issue number
3. Rerun deduplication before next filing

## See Also

- `.claude/skills/transformers-auto-triage/SKILL.md` - Skill documentation
- `.github/ISSUE_TEMPLATE/ai_agent_issue.md` - Issue template format
- `docs/reference/hf-coverage.md` - Test baseline records
- `.claude/skills/transformers-test/SKILL.md` - Original manual workflow
