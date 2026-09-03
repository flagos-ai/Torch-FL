---
name: transformers-full-sweep
description: End-to-end transformers testing with resilient mode and automatic issue filing
---

# Transformers Full Sweep Skill

**Purpose**: Run HuggingFace transformers tests from scratch (with resilient mode), automatically triage failures, verify, deduplicate, and file issues to GitHub.

**When to use**: When you want to test a transformers model on a new chip and automatically file all issues without manual intervention.

**Time**: Varies by model:
- Small models (bert, distilbert): ~15-20 minutes
- Medium models (gpt2, t5): ~30-40 minutes
- Large models (llama, qwen3): ~60-90 minutes

## Prerequisites

1. Chip/device configured (e.g., GCU, MUSA, Ascend)
2. torch_fl installed and device accessible
3. HuggingFace transformers installed
4. GitHub CLI (`gh`) authenticated

## Single Command Usage

```bash
bash scripts/transformers_auto_sweep.sh <model> <device> <chip> [repo]
```

**Parameters**:
- `<model>`: Model name (e.g., "bert", "qwen3")
- `<device>`: Device name for torch (e.g., "gcu", "musa")
- `<chip>`: Chip name for issue titles (e.g., "GCU", "MUSA")
- `[repo]`: GitHub repo (default: "flagos-ai/Torch-FL")

**Examples**:
```bash
# Test bert on GCU
bash scripts/transformers_auto_sweep.sh bert gcu GCU

# Test qwen3 on MUSA
bash scripts/transformers_auto_sweep.sh qwen3 musa MUSA

# Test on Ascend with custom repo
bash scripts/transformers_auto_sweep.sh gpt2 ascend Ascend my-org/my-repo
```

## Batch Mode (Multiple Models)

```bash
bash scripts/transformers_batch_sweep.sh <device> <chip> [repo]
```

Runs bert and qwen3 in sequence.

**Example**:
```bash
bash scripts/transformers_batch_sweep.sh gcu GCU
```

## What It Does

### Stage 1: Run Tests (Resilient Mode)

```bash
python tests/manual/transformers_hf_tests.py \
    --model <model> \
    --device <device> \
    --resilient \
    --batch-size 20 \
    --batch-timeout 900 \
    --out /tmp/transformers-auto-sweep-<model>/test-results.json
```

**Resilient mode features**:
- Tests run in batches of 20
- If a batch crashes, it's marked and the next batch continues
- Each batch has 15-minute timeout
- Device reset attempted after crash
- Incremental JSON output (results saved as they complete)

**Result**: Even if some tests crash, you get partial results for all completed tests.

### Stage 2: Triage

```bash
python scripts/transformers_triage.py \
    test-results.json \
    --out classified.json
```

Classifies failures into: OP_UNSUPPORTED, PRECISION, CRASH, FEATURE_UNSUPPORTED

### Stage 3: Verify

```bash
python scripts/transformers_verify.py \
    classified.json \
    --out verified.json \
    --test-source-dir tests/transformers/models/<model> \
    --workers 4
```

Reruns each failure in isolation with 4 parallel workers to separate real failures from collateral damage.

### Stage 4: Deduplicate

```bash
python scripts/transformers_deduplicate.py \
    verified.json \
    --out new.json \
    --coverage-file docs/reference/hf-coverage.md \
    --repo <repo>
```

Removes findings already in baseline or existing GitHub issues.

### Stage 5: Preview

```bash
python scripts/transformers_preview_issues.py \
    new.json \
    --chip <chip> \
    --transformers-version <version> \
    --torch-fl-commit <commit> \
    --issue-bodies-dir issues/ \
    --out preview.md
```

Generates individual issue bodies and consolidated preview.

### Stage 6: File Issues (Automatic)

```bash
python scripts/transformers_file_issues.py \
    new.json \
    --approve-all \
    --repo <repo>
```

Automatically files all new issues to GitHub with proper labels and updates baseline.

## Output

All artifacts saved in `/tmp/transformers-auto-sweep-<model>-<timestamp>/`:
```
test-results.json       # Raw test output
classified.json         # Triaged findings
verified.json          # Isolated verification results
new.json              # Deduplicated new issues
preview.md            # Issue preview
issues/
  issue-<fp1>.md      # Individual issue bodies
  issue-<fp2>.md
  ...
```

## Manual Mode (Step by Step)

If you need more control, run each stage manually:

```bash
MODEL=bert
DEVICE=gcu
CHIP=GCU
WORK_DIR=/tmp/transformers-sweep-${MODEL}
mkdir -p ${WORK_DIR}

# Stage 1: Test (resilient)
python tests/manual/transformers_hf_tests.py \
    --model ${MODEL} \
    --device ${DEVICE} \
    --resilient \
    --batch-size 20 \
    --batch-timeout 900 \
    --out ${WORK_DIR}/test-results.json

# Stage 2: Triage
python scripts/transformers_triage.py \
    ${WORK_DIR}/test-results.json \
    --out ${WORK_DIR}/classified.json

# Stage 3: Verify
python scripts/transformers_verify.py \
    ${WORK_DIR}/classified.json \
    --out ${WORK_DIR}/verified.json \
    --test-source-dir tests/transformers/models/${MODEL} \
    --workers 4

# Stage 4: Deduplicate
python scripts/transformers_deduplicate.py \
    ${WORK_DIR}/verified.json \
    --out ${WORK_DIR}/new.json \
    --coverage-file docs/reference/hf-coverage.md \
    --repo flagos-ai/Torch-FL

# Stage 5: Preview
python scripts/transformers_preview_issues.py \
    ${WORK_DIR}/new.json \
    --chip ${CHIP} \
    --transformers-version $(python -c "import transformers; print(transformers.__version__)") \
    --torch-fl-commit $(git rev-parse --short HEAD) \
    --issue-bodies-dir ${WORK_DIR}/issues \
    --out ${WORK_DIR}/preview.md

# Review
cat ${WORK_DIR}/preview.md

# Stage 6: File
python scripts/transformers_file_issues.py \
    ${WORK_DIR}/new.json \
    --approve-all \
    --repo flagos-ai/Torch-FL
```

## Resilient Mode Details

**Why resilient mode?**
New chips often have crashes/hangs that kill the entire test process. Resilient mode ensures:
- Partial results are always saved
- One crash doesn't block all other tests
- You can still triage and file issues for completed tests

**Configuration**:
- `--batch-size 20`: Each batch has 20 tests (adjustable)
- `--batch-timeout 900`: Each batch times out after 15 minutes (adjustable)

**Adjust for your environment**:
```bash
# Frequent crashes? Smaller batches
--batch-size 10 --batch-timeout 600

# Stable environment? Larger batches
--batch-size 50 --batch-timeout 1200

# Large models? Longer timeout
--batch-size 20 --batch-timeout 1800
```

## Supported Chips

All chips using `torch.flagos` backend:
- MUSA (Moore Threads)
- GCU (Enflame)
- Ascend (Huawei)
- MetaX
- PPU
- IPU (Graphcore)
- Gaudi (Habana)
- MLU (Cambricon)

No chip-specific code required. Universal crash detection patterns.

## Common Scenarios

### Scenario 1: First time testing on new chip

```bash
# Start with bert (small, fast)
bash scripts/transformers_auto_sweep.sh bert gcu GCU

# If successful, try qwen3 (larger)
bash scripts/transformers_auto_sweep.sh qwen3 gcu GCU
```

### Scenario 2: Batch test priority models

```bash
# Runs bert and qwen3 automatically
bash scripts/transformers_batch_sweep.sh gcu GCU
```

### Scenario 3: Test crashed, want to retry

Just re-run the same command. Resilient mode will:
- Skip batches that completed successfully (if using same output file)
- Retry crashed batches
- Continue from where it left off

### Scenario 4: Test without filing issues

Remove Stage 6 or use `--dry-run`:
```bash
python scripts/transformers_file_issues.py \
    new.json \
    --approve-all \
    --dry-run
```

## Troubleshooting

### Issue: No test results generated

**Cause**: Model name wrong or collect failed

**Fix**:
```bash
# List available models
python tests/manual/transformers_hf_tests.py --list-models | grep <model>

# Test collect only
python tests/manual/transformers_hf_tests.py --model bert --collect-only
```

### Issue: All batches crash

**Cause**: Environment problem (driver, dependencies)

**Fix**:
```bash
# Check device
python -c "import torch, torch_fl; print(torch.flagos.device_count())"

# Check dependencies
pip list | grep -E "transformers|torch"

# Sanity test
python -c "
import torch
import torch_fl
x = torch.randn(2, 3, device='flagos')
print('Device OK:', x.device)
"
```

### Issue: Batches very slow

**Cause**: Batch size too large or timeout too long

**Fix**: Reduce batch size and timeout
```bash
--batch-size 10 --batch-timeout 600
```

### Issue: Issues not filed

**Cause**: GitHub auth or permissions

**Fix**:
```bash
# Check auth
gh auth status

# Test with dry-run
python scripts/transformers_file_issues.py \
    new.json \
    --approve-all \
    --dry-run
```

## Comparison with transformers-auto-triage

| Feature | transformers-auto-triage | transformers-full-sweep |
|---------|-------------------------|------------------------|
| Test execution | Manual (you provide JSON) | Automatic (resilient mode) |
| Crash handling | N/A | Batch isolation + recovery |
| Steps | 5 (from existing results) | 6 (including testing) |
| Use case | You already ran tests | Start from scratch |

**When to use which?**
- **Use full-sweep**: Testing new chip, want end-to-end automation
- **Use auto-triage**: Already have test results, just want triage+filing

## Best Practices

1. **First run on new chip**: Use resilient mode with small batches (10-20)
2. **Stable environment**: Can disable resilient mode for speed
3. **Large models**: Increase batch timeout to 1800s (30 min)
4. **Debugging crashes**: Check `/tmp/transformers-auto-sweep-*/test-results.json` for crash details
5. **Reuse results**: Keep work directories for analysis

## Time Estimates

| Model | Test Duration | Total Duration |
|-------|--------------|----------------|
| bert | 10-15 min | 15-20 min |
| distilbert | 5-10 min | 10-15 min |
| gpt2 | 15-25 min | 25-35 min |
| t5 | 20-30 min | 30-40 min |
| qwen3 | 40-60 min | 60-90 min |
| llama | 50-80 min | 80-120 min |

Times are for resilient mode on typical hardware. Adjust based on your chip performance.

## Advanced: Custom Configuration

For chip-specific tuning, you can modify the scripts:

```bash
# Edit batch size per model
vi scripts/transformers_auto_sweep.sh
# Line 34: Change --batch-size 20

# Edit timeout per model  
# Line 35: Change --batch-timeout 900

# Add preflight checks
# Add before line 28
```

## Integration with CI/CD

The scripts return proper exit codes:
- `0`: Success
- `1`: Test failures (but issues filed)
- `2`: Environment error

```bash
# In CI
bash scripts/transformers_auto_sweep.sh bert gcu GCU || {
    echo "Tests failed but issues were filed"
    exit 0  # Don't fail CI
}
```

## Documentation

- **Quickstart**: `docs/workflows/resilient-testing-quickstart.md`
- **Design**: `docs/design/robust-harness-proposal.md`
- **Auto-triage (no testing)**: `.claude/skills/transformers-auto-triage/SKILL.md`
