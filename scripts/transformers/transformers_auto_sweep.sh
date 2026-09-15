#!/bin/bash
# End-to-end measurement: test → triage → verify → deduplicate → preview issues
# Copyright 2026 FlagOS Contributors

set -e

MODEL=$1
DEVICE=${2:-gcu}
CHIP=${3:-GCU}
REPO=${4:-flagos-ai/Torch-FL}

if [ -z "$MODEL" ]; then
    echo "Usage: $0 <model> [device] [chip] [repo]"
    echo ""
    echo "Examples:"
    echo "  $0 bert                              # Use defaults: gcu, GCU, flagos-ai/Torch-FL"
    echo "  $0 qwen3 gcu GCU flagos-ai/Torch-FL  # Explicit all params"
    echo "  $0 bert musa MUSA flagos-ai/Torch-FL"
    echo ""
    echo "Supported chips: MUSA, GCU, Ascend, MetaX, PPU, IPU, Gaudi, MLU"
    exit 1
fi

WORK_DIR=/tmp/transformers-auto-sweep-${MODEL}-$(date +%Y%m%d-%H%M%S)
mkdir -p ${WORK_DIR}

echo "======================================================================="
echo "Transformers Auto Sweep + Issue Preview"
echo "======================================================================="
echo "Model:    $MODEL"
echo "Device:   $DEVICE"
echo "Chip:     $CHIP"
echo "Repo:     $REPO"
echo "Work Dir: $WORK_DIR"
echo "======================================================================="
echo ""

# Step 1: Run tests (resilient mode)
echo "[1/6] Running tests (resilient mode)..."
echo "  Batch size: 20 tests"
echo "  Batch timeout: 15 minutes"
echo ""

python tests/manual/transformers_hf_tests.py \
    --model ${MODEL} \
    --device ${DEVICE} \
    --resilient \
    --batch-size 20 \
    --batch-timeout 900 \
    --out ${WORK_DIR}/test-results.json \
    || {
        echo ""
        echo "Warning: Tests completed with errors (this is expected in resilient mode)"
    }

# Check if results were generated
if [ ! -f ${WORK_DIR}/test-results.json ]; then
    echo ""
    echo "ERROR: No test results generated"
    echo "Check if the model name is correct: $MODEL"
    exit 1
fi

TEST_COUNT=$(python3 -c "import json; d=json.load(open('${WORK_DIR}/test-results.json')); print(len(d.get('tests', [])))" 2>/dev/null || echo "0")
echo ""
echo "✓ Captured ${TEST_COUNT} test results"

if [ "$TEST_COUNT" -eq 0 ]; then
    echo "ERROR: No tests completed successfully"
    exit 1
fi

# Step 2: Triage
echo ""
echo "[2/6] Triaging failures..."
python scripts/transformers/transformers_triage.py \
    ${WORK_DIR}/test-results.json \
    --out ${WORK_DIR}/classified.json

CLASSIFIED_COUNT=$(python3 -c "import json; d=json.load(open('${WORK_DIR}/classified.json')); print(len(d.get('findings', [])))" 2>/dev/null || echo "0")
echo "✓ Classified ${CLASSIFIED_COUNT} findings"

if [ "$CLASSIFIED_COUNT" -eq 0 ]; then
    echo ""
    echo "No failures to triage. All tests passed!"
    exit 0
fi

# Step 3: Verify (serial isolation)
echo ""
echo "[3/6] Verifying failures in isolation (serial)..."
python scripts/transformers/transformers_verify.py \
    ${WORK_DIR}/classified.json \
    --out ${WORK_DIR}/verified.json \
    --test-source-dir /root/.cache/torch_fl/hf-tests \
    --transformers-version "$(python3 -c 'import transformers; print(transformers.__version__)')" \
    --workers 1 \
    --timeout 120 \
    || {
        echo ""
        echo "Warning: Verification completed with some errors"
    }

VERIFIED_COUNT=$(python3 -c "import json; d=json.load(open('${WORK_DIR}/verified.json')); print(len(d.get('findings', [])))" 2>/dev/null || echo "0")
echo "✓ Verified ${VERIFIED_COUNT} findings"

# Step 4: Deduplicate
echo ""
echo "[4/6] Deduplicating against baseline and GitHub..."
python scripts/transformers/transformers_deduplicate.py \
    ${WORK_DIR}/verified.json \
    --out ${WORK_DIR}/new.json \
    --coverage-file docs/reference/hf-coverage.md \
    --repo ${REPO}

NEW_COUNT=$(python3 -c "import json; d=json.load(open('${WORK_DIR}/new.json')); print(len(d.get('findings', [])))" 2>/dev/null || echo "0")
echo "✓ Found ${NEW_COUNT} new findings"

if [ "$NEW_COUNT" -eq 0 ]; then
    echo ""
    echo "No new issues to file. All findings are known!"
    echo "Results saved in: ${WORK_DIR}"
    exit 0
fi

# Step 5: Preview
echo ""
echo "[5/6] Generating issue previews..."

# Get transformers version
TF_VERSION=$(python3 -c "import transformers; print(transformers.__version__)" 2>/dev/null || echo "unknown")

# Get torch_fl commit
TF_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

python scripts/transformers/transformers_preview_issues.py \
    ${WORK_DIR}/new.json \
    --chip ${CHIP} \
    --transformers-version ${TF_VERSION} \
    --torch-fl-commit ${TF_COMMIT} \
    --issue-bodies-dir ${WORK_DIR}/issues \
    --out ${WORK_DIR}/preview.md

echo ""
echo "======================================================================="
echo "Issue Preview (${NEW_COUNT} issues)"
echo "======================================================================="
cat ${WORK_DIR}/preview.md
echo ""
echo "======================================================================="

# Step 6: Stop for explicit authorization
echo ""
echo "[6/6] Issue previews are ready for human review."
echo "No GitHub writes were performed."
echo ""
echo "File only explicitly approved fingerprints, for example:"
echo "  python scripts/transformers/transformers_file_issues.py ${WORK_DIR}/new.json \\"
echo "      --approve <fingerprint> [<fingerprint> ...] \\"
echo "      --repo ${REPO} \\"
echo "      --issue-bodies-dir ${WORK_DIR}/issues"
echo ""
echo "======================================================================="
echo "✓ Measurement and preview complete"
echo "======================================================================="
echo "Results saved in: ${WORK_DIR}"
echo ""
echo "Files:"
echo "  - test-results.json   (raw test output)"
echo "  - classified.json     (triaged findings)"
echo "  - verified.json       (isolated verification)"
echo "  - new.json            (deduplicated new findings)"
echo "  - preview.md          (issue preview)"
echo "  - issues/*.md         (individual issue bodies)"
echo "======================================================================="
