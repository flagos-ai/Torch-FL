#!/bin/bash
# Batch run transformers tests for multiple models
# Copyright 2026 FlagOS Contributors
#
# Usage: $0 [chip] [repo]
#
# Each model's sweep reports one of three outcomes, and this script keeps them
# apart: clean (0), measured with findings to review (1), and nothing measured
# (2). Only the last one means the run is not a coverage result, so only the
# last one makes the batch fail; treating every non-zero exit as "the model is
# broken" hid both a broken interpreter and an unmeasured device.

set -uo pipefail

CHIP=${1:-GCU}
REPO=${2:-flagos-ai/Torch-FL}

echo "======================================================================="
echo "Batch Transformers Test Suite"
echo "======================================================================="
echo "Chip:   $CHIP"
echo "Repo:   $REPO"
echo "Models: bert, qwen3"
echo "======================================================================="
echo ""

MODELS=("bert" "qwen3")
CLEAN_COUNT=0
FINDINGS_COUNT=0
INVALID_COUNT=0
INVALID_MODELS=()
FINDING_MODELS=()

for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "======================================================================="
    echo "Processing: $MODEL"
    echo "======================================================================="

    bash scripts/transformers/transformers_auto_sweep.sh "$MODEL" "$CHIP" "$REPO"
    MODEL_STATUS=$?

    case ${MODEL_STATUS} in
        0)
            CLEAN_COUNT=$((CLEAN_COUNT + 1))
            echo "✓ $MODEL: measured, nothing to file"
            ;;
        1)
            FINDINGS_COUNT=$((FINDINGS_COUNT + 1))
            FINDING_MODELS+=("$MODEL")
            echo "⚠ $MODEL: measured, findings are awaiting review"
            ;;
        *)
            INVALID_COUNT=$((INVALID_COUNT + 1))
            INVALID_MODELS+=("$MODEL")
            echo "✗ $MODEL: nothing was measured (exit ${MODEL_STATUS}); its result is not coverage"
            ;;
    esac

    # Sleep between models to let device cool down
    if [ "$MODEL" != "${MODELS[-1]}" ]; then
        echo ""
        echo "Waiting 30 seconds before next model..."
        sleep 30
    fi
done

echo ""
echo "======================================================================="
echo "Batch Run Summary"
echo "======================================================================="
echo "Total models:     ${#MODELS[@]}"
echo "Nothing to file:  $CLEAN_COUNT"
echo "Findings:         $FINDINGS_COUNT"
echo "Not measured:     $INVALID_COUNT"

if [ $FINDINGS_COUNT -gt 0 ]; then
    echo ""
    echo "Models with findings (review the preview before filing anything):"
    for MODEL in "${FINDING_MODELS[@]}"; do
        echo "  - $MODEL"
    done
fi

if [ $INVALID_COUNT -gt 0 ]; then
    echo ""
    echo "Models that measured nothing (fix the environment and re-run them):"
    for MODEL in "${INVALID_MODELS[@]}"; do
        echo "  - $MODEL"
    done
fi

echo "======================================================================="

if [ $INVALID_COUNT -gt 0 ]; then
    exit 2
fi
if [ $FINDINGS_COUNT -gt 0 ]; then
    exit 1
fi
exit 0
