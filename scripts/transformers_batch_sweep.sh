#!/bin/bash
# Batch run transformers tests for multiple models
# Copyright 2026 FlagOS Contributors

DEVICE=${1:-gcu}
CHIP=${2:-GCU}
REPO=${3:-flagos-ai/Torch-FL}

echo "======================================================================="
echo "Batch Transformers Test Suite"
echo "======================================================================="
echo "Device: $DEVICE"
echo "Chip:   $CHIP"
echo "Repo:   $REPO"
echo "Models: bert, qwen3"
echo "======================================================================="
echo ""

MODELS=("bert" "qwen3")
SUCCESS_COUNT=0
FAIL_COUNT=0
FAILED_MODELS=()

for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "======================================================================="
    echo "Processing: $MODEL"
    echo "======================================================================="

    bash scripts/transformers_auto_sweep.sh $MODEL $DEVICE $CHIP $REPO

    if [ $? -eq 0 ]; then
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        echo "✓ $MODEL completed successfully"
    else
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAILED_MODELS+=("$MODEL")
        echo "✗ $MODEL failed (continuing with next model...)"
    fi

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
echo "Total models: ${#MODELS[@]}"
echo "Successful:   $SUCCESS_COUNT"
echo "Failed:       $FAIL_COUNT"

if [ $FAIL_COUNT -gt 0 ]; then
    echo ""
    echo "Failed models:"
    for MODEL in "${FAILED_MODELS[@]}"; do
        echo "  - $MODEL"
    done
fi

echo "======================================================================="

exit $FAIL_COUNT
