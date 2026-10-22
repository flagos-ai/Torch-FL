#!/bin/bash
# End-to-end measurement: test → triage → verify → deduplicate → preview issues
# Copyright 2026 FlagOS Contributors
#
# Usage: $0 <model> [chip] [repo]
#
# The run measures a device; it does not modify anything. Every step is checked
# explicitly rather than with ``set -e``, because the pipeline has to tell three
# outcomes apart: clean (0), measured-with-findings (1), and nothing-was-measured
# (2). Only the last one stops the pipeline, and it stops it loudly --- a broken
# environment must never be summarized as a clean sweep.

set -uo pipefail

MODEL=${1:-}
CHIP=${2:-GCU}
REPO=${3:-flagos-ai/Torch-FL}
PYTHON=${PYTHON:-python3}

if [ -z "$MODEL" ]; then
    echo "Usage: $0 <model> [chip] [repo]"
    echo ""
    echo "Examples:"
    echo "  $0 bert                             # defaults: GCU, flagos-ai/Torch-FL"
    echo "  $0 qwen3 'MUSA MTT S5000' flagos-ai/Torch-FL"
    echo ""
    echo "Supported chips: MUSA, GCU, Ascend, MetaX, PPU, IPU, Gaudi, MLU"
    echo "The device is not a parameter: it is whatever tests/manual/hf_device_spec.py"
    echo "registers, and the runner derives it from that file."
    echo ""
    echo "Set PYTHON to the interpreter that has the accelerator build installed."
    exit 1
fi

WORK_DIR=${TMPDIR:-/tmp}/transformers-auto-sweep-${MODEL}-$(date +%Y%m%d-%H%M%S)
mkdir -p "${WORK_DIR}"

echo "======================================================================="
echo "Transformers Auto Sweep + Issue Preview"
echo "======================================================================="
echo "Model:    $MODEL"
echo "Chip:     $CHIP"
echo "Repo:     $REPO"
echo "Python:   $PYTHON"
echo "Work Dir: $WORK_DIR"
echo "======================================================================="
echo ""

# Step 0: the interpreter. A box whose ``python`` has no torch, transformers or
# torch_fl otherwise fails much later, as "the model name must be wrong".
#
# The probe runs from an empty directory, and separates the verdict from the
# interpreter's own chatter, because either mistake makes this check disagree
# with the run it is supposed to predict:
#
#   * ``python -c`` puts the working directory on ``sys.path``. Probing from the
#     repository root therefore imports this checkout's ``torch_fl`` and reports
#     a healthy interpreter for a box where no test child --- every one of them
#     runs from a private work directory --- can import it.
#   * The verdict is printed on stdout. Merging stderr into it turns any warning
#     the device build emits while importing into a "cannot import" verdict, so a
#     correctly configured interpreter is rejected as a broken one.
#
# PYTHONPATH is read the same way the children read it, so exporting the
# repository root is what makes a checkout usable, and leaving it unset is what
# selects an installed build.
echo "[0/6] Checking ${PYTHON} for torch, transformers and torch_fl..."
PROBE_DIR=$(mktemp -d "${TMPDIR:-/tmp}/transformers-probe-XXXXXX")
PROBE_ERR="${PROBE_DIR}/stderr.txt"
trap 'rm -rf "${PROBE_DIR}"' EXIT

MISSING=$(
    cd "${PROBE_DIR}" || exit 1
    ${PYTHON} -c '
import importlib

missing = []
for name in ("torch", "transformers", "torch_fl"):
    try:
        importlib.import_module(name)
    except Exception as exc:
        missing.append(f"{name} ({exc})")
print("; ".join(missing))
' 2>"${PROBE_ERR}"
)
PROBE_STATUS=$?

if [ "${PROBE_STATUS}" -ne 0 ]; then
    echo "environment error: ${PYTHON} could not run the import check (exit ${PROBE_STATUS}):" >&2
    sed 's/^/  /' "${PROBE_ERR}" >&2
    echo "  Set PYTHON to an interpreter that has the accelerator build installed." >&2
    exit 2
fi
if [ -n "${MISSING}" ]; then
    echo "environment error: ${PYTHON} cannot import the test environment:" >&2
    echo "  ${MISSING}" >&2
    echo "  The interpreter is probed from an empty directory, because that is where" >&2
    echo "  every test child runs. Install the build into ${PYTHON}, or export" >&2
    echo "  PYTHONPATH pointing at an already installed torch_fl." >&2
    exit 2
fi
echo "✓ ${PYTHON} has torch, transformers and torch_fl"

# Step 1: Run tests (resilient mode)
echo ""
echo "[1/6] Running tests (resilient mode)..."
echo "  Batch size: 20 tests"
echo "  Batch timeout: 15 minutes"
echo ""

${PYTHON} tests/manual/transformers_hf_tests.py \
    --model "${MODEL}" \
    --resilient \
    --batch-size 20 \
    --batch-timeout 900 \
    --out "${WORK_DIR}/test-results.json"
TEST_STATUS=$?

case ${TEST_STATUS} in
    0)
        echo ""
        echo "✓ Measured: every test passed"
        ;;
    1)
        echo ""
        echo "✓ Measured: failures were recorded (resilient mode continued past them)"
        ;;
    2)
        echo "" >&2
        echo "environment error: nothing was measured" >&2
        echo "  The runner refused to report a result. Its preflight report is in" >&2
        echo "  ${WORK_DIR}/test-results.json" >&2
        exit 2
        ;;
    *)
        echo "" >&2
        echo "environment error: the runner exited ${TEST_STATUS}" >&2
        exit 2
        ;;
esac

if [ ! -f "${WORK_DIR}/test-results.json" ]; then
    echo "" >&2
    echo "environment error: the runner wrote no result for ${MODEL}" >&2
    exit 2
fi

RUN_STATUS=$(${PYTHON} -c "import json; print(json.load(open('${WORK_DIR}/test-results.json')).get('run', {}).get('status', 'unknown'))" 2>/dev/null || echo "unknown")
TEST_COUNT=$(${PYTHON} -c "import json; print(len(json.load(open('${WORK_DIR}/test-results.json')).get('tests', [])))" 2>/dev/null || echo "0")
echo ""
echo "✓ Captured ${TEST_COUNT} test results (run status: ${RUN_STATUS})"

if [ "${TEST_COUNT}" -eq 0 ]; then
    echo "" >&2
    echo "environment error: the runner recorded no tests (run status: ${RUN_STATUS})" >&2
    echo "  ${MODEL} is a registry key, so an empty run is an environment problem," >&2
    echo "  not a wrong model name. Check the preflight report in test-results.json." >&2
    exit 2
fi

# Step 2: Triage
echo ""
echo "[2/6] Triaging failures..."
if ! ${PYTHON} scripts/transformers/transformers_triage.py \
    "${WORK_DIR}/test-results.json" \
    --out "${WORK_DIR}/classified.json"; then
    echo "error: triage failed; the measurement is not classified" >&2
    exit 2
fi

read -r CLASSIFIED_COUNT ACTIONABLE_COUNT ENV_ERROR_COUNT < <(${PYTHON} -c "
import json

data = json.load(open('${WORK_DIR}/classified.json'))
findings = data.get('findings', [])
actionable = [f for f in findings if f.get('actionable', True)]
print(len(findings), len(actionable), data.get('summary', {}).get('environment_error', 0))
" 2>/dev/null || echo "0 0 0")

echo "✓ Classified ${CLASSIFIED_COUNT} findings (${ACTIONABLE_COUNT} actionable)"

if [ "${CLASSIFIED_COUNT}" -eq 0 ]; then
    echo ""
    echo "No failures to triage. All tests passed!"
    exit 0
fi

if [ "${ACTIONABLE_COUNT}" -eq 0 ]; then
    echo ""
    if [ "${ENV_ERROR_COUNT}" -gt 0 ]; then
        echo "environment error: ${ENV_ERROR_COUNT} test(s) never ran" >&2
        echo "  Fix the environment and re-run; this sweep is not a coverage result." >&2
        exit 2
    fi
    echo "Every failure was a test error, not a platform defect. Nothing to file."
    echo "Results saved in: ${WORK_DIR}"
    exit 0
fi

# Step 3: Verify (serial isolation)
echo ""
echo "[3/6] Verifying failures in isolation (serial)..."
if ! ${PYTHON} scripts/transformers/transformers_verify.py \
    "${WORK_DIR}/classified.json" \
    --out "${WORK_DIR}/verified.json" \
    --workers 1 \
    --timeout 120; then
    echo "error: verification failed; findings are not confirmed" >&2
    exit 2
fi

VERIFIED_COUNT=$(${PYTHON} -c "import json; print(len(json.load(open('${WORK_DIR}/verified.json')).get('findings', [])))" 2>/dev/null || echo "0")
echo "✓ Verified ${VERIFIED_COUNT} findings"

# Step 4: Deduplicate
echo ""
echo "[4/6] Deduplicating against baseline and GitHub..."
if ! ${PYTHON} scripts/transformers/transformers_deduplicate.py \
    "${WORK_DIR}/verified.json" \
    --out "${WORK_DIR}/new.json" \
    --coverage-file docs/reference/hf-coverage.md \
    --hardware "${CHIP}" \
    --repo "${REPO}"; then
    echo "error: deduplication failed; no finding may be treated as new" >&2
    exit 2
fi

# The counts live in ``summary.dedup``, not in the findings array: only NEW
# findings reach ``findings``, so counting them there would always report zero
# unchecked and the warning below could never fire.
read -r NEW_COUNT KNOWN_COUNT UNCHECKED_COUNT < <(${PYTHON} -c "
import json

dedup = json.load(open('${WORK_DIR}/new.json')).get('summary', {}).get('dedup', {})
unchecked = dedup.get('DEDUP_UNAVAILABLE', 0) + dedup.get('NOT_CHECKED', 0)
known = sum(dedup.get(name, 0) for name in ('IN_BASELINE', 'DUPLICATE', 'REVIEW_CANDIDATE'))
print(dedup.get('NEW', 0), known, unchecked)
" 2>/dev/null || echo "0 0 0")

echo "✓ Deduplication: ${NEW_COUNT} new, ${KNOWN_COUNT} already tracked, ${UNCHECKED_COUNT} unchecked"

if [ "${UNCHECKED_COUNT}" -gt 0 ]; then
    echo "" >&2
    echo "warning: ${UNCHECKED_COUNT} finding(s) were never checked against GitHub" >&2
    echo "  (dedup_status DEDUP_UNAVAILABLE or NOT_CHECKED). They are not known to be" >&2
    echo "  new, so this run must not be presented as a filing opportunity." >&2
fi

if [ "${NEW_COUNT}" -eq 0 ]; then
    echo ""
    if [ "${UNCHECKED_COUNT}" -gt 0 ]; then
        echo "No finding could be confirmed new: ${UNCHECKED_COUNT} of them were never" >&2
        echo "  checked against the tracker. This is not a clean deduplication result." >&2
        echo "Results saved in: ${WORK_DIR}"
        exit 1
    fi
    echo "No new issues to file. All findings are known!"
    echo "Results saved in: ${WORK_DIR}"
    exit 0
fi

# Step 5: Preview
echo ""
echo "[5/6] Generating issue previews..."

${PYTHON} scripts/transformers/transformers_preview_issues.py \
    "${WORK_DIR}/new.json" \
    --chip "${CHIP}" \
    --issue-bodies-dir "${WORK_DIR}/issues" \
    --out "${WORK_DIR}/preview.md" \
    || {
        echo "error: generating the preview failed" >&2
        exit 2
    }

echo ""
echo "======================================================================="
echo "Issue Preview (${NEW_COUNT} issues)"
echo "======================================================================="
cat "${WORK_DIR}/preview.md"
echo ""
echo "======================================================================="

# Step 6: Stop for explicit authorization
echo ""
echo "[6/6] Issue previews are ready for human review."
echo "No GitHub writes were performed."
echo ""

if [ "${UNCHECKED_COUNT}" -gt 0 ]; then
    echo "STOP: ${UNCHECKED_COUNT} finding(s) could not be deduplicated against GitHub."
    echo "Resolve the search failure before approving anything above."
    echo ""
fi

echo "File only explicitly approved fingerprints, for example:"
echo "  ${PYTHON} scripts/transformers/transformers_file_issues.py ${WORK_DIR}/new.json \\"
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
echo "  - test-results.json   (raw test output, includes the preflight report)"
echo "  - classified.json     (triaged findings)"
echo "  - verified.json       (isolated verification)"
echo "  - new.json            (deduplicated new findings)"
echo "  - preview.md          (issue preview)"
echo "  - issues/*.md         (individual issue bodies)"
echo "  - issues/*.json       (title and label sidecars the filer reads)"
echo "======================================================================="
echo ""
echo "Exit code 1: measured, with ${NEW_COUNT} finding(s) awaiting review."
echo "  The batch wrapper counts this as findings, not as a clean run. Without"
echo "  this code the caller cannot tell a clean measurement from one that"
echo "  produced drafts, and the batch summary reports the run as 'nothing to"
echo "  file' while a preview sits in ${WORK_DIR}."
exit 1
