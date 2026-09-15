# Transformers Auto-Triage Workflow

This workflow turns an official HuggingFace architecture report into verified,
deduplicated issue drafts. The automated path is report-only. GitHub issue
creation is a separate, explicitly authorized action after human review.

## Overview

```text
test-results.json
    -> triage
classified.json
    -> serial fresh-process verification
verified.json
    -> baseline and GitHub deduplication
new.json
    -> issue draft generation
preview.md + issues/*.md
    -> human completion and explicit fingerprint approval
optional GitHub issue creation
```

The stages are separate so intermediate evidence remains inspectable and the
workflow can resume without rerunning the hardware suite.

## Quick Start

The normal entry point runs the report-only pipeline:

```bash
bash scripts/transformers/transformers_auto_sweep.sh qwen3 flagos "MUSA MTT S5000"
```

For weak-model guardrails or an explicitly requested safe run:

```bash
python scripts/transformers/safe_transformers_wrapper.py \
    test qwen3 "MUSA MTT S5000" --device flagos
```

Both commands stop after preview generation.

## Stage 1: Official Test Run

The automatic sweep invokes the version-matched official runner in resilient
mode:

```bash
python tests/manual/transformers_hf_tests.py \
    --model qwen3 \
    --device flagos \
    --resilient \
    --out /tmp/qwen3-results.json
```

The runner:

- loads the device through `TRANSFORMERS_TEST_DEVICE_SPEC`;
- disables optional backend autoloading that can break collection;
- enables `FLAGOS_LOG_FALLBACK=1`;
- executes selected nodeids in isolated batches;
- preserves completed records when a batch crashes or times out;
- atomically rewrites the aggregate report after each batch.

A `BATCH_CRASHED` record means that nodeid did not report before its batch
stopped. It is not a confirmed per-test defect.

## Stage 2: Triage

```bash
python scripts/transformers/transformers_triage.py \
    /tmp/qwen3-results.json \
    --out /tmp/qwen3-classified.json
```

Triage recognizes these actionable classes:

| Class | Meaning |
| --- | --- |
| `OP_UNSUPPORTED` | An operator has no usable device implementation |
| `OP_CPU_FALLBACK` | Runtime logs show that an operator executed on the host |
| `FEATURE_UNSUPPORTED` | A non-operator API or device feature is unavailable |
| `PRECISION` | Device output differs from the CPU baseline |
| `CRASH` | Per-test evidence shows a fatal device/process failure or timeout |

`PRECISION_KNOWN_ISSUE` and `UNKNOWN` remain visible for review but should not be
published without further investigation.

### CPU fallback

A passing model assertion can still produce `OP_CPU_FALLBACK`. The operator is a
coverage gap because the measured computation did not stay on the accelerator.
The fallback log is direct evidence, so these findings are marked confirmed and
do not need another reproduction run.

### Crash attribution

A run-level poisoned-context flag is never applied to every failed test. Triage
requires per-test crash evidence such as an illegal memory access, device-side
assertion, fatal signal, or timeout. Later failures remain unclassified or are
classified by their own error text until they reproduce independently.

### Cause fingerprints

Findings are grouped with a hash of:

```text
failure class | responsible component | subject | normalized mechanism
```

Model names and nodeids are occurrences, not cause identity. Including the
component prevents unrelated platform backends from sharing a fingerprint.

## Stage 3: Serial Verification

```bash
TRANSFORMERS_VERSION=$(python -c 'import transformers; print(transformers.__version__)')

python scripts/transformers/transformers_verify.py \
    /tmp/qwen3-classified.json \
    --out /tmp/qwen3-verified.json \
    --test-source-dir /root/.cache/torch_fl/hf-tests \
    --transformers-version "${TRANSFORMERS_VERSION}" \
    --workers 1 \
    --timeout 120
```

The verifier recreates the official runner's pytest environment and runs exactly
one selected nodeid in each fresh subprocess. Passing both the architecture
directory and a nodeid is forbidden because pytest treats the selectors as a
union and runs the whole directory.

An isolation result is valid only when pytest collected exactly one test.

### Verdict mapping

| Isolated result | Verdict | Filing effect |
| --- | --- | --- |
| `FAIL` | `CONFIRMED` | May continue through the evidence gates |
| `TIMEOUT` | `CONFIRMED` | May continue, with timeout evidence |
| `PASS` | `COLLATERAL` | Blocked |
| `SKIP` | `COLLATERAL` | Blocked |
| runner/setup `ERROR` | `INCONCLUSIVE` | Blocked |

Verification is serial. Multiple subprocesses can still contend for one device
context and memory pool, so `--workers` values other than one are rejected until
per-worker device isolation exists.

## Stage 4: Deduplication

```bash
python scripts/transformers/transformers_deduplicate.py \
    /tmp/qwen3-verified.json \
    --out /tmp/qwen3-new.json \
    --coverage-file docs/reference/hf-coverage.md \
    --repo flagos-ai/Torch-FL
```

Deduplication checks:

1. exact fingerprints in the coverage record;
2. exact fingerprints in issue bodies;
3. exact fingerprints in issue comments;
4. semantic subject matches for older issues without fingerprints.

An exact match is a duplicate. A semantic match is marked
`REVIEW_CANDIDATE` and requires a human to compare the component and mechanism.
It is not silently treated as the same root cause.

Collateral and inconclusive findings are omitted from the output intended for
issue preview.

Use `--skip-github` only for local tests of the tooling. A real publication
workflow must search the tracker before filing.

## Stage 5: Preview Generation

```bash
python scripts/transformers/transformers_preview_issues.py \
    /tmp/qwen3-new.json \
    --chip "MUSA MTT S5000" \
    --transformers-version "${TRANSFORMERS_VERSION}" \
    --torch-fl-commit "$(git rev-parse --short HEAD)" \
    --issue-bodies-dir /tmp/qwen3-issues \
    --out /tmp/qwen3-preview.md
```

The tool writes one Markdown body per fingerprint and a consolidated preview.
Drafts follow the repository's AI issue template structure, but they are
intentionally incomplete. Before publication, a human or capable agent must add
and validate:

- the actual AI model and full environment;
- a minimal self-contained reproducer, or a defensible explanation of why the
  exact isolated upstream test is the smallest available reproduction;
- root-cause analysis rather than an error restatement;
- a specific proposed solution;
- responsible code locations with line numbers;
- a completed issue checklist.

The preview's proposed operator solution directs non-CUDA-compatible platforms
to their code generator rather than handwritten per-operator kernels.

## Stage 6: Optional Issue Filing

Issue creation is not part of the automatic or safe sweep. It is allowed only
after the user has reviewed a named set of findings and explicitly authorized
those fingerprints.

```bash
python scripts/transformers/transformers_file_issues.py \
    /tmp/qwen3-new.json \
    --issue-bodies-dir /tmp/qwen3-issues \
    --approve <fingerprint> [<fingerprint> ...] \
    --repo flagos-ai/Torch-FL
```

The filing tool rejects:

- fingerprints not present in the input;
- findings whose verdict is not `CONFIRMED`;
- drafts that still contain required review placeholders;
- missing body files.

There is no bulk `--approve-all` path. Approval for one finding does not cover
later findings or another tracker action.

When issues are created successfully, the tool appends their fingerprints and
issue numbers to `docs/reference/hf-coverage.md`. Commit that documentation
change through the normal fork-and-PR workflow; the script must not push a
branch itself.

## Failure Classes and Labels

| Class | Suggested labels |
| --- | --- |
| `OP_UNSUPPORTED` | `enhancement`, `ai-generated` |
| `OP_CPU_FALLBACK` | `enhancement`, `ai-generated` |
| `FEATURE_UNSUPPORTED` | `enhancement`, `ai-generated` |
| `PRECISION` | `bug`, `ai-generated` |
| `CRASH` | `bug`, `P0`, `ai-generated` |

Confirm the repository's current labels before publication.

## Baseline Semantics

A baseline is scoped to the measured hardware, device, Transformers version, and
torch_fl commit. It supports comparisons and regression claims; it is not a
permission gate that suppresses every first-sweep defect.

A first sweep may produce an issue when an individual finding has complete
evidence, independent reproduction where required, a named cause, deduplication,
a finished issue body, and explicit authorization. Describe it as observed on
the pinned tuple, not as a regression without an earlier matching measurement.

## Safe Wrapper Boundaries

The safe wrapper validates model, device, chip, and repository parameters and
invokes only the checked report-only scripts. It does not install dependencies,
edit source files, change the parent shell environment, or publish issues.

If preflight dependencies are missing, stop and report the environment problem
rather than modifying the torch installation during the measurement.

## Troubleshooting

### All findings are `UNKNOWN`

Inspect `representative_detail` in the classified JSON. Confirm that the official
runner captured the exception tail and that the record is a test failure rather
than a setup or collection error. Add a classifier pattern only after the
mechanism is understood.

### A verifier result is `ERROR`

Treat it as inconclusive. Check the exact source version, pytest root, device
specification, imports, and selected nodeid. Do not convert it to confirmed based
on the original suite failure.

### A nodeid rerun collects many tests

Remove the architecture directory from the pytest command. Pass the nodeid only
and confirm the output says one test was collected.

### A semantic duplicate candidate appears

Read both issue bodies and compare the responsible component, subject, dtype,
shape, and normalized mechanism. Mark it as an exact duplicate only after that
review.

### The issue body uses `[AI][Unknown]`

Regenerate the draft with `--chip`, or ensure the body contains either the
current `- **Platform**: ...` field or the legacy `- **Chip**: ...` field. The
filer supports both forms.

## Verification of the Tooling

```bash
ruff check
ruff format --check
pytest tests/unit/test_transformers_hf_tests.py \
       tests/unit/test_transformers_automation.py -q
python scripts/transformers/test_transformers_automation.py
bash -n scripts/transformers/transformers_auto_sweep.sh \
        scripts/transformers/transformers_batch_sweep.sh
```

The smoke test covers triage, deduplication, and preview generation. It
intentionally excludes GitHub filing because generated drafts require human
completion and explicit authorization.

## Related Documentation

- `.claude/skills/transformers-test/SKILL.md`
- `docs/workflows/resilient-testing-quickstart.md`
- `docs/design/robust-harness-proposal.md`
- `docs/reference/hf-coverage.md`
- `.github/ISSUE_TEMPLATE/ai_agent_issue.md`
