# Resilient Transformers Testing Quick Start

Resilient mode runs an official HuggingFace architecture suite in isolated
batches. If one batch crashes or times out, the runner preserves completed
records and continues with the remaining batches.

The automation is report-only: it generates evidence and issue drafts but never
publishes GitHub issues by itself.

## Quick Start

### One model

```bash
bash scripts/transformers/transformers_auto_sweep.sh bert gcu GCU
bash scripts/transformers/transformers_auto_sweep.sh qwen3 flagos "MUSA MTT S5000"
```

Arguments:

1. Model architecture name, such as `bert` or `qwen3`.
2. Torch device name. Use the device accepted by the runner; for the standard
   torch_fl PrivateUse1 flow this is normally `flagos`.
3. Hardware name used in the report and issue preview.
4. Optional repository for duplicate search, defaulting to
   `flagos-ai/Torch-FL`.

### Batch wrapper

```bash
bash scripts/transformers/transformers_batch_sweep.sh gcu GCU
```

The batch wrapper runs its configured model list one architecture at a time. It
still stops before issue publication.

## Manual Workflow

Use the individual tools when investigating a failure or reviewing intermediate
outputs:

```bash
MODEL=qwen3
DEVICE=flagos
CHIP="MUSA MTT S5000"
RESULT_ROOT=/tmp/transformers-${MODEL}
TRANSFORMERS_VERSION=$(python -c 'import transformers; print(transformers.__version__)')

# 1. Run official tests in resilient batches.
python tests/manual/transformers_hf_tests.py \
    --model "${MODEL}" \
    --device "${DEVICE}" \
    --resilient \
    --batch-size 20 \
    --batch-timeout 900 \
    --out "${RESULT_ROOT}-results.json"

# 2. Classify failures and measured CPU fallbacks.
python scripts/transformers/transformers_triage.py \
    "${RESULT_ROOT}-results.json" \
    --out "${RESULT_ROOT}-classified.json"

# 3. Verify candidate failures serially in fresh subprocesses.
python scripts/transformers/transformers_verify.py \
    "${RESULT_ROOT}-classified.json" \
    --out "${RESULT_ROOT}-verified.json" \
    --test-source-dir /root/.cache/torch_fl/hf-tests \
    --transformers-version "${TRANSFORMERS_VERSION}" \
    --workers 1

# 4. Check exact fingerprints and semantic duplicate candidates.
python scripts/transformers/transformers_deduplicate.py \
    "${RESULT_ROOT}-verified.json" \
    --out "${RESULT_ROOT}-new.json" \
    --coverage-file docs/reference/hf-coverage.md \
    --repo flagos-ai/Torch-FL

# 5. Generate incomplete drafts for human review.
python scripts/transformers/transformers_preview_issues.py \
    "${RESULT_ROOT}-new.json" \
    --chip "${CHIP}" \
    --transformers-version "${TRANSFORMERS_VERSION}" \
    --torch-fl-commit "$(git rev-parse --short HEAD)" \
    --issue-bodies-dir "${RESULT_ROOT}-issues" \
    --out "${RESULT_ROOT}-preview.md"
```

Review the preview and each body file. Complete the environment, reproducer,
root-cause analysis, solution, code locations, and checklist before requesting
publication.

After the user explicitly approves named fingerprints, file only that approved
set:

```bash
python scripts/transformers/transformers_file_issues.py \
    "${RESULT_ROOT}-new.json" \
    --issue-bodies-dir "${RESULT_ROOT}-issues" \
    --approve <fingerprint> [<fingerprint> ...] \
    --repo flagos-ai/Torch-FL
```

There is no `--approve-all` mode.

## Resilient Mode Options

### `--resilient`

Collect the architecture suite, split it into batches, and run each batch in a
fresh pytest subprocess.

### `--batch-size N`

Number of nodeids per batch. The default is 20.

- Use 10 when failures frequently crash or poison the device.
- Use 20 for a new or moderately stable backend.
- Use 50 only after the suite is stable enough that a crash is unlikely.

Smaller batches reduce collateral uncertainty but increase pytest startup cost.

### `--batch-timeout N`

Timeout in seconds for each batch. The default is 900.

- Small architectures: 600 seconds may be sufficient.
- Typical architectures: 900 seconds.
- Large or compilation-heavy architectures: 1800 seconds.

A timeout preserves any records already emitted by the batch and marks only the
remaining unreported nodeids as `BATCH_CRASHED`.

## Interpreting Results

### Completed records in a crashed batch

A crashed batch may contain genuine `PASS`, `FAIL`, or `SKIP` records written
before the process stopped. Those records are preserved. Do not replace the
whole batch with crash placeholders.

### `BATCH_CRASHED`

This status means the subprocess stopped before that nodeid produced a report.
It is not a confirmed per-test defect. Isolate the batch and then each candidate
nodeid in a fresh process.

### Context poisoning

A run-level poisoning warning invalidates later observations, but it does not
identify the triggering test. Only per-test evidence such as an illegal memory
access, device-side assertion, fatal signal, or independent isolated failure can
support a crash finding.

### CPU fallback

The runner enables `FLAGOS_LOG_FALLBACK=1`. Any operator listed under
`cpu_fallback_ops` is an accelerator coverage gap even if the model assertion
passed. Triage emits a confirmed `OP_CPU_FALLBACK` finding for each measured
operator.

### Verification verdicts

- `FAIL` or `TIMEOUT` in the isolated rerun: `CONFIRMED`.
- `PASS` or `SKIP`: `COLLATERAL`.
- pytest setup, import, collection, or runner error: `INCONCLUSIVE`.

Only confirmed findings can pass the filing gate.

## Exact Test Isolation

Do not combine an architecture directory with a nodeid:

```bash
# Wrong: pytest treats these as a union and runs the directory too.
pytest tests/models/qwen3 \
    tests/models/qwen3/test_modeling_qwen3.py::Qwen3ModelTest::test_example

# Correct: pass only the nodeid.
pytest \
    tests/models/qwen3/test_modeling_qwen3.py::Qwen3ModelTest::test_example
```

An isolation result is valid only when pytest collected exactly one test.

## Output Files

The automatic sweep writes a timestamped directory under
`/tmp/transformers-auto-sweep-<model>-*` containing:

```text
test-results.json  raw official-runner results
classified.json    cause-oriented findings
verified.json      isolation outcomes and verdicts
new.json           findings remaining after deduplication
preview.md          consolidated human review preview
issues/*.md         individual incomplete issue drafts
```

Keep these files together when investigating or citing a run.

## Troubleshooting

### No test results

Check that the architecture exists in the installed Transformers version and
that its exact source tree is available:

```bash
python tests/manual/transformers_hf_tests.py --list-models
python tests/manual/transformers_hf_tests.py --model bert --collect-only
```

A source-version mismatch or collection failure is an environment result, not a
backend finding.

### Every batch crashes

Validate the device before interpreting model results:

```bash
python - <<'PY'
import torch
import torch_fl

print("device count:", torch.flagos.device_count())
x = torch.randn(2, 3, device="flagos")
print("device:", x.device)
PY
```

Also record the driver, vendor SDK, torch, Transformers, and torch_fl versions.
Do not install or replace packages as an ad hoc fix during a coverage run.

### One batch is slow

Reduce `--batch-size` to isolate the slow or hanging nodeid. Increase
`--batch-timeout` only when the individual tests are expected to take longer;
do not use a longer timeout to hide a hang.

### Verification is rejected with multiple workers

Use `--workers 1`. Separate subprocesses are not independent when they share the
same accelerator and memory pool.

### An issue is not filed

The filer deliberately rejects:

- unknown or unapproved fingerprints;
- non-confirmed findings;
- missing body files;
- drafts containing mandatory review placeholders.

Complete the draft and obtain explicit fingerprint-level authorization before
retrying. A dry run still requires an approved fingerprint:

```bash
python scripts/transformers/transformers_file_issues.py \
    /tmp/qwen3-new.json \
    --issue-bodies-dir /tmp/qwen3-issues \
    --approve <fingerprint> \
    --dry-run
```

## Recommended Practice

1. Start a new platform with batches of 10 to 20 tests.
2. Preserve the original JSON and all isolated rerun output.
3. Verify candidates serially and require exactly one collected nodeid.
4. Treat fallback operators as missing accelerator coverage.
5. Review semantic duplicate candidates manually.
6. Generate previews first; never publish from the automated or safe wrapper.
7. File only completed, confirmed, explicitly approved findings.
