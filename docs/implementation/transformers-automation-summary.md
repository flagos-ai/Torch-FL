# Transformers Automation Implementation Summary

## Scope

This change adds a report-only pipeline around the official HuggingFace
architecture runner:

```text
official tests
    -> triage
    -> serial verification
    -> deduplication
    -> issue previews
    -> optional, explicitly authorized filing
```

It also adds resilient batch execution for platforms where a model test can
crash, hang, or poison the device context.

## Problems Addressed

### Suite-wide failure after one device fault

The original runner used one pytest process for an architecture. A fatal device
error could terminate that process, discard later coverage, and contaminate
subsequent observations.

### Invalid isolation selectors

A batch command passed both the architecture directory and selected nodeids.
Pytest treats these as a union, so a purported one-test rerun could execute the
entire directory. That made early isolation evidence invalid.

### Lost records from crashed batches

A crashed batch was previously replaced wholesale with `BATCH_CRASHED`
placeholders, even when the report plugin had already written valid results for
some tests.

### Hidden host execution

A test could pass while one or more operators used torch_fl's CPU fallback. The
result looked like accelerator coverage even though part of the workload ran on
the host.

### Unsafe verification and publication

The initial automation used concurrent accelerator reruns, treated runner errors
as reproduced failures, and offered a bulk issue-filing path. Generated issue
bodies could therefore publish incomplete or incorrectly attributed findings.

## Implemented Architecture

### Official runner

`tests/manual/transformers_hf_tests.py` now supports resilient batches while
preserving the same version-matched source and device contract as a normal run.
Important properties are:

- selected-nodeid commands omit the architecture directory;
- each batch has an independent process and timeout;
- per-test JSONL records are reduced after every batch;
- completed records survive a later batch crash;
- only unreported nodeids become `BATCH_CRASHED`;
- aggregate output is written incrementally;
- long tracebacks preserve both their beginning and exception tail;
- `FLAGOS_LOG_FALLBACK=1` is enabled and fallback operators are recorded.

### Triage

`scripts/transformers/transformers_triage.py` converts runner output into cause-oriented
findings. It supports:

- `OP_UNSUPPORTED`;
- `OP_CPU_FALLBACK`;
- `FEATURE_UNSUPPORTED`;
- `PRECISION`;
- `CRASH`;
- known precision patterns and unknown failures retained for review.

A run-level poison marker no longer classifies every failed test as a crash.
Crash attribution requires per-test evidence. CPU fallback findings can come
from otherwise passing tests and are confirmed directly by the runtime log.

Fingerprints include the failure class, responsible component, subject, and
normalized mechanism. Model names and nodeids are aggregated occurrences.

### Verification

`scripts/transformers/transformers_verify.py` reruns one representative nodeid per finding in
a fresh pytest subprocess. It reconstructs the official environment with:

- the exact Transformers source tree;
- the source `pyproject.toml` and root directory;
- `hf_device_spec.py`;
- source package symlinks and `PYTHONPATH` entries;
- backend autoload disabled;
- fallback logging enabled.

Verification defaults to one worker and rejects parallel execution because
separate processes may still share an accelerator and memory pool.

Verdicts are:

- `FAIL` or `TIMEOUT` -> `CONFIRMED`;
- `PASS` or `SKIP` -> `COLLATERAL`;
- setup, import, collection, or runner `ERROR` -> `INCONCLUSIVE`.

### Deduplication

`scripts/transformers/transformers_deduplicate.py` checks exact fingerprints in the coverage
record, issue bodies, and issue comments. It then searches by subject for older
issues that predate fingerprints.

Semantic matches are emitted as `REVIEW_CANDIDATE` and blocked from filing until
a human compares the component and mechanism. Collateral and inconclusive
findings are also blocked.

### Preview generation

`scripts/transformers/transformers_preview_issues.py` writes one Markdown draft per new
finding and a consolidated preview. Drafts follow the repository AI issue
template structure and include the captured evidence, fingerprint, isolated
command, proposed verification, and suggested labels.

They intentionally leave required review placeholders for:

- the actual AI model;
- complete hardware and software environment;
- a validated minimal reproducer;
- root-cause analysis;
- responsible code locations;
- human completion of the checklist.

For non-CUDA-compatible operator work, proposed solutions direct contributors to
the platform code generator rather than handwritten per-operator kernels.

### Filing

`scripts/transformers/transformers_file_issues.py` is a separate optional tool. It requires an
explicit list of fingerprints, rejects non-confirmed findings, and rejects
incomplete drafts. There is no bulk approval option.

The filer reads the current `Platform` field emitted by the preview tool and
retains compatibility with legacy drafts that used `Chip`.

### Safe and automatic wrappers

`scripts/transformers/transformers_auto_sweep.sh` executes the full measurement and preview
pipeline, then prints the command shape for a later explicitly authorized filing
action. It does not invoke the filer.

`scripts/transformers/safe_transformers_wrapper.py` validates parameters before invoking that
same report-only path. Safe mode cannot publish issues automatically.

## Files

### Core tooling

- `scripts/transformers/transformers_triage.py`
- `scripts/transformers/transformers_verify.py`
- `scripts/transformers/transformers_deduplicate.py`
- `scripts/transformers/transformers_preview_issues.py`
- `scripts/transformers/transformers_file_issues.py`
- `scripts/transformers/transformers_auto_sweep.sh`
- `scripts/transformers/transformers_batch_sweep.sh`
- `scripts/transformers/safe_transformers_wrapper.py`

### Runner and tests

- `tests/manual/transformers_hf_tests.py`
- `tests/unit/test_transformers_hf_tests.py`
- `tests/unit/test_transformers_automation.py`
- `scripts/transformers/test_transformers_automation.py`

### Documentation

- `.claude/skills/transformers-test/SKILL.md`
- `docs/design/robust-harness-proposal.md`
- `docs/workflows/resilient-testing-quickstart.md`
- `docs/workflows/transformers-auto-triage.md`
- `docs/reference/hf-coverage.md`

## Important Invariants

1. A selected-nodeid subprocess must not also receive the architecture directory.
2. A valid isolation rerun collects exactly one test.
3. Completed records from a crashed batch are never overwritten.
4. A run-level poison marker does not identify the triggering test.
5. A passing assertion with CPU fallback is not accelerator success.
6. Accelerator verification is serial until devices can be isolated per worker.
7. A runner error is inconclusive, not confirmed.
8. Semantic issue matches require review rather than automatic deduplication.
9. Automatic and safe modes stop before GitHub writes.
10. Publication requires complete issue content and explicit fingerprint-level
    authorization.

## Usage

Run one architecture through the report-only path:

```bash
bash scripts/transformers/transformers_auto_sweep.sh qwen3 flagos "MUSA MTT S5000"
```

Or run the stages individually:

```bash
python scripts/transformers/transformers_triage.py results.json --out classified.json

python scripts/transformers/transformers_verify.py \
    classified.json \
    --out verified.json \
    --test-source-dir /root/.cache/torch_fl/hf-tests \
    --transformers-version 5.16.1 \
    --workers 1

python scripts/transformers/transformers_deduplicate.py \
    verified.json \
    --out new.json \
    --coverage-file docs/reference/hf-coverage.md \
    --repo flagos-ai/Torch-FL

python scripts/transformers/transformers_preview_issues.py \
    new.json \
    --chip "MUSA MTT S5000" \
    --transformers-version 5.16.1 \
    --torch-fl-commit "$(git rev-parse --short HEAD)" \
    --issue-bodies-dir /tmp/qwen3-issues \
    --out /tmp/qwen3-preview.md
```

After reviewing and completing specific drafts, an explicitly authorized set can
be filed with:

```bash
python scripts/transformers/transformers_file_issues.py \
    new.json \
    --issue-bodies-dir /tmp/qwen3-issues \
    --approve <fingerprint> [<fingerprint> ...] \
    --repo flagos-ai/Torch-FL
```

## Verification

The focused automated checks are:

```bash
ruff check
ruff format --check
pytest tests/unit/test_transformers_hf_tests.py \
       tests/unit/test_transformers_automation.py -q
python scripts/transformers/test_transformers_automation.py
bash -n scripts/transformers/transformers_auto_sweep.sh \
        scripts/transformers/transformers_batch_sweep.sh
```

The smoke test covers triage, deduplication, and preview generation. Filing is
excluded because generated drafts are incomplete until human review and explicit
authorization.

## Hardware Evidence

The qwen3 MUSA investigation that motivated the hardening demonstrated why the
invariants matter:

- the original selector shape reran the architecture directory rather than one
  nodeid;
- corrected one-nodeid subprocesses separated five cause groups from 20 failure
  occurrences;
- failures included a model-parallel illegal memory access, non-contiguous
  softmax restrictions, ProcessGroupGloo's device gap, a CUDA-only
  TorchInductor/Triton dependency, and mudnn INT64 true division;
- a complete CPU-fallback inventory still requires a hardware rerun with the new
  `FLAGOS_LOG_FALLBACK=1` instrumentation.

The corrected evidence and associated issue references are recorded in
`docs/reference/hf-coverage.md`.

## Trade-offs and Follow-up

- Serial verification takes longer than concurrent reruns but produces stronger
  evidence on a shared accelerator.
- Resilient batches recover coverage after a process failure but cannot guarantee
  that a vendor driver has reset; fresh subprocess isolation remains necessary.
- Generated drafts reduce repetitive formatting but do not replace root-cause
  investigation.
- The next target-hardware run should explicitly audit the reported
  `cpu_fallback_ops` and create findings for every measured host fallback.
