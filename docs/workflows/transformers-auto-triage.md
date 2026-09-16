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
bash scripts/transformers/transformers_auto_sweep.sh qwen3 "MUSA MTT S5000"
```

For weak-model guardrails or an explicitly requested safe run:

```bash
python scripts/transformers/safe_transformers_wrapper.py \
    test qwen3 "MUSA MTT S5000"
```

Both commands stop after preview generation.

The sweep's exit code states what it measured: `0` measured and clean, `1`
measured with findings awaiting review, `2` nothing measured. Only `2` means the
run is not a coverage result. The code is the sweep's only channel to its caller
once the preview is written: every stage having succeeded is not the same
outcome as having nothing to report, so a sweep that produced drafts exits `1`
and the batch driver counts it under `Findings` rather than `Nothing to file`.

## Stage 1: Official Test Run

The automatic sweep invokes the version-matched official runner in resilient
mode:

```bash
python tests/manual/transformers_hf_tests.py \
    --model qwen3 \
    --resilient \
    --out /tmp/qwen3-results.json
```

There is no `--device` argument. The test device is the `DEVICE_NAME` in
`tests/manual/hf_device_spec.py`, which is the contract HuggingFace reads, and
the runner derives it from that file and records it as `environment.device`.

The runner:

- loads the device through `TRANSFORMERS_TEST_DEVICE_SPEC`;
- runs a preflight in a child process before the first batch and records it
  under `environment.preflight`;
- disables optional backend autoloading that can break collection;
- enables `FLAGOS_LOG_FALLBACK=1`;
- executes selected nodeids in isolated batches;
- preserves completed records when a batch crashes or times out;
- atomically rewrites the aggregate report after each batch.

The preflight checks that `torch_fl` imports, that the registered PrivateUse1
name equals the spec's `DEVICE_NAME`, that `torch.flagos.device_count()` is
positive, that the three HF hooks are callable, and that the installed
`transformers` version is the requested one. A run that fails it measured
nothing and exits `2`. Every check runs under one guard, so the child always
writes its report: a check that raises is recorded as `the environment checks
could not run` alongside whatever else it learned, rather than ending the
process with nothing to read.

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

Two further classes are reported but are not actionable:

| Class | Meaning |
| --- | --- |
| `TEST_ERROR` | The test's own setup, teardown, or collection failed |
| `ENVIRONMENT_ERROR` | A missing module, import failure, or absent accelerator |

They appear in the report and in `--out` with `actionable = false` and
`verification_required = false`, and they never reach the tracker. A run whose
every outcome is `ENVIRONMENT_ERROR` reports that count in its summary instead of
"all tests passed"; the runner exits `2` for it, because nothing was measured.

`PRECISION_KNOWN_ISSUE` and `UNKNOWN` remain visible for review but should not be
published without further investigation.

Triage carries the measured environment through to its output as `environment`,
taken from the run's own record (`--all` mode keeps it per model block, and the
first block that has one answers for the run). Verification reads the
`transformers` version from there, so a cache holding several source trees cannot
verify a finding against one that did not produce it.

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

There is one implementation of this, `generate_fingerprint` in
`scripts/transformers/transformers_triage.py`, together with the
`normalize_error` it hashes. It is idempotent: re-normalizing an already
normalized mechanism yields the same string, and the mechanism comes from the
exception's stable final line rather than a fixed window of the traceback, so
nothing inserted above the exception changes the hash. `compute_fingerprint`
remains as an alias for older callers.

The per-test value the runner writes is `occurrence_fingerprint`. It includes the
nodeid and identifies one test result, which is why it is not used for dedup.

## Stage 3: Serial Verification

```bash
TRANSFORMERS_VERSION=$(python -c 'import transformers; print(transformers.__version__)')

python scripts/transformers/transformers_verify.py \
    /tmp/qwen3-classified.json \
    --out /tmp/qwen3-verified.json \
    --transformers-version "${TRANSFORMERS_VERSION}" \
    --workers 1 \
    --timeout 120
```

`--test-source-dir` defaults to the runner's own cache root: `HF_COVERAGE_CACHE`
if set, else `~/.cache/torch_fl/hf-tests`. Pass it explicitly only to verify
against a tree the runner did not produce. The versions verified against come
from the findings JSON, not from the reviewing interpreter: triage records the
environment the measurement ran in, and the verifier reads `transformers` from
it. A findings file with no environment falls back to the newest cached source
tree and says so.

The verifier recreates the official runner's pytest environment and runs exactly
one selected nodeid in each fresh subprocess. Passing both the architecture
directory and a nodeid is forbidden because pytest treats the selectors as a
union and runs the whole directory.

That environment is the runner's `child_env`, entry for entry: the source tree
leads so that `tests.models...` resolves into the measured tree, the caller's
`PYTHONPATH` follows, and a repository root the caller supplied is moved to the
end rather than dropped. `hf_device_spec.py` imports `torch_fl`, so a checkout
that was never installed is importable only through that entry. Dropping it turns
every isolation into `ModuleNotFoundError: No module named 'torch_fl'` before a
single test is collected, and the verifier can only record that as `ERROR`, which
reads like an unhealthy device and is really an unhealthy harness.

An isolation result is valid only when pytest collected exactly one test, and the
verifier enforces it rather than trusting it: the outcome is read from pytest's
own summary line, and anything other than one test is recorded as `ERROR`. A
nodeid pytest cannot select exits with a usage error, which is a defect in the
harness and not per-test evidence. When every isolation of a run selected zero
tests, the verifier exits `2` instead of reporting the run as clean — otherwise
"no new findings" would stand for "nothing was checked".

The nodeids the runner records are canonicalized before verification. pytest
reports a nodeid it was given without the file part, so a batch selector of
`tests/models/bert/test_modeling_bert.py::BertModelTest::test_x` comes back as
`::BertModelTest::test_x`, which is not selectable. Each recorded nodeid is
restored from the nodeid that was selected, matched on its `::Class::test` tail;
a tail shared by two selected nodeids is left as reported, because guessing which
file was meant would attribute a result to the wrong one.

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
    --hardware "MUSA MTT S5000" \
    --repo flagos-ai/Torch-FL
```

`--hardware` is the board this run measured, and it is what scopes the baseline
read. It is not optional in practice: MetaX and MUSA both register their
PrivateUse1 device as `flagos`, so a finding's component cannot tell the two
apart, and without the label every `## Baseline:` section in the coverage record
is merged as if this run had measured on each of them. A finding measured on one
board would then be suppressed as already known by a section measured on another.
Sections measured elsewhere are skipped and reported; a label that matches no
section is reported too, because no finding can then be matched against an
earlier measurement. `transformers_auto_sweep.sh` passes its `--chip` argument
through as `--hardware`.

Deduplication checks:

1. exact fingerprints in the coverage record;
2. exact fingerprints in issue bodies;
3. exact fingerprints in issue comments;
4. semantic subject matches for older issues without fingerprints.

An exact match is a duplicate. A semantic match is marked
`REVIEW_CANDIDATE` and requires a human to compare the component and mechanism.
It is not silently treated as the same root cause.

The check fails closed. A `gh` call that is missing, times out, or exits
non-zero marks the finding `DEDUP_UNAVAILABLE` with `should_file = false`, and
`--skip-github` marks it `NOT_CHECKED`, also with `should_file = false`. Only a
search that ran and returned nothing yields `NEW`. An unreachable tracker is not
evidence that a defect is new.

Those counts are read from `summary.dedup`: only `NEW` findings are carried in
the findings array, so a caller that inspected the array alone could not tell
"nothing new" from "nothing was checked". `transformers_auto_sweep.sh` counts
them that way and exits `1` rather than reporting a clean dedup when findings
remain unchecked.

The coverage record is the cause table in `docs/reference/hf-coverage.md`, keyed
by its leading `Fingerprint` column; the standalone ``Fingerprint: `hash` ``
line is accepted too, because that is how issue bodies carry it. Rows that
predate the convention are left blank rather than back-filled with invented
hashes.

Collateral and inconclusive findings are omitted from the output intended for
issue preview, as are `TEST_ERROR` and `ENVIRONMENT_ERROR` findings and any
baseline hit.

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

The tool writes one Markdown body per fingerprint and a consolidated preview,
plus an `issue-<fingerprint>.json` sidecar per draft holding the title, labels,
class, subject, and the fingerprint the file is named for. The filer submits the
sidecar, so the preview and the filed issue cannot disagree.

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

Unfilled values are written as `<!-- UNFILLED: <field> -->` markers, because a
free-text placeholder cannot be checked mechanically. The filer refuses any draft
that still contains one.

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
- findings classified `TEST_ERROR` or `ENVIRONMENT_ERROR`;
- drafts that still contain required review placeholders;
- sidecars whose recorded fingerprint is not the one they are named for;
- missing body files or sidecars.

It pre-collects every problem across all drafts and reports them together,
before any GitHub write, so a run is never left half-filed.

There is no bulk `--approve-all` path. Approval for one finding does not cover
later findings or another tracker action.

When issues are created successfully, the tool appends their fingerprints and
issue numbers to the cause table of the last `## Baseline:` section in
`docs/reference/hf-coverage.md`. Commit that documentation change through the
normal fork-and-PR workflow; the script must not push a branch itself.

## Failure Classes and Labels

| Class | Suggested labels |
| --- | --- |
| `OP_UNSUPPORTED` | `enhancement`, `ai-generated` |
| `OP_CPU_FALLBACK` | `enhancement`, `ai-generated` |
| `FEATURE_UNSUPPORTED` | `enhancement`, `ai-generated` |
| `PRECISION` | `bug`, `ai-generated` |
| `CRASH` | `bug`, `ai-generated` |
| `TEST_ERROR` | `ai-generated`, not filed |
| `ENVIRONMENT_ERROR` | `ai-generated`, not filed |

Confirm the repository's current labels before publication. No priority label is
assigned automatically: the classifier cannot tell a P0 from a P2, and a wrong
priority on a filed issue is worse than none.

## Baseline Semantics

A baseline is scoped to the measured hardware, device, Transformers version, and
torch_fl commit. It supports comparisons and regression claims; it is not a
permission gate that suppresses every first-sweep defect.

The hardware scope is real, not documentation-only: deduplication reads only the
sections whose heading shares a word with the `--hardware` label, so a run on
`MetaX C550` is compared against the `MetaX` measurement and not against the
`MUSA MTT S5000` one. Two vendors can register the same device name, so the
section heading is the only place the boards can be told apart.

A first sweep may produce an issue when an individual finding has complete
evidence, independent reproduction where required, a named cause, deduplication,
a finished issue body, and explicit authorization. Describe it as observed on
the pinned tuple, not as a regression without an earlier matching measurement.

## Safe Wrapper Boundaries

The safe wrapper validates model, chip, and repository parameters and
invokes only the checked report-only scripts. It does not install dependencies,
edit source files, change the parent shell environment, or publish issues. It
has no device parameter: the device name comes from
`tests/manual/hf_device_spec.py`.

`--chip` is a hardware label, not a routing parameter, so it is validated by the
vendor it names and passed on exactly as written: `MetaX`, `MetaX C550`, and
`MUSA MTT S5000` are all accepted, and the label reaches the issue title in the
spelling the caller used.

If preflight dependencies are missing, stop and report the environment problem
rather than modifying the torch installation during the measurement.

## Troubleshooting

### Every result is `ENVIRONMENT_ERROR`

Nothing was measured. The runner exits `2` for this. Read
`environment.preflight` in the report: it names which check failed — the
`torch_fl` import, a PrivateUse1 name that does not match the spec's
`DEVICE_NAME`, a device count of zero, a missing HF hook, or a `transformers`
version that is not the requested one. Fix the environment and re-run; do not
treat the run as coverage, and do not file an issue from it.

A verdict of `the preflight published no verdict` means the check died before it
could report anything. That is itself a defect worth reporting, but the common
cause is a child that never started: the tests run from a private work
directory, so `torch_fl` has to be importable without the repository as the
working directory. `PYTHONPATH="$PWD"` selects a working tree, and leaving it
unset selects an installed build. Run the sweep's own gate to see the same check
the preflight performs:

```bash
PYTHON=/opt/conda/bin/python3 PYTHONPATH="$PWD" \
    bash scripts/transformers/transformers_auto_sweep.sh bert MetaX
```

### The sweep refuses to start with `cannot import the test environment`

Step 0 probes the interpreter from an empty directory, deliberately: that is
where every test child runs, and `python -c` would otherwise import the
repository's own `torch_fl` merely because the sweep was launched from the
repository root. The message names the module that could not be imported. Install
the accelerator build into `PYTHON`, or export `PYTHONPATH` pointing at the
checkout or the installed package that provides `torch_fl`.

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
and confirm the output says one test was collected. The verifier checks this
itself: a run reporting anything other than one test is recorded as `ERROR`, and
if no isolation of a run selected exactly one test the verifier exits `2`.

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
bash -n scripts/transformers/transformers_auto_sweep.sh \
        scripts/transformers/transformers_batch_sweep.sh
```

The regression suite covers triage on both output shapes, the
all-`ENVIRONMENT_ERROR` summary, fingerprint normalization and stability, the
baseline round-trip and its hardware scope, nodeid canonicalization, the
exactly-one-test isolation rule, dedup failure modes, preview/filer parity, and
the device contract. It intentionally excludes GitHub filing because generated
drafts require human completion and explicit authorization; the refusals are
asserted in the tests instead.

## Related Documentation

- `.claude/skills/transformers-test/SKILL.md`
- `docs/workflows/resilient-testing-quickstart.md`
- `docs/design/robust-harness-proposal.md`
- `docs/reference/hf-coverage.md`
- `.github/ISSUE_TEMPLATE/ai_agent_issue.md`
