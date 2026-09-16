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

### False green on a broken environment

A run that measured nothing and a run that measured everything and found nothing
were both reported the same way. `--all` aggregate output triaged to zero
findings because triage read only the flat shape, non-`FAIL` statuses were
dropped outright, and a model whose every test errored on a missing `torch.flagos`
hook was summarized as "all tests passed".

### A device name that could disagree with the tests

`--device` was recorded as provenance but never enforced. The test device is the
`DEVICE_NAME` in `tests/manual/hf_device_spec.py`; a second copy on the command
line was documented with the wrong default and omitted from the safe wrapper's
allowlist, so `reset_device_context()` — which compares against the same value —
could silently become a no-op and leave device-context poisoning undetected.

### A baseline that could never learn, and a dedup that failed open

The baseline writer emitted a table row with a `Fingerprint` column; the reader
matched only a standalone ``Fingerprint: `hash` `` line, so a round-trip through
`docs/reference/hf-coverage.md` returned nothing and every known cause looked new.
Separately, an unreachable or failing `gh` was treated as "no duplicate found",
which filed the same defect again.

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
- `FLAGOS_LOG_FALLBACK=1` is enabled and fallback operators are recorded;
- a preflight runs in a child process before the first batch and is recorded
  under `environment.preflight`;
- the device name is read from the spec with `ast`, so the parent process never
  claims the accelerator just to learn it;
- exit `0` means measured and clean, `1` measured with findings, and `2` nothing
  measured.

### Triage

`scripts/transformers/transformers_triage.py` converts runner output into cause-oriented
findings. It supports:

- `OP_UNSUPPORTED`;
- `OP_CPU_FALLBACK`;
- `FEATURE_UNSUPPORTED`;
- `PRECISION`;
- `CRASH`;
- `TEST_ERROR` and `ENVIRONMENT_ERROR`, reported but never actionable;
- known precision patterns and unknown failures retained for review.

A run-level poison marker no longer classifies every failed test as a crash.
Crash attribution requires per-test evidence. CPU fallback findings can come
from otherwise passing tests and are confirmed directly by the runtime log.

`TEST_ERROR` covers a failure in the test's own setup, teardown, or collection;
`ENVIRONMENT_ERROR` covers a missing dependency, an import failure, or an absent
accelerator. Both carry `actionable = false` and `verification_required = false`,
so they appear in the summary but cannot reach the tracker. The triage summary
counts them separately (`total_failures`, `actionable`, `environment_error`),
which is what keeps a run that measured nothing from being read as a clean run.

It also accepts the `--all` aggregate shape: `run_units()` walks `models[]` when
`mode == "all"` and reads each model's own `environment.device`, so a full sweep
triages the same findings a per-model run would.

Fingerprints include the failure class, responsible component, subject, and
normalized mechanism. Model names and nodeids are aggregated occurrences.
`generate_fingerprint` (aliased as `compute_fingerprint`) and `normalize_error`
live here and nowhere else; `normalize_error` is idempotent, so re-normalizing an
already-normalized mechanism produces the same fingerprint.

The per-test value the runner writes is named `occurrence_fingerprint`: it
includes the nodeid and identifies one test result, which is deliberately not
what dedup uses.

### Verification

`scripts/transformers/transformers_verify.py` reruns one representative nodeid per finding in
a fresh pytest subprocess. It reconstructs the official environment with:

- the exact Transformers source tree;
- the source `pyproject.toml` and root directory;
- `hf_device_spec.py`;
- source package symlinks and `PYTHONPATH` entries;
- backend autoload disabled;
- fallback logging enabled.

The subprocess environment is the runner's `child_env`, not a second definition of
it. The source tree leads so that HF's `tests.models...` imports resolve into the
measured tree, the caller's `PYTHONPATH` follows, and a repository root the caller
supplied is moved to the end instead of being dropped: `hf_device_spec.py` imports
`torch_fl`, and a checkout that was never installed is importable only through
that entry. Dropping it made every isolation fail to collect with
`ModuleNotFoundError: No module named 'torch_fl'`, which the verifier could only
report as `ERROR` and which reads like a device fault rather than a harness one.

Verification defaults to one worker and rejects parallel execution because
separate processes may still share an accelerator and memory pool.

Verdicts are:

- `FAIL` or `TIMEOUT` -> `CONFIRMED`;
- `PASS` or `SKIP` -> `COLLATERAL`;
- setup, import, collection, or runner `ERROR` -> `INCONCLUSIVE`.

An isolated `TIMEOUT` means the finding is a hang, so its filed class becomes
`CRASH` and an `isolation_note` records the reclassification; the class the
verifier otherwise preserves is the one triage assigned. `--test-source-dir`
defaults to the runner's own cache root (`HF_COVERAGE_CACHE`, else
`~/.cache/torch_fl/hf-tests`), and the versions verified against are the ones
recorded in the findings JSON --- triage carries the measured environment
through to its output, and the verifier reads `transformers` from there --- so a
multi-version cache cannot verify a finding against a tree that did not produce
it.

### Deduplication

`scripts/transformers/transformers_deduplicate.py` checks exact fingerprints in the coverage
record, issue bodies, and issue comments. It then searches by subject for older
issues that predate fingerprints.

The coverage record is read per board: `--hardware` selects the `## Baseline:`
sections that describe the hardware this run measured, and sections belonging to
another board are skipped and reported. Both vendors that register a `flagos`
PrivateUse1 device name would otherwise share one merged set of fingerprints.

Semantic matches are emitted as `REVIEW_CANDIDATE` and blocked from filing until
a human compares the component and mechanism. Collateral and inconclusive
findings are also blocked. Findings classified `TEST_ERROR` or
`ENVIRONMENT_ERROR` never reach GitHub at all, and neither does a baseline hit.

The check fails closed. A `gh` invocation that is missing, times out, or exits
non-zero raises `GitHubSearchUnavailable` and marks the finding
`DEDUP_UNAVAILABLE`; `--skip-github` marks it `NOT_CHECKED`. Both carry
`should_file = false`. Only a search that actually ran and returned nothing
yields `NEW`, so an unreachable tracker can no longer be mistaken for a clean
dedup pass and re-file a known defect.

The baseline is read from the cause table in `docs/reference/hf-coverage.md`,
keyed by its leading `Fingerprint` column, with the standalone
``Fingerprint: `hash` `` form still accepted because that is how issue bodies
carry it. Both forms resolve their reference from the row's own issue cell and
the nearest preceding `## Baseline:` heading.

### Preview generation

`scripts/transformers/transformers_preview_issues.py` writes one Markdown draft per new
finding and a consolidated preview. Drafts follow the repository AI issue
template structure and include the captured evidence, fingerprint, isolated
command, proposed verification, and suggested labels.

Each draft is paired with an `issue-<fingerprint>.json` sidecar recording its
title, labels, class, subject, and the fingerprint the file is named for. The
sidecar is the only place a title and a label set are decided, so the preview and
the filer cannot disagree, and the `## Issue Type` checkbox is chosen from the
same table that chooses the labels.

They intentionally leave required review placeholders for:

- the actual AI model;
- complete hardware and software environment;
- a validated minimal reproducer;
- root-cause analysis;
- responsible code locations;
- human completion of the checklist.

Placeholders are machine-checkable: each one is an
`<!-- UNFILLED: <field> -->` marker rather than prose, because a free-text
placeholder could pass a gate that only looked for two exact strings. The
traceback keeps the runner's head-and-tail convention instead of a one-sided
truncation, so the exception survives.

For non-CUDA-compatible operator work, proposed solutions direct contributors to
the platform code generator rather than handwritten per-operator kernels.

### Filing

`scripts/transformers/transformers_file_issues.py` is a separate optional tool. It requires an
explicit list of fingerprints, rejects non-confirmed findings, and rejects
incomplete drafts. There is no bulk approval option.

It submits exactly what the preview wrote: it reads each draft's sidecar for the
title and labels rather than rebuilding them from the body. Before any GitHub
write it pre-collects every problem across all drafts and reports them together,
so a run with three incomplete drafts names three problems instead of filing the
first two and failing on the third.

When it records a filed issue in the baseline it adds the `Fingerprint` column
and inserts the row into the last cause table in the file, so a second
`## Baseline:` section updates rather than accumulating a stray table after the
trailing prose.

The filer reads the current `Platform` field emitted by the preview tool and
retains compatibility with legacy drafts that used `Chip`.

### Safe and automatic wrappers

`scripts/transformers/transformers_auto_sweep.sh` executes the full measurement and preview
pipeline, then prints the command shape for a later explicitly authorized filing
action. It does not invoke the filer. It resolves one interpreter
(`${PYTHON:-python3}`) and checks that it can import `torch`, `transformers`, and
`torch_fl` before the first measurement, so a box without them fails loudly
rather than surfacing as a model-name error. The probe runs from an empty
directory, because that is where every test child runs and because `python -c`
puts the working directory on `sys.path`: probing from the repository root would
import the checkout's own `torch_fl` and call an interpreter healthy that no
child can use. The verdict is the probe's stdout and the interpreter's warnings
go to a file, because merging the two turns a working device build's import
warnings into a "cannot import" verdict. `PYTHONPATH` is therefore what selects
the build — export the repository root to measure the working tree, leave it
unset to measure an installed one — and the runner keeps that entry on the
children's path for the same reason. The sweep does not hardcode a cache path,
and its work directory honours `TMPDIR`.

Both sweeps consume the runner's three exit codes — `0` measured and clean, `1`
measured with findings, `2` nothing measured — and the batch driver reports
`Nothing to file`, `Findings`, and `Not measured` as separate counts, exiting `2`
when any model measured nothing.

The automatic sweep also refuses to present a preview as a filing opportunity
when the dedup check did not run. It counts `NEW` and the blocked statuses from
`summary.dedup`, not from the findings array, because only `NEW` findings appear
there and a wrapper that looked for `DEDUP_UNAVAILABLE` among them would always
count zero. A run with nothing new and at least one unchecked finding exits `1`
and says so, instead of reporting "all findings are known".

`scripts/transformers/safe_transformers_wrapper.py` validates parameters before invoking that
same report-only path. Safe mode cannot publish issues automatically. It validates
the model and chip against allowlists; there is no device parameter to validate,
because the device name comes from `tests/manual/hf_device_spec.py` and the
wrapper does not let the caller override it.

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
- `tests/manual/hf_device_spec.py`
- `tests/manual/transformers_hf_source.py`
- `tests/unit/test_transformers_hf_tests.py`
- `tests/unit/test_transformers_automation.py`

The automation regression tests live in `tests/unit/test_transformers_automation.py`.
Their fixtures are produced by running the runner's own `reduce_records()` over
synthetic pytest reports rather than hand-written JSON: the `--all` and
all-`ENVIRONMENT_ERROR` defects both shipped because the fixtures were
hand-written and had drifted from what the runner emits.

### Documentation

- `.claude/skills/transformers-test/SKILL.md`
- `docs/design/robust-harness-proposal.md`
- `docs/workflows/resilient-testing-quickstart.md`
- `docs/workflows/transformers-auto-triage.md`
- `docs/reference/hf-coverage.md`

## Important Invariants

1. A selected-nodeid subprocess must not also receive the architecture directory.
2. A valid isolation rerun collects exactly one test. The verifier reads that
   count from pytest's own summary line, records any other count as `ERROR`, and
   exits `2` when no isolation of a run selected a single test. A nodeid the
   runner recorded without its file part is restored from the nodeid that was
   selected, matched on its `::Class::test` tail; a tail two files share is left
   as reported.
3. Completed records from a crashed batch are never overwritten.
4. A run-level poison marker does not identify the triggering test.
5. A passing assertion with CPU fallback is not accelerator success.
6. Accelerator verification is serial until devices can be isolated per worker.
7. A runner error is inconclusive, not confirmed.
8. Semantic issue matches require review rather than automatic deduplication.
9. Automatic and safe modes stop before GitHub writes.
10. Publication requires complete issue content and explicit fingerprint-level
    authorization.
11. An environment error is never reported as a clean run. A result that failed
    preflight, or whose every outcome was `ENVIRONMENT_ERROR`, is a measurement
    failure with exit code `2`, and triage reports it as such.
12. The device name has exactly one source: `DEVICE_NAME` in
    `tests/manual/hf_device_spec.py`. No command-line flag sets it, and no
    recorded provenance can disagree with the spec the tests actually read.
13. A dedup check that could not run is not a new finding. `DEDUP_UNAVAILABLE`
    and `NOT_CHECKED` both mean `should_file = false`.
14. The environment gate asks the question the test children will ask. The
    interpreter probe runs from an empty directory — never the repository root,
    which `python -c` puts on `sys.path` but the children never see — and reads
    its verdict from stdout alone, so the device build's import warnings cannot
    be mistaken for missing modules. A preflight always publishes a named
    diagnosis; a check that raises while reporting a problem adds "the
    environment checks could not run" rather than killing the child, because "no
    verdict" is indistinguishable from a child that never started.
15. A baseline belongs to one board. Deduplication reads only the sections whose
    heading names the hardware a run measured, because two vendors can register
    the same PrivateUse1 device name, and an unscoped read would let one board's
    measurement suppress a finding measured on another.
16. The version a finding is verified against comes from the run that produced
    it. Triage carries the measured environment through; the newest cached
    source tree is a fallback that announces itself, not the default.
17. An isolation reproduces the environment that produced the finding. The
    verifier reuses the runner's `child_env` ordering --- source tree first, the
    caller's `PYTHONPATH` next, a caller-supplied repository root last --- so
    `hf_device_spec.py` can import `torch_fl` from a checkout that was never
    installed. A child that cannot import the device build dies before it
    collects a test, and that kind of failure must not sit beside genuine
    per-test evidence as one more `ERROR`.
18. A stage's outcome survives the hand-off to the stage that acts on it. The
    sweep exits `1` once it has written drafts for review, because the batch
    driver reads that code and nothing else: reaching the end of the preview
    successfully is not the same measurement as having nothing to report, and a
    sweep that fell off the end of step 6 was counted as `Nothing to file`
    while its preview sat in the work directory.

## Usage

Run one architecture through the report-only path:

```bash
# The working tree, when the repository is not installed:
PYTHON=/opt/conda/bin/python3 PYTHONPATH="$PWD" \
    bash scripts/transformers/transformers_auto_sweep.sh qwen3 "MUSA MTT S5000"

# An installed build, with no repository on the path:
PYTHON=/opt/conda/bin/python3 \
    bash scripts/transformers/transformers_auto_sweep.sh qwen3 "MUSA MTT S5000"
```

`PYTHON` selects the interpreter and `PYTHONPATH` selects the build, and step 0
refuses to start if that pair cannot import `torch`, `transformers` and
`torch_fl` from an empty directory. The sweep's exit code is the contract: `0`
measured and clean, `1` measured with findings to review, `2` nothing measured.

Or run the stages individually:

```bash
python scripts/transformers/transformers_triage.py results.json --out classified.json

python scripts/transformers/transformers_verify.py \
    classified.json \
    --out verified.json \
    --transformers-version 5.16.1 \
    --workers 1

python scripts/transformers/transformers_deduplicate.py \
    verified.json \
    --out new.json \
    --coverage-file docs/reference/hf-coverage.md \
    --hardware "MUSA MTT S5000" \
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
bash -n scripts/transformers/transformers_auto_sweep.sh \
        scripts/transformers/transformers_batch_sweep.sh
```

The regression suite covers triage on both output shapes, the
all-`ENVIRONMENT_ERROR` summary, fingerprint normalization and stability, the
baseline round-trip, dedup failure modes, preview/filer parity, and the device
contract. Filing against a live tracker is excluded because generated drafts are
incomplete until human review and explicit authorization; the refusals are
asserted in the unit tests instead.

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
