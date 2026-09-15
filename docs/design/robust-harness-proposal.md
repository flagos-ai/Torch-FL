# Robust Transformers Test Harness Design

## Problem

An accelerator fault can terminate or poison a HuggingFace architecture test run
before pytest reports every test. A single-process run then loses useful evidence
and can make later tests look like independent failures even when they are only
collateral damage.

The harness needs to:

1. Continue collecting evidence after a batch crashes or times out.
2. Preserve every report written before the process stopped.
3. Identify unreported tests without overwriting completed results.
4. Reproduce candidate failures in fresh subprocesses.
5. Keep all GitHub actions behind review and explicit authorization.

## Constraints

- Pytest combines a directory selector and explicit nodeids as a union. A batch
  command must therefore pass only the selected nodeids, not the architecture
  directory as an additional target.
- Separate Python processes can still share one accelerator and memory pool.
  Verification must remain serial until workers can be pinned to independent
  devices.
- A run-level poisoned-context flag invalidates later results but does not prove
  that every failed test caused the poison.
- CPU fallback can make an assertion pass while providing no accelerator
  implementation coverage. Fallback operators are findings, not passes.
- Setup, import, or collection errors do not confirm the original backend
  defect. They are inconclusive verification outcomes.

## Design

### 1. Version-matched collection

The runner resolves the official Transformers source tree for the installed
version and collects the architecture nodeids without executing them. Collection
uses the same device specification, source root, and pytest configuration as the
real run.

A collection or environment failure stops the architecture measurement. It is
not converted into operator findings.

### 2. Resilient batch execution

When `--resilient` is enabled, the runner splits collected nodeids into batches.
Each batch runs in a fresh pytest subprocess with its own timeout and JSONL
report. The command contains the nodeids only:

```python
pytest_command(
    target,
    marks,
    ["-p", "hf_report_plugin", *batch_nodeids],
    collect_only=False,
    include_target=False,
)
```

The report plugin writes a record after each test phase. After the subprocess
returns, the runner reduces those records and keeps every completed test result.
If the subprocess crashes or times out, only nodeids that produced no report are
added as `BATCH_CRASHED` placeholders.

The aggregate JSON is rewritten atomically after each batch. An interrupted run
therefore leaves a readable partial measurement.

### 3. Evidence preservation

Tracebacks are bounded without discarding the exception tail. For long output,
the plugin retains both the beginning and end:

```text
[first 4000 characters]
... <truncated> ...
[last 4000 characters]
```

This preserves setup context and the final exception while keeping the report
size manageable.

### 4. CPU fallback measurement

Every child process receives:

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0
TRANSFORMERS_TEST_DEVICE_SPEC=hf_device_spec.py
FLAGOS_LOG_FALLBACK=1
```

The plugin captures messages of the form:

```text
[flagos cpu_fallback] aten::<operator>
```

and records a deduplicated `cpu_fallback_ops` list on each test. Triage converts
each measured operator into a confirmed `OP_CPU_FALLBACK` finding even when the
test assertion passed.

### 5. Cause-oriented triage

Triage groups occurrences by a fingerprint computed from:

```text
failure class | responsible component | subject | normalized mechanism
```

The model name and nodeid are occurrence data and do not enter the fingerprint.
The component is required so unrelated vendor backends do not collapse into one
finding.

Run-level context poisoning is not applied to every failure. A test is classified
as a crash only when its own evidence contains a crash signature such as an
illegal memory access, device-side assertion, fatal signal, or timeout.

### 6. Fresh-process verification

Each finding that requires verification is rerun as exactly one nodeid in a fresh
subprocess. The verifier recreates the official runner environment, including the
source `pyproject.toml`, `--rootdir`, device specification, and package symlinks.

Verification is serial by default and rejects `--workers` values other than one.
The verdict mapping is:

| Isolation result | Finding verdict |
| --- | --- |
| `FAIL` or `TIMEOUT` | `CONFIRMED` |
| `PASS` or `SKIP` | `COLLATERAL` |
| runner/setup `ERROR` | `INCONCLUSIVE` |

CPU fallback findings do not need another rerun because the fallback log is the
direct runtime evidence.

### 7. Deduplication

Deduplication proceeds in this order:

1. Exact fingerprint match in the coverage record.
2. Exact fingerprint match in GitHub issue bodies or comments.
3. Semantic subject search for older issues created before fingerprints existed.

A semantic match is a `REVIEW_CANDIDATE`, not an automatic duplicate. Collateral
and inconclusive findings are blocked from filing.

### 8. Report-only automation

The automated and safe wrappers stop after generating issue previews. Generated
drafts intentionally retain required review placeholders for environment,
reproducer, root-cause analysis, code locations, and reviewer completion.

The filing tool requires explicit fingerprints:

```bash
python scripts/transformers/transformers_file_issues.py findings.json \
    --approve <fingerprint> [<fingerprint> ...]
```

It rejects findings without a `CONFIRMED` verdict and rejects drafts that still
contain mandatory placeholders. There is no bulk approval option.

## Failure Handling

| Failure | Harness behavior |
| --- | --- |
| Batch process exits normally | Keep all reported test records |
| Batch crashes | Keep reported records; mark only unreported nodeids `BATCH_CRASHED` |
| Batch times out | Same preservation rule as a crash |
| Device reset fails | Record the warning and continue with a fresh subprocess |
| Verification cannot start | Mark the finding `INCONCLUSIVE` |
| GitHub search returns a semantic candidate | Require human duplicate review |
| Draft is incomplete | Refuse to publish it |

A best-effort cache clear can help after a recoverable failure, but it is not
considered proof that a poisoned driver or device context has recovered. Fresh
subprocesses and isolated reruns provide the evidence boundary.

## Verification

The implementation is covered by unit tests for:

- exact-nodeid command construction without the directory/nodeid union;
- fallback logging and per-test operator aggregation;
- preservation of completed results from crashed batches;
- `BATCH_CRASHED` summary behavior;
- fallback-to-finding conversion;
- component-aware fingerprints;
- per-test crash classification despite a run-level poison marker;
- `ERROR` to `INCONCLUSIVE` verification mapping;
- exact and versioned Transformers source resolution;
- generated issue metadata parsing.

Manual validation should additionally run an architecture suite on the target
hardware, inspect the fallback operator list, and rerun every candidate nodeid in
a fresh process before any issue draft is completed.

## Trade-offs

- Smaller batches limit collateral damage but add pytest startup overhead.
- Serial verification is slower than concurrent reruns but avoids shared-device
  interference.
- Fingerprints improve exact deduplication but still require semantic review for
  historical issues.
- Report-only automation requires human work before publication, intentionally
  trading throughput for accurate and authorized tracker records.

## Success Criteria

1. A batch crash does not erase earlier test records.
2. A nodeid isolation command collects exactly one test.
3. A run-level poison marker does not turn unrelated failures into crashes.
4. Any measured CPU fallback is visible as an accelerator coverage finding.
5. Setup errors cannot become confirmed defect reports.
6. Automated and safe modes perform no GitHub writes.
7. Issue publication requires a completed draft and explicit fingerprint-level
   authorization.
