---
description: >
  Run the full automated HuggingFace unit-test harness for the `flagos`
  accelerator. Invoke with one transformers model, one diffusers pipeline/model,
  or `all` to sweep every suite, then read a per-target report that
  classifies each failure (unsupported operator, precision, crash) with a
  root-cause fingerprint. Use to measure HF support after enabling operators,
  after a transformers/diffusers version change, or to compare the vendor,
  boxing, and flaggems routes.
argument-hint: "<transformers-model> | <diffusers-pipeline> | all"
---

# transformers-test (torch_fl)

Fully automatic runner for HuggingFace official unit tests — the `transformers`
suite (`tests/models/<module>/`) and the `diffusers` suite
(`tests/pipelines/<name>/`, `tests/models/`) — on the flagos accelerator.
Unattended once invoked: it pins the environment, injects `flagos` as the test
device, isolates every suite in a subprocess, classifies failures, deduplicates
root causes by fingerprint, and writes a report. It never writes to GitHub.

## Target

The invocation target is: **$ARGUMENTS**

Interpret it as:
- a **transformers architecture** module name (e.g. `/transformers-test bert`,
  `/transformers-test qwen3`) — sweep `<transformers-src>/tests/models/<module>/`;
- a **diffusers pipeline or model directory** name (e.g.
  `/transformers-test audioldm2`) — sweep `<diffusers-src>/tests/pipelines/<name>/`;
- **`all`** — a full sweep of every suite. Default when no argument is provided.

Prefer the stable command form `/transformers-test <target>` over editing this
file. For targets whose upstream directories share a name between transformers
and diffusers, run the transformers sweep first and note the diffusers overlap
in the report rather than skipping either.

## Scope and prerequisites

Complete these before the first run on a machine:

- Device bring-up passes: `torch.flagos` registers, allocates, and copies.
  Vendor route: native vendor PyTorch / SDK environment (`device="cuda"`).
  Boxing and flaggems routes: `/opt/fl-envs/boxing/bin/python` with `torch_fl`.
- The exact test sources are available offline:
  - transformers source tree matching the pinned `transformers` version
    (`<transformers-src>`);
  - diffusers source tree matching the pinned `diffusers` version
    (`<diffusers-src>`);
  - `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` unless a fixture must be fetched.
- A cached seed of tiny-random Hub fixtures for the target if the run must
  stay fully offline. Missing fixtures are an environment gap, not a backend
  failure; recover connectivity first (proxy, then `HF_ENDPOINT`) before
  classifying the result.

## Step 1 — pin and record the environment

Two versions decide how a result is read: the transformers/diffusers version and
the torch_fl commit. A failure carries no meaning without both. Record beside
every result: Python executable and version, torch version, torch_fl commit,
transformers and diffusers versions, accelerator model, driver, SDK, vendor
library, and the device-injection mode used (spec module vs env var).

```bash
python - <<'PY'
import torch, torch_fl
print("python-exec", __import__("sys").executable)
print("torch", torch.__version__)
print("torch_fl", torch_fl.__file__)
try:
    print("device_count", torch.flagos.device_count())
except AttributeError:
    pass
PY
git rev-parse --short HEAD
```

For the vendor route, run the same block with the vendor Python and record the
chip model and SDK revision. A run that executed against the wrong runtime is an
environment failure, not a coverage data point.

## Step 2 — inject `flagos` (or the route device) as the HF test device

Route determines the device name and injection mechanism:

| Route | Interpreter | Device | Injection |
|---|---|---|---|
| vendor baseline | system / vendor Python | `cuda` | native vendor PyTorch with no torch_fl |
| boxing | `/opt/fl-envs/boxing/bin/python` | `flagos` | `TRANSFORMERS_TEST_DEVICE_SPEC=tests/manual/hf_device_spec.py` (transformers) or `DIFFUSERS_TEST_DEVICE=flagos` (diffusers) |
| flaggems | `/opt/fl-envs/boxing/bin/python` | `flagos` | boxing injection plus `FLAGOS_USE_FLAGGEMS=1 TRITON_BACKENDS_IN_TREE=1` |

Do not set `TRANSFORMERS_TEST_DEVICE=flagos` for transformers 5.16.1: that
release validates the variable with `torch.device()` before importing the spec,
so a custom PrivateUse1 name is not accepted at that point. The spec's
`DEVICE_NAME` becomes the effective test device after it is loaded. For built-in
devices, `TRANSFORMERS_TEST_DEVICE` remains available as an alternative to a
spec file.

The spec module must define `DEVICE_NAME` plus the three backend hooks HF
dispatches on; a missing hook raises at import unless HF has a default:

```python
DEVICE_NAME = "flagos"
MANUAL_SEED_FN = torch.flagos.manual_seed
EMPTY_CACHE_FN = torch.flagos.empty_cache
DEVICE_COUNT_FN = torch.flagos.device_count
```

Note the gating split this produces, and do not try to defeat it:
`require_torch_accelerator` passes for `flagos` (non-CPU, non-None), while
`require_torch_gpu` skips because it compares against `cuda` literally. Cases
skipped by `require_torch_gpu` are CUDA-specific and are not coverage gaps.

### Diffusers device injection

For the diffusers suite, inject through the diffusers device contract instead:

```bash
export DIFFUSERS_TEST_DEVICE=flagos
export PYTHONPATH=<diffusers-src>
export HF_HUB_OFFLINE=1
```

### Fixture connectivity

Some official tests download tiny fixtures from the Hugging Face Hub. If
`huggingface.co` is unreachable, do not classify the resulting retries or
network errors as backend failures. First check the local Hub cache and rerun
with `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` when all fixtures are present.
Otherwise load the configured proxy and retry the same command; if the origin
still cannot be reached, use the Hub mirror:

```bash
source ../proxy.sh
# Retry with HF_HUB_OFFLINE and TRANSFORMERS_OFFLINE unset.
python tests/manual/transformers_hf_tests.py --model <model> --out /tmp/<model>.json

export HF_ENDPOINT=https://hf-mirror.com
python tests/manual/transformers_hf_tests.py --model <model> --out /tmp/<model>-mirror.json
```

Use the proxy before `HF_ENDPOINT`; unset `HF_HUB_OFFLINE` and
`TRANSFORMERS_OFFLINE` for online retries. `HF_ENDPOINT` is inherited by the
runner's isolated pytest process and affects Hub fixture downloads only. Record
the endpoint mode used by the run, without exposing proxy credentials.

## Step 3 — run the official suite for the target

### Transformers model

```bash
export TRANSFORMERS_TEST_DEVICE_SPEC=tests/manual/hf_device_spec.py
python tests/manual/transformers_hf_tests.py --model <module> \
  --out /tmp/transformers-<module>-<route>.json
```

`--all` visits each test directory once, not each registry key, and rewrites the
aggregate JSON after every architecture so an interrupted sweep still holds its
measurements. A version-mismatched source tree produces failures that belong to
HF, not to torch_fl; the runner verifies the version declared by
`src/transformers/__init__.py` before running.

### Diffusers pipeline

```bash
export DIFFUSERS_TEST_DEVICE=flagos
export PYTHONPATH=<diffusers-src>
python -m pytest <diffusers-src>/tests/pipelines/<name>/ -q \
  -x --timeout 600 --timeout-method=thread \
  -p no:cacheprovider --disable-warnings \
  2>&1 | tee /tmp/diffusers-<name>-<route>.log
```

For a full diffusers sweep use the dedicated sweep entry point if one exists in
`tests/manual/`, otherwise drive `pytest` per directory from the target list and
aggregate into one JSON per route. Keep raw logs outside the repository.

## Step 4 — isolate every model or pipeline in a subprocess

Run each suite in its own subprocess with a timeout, even in single-target mode.
Accelerator faults are not contained: one illegal memory access poisons the
device context, after which every later operation fails with the same symptom.
Treat a poisoned run as one finding for the whole target, never as a list of
per-test failures:

```text
illegal memory access | device-side assert | unspecified launch failure
misaligned address | vmfault | acceleratorerror
```

Write results incrementally so an interrupted sweep resumes instead of
restarting, and keep raw stdout/stderr per target for auditing.

## Step 5 — classify the failure and name the root cause

Every failure must land in exactly one class:

| Class | Signal | Actionable |
|---|---|---|
| `OP_UNSUPPORTED` | `NOT_SUPPORTED`, `backend not registered`, `could not run 'aten::…'` | yes — enhancement |
| `FEATURE_UNSUPPORTED` | non-operator runtime or feature gap | yes — enhancement |
| `PRECISION` | ran, but disagrees with the CPU baseline | yes — bug |
| `CRASH` | segfault, poison, or timeout | yes — bug |
| `UNTESTED` | CPU baseline itself errors | no |
| `NOT_IN_VERSION` | target absent from the pinned version | no |
| `CUDA_SKIP` | `require_torch_gpu` skip | no |

For `OP_UNSUPPORTED` and `PRECISION`, re-run the failing layer under
`TorchDispatchMode` and capture the aten calls, then put the specific operator
in the finding. Without this attribution the report only says a model failed,
which a maintainer cannot act on.

Judge precision against CPU, same model, same seed, same dtype, with
dtype-scaled tolerances: float32 `rtol=atol=1e-5`, float16 `1e-3`, bfloat16
`rtol=1.6e-2 atol=1e-2`. Keep `NaN`/`Inf` separate from magnitude disagreement
and rank it higher.

## Step 6 — deduplicate root causes by fingerprint

The same root cause reappears across models and platforms. Compute the cause
fingerprint from the finding established in Step 5, not from raw pytest output:

```python
import hashlib, re

def normalize(text: str) -> str:
    t = re.sub(r"0x[0-9a-fA-F]+", "0xADDR", text)
    t = re.sub(r"/tmp/[^\s'\"]+", "/tmp/PATH", t)
    t = re.sub(r"(/[^\s'\"]*)?/(site-packages|torch_fl|tests)/", r"/PATH/\2/", t)
    t = re.sub(r"\b\d+\.\d+s\b", "TIMEs", t)
    t = re.sub(r"\[[\d,\s]+\]", "[SHAPE]", t)
    t = re.sub(r"(?<!err )(?<!code )(?<!errno )\b\d+\b", "N", t)
    return re.sub(r"\s+", " ", t).strip()[-200:]

def cause_fingerprint(failure_class, component, subject, mechanism):
    payload = "|".join(
        [failure_class, component, subject, normalize(mechanism)]
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:12]
```

`subject` is `aten::<op>.<overload>` when an operator was named, otherwise the
feature or module that failed. `component` identifies the implementation
responsible for the cause (`musa:SubTensorKernelMusa`, `ascend:generated-add`,
`shared:privateuse1-registration`, `flaggems:<op>`). `mechanism` is the shortest
verified statement that distinguishes the defect. Do not hash an entire
traceback. Two vendor backends that both report an operator unsupported are
separate implementation gaps and keep separate fingerprints.

State the chip, route, and model in the report title, plus the
transformers/diffusers version. For dedup to work across runs, every recorded
finding carries the value on its own line:

```text
Fingerprint: `cce6ae545772`
```

## Step 7 — write the per-target report

Write one report per run under `docs/` when the sweep is a tracked deliverable,
otherwise keep it in `/tmp`. Include:

- environment block from Step 1 and the route matrix columns that were actually
  measured (vendor baseline / boxing / flaggems);
- per-class counts and, per failure, the class, subject operator, mechanism,
  and cause fingerprint;
- models or pipelines skipped, timed out, or absent from the pinned version in
  their own category — never folded into failures;
- raw evidence location, without committing package caches, core dumps,
  credentials, or machine-local environment files. MUSA faults leave
  `core_*.mudmp` dumps in the working tree: move JSON to the evidence directory
  and delete the dumps before committing.

`docs/` must be written in English per `CLAUDE.md`. Report only what was
attempted on this machine in this run; never carry another cohort's numbers
forward or copy one chip's rate onto another. Unavailable hardware is marked
**not revalidated**.

## Step 8 — record measured evidence (baseline gate)

Compare each cause fingerprint with the pre-existing measurement in
`docs/reference/hf-coverage.md`:

- present in the baseline: a known cause;
- absent from the baseline but present in an open issue: a duplicate;
- absent from the baseline but present in a closed issue: report the issue and
  its closing disposition;
- absent from both: a regression or newly discovered cause.

A first sweep on a chip is the baseline: record it, perform no tracker writes,
and say so. Recording the current sweep immediately before filing does not turn
it into an earlier baseline. This runner is report-only: no GitHub write happens
inside the command. Any issue or comment is a separate outward-facing action
that the user performs or explicitly authorizes after reading the report.

Update `docs/reference/hf-coverage.md` with chip, run date, versions, torch_fl
commit, targets attempted, and per-class counts. Every finding gets a row with
its fingerprint, class, subject, and issue number if one was filed later.

## Step 9 — close out

- confirm the version and torch_fl commit are recorded with every result;
- confirm every `OP_UNSUPPORTED` and `PRECISION` finding names an operator;
- confirm every finding row carries its fingerprint;
- confirm `NOT_IN_VERSION`, `UNTESTED`, and CUDA-only skips are excluded from
  failure counts;
- confirm no tracker write occurred without explicit user authorization;
- confirm raw JSON and vendor core dumps are not in the commit;
- run `ruff check .` and `ruff format --check .` if any Python was written.

Summarize the run to the user with the route matrix, per-class counts, the
highest-frequency fingerprints, and the list of targets that need a decision
(new cause vs known vs duplicate).
