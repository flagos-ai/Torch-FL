---
name: transformers-test
description: >
  Fully automatic runner for HuggingFace official unit tests — both the
  transformers suite (tests/models/<module>/) and the diffusers suite
  (tests/pipelines/<name>/, tests/models/) — on the flagos accelerator.
  Use this to measure a platform's HF model/pipeline support, to re-measure
  after enabling operators, after a transformers/diffusers version change, or
  to produce a three-route (vendor / boxing / flaggems) UT report. The skill
  runs end to end without user interaction: pin the environment, inject the
  flagos device, sweep every architecture or pipeline in an isolated
  subprocess, classify each failure with a named root cause, deduplicate by
  cause fingerprint, and write a report. The only steps that are NOT automatic
  are GitHub writes (issues/PRs), which stay behind an explicit authorization
  gate. Do not use this to infer support from a routing table or from a
  passing operator suite.
---

# transformers-test (torch_fl) — automatic HF UT runner for transformers + diffusers

## Scope

This skill measures whether real HuggingFace models and pipelines pass their
**official unit tests** on the `flagos` device. It is a hardware measurement,
not a routing audit: an operator that passes a standalone dispatch test can
still fail inside a model through a shape, dtype, stride, or call-order the
operator suite never produces.

Two official suites are covered:

| Suite | What runs | Test root |
|---|---|---|
| transformers | every test under one architecture directory | `<transformers-src>/tests/models/<module>/` |
| diffusers | every test under one pipeline directory, plus `tests/models/` | `<diffusers-src>/tests/pipelines/<name>/`, `<diffusers-src>/tests/models/` |

"Automatic" means: once invoked with a target (one model, one pipeline, or a
full sweep), the skill carries the run from environment pinning through the
final report with no intermediate questions. It never asks "should I run the
next model". The single exception is any GitHub write, which remains gated by
the authorization rules at the end of this document.

Complete these first:

- [[runtime-bringup]] — the device and allocator contract must pass.
- [[native-op-backend]] or [[cuda-compat-vendor]] — an operator path must exist.
  A platform whose every compute op reaches `cpu_fallback` will "pass" this
  survey while measuring nothing about the accelerator.
- [[test-dependencies]] — install and validate the requested Transformers test
  packages without replacing the torch build used by torch_fl. In particular,
  install `accelerate` before official tests that exercise a device context,
  `device_map`, `tp_plan`, or `torch.set_default_device`.

The unit of work is **one architecture** (transformers) or **one pipeline /
model directory** (diffusers), each in its own subprocess. A full sweep walks
every directory once; it does not enumerate Hub checkpoints or download real
weights — HF's own tests use tiny random configs and dummy weights.

## Route matrix

Every measurement runs on up to three routes. A bug's route pattern is its
first attribution signal, so keep the routes comparable: same suite, same
source tree, same pytest arguments.

| Route | Interpreter | Device | Extra environment |
|---|---|---|---|
| vendor | system python3 + vendor torch | `cuda` | none (baseline) |
| boxing | the torch_fl venv (e.g. `/opt/fl-envs/boxing/bin/python`) | `flagos` | device injection, Step 2 |
| flaggems | boxing venv | `flagos` | `FLAGOS_USE_FLAGGEMS=1 TRITON_BACKENDS_IN_TREE=1` |

For the flaggems route add pytest-xdist crash isolation so one worker fault
cannot take down the whole run:

```bash
-n1 --dist load --max-worker-restart 8
```

A failure on all three routes including vendor is an upstream (HF or vendor
torch) issue, not a torch_fl defect. A failure only on flaggems points at the
FlagGems kernel or its routing. A failure on boxing + flaggems but not vendor
is the torch_fl integration layer.

## Step 1 — pin the environment and record it (automatic)

Before running anything, capture the versions that decide how a result must be
read. A failure carries no meaning without them.

```python
import torch, transformers, torch_fl
print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("device_count", torch.flagos.device_count())
try:
    import diffusers; print("diffusers", diffusers.__version__)
except ImportError:
    pass
try:
    import flag_gems; print("flag_gems", flag_gems.__version__)
except ImportError:
    pass
```

```bash
git rev-parse --short HEAD   # torch_fl commit
```

Also record: chip model, driver, vendor SDK revision, the venv path, and the
source-tree paths and versions for transformers and diffusers. Every table in
the final report cites this block.

Missing **test-only** Python dependencies (sentencepiece, protobuf, librosa,
…) may be pip-installed into the test venv when they are leaf test
dependencies. Never let a dependency resolver replace or upgrade `torch` —
that silently changes the ABI the extension was built against and invalidates
the run. Pin with `--no-deps` or an exact version when in doubt.

## Step 2 — inject `flagos` as the HF test device (automatic)

Both libraries support a third-party device through a device-spec module, and
both accept a plain device-name environment variable once `torch.device(
"flagos")` constructs — which requires `torch_fl` to be imported **before** the
library's testing utils validate the variable.

### transformers

Two working mechanisms:

1. **Device spec** (the upstream contract, used by
   `tests/manual/transformers_hf_tests.py`):

   ```bash
   export TRANSFORMERS_TEST_DEVICE_SPEC=tests/manual/hf_device_spec.py
   ```

   The spec imports `torch_fl`, sets `DEVICE_NAME = "flagos"`, and supplies
   `MANUAL_SEED_FN` / `EMPTY_CACHE_FN` / `DEVICE_COUNT_FN`. Newer transformers
   releases validate `TRANSFORMERS_TEST_DEVICE` with `torch.device()` before
   importing the spec, so prefer the spec over the bare variable.

2. **Bootstrap plugin + device variable** (used for direct pytest runs):

   ```bash
   PYTEST_PLUGINS=flagos_pytest_bootstrap TRANSFORMERS_TEST_DEVICE=flagos \
     python /root/bench/run_hf_ut.py <pytest args>
   ```

   `flagos_pytest_bootstrap` is a site-packages plugin that just does
   `import torch_fl`; `PYTEST_PLUGINS` loads it in every pytest process,
   including xdist workers, so the device name exists before validation.
   `run_hf_ut.py` is the equivalent for the main process: it imports
   `torch_fl`, then calls `pytest.main`.

Note the gating split this produces, and do not try to defeat it:
`require_torch_accelerator` passes for `flagos` (non-CPU, non-None), while
`require_torch_gpu` skips because it compares against `cuda` literally. Cases
skipped by `require_torch_gpu` are CUDA-specific and are not coverage gaps.

### diffusers

diffusers reads `DIFFUSERS_TEST_DEVICE` and `DIFFUSERS_TEST_DEVICE_SPEC` in
`diffusers/utils/testing_utils.py`. Two differences from transformers:

- the spec is a **file path**, not a module name (diffusers strips the `.py`
  and imports the remainder);
- the hook table is larger: in addition to `MANUAL_SEED_FN`,
  `EMPTY_CACHE_FN`, `DEVICE_COUNT_FN`, it merges `SUPPORTS_TRAINING`,
  `RESET_PEAK_MEMORY_STATS_FN`, `RESET_MAX_MEMORY_ALLOCATED_FN`, and
  `MAX_MEMORY_ALLOCATED_FN` when the spec defines them (each has a default, so
  an omitted hook falls back instead of raising).

```bash
PYTHONPATH=/root/hf-diffusers/src \
PYTEST_PLUGINS=flagos_pytest_bootstrap \
DIFFUSERS_TEST_DEVICE=flagos HF_HUB_OFFLINE=1 \
  python -m pytest tests/pipelines/audioldm2 -q
```

Run from the diffusers repository root so its `tests` package resolves.
Import diffusers through `PYTHONPATH=<src>` rather than reinstalling: the test
venv's transformers pins must stay exactly where the route matrix put them.

If `DIFFUSERS_TEST_DEVICE` and the spec's `DEVICE_NAME` disagree, diffusers
raises — set one or keep them equal.

## Step 3 — run the suites, one directory per subprocess (automatic)

### transformers

Use the in-repo official runner; it resolves a version-matched source tree,
injects the spec, isolates each architecture in a subprocess with a timeout,
and writes incremental JSON:

```bash
python tests/manual/transformers_hf_tests.py --model qwen3
python tests/manual/transformers_hf_tests.py --model qwen3 --out /tmp/qwen3.json
python tests/manual/transformers_hf_tests.py --all --out /tmp/transformers-all.json
python tests/manual/transformers_hf_tests.py --list-models
```

`--all` visits each test **directory** once, not each registry key: many keys
share a directory (`blip_text_model` and `blip_vision_model` both map to
`tests/models/blip`), and sweeping keys would count one defect several times.
The aggregate JSON is rewritten after every architecture, so an interrupted
sweep resumes with its measurements intact, and `attempted` is reported apart
from `completed` so a coverage rate never counts architectures that never ran.

Some official tests download tiny fixtures from the Hugging Face Hub. If
`huggingface.co` is unreachable, do not classify the resulting retries or
network errors as backend failures. First check the local Hub cache and rerun
with `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` when all fixtures are present.
Otherwise load the configured proxy and retry the same command; if the origin
still cannot be reached, use the Hub mirror:

```bash
source ../proxy.sh
# Retry the official runner with HF_HUB_OFFLINE and TRANSFORMERS_OFFLINE unset.
TRANSFORMERS_TEST_DEVICE_SPEC=tests/manual/hf_device_spec.py \
python tests/manual/transformers_hf_tests.py --model <model> --out /tmp/<model>.json

export HF_ENDPOINT=https://hf-mirror.com
TRANSFORMERS_TEST_DEVICE_SPEC=tests/manual/hf_device_spec.py \
python tests/manual/transformers_hf_tests.py --model <model> --out /tmp/<model>-mirror.json
```

Use the proxy before `HF_ENDPOINT`; unset `HF_HUB_OFFLINE` and
`TRANSFORMERS_OFFLINE` for online retries. `HF_ENDPOINT` is inherited by the
runner's isolated pytest process and affects Hub fixture downloads only. It does
not change the GitHub download used to populate the version-matched Transformers
source cache. For that source archive, use `--source-dir`, prepare the cache on
a machine with access, or use `--offline` once the cache is complete. Record the
endpoint mode used by the run, without exposing proxy credentials.

For a targeted re-run of specific nodes (e.g. only `test_training`), run
pytest directly through the bootstrap plugin as in Step 2, keeping the same
source tree.

### diffusers

There is no in-tree diffusers runner yet; drive pytest directly. Enumerate the
work items as directories and run each in its own subprocess with a timeout:

```bash
for d in /root/hf-diffusers/tests/pipelines/*/; do
  name=$(basename "$d")
  run_one "$name"   # the Step-2 command, target tests/pipelines/$name
done
```

Cover `tests/models/` (unets, transformers, vae, …) the same way. Do not
fabricate tests for a pipeline that has no upstream test directory — record it
as having no official UT (upstream may have deprecated it).

### Isolation rules (both suites)

- One directory per subprocess, with a timeout, always. Accelerator faults are
  not contained: one illegal memory access poisons the device context, after
  which every later test in the same process fails with the same symptom.
  This repository has measured a single fault turning one real failure into
  roughly eighty collateral ones.
- Treat a poisoned run as **one finding for the whole directory**, never as a
  list of per-test failures:

  ```text
  illegal memory access | device-side assert | unspecified launch failure
  misaligned address | vmfault | acceleratorerror
  ```

- Write results incrementally so an interrupted sweep resumes, and keep raw
  stdout/stderr per directory for auditing.
- If the requested model/pipeline does not exist in the pinned version, record
  `NOT_IN_VERSION`. That is neither a pass nor a failure.

## Step 4 — classify every result (automatic)

Reduce raw pytest output to one status per test node, then one verdict per
directory. A node reports up to three times (setup/call/teardown); the worst
outcome wins, and a setup/teardown failure is an `ERROR`, not a `FAIL` — the
test body never ran.

| Status | Meaning |
|---|---|
| `PASS` / `FAIL` / `ERROR` | ran, and the assertion/harness outcome |
| `XFAIL` / `XPASS` | expected-failure bookkeeping, never hides a FAIL |
| `SKIP_CUDA_ONLY` | skip reason matches CUDA/ROCm/multi-GPU/flash-attn gating — not a coverage gap |
| `SKIP_OTHER` | any other skip |
| `ENVIRONMENT_ERROR` | `ModuleNotFoundError`, missing optional library — the assertion never ran |
| `COLLECT_ERROR` | module failed at import/collection |
| `NOT_IN_VERSION` / `NO_TESTS_RUN` / `TIMEOUT` / `CRASH` | run-level verdicts |

Directory verdict precedence: `NOT_IN_VERSION` → `TIMEOUT` → `CRASH` →
`COLLECT_ERROR` → `FAIL` → `ENVIRONMENT_ERROR`/`NO_TESTS_RUN` → `PASS`.

`SKIP_CUDA_ONLY`, `ENVIRONMENT_ERROR`, `NOT_IN_VERSION`, and genuine upstream
skips are excluded from both the failure count and the pass-rate denominator.
Do not inflate a platform's rate by quietly keeping them in either.

## Step 5 — classify the failure and name the root cause (automatic)

Every `FAIL`/`ERROR` lands in exactly one cause class:

| Class | Signal | Labels |
|---|---|---|
| `OP_UNSUPPORTED` | `NOT_SUPPORTED`, `backend not registered`, `could not run 'aten::…'` | `enhancement`, `ai-generated` |
| `FEATURE_UNSUPPORTED` | non-operator runtime or feature gap (device whitelist, test-harness device gate, missing spec hook) | `enhancement`, `ai-generated` |
| `PRECISION` | ran, but disagrees with the CPU baseline | `bug`, `ai-generated` |
| `CRASH` | segfault, poison, or timeout | `bug`, `ai-generated` |

For `OP_UNSUPPORTED` and `PRECISION`, re-run the failing test under
`TorchDispatchMode` and capture the aten calls, then name the specific
operator in the finding. `tests/manual/op_called_summary.py` is the existing
collection pattern. Without this attribution the report only says a test
failed, which a maintainer cannot act on.

Before naming torch_fl as the cause, check the **vendor route**: if the vendor
baseline fails the same node the same way, the cause is upstream (HF test
code, HF modeling code, or vendor torch), and the finding is filed as
upstream-attributed, not as a torch_fl defect. A transformers-suite example of
this discipline: a whole `XLMRobertaForMultipleChoice` failure family traced
to `add_pooling_layer=False` vs `outputs[1]` in upstream modeling code — it
failed identically on CPU, vendor, boxing, and flaggems.

diffusers-specific cause notes:

- diffusers dummy-model tests build tiny random pipelines; a failure in
  `setUp`/weight download is an environment issue, not a pipeline result.
- Test-code device gates (`device_map` whitelists, hardcoded `cuda`, RNG
  factory device validation) are `FEATURE_UNSUPPORTED` against the **test or
  upstream library**, not model defects. Say so explicitly.
- Pipeline deprecated upstream with no test directory: record as "no official
  UT", never invent a test.

## Step 6 — judge precision against CPU, same dtype (automatic)

The baseline is the same model/pipeline, same seed, same dtype on CPU. A CUDA
baseline is not usable: most vendor boxes have no NVIDIA device, and the whole
point of the vendor route is a non-NVIDIA comparison.

Validate the CPU side first. If CPU itself errors, the case is `UNTESTED` —
not a device failure. Then compare with dtype-scaled tolerances:

| dtype | rtol | atol |
|---|---|---|
| float32 | 1e-5 | 1e-5 |
| float16 | 1e-3 | 1e-3 |
| bfloat16 | 1.6e-2 | 1e-2 |

For diffusers pipelines compare the same output slices the upstream test
compares (usually a small `[0, :4]`-style slice of the output tensor), not the
full output.

Keep `NaN`/`Inf` separate from magnitude disagreement and rank it higher. A
NaN is always a defect; a tolerance miss may be legitimate accumulation-order
variance. Reporting both as one class hides the real bug.

Watch for **harness tolerance fallbacks** before filing a precision finding:
HF's per-device tolerance tables may lack a `flagos` entry and fall back to a
stricter-than-fp16-ulp default, while `cuda` passes on its own entry. That is
a test-side tolerance gap (`FEATURE_UNSUPPORTED` in the harness), not a kernel
precision bug — verify against the vendor route and the actual max abs diff
first.

## Step 7 — deduplicate before filing anything (automatic)

The same root cause reappears across models, pipelines, and platforms. Filing
per occurrence buries the signal under duplicates.

Two fingerprints are in play — do not confuse them:

- the **occurrence fingerprint** that `transformers_hf_tests.py` writes into
  each test record: it includes model, device, and node ID, so it identifies
  one test result and is deliberately unsuitable for dedup;
- the **cause fingerprint** below, which drops model and node ID so the same
  defect reached from ten directories collapses to one value.

```python
import hashlib, re

def normalize(text: str) -> str:
    t = re.sub(r"0x[0-9a-fA-F]+", "0xADDR", text)
    t = re.sub(r"/tmp/[^\s'\"]+", "/tmp/PATH", t)
    t = re.sub(r"(/[^\s'\"]*)?/(site-packages|torch_fl|tests)/", r"/PATH/\2/", t)
    t = re.sub(r"\b\d+\.\d+s\b", "TIMEs", t)
    t = re.sub(r"\[[\d,\s]+\]", "[SHAPE]", t)
    # keep diagnostic codes; collapse every other literal number
    t = re.sub(r"(?<!err )(?<!code )(?<!errno )\b\d+\b", "N", t)
    return re.sub(r"\s+", " ", t).strip()[-200:]

def cause_fingerprint(failure_class, component, subject, mechanism):
    payload = "|".join(
        [failure_class, component, subject, normalize(mechanism)]
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:12]
```

`subject` is `aten::<op>.<overload>` when an operator was named, otherwise the
feature or module that failed (`sdpa`, `device_map`, `torch.compile`).
`component` identifies the implementation responsible: `musa:SubTensorKernelMusa`,
`ascend:generated-add`, `flaggems:cumsum`, `upstream:xlm-roberta-mc-head`, or
`shared:privateuse1-registration`. `mechanism` is the shortest verified
statement that distinguishes the defect — the vendor error's stable final line
or a normalized assertion. Do not hash a traceback: call stacks change without
changing the cause.

Two vendor backends both reporting an operator unsupported are **separate**
gaps (different components). A defect in shared torch_fl code uses a
`shared:*` component and has one fingerprint across all affected platforms.

Then, for each fingerprint, search before writing anything — issue bodies and
comments directly, since the search index lags writes:

```bash
FP=cce6ae545772
gh api repos/flagos-ai/Torch-FL/issues --paginate -X GET -f state=all \
  --jq ".[] | select(.pull_request | not) | select(.body // \"\" | contains(\"$FP\")) | \"issue #\(.number) \(.state) \(.title)\""
gh api repos/flagos-ai/Torch-FL/issues/comments --paginate -X GET \
  --jq ".[] | select(.body // \"\" | contains(\"$FP\")) | \"comment \(.html_url)\""
```

Also search by subject and component as a semantic fallback for pre-fingerprint
issues, and read every candidate before deciding. Outcomes:

- open issue match: duplicate; at most a new-evidence comment (gated below);
- closed issue match: report the match and its closing disposition; do not
  reopen without explicit authorization;
- no match: eligible for a new issue under the gates below.

Every issue or comment the skill writes must carry the fingerprint on its own
line:

```text
Fingerprint: `cce6ae545772`
```

## Step 8 — write the report (automatic)

The automatic run ends with a written report; it needs no authorization.

Per suite, produce a per-directory table with one row per route:

```text
| directory | route | passed | failed | error | skip(cuda) | skip(other) | env | verdict |
```

plus: the Step 1 environment block, the failure inventory grouped by cause
fingerprint (class, subject, component, mechanism, affected directories,
issue/PR number if one exists, **unfixed** marker otherwise), and the evidence
file paths. Route-pattern attribution (Step 5) is stated per cause.

Report only what was actually measured. Skipped, timed-out, and
absent-from-version directories are their own categories, never silent
omissions, and a previous cohort's numbers are never carried forward as
re-measured. Keep raw JSON and logs outside the repository; a sweep produces
roughly a megabyte per directory and crashing runs leave vendor core dumps
(`core_*.mudmp` on MUSA) in the working tree — move evidence out and delete
dumps before any commit. All committed docs are in English per `CLAUDE.md`;
Chinese working copies live at the repository root as `*_zh.md`, never under
`docs/`.

## Step 9 — tracker writes stay gated (the only non-automatic step)

Running suites, classifying, deduplicating, and writing the report are all
automatic. Creating issues, commenting, reopening, and filing PRs are not.
Each is a separate outward-facing action requiring an explicit user request in
the current session, for the named finding only:

- "file/open/create an issue" authorizes that one new issue — not comments,
  not reopens;
- "comment on issue #N" authorizes one comment on that issue;
- "run the survey", "find problems", and "summarize" authorize **no** tracker
  writes, and authorization for one finding never extends to the rest.

Before any tracker write, all four evidence conditions must hold:

1. **A pre-existing baseline exists for this chip** — an earlier dated
   measurement with cause fingerprints to diff against. The first sweep on a
   chip IS the baseline: record it, perform no tracker writes.
2. **The finding survived isolation** — reproduces in its own process with a
   passing CPU same-dtype baseline (and a vendor-route comparison done); for a
   poisoned run, confirmed as the first fault, not collateral.
3. **A standalone reproducer runs** — prefer under 30 lines importing only
   `torch` and `torch_fl`; if reduction is impossible, explain why and keep
   the exact isolated single-test command.
4. **The count is sane** — more than five apparently new causes from one sweep
   triggers another classification and dedup pass first.

When filing, follow `.github/AI_AGENT_GUIDE.md`,
`.github/CLAUDE_CODE_GUIDE.md`, and `.github/ISSUE_TEMPLATE/ai_agent_issue.md`
section by section; write the body to a file and use `gh issue create
--body-file`; everything in English; include the fingerprint line, the
verbatim error, the environment block, and a root cause that names
`file:line`. Use only labels that exist:

```bash
gh api repos/flagos-ai/Torch-FL/labels --paginate --jq '.[].name'
```

If authorization is missing or ambiguous, prepare the ready-to-file text and
stop. Do not ask repeatedly during a sweep; report the candidates at the end.

## Final checklist

- transformers and diffusers versions, torch, flag_gems, torch_fl commit, and
  chip/driver are recorded with every result;
- routes are comparable: same source tree and pytest args per directory;
- device injection used the correct contract per library (spec vs bootstrap +
  device variable) and was never "fixed" by patching upstream tests;
- one directory per subprocess, with timeout; poisoned runs are one finding
  per directory;
- CPU baselines validated before any device comparison; vendor route checked
  before attributing a cause to torch_fl;
- every `OP_UNSUPPORTED`/`PRECISION` finding names an operator or a
  `file:line` mechanism;
- `SKIP_CUDA_ONLY`, `ENVIRONMENT_ERROR`, and `NOT_IN_VERSION` are excluded
  from failure counts and denominators;
- every finding row carries its cause fingerprint, and fingerprints were
  searched in issue bodies and comments before any filing decision;
- no tracker write without both Step 9 gates and explicit per-action
  authorization;
- raw JSON, logs, and vendor core dumps are not in the commit;
- committed docs are English; Chinese copies are `*_zh.md` at repo root;
- `ruff check .` and `ruff format --check .` pass before any PR.

Coverage may be claimed only for the directories actually measured on that
chip. A passing operator suite, a populated routing table, and another
platform's rate are all not evidence.

## Related

[[runtime-bringup]] · [[native-op-backend]] · [[cuda-compat-vendor]] ·
[[test-dependencies]] · [[flaggems-integration]] · [[pre-pr-checks]]
