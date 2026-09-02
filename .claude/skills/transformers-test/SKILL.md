---
name: transformers-test
description: >
  Measure torch_fl transformers coverage on an accelerator one model at a time,
  and turn each new failure into an operator, feature, precision, or crash
  finding with a named root cause. Use this to survey a platform's HuggingFace
  model support, to check a model after enabling operators, or to re-measure
  after a transformers version change. Covers: per-model probing without real
  weights, the HF custom-device injection contract, CPU same-dtype precision
  baselines, aten attribution through TorchDispatchMode, root-cause fingerprint
  dedup, and a baseline-first gate for explicitly authorized tracker writes.
  Do not use this to infer model support from a routing table or from a passing
  operator suite.
---

# transformers-test (torch_fl)

## Scope and prerequisites

This skill measures whether real transformers models run on the `flagos`
device. It is a hardware measurement, not a routing audit: an operator that
passes a standalone dispatch test can still fail inside a model through a shape,
dtype, stride, or call-order the operator suite never produces.

Complete these first:

- [[runtime-bringup]] — the device and allocator contract must pass.
- [[native-op-backend]] or [[cuda-compat-vendor]] — an operator path must exist.
  A platform whose every compute op reaches `cpu_fallback` will "pass" this
  survey while measuring nothing about the accelerator.
- [[test-dependencies]] — install and validate the requested Transformers test
  packages without replacing the torch build used by torch_fl. In particular,
  install `accelerate` before official tests that exercise a device context,
  `device_map`, `tp_plan`, or `torch.set_default_device`.

The unit of work is **one architecture**. `transformers-test <model>` runs every
official test under that architecture's `tests/models/<module>/` directory. Use
`transformers-test --all` only when an explicit full sweep is wanted: it walks all
architecture keys exposed by the pinned Transformers registry, one isolated
process per architecture. This means model architecture tests, not every Hub
checkpoint or weight variant.

## Step 1 — pin the environment and record it

Two versions decide how a result must be read: the `transformers` version and
the torch_fl commit. A failure carries no meaning without both.

Default to the latest `transformers`. Pin an older line only to reproduce a
known-good state or to check a regression:

```bash
python tests/manual/transformers_model_probe.py --model qwen3
python tests/manual/transformers_model_probe.py --model qwen3 --transformers-version 4.50.2
```

Install each requested `transformers` into its own cached virtual environment
rather than the environment that holds torch_fl. `transformers` pulls its own
torch constraint, and letting it resolve inside the torch_fl environment can
replace the torch the extension was built against — which produces symbol
errors or silent ABI corruption, not a coverage result.

Record and keep with the run:

```bash
python - <<'PY'
import torch, transformers, torch_fl
print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("device_count", torch.flagos.device_count())
PY

git rev-parse --short HEAD
```

For a vendor box, also record the chip model, driver, SDK revision, and any
required environment. A model probe that ran against the wrong runtime is an
environment failure, not a coverage data point.

## Step 2 — inject `flagos` as the HF test device

`transformers` supports a third-party device through a device-spec module.
The spec imports `torch_fl`, registers the PrivateUse1 name, and supplies the
backend hooks that HuggingFace needs:

```bash
export TRANSFORMERS_TEST_DEVICE_SPEC=tests/manual/hf_device_spec.py
```

Do not set `TRANSFORMERS_TEST_DEVICE=flagos` for Transformers 5.16.1: that
release validates the variable with `torch.device()` before importing the spec,
so a custom PrivateUse1 name is not accepted at that point. The spec's
`DEVICE_NAME` becomes the effective test device after it is loaded.

For built-in devices, `TRANSFORMERS_TEST_DEVICE` remains available as an
alternative to a spec file.


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

The `transformers` wheel ships no `tests/` directory. The synthetic probe
(`tests/manual/transformers_model_probe.py`) does not need an HF source checkout.
To run HuggingFace's own assertions, use the official runner:

```bash
python tests/manual/transformers_hf_tests.py --model qwen3
python tests/manual/transformers_hf_tests.py --all --out /tmp/transformers-all.json
```

It obtains a cached source tree for the exact installed version, verifies the
version declared by `src/transformers/__init__.py`, and runs only
`tests/models/<module>/` for that architecture. Set `--source-dir` to use a
prepared tree, or use `--offline` to require an existing versioned cache. A
version-mismatched source tree produces failures that belong to HF, not to
torch_fl. The runner injects `flagos` through
`TRANSFORMERS_TEST_DEVICE_SPEC=tests/manual/hf_device_spec.py`.

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

`--all` visits each test directory once, not each registry key: many keys share
a directory (`blip_text_model` and `blip_vision_model` both map to
`tests/models/blip`), and sweeping keys would count one defect three times. It
rewrites the aggregate JSON after every architecture, so an interrupted sweep
still holds its measurements, and it reports `attempted` apart from `completed`
so a coverage rate never uses architectures that never ran as its denominator.

The actual device contract used here is `TRANSFORMERS_TEST_DEVICE_SPEC` and
the three hooks in the spec module. No third-party test switch or other
pass-oriented environment override is added.

The official runner preserves per-test JSON evidence and never writes to
GitHub. The skill may turn a verified finding into a tracker action only through
the explicitly authorized workflow in Steps 7 and 8; normal architecture tests
and `--all` sweeps remain report-only.

## Step 3 — probe one model in layers

Build the model from its `CONFIG_MAPPING` entry reduced to tiny dimensions and
randomly initialized. Real pretrained weights are not needed to find missing
operators, and requiring them makes the survey depend on downloads and device
memory.

Resolve the architecture key through `model_type_to_module_name()`. Keys and
module directories disagree for a substantial minority of architectures
(`blip-2` → `blip_2`, `audio-spectrogram-transformer` →
`audio_spectrogram_transformer`); comparing the raw key against directory names
misreports those models as absent.

Run the layers in order and stop the model at the first failing layer. A model
whose `.to(device)` fails produces meaningless forward and backward results:

| Layer | What runs | What a failure means |
|---|---|---|
| L0 | build tiny model, `.to("flagos")` | allocator, copy, or dtype transfer |
| L1 | forward, compare to CPU same dtype | missing operator or precision |
| L2 | backward plus one optimizer step | missing backward operator |
| L3 | short `generate()` | KV-cache, sampling, or control flow |

If the requested model does not exist in the pinned `transformers`, record
`NOT_IN_VERSION`. That is neither a pass nor a failure, and counting it either
way corrupts the platform's rate.

## Step 4 — isolate every model in a subprocess

Run each model in its own subprocess with a timeout, even in single-model mode.
Accelerator faults are not contained: one illegal memory access poisons the
device context, after which every later operation fails with the same symptom.
This has been measured in this repository — a single fault turned roughly one
real failure into roughly eighty collateral ones.

Treat a poisoned run as one finding for the whole model, never as a list of
per-layer failures:

```text
illegal memory access | device-side assert | unspecified launch failure
misaligned address | vmfault | acceleratorerror
```

Write results incrementally so an interrupted sweep resumes instead of
restarting, and keep raw stdout/stderr per model for auditing.

## Step 5 — classify the failure and name the root cause

Every failure must land in exactly one class. The class selects the issue label
and decides whether the finding is actionable:

| Class | Signal | Labels |
|---|---|---|
| `OP_UNSUPPORTED` | `NOT_SUPPORTED`, `backend not registered`, `could not run 'aten::…'` | `enhancement`, `ai-generated` |
| `FEATURE_UNSUPPORTED` | non-operator runtime or feature gap | `enhancement`, `ai-generated` |
| `PRECISION` | ran, but disagrees with the CPU baseline | `bug`, `ai-generated` |
| `CRASH` | segfault, poison, or timeout | `bug`, `ai-generated` |

Use only labels that exist in this repository. As of this writing the tracker
has `bug`, `enhancement`, `documentation`, `ai-generated`, `duplicate`,
`P0`/`P1`/`P2`, and the triage set; there is no `operator` label, so an
unsupported operator is filed as `enhancement` with `aten::<op>` in the title.
Confirm the current set before filing rather than trusting this list:

```bash
gh api repos/flagos-ai/Torch-FL/labels --paginate --jq '.[].name'
```

For `OP_UNSUPPORTED` and `PRECISION`, re-run the failing layer under
`TorchDispatchMode` and capture the aten calls, then put the specific operator
in the finding. Without this attribution the report only says a model failed,
which a maintainer cannot act on. `tests/manual/op_called_summary.py` is the
existing pattern for this collection.

## Step 6 — judge precision against CPU, same dtype

The baseline is the same model, same seed, same dtype on CPU. A CUDA baseline is
not usable here: most vendor boxes have no NVIDIA device, so it would make the
survey unrunnable exactly where it is needed.

Validate the CPU side first. If CPU itself errors, the case is `UNTESTED` — not
a device failure. Then compare with dtype-scaled tolerances:

| dtype | rtol | atol |
|---|---|---|
| float32 | 1e-5 | 1e-5 |
| float16 | 1e-3 | 1e-3 |
| bfloat16 | 1.6e-2 | 1e-2 |

Keep `NaN`/`Inf` separate from magnitude disagreement and rank it higher. A NaN
is always a defect; a tolerance miss may be legitimate accumulation-order
variance. Reporting both as one class hides the real bug.

## Step 7 — deduplicate before filing anything

The same root cause reappears across models and platforms. Filing per model and
per platform buries the signal under duplicates.

Two different fingerprints are in play, and confusing them is the usual mistake:

- the **occurrence fingerprint** that `tests/manual/transformers_hf_tests.py`
  writes into each test record. It includes the model, the device, and the test
  node ID, so it identifies one test result and is deliberately unsuitable for
  dedup;
- the **cause fingerprint** below, which drops model and node ID so that the same
  defect reached from ten models collapses to one value.

Compute the cause fingerprint from the finding you established in Step 6, not
from raw pytest output:

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
feature or module that failed (`sdpa`, `torch.compile`, `flash_attention_2`).
`component` identifies the implementation responsible for the cause: for
example `musa:SubTensorKernelMusa`, `ascend:generated-add`, or
`shared:privateuse1-registration`. `mechanism` is the shortest verified statement
that distinguishes the defect, preferably the vendor error's stable final line
or a normalized assertion. Do not hash an entire traceback: call stacks change
without changing the cause.

This makes the value stable across models, tests, transformers releases, and
chips that execute the same faulty implementation. It does **not** merge two
vendor backends merely because they both report that an operator is unsupported;
those are separate implementation gaps. Conversely, a defect in shared torch_fl
code uses a `shared:*` component and therefore has one fingerprint across all
affected platforms.

Then, for each fingerprint, search before writing anything. Search issue bodies
and comments directly rather than trusting only the search index, which can lag
behind new writes by minutes:

```bash
FP=cce6ae545772

gh api repos/flagos-ai/Torch-FL/issues --paginate -X GET -f state=all \
  --jq ".[] | select(.pull_request | not) | select(.body // \"\" | contains(\"$FP\")) | \"issue #\(.number) \(.state) \(.title)\""

gh api repos/flagos-ai/Torch-FL/issues/comments --paginate -X GET \
  --jq ".[] | select(.body // \"\" | contains(\"$FP\")) | \"comment \(.html_url)\""
```

Also search by the attributed subject and component to catch issues created before
this fingerprint convention. That semantic search is a fallback, not proof that
two causes are equal; read every candidate before deciding:

```bash
SUBJECT='aten::rsub.Scalar'
gh api search/issues -X GET \
  -f q="$SUBJECT repo:flagos-ai/Torch-FL is:issue" \
  --jq '.items[] | "#\(.number) \(.state) \(.title)"'
```

- a match in an open issue: it is a duplicate, not a new issue; prepare a comment
  carrying this platform's evidence and write it only if Step 8's authorization
  gate covers comments;
- a match in a closed issue: report the match and its closing disposition. Do not
  reopen it or comment unless the user explicitly authorizes that specific
  action; a maintainer may have closed it as intentional or unsupported;
- no match: it is eligible for a new issue under Step 8.

For dedup to work, every issue and every comment the skill writes must carry the
value in exactly this form, on its own line:

```text
Fingerprint: `cce6ae545772`
```

State the chip and the model in the title, and the `transformers` version, so
the title alone identifies the measurement:

```text
[AI][MUSA MTT S5000] qwen3: aten::index_copy_ not supported (transformers 5.16.1)
[AI][Ascend 910] gemma3: fp16 logits mismatch vs CPU, max abs diff 3.2e-2 (transformers 5.16.1)
```

Before any tracker write, read `.github/AI_AGENT_GUIDE.md`,
`.github/CLAUDE_CODE_GUIDE.md`, and
`.github/ISSUE_TEMPLATE/ai_agent_issue.md`. Include a self-contained reproducer
and the environment block, and write everything in English per `CLAUDE.md`.

## Step 8 — baseline first, then authorize each tracker action

A first sweep on a new platform can produce hundreds of failures at once.
Writing all of them to the tracker would hide the signal under a flood. Neither
harness writes to GitHub or promotes a baseline; the agent performs the steps
below deliberately from the report JSON.

### Evidence gate

All four conditions must hold before any tracker write is considered. If one
fails, stop and report instead:

1. **A pre-existing baseline exists for this chip.** Before the current run,
   `docs/reference/hf-coverage.md` already contained a measurement for this chip
   with an earlier run date and cause fingerprints. The first sweep is the
   baseline: record it, perform no tracker writes, and say so. Recording the
   current sweep immediately before filing does not turn it into an earlier
   baseline.
2. **The finding survived isolation.** It reproduces in its own process with a
   passing CPU same-dtype baseline and, for a poisoned run, is confirmed as the
   first fault rather than collateral damage.
3. **A standalone reproducer runs.** Prefer fewer than 30 lines importing only
   `torch` and `torch_fl`, with no `transformers` or pytest. If reduction is not
   possible, explain why and retain the exact isolated single-test command.
4. **The count is sane.** More than five apparently new causes from one sweep
   triggers another classification and dedup pass; do not publish a batch whose
   causes may have collapsed incorrectly.

After that gate, compare the cause fingerprint with both the pre-existing
baseline and the tracker:

- present in the baseline: a known cause, never a new issue;
- absent from the baseline but present in an open issue: a duplicate, optionally
  eligible for a new-evidence comment;
- absent from the baseline but present in a closed issue: report the issue and
  its closing disposition; optionally eligible for a comment or reopen request;
- absent from both: a regression or newly discovered cause eligible for a new
  issue.

### Authorization gate

Every GitHub write is a separate outward-facing action. Perform only the action
the user explicitly requests in the current session, and only for the named
finding or issue:

- "file/open/create an issue" authorizes creating the specified new issue; it
  does not authorize commenting on or reopening an existing issue;
- "comment on issue #N" authorizes one comment on that issue; it does not
  authorize reopening it;
- "reopen issue #N" authorizes changing that issue's state; add a comment only
  if commenting was also requested;
- "run the survey", "find problems", and "summarize" authorize no tracker
  writes, and authorization for one finding never extends to the rest.

Read-only duplicate searches need no write authorization. If authorization is
missing or ambiguous, prepare the issue body or comment without publishing it.
Do not repeatedly ask for authorization during a survey; report the candidates
at the end and let the user decide.

### Creating a new issue

Write the body to a file — never pass a multi-line body as a shell argument,
where a backtick or `$` in a traceback could be interpreted:

```bash
gh issue create \
  --repo flagos-ai/Torch-FL \
  --title "[AI][MUSA MTT S5000] bert: aten::rsub.Scalar illegal memory access (transformers 5.16.1)" \
  --body-file /tmp/issue-cce6ae545772.md \
  --label ai-generated --label bug
```

Map the finding onto `.github/ISSUE_TEMPLATE/ai_agent_issue.md` section by
section. The template's rejection criteria decide whether the issue is useful;
in particular:

- **Root Cause Analysis** — the template forbids "unknown". Name the mechanism
  and cite `file:line` of the code responsible, e.g. "`SubTensorKernelMusa`
  moves only `other` to the device (`csrc/aten/backends/musa/generated/musa_kernels.cc:760`)
  and allocates `out` from `self.options()` (line 764), so a CPU wrapped scalar
  in the `self` slot reaches muDNN as a host pointer." If the mechanism is not
  established, the finding is not ready to publish.
- **Expected vs Actual** — include the verbatim error text and relevant
  traceback, not a paraphrase.
- **Environment** — include the pinned Step 1 block: chip, driver, vendor
  library, Python, `torch`, `transformers`, and the torch_fl commit SHA.
- **Proposed Solution and Verification Plan** — give a concrete implementation
  direction and the unit, integration, and hardware regression checks needed.

Add the fingerprint line from Step 7 and state the model and test node ID that
led to the finding. Keep the raw JSON out of the issue; quote only the relevant
evidence. Assign an issue owner only if the user names one or repository policy
requires one; an issue assignee is not a PR reviewer.

Nothing here bypasses the report-only default. When in doubt, produce the
ready-to-file text and wait.

## Step 9 — record measured evidence

Update `docs/reference/hf-coverage.md` with the chip, run date, `transformers`
version, torch_fl commit, models attempted, and the per-class counts. Every
finding gets a row carrying its Step 7 cause fingerprint, its class, its subject
operator or feature, and its issue number if one was filed — that table is what
the next run diffs against, so a finding recorded without its fingerprint cannot
be deduplicated later.

Keep the raw JSON, outside the repository. A sweep produces roughly a megabyte
per model, and crashing runs leave vendor core dumps (`core_*.mudmp` on MUSA) in
the working tree; move the JSON to an evidence directory and delete the dumps
before committing.

Report only the models actually attempted. If a model was skipped, timed out, or
was absent from the pinned version, say so in its own category. Do not carry a
previous cohort's numbers forward as though they were re-measured, and do not
copy one chip's rate onto another.

Before opening a PR:

- the `transformers` version and torch_fl commit are recorded with every result;
- CPU baselines were validated before any device comparison;
- poisoned runs are one finding per model, not per layer;
- every `OP_UNSUPPORTED` and `PRECISION` finding names an operator;
- every finding row carries its cause fingerprint;
- `NOT_IN_VERSION`, `UNTESTED`, and CUDA-only skips are excluded from failures;
- fingerprints were searched in issue bodies and comments, with a semantic
  fallback for pre-fingerprint issues;
- duplicate issues were commented on or reopened only when that exact action was
  separately authorized;
- no tracker write occurred without both Step 8 gates passing, and the first
  sweep on a chip performed no tracker writes;
- raw JSON and vendor core dumps are not in the commit;
- unavailable hardware is marked **not revalidated**;
- `ruff check .` and `ruff format --check .` pass.

Coverage may be claimed only for the models measured on that chip. A passing
operator suite, a populated routing table, and another platform's rate are all
not evidence.

## Related

[[runtime-bringup]] · [[native-op-backend]] · [[cuda-compat-vendor]] ·
[[flaggems-integration]] · [[pre-pr-checks]]
