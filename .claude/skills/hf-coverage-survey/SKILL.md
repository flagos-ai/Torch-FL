---
name: hf-coverage-survey
description: >
  Measure torch_fl transformers coverage on an accelerator one model at a time,
  and turn each new failure into an operator, feature, precision, or crash
  finding with a named root cause. Use this to survey a platform's HuggingFace
  model support, to check a model after enabling operators, or to re-measure
  after a transformers version change. Covers: per-model probing without real
  weights, the HF custom-device injection contract, CPU same-dtype precision
  baselines, aten attribution through TorchDispatchMode, fingerprint dedup, and
  the baseline-first issue policy. Do not use this to infer model support from a
  routing table or from a passing operator suite.
---

# transformers coverage survey (torch_fl)

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

The unit of work is **one model**. A full sweep is a loop over single-model
runs, not a single long process. Prefer measuring the model you care about:
a per-model run finishes in minutes and produces an actionable finding, while a
full sweep across 300+ architectures mostly re-measures what the previous sweep
already recorded.

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
```

It obtains a cached source tree for the exact installed version, verifies the
version declared by `src/transformers/__init__.py`, and runs only
`tests/models/<module>/` for that architecture. Set `--source-dir` to use a
prepared tree, or use `--offline` to require an existing versioned cache. A
version-mismatched source tree produces failures that belong to HF, not to
torch_fl. The runner injects `flagos` through
`TRANSFORMERS_TEST_DEVICE_SPEC=tests/manual/hf_device_spec.py`.

The actual device contract used here is `TRANSFORMERS_TEST_DEVICE_SPEC` and
the three hooks in the spec module. No third-party test switch or other
pass-oriented environment override is added.

The official runner preserves per-test JSON evidence and does not create issues.
Baseline promotion, fingerprint deduplication, and GitHub issue creation remain
a separate opt-in follow-up.

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

| Class | Signal | Label |
|---|---|---|
| `OP_UNSUPPORTED` | `NOT_SUPPORTED`, `backend not registered`, `could not run 'aten::…'` | `operator` |
| `FEATURE_UNSUPPORTED` | non-operator runtime or feature gap | `enhancement` |
| `PRECISION` | ran, but disagrees with the CPU baseline | `bug` |
| `CRASH` | segfault, poison, or timeout | `bug` |

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

Fingerprint the cause, not the occurrence:

```text
sha256(failure_class + operator_or_module + normalized_error_signature)[:12]
```

Normalization must strip addresses, shapes, durations, and temporary paths.
Without it a differing pointer value creates a new issue for a known cause.

Then, for each fingerprint:

1. search existing open issues for the fingerprint;
2. if found, add a comment carrying this platform's evidence;
3. if not found, open one issue.

State the chip and the model in the title, and the `transformers` version, so
the title alone identifies the measurement:

```text
[AI][MUSA MTT S5000] qwen3: aten::index_copy_ not supported (transformers 5.16.1)
[AI][Ascend 910] gemma3: fp16 logits mismatch vs CPU, max abs diff 3.2e-2 (transformers 5.16.1)
```

Use `.github/ISSUE_TEMPLATE/ai_agent_issue.md`, include a self-contained
reproducer and the environment block, and write everything in English per
`CLAUDE.md`.

## Step 8 — baseline first, then only new failures

A first sweep on a new platform can produce hundreds of failures at once.
Filing those is not a report; it is a flood that hides every later regression.

The first run on a platform records a baseline and files nothing. Later runs
file only fingerprints absent from the baseline. Issue filing stays opt-in:

```bash
# first run on a platform: keep the JSON as the baseline, file nothing
python tests/manual/transformers_hf_tests.py --model qwen3 \
  --out results/musa/qwen3.json
```

Baseline promotion, fingerprint diffing, and issue filing are not implemented in
either harness yet. Both write per-result JSON with normalized failure
fingerprints; the reporter that compares against a baseline and writes to the
tracker is a separate change. Until it exists, filing is a manual decision made
from the JSON, not something a run performs.

Default to report-only. Opening issues writes to a shared tracker and cannot be
cleanly undone, so it must be requested explicitly rather than inferred.

## Step 9 — record measured evidence

Update `docs/reference/hf-coverage.md` with the chip, run date, `transformers`
version, torch_fl commit, models attempted, and the per-class counts. Keep the
raw JSON.

Report only the models actually attempted. If a model was skipped, timed out, or
was absent from the pinned version, say so in its own category. Do not carry a
previous cohort's numbers forward as though they were re-measured, and do not
copy one chip's rate onto another.

Before opening a PR:

- the `transformers` version and torch_fl commit are recorded with every result;
- CPU baselines were validated before any device comparison;
- poisoned runs are one finding per model, not per layer;
- every `OP_UNSUPPORTED` and `PRECISION` finding names an operator;
- `NOT_IN_VERSION`, `UNTESTED`, and CUDA-only skips are excluded from failures;
- fingerprints were searched before filing, and duplicates became comments;
- unavailable hardware is marked **not revalidated**;
- `ruff check .` and `ruff format --check .` pass.

Coverage may be claimed only for the models measured on that chip. A passing
operator suite, a populated routing table, and another platform's rate are all
not evidence.

## Related

[[runtime-bringup]] · [[native-op-backend]] · [[cuda-compat-vendor]] ·
[[flaggems-integration]] · [[pre-pr-checks]]
