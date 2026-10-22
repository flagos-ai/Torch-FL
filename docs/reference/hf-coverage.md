# Hugging Face Transformers Coverage

This document records measured Hugging Face Transformers architecture tests on
FlagOS devices. Results are hardware measurements and are not inferred from
operator routing tables.

## Baseline: MUSA MTT S5000

| Field | Value |
| --- | --- |
| Run date | 2026-09-02 |
| Chip | MUSA MTT S5000 |
| Device | `flagos` |
| Device count | 8 |
| PyTorch | 2.10.0+cpu |
| Transformers | 5.16.1 |
| torch_fl commit | `64e60dd` |
| Architecture | `qwen3` |
| Test source | Transformers 5.16.1 cached source |
| Command | `python tests/manual/transformers_hf_tests.py --model qwen3 --offline --out /tmp/qwen3.json` |
| Collected | 296 |
| Passed | 141 |
| Failed | 20 |
| CUDA-only skips | 3 |
| Other skips | 132 |
| Verdict | `CRASH` |
| Context poisoned | Yes |

The run used `TORCH_DEVICE_BACKEND_AUTOLOAD=0` to disable the unavailable
FlagCX backend extension during PyTorch import. The final runtime diagnostic
was:

```text
[flagos-musa] musaFree(0x10019600000) failed: an illegal memory access was encountered
```

Because the device context was poisoned, this baseline did not initially treat
later failures as independent findings. A corrected isolation pass subsequently
found that the resilient harness had passed both the architecture directory and
the selected nodeid to pytest, which silently reran all 297 collected tests.
After fixing that selector bug, all 20 nodeids were rerun one at a time in fresh
processes (`collected == 1` per invocation).

The corrected verification grouped the 20 occurrences into these tracked causes:

| Fingerprint | Class | Subject | Affected tests | Issue |
| --- | --- | --- | ---: | --- |
| | `CRASH` | qwen3 device context poisoning / model parallelism trigger | 1 | [#250](https://github.com/flagos-ai/Torch-FL/issues/250), [#265](https://github.com/flagos-ai/Torch-FL/issues/265) |
| | `OP_UNSUPPORTED` | mudnn softmax rejects non-contiguous input | 9 | [#262](https://github.com/flagos-ai/Torch-FL/issues/262), [#268](https://github.com/flagos-ai/Torch-FL/issues/268) |
| | `FEATURE_UNSUPPORTED` | ProcessGroupGloo rejects `flagos` tensors | 6 | [#263](https://github.com/flagos-ai/Torch-FL/issues/263) |
| | `FEATURE_UNSUPPORTED` | TorchInductor/Triton requires CUDA libraries | 2 | [#264](https://github.com/flagos-ai/Torch-FL/issues/264) |
| | `OP_UNSUPPORTED` | mudnn `TRUEDIV` with `INT64` | 2 | [#266](https://github.com/flagos-ai/Torch-FL/issues/266) |

`Affected tests` is the count this baseline measured on `64e60dd` and is not
updated here: the table's job is to say which occurrence a re-measurement is
looking at, so a count rewritten against a later tree would stop matching the run
it came from. Which rows are still live work is recorded in prose below, where a
closure can carry the commit that closed it.

The `Fingerprint` column is what deduplication keys on: `transformers_deduplicate.py`
reads it from each baseline's cause table, and `transformers_file_issues.py`
fills it in when it records a newly filed issue. The five rows above are left
blank on purpose. They were filed before the fingerprint convention existed and
inventing hashes for them would defeat the check they exist to serve, so a
re-measurement of this baseline is what fills them in. In the meantime a
fingerprint match is not the only dedup path: the subject and component are also
searched, and a semantic match is surfaced as `REVIEW_CANDIDATE` for a human to
compare rather than treated as an automatic duplicate.

Each `## Baseline:` section describes one board, and deduplication reads only the
sections belonging to the board a run measured — `transformers_deduplicate.py`
matches its `--hardware` label against the section heading and reports the
sections it skipped. This is not a formality: two vendors may register their
PrivateUse1 device under the same name, so the heading is the only place the
boards can be told apart, and an unscoped read would let a measurement on one
board be suppressed as already known by a measurement on another.

Issue #267 was closed and replaced by #268 because its first draft listed
incorrect parameterized nodeids. The initial baseline remains useful as the raw
suite measurement, but the corrected per-test isolation is the evidence used for
root-cause filing. Raw JSON remains outside the repository at
`/tmp/qwen3.json` and `/tmp/qwen3_isolated_results_v2.json`.

### What has changed since this baseline

All five causes have a fix. For one of them (#263) that is not the same thing as
all of its nodeids passing: the collective failure the row names is gone, and the
three nodeids that used to be blocked behind it now reach a separate defect in
FSDP2's stream setup instead.

- **#250, #265** (device context poisoning, and its `test_model_parallelism`
  trigger) — closed 2026-09-15, the same day PR #280 (`6b2af4b`, "stage
  cross-device copies in MudnnCopy instead of handing mudnn two devices") landed,
  with PR #282 ("order the peer device against a cross-device D2D copy") covering
  the ordering half.
- **#262, #268** (mudnn softmax rejects non-contiguous input) — PR #398
  (`6a9c101`, 2026-09-23, "materialize strided softmax operands before mudnn on
  MUSA"), which closed #262 the same day. #268 was closed three days later, on
  2026-09-26, against that fix after its borrowed nodeids were re-measured. The eight
  SDPA nodeids this row borrowed were re-measured on `main @ 48829b8` with the
  harness's device shims enabled and report `{"PASS": 8}`; the
  `RuntimeError: _softmax failed: INVALID_PARAMETER` signature the run recorded is
  raised by none of them. With the shims disabled
  (`HF_TEST_NO_DEVICE_SHIMS=1`) the same eight fail, but on the 80 % `allclose`
  check and not in the operator: `tests/test_modeling_common.py:486` falls through
  to fp32's `atol = 1e-7, rtol = 1e-4` for a device name transformers does not
  list, which is the fall-through #248 documents for GCU. That residual belongs
  with #248. Only those eight were re-run.
- **#266** (mudnn `TRUEDIV` with `INT64`) — PR #278 (`251c9c8`, 2026-09-15, "route
  MUSA integer division through mudnn and promote true division").
- **#264** (TorchInductor/Triton requires CUDA libraries) — PR #419 (`48829b8`,
  "ask torch, not the redirected probe, before aliasing cuda to flagos").
- **#263** (ProcessGroupGloo rejects `flagos` tensors) — PR #435 (2026-09-27, "answer gloo
  requests with the flagos backend and stage collectives over host memory"). gloo cannot be
  repaired from the gloo side, so the request is answered with the `flagos` backend before the
  backend config is built, and `ProcessGroupFlagOS` grew a last tier that stages
  collectives through host memory over a real gloo group
  (`docs/architecture/distributed-flagcx.md` §0.4 has the mechanism and the
  measurements). Re-measured with the same harness on this change over `05be850`:
  `{"PASS": 3, "FAIL": 3}` of the six FSDP2 nodeids, against six failures at
  baseline. The three that pass are `test_fsdp2_save_load`,
  `test_fsdp2_sharding_structure_0_untied` and `test_fsdp2_sharding_structure_1_tied`.
  The three that fail are `test_fsdp2_plan_vs_ddp_0_untied`, `test_fsdp2_plan_vs_ddp_1_tied`
  and `test_fsdp2_save_load_dcp`, and they fail *after* the collectives, in
  `_fsdp_param_group.py::FSDPCommContext.lazy_init`, with `RuntimeError: libascendcl.so
  not found. ACL runtime is required` — `torch_fl.flagos.Stream` sent every non-GCU
  platform without a CUDA runtime to the Ascend ACL stream. That is the
  `torch.flagos.Stream()` row of #451, not the collective path this row tracks. Fixed
  by PR #467 over `6b69999`: MUSA has its own stream class
  (`torch_fl/accelerator/musa/musa_stream.py`, a handle to the one stream
  `GetDefaultMusaStream` hands out, which is the queue mudnn and Triton already share),
  and the four vendor dispatches name their arms instead of ending in an `else` that
  meant Ascend, so an unlisted accelerator now raises instead of importing another
  vendor's runtime. The failing frame was re-run on the MTT S5000 in this change and
  passes, as does FSDP2 itself — `tests/manual/musa/test_fsdp2_musa.py --world-size 2`
  completed with finite, non-increasing loss across two MTT S5000 devices, over
  `ProcessGroupFlagOS`'s host-staged gloo tier (this environment has no vendor
  communicator, as the warning in that run records). The six nodeids were **not**
  re-run through the HF harness, so the `05be850` numbers above are what the harness
  last measured.

Because `Affected tests` and the baseline header above describe `64e60dd`, the
whole table is a dated snapshot and not a statement about the current tree.
Re-running the baseline is what refreshes it — and is also what fills in the
`Fingerprint` column, per the note above. The closures above are recorded from the
issues and the commits that landed; two re-measurements have re-run any of this
baseline's tests since — #268's eight borrowed nodeids and #263's six — so the rows
for #250/#265, #264 and #266 are **not revalidated** here.

### Failed test inventory

The 20 failed tests were grouped for follow-up investigation:

- **SDPA precision (8):**
  `test_eager_matches_sdpa_inference_00_fp16_pad_left_sdpa_kernels`,
  `test_eager_matches_sdpa_inference_01_fp16_pad_left`,
  `test_eager_matches_sdpa_inference_02_fp16_pad_left_no_attn_mask_sdpa_kernels`,
  `test_eager_matches_sdpa_inference_03_fp16_pad_left_no_attn_mask`,
  `test_eager_matches_sdpa_inference_04_fp16_pad_right_sdpa_kernels`,
  `test_eager_matches_sdpa_inference_05_fp16_pad_right`,
  `test_eager_matches_sdpa_inference_06_fp16_pad_right_no_attn_mask_sdpa_kernels`,
  `test_eager_matches_sdpa_inference_07_fp16_pad_right_no_attn_mask`.
- **FSDP2 distributed support (6):**
  `test_fsdp2_plan_vs_ddp_0_untied`,
  `test_fsdp2_plan_vs_ddp_1_tied`,
  `test_fsdp2_save_load`,
  `test_fsdp2_save_load_dcp`,
  `test_fsdp2_sharding_structure_0_untied`,
  `test_fsdp2_sharding_structure_1_tied`.
- **Torch compile and cache behavior (3):**
  `test_generate_compilation_all_outputs`,
  `test_generate_compile_model_forward_fullgraph`,
  `test_static_cache_no_recompile_with_smaller_length`.
- **Other model behaviors (3):**
  `test_custom_4d_attention_mask`,
  `test_model_parallelism`,
  `test_model_rope_scaling_frequencies`.

The inventory above records the original suite grouping. The corrected
single-nodeid reruns supersede its provisional labels; in particular, the eight
SDPA-named tests reproduced the same non-contiguous softmax failure rather than a
pure tolerance mismatch. Future measurements must also run with
`FLAGOS_LOG=fallback` and report any passing operation that used CPU fallback
as an accelerator coverage gap.

That last clause is worth revisiting now that #398 has landed. With the strided
operands materialized before mudnn, a tolerance mismatch is exactly what those
eight show: they pass under the harness's device shims and fail without them, at
the `allclose` check rather than in the operator (the re-measurement above). The
baseline's reading was right for the tree it measured — and the behaviour it ruled
out is what remains once the operator is fixed, which is why the residual is
tracked by #248 and not as a softmax defect.
