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
