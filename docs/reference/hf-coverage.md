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

Because the device context was poisoned, failures after the first accelerator
fault are not considered independently verified findings. The device-poisoning
finding was filed as issue #250 after the run-level poison signal was confirmed;
the remaining test failures were not filed individually. The raw JSON report
remains outside the repository at `/tmp/qwen3.json`.

This first sweep is both the initial baseline and the source of one verified
tracker finding. It is recorded as an observed defect on the pinned tuple, not
as a regression claim.

The issue reference is retained here so future runs can distinguish the known
poisoning cause from new findings:

| Fingerprint | Class | Subject | Issue |
| --- | --- | --- | --- |
| Pending cause-level fingerprint | `CRASH` | qwen3 device context poisoning | [#250](https://github.com/flagos-ai/Torch-FL/issues/250) |

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

The SDPA failures require an isolated rerun with a CPU same-dtype baseline and
aten attribution before classification. The remaining failures likewise need
isolated subprocess runs to distinguish the first fault from collateral
failures. Cause fingerprints and issue references will be added only after the
findings pass the evidence and deduplication gates.
