<!-- Copyright 2026 FlagOS Contributors -->
<!-- Licensed under the Apache License, Version 2.0 -- see the repository LICENSE. -->

# scripts/

Development tooling: operator code generation, vendor build helpers, the
Transformers triage pipeline, and standalone checks. Nothing here is imported at
runtime by the installed wheel except where noted; these are host-side tools.

## Layout

| Directory | Purpose |
|---|---|
| `codegen/` | Generators. Each reads a source of truth and writes into `csrc/aten/generated/`, `torch_fl/configs/`, or `torch_fl/tileops/generated/`. |
| `vendor/` | Per-vendor build and setup steps: bundling a vendor libtorch into the wheel, patching triton-ascend, preparing BPU/NPU toolchains. Called by `.github/scripts/set_env_*.sh` and by `setup.py`. |
| `transformers/` | The Transformers test → triage → verify → dedup → file-issue pipeline, plus its own smoke test. |
| `tools/` | Standalone checks that do not belong to the pipeline above: PR validation, a FlagGems-on-Ascend sweep, a MUSA failure recorder, and a pytest-free TileOPs check. |

Run every script from the repository root; the examples below assume that.

## codegen/

| Script | What it generates | Run |
|---|---|---|
| `codegen_ops.py` | `csrc/aten/generated/{ops.h,ops.cc,cuda_kernels.cc,flaggems_python_kernels.cc,register.inc}` for the CUDA/boxing path. Reads `torch_fl/configs/backends_cuda.conf` and torchgen. | `python scripts/codegen/codegen_ops.py` |
| `codegen_ascend.py` | Ascend aclnn backend sources under `csrc/aten/backends/ascend/`, and appends the covered ops to `backends_ascend.conf`. | `python scripts/codegen/codegen_ascend.py [--category NAME] [--no-conf]` |
| `codegen_gcu.py` | Enflame GCU (topsaten) backend sources, and the matching conf append. | `python scripts/codegen/codegen_gcu.py [--category NAME] [--no-conf]` |
| `codegen_mudnn.py` | Moore Threads MUSA (mudnn) backend sources, and the matching conf append. | `python scripts/codegen/codegen_mudnn.py [--category NAME] [--no-conf]` |
| `codegen_musa_flaggems.py` | The MUSA FlagGems registration list (`.inc`). | `python scripts/codegen/codegen_musa_flaggems.py [--check]` |
| `codegen_gcu_flaggems.py` | The GCU FlagGems registration list (`.inc`). | `python scripts/codegen/codegen_gcu_flaggems.py [--check]` |
| `codegen_autograd.py` | `csrc/aten/generated/variable_type.cc` — the `AutogradPrivateUse1` layer for ops a backend re-owns with a fused kernel. | `python scripts/codegen/codegen_autograd.py` |
| `codegen_tileops.py` | `csrc/aten/generated/tileops_python_kernels.cc`, `torch_fl/tileops/generated/` (routes, shims), `tests/integration/ops/test_tileops_generated.py`, and the `TILEOPS_OPS` block in `backend_coverage.py`. | `python scripts/codegen/codegen_tileops.py [--check]` |
| `gen_vendor_confs.py` | `torch_fl/configs/backends_<vendor>.conf`, one full-coverage conf per platform. | `python scripts/codegen/gen_vendor_confs.py [--check\|--stats]` |
| `extract_name_map.py` | `csrc/aten/generated/name_map.json` — op name → dispatcher symbol, parsed back out of the existing `csrc/aten/*.h`/`.cc`. | `python scripts/codegen/extract_name_map.py` |
| `backend_coverage.py` | Not a generator: the measured coverage sets the conf generator reads. `TILEOPS_OPS` is rewritten in place by `codegen_tileops.py`; the two FlagGems sets are hand-maintained. | — |

Adding a backend: see `.claude/skills/native-op-backend/SKILL.md`, which walks
through writing a `codegen/codegen_<vendor>.py`.

## vendor/

| Script | Purpose |
|---|---|
| `bundle_common.sh` | Shared helpers for self-contained wheel bundling. Sourced by the three `bundle_*_libtorch.sh` scripts, not run directly. |
| `bundle_dcu_libtorch.sh` | Copy DTK's libtorch device `.so` into `torch_fl/lib_dcu/`. |
| `bundle_maca_libtorch.sh` | Copy the MetaX-forked libtorch C++ `.so` into `torch_fl/lib_maca/`. |
| `bundle_ppu_libtorch.sh` | Copy the locally built PPU libtorch C++ `.so` into `torch_fl/lib_ppu/`. |
| `check_dcu_core_abi.py` | Assert that the DTK device libraries need no unaccounted vendor-core symbols. Run from `bundle_dcu_libtorch.sh` and from `setup.py`. |
| `patch_triton_ascend.py` | Patch triton-ascend so it works without a torch_npu dependency. Called from `.github/scripts/set_env_ascend.sh`. |
| `setup_torch_npu_stubs.sh` | Create minimal torch_npu header stubs for triton-ascend JIT compilation. Called from the Ascend CI workflow. |
| `setup_bpu_hbdk4.sh` | Install the hbdk4 BPU graph compiler (x86_64-only wheels) plus box64 where on-board compilation is needed. |
| `with_cuda_libtorch.sh` | Run any command with a version-matched `libtorch_cuda.so` preloaded via `LD_PRELOAD`. Superseded by the single-wheel runtime preload for normal use; still the way to run a CUDA test against an external libtorch. |

## transformers/

The pipeline is `test → triage → verify → deduplicate → preview/file issues`.

| Script | Purpose |
|---|---|
| `transformers_auto_sweep.sh` | End-to-end driver: runs all of the below for one model. `./transformers_auto_sweep.sh <model> [device] [chip] [repo]` |
| `transformers_batch_sweep.sh` | Runs the sweep across the built-in model list. `./transformers_batch_sweep.sh [device] [chip] [repo]` |
| `safe_transformers_wrapper.py` | Runs one model's test under a guard, designed to survive weak models. |
| `transformers_triage.py` | Classify a test report into failure classes. |
| `transformers_verify.py` | Re-check each finding in a fresh pytest subprocess. |
| `transformers_deduplicate.py` | Collapse findings that share a root cause. |
| `transformers_preview_issues.py` | Render the issues that would be filed, without filing them. |
| `transformers_file_issues.py` | File the issues. |
| `test_transformers_automation.py` | Smoke test for the pipeline itself; run it after changing any script in this directory. |

## tools/

| Script | Purpose |
|---|---|
| `validate_ai_pr.py` | Validate an AI-authored PR body against the repository's requirements. `python scripts/tools/validate_ai_pr.py --pr-body pr_description.md` |
| `verify_flaggems_ascend.py` | Per-op check of whether the FlagGems Triton path is numerically correct on Ascend. Slow (Triton JIT dominates, hours for a full sweep), supports `--shard i/n` and `--ops`. |
| `record_musa_flaggems_failures.py` | Record MUSA FlagGems ops that fail CI so they move into `NATIVE_TRITON_GAPS`, then regenerate the confs. |
| `run_tileops_checks.py` | TileOPs route checks with plain python, for hosts without pytest. `python scripts/tools/run_tileops_checks.py [--full] [--filter SUBSTR]` |

## Generated artifacts

Several files in the tree are written by these scripts. Do not hand-edit them:
change the generator (or its source of truth) and regenerate.

No CI job currently runs these generators or their `--check` modes — CI covers
`tests/integration/` only, and `tests/unit/` (which holds the conf check) is not
wired into any workflow. Until that changes, run the staleness column yourself
before opening a PR that touches a generator.

| Artifact | Generator | Staleness check |
|---|---|---|
| `csrc/aten/generated/{ops.h,ops.cc,cuda_kernels.cc,flaggems_python_kernels.cc,register.inc}` | `codegen_ops.py` | none — `tests/integration/ops/test_flaggems_conf_consistency.py` asserts the routing these files encode |
| `csrc/aten/generated/variable_type.cc` | `codegen_autograd.py` | none |
| `csrc/aten/generated/tileops_python_kernels.cc` | `codegen_tileops.py` | `codegen_tileops.py --check` |
| `csrc/aten/generated/name_map.json` | `extract_name_map.py` | none |
| `csrc/aten/backends/ascend/generated/*` | `codegen_ascend.py` | none — regenerate and require an empty diff |
| `csrc/aten/backends/gcu/generated/{gcu_kernels.cc,gcu_register.inc}` | `codegen_gcu.py` | none — regenerate and require an empty diff |
| `csrc/aten/backends/gcu/generated/gcu_flaggems_register.inc` | `codegen_gcu_flaggems.py` | `codegen_gcu_flaggems.py --check` |
| `csrc/aten/backends/musa/generated/{musa_kernels.cc,musa_register.inc}` | `codegen_mudnn.py` | none — regenerate and require an empty diff |
| `csrc/aten/backends/musa/generated/musa_flaggems_register.inc` | `codegen_musa_flaggems.py` | `codegen_musa_flaggems.py --check` |
| `torch_fl/configs/backends_*.conf` | `gen_vendor_confs.py` | `gen_vendor_confs.py --check`, and `tests/unit/test_gen_vendor_confs.py` enforces the same contract |
| `torch_fl/tileops/generated/*` | `codegen_tileops.py` | `codegen_tileops.py --check` |
| `tests/integration/ops/test_tileops_generated.py` | `codegen_tileops.py` | `codegen_tileops.py --check` |
| `TILEOPS_OPS` in `codegen/backend_coverage.py` | `codegen_tileops.py` | `codegen_tileops.py --check` |

Generated files carry an `@generated by scripts/... -- DO NOT EDIT.` banner.
When you change a generator, regenerate and commit the output in the same
change, then re-run the generator and require an empty diff.

The vendor generators (`codegen_ascend.py`, `codegen_gcu.py`, `codegen_mudnn.py`)
must run on a host with that vendor's ATen headers; the CUDA generators must run
with an importable `torch_fl` (built `torch_fl._C`) and, for the FlagGems sets, a
matching `flag_gems` install. Regenerating in the wrong environment silently
produces a smaller cohort — see `.claude/skills/flaggems-integration/SKILL.md`.
