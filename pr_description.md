## AI Agent Information
- **Agent/Tool**: Codex CLI
- **Model**: DeepSeek V4.1 Flash
- **Human Reviewer**: @zhaoyinglia
- **Session Summary**: Following the shared runtime work, the user requested one FlagGems integration flow and one set of test contracts across every configured platform. Operator coverage remains platform data; shared case definitions, execution, and CI completeness checks are common.

## Summary
This PR replaces chip-named FlagGems and SDPA test files with shared cases and per-platform data rows. All ten configured platforms—Ascend, CUDA, DCU, GCU, Iluvatar, Kunlunxin, MetaX, MPS, MUSA and PPU—use the same shared conformance paths. CI generates exact case inventories and fails when an expected case is missing, skipped, or unplanned. Platforms may still route different operators through FlagGems.

The Qwen3 smoke groups now use one model contract on all ten platforms. A host-only model job on each platform's runner downloads a pinned Qwen3 revision and validates its files, size, and safetensors layout. Linux platforms populate the same read-only `/models/Qwen3-0.6B` mount before Docker resolves the integration job's volume, while MPS receives the validated checkpoint through an artifact. The model, inference, and training groups run on every platform.

The common dependency setup now takes precedence over vendor image versions. The former native-vendor collection gate is removed, so existing vendor regression files remain selected on every operator platform. The FlagGems routing contract follows the actual dispatcher names, while public wrappers such as `embedding` and `index_add_` run as shared execution cases. The routing expectation also accounts for operations omitted from a wheel's native registration set. Device initialization now precedes FlagGems' dispatch repairs, so CUDA-ABI wheels bind the patched callables before the first routed operation.

## Change Type
- [x] Bug Fix
- [ ] New Feature
- [ ] Performance Optimization
- [x] Refactoring
- [x] Documentation
- [x] Testing
- [x] CI/Infrastructure
- [ ] Breaking Change

## Platforms Affected
- [x] CUDA
- [x] MetaX
- [x] Ascend
- [x] PPU
- [x] Platform-agnostic (all platforms)

The shared tests and inventory checks apply to Ascend, CUDA, DCU, GCU, Iluvatar, Kunlunxin, MetaX, MPS, MUSA and PPU. MPS omits the FlagGems kernel pass because its build contract forbids that kernel set; its native FlagGems contracts still run.

## Problem Analysis
### What was broken/missing?
FlagGems and SDPA coverage was duplicated across chip-named test files. A missing runtime, route, or case could be skipped in one platform's copy, leaving a partial run that looked successful. Adding a case in one file did not require the other platforms to run it.

The setup step shared test dependencies but could still read vendor image pip configurations. It treated an older exact pin as satisfied, did not repair newer or missing packages, and MetaX and PPU imported FlagGems before PyYAML was available. These differences surfaced during `Prepare platform environment` instead of producing a tested, consistent environment.

The FlagGems routing suite treated every public API in its call table as a dispatcher operator. `F.embedding` logs `index_select` instead, so its logs did not establish an `embedding` route. On CUDA-ABI wheels, some conf rows point to a native arm compiled with another kernel set; these operations fall back to CPU when absent, and must be expected from the wheel's registration rather than assumed to reach the native backend.

The FlagGems contract patches import their package callables lazily. Applying them before FlagGems binds those exports could leave the original callables in place. The repaired CUDA kernels also need to remain reachable through both the package and the C++ bridge.

The unit runner also waited indefinitely when a completed pytest session was followed by a stalled interpreter exit. On MPS, five BPU files reported their skipped results and then each waited the full 300-second file timeout, preventing later unit files from starting. The runner now records pytest's verdict when its finish hooks return and bounds only the subsequent interpreter exit.

Integration groups relied on the outer job timeout when an individual group stalled or left a process behind. That could end the whole platform job before its structured report was written. The integration runner now gives every group a shorter deadline, terminates its process group, returns exit code `124`, and retains reports from completed groups.

### Why did it happen?
The chip-specific files preserved a bring-up model after all configured platforms entered one CI pipeline. Case selection and skipping differed by filename and optional runtime behavior. The dependency installer used the image's package configuration and did not compare every installed version with the shared cohort. Route expectations used the public call name without distinguishing a dispatcher row from the operator a wrapper invokes internally.

### Investigation process:
1. Compared the FlagGems routing, execution, exclusion, device identity, and SDPA cases with the chip-named files and grouped shared contracts separately from platform route and numeric data.
2. Traced platform configuration, manifest composition, environment preflight, pytest launch, and report generation to find every place a shared case could disappear from a platform's result.
3. Inspected earlier CI logs for the preparation step. Docker images supplied different `typing-extensions`, `transformers`, and `filelock` versions. MetaX and PPU failed their FlagGems probe because `yaml` was missing.
4. Ran the unified routing and execution cases on an H100 through JMS. The public-wrapper case and the wheel-registration expectations now use the dispatcher logs and the checkout's registration files.
5. Audited existing vendor regression files and removed a speculative native-vendor collection gate that changed test selection outside the FlagGems migration.
6. Probed Ascend through JMS. Dependency setup reached the shared installer but failed on DNS resolution for `repo.huaweicloud.com`. Ascend results for the latest source therefore remain **not revalidated**.
7. Verified the current local source against the files, installed package, and shared test tree used on H100. All six changed FlagGems test and startup files matched the expected SHA-256 values.
8. Inspected the MPS job log from run `39731566307`. Five skipped BPU files printed their summary at `06:18:03` and were terminated five minutes later; the job reached its 60-minute timeout before the remaining files ran.
9. Inspected run `39740142613`. PPU failed its first integration group because `/models/Qwen3-0.6B` had no configuration file on the runner. Each job also receives a separate runner, so a directory created in one job is not available to the next container job without a shared host path.

## Solution Design
### Implementation approach:
`tests/integration/flaggems/run.py` provides shared `native` and `flaggems` kernel passes. Every platform shares the native routing, execution, exclusion, device identity, and SDPA case definitions. Platform route rows live in `flaggems/sdpa_caps.py` and `vendor_lanes/sdpa_lane_caps.py`; shared operator calls live in `shared_sdpa_caps.py`.

The manifest and integration loader generate exact case inventories, associate every report with its declaring group, and reject missing, extra, or skipped shared cases. The FlagGems groups receive `TORCH_FL_SOURCE_ROOT`, which names the checkout carrying the registration lists; tests still run from a wheel-only tree. The common `vendor_ops` selection remains unchanged, and the operator conftest does not block vendor regression files by build kernel set.

`FLAGGEMS_ROUTING_CALLS` names only dispatcher rows. `FLAGGEMS_RESULT_CALLS` are the CPU-equivalence cases, and each must have a routing row. `FLAGGEMS_EXECUTION_CALLS` contains the public wrappers; their child records the operators the wrapper actually logs. A fallback log's `aten::` prefix is matched to its conf row.

`route_reason()` reads each platform's committed registration includes and build record. A native route is expected only when that wheel registers the dispatcher operation; otherwise the expected behavior is a CPU fallback. FlagGems routes must still be registered and cannot be hidden by this rule.

`verify_shared_dependencies.py` uses `PIP_CONFIG_FILE=os.devnull` and removes both `PIP_CONSTRAINT` variables for the shared install while retaining `PIP_INDEX_URL`. It resolves the exact pins together, then force-reinstalls only packages whose installed metadata still differs. The MetaX and PPU hooks install PyYAML 6.0.1 before importing FlagGems. PPU child probes load `torch_fl` before `torch`, because the facade selects the PPU runtime only through the first import. The installed-wheel probe follows the current interpreter's `sys.path`.

Device initialization now occurs before the FlagGems contract patches. The repaired `where_self_out` and `slice_backward` callables are published in the package and `flag_gems.ops` namespaces without replacing a submodule object.

The unit runner launches each pytest file in its own process group. A temporary session marker records the pytest exit status after all `pytest_sessionfinish` hooks return, and records completion after all `pytest_unconfigure` hooks return. If the interpreter stays alive for 10 seconds after that work, the runner kills the process group and preserves pytest's status for the next file. The file's ordinary 300-second timeout still applies to the session itself, so a test that hangs during collection or execution is still reported as a timeout.

The integration runner derives a per-group timeout from the platform job timeout and launches each group in a new process session. At the deadline it terminates the session, then forces the remaining process group to exit and reads the output that was already written. A completed group's report remains available for the merged result.

The model job chooses storage from the platform config before downloading. Container platforms must provide a writable Linux host bind root; the checkpoint is written directly there and checked again before the container job starts. The checkpointless platform must be macOS and carries its checkpoint as a `qwen3-0.6b-<platform>` artifact to the integration job's workspace.

The checkpoint is pinned to Hugging Face revision `c1899de289a04d12100db370d81485cdf75e47ca`. Validation checks the required Transformers and tokenizer files, the exact size of `model.safetensors`, and the 8-byte safetensors header length, JSON document, and tensor entries. The integration container repeats the header check on every mounted copy.

### Key design decisions:
- Test contracts are shared; operator coverage remains configuration data. Different routes do not require copied tests.
- Missing, skipped, or unplanned shared cases fail CI. Platform support differences are represented explicitly in route rows and build configuration.
- Public wrapper execution and dispatcher routing have separate shared cases, so each reports the behavior it can establish.
- Native route expectations follow the registration set compiled into the wheel. FlagGems route expectations remain unconditional.
- The shared dependency cohort wins over image versions, while configured package indexes remain available for the install.
- A completed pytest session can leave stalled vendor cleanup without consuming the next file's budget. Crashes and test failures retain their existing classification.
- Docker receives a fully populated bind source from the runner. MPS uses an artifact because a validation job and an integration job do not share a temporary directory.
- A timed-out group fails with its own name and a report that preserves earlier completed cases.

### Code changes by file:
- `.github/configs/*.yml`: declare FlagGems kernel support, device facts, and shared test requirements for all ten platforms.
- `.github/integration-manifest.yml`: define common FlagGems native, kernel, and SDPA groups, select the same vendor and RNG cohorts on eight operator platforms, and remove chip-specific test references.
- `.github/scripts/{compose_integration_manifest.py,integration_inventory_plan.py,flaggems_cases.py}`: compose common commands and inspect exact source case inventories.
- `.github/scripts/{run_integration_tests.py,merge_pytest_reports.py,python_coverage.py,run_unit_tests.py}` and `.github/scripts/pytest_reporting/`: bind reports to their groups, preserve launch failures, and pass the registration source.
- `.github/scripts/run_unit_tests.py` and `tests/unit/test_run_unit_tests.py`: bound interpreter exit after pytest's finish hooks, clean up leftover child processes, and preserve failure status and reports.
- `.github/scripts/{check_platform_environment.py,verify_shared_dependencies.py,verify_shared_dependency_imports.py}`: validate the vendor environment and force-reinstall the exact shared Python and FlagGems runtime dependencies.
- `.github/workflows/integration-tests.yml` and `.github/scripts/run_integration_tests.py`: prepare one pinned model on each platform host, check the bind root before download, hand MPS its checkpoint, and bound each integration group.
- `.github/scripts/hooks/set_env_{metax,ppu}.sh`: make PyYAML available before either FlagGems import probe; PPU also receives its existing NumPy and SQLAlchemy prerequisites.
- `tests/integration/flaggems/`: define shared routing, execution, public-wrapper, exclusion, device identity, and SDPA contracts. Route data remains per platform.
- `tests/integration/shared_sdpa_caps.py` and `tests/integration/vendor_lanes/`: hold shared SDPA calls and platform route rows.
- Removed files under `tests/integration/ops/` and `tests/unit/`: chip-named FlagGems or SDPA cases migrated into the shared suites.
- `csrc/aten/{dispatcher.h,common.cc,common.h}` and `csrc/aten/register.cc`: validate input contracts before backend lookup and preserve native registration behavior.
- `torch_fl/__init__.py` and `torch_fl/flagos/__init__.py`: initialize FlagGems before applying its dispatch repairs and bind the repaired callables for the C++ bridge.
- `tests/unit/test_flaggems_route_reason.py` and related unit tests: cover wrapper routing, registration-aware expectations, registration-source propagation, and restored vendor test selection.
- `docs/reference/{operator-support,compatibility,environment-variables,platform-capability-matrix}.md`, `docs/architecture/torch-compile-integration.md`, and the vendor notes: record the unified flow and measured platform scope.

### Changes by commit:
1. `01672614` - `test: unify FlagGems integration across configured platforms`: replaces chip-specific cases with shared suites and exact inventories.
2. `00f51b4c` - `fix: align shared test inventories with source checkouts`: validates inventories from their declaring source tree.
3. `1075765f` - `fix: isolate vendor cases and restore the Ascend build`: keeps vendor regression cases selected and fixes the Ascend build path.
4. `2cb5a975` - `fix: prepare Ascend TLE before shared FlagGems import`: makes the Ascend preparation probe independent of the uninitialized device.
5. `b3d10c8a` - `fix: route PPU FlagGems gaps through boxing tests`: gives PPU's FlagGems-only wheel its applicable route checks.
6. `f0aba09f` - `test: initialize device before MPS imports`: initializes the MPS runtime before FlagGems imports.
7. `18d1569e` - `fix: restore Ascend generated registrations`: keeps generated metadata registrations in the Ascend build.
8. `f6dc5dfc` - `fix: align shared FlagGems cases with hardware contracts`: makes routing, wrapper execution, and test selection follow the actual wheel contract.
9. `d2cfa8ca` - `fix: initialize FlagGems before applying dispatch repairs`: separates dispatcher and wrapper cases, accounts for native registrations, restores the existing vendor selection, and binds the repaired CUDA callables.
10. `4596b7ba` - `merge: reconcile shared FlagGems integration with main`: incorporates main's convolution registration fix and updates the focused fallback regression test.
11. `9ec941aa` - `fix: bound the unit runner's post-session exit`: records pytest's completed session and does not let stalled vendor teardown consume every remaining file's timeout.
12. Pending commit - `fix: prepare the shared model outside platform containers`: downloads and validates one pinned Qwen3 revision on each platform's host, populates the fixed bind root before Docker starts, and runs the model, inference, and training groups on every platform.

## Verification
### Pre-submission Checklist
- [x] **Linting passed** (`ruff check`, `ruff format --check`)
- [ ] **Type checking passed** (the repository has no configured type-check command)
- [ ] **All tests pass** (the focused unit cohort and an H100 wheel run pass; accelerator jobs for the pushed commit remain to be observed, and the local source checkout has no compiled `torch_fl._C`)
- [ ] **Manual hardware testing completed** (CUDA-ABI FlagGems cases ran on H100 through JMS; Ascend is not revalidated for this revision)
- [x] **No debug/temporary code** (intentional CI progress output remains; no debug statements)
- [x] **Documentation updated** (shared flow, platform support records, and dependency behavior)
- [x] **Commit messages follow conventions** and use English
- [x] **All GitHub-facing text is in English**

### Linting Results
```text
$ /tmp/pr537-py312/bin/python -m ruff check
All checks passed!

$ /tmp/pr537-py312/bin/python -m ruff format --check
487 files already formatted

$ git diff --check
# No whitespace errors in the working or staged change.
```

The 213 Python files changed from `flagos/main`, including new files, also pass `ruff check` and `ruff format --check`.

### Test Results
Focused unit cohort on the final source:

```text
$ /tmp/pr537-py312/bin/python -m pytest \
    tests/unit/test_conf_registration_consistency.py \
    tests/unit/test_cuda_convolution_fallback.py \
    -q
66 passed in 0.12s

$ /tmp/pr537-py312/bin/python -m pytest \
    tests/unit/test_flaggems_probe.py \
    tests/unit/test_flaggems_route_reason.py \
    tests/unit/test_run_integration_tests.py \
    tests/unit/test_ascend_matmul_soc_dtype.py \
    tests/unit/test_integration_workdir.py \
    tests/unit/test_ops_conftest_platform.py \
    tests/unit/test_flaggems_cases.py \
    tests/unit/test_flaggems_run.py \
    tests/unit/test_integration_manifest.py \
    -q --tb=short -p no:cacheprovider
276 passed in 22.99s

$ /tmp/pr537-py312/bin/python -m pytest \
    tests/unit/test_run_unit_tests.py \
    tests/unit/test_run_integration_tests.py \
    tests/unit/test_flagcicd_reports.py \
    -q
106 passed in 12.59s

$ /tmp/pr537-py312/bin/python -m pytest \
    tests/unit/test_ci_pip_cache.py \
    tests/unit/test_ci_platform_routing.py \
    tests/unit/test_ci_matrix.py \
    -q
209 passed in 10.90s

$ /tmp/pr537-py312/bin/python -m pytest \
    tests/unit/test_device_tracer_selection.py \
    tests/unit/test_run_unit_tests.py \
    -q
58 passed in 4.48s
```

The final model, mount, safetensors, and timeout changes pass the following focused cohort:

```text
$ /tmp/pr537-py312/bin/python -m pytest -q \
    tests/unit/test_run_integration_tests.py \
    tests/unit/test_integration_manifest.py \
    tests/unit/test_ci_platform_routing.py \
    tests/unit/test_shared_dependencies.py \
    tests/unit/test_integration_workdir.py \
    tests/unit/test_check_platform_environment.py \
    tests/unit/test_ci_pip_cache.py \
    --tb=short -p no:cacheprovider
387 passed in 24.85s
```

The H100 verification used an installed FlagGems wheel and the shared tests from this branch. The SHA-256 values of all six changed FlagGems test and startup files matched the local source. The installed `torch_fl/__init__.py` and `torch_fl/flagos/__init__.py` also matched the local files.

```text
$ /usr/bin/python -m pytest \
    tests/integration/flaggems/test_flaggems_routing.py \
    tests/integration/flaggems/test_flaggems_execution.py \
    -v -s --tb=short
40 passed in 9.95s
```

The suite covers 23 dispatcher rows, one public-wrapper execution case, 15 CPU-equivalence cases, and the float64 `mm` boundary case.

```text
$ /usr/bin/python -m pytest \
    tests/integration/flaggems/test_flaggems_routing.py::test_public_wrapper_calls_execute \
    -v -s --tb=short
1 passed, 39 deselected in 8.40s
```

The local source checkout cannot load tests that require the compiled `torch_fl._C`; the corresponding runtime coverage comes from the H100 wheel run and the platform jobs.

An unfiltered `pytest tests/unit/` collection stops on the missing `torch_fl._C` extension and optional dependencies in this macOS checkout. The focused commands above include the runner, CI routing, reporting, and startup tests that can run without an installed accelerator wheel.

### Manual Verification
Earlier preparation failures included package versions that differed from the shared cohort:

```text
MPS: typing-extensions 4.16.0 installed; shared pin is 4.15.0
Iluvatar: transformers 4.57.6 installed; shared pin is 5.20.2
Ascend: filelock 3.32.3 installed; shared pin is 4.2.4
MetaX and PPU: ModuleNotFoundError: No module named 'yaml'
```


Run [39740142613](https://github.com/flagos-ai/Torch-FL/actions/runs/39740142613)
confirmed that the PPU dependency install and import check passed. Its first
integration group then found a missing model configuration:

```text
Integration test 'Check isolated PPU environment' failed with exit code 1
AssertionError: MODEL_PATH has no Transformers config.json: /models/Qwen3-0.6B
```

The PPU runner's host path no longer supplies the checkpoint. The new model
preparation job checks the bind root and downloads and validates the pinned
files on PPU before its container starts. That path is covered by local
workflow and safetensors tests; PPU tests are **not yet revalidated** until the
new CI run executes them.

The same run installed the exact shared cohort on MPS and reached its integration
tests. MPS had not completed that group when this description was updated, so
no MPS hardware result is claimed for this revision. The new host model job is
also unvalidated on MPS hardware.

The current CI run for `9ec941aa` still uses the older model setup. Its results
cannot establish the new `/models` mount or the universal model groups.

The dependency reconciliation uses pip resolution followed by a metadata check and a focused forced reinstall for remaining mismatches. Tests cover older, newer, and missing packages, clearing image constraints, preserving FlagTree Triton, locating the installed wheel through interpreter site paths, loading FlagGems before its probes, checking the startup capability only on applicable builds, and PyYAML preceding both FlagGems probes.

The H100 run exercises the CUDA-ABI FlagGems-only wheel. Its expected routes are derived from the generated FlagGems registration include, while public wrappers are checked for successful execution. The startup ordering and callable binding tests are covered locally.

JMS Ascend dependency setup failed on DNS lookup for `repo.huaweicloud.com` before the FlagGems cases could run. Ascend operator and startup results for this revision are therefore **not revalidated**. This evidence gap remains in the platform support record and must be closed by measured results; routing configuration alone is not presented as a hardware result.

Local runner tests cover an atexit sleep, a process left by teardown, and a SIGSEGV before session completion. The first test confirms a timeout in interpreter exit preserves pytest status 0; the second confirms the process group is cleaned up; the third remains a crash. Existing tests also confirm that a nonzero pytest status remains a failure and that its report is attached to the inventory.

## Performance Impact
Not a performance change. The shared tests add inventory, report, and dependency checks to CI and replace duplicate chip-specific suites with common ones.

## Breaking Changes
### What breaks:
No public torch_fl API change. CI configurations or scripts that invoke a deleted chip-specific test file must use the shared FlagGems or SDPA suites through the integration manifest.

### Migration path:
```bash
# Old chip-specific test path:
python -m pytest tests/integration/ops/test_metax_sdpa_route.py

# Shared FlagGems native contract:
python tests/integration/flaggems/run.py native

# Shared vendor SDPA cases:
python -m pytest tests/integration/vendor_lanes -v -s --tb=short
```

### Affected users:
CI maintainers and contributors who referenced removed chip-specific test files directly.

## Code Quality Verification
### Style Consistency
- [x] Followed the existing platform configuration and pytest fixture structure
- [x] Reused the existing backend configuration and manifest composer
- [x] Kept shared case definitions separate from platform data
- [x] Matched surrounding comments and error-reporting conventions

### Edge Cases Considered
1. A shared case that skips on one platform fails that platform's inventory check instead of shrinking the expected case set.
2. A public wrapper invokes a different dispatcher operation: its execution is checked, while the named dispatcher row has its own routing case.
3. A conf row names a native route absent from the wheel's registration set: the shared expectation is a CPU fallback, while a FlagGems route remains required.
4. A FlagGems callable is also a submodule name: patching updates the dict without replacing the importable submodule.
5. An image package is older, newer, or missing: the shared installer retains a satisfying newer version and replaces remaining mismatching pins without replacing torch.
6. MPS's build forbids the FlagGems kernel set: the common kernel group is omitted by that declared build contract while all native shared cases continue.
7. A completed pytest session hangs in interpreter exit or leaves a child process behind: the file is bounded after its finish hooks, its result remains available, and its process group is cleaned up.
8. A host runner retains a model directory from an earlier job: only the pinned revision is accepted, and a stale cache entry is checked before use.
9. The container job starts before its own first step: the model is staged by a host-only job on the same hardware runner, rather than by a step inside the container.

### Potential Risks
1. Some historical numeric SDPA bounds may need measured corrections on newly unified hardware. Those corrections belong in a platform row and do not fork the test contract.
2. Vendor images can continue to differ during provisioning. The shared installer rejects surviving version mismatches, so later setup changes must preserve the common cohort.
3. Vendor libraries may still delay their own finish or unconfigure hooks. Those hooks are allowed to complete; the post-session bound applies only after pytest has finished its reporting and plugin teardown.
4. The fixed bind root requires a writable `/var/lib/torch-fl-ci` directory on every Linux hardware runner. The host model job reports a clear failure if it cannot create the directory or move the checkpoint.
5. Downloading a 1.4 GB checkpoint across all ten platform runners adds network and disk time. The pinned cache reduces a warm run; the hardware results will show whether each platform's budget is sufficient.
6. MPS integration already approached its 60-minute job limit in the previous run. The model job adds a host download and validation step, and its result remains unvalidated until the new run completes.

### Rollback Plan
Revert the commits listed above, restoring the previous chip-specific tests and dependency setup. No generated operator kernels or production operator routes are changed by this work.

## Related Work
- Related to #534 (shared runtime CI wiring)
- Related to #535 (all-platform integration groups)
- Follows the shared runtime work that supplies the platform environment

## Explicitly Not Included
- Requiring every platform to support the same operator set. The process and test contracts are shared, while route coverage and measured results remain platform data.
- Fixing backend operator defects discovered by the unified tests. A failing shared case is reported as a failure for that platform.
- Adding accelerator hardware execution to this macOS checkout. Hardware verification comes from JMS and the PR's CI jobs.

## Human Review Notes
### Areas needing special attention:
1. `.github/scripts/verify_shared_dependencies.py` and the MetaX and PPU hooks: check the authoritative shared pins, constraint handling, and pre-probe dependency order.
2. `.github/integration-manifest.yml` and the platform configs: check that common groups reach every platform and that MPS omits only the build-inapplicable FlagGems kernel pass.
3. `.github/scripts/run_integration_tests.py`, reporting scripts, and `flaggems_cases.py`: check that a missing, skipped, or unplanned case fails the correct group.
4. `tests/integration/flaggems/`: check the separation of dispatcher routing, CPU-equivalent results, and public-wrapper execution, and the use of wheel registration data.
5. `torch_fl/flagos/__init__.py` and `torch_fl/__init__.py`: check that startup ordering and callable binding preserve existing imports and FlagGems' package behavior.
6. `.github/scripts/run_unit_tests.py`: check that the session marker is complete before the process group is killed and that pytest's status is preserved for report merging.

### Questions for reviewer:
1. Are the shared SDPA bounds acceptable as the common test definition, with measured platform corrections made in the platform row?
2. Does the registration-derived native route expectation match the intended behavior of each FlagGems-only wheel?
3. Can Ascend hardware access be restored so the new shared cases can be measured and the current DNS evidence gap can be closed?

## Additional Context
The branch is `ci/unify-flaggems-tests` from fork `lvyufeng`, targeting `flagos-ai/Torch-FL:main`. This revision contains ten branch commits, the bounded unit-runner fix, and a merge of the latest main. The hardware jobs for the pushed commit determine the current platform status; incomplete or unavailable jobs are not presented as validated.

The PR base moved from `4596b7ba` to `5699bb03` while the last run was in progress. Platform support records retain their measured scope and mark unavailable evidence explicitly; the PR does not substitute routing configuration for a hardware measurement.
