# Functional CI profiles

This CI verifies the installed Torch-FL wheel on the configured accelerators.
It reuses the existing diff router, platform configs and integration manifest.

| Profile | Hardware selection | Tests |
| --- | --- | --- |
| PR | Changed-file routing, with existing label overrides | Existing applicable functional groups |
| Nightly | Every configured platform, regardless of the diff | All existing applicable groups plus CUDA transfer regressions |
| Weekly | Reserved | Rejected before hardware dispatch until a workload is defined |

The daily nightly runs at 02:00 China Standard Time. A manually requested
nightly requires `platforms=all`. Ordinary push events keep the existing full
platform matrix. Shared lint, source and codegen checks still run for PRs.
Only the hardware matrix is narrowed by the diff router.

`profiles: [nightly]` on a manifest group keeps optional CUDA profiling out of
ordinary PR checks. Groups without that field run in both profiles. The CUDA
FlagGems route regression requires at least one passing test. The dtype-cast
performance regression reports a skip when PyTorch has no `libtorch_cuda.so`:
that build intentionally casts through the host, and portable cast correctness
is still checked in the functional groups. Runtime platform
guards retain build-table ids such as `cuda`; CI config names use vendor names
such as `nvidia`. Source-only contracts run in the agnostic/codegen checks.

The Hygon per-file unit runner reads those workflows' explicit, unfiltered
pytest file lists with `--exclude-shared-cpu`, so CPU-owned checks are not
repeated on hardware. The manifest retains `test_startup_subprocess.py` and
`test_vendor_libtorch_overlay.py`: they also contain installed-wheel checks
that the CPU environment cannot execute. Filtered pytest selections do not
claim a whole file. Test directories and device-dependent unit files stay in
place; direct invocations without the option still run every matching file.

CPU Torch installation may choose a faster mirror. If that installation fails,
the existing retry helper falls back to the pinned primary index with the same
version and options. An explicitly empty `TORCH_FL_CPU_TORCH_MIRROR_INDEX_URL`
disables mirror selection; a failed primary install remains a failure.

Groups run serially within a platform; separate platform jobs remain parallel.
Commands have a 30-minute default deadline, overridable with
`FLAGCICD_CASE_TIMEOUT`. Timeout and cancellation kill the command's process
group. A dedicated Linux child subreaper also retains workers that detach and
outlive their launcher, so they remain reachable for cleanup. Other POSIX hosts
retain process-group cleanup. The output reader shares the command deadline
instead of blocking on an inherited pipe. `groups.json` retains failed and
unstarted groups. Pytest reports retain
selected node ids, skips and failures. Missing sessions, missing selected test
results, incomplete execution or an entirely skipped run cannot pass the
functional evidence check. Existing documented vendor teardown handling stays
explicit in the runner and does not excuse failed assertions.

Hardware identity is retained with the test artifacts for diagnosis. An unknown
SDK label alone does not reject functional tests. The existing platform
preflight still checks the actual device and runtime requirements.

CUDA functional CI keeps FlagCX communication optional through `cuda-comm`.
It does not run a global `pip check` or claim collective-communication support.
Other platforms retain their existing dependency setup.

Release publication continues through the existing wheel-build and publishing
workflows, with platform names and boxing-mode propagation kept consistent with
CI. Candidate receipts, performance baselines, endurance and model release
acceptance are outside this PR. No whole-repository test-file registry or
cross-repository schema/adapter framework is required to add a functional test.
