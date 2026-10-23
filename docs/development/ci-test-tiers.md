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
ordinary PR checks. Groups without that field run in both profiles. The two
CUDA transfer regressions require at least one passing test. Runtime platform
guards retain build-table ids such as `cuda`; CI config names use vendor names
such as `nvidia`. Source-only contracts run in the agnostic/codegen checks.

Groups run serially within a platform; separate platform jobs remain parallel.
Commands have a 30-minute default deadline, overridable with
`FLAGCICD_CASE_TIMEOUT`. Timeout and cancellation kill the command's process
group. `groups.json` retains failed and unstarted groups. Pytest reports retain
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

Release publication, candidate receipts, performance baselines, endurance and
model release acceptance are outside this PR. The existing release workflow
is unchanged. No whole-repository test-file registry or cross-repository
schema/adapter framework is required to add a functional test.
