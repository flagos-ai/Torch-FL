# Functional CI profiles

This CI verifies installed Torch-FL wheels using the platform configs and
integration manifest.

| Profile | Hardware selection | Tests |
| --- | --- | --- |
| PR | Every configured platform, including the ten required lanes | A common regression suite plus each platform's applicable groups |
| Nightly | Every configured platform, regardless of the diff | All existing applicable groups plus CUDA transfer regressions |
| Weekly | Reserved | Rejected before hardware dispatch until a workload is defined |

The daily nightly runs at 02:00 China Standard Time. A manually requested
nightly requires `platforms=all`. Ordinary push events run the full matrix.
PR acceptance requires the common suite to pass on all ten lanes for the
same revision. Shared lint, source and codegen checks are also required.

`profiles: [nightly]` on a manifest group keeps optional CUDA profiling out of
ordinary PR checks. Groups without that field run in both profiles. The CUDA
FlagGems route regression runs on the NVIDIA vendor lane and requires five
passing cases, including the measured route and correctness cases; its optional
FlagGems device-name probe may skip on versions without that constant.
The ordinary NVIDIA wheel runs portable copy correctness checks. Transfer-free
dtype casts and strided copies are outside the scheduled coverage. The FlagGems
route regression counts `gpu_memcpy` activities independently of display names
and requires a known host-transfer positive control. Runtime platform guards
use build-table IDs such as `cuda`. CI configs use vendor names such as
`nvidia`. Source-only contracts run in the agnostic and codegen checks.

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

The wheel-build and publishing workflows use the same platform names as
functional CI.
