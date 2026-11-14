# CI selection and maintainer controls

CI schedules configured hardware platforms through the shared integration
workflow. Runner labels, images and device access come from each platform config;
test groups come from the shared integration manifest.

## Runtime environment preparation

Ascend, Iluvatar, Kunlunxin, MUSA, PPU, GCU, MetaX and Hygon use the same preparation layout:
`.github/configs/<platform>.yml` selects the digest-pinned runtime and device
access; `.github/scripts/set_env.sh` dispatches to
`.github/scripts/hooks/set_env_<platform>.sh`; and
`.github/requirements-<platform>.txt` declares each isolated environment's
Python tools. Platform-specific preparation functions live in their hook.
Shared package reuse, installation and cache utilities remain under
`.github/scripts/seed_runtime.py` and `.github/scripts/lib/`.

Each hook creates a separate venv, selectively copies compatible RECORD-owned
wheels, checks the declared requirements and installs missing packages through
the shared `ensure_runtime_requirements` helper. It keeps satisfied packages,
uses the platform's selected pip index for gaps, and rechecks each installation.
The runtime supplies system compilers and its SDK. Child shells retain the
venv through `GITHUB_ENV`/`GITHUB_PATH`; image startup scripts cannot reactivate
vendor Python or add `/opt/flagtree` to `PYTHONPATH`. A fresh venv and a download
cache have separate lifecycles: cached downloads never skip the native build,
wheel validation or tests. CPU-only provisioning and cache regression tests run
in the shared platform-agnostic job.

The eight migrated platforms use `.github/constraints-runtime-ci.txt` as their
shared CI-only version policy. It includes the existing global constraints and
sets compatibility ranges for Python build tools, test plugins and common
runtime dependencies. Platform requirements select which packages are needed
and retain narrower restrictions, such as CMake below 4 where required. A
compatible installed or copied package is retained instead of upgraded to a
common patch version: SQLAlchemy 2.0.55 and 2.0.57 both satisfy the shared
`>=2.0.55,<2.0.58` range. Bounds are maintained explicitly; these are CI
provisioning limits, not changes to the published wheel's supported versions.

Every migrated hook uses `create_runtime_venv` in `lib/set_env_common.sh`.
It checks target/source/checkout boundaries before clearing the target, creates
the venv with the selected interpreter (or the image's `uv` fallback), then
requires pip and verifies interpreter ownership and disabled system-site
inheritance through the common environment checker. CI never reuses a prebuilt
venv or selects an old development-image provisioning branch. Explicit
`TORCH_FL_VENDOR_PYTHON` and `TORCH_FL_VENV_ROOT` inputs remain available for
the runtime interpreter and job-owned destination.

Shared helpers also own repeated build-tool imports, CPU-Torch identity and
FlagTree provider-origin checks. Hooks retain SDK discovery, platform-specific
package restrictions and vendor-library staging. GCU/MUSA continue to defer
FlagCX; removing old environment branches does not enable unsupported binaries.

Runtime copying, installed-package dependency checks and pip all honor the
same constraints, including constraints on transitive dependencies. Constraints
do not select additional packages for installation. `PIP_CONSTRAINT` is persisted
through `GITHUB_ENV` so later test-dependency and reporting installs use the same
policy. Environment reports retain the actual versions used; compatible patch
differences remain visible rather than silently treated as identical. The
constraints file participates in both integration and manual-wheel cache keys.
Other platform entrypoints retain the existing global constraints.

Manual and release wheel builds keep the `build` frontend prepared by migrated
runtime hooks. The later build-tool upgrade applies only to other platforms.

### Final test environment validation

After installing the built wheel, platform test dependencies and reporting
plugins, the migrated lanes run `seed_runtime.py --finalize-environment` before
the hardware suites. This verifies required CI tools and shared ranges, runs
`pip check`, checks installed distributions against the wheel manifest, and
records the final package/tool inventory, SDK/vendor-library provenance,
source ref and wheel SHA256. The earlier preparation snapshot is not treated
as the final test environment. Validation failures retain their JSON artifact
and count as installation failures in execution evidence.

When FlagCX is installed, an isolated subprocess imports `torch_fl` before
FlagCX, checks backend registration and rejects vendor Torch extensions or
Torch libraries loaded from `/flagos`. This import check does not replace
multi-rank collective tests and adds no source build. MetaX's two lanes reuse
the matching `0.14.0+maca3.8.1.3` package, or install that published pin, against
the vendor Torch libraries already bundled by those lanes.

GCU and MUSA explicitly defer FlagCX until compatible binaries are available.
Static inspection of their published 0.14.0 and 0.14.1 extensions found a hard
dependency on `libtorch_gcu.so` or `libmusa_python.so.2`, respectively; changing
the Python backend selector does not remove those ELF dependencies. Do not
copy those vendor extensions into the isolated CPU-Torch environment, build a
new FlagCX from source, or remove the artifact's declared requirements to make
this check pass.

The final gate accepts only the exact missing-FlagCX diagnostic for Torch-FL
on these two platforms. Missing unrelated dependencies, version conflicts,
checker failures, and missing FlagCX recorded as present at build time remain
fatal. The raw nonzero checker results and the deferred requirement remain in
the artifact: `valid` is false while `ready_for_tests` can be true under this
explicit functional-CI policy. This is not a claim that the published wheel's
complete dependency contract or communication support has been validated.
MetaX, Ascend and Hygon do not receive this exception.

The interpreter, compiler backend and SDK follow each chosen runtime, while CPU
Torch and published package pins retain their existing lane contracts. PPU retains
its two existing lanes: default uses CPU Torch without a vendor bundle; boxing
bundles vendor libtorch after the existing native build. No additional native
build, source-built Torch or hardware test-selection exclusions are introduced
by this migration. Hygon's [DTK 26.10 trial](../vendors/dcu/installation.md#dtk-2610-runtime-ci-trial)
explicitly pairs CPU Torch 2.10 with vendor Torch 2.13 device libraries. It
bypasses the existing version-equality gate, retains ABI/dispatch/computation
checks and the two existing hardware suites, and excludes the separate
full-vendor-core legacy smoke. This experimental cohort is not an accepted
cross-version runtime contract.

### Organization of environment checks

Environment preparation, installed-wheel validation, hardware-environment
validation and computation tests are separate stages. Hooks own SDK discovery,
build inputs and vendor-library staging. The final dependency gate owns package
ranges and the wheel contract. The first integration group runs
`check_platform_environment.py` for every platform; its expectations come from
the platform's `environment_check` or `boxing_environment_check` mapping.
Computation and runtime-contract suites retain their existing groups and coverage.

All eight migrated platforms enable `venv_isolation` in both applicable lanes.
It verifies the active venv, disabled system-site inheritance, package locations
and absence of runtime Python search paths. Optional native checks are driven by
configuration: exact Torch metadata/module versions, C++ ABI, loaded Torch
libraries, forbidden ELF dependencies, and the compiler target. Unknown keys or
malformed nested policies fail before accelerator imports. These are values in
YAML, not embedded Python or shell commands.

| Platform | Preparation/build checks retained | Hardware checks and functional owner |
| --- | --- | --- |
| Ascend | CANN/driver assets and isolated Python | Common environment group; existing operator suites |
| Iluvatar | CoreX assets/header syntax probe; shared isolation assertions | Common group audits CPU core mappings, ELF dependencies and CoreX target; the existing `iluvatar_runtime` group retains both add computations |
| Enflame | TOPS assets and isolated Python | Common environment group; existing native and FlagGems suites |
| Kunlunxin | XPU/XDNN assets and XPU compiler origin | Common environment group; existing native and FlagGems suites |
| MUSA | MUSA/mudnn assets and isolated compiler | Common environment group; the existing MUSA runtime-contract test retains driver binding and compile-target validation |
| PPU | SDK and boxing-only vendor bundle | Common lane-specific environment group; existing computation suites |
| MetaX | Vendor bundle and non-mutating Torch facade checks | Common lane-specific environment group; existing computation suites |
| Hygon | DTK/driver assets, bundle ABI and dispatch checks | Common lane-specific environment group; existing trial and hardware gates |

Loaded-core policies must describe the actual lane. Iluvatar expects the
official CPU core; PPU/MetaX vendor-core combinations and Hygon's decoupled
trial must not inherit that policy. This reorganization moves Iluvatar's
existing native assertions into the common checker; it does not invent
unverified loaded-library allowlists for other vendors.

The Iluvatar hook owns the CoreX preparation probe and calls shared assertions
from the common checker. The original two computations run through pytest in
`tests/integration/test_iluvatar_runtime.py`, after the common environment group,
just as MUSA's runtime contract lives under `tests/integration/`. No standalone
platform verifier or legacy CLI remains. No operator selection or failure
exception is added by this reorganization.

The checker records expected values, observed results and failure reasons in
`hardware-environment-<platform>.json`, retained with the existing pytest report
artifact even when the environment group fails. The earlier dependency snapshot
remains a separate pre-test artifact. A failed assertion still fails its group;
reporting cannot convert it into a skip or pass.

Configuration loading and policy validation participate in the same failure
reporting path: malformed or missing policy data fails before accelerator
imports and still retains the diagnostic JSON.

### Device access for the runtime-migration lanes

The eight migrated platforms use all devices made available to their runner;
CI does not impose a fixed card selection or device-count upper bound. Both
vendor and boxing environment policies require at least two visible FlagOS
devices so existing cross-device tests cannot silently lose their second card.
Where a lane also checks the CUDA-compatible interface, it uses the same minimum.
This is a functional prerequisite, not a device allocator or a claim that every
visible card is exercised. The current test commands do not automatically shard
work across additional cards.

Seven platforms use privileged-container access. Iluvatar now follows this
model and no longer masks physical devices 8 and 9. Ascend's redundant
per-card mappings are omitted because privileged access already exposes its
device nodes; its driver mount remains required. PPU is the runner-specific
exception: DSW authZ prohibits privileged containers, so it retains explicit
compute/control node mappings. Device access is established once when each job
container starts and remains unchanged throughout preparation and testing.

## Download caching for the runtime-migration lanes

Ascend, Iluvatar, Kunlunxin, MUSA, PPU, GCU, MetaX and Hygon use the same cache-selection helper,
`.github/scripts/lib/pip_cache.sh`, before their platform environment hooks run.
The integration workflow prefers the existing runner-local
`/__w/.pip-cache/<platform>` directory. It checks ownership, directory access
and an actual temporary-file write, because an existing directory can still be
unusable by pip in a container with a different UID.

If that directory cannot be used, the helper tries a persistent sibling with a
`-uid-<uid>` suffix, then a platform directory under `RUNNER_TEMP`, and finally
an owned temporary directory. It leaves the old cache and its owner/permissions
untouched. A selected directory is exported through `GITHUB_ENV` so subsequent
shells and ordinary pip commands use the same cache. MUSA no longer treats its
own extra cache-owner check as a fatal build requirement. If every directory
fails, CI emits a warning and explicitly disables caching for that job instead
of skipping the build or tests.

A matching dependency key retains downloads across jobs on the same runner.
When the key changes, the selected cache's contents are discarded before the new
key is written; the original sibling key files remain readable for migration
from the older layout. An unowned or unwritable old cache is not purged. The key
includes platform requirements, hooks and the shared helper, so these changes
can cause a one-time cold cache. Other platform lanes retain their existing
cache-selection behavior.

The manual/release wheel workflow still transports downloads through its
existing Actions cache restore/save steps; these runtime lanes use the same helper
to select a usable directory before that transport and recheck it at setup.
Normal PR integration jobs keep using runner-local storage.

`PIP_RETRY_NO_CACHE=1` explicitly disables caching, and the exported
`PIP_NO_CACHE_DIR` makes that choice apply to plain pip commands too. When
caching is enabled, `PIP_NO_CACHE_DIR` is empty: pip also treats `0` and `false`
as disabling values. FlagTree's existing per-install cache opt-out is unchanged.
These are download caches, not reusable build environments: interpreter and
package isolation, native builds, wheel validation and test execution still run.

## Automatic platform selection

| Event | Hardware selection |
| --- | --- |
| PR with `platform:all` | Every configured platform |
| PR with one or more `platform:<name>` labels | The union of the labeled platforms |
| PR with an unknown routing label | Every platform, conservatively |
| PR without routing labels | Platforms affected by the changed files |
| Documentation-only PR without routing labels | No hardware jobs; shared checks still run |
| Push or daily schedule | Every platform |
| Manual dispatch | `platforms=all` or a comma-separated platform list |

Routing labels take precedence over changed-file filtering, following upstream
policy. For example, a shared workflow change with `platform:nvidia` selects
NVIDIA and its boxing lane. Without routing labels, shared code or unavailable
diffs select every platform. The plan records the diff audit fields and the
label's `label_priority` selection policy. The gate validates every lane in
that selected matrix; it does not imply a full-platform run.
Adding or removing labels reruns PR CI. Supported platform names are `apple`,
`ascend`, `enflame`, `hygon`, `iluvatar`, `kunlunxin`, `metax`, `mthreads`,
`nvidia` and `thead`.

Each `.github/configs/<platform>.yml` declares `paths_watch`: exact file names,
directory prefixes ending in `/`, or shell-style globs. Its own configuration
and dedicated workflow automatically belong to that lane. Shared code or an
unrecognized code path selects all lanes; a recognized vendor change must not
hide an unrelated shared change in the same PR. Missing or invalid watch
metadata also selects all lanes. A failed Git comparison selects all lanes.

`seed_runtime.py` is shared provisioning infrastructure. Its changes follow the
same conservative full-matrix policy as `set_env.sh` and `set_env_common.sh`;
do not list it as a vendor-owned watch path. Explicit routing labels retain
their existing precedence over file filtering.

Selection compares the immutable PR base and head commits and includes both
paths when files move between platforms. Watch lists include shared consumers:
CUDA runtime sources affect CUDA, DCU and PPU, while the CUDA allocator header
also affects MetaX. Keep these relationships aligned with
`cmake/flagos_platforms.json` and the runtime build configuration when adding or
changing a platform.

## Functional profiles

The default `pr` profile preserves the platform selection above. The daily
`nightly` profile runs every configured platform and all applicable existing
functional groups, including the dedicated CUDA transfer regressions. Manual
nightly requests require `platforms=all`. `weekly` is reserved and rejected
before hardware dispatch. Both CI and FlagCICD Entry expose these profile names.
See [Functional CI profiles](ci-test-tiers.md) for execution and evidence rules.

## One required check

The top-level job named **CI complete** runs after the shared checks and every
hardware pipeline. Shared checks and every selected platform must succeed.
Unselected platforms may skip. A selected platform that unexpectedly skips,
fails or is cancelled blocks the gate, as does missing or invalid discovery
output. A documentation-only PR can pass after its shared checks succeed.

A repository administrator can make **CI complete** the required status check
in the branch protection rule or ruleset. Adding the job does not change that
repository setting automatically. Do not substitute individual platform job
names for this aggregate check.

## `/ci` comments

Repository writers, maintainers and administrators can post a command on the
first line of a PR comment:

```text
/ci run
/ci run all
/ci run ascend
/ci run ascend cuda
/ci run ascend,cuda
```

The command validates platform names, keeps unrelated labels, replaces routing
labels and dispatches `ci.yml` on the base repository's default branch. That
trusted workflow resolves the open PR's current head repository and immutable
commit; every shared check, hardware checkout and manifest loader receives the
same source. This supports fork PRs without loading workflow definitions from
the fork. Authoring a PR alone does not authorize this privileged hardware
dispatch route.

A valid command is acknowledged only after dispatch succeeds. Invalid commands
from authorized users receive a usage response. Closed PRs and deleted fork
heads cannot dispatch. Separate concurrency groups keep a requested PR run from
cancelling main, nightly CI or another PR's requested run.

Requested runs provide diagnostics for the PR head. Their check run belongs to
the dispatch ref; they do not replace the ordinary PR's required **CI complete**
check or publish a synthetic passing check on the PR commit.

## Bounded network retry

`flaky-retry.yml` listens for completed, failed runs of `ci.yml`. It checks the
actual run and all failed root jobs using the GitHub API. It excludes the
dependent **CI complete** summary from root-cause classification.

A platform outcome job succeeds when it successfully describes an upstream
hardware failure; that failure remains in the report and blocks the shared gate.
Report validation, generation or upload failures still fail the outcome job and
remain subject to the same root-failure policy. This keeps a dependent report
from blocking an otherwise eligible network retry without ignoring real report
faults by job name.

Only recognized network failures are retried: for example pip read timeouts,
DNS failures, connection resets, selected curl transport failures and HTTP
502/503/504 responses. Every failed root job must qualify. Assertion, import,
compiler, missing-package and OOM failures take precedence over network noise.
Unknown failures, driver faults, collective hangs, plain timeouts, unavailable
logs and cancellations remain failed for investigation.

The policy requests `gh run rerun --failed` at most once, including dependent
jobs so the aggregate gate is recalculated. It checks the current attempt again
immediately before requesting a rerun; a duplicate event or manual rerun must
not consume another retry. GitHub CLI calls have a 120-second limit and the
supplemental workflow has a five-minute limit. Retry does not turn a failed CI
result into a pass: the rerun must succeed. No AI service credentials are needed.

Privileged control workflows checkout scripts from their own trusted event
commit and never execute a failed PR's scripts or downloaded artifacts. Both
[`issue_comment`](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#issue_comment)
and
[`workflow_run`](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#workflow_run)
require their workflow definitions to exist on the default branch.

## Validation and rollout

### Implementation discipline

The existing integration PR tracks upstream main and its CI results. Planning
changes are developed on a dependent branch and reviewed in stages. When the
integration branch advances, update the dependent branch and recheck its own
delta; after the integration PR merges, align the planning branch with main.

For each stage, audit existing CI capabilities before adding code. Reuse the
router, composer, configuration and reporting facilities wherever they fit.
Existing facilities may be corrected or simplified when evidence shows a defect
or an unreasonable design; explain the impact and add relevant regression checks.
Keep one authoritative implementation for each responsibility. Observation beside
the existing execution path is transitional: when plan consumption and shared
gating take over, remove the superseded decision paths rather than retaining
parallel routing, composition or verdict implementations. New abstractions must
solve a demonstrated need. Each stage must include behavior checks and cleanup;
test exclusions or weaker gates must not hide defects.

### Executable plans

Ordinary CI generates its matrix and final execution plan together in
`discover-platforms`. FlagCICD Entry generates and validates the same plan shape
in `plan`. Both retain `flagcicd-plan-<producer-attempt>` and forward its exact
artifact name and compact identity to `integration-tests.yml`. Plan creation and
upload must succeed before hardware can start. The transitional observation job
and the external entry's repeated preflight matrix have been removed; full
configuration validation now happens in the plan job, including L0/L1 requests.

`platform_config.py` is the shared platform parser. It validates scheduling and
build fields, invokes the existing manifest composer once per validated lane,
and snapshots the final runner/image/mount/timeout/setup outputs, shared test
environment and ordered groups. Standalone single-platform integration and wheel
builds compile with this same parser. There is no second group catalog or
executor, and no YAML parser is required in vendor images. `group_id` reuses the
composer's `manifest_id`; commands, group environments and `min_passed` are
preserved. Shell variables remain unexpanded and secrets are not copied.

For a planned lane, the CPU configuration job downloads that artifact and only
consumes its snapshot. It does not read platform YAML or recompose tests. It
checks the plan hash, source SHA, trusted control SHA, profile, current run and
requested lane against the producer's compact context. Missing, modified or
mismatched plans fail before hardware scheduling; they never trigger an implicit
standalone compilation. Trusted control scripts are checked out at
`github.workflow_sha` before source data is checked out. CPU planning and
configuration jobs use sibling `control/` and `source/` directories; a path or
symlink submitted inside the source tree cannot redirect the trusted checkout.
Configuration outputs remain relative to the hardware job's source checkout.

One planning call reads each platform file once. Routing, boxing selection,
compilation and configuration fingerprinting reuse that call's parsed data and
original bytes, without a process-global cache. Each lane receives its own copy
so default and boxing expectations cannot mutate one another. Runner labels must
be nonempty strings, including after whitespace is stripped.

A failed-job rerun may reuse the original producer's artifact in the same run;
the consumer still checks its exact context and hash, and rejects a future
producer attempt. A rerun that regenerates the plan passes the new artifact name
to its consumers. No cross-run artifact lookup is used.

The router's selection policy is unchanged. Documentation PRs keep an empty
hardware matrix and all ordinary shared checks. External requests retain their
L0/L1/L2 policy and explicit L2 empty-selection fallback. Regression tests run
conditionally when present in the tested source checkout.

`plan_hash` binds source repository/commit, comparison base and diff head,
control-code commit, profile, configuration identity, `changed_files`,
`matched_rules`, selection reasons and final execution content. Modifying either
audit field invalidates the artifact's hash. The potentially large audit arrays
stay in the artifact; compact contexts carry the hash instead of duplicating
them. `source_sha` is the checked-out commit; `diff_head_sha` may be
the PR head when testing a merge commit. Run, attempt, task and parent IDs are
excluded from the deterministic hash and run identity is checked separately.
Large commands and explanations stay in the artifact, outside compact contexts.

### Planned evidence and one gate

Both entries pass one `execution_plan_context` to the shared integration
workflow. The executor records stable group IDs, configured shared/group
environments, commands, outcomes and the producer's context in `groups.json`.
The pytest plugin records the same group, lane and context for each session;
it removes these reporting settings from the pytest process environment so
nested fixtures cannot pollute the parent lane's evidence.

The existing merger retains selected and observed node inventories. Its shared
`validate_execution` validator checks exact group IDs and order, commands,
environments, pytest requirements, command outcomes, session identities and
counts. The CPU platform reporter supplies the independently generated planned
group list to that same validator. A self-consistent execution inventory cannot
hide a group omitted from the plan's required execution.

`flagcicd_report.py` now handles both `CI complete` and `FlagCICD complete`.
Each gate downloads the exact full plan and platform outcomes, validates the
artifact against its compact context/current run/trusted control commit, then
checks the plan's `expected_shared_checks` and expected lanes. Report identities
include `plan_hash`, `control_sha`, source SHA, run, profile and configuration
hash. The summary rechecks each passing child's group/session evidence instead
of trusting its `passed` label. Missing, duplicate or extra lanes/groups,
modified commands/environments, missing sessions, zero collection, contradictory
counts and wrong identities cannot pass. Failed and unstarted groups remain in
the layered outcome alongside their raw execution evidence.

A report-only rerun may reference an earlier successful hardware attempt. The
execution attempt must equal the hardware job's recorded output, while the
report attempt can be newer. A rerun of the hardware job cannot borrow old
execution evidence. The summary retains the latest matching report per lane,
allowing unaffected lanes from earlier attempts of the same plan to remain
valid. Stale reports from another plan never supply missing current lanes.

### Functional evidence and conditional skips

Successful wrappers cannot hide failed pytest sessions or failed/error counts.
The existing manifest `min_passed` is enforced per group. The executor's early
minimum check and final gate read the same native pytest sessions and share the
inventory/count and minimum validators; console verbosity cannot change the
result. A minimum without structured session evidence fails closed. Groups without a
positive minimum retain their existing test-level capability/route skip rules:
a nonempty, fully accounted-for skipped session can be valid when another group
passed functional tests. Missing sessions and zero collection are not skips;
an entirely skipped lane cannot pass. No blanket per-group minimum, new skip
exemptions or workload exclusions are introduced. Existing explicit vendor
teardown tolerance still requires clean pytest evidence and a permitted fatal
signal; assertion failures cannot use it.

Coverage remains optional, per platform, and bound to the same plan context.
Incomplete or mismatched coverage is unavailable with a warning. Backend upload
failures do not replace the functional result; missing functional proof does.
The ordinary entry retains bounded paginated job diagnostics in the shared
reporter; failure to fetch optional diagnostics cannot change the gate verdict.

When an upstream shared check prevents ordinary CI from generating a plan, the
summary retains that check's failure/cancellation category. A missing plan is
still an error, but does not imply that successful upstream checks should have
been skipped. Successful platform reports reuse the shared validator's group
summary; diagnostic summaries are generated only for unsuccessful evidence.

Standalone integration dispatches without a plan keep their existing local
functional evidence check. Planned runs use the shared CPU reporter and gate;
the superseded ordinary inline verdict and duplicate context input are removed.
Older execution artifacts without stable IDs and plan identity cannot be treated
as passing evidence by the new gate. Real-runner rollout must test a source
commit containing this reporting protocol; no compatibility success is inferred.
Local simulations do not replace real-runner acceptance.

### Shared CPU check ownership

`agnostic-checks.yml` owns the stdlib-only vendor-conf generator's `--check` and
`test_gen_vendor_confs.py`. `codegen-checks.yml` owns the remaining platform code
generators and registration consistency checks that share its CPU-torch setup.
Both ordinary CI and the external entry retain their required shared-job gates.
A standalone codegen dispatch has this narrower scope; use full CI for both.

Explicit test files and generator checks must not be listed in both CPU
workflows. The routing contract tests check unique ownership and preserve the
required dependencies. This policy does not remove device or installed-wheel
checks: identical filenames on default/boxing or different vendor environments
can represent different validation. Reducing those checks requires evidence of
equivalent environments and behavior, not a filename-based exclusion list.

Platform-agnostic CI runs regression tests for selection, routing, required gate
results, checkout resolution, comment dispatch, network classification and retry
budgets. GitHub operations in these tests are simulated; tests do not post real
comments or start hardware runs.

Before considering rollout verified, run hardware CI, configure the required
check, then exercise comment dispatch for an actual fork PR and observe a real
network-failure retry. The local tests and Actionlint checks cannot verify runner
availability, repository write permissions or GitHub's event delivery.
