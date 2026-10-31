# CI selection and maintainer controls

CI schedules configured hardware platforms through the shared integration
workflow. Runner labels, images and device access come from each platform config;
test groups come from the shared integration manifest.

## Automatic platform selection

| Event | Hardware selection |
| --- | --- |
| PR with `platform:all` | Every configured platform |
| PR with one or more `platform:<name>` labels | The union of those platforms |
| PR with an unknown routing label | Every platform, conservatively |
| PR without routing labels | Platforms affected by the changed files |
| Documentation-only PR without routing labels | No hardware jobs; shared checks still run |
| Push or daily schedule | Every platform |
| Manual dispatch | `platforms=all` or a comma-separated platform list |

Routing labels take precedence over file filtering. Adding or removing labels
reruns PR CI. Supported platform names are `apple`, `ascend`, `enflame`, `hygon`,
`iluvatar`, `kunlunxin`, `metax`, `mthreads`, `nvidia` and `thead`.

Each `.github/configs/<platform>.yml` declares `paths_watch`: exact file names,
directory prefixes ending in `/`, or shell-style globs. Its own configuration
and dedicated workflow automatically belong to that lane. Shared code or an
unrecognized code path selects all lanes; a recognized vendor change must not
hide an unrelated shared change in the same PR. Missing or invalid watch
metadata also selects all lanes. A failed Git comparison selects all lanes.

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
control-code commit, profile, configuration identity, selection reasons and final
execution content. `source_sha` is the checked-out commit; `diff_head_sha` may be
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
