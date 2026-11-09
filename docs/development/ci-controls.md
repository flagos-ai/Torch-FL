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

### Functional evidence validation

The pytest merger, the ordinary hardware evidence check and the external platform
reporter share `validate_pytest_result`. Successful outer commands cannot hide a
nonzero pytest session exit code or failed/error counts. Conditional skipped
sessions remain valid when other functional tests passed; existing explicit
teardown-crash handling remains separate from assertion failures.

The ordinary gate reads paginated job diagnostics as a JSON array, with a bounded
API query. Optional diagnostic failures do not replace the required job verdicts.

Platform-agnostic CI runs regression tests for selection, routing, required gate
results, checkout resolution, comment dispatch, network classification and retry
budgets. GitHub operations in these tests are simulated; tests do not post real
comments or start hardware runs.

Before considering rollout verified, run hardware CI, configure the required
check, then exercise comment dispatch for an actual fork PR and observe a real
network-failure retry. The local tests and Actionlint checks cannot verify runner
availability, repository write permissions or GitHub's event delivery.
