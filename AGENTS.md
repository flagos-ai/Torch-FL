# Project conventions for AI coding agents

This file is the single source of guidance for AI agents working in this
repository. Most agents read `AGENTS.md` natively: Claude Code (v2.1.277 and
later), Codex, Cursor, Opencode, Zed, Copilot code review, Copilot cloud agent,
and VS Code Copilot Chat. `.github/copilot-instructions.md` is the one symlink
left, for the github.com Copilot Chat surface, which reads only that filename.

There is deliberately no `CLAUDE.md` and no `GEMINI.md`. Claude Code reads
`AGENTS.md` directly when no `CLAUDE.md` exists, so a symlink there would add a
second name for the same bytes and behave worse on a Windows checkout, where Git
turns a committed symlink into a plain text file unless `core.symlinks` is on.
Gemini CLI defaults to `GEMINI.md` and is not used in this repository.

## Language

**All GitHub-facing text must be written in English** — PR titles, PR descriptions,
commit messages, issue text, and code review comments. This repository's code,
comments, and existing history are all English, and PRs are read by contributors
who do not read Chinese.

**Everything under `docs/` must be written in English**, without exception —
design docs, plans, vendor integration notes, analysis write-ups. A doc drafted in
Chinese during a working session must be translated before it lands. The one
allowance is an explicitly localized top-level README (`README_zh.md`), which
exists as a translation of the English `README.md`.

Chat replies to the user in a session stay in whatever language the user is
using (usually Chinese). The rule is about what gets committed and published, not
about how you talk to the user.

If a PR description has already been opened in the wrong language, fix it with
`gh api -X PATCH repos/{owner}/{repo}/pulls/{n} -f body=@file` rather than
leaving it and noting the problem.

## Simplified workflow for routine tasks

**For routine development tasks, skip the full superpowers design workflow and implement
directly.** Only invoke brainstorming/writing-plans for tasks that genuinely require
upfront design:

- Complex feature additions with multiple valid architectural approaches
- Performance optimization requiring measurement and trade-off analysis
- Large-scale refactoring affecting many files or core abstractions
- Tasks where the user explicitly requests a design discussion

**Implement directly for:**
- Bug fixes with clear root cause
- Straightforward feature additions with obvious implementation
- Code cleanup and file reorganization
- Documentation updates
- Test additions
- Dependency updates

When implementing directly, still follow investigation-before-action (read relevant code
first) and verification-before-completion (run tests, check the build). The difference is
skipping the separate design phase when the path forward is clear.

## GitHub Templates and Issue/PR Guidelines

**When creating issues or pull requests, use the appropriate templates:**

### For AI Agents (Claude Code, Codex, Cursor, Copilot, etc.)

**READ THESE FIRST before filing any issue or PR:**
1. `.github/AI_AGENT_GUIDE.md` - Complete guidelines for AI agents
2. `.github/CLAUDE_CODE_GUIDE.md` - Claude Code specific instructions

**Required templates:**
- **Issues**: Use `.github/ISSUE_TEMPLATE/ai_agent_issue.md`
- **Pull Requests**: Use `.github/PULL_REQUEST_TEMPLATE/ai_agent_pr.md`

**Pre-submission validation:**
```bash
# Run this before creating any PR
python scripts/tools/validate_ai_pr.py --pr-body pr_description.md
```

**Mandatory requirements for AI PRs:**
- Root cause analysis (not just symptoms)
- Investigation process documented
- Actual test output pasted (not claims like "tests pass")
- Linting results shown (`ruff check`, `ruff format --check`)
- Edge cases identified and tested
- Human reviewer assigned
- Everything in English (enforced by validation script)

**Creating an AI PR:**
```bash
# After implementation and validation
gh pr create \
  --title "fix: <description>" \
  --body-file pr_description.md \
  --label "ai-generated"
```

**Mandatory PR completeness gate:**

Before creating or updating a PR, verify the complete diff against the intended
base branch and confirm that every intended file is present:

```bash
git diff --stat <base-branch>...HEAD
git diff --name-status <base-branch>...HEAD
gh api repos/{owner}/{repo}/pulls/{number}/files --paginate
```

Do not create or update the PR until all required files are included in the
branch diff. After any formatting or follow-up change, push the new commit and
re-check the PR file list through the API. The PR body must contain the actual
outputs of both `ruff check` and `ruff format --check`; a syntax check is not a
substitute when Ruff is available.

**PR body must be self-contained:**

The PR body must contain the full description text inline, not a reference to an
external file like `@pr_description.md`. File references are not rendered by
GitHub and leave the PR body effectively empty for human reviewers. When creating
or updating a PR:

```bash
# WRONG - leaves body as literal "@pr_description.md" text
gh pr create --body "@pr_description.md"
gh api -X PATCH repos/{owner}/{repo}/pulls/{n} -f body=@pr_description.md

# CORRECT - reads file content and passes it inline
gh pr create --body-file pr_description.md
gh api -X PATCH repos/{owner}/{repo}/pulls/{n} --field body=@pr_description.md
```

Use `--body-file` for `gh pr create` and `--field body=@file` (not `-f body=@file`)
for GitHub REST API calls. After updating, verify the body renders correctly with
`gh pr view {n} --json body -q '.body' | head -20`.

### For Human Contributors

Use standard templates:
- Bug reports: `.github/ISSUE_TEMPLATE/bug_report.md`
- Feature requests: `.github/ISSUE_TEMPLATE/feature_request.md`
- Platform support: `.github/ISSUE_TEMPLATE/platform_support.md`
- Operator requests: `.github/ISSUE_TEMPLATE/operator_support.md`
- Pull requests: `.github/PULL_REQUEST_TEMPLATE.md`

See `CONTRIBUTING.md` for detailed contribution guidelines.

### Upstream branch policy

The upstream repository may contain stable branches for supported PyTorch minor
lines, such as `2.9`. These stable branches are part of the release/support
contract and may be created or maintained on the upstream remote when explicitly
requested by the project maintainer. A stable branch must be named for a PyTorch
minor line and must not be used as a general-purpose development branch.

All ordinary development branches remain fork-only. This includes `docs/*`,
`feat/*`, `fix/*`, `perf/*`, `refactor/*`, `test/*`, `ci/*`, and experimental
work, even when the change is intended to land on a stable branch. The required
flow is:

```text
<contributor-fork>:<development-branch>
        -> PR -> <upstream>:main or <upstream>:<torch-minor-stable-branch>
```

Before pushing a branch:

1. Inspect `git remote -v` and classify the target as either an upstream stable
   branch or a contributor development branch.
2. For ordinary development, push only to the contributor fork (normally `origin`):
   ```bash
   git push origin <development-branch>
   ```
3. Create the pull request from `<fork-owner>:<development-branch>` into the
   intended upstream target, either `main` or an existing stable PyTorch minor
   branch:
   ```bash
   gh pr create \
     --repo <upstream-owner>/<upstream-repo> \
     --head <fork-owner>:<development-branch> \
     --base <main-or-torch-minor-stable-branch>
   ```
4. Creating or updating an upstream stable branch requires explicit maintainer
   authorization and must be limited to stable-branch setup or maintenance. Do not
   push ordinary commits directly to that branch; submit them through a fork PR.

Do **not** run `git push <upstream> <development-branch>` for ordinary development.
A successful push is not evidence that the push was appropriate. If a branch is
ambiguous, treat it as a development branch and use the contributor fork.

## Commit Message Format

All commits should follow this format:
```
<type>: <subject>

<body>

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
```

**Valid types:**
- `feat:` - New feature
- `fix:` - Bug fix
- `perf:` - Performance improvement
- `refactor:` - Code refactoring (no behavior change)
- `docs:` - Documentation changes
- `test:` - Test additions or modifications
- `ci:` - CI/CD changes
- `build:` - Build system changes

**Example:**
```
fix: resolve CUDA stream synchronization race in profiler

The CUPTI callbacks must be registered before the first CUDA context
is created. This change moves callback registration to module import
time, ensuring it happens before any CUDA operations.

Tested: pytest tests/integration/test_profiler.py - all pass

Fixes #123

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
```

## Code Quality Requirements

### Pre-commit Checks

Before committing any code changes:
```bash
# Linting (required)
ruff check
ruff format --check

# If format issues found, fix them:
ruff format

# Run relevant tests
pytest tests/unit/ -v
pytest tests/integration/ops/test_<relevant>.py -v
```

### Code Style

- Follow existing code style in the files you modify
- Match comment density of surrounding code
- Reuse existing utilities instead of reinventing
- Keep changes focused (avoid unnecessary refactoring)

### Operator integration on non-CUDA-compatible platforms

For accelerator platforms that are not CUDA-compatible, including Ascend and
Enflame GCU, operator integration **must be implemented through the platform's
code generator**. AI agents must extend the relevant operator registry,
category/template, and generated configuration instead of adding handwritten
per-operator kernels.

- Do not add a new handwritten operator implementation under
  `csrc/aten/backends/<platform>/` merely because it is faster to author.
- Regenerate the platform outputs and commit both the generator change and its
  generated artifacts.
- Run the generator a second time and require an empty diff to prove
  idempotency.
- A handwritten kernel is allowed only when the platform codegen cannot express
  the required runtime behavior. The PR must document that concrete limitation
  and receive explicit human approval before the handwritten implementation is
  added.
- CUDA-compatible platforms may follow their existing CUDA boxing/codegen path;
  this rule does not require vendor-specific native kernels for them.

## Testing Requirements

### Test Organization

- Unit tests: `tests/unit/` - Fast tests (< 1s each)
- Integration tests: `tests/integration/` - Full operator/model tests
- Use pytest marks to indicate platform requirements:
  - `@pytest.mark.anyplatform` - Runs on all platforms
  - `@pytest.mark.cuda` - CUDA specific
  - `@pytest.mark.ascend` - Ascend NPU specific
  - `@pytest.mark.flaggems` - FlagGems backend required

### Verification Before Completion

After implementing changes:
1. Run linting (ruff check, ruff format --check)
2. Run relevant unit tests
3. Run relevant integration tests
4. Manually verify the fix/feature works
5. Check that no debug code remains (no print statements, TODOs)

## Platform Considerations

When making changes, consider impact on all platforms:
- **CUDA** - Primary development platform
- **MetaX** - Boxing mode (reuses CUDA kernels)
- **Ascend** - ACL-based backend
- **PPU** - CUDA-compatible via PPU SDK

If a change only affects one platform, document why in the PR.

## Documentation

### Wheel compatibility guardrail

When changing wheel dependencies, platform defaults, SDK/toolkit selection, or
CI package versions, preserve the artifact contract in
`torch_fl/compatibility.json` (generated by `setup.py`). Do not hand-edit a
generated manifest or treat a CI-tested exact version as a universal supported
range. Set `FLAGOS_SDK_VERSION` only from verified build-environment evidence;
an unknown SDK must remain unknown. Every CI workflow that builds a wheel must
run `python scripts/tools/torch-fl-preflight --wheel ... --check-build-env` before installing
or publishing it. `tests/unit/test_wheel_preflight.py` enforces this workflow
rule. Keep the CLI under `scripts/tools/`, and package it as an installed script;
running it must not import `torch_fl` or load native libraries. Do not add a
root-level preflight module or package. For release artifacts, record
`FLAGOS_VENDOR_TORCH_VERSION` where vendor PyTorch supplies libraries, then run
the preflight with `--release` and the target `--sdk-version` before publication.

Update documentation when:
- Adding new features → Update README
- Changing APIs → Update docstrings
- Platform changes → Update platform-specific sections
- New conventions → Update this file (`AGENTS.md`)

All documentation must be in English.

### Operator support records

`docs/reference/operator-support.md` holds only the per-platform support counts.
A change that adds, enables, removes, disables, or reroutes an operator through a
vendor-native backend or FlagGems does **not** need to edit it: the counts are
read from the shipped `torch_fl/configs/backends_*.conf`, which is regenerated by
`scripts/codegen/gen_vendor_confs.py`. Refresh the numbers only when the user
asks, using `gen_vendor_confs.py --stats` and the conf tally, never by copying
old figures. Measured correctness evidence belongs in the PR and the survey
output, not in that page.

### Platform names: vendor first, chip only in source

A platform has two halves, and they are used at different layers.

**Vendor** — the company, and the `<lane>` of `flagos-pypi-<lane>`: `nvidia`,
`hygon`, `metax`, `mthreads`, `enflame`, `thead`, `iluvatar`, `kunlunxin`,
`ascend`, `tsingmicro`, `drobotics` (Nexus spells it `d-robotics`; identifiers
drop the hyphen), `apple`. Declared as `vendor` in
`cmake/flagos_platforms.json`. `FLAGOS_ACCELERATOR` accepts it.

**Platform id** — the key in the same table: `cuda`, `dcu`, `musa`, `gcu`,
`ppu`, `bpu`, `mps`, or the vendor name itself where they coincide. It is what
CMake, `setup.py`'s kernel-switch table and the CI runner configuration key on.
**Everything a reader names is the vendor**: the conf (`backends_<vendor>.conf`),
its route keys, the `register_flaggems_<vendor>.inc`, the `lib_<vendor>/` bundle,
the lane file, and the `ACCELERATOR` a wheel records. Both spellings of
`FLAGOS_ACCELERATOR` resolve to the same row; the platform id remains accepted
so older scripts and docs keep working.

The **wheel filename** keeps the SDK segment (`+cuda13.3`, `+dtk2604`,
`+maca3.8.1.3`, `+ppu2.0.0`): that is the build variant, it is what every
published `flagtree`/`torch_fl` artifact on `resource.flagos.net` uses, and the
lane already carries the vendor. `FLAGOS_BUILD_BOXING`-style switches and
`Backend` spellings follow the same rule: a name a user types resolves either
half; a name the tree produces is the vendor.

**Chip** — a specific part (`Ascend 310B`, `910B`, `A100`, …). Chip names are
**source-only**: a runtime predicate such as `csrc/aten/backends/ascend/
dtype_support.h`'s `IsFp16OnlyMatmulSoc`, keyed off the device's own SoC name.
They never appear in a conf, an accelerator id, a wheel name or a lane: one
wheel serves every chip of its platform, and the conf is selected before
`import torch`, when the device name is not yet readable. `tests/unit/
test_platform_vendor_naming.py` enforces this.

Do not introduce a third spelling. A new platform declares both a `vendor` and
an accelerator id, and the vendor must equal its `pypi_lane` where it publishes.

### Default confs never route boxing

Every default `torch_fl/configs/backends_<platform>.conf`, on every chip, may
only use these keys: `flaggems` / `flaggems_cpp`, `tileops`, the platform's own
vendor kernel name (`ascend`, `musa`, `gcu`, `kunlunxin`, `iluvatar`,
`tsingmicro`, `mps`, …), and `none`. It must **never** contain a `boxing` route —
nor the legacy `cuda` spelling of it.

- If FlagGems does not cover an operator and the vendor kernel is not wired up,
  the only allowed outcome is `none`, which reaches `cpu_fallback`. A `boxing`
  entry would hide the missing route behind the vendor's borrowed CUDA kernel and
  report coverage the vendor + FlagGems story does not have.
- A CUDA-ABI platform with no native operator library of its own (NVIDIA, MetaX,
  Hygon DCU, T-Head PPU on this tree) therefore has no vendor key in its default
  conf: everything FlagGems does not cover is `none`.
- `scripts/codegen/gen_vendor_confs.py` enforces this — `build_all()` exits if any
  default conf would emit `boxing`/`cuda`. Do not relax that check; if a route is
  wrong, fix the route.
- The only place a `boxing` key may appear is the opt-in day-0 file below.

### Day-0 boxing confs

`torch_fl/configs/backends_<platform>_boxing.conf` is a **temporary**,
boxing-only routing table for a CUDA-compatible platform — one that can
re-dispatch PrivateUse1 to CUDA through the vendor's own `libtorch` (via
`DeviceBoxingGuard`). Today that is NVIDIA (`cuda`), MetaX (`metax`), Hygon DCU
(`dcu`) and the T-Head PPU (`ppu`). Its only purpose is **day-0 model bring-up**: a
model whose ops FlagGems does not cover yet runs on the vendor's real CUDA-ABI
kernels instead of falling back to CPU per op, so bringing up a new model does
not wait on a FlagGems or vendor-native kernel for every op in it.

Rules for these files:

- **They are not the default.** Each platform's default conf stays
  `backends_<platform>.conf` (FlagGems-primary + measured gaps), and a wheel's
  import-time selection never picks a `_boxing` file. The user opts in explicitly:
  `FLAGOS_BACKEND_CONFIG=<install>/torch_fl/configs/backends_<platform>_boxing.conf`
  on a boxing build (`FLAGOS_BUILD_BOXING=1`).
- **Generated, never hand-edited.** They come from
  `scripts/codegen/gen_vendor_confs.py` (`BOXING_VARIANT_PLATFORMS`), same as the
  default confs; `--check` covers them. A hand edit is overwritten on the next
  regeneration.
- **FlagGems + boxing only.** The variant routes `flaggems` (and `flaggems_cpp`
  where the platform compiles that slot) and puts `boxing` behind everything else.
  Measured composite routes (for example `scaled_dot_product_attention` on DCU) are
  deliberately *not* applied — the file's premise is that FlagGems is not what a
  day-0 model is trusted to.
- **Do not extend the platform list casually.** Adding a row is a statement that
  the platform needs a day-0 escape hatch and a boxing wheel exists for it. It is
  not a way to make an operator test pass, and it is not a substitute for
  implementing the vendor or FlagGems route.
- **They go away.** Once a platform's vendor-native + FlagGems routes cover the
  day-0 models, its `_boxing` conf and its `BOXING_VARIANT_PLATFORMS` row are
  deleted. Do not build anything on their continued existence: no CI lane, no
  documentation, no test may depend on `backends_<platform>_boxing.conf` being
  present.

### The two CI lanes of a boxing platform

NVIDIA, Hygon DCU, MetaX and the T-Head PPU each run **two** integration lanes:
`<platform>` builds and tests the default (vendor + FlagGems) wheel, and
`<platform>-boxing` builds and tests the CUDA-boxing wheel of the same source.
The second lane is opted into by `boxing_lane: true` in `.github/configs/<p>.yml`
and always runs alongside the first; do not add a way to schedule one alone.

Rules for these lanes:

- **One manifest, two selections.** The boxing lane composes the same
  `.github/integration-manifest.yml` groups in the same order. A step whose
  selection differs per build is spelled per build, on the same entry: list the
  platforms in `boxing_variant` (`boxing_variant: [nvidia]`), and that entry then
  runs on their boxing build instead of their vendor build. `vendor_only: true`
  marks the other half -- a step that asserts routing only the vendor build can
  have. The composer runs one build per job; `validate_manifest` refuses a boxing
  platform that left one of these unscoped.
- **Two expectation blocks.** `.github/configs/<p>.yml` states
  `environment_check` for the vendor lane and `boxing_environment_check` for the
  boxing lane. The loader hands the boxing block to the environment step in place
  of the vendor one; the hook asserts the build record (`torch_fl/_build_config.py`)
  carries `boxing` in `KERNELS` on the boxing lane and not on the vendor lane.
- **The hook reads `BOXING_LANE`.** `set_env_<platform>.sh` sets
  `FLAGOS_BUILD_BOXING=1` and `FLAGOS_BACKEND_CONFIG=backends_<platform>_boxing.conf`
  only when `BOXING_LANE=1`, and the two venvs are separate
  (`torch-fl-<vendor>-<stage>-boxing`). Do not reintroduce a mode variable the
  build and the runtime both have to agree on.
- **Separate artifacts.** The boxing lane publishes as
  `wheel-<platform>-boxing` (`load-platform-config.yml` appends the suffix) so a
  release's two builds cannot overwrite each other.

### Dependency sources

FlagTree, FlagGems, and FlagCX are consumed as published wheels from
`https://resource.flagos.net/` (the `flagos-pypi-<vendor>` and
`flagos-pypi-hosted` lanes described in `docs/getting-started/installation.md`,
with the versions pinned in `.github/version-pins.env`). Do not build these from
source and do not pull their latest Git sources on your own initiative; that
changes the cohort under measurement and makes results incomparable with CI.
Build from source only when the user explicitly asks for it.

### Transient agent worktrees

Parallel agent sessions may create git worktrees under `.claude/worktrees/`.
That directory is gitignored and is scratch state, not source. Prune it when the
sessions that created it are done (`rm -rf .claude/worktrees`) so it does not
pollute recursive searches, globs, or code metrics run over the source tree, and
never commit anything from it.

## Related Documentation

- **CONTRIBUTING.md** - Detailed contribution workflow
- **README.md** - Project overview and build instructions
- **.github/AI_AGENT_GUIDE.md** - AI agent specific guidelines
- **.github/CLAUDE_CODE_GUIDE.md** - Claude Code integration guide
- **.github/TEMPLATES.md** - Template selection guide
