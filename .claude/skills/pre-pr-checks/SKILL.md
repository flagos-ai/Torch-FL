---
name: pre-pr-checks
description: >
  Run the checks CI enforces before opening or updating a pull request on
  torch_fl. Use this whenever you are about to `gh pr create`, push to a PR
  branch, or amend a commit that is already on a PR — and when a PR's `lint /
  Lint` job has gone red. Covers: rebasing onto the latest flagos/main first,
  the exact pinned ruff version CI uses, the two ruff commands that gate every
  job downstream, which interpreter to run them with on vendor boxes, and how
  to update an existing PR afterwards.
---

# Pre-PR checks (torch_fl)

## Order of operations

Every PR, no exceptions:

1. **Rebase onto the latest `flagos/main`** — fetch first, then rebase, then
   resolve conflicts. See below; this is the step that actually bites.
2. **Lint locally** with the pinned ruff (`ruff check .`, `ruff format --check .`).
3. **Run the tests** covering the change.
4. *Then* `gh pr create` / `git push`.

Doing 2 before 1 wastes the run: a rebase can reintroduce lint errors or pull in
regenerated files, so lint has to come after.

## Step 1: rebase onto the latest flagos/main

**Always `git fetch flagos main` before you rebase.** A local `flagos/main` ref
is a snapshot from whenever you last fetched, and this repo moves fast — on one
occasion ten commits landed in the two days between a fetch and a PR. Basing a
PR on a stale ref produces a diff full of conflicts that have nothing to do with
your change:

```bash
git fetch flagos main
git log --oneline -1 --format="%h %cd %s" flagos/main   # check the date, not just the sha
git rebase flagos/main
```

### Check whether your commits already landed upstream

The most confusing failure mode is a PR that conflicts with *itself*: a local
commit was merged upstream through an earlier PR (usually squashed, so the sha
differs and git cannot match them up), and the rebase tries to apply it a second
time.

```bash
git log --oneline <old-base>..flagos/main    # look for your own commit subjects
```

If you find one, drop that commit from the branch rather than fighting the
conflict — the upstream copy is authoritative.

### Regenerate rather than merge generated files

For conflicts inside generated artifacts (`csrc/aten/generated/*`,
`torch_fl/configs/backends_*.conf`), do not hand-merge conflict markers. Take
the upstream side, port only the *generator* change, and re-run codegen:

```bash
git checkout flagos/main -- csrc/aten/generated/ torch_fl/configs/
FLAGOS_CODEGEN_ALL=1 /usr/bin/python3 scripts/codegen/codegen_ops.py
```

Then confirm idempotency — a second run must produce no diff:

```bash
FLAGOS_CODEGEN_ALL=1 /usr/bin/python3 scripts/codegen/codegen_ops.py
git diff --quiet && echo "idempotent" || echo "generator is NOT idempotent"
```

### Do not clobber upstream changes to files you also touched

`git checkout <your-commit> -- <file>` on a hand-written file silently discards
whatever upstream did to it. Apply your side as a patch so a real conflict is
surfaced instead of swallowed:

```bash
git diff <base> <your-commit> -- path/to/file.py > /tmp/change.patch
git apply --3way /tmp/change.patch     # conflicts to resolve, not to lose
```

When resolving, keep **both** sides unless they genuinely contradict.

## Why the lint step exists

`.github/workflows/ci.yml` runs `lint` **first** and every other job declares
`needs: lint`. A formatting slip therefore does not just fail one check — it
skips the build and integration jobs entirely, so a red lint tells you nothing
about whether the actual change works. Lint is cheap (seconds) and it gates
everything, so run it before you push, not after CI complains.

## The gate

`.github/workflows/lint.yml` is the whole contract. Two commands, one pinned
tool version:

```bash
pip install ruff==0.15.12       # pin matters: formatting rules shift between
                                # ruff releases, so an unpinned local ruff can
                                # disagree with CI in both directions
ruff check .                    # lint rules
ruff format --check .           # formatting
```

Both must pass. Run them from the repo root — the config lives in
`pyproject.toml` and the paths are relative.

To fix rather than just report:

```bash
ruff check --fix .
ruff format .
```

Then re-run the two `--check` forms to confirm, and re-run any tests covering
files that were reformatted. Formatting should be semantically inert, but
confirming costs one command.

## Interpreter caveat on vendor boxes

On machines where the vendor torch lives in the **system** interpreter rather
than whatever `python` resolves to (Hygon DCU/DTK is one — bare `python` there
is a conda install with no torch), spell the interpreter out or you will
install ruff into the wrong environment and get "No module named ruff":

```bash
/usr/bin/python3 -m pip install ruff==0.15.12
/usr/bin/python3 -m ruff check .
/usr/bin/python3 -m ruff format --check .
```

## Reading a red lint job

`ruff format --check` reports only a count ("1 file would be reformatted") —
it does not name the file in the CI summary. Get the specifics locally:

```bash
ruff format --check .                        # names the files
ruff format --diff path/to/file.py           # shows the exact required change
```

Or pull the CI log directly:

```bash
gh run view <run-id> --repo <owner>/<repo> --log-failed
gh pr checks <pr-number> --repo <owner>/<repo>
```

## Updating a PR after fixing

This repo's PRs land as a **single commit**. Fold lint fixes into the existing
commit rather than stacking a "fix lint" commit on top:

```bash
git add <files>
git commit --amend --no-edit
git push --force-with-lease origin <branch>
```

`--force-with-lease` (not `--force`) so the push aborts if someone else has
pushed to the branch meanwhile. Force-pushing is appropriate here because the
branch is your own PR branch; it is not appropriate for shared or upstream
branches.

## Checklist

Before `gh pr create` or any push to a PR branch:

1. `git fetch flagos main`, then `git rebase flagos/main` — conflicts resolved,
   and any commit that already landed upstream dropped
2. `git log --oneline flagos/main..HEAD` — only the commits you mean to ship
3. `ruff check .` → "All checks passed!"
4. `ruff format --check .` → "N files already formatted", nothing to reformat
5. Tests covering your change still pass (and re-run any file ruff reformatted)
6. If codegen ran: a second `FLAGOS_CODEGEN_ALL=1` run leaves no diff
7. `git diff --stat flagos/main` — confirm the diff contains only what you
   intended, especially when generated files are involved
