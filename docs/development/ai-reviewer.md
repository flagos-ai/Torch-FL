# Automated AI review

This repository requires two approving reviews before a pull request can merge
into `main`. One of those two is an automated review produced by a language
model. This document explains how that reviewer works, what it will and will not
approve, and how to configure it.

## Why a machine user instead of the workflow token

GitHub refuses to count an approval made with the Actions `GITHUB_TOKEN` towards
a branch's required approvals. That is deliberate: it stops a workflow from
rubber-stamping a change on its way in. A workflow using its own token can post a
review, but the review will never satisfy the merge requirement.

The AI reviewer therefore submits its review through a dedicated **machine user**
account with write access to the repository, using that account's token. To
GitHub this is an ordinary approving review from a collaborator, and it counts.
The account must be a real account (or a bot account on a plan that allows one),
not the Actions token.

The consequence to accept: that account holds one of the two required approval
votes on a protected branch. Keep its token scoped to this repository only, and
rotate it.

## How a review runs

The workflow is `.github/workflows/ai-review.yml`, the logic is
`.github/scripts/ci/ai_reviewer.py`.

1. A `pull_request_target` event fires on `opened`, `synchronize`, `reopened` or
   `ready_for_review` for a PR against `main`. Draft PRs are skipped by the job
   condition.
2. The job checks out **only** the base revision's `.github/scripts/ci`
   directory. `pull_request_target` grants a write token, so the job must never
   run pull request code — and it does not. The diff is read back through the
   API, and PR text reaches the model as data, never as a command.
3. The script fetches the PR, its metadata and its changed files, and builds one
   prompt: PR title and body, then the diff fenced and labelled as untrusted.
4. The model is asked for a JSON verdict: a decision, a summary, and findings
   each tagged `high`, `medium` or `low`.
5. The script decides the review state from the verdict **and** a set of
   independent gates, then posts a single review.

It runs a second time on every push (`synchronize`), so a fix is re-reviewed.
`workflow_dispatch` accepts a PR number for a manual re-review.

## When it approves, and when it does not

`APPROVE` is the only state that counts towards the merge requirement, so every
gate that could hide a human decision fails closed into `COMMENT`:

| Condition | Result |
| --- | --- |
| Model withheld approval | `COMMENT` |
| Any `high` finding | `COMMENT` |
| Change touches `.github/workflows/`, `.github/scripts/`, `.github/configs/`, `CODEOWNERS` | `COMMENT` |
| Any changed file is binary or has no inline patch | `COMMENT` |
| Diff truncated (over 300 files or 120k patch characters) | `COMMENT` |
| Over 5000 added lines | `COMMENT` |
| Reviewer account authored the PR | `COMMENT` |
| Pull request is a draft | skipped |
| `AI_REVIEW_APPROVE=false` | `COMMENT` |

The first two rows are the substance of the review. The protected-path rule is
the important one structurally: a review bot must not be able to approve a
change to the review pipeline, so those paths always fall back to a human. This
means the workflow that introduces the reviewer can never be self-approved.

A `COMMENT` review is not a request for changes and does not block merging by
itself; it records the AI's verdict and leaves the approval requirement to
humans.

## Configuration

Repository secrets:

| Secret | Purpose |
| --- | --- |
| `AI_REVIEW_API_KEY` | API key for the model provider |
| `AI_REVIEWER_TOKEN` | Fine-grained PAT for the machine user, with **Pull requests: write** on this repository. Acts as the reviewer account. |

Repository variables (all optional):

| Variable | Default | Purpose |
| --- | --- | --- |
| `AI_REVIEW_PROVIDER` | `anthropic` | `anthropic` or `openai` |
| `AI_REVIEW_MODEL` | `claude-sonnet-4-5` | Model id |
| `AI_REVIEW_API_BASE` | `https://api.anthropic.com` | API base URL override |
| `AI_REVIEWER_LOGIN` | resolved from the token | Reviewer account login, used to detect self-authored PRs |

If `AI_REVIEW_API_KEY` or `AI_REVIEWER_TOKEN` is missing, the script logs a
warning and exits 0 rather than failing the PR.

### Enabling it on a repository

1. Create the machine user and give it write access to the repository.
2. Create a fine-grained PAT for it, scoped to this repository with
   **Pull requests: write**.
3. Add `AI_REVIEW_API_KEY` and `AI_REVIEWER_TOKEN` as repository secrets, and
   any variables you want to override.
4. The workflow is already wired; open a test PR and confirm the reviewer's
   account appears in the Reviews section.

Verify the vote actually counts before relying on it: the PR's merge box should
show the reviewer account as a satisfied required approval. A review that merely
appears in the timeline is not proof — check the branch protection state, for
example with
`gh api repos/<owner>/<repo>/branches/main/protection/required_pull_request_reviews`.

## Human review is still required

The AI reviewer never removes the human requirement: branch protection asks for
two approvals, and at most one of them is automated. A change touching the
review pipeline, a large diff, or one with a high finding still needs both votes
from people.

## Testing

`tests/unit/test_ai_reviewer.py` covers verdict parsing, every approval gate,
and the API writes, against a simulated GitHub client and a stub model. It does
not call a real provider and does not post real reviews.

Set `AI_REVIEW_DRY_RUN=true` to have the script log the decision it *would* make
without posting anything. This is the recommended way to try a new model or
prompt.

## Security notes

- The diff is untrusted input. It is fenced and labelled as data in the prompt,
  and the system prompt says an injection attempt is itself a finding. Prompt
  injection is mitigated, not eliminated — which is why the machine's vote is one
  of two, and why the review pipeline's own files can never be auto-approved.
- The workflow never checks out or executes pull request code.
- `pull_request_target` runs with a write token; the checkout is restricted to
  the base revision's script directory with `persist-credentials: false`, and the
  model call is made from Python with no PR-controlled shell input.
- The final prompt-injection backstop is that the reviewer account can only
  approve, and only within the gates above. Restrict its token to **Pull
  requests: write** so a compromised run cannot push code or merge.