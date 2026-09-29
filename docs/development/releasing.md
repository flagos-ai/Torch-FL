# Releasing

A release is a tag. Push `v2.10.0` and CI builds one wheel per platform and
uploads each to the vendor PyPI lane it belongs to.

```bash
git tag v2.10.0 <commit>
git push upstream v2.10.0
```

## What the tag starts

`.github/workflows/release.yml` runs on `v*` tags. It:

1. **Checks the tag is a release.** A version tag publishes, and so does a
   release candidate (`v2.10.0rc1`, `v2.10.0a1`, `v2.10.0b1`) — a candidate
   nobody can install cannot be tested, which is the only reason to cut one. A
   variant tag (`v2.10.0+cuda13.3`) builds nothing and says so, because the SDK
   it names is already in the wheel's filename. This mirrors the guard in
   `flagos-ai/build-infra/.github/workflows/upload-nexus.yml` for the stable
   case, so a tag that would be skipped there is not published here by accident
   either. `workflow_dispatch` with a `tag` input always proceeds — the escape
   hatch for a deliberate re-run. Dispatch it **from the tag's ref** (the "Use
   workflow from" selector), not from a branch: the wheels are built from the
   run's own ref and the candidate suffix comes from it too, so a branch dispatch
   would build a branch tip under the tag's name. The guard rejects that in
   seconds rather than after seven builds.

   The candidate suffix reaches the artifact through the tag: the build reads
   `GITHUB_REF_NAME` and exports `FLAGOS_WHEEL_PRERELEASE`, so `v2.10.0rc1`
   produces `torch_fl-2.10.0rc1+cuda13.3`. PEP 440 sorts that below `2.10.0`, so
   `pip install torch_fl==2.10.0` does not pick the candidate up. The uploader
   then requires the tag and the wheel to name the same version, which is what
   stops a candidate being published under the release's name or the reverse.
2. **Builds and uploads on every platform**, one job each, on the platform's own
   runner. The build is the same one CI runs: the `build-wheel-*.yml` wrapper (or
   `all-tests-common.yml` for GCU, which has no wrapper), the same image and
   setup hook, and the same `Verify wheel` step that installs the fresh artifact
   before it is offered to the uploader. The upload happens in that same job, so
   the wheel never leaves the platform's own network.
3. **Uploads each wheel to its lane** with
   `.github/scripts/publish_wheels.py`, run by `build-wheel-common.yml` on the
   build runner. It reads the target from the `pypi_lane` column of
   `cmake/flagos_platforms.json`:

   | Platform | Lane | Wheel |
   |---|---|---|
   | cuda | `flagos-pypi-nvidia` | `torch_fl-2.10.0+cuda13.3-cp312-cp312-linux_x86_64.whl` |
   | dcu | `flagos-pypi-hygon` | `torch_fl-2.10.0+dtk2604-cp310-cp310-linux_x86_64.whl` |
   | gcu | `flagos-pypi-enflame` | `torch_fl-2.10.0+tops1.9.10-cp312-cp312-linux_x86_64.whl` |
   | musa | `flagos-pypi-mthreads` | `torch_fl-2.10.0+musa5.2.0-cp310-cp310-linux_x86_64.whl` |
   | ppu | `flagos-pypi-thead` | `torch_fl-2.10.0+ppu2.0.0-cp312-cp312-linux_x86_64.whl` |
   | ascend | `flagos-pypi-ascend` | `torch_fl-2.10.0+cann9.0.0-cp311-cp311-linux_aarch64.whl` |
   | metax | `flagos-pypi-metax` | `torch_fl-2.10.0+maca3.8.1.3-cp312-cp312-linux_x86_64.whl` |

   TsingMicro and BPU do not publish.

   Uploading from the platform's own job, rather than collecting all seven wheels
   into one GitHub-hosted job, is a measured decision. On `v2.10.0rc1` the
   collecting job pulled the seven artifacts out of GitHub's store in 22 seconds,
   then spent 59m47s pushing 2.35 GB to Nexus before its own 60-minute timeout
   killed it — roughly 0.13 MB/s across that border. The same bytes upload from a
   runner in China in about a minute (13–23 MB/s). GitHub's artifact hop is not
   the cost; the last leg is, and it only exists in the collecting shape.
   `flagos-ai/build-infra` releases its wheels the same way.

The tag names the release and the local segment names the SDK, so the two must
agree on everything before the `+`; the uploader checks that and refuses to
publish a wheel whose base version is not the tag.

Re-running a release whose wheels are already in a lane skips them and uploads
only the missing ones, so a lane that failed can be completed without touching the
others. Because every platform publishes from its own job, re-running the workflow
rebuilds all seven platforms; to re-upload one lane without rebuilding anything,
see [Recovering a single lane](#recovering-a-single-lane). Note that a *partially*
published version cannot be completed later against the same version number if a
wheel's content has to change — Nexus does not overwrite — so a broken artifact
means cutting the next candidate.

### How the upload behaves

Each platform uploads its own wheel in its own job, so the seven uploads are
concurrent by construction: there is no shared process to serialise and nothing to
share, since every lane is a different repository. A lane that fails is reported by
that platform's job and the job exits non-zero, and it does not hold the others.

Each wheel gets a **timeout** (900 s by default, `--upload-timeout` to change it)
and one retry on failure, and the upload step itself is capped at 20 minutes so a
stall is reported by the step rather than by the job budget. The timeout and the
retry are both there because of what the route does: it answers an intermittent
`503`, and a stalled transfer hangs on a socket read with no output. For scale, the
same Nexus takes a 673 MB wheel in 34 s from a runner in China (~20 MB/s), so a
wheel taking many minutes is a stall and not slowness — which is exactly what
`v2.10.0rc1` hit when the link ran at ~0.13 MB/s and held the release for an hour.

The route to Nexus is decided per runner rather than assumed:
`build-wheel-common.yml` calls `prefer_direct_route` from
`.github/scripts/lib/proxy_route.sh`, which probes whether `resource.flagos.net`
answers *without* the runner's HTTP proxy and adds it to `NO_PROXY` when it does.
That matters because the pod proxies differ — the PPU pod's proxy refuses the
`CONNECT` while the MUSA runner's serves the same host. `TORCH_FL_PROXY_ROUTE=direct|proxy`
overrides the probe if a runner ever needs forcing.

## Credentials

The organization secret **`NEXUS_TOKEN`** in `user:token` form — the same one
`flagos-ai/build-infra` uses. Because each platform uploads its own wheel, every
vendor runner that publishes now receives it, and it has write access to every
vendor lane. That is the deliberate cost of keeping the upload inside the
country; the guards that travel with it are the version check in
`publish_wheels.py`, which refuses a wheel whose base version is not the tag, and
the ref check in the upload step, which lets nothing but a tag ref publish.

The secret is declared and mapped explicitly at every `uses:` hop —
`release.yml` → `build-wheel-<vendor>.yml` → `all-tests-common.yml` →
`build-wheel-common.yml` — because `secrets: inherit` is not transitive. Nothing
per-repository needs configuring as long as the organization secret is visible to
this repository. If it is not, each build job fails on an empty token *before* it
builds anything, rather than uploading nothing quietly.

A follow-up worth pursuing with the Nexus administrators is a token scoped to one
lane, or one that can only upload, so that a compromised vendor runner cannot
publish to another vendor's lane at all.

## Adding a platform

A platform that publishes needs four things, and the tests will say so if one is
missing:

1. `pypi_lane` in `cmake/flagos_platforms.json`.
2. A `.github/configs/<platform>.yml`, since a release reuses that pipeline.
3. A job in `.github/workflows/release.yml` — either a
   `build-wheel-<platform>.yml` wrapper or an explicit `platform:` input to
   `all-tests-common.yml`.
4. `publish: true` on that job, with the `NEXUS_TOKEN` mapping beside it. Without
   them the platform still builds and the release is green, but that wheel never
   reaches its lane — which nothing else would catch, so
   `tests/unit/test_release_publishing.py` fails when a build job is missing
   either.

## Testing the path without releasing

```bash
python3 .github/scripts/publish_wheels.py --wheels /tmp/wheels --tag v2.10.0 --dry-run
```

`--wheels` expects one directory per artifact, named `wheel-<platform>` — the same
layout the upload step assembles on the runner before calling the script.

## Recovering a single lane

Every wheel is still uploaded as a workflow artifact, so a lane that failed can be
completed from any machine, without rebuilding anything:

```bash
gh run download <run-id> --repo flagos-ai/Torch-FL --pattern 'wheel-<platform>' -D wheels
NEXUS_TOKEN=user:token python3 .github/scripts/publish_wheels.py \
  --wheels wheels --tag v2.10.0
```

Two things make this safe to run at any time: the version check refuses to publish
a wheel whose base version is not the tag, and the skip check makes it a no-op for
a lane that already has that version. This is how `v2.10.0rc1` was completed after
its publish job was killed — the seven artifacts were still in the run, and the six
lanes that had never been reached were filled in a minute each. Note that
`gh run download` fetches serially, so a full 2.4 GB set over a throttled link takes
hours; fetching one platform's artifact, or using several connections in parallel,
is the difference between minutes and hours.
