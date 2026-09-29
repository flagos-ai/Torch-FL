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
   hatch.

   The candidate suffix reaches the artifact through the tag: the build reads
   `GITHUB_REF_NAME` and exports `FLAGOS_WHEEL_PRERELEASE`, so `v2.10.0rc1`
   produces `torch_fl-2.10.0rc1+cuda13.3`. PEP 440 sorts that below `2.10.0`, so
   `pip install torch_fl==2.10.0` does not pick the candidate up. The uploader
   then requires the tag and the wheel to name the same version, which is what
   stops a candidate being published under the release's name or the reverse.
2. **Builds on every platform**, one job each, on the platform's own runner. The
   build is the same one CI runs: the `build-wheel-*.yml` wrapper (or
   `all-tests-common.yml` for GCU, which has no wrapper), the same image and
   setup hook, and the same `Verify wheel` step that installs the fresh artifact
   before it is offered to the uploader.
3. **Uploads each wheel to its lane** with
   `.github/scripts/publish_wheels.py`, which reads the target from the
   `pypi_lane` column of `cmake/flagos_platforms.json`:

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

The tag names the release and the local segment names the SDK, so the two must
agree on everything before the `+`; the uploader checks that and refuses to
publish a wheel whose base version is not the tag. Re-running a release whose
wheels are already in the lane skips them and uploads only the missing ones, so
a partial failure can be retried by re-running the workflow.

## Credentials

The organization secret **`NEXUS_TOKEN`** in `user:token` form — the same one
`flagos-ai/build-infra` uses. Only the `publish` job reads it, so no vendor
runner and no wheel build ever sees it, and nothing per-repository needs
configuring as long as the organization secret is visible to this repository. If
it is not, the job fails on an empty token rather than uploading nothing
quietly.

## Adding a platform

A platform that publishes needs three things and the tests will say so if one is
missing:

1. `pypi_lane` in `cmake/flagos_platforms.json`.
2. A `.github/configs/<platform>.yml`, since a release reuses that pipeline.
3. A job in `.github/workflows/release.yml` — either a
   `build-wheel-<platform>.yml` wrapper or an explicit `platform:` input to
   `all-tests-common.yml`.

## Testing the path without releasing

```bash
python3 .github/scripts/publish_wheels.py --wheels /tmp/wheels --tag v2.10.0 --dry-run
```

`--wheels` expects what `actions/download-artifact` produces: one directory per
artifact, named `wheel-<platform>`.
