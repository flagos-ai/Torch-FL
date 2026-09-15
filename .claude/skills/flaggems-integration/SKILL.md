---
name: flaggems-integration
description: >
  Enable and validate FlagGems Python or C++ operator routing on an accelerator
  after runtime bring-up and the platform operator path are ready. Use this for
  a new FlagGems vendor/compiler integration, a FlagGems version change, a
  generated-route mismatch, or a hardware support measurement. Covers the
  isolated vendor environment, torchgen and FlagGems discovery, generated
  configuration, Python/C++ dispatch selection, RNG state, and per-overload
  correctness survey. Do not use this to infer support from a routing table.
---

# FlagGems integration (torch_fl)

## Scope and prerequisites

FlagGems is a portable compiler-kernel path. It is an optional layer on top of
an already working device runtime and operator path; it does not replace device
allocation, streams, copies, synchronization, or vendor-native compatibility
work.

Complete these first:

- [[runtime-bringup]] — the device and allocator contract must pass on hardware.
- [[torch-version-port]] — generated ATen bindings must match the active torch
  minor line.
- [[cuda-compat-vendor]] — for a CUDA-shaped vendor, establish the boxing path
  before adding FlagGems. FlagGems and CUDA boxing may coexist per operator.
- [[native-op-backend]] — for a native vendor, establish the native backend and
  its fallback policy before routing additional operators to FlagGems.

The integration has three separate surfaces:

| Surface | Source of truth | What it does |
|---|---|---|
| Python kernels | `flag_gems._FULL_CONFIG` plus `scripts/codegen/codegen_ops.py` | Discovers compatible gems, emits wrappers, and registers `flagos_python` routes |
| C++ kernels | `csrc/aten/flaggems_cpp_kernels.cc` and generated registration | Routes the small explicit C++ FlagGems set without the Python GIL |
| Runtime selection | `torch_fl/__init__.py` and `torch_fl/configs/backends_*.conf` | Selects the platform config and installs the chosen dispatch registrations |

A route is only a candidate for testing. The support verdict comes from cases
that execute correctly against a CPU reference.

## Step 1 — identify the vendor/compiler environment

Keep the active torch environment and the vendor compiler environment explicit.
The Python torch used to build torch_fl must match the branch's minor line. Do
not replace a CPU-only torch with a vendor CUDA torch merely to make FlagGems
import or Triton discovery succeed.

Record these values before changing code:

```bash
python - <<'PY'
import sys
import torch
print("python", sys.version)
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("hip", torch.version.hip)
try:
    import flag_gems
    print("flag_gems", getattr(flag_gems, "__version__", "unknown"))
    print("full_config", len(flag_gems._FULL_CONFIG))
except Exception as exc:
    print("flag_gems import failed:", repr(exc))
PY

python -m pip show torch flag_gems triton 2>/dev/null || true
```

For a vendor box, also record the vendor Triton/compiler version, Python ABI,
SDK/runtime revision, device model, driver, and any required environment such
as `GEMS_VENDOR`, `XPU_ENABLE_PROFILER_TRACING`, or a vendor library path.
Vendor `PYTHONPATH` and `LD_LIBRARY_PATH` must not silently inject a second
PyTorch or Triton into a CPU-only codegen environment. Use an explicit wrapper
or a clean shell and print the resolved paths:

```bash
python - <<'PY'
import importlib.util
for name in ("torch", "flag_gems", "triton"):
    spec = importlib.util.find_spec(name)
    print(name, spec.origin if spec else "not found")
PY
```

A Python ABI mismatch, a failed `flag_gems` import, or an unresolved vendor
compiler is an environment failure. Stop the integration run and record it;
do not accept an empty generated Python-kernel file as a successful result.

## Step 2 — discover and classify the Python surface

`scripts/codegen/codegen_ops.py` reads the installed torchgen schemas and discovers
FlagGems functions from `flag_gems._FULL_CONFIG`. The generator filters a gem
by schema compatibility, positional/keyword arity, supported type annotations,
`out` behavior, and known recursive or RNG cases. Never hand-copy the complete
FlagGems config into a torch-fl config.

Run discovery in the exact environment that will build the generated files:

```bash
FLAGOS_CODEGEN_ALL=1 python scripts/codegen/codegen_ops.py 2>&1 | tee /tmp/flaggems-codegen.log
! grep -E 'SKIP|WARN|\[flaggems\] import failed' /tmp/flaggems-codegen.log
```

If CUDA boxing is used, the external CUDA assets may be preloaded only through
the repository's wrapper, while the active Python torch remains CPU-only:

```bash
FLAGOS_CODEGEN_ALL=1 bash scripts/vendor/with_cuda_libtorch.sh \
  python scripts/codegen/codegen_ops.py 2>&1 | tee /tmp/flaggems-codegen.log
```

Inspect the resulting counts and keep them with the run evidence:

```bash
grep -E 'flaggems|generated|route|SKIP|WARN' /tmp/flaggems-codegen.log || true
wc -l csrc/aten/generated/flaggems_python_kernels.cc
```

A generator change belongs in `scripts/codegen/codegen_ops.py` or its supporting
registry logic, not in generated C++ or config files. Regenerate all affected
artifacts together, including `flaggems_python_kernels.cc`, `register.inc`, and
`torch_fl/configs/backends_*.conf`.

## Step 3 — make generation idempotent

Compare complete generated patches, not the working tree against the branch
base: a valid torch or FlagGems version change is expected to produce a diff.
Run the generator twice in the same isolated environment:

```bash
git diff --binary > /tmp/flaggems-codegen-1.patch
FLAGOS_CODEGEN_ALL=1 python scripts/codegen/codegen_ops.py \
  > /tmp/flaggems-codegen-second.log 2>&1
! grep -E 'SKIP|WARN|\[flaggems\] import failed' /tmp/flaggems-codegen-second.log
git diff --binary > /tmp/flaggems-codegen-2.patch
cmp -s /tmp/flaggems-codegen-1.patch /tmp/flaggems-codegen-2.patch \
  && echo "FlagGems codegen is idempotent"
```

If the second patch differs, find whether the cause is unordered discovery,
unstable generated config ordering, a timestamp/path, or a mutable FlagGems
registry. Fix the generator and repeat. Record the exact torch, FlagGems, and
Triton versions because generated output is input-version-specific.

## Step 4 — wire and select the backend

Python FlagGems routes use `flagos_python`; the C++ path uses the explicit
FlagGems C++ backend. Check both sides of every route:

```bash
python tests/integration/ops/test_flaggems_conf_consistency.py -v
```

The runtime config is selected by `torch_fl/__init__.py`:

- `FLAGOS_USE_FLAGGEMS=1` selects the generic Python config.
- `FLAGOS_USE_FLAGGEMS_CPP=1` selects the explicit C++ config.
- Platform-specific combinations select the MetaX, DCU, or other vendor config.
- A platform config must not claim a Python route that its compiler cannot run.

For a new platform, follow the existing config pattern and keep the generated
block clearly identified. Ensure that:

1. every `flagos_python` route has a generated wrapper and dispatcher;
2. C++ routes are disjoint from Python routes unless the dispatch priority is
   intentional and tested;
3. unsupported dtypes and unsupported vendor operations fall back according to
   the platform policy rather than silently changing tensor device semantics;
4. the selected `GEMS_VENDOR` and Triton backend are compatible with the chip;
5. importing torch_fl with FlagGems disabled remains clean.

Do not add a handwritten per-operator kernel to bypass a generator limitation.
If the vendor API cannot be expressed by FlagGems, route that operator through
the CUDA-compatible or native backend and document the boundary.

## Step 5 — handle RNG and state explicitly

FlagGems RNG must be tested as a stateful interface, not only for numerical
shape. For each platform, determine whether generator-less calls use a shared
seed/offset stream and whether explicit generators remain isolated. Include
`rand`, `randn`, `randperm`, `random_`, `bernoulli`, and at least one `*_like` or
factory overload available in the active torch schema.

Use the repository RNG tests where available:

```bash
python -m pytest tests/integration/ops/test_rng_dispatch.py -q
```

Then run a focused mixed-route probe. Verify same-seed replay, different-seed
sensitivity, stream advancement across repeated calls, explicit-generator
isolation, and replay after mixing FlagGems with a native or CUDA route. A
successful call with the wrong stream or a reset generator is a correctness
failure.

## Step 6 — validate cases against CPU

Routing presence and import success are not operator support. Start with the
small dispatch suite, then run the manual overload survey on each affected
hardware platform:

```bash
python tests/integration/ops/test_flaggems_conf_consistency.py -v
python -m pytest tests/integration/ops/ -m "flaggems or flaggems_python" -q

python tests/manual/flaggems_overload_survey.py \
  --conf torch_fl/configs/backends_cuda.conf \
  --out /tmp/flaggems-overloads.json
```

There is no generic `backends_flaggems.conf` any more: FlagGems routing lives in
each platform's own conf, so survey the conf for the platform under test
(`backends_ascend.conf`, `backends_musa.conf`, ...).

The survey first rejects synthesized inputs that fail on CPU. Recompute each
operator verdict from the remaining cases:

- `STRICT`: every CPU-valid case passed;
- `BASIC_ONLY`: at least one passed and at least one failed;
- `FAILED`: valid cases existed and none passed;
- `UNTESTED`: no CPU-valid synthesized case existed.

Keep case-level `ERROR`, `WRONG`, `CRASH`, and `TIMEOUT` separate from operator
verdicts. Re-run crashes with the correct runtime environment before counting
them. A runtime startup failure is not an operator result.

For each claimed platform, capture the hardware model, run date, torch/FlagGems/
Triton revisions, config and active route-set hashes, survey harness version,
profiles per overload, raw JSON, and aggregate counts. If hardware is absent,
mark the row **not revalidated**. Do not copy another platform's rate.

Two standalone tools help when the survey is impractical:

- `scripts/tools/verify_flaggems_ascend.py` re-checks individual overloads on
  Ascend without a full sweep. Triton JIT dominates, so use `--ops a,b,c` or
  `--shard i/n` rather than the full run.
- `scripts/tools/record_musa_flaggems_failures.py` records MUSA ops that fail CI
  so they move into `NATIVE_TRITON_GAPS`, then regenerates `backends_musa.conf`.

## Step 7 — update evidence and review the boundary

Update `docs/reference/operator-support.md` in the same change when routing or
implementation changes. Keep generic FlagGems overload results separate from
native routes such as Ascend, GCU, or vendor-specific kernels. Update the
platform installation guide with prerequisites, environment isolation, build
flags, and the measured command.

Before opening a PR, verify:

- `flag_gems` imported successfully in the generation environment;
- no `SKIP`, `WARN`, or hidden import failure occurred;
- the second generator run produced an identical patch;
- config consistency passed;
- RNG state and mixed-route behavior passed where supported;
- the survey JSON and aggregate arithmetic agree;
- unavailable hardware is labeled **not revalidated**;
- `ruff check .` and `ruff format --check .` pass;
- no vendor runtime, driver, XMLIR, or Python environment was copied into a
  wheel without a measured minimal dependency set;
- all generated artifacts and the generator/config source change are committed.

FlagGems is ready to claim only the measured scope. A larger route table, a
successful import, or a passing smoke test is not evidence for a larger support
rate.

## Related

[[runtime-bringup]] · [[torch-version-port]] · [[cuda-compat-vendor]] ·
[[native-op-backend]] · [[pre-pr-checks]]
