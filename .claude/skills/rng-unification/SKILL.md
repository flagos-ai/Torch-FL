---
name: rng-unification
description: >
  Automatically diagnose and fix inconsistent random-number-generator behavior in
  torch_fl across CUDA boxing, vendor-native, and FlagGems routes. Use this when
  torch.manual_seed, generator=, RNG state, seed/offset advancement, replay,
  mixed-route execution, or multi-device random operations are missing or failing.
  The workflow traces the active route, reproduces stateful failures, applies the
  smallest codegen or backend fix, adds contract tests, and records measured
  hardware boundaries. Do not infer RNG support from a successful import or a
  plausible-looking sample.
---

# Automatic RNG unification (torch_fl)

## Purpose

This skill turns an RNG bug report or parity request into an evidence-driven
repository change. It is for **changing the implementation**, not merely running
the existing RNG suite. Use the existing dispatch tests for validation-only work;
this skill continues through diagnosis, implementation, regression coverage, and
measured evidence.

The workflow is:

```text
request
  -> resolve torch/vendor/device and active route
  -> reproduce the state failure in a fresh process
  -> identify the generator source and consumption contract
  -> choose the existing CUDA/codegen or native backend path
  -> implement the smallest complete fix
  -> regenerate generated artifacts and prove idempotency
  -> test replay, sensitivity, advancement, isolation, and distributions
  -> validate mixed routes and devices where available
  -> update measured support records and report exact boundaries
```

RNG is stateful. A random tensor with the right shape and a plausible mean is not
proof that the correct generator was used. A fix is complete only for the scope
that has passed the corresponding state, numerical, route, and hardware gates.

## Inputs and defaults

Accept any subset of:

- target accelerator and hardware (`ACCELERATOR`, model, device index);
- PyTorch minor line and vendor torch path;
- failing operator or operator family;
- active route (`= cuda`, vendor-native, `flagos_python`, C++ FlagGems, or unknown);
- requested contract (`torch.manual_seed`, explicit `generator=`, state round-trip,
  seed/offset parity, dropout, compile, or distributed workload);
- whether the user wants a commit or pull request.

Make routine defaults without blocking:

- device key: the torch_fl PrivateUse1 name, normally `flagos`;
- validation interpreter: the interpreter that imports the vendor torch used to
  build torch_fl;
- reference: a CPU result for distribution/range checks and a same-seed replay
  for exact state checks;
- target branch: the current stable PyTorch minor branch;
- implementation route: preserve the selected route unless measurement proves it
  is the failing layer;
- default scope: generator-less and explicit-generator behavior on one visible
  device, then expand to out, like, factory, mixed-route, and multi-device cases.

Ask a blocking question only when the target hardware, torch ABI, or requested
scope cannot be determined safely. Never substitute a CPU or NVIDIA wheel for a
vendor torch distribution merely to make an RNG test run.

## Non-negotiable boundaries

1. **No inferred support.** API import, a populated route table, a compiled symbol,
   or one random-looking output does not prove RNG correctness.
2. **No hidden skips.** A skipped, xfailed, crashed, or deselected case is an
   evidence gap. Record its reason and keep it out of the pass count.
3. **No generator conflation.** Distinguish the PrivateUse1 `flagos` generator,
   CUDA-shaped Philox generators, the default CPU generator, and any vendor
   seed/offset state. Similar names do not imply shared state.
4. **No route inference from a global switch.** `FLAGOS_USE_FLAGGEMS=1` does not
   mean every operator uses FlagGems. Read the selected config for the exact
   overload under test.
5. **No silent fallback.** A vendor fallback, CPU implementation, or ATen default
   generator must be identified; it cannot be reported as unified RNG behavior.
6. **No handwritten vendor kernels when codegen expresses the path.** For a
   non-CUDA-compatible backend, extend the category/template, mapping, generated
   source, and configuration. A handwritten kernel requires a documented vendor
   limitation and explicit human approval.
7. **No generated-file hand edits.** Change `scripts/codegen_ops.py` or the native
   backend generator first, regenerate every affected artifact, and inspect the
   complete diff.
8. **No state races.** Generator state reads and seed/offset reservations must use
   the generator's documented mutex/locking contract. Never clone, mutate, or
   advance a shared generator without synchronization.
9. **No cross-device claims from device zero.** Derive the device index from the
   tensor, requested factory device, or current device as appropriate, and test a
   nonzero device when hardware exposes one.
10. **No distributed claims from local replay.** DDP/FSDP2 or multi-rank RNG
    alignment needs a separate multi-process test and evidence record.
11. **No environment leakage.** Do not copy vendor drivers, SDKs, runtimes,
    XMLIR, credentials, or Python environments into the repository or wheel.
12. **Keep claims scoped.** Every report separates operator family, dtype, route,
    generator source, hardware, device index, distributed scope, and performance.

## Phase 0 — establish the development ledger

Before editing, create an in-memory ledger or temporary evidence file outside the
repository:

```text
request:
target_branch:
target_accelerator:
hardware_model:
device_key:
torch_version:
torch_path:
torch_fl_revision:
route_and_config:
operator_scope:
generator_contract:
original_failure:
files_examined:
commands_run:
changes_made:
verification_results:
unsupported_scope:
```

Start by proving the checkout and upstream state:

```bash
git status --short --branch
git fetch flagos <target-stable-branch>
git log --oneline -1 flagos/<target-stable-branch>
```

For ordinary development, work on a contributor branch based on the requested
stable branch. Do not push a development branch directly to the upstream remote.
Do not rebase or overwrite unrelated user changes.

Resolve the runtime with the interpreter that will build and load torch_fl:

```bash
python - <<'PY'
import importlib.util
import os
import sys
import torch

# Import order matters for CUDA-shaped vendor builds.
import torch_fl
from torch_fl._build_config import ACCELERATOR

print("python", sys.version.split()[0])
print("torch", torch.__version__)
print("torch_file", torch.__file__)
print("torch_fl_file", torch_fl.__file__)
print("build_accelerator", ACCELERATOR)
print("env_accelerator", os.environ.get("ACCELERATOR", "<unset>"))
print("device_key", torch._C._get_privateuse1_backend_name())
print("device_count", torch_fl.flagos.device_count())
print("backend_config", os.environ.get("FLAGOS_BACKEND_CONFIG", "<unset>"))
print("flag_gems", importlib.util.find_spec("flag_gems") is not None)
print("flagcx", importlib.util.find_spec("flagcx") is not None)
print("cuda_generators", len(torch.cuda.default_generators))
print("flagos_generators", len(torch.flagos.default_generators))
PY
```

Also record the vendor runtime/SDK/driver, visible-device mapping, compiler,
Python ABI, loaded extension path, and relevant `FLAGOS_*`, `GEMS_*`, `XPU_*`,
`CUDA_*`, and `TORCH_*` variables. Print the selected config path and its
resolved contents. If the build accelerator, device key, torch minor line, or
loaded extension does not match the requested target, stop implementation claims
and report the environment blocker.

## Phase 1 — map the complete RNG path before changing code

### 1. Identify the exact overload and route

Resolve the ATen overload rather than reasoning from a Python convenience name:

```bash
grep -R -n "<operator>" torch_fl/configs csrc/aten/generated scripts
python - <<'PY'
import os
import torch
import torch_fl  # noqa: F401

name = os.environ.get("RNG_OVERLOAD", "randn")
conf = os.environ.get("FLAGOS_BACKEND_CONFIG", "")
print("operator", name)
print("config", conf or "<implicit/default>")
if conf and os.path.exists(conf):
    for line in open(conf):
        op, sep, backend = line.partition("=")
        if sep and op.strip() == name:
            print("route", backend.strip())
            break
print("aten schemas", [str(x) for x in torch.ops.aten.__dir__() if name in x][:20])
PY
```

Inspect the generated kernel and the selected registration. For a CUDA-compatible
backend, follow `DeviceBoxingGuard` through the `at::<op>` call. For a native
backend, follow the generated category into the vendor wrapper and its seed or
Philox reservation helper. For FlagGems, inspect the generated Python caller and
the compiler's `philox_backend_seed_offset` or equivalent.

The route map must answer all of these questions:

- Which generator object is used when `generator=None`?
- Which object is used when `generator=` is supplied?
- Who seeds it when `torch.manual_seed` runs?
- How is state represented (mt19937, Philox seed/offset, or vendor state)?
- Who reserves or increments the offset, and by how much?
- Does an out/like/factory overload use a different generated template?
- Does dropout use an explicit generator or a composite with no generator arg?
- Does a nonzero device select the matching generator?

### 2. Search comparable implementations

Read the complete relevant files before editing:

- `tests/integration/ops/test_rng_dispatch.py`;
- `torch_fl/flagos/random.py` and `torch_fl/flagos/__init__.py`;
- the accelerator compatibility shim, such as
  `torch_fl/accelerator/cuda/_cuda_compat.py` or the vendor equivalent;
- `scripts/codegen_ops.py` and generated
  `csrc/aten/generated/cuda_kernels.cc`;
- `csrc/aten/backends/flagos/python_op_caller.{h,cc}`;
- `csrc/runtime/generator.{h,cc}`;
- the selected native backend generator and RNG implementation;
- `torch_fl/configs/backends_*.conf`;
- `docs/reference/operator-support.md` and the relevant vendor documentation.

Search existing fixes on other platforms before inventing a new state model. The
CUDA-compatible path may share a CUDA-shaped Philox generator with FlagGems;
GCU-style native paths may use a shared `{seed, offset}` reservation helper;
Ascend-style paths may seed vendor APIs from the default CPU generator. Reuse the
existing abstraction only when its state and locking semantics match the target.

## Phase 2 — reproduce and classify the failure

Reproduce every failure in a fresh process and capture stdout, stderr, exit
status, crash/signal status, hardware identity, and route. Run one minimal probe
per layer, not one large model script.

### Stateful contract probes

Use the logical device key and compare CPU-visible results:

```python
import torch
import torch_fl  # noqa: F401

DEVICE = "flagos:0"

def draw():
    return torch.randn(64, device=DEVICE).cpu()

torch.manual_seed(1234)
first = draw()
torch.manual_seed(1234)
second = draw()
assert torch.equal(first, second), "same-seed replay failed"

torch.manual_seed(1)
left = draw()
torch.manual_seed(2)
right = draw()
assert not torch.equal(left, right), "different seeds produced the same draw"
```

Then isolate the rest of the contract:

```python
# Consecutive calls must consume state, not return a constant draw.
torch.manual_seed(1234)
a = draw()
b = draw()
assert not torch.equal(a, b)

# State round-trip must replay the next draw.
torch.manual_seed(1234)
state = torch.flagos.get_rng_state()
a = draw()
torch.flagos.set_rng_state(state)
b = draw()
assert torch.equal(a, b)

# An explicit generator is reproducible and isolated from the default stream.
generator = torch.Generator(device="flagos").manual_seed(77)
a = draw_with_generator(generator)
torch.manual_seed(999)
generator.manual_seed(77)
b = draw_with_generator(generator)
assert torch.equal(a, b)
```

Adapt the explicit-generator device to the backend contract. A CUDA-boxing
backend normally needs a CUDA-shaped generator or translates a PrivateUse1
caller generator; a native backend may consume a `flagos` or CPU generator.
Passing an incompatible generator must raise a clear error, not recurse or
silently switch streams.

### Failure classification

Classify the primary failure before implementing:

| Symptom | Primary layer | First inspection |
|---|---|---|
| same seed gives different values | seed plumbing or wrong default generator | compatibility shim, `manual_seed`, generated call |
| different seeds give the same values | stale/constant seed or ignored reseed | state setter, seed reservation, FlagGems patch |
| repeated calls return the same values | state is reset each call or offset never advances | generator clone/copy, reservation mutex |
| `get_rng_state` round-trip fails | public state is disconnected | `torch_fl.random`, default-generator ownership |
| explicit generator is ignored | generator dropped in wrapper/codegen | schema, argument order, native helper |
| explicit generator causes recursion/crash | wrong dispatch key or unboxed PrivateUse1 generator | CUDA boxing prelude and generator translation |
| only `out` or `*_like` fails | wrong generated overload or insertion point | `gen_out_variant`, `gen_functional_pure` |
| only factory ops fail | factory device/index or generator overload mismatch | `gen_factory`, ATen schema |
| FlagGems-only failure | compiler route or Philox state adapter | selected config, generated Python caller |
| native-only failure | vendor seed/offset bridge | native RNG implementation |
| device 0 passes, device 1 fails | device guard/index mapping | requested device and per-device generator lookup |
| dropout fails while other RNG passes | dropout schema/composite has no generator | route-specific dropout implementation |
| ranks disagree or hang | distributed RNG ordering/state broadcast | process-group and workload-level contract |

Do not classify a distribution mismatch as a seed mismatch until replay and seed
sensitivity have been tested. Do not fix a route failure by changing the route
without proving that the route is the failing layer.

## Phase 3 — choose the smallest safe repair

Use this decision table after reproduction:

| Measured condition | Repair |
|---|---|
| public seed API does not reach the active default source | connect the existing `manual_seed_all`, state, and initial-seed API to that source |
| CUDA boxing native kernels use an unreachable ATen default CUDA generator | inject the shared per-device CUDA/Philox generator in the codegen template |
| a CUDA-boxed PrivateUse1 generator redispatches or is rejected incorrectly | translate it to a CUDA-typed clone using the documented seed-reservation contract, or fail clearly if translation is impossible |
| a native API takes `{seed, offset}` | reserve from the shared generator under lock and pass the exact reservation to the vendor API |
| FlagGems reads a different generator/state shape | adapt the state at the existing compatibility boundary; do not patch each operator separately |
| generator-less factory or out/like overload bypasses the generator sibling | derive the overload from the active torch schema and inject the shared generator at the schema-defined position |
| explicit generator is dropped by a Python wrapper | preserve it through the generated caller; never replace it with the default stream |
| a route has no generator argument, such as some dropout composites | document the limitation, add a route-aware test, and use a decomposition or backend change only when measured and approved |
| only a vendor-specific dtype is unsupported | fail clearly or mark that dtype not validated; do not cast to a different dtype silently |
| generator state is raced by a cache/clone | add the required mutex scope and test repeated/concurrent reservations where the runtime permits |

Prefer a shared helper or codegen predicate over an operator list. For example,
a CUDA-boxing implementation should make every schema carrying `Generator?` use
the same prelude, while generator-less `rand`, `randn`, `randint`, `randperm`, and
`*_like` overloads should be handled only when the active torch schema exposes a
compatible generator sibling. Never assume a newer PyTorch overload exists on
2.9; inspect torchgen's installed schema first.

For native non-CUDA-compatible hardware, follow [[native-op-backend]]: add the
RNG category/template or mapping and generated registration/configuration. Do not
add a per-operator `.cc` implementation merely because it is quicker.

## Phase 4 — implement and preserve generator semantics

### Default-generator invariants

Every generator-less operation must satisfy:

```text
one device -> one stable default source
manual_seed(seed) -> next draw is reproducible
manual_seed_all(seed) -> every visible device is reseeded
successive calls -> state advances exactly according to the route contract
get_rng_state/set_rng_state -> next draw replays exactly
```

Every explicit-generator operation must satisfy:

```text
explicit generator controls its own draw
re-seeding it replays its own next draw
manual_seed(default) does not perturb it
it does not unexpectedly advance the default generator
an incompatible device type fails clearly and finitely
```

When two paths intentionally share a source, test the sequence, not just each
path independently:

```text
seed
  -> FlagGems draw
  -> native/vendor draw
  -> FlagGems draw
```

Reset the source and replay the complete sequence. If the route contract promises
Philox seed/offset parity, record the initial and final states and verify the
expected offset reservation. Account for each implementation's documented
rounding (for example, a Philox reservation may round the increment to a block
multiple); do not compare raw offsets without understanding consumption units.

### CUDA-compatible codegen

For a CUDA-shaped vendor:

1. Modify `scripts/codegen_ops.py`, not
   `csrc/aten/generated/cuda_kernels.cc`.
2. Apply the generator prelude in every relevant shared template:
   functional, inplace, out, tuple-return, and factory.
3. Derive a tensor-input device index from the boxed tensor and a factory index
   from the requested device after validating PrivateUse1-to-CUDA conversion.
4. Keep caller-supplied generators intact. If a PrivateUse1 generator must be
   translated, reserve from its state exactly once, clone/reseed the shared
   CUDA generator under the required lock, and preserve the anti-redispatch
   device type.
5. Include the helper header and all necessary generator/locking headers through
   codegen, so a clean regeneration compiles.
6. Regenerate all affected files and inspect representative families including
   `normal_`, `randn`, `randint`, `randperm`, an out variant, a `*_like` variant,
   and a distribution op.

Do not inject a generator into an overload whose schema cannot accept it. For
`native_dropout` or another schema without `Generator?`, first determine whether
the selected route has an independent shared RNG mechanism. If not, keep the
failure explicit and route-aware rather than claiming global unification.

### Native backend codegen

For a native backend:

- use the platform's existing RNG helper and lock discipline;
- reserve one seed/offset from the shared public source for each logical call;
- honor an explicit generator by consuming that generator, without advancing the
  global source;
- preserve shape, dtype, layout, out aliases, and device index;
- make unsupported vendor dtypes fall back only according to the documented
  policy, with a test proving that fallback's generator source;
- add the operator to the generator mapping and generated registration/config,
  never only to a handwritten source file.

The native helper must not accidentally use `at::detail::getDefaultCPUGenerator`
when the platform promises a shared Philox stream, nor use a dead PrivateUse1
`GeneratorImpl` when FlagGems/native CUDA paths read a different source. The
opposite is also true: a CPU-seeded native API must not be forced onto a CUDA
Philox generator without a measured conversion contract.

### Python/FlagGems path

For FlagGems:

- read the active config for the exact overload;
- keep generator-less state access in one compatibility boundary;
- preserve explicit generator arguments when the gem supports them;
- verify the expected state shape before unpacking seed/offset;
- avoid cold-cache behavior that changes generator argument handling;
- keep vendor aliases normalized and generated from the canonical source;
- update the FlagGems route only when the compiler route itself is measured as
  the failure.

A global FlagGems switch is not sufficient evidence. Use route-aware marks or
checks, as `test_rng_dispatch.py` does for dropout and allowlist configurations.

## Phase 5 — add contract tests before broad validation

Extend `tests/integration/ops/test_rng_dispatch.py` when the cross-backend
contract applies. Add a focused platform-marked file only when the semantics are
vendor-specific. Every test must use the logical device key, compare CPU-visible
values where appropriate, and state its route/hardware boundary.

The minimum test matrix is:

| Contract | Required assertion |
|---|---|
| same-seed replay | two fresh draws after the same seed are exactly equal |
| seed sensitivity | two distinct seeds produce different draws |
| advancement | consecutive draws without reseeding are not the same constant stream |
| public state round-trip | saving/restoring state replays the next draw |
| explicit generator replay | reseeding the explicit generator reproduces its draw |
| explicit isolation | default reseeding does not perturb the explicit generator |
| default isolation | explicit draws do not unexpectedly advance the default source |
| factory | `rand`, `randn`, `randint`, and `randperm` use the correct device/index |
| out variants | `rand.*.out`, `randn.*.out`, integer and like out forms preserve output and state |
| in-place distributions | `normal_`, `uniform_`, `exponential_`, `bernoulli_`, and related forms |
| distribution validity | ranges, permutation property, and basic moments against CPU expectations |
| mixed route | FlagGems/native/vendor sequence replays from one seed |
| multi-device | device 0 and a nonzero visible device address independent sources |
| dropout | route-aware replay plus mask, scaling, and backward validity |
| unsupported generator | incompatible generator type fails clearly, without recursion/crash |

Use deterministic, moderate-size inputs. Distribution moment checks supplement,
but never replace, exact replay checks. Avoid asserting that two independent
devices must produce different values after the same seed; assert independent
state objects and per-device replay instead.

Tests must leave global state usable for later tests. Restore changed environment
variables/configuration, avoid unbounded generator iteration, and use precise
skip reasons for absent hardware or unsupported schemas. If a test is only valid
when an overload routes through FlagGems, inspect that overload's route rather
than gating on `FLAGOS_USE_FLAGGEMS` alone.

## Phase 6 — regenerate and prove idempotency

Run the generator in the exact build environment, with the vendor runtime setup
required by that platform. Never hand-edit generated output.

For a codegen change:

```bash
set -eu
FLAGOS_CODEGEN_ALL=1 python scripts/codegen_ops.py
# Review generated and untracked files before deciding the intended scope.
git diff --binary > /tmp/rng-codegen-first.patch
FLAGOS_CODEGEN_ALL=1 python scripts/codegen_ops.py
git diff --binary > /tmp/rng-codegen-second.patch
cmp -s /tmp/rng-codegen-first.patch /tmp/rng-codegen-second.patch
printf '%s\n' "RNG codegen idempotent"
```

Check the generator exit status before comparing patches. If the complete run
rewrites unrelated artifacts because torch or FlagGems differs, restore those
drifts, record the environment mismatch, and do not claim a clean regeneration.
A second identical failed run is not idempotency.

Inspect the generated diff for:

- exact ATen overload argument order;
- generator insertion points for factory, like, and out forms;
- tensor/device index expressions;
- explicit-generator pass-through;
- required includes and declarations;
- duplicate registrations or route widening;
- accidental changes to non-RNG kernels.

## Phase 7 — build and validate in layers

Build through `setup.py` or the repository's documented editable-install path,
not a hand-written bare CMake invocation. Print the loaded extension path after
building so a stale library cannot masquerade as a regression result:

```bash
<documented-build-command>
python -c 'import torch_fl, torch_fl._C; print(torch_fl.__file__); print(torch_fl._C.__file__)'
```

Run checks in this order:

1. **Environment:** intended torch, torch_fl extension, vendor runtime, device
   count, and active config.
2. **Public state:** `manual_seed`, `manual_seed_all`, initial seed, and state
   get/set round-trip.
3. **Primitive route:** one representative operation per generator source,
   including an explicit generator.
4. **Operator families:** in-place, functional, factory, like, out, and
   distribution overloads supported by the active schema.
5. **Mixed routes:** replay the same FlagGems/native/vendor sequence with both
   FlagGems disabled and enabled where the config supports both.
6. **Multi-device:** repeat on a nonzero device if at least two devices are
   visible; otherwise record `not validated`.
7. **Full RNG suite:**

   ```bash
   python -m pytest tests/integration/ops/test_rng_dispatch.py -v
   ```

8. **Focused route suites:** run the native, FlagGems, or vendor tests selected
   by the changed route, plus config consistency tests.
9. **Regression suite:** run the relevant operator/model tests. A known unrelated
   crash must be reproduced at the clean base before being excluded.
10. **Quality:**

    ```bash
    ruff check .
    ruff format --check .
    python -m compileall -q torch_fl tests
    git diff --check
    ```

Capture actual output and exit status. If the broad suite fails for a pre-existing
vendor/runtime reason, run the same command at the base or a clean worktree and
report both counts. Never turn an xfail, crash, or base-equivalent failure into a
pass claim.

For a distributed request, add a separate multi-process test that proves rank
ordering, seed initialization, per-rank/device state, and workload behavior. A
single-process mixed-route replay is not distributed evidence. Use
[[flagcx-integration]] for communication-specific setup and fail closed if the
intended communication backend is replaced by a fallback.

## Phase 8 — update measured evidence

Update `docs/reference/operator-support.md` in the same change when the RNG route,
implementation, or claimed hardware support changes. Keep native vendor and
FlagGems cohorts separate. For every measured platform record:

- date and torch-fl revision/branch;
- exact torch, FlagGems, Triton/vendor SDK, and runtime versions;
- hardware model, visible count, and device mapping;
- selected config and active route-set hash when applicable;
- generator source/state representation and explicit-generator contract;
- exact commands, pass/skip/xfail/failure counts, and exit status;
- whether mixed-route, multi-device, distributed, performance, and dropout scope
  was actually tested.

If hardware is unavailable, write **not revalidated** and record the evidence gap.
Do not retain old measurements as if they covered a new implementation or torch
minor line. Do not put raw vendor environments, credentials, or proxy contents in
the evidence file.

A support statement should look like this:

```text
Unified generator behavior: validated for generator-less and explicit-generator
normal/randn/randint/randperm operations on <hardware>, route <config>, devices
0..N-1. Same-seed replay, seed sensitivity, state round-trip, mixed-route replay,
and <explicit list> passed. Dropout/distributed/performance: not validated.
FlagGems generic cohort: not revalidated unless its survey was rerun.
```

## Phase 9 — automated completion gate

Before declaring the task complete, evaluate every row directly:

| Gate | Required proof | Status label |
|---|---|---|
| Environment | matching torch/extension/runtime/device/config printed | environment resolved / blocked |
| Route | exact overload route observed or read from selected config and confirmed at runtime | route identified |
| Default source | generator object/state source is traced and seeded by the public API | default source validated |
| Replay | same seed and state round-trip pass | replay validated |
| Sensitivity | different seeds produce different draws | seed sensitivity validated |
| Advancement | successive calls consume state with expected reservation behavior | advancement validated |
| Explicit generator | replay, isolation, and mismatch behavior pass | explicit generator validated |
| Families | tested in-place, functional, factory, like, out, and distributions in scope | operator families validated |
| Mixed route | complete interleaved sequence replays across selected routes | mixed-route validated |
| Multi-device | nonzero device source/index tested | multi-device validated / not validated |
| Dropout | route-aware mask/replay/backward contract tested | dropout validated / not validated |
| Distributed | multi-rank state/order/workload test passes | distributed validated / not validated |
| Generation | second successful generator run has identical patch | codegen idempotent / not applicable |
| Quality | lint, format, compile, and relevant tests pass | quality checks passed |
| Records | docs state measured scope and evidence gaps | support records updated |

A missing gate must be reported as `not validated`, `blocked`, or `not revalidated`,
never as supported. The final report must include:

- files changed and why;
- the root cause and fixed layer;
- generator sources before and after, including state/offset semantics;
- exact commands and actual output summaries with exit status;
- hardware/interpreter/torch/runtime/config identity;
- generated artifact and idempotency result;
- known failures and whether they pre-date the change;
- explicit operator, dtype, route, device, distributed, and performance boundaries.

## Phase 10 — commit and PR automation

Only when the user requests submission:

1. Read `.github/AI_AGENT_GUIDE.md`, `.github/CLAUDE_CODE_GUIDE.md`, and
   `.github/PULL_REQUEST_TEMPLATE/ai_agent_pr.md`.
2. Fetch and rebase onto the requested upstream stable branch before linting.
3. Confirm the diff contains only the RNG fix, tests, generator/config changes,
   and measured documentation intended for the PR.
4. Use an English conventional commit with the required Claude co-author trailer.
5. Validate the AI PR body:

   ```bash
   python scripts/validate_ai_pr.py --pr-body <body-file>
   ```

6. Include root cause, investigation steps, actual test/lint output, edge cases,
   hardware identity, and all unvalidated scope.
7. Push the ordinary development branch only to the contributor fork, then create
   the PR against the requested upstream branch with the `ai-generated` label and
   a human reviewer who is not the PR author.

Do not expose authentication material, force-push a shared branch, or claim
hardware evidence that was not measured.

## Failure recovery loop

When a gate fails:

```text
failure
  -> preserve complete log and exit status
  -> reduce to one overload and one generator source
  -> classify seed/state/offset/route/device/locking failure
  -> inspect a comparable working backend
  -> implement one focused fix
  -> regenerate and check idempotency
  -> rerun the failed gate and all dependent gates
```

Do not patch seeding, codegen, route configuration, and vendor kernels at once
without isolating the cause. If hardware is unavailable, finish static/codegen
and unit work that does not require it, then report runtime-dependent gates as
not validated.

## Related skills

[[runtime-bringup]] · [[torch-version-port]] · [[cuda-compat-vendor]] ·
[[native-op-backend]] · [[flaggems-integration]] · [[flagcx-integration]] ·
[[pre-pr-checks]]
