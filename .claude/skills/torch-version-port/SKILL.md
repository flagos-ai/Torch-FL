---
name: torch-version-port
description: >
  Port torch_fl to a different PyTorch minor version (e.g. create the 2.9 branch,
  or bump to 2.11/2.12) — hardware-independent work covering the CPU-only torch
  pin, ATen codegen regeneration against the new native_functions.yaml, and the
  per-operator IListRef/ArrayRef signature reconciliation. Use this when creating
  a new version branch, when `import torch_fl` fails with "Mismatch in kernel C++
  signatures" after a torch bump, or when csrc/aten/generated/ needs regenerating.
  Do NOT use this for chip enablement — see runtime-bringup first for that.
---

# Torch version port (torch_fl, hardware-independent)

## Scope: one axis only

This skill moves torch_fl onto a different `torch==X.Y` line. Everything it
touches is schema- and ABI-level. **Nothing in it knows or cares which chip will
run the kernels.**

That separation is the point. Collapsing version work into chip work means every
torch bump re-derives chip facts and every new chip re-derives version facts.
Keep them apart:

- version axis → this skill
- the always-required device runtime → [[runtime-bringup]]
- operators → [[cuda-compat-vendor]] or [[native-op-backend]]

## Branch model

Branches are **per-torch-minor and are siblings, not descendants**. `main` tracks
one line (currently 2.10.x, per `docs/reference/compatibility.md`); `2.9`,
`2.12`, `2.13` are peers. A chip port targets whichever version branch it needs
and is never the reason a version branch exists.

```bash
git fetch flagos main
git switch -c 2.9 flagos/main     # branch from main, not from another version branch
```

Branching from another version branch inherits that version's signature fixes,
which are exactly what you are trying to re-derive. Start from `main`.

Note that `main` may not carry codegen infrastructure at all — at time of writing
`2.13` is the reference implementation for it. If `scripts/codegen/codegen_ops.py` is
absent on your base, port the infra first (see [[cuda-op-integration]] Step 0,
which lists the precise file set to `git checkout 2.13 --`).

## Step 1 — pin a CPU-only torch, deliberately

```bash
conda create -n fl29 python=3.11 -y && conda activate fl29
pip install torch==2.9.0+cpu --index-url https://download.pytorch.org/whl/cpu
python -c "import torch; print(torch.__version__)"   # must print 2.9.0+cpu
```

**CPU-only is a requirement, not a convenience.** The CUDA-compatible operator
path supplies GPU symbols out-of-band through a preloaded external
`libtorch_cuda.so`; a pip CUDA torch in the same env introduces a second,
possibly mismatched copy of those symbols and the resulting failures are
attributed to codegen. `torch.__version__` must keep its `+cpu` suffix through
the entire port — check it again at the end.

## Step 2 — regenerate ATen codegen

`scripts/codegen/codegen_ops.py` reads torchgen's *packaged* `native_functions.yaml`,
which means its output is a function of the installed torch version. Regenerate
after the pin, never before:

```bash
FLAGOS_CODEGEN_ALL=1 python scripts/codegen/codegen_ops.py
```

Emitted into `csrc/aten/generated/`: `ops.h` (typedefs + `DECLARE_DISPATCHER`),
`ops.cc` (`ADD_IMPL_TO_DISPATCHER`), `cuda_kernels.cc` (boxing kernels), and
`register.inc` (wrappers + `m.impl()` lines).

Two things to check on the output, both cheap and both catching real problems:

Run codegen through the same isolated environment used for the build. For a CUDA
boxing validation, preload only the CUDA-specific libraries from the matching
CUDA torch wheel; keep the active pip torch CPU-only. Do not inherit vendor-box
`PYTHONPATH` or `LD_LIBRARY_PATH` entries, because they can import a second torch
or vendor shim and produce false registration failures.

```bash
FLAGOS_CODEGEN_ALL=1 bash scripts/vendor/with_cuda_libtorch.sh python scripts/codegen/codegen_ops.py \
  2>&1 | tee /tmp/torch-codegen.log

# 1. No dropped operators. Any "SKIP <op>" or "WARN" means an op failed to
#    generate and must be investigated.
! grep -E 'SKIP|WARN' /tmp/torch-codegen.log

# A failed FlagGems import is also a hard failure. It must not be accepted as a
# successful run: otherwise the generator writes an empty Python-kernel file and
# removes the FlagGems routes from every generated config.
! grep -F '[flaggems] import failed' /tmp/torch-codegen.log

# 2. Idempotency. Compare two complete generated patches, not the working tree
#    against HEAD (a valid version port is expected to differ from HEAD).
git diff --binary > /tmp/codegen-1.patch
FLAGOS_CODEGEN_ALL=1 bash scripts/vendor/with_cuda_libtorch.sh python scripts/codegen/codegen_ops.py \
  > /tmp/torch-codegen-second.log 2>&1
git diff --binary > /tmp/codegen-2.patch
cmp -s /tmp/codegen-1.patch /tmp/codegen-2.patch && echo idempotent
```

A non-idempotent generator (dict ordering, absolute paths, timestamps) makes
every future rebase a conflict, so fix it here rather than downstream. Record
the discovered FlagGems count and the exact FlagGems/Triton versions used for
the regeneration; a different FlagGems release is a different generated input,
not merely a runtime dependency refresh.

## Step 3 — the signature reconciliation, which is the actual work

Steps 1–2 are mechanical. This is where a version port takes its time.

### The failure

```
RuntimeError: Mismatch in kernel C++ signatures
  operator: aten::cat
  kernel 1: ... at::ArrayRef<at::Tensor> ...
  kernel 2: ... at::IListRef<at::Tensor> ...
```

A TensorList argument is spelled either `ArrayRef<Tensor>` or `IListRef<Tensor>`
in the dispatcher signature, and **which one is per-operator, decided by
torchgen's own rule** — not a global setting. torch_fl mirrors that rule in
`scripts/codegen/codegen_ops.py` (see the comment block around line 56 and
`use_arrayref`-style logic near line 2065). When PyTorch reclassifies an op
between minors, the mirrored rule goes stale for that op and registration aborts
at import.

### How to resolve it

Fix the **rule or its operator list in the generator**, then regenerate. Never
hand-edit `csrc/aten/generated/*` — the next codegen run reverts it and the bug
returns wearing a different hat.

```bash
# Ground truth for one operator, straight from the installed torch:
python - <<'PY'
from torchgen.model import NativeFunction
# read the schema for the failing op out of torchgen's packaged
# native_functions.yaml and inspect whether it is part of a structured group
PY
```

The empirical rule as of the versions checked: an op belonging to a structured
group takes `IListRef` (e.g. `aten::cat`); a standalone op takes `ArrayRef`. Do
not port the *list* of exceptions across versions blindly — re-derive it from the
installed torchgen. Reassuringly, the delta is often empty: 2.13 → 2.12 needed no
change to the ArrayRef operator set at all, so an empty diff here is a plausible
correct outcome, not evidence you skipped the step.

### Iterate against the real error, one op at a time

The import aborts on the *first* mismatch, so this is a loop, not a batch fix:

```bash
python -c "import torch_fl" 2>&1 | head -20   # names one op
# adjust the generator's rule/list for that op, regenerate, repeat
```

## Step 4 — build and verify

Build with the CUDA operator path on, since that is what codegen produces:

```bash
FLAGOS_BUILD_FLAGGEMS_CPP=OFF FLAGOS_BUILD_FLAGGEMS=OFF pip install -e . --no-build-isolation
```

The generated C++ sources link against the CPU torch libraries, while CUDA
symbols resolve at runtime from the matching external CUDA assets. On this
repository's CUDA path, the top-level CMake project still enables the CUDA
language and may run nvcc compiler identification during configuration; that
is expected for the existing build system and does not mean a CUDA torch was
installed. The important invariant is that the active Python torch remains
CPU-only and no second core libtorch is preloaded. If a CUDA torch is installed
in the environment, stop and recreate the environment before attributing any
error to codegen.

Then, through the preload wrapper:

```bash
bash scripts/vendor/with_cuda_libtorch.sh python -c "import torch_fl; print('ok')"
bash scripts/vendor/with_cuda_libtorch.sh python -m pytest tests/integration/ops/ \
    -m "not flaggems and not flaggems_python" -q
```

### Known-benign failures

Under the external-libtorch scheme (CPU pip torch + preload), the two
`*_out_cuda_override` dispatch-log tests can fail with `Allocator not initialized
for device` from `CUDACachingAllocator.cpp`. Cause: PyTorch primes the CUDA
caching allocator inside `torch.cuda._lazy_init()`, which this scheme
deliberately never calls; a fresh subprocess doing *nothing but* an out-variant op
allocates into a caller-provided `out` before anything warms the allocator.

This is an environment artifact, not a codegen defect — with a real pip CUDA
torch installed the same tests pass. Accept or xfail them; do not "fix" it by
reaching into `torch.cuda` internals, which is precisely what the scheme avoids.
Ascend-marked tests skipping is also expected.

## Step 5 — record the version in the docs that assert it

A version port that leaves the docs claiming the old range is half-done. Update:

- `docs/reference/compatibility.md` — the PyTorch row of "Project Compatibility"
  states the supported range (`>=2.10,<2.11` style) and notes that generated ATen
  bindings are tied to the minor line
- `README.md` — the PyTorch badge
- `pyproject.toml` / `setup.py` — any torch version constraint

## Done criteria

- `torch.__version__` still ends in `+cpu`
- `FLAGOS_CODEGEN_ALL=1 python scripts/codegen/codegen_ops.py` runs clean: no `SKIP`, no
  `WARN`, and a second run leaves no diff
- Operator count matches what the conf lists (no silent drops vs the base branch)
- `import torch_fl` is clean through `scripts/vendor/with_cuda_libtorch.sh`
- `tests/integration/ops/` passes modulo the allocator cold-start tests above
- Version claims in `compatibility.md` / `README.md` match reality
- `ruff check .` and `ruff format --check .` pass — see [[pre-pr-checks]]

## Pitfalls

**Regenerating before pinning torch.** Codegen output is a function of the
installed torch. Pin first, or you regenerate against the old schema and the
diff is noise.

**Hand-editing generated files to clear a signature mismatch.** It works until
the next codegen run. Fix the generator.

**Porting the ArrayRef exception list from another version branch.** That list
*is* the version-specific knowledge. Re-derive it.

**Installing CUDA torch to make an error go away.** It will make errors go away,
including the ones that indicate the external-libtorch path is broken. Keep the
env CPU-only.

**Treating a green suite on one chip as a green port.** This skill is
hardware-independent, but its verification runs on whatever chip you have. Other
platforms are re-validated by their own bringup, and per CLAUDE.md an unvalidated
platform must be marked **not revalidated** rather than presented as measured.
