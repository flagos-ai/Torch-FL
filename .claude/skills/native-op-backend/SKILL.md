---
name: native-op-backend
description: >
  Enable operators on an accelerator that is NOT CUDA-compatible, by binding its
  native operator library (Ascend ACLNN, Enflame topsaten, Moore Threads mudnn,
  Kunlun XDNN, …) per operator through category-based codegen. Use this after
  runtime-bringup passes and after cuda-compat-vendor has been ruled out by
  measurement. Covers: the category system that makes codegen tractable, writing
  the vendor op_api_common/op_preparation layer, the OPS mapping table, the
  kAscend-style backend slot and conf routing, and per-op CPU comparison tests.
---

# Native operator backend (torch_fl, non-CUDA-compatible chip)

## When this is the right path

Only after [[cuda-compat-vendor]] Step 1 has been *measured* and failed. This
route is roughly an order of magnitude more work, so the cheap check is always
worth running first.

**Prerequisite:** [[runtime-bringup]] green. Every kernel here allocates its
output through the flagos allocator and runs on a flagos stream; neither exists
until the 28-function contract is implemented.

## Why the CUDA codegen cannot be copied

On the CUDA-compatible path a generated kernel body is a single line —
`at::op(args)` — because `DeviceBoxingGuard` rewrites device metadata and
PyTorch's own registered kernel does the work. **That shortcut does not exist
here.** A native kernel body must call the vendor C ABI itself, which means it has
to marshal arguments into vendor wrapper types, **allocate its own output and
infer the output shape and dtype**, and then run the vendor's call sequence
(often two-stage: query workspace size, then execute).

| | CUDA-compatible codegen | native codegen |
|---|---|---|
| Kernel body | one line, `at::op(args)` | marshalling + output allocation + vendor call |
| Information source | entirely in the ATen schema | the schema **does not carry** the vendor API name or marshalling rules |
| Coverage strategy | enumerate every operator | **by category + mapping table**, expanded category by category |

`docs/vendors/ascend/aclnn-codegen.md` is the reference write-up for this whole
approach — read it before starting. It also documents two dead ends that were
measured and closed off (boxing into torch_npu's key, and intercept-based
routing), so you do not need to re-explore them.

## The central insight: categories, not operators

Hand-writing 138 kernels is intractable; generating 138 kernels from 63
**categories** is not. The vendor calling convention is highly uniform, and the
real variation — how arguments are marshalled and how the output is allocated —
is **completely consistent within a category**.

Ascend reached 138 operators from 63 categories. Adding a category costs one
template; adding an operator to an existing category costs **one line** in a
mapping table.

Representative categories (see the Ascend doc for the full table):

| category | Criterion | Output shape / dtype | Body shape |
|---|---|---|---|
| `unary` | 1 Tensor in, Tensor out | = input | `vendorOp(self, out)` |
| `unary_bool` | 1 Tensor in, predicate | input shape, **bool** out | `vendorOp(self, out)` |
| `binary` | 2 Tensors, broadcasting | `at::infer_size` | `vendorOp(self, other, out)` |
| `binary_scalar` | Tensor + Scalar | = input | scalar wrapper + `vendorOp` |
| `reduce` | dim list + keepdim | reduced shape | int-array wrapper + `vendorOp` |
| `gemm` | matmul family | contracted shape | often needs transpose flags |

**Do not start by picking your favourite operators.** Start with `unary`, get one
operator through the entire pipeline to a passing CPU-comparison test, and only
then widen. The first operator pays all the infrastructure cost; the second
should cost minutes.

## Step 1 — SDK reconnaissance

Answer these five questions from the headers before writing any code. Each one
determines a template decision, and guessing any of them means rewriting every
category later.

```bash
find /opt/<vendor> -name '*.h' | xargs grep -l 'Tensor\|tensor' | head
nm -D --defined-only /opt/<vendor>/lib/lib<vendorops>.so | grep ' T ' | head -40
```

1. **Tensor descriptor type** — what struct describes a tensor (shape, strides,
   dtype, data pointer)? This becomes your `<Vendor>TensorWrapper`.
2. **Call convention** — single call, or two-stage
   `GetWorkspaceSize` + `Execute`? Two-stage needs a workspace allocation in the
   macro.
3. **dtype enum mapping** — `torch.float32` → which vendor enum? Enumerate all
   dtypes you intend to support; a missing mapping must raise, not silently pick a
   default.
4. **Output ownership** — does the vendor allocate the output, or does the caller?
   (Caller, almost always — hence `op_preparation`.)
5. **Stream/context argument** — what does the op take, and how do you get it
   from the flagos stream? This is where [[runtime-bringup]]'s
   `GetDefault<Vendor>Stream` accessor gets used.

Record the answers in `docs/vendors/<vendor>/` as you go. Reconstructing them
later from generated code is far harder than writing them down now.

## Step 2 — the vendor support layer (hand-written, once)

Two headers, mirroring the Ascend layout:

```
csrc/aten/backends/<vendor>/
├── op_api_common.h      # <Vendor>TensorWrapper / ScalarWrapper /
│                        # IntArrayWrapper + the EXEC_<VENDOR>_CMD macro
├── op_preparation.h     # apply_tensor_without_format(): allocate an output
│                        # tensor with a given shape/dtype on the flagos device
└── generated/
    └── <vendor>_kernels.cc   # emitted by the generator, never hand-edited
```

`EXEC_<VENDOR>_CMD` must hide the entire call sequence — workspace query,
workspace allocation, execute, error check — so that a category template stays
one readable line. If a template needs to know about workspaces, the macro is not
doing its job.

Error handling: convert every vendor status into a `TORCH_CHECK` with the
operator name and the vendor error string. A native backend without this is
extremely unpleasant to debug, because failures surface as wrong numbers rather
than exceptions.

## Step 3 — the generator

Create `scripts/codegen/codegen_<vendor>.py` modelled on `scripts/codegen/codegen_ascend.py`,
which has four parts:

- **`OPS`** — `schema op name → (category, vendor-name override)`. **The only
  hand-maintained data.**
- **`SKIP`** — ops deliberately hand-written for this backend. A duplicate
  registration in the same backend slot **crashes at import**, so an op is either
  generated or hand-written, never both.
- **`CATEGORIES`** — `category → kernel body template`.
- **Name reuse** — import `schema_to_cpp_name` from `scripts/codegen/codegen_ops.py`.
  If `XxxFn`/`xxx_dispatcher` names diverge from `csrc/aten/generated/ops.h`, the
  link fails.

### Declare no dispatchers of your own

This is the most common structural mistake. `csrc/aten/generated/ops.h` already
has the `XxxFn` typedefs and `DECLARE_DISPATCHER`; `ops.cc` already has
`ADD_IMPL_TO_DISPATCHER`; `register.inc` already binds the ATen operator to
`xxx_dispatcher`. Your generator emits **only** the kernel plus:

```cpp
REGISTER_IMPL_TO_DISPATCHER(XxxFn, xxx_dispatcher,
                            Backend::k<Vendor>, XxxKernel<Vendor>);
```

This hangs your kernel in the `k<Vendor>` slot of a dispatcher that already
exists. Declaring your own typedefs produces either a duplicate-symbol link
failure or, worse, a second dispatcher nothing routes to — so the kernel compiles,
registers, and never runs.

Reuse `schema_to_cpp_name` from `scripts/codegen/codegen_ops.py` rather than
reimplementing the name mangling.

## Step 4 — backend slot and routing

Three edits outside the generator:

1. **Backend enum** — add `k<Vendor>` alongside `kAscend` in the `Backend` enum
   (`csrc/aten/dispatcher.h`), and teach `GetBackendForOp` to resolve it.
2. **Conf file** — `torch_fl/configs/backends_<vendor>.conf`, with `op = <vendor>`
   lines. This is what routes an op to your slot **at runtime**; a registered
   kernel with no conf line never executes. The generator should rewrite a
   `# --- generated ---` block at the end of this file idempotently.
3. **CMake** — generated sources under `csrc/aten/backends/<vendor>/` must be
   excluded from other platforms' builds. They are covered by the generic
   `VENDOR_KERNEL` gate in `csrc/CMakeLists.txt` (which keeps only the
   `ACCELERATOR` vendor's directory): add `<vendor>` to its directory list,
   no per-vendor switch needed.

Registration and routing are **two separate mechanisms**. A kernel that is
registered but unrouted is the most common "my kernel does nothing" cause; check
the conf before debugging the kernel.

## Step 5 — verify each operator against CPU

Per-operator CPU comparison is the only trustworthy check — a native kernel that
returns plausible-looking wrong numbers is the characteristic failure of this path.

```bash
ACCELERATOR=<vendor> VENDOR_KERNEL=1 FLAGGEMS_CPP=0 \
  pip install -e . --no-build-isolation

FLAGOS_BACKEND_CONFIG=torch_fl/configs/backends_<vendor>.conf \
  pytest tests/integration/ops/ -m "<vendor>" -v
```

Pass **all** the build env vars explicitly. `setup.py` defaults to
`ACCELERATOR=cuda`, and omitting them fails at the cmake configure stage with an
error that does not obviously point back at the missing variable.

Tolerances measured on Ascend, useful as a sanity reference: unary ≤4.4e-5,
binary ≤4.7e-6, comparison ops exact. A unary op off by 1e-2 is a bug, not
hardware noise.

Cover per operator: several shapes (including empty and 1-element), every dtype
you claim, non-contiguous inputs (a wrapper that ignores strides passes contiguous
tests and fails here), and broadcasting for binaries.

## Growth strategy

1. `unary` category, one operator, end to end until the CPU comparison passes.
2. Rest of `unary` — should be one `OPS` line each.
3. `binary` + `binary_scalar` (introduces broadcasting and `at::infer_size`).
4. `reduce` (introduces int-array marshalling and keepdim shape math).
5. `gemm` and the rest, by workload need.

After each category: run the suite, update `docs/reference/operator-support.md`
from **measured** results, commit. Do not accumulate five categories of unverified
kernels — attributing a failure across them costs more than the incremental runs.

## Done criteria

- Generator runs clean, is idempotent (second run leaves no diff), emits no
  duplicate `k<Vendor>` registration
- `SKIP` and the generated set are disjoint — an op is generated **or**
  hand-written, never both, or import crashes
- Every generated op has a conf line routing to it
- Every claimed op has a passing CPU-comparison test at the tolerances above
- `docs/vendors/<vendor>/` records the five Step-1 answers and the category table
- `docs/reference/operator-support.md` updated from measured results per CLAUDE.md

## Related

[[runtime-bringup]] (prerequisite) · [[cuda-compat-vendor]] (must be ruled out
first) · [[torch-version-port]] (supplies `ops.h`/`register.inc`) ·
[[pre-pr-checks]] (before opening the PR)
