# Integration harness design: four commands for bringing up a new target

> Status: design proposal. 2026-08-13. Motivated by Kunlun P800, which torch-fl
> currently does not support at all, but written so the harness is reusable for
> every subsequent target rather than being P800-specific.

## 1. Why four commands and not one

An earlier draft of this proposal modelled new-target bringup as a single
end-to-end command. That is the wrong decomposition, because the work splits
along two independent axes that do not co-vary:

- **Which torch version are we building against?** This is hardware-independent.
  It is about ATen schemas, dispatcher signatures, and which `torch==X+cpu` wheel
  the build resolves against.
- **How does this chip execute an operator?** This is hardware-specific and has
  two fundamentally different answers — reuse a vendor-supplied CUDA-compatible
  `libtorch_cuda.so`, or call the vendor's own operator library directly.

Collapsing these into one command means every torch bump re-runs chip discovery,
and every new chip re-derives torch-version facts that are already known. It also
hides the one piece of work that is *unconditionally* required: the device
runtime.

The runtime is the honest common denominator. Whether a chip is CUDA-compatible
or not, torch-fl cannot allocate a tensor on it until the `flagos` runtime
contract is implemented. So the runtime gets its own command, and it is a
prerequisite for both operator paths.

```
                    ┌───────────────────────────┐
                    │ /torch-version-port       │  hardware-independent
                    │ (axis: torch version)     │  runs on its own cadence
                    └───────────────────────────┘

                    ┌───────────────────────────┐
                    │ /runtime-bringup          │  ALWAYS required
                    │ (the ~40-function floor)  │  chip-specific, op-agnostic
                    └─────────────┬─────────────┘
                                  │ prerequisite
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
        ┌───────────────────────┐   ┌───────────────────────┐
        │ /cuda-compat-vendor   │   │ /native-op-backend    │
        │ extract + bundle the  │   │ bind the vendor op    │
        │ vendor libtorch_cuda  │   │ library per operator  │
        └───────────────────────┘   └───────────────────────┘
              CUDA-compatible            not CUDA-compatible
              (metax, dcu, ppu)          (ascend, gcu, musa)
```

`/runtime-bringup` is not a subroutine of the other two — it is a gate. Its exit
criterion (tensors allocate, copy H2D/D2H, streams and events synchronize) is
checkable without a single operator being registered, and both operator commands
assume it has already passed.

## 2. The runtime contract, measured

The "basic 40 functions" is close to exact. Counted from the tree rather than
from memory:

| Surface | File | Count |
|---|---|---|
| C ABI runtime functions | `csrc/include/flagos.h` (`FLAGOS_EXPORT Error_t`) | **28** |
| Caching-allocator virtuals | `csrc/runtime/allocator/device_memory_interface.h` (`= 0;`) | **10** |
| Vendor stream accessor | `GetDefault<Vendor>Stream` / `GetCurrentStream` | 1–2 |
| Build wiring | `csrc/runtime/accelerator/CMakeLists.txt` + `setup.py` selector | 2 sites |

So **38 functions plus wiring** — call it 40. The 28 C functions land in exactly
three files per vendor, and every existing backend implements all 28 (verified:
cuda, musa, gcu, ascend, tsingmicro, bpu each expose 28):

```
csrc/runtime/accelerator/<vendor>/
├── device.cc   #  5  GetDeviceCount, GetDevice, SetDevice,
│               #     DeviceGetStreamPriorityRange, DeviceSynchronize
├── memory.cc   #  9  Malloc, Free, MallocHost, FreeHost, Memcpy, MemcpyAsync,
│               #     PointerGetAttributes, Memset, MemsetAsync
└── stream.cc   # 14  Stream{CreateWithPriority,Create,GetPriority,Destroy,
                #     Query,Synchronize,WaitEvent}
                #     Event{CreateWithFlags,Create,Destroy,Record,
                #     Synchronize,Query,ElapsedTime}
```

Plus `csrc/runtime/allocator/backends/<vendor>_memory.h` implementing the 10
virtuals (`device_malloc`, `device_free`, `get_device_index`, `set_device`,
`get_memory_info`, `event_create`, `event_destroy`, `event_record`,
`event_query`, `memcpy`), with the optional `provides_caching()` escape hatch for
platforms that already ship a mature caching allocator.

This is a **fixed, enumerable, mechanically checkable** contract. That is what
makes it a good harness target: the agent is not designing anything, it is
filling in a known table from a vendor SDK header. Completeness is verifiable by
counting symbols, not by judgement.

---

## 3. Command 1 — `/torch-version-port`

**Axis:** torch version. **Hardware:** none.

### Scope

Adapt torch-fl to a different `torch==X.Y+cpu` line. Everything this command
touches is schema- and ABI-level; nothing in it knows what chip will eventually
run the kernels.

```bash
/torch-version-port 2.9
```

### What it does

1. **Create/checkout the version branch.** Branches are per-torch-minor in this
   repo (`main` tracks 2.10.x; `2.9` is a sibling, not a descendant of a chip
   port). Establish the branch before any other command runs.
2. **Pin the environment.** `torch==2.9.x+cpu` — CPU-only, deliberately. The
   CUDA-compatible path supplies GPU symbols out-of-band (command 3), so a CUDA
   pip torch in the env is a liability, not a help.
3. **Re-run ATen codegen against the new schema.** `scripts/codegen/codegen_ops.py`
   reads torchgen's packaged `native_functions.yaml`, so a torch bump changes its
   output. Regenerate `csrc/aten/generated/{ops.h,ops.cc,cuda_kernels.cc,register.inc}`.
4. **Reconcile per-operator signature splits.** The known failure mode is
   `Mismatch in kernel C++ signatures` at `import torch_fl`, caused by ops whose
   dispatcher signature uses `IListRef` vs `ArrayRef` differently between
   versions. The existing `cuda-op-integration` skill documents this; this
   command inherits that procedure and its `ARRAYREF_OPS` list.
5. **Prove codegen idempotency.** A second run must produce no diff. This is a
   hard gate — a non-idempotent generator makes every future rebase a conflict.
6. **Update the declared range.** `pyproject.toml`, `docs/reference/compatibility.md`
   (the `PyTorch` row currently reads `>=2.10,<2.11`), and the README badge.

### Exit criteria

- `python scripts/codegen/codegen_ops.py` emits the full conf op count with no WARNINGs
- second codegen run leaves `git diff` empty
- `import torch_fl` clean, no signature mismatch
- `torch.__version__` still ends in `+cpu`
- existing platform tests for at least one already-supported chip still pass

### Relationship to existing skills

This command is a **generalization of `cuda-op-integration`**, which today is
written specifically around CUDA boxing on a new torch branch. The version-port
concerns (codegen, signature splits, idempotency) belong here; the
CUDA-boxing-specific concerns (external `libtorch_cuda.so`, `LD_PRELOAD`) belong
in command 3. Splitting them is the main refactor this design implies for the
existing skill.

---

## 4. Command 2 — `/runtime-bringup`

**Axis:** hardware. **Operator-agnostic.** Required for every target.

### Scope

Implement the 38-function contract in §2 for a new chip and nothing more. No
operators, no boxing, no kernel library. The deliverable is a build in which a
`flagos` tensor can be allocated, filled from host memory, copied back, and
synchronized — with every operator still falling back to CPU.

```bash
/runtime-bringup kunlun --sdk-path /opt/kunlun/xre
```

### Naming note

Do not use `xpu` as the `ACCELERATOR` value for Kunlun. `torch.xpu` is PyTorch's
own Intel GPU namespace, and a `ACCELERATOR=xpu` selector inside a PyTorch
plugin will be read as Intel by every future reader. Use `kunlun`.

### Phase 2a — SDK reconnaissance (read-only)

Discover, from the vendor SDK, the concrete answer to each of the 38 slots.
Output is a structured mapping report, committed as
`docs/vendors/<vendor>/runtime-api-map.md`:

- the runtime shared library and its header set (for Kunlun: XRE — `libxpurt.so`
  and `xpu/runtime.h`, to be confirmed against the actual SDK, not assumed)
- device: enumerate / get / set / synchronize entry points
- memory: device alloc/free, pinned-host alloc/free, sync + async memcpy, memset,
  pointer attribute query, and free/total memory query
- stream & event: create-with-priority, query, wait, elapsed-time. Priority range
  is the slot most often absent; record the clamp behaviour rather than inventing one
- error enum → `Error_t` mapping (`Success`, `ErrorNotReady`, `ErrorInvalidDevice`,
  `ErrorMemoryAllocation`, `ErrorUnknown`)

Two findings must be recorded explicitly because they change the generated code
rather than just filling it in:

1. **Missing slots.** If the SDK has no event-elapsed-time, or no async memset,
   that is a real capability gap. It must be recorded and surfaced, not silently
   stubbed with `return Success`. A stub that lies about completion produces
   silent wrong results downstream.
2. **Whether the vendor ships a caching allocator.** Determines
   `provides_caching()`. Getting this wrong costs either performance (redundant
   double-caching) or correctness of memory stats.

### Phase 2b — Generate the three files plus the allocator

Emit `device.cc` / `memory.cc` / `stream.cc` and
`allocator/backends/<vendor>_memory.h` from the mapping report, using an existing
backend as the structural template. Pick the template by shape, not
alphabetically: `musa` for a CUDA-shaped runtime with renamed symbols, `ascend`
for a runtime with its own idioms and a self-managed allocator.

### Phase 2c — Build wiring

- `csrc/runtime/accelerator/CMakeLists.txt` — an `elseif(ACCELERATOR STREQUAL
  "<vendor>")` arm for the source subdirectory and one for include/link paths
  (both lists exist separately in that file; both need the arm)
- `setup.py` — the `ACCELERATOR` branch for SDK detection, with a clear
  actionable error when the SDK is absent, matching the DTK/BPU precedent
- `torch_fl/configs/backends_<vendor>.conf` — created, and for this command
  deliberately near-empty: every operator falls back to CPU

### Exit criteria — checkable without any operator

```python
import torch, torch_fl
t = torch.empty(4, 4, device="flagos")            # allocator reached
h = torch.arange(16, dtype=torch.float32).reshape(4, 4)
d = h.to("flagos"); back = d.cpu()                # H2D + D2H round-trip
assert torch.equal(h, back)
torch_fl.synchronize()                            # stream/event path live
print(torch_fl.device_count(), torch_fl.memory_allocated())
```

Symbol-count check as a mechanical gate:

```bash
grep -c 'FLAGOS_EXPORT Error_t' csrc/include/flagos.h     # expect 28
grep -oE '^Error_t [A-Za-z]+\(' csrc/runtime/accelerator/<vendor>/*.cc | wc -l
```

The two numbers must agree, and every unimplemented slot must be an explicit
documented gap rather than a missing symbol found at link time.

---

## 5. Command 3 — `/cuda-compat-vendor`

**Axis:** hardware. **Precondition:** command 2 passed.

### Scope

For a chip whose vendor ships a CUDA-compatible PyTorch build, harvest the
vendor's own `libtorch_cuda.so` and reuse PyTorch's CUDA kernels through zero-copy
device-metadata boxing. Zero kernels are written.

```bash
/cuda-compat-vendor kunlun --vendor-torch /opt/kunlun/torch
```

### Why this works, and its one hard constraint

`docs/vendors/cuda/external-libtorch-cuda.md` records the measured result: a
CPU-only pip torch plus an externally supplied `libtorch_cuda.so` registers real
CUDA implementations for `aten::mm` / `add` / `_softmax` / `bmm` and computes
correct results (`mm max_err = 9.5e-06`). The hard constraint is **load timing**:
the library must be loaded before `import torch`, hence `LD_PRELOAD` and
`scripts/vendor/with_cuda_libtorch.sh`.

### Phase 3a — Establish CUDA compatibility (a gate, not an assumption)

This command must not be entered on a hunch. The check is concrete: take the
vendor's torch install and confirm it registers a **CUDA** dispatch key rather
than occupying PrivateUse1.

```bash
python -c "
import torch
print(torch.__version__)
print(torch._C._dispatch_dump('aten::mm'))
"
```

- **CUDA key present** → this command applies (the metax / dcu / ppu pattern).
- **PrivateUse1** → stop and use command 4. torch-fl itself occupies
  PrivateUse1, so there is no key left to box into. This is not a theory: the
  Ascend write-up records both `libtorch_npu` fallback and dispatch interception
  as measured and closed off.

For Kunlun specifically, this gate decides the whole route and its answer is not
yet known in this repo. It must be measured on a real P800 with the real vendor
SDK before either operator command is picked. Do not infer it from marketing
material about CUDA source compatibility — source compatibility of kernel code
says nothing about which dispatch key the vendor's torch registers.

### Phase 3b — Harvest and bundle

The extraction target is the vendor's torch install (or wheel), not an NVIDIA
wheel:

1. Locate `libtorch_cuda.so` + `libc10_cuda.so` in the vendor's `torch/lib`.
2. **Verify the exact version match** against the branch's torch version.
   Constraint 3 of the CUDA doc: versions must match exactly. A vendor torch
   built on 2.9.0 pairs only with the `2.9` branch — this is precisely why
   command 1 exists as a separate axis, and why the branch is chosen first.
3. Stage into the assets dir and bundle. `setup.py::_bundle_cuda_assets()`
   already implements this, gated on `ACCELERATOR == "cuda"` and reading
   `FLAGOS_CUDA_ASSETS_DIR`. Extending that gate to accept a vendor selector is
   the code change this command needs — the copy logic itself is done, including
   the size-comparison skip that avoids re-copying ~1GB.
4. Record the provenance: vendor SDK version, torch version, `.so` sizes and
   hashes. A bundled 1GB binary of unclear origin is a supply-chain problem, and
   redistribution terms for a vendor `.so` are a licensing question for a human,
   not for the agent. The harness stages and documents; it does not decide.

### Phase 3c — Route operators through boxing

Populate `backends_<vendor>.conf` from `backends_cuda.conf`, then narrow it by
measurement. Vendor CUDA-compatibility is rarely total; the honest output is a
conf where each entry was tested, and unsupported ops fall back explicitly.

### Exit criteria

- vendor `.so` version matches the branch torch version exactly
- `import torch_fl` clean through `with_cuda_libtorch.sh`
- `torch.__version__` still `+cpu` — no vendor CUDA torch installed into the env
- `tests/integration/ops/` passing for routed ops, with failures either fixed or
  removed from the conf, never left silently routed
- provenance recorded in `docs/vendors/<vendor>/installation.md`

### Known cold-start artifact

The first CUDA op in a fresh process can hit `Allocator not initialized for
device`, because this scheme deliberately never calls
`torch.cuda._lazy_init()`. It surfaces on out-variants (`mm.out`, `bmm.out`) run
as the very first op. It is an artifact of the external-libtorch scheme, not a
defect in the port; do not "fix" it by reaching into `torch.cuda` internals,
which is exactly what the scheme avoids.

---

## 6. Command 4 — `/native-op-backend`

**Axis:** hardware. **Precondition:** command 2 passed and command 3's gate
returned "not CUDA-compatible".

### Scope

Bind the vendor's own operator library per operator. Each kernel marshals its
arguments, allocates and shape-infers its own output, and invokes the vendor
call. This is the Ascend/GCU/MUSA route.

```bash
/native-op-backend kunlun --op-lib libxdnn.so --category unary
```

### Reuse what already exists

The Ascend aclnn codegen is a working precedent and its architecture transfers
directly. Three properties matter:

- **Dispatcher declarations are reused, not re-declared.** `generated/ops.h`
  already has the `XxxFn` typedefs and `DECLARE_DISPATCHER`; `ops.cc` has
  `ADD_IMPL_TO_DISPATCHER`; `register.inc` binds the ATen op. A native backend
  emits only `REGISTER_IMPL_TO_DISPATCHER(XxxFn, xxx_dispatcher,
  Backend::k<Vendor>, XxxKernel<Vendor>)`.
- **Symbol names must agree with the ATen codegen.** Reuse
  `codegen_ops.py:schema_to_cpp_name()` or the link fails.
- **Category-driven, not operator-driven.** The vendor calling convention is
  uniform; the real variation is argument marshalling and output allocation,
  which is consistent *within* a category. Ascend reached 63 categories covering
  138 operators this way. Expand category by category, verifying each against
  CPU on real hardware before starting the next.

### Phase 4a — Recon the operator library

The critical asymmetry versus command 3: **the ATen schema does not carry the
vendor API name or the marshalling rules.** A native backend therefore needs a
hand-maintained mapping table, and the agent's job is to propose entries and
verify them, not to guess them. Record: the two-stage vs single-call convention,
the tensor/scalar/int-array wrapper types, the dtype enum mapping, and how
workspace is requested.

### Phase 4b — Emit kernels by category

Generate into `csrc/aten/backends/<vendor>/generated/<vendor>_kernels.cc`.
Hand-written and generated registrations are **mutually exclusive** — a duplicate
registration for the same backend slot is a hard error at import — so the
generator reads a skip list. The Ascend principle applies: anything a category
can express goes through codegen; hand-writing is reserved for what codegen
cannot express.

### Phase 4c — Route and verify per operator

Add `op = <vendor>` lines to `backends_<vendor>.conf` only for operators that
have passed a CPU-comparison test on real hardware. Unrouted operators reach
`cpu_fallback`.

### Exit criteria

- each claimed operator has a recorded max-error against CPU on real hardware
  (the Ascend bar: unary ≤4.4e-5, binary ≤4.7e-6, comparisons exact)
- no duplicate backend-slot registration
- conf regeneration idempotent
- `docs/reference/operator-support.md` updated from **measured** results, per
  CLAUDE.md's operator-support rule — routing configuration is not evidence

---

## 7. Applying this to Kunlun P800

The sequence is fixed by the dependency graph, and the branching decision sits in
the middle where it can be made from evidence:

1. `/torch-version-port 2.9` — create the `2.9` branch, pin `torch==2.9.x+cpu`,
   regenerate ATen bindings. No P800 needed; can start immediately.
2. `/runtime-bringup kunlun --sdk-path <XRE>` — the 38-function floor. Needs the
   SDK; needs a P800 only for the final round-trip check.
3. **Run command 3's Phase 3a gate on real hardware.** Dump the dispatch table of
   the vendor's torch. This is the one unknown that determines everything after
   it, and it is cheap to answer once hardware is in hand.
4. Then either `/cuda-compat-vendor kunlun` or `/native-op-backend kunlun`.

Two things block on access rather than on design: the XRE/XDNN SDK, and a P800
for the gate in step 3. Steps 1 and 2a can proceed without either — the
reconnaissance phase is read-only over headers.

## 8. What changes in the existing skills

- `cuda-op-integration` splits: version-port concerns move to command 1, the
  external-libtorch and boxing concerns become command 3. The skill as written
  conflates them, which is why it reads as CUDA-only despite most of its content
  being version-generic.
- `pre-pr-checks` is unchanged and remains the final step of all four commands.
- The four commands share the reconnaissance-report convention
  (`docs/vendors/<vendor>/*-api-map.md`): a committed, human-reviewable artifact
  produced before any code is generated. It is the review surface — reviewing a
  mapping table is tractable, reviewing 138 generated kernels is not.

## 9. Design principles carried across all four

1. **Recon is read-only and its output is committed.** Generation never happens
   in the same step as discovery.
2. **Gaps are recorded, never stubbed.** A missing SDK slot is documented as a
   capability gap. A stub returning `Success` for an operation that did not
   happen is worse than a link error.
3. **Support claims come from measurement.** Every "supported" entry traces to a
   test run on real hardware, per CLAUDE.md. Routing config is not evidence.
4. **Idempotency is a gate, not a nicety.** Every generator's second run must
   produce no diff.
5. **The agent stages, humans decide.** Redistributing a vendor `.so`, promoting
   a platform's status, merging a branch — the harness prepares these with
   provenance and stops.
