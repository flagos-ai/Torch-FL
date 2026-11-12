# Compiling FlagGems' Python host logic into native Torch-FL operators

Status: design proposal for #822, revised against review on #823. Nothing here is
implemented. Phase 0 is a sequence of gates whose purpose is to fail early and cheaply.

## Problem

An eager ATen call on `flagos` reaches FlagGems through a generic caller that acquires
the GIL, constructs Python arguments and invokes the resolved Python callable. Everything
the wrapper does around the kernel - argument normalisation, metadata arithmetic, dispatch
and specialisation, allocation - runs in Python on every call.

#820 removed part of that cost by recording one concrete launch and replaying it from the
caller. Measured there on H100: `mul` 512x512 went from 82.55 us (wrapper) to 6.81 us
(replay), against 7.27 us for the vendor operator. But that result rests on two things this
project may not keep: a persistent per-plan pool of output tensors, and a plan that is a
snapshot of one observed call with concrete dimensions.

## Positioning

The feature is: **inside Torch-FL's existing eager dispatch, transparently compile the host
logic of an individual FlagGems operator.** A user script stays byte-for-byte unchanged.

- No `torch.compile`, no decorator, no capture context, no dedicated API, no manual warmup.
- Users keep running plain Python eager. Nothing is required of them to opt in, and nothing
  is required of them to keep working.
- "Removing Python overhead" is scoped precisely: once a supported call has entered the
  backend, the cache-hit path of preparation, allocation and submission runs without
  Python. It is not a claim about the model, the training loop or the dispatcher.

## Bounded support and eager semantics

The initial scope is metadata-driven, functional operators with no input gradients. This is
a statement about what is accelerated, not a requirement on the user: **callers are not
asked to enter `inference_mode()`** or otherwise change how they call the operator.

Sizes and scalars may be symbolic, or explicitly specialised with a guard. What must not
happen is freezing the first call's values silently.

Autograd, mutation, RNG, complex aliasing and tensor subclasses fall back transparently
until each is verified. The native path must use the correct allocator, device and current
stream, and preserve output count, dtype, shape, strides, lifetime and error semantics.

The fallback decision happens before anything is submitted. Once an operator has partially
executed, the original implementation is never re-run unconditionally: that would replay
side effects.

## Performance model

A warm call's budget, from #820's measurements on H100, same host and stack: the launch
itself was 4.33 us inside C++, and a plugin dispatch entry adds about 1 us.

Allocation is the part that decides the ceiling. `at::empty` on the `flagos` device
measured about **4.0 us**, and the vendor's 7.27 us for `mul` already includes an
allocation of its own, so comparing our unoptimised allocation against the vendor's total
is not like for like:

| composition | cost | ratio to vendor (7.27 us) |
| --- | ---: | ---: |
| launch only (#820) | 4.33 us | 0.60x |
| launch + dispatch entry | ~5.3 us | 0.73x |
| launch + entry + `at::empty` (4.0 us) | ~10.6 us | 1.45x |
| launch + entry + allocation at 1.0-1.5 us | 6.8-7.8 us | 0.94-1.07x |

Two consequences for the acceptance criteria below. First, an allocation prototype is part
of the measurements, not a later optimisation. Second, **vendor parity is a separate goal**:
this project's acceptance is that the warm CPU submission cost improves by more than
measurement noise and that the synchronised end-to-end time does not regress.

## Architecture

```text
ATen eager call on flagos
  -> backend caller, before the GIL is taken and Python arguments are built
     -> lookup a native host program for this operator and this call
        hit:  native path
                guard: dtype/rank/layout/offset/alignment/mode/device
                compute shapes and scalars from this call's metadata
                select device variant, allocate, submit 1..N launches
                return ATen-compatible results
        miss/unsupported: the existing FlagGems Python route
```

### Hook point

The lookup happens in `CallPythonOp_Generic`, before the GIL is acquired and before Python
arguments are constructed - the position #820 established for its replay. It must cover
positional arguments, keyword arguments, tuple and multi-output returns, and preserve the
ATen schema, overload and outer dispatcher layering.

### Capture protocol

The first call behaves exactly as it does today. Capture happens around it, and never
touches the real inputs:

1. The first call executes the existing implementation and its result is kept and returned.
2. Symbolic capture follows, using metadata and the actual callable, with FakeTensor or
   symbolic metadata. **No kernel is submitted against real inputs during capture.**
3. If capture fails, the first call's result is still what the caller receives, and the
   operator keeps its existing route. The operator is not re-run.
4. An internal re-entrancy barrier prevents helper ATen operations issued during capture
   from triggering compilation again.
5. Autotune trial runs are recorded separately from the operator's real launch sequence; a
   tuning run is not the operator's launch.
6. A capture failure leaves neither the current result nor later state modified.

Only host paths confirmed safe to capture are admitted. A callable with non-isolatable
Python side effects, unreconstructable state, or data-dependent behaviour keeps its
existing implementation.

### What a host program contains

- Expressions over `shape`, `strides`, `numel`, `storage_offset`, alignment and dtype, and
  the scalar arguments of the call - evaluated per call, never frozen to the first call's
  values.
- Allocation, layout and view construction, and the metadata-dependent branches around
  them, each with the guard that decides it.
- The original Triton kernel, its grid, its argument mapping, its constexprs and the
  compile conditions under which it is valid.

An artifact that is only a concrete launch snapshot, or that contains an opaque Python
call, does not satisfy this: it is what #820 already did.

### Frontend

Dynamo with symbolic shapes and FX is the first experimental frontend, applied to the real
FlagGems callable. Raw Triton calls have to be exposed through `wrap_triton` or an
equivalent mechanism so they enter the tracing path rather than disappearing as opaque
calls. `TorchDispatchMode` is used to observe ATen operations, not as the capture mechanism.

FlagGems wraps many operators in `LibEntry`, heuristics and autotuner machinery. The plan is
a shared adapter per wrapper type, not per operator. **Whether the internal Dynamo/Inductor
route works at all under CPU-torch + PrivateUse1 is a Phase 0 hypothesis**, not an
assumption.

Caching a compiled function that still needs Python to execute is not success. Neither is
having Inductor replace the operator's device kernel: the kernel stays FlagGems' Triton
kernel.

### Native exit

Prefer reusing an existing native-wrapper mechanism over writing a new one. If one can be
reused, the backend hook is connected to the ordinary eager entry and the design collapses
into a shared adapter layer - which is the cheapest outcome and the one Phase 0 looks for
first.

Only if capture succeeds and the ready-made codegen is blocked by integration problems does
a restricted host IR plus a small native backend become the topic. That IR is not built
ahead of that finding.

### Two cache layers

| layer | holds | keyed by |
| --- | --- | --- |
| host program | expressions, allocation, variant selection | operator + its parameterised class |
| device variant | concrete constexprs, specialisation, ABI, target, compile options | the variant the host program selects |

A change of size reuses the host program and may select a different device variant. A new
variant may take the cold path, but an existing variant's guard and selection must be
native. Both layers and a failure cache are separately bounded; eviction never affects a
call in flight, and retired resources stay accounted until nothing references them. No old
inputs or outputs are retained to make a hit cheap.

## Phase 0 gates

Phase 0 does not build a general IR. Each gate has an acceptance criterion, and failing one
stops or redirects the work rather than leading into the next.

| gate | experiment | acceptance |
| --- | --- | --- |
| **0. Freeze the contract** | Pin dependencies, the real callables, device, build, workload and measurement method | Cold-start budget, cache capacity and measurement tolerance are fixed before measuring; no performance conclusion is extrapolated across different workloads |
| **1. User code unchanged** | One native eager script, run through the real dispatcher; experiment groups switched by internal configuration only | No new user-side steps; the first call is not run twice; a capture failure returns normally; helper operations do not recursively trigger compilation |
| **2. Capture complete** | Real `mul` first, then `softmax` with size expressions and branches | Expressions, guards, allocation and the original kernel call are captured; symbolic capture submits no real GPU work. A concrete launch snapshot or an opaque Python call fails this gate |
| **3. Native exit works** | Reuse a native wrapper if possible, then connect to the ordinary eager entry | At least one operator runs end to end under CPU-torch + PrivateUse1; the original kernel is preserved; on a cache hit the backend path does not enter Python, shown by both a profiler and entry counters |
| **4. Parameterisation and fallback** | Vary sizes, layout, offset, broadcasting, scalars, default arguments and empty inputs; cross constexpr boundaries; keep several live outputs | Results, layout, exceptions and lifetime are correct; the host program does not gain an artifact per size; a variant is selected or the call falls back before submission. Unsupported training, mutation and stream cases still behave correctly |
| **5. Coverage potential** | Support table for pointwise, inner and non-inner `softmax`, a multi-launch reduction, and a multi-output operator | Each supported case is explained by shared capture rules, and each failure is attributed to expressions, control flow, wrappers, ABI or codegen |
| **6. Performance value** | Same workload: existing route, full native route and the allocation prototype; the vendor operator reported separately | Warm CPU submission improves by more than the measurement noise; synchronised end to end does not regress beyond the agreed tolerance; cold start and cache growth stay within the budget from gate 0 |

Performance acceptance includes the guard, variant selection, allocation and release.
First call, compile time, warm CPU submission and synchronised end to end are reported
separately.

### Phase 0 exit

- **Capture complete and a native wrapper reusable:** converge on a shared adapter layer and
  shrink the self-built backend.
- **Capture complete but ready-made codegen blocked by integration:** only then evaluate a
  restricted host IR and a small native backend.
- **Expressions or guards unrecoverable, or the user's eager usage would have to change:**
  the route fails; stop expanding it. A new backend cannot restore semantics lost during
  capture.

## Phase 0 result: gates 3 and 6, measured on `softmax` and `mul`

Phase 0 has been run on H100 against a real build. This section states what was measured and
what it says about the gates, and is written after the fact rather than as a plan.

### Gate 3, native exit: passed

An ordinary eager call, with no capture and no extra user steps, reaches a parameterised host
program in C++ and submits the original Triton kernel through the driver API. With a plan
registered for `flag_gems.mul` and a 4096x4096 float32 `a * b`:

| arm | warm CPU submission |
| --- | ---: |
| eager `torch.mul`, native host program | **8.24 us** |
| eager `torch.mul`, Python route | 56.19 us |
| direct `flag_gems.mul` call | 51.32 us |

Every one of the 230 benchmark calls took the host program, checked by an entry counter, and
every result matched the reference. The 8.24 us is the whole chain an eager call pays: ATen
dispatch, the plugin's own dispatcher, host-program lookup, a C++ `at::empty_like`, and the
driver submission.

An earlier stage of the same phase measured the parameterised host program building its
geometry from tensor metadata across six shapes from 2-D to 4-D (grids 256, 21, 64, 1, 15 and
32768), one artifact for all of them, with the results matching. That is the gate 4 property
that mattered most: no artifact per size.

### Gate 6, performance value: passed on submission

The vendor operator measured on the same machine was `torch.softmax` at 8.95 us, and `mul` is
the heavier of the two. The measured 8.24 us is therefore at vendor-kernel parity for an eager
call, which is the claim this design exists to support.

The decomposition behind that number, measured separately rather than assumed:

| component | measured |
| --- | ---: |
| Triton Python launcher for the same kernel | 15.43 us |
| direct `cuLaunchKernel` from Python | ~6.0 us |
| `at::empty_like` on `flagos`, 1 element | 3.97 us |
| `at::empty_like` on `flagos`, 4 MiB | 4.10 us |
| `at::empty(1)` on CPU torch, for scale | 3.00 us |
| C++ `at::empty` through a dispatcher entry | 4.65 us |

Two corrections to the model above came out of this, and both are the reason the numbers in
this section differ from the estimate it was written with:

- **Allocation is mostly not allocation.** One element and four MiB differ by 0.11 us, so the
  allocator's own work is about a tenth of a microsecond and the rest is the fixed cost of
  calling `torch.empty` at all - which CPU torch pays too, 3.00 us for a single element.
  Moving allocation into C++ did not reduce it (4.65 us through a dispatcher entry against
  4.05 us from Python). The 1.0-1.5 us allocation row in the table above is therefore not
  reachable this way, and the report should stop treating it as a planned win.
- **Submission is where the whole win is.** 15.43 us of Python launcher became about 6.0 us of
  direct submission, and that saving is the same for every operator and every shape.


### Generalisation: what one artifact covers

Two operators were taken through the same registry, and they behave differently for a reason
worth stating in the design rather than discovering in the field.

**`mul`, a pointwise binary operator - one artifact serves every shape measured.** One
registered artifact, six input shapes including odd sizes and a five-dimensional case:

| shape | host program | Python route | ratio | correct |
| --- | ---: | ---: | ---: | --- |
| (1024, 1024) | 8.41 us | 57.42 us | 6.8x | yes |
| (4096, 4096) | 8.20 us | 56.84 us | 6.9x | yes |
| (2, 3, 5, 7, 11) | 8.44 us | 56.73 us | 6.7x | yes |
| (999,) | 8.35 us | 57.14 us | 6.8x | yes |
| (64, 1000) | 8.58 us | 57.57 us | 6.7x | yes |
| (7, 13, 101) | 8.42 us | 57.83 us | 6.9x | yes |

The ratio is flat at about 6.8x because both sides are fixed overheads: the host program's cost
is a lookup, an allocation and a submission, and the Python route's cost is the interpreter, the
GIL and the launcher - neither scales with the element count in this range. That is the useful
shape of the result: the win is per call, not per byte.

The cases the guard refuses are refused correctly and, more importantly, silently fall back and
stay correct:

| case | took the host program | result |
| --- | --- | --- |
| broadcasting operands | no | correct |
| `float16` operands | no | correct |
| non-contiguous operands | no | correct |

**`softmax`, a reduction - one artifact serves only the shapes its constexprs were compiled
for.** A single artifact, registered for the shape it was warmed on, gives 8.90 us against
29.42 us on the Python route (3.3x) and matches the reference. Reusing that same artifact for
other widths does not degrade gracefully; it corrupts the device and the next CUDA call reports
a sticky failure.

The mechanism is in the kernel's own signature. `softmax_kernel_inner` takes `TILE_N` and
`ONE_TILE_PER_CTA` as constexprs chosen from `N` at launch time, so an artifact compiled for
`N = 4096` is not a valid launch for `N = 1024`. `mul_kernel` has only `BLOCK_SIZE`, which the
grid can be computed against at any size, so it generalises.

This is the design's gate 4 in concrete form, and it settles how the host compiler must treat
constexprs: **a constexpr derived from a tensor's metadata is part of the artifact's identity,
not a launch-time value.** The host program has to either hold the variants it may need and
select between them on the same predicates capture recorded, or decline and fall back. Holding
one artifact and assuming it covers a shape class is exactly the failure this measured.

For `mul` the selection is trivial because its only constexpr is fixed; for `softmax` it is not,
and that difference - not the operator's arithmetic - is what decides whether one artifact is
enough.


### Multi-artifact selection, implemented and measured

The previous section states the rule; this one implements it. A registered operator now holds a
list of artifacts, each with the metadata range it is valid for, and the call selects by the
same value the constexpr was derived from:

```cpp
struct HostVariant {
  CUfunction fn;
  int num_warps;
  int64_t shared;
  int64_t min_n;   // the artifact claims [min_n, max_n] of the constexpr's driving value
  int64_t max_n;
};
```

Selection happens *before* the allocation, so a call no artifact covers costs a lookup and
falls back; it is counted as `no_variant` rather than silently declining. Registering a bare
program still works and installs one artifact claiming every size - correct for `mul`, whose
only constexpr is a fixed `BLOCK_SIZE`.

For `softmax` two artifacts were compiled, one per reduction width, and each registered for
exactly the width it was built for. They are genuinely different specialisations - the 4096
artifact has 16 warps and 64 bytes of shared memory, the 1024 artifact 4 warps and 16 bytes -
which is the point: one artifact cannot serve the other.

| shape | host program | Python route | ratio | correct |
| --- | ---: | ---: | ---: | --- |
| (4, 4096) | 8.86 us | 30.71 us | 3.5x | yes |
| (128, 4096) | 9.03 us | 30.37 us | 3.4x | yes |
| (4, 1024) | 9.00 us | 29.34 us | 3.3x | yes |
| (128, 1024) | 9.00 us | 30.12 us | 3.4x | yes |
| (2, 128, 4096) | 9.00 us | 30.39 us | 3.4x | yes |

Every call took its variant and every result matched. The cost is flat at about 9.0 us across
widths and row counts, which is what a fixed-overhead win looks like.

A width no artifact covers behaves the way the design requires:

| case | took the host program | counter | result |
| --- | --- | --- | --- |
| width 512, no artifact | no | `no_variant` + 1 | correct |

So the two ends are both pinned down: a covered shape runs native, an uncovered one declines
through a counted path and the Python route answers correctly. What remains for real coverage
is the part a compiler owns rather than a prototype - deciding, before compiling, which
constexpr classes a shape population will need, and compiling the variants up front instead of
on the first call that wants one.

### What this says about the design

The exit branch this lands in is the first one: capture is complete and a native entry point
works, so the work should converge on the adapter layer rather than on a self-built host IR.
Two facts argue for that rather than against.

The capture side is genuinely hard - `pointwise_dynamic`, `LibEntry` and the autotuner are
exactly where tracing was expected to break - and it was solved with a shim plus routing
changes, not with a new IR. And the native entry needed nothing from an IR: it takes the
compiled kernel's handle, its launch geometry and the parameter values, all of which are
reachable from the compiled artifact.

The most expensive defect in the phase is worth recording because it generalises. The host
program reproduced the kernel's signature from the source-level `arg_names`, the `signature`
string and the metadata, and all three said four parameters. The compiled entry wanted six:
Triton appends a global-scratch and a profile-scratch pointer after the user arguments. The
disassembly of the compiled artifact was the only authority, and reading it was the last thing
tried. **Any component that reproduces a compiled entry's ABI should read the entry's own
parameter list first.** For the host compiler this is a requirement, not a lesson: the ABI it
emits against is the one in the artifact.

## Allocation

Three layers, in the order they are worth doing. The first is part of the gate 6
measurement; the second is independent of this project; the third is not in scope now.

1. **Direct allocator in the native path.** Allocate through `c10::GetAllocator` and
   construct the tensor directly, skipping the dispatcher hop while still allocating and
   freeing every call through the real allocator. Not a pool. Target 1.0-1.5 us.
2. **Fix the plugin's own allocation path.** If allocation on `flagos` is materially slower
   than on `cuda`, the cause is in the plugin's allocator path, and fixing it benefits every
   operator independently of this compiler.
3. **Bounded process-wide reuse.** Only if measurements justify it later. #822 permits
   caching with an explicit per-device byte budget and safe stream/lifetime management; it
   forbids a per-plan pool.

Layer 1 must be indistinguishable from `at::empty` in the properties callers observe:
dispatch key set including the autograd branch, version counter, resizable flag, memory
format, storage offset, alignment and allocator stream association. These fail silently
rather than loudly, so the equivalence tests are an entry condition, not a follow-up.

## FAQ: how a host program differs from the replay plan cache

Both are caches, so the distinction is not cache versus no cache. A replay plan memoises
**values**; a host program memoises **code**. #822 states the same thing as its reason for
existing: "recording one concrete execution captures values, not necessarily the expressions
and branch conditions that produced those values".

The captured `softmax` graph makes the difference concrete. What capture produced for
`(2, 128, 4096)` was:

```text
empty_like(self, dtype=float32)
triton_kernel_wrapper_mutation(grid=(256, 1, 1), kernel_idx=1, kwargs={output_ptr, input_ptr})
```

Read as a replay plan that is: *the grid is `(256, 1, 1)`, the output is pool slot 3, the
input is the tensor at this address*. It holds for that shape and those addresses only. A
different shape needs a different plan or a refusal, and because the output came from a
persistent pool, its lifetime is managed by a reference count rather than by ATen.

Read as a host program it is: *`rows = prod(shape[:-1])`, `grid = (rows, 1, 1)`, allocate an
output of the input's shape with dtype float32, then launch the kernel with these parameter
mappings*. The dimensions are inputs to the program, not constants in it, so one artifact
serves every shape in its specialisation class, and allocation goes through the real
allocator.

| dimension | replay plan (#820) | host program (#822) |
| --- | --- | --- |
| what is stored | values: concrete dimensions, grid, pointers, output slot | code: metadata expressions, guards, allocation, launch mapping |
| cache key | the concrete call, exact sizes included | the specialisation class (dtype, rank, layout, mode) |
| a new shape | a new plan, so entries grow with shape churn | the same artifact; only the grid and scalars are recomputed |
| allocation | persistent output pool, which is where its speed came from | the real allocator on every call |
| output semantics | pooled buffers, kept distinct by reference counting | ordinary ATen allocation and lifetime |
| branches | only the path that was observed | guards plus multiple launch sites, chosen per call |
| analysable | no: it is data | yes: it is a program with structure |
| failure mode | degrades or refuses on an unobserved shape | handles it, until the specialisation class changes |

The relationship is a continuum rather than two rival designs. Advancing a plan into a host
program takes three substitutions:

1. replace baked dimensions, grid and scalars with expressions over metadata - parameterise;
2. add guards that choose the path - specialise;
3. replace the pooled output with a real allocation - correct semantics.

#822 asks for exactly those three. #820 stops before the first, which is why it is fast and
why it only covers what it recorded.

This also explains why the 7 us cannot be inherited. Two different savings were being summed:
skipping Python, which a host program also does, and skipping allocation, which #820 did by
pooling. Pooling rests on a plan being tied to a fixed output shape, so the moment that tie
is removed the saving goes with it. That is about 4 us of the 7. A host program has to
recover it through the allocation fast path instead, which is a separate piece of work.

## Open questions

- Runtime host compilation: this design assumes a host C++ compiler is available at run
  time, as `torch.compile` already requires. Is that acceptable for the supported
  deployment, or must artifacts be produced at wheel build time?
- Does the CPU-torch + PrivateUse1 configuration permit the Dynamo/Inductor route at all,
  and does generated code stay free of `c10::cuda`/`at::cuda` symbols? #820 had to take its
  stream from the plugin's registry to satisfy that contract; generated code must keep the
  same property.
- Should the plugin-allocator work (allocation layer 2) proceed as its own change
  regardless of this project's timeline?

## Risks, in the order they can sink this

1. **Capture yield is too low.** FlagGems' `pointwise_dynamic`, `LibEntry` and autotuner
   wrappers are exactly where tracing is expected to break. Gate 2 and gate 5 measure this
   on real callables instead of assuming.
2. **No native wrapper can be reused**, forcing a self-built IR and backend before any
   performance evidence exists. Gate 3 and the exit criteria exist to make that decision on
   evidence.
3. **Allocation cannot be brought near the vendor's**, leaving the ceiling at 1.45x. That
   would not fail the acceptance criteria - parity is a separate goal - but it would limit
   the project's value, which is why the prototype is measured in gate 6.
4. **Generated code violates the CPU-torch contract** by naming `c10::cuda`, which fails at
   load time rather than at run time.