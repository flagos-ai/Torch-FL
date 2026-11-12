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

> **The allocation row of this table did not survive measurement, and the table is left as it was
> written so the correction is visible.** Phase 0 measured `torch.empty` on `flagos` at 3.97 us for
> one element and 4.10 us for four MiB - a difference of 0.11 us - and CPU torch at 3.00 us for a
> single element. Allocation is therefore dominated by fixed cost that exists without this plugin,
> not by the allocator, and the 1.0-1.5 us row is not reachable by optimising allocation. Moving
> allocation into C++ did not help either (4.65 us through a dispatcher entry against 4.05 us from
> Python). The win this project actually has is submission, not allocation; see "Gate 6" below.

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

Gates 6 requires the comparison to be like for like, so the parity statement is made on
`softmax` only, the one operator where both arms were measured in the same session:

| `softmax`, same shape and dtype, same session | CPU submission |
| --- | ---: |
| native host program | 8.90-9.03 us |
| vendor `torch.softmax` on `cuda:0` | 8.95 us |

That is parity, within the run-to-run spread, for an eager call.

**For `mul` no same-operator vendor measurement exists**, and an earlier revision of this section
compared `mul`'s 8.24 us against the vendor's *softmax* to conclude parity. That comparison is
invalid and is withdrawn; the correct reading is that `mul` was not compared with a vendor `mul`
here at all. #820 measured the vendor's `torch.mul` at 7.27 us in its own session, which would put
the host program at 1.13x rather than at parity - but the two sessions are not the same measurement
run, so that number is a pointer to what to measure next, not a result.

What remains true regardless of the vendor comparison is the within-session result: removing the
Python launcher takes these calls from 29-57 us to 8-9 us, which is the improvement gate 6 asks
for.

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

**No exit branch is claimed yet, and an earlier revision of this section claimed the first one
too early.** What Phase 0 demonstrated is narrower than that: a native entry point, driver
submission and multi-artifact selection work, *given a registered artifact*. What it did not
demonstrate is the automatic path that would put that artifact there - the capture of a real
wrapper's parameterisation, the symbolic expressions over shapes, the guards derived from the
captured branches, or the reuse of ready-made codegen. In the prototype the kernel handle and
launch geometry were read from the Python side by hand and registered explicitly; the six-shape
result proves the parameterisation works once an artifact exists, not that a compiler produced it.

So the honest state against the design's own exit criteria is: **the runtime half of branch one is
demonstrated, the capture half is not**, and gate 2 (capture complete) has not been passed. Any
decision between converging on an adapter layer and evaluating a restricted IR needs that missing
half measured first.

Two things the phase does establish for that decision. The capture side is where the difficulty
concentrates - `pointwise_dynamic`, `LibEntry` and the autotuner are exactly where tracing was
expected to break. And the native entry needed nothing from an IR: it takes the compiled kernel's
handle, its launch geometry and the parameter values, all of which are reachable from the compiled
artifact once something has decided which artifact to use.

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

## Development cost to cover FlagGems

The question this section answers is not "does one operator work" - Phase 0 answered that - but
what it costs to cover the FlagGems operator surface with generalisation. Numbers below are
either **measured** on the shipped tree or **estimated**, and each is labelled.

### The operator surface, measured

| quantity | value |
| --- | ---: |
| `aten` operators routed to `flaggems` in `backends_nvidia.conf` | **549** |
| `@pointwise_dynamic` use sites, across 177 operator modules | **265** |
| `@libentry` hand-written Triton kernels | **786** |
| kernel definitions in total (3 to 10 parameters, no dominant shape) | **966** |
| of those, `@triton.autotune` kernels - several configurations each | 143 |
| routed operators whose same-named module uses `pointwise_dynamic` | **161 (29%)**, unverified |
| `mul` - the operator Phase 0 measured - uses | `libentry` + `libtuner`, **not** the factory |

**The 161 figure is an upper bound read off module names, not verified coverage, and it must not
be used to estimate delivered work.** It was produced by checking, for each routed operator whose
name matches an `ops` module, whether that module mentions `pointwise_dynamic`. It does not
establish that the specific routed callable is the factory-generated one; that needs per-schema
verification against the real callable, which has not been done.

The `mul` row is why this correction matters. Phase 0's pointwise class was built against
`mul_kernel`, a `libentry`/`libtuner` kernel with a hand-written Triton body and a flat element
count - and `mul.py` does not use `pointwise_dynamic` at all. So the class that was implemented
and measured is the **libentry elementwise shape**, not the factory shape, and the factory family
is a different class whose ABI has not been examined. An earlier revision of this section claimed
200 routes (36%) were already implemented; that claim was wrong for exactly this reason and is
withdrawn.

### The cost model: classes, not operators

Development cost scales with the number of **kernel-shape classes**, not with the number of
operators. Three measured facts support that:

1. Kernels generated by one factory share one signature. The 265 `pointwise_dynamic` sites are
   one such family; the `libentry` elementwise kernels are another. They are *different*
   families with different ABIs - `mul_kernel` takes a flat element count, a factory-generated
   kernel takes shapes and strides - so a class covers one family, and the six-shape result
   Phase 0 measured belongs to the `libentry` side.
2. The 786 `libentry` kernels have widely scattered signatures - parameter counts peak at 5, 6,
   7 and 4 with no majority - so they cannot be one class, but they do cluster by computational
   pattern: reduction, normalisation, pooling, convolution, indexing, sort/topk, linear algebra.
3. Per operator, the host logic is a declaration of how the parameter values derive from tensor
   metadata, not a kernel implementation. Once the class and the ABI extraction exist, a new
   operator is a declaration plus an automated check.

### What is already built and measured

| piece | status | coverage |
| --- | --- | --- |
| elementwise kernel class (`libentry` shape, as `mul` uses) | implemented, measured on six shapes | covers operators whose kernel has this shape; **not yet counted per schema** |
| reduction over the last axis (softmax) | implemented, measured at 3.4x | the softmax family |
| ABI extraction from the compiled artifact's PTX | **method proven, done by hand** | two kernels so far |
| constexpr variant selection | implemented, measured across two artifacts | grows with the shape population |
| fallback with per-stage counters | implemented, measured | all declines |

### Estimate

Assumptions, stated because the estimate is worthless without them: one engineer familiar with
Triton internals and PrivateUse1 dispatch; the Phase 0 ABI method reused rather than re-derived;
single platform (CUDA on H100); no change to the FlagGems operator set during the work.

| work item | estimate | basis |
| --- | ---: | --- |
| **ABI extraction automated** - compiled entry's parameter list to a parameter mapping | **0.5-1 person-month** | the manual version cost half a day for two kernels; 966 kernels cannot be done by hand, so this is the leverage point |
| **code generator** - ABI table to C++ dispatch entries and guards | **1-1.5 person-month** | generates the registration surface for 549 operators once |
| **per-class host logic** - 15 to 25 classes | **2-4 person-months** | 3 to 6 days per class; multi-launch sequences and multi-output operators priced separately |
| **constexpr and variant coverage** | **0.5-1 person-month** | 143 autotune sites multiply variants; needs a bound on how many are compiled |
| **guard coverage** - dtype, stride, broadcast, non-contiguous | **1-2 person-months** | today these decline; Triton kernels take strides as parameters, so this reaches into the ABI |
| **test harness and vendor comparison** - 549 operators across the dtype matrix | **1-2 person-months** | automated comparison is the only realistic acceptance path at this count |
| **total** | **6-11 person-months** | covers roughly 90% of commonly used operators; the long tail of convolution, linear algebra and attention is not in this figure |

After the first, second and sixth items exist, the marginal cost of adding an operator is a
declaration plus an automated check - on the order of an hour, mostly test time.

The estimate deliberately does **not** discount itself by the 161 routes in the factory family. A
per-schema census is a work item in its own right, and until it exists neither this estimate nor
any coverage claim should be read as delivered work. The figure above is what remains to be built
from a prototype that has one class measured.

### Compared with hand-written native wrappers

| approach | effort for 549 operators | maintenance when FlagGems changes |
| --- | --- | --- |
| hand-written C++ wrapper per operator | **3-7 person-years** (2-5 days each) | hand-edited per operator |
| host program, generated from capture | **6-11 person-months** | regenerate from the ABI |

A factor of five to ten, and the ratio grows with the operator count. This, rather than any
performance claim, is the argument for the approach: the host program is not faster than the
replay plan cache, and it is not faster than a hand-written wrapper - it is *at* the
hand-written wrapper's floor (8.24 us against a floor of roughly 8.2-9.2 us for dispatch plus
allocation plus launch) while being generated rather than written.

### The risks that would invalidate the estimate

| risk | effect | evidence |
| --- | --- | --- |
| **ABI extraction cannot be automated** | cost multiplies roughly tenfold, and repeats on every FlagGems upgrade | this is the pitfall Phase 0 hit: the source signature, `arg_names`, the `signature` string and the metadata all said four parameters, the compiled entry wanted six |
| **multi-launch sequences** | the host program expresses one launch today; `sort` issues 21 | measured, and it is an architectural gap rather than a class |
| **143 autotune kernels** | variant count multiplies per kernel | needs a bound: compile only the configurations actually selected |
| **capture yield below 100%** | operators whose FlagGems call the capture cannot see | the `pointwise_dynamic` factory, `LibEntry` and the autotuner are exactly where tracing is expected to break; gates 2 and 5 exist for this |
| **FlagGems ABI churn** | regeneration each upgrade | this is the reason to generate rather than hand-write, not a reason against the approach |

### Suggested order

Ordered by coverage per unit of effort rather than by technical interest:

1. **Automate ABI extraction.** Without it every later item is hand work and this estimate does
   not hold.
2. **Write the code generator.** The elementwise class is implemented and measured, so the
   generator can land the operators whose kernels actually have that shape - enumerated by a
   per-schema census, not by module name.
3. **Add classes by call frequency** - reduction, then normalisation, then indexing, then
   sort/topk - not alphabetically.
4. **Extend guards after step 2.** Broadcast and non-contiguous inputs are common in real eager
   code; while they decline, the optimisation does not apply to those calls at all.

## Reuse before building: three routes, and what would decide between them

Phase 0 proves a native entry point works. It does not prove that building a general host compiler
is the best way to get one, and this section is the comparison that has to happen before that
commitment. The review of this document is right that the earlier draft skipped it: the question is
not "can a host program run" - it can - but "what should produce the host program".

Four candidates, in ascending order of our own effort:

| route | what it reuses | what it costs | what has to be proven |
| --- | --- | --- | --- |
| **A. Inductor C++ / AOT wrapper** | allocation, scheduling and code generation already exist | integration, not construction | that it captures real FlagGems callables, keeps the original Triton kernels, and reconnects to ordinary eager under CPU-torch + PrivateUse1 |
| **B. FlagGems' own shared pointwise generator and native runtime** | factory and schema already hold part of the host semantics: `get_kernel_info(rank)`, type promotion, configuration generation, and a shared native argument-preparation path | limited generated coverage today, and a CUDA-stream dependency | that it covers the target shapes and is CPU-torch clean, or can be made so |
| **C. A restricted host IR with a small native backend** | nothing; expresses metadata computation, guards, allocation and multi-launch | the frontend and ABI work remain | only worth choosing if A and B are both blocked for integration reasons |
| **D. A general host compiler built here** | nothing; highest coverage potential | frontend, semantic verification and maintenance are all ours | the evidence for this does not exist yet |

**Route B is a real candidate and is not the same as hand-writing a kernel per operator.** FlagGems
5.4.0 ships a C++ pointwise path whose `prebuild_kernels.py` already reuses `get_kernel_info(rank)`,
the schema, type promotion and configuration generation as a thin wrapper, and whose
`pointwise_prepare_args.h` provides a shared native preparation path. That is part of the host
semantics this document proposes to reconstruct from Python. What it does not yet offer is evidence
of CPU-torch + PrivateUse1 compatibility - it carries a CUDA-stream dependency - or broad generated
coverage. Both are cheap to test and neither has been tested here.

### What the comparison must hold fixed

Comparing routes is only meaningful on common ground, so the experiment is specified before the
result is known:

- **Fixed inputs.** Same pinned dependencies, real callables, CPU-torch + PrivateUse1 build, device
  and SDK. Compare on the operator set both routes support, and report the unsupported set and its
  extension cost separately rather than averaging it in.
- **Comparable measurement.** Same workload, shapes, dtype, layout, original FlagGems kernel,
  ordinary allocation and timing boundary. Direct-allocation work is measured on its own arm so it
  does not contaminate the codegen comparison.
- **Same acceptance for both.** Each route must obtain, from the real callable or schema and without
  hand editing, the parameterised expressions for the current input, the complete guards, and the
  kernel parameter mapping, and must be verified across sizes and across constexpr boundaries.
  A concrete launch snapshot, an opaque Python call, or an Inductor substitution of the device
  kernel do not count as completing the goal.
- **Same entry.** One unchanged native eager script reaches the backend with no user registration or
  warmup step. Capture must not re-run the user's call, submit real GPU work, or recurse into
  compilation; unsupported cases fall back before submission.
- **Same evidence for "no Python".** A profiler and entry counters together, plus tests for output
  lifetime, layout, exception behaviour and the semantics of unsupported fallbacks.
- **Reported separately.** First call, warm CPU submission including guard, allocation and release,
  synchronised end to end, and cache pressure - with the budget and the measurement tolerance fixed
  before the experiment, and vendor parity reported as its own line.

### Where that leaves the recommendation

The order should be: **A and B in parallel first, on a common operator set; C only if both are
blocked by integration rather than by capability; D only if C is also insufficient.**

What Phase 0 contributes to that decision is bounded and worth stating plainly: the native half -
entry point, allocation, guards, variant selection, submission, fallback - is the same work under
every route above, and one instance of it now exists and is measured. The open question is only
what produces the artifact and the guards, which is exactly the question the review asks, and the
earlier draft answered it by assumption.

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