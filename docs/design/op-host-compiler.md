# Compiling FlagGems' Python host logic into native Torch-FL operators

Status: design proposal for #822. Nothing here is implemented; Phase 0 exists to
falsify the assumptions this document is built on.

## Problem

An eager ATen call on `flagos` reaches FlagGems through a generic caller that
acquires the GIL, builds Python arguments and invokes the resolved Python
callable. Everything the wrapper does around the kernel — argument
normalisation, metadata arithmetic, dispatch and specialisation, allocation —
runs in Python on every call.

#820 removed part of that cost by recording one concrete launch and replaying it
from the caller. Measured there, on H100: `mul` 512x512 went from 82.55 us
(wrapper) to 6.81 us (replay), against 7.27 us for the vendor operator. But that
result rests on two things this project may not keep: a persistent per-plan pool
of output tensors, and a plan that is a snapshot of one observed call with
concrete dimensions. Both are excluded here — the target is a *parameterised
host program* that recomputes its shapes, guards its specialisation and
allocates normally on every call.

## Goals

- Ordinary eager calls keep working; no model rewrite, no user `torch.compile`.
- After warmup, a supported call completes without entering Python: guard,
  specialisation selection, shape/scalar arithmetic, allocation and one or more
  kernel launches are all native.
- Device kernels stay FlagGems' Triton kernels. What is compiled is the host
  side.
- Coverage grows through shared lowering rules, not per-operator C++.

## Non-goals

- Handwritten C++ per operator, or upstream FlagGems changes to make tracing
  easier.
- Per-plan persistent output pools; retaining user inputs, input pointers, prior
  outputs or a fixed stream to make replay cheap.
- A promise of zero compilation cost, zero extra memory, or optimal performance
  on every input.
- Autograd, RNG, data-dependent output sizes or mutation in the first slice.
  These must be declined explicitly, not silently treated as constant.

## Performance model

The budget for one warm call, from #820's measurements on H100 (all same host,
same stack): the kernel launch itself was 4.33 us inside C++, and a plugin
dispatch entry adds about 1 us.

Allocation is the open question, and it is large enough to decide the project.
`at::empty` on the `flagos` device measured about **4.0 us**, and the vendor's
7.27 us for `mul` includes an allocation of its own. Comparing our unoptimised
allocation against the vendor's total is not a like-for-like comparison:

| composition | cost | ratio to vendor (7.27 us) |
| --- | ---: | ---: |
| launch only (#820) | 4.33 us | 0.60x |
| launch + dispatch entry | ~5.3 us | 0.73x |
| launch + entry + `at::empty` (4.0 us) | ~10.6 us | **1.45x** |
| launch + entry + allocation at 1.0-1.5 us | 6.8-7.8 us | **0.94-1.07x** |

So parity is reachable inside this project's constraints, but only if allocation
stops costing 4 us. That makes allocation a first-class workstream rather than a
later optimisation, and it is the reason Phase 0 measures it before anything
else.

## Architecture

```text
ATen eager call on flagos
  -> native entry (per operator, registered)
     -> guard: dtype/rank/layout/mode/device of this call
        -> hit:  host program
                   compute shapes and scalars from this call's metadata
                   allocate outputs through the device allocator
                   launch 1..N kernels
                   return ATen-compatible results
        -> miss: capture, compile, register, then run as above
     -> unsupported: the existing Python route, at the dispatch boundary
```

### Frontend

Evaluate TorchDynamo + FX on real FlagGems callables first. FlagGems operators
are commonly wrapped in decorator machinery (`pointwise_dynamic`, autotuners,
`libentry`), which is where tracing is expected to break; the support table from
Phase 0 records where and why. A `TorchDispatchMode` recorder is the fallback
frontend, because it collects calls at the boundary this project cares about
without depending on Dynamo internals. Compiling the Python source directly is
rejected: it degrades into per-operator pattern matching, which is the thing
this project is meant to stop doing.

### Representing the host program

A small typed IR, as #822 sketches: read tensor metadata, checked integer
arithmetic, comparisons and guards, allocation, layout/view construction,
tuple returns, and kernel invocations with explicit argument mapping and launch
geometry. Shapes and scalars are *expressions over this call's metadata*, never
constants from the first call. Control flow is supported only as guarded
branches: a mismatched guard compiles or selects another variant rather than
silently taking the observed path.

The IR is then emitted as C++, compiled with the host compiler, and loaded. An
interpreter over the IR is kept for debugging only — the issue is explicit that
performance acceptance requires a native host path.

### Kernel ABI adapter

Use the compiler's own post-specialisation description — the runtime parameter
list, types and widths, constants and alignment, grid/block, shared memory,
scratch and dependency fingerprints — as a versioned adapter (`v1` for the
pinned Triton/FlagTree). #820's PTX parameter-count inference is demoted to a
diagnostic and a cross-check; it is not an ABI contract.

### Allocation

Three layers, in the order they are worth doing:

1. **Direct allocator in the host program (Phase 1).** `c10::GetAllocator` plus
   tensor construction, skipping the dispatcher hop while still allocating and
   freeing through the real allocator every call. Not a pool. Target 1.0-1.5 us.
2. **Fix the plugin's own allocation path (Phase 0 measurement, may land first).**
   If allocation on `flagos` is materially slower than on `cuda`, the reason is
   in the plugin's allocator path, and fixing it benefits every operator
   independently of this compiler.
3. **Bounded process-wide reuse (later, only if measurements justify it).** #822
   permits caching with an explicit per-device byte budget and safe
   stream/lifetime management; it forbids a per-plan pool. Layer 1 and 2 come
   first.

Layer 1 has to be indistinguishable from `at::empty` in the properties callers
observe: dispatch key set including the autograd branch, version counter,
resizable flag, memory format, storage offset, alignment, and allocator stream
association. The equivalence tests are an entry condition for Phase 1, not a
follow-up: a tensor that differs in any of these fails silently rather than
loudly.

### Cache and lifetime

Specialisation classes, not concrete sizes, key the artifacts: operator,
dtype combination, rank, contiguity and alignment class, execution mode. Sizes
stay inside the host program as arithmetic over metadata, so many shapes share
one artifact.

Bounded by per-operator and global entry counts plus a byte budget, with
admission policy, eviction, compile-rate limiting and deduplication of
concurrent compilation of the same variant. Eviction is separate from
destruction: in-flight launches and live results retain their resources, and
retired-but-live resources stay accounted. Counters for captures, compiles,
native hits, misses with reasons, variant count, bytes and evictions exist from
the first slice.

### Fallback

At the dispatch boundary, before any allocation, launch or mutation. No path may
launch part of an operator, then fall back and re-run it: that would replay
side effects. Unsupported constructs are declined with a reason that is
observable.

## Phase plan

### Phase 0 — falsification and baselines

1. **Allocation.** `at::empty` on `flagos:0` versus `cuda:0`, decomposed into
   dispatcher, allocator and tensor construction. Prototype of the direct
   allocator path, to see whether 1.0-1.5 us is real. Decide whether the plugin
   allocator itself is fixable work worth doing first.
2. **Frontend spike.** Whether Inductor's `cpp_wrapper` / AOTInductor can emit a
   host artifact for a one-operator graph on `flagos`. If it can, most of this
   document is replaced by an adapter, and that is the finding that matters most.
   Also check whether its generated code introduces `c10::cuda`/`at::cuda`
   symbols, which the CPU-torch build contract forbids.
3. **Capture support table.** Four categories on real FlagGems callables:
   pointwise `mul`, inner and non-inner `softmax`, a reduction verified to use
   several launches, and a multi-output operator such as `max.dim`. Record the
   frontend's graph breaks and reasons per operator.
4. **Kernel ABI.** What the pinned Triton/FlagTree exposes after
   specialisation, and whether it is enough to generate a launch without
   inference.
5. **Baselines, four arms, same process where possible:** wrapper, native launch
   plus normal allocation, native launch plus layer-1 allocation, and the vendor
   operator in a separate process as the end-user reference.

### Phase 1 — one vertical slice

One operator end to end: real Python implementation, capture, host artifact,
registration, ordinary ATen call. Layer-1 allocation with its equivalence
tests. A size sweep across one specialisation class demonstrating artifact
reuse rather than one artifact per dimension. Diagnostics, bounded cache and
disable/invalidate from the start.

The operator is chosen from the Phase 0 support table — the one whose host logic
is cleanest, not the most familiar. `mul` is a poor candidate precisely because
its host logic lives inside the `pointwise_dynamic` factory.

### Phase 2 — shared semantic coverage

Shape and layout arithmetic, keyword and default handling, views and tuples,
multiple launches and temporaries through generic rules. The four categories
from Phase 0 covered without handwritten operator bodies. Explicit statement of
what remains unsupported, with fallback verified.

### Phase 3 — hardening

Concurrency, multiple devices, non-default streams, cross-stream consumers,
live-output stability, eviction with in-flight work, failure cleanup, module
lifetime. The supported NVIDIA CPU-torch configuration and a safe
unsupported-platform build. Capability boundary for a second backend, added only
with execution evidence.

## Risks, in the order they can sink this

1. **Inductor's `cpp_wrapper` works.** This is upside-down as a risk: if it works
   on `flagos`, building a compiler here would be duplicated work. Phase 0 item 2
   exists to find this out before any IR is written.
2. **Tracing yield is low** because of FlagGems' decorator machinery, leaving too
   few operators for the approach to be general. The support table answers this
   with numbers.
3. **Allocation cannot be brought near the vendor's**, leaving the project's
   ceiling at 1.45x. Phase 0 item 1 answers this, and it is why it is first.
4. **Generated host code violates the CPU-torch contract** by naming `c10::cuda`.
   #820 hit exactly this and solved it by taking the stream from the plugin's own
   registry; the codegen has to keep that property.

## Open questions

- Performance target wording: is "no Python on the warm path" the acceptance
  criterion, with parity as a separate goal that depends on allocation work?
- Runtime host compilation: this design assumes a host C++ compiler is available
  at run time, as `torch.compile` already does. Is that acceptable for the
  supported deployment, or must the artifact be produced at wheel build time?
- Should the plugin-allocator work (layer 2) proceed as its own change
  regardless of this project's timeline?