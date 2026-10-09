# Torch-FL CUDA-facing compatibility

Status: **draft, for discussion on [#486](https://github.com/flagos-ai/Torch-FL/issues/486)**.
This page defines the problem, the platform classes, the compatibility contract and
the acceptance gates. It does not yet commit to the device-identity decision; that
is called out as an explicit architecture choice below.

## Context

Torch-FL's logical device is `flagos` (PrivateUse1). CUDA-oriented application code
and frameworks make control-plane decisions on **CUDA identity** long before any ATen
operator dispatches: accelerator discovery, `device.type == "cuda"`, the AMP device,
stream/event construction, the default distributed backend, and DDP hook
registration. None of those consult the dispatcher, so remapping device arguments is
not enough to make them agree with the logical `flagos` device.

What exists today is an **input alias**, not a compatibility contract:

- `torch_fl.__init__._alias_cuda_to_flagos` conditionally remaps `cuda` device
  arguments through a `TorchFunctionMode`, wraps the `torch.device` constructor, and
  redirects `Tensor.cuda()` / `Module.cuda()`.
- It stands down when the original PyTorch probe reports real CUDA, and can be
  disabled with `FLAGOS_ALIAS_CUDA=0`; `torch_fl._cuda_alias_active` reports whether
  it is on.
- Per-vendor `torch.cuda` facades (CUDA-ABI vendors) supply a `torch.cuda` surface.

The output of an alias is still a logical `flagos`/PrivateUse1 tensor. Frameworks
that branch on the accelerator class, `device.type == "cuda"`, CUDA-only AMP
construction, or CUDA stream/collective APIs therefore do not follow it. #486 lists
eleven concrete examples; PyTorch Lightning is the motivating consumer, not the
scope.

## Goals

- **One documented compatibility initialization step** after which existing
  CUDA-facing placement and configuration (`device="cuda"`, `.cuda()`,
  `accelerator="gpu"`) has consistent, supported behavior on the selected FlagOS
  backend, without editing business-model or third-party framework source.
- A **per-backend capability contract**: for each boundary below, state supported /
  partially supported / unsupported, and make unsupported fail clearly rather than
  silently selecting an unintended CPU path.
- **Explicit isolation** between native CUDA and native `flagos` so the two can
  coexist and each can be validated.

## Non-goals

- Full CUDA **tensor identity** (a `flagos` tensor reporting `.is_cuda` /
  `device.type == "cuda"`) — see the architecture decision below; not adopted here.
- **Arbitrary CUDA binary-extension ABI** compatibility across vendors.
- **Cross-platform binary compatibility**; each platform class has its own gate.
- Replacing the dispatcher/routing work: this is the framework-facing contract, not
  the per-op route decision (see #444).

## Platform classes

Compatibility is asserted per class, never globally. The facade a class can honestly
provide differs:

| Class | `FLAGOS_ACCELERATOR` | Runtime reality | CUDA facade today |
|---|---|---|---|
| Native NVIDIA CUDA | `cuda` | Real CUDA runtime and kernels | none needed (it *is* CUDA) |
| Vendor CUDA-ABI | `dcu`, `metax`, `ppu` | Vendor forked libtorch; flagos storage is a zero-copy CUDA alias | `torch.cuda` shim |
| Native non-CUDA | `ascend`, `gcu`, `musa`, `tsingmicro`, `bpu` | No CUDA runtime; vendor-native ops / CPU fallback | none (must not pretend CUDA) |

A CUDA label or a zero-copy physical view does **not** make guards, allocator
ownership, gradients, stream ordering or extension ABI automatically compatible; the
vendor CUDA-ABI class still needs a measured stream/collective gate.

## Two modes

1. **Native `flagos` mode (default).** `import torch_fl`, logical `flagos` device.
   Frameworks reach it through native discovery or, where offered, a first-party
   integration. No CUDA pretence.
2. **CUDA-facing compatibility mode.** One documented initialization step (reusing
   the existing alias/facade, gated like `FLAGOS_STARTUP_PROFILE`/`FLAGOS_ALIAS_CUDA`)
   under which the contract below holds for the selected backend.

The two modes must be explicitly selectable and testable; a process must not
silently be in both.

## The compatibility contract

For each boundary, the CUDA-facing mode states a guarantee, the mechanism that
delivers it, and the failure mode when it is unsupported.

| Boundary | Guarantee in CUDA-facing mode | Mechanism |
|---|---|---|
| `torch.device("cuda[:i]")`, `device=` kwargs | Resolves to the selected backend's device | existing `_remap` / constructor wrap |
| `Tensor.cuda()`, `Module.cuda()` | Places on the selected backend | existing redirect |
| Accelerator discovery (Lightning `accelerator="gpu"/"auto"`) | Selects a working accelerator for the backend, or fails naming the supported values | first-party registration or a documented adapter |
| `device.type` / `is_cuda` on a tensor | **Unchanged** (logical `flagos`), documented | not faked |
| AMP (`autocast`, `GradScaler`) | Matches the backend's precision path and is validated per platform | `AutocastPrivateUse1` + scaler semantics |
| Streams / events | Preserves real ordering for multi-stream use | vendor stream/event implementation, not a no-op shim |
| Distributed default backend | Derived from the device actually in use | `torch._C._get_privateuse1_backend_name()` |
| DDP communication hooks | Registered for the logical device, not gated on `type == "cuda"` | `ProcessGroupFlagOS` + hook registration |
| FSDP / HSDP | Supported only where the native stream/sharding path is validated | per-platform gate |
| Serialization | Round-trips a logical device | documented tag |
| Compiler target | `torch.compile` targets the backend's Triton/Inductor path | compile integration doc |
| Telemetry (`flops`, device specs) | Available where vendor metadata is trustworthy, else `None` | vendor properties |

Discovery success is not proof of working collectives or sharding: each row is gated
by a measurement, not by the API name existing.

## Architecture decision: device identity and dispatch

Two mutually exclusive directions; this doc recommends A now and leaves B as a
separate decision.

- **A — logical `flagos` tensors with a CUDA-facing API/control-plane contract**
  (recommended). Tensors stay PrivateUse1; the compatibility mode makes the
  framework *decisions* above agree with the logical device. Explicitly handles
  framework CUDA-only branches instead of faking identity.
- **B — CUDA tensor identity.** A `flagos` tensor would report CUDA identity and
  reach CUDA dispatch/guard/autograd/allocator/runtime/compiler bridging. This is a
  much larger architecture decision and must not be done by faking `.device.type` /
  `.is_cuda` or hijacking the CUDA dispatch key independently of those semantics.

## Adapters, not more global patches

The compatibility mode is assembled from **scoped, version-tested adapters** (#443),
not from new process-global patches. Concretely:

- Prefer public registration/extension hooks over replacing `torch.device`,
  `Tensor.cuda`, DDP `__init__`, a permanent `TorchFunctionMode`, or private
  Inductor methods.
- Where a patch is unavoidable, centralize it behind an idempotent, version-guarded
  adapter with explicit ordering and a diagnostic on upstream API change.
- Remove request-scoped global state changes (e.g. temporary DDP optimization-mode
  flips) or make them safe under concurrency.
- Reuse the `FLAGOS_STARTUP_PROFILE` hook structure so the compatibility hooks are
  independently activatable and individually diagnostic.

## Acceptance gates

Per platform class, a subprocess conformance test that validates the **actual**
behavior, not API labels:

1. Placement: `torch.ones(..., device="cuda")`, `model.cuda()` land on the backend.
2. Dispatch: the overload reaches the intended route (boxing / native / FlagGems),
   traced via `FLAGOS_LOG=dispatch`.
3. AMP: autocast + scaler produce the expected dtype and finite/overflow behavior.
4. Streams: multi-stream ordering is real (not a no-op `wait_stream`).
5. Distributed: default backend + a hook are reached, and a small collective
   transfers the right data.
6. Isolation: with real CUDA present, alias activation and stand-down behave as
   documented; native `flagos` and CUDA-facing modes do not both engage.

The gates for a platform are the definition of its "supported" row in the matrix.

## Rollout

1. **Design + matrix** (this doc): agree the contract, the platform classes and the
   device-identity decision; publish the per-platform support matrix.
2. **Native integration (complementary).** A first-party `flagos` accelerator/strategy
   for the motivating framework, so native mode is usable without per-model code.
3. **CUDA-facing mode + adapters.** Implement the contract for the boundaries agreed
   supported, on top of #443.
4. **Conformance gate in CI.** Land the acceptance tests per class; the matrix is
   generated/validated from them, feeding the support evidence in #203/#201.

## Open questions

1. Is **API/control-plane** compatibility sufficient, or is **CUDA tensor identity**
   (option B) required by a real consumer? This decides the size of the effort.
2. Which distributed backend does each mode expose by default (FlagCX vs native
   NCCL/RCCL/HCCL vs Gloo), and how is that surfaced to the framework?
3. What is the AMP contract on native non-CUDA platforms (BF16/FP16 autocast and
   scaler), and how is it validated?
4. Where the CUDA-facing mode cannot be honest (native non-CUDA class), does the
   framework get a clear "unsupported" error or a first-party integration path?
