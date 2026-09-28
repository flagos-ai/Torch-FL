# FlagCX integration with an NCCL fallback: unified distributed communication across vendors

This document describes how `torch_fl.distributed` integrates FlagCX uniformly on flagos
(PrivateUse1) devices, falling back to each hardware vendor's native communication backend when
FlagCX is unavailable. The goal is a single upper-layer API shared by nvidia, metax, and ascend,
while closing the correctness gaps in communication.

> Status: the native ProcessGroup implementation is active. Enflame GCU integration has been
> validated on two S60 devices; other vendor paths retain the validation status documented below.

---

## 0. Architecture evolution (current implementation; supersedes §4)

Sections 1–8 record the original design based on monkeypatching (`_resolve_backend` /
`_patch_dist_collectives` / `_register_privateuseone_backend`). That design has been
**superseded** and is retained as background. The current implementation is a native ProcessGroup
backend:

- **`torch_fl/comm/process_group.py :: ProcessGroupFlagOS`**
  Subclasses `torch.distributed.ProcessGroup` and overrides every collective virtual function
  (allreduce / allgather / reduce_scatter / alltoall / broadcast / gather / scatter / reduce /
  send / recv / barrier, …). Each override converts privateuseone tensors into the device view
  the inner backend needs (`_C._flagos_to_cuda_view`), delegates to `self._inner`, and returns
  the inner backend's Work. Inner backend priority: FlagCX → the vendor's native backend
  (HCCL for ascend, NCCL for nvidia/metax, MCCL for MUSA) → host-staged gloo, the last tier
  and the only one that needs no vendor library (§0.4).

- **Registration**: `import torch_fl` calls `register_flagos_backend()`, which runs
  `Backend.register_backend("flagos", creator, devices=["privateuseone"])` and sets
  `default_device_backend_map["privateuseone"] = "flagos"`. After that, the standard
  `torch.distributed.init_process_group("flagos")` (or automatic detection via
  `device_id=torch.device("privateuseone:0")`) works with no `torch.distributed.*` monkeypatching
  at all.

- **DDP**: `import torch_fl` patches
  `torch.nn.parallel.DistributedDataParallel.__init__`. When the model lives on privateuseone, it
  forces `python_reducer` (bypassing the C++ Reducer's CUDA assertion) and replaces the default
  accum-grad hook (which uses functional collectives, and privateuseone has no dispatch for
  those) with a version that goes through `dist.all_reduce` → ProcessGroupFlagOS.

### 0.1 The real FlagCX integration contract (GitHub main, v0.13.0, verified 2026-07)

Integrate against this contract; do not guess:

1. On `import flagcx`, its C++ side (in the `backend_flagcx.cpp` constructor) calls
   `torch.distributed.Backend.register_backend("flagcx", createFlagcxBackend,
   devices=(devName,), extended_api=True)` **itself**. `devName` is fixed at compile time by the
   adaptor: nvidia/metax/du/klx → `"cuda"`, ascend → `"npu"`, musa → `"musa"`, etc.
   **The registered device is cuda (or the vendor accelerator), not privateuseone.**
2. The backend name is fixed as `FLAGCX_BACKEND_NAME = "flagcx"`.
3. `dist.ProcessGroupFlagCX` is exposed through pybind only when
   `USE_NVIDIA_ADAPTOR || USE_METAX_ADAPTOR` and torch>=2.5; it subclasses
   `torch._C._distributed_c10d.Backend`.
4. Its construction takes the `extended_api=True` form: the creator is
   `flagcx.createFlagcxBackend(DistributedBackendOptions, Options)`, **not**
   `(store, rank, world_size, opts)`. So `ProcessGroupFlagOS`, in `_try_build_flagcx`, fills a
   `torch._C._distributed_c10d._DistributedBackendOptions` with store / group_rank / group_size /
   group_id / global_ranks_in_group / timeout and passes it to the creator. `Options`
   (`enable_tuner` / `tune_group_idx`) comes from `ProcessGroupFlagCX.Options`.
5. The FlagCX plugin's `__init__.py` also hacks PrefixStore via `replace_prefix`
   (`cuda→flagcx_dev`) and overrides `batch_isend_irecv` on torch>=2.7. These are flagcx's own
   behaviors and are unrelated to ProcessGroupFlagOS.
6. When flagcx is not installed, `_try_build_flagcx` returns False and the code falls back to
   HCCL/NCCL automatically.

### 0.2 GCU integration status (verified on two Enflame S60 devices)

- **Profile**: `"enflame"` uses device `"flagos"` with the plain FlagCX creator signature when the torch-fl-compatible build is selected.
- **Backend selection**: `ProcessGroupFlagOS` sets `FLAGCX_TORCH_BACKEND=flagos` before importing FlagCX when `GEMS_VENDOR=enflame`, while preserving an explicit user value.
- **Torch-fl compatibility mode**: FlagCX can be built with `FLAGCX_TORCH_BACKEND=flagos`; this mode links `libflagos.so` and `libtopsrt.so`, uses raw Tops stream/event calls, and does not import or link `torch_gcu`. The default selector remains available for existing vendor installations.
- **Stream interoperation**: torch-fl exports its current-stream registry through `flagos.h`. Topsaten, FSDP2, and FlagCX use that process-wide ABI so communication and compute can establish event dependencies without relying on vendor `torch_gcu` objects.
- **Device guard**: GCU-scoped guards force the current device to match the communicator and operands because GCU streams and pointers are device-scoped. A test that deliberately selects the other device before `all_reduce` passed on both ranks.
- **View conversion**: none. The profile sets `direct=True`, so `_resolve_view` returns the identity and FlagCX receives the privateuseone tensor directly.
- **Fallback backend**: No ECCL Python package; FlagCX is the only communication path for GCU.
- **Unit tests**: Profile selection, plain creator signature, backend-selector defaulting, explicit-selector preservation, and device guard logic are covered in `tests/unit/test_vendor_routing.py`.
- **Measured basic coverage**: `tests/manual/test_flagos_dist_gcu.py` passed on two S60 devices for all-reduce, broadcast, list all-gather, all-gather-into-tensor, reduce-scatter-tensor, barrier, DDP forward/backward, gradient synchronization, and the wrong-current-device guard.
- **Measured FSDP2 coverage**: per-layer `fully_shard`, training, bf16 parameters with fp32 reduction, gradient clipping, and sharded state-dict save/load passed on both ranks. The measured losses were `95.0701` (fp32) and `95.5000` (mixed precision), and the gradient norm was `1451.6295` on both ranks.

### 0.3 Remaining GCU validation

The manual suite does not yet exercise point-to-point operations, gather/scatter roots, all-to-all, multi-node rendezvous, process failure recovery, or more than two devices. Those paths remain unvalidated on Enflame hardware.

### 0.4 Host-staged gloo and gloo redirection (measured on MUSA MTT S5000)

A host with no FlagCX, no vendor torch and no vendor communication library used to have no
working collective at all: `ProcessGroupFlagOS` raised "no suitable inner backend" from
`backend="flagos"`, and `backend="gloo"` — what torchrun and the transformers FSDP2 tests ask
for — built a `ProcessGroupGloo` whose C++ collectives reject flagos tensors with
`ProcessGroupGloo::allreduce: unsupported device type flagos` (issue
[#263](https://github.com/flagos-ai/Torch-FL/issues/263)).

Two additions close that gap. Both are in `torch_fl/comm/process_group.py` and both are gated
by a runtime switch:

- **`_try_build_staged_gloo` — the last tier of `_build_inner`.** gloo ships inside c10d, so it
  is always present; this tier builds a real `ProcessGroupGloo` and wraps it in
  `_HostStagedGloo`, a `ProcessGroup` facade that moves every flagos operand through host memory
  (device → host → collective → device) and synthesises the two virtuals gloo does not implement
  (`allgather_into_tensor_coalesced`, `reduce_scatter_tensor_coalesced`) from gloo's own
  `_allgather_base` / `_reduce_scatter_base`. It costs a copy per collective, so it runs only
  after FlagCX and the vendor-native backend have declined, and it warns once when it is used.
  `FLAGOS_DIST_STAGED_GLOO=0` declines the tier so the failure is explicit instead.
- **`redirect_gloo_requests` — answered on the requested side.** gloo cannot be repaired from
  the gloo side: gloo's own `BackendConfig` expansion is what registers a `ProcessGroupGloo`
  under the flagos device type (`pg._device_types` reads `[('flagos', 'flagos'), ('cpu',
  'cpu')]` on a plain gloo group), its creator branch asserts `isinstance(backend_class,
  ProcessGroupGloo)`, and `c10d.Backend` cannot be subclassed from Python at all. So the request
  is answered with the `flagos` backend *before* the backend config is built: `import torch_fl`
  interposes on `torch.distributed.init_process_group` and `torch.distributed.new_group` and
  rewrites a `"gloo"` backend argument to `"flagos"`, but only when the process accelerator is
  the flagos device (`torch._C._get_accelerator().type`), because gloo is a legitimate choice on
  a cpu/cuda host. `FLAGOS_DIST_REDIRECT_GLOO=0` keeps the requested backend verbatim while
  debugging a distributed problem.

**Measured** (two ranks, no FlagCX / no `torch_musa.distributed` / no NCCL, i.e. the worst case;
`torch_fl/comm/process_group.py` as landed):

| Request | Result |
| --- | --- |
| `init_process_group(backend="flagos")` | `ProcessGroupFlagOS` over `_HostStagedGloo`; `all_reduce` → `[3.0, 3.0]` on both ranks |
| `init_process_group(backend="gloo")` | redirected to `flagos`; `all_reduce`, `all_gather_into_tensor`, `reduce_scatter_tensor`, cpu `all_reduce`, `barrier`, and `torch.ops._c10d_functional.all_gather_into_tensor` all correct on both ranks |
| `new_group([0, 1], backend="gloo")` | redirected to `flagos`; subgroup `all_reduce` → `[3.0, 3.0]` on both ranks |
| `FLAGOS_DIST_REDIRECT_GLOO=0` | both ranks get a plain `ProcessGroupGloo` again and fail on the flagos tensor — the switch is load-bearing |

**Partially validated on the FSDP2 suite.** Of the six `Qwen3ModelTest::test_fsdp2_*` nodeids
that #263 counted, three now pass (`test_fsdp2_save_load`, `test_fsdp2_sharding_structure_0_untied`,
`test_fsdp2_sharding_structure_1_tied`); the other three got past the collectives and then failed in
FSDP2's stream setup on a **separate** defect — `torch_fl.flagos.Stream.__init__` routed every
non-GCU platform without a CUDA runtime to the Ascend ACL stream, so on MUSA it raised
`RuntimeError: libascendcl.so not found. ACL runtime is required` at
`_fsdp_param_group.py::FSDPCommContext.lazy_init`. MUSA had no Python-side stream/event
implementation (`torch_fl/accelerator/` had `ascend`, `gcu`, `metax`, `dcu`, `cuda`, `bpu`,
`ppu`), so those three were blocked on that work, not on the collective path.

That work landed as PR #467 over `6b69999`, and with it the three nodeids are no longer blocked by
it: MUSA now has `torch_fl/accelerator/musa/musa_stream.py`, and the vendor dispatches name their
arms rather than ending in an `else` that meant Ascend. On an MTT S5000 the failing frame
(`FSDPCommContext.lazy_init`) and FSDP2 itself now run — `tests/manual/musa/test_fsdp2_musa.py
--world-size 2` trained to a finite, non-increasing loss across two MTT S5000 devices. That run's
collectives went over the host-staged gloo tier above, since this environment has no vendor
communicator; the MCCL path this document is about was not exercised by it, and the six nodeids were
not re-run through the HF harness.

---

## 1. Background and prior state

### 1.1 Zero-copy bridging

flagos tensors and CUDA tensors share the same device memory. `flagos_to_cuda_view_impl` in
`torch_fl/csrc/module.cc` builds a tensor carrying `DispatchKey::CUDA` over the same `data_ptr`
(holding a reference to the original flagos tensor so the memory is not freed). The communication
backend therefore receives what is, to it, a CUDA tensor, entirely unaware of flagos.

### 1.2 The original communication path (before the refactor)

`init_process_group` in `torch_fl/distributed.py` had two legs:

- `backend="nccl"`: an ordinary `dist.init_process_group("nccl")`, then
  `_register_privateuseone_backend` copies the cuda backend registration onto the privateuseone
  device, and `_patch_dist_collectives` monkeypatches 5 collective APIs into "convert to a cuda
  view first, then call the original".
- `backend="flagcx"`: `import flagcx` triggers the entry-point registration, and initialization
  uses `backend="cpu:gloo,cuda:flagcx"`.

`_patch_dist_collectives` covered only 5 APIs: `all_reduce`, `broadcast`, `reduce`,
`all_gather_into_tensor`, `reduce_scatter_tensor`.

### 1.3 The authoritative source for vendor detection

`_patch_flaggems_codegen_config()` in `torch_fl/__init__.py` sets the `GEMS_VENDOR` environment
variable to one of `nvidia` / `metax` / `ascend` at import time. The distributed layer should
**reuse that variable directly** rather than starting a second hardware detection scheme.

---

## 2. Identified problems

1. **BackendType hardcoded to NCCL**: `_register_privateuseone_backend` hardcodes
   `BackendType.NCCL`, which does not hold on ascend (HCCL is not the NCCL strong type).
2. **View target hardcoded to cuda**: `_ensure_cuda` blindly does flagos→cuda. That works on
   nvidia/metax (metax goes through maca's libtorch_cuda compatibility layer), but **ascend has
   no CUDA compatibility layer**, so this path simply does not work there.
3. **API coverage gaps**: any collective API that was not patched crashes as soon as it is called
   with a privateuseone tensor. The gaps include `all_gather` (the list form), `gather`/`scatter`,
   `all_to_all[_single]`, `send`/`recv`/`isend`/`irecv`, `barrier`, and — most insidiously —
   `torch.ops._c10d_functional.*` (used by the torch.compile / DTensor / FSDP2 compiled paths).
4. **Missing fallback logic**: no automatic downgrade to the vendor's native backend when flagcx
   is unavailable.

---

## 3. Target architecture

```
                flagos_dist.init_process_group(backend="auto")
                                │
                ┌───────────────┴────────────────┐
                │   _resolve_backend()           │  reads GEMS_VENDOR + the user request
                │   FlagCX first, vendor native  │
                └───────────────┬────────────────┘
     ┌──────────────────┬───────┴────────┬──────────────────┐
  flagcx available?   nvidia            metax             ascend
     │ yes              │ nccl fallback   │ mccl fallback    │ hccl fallback
unified flagcx backend  │ (NCCL type)     │ (via maca        │ (CUSTOM type,
(CUSTOM type)           │                 │  libtorch_cuda)  │  torch_npu)
     │                  └────────┬────────┘                  │
     │            flagos→cuda view (shared data_ptr)     flagos→npu view
     │                           │                    (no CUDA compat layer!)
     └────────── natively accepts privateuseone, no view needed ───────────┘
```

---

## 4. Three changes

### 4.1 `_resolve_backend()`: backend resolution and fallback (fallback semantics A)

Concentrate all "which one / when to downgrade" logic into one pure-Python function:

- Read `GEMS_VENDOR` to get the vendor and map it to the vendor's native backend:
  - `nvidia` → `nccl`
  - `metax`  → `nccl` (maca's mccl looks like the NCCL backend to PyTorch; it links mccl
    underneath)
  - `ascend` → `hccl`
- Respect an explicit user choice of `nccl`/`hccl`/`flagcx`.
- `auto` (the default) prefers flagcx.
- If `import flagcx` fails on the flagcx path, warn and fall back to the vendor's native backend.

Returns `(actual backend string, vendor)`.

### 4.2 `_register_privateuseone_backend()`: BackendType routed by vendor

- BackendType mapping: `flagcx`→`CUSTOM`, `nccl`→`NCCL`, `hccl`→`CUSTOM`.
- Detect the original backend's device: take it from `privateuseone` on ascend, from `cuda`
  otherwise.

### 4.3 `_patch_dist_collectives()`: view routing and full API coverage (fallback semantics B)

**(a) Route the view target by vendor**, abstracted as `_ensure_comm_tensor(t, vendor)`:

- `ascend` → `_flagos_to_npu_view(t)` (needs a new C++ implementation, or use native flagcx
  registration to avoid views entirely)
- everything else → `_flagos_to_cuda_view(t)` (reuses the existing one)

**(b) Complete the API coverage**: use a generic patch generator that walks a table of
`(function name, which positional/keyword arguments are tensors)` and wraps them in bulk, instead
of hand-writing each function. It must cover at least:

- `all_reduce`, `broadcast`, `reduce`, `all_gather_into_tensor`, `reduce_scatter_tensor` (the
  original 5)
- `all_gather` (the list form), `gather`, `scatter`
- `all_to_all`, `all_to_all_single`
- `send`, `recv`, `isend`, `irecv`
- `barrier` (the device_ids argument)
- `torch.ops._c10d_functional.*` (functional collectives)

---

## 5. Native FlagCX registration (no views) — the hypothesis to verify first

If flagcx's adaptor only cares about `data_ptr + stream` and does not validate the device type,
then the flagcx backend can be registered directly for privateuseone, **eliminating both the
patches and the views** and covering every collective API by construction. This matters most for
ascend (it removes the `_flagos_to_npu_view` C++ work). Once a flagcx environment is available,
this is the first thing to verify.

---

## 6. Per-vendor status

| Vendor  | flagcx path | Native fallback | View conversion | Main gap |
|---------|------------|-----------------|-----------------|----------|
| nvidia  | works today | nccl (works today) | flagos→cuda (works today) | API coverage only |
| metax   | should be reusable | "nccl"@maca | flagos→cuda (via maca) | mccl needs measuring |
| ascend  | **recommended primary path** | hccl (CUSTOM) | **flagos→npu missing** | either the view or native registration |
| enflame | primary path (2026-08) | none (FlagCX only) | none needed (`direct=True`) | P2P, root-based collectives, and larger deployments remain unvalidated |

---

## 7. Implementation order

1. **Refactor `distributed.py`** (without changing nvidia behavior): extract `_resolve_backend`,
   the vendor-routed `_register_privateuseone_backend`, and `_ensure_comm_tensor` to establish
   the architectural skeleton.
2. **Complete the collective API coverage** (including functional collectives): pure Python, no
   hardware dependency, verifiable for correctness in the current environment.
3. **Verify in a flagcx environment**: test the no-view native registration hypothesis, then
   decide whether ascend uses native registration or needs `_flagos_to_npu_view`.
4. **Verify the fallback paths on metax / ascend hardware.**

---

## 8. Public API (after the refactor)

```python
import torch_fl.distributed as flagos_dist

# backend values:
#   "auto"   -> flagcx first, falling back to the vendor native backend (recommended)
#   "flagcx" -> force flagcx, falling back to the vendor native backend
#   "nccl"   -> force nccl (nvidia/metax)
#   "hccl"   -> force hccl (ascend)
flagos_dist.init_process_group(backend="auto")

model = flagos_dist.DistributedDataParallel(model)
flagos_dist.move_buffers_to_device(model, "flagos:0")
```
