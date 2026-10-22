---
name: runtime-bringup
description: >
  Implement the torch_fl device runtime contract for an accelerator that has no
  backend at all yet — the ~38-function floor (28 C ABI functions +
  10 allocator virtuals) that must exist before any operator can run. Use this
  as the FIRST step for any new chip (Kunlun XPU, or any other), regardless of
  whether it will later use CUDA-compatible boxing or a native operator library.
  Covers: the exact function inventory and which file each goes in, the
  USE_<VENDOR> macro wiring across three CMake sites, the allocator backend
  selection block, and the operator-free smoke test that proves the runtime works.
---

# Device runtime bringup (torch_fl new accelerator)

## What this achieves and what it deliberately does not

This skill brings a chip from "torch_fl has never heard of it" to "torch_fl can
allocate a tensor on it, copy H2D/D2H, and synchronize streams and events" —
**with zero operators registered**. That intermediate state is real, testable,
and the correct place to stop.

It is a **gate, not a subroutine**. Both operator strategies
([[cuda-compat-vendor]] for CUDA-compatible chips, [[native-op-backend]] for
chips with their own operator library) assume this has already passed. Do not
start either until the smoke test below is green, because an operator failure on
top of a broken runtime is nearly impossible to attribute.

Nothing here depends on torch version — that axis belongs to
[[torch-version-port]].

## The contract, counted from the tree

Do not estimate this inventory. It is fixed and enumerable:

| Surface | File | Count |
|---|---|---|
| C ABI runtime functions | `csrc/include/flagos.h` (`FLAGOS_EXPORT Error_t`) | **28** |
| Allocator virtuals | `csrc/runtime/allocator/device_memory_interface.h` (`= 0;`) | **10** |
| Vendor stream accessor | `GetDefault<Vendor>Stream` (optional, see below) | 0–2 |

Verify the count yourself before starting, since `flagos.h` may have grown:

```bash
grep -c 'FLAGOS_EXPORT Error_t' csrc/include/flagos.h        # expect 28
grep -c '= 0;' csrc/runtime/allocator/device_memory_interface.h  # expect 10
```

Every existing backend implements all 28 — cuda, musa, gcu, ascend, tsingmicro
and bpu each expose exactly 28. There is no "partial runtime" precedent, and
`torch.empty` on the device will fault if any are stubbed to return `Success`
without doing the work.

## Step 1 — read the reference implementation that matches your chip's shape

Pick the closest existing vendor and read all three of its files end to end
before writing anything. The choice matters more than it looks:

| If the SDK … | Read | Because |
|---|---|---|
| ships a `libcudart` shim (`cuda_runtime.h` works) | `accelerator/cuda/*.cc` — and reuse it verbatim via CMake, like `dcu` does | zero new runtime code; DCU adds no `dcu/` directory at all |
| is CUDA-shaped but renamed (`xxxMalloc`, `xxxStream_t`) | `accelerator/musa/*.cc` | a mechanical 1:1 port of the CUDA sources onto a renamed C API |
| is a different model entirely (two-stage, context objects) | `accelerator/ascend/*.cc` + `accelerator/gcu/*.cc` | shows how to keep a shared default stream and drain it before host-visible memcpy |
| has no stream concept | `accelerator/bpu/*.cc` | the degenerate case: streams emulated, `Event*` mostly no-ops |

Kunlun XPU (XRE) is a *renamed-CUDA-shape* SDK (`xpu_malloc` / `XPUStream`), so
`musa/` is the reference. Confirm that from the headers rather than assuming it.

## Step 2 — the three files, and exactly what goes in each

The 28 functions partition into three files per vendor. Keep the split; the CMake
glob and every other backend depend on it.

```
csrc/runtime/accelerator/<vendor>/
├── device.cc   #  5
├── memory.cc   #  9
└── stream.cc   # 14
```

`device.cc` (5): `GetDeviceCount`, `GetDevice`, `SetDevice`,
`DeviceGetStreamPriorityRange`, `DeviceSynchronize`

`memory.cc` (9): `Malloc`, `Free`, `MallocHost`, `FreeHost`, `Memcpy`,
`MemcpyAsync`, `PointerGetAttributes`, `Memset`, `MemsetAsync`

`stream.cc` (14): `StreamCreateWithPriority`, `StreamCreate`,
`StreamGetPriority`, `StreamDestroy`, `StreamQuery`, `StreamSynchronize`,
`StreamWaitEvent`, `EventCreateWithFlags`, `EventCreate`, `EventDestroy`,
`EventRecord`, `EventSynchronize`, `EventQuery`, `EventElapsedTime`

### The error mapping is the part that gets rushed

`Error_t` has exactly five values (`Success`, `ErrorUnknown`,
`ErrorNotReady`, `ErrorInvalidDevice`, `ErrorMemoryAllocation`). Two mappings are
load-bearing and silently break things if collapsed into `ErrorUnknown`:

- **`ErrorNotReady`** — `StreamQuery` and `EventQuery` MUST return this (not an
  error) for "still running". The caching allocator's block-reuse path treats
  `ErrorNotReady` as "not safe to reuse yet" and anything else as a hard failure,
  so collapsing it either deadlocks or corrupts.
- **`ErrorMemoryAllocation`** — OOM must be distinguishable, or torch's
  retry-after-`empty_cache` path never triggers and users get a crash where they
  should get a shrunk workspace.

### `PointerGetAttributes` is not optional

It fills `MemoryType` (`Unmanaged`/`Host`/`Device`) plus device index. `copy_`
uses it to decide whether a pointer needs a device copy or a plain `memcpy`. If
the SDK has no equivalent query, track allocations in a side table in `memory.cc`
rather than returning a fixed `MemoryTypeDevice` — a wrong answer here shows up
much later as a wrong-results bug in `.cpu()`, not as a runtime error.

### The async-dispatch memcpy hazard

If operators on this chip dispatch asynchronously on a shared default stream,
`Memcpy` (the synchronous one) must **drain that stream first**, or `.cpu()`
races the kernel that produces the data. Ascend hit this and solved it in
`accelerator/ascend/acl_stream.h` + a drain at the top of `memory.cc`. Read that
before writing `Memcpy`; the failure is nondeterministic and looks like a kernel
bug.

## Step 3 — the allocator backend (10 virtuals)

Create `csrc/runtime/allocator/backends/<vendor>_memory.h` implementing
`DeviceMemoryInterface`: `device_malloc`, `device_free`, `get_device_index`,
`set_device`, `get_memory_info`, `event_create`, `event_destroy`, `event_record`,
`event_query`, `memcpy`.

These mostly forward to the C functions from step 2. Two decisions are real:

**`get_memory_info(free, total)`** — if the SDK cannot report free bytes, do not
invent a number. Report total and a conservative free; `torch.<device>.mem_get_info`
and OOM-retry logic read this.

**`provides_caching()`** — the escape hatch. Return `true` only if the platform
ships a mature caching allocator that does **not** claim the PrivateUse1
allocator slot torch_fl registers. Getting this wrong is a known trap:

- DCU returns `true` and delegates through the device-generic registry (c10::hip
  underneath, no c10::cuda symbols) — see `backends/dcu_memory.h`.
- MUSA deliberately returns `false` even though a `MUSACachingAllocator` exists,
  because it claims the same PrivateUse1 slot flagos registers — see the comment
  at `caching_device_allocator.cc:551`.

**Default to `false`** for a new chip. The built-in block pool works, and you can
delegate later once the vendor allocator's slot behaviour is understood.

## Step 4 — wire the build (three sites, all mandatory)

Missing any one of these produces a confusing failure rather than a clean error.

**Site 1 — `csrc/runtime/accelerator/CMakeLists.txt`.** Two separate blocks:
the `project()` selector near the top (non-CUDA vendors declare `CXX C` so cmake
does not demand nvcc), and the per-vendor source/link block below.

```cmake
elseif(FLAGOS_ACCELERATOR STREQUAL "<vendor>")
  project(FLAGOS_RUNTIME CXX C)
...
elseif(FLAGOS_ACCELERATOR STREQUAL "<vendor>")
  file(GLOB SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/<vendor>/*.cc")
  add_library(${LIBRARY_NAME} SHARED ${SOURCE_FILES})
  target_include_directories(${LIBRARY_NAME} PUBLIC
      ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR} ${<VENDOR>_HOME}/include)
  target_link_directories(${LIBRARY_NAME} PRIVATE ${<VENDOR>_HOME}/lib)
  target_link_libraries(${LIBRARY_NAME} PRIVATE <vendor_runtime_lib>)
```

Emit a `FATAL_ERROR` when the SDK is not found, naming the env var to set — copy
the wording style of the bpu/ascend blocks. A link error 40 lines into a build is
a worse experience than a configure-time message.

**Site 2 — `csrc/CMakeLists.txt`**, define the macro:

```cmake
target_compile_definitions(${LIBRARY_NAME} PRIVATE USE_<VENDOR>=1)
```

**Site 3 — `csrc/runtime/allocator/caching_device_allocator.cc`**, two edits.
The include block at the top, and the selection block in
`GetCachingAllocator()`. Note the CUDA include is guarded by a **negative**
condition listing every non-CUDA vendor:

```cpp
#if !defined(USE_ASCEND) && !defined(USE_TSINGMICRO) && !defined(USE_DCU) && \
    !defined(USE_GCU) && !defined(USE_MUSA) && !defined(USE_BPU)
#include "backends/cuda_memory.h"
#endif
```

**Add your `USE_<VENDOR>` to that negative list.** Forgetting it is the single
most likely mistake in this whole skill: both `cuda_memory.h` and your header get
included, and you get a duplicate-definition or wrong-backend selection that
points nowhere near the real cause.

Then add the `#elif defined(USE_<VENDOR>)` arm to `GetCachingAllocator()`, with a
comment saying *why* your `provides_caching()` choice is what it is — every
existing arm does, and those comments are the reason the DCU/MUSA divergence is
understandable at all.

**`setup.py`** — add the `FLAGOS_ACCELERATOR == "<vendor>"` arm for SDK discovery and
cmake args. Follow the `musa`/`gcu` arms.

## Step 5 — the operator-free smoke test

Two gates. Symbol completeness first, because it is instant:

```bash
nm -D torch_fl/lib/libflagos.so | grep -c ' T '   # every one of the 28 exported
```

Then behaviour. This must pass with **no operators registered** — that is the
point. Anything that reaches a kernel does not belong in this test:

```python
import torch, torch_fl
assert torch.flagos.device_count() > 0
x = torch.empty(4, 4, device="flagos")          # Malloc
h = torch.arange(16, dtype=torch.float32).reshape(4, 4)
d = h.to("flagos")                              # Memcpy H2D
assert torch.equal(d.cpu(), h)                  # Memcpy D2H
s = torch.flagos.Stream(); e = torch.flagos.Event(enable_timing=True)
with torch.flagos.stream(s):
    y = torch.empty(1 << 20, device="flagos")
e.record(s); s.synchronize(); e.synchronize()
assert e.query()
del x, d, y
torch.flagos.empty_cache()                      # block pool churn
print(torch.flagos.memory_allocated(), torch.flagos.max_memory_allocated())
```

Add it as `tests/integration/test_<vendor>_runtime.py` with a
`@pytest.mark.<vendor>` marker registered in `pyproject.toml`.

Also exercise the allocator under churn — allocate/free in a loop across two
streams. Block reuse is where an `ErrorNotReady` mapping mistake surfaces, and a
single-allocation test will never catch it.

## Done criteria

- `grep -c 'FLAGOS_EXPORT Error_t' csrc/include/flagos.h` functions all
  implemented, none stubbed to a bare `return Success;`
- All 10 allocator virtuals implemented; `provides_caching()` decision commented
- `USE_<VENDOR>` added at all three CMake/allocator sites, **including the
  negative CUDA-include guard**
- `FLAGOS_ACCELERATOR=<vendor> pip install -e .` builds
- Smoke test passes with zero operators registered
- Allocator churn loop across two streams passes

## What to hand off

State explicitly which operator path comes next and on what evidence. If the
vendor ships a torch wheel containing `libtorch_cuda.so` whose `nm` shows
`at::add`/`at::mm`, go to [[cuda-compat-vendor]] — it is roughly an order of
magnitude less work. Otherwise [[native-op-backend]]. Decide from the SDK, not
from marketing claims about CUDA compatibility.
