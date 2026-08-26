---
name: flagcx-integration
description: >
  Integrate and validate FlagCX distributed communication for torch_fl after the
  device runtime and vendor communication path are available. Use this for a
  new FlagCX vendor adaptor, a FlagCX or PyTorch version change, a process-group
  routing change, or a distributed correctness survey. Covers the FlagCX
  registration contract, vendor profiles, NCCL/HCCL/MCCL fallback, zero-copy
  tensor views, process-group and DDP automation, multi-process collectives,
  and evidence boundaries. Do not infer distributed support from import success.
---

# FlagCX integration (torch_fl)

## Scope and prerequisites

FlagCX is the heterogeneous distributed-communication layer. It is an optional
inner backend below torch-fl's `ProcessGroupFlagOS`; it does not replace device
allocation, streams, copies, synchronization, vendor kernels, or rendezvous.
A FlagCX integration is ready only when a real multi-process collective reaches
the intended adaptor and produces the expected result.

Complete these first:

- [[runtime-bringup]] — device discovery, allocation, streams, and copies pass on
  the target hardware.
- [[torch-version-port]] — the active PyTorch minor line matches the generated
  bindings and the c10d creator ABI.
- [[cuda-compat-vendor]] or [[native-op-backend]] — the vendor's compute path and
  its communication fallback are understood before adding distributed routing.
- [[pre-pr-checks]] — fetch and rebase before linting or opening a PR.

The integration has four separate surfaces:

| Surface | Source of truth | What it does |
|---|---|---|
| FlagCX package/adaptor | `import flagcx` and the vendor FlagCX build | Registers the `flagcx` c10d backend and exposes its creator/inner process group |
| torch-fl process group | `torch_fl/comm/process_group.py` | Tries FlagCX first, converts tensors for the selected backend, and falls back to the vendor native backend |
| vendor profile | `_VENDOR_PROFILES` in `process_group.py` | Maps `GEMS_VENDOR` to FlagCX device name, tensor view, direct/native mode, and fallback |
| public registration | `torch_fl/__init__.py`, `torch_fl/distributed.py` | Registers `flagos` for PrivateUse1 and makes standard c10d/DDP APIs work |

A backend import is discovery, not validation. A process-group constructor is
also not validation. The minimum correctness claim is a two-rank collective
against a CPU-computed expected result, followed by the DDP gradient path when
that platform supports it.

## Non-negotiable correctness contract

A FlagCX integration is **correctly connected** only when all of these gates pass:

1. The intended `flagcx` package and shared libraries are loaded by the same
   interpreter that loads torch-fl.
2. Importing `flagcx` registers the expected c10d backend and the adaptor's
   actual compile-time device name is recorded.
3. A strict live run proves that `ProcessGroupFlagOS` selected FlagCX. A native
   fallback must be made fatal in this run; a successful NCCL/HCCL/MCCL fallback
   is not FlagCX evidence.
4. At least two independent ranks complete `all_reduce`, `broadcast`,
   `all_gather`, `_allgather_base`, `_reduce_scatter_base`, and `barrier` with
   CPU-computed expected values and a zero process exit status.
5. DDP forward/backward completes when the platform claims DDP support, and
   rank-wise gradients agree within the documented tolerance.
6. A separate fallback run disables or rejects FlagCX and proves the documented
   native backend path. Its output is reported separately from the FlagCX run.
7. The harness fails closed: missing FlagCX, wrong adaptor selection, a fallback
   call, a rank mismatch, a timeout, or an unchecked assertion makes the command
   nonzero.

If any gate is missing, report **FlagCX not validated**. Never use the phrase
"FlagCX works" when only the routing unit tests, import check, process-group
construction, or native fallback passed.

The final status must be one of these, with the corresponding evidence attached:

| Status | Required evidence |
|---|---|
| `FlagCX validated` | strict two-rank FlagCX selection proof + collective matrix + DDP where claimed + zero exit |
| `FlagCX validated; native fallback validated` | all of the above plus an independent forced-fallback matrix |
| `FlagCX not validated; native fallback validated` | strict run blocked or failed, independent fallback matrix passes |
| `Distributed communication not validated` | neither a strict FlagCX matrix nor an independent fallback matrix passes |

Never collapse these statuses into one generic "distributed supported" label.

## Step 0 — make backend selection observable and fail closed

The live harness must distinguish "FlagCX selected" from "some backend selected".
Do not infer this from a class name, a successful collective, or the absence of
a warning: a CUDA/NCCL fallback can produce the same result. Use one of these
mechanical proofs:

- **Preferred:** expose a diagnostic `ProcessGroupFlagOS._inner_backend` value
  (`"flagcx"` or `"native:<builder>"`) and assert it on every rank.
- **Required when the diagnostic field is not available:** monkeypatch the native
  builders to raise a sentinel error in the strict FlagCX run, and wrap
  `_try_build_flagcx` so a false return raises immediately. This makes fallback
  impossible to mistake for success:

```python
from torch_fl.comm.process_group import ProcessGroupFlagOS

_NATIVE = ("_try_build_nccl", "_try_build_hccl", "_try_build_mccl")
_original_flagcx = ProcessGroupFlagOS._try_build_flagcx

def _require_flagcx(self, store, rank, world_size, timeout):
    selected = _original_flagcx(self, store, rank, world_size, timeout)
    if not selected:
        raise AssertionError("FlagCX was unavailable or not selected")
    return True

ProcessGroupFlagOS._try_build_flagcx = _require_flagcx
for _name in _NATIVE:
    setattr(
        ProcessGroupFlagOS,
        _name,
        lambda *args, _name=_name, **kwargs: (_ for _ in ()).throw(
            AssertionError(f"native fallback {_name} was selected")
        ),
    )
```

Apply this in every spawned rank before `init_process_group`. Do not catch the
sentinel. The strict run must fail if `flagcx` is missing, its creator returns
false, or any fallback builder is reached. The fallback run uses the inverse
proof: force `_try_build_flagcx` to return false, assert the expected native
builder was called, and label the result as fallback evidence.

## Step 1 — identify the communication environment

Keep the torch, FlagCX, vendor runtime, and launcher explicit. The torch used
to build torch-fl must match the branch's PyTorch minor line. Do not install a
CPU-only or NVIDIA wheel merely to make `import flagcx` succeed on a vendor box.

Record the environment before changing code:

```bash
python - <<'PY'
import importlib.util
import os
import sys
import torch

print("python", sys.version)
print("torch", torch.__version__)
print("torch file", torch.__file__)
print("cuda", torch.version.cuda)
print("GEMS_VENDOR", os.environ.get("GEMS_VENDOR", "<unset>"))
print("FLAGCX_TORCH_BACKEND", os.environ.get("FLAGCX_TORCH_BACKEND", "<unset>"))
for name in ("flagcx", "torch.distributed"):
    spec = importlib.util.find_spec(name)
    print(name, spec.origin if spec else "not found")
try:
    import flagcx
    print("flagcx file", flagcx.__file__)
    print("creator", getattr(flagcx, "createFlagcxBackend", None))
    print("pg class", getattr(flagcx, "ProcessGroupFlagCX", None))
except Exception as exc:
    print("flagcx import failed:", repr(exc))
PY

python -m pip show torch flagcx torch-fl 2>/dev/null || true
```

For a vendor box, also record:

- vendor and adaptor name (`GEMS_VENDOR`, `USE_NVIDIA_ADAPTOR`, `USE_ASCEND_ADAPTOR`, etc.);
- FlagCX revision and build options;
- PyTorch version and c10d ABI;
- vendor runtime/driver/SDK revision;
- physical device count and visible-device mapping;
- launcher (`torchrun`, `mp.spawn`, scheduler) and rendezvous settings;
- `LD_LIBRARY_PATH`, `PYTHONPATH`, and any `FLAGCX_*` variables.

A missing package is not the end of discovery. If `import flagcx` fails, fetch
FlagCX from its upstream GitHub repository and build the matching C++ library and
PyTorch plugin before declaring the FlagCX path unavailable. A genuinely missing
vendor SDK/CCL, unresolved shared library, ABI mismatch, or failed source build
is an environment failure; record the exact blocker and only then test the native
fallback. Never silently report a successful FlagCX integration.

## Step 2 — fetch and build FlagCX when the package is absent

Use the upstream repository, not an unrelated PyPI placeholder:

```bash
export FLAGCX_SRC=${FLAGCX_SRC:-$PWD/.vendor/FlagCX}
if ! python -c 'import flagcx' >/dev/null 2>&1; then
  if [ ! -d "$FLAGCX_SRC/.git" ]; then
    git clone --recurse-submodules https://github.com/FlagOpen/FlagCX.git "$FLAGCX_SRC"
  else
    git -C "$FLAGCX_SRC" fetch --tags origin
    git -C "$FLAGCX_SRC" pull --ff-only
    git -C "$FLAGCX_SRC" submodule update --init --recursive
  fi
fi
```

Select the adaptor from the actual hardware and the FlagCX build table. For a
Kunlun P800, the upstream source currently uses `USE_KUNLUNXIN=1`,
`DEVICE_HOME=/usr/local/xpu`, and `CCL_HOME=/usr/local/xccl`; the CCL SDK must
provide `include/` and `so/libbkcl.so`. Do not substitute `/usr/local/xpu` for
`/usr/local/xccl`: the device runtime and the collective library are separate
inputs.

The vendor torch may ship BKCL inside its Python package rather than in the
standard `/usr/local/xccl` layout. It is valid to stage a temporary compatibility
layout with symlinks, without copying or modifying vendor files:

```bash
TORCH_XMLIR=$(python -c 'import torch_xmlir, pathlib; print(pathlib.Path(torch_xmlir.__file__).parent)')
export FLAGCX_CCL_STAGING=${FLAGCX_CCL_STAGING:-/tmp/flagcx-ccl}
mkdir -p "$FLAGCX_CCL_STAGING/include" "$FLAGCX_CCL_STAGING/so"
ln -sf "$TORCH_XMLIR/xccl/include/bkcl.h" "$FLAGCX_CCL_STAGING/include/bkcl.h"
ln -sf "$TORCH_XMLIR/libbkcl.so" "$FLAGCX_CCL_STAGING/so/libbkcl.so"
```

Before building, assert both files exist and inspect their dependencies with
`ldd`; a header-only directory is not a usable CCL SDK. If the SDK is absent,
stop with an explicit build gap instead of linking a different vendor's CCL.

Build the C++ library first, then build the torch plugin against the same source,
interpreter, torch, and adaptor:

```bash
cd "$FLAGCX_SRC"
git submodule update --init --recursive
make USE_KUNLUNXIN=1 DEVICE_HOME=/usr/local/xpu \
  CCL_HOME="${FLAGCX_CCL_HOME:-$FLAGCX_CCL_STAGING}" \
  PLATFORM_EXTRA_SRCS=flagcx/adaptor/device_api/default_dev_api_backend.cc \
  -j"${FLAGCX_BUILD_JOBS:-$(nproc)}"

cd plugin/torch
CUDA_HOME=/usr/local/cuda-12.9 \
CXXFLAGS="-I/usr/local/cuda-12.9/targets/x86_64-linux/include -I/usr/local/xpu/include" \
CPPFLAGS="$CXXFLAGS" \
LDFLAGS="-L/usr/local/xcudart/lib -L/usr/local/xpu/so -L${FLAGCX_CCL_HOME:-$FLAGCX_CCL_STAGING}/so" \
LIBRARY_PATH="/usr/local/xcudart/lib:/usr/local/xpu/so:${FLAGCX_CCL_HOME:-$FLAGCX_CCL_STAGING}/so" \
FLAGCX_ADAPTOR=klx \
python -m pip install . --no-build-isolation
```

The explicit `PLATFORM_EXTRA_SRCS` is required for current FlagCX source when
the Kunlun makefile leaves the device API backend empty; without it
`libflagcx.so` can build with an unresolved `devApiBackend` and the Python
plugin fails at import. Keep this workaround in the evidence and re-check it
when the upstream makefile changes.

Use the adaptor name expected by `plugin/torch/_build_config.py`; the root Make
variable (`USE_KUNLUNXIN`) and the torch plugin adaptor (`klx`) are not always
the same string. For other vendors, read the corresponding `makefiles/*.mk`
and `ADAPTOR_MAP` entry instead of guessing paths or flags.

After installation, point the loader and Python at the artifacts and verify the
exact files loaded:

```bash
export PYTHONPATH="$FLAGCX_SRC/plugin/torch:$PYTHONPATH"
export LD_LIBRARY_PATH="$FLAGCX_SRC/build/lib:$FLAGCX_SRC/build/lib64:$LD_LIBRARY_PATH"
python - <<'PY'
import flagcx
import torch
import torch.distributed as dist
print("flagcx", flagcx.__file__)
print("torch", torch.__version__, torch.__file__)
print("creator", getattr(flagcx, "createFlagcxBackend", None))
print("backend map", getattr(dist.Backend, "default_device_backend_map", {}))
PY
```

If the vendor CCL SDK is absent, stop the build with an explicit evidence gap:
`FlagCX source fetched but not buildable: missing <path>`. Do not weaken the
adaptor to compile against a different vendor or mark the package as installed
just because the Python directory exists. Re-run this whole step after every
FlagCX or torch minor-line change.

## Step 3 — verify the FlagCX registration contract

Do not guess the contract from an older integration or from a package name.
For the current FlagCX build, inspect the actual registration and creator:

```bash
python - <<'PY'
import torch
import torch.distributed as dist
import flagcx

print("flagcx", flagcx.__file__)
print("creator", flagcx.createFlagcxBackend)
print("flagcx ProcessGroup", getattr(flagcx, "ProcessGroupFlagCX", None))
print("registered backends", getattr(dist.Backend, "default_device_backend_map", {}))
PY
```

The torch-fl integration currently relies on this contract:

1. `import flagcx` self-registers the c10d backend named `flagcx` through
   `Backend.register_backend`.
2. The adaptor's compile-time device name is not necessarily `privateuseone`.
   NVIDIA/MetaX CUDA adaptors use `cuda`; other adaptors may use `cann`, `musa`,
   `mlu`, `flagos`, or another vendor device name. That value belongs in the
   vendor profile, not in a global constant.
3. NVIDIA and MetaX builds on current PyTorch expose the extended creator form:
   `createFlagcxBackend(_DistributedBackendOptions, Options)`.
4. Other adaptors may expose the plain c10d form:
   `createFlagcxBackend(store, rank, world_size, timeout)`.
5. `ProcessGroupFlagOS._try_build_flagcx` must try the extended form, retry the
   plain form only for a signature `TypeError`, and warn/fall back for real
   initialization failures.
6. The creator's registered device and the tensor view passed to the inner
   backend are separate decisions. A backend registered for `cuda` can consume
   a zero-copy CUDA view; a native FlagCX adaptor may consume a PrivateUse1
   tensor directly.

Capture the exact creator signature and registered device for the adaptor under
test. A passing import with a different device name or creator ABI is not enough.

## Step 3a — define the vendor profile before wiring code

Add or review one row in `_VENDOR_PROFILES` rather than adding vendor-specific
branches throughout `ProcessGroupFlagOS`:

| Field | Meaning |
|---|---|
| `flagcx_dev` | Device name under which the adaptor registers FlagCX |
| `view` | `torch_fl._C` helper that creates the zero-copy physical-device view, or `None` |
| `native` | Native fallback builder such as `_try_build_nccl`, `_try_build_hccl`, or `_try_build_mccl` |
| `flagcx_native` | FlagCX adaptor accepts PrivateUse1 tensors directly; no view is needed on that path |
| `direct` | The inner backend receives the PrivateUse1 tensor directly for all paths; opt in only after measurement |

Examples in the current implementation:

- CUDA-ABI vendors (`nvidia`, `metax`, `hygon`, `kunlunxin`, `thead`) use
  `flagcx_dev="cuda"`, `_flagos_to_cuda_view`, and an NCCL/RCCL fallback.
- Ascend uses `flagcx_dev="cann"`, no view, `flagcx_native=True`, and HCCL as a
  fallback. The HCCL fallback must not receive a raw PrivateUse1 tensor unless a
  measured view is added.
- Enflame uses `flagcx_dev="flagos"`, `direct=True`, and no native fallback in
  the torch-fl-compatible FlagCX mode.
- MUSA uses `flagcx_dev="musa"`, `_flagos_identity_view`, and MCCL fallback.

For a CUDA-compatible vendor, prove the zero-copy view preserves `data_ptr`,
device index, lifetime, and stream semantics before using it for collectives. For
a native adaptor, prove that the adaptor consumes the PrivateUse1 tensor itself
before setting `flagcx_native` or `direct`. Otherwise fail loudly rather than
passing a tensor with the wrong device type to a communication library.

## Step 4 — implement automatic selection and fallback

The required selection order is:

```text
FlagCX available and constructible
    -> use FlagCX and the profile's FlagCX view/direct mode
FlagCX absent or initialization fails
    -> use the profile's native backend and view
No suitable inner backend
    -> raise a diagnostic RuntimeError
```

Keep the logic table-driven in `ProcessGroupFlagOS._build_inner`:

```python
if self._try_build_flagcx(...):
    return self._resolve_view(profile, vendor, backend="flagcx")
if profile.native is not None and getattr(self, profile.native)(...):
    return self._resolve_view(profile, vendor, backend="native")
raise RuntimeError(...)
```

Requirements for automatic behavior:

1. `import torch_fl` must register `flagos` for `privateuseone` without requiring
   callers to monkeypatch every `torch.distributed` collective.
2. `init_process_group(backend="auto")` and `backend="flagos"` must select the
   torch-fl process group.
3. FlagCX import failure must be a controlled fallback, not process-group
   construction failure when a native backend is available.
4. A FlagCX creator signature mismatch must retry the supported alternate form;
   arbitrary runtime errors must emit a warning and then use the fallback.
5. An absent tensor view must raise a clear `NotImplementedError` for a native
   fallback, not silently hand raw PrivateUse1 memory to HCCL/NCCL/MCCL.
6. DDP must use the torch-fl Python reducer and its all-reduce hooks when the
   model is on `privateuseone`.
7. Device guards must make the communicator's device current before a collective
   when the vendor's streams or pointers are device-scoped.

Do not add a second hardware-detection scheme in the distributed layer. Reuse
`GEMS_VENDOR` selected by `torch_fl` and preserve an explicit
`FLAGCX_TORCH_BACKEND` choice. For Enflame's torch-fl-compatible build, default
`FLAGCX_TORCH_BACKEND=flagos` before importing FlagCX; never overwrite an
explicit user value.

## Step 5 — test the automation without hardware first

The first automated gate is pure Python and must run on every platform:

```bash
pytest tests/unit/test_vendor_routing.py -q
```

These tests should cover at least:

- every profile has a coherent device/view/fallback combination;
- known CUDA-ABI and native vendors resolve without warnings;
- unknown vendors warn and use the documented default;
- FlagCX is attempted before the native fallback;
- both extended and plain creator signatures are exercised;
- hard creator failures warn and fall back;
- no-backend errors are diagnostic;
- missing view helpers fail loudly;
- direct PrivateUse1 mode is opt-in;
- all base collective virtuals (`_allgather_base`, `_reduce_scatter_base`) remain
  overridden and apply the selected view;
- vendor-specific environment defaults preserve explicit overrides.

Use fakes for FlagCX and native builders. Do not make unit tests import a vendor
runtime, allocate a device, or depend on a particular FlagCX wheel. The fake
creator must reject the wrong signature exactly as the real pybind binding does,
so a test that only accepts `*args` is insufficient.

Then run the static/public registration checks:

```bash
pytest tests/unit/test_vendor_routing.py tests/integration/ops/test_flaggems_conf_consistency.py -q
python -m compileall -q torch_fl tests
```

A pure-Python pass proves selection and fallback automation only. It does not
prove a communicator can exchange data.

## Step 6 — run the two-rank live collective harness

Use the repository harness or an equivalent `torchrun` command. Keep the
interpreter, library path, visible devices, profiler settings, and rendezvous
variables explicit. On the measured Kunlun XPU-RT stack, do not set
`XPU_CUPTI_ENABLE_DEVICE` for a multi-process run: selecting devices for CUPTI
event sampling can make `etiEventSamplingSetMode` fail with error 17 before
ProcessGroup construction. Leave it unset unless the vendor profiler setup has
been independently validated.

```bash
unset XPU_CUPTI_ENABLE_DEVICE
XPU_ENABLE_PROFILER_TRACING=1 \
MASTER_ADDR=127.0.0.1 MASTER_PORT=29531 \
  torchrun --standalone --nproc-per-node=2 \
  tests/manual/test_flagos_dist_live.py --world-size 2 --require-flagcx
```

The checked-in harness also supports `--force-native` for an independent
fallback run. Do not replace `--require-flagcx` with a successful fallback run.
```bash
unset XPU_CUPTI_ENABLE_DEVICE
XPU_ENABLE_PROFILER_TRACING=1 \
  tests/manual/test_flagos_dist_live.py --world-size 2 --force-native
```

The minimum live matrix is:

| Check | Required assertion |
|---|---|
| process-group initialization | both ranks construct `ProcessGroupFlagOS` |
| FlagCX path | every rank proves `inner_backend == "flagcx"`; a native fallback is fatal |
| native fallback | a separate run disables or rejects FlagCX and proves the expected native builder |
| `all_reduce` | every rank receives the CPU-expected sum |
| `broadcast` | every rank receives rank-0 data |
| `all_gather` | list output has one expected value per rank |
| `all_gather_into_tensor` | `_allgather_base` path returns ordered values |
| `reduce_scatter_tensor` | each rank receives its expected shard |
| barrier | all ranks complete without a hang |
| DDP | forward/backward completes and gradients agree across ranks |
| device guard | deliberately select a different current device before a collective and still pass, when the vendor requires it |

A harness that prints `OK` without asserting exit status, tests only one rank,
or merely observes that a collective returned is not evidence. The strict
FlagCX harness must:

- run with `world_size >= 2` and a finite timeout;
- assert the selected inner backend on every rank before the first collective;
- make native fallback selection raise a sentinel error;
- compare every result with a CPU-computed expected value, including rank order;
- synchronize at the end and emit one all-ranks success marker;
- propagate any child exception, timeout, assertion, or nonzero exit status.

Capture stdout/stderr, the selected backend proof from every rank, device model,
FlagCX revision, creator ABI, and environment in the evidence file. A result
without the backend proof is **not validated**, even if all numerical checks pass.

For a new vendor adaptor, run two mechanically separate matrices:

1. **Strict FlagCX run:** FlagCX must import, construct, be selected on every
   rank, and complete the matrix. Any fallback is a failure.
2. **Fallback run:** force `_try_build_flagcx` to return false or hide the package,
   assert the expected native builder was selected, and complete the matrix if a
   native backend is supported.

If FlagCX is unavailable on the target, report the FlagCX path as **not
validated** and still run the fallback path if the vendor provides one. Never
substitute a successful NCCL/HCCL/MCCL run for FlagCX evidence.

## Step 7 — extend beyond basic collectives

After the two-rank smoke passes, test the communication operations used by the
actual workloads:

- list and tensor forms of all-gather and reduce-scatter;
- all-to-all and all-to-all-single;
- gather/scatter with roots;
- send/recv/isend/irecv;
- barrier with device IDs;
- functional collectives used by DTensor, `torch.compile`, FSDP2, or ZeRO;
- DDP gradient synchronization and optimizer step;
- FSDP2 parameter all-gather, gradient reduce-scatter, and sharded state dict;
- process failure, timeout, and repeated init/destroy where production requires.

Test both contiguous and non-contiguous tensors, multiple dtypes supported by
the adaptor, nonzero device indices, and repeated operations on the same stream.
For CUDA-compatible vendors, compare the physical pointer and stream behavior of
the view. For direct native adaptors, test that the adaptor receives the expected
device type and current device.

Do not claim multi-node, failure recovery, point-to-point, root collectives, or
FSDP2 from a basic two-rank all-reduce. Keep those as separate evidence rows.

## Step 8 — record evidence and review the boundary

For each claimed vendor, record:

- torch-fl revision and branch;
- FlagCX revision, adaptor, and build options;
- PyTorch version and creator ABI;
- vendor runtime/driver/SDK and device model;
- visible device count and rank mapping;
- selected `GEMS_VENDOR`, `FLAGCX_*`, and library paths;
- whether FlagCX or the native fallback was selected;
- exact launcher and command;
- all live test output, exit status, and timeout policy;
- operations, dtypes, layouts, world sizes, and device indices exercised.

Keep FlagCX evidence separate from native fallback evidence. A vendor row may say:

```text
FlagCX: validated for two-rank all_reduce/broadcast/all_gather and DDP
Native fallback: validated separately for the same basic matrix
FSDP2 / P2P / multi-node / failure recovery: not validated
```

If FlagCX is missing, say **FlagCX not validated; native fallback tested** or
**distributed communication not validated**. Do not convert an import failure
into a zero, and do not infer support from `_VENDOR_PROFILES` or a registered
backend name.

Update `docs/architecture/distributed-flagcx.md` and the platform compatibility
or installation document in the same change when the supported scope changes.
Do not copy vendor runtimes, drivers, XMLIR, or Python environments into a wheel.

## Step 9 — pre-PR automation gate

Before opening a PR:

```bash
git fetch flagos main
git rebase flagos/main
ruff check .
ruff format --check .
pytest tests/unit/test_vendor_routing.py -q
pytest tests/unit/test_vendor_routing.py tests/integration/ops/test_flaggems_conf_consistency.py -q
```

Then verify:

- the second run of every generator or environment setup is idempotent;
- the loaded torch-fl extension and FlagCX package are the intended artifacts;
- unit tests exercise both creator signatures and fallback paths;
- the strict live harness proves FlagCX was selected on every rank and fails if fallback occurs;
- the fallback harness separately proves the expected native builder;
- the live harness has a nonzero failure path and asserts every result;
- FlagCX and native fallback evidence are not mixed;
- unsupported capabilities are explicitly marked **not validated**;
- the PR includes exact test output, environment versions, and human reviewer;
- all GitHub-facing text and documentation are in English.

## Common failure modes

| Symptom | Likely cause | Correct response |
|---|---|---|
| `No module named flagcx` | FlagCX is not installed in the active interpreter | Fetch the matching source from `https://github.com/FlagOpen/FlagCX.git`, build the vendor adaptor and torch plugin, then record the exact SDK/build blocker if the source build cannot complete |
| `createFlagcxBackend` missing | Old or incomplete FlagCX build | Inspect the build/revision; treat the path as unavailable and fall back |
| `incompatible function arguments` | Plain adaptor called with extended creator or vice versa | Retry the alternate supported signature; add a fake-signature unit test |
| Process group initializes but collectives fail | Wrong adaptor device name, tensor view, stream, or current device | Inspect profile, pointer/view, stream, and device guard; rerun a two-rank harness |
| `No backend type associated with device type flagos` | PrivateUse1 backend registration or base virtual override missing | Check `register_flagos_backend()` and `_allgather_base`/`_reduce_scatter_base` overrides |
| HCCL/NCCL receives raw PrivateUse1 tensors | Missing or unsafe view conversion | Fail loudly or use a measured native FlagCX adaptor; never reinterpret blindly |
| FlagCX silently becomes NCCL/HCCL | Import/creator failure swallowed as success | Emit selected-backend evidence and test FlagCX available and unavailable separately |
| DDP hangs or gradients differ | Reducer hooks or stream ordering bypass ProcessGroupFlagOS | Run DDP with rank-wise gradient comparison and inspect device guards |
| Tests pass only with one rank | Harness does not exercise communication | Require `world_size >= 2`, expected values, and rank synchronization |

## Related

[[runtime-bringup]] · [[torch-version-port]] · [[cuda-compat-vendor]] ·
[[native-op-backend]] · [[pre-pr-checks]]
