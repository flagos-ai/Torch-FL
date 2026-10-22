# Real streams for boxing-enabled DCU builds

## Scope and current status

This is an **opt-in boxing-build provider**, not a native HIP implementation for
FlagGems-primary/boxing-OFF builds. It requires a matching vendor torch HIP
runtime alongside the official torch core. It is intended to address the missing
DCU Stream/Event plumbing discussed in issue #472, not to certify all FSDP/HSDP
or third-party native-view use cases.

The provider is compiled only when both `FLAGOS_ACCELERATOR=dcu` and
`FLAGOS_BUILD_BOXING=1` are selected. Other builds retain their existing stream
implementation. Python selects the provider by the compiled `_dcu_stream_info`
capability, not by a runtime environment override. The default remains boxing
OFF; this change does not claim to add multistream support to that default.

This change must remain draft until the exact candidate's ON/OFF wheel builds
and device tests have completed. Local unit tests and syntax checks are not a
substitute for those gates. The latest source-built ON fixture passed all 13
device cases; a complete DCU wheel is still blocked by the separate
`_flagos_nccl` extension build.

## Execution and lifetime contract

```text
flagos Stream / Event and PrivateUse1 DeviceGuard
  -> vendor CUDA-tag DeviceGuard backed by HIP
  -> one vendor per-thread/per-device stream registry and stream pool
  -> actual hipStream_t for eager, Triton and Inductor launches
```

Opaque `c10::StreamId` values are not raw HIP pointers. Conversion to/from raw
handles uses the matching `c10::hip` interfaces in a dedicated translation unit.
Only that unit consumes HIP headers; it must not include the DTK CUDA facade
headers in the same translation unit. CMake stages the selected vendor's
`c10/hip` subtree and optional generated `ATen/dtk_macros.h` without replacing
official ATen/c10 core headers. The selected headers and vendor libraries must
match; arbitrary vendor releases are not ABI-certified by this change.

The generic PyTorch Stream/Event types retain the public flagos device tag.
Contexts restore both device and per-device stream state on normal exit or an
exception. External streams are borrowed: their owner must keep them alive.
Timing is delegated to vendor events; unsupported IPC/graph-external event
options fail explicitly.

Allocations retain the vendor DataPtr's context and deleter so
`Tensor.record_stream()` can reach the allocator's actual lifetime tracking.
Event/wait orders producer and consumer execution; it does not replace
`record_stream()` when an allocation is consumed on another stream.

Clone, blocking device copies and Storage growth use the selected current
streams. They **remain blocking operations**; this is not an asynchronous-copy
performance claim. Exact TensorImpl self-copy avoids a redundant transfer, but
ordinary Storage aliases still follow normal copy semantics.

## Build and acceptance

Use separate source/build directories and isolated installations for:

1. Unmodified upstream, boxing OFF.
2. This revision, boxing OFF, without vendor torch on the loader search path.
3. This revision, boxing ON, with matching vendor device libraries.

For ON, set `FLAGOS_ACCELERATOR=dcu`, `FLAGOS_BUILD_BOXING=1`, `ROCM_PATH` to
the DTK root and `FLAGOS_VENDOR_TORCH_LIB` to the matching vendor `torch/lib`.
A direct CMake build can explicitly set `FLAGOS_DCU_HIP_INCLUDE_DIR` to that
vendor torch's include directory. OFF must not need that directory. Follow the
existing DCU installation instructions for the official core, runtime loader
and compiler dependencies. Do not replace official `libtorch_python.so` with a
vendor copy to make an import pass.

CPU regressions:

```bash
python -m pytest -q tests/unit/test_dcu_stream_contract.py \
  tests/unit/test_dcu_boxing_build_gate.py tests/unit/test_dcu_cuda_shim.py
ruff check
ruff format --check
```

Device regression (at least two DCUs, both boxing and mixed runtime routes):

```bash
python -c 'import torch_fl; assert hasattr(torch_fl._C, "_dcu_stream_info")'
python -c 'import torch_fl,pytest,sys; sys.exit(pytest.main(sys.argv[1:]))' \
  -q tests/integration/runtime/test_dcu_boxing_streams.py
```

Run against the installed wheel, not an unbuilt source package. Require zero
skips in this targeted ON gate. It checks independent pending queues, event
ordering, TLS, exception restoration, borrowed streams, allocator reuse,
current-stream clone, native derived-view device tags, compile launch,
self-copy and Storage resize. A process timeout is required because a broken
stream/query implementation can block on the deliberately held queue.

For OFF, inspect `compile_commands.json` and the dynamic symbol table: the new
provider must be absent, and the new Python `_dcu_*` methods must not be exported.
Then compare import and existing runtime tests with the unmodified OFF baseline.
Do not turn a failing baseline into a claim that the OFF runtime is validated.

## Not included

- No global shared-Storage bridge replacement or cross-tag differentiable API.
- No ProcessGroup/Work ownership rewrite and no HSDP certification.
- No new boxing-OFF allocator or stream provider.
- No profiler fixes, default-route changes, or Apex/TileOPs consumer migration.
- No throughput claim or guarantee that all runtime host waits are eliminated.

Native communication streams and any native borrowed Tensor views still need
separate end-to-end ownership and alias validation before using this change as
a complete FSDP/HSDP fix. Scope acceptance from maintainers is also outstanding.
