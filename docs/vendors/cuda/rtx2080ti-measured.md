# torch-fl on RTX 2080 Ti — measured run record

> Host: 4x RTX 2080 Ti (22 GB each), driver 580.178.04, Ubuntu 22.04.5, glibc 2.35
> Repository state: `flagos-ai/Torch-FL` @ `2308f09` (upstream/main)
> Build: default CUDA wheel shape — FlagGems-primary, boxing OFF,
> `FLAGOS_BUILD_FLAGGEMS_CPP=0`, torch 2.10.0+cpu, FlagTree 0.7.0 (Triton 3.6.0)

## Verdict

torch_fl builds and runs on a 2080 Ti. Compute, inference, and training are
correct. One real defect was found and it is **not 2080 Ti specific**.

## Environment workaround required (glibc 2.35 host)

The NVIDIA FlagTree wheel references `GLIBC_2.38` (`__isoc23_strto*`) and
`GLIBC_2.36` (`arc4random`). The CI lane requires glibc >= 2.38, so a 22.04 host
cannot load `libtriton.so` as shipped. Two pieces make it load:

1. `libglibc_shim.so` — defines `__isoc23_strtol/strtoll/strtoull` and
   `arc4random/arc4random_uniform`, stamped `@GLIBC_2.34` via `.symver` plus a
   version script. Loaded with `LD_PRELOAD`.
2. `fix_verneed.py` — rewrites the `.gnu.version_r` requirement records in each
   `triton/**/*.so` from `GLIBC_2.38`/`GLIBC_2.36` to `GLIBC_2.34`, including the
   `vna_hash` field (glibc matches on the hash, not the string).

Both are workarounds for the host, not for the hardware. A glibc >= 2.38 image
needs neither.

Python 3.12 (required by FlagTree's cp312 wheel) came from `uv`.

## Results

| Suite | Result |
|---|---|
| Core CUDA op group (`main_ops`, no flaggems marks) | 134 passed, 144 skipped, 0 failed |
| Full integration, 2080 Ti-sized training params | 1183 passed, 4 failed, 3 xfailed, 5 xpassed |
| Qwen3-0.6B inference | 4/4 passed |
| Qwen3-0.6B training (`--batch-size 1 --seq-len 512`) | 3/3 passed |
| Unit tests | 1797 passed, 18 failed |
| Ad-hoc op checks | `mm max_err` 1.3e-4 (512x512 fp32), `add` exact, softmax rows sum to 1.0 |

The 1183/4 integration row is `-m "not gcu"` with seven groups deselected
(94 tests). Without those deselects the same `-m "not gcu"` run is
**1215 passed, 16 failed**, and each of the 12 extra failures is a row in the
table below. All rows are from this host, so these are a 2080 Ti record and are
not comparable to the A100 CI figures.

## Defect found: `slice_backward` ignores zero-stride input

`flag_gems/ops/slice_backward.py` reads `grad_output` with linear offsets
(`tl.load(grad_output_ptr + offsets)`) and never checks that `grad_output` is
contiguous. A broadcast gradient — `scalar.expand(input_shape)`, exactly what
`SumBackward0` passes — has only one materialised element, so the kernel
reads out of bounds past it.

Reproduction:

```python
import torch, torch_fl
scan = torch.tensor(1.0, device="flagos").expand(2, 8)   # stride (0, 0)
torch.ops.aten.slice_backward(scan, [4, 8], 0, 0, 2, 1).cpu()
# flagos: [[1,0,0,0,0,0,0,0], [0,...]]   <- wrong
# cpu   : [[1,1,1,1,1,1,1,1], [1,...]]   <- correct
```

User-visible effect: every `.sum().backward()` over a slice produces a wrong
gradient.

```python
x = torch.arange(32.0).reshape(4, 8)
a = x.clone().to("flagos").requires_grad_(True)
a[0:2].sum().backward()
# a.grad[0] == [1, 22, 38, 54, 0, 0, 0, 0]   (22/38/54 are uninitialised memory)
# expected   == [1, 1, 1, 1, 1, 1, 1, 1]
```

Scope: the same implementation pattern appears in the kunlunxin, gcu300,
gcu400, cambricon, and tsingmicro backends with no contiguity check either. This
is a FlagGems-wide defect, not a 2080 Ti or CUDA one. Reported upstream as
flagos-ai/FlagGems#7396.

Fix that took `test_narrow_dispatch.py` from failing to 17/17 passing:

```python
logger.debug("GEMS SLICE_BACKWARD")
if not grad_output.is_contiguous():
    grad_output = grad_output.contiguous()
```

## Remaining failures — none attributable to sm_75

| Failure | Cause |
|---|---|
| `test_conv1d_dispatch.py` (3 fwd + 3 bwd) | `convolution_overrideable not implemented`. `backends_nvidia.conf` routes `convolution*` to `cuda`, and on a boxing-off build that key has no kernel, so the call reaches `cpu_fallback` and raises. Expected for this wheel shape, not a 2080 Ti result |
| `*_dispatch_log_flaggems_python` (cat, embedding, embedding_dense_backward, nll_loss, sum) | assert the route resolves to `flagos_python`; this build routes to `flaggems` |
| `TestNewOnesGcuRoute` (4) | marked `@pytest.mark.gcu`; GCU-only |
| `test_where_dispatch.py::test_out_grows_from_empty` | FlagGems `where` rejects a 0-size `out` |
| `test_torch_backends_entry_point_is_registered` | passes after an editable install (metadata, not code) |
| `test_fallback_trace_train` / Qwen3 at default params | 22 GB is not enough at `batch=2, seq_len=1024` (peak ~18 GB + optimizer + dataset) |
| `test_profiler_privateuse1.py` (2) | expects cuBLAS `sgemm`; this build captures Triton `mm_kernel_general` |
| `test_flagcicd_*.py` (16) | CI tooling tests, independent of device |

## FlagGems arch specialization

`flag_gems/runtime/backend/_nvidia/__init__.py` (FlagGems 5.4.0) maps
`ARCH_MAP = {"9": "hopper", "8": "ampere"}`. 7.5 is not a key, so every 2080 Ti
run prints:

```
[INFO] : FlagGems Unsupported GPU arch 7.5 specialization
```

and falls back to the generic kernels. Correctness is unaffected — the whole
suite above ran on that fallback.

There is a `_nvidia/turing/` directory, so the map looks like an omission, but
it is a narrow one: in 5.4.0 that directory holds a single override
(`ops/_safe_softmax.py`), and `backends_nvidia.conf` routes `_safe_softmax` to
`cuda`, which on this boxing-off wheel reaches `cpu_fallback` rather than
FlagGems. Adding `"7": "turing"` would enable that one override and no more.
`ARCH=turing` is not a workaround either: the lookup derives its key with
`arch_string.split("_")[-1][0]`, i.e. `"t"`, which is absent from the map.

This is FlagGems' table, not torch-fl's, so nothing in this repository changes
it. Recorded here so the log line is not mistaken for a torch-fl problem.

## bf16

`torch.cuda.is_bf16_supported()` is False on sm_75, and
`FlagOSDeviceInterface.is_bf16_supported` reads exactly that on a CUDA-like
profile. Fine for the suites above — `test_qwen3_infer.py` uses fp16 and
`test_qwen3_train.py` uses fp32 — but any bf16-dependent path will report
unsupported on this hardware.