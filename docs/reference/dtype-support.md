# Dtype Support

FlagOS preserves the requested dtype for tensor storage and follows PyTorch's
promotion rules for tensor-tensor operations. Compute coverage is still bounded
by the vendor library used by each backend. A dtype appearing in a backend's
storage table does not imply that every operator accepts it.

## AMP

`torch.autocast("flagos")` supports `torch.float16` and `torch.bfloat16` as
lower-precision targets. The standard PyTorch autocast policy groups are used:
matmul and convolution prefer the selected lower-precision dtype, numerical
operations such as logarithm and normalization use float32, and mixed inputs
follow the promote policy. Float32 and float64 are not valid autocast targets.

AMP target support is separate from eager dtype support. For example, Ascend
can store and perform many elementwise operations on float64 even though float64
is not an AMP target and is not accepted by its native matmul API.

## Backend Matrix

| Backend | Storage and copy | Eager elementwise | Matmul family | AMP targets |
| --- | --- | --- | --- | --- |
| CUDA | Native PyTorch CUDA dtype support | Native CUDA coverage | Native CUDA coverage | float16, bfloat16 |
| Ascend | float16, bfloat16, float32, float64, integer, uint8, bool | Vendor coverage; unsupported ACLNN combinations use the CPU fallback | float16, bfloat16, float32 natively; float64 and unsupported types use the CPU fallback | float16, bfloat16 |
| MetaX boxing | MACA libtorch CUDA-compatible coverage | MACA libtorch CUDA-compatible coverage | MACA libtorch CUDA-compatible coverage | float16, bfloat16 |
| DCU | Vendor library coverage | Vendor library coverage | Vendor library coverage | Backend-specific |
| MUSA | float16, bfloat16, float32, float64, integer, bool | mudnn coverage; unrouted operations use the CPU fallback | float16, bfloat16, and float32 measured for AMP paths; broader support is vendor-dependent | float16, bfloat16 |
| Enflame GCU | float16, bfloat16, float32, float64, integer, bool | topsaten coverage; int64, float64, and unrouted operations use the CPU fallback | float16, bfloat16, float32 measured for AMP paths on S60 | float16, bfloat16 |

The GCU AMP row is measured on S60 with the installed TopsRider release: the full
AMP integration suite passes, including both autocast dtypes and the GradScaler
finite/overflow paths. topsaten has no float64 kernels, so float64 is stored
natively but computed through the CPU fallback.

The MetaX AMP entry is measured for the CUDA-boxing path on C550 with MACA
3.8.0; it does not cover the legacy handwritten MetaX kernel mode. The DCU entry
remains vendor-dependent. MUSA AMP coverage below is measured for the specific
S5000 and mudnn release rather than inferred from its route configuration.

## GCU Boundaries

GCU uses the shared `AutocastPrivateUse1` policy lists. The generated native
`_amp_foreach_non_finite_check_and_unscale_` routes call topsaten's AMP API for
contiguous, supported lists; unsupported layouts and dtypes use the correctness-
first CPU fallback and copy mutations back to the GCU tensors. On the measured
S60 host, both float16 and bfloat16 execute the lower-precision matmul, linear,
and convolution policies; logarithm, layer normalization, MSE loss, and the
default softmax policy return float32; an explicit softmax dtype is preserved;
mixed inputs use the promote policy. GradScaler finite scale growth, overflow
backoff, and optimizer-step skipping were measured on the same host.

Measured topsaten dtype boundaries behind these routes:

- No float64 kernels. Every op returns `NOT_SUPPORT` for an F64 operand
  (measured across add, abs, and reciprocal), so float64 is stored natively but
  computed on CPU. GradScaler depends on this: it derives the inverse scale via
  `scale.double().reciprocal().float()`.
- No int64 kernels, handled by the same fallback.
- `topsatenNeg` additionally rejects uint8 and bool. Those two cases are not yet
  routed to the fallback and still raise; they are outside the AMP contract and
  tracked separately.

`convolution_overrideable` uses the native `topsatenConvolution` and matches the
CPU reference to within 3.9e-6 across stride, padding, dilation, group, and bias
variants. `convolution_backward_overrideable` is a CPU fallback:
`topsatenConvolutionBackward` is exported by the installed `libtopsaten.so.3`
but returns `NOT_SUPPORT` for every input measured (fp32 and fp16, grouped and
ungrouped, padded and unpadded, with both the caller's `output_mask` and an
all-true mask). Both routes are required because `aten::convolution` dispatches
PrivateUse1 to `convolution_overrideable`, which has no composite fallback.

## MUSA Boundaries

MUSA AMP uses the shared `AutocastPrivateUse1` policy lists with the native
mudnn operator routes. On the measured MTT S5000 and mudnn v3300 setup, both
float16 and bfloat16 execute the lower-precision matmul, linear, and convolution
policies. Logarithm, layer normalization, MSE loss, and the default softmax
policy return float32; an explicit softmax dtype is preserved; mixed inputs use
the promote policy.

`torch.amp.GradScaler("flagos")` uses the existing PrivateUse1 fallback for
`_amp_foreach_non_finite_check_and_unscale_`. That fallback moves the list and
scalar operands to CPU, runs the reference kernel, and copies the mutated values
and `found_inf` flag back to MUSA. This is correctness-oriented rather than a
claim of native mudnn AMP-foreach acceleration. Finite scale growth, overflow
backoff, optimizer-step skipping, and FP16/BF16 autocast training were measured
on the same S5000 host.

## Ascend Boundaries

Ascend CANN 9.0 exposes the ACL dtype enums for float64 and all of the common
integer types, but individual ACLNN operators have narrower contracts:

- `aclnnMatmul`, `aclnnMm`, and `aclnnBatchMatMul` reject float64 and integer
  inputs. FlagOS uses the existing CPU fallback and copies the correctly typed
  result back to the Ascend device.
- `aclnnNeg` rejects int16, uint8, and bool. FlagOS uses the CPU reference for
  those inputs, preserving PyTorch's integer wraparound behavior and bool error.
- Ascend float64 storage, device copies, casts, and elementwise operations remain
  float64. The implementation must not clamp float64 to float32 during `_to_copy`.
- Complex and quantized dtypes are not currently mapped by the Ascend ACL tensor
  wrapper and are outside the supported contract.

The fallback is correctness-oriented and may be slower than a native kernel.
New Ascend operator routes must be added through `scripts/codegen_ascend.py` and
regenerated; generated output is not an independent source of truth.

## Testing Contract

The integration tests cover factory, unary, binary, reduction, indexing,
comparison, copy, promotion, AMP, and GradScaler behavior. Dtype tests compare
results with CPU references where a native Ascend kernel is unavailable. A
passing test means the operation follows the documented PyTorch contract, not
that the operation necessarily uses a native vendor kernel.
