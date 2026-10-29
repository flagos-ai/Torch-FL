# Probes: dynamic-dimension .hbm on the S600

Working notes from establishing whether `torch_fl` can drive the BPU with
eager operators, and what the board does and does not allow for dynamic
shapes. Everything here was run on the board (`s600`, RDK S600, HBRT 4.7.5,
UCP 3.13.6).

## What compiles

`hbdk4` accepts a dynamic axis through a tensor-type attribute, written the way
its own IR dump prints it:

```mlir
func.func @dyn_relu(%a: tensor<1x512xf32, {dynamic_dims = [-1]}>)
    -> tensor<1x512xf32, {dynamic_dims = [-1]}> {
  %0 = "hbir.relu"(%a) : (tensor<1x512xf32, {dynamic_dims = [-1]}>)
                     -> tensor<1x512xf32, {dynamic_dims = [-1]}>
  return %0 : tensor<1x512xf32, {dynamic_dims = [-1]}>
}
```

Notes from getting there:

* MLIR's own `?` dimension is rejected outright:
  `'func.func' op Tensor shape cannot contain zero, got 1, -9223372036854775808`.
* `dynamic_dims` cannot be injected through `RankedTensorType.get(shape, type)`
  — its third parameter is *encoding*, and a dict encoding prints as
  `array<i64: 1>`, which the verifier then rejects. The attribute has to be on
  the tensor type in IR text.
* Not every op accepts a dynamic axis. `hbir.relu` converts.
  `hbir.add` fails with `kernel native::Add config function call failure`,
  and `hbir.quantize` does not exist as a registered op.

`convert()` succeeds; `compile()` writes the `.hbm` and then throws
`ResourceExhausted: Cannot malloc bpu memory` from its own post-compile
read-back validation. The artifact is complete at that point, which matches the
known emulation caveat in `docs/vendors/bpu/integration.md`.

## What the artifact declares

The dynamic axis survives into the artifact but the capacity does not disappear
— it is baked in:

| compiled capacity | `alignedByteSize` reported at runtime |
| ---: | ---: |
| 64 | 512 |
| 128 | 768 |
| 512 | 2304 |

That is `capacity * 4 + 256`. The 256 is a per-tensor dynamic-info block, and
it matches `hbrt4DynamicInfoGetShapeOffset` / `GetStrideOffset` /
`GetMemspace` on `libhbrt4`. All three artifacts are 4712 bytes on disk, so the
capacity lives in the declarations, not in extra code.

The declared tensor is:

```
dims   = [1, -1]
stride = [0, 4]      <- the dynamic axis carries stride 0
abs    = capacity*4 + 256
```

## Where it stops

`hbDNNInferV2` will not accept `stride[0] == 0`:

```
Model dyn_relu 0th dimension of the 0th input stride should be at least 64,
but get 0. Model min stride: (64,4), given: (0,4)
```

Supplying a concrete stride (`L*4`, or padded to 64) moves the failure to the
next check:

```
Do not support padding data for dynamic input now
Model dyn_relu validate input[0] failed!
```

That message means `memSize != stride[0] * dim[0]`, so the buffer must be
exactly one row at the declared stride. When the input is sized that way the
input passes and the output then complains:

```
Model dyn_relu output index 0's sys mem size is not enough,
required: 2304, given: 2048
```

Sweeping stride and `alignedByteSize` combinations across
`{L*4, align64(L*4)} x {L*4, stride0, capacity*4+256, stride0+256, 2304}`
produced no accepted configuration. The blocker is structural: the compiled
artifact declares a dynamic stride of 0, while the task validator requires at
least 64 and simultaneously rejects any mismatch between buffer size and the
stride extent. There is no value a caller can pass that satisfies both.

## Where this leaves the approach

Dynamic shapes are supported by the *loader* (`hbrt4VariableGetTensorDynamicInfo`
parses them) but not usable end to end from a caller on HBRT 4.7.5. A dynamic
axis is therefore not a route to running arbitrary-shape operators from a
prebuilt artifact.

## Native C++ on the board

Independent of the above: the board has an aarch64 `g++ 13.3`, and
`/usr/hobot/lib/libhbtl.so` exports `hbtl::native::*` (353 symbols),
`hbtl::ucp::*` (28) and `hbtl::b30::intrinsic::*`. Those are callable directly
with no cross-compilation, but they are **CPU implementations**: `ldd` shows
only libc/libstdc++/libm, and a 1M-element `ucp::Add` runs at `cpu/wall = 1.00`.
`internal/B30/FpCmodel/` is a cycle model. They are the reference path hbdk4
itself uses for verification, not a hardware entry point.

`hbtl::native::Mul` needs byte-unit strides (`getStrides(sizes, byteSize)`);
element strides silently compute the first element only. The B30 elementwise
kernels take uppercase mode names (`FADD`, `FMUL`, `MAX`, `MIN`) and reject
`f32` as an element type.