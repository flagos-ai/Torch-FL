## Summary

`torch.rsub(u, 1)` for a device tensor `u` faults with "an illegal memory access was encountered" and poisons the MUSA context for the rest of the process (#238).

`rsub.Scalar(self, other, alpha)` decomposes in ATen to `sub.Tensor(wrapped_scalar_tensor(other), self, alpha)` — the CPU-origin wrapped scalar lands in the `self` slot, and the real device tensor lands in `other`. Every binary mudnn kernel (`SubTensorKernelMusa`, `AddTensorKernelMusa`, `MulTensorKernelMusa`, ...) assumed `self` was always the device tensor: it allocated `out` from `self.options()` and moved only `other` onto `self.device()`. With `self` on CPU, that put a host pointer in `out` and handed it to mudnn as if it were a device pointer.

Fixed at the codegen template level (`_BINARY_PROLOGUE`, `T_BINARY`, `T_BINARY_ALPHA`, `T_BINARY_CMP` in `scripts/codegen_mudnn.py`), since the affected file is generated: the compute device is now picked from whichever operand isn't CPU, and both operands (plus `out`) are moved there. `.to()` is a no-op when a tensor already has the target device/dtype, so the ordinary `self`-is-device call pays no extra copy. The CPU dtype-fallback branch and the `EXEC_MUDNN_CMD` device guard tag were updated the same way, since they had the same `self.device()` assumption.

## Test

Verified on MTT S5000 (mudnn v3300):

- Regenerated `csrc/aten/backends/musa/generated/musa_kernels.cc` from the updated template; regenerating twice produces identical output.
- Reproduced the crash against the pre-fix binary (`torch.rsub(torch.ones(4, dtype=torch.long, device="flagos:0"), 1)` -> illegal memory access), then confirmed it returns the correct result after rebuilding with the fix.
- Added `TestMusaMixedDeviceOperandOrder` to `tests/integration/ops/test_musa_dispatch.py`, covering `rsub.Scalar`, `sub.Tensor` with a CPU `self`, and a sweep of `-`, `+`, `*`, `maximum`, `>`, `==` with a CPU-tensor `self` and device `other`.
- Full `tests/integration/ops/test_musa_dispatch.py`: 102 pass (was 94; +8 new).
- `tests/unit`: 264 pass, 2 pre-existing failures unrelated to this change (confirmed present on `main` before this fix).
- `ruff check` clean on the changed files.

## Note for reviewers

This is a template fix, not a source-level patch to the generated `.cc` — patching the file directly would be overwritten on the next `codegen_mudnn.py` run. The generated diff touches every `binary`/`binary_alpha`/`binary_cmp` kernel (`AddTensorKernelMusa`, `SubTensorKernelMusa`, `MulTensorKernelMusa`, `DivTensorKernelMusa`, `MaximumKernelMusa`, `MinimumKernelMusa`, `RemainderTensorKernelMusa`, `FmodTensorKernelMusa`, `PowTensorTensorKernelMusa`, `EqTensorKernelMusa`, `NeTensorKernelMusa`, `LtTensorKernelMusa`, `GtTensorKernelMusa`, `LeTensorKernelMusa`, `GeTensorKernelMusa`, `LogicalAndKernelMusa`, `LogicalOrKernelMusa`, `LogicalXorKernelMusa`, `FloorDivideKernelMusa`), all of which shared the same asymmetric-device assumption and were reachable through the same `wrapped_scalar_tensor` decomposition path for their respective r-ops (`radd`, `rmul`, etc. where applicable).

Generated with `claude-sonnet-5`.
