# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Paired vendor/native precision checks for shapes, strides and stream semantics."""

import argparse
import ctypes
import json
from pathlib import Path

import torch

from logits_validation import math_settings


class NativeNonblockingStream:
    """Exercise the public FlagOS stream C API without default-stream ordering."""

    def __init__(self):
        import torch_fl

        self.library = ctypes.CDLL(
            str(Path(torch_fl.__file__).parent / "lib/libflagos.so")
        )
        self.library.StreamCreateWithPriority.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_uint,
            ctypes.c_int,
        ]
        self.library.StreamCreateWithPriority.restype = ctypes.c_int
        self.library.GetCurrentStreamForDevice.argtypes = [ctypes.c_int]
        self.library.GetCurrentStreamForDevice.restype = ctypes.c_void_p
        self.library.SetCurrentStreamForDevice.argtypes = [
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self.library.SetCurrentStreamForDevice.restype = ctypes.c_int
        for name in ("StreamSynchronize", "StreamDestroy"):
            function = getattr(self.library, name)
            function.argtypes = [ctypes.c_void_p]
            function.restype = ctypes.c_int
        handle = ctypes.c_void_p()
        assert self.library.StreamCreateWithPriority(ctypes.byref(handle), 1, 0) == 0
        self.stream_id = handle.value

    def __enter__(self):
        self.previous = self.library.GetCurrentStreamForDevice(0)
        assert self.library.SetCurrentStreamForDevice(0, self.stream_id) == 0
        return self

    def __exit__(self, exc_type, exc, traceback):
        status = self.library.StreamSynchronize(self.stream_id)
        assert self.library.SetCurrentStreamForDevice(0, self.previous) == 0
        assert self.library.StreamDestroy(self.stream_id) == 0
        if exc_type is None:
            assert status == 0


def cases(device):
    generator = torch.Generator().manual_seed(3817)

    def data(shape, dtype=torch.float32):
        return (torch.rand(shape, generator=generator) + 0.25).to(dtype).to(device)

    def mean(tensor, dims, keepdim, dtype):
        return torch.ops.aten.mean.dim(tensor, dims, keepdim, dtype=dtype)

    def matrix_with_reduction(a, b, enabled):
        previous = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = enabled
        try:
            return torch.mm(a, b) if a.ndim == 2 else torch.bmm(a, b)
        finally:
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = previous

    ops = torch.ops.aten
    result = []
    for name, tensor in (
        ("vector", data((129,))),
        ("transposed", data((7, 13)).t()),
        ("strided", data((5, 22))[:, ::2]),
        ("scalar", data(())),
        ("empty", data((0, 13))),
    ):
        result.append((f"rsqrt/{name}", ops.rsqrt.default, (tensor,)))
        for exponent in (0, 1, 2, 0.5, -1, 2.5):
            result.append(
                (f"pow/{name}/{exponent}", ops.pow.Tensor_Scalar, (tensor, exponent))
            )
    special = torch.tensor([0.0, -1.0, float("inf"), float("nan")], device=device)
    result.append(("rsqrt/special", ops.rsqrt.default, (special,)))
    for name, tensor, dim in (
        ("last", data((3, 129)), -1),
        ("first", data((7, 13)), 0),
        ("strided", data((11, 17)).t(), -1),
        ("empty", data((0, 13)), -1),
        ("scalar", data(()), 0),
    ):
        result.append((f"softmax/{name}", ops._softmax.default, (tensor, dim, False)))
    for name, tensor, dims in (
        ("last", data((3, 128)), [-1]),
        ("axes", data((4, 7, 13)), [0, 2]),
        ("all", data((7, 13)).t(), None),
        ("empty", data((0, 13)), [0]),
        ("empty_output", data((0, 13)), [1]),
        ("scalar", data(()), []),
    ):
        for keepdim in (False, True):
            for dtype in (None, torch.float64):
                result.append(
                    (
                        f"mean/{name}/{keepdim}/{dtype}",
                        mean,
                        (tensor, dims, keepdim, dtype),
                    )
                )
    bf16 = torch.bfloat16
    for rank in (2, 3):
        op = ops.mm.default if rank == 2 else ops.bmm.default
        for batch in [None] if rank == 2 else [0, 1, 3]:
            prefix = () if batch is None else (batch,)
            for m, k, n in (
                (5, 17, 7),
                (1, 13, 7),
                (7, 13, 1),
                (3, 1, 4),
                (0, 5, 7),
                (3, 0, 7),
            ):
                for layout in (
                    "contiguous",
                    "transpose",
                    "padded",
                    "strided",
                    "expanded",
                ):
                    if layout == "transpose":
                        a, b = (
                            data(prefix + (k, m), bf16).transpose(-1, -2),
                            data(prefix + (n, k), bf16).transpose(-1, -2),
                        )
                    elif layout == "padded":
                        a, b = (
                            data(prefix + (m, k + 3), bf16)[..., :k],
                            data(prefix + (k, n + 3), bf16)[..., :n],
                        )
                    elif layout == "strided":
                        a, b = (
                            data(prefix + (m, 2 * k), bf16)[..., ::2],
                            data(prefix + (k, 2 * n), bf16)[..., ::2],
                        )
                    elif layout == "expanded":
                        a = data(prefix + (1, k), bf16).expand(prefix + (m, k))
                        b = data(prefix + (k, n), bf16)
                    else:
                        a, b = data(prefix + (m, k), bf16), data(prefix + (k, n), bf16)
                    result.append(
                        (f"matrix/{rank}/{batch}/{m},{k},{n}/{layout}", op, (a, b))
                    )
    # Interleaved attention heads, with broadcast batches on the right.
    result.append(
        (
            "bmm/interleaved",
            ops.bmm.default,
            (
                data((9, 4, 13), bf16).transpose(0, 1),
                data((1, 13, 7), bf16).expand(4, 13, 7),
            ),
        )
    )
    for index, enabled in enumerate((False, True, False, True)):
        for batch in (None, 1, 3):
            prefix = () if batch is None else (batch,)
            result.append(
                (
                    f"blas_math/{index}/{batch}",
                    matrix_with_reduction,
                    (
                        data(prefix + (5, 3072), bf16),
                        data(prefix + (3072, 31), bf16),
                        enabled,
                    ),
                )
            )
    return result


def run(device, stream):
    outputs = {}
    module = torch.get_device_module(device)
    with stream, torch.no_grad():
        assert (
            torch.accelerator.current_stream(torch.device(device)).stream_id
            == stream.stream_id
        )
        for name, function, args in cases(device):
            tensors = [value for value in args if isinstance(value, torch.Tensor)]
            identity = [
                (t.data_ptr(), t.device, str(torch._C._dispatch_keys(t)), t._version)
                for t in tensors
            ]
            try:
                output = function(*args)
            except Exception as error:
                raise RuntimeError(f"native vendor case failed: {name}") from error
            assert output.device == torch.device(device), name
            assert identity == [
                (t.data_ptr(), t.device, str(torch._C._dispatch_keys(t)), t._version)
                for t in tensors
            ], name
            outputs[name] = {"tensor": output.cpu(), "stride": list(output.stride())}
        if isinstance(stream, NativeNonblockingStream):
            # Queue a producer backlog so an accidental launch on CUDA's default
            # stream cannot pass by relying on legacy stream synchronization.
            value = torch.empty(8 * 1024 * 1024, dtype=torch.float32, device=device)
            for _ in range(32):
                value.fill_(1)
            value.fill_(4)
            assert torch.all(torch.rsqrt(value).cpu() == 0.5)
    module.synchronize()
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cuda:0", "flagos:0"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    if args.device.startswith("flagos"):
        import torch_fl

        assert torch_fl._build_config.NATIVE_VENDOR
        from torch_fl.native_host._generated import _vendor

        name, abi, address, _ = _vendor.bindings()[0]
        register = torch.ops.flagos.native_vendor_register
        register("nvidia", name, abi, address)  # Idempotent activation is allowed.
        for values in (
            ("unknown", name, abi, address),
            ("nvidia", "missing", abi, address),
            ("nvidia", name, "wrong ABI", address),
            ("nvidia", name, abi, 0),
            ("nvidia", name, abi, address + 1),
        ):
            try:
                register(*values)
            except RuntimeError:
                continue
            raise AssertionError(
                f"native vendor registry accepted invalid binding: {values[:2]}"
            )
    reference = (
        torch.load(args.reference, weights_only=True) if args.reference else None
    )
    if reference:
        assert reference["producer"] == "vendor"
        assert reference["math_settings"] == math_settings()
        assert reference["torch_git_version"] == torch.version.git_version
    reports = []
    first = None
    streams = (
        torch.accelerator.current_stream(torch.device(args.device)),
        torch.Stream(device=args.device),
    )
    if args.device.startswith("flagos"):
        streams += (NativeNonblockingStream(),)
    assert streams[0].stream_id != streams[1].stream_id
    for stream in streams:
        actual = run(args.device, stream)
        expected = reference["outputs"] if reference else first
        if expected:
            assert actual.keys() == expected.keys()
            for name, value in actual.items():
                torch.testing.assert_close(
                    value["tensor"],
                    expected[name]["tensor"],
                    atol=0,
                    rtol=0,
                    equal_nan=True,
                    msg=name,
                )
                assert value["stride"] == expected[name]["stride"], name
        first = actual
        reports.append(
            {"cases": len(actual), "stream": int(stream.stream_id), "passed": True}
        )
    report = {
        "torch": str(torch.__version__),
        "torch_git_version": torch.version.git_version,
        "producer": "vendor" if args.device.startswith("cuda") else "torch_fl",
        "math_settings": math_settings(),
        "streams": reports,
    }
    if args.device.startswith("flagos"):
        from torch_fl.native_host import vendor

        before = {name: hits for name, _, _, hits in _vendor.bindings()}
        with torch.no_grad():
            value = torch.ones((4, 4), dtype=torch.float16, device=args.device)
            for result in (value.pow(2), value.mean(dim=-1), value.rsqrt()):
                assert torch.equal(result.cpu(), torch.ones_like(result.cpu()))
            assert torch.equal(
                value.softmax(-1).cpu(), torch.full((4, 4), 0.25, dtype=torch.float16)
            )
            matrix = value.float()
            assert torch.equal((matrix @ matrix).cpu(), torch.full((4, 4), 4.0))
            assert torch.equal(
                torch.bmm(matrix.unsqueeze(0), matrix.unsqueeze(0)).cpu(),
                torch.full((1, 4, 4), 4.0),
            )
        assert before == {name: hits for name, _, _, hits in _vendor.bindings()}
        report["unmeasured_dtypes_stay_flaggems"] = True
        report["vendor_audit"] = vendor.audit()
        assert all(report["vendor_audit"]["hits"].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({**report, "outputs": first}, args.output.with_suffix(".pt"))
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(reports), flush=True)


if __name__ == "__main__":
    main()
