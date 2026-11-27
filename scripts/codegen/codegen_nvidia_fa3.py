# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Adapt the published Hopper FA3 provider to the Torch-FL device and stream."""

import argparse
from hashlib import sha256
from pathlib import Path


_SOURCE_DIGESTS = {
    "flash_api_stable.cpp": (
        "9755d46d7090dd851cbbc2b24494300348b7e93349dec9bab63de7f4fbd8b645"
    ),
    "flash_attn_interface.py": (
        "4af7735ad8bcb7f9819be9e71c66ebab2001f594e98106a5e78f453de5956cad"
    ),
    "__init__.py": "da997374411bdb814b8c4d88d97c8105c2443d858771c34be1282d2c2ffa7ab8",
}
_DISABLED = ("BACKWARD", "SPLIT", "PAGEDKV", "APPENDKV", "SOFTCAP", "FP8", "SM8x")

_CHECK_FORWARD = """\
    if any(x is not None for x in (k_new, v_new, out_, scheduler_metadata)):
        raise ValueError("FlagOS FA3 supports read-only forward without an output buffer or external scheduler metadata")
    stream = torch.accelerator.current_stream(q.device.index)
    for tensor in (
        q, k, v, qv, cu_seqlens_q, cu_seqlens_k, cu_seqlens_k_new,
        seqused_q, seqused_k, page_table, kv_batch_idx, leftpad_k,
        rotary_cos, rotary_sin, seqlens_rotary, q_descale, k_descale, v_descale,
    ):
        if tensor is None:
            continue
        if tensor.device != q.device:
            raise ValueError("FlagOS FA3 expects all tensors on the query device")
        tensor.record_stream(stream)
"""


def _read(source: Path, name: str) -> str:
    data = (source / name).read_bytes()
    if sha256(data).hexdigest() != _SOURCE_DIGESTS[name]:
        raise ValueError(f"{name} must match the published FA3 v1.0.3 sources")
    return data.decode()


def _write(path: Path, content: str) -> None:
    if not path.exists() or path.read_text() != content:
        path.write_text(content)


def generate(source_root: Path, output: Path) -> None:
    source = source_root / "src" / "flash_attn_3_nv"
    package = output / "flash_attn_3_nv"
    package.mkdir(parents=True, exist_ok=True)
    host = _read(source, "flash_api_stable.cpp")
    start = host.index("// Declare the CUDA stream function")
    end = host.index("#include <torch/headeronly/core/ScalarType.h>", start)
    host = host[:start] + "#include <flagos.h>\n\n" + host[end:]
    start = host.index('extern "C" {')
    end = host.index("#define CHECK_DEVICE", start)
    host = host[:start] + host[end:]
    host = host.replace("#include <Python.h>\n", "")
    host = host.replace(
        'x.is_cuda(), #x " must be on CUDA"',
        "x.device().type() == torch::headeronly::DeviceType::PrivateUse1,"
        ' #x " must be on FlagOS"',
    )
    host = host.replace(
        "void* stream_ptr = nullptr;\n"
        "        TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(device_idx, &stream_ptr));",
        "void* stream_ptr = static_cast<void*>(GetCurrentStreamForDevice(device_idx));",
    )
    host = host.replace(
        "STABLE_TORCH_LIBRARY(flash_attn_3, m)", "STABLE_TORCH_LIBRARY(torch_fl_fa3, m)"
    )
    host = host.replace(
        "STABLE_TORCH_LIBRARY_IMPL(flash_attn_3, CUDA, m)",
        "STABLE_TORCH_LIBRARY_IMPL(torch_fl_fa3, PrivateUse1, m)",
    )
    _write(output / "flash_api_flagos.cpp", host)

    interface = _read(source, "flash_attn_interface.py")
    start = interface.index("# isort: off")
    end = interface.index("flash_attn_3_cuda =", start)
    interface = interface[:start] + interface[end:]
    interface = interface.replace("torch.ops.flash_attn_3", "torch.ops.torch_fl_fa3")
    # Keep the public custom-op schemas so existing compiled graphs resolve the
    # same forward op. Only its raw kernel provider uses the private namespace.
    interface = interface.replace('device_types="cuda"', 'device_types="privateuseone"')
    forward_body = interface.index("    q, k, k_new, v_new =")
    interface = interface[:forward_body] + _CHECK_FORWARD + interface[forward_body:]
    _write(package / "flash_attn_interface.py", interface)
    _write(package / "__init__.py", _read(source, "__init__.py"))
    config = {
        f"FLASHATTENTION_DISABLE_{flag}": flag in _DISABLED
        for flag in ("HDIM64", "HDIM96", "HDIM128", "HDIM192", "HDIM256", *_DISABLED)
    }
    _write(
        package / "flash_attn_config.py", f"CONFIG = {{'build_flags': {config!r}}}\n"
    )
    _write(output / "features.cmake", "set(FA3_DISABLED " + " ".join(_DISABLED) + ")\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    generate(args.source, args.output)
