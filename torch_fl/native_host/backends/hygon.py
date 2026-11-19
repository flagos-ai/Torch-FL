# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""HCU launch mechanics for the published DTK FlagTree backend."""

import os
from pathlib import Path


def _published_codec(source):
    """Reuse the pinned driver's descriptor layout and encoder unchanged."""
    start = source.index("typedef struct {")
    end = source.index("} TDMDescriptor;", start) + len("} TDMDescriptor;")
    function = source.index("static bool encodeTDMDescriptor(")
    body = source.index("{", function)
    depth = 1
    stop = body + 1
    while depth:
        depth += (source[stop] == "{") - (source[stop] == "}")
        stop += 1
    return source[start:end] + "\n" + source[function:stop] + "\n"


def _substitute(source, marker, value):
    if source.count(marker) != 1:
        raise RuntimeError("unrecognized published HCU utility source: " + marker)
    return source.replace(marker, value, 1)


class Backend:
    vendor = "hygon"
    compiler_backend = "hip"
    compiler_distribution = "flagtree"
    driver_directory = "hcu"
    driver_files = ("driver.py", "driver.c", "compiler.py", "compiler_hcu.py")
    driver_symbols = ("HIPDriver", "HIPUtils", "HIPLauncher")
    utility_module = "hip_utils"
    utility_exports = ("PyTDMDescriptor",)
    # Software/TDM descriptors, profile scratch, cooperative/cluster launch,
    # stream-capture query. The published HCU ABI always passes null global scratch.
    capabilities = 187
    scratch_alignment = 256

    def extension_options(self, driver, source_dir):
        sdk = Path(os.environ.get("ROCM_PATH") or "/opt/dtk")
        runtime = sdk / "lib/libgalaxyhip.so"
        if not runtime.is_file():
            raise RuntimeError(
                "HCU native hosts require DTK libgalaxyhip.so: " + str(runtime)
            )
        source = (driver / "driver.c").read_text()
        generated = source_dir / "hcu"
        generated.mkdir(parents=True, exist_ok=True)
        (generated / "hcu_descriptor.h").write_text(_published_codec(source))
        source = _substitute(source, "/*py_libhip_search_path*/", str(runtime))
        source = _substitute(
            source,
            "/*py_hip_get_device_properties_sym*/ NULL",
            '"hipGetDeviceProperties_v2"',
        )
        utility = generated / "driver.c"
        utility.write_text(source)
        include = str(driver / "include")
        return {
            "depends": [str(generated / "hcu_descriptor.h")],
            "include_dirs": [include, str(generated)],
            "system_libraries": ["dl"],
            "define_macros": [
                ("__HIP_PLATFORM_AMD__", "1"),
                ("FLAGOS_HCU_RUNTIME_LIBRARY", f'"{runtime}"'),
            ],
        }, {
            "sources": [str(utility)],
            "include_dirs": [include],
            "libraries": ["dl", "m"],
        }
