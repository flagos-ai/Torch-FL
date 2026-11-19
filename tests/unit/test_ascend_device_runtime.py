# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Exercise the actual device runtime against an instrumented ACL boundary."""

from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def device_runtime(tmp_path_factory):
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ compiler is unavailable")
    directory = tmp_path_factory.mktemp("ascend-device")
    (directory / "acl").mkdir()
    (directory / "flagos.h").write_text(
        "#pragma once\n"
        "enum Error_t { Success, ErrorUnknown, ErrorInvalidDevice };\n"
        "Error_t GetDevice(int*); Error_t SetDevice(int);\n"
    )
    (directory / "acl/acl.h").write_text(
        "#pragma once\n#include <cstdint>\n"
        "using aclError = int;\n"
        "constexpr int ACL_SUCCESS=0, ACL_ERROR_REPEAT_INITIALIZE=1;\n"
        "aclError aclInit(const char*);\n"
        "aclError aclrtGetDeviceCount(uint32_t*);\n"
        "aclError aclrtSetDevice(int); aclError aclrtSynchronizeDevice();\n"
    )
    (directory / "acl/acl_rt.h").write_text('#include "acl.h"\n')
    source = directory / "probe.cc"
    source.write_text(
        r"""
#include <flagos.h>
#include <acl/acl.h>
#include <atomic>
#include <cassert>
#include <string>
#include <thread>
Error_t GetDeviceCount(int*);
std::atomic<int> init_calls{0}, count_calls{0}, set_calls{0};
int init_error=0, count_error=0, set_error=0;
thread_local int bound=-1;
aclError aclInit(const char*) { ++init_calls; return init_error; }
aclError aclrtGetDeviceCount(uint32_t* count) {
  ++count_calls; *count=2; return count_error;
}
aclError aclrtSetDevice(int device) {
  ++set_calls; if (!set_error) bound=device; return set_error;
}
aclError aclrtSynchronizeDevice() { return 0; }
int main(int argc, char** argv) {
  assert(argc==2);
  const std::string mode=argv[1];
  int device=-1;
  assert(GetDevice(nullptr)==ErrorUnknown);
  assert(GetDeviceCount(nullptr)==ErrorUnknown);
  if (mode=="init-error") {
    init_error=42;
    assert(GetDevice(&device)==ErrorUnknown && device==-1);
    int count=99;
    assert(GetDeviceCount(&count)==ErrorUnknown && count==0);
    assert(SetDevice(0)==ErrorUnknown);
    assert(count_calls==0 && set_calls==0);
    return 0;
  }
  if (mode=="repeat-init") init_error=ACL_ERROR_REPEAT_INITIALIZE;
  if (mode=="count-error") {
    count_error=42;
    assert(GetDevice(&device)==ErrorUnknown && device==-1);
    assert(set_calls==0);
    count_error=0;
  }
  if (mode=="set-error") {
    set_error=42;
    assert(GetDevice(&device)==ErrorUnknown && device==-1 && bound==-1);
    set_error=0;
  }
  assert(GetDevice(&device)==Success && device==0 && bound==0);
  const int counts=count_calls, sets=set_calls;
  for (int i=0; i<100; ++i) {
    assert(SetDevice(0)==Success);
    assert(GetDevice(&device)==Success && device==0);
  }
  assert(count_calls==counts && set_calls==sets);
  assert(SetDevice(-1)==ErrorInvalidDevice);
  assert(SetDevice(2)==ErrorInvalidDevice);
  assert(bound==0);
  assert(SetDevice(1)==Success && bound==1);
  assert(GetDevice(&device)==Success && device==1);
  set_error=42;
  assert(SetDevice(0)==ErrorUnknown);
  assert(GetDevice(&device)==Success && device==1 && bound==1);
  set_error=0;
  assert(SetDevice(0)==Success && bound==0);
  std::thread worker([] {
    int current=-1;
    assert(bound==-1);
    assert(GetDevice(&current)==Success && current==0 && bound==0);
    const int before=set_calls;
    assert(SetDevice(0)==Success && set_calls==before);
    assert(SetDevice(1)==Success && bound==1);
  });
  worker.join();
  assert(GetDevice(&device)==Success && device==0 && bound==0);
  assert(init_calls==1);
}
"""
    )
    executable = directory / "probe"
    result = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-pthread",
            "-Werror",
            "-I" + str(directory),
            str(ROOT / "csrc/runtime/accelerator/ascend/device.cc"),
            str(source),
            "-o",
            str(executable),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return executable


@pytest.mark.parametrize(
    "mode", ["normal", "repeat-init", "init-error", "count-error", "set-error"]
)
def test_device_context_binding_and_driver_failures(device_runtime, mode):
    result = subprocess.run(
        [str(device_runtime), mode], capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 0, result.stdout + result.stderr
