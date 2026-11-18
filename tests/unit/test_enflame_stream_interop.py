# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Exercise the real stream producer state machine against a tiny SDK stub."""

from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.parametrize("abi", [1, 2])
def test_barriers_only_at_tracked_producer_transitions(tmp_path, abi):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a C++ compiler")
    root = Path(__file__).resolve().parents[2]
    (tmp_path / "tops_runtime_api.h").write_text(
        """
#pragma once
using topsStream_t = void*;
using topsEvent_t = void*;
constexpr unsigned topsEventDisableTiming=2;
enum topsError_t { topsSuccess=0, topsErrorInvalidValue=1 };
extern int stream_syncs, device_syncs;
extern int records, waits, destroys;
extern bool fail_wait;
inline topsError_t topsDeviceSynchronize() { ++device_syncs; return topsSuccess; }
inline topsError_t topsStreamSynchronize(topsStream_t) { ++stream_syncs; return topsSuccess; }
inline topsError_t topsEventCreateWithFlags(topsEvent_t* event, unsigned) {
  *event=reinterpret_cast<void*>(64); return topsSuccess;
}
inline topsError_t topsEventRecord(topsEvent_t, topsStream_t) { ++records; return topsSuccess; }
inline topsError_t topsStreamWaitEvent(topsStream_t, topsEvent_t, unsigned) {
  ++waits; return fail_wait ? topsErrorInvalidValue : topsSuccess;
}
inline topsError_t topsEventDestroy(topsEvent_t) { ++destroys; return topsSuccess; }
"""
    )
    source = tmp_path / "check.cc"
    source.write_text(
        """
#include "runtime/accelerator/enflame/interop.h"
#include <atomic>
#include <cassert>
#include <chrono>
#include <future>
int stream_syncs=0, device_syncs=0, records=0, waits=0, destroys=0;
bool fail_wait=false;
int main(int argc, char** argv) {
  const bool events=argv[1][0]=='2';
  auto barriers=[&](int expected) {
    assert(stream_syncs==(events ? 2 : expected));
    assert(records==(events ? expected-2 : 0));
    assert(waits==records);
  };
  auto a=reinterpret_cast<void*>(16), b=reinterpret_cast<void*>(32);
  assert(FlagosEnflameBeginLaunch(a,1)==0);
  assert(FlagosEnflameEndLaunch(a)==0);
  assert(stream_syncs==2);  // ordinary wheels retain both legacy barriers
  assert(FlagosEnflameEnableInterop(3)!=0);
  assert(FlagosEnflameEnableInterop(events ? 2 : 1)==0 && device_syncs==1);
  assert(FlagosEnflameEnableInterop(events ? 2 : 1)==0 && device_syncs==1);
  assert(FlagosEnflameEnableInterop(events ? 1 : 2)!=0);
  for(int i=0;i<4;++i) {
    assert(FlagosEnflameBeginLaunch(a,1)==0);
    assert(FlagosEnflameEndLaunch(a)==0);
  }
  barriers(2);  // queued vendor work stays on its stream
  assert(FlagosEnflameBeginLaunch(a,2)==0);
  assert(FlagosEnflameEndLaunch(a)==0);
  barriers(3);
  assert(FlagosEnflameBeginLaunch(b,1)==0);
  assert(FlagosEnflameEndLaunch(b)==0);
  barriers(3);
  assert(FlagosEnflameBeginLaunch(a,1)==0);
  barriers(4);
  std::atomic<bool> started{false};
  auto other=std::async(std::launch::async,[&] {
    started.store(true);
    assert(FlagosEnflameBeginLaunch(a,2)==0);
    assert(FlagosEnflameEndLaunch(a)==0);
  });
  while(!started.load()) {}
  assert(other.wait_for(std::chrono::milliseconds(10))==std::future_status::timeout);
  assert(FlagosEnflameEndLaunch(a)==0);
  other.get();
  barriers(5);  // no other thread enqueues across the open submission
  FlagosEnflameForgetStream(a);
  assert(destroys==(events ? 1 : 0));
  assert(FlagosEnflameBeginLaunch(a,1)==0);
  assert(FlagosEnflameEndLaunch(a)==0);
  barriers(5);
  assert(FlagosEnflameBeginLaunch(a,3)!=0);
  assert(FlagosEnflameEndLaunch(a)!=0);
  if(events) {
    fail_wait=true;
    assert(FlagosEnflameBeginLaunch(a,2)!=0);
    assert(FlagosEnflameEndLaunch(a)!=0);
    fail_wait=false;
    // A failed fence must release the lock and retain the previous producer.
    auto retry=std::async(std::launch::async,[&] {
      assert(FlagosEnflameBeginLaunch(a,2)==0);
      assert(FlagosEnflameEndLaunch(a)==0);
    });
    retry.get();
  }
}
"""
    )
    binary = tmp_path / "check"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-pthread",
            f"-I{tmp_path}",
            f"-I{root / 'csrc'}",
            f"-I{root / 'csrc/include'}",
            str(source),
            str(root / "csrc/runtime/accelerator/enflame/interop.cc"),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run([str(binary), str(abi)], check=True, timeout=10)
