// Copyright 2026 FlagOS Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Shared Metal state for the flagos device runtime.
//
// device.mm, memory.mm and stream.mm all reach the GPU through one MTLDevice,
// one lazily built default command queue, and one registry of live queues, so
// the pieces live here rather than in any single translation unit. There is no
// Objective-C ARC: this file builds with the compiler default (manual retain /
// release), which is what lets the plain C++ side tables elsewhere hold
// MTLBuffer and MTLCommandBuffer references without ownership qualifiers. Every
// +1 object stored in those tables is retained on the way in and released on the
// way out.

#include "mps.h"
#include "mps_runtime.h"

// _XOPEN_SOURCE is what makes <ucontext.h> declare ucontext_t at all: on this
// SDK the header is an #error without it. Define before the first system
// include, and only for this translation unit.
#ifndef _XOPEN_SOURCE
#define _XOPEN_SOURCE 700
#endif

#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <execinfo.h>
#include <unistd.h>
#include <ucontext.h>
#include <mutex>
#include <set>
#include <vector>
#include <dlfcn.h>
#include <mach-o/dyld.h>
#include <mach-o/loader.h>

namespace torch_fl_mps {
namespace {

std::mutex gDeviceMutex;
id<MTLDevice> gDevice = nil;
bool gDeviceResolved = false;

// Optional native backtrace on a fatal signal, off unless the environment asks
// for it.
//
// Why this exists: a segfault inside an MPS op reaches the interpreter as a
// Python traceback that names only `torch_fl/__init__.py`'s __torch_function__
// trampoline -- the crash is several frames *below* the last Python frame, so
// the log says which aten call was in flight and nothing about where in the
// Metal or allocator code it died. That is enough to know an op crashed and not
// enough to fix it, which is exactly the position a hosted macOS runner leaves
// you in: no debugger, no core, and a host whose GPU and OS differ from the
// developer's.
//
// So the handler is installed in-process, and only when the operator asks for
// it, because a signal handler that prints is not something to enable by
// default: it can deadlock (the crashing thread may hold the malloc or stdio
// lock backtrace_symbols needs) and it changes the post-crash behaviour a test
// might observe. Ungated it would also be dead weight in every process that
// merely links the backend.
//
// It is a diagnostic, not a fix: it prints and then re-raises with the default
// disposition so the process still dies from the original signal and CI still
// reports the failure.
//
// The report is deliberately more than a frame list, because the frame list is
// not always there. `backtrace()` walks the frame-pointer chain, and a fault
// that has damaged that chain returns zero frames -- which is itself a finding
// (it says the crash is not an ordinary call into a bad pointer) but leaves the
// address unknown. So the faulting address and the register file are printed
// too: they come from the kernel's signal context, not from the stack, and are
// available whatever happened to the stack.
ssize_t CrashWrite(const char* text) {
  // write(2) rather than printf: stdio is not async-signal-safe, and the
  // buffered half of a message is exactly what gets lost on a fatal signal.
  return ::write(2, text, ::strlen(text));
}

void CrashWriteHex(const char* label, uintptr_t value) {
  char line[64];
  int n = 0;
  for (const char* p = label; *p != '\0' && n < 40; ++p) {
    line[n++] = *p;
  }
  const char* digits = "0123456789abcdef";
  line[n++] = '0';
  line[n++] = 'x';
  // 16 nibbles, high to low, with the leading zeros kept: a fault address of
  // 0x0 is a different finding from one of 0x1f3a..., so nothing is trimmed.
  for (int shift = 60; shift >= 0; shift -= 4) {
    line[n++] = digits[(value >> shift) & 0xf];
  }
  line[n++] = '\n';
  ::write(2, line, static_cast<size_t>(n));
}

void CrashTraceHandler(int signo, siginfo_t* info, void* context) {
  CrashWrite("\n=== FLAGOS_MPS_CRASH_TRACE: begin ===\n");
  CrashWrite("signal: ");
  const char* name = "?";
  switch (signo) {
    case SIGSEGV: name = "SIGSEGV"; break;
    case SIGBUS:  name = "SIGBUS";  break;
    case SIGABRT: name = "SIGABRT"; break;
    case SIGILL:  name = "SIGILL";  break;
    case SIGFPE:  name = "SIGFPE";  break;
    default: break;
  }
  CrashWrite(name);
  CrashWrite("\n");

  // si_addr is the faulting address for the memory-fault signals, which is the
  // single most useful number here: it distinguishes a null dereference from a
  // wild pointer from a guard-page strike.
  if (info != nullptr) {
    CrashWriteHex("fault address: ", reinterpret_cast<uintptr_t>(info->si_addr));
    CrashWriteHex("si_code: ", static_cast<uintptr_t>(info->si_code));
  }

  // backtrace() returned nothing on the runner -- the frame chain is gone by the
  // time the handler runs, most likely because the faulting frame had already
  // pushed a large stack adjustment or unwound through a leaf. The ucontext
  // still carries the register file the kernel saved at the fault, so the
  // program counter and the frame-pointer chain are read straight out of it and
  // walked by hand. That is the only way to name the caller of a call that died
  // inside a library with a broken/unwind-hostile prologue.
  if (context != nullptr) {
    const ucontext_t* uc = static_cast<const ucontext_t*>(context);
#if defined(__arm64__) || defined(__aarch64__)
    const uintptr_t pc = static_cast<uintptr_t>(uc->uc_mcontext->__ss.__pc);
    const uintptr_t fp = static_cast<uintptr_t>(uc->uc_mcontext->__ss.__fp);
    const uintptr_t lr = static_cast<uintptr_t>(uc->uc_mcontext->__ss.__lr);
    CrashWriteHex("pc: ", pc);
    CrashWriteHex("lr: ", lr);
#elif defined(__x86_64__)
    const uintptr_t pc = static_cast<uintptr_t>(uc->uc_mcontext->__ss.__rip);
    const uintptr_t fp = static_cast<uintptr_t>(uc->uc_mcontext->__ss.__rbp);
    CrashWriteHex("pc: ", pc);
#else
    const uintptr_t pc = 0;
    const uintptr_t fp = 0;
#endif
    CrashWrite("=== FLAGOS_MPS_CRASH_TRACE: frame-pointer walk ===\n");
    CrashWriteHex("frame00: ", pc);
    // A frame-pointer walk over a destroyed stack can read anything; every
    // dereference is a fault away from a nested crash, which on a broken stack
    // is exactly what makes this useful rather than reckless -- a nested fault
    // here re-raises the original signal through the same handler and prints
    // "begin" a second time, which is itself evidence.
    uintptr_t frame = fp;
    for (int i = 1; i < 48 && frame != 0; ++i) {
      const uintptr_t* words = reinterpret_cast<const uintptr_t*>(frame);
      const uintptr_t next = words[0];
      const uintptr_t ret = words[1];
      CrashWriteHex("frame", ret);
      // Name the image each return address lands in, so a frame list whose
      // addresses are all in a framework the developer has no symbols for still
      // says *which* framework without a separate atos pass over a core that
      // does not exist.
      Dl_info info;
      if (ret != 0 && ::dladdr(reinterpret_cast<void*>(ret), &info) != 0 &&
          info.dli_fname != nullptr) {
        CrashWrite("  in ");
        CrashWrite(info.dli_fname);
        CrashWrite("\n");
      }
      if (next <= frame) {
        break;
      }
      frame = next;
    }
    // The image table with each library's load address, the input `atos -o <lib>
    // -l <addr>` needs. Printed after the frames so a truncated log still has
    // the frame list first.
    CrashWrite("=== FLAGOS_MPS_CRASH_TRACE: images ===\n");
    const uint32_t image_count = _dyld_image_count();
    for (uint32_t i = 0; i < image_count; ++i) {
      const char* name = _dyld_get_image_name(i);
      if (name == nullptr) {
        continue;
      }
      CrashWriteHex("image load: ", static_cast<uintptr_t>(
          reinterpret_cast<uintptr_t>(_dyld_get_image_header(i))));
      CrashWrite(" ");
      CrashWrite(name);
      CrashWrite("\n");
    }
  }

  void* frames[128];
  const int count = ::backtrace(frames, 128);
  CrashWrite("backtrace frames: ");
  CrashWriteHex("", static_cast<uintptr_t>(count));
  CrashWrite("=== FLAGOS_MPS_CRASH_TRACE: native backtrace ===\n");
  if (count > 0) {
    ::backtrace_symbols_fd(frames, count, 2);
  }
  CrashWrite("=== FLAGOS_MPS_CRASH_TRACE: end ===\n");

  // Default disposition, so the signal that killed us is the one reported and
  // the exit status is the same as it would have been without this handler.
  ::signal(signo, SIG_DFL);
  ::raise(signo);
}

void MaybeInstallCrashTrace() {
  static bool installed = false;
  if (installed) {
    return;
  }
  installed = true;
  const char* wanted = ::getenv("FLAGOS_MPS_CRASH_TRACE");
  if (wanted == nullptr || wanted[0] == '\0' || wanted[0] == '0') {
    return;
  }
  // SA_SIGINFO is what makes the fault address available; SA_RESETHAND would
  // restore the default disposition on delivery for free, but the handler
  // re-raises explicitly so the intent stays legible at the call site.
  struct sigaction action;
  ::memset(&action, 0, sizeof(action));
  action.sa_sigaction = CrashTraceHandler;
  action.sa_flags = SA_SIGINFO | SA_NODEFER;
  // Not `::sigemptyset`: it is a macro that expands to an expression, so the
  // qualified spelling is a syntax error. `sigaction` is a real function and
  // keeps the `::`.
  sigemptyset(&action.sa_mask);
  ::sigaction(SIGSEGV, &action, nullptr);
  ::sigaction(SIGBUS, &action, nullptr);
  ::sigaction(SIGABRT, &action, nullptr);
  ::sigaction(SIGILL, &action, nullptr);
  ::sigaction(SIGFPE, &action, nullptr);
}

std::mutex gDefaultStreamMutex;
StreamState* gDefaultStream = nullptr;

// Every queue DrainAll() must wait on. The default queue is one entry; each
// StreamCreate() adds another and StreamDestroy() takes it out.
std::mutex gRegistryMutex;
std::set<StreamState*> gLiveStreams;

// The shutdown drain, registered the first time a queue is created.
//
// Ops commit their MTLCommandBuffer and return; nothing on the ordinary path
// ever waits for one to finish, because a Metal queue runs in order and the
// next host read is what forces the wait. That is fine while the process keeps
// running and wrong at the very end: an MTLCommandBuffer still in flight when
// libmetal / MTLCompilerService tear themselves down faults inside the encode
// machinery, which is a crash the operator sees as an inexplicable segfault in
// a program whose last GPU op finished correctly.
//
// atexit runs while the C++ runtime and the still-loaded Metal frameworks are
// intact, which is early enough to drain and late enough that no later user
// code can commit more work. Draining twice is harmless -- waitUntilCompleted
// on a finished buffer returns immediately, and DrainAll is already the
// implementation of DeviceSynchronize.
//
// The flag is deliberate: if the drain itself faults (a queue already torn
// down by a framework that unloads early), the second invocation must not run
// it again and turn one fault into a crash loop inside exit().
bool gShutdownDrainRegistered = false;
bool gShutdownDrainRunning = false;

void ShutdownDrain() {
  if (gShutdownDrainRunning) {
    return;
  }
  gShutdownDrainRunning = true;
  DrainAll();
}

void MaybeRegisterShutdownDrain() {
  if (gShutdownDrainRegistered) {
    return;
  }
  gShutdownDrainRegistered = true;
  ::atexit(&ShutdownDrain);
}

} // namespace

id<MTLDevice> MetalDevice() {
  std::lock_guard<std::mutex> lock(gDeviceMutex);
  if (!gDeviceResolved) {
    MaybeInstallCrashTrace();
    // +1 (Create rule). Held for the life of the process; a Mac has one GPU and
    // re-querying here would only rebuild the same object.
    gDevice = MTLCreateSystemDefaultDevice();
    gDeviceResolved = true;
  }
  return gDevice;
}

bool MetalAvailable() {
  id<MTLDevice> device = MetalDevice();
  // A non-nil device is not enough: an Intel Mac with a discrete AMD GPU also
  // answers MTLCreateSystemDefaultDevice(), but MPS (and the shared-memory
  // assumption this runtime is built on) needs unified memory.
  return device != nil && device.hasUnifiedMemory;
}

bool MetalGraphUsable() {
  id<MTLDevice> device = MetalDevice();
  // The paravirtualized family is the one MPSGraph encode is known to fault on,
  // and the device class is the only thing that distinguishes those hosts from a
  // physical GPU: the same OS build, the same MPSGraph, the same unified memory.
  // `MTLDevice.name` is the family string ("Apple Paravirtual device"), and the
  // loaded IOKit personality is AppleParavirtGPUMetalIOGPUFamily. The name is
  // checked rather than the loaded bundle because it is the documented property
  // and does not depend on an internal framework's filename.
  NSString* name = device != nil ? device.name : nil;
  const bool paravirtual =
      name != nil &&
      [name rangeOfString:@"paravirtual"
                  options:NSCaseInsensitiveSearch].location != NSNotFound;
  const bool usable =
      device != nil && device.hasUnifiedMemory && !paravirtual;
  // DIAGNOSTIC (reverted once it has answered): the gate is a name test, and
  // the one host it has to trip is a CI runner whose name string has never been
  // read. Printing the raw device facts from the process doing the deciding
  // turns "the gate fired" from an inference into an observation, and it
  // separates "this host is genuinely paravirtualized" from "the name test is
  // matching something that is not", which are the two readings a green MPS
  // suite could otherwise hide behind.
  if (::getenv("FLAGOS_MPS_CRASH_TRACE") != nullptr && name != nil) {
    std::fprintf(stderr,
                 "[mps-device] name=\"%s\" unified=%d paravirtual=%d "
                 "graph_usable=%d\n",
                 name.UTF8String,
                 static_cast<int>(device.hasUnifiedMemory),
                 static_cast<int>(paravirtual),
                 static_cast<int>(usable));
    std::fflush(stderr);
  }
  return usable;
}

StreamState* DefaultStream() {
  std::lock_guard<std::mutex> lock(gDefaultStreamMutex);
  if (gDefaultStream == nullptr) {
    id<MTLDevice> device = MetalDevice();
    if (device == nil) {
      return nullptr;
    }
    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (queue == nil) {
      return nullptr;
    }
    auto* state = new StreamState();
    state->queue = queue; // +1, owned by the state for the process lifetime
    gDefaultStream = state;
    RegisterStream(state);
    // A queue exists from here on, so committed work can outlive main.
    MaybeRegisterShutdownDrain();
  }
  return gDefaultStream;
}

StreamState* AsStream(Stream_t stream) {
  // A null handle is the default stream. csrc/runtime/guard.h synthesizes
  // c10::Stream(UNSAFE, device, 0) on this backend, so id 0 -- a null Stream_t
  // -- is the stream every aten call and every allocator event arrives on.
  if (stream == nullptr) {
    return DefaultStream();
  }
  return reinterpret_cast<StreamState*>(stream);
}

void Commit(StreamState* stream, id<MTLCommandBuffer> buffer) {
  if (stream == nullptr || buffer == nil) {
    return;
  }
  // The buffer usually comes from -[MTLCommandQueue commandBuffer], i.e. +0,
  // and the caller's autorelease pool may drain before the next wait. Take a
  // reference so "the newest committed buffer" stays valid for Drain().
  [buffer retain];
  [buffer commit];
  id<MTLCommandBuffer> previous = stream->last_committed;
  stream->last_committed = buffer;
  if (previous != nil) {
    [previous release];
  }
}

void Drain(StreamState* stream) {
  if (stream == nullptr) {
    return;
  }
  id<MTLCommandBuffer> buffer = stream->last_committed;
  if (buffer == nil) {
    return;
  }
  // Metal executes a queue strictly in order, so waiting on the newest buffer
  // waits on everything before it. waitUntilCompleted on an already-finished
  // buffer returns immediately; on one that was never committed it would raise,
  // which is why last_committed only ever names a committed buffer.
  const MTLCommandBufferStatus status = buffer.status;
  if (status == MTLCommandBufferStatusCommitted ||
      status == MTLCommandBufferStatusScheduled) {
    [buffer waitUntilCompleted];
  }
}

void DrainAll() {
  // Snapshot under the lock, then wait outside it: draining can block for the
  // duration of a kernel, and RegisterStream on another thread must not stall
  // behind that.
  std::vector<StreamState*> streams;
  {
    std::lock_guard<std::mutex> lock(gRegistryMutex);
    streams.assign(gLiveStreams.begin(), gLiveStreams.end());
  }
  for (StreamState* stream : streams) {
    Drain(stream);
  }
}

void RegisterStream(StreamState* stream) {
  if (stream == nullptr) {
    return;
  }
  std::lock_guard<std::mutex> lock(gRegistryMutex);
  gLiveStreams.insert(stream);
}

void UnregisterStream(StreamState* stream) {
  if (stream == nullptr) {
    return;
  }
  std::lock_guard<std::mutex> lock(gRegistryMutex);
  gLiveStreams.erase(stream);
}

EventState* AsEvent(Event_t event) {
  return reinterpret_cast<EventState*>(event);
}

} // namespace torch_fl_mps

// Non-contract helper for the operator layer. Declared in mps.h. The operator
// layer is Objective-C++ and could reach AsStream itself if it included this
// header, but it lives in a different component (csrc/aten/) that must not
// depend on the runtime's internal header, so the one call it needs is exported
// through the same side door as FlagosMPSDeviceMemoryInfo.
extern "C" void* FlagosMPSCommandQueue(Stream_t stream) {
  torch_fl_mps::StreamState* state = torch_fl_mps::AsStream(stream);
  return state == nullptr ? nullptr : (void*)state->queue;
}

// See mps.h. The buffer is committed and released immediately: Commit() hands
// the state its own reference, which is the one Drain() later waits on, so
// nothing here outlives the call.
extern "C" void FlagosMPSStreamSentinel(Stream_t stream) {
  torch_fl_mps::StreamState* state = torch_fl_mps::AsStream(stream);
  if (state == nullptr || state->queue == nil) {
    return;
  }
  id<MTLCommandBuffer> buffer = [state->queue commandBuffer];
  if (buffer == nil) {
    return;
  }
  torch_fl_mps::Commit(state, buffer);
}
