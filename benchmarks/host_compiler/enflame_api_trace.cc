// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

// Optional LD_PRELOAD observer. It forwards every call unchanged; it never
// substitutes a kernel, suppresses a wait, or participates in acceptance timing.
#include <tops/tops_runtime.h>
#include <dlfcn.h>
#include <execinfo.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace {
struct Frame { std::string library; uint64_t offset; };
struct Event {
  const char* name;
  uint64_t start, end, bytes;
  int kind, thread;
  std::vector<Frame> frames;
};
std::atomic<bool> enabled{false};
std::mutex mutex;
std::vector<Event> events;
thread_local unsigned depth = 0;
uint64_t Now() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::system_clock::now().time_since_epoch()).count();
}
struct Observe {
  bool participating;
  bool active;
  Event event;
  Observe(const char* name, uint64_t bytes=0, int kind=-1)
      : participating(enabled.load(std::memory_order_relaxed)),
        active(participating && depth++ == 0),
        event{name, 0, 0, bytes, kind, 0, {}} {
    if (active) {
      static thread_local int thread = static_cast<int>(syscall(SYS_gettid));
      event.thread = thread;
      if (kind >= 0) {
        void* stack[24];
        int count = backtrace(stack, 24);
        for (int i=1; i<count; ++i) {
          Dl_info info{};
          if (dladdr(stack[i], &info) && info.dli_fname && info.dli_fbase) {
            event.frames.push_back({info.dli_fname,
                reinterpret_cast<uintptr_t>(stack[i]) - reinterpret_cast<uintptr_t>(info.dli_fbase)});
          }
        }
      }
      event.start = Now();
    }
  }
  ~Observe() {
    if (participating) --depth;
    if (active) {
      event.end = Now();
      std::lock_guard<std::mutex> lock(mutex);
      events.push_back(std::move(event));
    }
  }
};
std::string Escape(const std::string& value) {
  std::string out;
  for (char c : value) {
    if (c == '"' || c == '\\') out += '\\';
    out += c;
  }
  return out;
}
}

extern "C" void flagos_api_trace_begin() {
  std::lock_guard<std::mutex> lock(mutex);
  events.clear();
  events.reserve(50000);
  enabled.store(true, std::memory_order_release);
}
extern "C" int flagos_api_trace_end(const char* path) {
  enabled.store(false, std::memory_order_release);
  std::lock_guard<std::mutex> lock(mutex);
  FILE* file = std::fopen(path, "w");
  if (!file) return 1;
  std::fprintf(file, "{\"schema_version\":1,\"pid\":%d,\"events\":[", getpid());
  bool first = true;
  for (const auto& e : events) {
    std::fprintf(file, "%s{\"api\":\"%s\",\"start_ns\":%lu,\"end_ns\":%lu,\"thread\":%d,\"bytes\":%lu,\"kind\":%d,\"frames\":[",
        first ? "" : ",", e.name, e.start, e.end, e.thread, e.bytes, e.kind);
    first = false;
    for (size_t i=0; i<e.frames.size(); ++i) {
      const auto& f = e.frames[i];
      std::fprintf(file, "%s{\"library\":\"%s\",\"offset\":%lu}",
          i ? "," : "", Escape(f.library).c_str(), f.offset);
    }
    std::fputs("]}", file);
  }
  std::fputs("]}\n", file);
  return std::fclose(file) != 0;
}

#define NEXT(name) static auto real = reinterpret_cast<decltype(&name)>(dlsym(RTLD_NEXT, #name))
extern "C" topsError_t topsGetDevice(int* device) {
  NEXT(topsGetDevice); Observe observation("topsGetDevice"); return real(device);
}
extern "C" topsError_t topsSetDevice(int device) {
  NEXT(topsSetDevice); Observe observation("topsSetDevice"); return real(device);
}
extern "C" topsError_t topsStreamSynchronize(topsStream_t stream) {
  NEXT(topsStreamSynchronize); Observe observation("topsStreamSynchronize"); return real(stream);
}
extern "C" topsError_t topsDeviceSynchronize() {
  NEXT(topsDeviceSynchronize); Observe observation("topsDeviceSynchronize"); return real();
}
extern "C" topsError_t topsEventRecord(topsEvent_t event, topsStream_t stream) {
  NEXT(topsEventRecord); Observe observation("topsEventRecord"); return real(event, stream);
}
extern "C" topsError_t topsStreamWaitEvent(topsStream_t stream, topsEvent_t event, unsigned flags) {
  NEXT(topsStreamWaitEvent); Observe observation("topsStreamWaitEvent"); return real(stream, event, flags);
}
extern "C" topsError_t topsMemcpy(void* dst, const void* src, size_t bytes, topsMemcpyKind kind) {
  NEXT(topsMemcpy); Observe observation("topsMemcpy", bytes, kind); return real(dst, src, bytes, kind);
}
extern "C" topsError_t topsLaunchKernel(const void* fn, dim3 grid, dim3 block,
    void** args, size_t shared, topsStream_t stream) {
  NEXT(topsLaunchKernel); Observe observation("topsLaunchKernel");
  return real(fn, grid, block, args, shared, stream);
}
extern "C" topsError_t topsModuleLaunchKernel(topsFunction_t fn, unsigned gx, unsigned gy,
    unsigned gz, unsigned bx, unsigned by, unsigned bz, unsigned shared,
    topsStream_t stream, void** args, void** extra) {
  NEXT(topsModuleLaunchKernel); Observe observation("topsModuleLaunchKernel");
  return real(fn, gx, gy, gz, bx, by, bz, shared, stream, args, extra);
}
