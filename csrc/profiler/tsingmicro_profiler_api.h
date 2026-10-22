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
//
// C API between the standalone rocprofiler tool (libtorch_fl_profiler.so) and
// the Kineto integration that lives in libtorch_fl.so.
//
// The two are separate libraries on purpose: rocprofiler dlopens its tools
// during its own initialization, and libtorch_fl.so depends on libflagos.so ->
// libhpgr.so, so a tool living in the main library would make rocprofiler pull
// the runtime in while it is still initializing. The tool therefore links
// rocprofiler-sdk and nothing else, and this header is the only thing the two
// sides share.
//
// Pure C: the tool side must be compilable without any torch headers.

#pragma once

#include <stddef.h>
#include <stdint.h>

// The build compiles with -fvisibility=hidden (CMAKE_CXX_VISIBILITY_PRESET), so
// every entry point has to opt back in -- otherwise the tool's functions are
// local symbols and the tracer's dlsym finds none of them.
#if defined(__GNUC__)
#define TORCH_FL_PROFILER_API __attribute__((visibility("default")))
#else
#define TORCH_FL_PROFILER_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Opens the collection window and samples the clock. Returns 0 on success.
// The rocprofiler tracing services themselves are configured once, when the
// tool is first initialized; this only toggles the context.
TORCH_FL_PROFILER_API int torch_fl_profiler_start(void);

// Closes the window: stops the context and flushes the buffer, so records are
// in hand before the caller reads them.
TORCH_FL_PROFILER_API int torch_fl_profiler_stop(void);

// rocprofiler's clock -> Unix epoch nanoseconds, as sampled by start().
// Kernel timestamps come from rocprofiler's clock; the trace needs them on the
// same axis as the CPU ops, which PyTorch stamps with Unix epoch ns.
TORCH_FL_PROFILER_API int64_t torch_fl_profiler_clock_offset_ns(void);

// Brackets one op, so a kernel can be attributed back to it. Called from the
// Kineto session for every op. This pushes the op's id as the thread-local
// external correlation id, which rocprofiler copies into the external field of
// the kernel dispatch records launched by that op, so the caller can read the
// op id straight off the record.
TORCH_FL_PROFILER_API void torch_fl_profiler_push_correlation(uint64_t id);
TORCH_FL_PROFILER_API void torch_fl_profiler_pop_correlation(void);

// Timestamps (rocprofiler clock) of the launch API call that carries this
// internal correlation id -- the id already on the kernel dispatch record.
// Kineto draws the CPU -> kernel arrow from a flow pair, and CUDA's start
// point is the cudaLaunchKernel record; this is the equivalent, so the arrow
// starts from a real event rather than a synthetic zero-length one.
// Returns 0 on success, -1 when the window was not seen.
TORCH_FL_PROFILER_API int torch_fl_profiler_lookup_launch(
    uint64_t internal_correlation_id,
    uint64_t* start_ns,
    uint64_t* end_ns);

// Kernel dispatch records captured so far. `take` hands ownership to the
// caller and empties the tool's copy; release it with `release`.
typedef struct torch_fl_profiler_records_t {
  void* data;     // rocprofiler_buffer_tracing_tsm_kernel_dispatch_record_t[]
  size_t count;   // number of records
  size_t stride;  // sizeof one record
} torch_fl_profiler_records_t;

TORCH_FL_PROFILER_API int torch_fl_profiler_take_records(
    torch_fl_profiler_records_t* out);
TORCH_FL_PROFILER_API void torch_fl_profiler_release_records(
    torch_fl_profiler_records_t* records);

// Counters kept for diagnosing a backend that stops reporting correlation.
TORCH_FL_PROFILER_API int64_t torch_fl_profiler_push_count(void);
TORCH_FL_PROFILER_API int64_t torch_fl_profiler_correlation_cb_count(void);
TORCH_FL_PROFILER_API int64_t torch_fl_profiler_api_cb_count(void);

#ifdef __cplusplus
}
#endif
