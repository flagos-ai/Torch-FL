// Copyright (c) 2026, BAAI. All rights reserved.
//
// Centralized ACL stream management for the Ascend backend.
// All Ascend ops (both native aclnn and FlagGems/TritonJIT) share the same
// default stream to ensure implicit ordering without explicit synchronization.

#pragma once

#ifdef USE_ASCEND

#include <acl/acl_rt.h>

#include <macros.h>
#include <string>

namespace at::native::flagos::ascend {

// Returns the default ACL stream of the *current* device. Every Ascend op on
// that device shares it, so implicit ordering holds without explicit
// synchronization.
//
// These MUST be single external-linkage, default-visibility symbols defined
// once (in libflagos.so). `GetDefaultAclStream` used to be an `inline` function
// with a function-local `static`; under -fvisibility=hidden that produced a
// SEPARATE stream instance per shared object (libflagos.so vs libtorch_fl.so).
// The aten kernels in libtorch_fl.so then enqueued ops on one stream while the
// drain-before-read in libflagos.so's memory.cc synchronized a DIFFERENT
// stream, so host-visible D2H reads never waited for the producing kernels ->
// silent corruption under async dispatch. Keep it a plain exported function.
//
// The stream is ALSO device-scoped, and an aclrtStream belongs to the device
// that was current when it was created. A single process-wide stream therefore
// made every device but the first one either return zeros or fail outright
// (the op enqueued into another device's context, and the drain-before-read
// synchronized that other device's stream). Hence one default stream per
// device, created lazily on its own device.
FLAGOS_EXPORT aclrtStream GetDefaultAclStream();
FLAGOS_EXPORT aclrtStream GetDefaultAclStreamForDevice(int device_index);

// Wait for every per-device default stream that has been created so far.
//
// Called before a blocking (non-stream-ordered) transfer that touches device
// memory. The caller does not always know which device owns the pointer, so
// all of them are drained; only devices that actually ran an op have a stream
// here, which in practice is the one or two devices in use.
FLAGOS_EXPORT void DrainDefaultAclStreams();

// Return the stream selected for the calling thread and device. If no
// auxiliary stream has been selected, this returns the shared default stream.
FLAGOS_EXPORT aclrtStream GetCurrentAclStream();
FLAGOS_EXPORT aclrtStream GetCurrentAclStreamForDevice(int device_index);
FLAGOS_EXPORT void SetCurrentAclStream(aclrtStream stream);
FLAGOS_EXPORT void SetCurrentAclStreamForDevice(int device_index, aclrtStream stream);

// The SoC name of the current device, e.g. "Ascend910_9382" or "Ascend310B1",
// or an empty std::string when ACL cannot report one.
//
// CANN's aclnn operator set is not uniform across SoCs, so a kernel that sends
// every dtype upstream regardless of hardware fails at GetWorkspaceSize with an
// opaque status instead of falling back. Callers use this to decide what the
// installed CANN can actually run here.
//
// Exported for the same reason the stream accessors are: the aten kernels live
// in a different shared object, and an `inline` function-local static would give
// each object its own cache.
FLAGOS_EXPORT std::string GetSocName();

} // namespace at::native::flagos::ascend

#endif // USE_ASCEND
