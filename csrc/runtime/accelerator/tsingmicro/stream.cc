#include <flagos.h>
#include <tx_runtime.h>

#include <mutex>
#include <unordered_map>

// ---------------------------------------------------------------------------
// Per-device stream registry
//
// The framework contract -- and the FlagCX torch plugin, which is built against
// libflagos.so -- resolves the stream a collective or a kernel should run on
// through GetCurrentStreamForDevice / SetCurrentStreamForDevice. Those are the
// functions the whole device abstraction is expressed in: the guard's
// getStream/exchangeStream and flagcxStreamGuard's stream swap both go through
// them. A txStream_t is bound to the device that was current when it was
// created (see GetDefaultStreamForDevice below), so they have to be per-device
// maps rather than one global handle.
//
// The shapes here follow runtime/accelerator/gcu/stream.cc, the other
// device-scoped vendor: `default_streams` is process-wide under a mutex (one
// lazily created stream per device, shared by every thread) and
// `current_streams` is thread_local, because "the current stream" is a
// per-thread notion -- a thread's exchangeStream() must not move the stream
// another thread is issuing into.
// ---------------------------------------------------------------------------

namespace {

std::mutex stream_mutex;
std::unordered_map<int, txStream_t> default_streams;
thread_local std::unordered_map<int, txStream_t> current_streams;

txStream_t get_default_stream(int device) {
  std::lock_guard<std::mutex> lock(stream_mutex);
  auto it = default_streams.find(device);
  if (it != default_streams.end()) {
    return it->second;
  }

  // txStreamCreate binds the new stream to the device that is current at the
  // call, so the creating call has to run on `device` even when another one is
  // current -- the normal case, since the registry is queried for a device
  // other than the caller's on every multi-device run. Restore the caller's
  // device afterwards either way.
  uint32_t previous_device = 0;
  if (txGetDevice(&previous_device) != TX_SUCCESS) {
    return nullptr;
  }
  if (previous_device != static_cast<uint32_t>(device) &&
      txSetDevice(static_cast<uint32_t>(device)) != TX_SUCCESS) {
    return nullptr;
  }

  txStream_t stream = nullptr;
  const txError_t create_err = txStreamCreate(&stream);

  if (previous_device != static_cast<uint32_t>(device)) {
    txSetDevice(previous_device);
  }

  if (create_err != TX_SUCCESS) {
    // Do NOT cache the failure: a null pinned here would be returned by every
    // later lookup for this device, with no retry, even after a transient
    // create failure clears.
    return nullptr;
  }

  default_streams.emplace(device, stream);
  return stream;
}

} // namespace

Stream_t GetDefaultStreamForDevice(int device) {
  return reinterpret_cast<Stream_t>(get_default_stream(device));
}

Stream_t GetCurrentStreamForDevice(int device) {
  auto it = current_streams.find(device);
  if (it != current_streams.end()) {
    return reinterpret_cast<Stream_t>(it->second);
  }
  return GetDefaultStreamForDevice(device);
}

Error_t SetCurrentStreamForDevice(int device, Stream_t stream) {
  current_streams[device] = reinterpret_cast<txStream_t>(stream);
  return Success;
}

Error_t StreamCreateWithPriority(
    Stream_t* stream,
    unsigned int /*flags*/,
    int /*priority*/) {
  // TX runtime has no stream priority support
  txError_t err = txStreamCreate(reinterpret_cast<txStream_t*>(stream));
  return (err == TX_SUCCESS) ? Success : ErrorUnknown;
}

Error_t StreamCreate(Stream_t* stream) {
  txError_t err = txStreamCreate(reinterpret_cast<txStream_t*>(stream));
  return (err == TX_SUCCESS) ? Success : ErrorUnknown;
}

Error_t StreamGetPriority(Stream_t /*stream*/, int* priority) {
  // TX runtime has no stream priority query
  *priority = 0;
  return Success;
}

Error_t StreamDestroy(Stream_t stream) {
  txError_t err = txStreamDestroy(reinterpret_cast<txStream_t>(stream));
  return (err == TX_SUCCESS) ? Success : ErrorUnknown;
}

Error_t StreamQuery(Stream_t stream) {
  txError_t err = txStreamQuery(reinterpret_cast<txStream_t>(stream));
  if (err == TX_SUCCESS) {
    return Success;
  } else if (err == TX_ERROR_NOT_READY) {
    return ErrorNotReady;
  }
  return ErrorUnknown;
}

Error_t StreamSynchronize(Stream_t stream) {
  txError_t err = txStreamSynchronize(reinterpret_cast<txStream_t>(stream));
  return (err == TX_SUCCESS) ? Success : ErrorUnknown;
}

Error_t StreamWaitEvent(Stream_t stream, Event_t event, unsigned int /*flags*/) {
  // TX runtime's StreamWaitEvent has no flags parameter
  txError_t err = txStreamWaitEvent(
      reinterpret_cast<txStream_t>(stream),
      reinterpret_cast<txEvent_t>(event));
  return (err == TX_SUCCESS) ? Success : ErrorUnknown;
}

Error_t EventCreateWithFlags(Event_t* event, unsigned int /*flags*/) {
  // TX runtime has no event flags support
  txError_t err = txEventCreate(reinterpret_cast<txEvent_t*>(event));
  return (err == TX_SUCCESS) ? Success : ErrorUnknown;
}

Error_t EventCreate(Event_t* event) {
  txError_t err = txEventCreate(reinterpret_cast<txEvent_t*>(event));
  return (err == TX_SUCCESS) ? Success : ErrorUnknown;
}

Error_t EventDestroy(Event_t event) {
  txError_t err = txEventDestroy(reinterpret_cast<txEvent_t>(event));
  return (err == TX_SUCCESS) ? Success : ErrorUnknown;
}

Error_t EventRecord(Event_t event, Stream_t stream) {
  txError_t err = txEventRecord(
      reinterpret_cast<txEvent_t>(event),
      reinterpret_cast<txStream_t>(stream));
  return (err == TX_SUCCESS) ? Success : ErrorUnknown;
}

Error_t EventSynchronize(Event_t event) {
  txError_t err = txEventSynchronize(reinterpret_cast<txEvent_t>(event));
  return (err == TX_SUCCESS) ? Success : ErrorUnknown;
}

Error_t EventQuery(Event_t event) {
  txError_t err = txEventQuery(reinterpret_cast<txEvent_t>(event));
  if (err == TX_SUCCESS) {
    return Success;
  } else if (err == TX_ERROR_NOT_READY) {
    return ErrorNotReady;
  }
  return ErrorUnknown;
}

Error_t EventElapsedTime(float* ms, Event_t start, Event_t end) {
  txError_t err = txEventElapsedTime(
      ms,
      reinterpret_cast<txEvent_t>(start),
      reinterpret_cast<txEvent_t>(end));
  return (err == TX_SUCCESS) ? Success : ErrorUnknown;
}