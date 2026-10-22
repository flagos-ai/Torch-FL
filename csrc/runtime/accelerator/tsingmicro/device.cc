#include <flagos.h>
#include <tx_runtime.h>

namespace {

// Current device index (thread local for multi-threading support)
thread_local int gCurrentDevice = 0;

} // namespace

Error_t GetDeviceCount(int* count) {
  if (!count) {
    return ErrorUnknown;
  }

  uint32_t tx_count = 0;
  txError_t err = txGetDeviceCount(&tx_count);
  if (err != TX_SUCCESS) {
    *count = 0;
    return ErrorUnknown;
  }

  *count = static_cast<int>(tx_count);
  return Success;
}

Error_t GetDevice(int* device) {
  if (!device) {
    return ErrorUnknown;
  }

  *device = gCurrentDevice;
  return Success;
}

Error_t SetDevice(int device) {
  int count = 0;
  GetDeviceCount(&count);

  if (device < 0 || device >= count) {
    return ErrorInvalidDevice;
  }

  txError_t err = txSetDevice(static_cast<uint32_t>(device));
  if (err != TX_SUCCESS) {
    return ErrorUnknown;
  }

  gCurrentDevice = device;
  return Success;
}

Error_t DeviceGetStreamPriorityRange(int* leastPriority, int* greatestPriority) {
  // TX runtime has no stream priority API
  *leastPriority = 0;
  *greatestPriority = 0;
  return Success;
}

Error_t DeviceSynchronize(void) {
  txError_t err = txDeviceSynchronize();
  return (err == TX_SUCCESS) ? Success : ErrorUnknown;
}