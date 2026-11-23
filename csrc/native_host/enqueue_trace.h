// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <utility>
#include <vector>

namespace flagos::native_host {

class EnqueueTrace {
 public:
  using Events = std::vector<std::pair<uint64_t, bool>>;

  EnqueueTrace() : previous_(std::exchange(current_, this)) {}
  ~EnqueueTrace() { current_ = previous_; }
  EnqueueTrace(const EnqueueTrace&) = delete;
  EnqueueTrace& operator=(const EnqueueTrace&) = delete;

  // The bit records an enqueue from a compiled operator host.
  static void record(uint64_t stream, bool compiled_host) {
    if (current_) current_->events_.emplace_back(stream, compiled_host);
  }

  const Events& events() const { return events_; }

 private:
  // An AOT extension and the operator library can submit into the same
  // scoped trace. Keep the thread-local symbol visible to the Linux loader.
#if defined(__linux__)
  __attribute__((visibility("default")))
#endif
  inline static thread_local EnqueueTrace* current_ = nullptr;
  EnqueueTrace* previous_;
  Events events_;
};

} // namespace flagos::native_host
