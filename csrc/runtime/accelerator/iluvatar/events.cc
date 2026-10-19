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
// corex's event API. The stream half of the CUDA runtime contract lived in
// this file when it was the shared cuda/stream.cc; it now lives in stream.cc
// beside the registry, because iluvatar owns its streams rather than reading
// torch's (see the header comment there). What remains is the thin cuda_*
// event mapping, unchanged from the CUDA sources.

#include <flagos.h>
#include <cuda_runtime.h>
#include "iluvatar_context.h"

Error_t EventCreateWithFlags(Event_t* event, unsigned int flags) {
  c10::flagos::iluvatar::EnsureIluvatarContext();
  cudaError_t err = cudaEventCreateWithFlags((cudaEvent_t*)event, flags);
  return (err == cudaSuccess) ? Success : ErrorUnknown;
}

Error_t EventCreate(Event_t* event) {
  c10::flagos::iluvatar::EnsureIluvatarContext();
  cudaError_t err = cudaEventCreate((cudaEvent_t*)event);
  return (err == cudaSuccess) ? Success : ErrorUnknown;
}

Error_t EventDestroy(Event_t event) {
  c10::flagos::iluvatar::EnsureIluvatarContext();
  cudaError_t err = cudaEventDestroy((cudaEvent_t)event);
  return (err == cudaSuccess) ? Success : ErrorUnknown;
}

Error_t EventRecord(Event_t event, Stream_t stream) {
  c10::flagos::iluvatar::EnsureIluvatarContext();
  cudaError_t err = cudaEventRecord((cudaEvent_t)event, (cudaStream_t)stream);
  return (err == cudaSuccess) ? Success : ErrorUnknown;
}

Error_t EventSynchronize(Event_t event) {
  c10::flagos::iluvatar::EnsureIluvatarContext();
  cudaError_t err = cudaEventSynchronize((cudaEvent_t)event);
  return (err == cudaSuccess) ? Success : ErrorUnknown;
}

Error_t EventQuery(Event_t event) {
  c10::flagos::iluvatar::EnsureIluvatarContext();
  cudaError_t err = cudaEventQuery((cudaEvent_t)event);
  if (err == cudaSuccess) {
    return Success;
  } else if (err == cudaErrorNotReady) {
    return ErrorNotReady;
  }
  return ErrorUnknown;
}

Error_t EventElapsedTime(float* ms, Event_t start, Event_t end) {
  c10::flagos::iluvatar::EnsureIluvatarContext();
  cudaError_t err = cudaEventElapsedTime(ms, (cudaEvent_t)start, (cudaEvent_t)end);
  return (err == cudaSuccess) ? Success : ErrorUnknown;
}
