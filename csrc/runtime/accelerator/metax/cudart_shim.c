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

/*
 * Complete CUDA Runtime API shim for MetaX compatibility.
 *
 * PyTorch's libc10_cuda.so and libtorch_cuda.so require ~85 symbols from
 * libcudart.so.12 with @@libcudart.so.12 version tags.  MetaX's
 * libsymbol_cu.so provides 75 of these but without version tags.
 *
 * This file:
 *   1. Forwards 75 symbols to libsymbol_cu.so via dlsym (lazy init)
 *   2. Stubs 10 symbols that don't exist in MetaX at all
 *   3. Overrides cudaDriverGetVersion / cudaRuntimeGetVersion to report
 *      CUDA 12.8 (matching the PyTorch build)
 *
 * Build with the version script (csrc/libcudart.version) and
 * -Wl,-soname,libcudart.so.12 so every exported symbol gets the
 * @@libcudart.so.12 tag.
 */

#include <dlfcn.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* Opaque CUDA types — we only pass them through, never inspect them. */
/* ------------------------------------------------------------------ */
typedef int   cudaError_t;
typedef void* cudaStream_t;
typedef void* cudaEvent_t;
typedef void* cudaMemPool_t;
typedef void* cudaGraph_t;
typedef void* cudaGraphExec_t;
typedef void* cudaGraphNode_t;
typedef void (*cudaStreamCallback_t)(cudaStream_t, cudaError_t, void*);

typedef struct { int x, y, z; } dim3;
typedef struct { char _[64]; } cudaIpcEventHandle_t;
typedef struct { char _[64]; } cudaIpcMemHandle_t;
typedef struct { char _pad[1024]; } cudaDeviceProp;

/* Forward-declare the config struct (CUDA 12 launch API) */
typedef struct cudaLaunchConfig_st {
    char _opaque[128];
} cudaLaunchConfig_t;

/* Enums used in signatures — actual values don't matter for forwarding. */
typedef int cudaMemcpyKind;
typedef int cudaDeviceAttr;
typedef int cudaMemPoolAttr;
typedef int cudaFuncAttribute;
typedef int cudaStreamCaptureMode;
typedef int cudaStreamCaptureStatus;
typedef int cudaGraphNodeType;
typedef int cudaDriverEntryPointQueryResult;

typedef struct cudaFuncAttributes_st { char _[128]; } cudaFuncAttributes;
typedef struct cudaPointerAttributes_st { char _[64]; } cudaPointerAttributes;
typedef struct cudaMemAccessDesc_st { char _[32]; } cudaMemAccessDesc;

/* ------------------------------------------------------------------ */
/* Lazy-init handle for libsymbol_cu.so                               */
/* ------------------------------------------------------------------ */
static void *_metax_lib = NULL;

static void _ensure_metax_lib(void) {
    if (_metax_lib) return;
    _metax_lib = dlopen("libsymbol_cu.so", RTLD_LAZY | RTLD_GLOBAL);
    if (!_metax_lib) {
        const char *sdk_path = getenv("MACA_PATH");
        if (!sdk_path)
            sdk_path = getenv("MACA_HOME");
        if (sdk_path) {
            char p[512];
            snprintf(p, sizeof(p), "%s/lib/libsymbol_cu.so", sdk_path);
            _metax_lib = dlopen(p, RTLD_LAZY | RTLD_GLOBAL);
        }
    }
    if (!_metax_lib)
        _metax_lib = dlopen("/opt/maca-3.3.0/lib/libsymbol_cu.so", RTLD_LAZY | RTLD_GLOBAL);
    if (!_metax_lib)
        fprintf(stderr, "cudart_shim: cannot open libsymbol_cu.so: %s\n", dlerror());
}

/* ------------------------------------------------------------------ */
/* Macro: generate a forwarding wrapper                               */
/*                                                                    */
/* FORWARD(ret, name, params, args)                                   */
/*   ret    — return type                                             */
/*   name   — function name                                           */
/*   params — parameter list including parens                         */
/*   args   — argument forwarding list including parens               */
/* ------------------------------------------------------------------ */
#define FORWARD(ret, name, params, args)                               \
    ret name params {                                                  \
        static ret (*_fn) params = NULL;                               \
        if (!_fn) {                                                    \
            _ensure_metax_lib();                                        \
            if (_metax_lib) _fn = (ret (*) params) dlsym(_metax_lib, #name); \
            if (!_fn) {                                                \
                fprintf(stderr, "cudart_shim: %s not found\n", #name); \
                return (ret)0;                                         \
            }                                                          \
        }                                                              \
        return _fn args;                                               \
    }

/* Same but for void-returning functions */
#define FORWARD_VOID(name, params, args)                               \
    void name params {                                                 \
        static void (*_fn) params = NULL;                              \
        if (!_fn) {                                                    \
            _ensure_metax_lib();                                        \
            if (_metax_lib) _fn = (void (*) params) dlsym(_metax_lib, #name); \
        }                                                              \
        if (_fn) _fn args;                                             \
    }

/* For functions returning const char* */
#define FORWARD_STR(name, params, args)                                \
    const char* name params {                                          \
        static const char* (*_fn) params = NULL;                       \
        if (!_fn) {                                                    \
            _ensure_metax_lib();                                        \
            if (_metax_lib) _fn = (const char* (*) params) dlsym(_metax_lib, #name); \
            if (!_fn) return "unknown";                                \
        }                                                              \
        return _fn args;                                               \
    }


/* ================================================================== */
/* Part 1: 75 symbols forwarded to libsymbol_cu.so                    */
/* ================================================================== */

/* --- Device Management --- */
FORWARD(cudaError_t, cudaDeviceCanAccessPeer,
    (int *canAccessPeer, int device, int peerDevice),
    (canAccessPeer, device, peerDevice))

FORWARD(cudaError_t, cudaDeviceEnablePeerAccess,
    (int peerDevice, unsigned int flags),
    (peerDevice, flags))

FORWARD(cudaError_t, cudaDeviceGetAttribute,
    (int *value, cudaDeviceAttr attr, int device),
    (value, attr, device))

FORWARD(cudaError_t, cudaDeviceGetDefaultMemPool,
    (cudaMemPool_t *memPool, int device),
    (memPool, device))

FORWARD(cudaError_t, cudaDeviceGetPCIBusId,
    (char *pciBusId, int len, int device),
    (pciBusId, len, device))

FORWARD(cudaError_t, cudaDeviceGetStreamPriorityRange,
    (int *leastPriority, int *greatestPriority),
    (leastPriority, greatestPriority))

FORWARD(cudaError_t, cudaDeviceSynchronize, (void), ())

/* --- Version (override to report CUDA 12.8) --- */
cudaError_t cudaDriverGetVersion(int *driverVersion) {
    if (driverVersion) *driverVersion = 12080;
    return 0;
}

cudaError_t cudaRuntimeGetVersion(int *runtimeVersion) {
    if (runtimeVersion) *runtimeVersion = 12080;
    return 0;
}

/* Also override cuDriverGetVersion (driver API, used by some paths) */
int cuDriverGetVersion(int *driverVersion) {
    if (driverVersion) *driverVersion = 12080;
    return 0;
}

/* --- Event Management --- */
FORWARD(cudaError_t, cudaEventCreate,
    (cudaEvent_t *event),
    (event))

FORWARD(cudaError_t, cudaEventCreateWithFlags,
    (cudaEvent_t *event, unsigned int flags),
    (event, flags))

FORWARD(cudaError_t, cudaEventDestroy,
    (cudaEvent_t event),
    (event))

FORWARD(cudaError_t, cudaEventElapsedTime,
    (float *ms, cudaEvent_t start, cudaEvent_t end),
    (ms, start, end))

FORWARD(cudaError_t, cudaEventQuery,
    (cudaEvent_t event),
    (event))

FORWARD(cudaError_t, cudaEventRecord,
    (cudaEvent_t event, cudaStream_t stream),
    (event, stream))

FORWARD(cudaError_t, cudaEventRecordWithFlags,
    (cudaEvent_t event, cudaStream_t stream, unsigned int flags),
    (event, stream, flags))

FORWARD(cudaError_t, cudaEventSynchronize,
    (cudaEvent_t event),
    (event))

/* --- Memory Management --- */
FORWARD(cudaError_t, cudaFree,
    (void *devPtr),
    (devPtr))

FORWARD(cudaError_t, cudaFreeAsync,
    (void *devPtr, cudaStream_t hStream),
    (devPtr, hStream))

FORWARD(cudaError_t, cudaFreeHost,
    (void *ptr),
    (ptr))

FORWARD(cudaError_t, cudaHostAlloc,
    (void **pHost, size_t size, unsigned int flags),
    (pHost, size, flags))

FORWARD(cudaError_t, cudaHostRegister,
    (void *ptr, size_t size, unsigned int flags),
    (ptr, size, flags))

FORWARD(cudaError_t, cudaHostUnregister,
    (void *ptr),
    (ptr))

FORWARD(cudaError_t, cudaMalloc,
    (void **devPtr, size_t size),
    (devPtr, size))

FORWARD(cudaError_t, cudaMallocAsync,
    (void **devPtr, size_t size, cudaStream_t hStream),
    (devPtr, size, hStream))

FORWARD(cudaError_t, cudaMemGetInfo,
    (size_t *free_mem, size_t *total),
    (free_mem, total))

FORWARD(cudaError_t, cudaMemcpy,
    (void *dst, const void *src, size_t count, cudaMemcpyKind kind),
    (dst, src, count, kind))

FORWARD(cudaError_t, cudaMemcpyAsync,
    (void *dst, const void *src, size_t count, cudaMemcpyKind kind, cudaStream_t stream),
    (dst, src, count, kind, stream))

FORWARD(cudaError_t, cudaMemcpyPeerAsync,
    (void *dst, int dstDevice, const void *src, int srcDevice, size_t count, cudaStream_t stream),
    (dst, dstDevice, src, srcDevice, count, stream))

FORWARD(cudaError_t, cudaMemset,
    (void *devPtr, int value, size_t count),
    (devPtr, value, count))

FORWARD(cudaError_t, cudaMemsetAsync,
    (void *devPtr, int value, size_t count, cudaStream_t stream),
    (devPtr, value, count, stream))

/* --- Memory Pool --- */
FORWARD(cudaError_t, cudaMemPoolGetAttribute,
    (cudaMemPool_t memPool, cudaMemPoolAttr attr, void *value),
    (memPool, attr, value))

FORWARD(cudaError_t, cudaMemPoolSetAccess,
    (cudaMemPool_t memPool, const cudaMemAccessDesc *descList, size_t count),
    (memPool, descList, count))

FORWARD(cudaError_t, cudaMemPoolSetAttribute,
    (cudaMemPool_t memPool, cudaMemPoolAttr attr, void *value),
    (memPool, attr, value))

FORWARD(cudaError_t, cudaMemPoolTrimTo,
    (cudaMemPool_t memPool, size_t minBytesToKeep),
    (memPool, minBytesToKeep))

/* --- IPC --- */
FORWARD(cudaError_t, cudaIpcCloseMemHandle,
    (void *devPtr),
    (devPtr))

FORWARD(cudaError_t, cudaIpcGetEventHandle,
    (cudaIpcEventHandle_t *handle, cudaEvent_t event),
    (handle, event))

FORWARD(cudaError_t, cudaIpcGetMemHandle,
    (cudaIpcMemHandle_t *handle, void *devPtr),
    (handle, devPtr))

FORWARD(cudaError_t, cudaIpcOpenEventHandle,
    (cudaEvent_t *event, cudaIpcEventHandle_t handle),
    (event, handle))

FORWARD(cudaError_t, cudaIpcOpenMemHandle,
    (void **devPtr, cudaIpcMemHandle_t handle, unsigned int flags),
    (devPtr, handle, flags))

/* --- Execution Control --- */
FORWARD(cudaError_t, cudaFuncGetAttributes,
    (cudaFuncAttributes *attr, const void *func),
    (attr, func))

FORWARD(cudaError_t, cudaFuncSetAttribute,
    (const void *func, cudaFuncAttribute attr, int value),
    (func, attr, value))

FORWARD(cudaError_t, cudaLaunchKernel,
    (const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream),
    (func, gridDim, blockDim, args, sharedMem, stream))

FORWARD(cudaError_t, cudaLaunchKernelExC,
    (const cudaLaunchConfig_t *config, const void *func, void **args),
    (config, func, args))

FORWARD(cudaError_t, cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags,
    (int *numBlocks, const void *func, int blockSize, size_t dynamicSMemSize, unsigned int flags),
    (numBlocks, func, blockSize, dynamicSMemSize, flags))

/* --- Device Management (continued) --- */
FORWARD(cudaError_t, cudaGetDevice, (int *device), (device))
FORWARD(cudaError_t, cudaGetDeviceCount, (int *count), (count))
FORWARD(cudaError_t, cudaSetDevice, (int device), (device))

FORWARD(cudaError_t, cudaGetDriverEntryPoint,
    (const char *symbol, void **funcPtr, unsigned long long flags, cudaDriverEntryPointQueryResult *driverStatus),
    (symbol, funcPtr, flags, driverStatus))

/* --- Error Handling --- */
FORWARD_STR(cudaGetErrorName,
    (cudaError_t error),
    (error))

FORWARD_STR(cudaGetErrorString,
    (cudaError_t error),
    (error))

FORWARD(cudaError_t, cudaGetLastError, (void), ())
FORWARD(cudaError_t, cudaPeekAtLastError, (void), ())

/* --- Pointer Attributes --- */
FORWARD(cudaError_t, cudaPointerGetAttributes,
    (cudaPointerAttributes *attributes, const void *ptr),
    (attributes, ptr))

/* --- Stream Management --- */
FORWARD(cudaError_t, cudaStreamAddCallback,
    (cudaStream_t stream, cudaStreamCallback_t callback, void *userData, unsigned int flags),
    (stream, callback, userData, flags))

FORWARD(cudaError_t, cudaStreamBeginCapture,
    (cudaStream_t stream, cudaStreamCaptureMode mode),
    (stream, mode))

FORWARD(cudaError_t, cudaStreamCreateWithPriority,
    (cudaStream_t *pStream, unsigned int flags, int priority),
    (pStream, flags, priority))

FORWARD(cudaError_t, cudaStreamDestroy,
    (cudaStream_t stream),
    (stream))

FORWARD(cudaError_t, cudaStreamEndCapture,
    (cudaStream_t stream, cudaGraph_t *pGraph),
    (stream, pGraph))

FORWARD(cudaError_t, cudaStreamIsCapturing,
    (cudaStream_t stream, cudaStreamCaptureStatus *pCaptureStatus),
    (stream, pCaptureStatus))

FORWARD(cudaError_t, cudaStreamQuery,
    (cudaStream_t stream),
    (stream))

FORWARD(cudaError_t, cudaStreamSynchronize,
    (cudaStream_t stream),
    (stream))

FORWARD(cudaError_t, cudaStreamUpdateCaptureDependencies,
    (cudaStream_t stream, cudaGraphNode_t *dependencies, size_t numDependencies, unsigned int flags),
    (stream, dependencies, numDependencies, flags))

FORWARD(cudaError_t, cudaStreamWaitEvent,
    (cudaStream_t stream, cudaEvent_t event, unsigned int flags),
    (stream, event, flags))

FORWARD(cudaError_t, cudaThreadExchangeStreamCaptureMode,
    (cudaStreamCaptureMode *mode),
    (mode))

/* --- Graph Management --- */
FORWARD(cudaError_t, cudaGraphAddEmptyNode,
    (cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies),
    (pGraphNode, graph, pDependencies, numDependencies))

FORWARD(cudaError_t, cudaGraphDebugDotPrint,
    (cudaGraph_t graph, const char *path, unsigned int flags),
    (graph, path, flags))

FORWARD(cudaError_t, cudaGraphDestroy,
    (cudaGraph_t graph),
    (graph))

FORWARD(cudaError_t, cudaGraphExecDestroy,
    (cudaGraphExec_t graphExec),
    (graphExec))

FORWARD(cudaError_t, cudaGraphGetNodes,
    (cudaGraph_t graph, cudaGraphNode_t *nodes, size_t *numNodes),
    (graph, nodes, numNodes))

FORWARD(cudaError_t, cudaGraphInstantiate,
    (cudaGraphExec_t *pGraphExec, cudaGraph_t graph, unsigned long long flags),
    (pGraphExec, graph, flags))

FORWARD(cudaError_t, cudaGraphInstantiateWithFlags,
    (cudaGraphExec_t *pGraphExec, cudaGraph_t graph, unsigned long long flags),
    (pGraphExec, graph, flags))

FORWARD(cudaError_t, cudaGraphLaunch,
    (cudaGraphExec_t graphExec, cudaStream_t stream),
    (graphExec, stream))

FORWARD(cudaError_t, cudaGraphNodeGetDependencies,
    (cudaGraphNode_t node, cudaGraphNode_t *pDependencies, size_t *pNumDependencies),
    (node, pDependencies, pNumDependencies))

FORWARD(cudaError_t, cudaGraphNodeGetType,
    (cudaGraphNode_t node, cudaGraphNodeType *pType),
    (node, pType))

/* --- Additional forwarded symbols (from libtorch_python.so etc.) --- */
FORWARD(cudaError_t, cudaLaunchHostFunc,
    (cudaStream_t stream, void (*fn)(void*), void *userData),
    (stream, fn, userData))

FORWARD(cudaError_t, cudaMemcpy2DAsync,
    (void *dst, size_t dpitch, const void *src, size_t spitch,
     size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream),
    (dst, dpitch, src, spitch, width, height, kind, stream))

FORWARD(cudaError_t, cudaStreamCreate,
    (cudaStream_t *pStream),
    (pStream))

FORWARD(cudaError_t, cudaStreamGetPriority,
    (cudaStream_t hStream, int *priority),
    (hStream, priority))


/* ================================================================== */
/* Part 2: Stub symbols (not in libsymbol_cu.so)                      */
/* ================================================================== */

/* --- CUDA Kernel Registration (no-op on MetaX) --- */
void** __cudaRegisterFatBinary(void *fatCubin) {
    static void* dummy = NULL;
    (void)fatCubin;
    return &dummy;
}

void __cudaRegisterFatBinaryEnd(void **fatCubinHandle) {
    (void)fatCubinHandle;
}

void __cudaUnregisterFatBinary(void **fatCubinHandle) {
    (void)fatCubinHandle;
}

void __cudaRegisterFunction(
    void **fatCubinHandle, const char *hostFun, char *deviceFun,
    const char *deviceName, int thread_limit, void *tid, void *bid,
    void *bDim, void *gDim, int *wSize)
{
    (void)fatCubinHandle; (void)hostFun; (void)deviceFun;
    (void)deviceName; (void)thread_limit; (void)tid; (void)bid;
    (void)bDim; (void)gDim; (void)wSize;
}

void __cudaInitModule(void **fatCubinHandle) {
    (void)fatCubinHandle;
}

void __cudaRegisterVar(
    void **fatCubinHandle, char *hostVar, char *deviceAddress,
    const char *deviceName, int ext, size_t size, int constant, int global)
{
    (void)fatCubinHandle; (void)hostVar; (void)deviceAddress;
    (void)deviceName; (void)ext; (void)size; (void)constant; (void)global;
}

unsigned __cudaPushCallConfiguration(
    unsigned gridDimX, unsigned gridDimY, unsigned gridDimZ,
    unsigned blockDimX, unsigned blockDimY, unsigned blockDimZ,
    size_t sharedMem, void *stream)
{
    (void)gridDimX; (void)gridDimY; (void)gridDimZ;
    (void)blockDimX; (void)blockDimY; (void)blockDimZ;
    (void)sharedMem; (void)stream;
    return 0;
}

unsigned __cudaPopCallConfiguration(
    void *gridDim, void *blockDim, size_t *sharedMem, void *stream)
{
    (void)gridDim; (void)blockDim; (void)sharedMem; (void)stream;
    return 0;
}

/* --- cudaGetDeviceProperties_v2: return zeroed struct (Python-level
 *     patches in _metax_compat.py provide real values via MetaX API) --- */
cudaError_t cudaGetDeviceProperties_v2(cudaDeviceProp *prop, int device) {
    (void)device;
    if (prop) memset(prop, 0, sizeof(*prop));
    return 0;
}

/* --- cudaGetDriverEntryPointByVersion: CUDA 12 API, return success
 *     with NULL pointer (no driver entry points available on MetaX) --- */
cudaError_t cudaGetDriverEntryPointByVersion(
    const char *symbol, void **funcPtr, unsigned int cudaVersion,
    unsigned long long flags, void *symbolStatus)
{
    (void)symbol; (void)cudaVersion; (void)flags; (void)symbolStatus;
    if (funcPtr) *funcPtr = NULL;
    return 0;
}

/* --- cudaStreamGetCaptureInfo_v2: return "not capturing" --- */
cudaError_t cudaStreamGetCaptureInfo_v2(
    cudaStream_t stream, cudaStreamCaptureStatus *captureStatus_out,
    unsigned long long *id_out, cudaGraph_t *graph_out,
    const cudaGraphNode_t **dependencies_out, size_t *numDependencies_out)
{
    (void)stream;
    if (captureStatus_out) *captureStatus_out = 0;  /* cudaStreamCaptureStatusNone */
    if (id_out) *id_out = 0;
    if (graph_out) *graph_out = NULL;
    if (dependencies_out) *dependencies_out = NULL;
    if (numDependencies_out) *numDependencies_out = 0;
    return 0;
}

/* --- Profiler (no-op) --- */
cudaError_t cudaProfilerStart(void) { return 0; }
cudaError_t cudaProfilerStop(void) { return 0; }
