# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set(LIBRARY_NAME torch_fl)

option(CUDA_KERNEL "Build CUDA kernel implementations" ON)
option(FLAGGEMS_KERNEL "Build FlagGems C++ kernel wrappers (requires liboperators.so)" ON)
option(FLAGGEMS_PYTHON "Build FlagGems Python kernel wrappers (calls via Python, no C++ linking)" ON)
option(TILEOPS_KERNEL "Build TileOPs kernel wrappers (calls via Python, no C++ linking)" ON)
option(METAX_KERNEL "Build MetaX kernel implementations" OFF)
option(ASCEND_KERNEL "Build Ascend kernel implementations" OFF)
option(GCU_KERNEL "Build Enflame GCU (topsaten) kernel implementations" OFF)
option(MUSA_KERNEL "Build Moore Threads MUSA kernel implementations (mudnn)" OFF)

file(GLOB_RECURSE SOURCE_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cu"
)

# Exclude runtime/accelerator (built separately as subdirectory)
list(FILTER SOURCE_FILES EXCLUDE REGEX ".*/runtime/accelerator/.*")
# stream_api is built as a separate shared lib
list(FILTER SOURCE_FILES EXCLUDE REGEX ".*/runtime/accelerator/ascend/stream_api\\.cc")

# Select exactly one vendor-specific DeviceTracer implementation. CUDA and PPU
# use the CUPTI-compatible tracer, MetaX uses that same tracer against MCPTI,
# DCU uses ROCtracer, Ascend uses CANN MSPTI, and MUSA uses MUPTI. Platforms
# without a supported activity API use the unavailable implementation. Each
# source defines the same factory.
if(ACCELERATOR STREQUAL "cuda" OR ACCELERATOR STREQUAL "metax")
  list(FILTER SOURCE_FILES EXCLUDE REGEX ".*/profiler/roctracer_device_tracer\\.cc$")
  list(FILTER SOURCE_FILES EXCLUDE REGEX ".*/profiler/cann_device_tracer\\.cc$")
  list(FILTER SOURCE_FILES EXCLUDE REGEX ".*/profiler/musa_mupti_device_tracer\\.cc$")
  list(FILTER SOURCE_FILES EXCLUDE REGEX ".*/profiler/gcu_topspti_device_tracer\\.cc$")
  list(FILTER SOURCE_FILES EXCLUDE REGEX ".*/profiler/unavailable_device_tracer\\.cc$")
elseif(ACCELERATOR STREQUAL "dcu")
  list(FILTER SOURCE_FILES EXCLUDE REGEX ".*/profiler/cupti_device_tracer\\.cc$")
  list(FILTER SOURCE_FILES EXCLUDE REGEX ".*/profiler/cann_device_tracer\\.cc$")
  list(FILTER SOURCE_FILES EXCLUDE REGEX ".*/profiler/musa_mupti_device_tracer\\.cc$")
  list(FILTER SOURCE_FILES EXCLUDE REGEX ".*/profiler/gcu_topspti_device_tracer\\.cc$")
  list(FILTER SOURCE_FILES EXCLUDE REGEX ".*/profiler/unavailable_device_tracer\\.cc$")
elseif(ACCELERATOR STREQUAL "ascend")
  list(FILTER SOURCE_FILES EXCLUDE REGEX ".*/profiler/cupti_device_tracer\\.cc$")
  list(FILTER SOURCE_FILES EXCLUDE REGEX ".*/profiler/roctracer_device_tracer\\.cc$")
  list(FILTER SOURCE_FILES EXCLUDE REGEX ".*/profiler/musa_mupti_device_tracer\\.cc$")
  list(FILTER SOURCE_FILES EXCLUDE REGEX ".*/profiler/gcu_topspti_device_tracer\\.cc$")
  list(FILTER SOURCE_FILES EXCLUDE REGEX ".*/profiler/unavailable_device_tracer\\.cc$")
elseif(ACCELERATOR STREQUAL "musa")
  list(FILTER SOURCE_FILES EXCLUDE REGEX ".*/profiler/cupti_device_tracer\\.cc$")
  list(FILTER SOURCE_FILES EXCLUDE REGEX ".*/profiler/roctracer_device_tracer\\.cc$")
  list(FILTER SOURCE_FILES EXCLUDE REGEX ".*/profiler/cann_device_tracer\\.cc$")
  list(FILTER SOURCE_FILES EXCLUDE REGEX ".*/profiler/unavailable_device_tracer\\.cc$")
  list(FILTER SOURCE_FILES EXCLUDE REGEX ".*/profiler/gcu_topspti_device_tracer\\.cc$")
elseif(ACCELERATOR STREQUAL "gcu")
  list(FILTER SOURCE_FILES EXCLUDE REGEX ".*/profiler/cupti_device_tracer\\.cc$")
  list(FILTER SOURCE_FILES EXCLUDE REGEX ".*/profiler/roctracer_device_tracer\\.cc$")
  list(FILTER SOURCE_FILES EXCLUDE REGEX ".*/profiler/cann_device_tracer\\.cc$")
  list(FILTER SOURCE_FILES EXCLUDE REGEX ".*/profiler/musa_mupti_device_tracer\\.cc$")
  list(FILTER SOURCE_FILES EXCLUDE REGEX ".*/profiler/unavailable_device_tracer\\.cc$")
else()
  list(FILTER SOURCE_FILES EXCLUDE REGEX ".*/profiler/cupti_device_tracer\\.cc$")
  list(FILTER SOURCE_FILES EXCLUDE REGEX ".*/profiler/roctracer_device_tracer\\.cc$")
  list(FILTER SOURCE_FILES EXCLUDE REGEX ".*/profiler/cann_device_tracer\\.cc$")
  list(FILTER SOURCE_FILES EXCLUDE REGEX ".*/profiler/musa_mupti_device_tracer\\.cc$")
  list(FILTER SOURCE_FILES EXCLUDE REGEX ".*/profiler/gcu_topspti_device_tracer\\.cc$")
endif()

# CANN's public MSPTI headers are optional. The tracer is still compiled with a
# small unavailable implementation when a CPU-only build has no toolkit, while
# the Python import path likewise treats libmspti.so as optional.
if(ACCELERATOR STREQUAL "ascend")
  find_path(MSPTI_INCLUDE_DIR
    NAMES mspti.h
    PATHS
      "$ENV{ASCEND_HOME}/tools/mspti/include"
      "$ENV{ASCEND_HOME_PATH}/tools/mspti/include"
      /usr/local/Ascend/ascend-toolkit/latest/tools/mspti/include
      /usr/local/Ascend/cann-9.0.0/tools/mspti/include)
  if(MSPTI_INCLUDE_DIR)
    include_directories(${MSPTI_INCLUDE_DIR})
    add_compile_definitions(FLAGOS_HAVE_MSPTI=1)
    message(STATUS "mspti.h found: ${MSPTI_INCLUDE_DIR}")
  else()
    message(STATUS "mspti.h NOT found; Ascend profiler will be unavailable")
  endif()
elseif(ACCELERATOR STREQUAL "musa")
  find_path(MUPTI_INCLUDE_DIR
    NAMES mupti_activity.h
    PATHS
      "$ENV{MUSA_HOME}/include"
      /usr/local/musa/include
      /usr/local/musa-5.1.0/include)
  if(MUPTI_INCLUDE_DIR)
    include_directories(${MUPTI_INCLUDE_DIR})
    add_compile_definitions(FLAGOS_HAVE_MUPTI=1)
    message(STATUS "mupti_activity.h found: ${MUPTI_INCLUDE_DIR}")
  else()
    message(STATUS "mupti_activity.h NOT found; MUSA profiler will be unavailable")
  endif()
elseif(ACCELERATOR STREQUAL "gcu")
  find_path(TOPSPTI_INCLUDE_DIR
    NAMES topspti_activity.h
    PATHS
      "$ENV{TOPS_HOME}/extras/TOPSPTI/include"
      "$ENV{TOPS_HOME}/include/gcu/topspti"
      /opt/tops/extras/TOPSPTI/include
      /opt/tops/include/gcu/topspti)
  if(TOPSPTI_INCLUDE_DIR)
    include_directories(${TOPSPTI_INCLUDE_DIR})
    add_compile_definitions(FLAGOS_HAVE_TOPSPTI=1)
    message(STATUS "topspti_activity.h found: ${TOPSPTI_INCLUDE_DIR}")
  else()
    message(STATUS "topspti_activity.h NOT found; GCU profiler will be unavailable")
  endif()
endif()

# Exclude backend kernel directories based on options
if(NOT CUDA_KERNEL)
  list(FILTER SOURCE_FILES EXCLUDE REGEX ".*/aten/backends/cuda/.*")
endif()
# FLAGGEMS_PYTHON path: keep csrc/aten/backends/flagos/python_op_caller.cc and
# build csrc/aten/generated/flaggems_python_kernels.cc with FLAGOS_FLAGGEMS_PYTHON
# defined (below). The stale C++/python_wrapper files were removed; nothing to
# exclude here.
#
# The flagos backend (python_op_caller.{h,cc}) is excluded further down, once the
# per-accelerator rules have settled whether generated/cuda_kernels.cc is in the
# build -- it calls GetFlagosDefaultCudaGenerator(), which python_op_caller.cc
# defines.
if(NOT METAX_KERNEL)
  list(FILTER SOURCE_FILES EXCLUDE REGEX ".*/aten/backends/metax/.*")
endif()
if(NOT ASCEND_KERNEL)
  list(FILTER SOURCE_FILES EXCLUDE REGEX ".*/aten/backends/ascend/.*")
endif()
if(NOT GCU_KERNEL)
  list(FILTER SOURCE_FILES EXCLUDE REGEX ".*/aten/backends/gcu/.*")
endif()
if(NOT MUSA_KERNEL)
  list(FILTER SOURCE_FILES EXCLUDE REGEX ".*/aten/backends/musa/.*")
endif()

# The DCU SDK-native operators are always a SEPARATE artifact
# (libdcu_aten_ops.so), never part of libtorch_fl.so -- that separation is the
# whole point: the plugin links the official core plus the DTK SDK and must not
# inherit libtorch_fl.so's link line. The recursive glob above would otherwise
# pull these sources in on every platform, so filter them unconditionally. The
# plugin target is defined further down, guarded by ACCELERATOR=dcu.
# dcu_sdk_registration.cc (the core-side bridge) deliberately lives outside this
# directory, in csrc/aten/, so it stays in libtorch_fl.so.
list(FILTER SOURCE_FILES EXCLUDE REGEX ".*/aten/backends/dcu_sdk/.*")

if(ACCELERATOR STREQUAL "metax")
  list(FILTER SOURCE_FILES EXCLUDE REGEX ".*/aten/backends/cuda/.*")
  list(FILTER SOURCE_FILES EXCLUDE REGEX ".*/aten/backends/metax/.*\\.cu$")
endif()

if(ACCELERATOR STREQUAL "gcu")
  # The generated CUDA boxing kernels re-dispatch PrivateUse1 -> CUDA, which
  # needs a CUDA-capable libtorch. GCU has no CUDA runtime at all (the tops
  # stack exports no cuda symbols), so drop them along with the CUDA backend.
  list(FILTER SOURCE_FILES EXCLUDE REGEX ".*/aten/backends/cuda/.*")
  list(FILTER SOURCE_FILES EXCLUDE REGEX ".*/aten/generated/cuda_kernels\\.cc$")
endif()

if(ACCELERATOR STREQUAL "musa")
  # Same reasoning as gcu: torch_musa registers under PrivateUse1, never the
  # CUDA key, and the MUSA toolkit exports no cuda symbols -- so the generated
  # CUDA boxing kernels have nothing to dispatch into. The musa wrappers under
  # aten/backends/musa/generated/ replace them.
  list(FILTER SOURCE_FILES EXCLUDE REGEX ".*/aten/backends/cuda/.*")
  list(FILTER SOURCE_FILES EXCLUDE REGEX ".*/aten/generated/cuda_kernels\\.cc$")
endif()

# Drop the flagos backend (only python_op_caller.{h,cc} lives here) when nothing
# left in SOURCE_FILES needs it. Three independent consumers:
#
#   * the FlagGems dispatch paths (FLAGGEMS_KERNEL / FLAGGEMS_PYTHON), which call
#     into Python through it;
#   * the TileOPs path (TILEOPS_KERNEL), which reaches its TileLang kernels the
#     same way;
#   * generated/cuda_kernels.cc, whose native RNG kernels call
#     GetFlagosDefaultCudaGenerator() unconditionally since the unified-RNG
#     generator injection, regardless of the FlagGems dispatch path;
#   * the MetaX boxing build, whose vendor runtime calls the same provider even
#     with FlagGems off.
#
# The second consumer is why this test runs *here* and looks at the source list
# instead of at CUDA_KERNEL: cuda_kernels.cc is generated code under
# aten/generated/, so the CUDA_KERNEL option does not govern it -- the gcu/musa
# blocks above do, by accelerator. Keying on CUDA_KERNEL instead left dcu
# (CUDA_KERNEL=OFF, both FLAGGEMS_* OFF, cuda_kernels.cc still compiled, since
# the DTK torch wheel is hipified and registers HIP kernels under the CUDA key)
# linking with an undefined GetFlagosDefaultCudaGenerator.
set(_needs_python_op_caller OFF)
if(FLAGGEMS_KERNEL OR FLAGGEMS_PYTHON OR TILEOPS_KERNEL)
  set(_needs_python_op_caller ON)
endif()
if(ACCELERATOR STREQUAL "metax" AND FLAGOS_METAX_BOXING)
  set(_needs_python_op_caller ON)
endif()
if(ACCELERATOR STREQUAL "gcu")
  set(_needs_python_op_caller ON)
endif()
foreach(_src IN LISTS SOURCE_FILES)
  if(_src MATCHES ".*/aten/generated/cuda_kernels\\.cc$")
    set(_needs_python_op_caller ON)
  endif()
endforeach()
if(NOT _needs_python_op_caller)
  list(FILTER SOURCE_FILES EXCLUDE REGEX ".*/aten/backends/flagos/.*")
endif()

set(_torch_fl_sources ${SOURCE_FILES})

if(ACCELERATOR STREQUAL "metax" AND METAX_KERNEL)
  include(${CMAKE_SOURCE_DIR}/cmake/MetaxKernels.cmake)
  flagos_add_metax_kernel_objects(_flagos_metax_kernel_objs)
  list(APPEND _torch_fl_sources ${_flagos_metax_kernel_objs})
endif()

add_library(${LIBRARY_NAME} SHARED ${_torch_fl_sources})
set_target_properties(${LIBRARY_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)

# CUPTI headers for the profiler child (Stage B). Runtime symbols are dlopen'd,
# so we only need the include path, not the link library.
#
# CRITICAL: the header MUST match the CUDA runtime actually running in the
# process (cu12.8 via the external libtorch_cuda.so + pip nvidia-*-cu12 wheels),
# NOT the system CUDA-13.0 toolkit. The activity-record struct layouts and the
# CUpti_ActivityKind enum values DIFFER between major versions (e.g.
# CONCURRENT_KERNEL is 10 in cu12 but 9 elsewhere); using the wrong header
# decodes kernel timestamps/names as garbage. So probe the pip cu12 CUPTI
# include dir first, then fall back to system paths.
execute_process(
  COMMAND "${PYTHON_EXECUTABLE}" -c
    "import importlib.util,os,sys; s=importlib.util.find_spec('nvidia'); \
d=[os.path.join(b,'cuda_cupti','include') for b in (s.submodule_search_locations or []) \
if os.path.exists(os.path.join(b,'cuda_cupti','include','cupti_activity.h'))]; \
sys.stdout.write(d[0] if d else '')"
  OUTPUT_VARIABLE CUPTI_CU12_INCLUDE_DIR
  ERROR_QUIET)
find_path(CUPTI_INCLUDE_DIR cupti_activity.h
  PATHS ${CUPTI_CU12_INCLUDE_DIR}
        /usr/local/cuda-13.0/targets/x86_64-linux/include
        /usr/local/cuda/extras/CUPTI/include
  NO_DEFAULT_PATH)
find_path(CUPTI_INCLUDE_DIR cupti_activity.h
  PATHS /usr/local/cuda-13.0/targets/x86_64-linux/include
        /usr/local/cuda/extras/CUPTI/include)
if(CUPTI_INCLUDE_DIR)
  target_include_directories(${LIBRARY_NAME} PRIVATE ${CUPTI_INCLUDE_DIR})
  if(ACCELERATOR STREQUAL "cuda" OR ACCELERATOR STREQUAL "metax")
    target_compile_definitions(${LIBRARY_NAME} PRIVATE FLAGOS_HAVE_CUPTI=1)
  endif()
endif()

# cuda_occupancy.h -- CUDA's occupancy calculator, used by the device tracer to
# compute the kernel "est. achieved occupancy %" that torch-cuda's trace carries.
#
# Unlike the CUPTI headers above this is genuinely version-INsensitive: it is a
# pure-header calculator with no linking, no CUDA context, and no dependence on
# activity-record layouts -- it takes device attributes and a launch config as
# plain integers. So any toolkit's copy will do, and no cu12-vs-cu13 hazard
# applies. Note it ships in the CUDA *runtime* include dir, NOT alongside the
# CUPTI headers, hence a separate search.
#
# Optional by design: without it we simply omit the three occupancy fields
# rather than fail the build, so a machine with no CUDA toolkit headers still
# gets everything else in the trace.
set(CUDA_OCCUPANCY_SEARCH_PATHS
  "$ENV{CUDA_HOME}/include"
  "$ENV{CUDA_HOME}/targets/x86_64-linux/include"
  /usr/local/cuda/include
  /usr/local/cuda/targets/x86_64-linux/include
  /usr/local/cuda-13.0/targets/x86_64-linux/include)
if(ACCELERATOR STREQUAL "metax")
  list(APPEND CUDA_OCCUPANCY_SEARCH_PATHS
    "${METAX_PATH}/tools/cu-bridge/include")
endif()
find_path(CUDA_OCCUPANCY_INCLUDE_DIR cuda_occupancy.h
  PATHS ${CUDA_OCCUPANCY_SEARCH_PATHS})
if(CUDA_OCCUPANCY_INCLUDE_DIR)
  target_include_directories(${LIBRARY_NAME} PRIVATE ${CUDA_OCCUPANCY_INCLUDE_DIR})
  target_compile_definitions(${LIBRARY_NAME} PRIVATE FLAGOS_HAVE_CUDA_OCCUPANCY=1)
  message(STATUS "cuda_occupancy.h found: ${CUDA_OCCUPANCY_INCLUDE_DIR}")
else()
  message(STATUS "cuda_occupancy.h NOT found; kernel occupancy fields will be omitted")
endif()

# ROCtracer headers for DCU profiler support. Similar to CUPTI above, we only
# need headers at build time; runtime symbols are resolved via dlopen.
#
# ROCtracer ships with the DTK (DCU ToolKit) at /opt/dtk. The headers live in
# /opt/dtk/roctracer/include, and the library is at /opt/dtk/lib/libroctracer64.so.
# Unlike CUPTI's version-sensitivity, ROCtracer 4.1 is stable across DTK releases.
if(ACCELERATOR STREQUAL "dcu")
  # In SDK-only mode, exclude CUDA boxing kernels that would pull in DTK's
  # libtorch_hip.so as a hard dependency. The official CPU wheel has no CUDA
  # runtime, and the SDK plugin (libdcu_aten_ops.so) implements the GEMM
  # cohort independently without requiring libtorch_hip.
  list(FILTER SOURCE_FILES EXCLUDE REGEX ".*/aten/backends/cuda/.*")
  list(FILTER SOURCE_FILES EXCLUDE REGEX ".*/aten/generated/cuda_kernels\\.cc$")

  find_path(ROCTRACER_INCLUDE_DIR
    NAMES roctracer.h
    PATHS
      $ENV{ROCM_PATH}/roctracer/include
      $ENV{DTK_ROOT}/roctracer/include
      /opt/dtk/roctracer/include
      /opt/rocm/roctracer/include
    NO_DEFAULT_PATH)
  find_path(ROCTRACER_INCLUDE_DIR
    NAMES roctracer.h
    PATHS
      /opt/dtk/roctracer/include
      /opt/rocm/roctracer/include)

  if(ROCTRACER_INCLUDE_DIR)
    target_include_directories(${LIBRARY_NAME} PRIVATE ${ROCTRACER_INCLUDE_DIR})
    target_compile_definitions(${LIBRARY_NAME} PRIVATE FLAGOS_HAVE_ROCTRACER=1)
    target_compile_definitions(${LIBRARY_NAME} PRIVATE __HIP_PLATFORM_AMD__=1)
    message(STATUS "roctracer.h found: ${ROCTRACER_INCLUDE_DIR}")

    # Find the roctracer library for linking
    find_library(ROCTRACER_LIBRARY
      NAMES roctracer64
      PATHS
        $ENV{ROCM_PATH}/lib
        $ENV{DTK_ROOT}/lib
        /opt/dtk/lib
        /opt/rocm/lib
      NO_DEFAULT_PATH)
    find_library(ROCTRACER_LIBRARY
      NAMES roctracer64
      PATHS
        /opt/dtk/lib
        /opt/rocm/lib)

    if(ROCTRACER_LIBRARY)
      message(STATUS "libroctracer64.so found: ${ROCTRACER_LIBRARY}")
    else()
      message(WARNING "libroctracer64.so NOT found; profiler may fail at runtime")
    endif()
  else()
    message(STATUS "roctracer.h NOT found; DCU profiler will be unavailable")
  endif()
endif()

# Build stream_api as a standalone shared lib so liboperators.so can resolve it at runtime
if(FLAGGEMS_KERNEL AND ACCELERATOR STREQUAL "ascend")
  add_library(stream_api SHARED "${CMAKE_CURRENT_SOURCE_DIR}/runtime/accelerator/ascend/stream_api.cc")
  target_compile_definitions(stream_api PRIVATE USE_ASCEND=1)
  target_include_directories(stream_api PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${ASCEND_HOME}/include
  )
  find_library(ASCENDCL_LIB ascendcl
      PATHS ${ASCEND_HOME}/lib64)
  target_link_libraries(stream_api PRIVATE torch_cpu_library ${ASCENDCL_LIB})
  install(TARGETS stream_api
      LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )
endif()

target_compile_definitions(${LIBRARY_NAME} PRIVATE
    FLAGOS_SOURCE_ROOT="${CMAKE_SOURCE_DIR}"
)

# Compile the generated FlagGems Python-path kernels (guarded by this macro) and
# python_op_caller only when the Python FlagGems path is requested.
if(FLAGGEMS_PYTHON)
  target_compile_definitions(${LIBRARY_NAME} PRIVATE FLAGOS_FLAGGEMS_PYTHON)
endif()

# Compile the FlagGems C++ dispatch kernels (flaggems_cpp_kernels.cc, guarded by
# this macro) only when linking against liboperators.so is requested.
if(FLAGGEMS_KERNEL)
  target_compile_definitions(${LIBRARY_NAME} PRIVATE FLAGOS_FLAGGEMS_CPP)
endif()

# Compile the generated TileOPs kernels (guarded by this macro). Like the
# FlagGems Python path they call into the interpreter via python_op_caller, so
# there is nothing to link against -- TileOPs is a pure-Python (TileLang)
# package, and its absence at runtime is handled by the shims falling back to
# aten. Building them in costs nothing when unused: the conf files decide
# whether any op actually routes to Backend::kTileOps.
if(TILEOPS_KERNEL)
  target_compile_definitions(${LIBRARY_NAME} PRIVATE FLAGOS_TILEOPS)
endif()

# The CPU-only torch wheel ships c10/cuda headers but not the CUDA-build-only
# c10/cuda/impl/cuda_cmake_macros.h. cuda_memory.h includes CUDACachingAllocator.h
# to delegate flagos allocation to the CUDA caching allocator; this macro tells
# CUDAMacros.h to skip that generated configure file. Not needed for Ascend
# (which does not include c10/cuda headers).
if(NOT ACCELERATOR STREQUAL "ascend")
  target_compile_definitions(${LIBRARY_NAME} PRIVATE C10_CUDA_NO_CMAKE_CONFIGURE_FILE)
endif()

if(ACCELERATOR STREQUAL "metax")
  # maca's torch headers are a hard fork gated on USE_MACA (e.g. C10_WARP_SIZE,
  # Context.h allow_tf32_cudnn). Without it the headers hit static_assert(0).
  # MCPTI callback ids use a different namespace from NVIDIA CUPTI ids.
  target_compile_definitions(${LIBRARY_NAME} PRIVATE
    USE_MACA=1
    FLAGOS_METAX_MCPTI=1
    FLAGOS_SOFT_LOWP=1)
elseif(ACCELERATOR STREQUAL "dcu")
  target_compile_definitions(${LIBRARY_NAME} PRIVATE FLAGOS_SOFT_LOWP=1)
endif()

set(_torch_fl_link_libs torch_cpu_library torch_python_library flagos ${CUDA_RUNTIME_LIB})
if(FLAGGEMS_KERNEL)
  if(TARGET FlagGems::operators)
    list(APPEND _torch_fl_link_libs FlagGems::operators)
  elseif(TARGET operators)
    list(APPEND _torch_fl_link_libs operators)
  endif()
endif()

if(ACCELERATOR STREQUAL "metax" AND FLAGOS_METAX_BOXING)
  # The CPU TorchConfig used for the isolated Python environment does not
  # declare MetaX's forked CUDA runtime. Link it directly so vendor-only ATen
  # symbols are resolved by ELF instead of being left to process-global lookup.
  set(_metax_torch_lib "$ENV{FLAGOS_MACA_TORCH_LIB}")
  if(IS_DIRECTORY "${_metax_torch_lib}")
    if(EXISTS "${_metax_torch_lib}/libtorch_cuda.so")
      list(APPEND _torch_fl_link_libs
        "${_metax_torch_lib}/libtorch_cuda.so")
      target_link_options(${LIBRARY_NAME} PRIVATE "-Wl,--no-as-needed")
    endif()
  endif()
endif()

if(ACCELERATOR STREQUAL "ascend")
  target_compile_definitions(${LIBRARY_NAME} PRIVATE USE_ASCEND=1)
  find_library(OPAPI_LIB opapi
      PATHS $ENV{ASCEND_HOME}/lib64
            /usr/local/Ascend/ascend-toolkit/latest/lib64)
  find_library(NNOPBASE_LIB nnopbase
      PATHS $ENV{ASCEND_HOME}/lib64
            /usr/local/Ascend/ascend-toolkit/latest/lib64)
  target_link_libraries(${LIBRARY_NAME} PRIVATE ${_torch_fl_link_libs})
  if(OPAPI_LIB)
    target_link_libraries(${LIBRARY_NAME} PRIVATE ${OPAPI_LIB})
  endif()
  if(NNOPBASE_LIB)
    target_link_libraries(${LIBRARY_NAME} PRIVATE ${NNOPBASE_LIB})
  endif()
elseif(ACCELERATOR STREQUAL "tsingmicro")
  target_compile_definitions(${LIBRARY_NAME} PRIVATE USE_TSINGMICRO=1)
  target_link_libraries(${LIBRARY_NAME} PRIVATE ${_torch_fl_link_libs})
elseif(ACCELERATOR STREQUAL "dcu")
  # Selects backends/dcu_memory.h (CUDA-compat runtime, no c10::cuda caching
  # delegation -- the DCU wheel exports c10::hip, not c10::cuda).
  target_compile_definitions(${LIBRARY_NAME} PRIVATE USE_DCU=1)
  target_link_libraries(${LIBRARY_NAME} PRIVATE ${_torch_fl_link_libs})
  # Link roctracer library for profiler support
  if(ROCTRACER_LIBRARY)
    target_link_libraries(${LIBRARY_NAME} PRIVATE ${ROCTRACER_LIBRARY})
  endif()
elseif(ACCELERATOR STREQUAL "musa")
  target_compile_definitions(${LIBRARY_NAME} PRIVATE USE_MUSA=1)
  target_link_libraries(${LIBRARY_NAME} PRIVATE ${_torch_fl_link_libs})
  if(MUSA_KERNEL)
    # FLAGOS_MUSA_KERNEL gates the m.impl subset in register.cc, mirroring
    # FLAGOS_GCU_KERNEL.
    target_compile_definitions(${LIBRARY_NAME} PRIVATE FLAGOS_MUSA_KERNEL=1)
    # mudnn's headers and library both live in the MUSA toolkit, already on the
    # include path via cuda_runtime_compat. Nothing vendor-specific to stage: no
    # torch_musa tree, and so none of the header-shadowing hazards it brought
    # (it re-shipped ~1341 of torch's own ATen/ops/*_ops.h frozen at the
    # vendor's torch version).
    target_link_libraries(${LIBRARY_NAME} PRIVATE ${MUDNN_LIB} ${MURAND_LIB})
  endif()
elseif(ACCELERATOR STREQUAL "bpu")
  # Selects backends/bpu_memory.h and, like tsingmicro, skips the generated
  # PrivateUse1 op registrations: the BPU executes whole compiled graphs, so
  # there is no per-op kernel to register and every aten call reaches
  # cpu_fallback. Acceleration comes from the torch.compile backend instead.
  target_compile_definitions(${LIBRARY_NAME} PRIVATE USE_BPU=1)
  target_link_libraries(${LIBRARY_NAME} PRIVATE ${_torch_fl_link_libs})
  find_path(HOBOT_UCP_INCLUDE_DIR hb_ucp_sys.h
      PATHS ${HOBOT_INCLUDE_DIR} /usr/include/hobot)
  if(HOBOT_UCP_INCLUDE_DIR)
    target_include_directories(${LIBRARY_NAME} PRIVATE
        ${HOBOT_UCP_INCLUDE_DIR} ${HOBOT_UCP_INCLUDE_DIR}/dnn)
  endif()
elseif(ACCELERATOR STREQUAL "gcu")
  target_compile_definitions(${LIBRARY_NAME} PRIVATE USE_GCU=1)
  target_link_libraries(${LIBRARY_NAME} PRIVATE ${_torch_fl_link_libs})
  if(GCU_KERNEL)
    # FLAGOS_GCU_KERNEL gates the m.impl subset in register.cc: without the
    # topsaten kernels there is nothing to register on PrivateUse1.
    target_compile_definitions(${LIBRARY_NAME} PRIVATE FLAGOS_GCU_KERNEL=1)
    target_include_directories(${LIBRARY_NAME} PRIVATE
        ${TOPSATEN_INCLUDE_DIRS})
    target_link_libraries(${LIBRARY_NAME} PRIVATE ${TOPSATEN_LIB})
  endif()
else()
  target_link_libraries(${LIBRARY_NAME} PRIVATE ${_torch_fl_link_libs})
endif()

target_include_directories(${LIBRARY_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if(ACCELERATOR STREQUAL "metax")
  target_include_directories(${LIBRARY_NAME} PRIVATE
    "${METAX_PATH}/tools/cu-bridge/include"
    "${METAX_PATH}/include"
    "${METAX_PATH}/include/mcr"
    "${CMAKE_CURRENT_SOURCE_DIR}/aten/backends/metax")

  find_library(MCBLAS_LIB mcblas
      PATHS "${METAX_PATH}/lib"
            "${METAX_PATH}/lib64"
            "/opt/maca/lib"
            "/opt/maca/lib64")
  if(MCBLAS_LIB)
    target_link_libraries(${LIBRARY_NAME} PRIVATE ${MCBLAS_LIB})
  else()
    message(WARNING "mcblas not found, MetaX mm will fallback to naive kernel")
  endif()

endif()

# Self-contained wheel: the installed libtorch_fl.so must NOT bake the build
# machine's absolute torch/lib path into its RUNPATH -- that path does not exist
# on a target machine, and on one where it does exist it silently overrides the
# interpreter actually running. Resolve a bundled forked libtorch via $ORIGIN
# instead. Everything else the backend put on CMAKE_INSTALL_RPATH is kept,
# including FlagGems' liboperators.so dir, so the C++ FlagGems path
# (FLAGGEMS_KERNEL, kFlagOs) still resolves without LD_LIBRARY_PATH. Mirrors the
# patchelf pass in scripts/bundle_*_libtorch.sh.
flagos_set_portable_rpath(${LIBRARY_NAME})

install(TARGETS ${LIBRARY_NAME}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

# DCU SDK-native operator plugin. Keep this target independent from the main
# torch_fl library: it is loaded only when FLAGOS_DCU_SDK_OPS=1 and must not
# acquire torch_fl's CUDA boxing/device-library dependencies transitively.
if(ACCELERATOR STREQUAL "dcu")
  option(FLAGOS_DCU_SDK_OPS_BUILD
      "Build the DTK SDK-native DCU operator plugin" ON)
  if(FLAGOS_DCU_SDK_OPS_BUILD)
    set(_dcu_sdk_root "${FLAGOS_DCU_SDK_ROOT}")
    if(NOT _dcu_sdk_root)
      set(_dcu_sdk_root "$ENV{FLAGOS_DCU_SDK_ROOT}")
    endif()
    if(NOT _dcu_sdk_root)
      set(_dcu_sdk_root "${DTK_ROOT}")
    endif()
    if(NOT _dcu_sdk_root)
      set(_dcu_sdk_root "$ENV{DTK_ROOT}")
    endif()
    if(NOT _dcu_sdk_root)
      set(_dcu_sdk_root "$ENV{ROCM_PATH}")
    endif()
    if(NOT _dcu_sdk_root)
      set(_dcu_sdk_root "/opt/dtk")
    endif()

    find_path(DCU_SDK_ROCBLAS_INCLUDE_DIR
        NAMES rocblas/rocblas.h
        PATHS "${_dcu_sdk_root}/include"
        NO_DEFAULT_PATH)
    find_library(DCU_SDK_ROCBLAS_LIBRARY
        NAMES rocblas
        PATHS "${_dcu_sdk_root}/lib" "${_dcu_sdk_root}/lib64"
        NO_DEFAULT_PATH)
    find_path(DCU_SDK_HIP_INCLUDE_DIR
        NAMES hip/hip_runtime_api.h
        PATHS "${_dcu_sdk_root}/include"
        NO_DEFAULT_PATH)
    if(NOT DCU_SDK_ROCBLAS_INCLUDE_DIR OR
       NOT DCU_SDK_ROCBLAS_LIBRARY OR NOT DCU_SDK_HIP_INCLUDE_DIR)
      message(FATAL_ERROR
          "FLAGOS_DCU_SDK_OPS_BUILD=ON requires DTK rocBLAS and HIP headers. "
          "Searched ${_dcu_sdk_root}/include and ${_dcu_sdk_root}/lib; set "
          "FLAGOS_DCU_SDK_ROOT, DTK_ROOT, or ROCM_PATH to a valid SDK.")
    endif()

    # Device code is compiled with DTK's hipcc rather than CMake's CUDA language.
    # This keeps the main project g++-only while still allowing the independent
    # plugin to carry HIP kernels. The object contains no ATen code.
    set(_dcu_hipcc "${_dcu_sdk_root}/bin/hipcc")
    if(NOT EXISTS "${_dcu_hipcc}")
      set(_dcu_hipcc "${_dcu_sdk_root}/hip/bin/hipcc")
    endif()
    if(NOT EXISTS "${_dcu_hipcc}")
      message(FATAL_ERROR
          "DCU SDK-native plugin requires hipcc under ${_dcu_sdk_root}/bin or "
          "${_dcu_sdk_root}/hip/bin")
    endif()
    set(_dcu_hip_arch "gfx936")
    if(DEFINED FLAGOS_DCU_GPU_ARCH AND NOT FLAGOS_DCU_GPU_ARCH STREQUAL "")
      set(_dcu_hip_arch "${FLAGOS_DCU_GPU_ARCH}")
    elseif(DEFINED ENV{FLAGOS_DCU_GPU_ARCH} AND NOT "$ENV{FLAGOS_DCU_GPU_ARCH}" STREQUAL "")
      set(_dcu_hip_arch "$ENV{FLAGOS_DCU_GPU_ARCH}")
    endif()
    set(_dcu_hip_obj "${CMAKE_CURRENT_BINARY_DIR}/dcu_sdk/dtk_elementwise.hip.o")
    add_custom_command(
        OUTPUT "${_dcu_hip_obj}"
        COMMAND ${CMAKE_COMMAND} -E make_directory
            "${CMAKE_CURRENT_BINARY_DIR}/dcu_sdk"
        COMMAND "${_dcu_hipcc}" -std=c++17 --offload-arch=${_dcu_hip_arch}
            -fPIC -O3 -I${CMAKE_SOURCE_DIR}/csrc
            -I${CMAKE_SOURCE_DIR}/csrc/include
            -I${CMAKE_SOURCE_DIR}
            -I${PYTORCH_INSTALL_DIR}/include
            -I${PYTORCH_INSTALL_DIR}/include/torch/csrc/api/include
            -I${DCU_SDK_HIP_INCLUDE_DIR}
            -c "${CMAKE_CURRENT_SOURCE_DIR}/aten/backends/dcu_sdk/dtk_elementwise.hip"
            -o "${_dcu_hip_obj}"
        DEPENDS
            "${CMAKE_CURRENT_SOURCE_DIR}/aten/backends/dcu_sdk/dtk_elementwise.hip"
            "${CMAKE_CURRENT_SOURCE_DIR}/aten/backends/dcu_sdk/dtk_launch.h"
        COMMENT "DTK hipcc: dtk_elementwise.hip (${_dcu_hip_arch})"
        VERBATIM)

    add_library(libdcu_aten_ops SHARED
        "${CMAKE_CURRENT_SOURCE_DIR}/aten/backends/dcu_sdk/dcu_sdk_plugin.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/aten/backends/dcu_sdk/generated/dtk_kernels.cc"
        "${_dcu_hip_obj}")
    set_target_properties(libdcu_aten_ops PROPERTIES
        OUTPUT_NAME dcu_aten_ops
        POSITION_INDEPENDENT_CODE ON
        CXX_VISIBILITY_PRESET hidden)
    target_compile_definitions(libdcu_aten_ops PRIVATE
        __HIP_PLATFORM_AMD__=1
        C10_CUDA_NO_CMAKE_CONFIGURE_FILE=1)
    target_include_directories(libdcu_aten_ops PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}
        ${DCU_SDK_ROCBLAS_INCLUDE_DIR}
        ${DCU_SDK_HIP_INCLUDE_DIR})
    # Link only the exact official core interfaces needed for at::Tensor and
    # c10's ABI. Do NOT use ${TORCH_LIBRARIES}: TorchConfig.cmake defines that
    # umbrella list with the libtorch.so target, which is explicitly forbidden
    # in SDK-only mode.
    # Keep rocBLAS in DT_NEEDED even when all currently generated host wrappers
    # are elementwise: the plugin's SDK binding is an intentional contract, and
    # later category templates (GEMM/MIOpen/etc.) must not silently turn the ELF
    # audit into a false negative because the linker dropped an unused library.
    target_link_libraries(libdcu_aten_ops PRIVATE
        torch_cpu c10 flagos)
    target_link_options(libdcu_aten_ops PRIVATE
        -Wl,--no-as-needed ${DCU_SDK_ROCBLAS_LIBRARY} -Wl,--as-needed)
    # `${DCU_SDK_ROCBLAS_LIBRARY}` is deliberately passed as a link option above
    # rather than through target_link_libraries: CMake's transitive dependency
    # machinery places all libraries after its own --as-needed boundary, which
    # drops rocBLAS when the current generated cohort is header-only on the host.
    target_link_directories(libdcu_aten_ops PRIVATE
        "${_dcu_sdk_root}/lib" "${_dcu_sdk_root}/lib64"
        "${_dcu_sdk_root}/hip/lib")
    flagos_set_portable_rpath(libdcu_aten_ops
        EXTRA_DIRS "${_dcu_sdk_root}/lib" "${_dcu_sdk_root}/lib64"
                   "${_dcu_sdk_root}/hip/lib" "${_dcu_sdk_root}/cuda/cuda-12/lib64")
    install(TARGETS libdcu_aten_ops
        LIBRARY DESTINATION lib_dcu)
    set(_dcu_sdk_kernel_header
        "${CMAKE_CURRENT_SOURCE_DIR}/aten/backends/dcu_sdk/generated/dtk_kernels_decl.h")
    set(_dcu_sdk_manifest_template
        "${CMAKE_CURRENT_SOURCE_DIR}/aten/backends/dcu_sdk/generated/dcu_sdk_manifest.json")
    if(EXISTS "${_dcu_sdk_manifest_template}")
      install(FILES "${_dcu_sdk_manifest_template}" DESTINATION lib_dcu)
    endif()
    # The manifest states the currently generated plugin rows. Route-level
    # coverage metadata is emitted separately by codegen_dtk.py; this fallback
    # keeps source builds usable until that generated manifest is installed.
    set(_dcu_sdk_manifest "${CMAKE_CURRENT_BINARY_DIR}/dcu_sdk_manifest.json")
    set(_dcu_sdk_manifest_source
        "${CMAKE_SOURCE_DIR}/torch_fl/configs/dcu_sdk_manifest.json")
    if(EXISTS "${_dcu_sdk_manifest_source}")
      configure_file("${_dcu_sdk_manifest_source}" "${_dcu_sdk_manifest}" COPYONLY)
    else()
      message(FATAL_ERROR
          "DCU SDK-native manifest is missing: ${_dcu_sdk_manifest_source}. "
          "Run scripts/codegen_dtk.py --emit all before configuring.")
    endif()
    install(FILES "${_dcu_sdk_manifest}" DESTINATION lib_dcu)
    message(STATUS "DCU SDK-native plugin manifest: ${_dcu_sdk_manifest}")

    # The generated HIP object and host kernel table are the plugin's complete
    # source boundary. Keep a dependency on the declaration header so changing
    # the generated row set rebuilds the plugin even when the CMake source glob
    # itself is unchanged.
    set_property(TARGET libdcu_aten_ops APPEND PROPERTY OBJECT_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/aten/backends/dcu_sdk/generated/dtk_kernels_decl.h")
    add_dependencies(libdcu_aten_ops ${LIBRARY_NAME})

    # CMake's post-build checker runs before installation and therefore catches a
    # forbidden DT_NEEDED edge at the point it is introduced.
    add_custom_command(TARGET libdcu_aten_ops POST_BUILD
        COMMAND ${PYTHON_EXECUTABLE} "${CMAKE_SOURCE_DIR}/scripts/check_dcu_sdk_abi.py"
            "$<TARGET_FILE:libdcu_aten_ops>"
        COMMENT "Checking DCU SDK plugin dependencies")
    message(STATUS "DCU SDK-native route manifest: ${_dcu_sdk_manifest_source}")
  endif()
endif()



if(ACCELERATOR STREQUAL "gcu" OR ACCELERATOR STREQUAL "musa"
   OR ACCELERATOR STREQUAL "bpu" OR ACCELERATOR STREQUAL "ascend")
  # torch_fl/__init__.py pins backends_cuda.conf by default, which on
  # GCU/MUSA/BPU/Ascend would route every op to the (absent) CUDA kernels.
  # Record the platform next to the libs so the Python side can pick
  # backends_<platform>.conf instead. BPU additionally uses the marker to decide
  # whether to register the "bpu" torch.compile backend. Ascend previously
  # relied on runtime /dev/davinci* detection in torch_fl/__init__.py instead of
  # this marker (see _select_backend_config()); that detection can silently
  # fall through on a build/host where /dev is not enumerable, which also
  # makes tests/integration/platform_support.py misclassify the platform as
  # "cuda". Writing the marker makes Ascend identification explicit, matching
  # the other native platforms, while the davinci-based detection stays as a
  # fallback for wheels built before this change.
  set(_flagos_platform_marker "${CMAKE_CURRENT_BINARY_DIR}/flagos_platform")
  file(WRITE "${_flagos_platform_marker}" "${ACCELERATOR}\n")
  install(FILES "${_flagos_platform_marker}"
      DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()
