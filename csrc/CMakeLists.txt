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

# Profiler device tracer selection: exclude the wrong tracer based on ACCELERATOR.
# CUPTI tracer is for CUDA/NVIDIA; ROCtracer is for DCU/AMD; everything else gets
# the stub (neither). Both tracer files implement the same MakeDeviceTracer() factory,
# so only one can be in the build.
if(ACCELERATOR STREQUAL "dcu")
  list(FILTER SOURCE_FILES EXCLUDE REGEX ".*/profiler/cupti_device_tracer\\.cc$")
else()
  list(FILTER SOURCE_FILES EXCLUDE REGEX ".*/profiler/roctracer_device_tracer\\.cc$")
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
if(FLAGGEMS_KERNEL OR FLAGGEMS_PYTHON)
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
  target_compile_definitions(${LIBRARY_NAME} PRIVATE FLAGOS_HAVE_CUPTI=1)
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
find_path(CUDA_OCCUPANCY_INCLUDE_DIR cuda_occupancy.h
  PATHS "$ENV{CUDA_HOME}/include"
        "$ENV{CUDA_HOME}/targets/x86_64-linux/include"
        /usr/local/cuda/include
        /usr/local/cuda/targets/x86_64-linux/include
        /usr/local/cuda-13.0/targets/x86_64-linux/include)
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

# The CPU-only torch wheel ships c10/cuda headers but not the CUDA-build-only
# c10/cuda/impl/cuda_cmake_macros.h. cuda_memory.h includes CUDACachingAllocator.h
# to delegate flagos allocation to the CUDA caching allocator; this macro tells
# CUDAMacros.h to skip that generated configure file. Not needed for Ascend
# (which does not include c10/cuda headers).
if(NOT ACCELERATOR STREQUAL "ascend")
  target_compile_definitions(${LIBRARY_NAME} PRIVATE C10_CUDA_NO_CMAKE_CONFIGURE_FILE)
endif()

if(ACCELERATOR STREQUAL "dcu" OR ACCELERATOR STREQUAL "kunlun")
  # Register the generic AutocastPrivateUse1 policy table for CUDA-compatible
  # vendor runtimes. The policy redispatches to existing backend kernels and
  # does not add handwritten vendor-specific operator implementations.
  target_compile_definitions(${LIBRARY_NAME} PRIVATE USE_FLAGOS_AUTOCAST=1)
endif()

if(ACCELERATOR STREQUAL "metax")
  # maca's torch headers are a hard fork gated on USE_MACA (e.g. C10_WARP_SIZE,
  # Context.h allow_tf32_cudnn). Without it the headers hit static_assert(0).
  target_compile_definitions(${LIBRARY_NAME} PRIVATE USE_MACA=1)
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
elseif(ACCELERATOR STREQUAL "kunlun")
  # Kunlun's vendor torch exposes CUDA registrations and uses the CUDA-shaped
  # runtime, so the standard allocator and generated CUDA boxing path apply.
  target_compile_definitions(${LIBRARY_NAME} PRIVATE USE_KUNLUN=1)
  target_link_libraries(${LIBRARY_NAME} PRIVATE ${_torch_fl_link_libs})
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
    target_link_libraries(${LIBRARY_NAME} PRIVATE ${MUDNN_LIB})
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

if(ACCELERATOR STREQUAL "gcu" OR ACCELERATOR STREQUAL "musa"
   OR ACCELERATOR STREQUAL "bpu")
  # torch_fl/__init__.py pins backends_cuda.conf by default, which on
  # GCU/MUSA/BPU would route every op to the (absent) CUDA kernels. Record the
  # platform next to the libs so the Python side can pick
  # backends_<platform>.conf instead. BPU additionally uses the marker to decide
  # whether to register the "bpu" torch.compile backend.
  set(_flagos_platform_marker "${CMAKE_CURRENT_BINARY_DIR}/flagos_platform")
  file(WRITE "${_flagos_platform_marker}" "${ACCELERATOR}\n")
  install(FILES "${_flagos_platform_marker}"
      DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()
