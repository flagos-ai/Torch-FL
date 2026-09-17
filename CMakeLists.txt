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

cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

set(ACCELERATOR "cuda" CACHE STRING "Accelerator platform: cuda, metax, ascend, tsingmicro, dcu, gcu, musa, or bpu")

# Directory inside the wheel holding a bundled forked libtorch, when the backend
# ships one (see scripts/vendor/bundle_*_libtorch.sh). "lib" means "no separate bundle
# dir": the CUDA backend drops its extra .so straight into torch_fl/lib/.
# Set per backend below; consumed by cmake/FlagosRpath.cmake.
set(FLAGOS_BUNDLE_LIBDIR "lib")
# Vendor driver/runtime dirs to embed in RUNPATH. Never bundled into the wheel
# (a machine with the card has the driver), mirroring MetaX's /opt/maca.
set(FLAGOS_VENDOR_RPATH_DIRS "")

# Kernel build options (can be overridden via -D flags from setup.py).
# Chip selection is ACCELERATOR's job alone -- there are no per-chip switches.
option(VENDOR_KERNEL "Build the ACCELERATOR vendor's native kernels (no-op where the vendor ships none)" ON)
option(FLAGGEMS_KERNEL "Build FlagGems Python kernel wrappers (calls via Python, no C++ linking)" ON)
option(BOXING_KERNEL "Build generated CUDA boxing kernels (CUDA-ABI vendor path)" ON)
option(FLAGGEMS_CPP "Build FlagGems C++ kernel wrappers (requires liboperators.so)" ON)

# Resolve one accelerator's SDK/toolkit root into a CMake variable.
#
#   flagos_resolve_sdk_root(<out_var> <default_path> <vendor_env_name>...)
#
# Order: an explicit -D<out_var> (already set) wins, then FLAGOS_SDK_ROOT, then
# the accelerator's vendor-native environment variable(s), then the default.
#
# Vendors export their own name (CUDA_HOME, ASCEND_HOME, MUSA_HOME, TOPS_HOME,
# METAX_PATH/MACA_PATH, ROCM_PATH, PPU_SDK, ...), so those must keep working --
# that is what a vendor's own set_env script writes. FLAGOS_SDK_ROOT sits on top
# as the one uniform override for a script that does not know which vendor is
# active, so cross-platform automation has a single knob instead of nine.
function(flagos_resolve_sdk_root out_var default_path)
  if(DEFINED ${out_var} AND NOT "${${out_var}}" STREQUAL "")
    return()
  endif()
  if(DEFINED ENV{FLAGOS_SDK_ROOT} AND NOT "$ENV{FLAGOS_SDK_ROOT}" STREQUAL "")
    set(${out_var} "$ENV{FLAGOS_SDK_ROOT}" PARENT_SCOPE)
    return()
  endif()
  foreach(_flagos_sdk_candidate IN LISTS ARGN)
    if(DEFINED ENV{${_flagos_sdk_candidate}}
       AND NOT "$ENV{${_flagos_sdk_candidate}}" STREQUAL "")
      set(${out_var} "$ENV{${_flagos_sdk_candidate}}" PARENT_SCOPE)
      return()
    endif()
  endforeach()
  set(${out_var} "${default_path}" PARENT_SCOPE)
endfunction()

if(ACCELERATOR STREQUAL "ascend")
  project(TORCH_FLAGOS CXX C)
elseif(ACCELERATOR STREQUAL "metax")
  flagos_resolve_sdk_root(METAX_PATH "/opt/maca" METAX_PATH METAX_HOME MACA_PATH MACA_HOME)
  # MetaX is a CUDA-boxing build: the generated CUDA boxing kernels
  # (csrc/aten/generated/cuda_kernels.cc, host g++, no mxcc) re-dispatch
  # PrivateUse1 -> CUDA into maca's libtorch_cuda.so. The hand-written mxcc
  # kernels under backends/metax/ are not built -- the dead native path is
  # retired, so there is no boxing/native mode variable at all.
  #
  # "Which kernel set compiles" is stated by the inclusion switches alone:
  # setup.py passes VENDOR_KERNEL=OFF (no native kernels) plus FLAGGEMS_CPP=OFF.
  # The routing that follows from that is a runtime concern -- torch_fl derives
  # the conf from the accelerator record -- so no mode belongs to the build.
  #
  # No MACA-built liboperators by default: keep the C++ wrappers off unless
  # explicitly requested, or a plain metax configure dies looking for them.
  if(NOT FLAGGEMS_CPP)
    set(FLAGGEMS_CPP OFF CACHE BOOL "Build FlagGems C++ kernel wrappers" FORCE)
  endif()
  # Self-contained wheel: forked libtorch bundled by
  # scripts/vendor/bundle_maca_libtorch.sh (the maca runtime stays on the target).
  set(FLAGOS_BUNDLE_LIBDIR "lib_maca")
  project(TORCH_FLAGOS CXX C)
elseif(ACCELERATOR STREQUAL "tsingmicro")
  project(TORCH_FLAGOS CXX C)
elseif(ACCELERATOR STREQUAL "dcu")
  # Hygon DCU (DTK). The vendor torch wheel is a hipified build that registers
  # its HIP kernels under the CUDA dispatch key (tensors report device type
  # kCUDA), so the generated PrivateUse1 -> CUDA boxing kernels dispatch into
  # libtorch_hip.so unchanged. DTK also ships a CUDA compatibility toolkit whose
  # libcudart.so.12 is a thin shim over libgalaxyhip.so, so the runtime sources
  # under csrc/runtime/accelerator/cuda/ compile as-is against host g++.
  # No nvcc and no CUDA language: nothing outside backends/metax/ is a .cu.
  set(FLAGGEMS_CPP OFF CACHE BOOL "Build FlagGems C++ kernel wrappers" FORCE)
  # Self-contained wheel: DTK's forked libtorch (libtorch_cpu.so carries hip
  # symbols and needs libgalaxyhip.so.5) is bundled here by
  # scripts/vendor/bundle_dcu_libtorch.sh. The DTK driver itself stays on the target.
  set(FLAGOS_BUNDLE_LIBDIR "lib_dcu")
  project(TORCH_FLAGOS CXX C)
elseif(ACCELERATOR STREQUAL "gcu")
  project(TORCH_FLAGOS CXX C)
elseif(ACCELERATOR STREQUAL "musa")
  # Moore Threads MUSA takes the native operator route (like Ascend/GCU), not
  # CUDA boxing: the MUSA toolkit ships no CUDA runtime, so there is no vendor
  # dispatch key to box into. The generated kernels call mudnn -- the vendor's
  # torch-independent kernel library -- directly. FLAGGEMS_KERNEL remains a
  # caller-controlled option so the same build can include the optional hybrid
  # dispatchers without requiring the FlagGems C++ runtime.
  set(VENDOR_KERNEL ON CACHE BOOL "Build the ACCELERATOR vendor's native kernels" FORCE)
  set(FLAGGEMS_CPP OFF CACHE BOOL "Build FlagGems C++ kernel wrappers" FORCE)
  project(TORCH_FLAGOS CXX C)
elseif(ACCELERATOR STREQUAL "bpu")
  # D-Robotics RDK BPU. No CUDA runtime and no per-op kernels: the BPU executes
  # whole compiled graphs, so eager ops fall back to CPU and acceleration comes
  # from the torch.compile backend. Only the runtime layer (UCP allocator,
  # device/stream stubs) is native, all plain host C++.
  set(FLAGGEMS_CPP OFF CACHE BOOL "Build FlagGems C++ kernel wrappers" FORCE)
  project(TORCH_FLAGOS CXX C)
elseif(ACCELERATOR STREQUAL "cuda" OR ACCELERATOR STREQUAL "ppu")
  # PPU builds against PPU_SDK/CUDA_SDK, so it rides the CUDA toolchain (CUDA
  # boxing kernels, project(CUDA)); it differs only in the libtorch it bundles
  # -- a local PPU build, not upstream's. Its own ACCELERATOR value makes it a
  # first-class chip like every other vendor (previously it was told apart from
  # cuda by the PPU_SDK environment, i.e. env-based chip detection).
  if(ACCELERATOR STREQUAL "ppu")
    set(FLAGOS_BUNDLE_LIBDIR "lib_ppu")
    message(STATUS "PPU build: CUDA toolchain + bundled PPU libtorch (lib_ppu)")
  endif()
  # project(... CUDA) runs nvcc compiler-id before find_package(CUDAToolkit); conda
  # layouts need CUDAToolkit_ROOT (headers under targets/x86_64-linux/include).
  #
  # FLAGOS_SDK_ROOT overrides the toolkit root here too: it is the CUDA toolkit
  # itself for a cuda build, and PPU_SDK's CUDA_SDK for a ppu build. The
  # existence check is what makes the candidate usable, so a value that does not
  # actually hold cuda_runtime.h falls through to CUDA_HOME / conda.
  if(NOT CUDAToolkit_ROOT)
    if(DEFINED ENV{FLAGOS_SDK_ROOT} AND EXISTS "$ENV{FLAGOS_SDK_ROOT}/include/cuda_runtime.h")
      set(CUDAToolkit_ROOT "$ENV{FLAGOS_SDK_ROOT}" CACHE PATH "CUDA toolkit root")
    elseif(DEFINED ENV{FLAGOS_SDK_ROOT} AND EXISTS "$ENV{FLAGOS_SDK_ROOT}/CUDA_SDK/include/cuda_runtime.h")
      set(CUDAToolkit_ROOT "$ENV{FLAGOS_SDK_ROOT}/CUDA_SDK" CACHE PATH "CUDA toolkit root")
    elseif(DEFINED ENV{CUDA_HOME} AND EXISTS "$ENV{CUDA_HOME}/include/cuda_runtime.h")
      set(CUDAToolkit_ROOT "$ENV{CUDA_HOME}" CACHE PATH "CUDA toolkit root")
    elseif(DEFINED ENV{PPU_SDK} AND EXISTS "$ENV{PPU_SDK}/CUDA_SDK/include/cuda_runtime.h")
      set(CUDAToolkit_ROOT "$ENV{PPU_SDK}/CUDA_SDK" CACHE PATH "CUDA toolkit root")
    elseif(DEFINED ENV{CONDA_PREFIX} AND EXISTS "$ENV{CONDA_PREFIX}/targets/x86_64-linux/include/cuda_runtime.h")
      set(CUDAToolkit_ROOT "$ENV{CONDA_PREFIX}/targets/x86_64-linux" CACHE PATH "CUDA toolkit root")
    endif()
  endif()
  if(NOT CMAKE_CUDA_COMPILER)
    if(DEFINED ENV{CONDA_PREFIX} AND EXISTS "$ENV{CONDA_PREFIX}/bin/nvcc")
      set(CMAKE_CUDA_COMPILER "$ENV{CONDA_PREFIX}/bin/nvcc" CACHE FILEPATH "")
    elseif(CUDAToolkit_ROOT AND EXISTS "${CUDAToolkit_ROOT}/bin/nvcc")
      set(CMAKE_CUDA_COMPILER "${CUDAToolkit_ROOT}/bin/nvcc" CACHE FILEPATH "")
    endif()
  endif()
  project(TORCH_FLAGOS CUDA CXX C)
else()
  message(FATAL_ERROR
    "Unknown ACCELERATOR '${ACCELERATOR}'. Expected one of: cuda, ppu, metax, "
    "ascend, dcu, gcu, musa, bpu, tsingmicro.")
endif()

include(GNUInstallDirs)
include(CheckCXXCompilerFlag)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT ACCELERATOR STREQUAL "metax" AND NOT ACCELERATOR STREQUAL "ascend"
   AND NOT ACCELERATOR STREQUAL "tsingmicro" AND NOT ACCELERATOR STREQUAL "dcu"
   AND NOT ACCELERATOR STREQUAL "gcu" AND NOT ACCELERATOR STREQUAL "musa"
   AND NOT ACCELERATOR STREQUAL "bpu")
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_SKIP_BUILD_RPATH  FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH FALSE)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)

set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)

if(APPLE)
  set(CMAKE_INSTALL_RPATH "@loader_path/lib;@loader_path")
elseif(UNIX)
  set(CMAKE_INSTALL_RPATH "$ORIGIN/lib:$ORIGIN")
elseif(WIN32)
  set(CMAKE_INSTALL_RPATH "")
endif()
set(CMAKE_INSTALL_LIBDIR lib)
set(CMAKE_INSTALL_MESSAGE NEVER)

set(Torch_DIR ${PYTORCH_INSTALL_DIR}/share/cmake/Torch)

if(ACCELERATOR STREQUAL "ascend")
  # Huawei CANN SDK
  flagos_resolve_sdk_root(ASCEND_HOME "/usr/local/Ascend/ascend-toolkit/latest" ASCEND_HOME)
  message(STATUS "Huawei CANN SDK path: ${ASCEND_HOME}")

  set(VENDOR_KERNEL ON CACHE BOOL "Build the ACCELERATOR vendor's native kernels" FORCE)

  add_library(cuda_runtime_compat INTERFACE IMPORTED)
  set_target_properties(cuda_runtime_compat PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${ASCEND_HOME}/include"
  )
  target_link_directories(cuda_runtime_compat INTERFACE "${ASCEND_HOME}/lib64")
  target_link_libraries(cuda_runtime_compat INTERFACE ascendcl)

  set(CUDA_RUNTIME_LIB cuda_runtime_compat)

  add_library(torch::cudart INTERFACE IMPORTED)
  if(NOT TARGET CUDA::nvToolsExt)
    add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
  endif()
elseif(ACCELERATOR STREQUAL "tsingmicro")
  # TsingMicro Kuiper SDK
  set(KUIPER_ROOT "/usr/local/kuiper" CACHE PATH "Kuiper SDK root")
  message(STATUS "TsingMicro Kuiper SDK path: ${KUIPER_ROOT}")

  add_library(cuda_runtime_compat INTERFACE IMPORTED)
  set_target_properties(cuda_runtime_compat PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${KUIPER_ROOT}/include"
  )
  target_link_directories(cuda_runtime_compat INTERFACE "${KUIPER_ROOT}/lib")
  target_link_libraries(cuda_runtime_compat INTERFACE hpgr)

  set(CUDA_RUNTIME_LIB cuda_runtime_compat)

  # Embed Kuiper library path in RPATH so LD_LIBRARY_PATH is not needed at runtime.
  list(APPEND CMAKE_INSTALL_RPATH "${KUIPER_ROOT}/lib")

  add_library(torch::cudart INTERFACE IMPORTED)
  if(NOT TARGET CUDA::nvToolsExt)
    add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
  endif()
elseif(ACCELERATOR STREQUAL "dcu")
  # Hygon DTK. ROCM_PATH is what the DTK env.sh exports.
  flagos_resolve_sdk_root(DTK_ROOT "/opt/dtk" DTK_ROOT ROCM_PATH)
  message(STATUS "Hygon DTK path: ${DTK_ROOT}")

  # DTK's CUDA compatibility toolkit. Its libcudart.so.12 is a shim over
  # libgalaxyhip.so -- the same runtime libtorch_hip.so uses -- so there is one
  # driver state, not two.
  file(GLOB _DCU_CUDA_CANDIDATES "${DTK_ROOT}/cuda/cuda-*")
  set(DCU_CUDA_ROOT "")
  foreach(_cand IN LISTS _DCU_CUDA_CANDIDATES)
    if(EXISTS "${_cand}/include/cuda_runtime.h")
      set(DCU_CUDA_ROOT "${_cand}")
      break()
    endif()
  endforeach()
  if(NOT DCU_CUDA_ROOT)
    message(FATAL_ERROR
        "ACCELERATOR=dcu selected, but the DTK CUDA compatibility toolkit was "
        "not found under ${DTK_ROOT}/cuda/. Set DTK_ROOT (or ROCM_PATH) to a "
        "DTK installation that provides cuda-*/include/cuda_runtime.h.")
  endif()
  message(STATUS "DTK CUDA compatibility toolkit: ${DCU_CUDA_ROOT}")

  add_library(cuda_runtime_compat INTERFACE IMPORTED)
  set_target_properties(cuda_runtime_compat PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${DCU_CUDA_ROOT}/include"
  )
  target_link_directories(cuda_runtime_compat INTERFACE "${DCU_CUDA_ROOT}/lib64")
  target_link_libraries(cuda_runtime_compat INTERFACE cudart)

  set(CUDA_RUNTIME_LIB cuda_runtime_compat)

  # Embed the compat + DTK library paths so LD_LIBRARY_PATH is not needed.
  # Measured layout of the deps the forked libtorch actually pulls. They are all
  # on the DTK container's LD_LIBRARY_PATH, which a clean env does NOT inherit --
  # so the RUNPATH has to name them or a self-contained wheel gets "not found":
  #   cuda/cuda-*/lib64      libcudart.so.12 (shim over libgalaxyhip)
  #   lib                    libgalaxyhip.so.5 libhipnn librocfft.so.0
  #                          librocrand.so.1 librocsparse.so.1 libunwind.so.8
  #                          libMIOpen-recommend.so
  #   hip/lib, lib64         hip runtime
  #   aillvm/lib             libomp.so
  #   .hyhal/rocm_smi/lib    librocm_smi64.so.2
  # Keep in sync with VENDOR_RPATH in scripts/vendor/bundle_dcu_libtorch.sh.
  set(_dcu_rpath_dirs
      "${DCU_CUDA_ROOT}/lib64" "${DTK_ROOT}/lib" "${DTK_ROOT}/hip/lib"
      "${DTK_ROOT}/lib64" "${DTK_ROOT}/aillvm/lib" "${DTK_ROOT}/llvm/lib"
      "${DTK_ROOT}/.hyhal/rocm_smi/lib" "/opt/hyhal/lib")
  list(APPEND CMAKE_INSTALL_RPATH ${_dcu_rpath_dirs})
  set(FLAGOS_VENDOR_RPATH_DIRS ${_dcu_rpath_dirs})

  # Pre-create torch::cudart so PyTorch's cuda.cmake is skipped entirely
  # (guard: if(TARGET torch::cudart) return()).
elseif(ACCELERATOR STREQUAL "gcu")
  # Enflame TopsRider SDK: tops runtime (libtopsrt) + topsaten operator library.
  flagos_resolve_sdk_root(TOPS_ROOT "/opt/tops" TOPS_HOME)
  message(STATUS "Enflame TopsRider SDK path: ${TOPS_ROOT}")

  add_library(cuda_runtime_compat INTERFACE IMPORTED)
  set_target_properties(cuda_runtime_compat PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${TOPS_ROOT}/include"
  )
  target_link_directories(cuda_runtime_compat INTERFACE "${TOPS_ROOT}/lib")
  target_link_libraries(cuda_runtime_compat INTERFACE topsrt)

  set(CUDA_RUNTIME_LIB cuda_runtime_compat)

  # Embed the tops library path in RPATH so LD_LIBRARY_PATH is not needed at runtime.
  list(APPEND CMAKE_INSTALL_RPATH "${TOPS_ROOT}/lib")

  # topsaten (the aten-style op library) ships in the distro paths, not under
  # ${TOPS_ROOT}/lib; its headers live in ${TOPS_ROOT}/include/gcu.
  if(VENDOR_KERNEL)
    find_library(TOPSATEN_LIB topsaten
        PATHS "${TOPS_ROOT}/lib" /usr/lib /usr/lib64)
    if(NOT TOPSATEN_LIB)
      message(FATAL_ERROR
        "VENDOR_KERNEL=ON but libtopsaten.so was not found (searched "
        "${TOPS_ROOT}/lib, /usr/lib, /usr/lib64). Install the TopsRider "
        "topsaten package, or build with -DVENDOR_KERNEL=OFF to fall back to CPU "
        "for all compute ops.")
    endif()
    find_path(TOPSATEN_INCLUDE_DIR topsaten/topsaten.h
        PATHS "${TOPS_ROOT}/include/gcu" /usr/include/gcu)
    if(NOT TOPSATEN_INCLUDE_DIR)
      message(FATAL_ERROR
        "VENDOR_KERNEL=ON but topsaten/topsaten.h was not found (searched "
        "${TOPS_ROOT}/include/gcu, /usr/include/gcu).")
    endif()
    set(TOPSATEN_INCLUDE_DIRS "${TOPSATEN_INCLUDE_DIR}" "${TOPS_ROOT}/include")
    message(STATUS "topsaten library: ${TOPSATEN_LIB}")
    message(STATUS "topsaten include: ${TOPSATEN_INCLUDE_DIR}")
    get_filename_component(_topsaten_libdir "${TOPSATEN_LIB}" DIRECTORY)
    list(APPEND CMAKE_INSTALL_RPATH "${_topsaten_libdir}")
  endif()

  add_library(torch::cudart INTERFACE IMPORTED)
  if(NOT TARGET CUDA::nvToolsExt)
    add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
  endif()
elseif(ACCELERATOR STREQUAL "musa")
  # Moore Threads MUSA toolkit: musart runtime. There is no libcudart shim here
  # (unlike MetaX/DCU), so nothing pretends to be CUDA -- cuda_runtime_compat
  # points at the native musa* headers and libmusart, and the runtime sources
  # under csrc/runtime/accelerator/musa/ call musaMalloc/musaStreamCreate/etc.
  flagos_resolve_sdk_root(MUSA_HOME "/usr/local/musa" MUSA_HOME)
  if(NOT EXISTS "${MUSA_HOME}/include/musa_runtime.h")
    message(FATAL_ERROR
        "ACCELERATOR=musa selected, but the MUSA toolkit was not found "
        "(no ${MUSA_HOME}/include/musa_runtime.h). Set MUSA_HOME to a valid "
        "MUSA installation.")
  endif()
  message(STATUS "Moore Threads MUSA toolkit path: ${MUSA_HOME}")

  add_library(cuda_runtime_compat INTERFACE IMPORTED)
  set_target_properties(cuda_runtime_compat PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${MUSA_HOME}/include"
  )
  target_link_directories(cuda_runtime_compat INTERFACE "${MUSA_HOME}/lib")
  target_link_libraries(cuda_runtime_compat INTERFACE musart)

  set(CUDA_RUNTIME_LIB cuda_runtime_compat)

  # Embed the MUSA library path in RPATH so LD_LIBRARY_PATH is not needed.
  list(APPEND CMAKE_INSTALL_RPATH "${MUSA_HOME}/lib")

  # Compute ops go through mudnn (libmudnn.so), the vendor's kernel library.
  # It links against musart only and pulls in no torch symbols, so it ships in
  # the MUSA toolkit next to musart and needs no separate root: no torch_musa
  # package, no extracted tree, no version pinning against the active torch.
  #
  # (The earlier route called torch_musa's flat at::musa::* API from
  # libmusa_python.so. That library links against torch and embeds its C++
  # object layout, which pinned the plugin to one exact torch build --
  # sizeof(c10::MessageLogger) changed 408 -> 400 between 2.9.1 and 2.10 and
  # corrupted the vendor .so's stack. mudnn has no such coupling.)
  if(VENDOR_KERNEL)
    find_library(MUDNN_LIB mudnn PATHS "${MUSA_HOME}/lib" NO_DEFAULT_PATH)
    if(NOT MUDNN_LIB)
      message(FATAL_ERROR
          "VENDOR_KERNEL=ON but libmudnn.so was not found under ${MUSA_HOME}/lib. "
          "Install the mudnn component of the MUSA toolkit, or build with "
          "-DVENDOR_KERNEL=OFF to fall back to CPU for compute ops.")
    endif()
    if(NOT EXISTS "${MUSA_HOME}/include/mudnncxx/mudnn.h")
      message(FATAL_ERROR
          "VENDOR_KERNEL=ON but the mudnn C++ headers were not found "
          "(no ${MUSA_HOME}/include/mudnncxx/mudnn.h).")
    endif()
    message(STATUS "mudnn op library: ${MUDNN_LIB}")
    find_library(MURAND_LIB murand PATHS "${MUSA_HOME}/lib" NO_DEFAULT_PATH)
    if(NOT MURAND_LIB)
      message(FATAL_ERROR
          "VENDOR_KERNEL=ON but libmurand.so was not found under ${MUSA_HOME}/lib. "
          "Install the murand component of the MUSA toolkit.")
    endif()
    if(NOT EXISTS "${MUSA_HOME}/include/murand.h")
      message(FATAL_ERROR
          "VENDOR_KERNEL=ON but murand.h was not found under ${MUSA_HOME}/include.")
    endif()
    message(STATUS "murand RNG library: ${MURAND_LIB}")
  endif()

  add_library(torch::cudart INTERFACE IMPORTED)
  if(NOT TARGET CUDA::nvToolsExt)
    add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
  endif()
elseif(ACCELERATOR STREQUAL "bpu")
  # D-Robotics RDK BPU. There is no CUDA-compat layer here at all: device
  # memory comes from the Horizon UCP allocator (libhbucp) and the core/task API
  # from libbpu, both linked by csrc/runtime/accelerator/bpu directly. So there
  # is no CUDA_RUNTIME_LIB to hand to the other targets.
  set(HOBOT_LIB_DIR "/usr/hobot/lib" CACHE PATH "Horizon runtime library dir")
  set(HOBOT_INCLUDE_DIR "/usr/include/hobot" CACHE PATH "Horizon runtime include dir")
  message(STATUS "Horizon runtime: ${HOBOT_LIB_DIR} (headers ${HOBOT_INCLUDE_DIR})")

  set(CUDA_RUNTIME_LIB "")

  # Embed the Horizon library path so LD_LIBRARY_PATH is not needed at runtime.
  list(APPEND CMAKE_INSTALL_RPATH "${HOBOT_LIB_DIR}")

  # Pre-create torch::cudart so PyTorch's cuda.cmake returns early
  # (guard: if(TARGET torch::cudart) return()).
  add_library(torch::cudart INTERFACE IMPORTED)
  if(NOT TARGET CUDA::nvToolsExt)
    add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
  endif()
elseif(ACCELERATOR STREQUAL "metax")
  # MetaX SDK: use cu-bridge headers and runtime_cu library.
  # METAX_PATH was resolved in the metax branch above (METAX_PATH ->
  # METAX_HOME -> MACA_PATH -> MACA_HOME -> /opt/maca, with FLAGOS_SDK_ROOT on
  # top), so it is reported here for the build log.
  message(STATUS "MetaX SDK path: ${METAX_PATH}")

  # The maca runtime (mcblas/mcdnn/...) comes from the target's /opt/maca driver
  # install; it is never bundled. /opt/maca is kept as an explicit fallback in
  # case the build machine's METAX_PATH is a non-default location.
  set(FLAGOS_VENDOR_RPATH_DIRS
      "${METAX_PATH}/lib" "${METAX_PATH}/lib64" "/opt/maca/lib" "/opt/maca/lib64")

  add_library(cuda_runtime_compat INTERFACE IMPORTED)
  set_target_properties(cuda_runtime_compat PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${METAX_PATH}/tools/cu-bridge/include;${METAX_PATH}/include;${METAX_PATH}/include/mcr"
  )
  target_link_directories(cuda_runtime_compat INTERFACE "${METAX_PATH}/lib")
  target_link_libraries(cuda_runtime_compat INTERFACE runtime_cu)

  set(CUDA_RUNTIME_LIB cuda_runtime_compat)

  # Pre-create torch::cudart so PyTorch's cuda.cmake is skipped entirely.
  # cuda.cmake guard: if(TARGET torch::cudart) return() endif()
  add_library(torch::cudart INTERFACE IMPORTED)

  if(NOT TARGET CUDA::nvToolsExt)
    add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
  endif()
else()
  # Workaround for CUDA 12.x where nvToolsExt is header-only (no separate library)
  # PyTorch's CMake expects CUDA::nvToolsExt target which doesn't exist in CUDA 12.x
  find_package(CUDAToolkit REQUIRED)
  if(NOT TARGET CUDA::nvToolsExt)
    # Create a dummy IMPORTED target for nvToolsExt (NVTX3 is header-only in CUDA 12.x)
    add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
    if(TARGET CUDA::nvtx3)
      set_target_properties(CUDA::nvToolsExt PROPERTIES
        INTERFACE_LINK_LIBRARIES CUDA::nvtx3)
    endif()
  endif()

  set(CUDA_RUNTIME_LIB CUDA::cudart)
endif()

find_package(Torch REQUIRED)

if(ACCELERATOR STREQUAL "dcu")
  # DTK exports MIOpen with a build-machine absolute path baked into its link
  # interface: /usr/lib/x86_64-linux-gnu/librt.so. glibc >= 2.34 folded librt
  # into libc, so current distros ship only librt.a + librt.so.1 -- that .so
  # symlink no longer exists and the generated makefiles die with
  # "No rule to make target '/usr/lib/x86_64-linux-gnu/librt.so'".
  # Rewrite any dangling absolute path to its bare -l name; the linker resolves
  # `rt` against the stub archive. Nothing to do on hosts where the path exists.
  foreach(_dcu_target MIOpen)
    if(TARGET ${_dcu_target})
      get_target_property(_dcu_libs ${_dcu_target} INTERFACE_LINK_LIBRARIES)
      if(_dcu_libs)
        set(_dcu_fixed_libs "")
        foreach(_dcu_lib IN LISTS _dcu_libs)
          if(IS_ABSOLUTE "${_dcu_lib}" AND NOT EXISTS "${_dcu_lib}")
            get_filename_component(_dcu_name "${_dcu_lib}" NAME_WE)
            string(REGEX REPLACE "^lib" "" _dcu_name "${_dcu_name}")
            message(STATUS
                "DCU: ${_dcu_target} links missing ${_dcu_lib}, using -l${_dcu_name}")
            list(APPEND _dcu_fixed_libs "${_dcu_name}")
          else()
            list(APPEND _dcu_fixed_libs "${_dcu_lib}")
          endif()
        endforeach()
        set_target_properties(${_dcu_target} PROPERTIES
            INTERFACE_LINK_LIBRARIES "${_dcu_fixed_libs}")
      endif()
    endif()
  endforeach()
endif()

if(FLAGGEMS_CPP)
  # Allow override via cmake var or env var
  # setup.py passes -DFlagGems_DIR=..., CMakeLists uses FLAGGEMS_DIR internally
  if(FlagGems_DIR AND NOT FLAGGEMS_DIR)
    set(FLAGGEMS_DIR "${FlagGems_DIR}")
  endif()
  if(NOT FLAGGEMS_DIR)
    set(FLAGGEMS_DIR "$ENV{FLAGGEMS_DIR}")
  endif()
  if(NOT FLAGGEMS_DIR)
    # Try FLAGGEMS_SOURCE_DIR (cmake var from setup.py or env var)
    if(NOT FLAGGEMS_SOURCE_DIR)
      set(FLAGGEMS_SOURCE_DIR "$ENV{FLAGGEMS_SOURCE_DIR}")
    endif()
    if(FLAGGEMS_SOURCE_DIR)
      # Check if it points to the repo root (has build/ dir with FlagGemsConfig.cmake)
      file(GLOB _FLAGGEMS_BUILD_CONFIGS "${FLAGGEMS_SOURCE_DIR}/build/*/FlagGemsConfig.cmake")
      if(_FLAGGEMS_BUILD_CONFIGS)
        list(GET _FLAGGEMS_BUILD_CONFIGS 0 _FLAGGEMS_CONFIG)
        get_filename_component(FLAGGEMS_DIR "${_FLAGGEMS_CONFIG}" DIRECTORY)
      endif()
    endif()
  endif()
  if(NOT FLAGGEMS_DIR)
    # Auto-detect from installed flag_gems package
    execute_process(
      COMMAND ${PYTHON_EXECUTABLE} -c "import flag_gems; print(flag_gems.__path__[0])"
      OUTPUT_VARIABLE _FLAGGEMS_PKG_PATH
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
      RESULT_VARIABLE _FLAGGEMS_DETECT_RET
    )
    if(_FLAGGEMS_DETECT_RET EQUAL 0 AND _FLAGGEMS_PKG_PATH)
      set(FLAGGEMS_DIR "${_FLAGGEMS_PKG_PATH}/lib/cmake/FlagGems")
    endif()
  endif()
  if(NOT FLAGGEMS_DIR)
    message(FATAL_ERROR "Cannot find FlagGems. Set FLAGGEMS_DIR or FLAGGEMS_SOURCE_DIR.")
  endif()
  message(STATUS "FlagGems cmake dir: ${FLAGGEMS_DIR}")
  # FlagGemsConfig.cmake calls find_dependency(Torch MODULE), which needs FindTorch.cmake in MODULE_PATH
  list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
  if(FLAGGEMS_SOURCE_DIR AND EXISTS "${FLAGGEMS_SOURCE_DIR}/cmake")
    list(APPEND CMAKE_MODULE_PATH "${FLAGGEMS_SOURCE_DIR}/cmake")
  endif()
  list(APPEND CMAKE_MODULE_PATH "${FLAGGEMS_DIR}")
  find_package(FlagGems REQUIRED PATHS "${FLAGGEMS_DIR}" NO_DEFAULT_PATH)

  # Embed liboperators.so dir into RPATH so it's found at runtime without LD_LIBRARY_PATH
  if(TARGET FlagGems::operators)
    set(_flaggems_operators_target FlagGems::operators)
  elseif(TARGET operators)
    set(_flaggems_operators_target operators)
  else()
    message(FATAL_ERROR "FlagGems package does not export an operators target")
  endif()

  get_filename_component(_flaggems_libdir "${FLAGGEMS_DIR}/../.." ABSOLUTE)
  get_target_property(
    _flaggems_lib_location
    ${_flaggems_operators_target}
    IMPORTED_LOCATION_RELEASE
  )
  if(NOT _flaggems_lib_location)
    get_target_property(
      _flaggems_lib_location
      ${_flaggems_operators_target}
      IMPORTED_LOCATION
    )
  endif()
  if(_flaggems_lib_location)
    get_filename_component(_flaggems_libdir "${_flaggems_lib_location}" DIRECTORY)
  endif()
  message(STATUS "FlagGems liboperators RPATH: ${_flaggems_libdir}")
  list(APPEND CMAKE_INSTALL_RPATH "${_flaggems_libdir}")
elseif(ACCELERATOR STREQUAL "metax")
  message(STATUS "MetaX build: CUDA boxing kernels + bundled maca libtorch (no native mxcc)")
endif()

if(DEFINED PYTHON_INCLUDE_DIR)
  include_directories(${PYTHON_INCLUDE_DIR})
else()
  message(FATAL_ERROR "Cannot find Python directory")
endif()

if(PYTORCH_INSTALL_DIR)
  # Needed for the *build* link step (and for an in-place/editable install, where
  # the built .so stay in the source tree). flagos_set_portable_rpath() strips it
  # from the installed targets' RUNPATH so the wheel is relocatable.
  list(APPEND CMAKE_INSTALL_RPATH "${PYTORCH_INSTALL_DIR}/lib")
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
# Public C ABI headers (flagos.h / macros.h), included as <flagos.h>.
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/csrc/include)
include(${PROJECT_SOURCE_DIR}/cmake/TorchPythonTargets.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/FlagosRpath.cmake)

add_subdirectory(${PROJECT_SOURCE_DIR}/csrc/runtime/accelerator)
add_subdirectory(${PROJECT_SOURCE_DIR}/csrc)
add_subdirectory(${PROJECT_SOURCE_DIR}/torch_fl/csrc)
