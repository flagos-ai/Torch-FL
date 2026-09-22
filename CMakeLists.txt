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

cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

set(FLAGOS_ACCELERATOR "cuda" CACHE STRING "Accelerator platform: cuda, metax, ascend, tsingmicro, dcu, gcu, musa, or bpu")

# Vendor driver/runtime dirs to embed in RUNPATH. Never bundled into the wheel
# (a machine with the card has the driver), mirroring MetaX's /opt/maca.
set(FLAGOS_VENDOR_RPATH_DIRS "")

# The platform/build matrix, shared with setup.py: cmake/flagos_platforms.json.
# Publishes FLAGOS_KERNEL_SWITCHES, FLAGOS_PLATFORM_LANGUAGES, the runtime source
# dir, the USE_* macro, the platform-marker flag, the C10_CUDA flag and
# FLAGOS_BUNDLE_LIBDIR -- the dir inside the wheel holding a bundled forked
# libtorch when the backend ships one (see scripts/vendor/bundle_*_libtorch.sh);
# "lib" means no separate bundle dir, the CUDA backend drops its extra .so
# straight into torch_fl/lib/. Consumed by cmake/FlagosRpath.cmake. Also rejects
# an unknown FLAGOS_ACCELERATOR.
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/FlagosPlatforms.cmake)

# Kernel build options (can be overridden via -D flags from setup.py).
# Chip selection is FLAGOS_ACCELERATOR's job alone -- there are no per-chip switches.
#
# A -D on the command line lands in the cache before option() runs, so this loop
# is the last point at which an explicit request can still be told apart from the
# option() default. flagos_apply_platform_kernel_policy() below uses it to refuse
# a contradictory -D instead of force-set()ting it away: setup.py resolves the
# same combination into the build record, so a switch quietly overridden here
# would ship a wheel describing a kernel set it does not contain.
foreach(_switch IN LISTS FLAGOS_KERNEL_SWITCHES)
  if(DEFINED ${_switch})
    set(${_switch}_EXPLICIT TRUE)
  endif()
endforeach()
unset(_switch)

option(FLAGOS_BUILD_VENDOR "Build the FLAGOS_ACCELERATOR vendor's native kernels (no-op where the vendor ships none)" ON)
option(FLAGOS_BUILD_FLAGGEMS "Build FlagGems Python kernel wrappers (calls via Python, no C++ linking)" ON)
option(FLAGOS_BUILD_BOXING "Build generated CUDA boxing kernels (CUDA-ABI vendor path)" ON)
option(FLAGOS_BUILD_FLAGGEMS_CPP "Build FlagGems C++ kernel wrappers (requires liboperators.so)" ON)
option(FLAGOS_BUILD_TILEOPS "Build TileOPs kernel wrappers (calls via Python, no C++ linking)" ON)

# Pin one kernel build switch for the accelerator selected below.
#
# A pin is a property of the platform, not a default, so an explicit -D asking
# for the other value is a hard error rather than something to override: there is
# nothing in this toolchain an "on" could link against, and setup.py refuses the
# same combination (the pins live in cmake/flagos_platforms.json).
function(flagos_pin_build_switch name value doc reason)
  if(${name}_EXPLICIT)
    if(NOT "${${name}}" STREQUAL "${value}")
      message(FATAL_ERROR
        "-D${name}=${${name}} is not possible with FLAGOS_ACCELERATOR=${FLAGOS_ACCELERATOR}: "
        "${reason}, so this build always has ${name}=${value}")
    endif()
  endif()
  set(${name} ${value} CACHE BOOL "${doc}" FORCE)
endfunction()

# Resolve one accelerator's SDK/toolkit root into a CMake variable.
#
#   flagos_resolve_sdk_root(<out_var> <default_path> <vendor_env_name>...)
#
# Order: an explicit -D<out_var> (already set) wins, then the accelerator's
# vendor-native environment variable(s), then the default.
#
# The environment names are the vendor's own: CUDA_HOME, ASCEND_HOME, MUSA_HOME,
# TOPS_HOME, MACA_PATH/MACA_HOME, ROCM_PATH, PPU_SDK. Those are what a vendor's
# set_env script writes, so they are the contract. No FLAGOS_-prefixed alias is
# accepted -- one name per vendor, the vendor's.
function(flagos_resolve_sdk_root out_var default_path)
  if(DEFINED ${out_var} AND NOT "${${out_var}}" STREQUAL "")
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

# Apply the table's per-accelerator kernel defaults and pins. Done after
# option() so the *_EXPLICIT loop above can still tell a -D from a default.
flagos_apply_platform_kernel_policy()

# Pre-project accelerator setup: the SDK/toolchain roots a project() call needs
# before it runs nvcc, plus MetaX's MACA_PATH (reported in the SDK block below).
if(FLAGOS_ACCELERATOR STREQUAL "metax")
  flagos_resolve_sdk_root(MACA_PATH "/opt/maca" MACA_PATH MACA_HOME)
endif()

if(FLAGOS_ACCELERATOR STREQUAL "cuda" OR FLAGOS_ACCELERATOR STREQUAL "ppu")
  # PPU builds against PPU_SDK/CUDA_SDK, so it rides the CUDA toolchain (CUDA
  # boxing kernels, project(CUDA)); it differs only in the libtorch it bundles
  # -- a local PPU build, not upstream's (FLAGOS_BUNDLE_LIBDIR from the table).
  #
  # project(... CUDA) runs nvcc compiler-id before find_package(CUDAToolkit); conda
  # layouts need CUDAToolkit_ROOT (headers under targets/x86_64-linux/include).
  #
  # Only vendor names: CUDA_HOME for a cuda build, PPU_SDK's CUDA_SDK for a ppu
  # build, CONDA_PREFIX for a conda toolkit. The existence check is what makes a
  # candidate usable, so one that does not actually hold cuda_runtime.h falls
  # through to the next.
  if(NOT CUDAToolkit_ROOT)
    if(DEFINED ENV{CUDA_HOME} AND EXISTS "$ENV{CUDA_HOME}/include/cuda_runtime.h")
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
  if(FLAGOS_ACCELERATOR STREQUAL "ppu")
    message(STATUS "PPU build: CUDA toolchain + bundled PPU libtorch (${FLAGOS_BUNDLE_LIBDIR})")
  endif()
endif()

# Per-platform kernel policy (why MetaX has no native kernels, why Ascend/GCU/MUSA
# turn boxing off, why MUSA pins the vendor on, ...) is documented in the table,
# cmake/flagos_platforms.json, next to the values it explains.
project(TORCH_FLAGOS ${FLAGOS_PLATFORM_LANGUAGES})

include(GNUInstallDirs)
include(CheckCXXCompilerFlag)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)

if("CUDA" IN_LIST FLAGOS_PLATFORM_LANGUAGES)
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

if(FLAGOS_ACCELERATOR STREQUAL "ascend")
  # Huawei CANN SDK
  flagos_resolve_sdk_root(ASCEND_HOME "/usr/local/Ascend/ascend-toolkit/latest" ASCEND_HOME)
  message(STATUS "Huawei CANN SDK path: ${ASCEND_HOME}")

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
elseif(FLAGOS_ACCELERATOR STREQUAL "tsingmicro")
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
elseif(FLAGOS_ACCELERATOR STREQUAL "dcu")
  # Hygon DTK. ROCM_PATH is what the DTK env.sh exports.
  flagos_resolve_sdk_root(ROCM_PATH "/opt/dtk" ROCM_PATH)
  message(STATUS "Hygon DTK path: ${ROCM_PATH}")

  # DTK's CUDA compatibility toolkit. Its libcudart.so.12 is a shim over
  # libgalaxyhip.so -- the same runtime libtorch_hip.so uses -- so there is one
  # driver state, not two.
  file(GLOB _DCU_CUDA_CANDIDATES "${ROCM_PATH}/cuda/cuda-*")
  set(DCU_CUDA_ROOT "")
  foreach(_cand IN LISTS _DCU_CUDA_CANDIDATES)
    if(EXISTS "${_cand}/include/cuda_runtime.h")
      set(DCU_CUDA_ROOT "${_cand}")
      break()
    endif()
  endforeach()
  if(NOT DCU_CUDA_ROOT)
    message(FATAL_ERROR
        "FLAGOS_ACCELERATOR=dcu selected, but the DTK CUDA compatibility toolkit was "
        "not found under ${ROCM_PATH}/cuda/. Set ROCM_PATH (or ROCM_PATH) to a "
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
      "${DCU_CUDA_ROOT}/lib64" "${ROCM_PATH}/lib" "${ROCM_PATH}/hip/lib"
      "${ROCM_PATH}/lib64" "${ROCM_PATH}/aillvm/lib" "${ROCM_PATH}/llvm/lib"
      "${ROCM_PATH}/.hyhal/rocm_smi/lib" "/opt/hyhal/lib")
  list(APPEND CMAKE_INSTALL_RPATH ${_dcu_rpath_dirs})
  set(FLAGOS_VENDOR_RPATH_DIRS ${_dcu_rpath_dirs})

  # Pre-create torch::cudart so PyTorch's cuda.cmake is skipped entirely
  # (guard: if(TARGET torch::cudart) return()).
elseif(FLAGOS_ACCELERATOR STREQUAL "gcu")
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
  if(FLAGOS_BUILD_VENDOR)
    find_library(TOPSATEN_LIB topsaten
        PATHS "${TOPS_ROOT}/lib" /usr/lib /usr/lib64)
    if(NOT TOPSATEN_LIB)
      message(FATAL_ERROR
        "FLAGOS_BUILD_VENDOR=ON but libtopsaten.so was not found (searched "
        "${TOPS_ROOT}/lib, /usr/lib, /usr/lib64). Install the TopsRider "
        "topsaten package, or build with -DFLAGOS_BUILD_VENDOR=OFF to fall back to CPU "
        "for all compute ops.")
    endif()
    find_path(TOPSATEN_INCLUDE_DIR topsaten/topsaten.h
        PATHS "${TOPS_ROOT}/include/gcu" /usr/include/gcu)
    if(NOT TOPSATEN_INCLUDE_DIR)
      message(FATAL_ERROR
        "FLAGOS_BUILD_VENDOR=ON but topsaten/topsaten.h was not found (searched "
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
elseif(FLAGOS_ACCELERATOR STREQUAL "musa")
  # Moore Threads MUSA toolkit: musart runtime. There is no libcudart shim here
  # (unlike MetaX/DCU), so nothing pretends to be CUDA -- cuda_runtime_compat
  # points at the native musa* headers and libmusart, and the runtime sources
  # under csrc/runtime/accelerator/musa/ call musaMalloc/musaStreamCreate/etc.
  flagos_resolve_sdk_root(MUSA_HOME "/usr/local/musa" MUSA_HOME)
  if(NOT EXISTS "${MUSA_HOME}/include/musa_runtime.h")
    message(FATAL_ERROR
        "FLAGOS_ACCELERATOR=musa selected, but the MUSA toolkit was not found "
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
  if(FLAGOS_BUILD_VENDOR)
    find_library(MUDNN_LIB mudnn PATHS "${MUSA_HOME}/lib" NO_DEFAULT_PATH)
    if(NOT MUDNN_LIB)
      message(FATAL_ERROR
          "FLAGOS_BUILD_VENDOR=ON but libmudnn.so was not found under ${MUSA_HOME}/lib. "
          "Install the mudnn component of the MUSA toolkit, or build with "
          "-DFLAGOS_BUILD_VENDOR=OFF to fall back to CPU for compute ops.")
    endif()
    if(NOT EXISTS "${MUSA_HOME}/include/mudnncxx/mudnn.h")
      message(FATAL_ERROR
          "FLAGOS_BUILD_VENDOR=ON but the mudnn C++ headers were not found "
          "(no ${MUSA_HOME}/include/mudnncxx/mudnn.h).")
    endif()
    message(STATUS "mudnn op library: ${MUDNN_LIB}")
    find_library(MURAND_LIB murand PATHS "${MUSA_HOME}/lib" NO_DEFAULT_PATH)
    if(NOT MURAND_LIB)
      message(FATAL_ERROR
          "FLAGOS_BUILD_VENDOR=ON but libmurand.so was not found under ${MUSA_HOME}/lib. "
          "Install the murand component of the MUSA toolkit.")
    endif()
    if(NOT EXISTS "${MUSA_HOME}/include/murand.h")
      message(FATAL_ERROR
          "FLAGOS_BUILD_VENDOR=ON but murand.h was not found under ${MUSA_HOME}/include.")
    endif()
    message(STATUS "murand RNG library: ${MURAND_LIB}")
  endif()

  add_library(torch::cudart INTERFACE IMPORTED)
  if(NOT TARGET CUDA::nvToolsExt)
    add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
  endif()
elseif(FLAGOS_ACCELERATOR STREQUAL "bpu")
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
elseif(FLAGOS_ACCELERATOR STREQUAL "metax")
  # MetaX SDK: use cu-bridge headers and runtime_cu library.
  # MACA_PATH was resolved in the metax branch above (MACA_PATH -> MACA_HOME ->
  # /opt/maca), and is reported here for the build log.
  message(STATUS "MetaX SDK path: ${MACA_PATH}")

  # The maca runtime (mcblas/mcdnn/...) comes from the target's /opt/maca driver
  # install; it is never bundled. /opt/maca is kept as an explicit fallback in
  # case the build machine's MACA_PATH is a non-default location.
  set(FLAGOS_VENDOR_RPATH_DIRS
      "${MACA_PATH}/lib" "${MACA_PATH}/lib64" "/opt/maca/lib" "/opt/maca/lib64")

  add_library(cuda_runtime_compat INTERFACE IMPORTED)
  set_target_properties(cuda_runtime_compat PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${MACA_PATH}/tools/cu-bridge/include;${MACA_PATH}/include;${MACA_PATH}/include/mcr"
  )
  target_link_directories(cuda_runtime_compat INTERFACE "${MACA_PATH}/lib")
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

if(FLAGOS_ACCELERATOR STREQUAL "dcu")
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

if(FLAGOS_BUILD_FLAGGEMS_CPP)
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
elseif(FLAGOS_ACCELERATOR STREQUAL "metax")
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
