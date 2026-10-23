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

set(FLAGOS_ACCELERATOR "cuda" CACHE STRING "Accelerator platform: cuda, metax, ascend, tsingmicro, dcu, gcu, musa, kunlunxin, iluvatar, or bpu")

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

# Which platform this build is, asked once: both spellings are accepted, because
# FLAGOS_ACCELERATOR carries whichever the build was invoked with (`cuda` or its
# vendor `nvidia`, `mps` or `apple`, or a vendor that is another platform's id
# alias, `metax`). Every gate below reads these instead of comparing the raw
# variable, which held for one spelling only -- a lane on the vendor name fell
# through to the CUDA branch, or skipped its own.
macro(_flagos_resolve_platforms)
  foreach(_name ascend metax enflame tsingmicro dcu kunlunxin iluvatar mps cuda ppu
               mthreads musa gcu bpu hygon nvidia thead apple)
    string(JSON _name_vendor ERROR_VARIABLE _name_err GET
           "${_flagos_platforms}" accelerators "${_name}" vendor)
    if(_name_err)
      # A vendor name with no row of its own (drobotics) is nobody's spelling.
      set(_name_vendor "")
    endif()
    set(_flagos_is_${_name} FALSE)
    if(FLAGOS_ACCELERATOR STREQUAL "${_name}"
       OR FLAGOS_PLATFORM_ID STREQUAL "${_name}"
       OR (NOT _name_vendor STREQUAL ""
           AND FLAGOS_PLATFORM_VENDOR STREQUAL "${_name_vendor}"))
      set(_flagos_is_${_name} TRUE)
    endif()
  endforeach()
endmacro()
_flagos_resolve_platforms()

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
if(_flagos_is_metax)
  flagos_resolve_sdk_root(MACA_PATH "/opt/maca" MACA_PATH MACA_HOME)
  # A MetaX image may carry only the driver: then there is no MACA toolkit to
  # compile the vendor kernels against, and the FlagGems-only build falls back
  # to the CUDA toolkit for the runtime sources. The image's CUDA_HOME is that
  # toolkit. A MACA install keeps its own cu-bridge path untouched.
  if(NOT EXISTS "${MACA_PATH}/tools/cu-bridge/include"
     AND DEFINED ENV{CUDA_HOME}
     AND EXISTS "$ENV{CUDA_HOME}/include/cuda_runtime.h")
    set(CUDAToolkit_ROOT "$ENV{CUDA_HOME}")
  endif()
endif()

if(_flagos_is_cuda OR _flagos_is_ppu OR (_flagos_is_metax AND CUDAToolkit_ROOT))
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
  if(_flagos_is_ppu)
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

if(_flagos_is_ascend)
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
elseif(_flagos_is_tsingmicro)
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
elseif(_flagos_is_dcu)
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
  # Keep in sync with VENDOR_RPATH in scripts/vendor/bundle_hygon_libtorch.sh.
  set(_dcu_rpath_dirs
      "${DCU_CUDA_ROOT}/lib64" "${ROCM_PATH}/lib" "${ROCM_PATH}/hip/lib"
      "${ROCM_PATH}/lib64" "${ROCM_PATH}/aillvm/lib" "${ROCM_PATH}/llvm/lib"
      "${ROCM_PATH}/.hyhal/rocm_smi/lib" "/opt/hyhal/lib")
  list(APPEND CMAKE_INSTALL_RPATH ${_dcu_rpath_dirs})
  set(FLAGOS_VENDOR_RPATH_DIRS ${_dcu_rpath_dirs})

  # Pre-create torch::cudart so PyTorch's cuda.cmake is skipped entirely
  # (guard: if(TARGET torch::cudart) return()).
elseif(_flagos_is_gcu)
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
elseif(_flagos_is_musa)
  # Moore Threads MUSA toolkit: musart runtime. There is no libcudart shim here
  # (unlike MetaX/DCU), so nothing pretends to be CUDA -- cuda_runtime_compat
  # points at the native musa* headers and libmusart, and the runtime sources
  # under csrc/runtime/accelerator/mthreads/ call musaMalloc/musaStreamCreate/etc.
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
elseif(_flagos_is_bpu)
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
elseif(_flagos_is_kunlunxin)
  # Kunlunxin XPU SDK. The backend is native, not CUDA-boxing: the runtime
  # sources under csrc/runtime/accelerator/kunlunxin/ and the native op wrappers
  # call the SDK's own C API -- libxpurt
  # (<xpu/runtime.h>) for the runtime and the vendor xdnn library for compute --
  # so no tensor is ever re-dispatched onto the CUDA dispatch key. XPU_ROOT is
  # therefore resolved for the xpu/ headers (never copied into the wheel).
  #
  # The SDK's CUDA-shaped surface is kept for exactly one caller and one reason:
  # csrc/runtime/hooks.h's isPinnedPtr() probes cudaPointerGetAttributes as the
  # DDP fallback for pointers the flagos host allocator did not register, and
  # that is why cuda_runtime.h and the libcudart shim are still on the include
  # path and the link line. Neither carries a kernel and neither is reachable
  # from the routing layer, so neither can re-introduce boxing.
  flagos_resolve_sdk_root(XPU_ROOT "/usr/local/xpu" XPU_ROOT)
  flagos_resolve_sdk_root(XCUDART_ROOT "${XPU_ROOT}/so" XCUDART_ROOT)
  if(NOT EXISTS "${XPU_ROOT}/include/xpu/runtime.h")
    message(FATAL_ERROR
      "Kunlunxin XPU SDK not found: ${XPU_ROOT}/include/xpu/runtime.h does not "
      "exist. Set XPU_ROOT to the Kunlun XPU SDK root.")
  endif()
  if(NOT EXISTS "${XPU_ROOT}/include/cuda_runtime.h")
    message(FATAL_ERROR
      "Kunlunxin XPU SDK not found: ${XPU_ROOT}/include/cuda_runtime.h does not "
      "exist (hooks.h's isPinnedPtr probe needs the CUDA-shaped headers). Set "
      "XPU_ROOT to the Kunlun XPU SDK root.")
  endif()
  # The shim's soname tracks the SDK's CUDA-compat version (libcudart.so.11.0 on
  # XPU SDK 5.0.21) rather than a fixed one, and it sits either directly in
  # XCUDART_ROOT or one level down in lib/, so match the whole family in both
  # spots instead of pinning a filename.
  file(GLOB _kunlun_cudart_libs
    "${XCUDART_ROOT}/libcudart.so*" "${XCUDART_ROOT}/lib/libcudart.so*")
  if(NOT _kunlun_cudart_libs)
    message(FATAL_ERROR
      "Kunlun CUDA runtime not found: no libcudart.so* under ${XCUDART_ROOT} or "
      "${XCUDART_ROOT}/lib. Set XCUDART_ROOT to the directory holding the XPU "
      "CUDA-compat runtime.")
  endif()
  list(GET _kunlun_cudart_libs 0 _kunlun_cudart_lib)
  get_filename_component(_kunlun_cudart_dir "${_kunlun_cudart_lib}" DIRECTORY)
  message(STATUS "Kunlunxin XPU SDK path: ${XPU_ROOT}")
  message(STATUS "Kunlun CUDA runtime path: ${_kunlun_cudart_dir}")

  # -DXPU_LIB_DIR lets a target whose SDK keeps libxpurt elsewhere override the
  # so/ directory; the P800 ships it at ${XPU_ROOT}/so next to the cudart shim.
  set(XPU_LIB_DIR "${XPU_ROOT}/so" CACHE PATH "Kunlun XPU runtime library dir")

  add_library(cuda_runtime_compat INTERFACE IMPORTED)
  set_target_properties(cuda_runtime_compat PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${XPU_ROOT}/include"
  )
  target_link_directories(cuda_runtime_compat INTERFACE
      "${_kunlun_cudart_dir}" "${XPU_LIB_DIR}")
  # xpurt is the native runtime the kunlunxin/ sources and the xdnn op library
  # actually call; cudart is the pinned-pointer probe in hooks.h. libxpurt.so
  # carries SONAME libxpurt.so.2, so plain `xpurt` resolves against the SDK.
  target_link_libraries(cuda_runtime_compat INTERFACE cudart xpurt)

  set(CUDA_RUNTIME_LIB cuda_runtime_compat)

  # Embed both runtime paths so LD_LIBRARY_PATH is not needed at runtime.
  list(APPEND CMAKE_INSTALL_RPATH "${_kunlun_cudart_dir}" "${XPU_LIB_DIR}")
  set(FLAGOS_VENDOR_RPATH_DIRS "${_kunlun_cudart_dir}" "${XPU_LIB_DIR}")

  # Compute kernels come from xdnn (libxdnn_pytorch.so), the vendor's XPU op
  # library -- the counterpart of topsaten on GCU and mudnn on MUSA. It does not
  # ship in the XPU SDK root: it is installed inside the vendor torch package
  # (torch_xmlir/), which is why XDNN_ROOT is its own knob rather than something
  # derived from XPU_ROOT. setup.py resolves it (from $XDNN_ROOT, else the vendor
  # torch package in site-packages) and passes -DXDNN_ROOT; a direct cmake
  # invocation can set it instead, or point FLAGOS_VENDOR_TORCH_LIB at a
  # directory holding the .so (the package root and the site-packages lib/
  # install both land that way).
  if(FLAGOS_BUILD_VENDOR)
    if(NOT DEFINED XDNN_ROOT)
      set(XDNN_ROOT "$ENV{XDNN_ROOT}")
    endif()
    if(XDNN_ROOT STREQUAL "" AND DEFINED ENV{FLAGOS_VENDOR_TORCH_LIB})
      set(_kunlun_xdnn_hint "$ENV{FLAGOS_VENDOR_TORCH_LIB}")
      if(EXISTS "${_kunlun_xdnn_hint}/libxdnn_pytorch.so")
        set(XDNN_ROOT "${_kunlun_xdnn_hint}")
      elseif(EXISTS "${_kunlun_xdnn_hint}/../libxdnn_pytorch.so")
        get_filename_component(XDNN_ROOT "${_kunlun_xdnn_hint}/.." ABSOLUTE)
      endif()
    endif()
    set(XDNN_ROOT "${XDNN_ROOT}" CACHE PATH "Kunlun xdnn (vendor op library) package root")

    find_library(XDNN_PYTORCH_LIB xdnn_pytorch
        PATHS "${XDNN_ROOT}" NO_DEFAULT_PATH)
    find_path(XDNN_PYTORCH_INCLUDE_DIR xdnn_pytorch/xdnn_pytorch.h
        PATHS "${XDNN_ROOT}/include" NO_DEFAULT_PATH)
    if(NOT XDNN_PYTORCH_LIB OR NOT XDNN_PYTORCH_INCLUDE_DIR)
      message(FATAL_ERROR
        "FLAGOS_BUILD_VENDOR=ON for kunlunxin but the vendor xdnn op library was "
        "not found: need libxdnn_pytorch.so and include/xdnn_pytorch/xdnn_pytorch.h "
        "under XDNN_ROOT (currently '${XDNN_ROOT}'). xdnn ships inside the vendor "
        "torch package, not the XPU SDK, so set XDNN_ROOT to that package root, or "
        "FLAGOS_VENDOR_TORCH_LIB to a directory holding libxdnn_pytorch.so, or "
        "choose another accelerator: xdnn_pytorch is the only operator library the "
        "XPU SDK provides, so a kunlunxin build always has FLAGOS_BUILD_VENDOR=ON.")
    endif()
    set(XDNN_PYTORCH_INCLUDE_DIRS "${XDNN_PYTORCH_INCLUDE_DIR}")
    message(STATUS "xdnn op library: ${XDNN_PYTORCH_LIB}")
    message(STATUS "xdnn include: ${XDNN_PYTORCH_INCLUDE_DIR}")
    get_filename_component(_kunlun_xdnn_libdir "${XDNN_PYTORCH_LIB}" DIRECTORY)
    list(APPEND CMAKE_INSTALL_RPATH "${_kunlun_xdnn_libdir}")
    list(APPEND FLAGOS_VENDOR_RPATH_DIRS "${_kunlun_xdnn_libdir}")

    # libxdnn_pytorch.so is not self-contained: its DT_NEEDED names the vendor's
    # own CUDA-compat runtime and its component libraries, which the vendor torch
    # package keeps in a nested dependency directory rather than next to the op
    # library. Without that directory on the runtime search path the flagos
    # module fails to load (unresolved libcudart.so.12 / libxpuml.so.1) no matter
    # what the kernels do. XCUDART_ROOT does not cover it: the XPU SDK ships only
    # its own CUDA-11-shaped shim, so this is the one place the CUDA-12-compat
    # runtime comes from.
    file(GLOB _kunlun_xdnn_dep_dirs
        "${_kunlun_xdnn_libdir}/xhpc/*/dependency_so")
    foreach(_kunlun_dep_dir IN LISTS _kunlun_xdnn_dep_dirs)
      if(IS_DIRECTORY "${_kunlun_dep_dir}")
        list(APPEND CMAKE_INSTALL_RPATH "${_kunlun_dep_dir}")
        list(APPEND FLAGOS_VENDOR_RPATH_DIRS "${_kunlun_dep_dir}")
      endif()
    endforeach()
  endif()

  # Pre-create torch::cudart so PyTorch's cuda.cmake returns early while the
  # host-only runtime/op sources are compiled (no NVIDIA toolkit probe).
  add_library(torch::cudart INTERFACE IMPORTED)
  if(NOT TARGET CUDA::nvToolsExt)
    add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
  endif()
elseif(_flagos_is_iluvatar)
  # Iluvatar corex SDK. Like the Kunlunxin XPU SDK it is CUDA-shaped and ships
  # everything already: cuda_runtime.h under <corex>/include and
  # libcudart.so.10.2 / libcupti.so.10.2 under <corex>/lib64 (corex 4.5.0).
  # Nothing is copied into the wheel -- the vendor torch links the same files.
  # Crucially, project_languages stays CXX/C: there is no project(CUDA) and so
  # no nvcc compiler-id probe. That probe is what fails here, because corex's
  # /usr/local/corex/bin/nvcc is a 194-byte banner stub and the real device
  # compiler (clang, triple bi-iluvatar-ilurt) is not a CUDA compiler id.
  flagos_resolve_sdk_root(COREX_PATH "/usr/local/corex" COREX_PATH COREX_HOME)
  if(NOT EXISTS "${COREX_PATH}/include/cuda_runtime.h")
    message(FATAL_ERROR
      "Iluvatar corex SDK not found: ${COREX_PATH}/include/cuda_runtime.h does "
      "not exist. Set COREX_PATH to the corex SDK root.")
  endif()
  file(GLOB _iluvatar_cudart_libs
    "${COREX_PATH}/lib64/libcudart.so*" "${COREX_PATH}/lib/libcudart.so*")
  if(NOT _iluvatar_cudart_libs)
    message(FATAL_ERROR
      "Iluvatar CUDA runtime not found: no libcudart.so* under ${COREX_PATH}/lib64 "
      "or ${COREX_PATH}/lib. Set COREX_PATH to the corex SDK root.")
  endif()
  list(GET _iluvatar_cudart_libs 0 _iluvatar_cudart_lib)
  get_filename_component(_iluvatar_cudart_dir "${_iluvatar_cudart_lib}" DIRECTORY)
  message(STATUS "Iluvatar corex SDK path: ${COREX_PATH}")
  message(STATUS "Iluvatar CUDA runtime path: ${_iluvatar_cudart_dir}")

  add_library(cuda_runtime_compat INTERFACE IMPORTED)
  set_target_properties(cuda_runtime_compat PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${COREX_PATH}/include"
  )
  target_link_directories(cuda_runtime_compat INTERFACE "${_iluvatar_cudart_dir}")
  target_link_libraries(cuda_runtime_compat INTERFACE cudart)

  set(CUDA_RUNTIME_LIB cuda_runtime_compat)

  # Embed the CUDA-compat runtime path so LD_LIBRARY_PATH is not needed at runtime.
  list(APPEND CMAKE_INSTALL_RPATH "${_iluvatar_cudart_dir}")
  set(FLAGOS_VENDOR_RPATH_DIRS "${_iluvatar_cudart_dir}")

  # Native, non-boxing vendor kernels. corex's CUDA-named C ABI *is* the vendor
  # implementation: cudnn.h maps to ixdnn (198 exported ixdnn* symbols in
  # libcudnn.so.7) and cublas_v2.h to ixblas (272 in libcublas), so
  # aten/backends/iluvatar/ includes <cudnn.h>/<cublas_v2.h> and calls the C API
  # directly against the flagos tensor's data_ptr() -- no device retag, no CUDA
  # dispatch key. Both live in the same corex lib64 the cudart shim does, so the
  # directory is already on the RPATH; -lcudnn/-lcublas resolve there.
  if(FLAGOS_BUILD_VENDOR)
    find_library(FLAGOS_ILUVATAR_CUDNN_LIB cudnn
        PATHS "${COREX_PATH}/lib64" "${COREX_PATH}/lib" NO_DEFAULT_PATH)
    find_library(FLAGOS_ILUVATAR_CUBLAS_LIB cublas
        PATHS "${COREX_PATH}/lib64" "${COREX_PATH}/lib" NO_DEFAULT_PATH)
    find_path(FLAGOS_ILUVATAR_VENDOR_INCLUDE_DIR cudnn.h
        PATHS "${COREX_PATH}/include" NO_DEFAULT_PATH)
    if(NOT FLAGOS_ILUVATAR_CUDNN_LIB OR NOT FLAGOS_ILUVATAR_CUBLAS_LIB
       OR NOT FLAGOS_ILUVATAR_VENDOR_INCLUDE_DIR)
      message(FATAL_ERROR
        "FLAGOS_BUILD_VENDOR=ON for iluvatar but the corex vendor op libraries "
        "were not found: need libcudnn.so + libcublas.so under "
        "${COREX_PATH}/lib64 (or lib) and cudnn.h under ${COREX_PATH}/include. "
        "corex's CUDA-named C ABI (ixdnn/ixblas) is the only operator library "
        "the SDK provides, so set COREX_PATH to the corex SDK root, or build "
        "with -DFLAGOS_BUILD_VENDOR=OFF to route every op to FlagGems/CPU.")
    endif()
    message(STATUS "Iluvatar vendor cudnn: ${FLAGOS_ILUVATAR_CUDNN_LIB}")
    message(STATUS "Iluvatar vendor cublas: ${FLAGOS_ILUVATAR_CUBLAS_LIB}")
    message(STATUS "Iluvatar vendor include: ${FLAGOS_ILUVATAR_VENDOR_INCLUDE_DIR}")
    get_filename_component(_iluvatar_vendor_libdir
        "${FLAGOS_ILUVATAR_CUDNN_LIB}" DIRECTORY)
    list(APPEND CMAKE_INSTALL_RPATH "${_iluvatar_vendor_libdir}")
    list(APPEND FLAGOS_VENDOR_RPATH_DIRS "${_iluvatar_vendor_libdir}")
  endif()

  # Pre-create torch::cudart so PyTorch's cuda.cmake returns early while the
  # host-only runtime sources are compiled (no NVIDIA toolkit probe).
  add_library(torch::cudart INTERFACE IMPORTED)
  # The ABI is the *linked* torch's, not the vendor torch's. Iluvatar builds
  # against the CPU torch wheel, and the two used to disagree: corex's torch is a
  # real CUDA build reporting torch._C._GLIBCXX_USE_CXX11_ABI == False, so a
  # hardcoded -D_GLIBCXX_USE_CXX11_ABI=0 made libtorch_fl.so reference the
  # pre-cxx11 mangling of e.g. c10::ivalue::ConstantString::create(std::string)
  # -- which the CPU wheel (new ABI) does not define, so the import dies with
  # "undefined symbol: _ZN3c106ivalue14ConstantString6createENSt7__cxx1112...".
  # Read the value from the interpreter that will own the build instead, exactly
  # as setup.py::_torch_abi does, so the CMake and setup.py paths agree. Guarded
  # because a configure that never builds a torch-linked target still runs here.
  if(DEFINED FLAGOS_TORCH_CXX11_ABI)
    set(_iluvatar_cxx11_abi "${FLAGOS_TORCH_CXX11_ABI}")
  else()
    execute_process(
      COMMAND "${PYTHON_EXECUTABLE}" -c
              "import torch; print('1' if torch._C._GLIBCXX_USE_CXX11_ABI else '0')"
      OUTPUT_VARIABLE _iluvatar_cxx11_abi
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
      RESULT_VARIABLE _iluvatar_cxx11_ret)
    if(NOT _iluvatar_cxx11_ret EQUAL 0 OR NOT _iluvatar_cxx11_abi MATCHES "^[01]$")
      message(FATAL_ERROR
          "Iluvatar build: cannot read torch._C._GLIBCXX_USE_CXX11_ABI from "
          "${PYTHON_EXECUTABLE}. Pass -DFLAGOS_TORCH_CXX11_ABI=0/1, or build "
          "through setup.py, which resolves it from the same interpreter.")
    endif()
  endif()
  if(_iluvatar_cxx11_abi STREQUAL "0")
    add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)
  endif()
  # __ILUVATAR__ exists for corex's *torch* headers (c10/cuda/CUDAException.h keys
  # C10_CUDA_CHECK on it to pick the 6-argument c10_cuda_check_implementation
  # libc10_cuda.so exports). This build includes no c10/cuda header and links no
  # libc10_cuda.so, so the define would select an ABI no linked library backs;
  # it is deliberately not added here.
  if(NOT TARGET CUDA::nvToolsExt)
    add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
  endif()
elseif(_flagos_is_metax)
  # MetaX SDK: use cu-bridge headers and runtime_cu library.
  # MACA_PATH was resolved in the metax branch above (MACA_PATH -> MACA_HOME ->
  # /opt/maca), and is reported here for the build log.
  message(STATUS "MetaX SDK path: ${MACA_PATH}")

  # The maca runtime (mcblas/mcdnn/...) comes from the target's /opt/maca driver
  # install; it is never bundled. /opt/maca is kept as an explicit fallback in
  # case the build machine's MACA_PATH is a non-default location.
  set(FLAGOS_VENDOR_RPATH_DIRS
      "${MACA_PATH}/lib" "${MACA_PATH}/lib64" "/opt/maca/lib" "/opt/maca/lib64")

  # The MACA SDK itself is only needed when vendor kernels are built: they are
  # the CUDA tree compiled against its cu-bridge headers. Without them the
  # build still compiles -- the NVIDIA CUDA toolkit the runtime sources include
  # is enough -- and pointing the compat target at an SDK root that is not
  # installed fails configure, which is what a Linux image holding only the
  # driver does.
  # An INTERFACE target may only name directories that exist, so each one is
  # added when it is there rather than listed and reported missing at
  # generate time.
  set(_metax_compat_include "")
  foreach(_dir "${MACA_PATH}/tools/cu-bridge/include" "${MACA_PATH}/include"
               "${MACA_PATH}/include/mcr")
    if(EXISTS "${_dir}")
      list(APPEND _metax_compat_include "${_dir}")
    endif()
  endforeach()

  add_library(cuda_runtime_compat INTERFACE IMPORTED)
  set_target_properties(cuda_runtime_compat PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${_metax_compat_include}"
  )
  if(FLAGOS_BUILD_VENDOR)
    # The vendor kernels are the CUDA tree built against cu-bridge, which is
    # what runtime_cu names; no SDK root means nothing to build them with.
    if(NOT EXISTS "${MACA_PATH}/tools/cu-bridge/include")
      message(FATAL_ERROR
          "FLAGOS_BUILD_VENDOR=ON but the MetaX cu-bridge headers were not "
          "found (${MACA_PATH}/tools/cu-bridge/include). Install the MACA "
          "toolkit and set MACA_PATH, or build with -DFLAGOS_BUILD_VENDOR=OFF.")
    endif()
    target_link_directories(cuda_runtime_compat INTERFACE "${MACA_PATH}/lib")
    target_link_libraries(cuda_runtime_compat INTERFACE runtime_cu)
  else()
    # FlagGems-only: the runtime sources include cuda_runtime.h, which cu-bridge
    # provides (a CUDA-compatible runtime API on top of MACA), and the MACA
    # driver supplies the symbols at run time. A CUDA toolkit is used only when
    # the build was given one; requiring it would fail every MetaX image that
    # has MACA and no CUDA.
    if(CUDAToolkit_ROOT)
      find_package(CUDAToolkit REQUIRED)
      target_link_libraries(cuda_runtime_compat INTERFACE CUDA::cudart)
    else()
      message(STATUS "MetaX without MACA: no CUDA toolkit either, no runtime lib added")
    endif()
  endif()

  set(CUDA_RUNTIME_LIB cuda_runtime_compat)

  # Pre-create torch::cudart so PyTorch's cuda.cmake is skipped entirely.
  # cuda.cmake guard: if(TARGET torch::cudart) return() endif()
  add_library(torch::cudart INTERFACE IMPORTED)

  if(NOT TARGET CUDA::nvToolsExt)
    add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
  endif()
elseif(_flagos_is_mps)
  # Apple Silicon (M-series) via Metal. There is no vendor SDK to locate and no
  # CUDA-compat layer: the runtime talks to Metal/Foundation, which the macOS
  # SDK provides, and there is no library under the wheel to link against. So
  # there is no CUDA_RUNTIME_LIB to hand the other targets.
  if(NOT APPLE)
    message(FATAL_ERROR
        "FLAGOS_ACCELERATOR=mps selects the Metal backend, which builds only on "
        "an Apple host with the macOS SDK.")
  endif()

  set(CUDA_RUNTIME_LIB "")

  # Pre-create torch::cudart so PyTorch's cuda.cmake returns early
  # (guard: if(TARGET torch::cudart) return()) rather than probing for an NVIDIA
  # toolkit that is not there.
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

if(_flagos_is_dcu)
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
elseif(_flagos_is_metax)
  message(STATUS "MetaX build: CUDA boxing kernels + bundled maca libtorch (no native mxcc)")
  if(FLAGOS_BUILD_VENDOR)
    # The vendor kernels are the CUDA tree built against cu-bridge; MACA's
    # compat layer is what they call, so its include and lib dirs are needed
    # on the compile and link lines. Without FLAGOS_BUILD_VENDOR nothing here
    # refers to MACA and none of these is added.
    include_directories("${MACA_PATH}/tools/cu-bridge/include"
                        "${MACA_PATH}/include" "${MACA_PATH}/include/mcr")
    link_directories("${MACA_PATH}/lib")
  endif()
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
