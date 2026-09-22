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

# project() languages and the source subdir come from the shared platform table
# (cmake/flagos_platforms.json, loaded by the top-level CMakeLists): CUDA CXX C
# for the CUDA-toolchain platforms, CXX C otherwise.
project(FLAGOS_RUNTIME ${FLAGOS_PLATFORM_LANGUAGES})

set(LIBRARY_NAME flagos)

# The runtime sources live under <runtime_dir>/, except DCU which reuses the CUDA
# runtime sources (DTK's CUDA compatibility toolkit provides cuda_runtime.h and a
# libcudart shim over libgalaxyhip, so they build unchanged) and adds its own
# DTK-core ABI shim below.
file(GLOB_RECURSE SOURCE_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/${FLAGOS_PLATFORM_RUNTIME_DIR}/*.cc"
)

add_library(${LIBRARY_NAME} SHARED ${SOURCE_FILES})

# The platform's USE_* macro (USE_ASCEND, USE_DCU, ...), from the table: the
# runtime sources gate their per-platform headers on it.
if(NOT FLAGOS_PLATFORM_USE_MACRO STREQUAL "")
  target_compile_definitions(${LIBRARY_NAME} PRIVATE ${FLAGOS_PLATFORM_USE_MACRO}=1)
endif()

if(FLAGOS_ACCELERATOR STREQUAL "ascend")
  find_path(CANN_INCLUDE_DIR acl/acl.h
      PATHS $ENV{ASCEND_HOME}/include
            /usr/local/Ascend/ascend-toolkit/latest/include)
  find_library(ACL_LIB ascendcl
      PATHS $ENV{ASCEND_HOME}/lib64
            /usr/local/Ascend/ascend-toolkit/latest/lib64)

  if(NOT CANN_INCLUDE_DIR OR CANN_INCLUDE_DIR STREQUAL "CANN_INCLUDE_DIR-NOTFOUND")
    message(FATAL_ERROR
        "Ascend runtime selected, but CANN headers were not found. "
        "Set ASCEND_HOME to a valid Ascend installation or install the toolkit "
        "under /usr/local/Ascend/ascend-toolkit/latest.")
  endif()
  if(NOT ACL_LIB OR ACL_LIB STREQUAL "ACL_LIB-NOTFOUND")
    message(FATAL_ERROR
        "Ascend runtime selected, but the ascendcl library was not found. "
        "Set ASCEND_HOME to a valid Ascend installation or install the toolkit "
        "under /usr/local/Ascend/ascend-toolkit/latest.")
  endif()

  target_include_directories(${LIBRARY_NAME} PUBLIC
      ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR} ${CANN_INCLUDE_DIR})
  # memory.cc drains the shared default ACL stream (acl_stream.h) before any
  # host-visible memcpy now that ops dispatch asynchronously.
  target_link_libraries(${LIBRARY_NAME} PRIVATE ${ACL_LIB})
elseif(FLAGOS_ACCELERATOR STREQUAL "metax")
  target_include_directories(${LIBRARY_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR})
  target_link_libraries(${LIBRARY_NAME} PRIVATE ${CUDA_RUNTIME_LIB})
elseif(FLAGOS_ACCELERATOR STREQUAL "tsingmicro")
  target_include_directories(${LIBRARY_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR} /usr/local/kuiper/include)
  target_link_directories(${LIBRARY_NAME} PRIVATE /usr/local/kuiper/lib)
  target_link_libraries(${LIBRARY_NAME} PRIVATE hpgr)
elseif(FLAGOS_ACCELERATOR STREQUAL "dcu")
  # dcu/ contains only the DTK-core ABI shim below; the runtime sources themselves
  # are the CUDA ones (FLAGOS_PLATFORM_RUNTIME_DIR == "cuda").
  target_include_directories(${LIBRARY_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR})
  target_link_libraries(${LIBRARY_NAME} PRIVATE ${CUDA_RUNTIME_LIB})

  add_library(flagos_dtk_core_compat SHARED
      "${CMAKE_CURRENT_SOURCE_DIR}/dcu/dtk_core_compat.cc")
  set_target_properties(flagos_dtk_core_compat PROPERTIES
      POSITION_INDEPENDENT_CODE ON)
  flagos_set_portable_rpath(flagos_dtk_core_compat)
  install(TARGETS flagos_dtk_core_compat
      LIBRARY DESTINATION lib_dcu)
elseif(FLAGOS_ACCELERATOR STREQUAL "gcu")
  target_include_directories(${LIBRARY_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR} ${TOPS_ROOT}/include)
  target_link_directories(${LIBRARY_NAME} PRIVATE ${TOPS_ROOT}/lib)
  target_link_libraries(${LIBRARY_NAME} PRIVATE topsrt)
elseif(FLAGOS_ACCELERATOR STREQUAL "musa")
  # Moore Threads MUSA. The toolkit ships no libcudart shim (unlike MetaX/DCU),
  # so musa/ holds a native port of the CUDA runtime sources onto the musa* API.
  # musa_runtime.h is plain C, so host g++ compiles it -- no mcc, no CUDA
  # language, and libmusart.so.5 is the only runtime dependency.
  target_include_directories(${LIBRARY_NAME} PUBLIC
      ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR} ${MUSA_HOME}/include)
  target_link_directories(${LIBRARY_NAME} PRIVATE ${MUSA_HOME}/lib)
  target_link_libraries(${LIBRARY_NAME} PRIVATE musart)
elseif(FLAGOS_ACCELERATOR STREQUAL "bpu")
  # D-Robotics RDK BPU. The UCP allocator (libhbucp) provides device memory and
  # libbpu the core/task API; both ship in the board image under /usr/hobot/lib
  # with headers at /usr/include/hobot, so there is no SDK root to configure.
  find_library(HBUCP_LIB hbucp PATHS ${HOBOT_LIB_DIR} /usr/hobot/lib)
  find_library(LIBBPU_LIB bpu PATHS ${HOBOT_LIB_DIR} /usr/hobot/lib)
  find_path(HOBOT_UCP_INCLUDE_DIR hb_ucp_sys.h
      PATHS ${HOBOT_INCLUDE_DIR} /usr/include/hobot)

  if(NOT HBUCP_LIB OR NOT LIBBPU_LIB OR NOT HOBOT_UCP_INCLUDE_DIR)
    message(FATAL_ERROR
        "FLAGOS_ACCELERATOR=bpu selected, but the Horizon runtime was not found. "
        "Expected libhbucp/libbpu under /usr/hobot/lib and hb_ucp_sys.h under "
        "/usr/include/hobot (set HOBOT_LIB_DIR / HOBOT_INCLUDE_DIR to override). "
        "These ship in the RDK BPU board image.")
  endif()

  target_include_directories(${LIBRARY_NAME} PUBLIC
      ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}
      ${HOBOT_UCP_INCLUDE_DIR} ${HOBOT_UCP_INCLUDE_DIR}/dnn)
  target_link_libraries(${LIBRARY_NAME} PRIVATE ${HBUCP_LIB} ${LIBBPU_LIB})
else()
  target_include_directories(${LIBRARY_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR})

  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(${LIBRARY_NAME} PRIVATE CUDA::cudart)
endif()

# libflagos.so needs no torch symbols at all (only the vendor runtime), so the
# inherited ${PYTORCH_INSTALL_DIR}/lib RUNPATH entry is pure build-machine
# leakage. Vendor driver dirs are preserved.
flagos_set_portable_rpath(${LIBRARY_NAME})

install(TARGETS ${LIBRARY_NAME}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(FILES ${CMAKE_SOURCE_DIR}/csrc/include/flagos.h
    DESTINATION include
)
