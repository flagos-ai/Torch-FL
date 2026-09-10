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

if(ACCELERATOR STREQUAL "ascend")
  project(FLAGOS_RUNTIME CXX C)
elseif(ACCELERATOR STREQUAL "metax")
  project(FLAGOS_RUNTIME CXX C)
elseif(ACCELERATOR STREQUAL "tsingmicro")
  project(FLAGOS_RUNTIME CXX C)
elseif(ACCELERATOR STREQUAL "bpu")
  project(FLAGOS_RUNTIME CXX C)
elseif(ACCELERATOR STREQUAL "dcu")
  project(FLAGOS_RUNTIME CXX C)
elseif(ACCELERATOR STREQUAL "gcu")
  project(FLAGOS_RUNTIME CXX C)
elseif(ACCELERATOR STREQUAL "musa")
  project(FLAGOS_RUNTIME CXX C)
else()
  project(FLAGOS_RUNTIME CUDA CXX C)
endif()

set(LIBRARY_NAME flagos)

if(ACCELERATOR STREQUAL "ascend")
  file(GLOB_RECURSE SOURCE_FILES
      "${CMAKE_CURRENT_SOURCE_DIR}/ascend/*.cc"
  )

  add_library(${LIBRARY_NAME} SHARED ${SOURCE_FILES})

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
  # host-visible memcpy now that ops dispatch asynchronously; acl_stream.h is
  # guarded by USE_ASCEND, so this target must define it.
  target_compile_definitions(${LIBRARY_NAME} PRIVATE USE_ASCEND=1)
  target_link_libraries(${LIBRARY_NAME} PRIVATE ${ACL_LIB})
elseif(ACCELERATOR STREQUAL "metax")
  file(GLOB SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/metax/*.cc")

  add_library(${LIBRARY_NAME} SHARED ${SOURCE_FILES})

  target_include_directories(${LIBRARY_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR})
  target_link_libraries(${LIBRARY_NAME} PRIVATE ${CUDA_RUNTIME_LIB})
elseif(ACCELERATOR STREQUAL "tsingmicro")
  file(GLOB SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/tsingmicro/*.cc")

  add_library(${LIBRARY_NAME} SHARED ${SOURCE_FILES})

  target_include_directories(${LIBRARY_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR} /usr/local/kuiper/include)
  target_link_directories(${LIBRARY_NAME} PRIVATE /usr/local/kuiper/lib)
  target_link_libraries(${LIBRARY_NAME} PRIVATE hpgr)
elseif(ACCELERATOR STREQUAL "dcu")
  # DTK's CUDA compatibility toolkit provides cuda_runtime.h and a libcudart
  # shim over libgalaxyhip, so the CUDA runtime sources build unchanged --
  # no nvcc (plain host g++). dcu/ adds the DTK-core ABI shim (a separate target
  # below) plus stream_registry.cc, which supplies the per-device current-stream
  # entry points the CUDA sources leave to c10::cuda on NVIDIA -- see the file's
  # header comment and runtime/guard.h's USE_DCU branches.
  file(GLOB SOURCE_FILES
      "${CMAKE_CURRENT_SOURCE_DIR}/cuda/*.cc"
      "${CMAKE_CURRENT_SOURCE_DIR}/dcu/stream_registry.cc")

  add_library(${LIBRARY_NAME} SHARED ${SOURCE_FILES})

  target_include_directories(${LIBRARY_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR})
  target_link_libraries(${LIBRARY_NAME} PRIVATE ${CUDA_RUNTIME_LIB})

  add_library(flagos_dtk_core_compat SHARED
      "${CMAKE_CURRENT_SOURCE_DIR}/dcu/dtk_core_compat.cc")
  set_target_properties(flagos_dtk_core_compat PROPERTIES
      POSITION_INDEPENDENT_CODE ON)
  flagos_set_portable_rpath(flagos_dtk_core_compat)
  install(TARGETS flagos_dtk_core_compat
      LIBRARY DESTINATION lib_dcu)
elseif(ACCELERATOR STREQUAL "gcu")
  file(GLOB SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/gcu/*.cc")

  add_library(${LIBRARY_NAME} SHARED ${SOURCE_FILES})

  target_include_directories(${LIBRARY_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR} ${TOPS_ROOT}/include)
  target_link_directories(${LIBRARY_NAME} PRIVATE ${TOPS_ROOT}/lib)
  target_link_libraries(${LIBRARY_NAME} PRIVATE topsrt)
elseif(ACCELERATOR STREQUAL "musa")
  # Moore Threads MUSA. The toolkit ships no libcudart shim (unlike MetaX/DCU),
  # so musa/ holds a native port of the CUDA runtime sources onto the musa* API.
  # musa_runtime.h is plain C, so host g++ compiles it -- no mcc, no CUDA
  # language, and libmusart.so.5 is the only runtime dependency.
  file(GLOB SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/musa/*.cc")

  add_library(${LIBRARY_NAME} SHARED ${SOURCE_FILES})

  target_include_directories(${LIBRARY_NAME} PUBLIC
      ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR} ${MUSA_HOME}/include)
  target_link_directories(${LIBRARY_NAME} PRIVATE ${MUSA_HOME}/lib)
  target_link_libraries(${LIBRARY_NAME} PRIVATE musart)
elseif(ACCELERATOR STREQUAL "bpu")
  # D-Robotics RDK BPU. The UCP allocator (libhbucp) provides device memory and
  # libbpu the core/task API; both ship in the board image under /usr/hobot/lib
  # with headers at /usr/include/hobot, so there is no SDK root to configure.
  file(GLOB SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/bpu/*.cc")

  add_library(${LIBRARY_NAME} SHARED ${SOURCE_FILES})

  find_library(HBUCP_LIB hbucp PATHS ${HOBOT_LIB_DIR} /usr/hobot/lib)
  find_library(LIBBPU_LIB bpu PATHS ${HOBOT_LIB_DIR} /usr/hobot/lib)
  find_path(HOBOT_UCP_INCLUDE_DIR hb_ucp_sys.h
      PATHS ${HOBOT_INCLUDE_DIR} /usr/include/hobot)

  if(NOT HBUCP_LIB OR NOT LIBBPU_LIB OR NOT HOBOT_UCP_INCLUDE_DIR)
    message(FATAL_ERROR
        "ACCELERATOR=bpu selected, but the Horizon runtime was not found. "
        "Expected libhbucp/libbpu under /usr/hobot/lib and hb_ucp_sys.h under "
        "/usr/include/hobot (set HOBOT_LIB_DIR / HOBOT_INCLUDE_DIR to override). "
        "These ship in the RDK BPU board image.")
  endif()

  target_include_directories(${LIBRARY_NAME} PUBLIC
      ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}
      ${HOBOT_UCP_INCLUDE_DIR} ${HOBOT_UCP_INCLUDE_DIR}/dnn)
  target_link_libraries(${LIBRARY_NAME} PRIVATE ${HBUCP_LIB} ${LIBBPU_LIB})
else()
  file(GLOB SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/cuda/*.cc")

  add_library(${LIBRARY_NAME} SHARED ${SOURCE_FILES})

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
