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

set(LIBRARY_NAME torch_bindings)

file(GLOB_RECURSE SOURCE_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cc"
)

add_library(${LIBRARY_NAME} SHARED ${SOURCE_FILES})

if(FLAGOS_ACCELERATOR STREQUAL "metax")
  # maca torch headers are gated on USE_MACA (see csrc/CMakeLists.txt).
  target_compile_definitions(${LIBRARY_NAME} PRIVATE USE_MACA=1)
elseif(FLAGOS_ACCELERATOR STREQUAL "musa")
  target_compile_definitions(${LIBRARY_NAME} PRIVATE USE_MUSA=1)
  if(NOT MUSA_HOME)
    set(MUSA_HOME "$ENV{MUSA_HOME}")
  endif()
  if(NOT MUSA_HOME)
    set(MUSA_HOME "/usr/local/musa")
  endif()
  target_include_directories(${LIBRARY_NAME} PRIVATE "${MUSA_HOME}/include")
  target_link_directories(${LIBRARY_NAME} PRIVATE "${MUSA_HOME}/lib")
endif()

target_link_libraries(${LIBRARY_NAME} PRIVATE torch_python_library torch_fl)

if(FLAGOS_ACCELERATOR STREQUAL "musa")
  target_link_libraries(${LIBRARY_NAME} PRIVATE musart)
endif()

if(WIN32)
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    target_link_libraries(${LIBRARY_NAME} PRIVATE ${Python3_LIBRARIES})
elseif(APPLE)
    set_target_properties(${LIBRARY_NAME} PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
endif()

target_link_directories(${LIBRARY_NAME} PRIVATE ${PYTORCH_INSTALL_DIR}/lib)

# Same reasoning as csrc/CMakeLists.txt: keep the build machine's torch/lib out
# of the installed RUNPATH and resolve a bundled forked libtorch via $ORIGIN.
# libtorch_bindings.so is loaded as torch_fl._C, i.e. always after
# torch_fl/__init__.py has imported torch.
flagos_set_portable_rpath(${LIBRARY_NAME})

install(TARGETS ${LIBRARY_NAME}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
