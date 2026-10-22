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

# Portable RUNPATH for the installed native libs.
#
# The problem: CMakeLists.txt appends "${PYTORCH_INSTALL_DIR}/lib" to
# CMAKE_INSTALL_RPATH so an in-place build finds the active torch wheel. That is
# the *build machine's* interpreter path (e.g.
# /nfs/.../envs/torch-fl-210/lib/python3.12/site-packages/torch/lib), which does
# not exist on a target machine -- and worse, on a machine that does have that
# path it silently wins over the interpreter actually running, so a py3.12 env
# can load another env's libtorch.
#
# Dropping it is safe: libtorch_fl.so is only ever loaded through
# `import torch_fl` -> torch_fl._C -> libtorch_bindings.so, which happens after
# torch_fl/__init__.py has already run `import torch`. libc10 / libtorch_cpu are
# mapped by then, so the loader satisfies those DT_NEEDED entries by soname from
# the already-loaded set. This is why the MetaX self-contained wheel works today.
#
# Anything else the backend put on CMAKE_INSTALL_RPATH (vendor driver dirs, the
# FlagGems liboperators dir) is preserved, so in-place builds keep working.
#
# flagos_set_portable_rpath(<target> [EXTRA_DIRS <dir>...])
function(flagos_set_portable_rpath _target)
  cmake_parse_arguments(_ARG "" "" "EXTRA_DIRS" ${ARGN})

  if(NOT UNIX OR APPLE)
    return()
  endif()
  if(NOT TARGET ${_target})
    return()
  endif()

  # $ORIGIN is torch_fl/lib/ for every installed target. A sibling bundle dir
  # (torch_fl/lib_maca, lib_dcu, lib_ppu) is reached via $ORIGIN/../<name>.
  set(_rpath "$ORIGIN" "$ORIGIN/lib")
  if(FLAGOS_BUNDLE_LIBDIR AND NOT FLAGOS_BUNDLE_LIBDIR STREQUAL "lib")
    list(APPEND _rpath "$ORIGIN/../${FLAGOS_BUNDLE_LIBDIR}")
  endif()

  foreach(_dir IN LISTS _ARG_EXTRA_DIRS FLAGOS_VENDOR_RPATH_DIRS)
    if(_dir)
      list(APPEND _rpath "${_dir}")
    endif()
  endforeach()

  # Inherit whatever the backend branches appended, minus the build machine's
  # torch/lib and minus the $ORIGIN entries already placed above.
  foreach(_dir IN LISTS CMAKE_INSTALL_RPATH)
    if(NOT _dir MATCHES "^\\$ORIGIN"
       AND NOT (PYTORCH_INSTALL_DIR AND _dir STREQUAL "${PYTORCH_INSTALL_DIR}/lib"))
      list(APPEND _rpath "${_dir}")
    endif()
  endforeach()

  list(REMOVE_DUPLICATES _rpath)
  string(REPLACE ";" ":" _rpath_str "${_rpath}")

  set_target_properties(${_target} PROPERTIES
    INSTALL_RPATH "${_rpath_str}"
    # Stop CMake appending the imported torch target's build-machine link dir.
    INSTALL_RPATH_USE_LINK_PATH OFF
    BUILD_WITH_INSTALL_RPATH ON
  )
endfunction()
