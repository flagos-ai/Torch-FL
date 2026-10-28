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

# Platform/build matrix loaded from cmake/flagos_platforms.json.
#
# Include this from the top-level CMakeLists after FLAGOS_ACCELERATOR is set and
# after flagos_pin_build_switch() is defined. It publishes, in the including
# scope (and therefore to every add_subdirectory below it):
#
#   FLAGOS_KERNEL_SWITCHES            the five FLAGOS_BUILD_* names, in order
#   FLAGOS_PLATFORM_LANGUAGES         project() language list
#   FLAGOS_PLATFORM_VENDOR            the platform's vendor (apple, nvidia, ...)
#   FLAGOS_ATEN_VENDOR_DIRS           accelerator keys owning an
#                                     aten/backends/<key>/ kernel tree
#   FLAGOS_PLATFORM_ATEN_DIRS         the ones this platform's kernels are in
#   FLAGOS_PLATFORM_RUNTIME_DIR       csrc/runtime/accelerator source subdir
#   FLAGOS_PLATFORM_USE_MACRO         the platform's USE_* define ("" if none)
#   FLAGOS_PLATFORM_WRITES_MARKER     ON/OFF: install the flagos_platform file
#   FLAGOS_PLATFORM_C10_CUDA_CONFIGURE ON/OFF: define
#                                     C10_CUDA_NO_CMAKE_CONFIGURE_FILE
#   FLAGOS_BUNDLE_LIBDIR              bundled-libtorch dir inside the wheel
#
# It also defines flagos_apply_platform_kernel_policy(), which the top-level
# CMakeLists calls after option() so an explicit -D can still be told apart from
# the option() default (see the *_EXPLICIT loop there).
#
# Only the accelerator's own SDK resolution (env var names, default install
# dirs) stays in the CMakeLists files; that is the SDK-roots phase of #376.

set(_flagos_platforms_json "${CMAKE_CURRENT_LIST_DIR}/flagos_platforms.json")
if(NOT EXISTS "${_flagos_platforms_json}")
  message(FATAL_ERROR "flagos platform table not found: ${_flagos_platforms_json}")
endif()
file(READ "${_flagos_platforms_json}" _flagos_platforms)

# JSON array at <path>... -> a CMake list in <out>.
function(flagos_json_list out json)
  set(_path ${ARGN})
  string(JSON _len LENGTH "${json}" ${_path})
  set(_items "")
  if(_len GREATER 0)
    math(EXPR _last "${_len} - 1")
    foreach(_i RANGE ${_last})
      string(JSON _item GET "${json}" ${_path} ${_i})
      list(APPEND _items "${_item}")
    endforeach()
  endif()
  set(${out} "${_items}" PARENT_SCOPE)
endfunction()

# Reject an unknown accelerator here, the way setup.py does, instead of falling
# through to the CUDA toolchain. The list in the message is the table's own.
#
# A vendor name is a legal value even though only the platform id keys a row
# (`FLAGOS_ACCELERATOR=apple` is the mps build); setup.py resolves it before
# configuring, and a direct cmake invocation resolves it here, so every read
# below can stay keyed on the row.
string(JSON _flagos_accel ERROR_VARIABLE _flagos_accel_err
       GET "${_flagos_platforms}" accelerators "${FLAGOS_ACCELERATOR}")
if(_flagos_accel_err)
  string(JSON _flagos_accel_count LENGTH "${_flagos_platforms}" accelerators)
  math(EXPR _flagos_accel_last "${_flagos_accel_count} - 1")
  set(_flagos_accel_by_vendor "")
  foreach(_i RANGE ${_flagos_accel_last})
    string(JSON _name MEMBER "${_flagos_platforms}" accelerators ${_i})
    string(JSON _name_vendor GET "${_flagos_platforms}" accelerators "${_name}" vendor)
    if(_name_vendor STREQUAL "${FLAGOS_ACCELERATOR}")
      set(_flagos_accel_by_vendor "${_name}")
    endif()
  endforeach()
  if(NOT _flagos_accel_by_vendor STREQUAL "")
    set(FLAGOS_ACCELERATOR "${_flagos_accel_by_vendor}")
    set(_flagos_accel_err "")
  endif()
endif()
if(_flagos_accel_err)
  string(JSON _flagos_accel_count LENGTH "${_flagos_platforms}" accelerators)
  math(EXPR _flagos_accel_last "${_flagos_accel_count} - 1")
  set(_flagos_known "")
  foreach(_i RANGE ${_flagos_accel_last})
    string(JSON _name MEMBER "${_flagos_platforms}" accelerators ${_i})
    list(APPEND _flagos_known "${_name}")
  endforeach()
  list(JOIN _flagos_known ", " _flagos_known_str)
  message(FATAL_ERROR
    "Unknown FLAGOS_ACCELERATOR '${FLAGOS_ACCELERATOR}'. Expected one of: "
    "${_flagos_known_str}.")
endif()

# The table row this build belongs to, and every other name for it that an
# environment may carry. setup.py records them (torch_fl/_platform_alias.py) and
# passes them here; this is their CMake-side twin, so a wheel built for a
# platform enters that platform's arms whichever half of the name it was told.
# `_flagos_is_<name>` below is what the target must match, and for the row whose
# vendor is also its id that is the vendor's own name.
#
# FLAGOS_ACCELERATOR itself is read by the branches further down -- the MetaX
# target name, the per-SDK messages -- so it keeps the name the build was given
# and the row is resolved into FLAGOS_PLATFORM_ID beneath it.
set(FLAGOS_PLATFORM_ID "${FLAGOS_ACCELERATOR}")
if(DEFINED FLAGOS_ACCELERATOR_ALIASES)
  list(FIND FLAGOS_ACCELERATOR_ALIASES "${FLAGOS_ACCELERATOR}" _alias_index)
  if(NOT _alias_index EQUAL -1)
    list(GET FLAGOS_ACCELERATOR_ALIASES -1 FLAGOS_PLATFORM_ID)
  endif()
endif()

# Every accelerator key that owns an aten/backends/<key>/ kernel tree, whatever
# spelling this build was given, as one alternation: the vendor-kernel filter
# below builds a single regex out of it, and a `;`-joined list inside a quoted
# regex matches nothing.

flagos_json_list(FLAGOS_KERNEL_SWITCHES "${_flagos_platforms}" kernel_switches)
flagos_json_list(FLAGOS_PLATFORM_LANGUAGES "${_flagos_platforms}"
    accelerators "${FLAGOS_ACCELERATOR}" project_languages)

string(JSON FLAGOS_PLATFORM_VENDOR GET "${_flagos_platforms}"
       accelerators "${FLAGOS_ACCELERATOR}" vendor)

# The accelerator keys whose aten/backends/<key>/ tree this build keeps: every
# key that has one, so a source filter can drop the other vendors' kernels
# whichever spelling FLAGOS_ACCELERATOR was given. The filter used to compare
# FLAGOS_ACCELERATOR against these names directly, which holds only while a lane
# passes the table id -- and it compared an unquoted string (`STREQUAL
# _vendor_dir`), which is the literal "_vendor_dir" and never a key, so the
# filter excluded nothing and an apple build compiled every vendor's .cc.
set(FLAGOS_ATEN_VENDOR_DIRS ascend;enflame;gcu;iluvatar;kunlunxin;mthreads;apple;tsingmicro;bpu;musa)

# The aten/backends trees this platform's own op wrappers live in. Not always a
# name the build was given: a row whose kernels are another platform's names
# that one in `aten_backends` (MetaX: `metax` and `cuda`, runtime_dir `nvidia`).
# Elsewhere the name itself is the tree when it is one of the keys above, and
# MetaX keeps `metax` as the fallback because it is the row known to borrow --
# so a build named after it cannot silently compile every vendor's tree.
string(JSON _flagos_aten_dirs ERROR_VARIABLE _flagos_aten_dirs_err GET
       "${_flagos_platforms}" accelerators "${FLAGOS_ACCELERATOR}" aten_backends)
if(_flagos_aten_dirs_err)
  set(FLAGOS_PLATFORM_ATEN_DIRS "")
else()
  # string(JSON) returns a JSON array literal; split it into a cmake list,
  # otherwise a single-element ["bpu"] stays one string and no vendor dir
  # matches it, so nothing is excluded.
  string(JSON _flagos_aten_count ERROR_VARIABLE _flagos_aten_n_err LENGTH
         "${_flagos_platforms}" accelerators "${FLAGOS_ACCELERATOR}" aten_backends)
  if(_flagos_aten_n_err)
    set(FLAGOS_PLATFORM_ATEN_DIRS "")
  else()
    set(FLAGOS_PLATFORM_ATEN_DIRS "")
    if(_flagos_aten_count GREATER 0)
      math(EXPR _flagos_aten_last "${_flagos_aten_count} - 1")
      foreach(_i RANGE ${_flagos_aten_last})
        string(JSON _flagos_aten_item GET "${_flagos_platforms}" accelerators
               "${FLAGOS_ACCELERATOR}" aten_backends ${_i})
        list(APPEND FLAGOS_PLATFORM_ATEN_DIRS ${_flagos_aten_item})
      endforeach()
    endif()
  endif()
endif()
if(NOT FLAGOS_PLATFORM_ATEN_DIRS)
  set(FLAGOS_PLATFORM_ATEN_DIRS metax)
  foreach(_name ${FLAGOS_ACCELERATOR};${FLAGOS_PLATFORM_ID};${FLAGOS_PLATFORM_VENDOR})
    foreach(_dir ${FLAGOS_ATEN_VENDOR_DIRS})
      if(_name STREQUAL "${_dir}")
        set(FLAGOS_PLATFORM_ATEN_DIRS "${_name}")
      endif()
    endforeach()
  endforeach()
endif()

string(JSON FLAGOS_PLATFORM_RUNTIME_DIR GET "${_flagos_platforms}"
       accelerators "${FLAGOS_ACCELERATOR}" runtime_dir)
string(JSON FLAGOS_PLATFORM_USE_MACRO GET "${_flagos_platforms}"
       accelerators "${FLAGOS_ACCELERATOR}" use_macro)
string(JSON FLAGOS_PLATFORM_WRITES_MARKER GET "${_flagos_platforms}"
       accelerators "${FLAGOS_ACCELERATOR}" writes_platform_marker)
string(JSON FLAGOS_PLATFORM_C10_CUDA_CONFIGURE GET "${_flagos_platforms}"
       accelerators "${FLAGOS_ACCELERATOR}" c10_cuda_no_cmake_configure)
string(JSON FLAGOS_BUNDLE_LIBDIR GET "${_flagos_platforms}"
       accelerators "${FLAGOS_ACCELERATOR}" bundle_libdir)

# Apply the platform's kernel defaults and pins. Called by the top-level
# CMakeLists after option(): a default is only forced when the switch was not
# passed as -D, and a pin refuses a contradictory -D (setup.py refuses the same
# combination). Both come from the table, so a direct cmake invocation resolves
# the same kernel set setup.py would.
function(flagos_apply_platform_kernel_policy)
  foreach(_switch IN LISTS FLAGOS_KERNEL_SWITCHES)
    string(JSON _default GET "${_flagos_platforms}"
           accelerators "${FLAGOS_ACCELERATOR}" kernel_defaults "${_switch}")
    if(NOT ${_switch}_EXPLICIT)
      set(${_switch} ${_default} CACHE BOOL
          "Build switch for ${FLAGOS_ACCELERATOR}" FORCE)
    endif()
  endforeach()

  string(JSON _pin_count LENGTH "${_flagos_platforms}"
         accelerators "${FLAGOS_ACCELERATOR}" pins)
  if(_pin_count GREATER 0)
    math(EXPR _pin_last "${_pin_count} - 1")
    foreach(_i RANGE ${_pin_last})
      string(JSON _pin_switch MEMBER "${_flagos_platforms}"
             accelerators "${FLAGOS_ACCELERATOR}" pins ${_i})
      string(JSON _pin_value GET "${_flagos_platforms}"
             accelerators "${FLAGOS_ACCELERATOR}" pins "${_pin_switch}" value)
      string(JSON _pin_reason GET "${_flagos_platforms}"
             accelerators "${FLAGOS_ACCELERATOR}" pins "${_pin_switch}" reason)
      flagos_pin_build_switch(${_pin_switch} ${_pin_value}
        "Build switch pinned for ${FLAGOS_ACCELERATOR}" "${_pin_reason}")
    endforeach()
  endif()
endfunction()
