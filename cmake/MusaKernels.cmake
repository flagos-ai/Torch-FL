# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

# Some toolkits ship libmudnn.so without its C++ headers. The effective switch
# controls source selection, registration and the build record. Keep the
# request in the cache so reconfiguring with a complete SDK enables the slot.
if(FLAGOS_BUILD_VENDOR)
  find_library(MUDNN_LIB mudnn PATHS "${MUSA_HOME}/lib" NO_DEFAULT_PATH)
  if(NOT MUDNN_LIB OR NOT EXISTS "${MUSA_HOME}/include/mudnncxx/mudnn.h")
    set(_mudnn_error
        "MUSA toolkit has no mudnn C++ SDK in ${MUSA_HOME} (libmudnn: '${MUDNN_LIB}', header: include/mudnncxx/mudnn.h)")
    if(FLAGOS_MUSA_REQUIRE_VENDOR)
      message(FATAL_ERROR "FLAGOS_MUSA_REQUIRE_VENDOR=ON: ${_mudnn_error}")
    endif()
    message(WARNING "${_mudnn_error}. Disabling MUSA vendor kernels.")
    set(FLAGOS_BUILD_VENDOR OFF)
  endif()
endif()

if(FLAGOS_BUILD_VENDOR)
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
