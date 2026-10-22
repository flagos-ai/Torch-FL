# Copyright (c) 2026, BAAI. All rights reserved.
#
# Adopted from https://github.com/pytorch/pytorch/tree/main/test/cpp_extensions/open_registration_extension/torch_openreg
# Below is the original copyright:
# Copyright (c) Meta Platforms, Inc. and affiliates.

# TorchPythonTargets.cmake - Find and configure PyTorch Python libraries

# Find torch libraries
if(NOT TARGET torch_cpu_library)
    add_library(torch_cpu_library INTERFACE IMPORTED)
    set_target_properties(torch_cpu_library PROPERTIES
        INTERFACE_LINK_LIBRARIES "${TORCH_LIBRARIES}"
        INTERFACE_INCLUDE_DIRECTORIES "${TORCH_INCLUDE_DIRS}"
    )
endif()

if(NOT TARGET torch_python_library)
    # Find the torch_python library
    find_library(TORCH_PYTHON_LIBRARY torch_python
        PATHS ${PYTORCH_INSTALL_DIR}/lib
        NO_DEFAULT_PATH
    )

    if(TORCH_PYTHON_LIBRARY)
        add_library(torch_python_library INTERFACE IMPORTED)
        set_target_properties(torch_python_library PROPERTIES
            INTERFACE_LINK_LIBRARIES "${TORCH_PYTHON_LIBRARY};${TORCH_LIBRARIES}"
            INTERFACE_INCLUDE_DIRECTORIES "${TORCH_INCLUDE_DIRS}"
        )
    else()
        message(WARNING "torch_python library not found, using torch_cpu_library instead")
        add_library(torch_python_library ALIAS torch_cpu_library)
    endif()
endif()
