# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

function(flagos_add_nvidia_fa3 torch_fl_target source_root)
  set(provider "${source_root}/src/flash_attn_3_nv")
  set(cutlass "${source_root}/third_party/cutlass")
  set(output "${CMAKE_CURRENT_BINARY_DIR}/torch_fl_fa3")
  set(generator "${CMAKE_SOURCE_DIR}/scripts/codegen/codegen_nvidia_fa3.py")
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${generator}")
  execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" "${generator}"
      --source "${source_root}" --output "${output}"
    COMMAND_ERROR_IS_FATAL ANY
  )
  include("${output}/features.cmake")

  # Keep the Hopper cubins separate from the baseline Ampere/Hopper build flags.
  set(kernels "${provider}/flash_prepare_scheduler.cu")
  foreach(width 64 96 128 192 256 64_256 64_512 192_128)
    foreach(dtype bf16 fp16)
      list(APPEND kernels
        "${provider}/instantiations/flash_fwd_hdim${width}_${dtype}_sm90.cu"
        "${provider}/instantiations/flash_fwd_hdim${width}_${dtype}_packgqa_sm90.cu")
    endforeach()
  endforeach()
  add_library(torch_fl_fa3_objects OBJECT "${output}/flash_api_flagos.cpp" ${kernels})
  set_target_properties(torch_fl_fa3_objects PROPERTIES POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 17 CUDA_STANDARD 17 CUDA_ARCHITECTURES OFF)
  target_include_directories(torch_fl_fa3_objects PRIVATE "${provider}"
    "${cutlass}/include" "${CMAKE_SOURCE_DIR}/csrc/include" ${CUDAToolkit_INCLUDE_DIRS})
  target_link_libraries(torch_fl_fa3_objects PRIVATE torch_cpu_library)
  target_compile_definitions(torch_fl_fa3_objects PRIVATE TORCH_STABLE_ONLY NDEBUG)
  foreach(feature IN LISTS FA3_DISABLED)
    target_compile_definitions(torch_fl_fa3_objects PRIVATE "FLASHATTENTION_DISABLE_${feature}")
  endforeach()
  set(cuda_flags -O3 --use_fast_math --expt-relaxed-constexpr --expt-extended-lambda
    -gencode=arch=compute_90a,code=sm_90a --threads=1
    -DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED -DCUTLASS_ENABLE_GDC_FOR_SM90
    -DCUTLASS_DEBUG_TRACE_LEVEL=0)
  target_compile_options(torch_fl_fa3_objects PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${cuda_flags}>")
  target_sources("${torch_fl_target}" PRIVATE $<TARGET_OBJECTS:torch_fl_fa3_objects>)

  set(package "${CMAKE_INSTALL_LIBDIR}/flash_attn_3_nv")
  install(FILES
    "${output}/flash_attn_3_nv/__init__.py"
    "${output}/flash_attn_3_nv/flash_attn_interface.py"
    "${output}/flash_attn_3_nv/flash_attn_config.py"
    DESTINATION "${package}")
  install(FILES "${source_root}/LICENSE" DESTINATION "${package}" RENAME LICENSE.FA3)
  install(FILES "${cutlass}/LICENSE.txt" DESTINATION "${package}" RENAME LICENSE.CUTLASS)
endfunction()
