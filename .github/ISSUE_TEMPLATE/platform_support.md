---
name: Platform Support Request
about: Request support for a new hardware platform or vendor
title: "[PLATFORM] Support for <Platform Name>"
labels: platform, enhancement
assignees: ''

---

## Platform Information
- **Platform/Vendor Name**: <!-- e.g., New AI Accelerator XYZ -->
- **Architecture**: <!-- e.g., CUDA-compatible, custom ISA -->
- **SDK/Toolkit**: <!-- Name and version of the vendor SDK -->
- **Official PyTorch Support**: <!-- Yes/No, and version if available -->
- **Triton Support**: <!-- Yes/No, custom Triton backend name if available -->

## Integration Approach
<!-- Which integration path is appropriate for this platform? -->
- [ ] **CUDA Boxing** (CUDA-compatible GPU, reuse CUDA kernels via libtorch_cuda.so)
- [ ] **Custom Backend** (vendor-specific SDK, requires new csrc/aten/backends/<vendor>/)
- [ ] **FlagGems Triton** (Triton backend available, use flagos_python path)
- [ ] **Hybrid** (combination of above)

## Technical Details
<!-- Provide information about the platform's capabilities -->

### Runtime API
- **Memory management**: <!-- e.g., cudaMalloc-compatible, custom API -->
- **Stream/Event model**: <!-- e.g., CUDA streams, custom async model -->
- **Kernel execution**: <!-- e.g., CUDA launch API, vendor-specific -->

### Toolchain
- **Compiler**: <!-- e.g., nvcc, vendor compiler -->
- **Standard Library Compatibility**: <!-- C++17, CUDA API compatibility -->
- **RPATH/Linking**: <!-- Special requirements for shared libraries -->

## Available Resources
<!-- What resources can you provide? -->
- [ ] Hardware access for testing
- [ ] SDK documentation
- [ ] Sample code/reference implementation
- [ ] Vendor technical support

## Motivation
<!-- Why should this platform be supported? -->

## Timeline
<!-- When do you need this support? -->

## Checklist
- [ ] I have provided complete platform information
- [ ] I have identified the appropriate integration approach
- [ ] I have checked for existing platform support in the codebase
- [ ] I can provide testing resources or hardware access
