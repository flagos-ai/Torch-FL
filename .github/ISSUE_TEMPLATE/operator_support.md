---
name: Operator Support Request
about: Request implementation of a missing PyTorch operator
title: "[OP] torch.<operator_name>"
labels: operator, enhancement
assignees: ''

---

## Operator Information
- **Operator Name**: <!-- e.g., torch.nn.functional.scaled_dot_product_attention -->
- **PyTorch Documentation**: <!-- Link to official PyTorch docs -->
- **Use Case**: <!-- Where is this operator needed? e.g., specific model, workload -->

## Current Status
<!-- What happens when you try to use this operator? -->
- [ ] Operator not registered (crashes with "not implemented for PrivateUse1")
- [ ] Falls back to CPU (slow)
- [ ] Incorrect output (numerical issue)
- [ ] Other: <!-- specify -->

## Minimal Example
```python
import torch
import torch_fl

# Code that triggers the missing operator
x = torch.randn(2, 2, device='flagos')
# torch.some_op(x)  # This fails or falls back to CPU
```

## Target Backend
<!-- Which backend(s) should implement this operator? -->
- [ ] FlagGems (Triton kernel)
- [ ] CUDA (cuBLAS/cuDNN/custom CUDA kernel)
- [ ] MetaX (MACA API)
- [ ] Ascend (ACL NN API)
- [ ] Generic (CPU fallback acceptable)

## Implementation Approach
<!-- If you have ideas on how to implement this -->
- **Underlying Primitive**: <!-- e.g., can be decomposed into matmul + softmax -->
- **Vendor Library**: <!-- e.g., cuBLAS gemm, ACL aclnnMatmul -->
- **Complexity**: <!-- Simple/Medium/Complex -->

## Priority
<!-- How urgent is this? -->
- [ ] Blocker (prevents model from running)
- [ ] High (significant performance impact)
- [ ] Medium (nice to have)
- [ ] Low (rare use case)

## Related Operators
<!-- Are there related operators that should be implemented together? -->

## Checklist
- [ ] I have verified this operator is missing (not just misconfigured)
- [ ] I have provided a minimal reproducer
- [ ] I have checked if FlagGems already has this operator
- [ ] I have specified target backend(s)
