---
name: Bug Report
about: Report a bug or unexpected behavior
title: "[BUG] "
labels: bug
assignees: ''

---

## Bug Description
<!-- A clear and concise description of what the bug is -->

## Environment
- **Platform**: <!-- CUDA / MetaX / Ascend / PPU -->
- **Python Version**: <!-- e.g., 3.12 -->
- **PyTorch Version**: <!-- e.g., 2.10.0 / 2.11.0 -->
- **torch_fl Version/Branch**: <!-- e.g., 0.1.0, commit hash, or branch name -->
- **CUDA/MACA/CANN Version**: <!-- e.g., CUDA 12.8, MACA 3.8.1, CANN 8.0.RC1 -->
- **FlagGems Version**: <!-- if applicable, e.g., 5.0.2 -->

## Build Configuration
<!-- Include relevant build flags used -->
```bash
# Example:
ACCELERATOR=cuda FLAGGEMS_CPP=1 pip install -e .
```

## Runtime Configuration
<!-- Include relevant environment variables -->
```bash
# Example:
export FLAGOS_BACKEND_CONFIG=torch_fl/backends_cuda.conf
export FLAGGEMS_SOURCE_DIR=/path/to/FlagGems
```

## Steps to Reproduce
<!-- Provide a minimal reproducible example -->
```python
import torch
import torch_fl

# Your code here
```

## Expected Behavior
<!-- What you expected to happen -->

## Actual Behavior
<!-- What actually happened, include error messages/tracebacks -->
```
# Error output here
```

## Additional Context
<!-- Add any other context, logs, or screenshots -->

## Checklist
- [ ] I have searched existing issues to avoid duplicates
- [ ] I have provided complete environment information
- [ ] I have included a minimal reproducible example
- [ ] I have included the full error traceback
