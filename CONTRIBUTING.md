# Contributing to torch_fl

Thank you for your interest in contributing to torch_fl. This guide outlines how to contribute operators, runtime integrations, compiler support, and documentation.

## Ways to Contribute

- **Operators**: Add missing operators or optimize existing implementations for specific backends.
- **Runtime and platform integration**: Port torch_fl to new accelerators or improve existing backend support.
- **Compiler integration**: Extend torch.compile support, improve Triton codegen, or add new compilation backends.
- **Distributed and profiler work**: Enhance collective communication backends (NCCL, FlagCX, HCCL) or profiling integration (CUPTI, vendor profilers).
- **Tests**: Add operator correctness tests, model integration tests, or performance benchmarks.
- **Documentation**: Improve setup guides, troubleshooting docs, or vendor-specific integration notes.

## Before You Start

### Check Support Boundaries

torch_fl targets PyTorch `PrivateUse1` extension points and upstream-compatible operator schemas. Contributions should work within these boundaries:

- Operators must match PyTorch's public API (schema, semantics, tensor layout conventions).
- Platform integrations should use vendor SDKs and official hardware specs, not reverse-engineered APIs.
- Avoid claiming inherited platform support (e.g., "this works on all CUDA-compatible devices") without evidence.

### Discuss Architectural Changes

For broad changes that affect multiple backends or core dispatch logic, open an issue first to discuss:

- New backend dispatch modes or routing strategies
- Changes to the build system affecting all platforms
- Breaking changes to environment variables or configuration files
- New dependencies or external libraries

Bug fixes, individual operator additions, and documentation improvements do not need upfront discussion.

## Development Workflow

### 1. Branch

Work on a feature branch rather than directly on `main`:

```bash
git checkout -b feature/my-contribution
```

### 2. Make Scoped Changes

Keep commits focused. A single PR should address one feature, bug, or documentation improvement. If your work involves multiple independent changes, submit separate PRs.

### 3. Regenerate Bindings When Needed

If you modify operator schemas, add new operators, or change backend registration:

```bash
python scripts/codegen/codegen_ops.py
```

For platform-specific kernels, run the appropriate codegen script (see [Testing Guide](docs/development/testing.md#code-generation)).

### 4. Run Focused Tests

Before pushing, run tests relevant to your changes:

```bash
# Unit tests (always run these)
pytest tests/unit/ -v

# Operator tests (if you modified operator implementations)
pytest tests/integration/ops/test_<operator>.py -v

# Platform-specific tests (if you changed backend routing)
pytest tests/integration/ops/ -m <platform> -v

# Model tests (if you changed runtime behavior)
pytest tests/integration/test_inference.py --model <model-path> -v
```

### 5. Run Lint and Diff Checks

CI requires passing lint checks. Run these commands locally before pushing:

```bash
ruff check .
ruff format --check .
git diff --check
```

Fix issues with:

```bash
ruff format .
ruff check --fix .
```

Install the pinned ruff version CI uses:

```bash
pip install ruff==0.15.12
```

## Adding an Accelerator Backend

New platform integrations require several components. Use existing backends (CUDA, MetaX, Ascend) as reference implementations.

### Required Components

1. **Runtime compatibility layer**: Device registration, stream/event abstractions, and memory management hooks.
2. **Operator routing**: Either CUDA boxing kernels (reuse NVIDIA implementations) or native vendor kernels (ACL, mudnn, topsaten, etc.).
3. **Backend configuration**: A `backends_<platform>.conf` file mapping operators to backend dispatch targets.
4. **Platform detection**: Add hardware detection logic to `torch_fl/__init__.py` and `tests/integration/ops/conftest.py`.
5. **Build integration**: Extend `CMakeLists.txt` to find the vendor SDK and compile platform-specific kernels.
6. **Documentation**: Setup guide, environment variables, and known limitations.

### Documentation Requirements

Platform integrations must include:

- **Compatibility matrix**: Supported hardware models, SDK versions, and PyTorch versions (see [Compatibility](docs/reference/compatibility.md)).
- **Setup guide**: Installation steps, environment configuration, and validation commands (see [vendor guides](docs/vendors/)).
- **Environment variables**: Document platform-specific variables in the vendor guide, not the main environment reference.
- **Testing results**: Evidence that the runtime and operator tests pass on the target hardware.

### Explicit Runtime-only Scope

If the integration provides device/stream/memory abstractions but no operator implementations (CPU fallback for all compute), state this limitation clearly in the PR description and platform documentation.

## Pull Requests

### Title and Description (English Required)

**All PR titles and descriptions must be written in English.** This repository's code, comments, and history are in English, and PRs are reviewed by contributors who do not read other languages.

If you drafted text in another language during development, translate it before opening the PR. A PR opened in the wrong language must be updated before review.

### Include

- **Exact commands**: Show the build commands, test invocations, and their output (success or failure).
- **Hardware and SDK**: Specify the accelerator model, SDK version, PyTorch version, and torch_fl build configuration.
- **Validation evidence**: Paste test results, operator correctness checks, or benchmark numbers.
- **Known limitations**: Document unsupported operators, performance gaps, or environment constraints.

### Exclude

- **Credentials**: No API keys, tokens, registry passwords, or SSH keys.
- **Machine-local paths**: No `/nfs/`, `/home/<user>/`, `/root/`, or organization-specific hostnames (e.g., private registries, internal CI servers).
- **Unrelated changes**: No reformatting unrelated files, dependency bumps outside the scope, or "drive-by" cleanups.

### Example PR Description

```
## Summary

Add operator correctness tests for `torch.nn.functional.gelu` on the MetaX backend.

## Changes

- `tests/integration/ops/test_gelu.py`: Test GELU forward and backward against PyTorch reference
- Mark as `@pytest.mark.metax` for MetaX-specific routing assertions

## Validation

Hardware: MetaX C550  
SDK: MACA 2.8.0  
PyTorch: 2.6.0  
torch_fl build: `ACCELERATOR=metax` (boxing; `VENDOR_KERNEL=OFF` is forced)

Commands:
```bash
pytest tests/integration/ops/test_gelu.py -v
```

Results:
```
test_gelu_forward PASSED
test_gelu_backward PASSED
```

## Known Limitations

None.
```

### Review and Iteration

Reviewers may request changes to code style, test coverage, or documentation clarity. Address feedback by pushing new commits to the same branch (no force-push unless requested).

## License

By contributing to torch_fl, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE). All new files must include the Apache 2.0 license header (see existing files for the template).

