# README redesign — design document

Date: 2026-08-11
Status: Approved for implementation planning

## 1. Goal

Replace the current root `README.md` with a concise project landing page that explains what torch-fl is, why it exists, how it supports multiple accelerator families, what capabilities and versions are validated, and where users should go next.

The root README must not remain an exhaustive build manual. Platform-specific installation, runtime configuration, testing, and troubleshooting belong under `docs/` and must be linked from the README.

## 2. Audience and success criteria

The README serves three audiences, in this order:

1. PyTorch users deciding whether torch-fl supports their accelerator and workload.
2. Framework and platform engineers evaluating torch-fl's architecture and capability boundaries.
3. Contributors adding operators, runtime features, or a new accelerator backend.

A reader should be able to answer the following within the first few sections:

- What is torch-fl?
- Why is it different from a vendor-specific PyTorch fork or a conventional PrivateUse1 plugin?
- Which accelerator platforms are supported, through which execution path, and at what maturity level?
- Which PyTorch and Python versions are supported?
- Does the project cover eager execution, autograd, compilation, distributed execution, and profiling?
- Where are the installation and contribution instructions?

The target size for the root README is approximately 250-350 lines. This is a guideline rather than a hard limit; clarity and accurate support claims take priority.

## 3. Reference projects and adopted conventions

The information architecture combines conventions from several established projects:

- vLLM and SGLang keep their root READMEs focused on positioning, capabilities, a short getting-started path, and links to detailed documentation.
- PyTorch explains the design philosophy before directing readers to installation, resources, releases, and contribution material.
- FlagCX and FlagScale use explicit support tables to make multi-backend scope discoverable.
- FlagGems uses a compact About → Features → Getting Started → Contribution structure that is consistent with the wider FlagOS ecosystem.

The design deliberately does not copy PyTorch's long source-build section or FlagTree's platform installation details into the root README.

## 4. Root README information architecture

The English `README.md` will use the following section order:

```text
# torch-fl
  Badges · Documentation · Chinese README

Project positioning

## Overview

## Design Philosophy

## Architecture

## Capabilities

## Hardware Support

## Compatibility

## Quick Start

## Documentation

## Contributing

## Acknowledgements

## License
```

### 4.1 Header and positioning

The header will identify the project as `torch-fl` and describe it as the PyTorch device plugin for the FlagOS software stack. The opening paragraph must state that torch-fl exposes a single `flagos` device while selecting among reusable native kernels, portable compiler kernels, and vendor-native implementations.

The header may include only useful, maintainable links and badges. It must not add decorative or unverified status badges.

### 4.2 Overview

The overview will describe the ecosystem problem: accelerator vendors expose different runtimes, compiler stacks, and PyTorch integration strategies. torch-fl provides a PyTorch-native device surface and isolates these differences behind a common runtime and operator-routing layer.

This section should describe the project rather than make unsupported performance or completeness claims.

### 4.3 Design Philosophy

The section will explain five principles:

1. **PyTorch-native interface** — users program against standard PyTorch APIs and the `flagos` device.
2. **One logical device** — vendor differences do not create a different user-facing device name for every accelerator.
3. **Layered operator backends** — routing may select different implementations per operator.
4. **Reuse before reimplementation** — reuse established kernels and compiler stacks where their dispatch and ABI boundaries permit it.
5. **Explicit capability boundaries** — unsupported paths and CPU fallbacks are documented instead of being presented as full native coverage.

It will also define the three principal execution paths:

- **Native vendor kernels:** direct calls into vendor runtime and operator libraries, such as ACLNN, mudnn, or topsaten.
- **Compatibility-key boxing:** zero-copy metadata conversion into an independent PyTorch dispatch key when the vendor stack exposes one that can coexist with PrivateUse1.
- **Portable compiler kernels:** FlagGems kernels generated through Triton or a compatible compiler backend.

These paths are implementation strategies, not user-selectable product tiers. A platform may use more than one path.

### 4.4 Architecture

The README will include one compact diagram showing this flow:

```text
PyTorch API
    ↓
flagos device (PrivateUse1)
    ↓
device runtime + per-operator routing
    ↓
FlagGems/compiler kernels | compatibility boxing | native vendor kernels | CPU fallback
    ↓
accelerator runtime
```

The diagram must remain conceptual. Detailed component diagrams, dispatch internals, and profiler/distributed designs belong in `docs/architecture/`.

### 4.5 Capabilities

The README will summarize the project-wide capability surface:

- PyTorch eager tensor operations and device management
- autograd and model training
- `torch.compile`
- distributed collectives and DDP through `ProcessGroupFlagOS`
- `torch.profiler` integration
- FlagGems/Triton operator integration
- explicit CPU fallback for uncovered operations where PyTorch semantics permit it

The section must distinguish project support from platform validation. A capability existing in torch-fl does not imply that every platform implements or validates it. Per-platform capability detail belongs in the compatibility reference.

### 4.6 Hardware Support

The root support table will have five columns:

```text
Platform | Execution path | Validated capabilities | Status | Guide
```

The table will list accelerator platforms supported by the current codebase, using names consistent with the build configuration and runtime detection. Chip families must not be added solely because FlagGems or FlagTree supports them; torch-fl itself must have an integration path in the repository.

Support status has these precise meanings:

- **Stable:** critical paths are continuously tested and the supported version combination is documented.
- **Beta:** the primary path has been validated, but coverage, packaging, or release procedures are not yet stable.
- **Experimental:** validation exists for a specific setup, model, or hardware environment, and interfaces or build procedures may change.
- **Runtime only:** the device runtime is integrated, but the platform is not a general eager operator backend.

Status assignments must be derived from code, tests, and documented live validation. Uncertain claims default to the less mature status.

The root table remains compact. A detailed matrix covering eager execution, training, compile, distributed, profiler, and FlagGems support will live in `docs/reference/compatibility.md`.

### 4.7 Compatibility

The README will state the supported Python and PyTorch range from package metadata, currently Python 3.8 or later and PyTorch 2.10.x. It will explain that generated ATen bindings are pinned to a PyTorch minor line and that vendor SDK or compiler versions vary by platform.

It must not preserve the stale global requirements table that claims Python 3.12, PyTorch 2.11.0, CUDA 12.8, and FlagGems 5.0.2 for all platforms. Platform-specific validated combinations belong in the compatibility reference and installation guides.

### 4.8 Quick Start

The root README will provide one short, platform-independent Python example using `device="flagos"`. It will not include platform build commands, SDK paths, environment-variable tables, import-order troubleshooting, or test invocations.

Before the example, the section will direct users to choose their platform in the installation guide. The example must avoid claiming that every operation always uses FlagGems, because routing depends on the platform and configuration.

### 4.9 Documentation

The documentation section will be a curated navigation list, not a dump of every file. It will link to:

- installation and platform selection,
- compatibility and support matrix,
- architecture documents,
- backend configuration or environment-variable reference,
- development and testing guidance.

Vendor-specific design analyses may be reached through their platform guide rather than all being linked directly from the root README.

### 4.10 Contributing

The section will welcome operator, runtime, compiler, documentation, and accelerator-backend contributions. It will link to a contribution guide if one exists; otherwise implementation must create a concise `CONTRIBUTING.md` or a development guide rather than embedding the complete workflow in the README.

Contribution claims must match the actual lint, code generation, and test commands in the repository.

### 4.11 Acknowledgements and license

Acknowledgements will identify the upstream systems the project materially builds on, including PyTorch, FlagGems, FlagTree/Triton where applicable, and vendor runtime or operator libraries. It must avoid implying endorsement by vendors or projects.

The license section will link to the repository's Apache-2.0 `LICENSE` file.

## 5. Documentation structure and migration

The redesign introduces stable documentation entry points:

```text
docs/
├── getting-started/
│   ├── installation.md
│   └── quickstart.md
├── reference/
│   ├── compatibility.md
│   └── environment-variables.md
└── vendors/
    ├── cuda/
    │   └── installation.md
    ├── metax/
    │   └── installation.md
    ├── ascend/
    │   └── installation.md
    ├── ppu/
    │   └── installation.md
    ├── dcu/
    │   └── installation.md
    ├── gcu/
    │   └── installation.md
    ├── musa/
    │   └── installation.md
    ├── bpu/
    │   └── installation.md
    └── tsingmicro/
        └── installation.md
```

Only platforms supported by the latest code and for which accurate instructions can be recovered from the current README or repository will receive an installation page. Empty placeholder guides are prohibited. If a platform lacks sufficient verified setup information, its support-table guide will point to a scoped existing document or omit the guide link while clearly marking the limitation.

Existing documents will be preserved and linked rather than duplicated:

- `docs/vendors/ascend/aclnn-codegen.md`, `external-libtorch-npu.md`, and `npu-plan.md` remain detailed design and analysis documents linked from the Ascend guide.
- `docs/vendors/bpu/integration.md` remains the detailed BPU architecture and operating guide; the BPU installation entry will link to it and avoid copying its full content.
- `docs/vendors/cuda/external-libtorch-cuda.md` remains the external-libtorch design explanation linked from the CUDA guide.
- `docs/vendors/flaggems/` remains focused on FlagGems routing analyses.
- `docs/architecture/` remains focused on cross-cutting architecture.
- `docs/superpowers/` historical design and implementation material is not part of the end-user navigation hierarchy and will not be reorganized in this change.

The migration will extract and edit current README material rather than blindly copying it. Stale paths, obsolete project trees, conflicting version claims, and machine-specific assumptions must be removed.

## 6. Scope boundaries

This redesign includes:

- rewriting the English `README.md`,
- creating or consolidating English installation and compatibility documentation needed by the new README,
- moving platform-specific build, runtime, testing, and troubleshooting material out of the root README,
- adding navigation links among the new entry points and existing architecture/vendor documents,
- verifying support claims against the current code, tests, package metadata, and existing measured documentation,
- validating Markdown, local links, language policy, and repository formatting checks.

This redesign does not include:

- changing runtime or operator behavior,
- adding support for a new accelerator,
- claiming support inherited only from another FlagOS component,
- reorganizing `docs/superpowers/`,
- rewriting `README_zh.md` in the same change.

`README_zh.md` is currently stale and will no longer be presented as an exact mirror until it receives a separate translation update. The English README may retain a clearly labelled link to it only if that does not imply synchronized content; otherwise the link will be omitted until translation is updated.

## 7. Content integrity and error handling

Documentation must fail conservatively:

- If support status cannot be verified, use the less mature category.
- If a platform guide is incomplete, state the missing validation instead of inventing commands.
- If metadata and the old README disagree, package/build metadata and current tested code take precedence.
- If an existing README instruction conflicts with current source behavior, it must not be migrated without verification.
- Machine-local absolute paths, credentials, and private proxy details must not appear in committed files.
- Everything under `docs/` and all GitHub-facing text must remain in English.

## 8. Verification

Implementation is complete only after all of the following pass:

1. Every relative Markdown link in `README.md` and the new or changed documentation resolves.
2. The support and version tables agree with `pyproject.toml`, `setup.py`, build configuration, and platform routing code.
3. No platform-specific build procedure remains in the root README beyond a link to its guide.
4. No stale root paths such as the removed `include/` or `debug/` layout remain.
5. No CJK characters appear under `docs/` or in the English `README.md`.
6. No machine-local absolute paths or proxy credentials are introduced.
7. `git diff --check` passes.
8. Repository Markdown/lint checks pass where available.
9. The final diff is documentation-only unless a discovered broken documentation check requires a narrowly scoped fix.

## 9. Implementation sequence

The implementation plan should proceed in this order:

1. Inventory and verify platform/version/capability claims from source, tests, and existing documents.
2. Create the installation, compatibility, and reference entry points.
3. Migrate and clean platform-specific content from the current README.
4. Rewrite the root README against the approved information architecture.
5. Validate all claims and links, then run documentation and repository checks.

This order prevents the root README from linking to incomplete destinations and keeps support claims grounded in the detailed guides.
