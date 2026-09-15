# README Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the root README with a concise, evidence-based project landing page and move accurate platform installation, compatibility, runtime configuration, testing, and troubleshooting material into stable English documentation entry points.

**Architecture:** Treat package metadata, build branches, runtime routing, CI manifests, tests, and measured design documents as the source of truth. Build the detailed documentation destinations first, then rewrite `README.md` as a curated summary that links to those destinations; unsupported or weakly validated claims are downgraded or omitted instead of inferred from other FlagOS projects.

**Tech Stack:** GitHub-flavored Markdown, Python 3, shell validation with `rg`, repository metadata in `pyproject.toml` and `setup.py`, CMake build configuration, pytest/CI manifests, and Git.

## Global Constraints

- All GitHub-facing text and every file under `docs/` must be written in English; `README_zh.md` is the only localized top-level exception and is out of scope for this change.
- The root README is a project landing page, not an exhaustive build, runtime, testing, or troubleshooting manual; target approximately 250-350 lines unless accuracy requires otherwise.
- The user-facing device is always `flagos`; execution paths are implementation strategies, not user-selectable product tiers.
- Hardware support claims must come from this repository's current build/runtime integration, tests, CI, or measured documentation, not from support claimed by FlagGems, FlagTree, or another FlagOS component.
- Support statuses use only **Stable**, **Beta**, **Experimental**, and **Runtime only**, with uncertain claims assigned the less mature status.
- Project-wide compatibility is Python 3.8 or later and PyTorch 2.10.x (`torch>=2.10,<2.11`); platform-specific SDK/compiler combinations belong in the detailed compatibility reference.
- Generated ATen bindings are pinned to a PyTorch minor line; do not reproduce the stale global Python 3.12, PyTorch 2.11.0, CUDA 12.8, and FlagGems 5.0.2 table.
- Preserve and link existing architecture/vendor analyses rather than duplicating them; do not reorganize `docs/superpowers/`.
- Create no empty or speculative platform guide. When setup information is insufficient, omit the guide link and state the limitation in the compatibility matrix.
- Do not change runtime or operator behavior, add a platform, rewrite `README_zh.md`, or broaden this work into source-tree reorganization.
- Do not commit machine-local absolute paths, credentials, private proxy details, private registry names, mounted model paths, or host-specific CI paths.
- The final implementation diff must be documentation-only unless a documentation check itself is demonstrably broken and needs a narrowly scoped fix.

---

## File Map

**Create:**

- `docs/getting-started/installation.md` — platform selector, common prerequisites, source-install contract, and links to verified platform guides.
- `docs/getting-started/quickstart.md` — platform-independent first tensor, device management, transfers, and status-query examples.
- `docs/reference/compatibility.md` — source-backed platform execution-path, capability, maturity, version, and limitation matrix.
- `docs/reference/environment-variables.md` — shared build/runtime routing variables, defaults, scope, and links to platform-specific variables.
- `docs/development/testing.md` — contributor test layout, pytest marks, platform test commands, code generation, and lint guidance.
- `docs/vendors/cuda/installation.md` — NVIDIA CUDA build/install/runtime guide and CUDA-specific validation.
- `docs/vendors/metax/installation.md` — MetaX self-contained boxing-wheel workflow, optional FlagGems route, import/runtime rules, and validation.
- `docs/vendors/ascend/installation.md` — CANN/ACLNN installation, optional FlagGems/triton-ascend route, validation, and links to existing Ascend analyses.
- `docs/vendors/ppu/installation.md` — PPU CUDA-compatible build, vendor Triton/FlagGems option, NCCL/FlagCX notes, and validation.
- `docs/vendors/dcu/installation.md` — DTK boxing build, optional Hygon FlagGems route, RCCL/FlagCX notes, and validation.
- `docs/vendors/gcu/installation.md` — TopsRider/topsaten native-kernel build, fallback boundaries, and source-backed validation guidance.
- `docs/vendors/musa/installation.md` — MUSA/mudnn native-kernel build, import-order rule, fallback boundaries, and validation guidance.
- `docs/vendors/bpu/installation.md` — BPU runtime and `torch.compile` entry point linked to the existing detailed integration guide.
- `CONTRIBUTING.md` — concise contribution workflow for runtime, operator, compiler, docs, and backend changes.

**Modify:**

- `README.md` — replace the current manual with the approved landing-page information architecture.
- Existing docs only when a newly created entry point reveals a broken relative link or requires a small reciprocal navigation link; do not rewrite their substantive content.

**Do not create:**

- `docs/vendors/tsingmicro/installation.md` — current source proves a Kuiper runtime build branch exists, but the repository has no complete, current installation/test procedure to migrate.
- Any placeholder guide for a platform listed only by another FlagOS project.

---

### Task 1: Record the Evidence-backed Support Contract

**Files:**
- Create: `docs/reference/compatibility.md`
- Inspect: `pyproject.toml`
- Inspect: `setup.py`
- Inspect: `CMakeLists.txt`
- Inspect: `.github/configs/{cuda,metax,ascend,dcu}.yml`
- Inspect: `torch_fl/comm/process_group.py`
- Inspect: `tests/integration/`
- Inspect: `tests/unit/bpu/`
- Inspect: `docs/architecture/`
- Inspect: `docs/vendors/`

**Interfaces:**
- Consumes: Package/runtime/CI/test evidence already present in the repository.
- Produces: Canonical support statuses, capability labels, version rules, and limitations that every guide and the root README must summarize without strengthening.

- [ ] **Step 1: Capture the authoritative metadata and platform inventory**

Run:

```bash
rg -n 'requires-python|torch>=2\.10,<2\.11|TORCH_PIN|ACCELERATOR.*cuda|ACCELERATOR STREQUAL' \
  pyproject.toml setup.py CMakeLists.txt
rg -n 'display_name:|integration_tests:|Run .*tests|Profiler parity|Qwen3' \
  .github/configs/*.yml
rg -n '^    "(nvidia|metax|thead|hygon|ascend|musa|cambricon)"' \
  torch_fl/comm/process_group.py
```

Expected: the first command proves Python `>=3.8`, PyTorch `>=2.10,<2.11`, and the eight explicit `ACCELERATOR` values; the second identifies continuous validation for CUDA, MetaX, Ascend, and DCU; the third identifies distributed routing boundaries.

- [ ] **Step 2: Create the compatibility reference with explicit semantics**

Write `docs/reference/compatibility.md` with these sections and contracts:

```markdown
# Compatibility and Platform Support

## Status Definitions

| Status | Meaning |
|---|---|
| Stable | Critical paths are continuously tested and the supported version combination is documented. |
| Beta | The primary path is validated, but coverage, packaging, or release procedures are not yet stable. |
| Experimental | Validation exists for a specific setup, model, or hardware environment; interfaces or build procedures may change. |
| Runtime only | Device runtime support exists, but the platform is not a general eager operator backend. |

## Project Compatibility

| Component | Supported range | Notes |
|---|---|---|
| Python | 3.8 or later | From package metadata. Platform SDKs and available wheels may impose a narrower range. |
| PyTorch | 2.10.x (`>=2.10,<2.11`) | Generated ATen bindings are tied to this minor line. |
| FlagGems | Platform dependent | Installed from PyPI or a vendor-compatible build only where the platform route uses it. |
| Triton/compiler | Platform dependent | Use the compiler distribution required by the selected accelerator. |

## Platform Matrix

| Platform | Build selector | Execution path | Eager and autograd | `torch.compile` | Distributed | Profiler | FlagGems | Status |
|---|---|---|---|---|---|---|---|---|
```

Populate all platforms integrated by this repository: NVIDIA CUDA, MetaX, Ascend, PPU, Hygon DCU, Enflame GCU, Moore Threads MUSA, D-Robotics BPU, and TsingMicro. Use the following conservative evidence rules:

- CUDA, MetaX, Ascend, and DCU may cite continuous CI, but individual capabilities must reflect each CI manifest. In particular, do not claim Ascend profiler parity or model-level CI while `.github/configs/ascend.yml` explicitly defers them.
- PPU is `ACCELERATOR=cuda` with `PPU_SDK` detection, not a separate build selector.
- GCU and MUSA use native vendor kernels with documented CPU fallback outside generated coverage; mark capabilities not represented in tests/docs as unvalidated.
- BPU is **Runtime only** for eager execution: the device runtime is native, eager compute falls back to CPU, and acceleration is graph-level through its BPU compile backend or dedicated prebuilt-HBM runtime.
- TsingMicro has a runtime integration/build selector but no recoverable current install/test guide; mark it **Runtime only** and state that setup and operator validation are not currently documented.
- For distributed support, distinguish validated paths from architectural routing. A `_VENDOR_PROFILES` row is not proof that collectives were tested on that platform.
- For profiler support, distinguish project capability from vendor tracer validation.

After the matrix, add `## Platform Notes` with one concise subsection per platform. Each subsection must cite repository-relative evidence links and state material limitations. Add `## Reading the Matrix` clarifying that project-wide capability does not imply platform-wide validation.

- [ ] **Step 3: Verify every matrix claim against its evidence**

Run:

```bash
rg -n 'Stable|Beta|Experimental|Runtime only' docs/reference/compatibility.md
rg -n 'CUDA|MetaX|Ascend|PPU|DCU|GCU|MUSA|BPU|TsingMicro' \
  docs/reference/compatibility.md
rg -n '2\.11|Python 3\.12|CUDA 12\.8.*all|FlagGems 5\.0\.2.*all' \
  docs/reference/compatibility.md
```

Expected: all nine platform names and only the four approved status labels appear; the stale global version claims produce no matches.

- [ ] **Step 4: Check links in the compatibility reference**

Run:

```bash
python - <<'PY'
from pathlib import Path
import re

path = Path('docs/reference/compatibility.md')
for target in re.findall(r'\[[^]]+\]\(([^)]+)\)', path.read_text()):
    if '://' in target or target.startswith('#'):
        continue
    resolved = (path.parent / target.split('#', 1)[0]).resolve()
    assert resolved.exists(), f'{path}: broken link: {target}'
print('compatibility links: OK')
PY
```

Expected: `compatibility links: OK`.

- [ ] **Step 5: Commit the support contract**

```bash
git add docs/reference/compatibility.md
git commit -m "docs: define platform compatibility matrix"
```

---

### Task 2: Create Shared Installation and Quick-start Entry Points

**Files:**
- Create: `docs/getting-started/installation.md`
- Create: `docs/getting-started/quickstart.md`
- Reference: `docs/reference/compatibility.md`
- Reference: `pyproject.toml`
- Reference: `setup.py`

**Interfaces:**
- Consumes: The platform names, status contract, and project compatibility range from Task 1.
- Produces: Stable README destinations for installation selection and platform-independent first use.

- [ ] **Step 1: Create the installation selector**

Write `docs/getting-started/installation.md` with:

1. `# Installation`
2. `## Choose a Platform`, containing a table with `Platform | Build selector | Execution path | Installation guide`.
3. Links for CUDA, MetaX, Ascend, PPU, DCU, GCU, MUSA, and BPU; these links may temporarily point to files created by Tasks 4-6 and will be validated after those tasks.
4. A TsingMicro row with no guide link and the explicit text: “The runtime build branch exists, but a current end-to-end installation and validation procedure is not documented.”
5. `## Common Requirements`, stating Python 3.8+, PyTorch 2.10.x, CMake 3.18+, a C++ build toolchain, and a platform SDK/runtime.
6. `## Source Installation Contract`, explaining that platform guides define the required `ACCELERATOR` and SDK/compiler variables, and that `--no-build-isolation` is used so generated native bindings compile against the selected PyTorch installation.
7. `## After Installation`, linking to `quickstart.md`, the compatibility matrix, environment-variable reference, and testing guide.

Do not include full vendor command sequences here.

- [ ] **Step 2: Create the platform-independent quick start**

Write `docs/getting-started/quickstart.md` with:

```python
import torch
import torch_fl

x = torch.randn(4, 4, device="flagos:0")
y = torch.relu(x @ x)
print(y.cpu())
```

Then add short sections for:

- moving tensors with `.to("flagos")` and `.cpu()`,
- selecting a device with `torch.flagos.device(0)`,
- synchronizing with `torch.flagos.synchronize()`,
- querying `is_available()`, `device_count()`, and `current_device()`,
- a routing note that an operation may use a portable compiler kernel, compatibility boxing, a native vendor kernel, or documented CPU fallback depending on platform and configuration.

Do not say every operation uses FlagGems. Do not include vendor SDK paths or build commands.

- [ ] **Step 3: Verify shared docs contain no platform manual leakage**

Run:

```bash
rg -n 'ACCELERATOR=.*pip install|/opt/|/usr/local/|LD_LIBRARY_PATH|LD_PRELOAD|pytest ' \
  docs/getting-started/*.md
```

Expected: no matches except a generic mention of `ACCELERATOR` without a command or machine path; edit any leaking detail out.

- [ ] **Step 4: Verify language and Markdown shape**

Run:

```bash
python - <<'PY'
from pathlib import Path
import re

for path in Path('docs/getting-started').glob('*.md'):
    text = path.read_text()
    assert text.startswith('# '), path
    assert not any(0x3400 <= ord(char) <= 0x9FFF for char in text), path
print('getting-started docs: OK')
PY
```

Expected: `getting-started docs: OK`.

- [ ] **Step 5: Commit shared onboarding docs**

```bash
git add docs/getting-started/installation.md docs/getting-started/quickstart.md
git commit -m "docs: add shared installation and quickstart guides"
```

---

### Task 3: Extract Environment and Contributor References

**Files:**
- Create: `docs/reference/environment-variables.md`
- Create: `docs/development/testing.md`
- Create: `CONTRIBUTING.md`
- Reference: `setup.py`
- Reference: `CMakeLists.txt`
- Reference: `torch_fl/__init__.py`
- Reference: `torch_fl/compile/README.md`
- Reference: `torch_fl/comm/process_group.py`
- Reference: `tests/integration/conftest.py`
- Reference: `tests/integration/ops/conftest.py`
- Reference: `.github/workflows/lint.yml`
- Reference: `scripts/codegen/codegen_*.py`

**Interfaces:**
- Consumes: Current build/runtime variable definitions and actual test/lint/codegen commands.
- Produces: Stable operational and contribution links for all platform guides and the root README.

- [ ] **Step 1: Inventory variables from source, not only the old README**

Run:

```bash
rg -o 'os\.environ(?:\.get)?\("[A-Z0-9_]+"|os\.getenv\("[A-Z0-9_]+' \
  setup.py torch_fl scripts .github/scripts | sort -u
rg -n '\$ENV\{[A-Z0-9_]+\}|CACHE (STRING|BOOL|PATH)' CMakeLists.txt csrc/CMakeLists.txt
```

Expected: source-defined build and runtime variables are listed. Compare them with `README.md:680-721`; do not migrate old descriptions that source contradicts.

- [ ] **Step 2: Create the environment-variable reference**

Write `docs/reference/environment-variables.md` with these sections:

- `## Build Selection`: `ACCELERATOR`, `FLAGOS_BUILD_JOBS`, and the kernel switches.
- `## SDK and Compiler Discovery`: shared variables with platform and default/precedence where source defines one.
- `## Operator Routing`: `FLAGOS_USE_FLAGGEMS`, `FLAGOS_BACKEND_CONFIG`, `FLAGOS_OP_<name>`, `FLAGOS_LOG_DISPATCH`, `FLAGGEMS_SOURCE_DIR`, and `GEMS_VENDOR`; explain that auto-detection/build metadata normally sets the vendor and users should override only deliberately.
- `## Runtime and Packaging`: asset/bundle switches and import/runtime compatibility controls.
- `## Compiler and Feature Backends`: compile and BPU-specific variables, linking to the relevant detailed guides.
- `## Platform-specific Variables`: links to vendor guides rather than duplicating long setup blocks.

For each variable, use `Variable | Scope | Default or auto-detection | Purpose`. If source does not define a stable default, write “No global default” instead of guessing.

- [ ] **Step 3: Create the development and testing guide**

Write `docs/development/testing.md` with:

- test directory responsibilities: `tests/unit/`, `tests/integration/`, `tests/integration/ops/`, `tests/manual/`, and `tests/perf/`;
- pytest marks read from `tests/integration/ops/conftest.py`, including `main_ops`, `anyplatform`, `cuda`, `ascend`, `flaggems`, `flaggems_python`, and `flaggems_cpp` only if currently registered;
- generic commands for unit tests and platform-filtered operator tests;
- model-test commands that use `<model-path>` rather than a local filesystem path;
- profiler and compile test entry points;
- code generation commands mapped to their source files, with a warning not to hand-edit `csrc/aten/generated/`;
- exact lint commands from CI:

```bash
ruff check .
ruff format --check .
git diff --check
```

State hardware and SDK prerequisites next to commands that need them; do not imply they run in a CPU-only environment.

- [ ] **Step 4: Create a concise contribution guide**

Write `CONTRIBUTING.md` with:

- `## Ways to Contribute`: operators, runtime/platform integration, compiler integration, distributed/profiler work, tests, and documentation;
- `## Before You Start`: check support boundaries, avoid claiming inherited platform support, and discuss broad architectural changes first;
- `## Development Workflow`: branch, make scoped changes, regenerate bindings when needed, run focused tests, run lint/diff checks;
- `## Adding an Accelerator Backend`: link to compatibility, installation, environment, testing, and relevant architecture docs; require a runtime path, operator route or explicit runtime-only scope, platform guide, and validation evidence;
- `## Pull Requests`: English title/body, exact commands/results, hardware/SDK combination, limitations, no credentials or machine-local paths;
- `## License`: contributions are under Apache-2.0.

Keep it concise; detailed commands remain in `docs/development/testing.md`.

- [ ] **Step 5: Verify variable and test documentation against source**

Run:

```bash
rg -n 'FLAGOS_USE_FLAGGEMS|FLAGOS_BACKEND_CONFIG|FLAGOS_OP_<name>|GEMS_VENDOR' \
  docs/reference/environment-variables.md
rg -n 'main_ops|anyplatform|flaggems_python|flaggems_cpp|ruff check|ruff format' \
  docs/development/testing.md
rg -n '/nfs/|/home/[^<]|/root/|harbor\.|MODEL_PATH: /' \
  docs/reference/environment-variables.md docs/development/testing.md CONTRIBUTING.md
```

Expected: required routing/test terms are present; the local/private path scan has no matches.

- [ ] **Step 6: Commit reference and contribution docs**

```bash
git add docs/reference/environment-variables.md docs/development/testing.md CONTRIBUTING.md
git commit -m "docs: add configuration and contribution references"
```

---

### Task 4: Migrate CUDA-compatible Platform Guides

**Files:**
- Create: `docs/vendors/cuda/installation.md`
- Create: `docs/vendors/metax/installation.md`
- Create: `docs/vendors/ppu/installation.md`
- Create: `docs/vendors/dcu/installation.md`
- Reference: `README.md:34-112`
- Reference: `README.md:224-465`
- Reference: `.github/configs/{cuda,metax,dcu}.yml`
- Reference: `.github/scripts/set_env_{cuda,metax,dcu}.sh`
- Reference: `scripts/vendor/with_cuda_libtorch.sh`
- Reference: `scripts/vendor/bundle_maca_libtorch.sh`
- Reference: `scripts/vendor/bundle_ppu_libtorch.sh`
- Reference: `scripts/vendor/bundle_dcu_libtorch.sh`
- Reference: `docs/vendors/cuda/external-libtorch-cuda.md`

**Interfaces:**
- Consumes: Shared compatibility/configuration/testing docs and current boxing build scripts.
- Produces: Detailed installation destinations for the compatibility-key boxing family.

- [ ] **Step 1: Write the NVIDIA CUDA guide**

Create `docs/vendors/cuda/installation.md` with:

- supported route: CPU PyTorch plus bundled/external CUDA libtorch compatibility-key boxing, with optional FlagGems compiler kernels;
- prerequisites and exact source installation flow verified against `.github/scripts/set_env_cuda.sh` and `setup.py`;
- a basic verification command using `device="flagos:0"`;
- operator, FlagGems, Qwen3, compile, distributed, and profiler validation commands only where current tests/CI support them;
- troubleshooting limited to verified import/asset and version-pin failures;
- `## Further Reading` linking to `external-libtorch-cuda.md`, compile, distributed, profiler, environment, and testing docs.

Use placeholders such as `<path-to-FlagGems>` and `<model-path>` rather than machine-local examples.

- [ ] **Step 2: Write the MetaX guide**

Create `docs/vendors/metax/installation.md` by editing, not blindly copying, `README.md:44-112` and `README.md:756-789`. Include:

- build-host versus target-host requirements;
- self-contained boxing wheel steps using repository scripts;
- import-order and MACA runtime rules;
- pure boxing versus optional `triton-metax`/FlagGems routing;
- representative operator/factory validation matching `.github/configs/metax.yml`;
- clearly label model training/inference and distributed claims as manual/experimental unless backed by current CI;
- replace `/opt/maca` only where it is the documented SDK default; replace user/repository/model paths with angle-bracket placeholders.

- [ ] **Step 3: Write the PPU guide**

Create `docs/vendors/ppu/installation.md` from the verified PPU material in `README.md:224-337`. Preserve these distinctions:

- `PPU_SDK` triggers a CUDA-compatible route; the build selector remains `ACCELERATOR=cuda`;
- vendor PyTorch supplies the CUDA dispatch-key implementation, so bundled CUDA assets are disabled;
- vendor Triton versioning may require manual dependency installation;
- NCCL is the default distributed route and FlagCX is optional;
- status remains Experimental because validation is setup-specific and no PPU CI manifest exists.

Remove temporary troubleshooting detail that cannot be reproduced from repository scripts. Link to shared distributed/testing references rather than repeating complete manuals.

- [ ] **Step 4: Write the DCU guide**

Create `docs/vendors/dcu/installation.md` from `README.md:339-465`, `setup.py`, the DCU bundle script, and `.github/configs/dcu.yml`. Include:

- DTK/hipified-PyTorch compatibility-key boxing architecture;
- `ACCELERATOR=dcu` installation and `DTK_ROOT`/`ROCM_PATH` precedence;
- optional DTK-provided Triton/FlagGems route;
- RCCL exposed through the NCCL API and optional FlagCX;
- CI-backed operator, factory, FlagGems, and profiler commands;
- current limits such as `record_stream` only if still true in source;
- no fixed machine path except documented vendor defaults.

- [ ] **Step 5: Verify all compatibility-boxing guides share the right boundaries**

Run:

```bash
for f in docs/vendors/{cuda,metax,ppu,dcu}/installation.md; do
  printf '%s: ' "$f"
  rg -q '^## (Prerequisites|Requirements)' "$f" && \
  rg -q '^## (Install|Build)' "$f" && \
  rg -q '^## Verif' "$f" && echo OK
done
rg -n '/nfs/|/home/[^<]|/root/|harbor\.|/models/' \
  docs/vendors/{cuda,metax,ppu,dcu}/installation.md
```

Expected: four `OK` lines; no local/private path matches.

- [ ] **Step 6: Check vendor-guide links**

Run:

```bash
python - <<'PY'
from pathlib import Path
import re

for path in Path('docs/vendors').glob('*/installation.md'):
    for target in re.findall(r'\[[^]]+\]\(([^)]+)\)', path.read_text()):
        if '://' in target or target.startswith('#'):
            continue
        resolved = (path.parent / target.split('#', 1)[0]).resolve()
        assert resolved.exists(), f'{path}: broken link: {target}'
print('boxing guide links: OK')
PY
```

Expected: `boxing guide links: OK`.

- [ ] **Step 7: Commit compatibility-boxing guides**

```bash
git add docs/vendors/cuda/installation.md docs/vendors/metax/installation.md \
  docs/vendors/ppu/installation.md docs/vendors/dcu/installation.md
git commit -m "docs: migrate compatibility platform installation guides"
```

---

### Task 5: Migrate Native-kernel Platform Guides

**Files:**
- Create: `docs/vendors/ascend/installation.md`
- Create: `docs/vendors/gcu/installation.md`
- Create: `docs/vendors/musa/installation.md`
- Reference: `README.md:113-223`
- Reference: `README.md:467-611`
- Reference: `.github/configs/ascend.yml`
- Reference: `.github/scripts/set_env_ascend.sh`
- Reference: `scripts/codegen/codegen_ascend.py`
- Reference: `scripts/codegen/codegen_gcu.py`
- Reference: `scripts/codegen/codegen_mudnn.py`
- Reference: `docs/vendors/ascend/{aclnn-codegen,external-libtorch-npu,npu-plan}.md`

**Interfaces:**
- Consumes: Shared compatibility/configuration/testing docs and native runtime/operator codegen sources.
- Produces: Detailed installation destinations for platforms that call vendor runtime/operator libraries directly.

- [ ] **Step 1: Write the Ascend guide**

Create `docs/vendors/ascend/installation.md` with:

- CPU PyTorch plus CANN prerequisites;
- `ACCELERATOR=ascend`/ACLNN native-kernel build as the default path;
- optional FlagGems plus triton-ascend path, including why torch_npu cannot be loaded as a compatibility-key fallback;
- import order, patch script, verification, operator/factory/RNG test commands matching current CI;
- explicit limitations: no model mount in current CI, no device-side profiler parity yet, and distributed capability depends on a viable FlagCX/HCCL integration rather than being implied globally;
- links to all three existing Ascend analyses.

Do not carry stale private fork URLs forward unless the source/install tooling still requires and documents them; prefer current upstream project links.

- [ ] **Step 2: Write the GCU guide**

Create `docs/vendors/gcu/installation.md` with:

- CPU PyTorch plus TopsRider/topsaten prerequisites;
- `ACCELERATOR=gcu` source installation;
- native `topsrt` runtime and generated topsaten operator path;
- explicit CPU fallback for missing kernels and int64 limitations;
- optional FlagGems/Triton-GCU only as an experimental route if current source still enables it;
- verification based on generic factory/operator smoke tests, clearly labelled as manual because there is no GCU CI config;
- codegen contributor link to `scripts/codegen/codegen_gcu.py` and shared testing docs.

- [ ] **Step 3: Write the MUSA guide**

Create `docs/vendors/musa/installation.md` with:

- CPU PyTorch plus MUSA toolkit/mudnn prerequisites;
- `ACCELERATOR=musa` source installation and required no-build-isolation behavior;
- native musart runtime and generated mudnn operators;
- `torch_fl`-before-`torch` rule when torch_musa is installed and the `TORCH_DEVICE_BACKEND_AUTOLOAD` interaction;
- generated-op and convolution/fallback boundaries;
- manual validation using `tests/integration/ops/test_musa_dispatch.py` plus focused common operator tests;
- distributed support described as FlagCX-only architecture unless live evidence proves more;
- no claim of continuous validation because no MUSA CI config exists.

- [ ] **Step 4: Compare all native guides against build forcing rules**

Run:

```bash
rg -n 'ACCELERATOR == "(ascend|gcu|musa)"|-(DASCEND_KERNEL|DGCU_KERNEL|DMUSA_KERNEL)' setup.py
rg -n 'ACCELERATOR=(ascend|gcu|musa)|ACLNN|topsaten|mudnn|CPU fallback' \
  docs/vendors/{ascend,gcu,musa}/installation.md
```

Expected: every guide's selected native kernel route agrees with `setup.py`; no guide describes CUDA boxing for these platforms.

- [ ] **Step 5: Verify no unsupported profiler/distributed claims slipped in**

Run:

```bash
rg -n 'full profiler parity|continuously tested|Stable|NCCL fallback' \
  docs/vendors/{ascend,gcu,musa}/installation.md
```

Expected: no matches unless immediately negated as an explicit limitation; revise ambiguous claims.

- [ ] **Step 6: Commit native-kernel guides**

```bash
git add docs/vendors/ascend/installation.md docs/vendors/gcu/installation.md \
  docs/vendors/musa/installation.md
git commit -m "docs: migrate native platform installation guides"
```

---

### Task 6: Create the BPU Runtime-only Entry Point

**Files:**
- Create: `docs/vendors/bpu/installation.md`
- Preserve: `docs/vendors/bpu/integration.md`
- Reference: `README.md:612-678`
- Reference: `setup.py:398-413`
- Reference: `torch_fl/accelerator/bpu/`
- Reference: `tests/unit/bpu/`
- Reference: `benchmarks/README.md`

**Interfaces:**
- Consumes: The **Runtime only** definition and BPU capability boundary from Task 1.
- Produces: A concise BPU installation entry point without duplicating its extensive architecture and benchmark analysis.

- [ ] **Step 1: Write the BPU entry guide**

Create `docs/vendors/bpu/installation.md` with:

- supported target and runtime prerequisites;
- `ACCELERATOR=bpu` installation command;
- explicit behavior boundary: native device memory/runtime, CPU fallback for eager operators, graph acceleration through `torch.compile(backend="bpu")`, and a separate prebuilt-HBM LLM runtime;
- hbdk4/box64 setup via `scripts/vendor/setup_bpu_hbdk4.sh` only where compilation is needed;
- short verification for import/device runtime and links to focused unit tests;
- links to `integration.md` for partitioning, quantization, cache, zero-copy, fixed-shape limits, benchmark results, and LLM runtime details;
- no duplicated performance numbers in the installation guide unless they are necessary to explain validation status.

- [ ] **Step 2: Verify the guide does not imply eager acceleration**

Run:

```bash
rg -n 'eager|CPU fallback|torch\.compile|Runtime only|\.hbm' \
  docs/vendors/bpu/installation.md
```

Expected: all five concepts are present and explicitly distinguish the execution modes.

- [ ] **Step 3: Verify the existing detailed guide remains the canonical analysis**

Run:

```bash
rg -n 'integration\.md' docs/vendors/bpu/installation.md
rg -n '1\.356 ms|82\.1 tok/s|19\.2x' docs/vendors/bpu/installation.md
```

Expected: the first command finds the link; the second produces no output, avoiding duplication of measured detail.

- [ ] **Step 4: Commit the BPU entry point**

```bash
git add docs/vendors/bpu/installation.md
git commit -m "docs: add BPU runtime installation entry point"
```

---

### Task 7: Rewrite the Root README as the Project Landing Page

**Files:**
- Modify: `README.md`
- Reference: `docs/superpowers/specs/2026-08-11-readme-redesign-design.md`
- Reference: all documentation created in Tasks 1-6
- Reference: `LICENSE`

**Interfaces:**
- Consumes: Canonical compatibility/status claims and complete documentation destinations.
- Produces: The public project landing page; it must summarize, never strengthen, detailed documentation.

- [ ] **Step 1: Replace the root README using the approved section order**

Rewrite `README.md` with exactly this top-level information architecture:

```markdown
# torch-fl

[useful badges and documentation links only]

[one-paragraph project positioning]

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

Header rules:

- Use the display name `torch-fl`; code/package imports remain `torch_fl`.
- Describe it as the PyTorch device plugin for the FlagOS software stack.
- State that one `flagos` device routes operators among reusable native kernels, portable compiler kernels, vendor-native implementations, and explicit CPU fallback.
- Include only verifiable badges, such as license and supported Python/PyTorch metadata. Do not add coverage, download, release, or build-status badges without a stable public target.
- Omit the stale `README_zh.md` link unless labelled “Chinese (translation may lag behind this README)”; prefer omission to implying synchronization.

- [ ] **Step 2: Explain the design without overclaiming**

In `## Design Philosophy`, cover these five principles:

1. PyTorch-native interface.
2. One logical device.
3. Layered per-operator backends.
4. Reuse before reimplementation.
5. Explicit capability boundaries.

Define these three implementation paths:

- native vendor kernels;
- compatibility-key boxing;
- portable compiler kernels.

Clarify that a platform may combine paths and users do not choose product tiers.

- [ ] **Step 3: Add the compact conceptual architecture diagram**

Use a plain text diagram, not a detailed component inventory:

```text
PyTorch API
    |
flagos device (PrivateUse1)
    |
device runtime + per-operator routing
    |
FlagGems/compiler kernels | compatibility boxing | native vendor kernels | CPU fallback
    |
accelerator runtime
```

Link architecture detail to `docs/architecture/` documents instead of reproducing dispatch, distributed, compile, or profiler internals.

- [ ] **Step 4: Summarize capabilities and platform support conservatively**

The capabilities section must list project-level support for:

- eager tensor operations and device management;
- autograd and training;
- `torch.compile`;
- distributed collectives and DDP through `ProcessGroupFlagOS`;
- `torch.profiler` integration;
- FlagGems/Triton integration;
- explicit CPU fallback.

Immediately state that availability and validation vary by platform.

Create the compact support table:

```markdown
| Platform | Execution path | Validated capabilities | Status | Guide |
|---|---|---|---|---|
```

Copy status and capability wording from `docs/reference/compatibility.md`. Use no guide link for TsingMicro. Keep BPU labelled **Runtime only** and avoid presenting its eager CPU fallback as accelerator operator coverage.

- [ ] **Step 5: Add compatibility, quick start, docs, contribution, acknowledgements, and license**

- Compatibility: state Python 3.8+ and PyTorch 2.10.x, explain generated ATen minor-line pinning, and link the detailed matrix.
- Quick Start: direct readers to the installation selector, then show only a short `flagos:0` example whose comment does not prescribe the backend route.
- Documentation: curate links to installation, quickstart, compatibility, architecture, environment variables, testing, and contribution docs.
- Contributing: summarize accepted contribution areas and link `CONTRIBUTING.md`.
- Acknowledgements: name PyTorch, FlagGems, FlagTree/Triton where applicable, FlagCX, and vendor runtime/operator libraries without implying endorsement.
- License: link `LICENSE` and state Apache-2.0.

- [ ] **Step 6: Enforce README scope and size**

Run:

```bash
wc -l README.md
rg -n '^### Build|ACCELERATOR=.*pip install|LD_LIBRARY_PATH|LD_PRELOAD|pytest |/opt/|/usr/local/' README.md
rg -n 'Python 3\.12|PyTorch 2\.11\.0|CUDA 12\.8|All operations automatically use FlagGems' README.md
rg -n 'include/|debug/' README.md
```

Expected: approximately 250-350 lines; no platform build/runtime/test commands, stale global claims, or removed root paths. A platform name in the support table is fine; procedural details are not.

- [ ] **Step 7: Check README links**

Run:

```bash
python - <<'PY'
from pathlib import Path
import re

path = Path('README.md')
for target in re.findall(r'\[[^]]+\]\(([^)]+)\)', path.read_text()):
    if '://' in target or target.startswith('#'):
        continue
    resolved = (path.parent / target.split('#', 1)[0]).resolve()
    assert resolved.exists(), f'{path}: broken link: {target}'
print('README links: OK')
PY
```

Expected: `README links: OK`.

- [ ] **Step 8: Commit the root README rewrite**

```bash
git add README.md
git commit -m "docs: redesign project README"
```

---

### Task 8: Validate Navigation, Language, Claims, and Diff Scope

**Files:**
- Modify: only documentation files with defects found by these checks
- Validate: `README.md`
- Validate: `CONTRIBUTING.md`
- Validate: `docs/getting-started/`
- Validate: `docs/reference/`
- Validate: `docs/development/`
- Validate: `docs/vendors/*/installation.md`

**Interfaces:**
- Consumes: The complete documentation implementation.
- Produces: A review-ready documentation-only diff satisfying every spec acceptance criterion.

- [ ] **Step 1: Validate every relative Markdown link in changed public docs**

Run:

```bash
python - <<'PY'
from pathlib import Path
import re

paths = [Path('README.md'), Path('CONTRIBUTING.md')]
paths += list(Path('docs/getting-started').glob('*.md'))
paths += list(Path('docs/reference').glob('*.md'))
paths += list(Path('docs/development').glob('*.md'))
paths += list(Path('docs/vendors').glob('*/installation.md'))

broken = []
for path in paths:
    text = path.read_text()
    for target in re.findall(r'\[[^]]+\]\(([^)]+)\)', text):
        if '://' in target or target.startswith('#') or target.startswith('mailto:'):
            continue
        file_target = target.split('#', 1)[0]
        if file_target and not (path.parent / file_target).resolve().exists():
            broken.append(f'{path}: {target}')
assert not broken, 'Broken links:\n' + '\n'.join(broken)
print(f'checked {len(paths)} Markdown files: all local links resolve')
PY
```

Expected: all local links resolve.

- [ ] **Step 2: Validate English-only publication policy**

Run:

```bash
python - <<'PY'
from pathlib import Path
import re

paths = [Path('README.md'), Path('CONTRIBUTING.md')]
paths += list(Path('docs').rglob('*.md'))
bad = [str(p) for p in paths if any(0x3400 <= ord(char) <= 0x9FFF for char in p.read_text())]
assert not bad, 'CJK found in English docs:\n' + '\n'.join(bad)
print('English-language scan: OK')
PY
```

Expected: `English-language scan: OK`. Historical `docs/superpowers/` files are included because the project policy covers all `docs/`; do not weaken the scan.

- [ ] **Step 3: Scan for secrets, private infrastructure, and machine-local paths**

Run:

```bash
rg -n '/nfs/|/home/[A-Za-z0-9_-]+/|/root/|harbor\.|https?://[^ ]+:[^ @]+@|proxy(_url)?=' \
  README.md CONTRIBUTING.md docs --glob '*.md'
```

Expected: no matches. Documented vendor defaults such as `/opt/dtk`, `/opt/maca`, `/opt/tops`, `/usr/local/musa`, and `/usr/local/Ascend` are allowed only inside the relevant vendor guide and must not contain a user or private-host component.

- [ ] **Step 4: Cross-check package compatibility and platform names**

Run:

```bash
python - <<'PY'
from pathlib import Path

readme = Path('README.md').read_text()
compat = Path('docs/reference/compatibility.md').read_text()
assert 'Python 3.8' in readme and 'Python 3.8' in compat
assert '2.10' in readme and '>=2.10,<2.11' in compat
for platform in ['CUDA', 'MetaX', 'Ascend', 'PPU', 'DCU', 'GCU', 'MUSA', 'BPU', 'TsingMicro']:
    assert platform in readme, f'README missing {platform}'
    assert platform in compat, f'compatibility matrix missing {platform}'
assert 'Runtime only' in compat
print('compatibility summaries agree: OK')
PY
```

Expected: `compatibility summaries agree: OK`.

- [ ] **Step 5: Recheck root README boundary and stale paths**

Run:

```bash
rg -n '^### Build from Source|ACCELERATOR=.*pip install|source /|export (LD_|PATH=)|pytest ' README.md
rg -n '├── include/|├── debug/|path_to_repos|/path/to/' README.md
```

Expected: no matches.

- [ ] **Step 6: Run repository formatting checks**

Run:

```bash
git diff --check
ruff check .
ruff format --check .
```

Expected: all commands exit 0. Markdown-only changes should not alter Ruff results; if Ruff fails on a pre-existing source issue, capture the exact failure and do not edit unrelated source merely to make this documentation change green.

- [ ] **Step 7: Confirm the final diff is documentation-only**

Run:

```bash
git diff --name-only flagos/main...HEAD
```

Expected: only `README.md`, `CONTRIBUTING.md`, and files under `docs/` are listed, including the approved spec and this plan. No source, test, build, generated, or workflow file appears.

- [ ] **Step 8: Review the final support-language diff**

Run:

```bash
git diff --stat flagos/main...HEAD
git diff --word-diff=plain flagos/main...HEAD -- README.md docs/reference/compatibility.md
```

Expected: the reviewer can trace every compact README claim to equal or more conservative wording in the compatibility reference. Correct any mismatch before committing.

- [ ] **Step 9: Commit validation fixes if needed**

If validation required documentation edits:

```bash
git add README.md CONTRIBUTING.md docs
git commit -m "docs: fix README navigation and validation issues"
```

If every check passed without edits, do not create an empty commit.

- [ ] **Step 10: Record final verification for handoff**

Run:

```bash
git status --short
git log --oneline flagos/main..HEAD
```

Expected: clean working tree and a reviewable sequence of documentation commits: design spec, compatibility contract, shared onboarding/reference docs, platform guides, README rewrite, and optional validation fixes.
