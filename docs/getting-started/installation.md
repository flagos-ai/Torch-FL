# Installation

## Choose a Platform

| Platform | Build selector | Execution path | Installation guide |
|---|---|---|---|
| NVIDIA CUDA | `FLAGOS_ACCELERATOR=cuda` (default) | CUDA boxing over an external `libtorch_cuda.so` | [CUDA Installation](../vendors/cuda/installation.md) |
| MetaX | `FLAGOS_ACCELERATOR=metax` | CUDA boxing via `cu-bridge` against the vendor libtorch | [MetaX Installation](../vendors/metax/installation.md) |
| Ascend | `FLAGOS_ACCELERATOR=ascend` | Native ACLNN operator backend, FlagGems via FlagTree (Triton 3.5) | [Ascend Installation](../vendors/ascend/installation.md) |
| PPU | `FLAGOS_ACCELERATOR=ppu` | Same CUDA-boxing path as NVIDIA CUDA, against the PPU's CUDA-13-compatible SDK, bundling its own libtorch into `lib_ppu/` | [PPU Installation](../vendors/ppu/installation.md) |
| Hygon DCU | `FLAGOS_ACCELERATOR=dcu` | CUDA boxing over the hipified DTK torch build (HIP kernels under the CUDA dispatch key) | [DCU Installation](../vendors/dcu/installation.md) |
| Enflame GCU | `FLAGOS_ACCELERATOR=gcu` | Native `libtopsaten.so` operator backend, with CPU fallback for unrouted/int64 ops | [GCU Installation](../vendors/gcu/installation.md) |
| Moore Threads MUSA | `FLAGOS_ACCELERATOR=musa` | FlagGems-first Triton kernels, native `mudnn` backend as fallback, with CPU fallback for unrouted ops | [MUSA Installation](../vendors/musa/installation.md) |
| D-Robotics BPU | `FLAGOS_ACCELERATOR=bpu` | No eager kernel sets are built; eager ops run on CPU | [BPU Installation](../vendors/bpu/installation.md) |
| TsingMicro | `FLAGOS_ACCELERATOR=tsingmicro` | Runtime/build selector present; no per-op kernel set documented | The runtime build branch exists, but a current end-to-end installation and validation procedure is not documented. |

## Common Requirements

All platforms require:

- **Python**: 3.8 or later
- **PyTorch**: 2.10.x (`>=2.10,<2.11`) — generated ATen bindings are tied to this minor line
- **CMake**: 3.18 or later
- **C++ build toolchain**: A working C++17 compiler (GCC 7+, Clang 5+, or MSVC 2017+)
- **Platform SDK/runtime**: The vendor-specific SDK, compiler, and runtime libraries for your accelerator

Platform-specific requirements (CUDA toolkit version, vendor SDK paths, additional dependencies) are documented in each platform's installation guide.

## Runtime Dependencies and the Package Index

A wheel declares the four things it needs and cannot work without: `torch`, and
the three packages built outside this repository — **FlagTree** (the Triton build
carrying the vendor's backend), **FlagGems** (the operator source) and **FlagCX**
(the distributed backend). They are pinned to the exact versions the wheel was
built against, taken from
[`.github/version-pins.env`](../../.github/version-pins.env), which is the same
file the CI setup scripts read. Nothing about them is a range: `flagtree` and
`flagcx` are not on PyPI at all, `flag_gems` there is an older cohort than the
one the per-op routing tables in `torch_fl/configs/backends_*.conf` were
generated against, and a FlagTree build is per-platform (its package name carries
the vendor's Triton backend, e.g. `0.7.0rc2+hcu3.6` for DCU).

That means the index has to carry more than one location. Which one serves what:

| Requirement | Where it is published |
|---|---|
| `torch_fl` | `flagos-pypi-<vendor>` — the lane named by the wheel's local version (`2.10.0+hygon` → `flagos-pypi-hygon`) |
| `flag_gems`, `flagcx` | the same vendor lane |
| `flagtree` | `flagos-pypi-hosted`, for every platform |
| `torch==2.10.0+cpu` | `https://download.pytorch.org/whl/cpu` |
| everything else (`packaging`, `PyYAML`, `numpy`, …) | PyPI (or a mirror) |

So a single `--index-url` has to name a **group repository** that contains all of
those. Where one is not configured, list them instead — this is the DCU case, and
it resolves (torch_fl 2.10.0+hygon on a cp310 target):

```bash
BASE=https://resource.flagos.net/repository
pip install \
  --index-url       "$BASE/flagos-pypi-hygon/simple/" \
  --extra-index-url "$BASE/flagos-pypi-hosted/simple/" \
  --extra-index-url "$BASE/pypi-proxy/simple/" \
  --extra-index-url "https://download.pytorch.org/whl/cpu" \
  torch_fl==2.10.0+hygon
```

Two things that are easy to get wrong here:

- **The vendor lane alone is not enough**, even for the platforms whose lane
  already carries all three FlagOS packages: `flag_gems` itself declares
  `packaging>=26.0` and `PyYAML==6.0.1`, which the lanes do not serve.
- **`flagtree` is not in most lanes.** It is fetched from `flagos-pypi-hosted`,
  which is how the CI scripts are already split (`FLAGTREE_INDEX_URL` versus
  `FLAGGEMS_INDEX_URL` in `.github/scripts/hooks/set_env_*.sh`).

If your environment provisions the stack itself — as CI does, installing the
wheel with `--no-deps` and each sibling through its own hook — pass `--no-deps`
and keep that arrangement.

## Source Installation Contract

Each platform installation guide defines:

1. The required `FLAGOS_ACCELERATOR` value for that platform
2. SDK/compiler environment variables (e.g., `CUDA_HOME`, `MACA_PATH`, `ASCEND_HOME`)
3. Any platform-specific build flags or dependencies

The general installation pattern is:

```bash
FLAGOS_ACCELERATOR=<platform> pip install --no-build-isolation -e .
```

The `--no-build-isolation` flag is required so that generated native bindings compile against the PyTorch installation visible in your current environment. Without it, pip creates an isolated build environment that may not see your platform SDK or the correct PyTorch installation.

## After Installation

- **First steps**: See the [Quick Start Guide](quickstart.md) for platform-independent usage examples.
- **Platform support**: Review the [Compatibility Matrix](../reference/compatibility.md) to understand what capabilities are validated for your platform.
- **Configuration**: Environment variables controlling runtime behavior are documented in the [Environment Variables Reference](../reference/environment-variables.md).
- **Testing**: Run the test suite following the [Testing Guide](../development/testing.md) to verify your installation.
