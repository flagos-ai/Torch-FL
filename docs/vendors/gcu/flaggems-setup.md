# Enflame GCU FlagGems Setup Guide

This guide covers running FlagGems on Enflame GCU hardware through `torch_fl`. A
generated vendor conf routes every operator FlagGems can execute to a FlagGems
Triton kernel and everything else to the native `topsaten` kernel, so the setup
below is about getting a working FlagGems stack under `torch_fl`, not about
opting operators in one at a time.

The measured routing table and the per-operator results live in
[flaggems-test-results.md](flaggems-test-results.md).

## Prerequisites

- **TopsRider SDK** discoverable under `TOPS_HOME` (default `/opt/tops`),
  providing `libtopsrt.so` and `libtopsaten.so`
- **CPU PyTorch 2.10.x** (`torch==2.10.0` from the upstream CPU index); GCU does
  not use CUDA boxing, so no CUDA-enabled wheel is needed
- **Python 3.12**: the published FlagTree GCU builds are 3.12 wheels
- **FlagTree** (`0.6.1+enflame3.6`), which installs a Triton 3.6 carrying the
  `enflame` backend
- **FlagGems** from source, pinned to the master revision validated on the S60
  (`3c6f7537d2d5d3aa680c55bbee5c70f2100c5b85`)

FlagTree is the packaging CI uses. The older `triton_gcu` plugin plus its
`/opt/triton_gcu` compiler toolchain is still supported by the compatibility
layer but is no longer what the setup script installs.

## Installation

### 1. Create the environment and install CPU PyTorch

```bash
python3.12 -m venv /opt/torch-fl-gcu
source /opt/torch-fl-gcu/bin/activate
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
```

### 2. Install FlagTree (Triton with the enflame backend)

```bash
python3 -m pip uninstall -y triton   # repeat until fully uninstalled
RES="--index-url=https://resource.flagos.net/repository/flagos-pypi-hosted/simple"
python3 -m pip install "flagtree===0.6.1+enflame3.6" $RES
```

FlagTree installs its own `triton` package rather than registering a `triton`
distribution, so `pip show triton` can still report a leftover NVIDIA wheel
after the install. Run the uninstall until it reports "not installed": an NVIDIA
Triton has no `enflame` backend, `flag_gems`' backend discovery fails against
it, and every FlagGems route in the conf then raises at import. FlagTree may
also pull `torch_gcu` in as a dependency; remove it, because that wheel claims
`PrivateUse1` for itself and cannot coexist with `torch_fl`.

```bash
python3 -m pip uninstall -y torch_gcu triton_gcu
```

### 3. Build and install torch_fl

```bash
git clone https://github.com/flagos-ai/PyTorch-Plugin-FL.git
cd PyTorch-Plugin-FL

FLAGOS_ACCELERATOR=gcu FLAGOS_BUILD_FLAGGEMS=1 pip install --no-build-isolation -v -e .
```

`FLAGOS_BUILD_FLAGGEMS=1` must be exported even though `setup.py` turns it on for
`FLAGOS_ACCELERATOR=gcu`: environment variables are applied *after* the per-accelerator
defaults, so exporting `0` switches it back off and produces a wheel whose conf
routes operators to dispatcher slots that were never compiled in.

### 4. Install FlagGems from source

Install FlagGems **without** its dependencies so FlagTree's Triton survives;
FlagGems' `triton` requirement otherwise pulls the NVIDIA wheel back in.

FlagGems moves faster than the FlagTree Triton it needs, so `set_env_gcu.sh`
pins the revision rather than tracking master. The pin that was validated on the
S60 is `3c6f7537d2d5d3aa680c55bbee5c70f2100c5b85`; see
[flaggems-test-results.md](flaggems-test-results.md) for the measured routing it
produced.

```bash
cd /tmp
git clone https://github.com/FlagOpen/FlagGems.git
cd FlagGems
git checkout 3c6f7537d2d5d3aa680c55bbee5c70f2100c5b85
pip install --no-deps -e .
pip install packaging 'PyYAML==6.0.1' 'sqlalchemy==2.0.48'
```

### 5. Rebuild the in-place extension after source changes

Any change under `csrc/` or `scripts/` requires rebuilding the extension, and a
change to the generator also requires re-running it:

```bash
cd /path/to/PyTorch-Plugin-FL
python3 scripts/codegen/codegen_gcu_flaggems.py
python3 scripts/codegen/gen_vendor_confs.py
FLAGOS_ACCELERATOR=gcu python setup.py build_ext --inplace
```

## Environment variables

```bash
export TOPS_HOME=/opt/tops
export TOPSATEN_LIB=/usr/lib/libtopsaten.so
export FLAGOS_ACCELERATOR=gcu FLAGOS_BUILD_VENDOR=1 FLAGOS_BUILD_FLAGGEMS=1 FLAGOS_BUILD_FLAGGEMS_CPP=0
export LD_LIBRARY_PATH=$TOPS_HOME/lib:$(dirname "$(readlink -f $TOPSATEN_LIB)"):$LD_LIBRARY_PATH
```

| Variable | Required | Purpose |
|---|---|---|
| `TOPS_HOME` | yes | Locates `libtopsrt.so` / `libtopsaten.so` at build and run time |
| `TOPSATEN_LIB` | no | Overrides the `libtopsaten.so` path when it is outside `TOPS_HOME/lib` |
| `FLAGOS_ACCELERATOR=gcu` | build time | Selects the GCU backend when building |
| `LD_LIBRARY_PATH` | yes | Must include `$TOPS_HOME/lib` and the `libtopsaten.so` directory |
| `FLAGOS_BUILD_VENDOR` / `FLAGOS_BUILD_FLAGGEMS` / `FLAGOS_BUILD_FLAGGEMS_CPP` | yes | Select the kernel paths the wheel was built with |
| `GEMS_VENDOR=enflame` | **no** | Set automatically by `torch_fl` at import (`torch_fl/__init__.py`) when a GCU Triton backend is importable and the selected conf routes anything to FlagGems |
| `FLAGOS_BACKEND_CONFIG` | **no** | `torch_fl` selects `configs/backends_gcu.conf` itself; set this only to force a different conf for testing |
| `FLAGOS_LOG=dispatch` | no | Logs `[flagos dispatch] <op> -> <backend>` for every dispatch |
| `FLAGOS_OP_<op>=<backend>` | no | Per-operator override; dots in an op name become double underscores (`FLAGOS_OP_add__Tensor=flaggems`) |

## Verifying the installation

`torch_fl` must be imported before `flag_gems`. FlagGems 5.x resolves its vendor
from the `torch_fl` device surface, so a bare `import flag_gems` against stock
CPU PyTorch raises `No device were detected on your machine !`. That is why
`set_env_gcu.sh` only asserts `importlib.util.find_spec("flag_gems")` and leaves
the real import to the integration job, which runs after the wheel is installed.

```bash
python -c "
import torch, torch_fl
import flag_gems, triton
print('triton  ', triton.__version__, sorted(triton.backends.backends))
print('flag_gems', flag_gems.__version__, flag_gems.vendor_name)
x = torch.randn(64, 64, device='flagos:0')
print('add     ', (x + 1.0).sum().item())
print('abs     ', torch.allclose(x.abs().cpu(), x.cpu().abs()))
"
```

## Routing configuration

`torch_fl/configs/backends_gcu.conf` is generated by `scripts/codegen/gen_vendor_confs.py`
and is FlagGems-first by construction: for every op registered on the `flagos`
device the generator picks the first available backend in the order
`flaggems_cpp > flaggems > tileops > gcu > none`, then re-routes the ops listed
in `NATIVE_TRITON_GAPS["gcu"]` to the native `topsaten` kernel or to `none`.

The `flaggems` routes need a matching entry in
`csrc/aten/backends/gcu/generated/gcu_flaggems_register.inc`, which
`scripts/codegen/codegen_gcu_flaggems.py` writes from the same gap set:
`FLAGGEMS_PYTHON_OPS - NATIVE_TRITON_GAPS["gcu"] - <native GCU ops>`. A `flaggems`
route without a registration raises
*"routed to 'flaggems' but the op is registered on PrivateUse1"* at dispatch, so
the two generators must be run together.

Regenerate and verify:

```bash
python3 scripts/codegen/codegen_gcu_flaggems.py            # rewrite the .inc
python3 scripts/codegen/codegen_gcu_flaggems.py --check    # fail if it is stale
python3 scripts/codegen/gen_vendor_confs.py                # rewrite the conf
python3 scripts/codegen/gen_vendor_confs.py --check        # fail if it is stale
```

An op belongs in `NATIVE_TRITON_GAPS["gcu"]` when FlagGems cannot execute it
correctly on the enflame Triton stack *and* the alternative is a real fallback
(`gcu` where `topsaten` has a kernel, `none` where it does not, which reaches
`cpu_fallback`). FlagGems itself is not patched or forked for GCU. Run
`tests/manual/flaggems_overload_survey.py` on the S60 to decide an entry; do not
infer it from routing configuration alone.

## Running the tests

```bash
pytest tests/integration/ops/ -m "flaggems and main_ops" -v
pytest tests/integration/ops/ -m "(anyplatform or main_ops) and not flaggems_python and not flaggems_cpp" -v
pytest tests/unit/test_gen_vendor_confs.py -v
pytest tests/integration/ops/test_flaggems_conf_consistency.py -v
```

The `flaggems`-marked cases are collected unconditionally; routing comes from `backends_gcu.conf`, so no environment variable is needed. See the note in `.github/configs/gcu.yml`.

## Troubleshooting

### `64-bit data type not supported on GCU300!`

The GCU300 kernel front end rejects any 64-bit type in the kernel IR. The
rejected type is usually a *pointer parameter*, so it is the operand dtype that
has to be avoided, not the operator: a FlagGems kernel is correct for float32,
int32 and bool and fails only for an integral or float64 operand. ATen raises it
as `RuntimeError: Pipeline run failed: PassManager execution failed`, which hides
the actual message; the underlying `loc("<file>":<line>:0): error: 64-bit data
type not supported on GCU300!` is printed by the compiler on stderr.

Routing is per operator, so any operator a real caller hands an int64 tensor has
to move as a whole. That is why the comparison, bitwise, fill, masked and scan
families are all in `NATIVE_TRITON_GAPS["gcu"]`. Reproduce a suspect entry by
forcing it back onto FlagGems:

```bash
FLAGOS_OP_where__self=flaggems pytest tests/integration/ops/ -k where -v
```

### `No device were detected on your machine !`

Import `torch_fl` before `flag_gems`. See "Verifying the installation" above.

### `RuntimeError: Pipeline run failed: PassManager execution failed`

The generic form of the GCU300 64-bit rejection above. Compile with
`TRITON_ALWAYS_COMPILE=1` and read the compiler's stderr, or set
`COMPILE_ARCH=gcu300` to skip the driver's arch lookup, which is occasionally the
part that fails.

### `version 'GLIBCXX_3.4.32' not found`

The FlagTree wheel is built against a newer libstdc++ than a stock Ubuntu 22.04
provides. Preload a newer one, or run on the 24.04 base image the CI uses:

```bash
export LD_PRELOAD=/path/to/conda/lib/libstdc++.so.6
```

### `Receive Sip error message` / `Receive Abort message from KMD: Sip exception`

The vendor Triton driver reported a different current device than `torch_fl`
does, so a kernel launched against device 0 while its operands live on another
device. Since a tops pointer only resolves against the current device, this
usually presents as a hang rather than an error, and kills the process after
enough launches. `_gcu_compat._pin_vendor_driver_to_flagos()` exists to prevent
exactly this; if it regresses, check that the driver still reports
`flagos.current_device()`.

### `active Triton backend does not provide a replay benchmarker`

A `RuntimeWarning` from the enflame Triton backend, which does not implement the
replay benchmarking API. Timing falls back to event measurement; correctness is
unaffected.

## References

- [FlagGems Repository](https://github.com/FlagOpen/FlagGems)
- [FlagTree Repository](https://github.com/flagos-ai/FlagTree)
- [FlagGems Non-NVIDIA Hardware Guide](https://flagos-ai.github.io/FlagGems/usage/non-nvidia/)
