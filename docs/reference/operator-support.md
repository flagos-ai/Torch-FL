# Operator Support

Per-platform operator support counts, read from the shipped routing
configuration of each accelerator: the number of ops routed to FlagGems or to
the platform's own vendor kernels, against the ops the platform leaves on
`cpu_fallback` (`none`).

An op is counted as supported when its final route is anything but `none`; a
`# <backend>` annotation does not change the route, so an op routed to FlagGems
and annotated with the vendor's name counts once, as a FlagGems route. The
denominator is the generated op list, the same for every platform that ships a
full-coverage conf.

| Platform | `FLAGOS_ACCELERATOR` | FlagGems | Vendor native | Supported / total | Supported rate |
|---|---|---:|---:|---:|---:|
| NVIDIA CUDA | `cuda` | 454 | 1583 | 2037 / 2037 | 100.0% |
| PPU | `ppu` | 592 | 1445 | 2037 / 2037 | 100.0% |
| MetaX | `metax` | 604 | 1433 | 2037 / 2037 | 100.0% |
| Hygon DCU | `dcu` | 454 | 1583 | 2037 / 2037 | 100.0% |
| Kunlunxin | `kunlunxin` | 0 | 57 | 57 / 2037 | 2.8% |
| Ascend | `ascend` | 221 | 157 | 378 / 2037 | 18.6% |
| Enflame GCU | `gcu` | 251 | 183 | 434 / 2037 | 21.3% |
| Moore Threads MUSA | `musa` | 466 | 53 | 519 / 2037 | 25.5% |
| Iluvatar | `iluvatar` | 585 | 11 | 596 / 2037 | 29.3% |
| Apple MPS | `mps` | 0 | 63 | 63 / 2037 | 3.1% |
| TsingMicro | `tsingmicro` | 67 | 14 | 81 / 81 | 100.0% |
| D-Robotics BPU | `bpu` | 0 | 0 | 0 | no per-op kernels |

## How the counts are read

For the platforms whose conf is generated full-coverage (`cuda`, `ppu`, `metax`,
`dcu`, `kunlunxin`, `ascend`, `gcu`, `musa`, `iluvatar`, `mps`), the columns are
a tally of `torch_fl/configs/backends_<platform>.conf`:

- **FlagGems** is `flaggems` plus `flaggems_cpp` (`metax` carries the only
  `flaggems_cpp` routes, 10 of its 604).
- **Vendor native** is the platform's own route value — `cuda` for the CUDA-ABI
  platforms (`cuda`, `ppu`, `metax`, `dcu`), and the platform name otherwise
  (`kunlunxin`, `ascend`, `gcu`, `musa`, `iluvatar`, `mps`).
- `ppu` and `dcu` share the CUDA route set with `cuda`, so their vendor column
  equals CUDA's.

`tsingmicro` and `bpu` are the two hand-maintained confs and are not
full-coverage, so their denominator is their own file rather than the generated
op list: `tsingmicro` lists 81 ops (65 `flaggems` + 2 `flagos_python` + 14
`tsingmicro` metadata-view routes); `bpu` builds no kernel set at all and
unlisted ops default to FlagGems, so it has no count to report.

Regenerate the confs and re-read the tally with:

```bash
python3 scripts/codegen/gen_vendor_confs.py            # rewrite the confs
python3 scripts/codegen/gen_vendor_confs.py --stats    # coverage table
```

## What this table does and does not say

A route is a configuration value, not a measurement. These counts say which
backend each op is **sent to**; they do not say that the op executes correctly
on that hardware. Correctness is measured per platform by
`tests/manual/flaggems_overload_survey.py` against the platform's conf, and the
resulting verdicts are recorded in that run's evidence, not here. The
[Compatibility Matrix](compatibility.md) records the platform-level validation
status.