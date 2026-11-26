# Benchmarks

## Qwen3-0.6B training and generation versus vendor Torch

Run `qwen3_vendor_parity.py` in separate vendor Torch and Torch-FL
environments on the same idle accelerator. Use Transformers 5.19.0 on
every accelerator. Use the same local checkpoint and safetensors version
in both environments.
Set `QWEN3_BENCH_HOST` to the same machine and physical device identifier when
the two processes run in containers with different hostnames.
Use an installed FlagGems Native Host wheel (`FLAGOS_BUILD_NATIVE_HOST=1`)
for the Linux candidate. Strict routing rejects unsupported numerical CPU
fallbacks. Every warmup and timed round
must record FlagGems launches from C++ Native Host and zero launches using
Python argument submission. The process-wide Python count also covers
setup and untimed accuracy collection. The report includes enqueue counts
and vendor routes. On Apple, use the Torch-FL Metal wheel and PyTorch's
native MPS reference.

For Native Host acceptance, stage the binaries and launch metadata from the
published compiler's isolated build outputs using
`torch_fl/native_host/device_artifacts.py`. Pass each build report with
`--pointwise-report` to include the generated pointwise modules. Set
`FLAGOS_NATIVE_HOST_DEVICE_BUNDLE` to that directory when building the wheel.
Use `--record-backend-fingerprints` in the pinned compile environment when
the backend checks its compiler version as part of specialization lookup.
The candidate loads the packaged specialization matching the current GPU
and launch signature. A missing specialization fails before runtime code
generation. The wheel compiles pointwise bytecode during the build and
loads it directly during execution. It also packages bytecode for transformed
published host modules. `--diagnose-device-kernels` records
pointwise build outputs and marks the diagnostic run ineligible for acceptance.

```bash
for mode in train infer; do
    "$vendor_python" benchmarks/qwen3_vendor_parity.py \
        --model "$checkpoint" --device "$vendor_device" --mode "$mode" \
        --output "vendor-$mode.json"
    "$flagos_python" benchmarks/qwen3_vendor_parity.py \
        --model "$checkpoint" --device flagos:0 --mode "$mode" \
        --reference "vendor-$mode.json" --output "flagos-$mode.json"
done
```

Use `--vendor-module torch_npu` or the corresponding vendor module when it
registers the reference device. Both environments run the same eager model
calls. Generation uses the usual dynamic KV cache with compilation disabled.
Each build runs the same workload during warmup and timing, without additional
graph capture, model compilation, or caches of model inputs or results.
Defaults are BF16, batch one, 128 input tokens, 64 generated tokens, three
warmups and five timed rounds. Reports keep the warmup times separately and
compute steady-state throughput from the median of the timed rounds.
Training times the forward pass, backward pass and complete AdamW update.
The vendor process records its default AdamW foreach selection, and the
Torch-FL process uses the same selection. Native Host builds with vendor
kernels use batched implementations of individual BF16 foreach operators on
MetaX and Iluvatar. Kunlunxin uses its SDK's tensor-list API. These paths keep
AdamW's eager operation order and the per-operator BF16 rounding points.
For deterministic eager runs, pass `--deterministic` to both processes.
On the measured MTT S5000 vendor build, this setting stabilizes the BF16
language-model-head matrix product in backward. The reports record the setting
and require agreement before comparing the two runs.
Every loss must agree within 2% relative error. The first and last step
sample up to 128 evenly spaced values per parameter in both gradients and
optimizer updates, with a relative L2 limit of 10% per tensor.

Inference times fixed-length greedy generation, including prefill and token
selection. It checks exact generated token IDs and raw logits at every
prefill position and generation step. BF16 uses the repository's
[`bf16-v2` policy](host_compiler/logits_validation.py) and FP32 uses elementwise
absolute and relative tolerances of `1e-3`. Keep the
`*.logits.safetensors` file beside its JSON report for comparison. Both modes
synchronize at the timing boundaries. For the Linux accelerator lanes,
training requires at least `1.00×` the vendor throughput and inference
requires at least `0.95×`. The Apple lane records throughput and requires
numerical agreement.

## BPU Qwen3-0.6B vs the vendor `llm` demo

`bpu_qwen3_bench.py` runs D-Robotics' own two-graph Qwen3-0.6B `.hbm` through
`torch_fl.accelerator.bpu.infer`, so the comparison isolates the runtime: same
artifact, same four cores, same weights, only a different caller.

```bash
python benchmarks/bpu_qwen3_bench.py \
    --hbm ~/llm_sdk/.../Qwen3-0.6B_language_chunk_512_cache_4096_w8_nash-p_corenum_4_4.hbm \
    --tokenizer ~/llm_sdk/.../configs/Qwen3_config
```

**82.1 tok/s decode against the vendor demo's 84.8–87.7**, and 6881 tok/s
prefill against 5626–7014. The vendor number comes from its own demo, which
prints it: `cd oellm_runtime/examples/llm_demo && ./llm -c qwen3_0.6b_config.json`.

Prefill is quoted per 512-token chunk, which is what the graph computes
regardless of prompt length — a 17-token prompt reads as 219 tok/s for the same
78 ms of work. Details and the KV cache layout in
[docs/vendors/bpu/integration.md](../docs/vendors/bpu/integration.md#qwen3-06b-against-the-vendor-llm-demo).

## BPU ResNet-18 vs the vendor artifact

`bpu_resnet18_bench.py` compares three paths for ResNet-18 at 224x224 on an RDK
board: D-Robotics' own `resnet18_224x224_nv12.hbm`, eager CPU, and our
`torch.compile(backend="bpu")`.

```bash
export FLAGOS_BPU_X86_PYTHON=~/hbdk4-x86/python/bin/python3.11
export FLAGOS_BPU_X86_EMULATOR=~/hbdk4-x86/bin/box64
python benchmarks/bpu_resnet18_bench.py
```

The first run compiles under box64 (~25 min); afterwards the `.hbm` is cached in
`~/.cache/torch_fl_bpu` and startup is seconds.

The vendor artifact is the honest upper bound for this board — eager CPU only
proves the offload happened. The two are not the same workload, and the script
says so: the official one takes NV12 (74 KB) where ours takes float32 NCHW
(588 KB), and it was quantized by HMCT with real calibration data. Numbers and
analysis in [docs/vendors/bpu/integration.md](../docs/vendors/bpu/integration.md#measured-performance).

## FlagGems dispatch-overhead microbenchmark

Compares the **host-side dispatch cost** of three paths for the same op on the
same device (`flagos:0`, shared GPU memory):

- **cuda** — the platform conf as shipped (`backends_nvidia.conf`), pure C++
  dispatch into the vendor `libtorch_cuda` kernel.
- **gems_py** — `FLAGOS_FORCE_BACKEND=flaggems`, Python dispatch: `kFlagOsPython`
  crosses into CPython + pybind to launch the FlagGems Triton kernel.
- **gems_cpp** — the same force plus `FLAGOS_OP_<op>=flaggems_cpp` on the 18
  Stage-A ops, C++ dispatch: `kFlagOs` boxes flagos→cuda metadata and calls the
  FlagGems C++ runtime (`liboperators.so`, TritonJIT, **no GIL/pybind**). Only
  those ops route here; the rest fall back to `gems_py`.

FlagGems Python and FlagGems C++ are per-op keys in the one platform conf
(`= flaggems` / `= flaggems_cpp`), not separate conf files; the driver uses
`FLAGOS_FORCE_BACKEND` and `FLAGOS_OP_<op>` so each column differs from the
others in the dispatch path alone.

```bash
python benchmarks/flaggems_dispatch_bench.py            # defaults: submit=300 e2e=100

# gems_cpp column additionally needs a FLAGOS_BUILD_FLAGGEMS_CPP=ON wheel + (one line):
FLAGGEMS_SOURCE_DIR=/path/to/FlagGems-src/src/flag_gems \
LD_LIBRARY_PATH=/path/to/FlagGems-src/cpp/build/lib:$LD_LIBRARY_PATH \
python benchmarks/flaggems_dispatch_bench.py --submit 500 --e2e 200
```

The driver runs `flaggems_dispatch_overhead.py` once per backend in its own
subprocess (routing is fixed at import). If the C++ runtime env is missing it
skips `gems_cpp` instead of failing.

### Metrics

- **submit** — issue N ops back-to-back, synchronize once at the end. The async
  queue hides kernel time, so this isolates host dispatch cost: GIL + pybind +
  a CPython frame (gems_py) vs a thin C++ dispatcher (cuda / gems_cpp).
- **e2e** — synchronize after every op: full per-op wall time including kernel.
- **py_tax** = `gems_py − cuda` — the Python dispatch overhead vs vendor.
- **cpp_save** = `gems_py − gems_cpp` — how much of that tax the C++ dispatch
  actually recovers (only meaningful on ops marked `*`).

Inputs are built on CPU and moved to `flagos:0` to sidestep the RNG shim (a
CPU-torch build cannot create a CUDA generator directly).

### Measured (NVIDIA A100, torch 2.10.0+cpu + flagos PrivateUse1, µs/op, submit=500)

`*` = op pinned to the C++ `kFlagOs` backend with `FLAGOS_OP_<op>=flaggems_cpp`.

| op | cat | cuda | gems_py | gems_cpp | py_tax | cpp_save | gpy_e2e | gcpp_e2e |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| add[512²] | elementwise | 8 | 92 | 95 | 84 | — | 103 | 107 |
| add[4096²] | elementwise | 149 | 150 | 150 | 1 | — | 240 | 244 |
| mul[512²] | elementwise | 8 | 17 | 19 | 9 | — | 25 | 27 |
| abs[512²] | elementwise | 10 | 80 | 83 | 70 | — | 86 | 88 |
| neg[512²] | elementwise | 8 | 78 | 81 | 70 | — | 84 | 87 |
| sum.dim[1024²] `*` | reduction | 10 | 49 | **11** | 38 | **38** | 55 | **21** |
| softmax[1024²] `*` | reduction | 8 | 40 | **11** | 32 | **30** | 49 | **21** |
| mm[512²] `*` | matmul | 27 | 65 | 110 | 38 | **−45** | 123 | 128 |
| mm[4096²] `*` | matmul | 7219 | 8059 | 8176 | 840 | **−117** | 8140 | 8201 |

(unmarked ops are still on the Python path in the cpp column, so their tiny
cpp_save is measurement noise — shown as `—`.)

### Reading the numbers

- **Python dispatch tax is real and roughly fixed: ~70–90 µs/op.** On small
  elementwise ops vendor submits in ~8 µs; the Python path needs ~80–100 µs —
  a ~10× host gap.
- **C++ dispatch delivers big on host-bound reductions.** `sum.dim` and
  `softmax` drop from ~49/40 µs to ~11 µs — `gems_cpp` submit essentially
  matches vendor (10 µs), recovering ~30–38 µs (the whole py_tax). End-to-end
  they more than halve (55→21, 49→21) because the C++ path also skips the
  Python-side launch bookkeeping the per-op sync would otherwise expose.
- **C++ dispatch is a net loss on `mm`** (`cpp_save` **negative**). The C++
  TritonJIT `mm` path here is *slower to submit* than the Python one (110 vs
  65 µs at 512²; 8176 vs 8059 µs at 4096²). The matmul kernel dominates either
  way, so the host difference is noise relative to a 7 ms kernel — but it shows
  the C++ dispatch is not automatically cheaper: for compute-bound ops whose
  kernel already hides the host cost, routing them to `kFlagOs` buys nothing and
  can cost a little (extra boxing + a C++ TritonJIT launch that isn't autotuned
  the way the Python FlagGems path is).
- **e2e reflects kernel quality, not just dispatch.** FlagGems Triton kernels
  are not faster than cuBLAS/vendor for these shapes, so both gems columns trail
  `cuda` end-to-end; C++ dispatch removes the host tax, not a kernel-speed gap.

**Takeaway:** the win is exactly where Stage A aimed once you look per-category —
**host-bound reductions/elementwise ops in eager per-op loops** (sum/softmax:
~3–4× lower submit, ~2.5× lower e2e). For matmul and large reductions the async
pipeline already hides the Python cost, so `kFlagOs` should *not* be enabled for
them — which is why the C++ route is a per-op key in the conf, checked op by op,
rather than a blanket switch. The negative `mm` result is a concrete argument for
keeping the per-op routing table selective (Stage C) rather than
C++-dispatching everything.
