# Unified RNG for torch_fl (flagos backend) — Design

Date: 2026-07-30
Branch base: `flagos/main` (tip `3fa2057`)
Status: design, pending implementation-plan

## Problem

The flagos (PrivateUse1) backend has **two disjoint RNG worlds**, and
`torch.manual_seed` reaches only one of them:

- **FlagGems (philox) world** — ops routed to `flagos_python`
  (`rand`/`randn`/`uniform_`/`exponential_`/`multinomial`/`native_dropout`/
  `bernoulli_.float`/`rand_like`/`randn_like`). These read seed+offset from
  `torch.cuda.default_generators[device]`, a per-device `torch.Generator(
  device="cuda")` installed by the vendor compat shim
  (`torch_fl/accelerator/cuda/_cuda_compat.py`). `torch.manual_seed ->
  torch.cuda.manual_seed_all` reseeds these, so they ARE reproducible.

- **Native CUDA world** — ops routed to `= cuda` in the backends conf
  (`normal_`/`randint`/`bernoulli.Tensor`/`random_`/`log_normal_`/`cauchy_`/
  `uniform`(functional)/`poisson`/... and `randperm` transitively). Their
  boxing kernels call `at::<op>(..., generator)` with `generator == nullopt`,
  so ATen falls back to **its own default CUDA generator**
  (`getDefaultCUDAGenerator()`). That generator is unreachable from Python on
  the CPU-torch wheel — `torch._C` has no `_cuda_manualSeed` /
  `_cuda_getRNGState` bindings (verified: `hasattr(...) == False`). So
  `torch.manual_seed` cannot seed it, and these ops are NOT reproducible.

Goal: **a single generator source.** All RNG — FlagGems, native CUDA, and any
C++ path — draws randomness from ONE generator, so `torch.manual_seed` makes
everything reproducible and the two-worlds split is eliminated.

## Why "seed the ATen default generator" is impossible here

The chosen target ("make ATen's default CUDA generator the single source") is
NOT reachable from Python on CPU-torch: the `_cuda_manualSeed`-family C++→Python
bindings are compiled only into CUDA-build torch, and we run the CPU wheel +
external `libtorch_cuda.so`. Probed 2026-07-30: those attributes are absent, and
seeding our shim generator does NOT affect native `normal_` (they are different
objects).

## Chosen approach — B: C++-level explicit generator injection

**Verified anchor (2026-07-30):** the flagos→CUDA boxing path *honors an
explicitly-passed generator*. `torch.empty(n, device="flagos").normal_(
generator=g)` is fully reproducible under `g.manual_seed(s)`, and `g` can be the
very object FlagGems reads (`torch.cuda.default_generators[0]`). So if every
native RNG kernel injects that shared generator when the caller passed none,
both worlds draw from one generator.

### Mechanism

In each generated native-CUDA RNG kernel body, before the `at::<op>(...)` call:

```cpp
if (!generator.has_value()) {
    generator = GetFlagosDefaultCudaGenerator(<device_index>);
}
```

`GetFlagosDefaultCudaGenerator(int64_t device_index)` is a new C++ helper
(added alongside the existing `python_op_caller`) that, under
`py::gil_scoped_acquire`, fetches `torch.cuda.default_generators[device_index]`
(the shim's per-device CUDA generator — the SAME object FlagGems reads) and
converts it to `at::Generator`. Using the Python shim object (not the C++
`csrc/runtime/generator.cc` World-A `GeneratorImpl`, which is dead code for RNG)
guarantees FlagGems and native share one generator.

When the caller *did* pass a generator, `has_value()` is true and injection is
skipped — fully backward compatible.

### Deriving `device_index` inside each body

Two body shapes, both able to determine the index in-body:

- **A. Tensor-input kernels** (inplace / functional / out / tuple_return): have
  a `DeviceBoxingGuard(self, ...)`. Use the boxed input tensor's device index.
- **B. Factory RNG kernels** (`rand`/`randint`/`randperm`/`randn` `.generator`
  overloads): no input tensor, but already compute `_cuda_dev`. Use
  `_cuda_dev.index()`.

### randperm falls out for free

`randperm` routes to `flagos_python` but its dominant randomness is an internal
`torch.randint(low=, high=)` that hits `RandintLowGeneratorKernelCuda` (native).
Injecting there makes randperm reproducible with no randperm-specific code.

## Scope — exhaustive

ALL 80 native-CUDA kernels carrying `std::optional<at::Generator> generator`
(enumerated from `csrc/aten/generated/cuda_kernels.cc`). Families:

- Normal: `NormalInplace`, `NormalFunctional`, `NormalOut`,
  `Normal{FloatFloat,FloatTensor,TensorFloat,TensorTensor}(+Out)`
- Bernoulli: `Bernoulli`, `BernoulliTensor(+Out)`, `BernoulliFloatOut`,
  `BernoulliOut`, `BernoulliInplace{Float,Tensor}`
- Rand/Randn factory: `Rand{,Like}Generator(+Out)`,
  `Rand{,Like}GeneratorWithNames(+Out)`, `Randn{,Like}Generator(+Out)`,
  `RandnGeneratorWithNames(+Out)`
- Randint: `RandintGenerator(+Out)`, `RandintLowGenerator(+Out)`,
  `RandintLike{,Tensor,LowGeneratorDtype}Generator(+Out)`
- Randperm: `RandpermGenerator(+Out)`
- Random_: `Random{,From,To}(+Out)`, `RandomInplace{,From,To}`
- Distributions: `Uniform{,Inplace,Out}`, `Exponential{,Inplace,Out}`,
  `Geometric{,Inplace,Out}`, `LogNormal{,Inplace,Out}`, `Cauchy{,Inplace,Out}`,
  `Poisson(+Out)`, `Binomial(+Out)`, `Multinomial(+Out)`,
  `PrivStandardGamma(+Out)`, `PrivSampleDirichlet(+Out)`, `PrivFusedDropout(+Out)`,
  `RreluWithNoise{,Inplace,Functional,Out}`

Injection is a no-op when the caller passes a generator, so applying it to the
explicit `.generator` overloads too is safe and uniform.

## Implementation surface

1. **C++ helper** `GetFlagosDefaultCudaGenerator(int64_t)` — new function in the
   flagos backend (near `python_op_caller.{h,cc}`): GIL-acquire, read
   `torch.cuda.default_generators[idx]`, return `at::Generator`. Consider a
   process-lifetime cache keyed by index (the shim generator is stable) to avoid
   a Python round-trip per RNG call; must stay correct across
   `torch.cuda.manual_seed` (which mutates the same object in place, so a cached
   `at::Generator` handle remains valid).
2. **Codegen** `scripts/codegen/codegen_ops.py` — add a predicate "native kernel with a
   `Generator?` arg" and emit the injection line in the affected body templates
   (`gen_functional_pure`, `gen_inplace`, `gen_out_variant`, `gen_tuple_return`,
   and the factory-RNG template). Regenerate `csrc/aten/generated/cuda_kernels.cc`.
3. **Rebuild** `_C.so` (CUDA_KERNEL=ON, single-wheel scheme).
4. **Config** — no routing changes required; native RNG stays `= cuda`. (The
   unification is now internal to those kernels.)

## Testing

- Extend `tests/integration/ops/test_rng_dispatch.py` (or a sibling) to assert
  reproducibility + seed-sensitivity under `torch.manual_seed` for the native
  families now unified: `normal_`, `randint`/`randint.low`, `bernoulli.Tensor`,
  `uniform`(functional), `log_normal_`, `cauchy_`, `exponential`(native),
  `multinomial.out`, `poisson`, and **`randperm`**.
- Cross-world consistency: with FlagGems on, seed once and confirm a FlagGems op
  and a native op both advance/consume the shared generator deterministically.
- Regression: full native suite + flaggems_python suite show no new failures.
  Exclude the pre-existing `test_conv1d_dispatch.py::test_conv1d_with_bias`
  segfault (unrelated, fails on clean tree).
- Verify backward compat: a user-supplied `generator=` still takes effect
  (injection skipped).

## Risks / open items

- **Per-call GIL cost.** Reading Python `default_generators` under the GIL on
  every native RNG call adds overhead. Mitigation: cache the `at::Generator` per
  device index in C++ after first fetch.
- **Generator device-type assert.** ATen `check_generator` asserts the generator
  device type matches the op. We inject a CUDA generator into ops running (post-
  boxing) on CUDA, so this holds — but must be verified per family, especially
  factory ops where the op nominally targets `_cuda_dev`.
- **Ops that hardcode `generator=None` downstream.** Some code paths ignore a
  passed generator (noted in codegen comments for certain gems ops). For native
  `at::<op>` this is not expected, but the out/functional variants should be
  spot-checked that ATen actually threads the injected generator through.
- **Multi-device.** Index derivation must be correct for non-zero device
  indices; tests should cover at least device 0 and 1 if the host has >1 GPU.
