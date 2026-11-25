# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""flagos unified-RNG regression coverage.

`torch.manual_seed(s)` must make flagos draws reproducible across the
native and FlagGems RNG kernels. Explicit generators must produce their own
reproducible sequence and stay independent of the default device generator.
The acceptable generator device depends on the installed kernel.

Every test carries `main_ops` because CI's operator jobs select on it; a test
without that marker never runs in CI.

Usage:
    pytest tests/integration/ops/test_rng_dispatch.py -v
"""

import warnings

import pytest

import torch
import torch_fl  # noqa: F401


def _is_musa() -> bool:
    return torch_fl._build_accelerator() == "musa"


def _supports_flagos_generator() -> bool:
    """Backends whose native RNG kernels consume the flagos generator."""
    return torch_fl._build_accelerator() in ("ascend", "gcu", "musa", "kunlunxin")


#: The conf backends whose route ends in a FlagGems kernel.
_FLAGGEMS_BACKENDS = frozenset({"flaggems", "flaggems_cpp", "flaggems_python"})


def _conf_route(op: str) -> str:
    """The backend the conf this wheel resolved routes ``op`` to, or "".

    Parsed the way the C++ reader does: drop anything after ``#``, split the
    rest on the first ``=``. An empty answer covers "unlisted" and "no conf
    resolved" together, which is all the caller needs to tell apart from a
    named backend.
    """
    path = torch_fl.backend_config_path()
    if not path:
        return ""
    with open(path) as conf:
        for line in conf:
            name, sep, value = line.split("#")[0].partition("=")
            if sep and name.strip() == op:
                return value.strip()
    return ""


def _skip_if_random_leaves_the_device(op):
    """Skip a draw whose op this wheel does not serve on the device at all.

    The two reproducibility cases below draw from every op in the catalogue,
    and a draw whose op is unregistered here does not come back flagged: it
    reaches ATen's `random()` dispatch stub through cpu_fallback, and a
    CPU-only torch has no kernel there -- "DispatchStub: missing kernel for
    cuda" (NVIDIA lane, run 40146671425: 29 failures, `multinomial`, `random_`,
    `randint`, `randperm` and their out variants).

    `op()` alone answers a different question, and it answered wrong: the
    draws that got past it failed in the warmup's `.float()` (run 40147673906,
    the same 29), which is where the boxed read of the drawn tensor -- not
    the draw -- reached the stub. So the probe runs the draw exactly as the
    cases do. That is the narrowest form of the question and the one that
    cannot go stale: a draw this wheel cannot serve raises, a served one
    returns.
    """
    try:
        _draw(op, SEED)
    except (NotImplementedError, RuntimeError) as exc:
        if "not implemented" not in str(exc) and "DispatchStub" not in str(exc):
            raise
        pytest.skip(f"draw reaches cpu_fallback on this wheel: {exc}")


def _factory_rng_served_by_flag_gems() -> bool:
    """True when this wheel answers the factory RNG path with a FlagGems kernel.

    ``torch.rand(..., generator=...)`` reaches the device through ``uniform_``
    (see the module docstring), so the factory path is exactly as unclaimed as
    the conf's ``uniform_`` route. When that route is a FlagGems arm the call is
    served -- the generated wrapper takes the explicit generator instead of
    rejecting it -- and the rejection this file asserts for an unrouted
    generator has nothing to hold on. Three facts about the wheel, no platform
    name: the route the conf states, the kernel set the build record lists, and
    whether the package that arm calls is importable.
    """
    if _conf_route("uniform_") not in _FLAGGEMS_BACKENDS:
        return False
    if "flaggems" not in torch_fl._env.build_kernels():
        return False
    try:
        import flag_gems  # noqa: F401
    except Exception:  # noqa: BLE001 - an unimportable arm cannot serve the call
        return False
    return True


_NEEDS_DEVICE_RNG = pytest.mark.skipif(
    not _supports_flagos_generator(),
    reason="requires a device RNG that reads the state from torch.flagos",
)


DEVICE = "flagos:0"
SEED = 1234
# Far enough from SEED that a collision is not plausible, but the value itself is
# arbitrary -- any different seed must produce a different draw.
OTHER_SEED = SEED + 977

aten = torch.ops.aten


def _empty(n=64, dtype=torch.float32, device=DEVICE):
    return torch.empty(n, dtype=dtype, device=device)


def _require_device_arithmetic(*operations):
    """Skip when a device dtype conversion or scalar multiply leaves the device.

    Comparing results and scaling the dropout mask need dense arithmetic. On
    a wheel whose conf routes these operations to ``none``, each of
    them reaches cpu_fallback, and a CPU-only torch has no copy or mul kernel
    for a non-CPU tensor there. Probe the exact expressions rather than the ops
    by name: a route that changes still answers correctly, and the ones that
    cannot run skip with the exception that proves it.
    """
    for operation in operations:
        try:
            operation()
        except (NotImplementedError, RuntimeError) as exc:
            if "not implemented" not in str(exc) and "DispatchStub" not in str(exc):
                raise
            pytest.skip(f"device arithmetic reaches cpu_fallback on this wheel: {exc}")


def _draw(op, seed):
    torch.manual_seed(seed)
    return op().float().cpu()


def _generator_device_rejection() -> RuntimeError | None:
    """The rejection this wheel gives a flagos generator on a device RNG op, if any.

    Asked, not assumed: this backend's RNG either accepts ``Generator(device=
    "flagos")`` or refuses it, and which of the two it is does not follow from
    the build record on every vendor. A draw that returns here means the wheel
    accepts the generator, which is the outcome the three generator cases below
    record rather than contradict -- the contract they guard is that the
    generator is honoured, not that it is refused. A refusal still has to be
    the documented one.
    """
    gen = torch.Generator(device="flagos").manual_seed(99)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            _empty(8).normal_(generator=gen)
        except RuntimeError as exc:
            if "device type for generator" not in str(exc):
                raise
            return exc
    return None


def _skip_if_generator_is_accepted(rejection) -> None:
    """Skip a rejection case on a wheel whose RNG accepts the flagos generator."""
    if rejection is None:
        pytest.skip(
            "this wheel's device RNG accepts a flagos generator, so there is "
            "no rejection to assert; the generator-honouring cases below cover "
            "the acceptance"
        )


def _explicit_generator(seed):
    """A caller-supplied generator that this backend's RNG kernels accept.

    A native PrivateUse1 kernel may consume a flagos generator. Probe other
    wheels for acceptance before choosing a CPU generator.
    """
    if _supports_flagos_generator() or _generator_device_rejection() is None:
        return torch.Generator(device="flagos").manual_seed(seed)
    return torch.Generator().manual_seed(seed)


# --- op catalogue -----------------------------------------------------------
#
# Grouped by the mechanism each group stresses, because the injection bug this
# file guards against reappeared once per codegen template: an op whose schema
# lacks `Generator?` gets no injection and silently falls back to ATen's own
# default generator. Factory, `*_like` and out-variant overloads each needed a
# separate fix, so each is represented here.

# Elementwise / reduction RNG that FlagGems implements in Triton (and that the
# vendor config routes to the native kernels).
_GEMS_PATH_OPS = {
    "rand": lambda: torch.rand(64, device=DEVICE),
    "randn": lambda: torch.randn(64, device=DEVICE),
    "rand_like": lambda: torch.rand_like(_empty()),
    "randn_like": lambda: torch.randn_like(_empty()),
    "uniform_": lambda: _empty().uniform_(),
    "exponential_": lambda: _empty().exponential_(),
    "bernoulli_.float": lambda: _empty().bernoulli_(0.5),
    "multinomial": lambda: torch.multinomial(
        torch.ones(10, device=DEVICE), 5, replacement=True
    ),
}

# `Generator?`-carrying native kernels: the `if (!generator.has_value())` form.
_NATIVE_GENERATOR_OPS = {
    "normal_": lambda: _empty().normal_(),
    "normal_.mean_std": lambda: _empty().normal_(2.0, 3.0),
    "normal.float_float": lambda: torch.normal(0.0, 1.0, (64,), device=DEVICE),
    "normal.Tensor_float": lambda: torch.normal(torch.zeros(64, device=DEVICE), 1.0),
    "normal.Tensor_Tensor": lambda: torch.normal(
        torch.zeros(64, device=DEVICE), torch.ones(64, device=DEVICE)
    ),
    "bernoulli.Tensor": lambda: torch.bernoulli(torch.full((64,), 0.5, device=DEVICE)),
    "bernoulli_.Tensor": lambda: _empty().bernoulli_(
        torch.full((64,), 0.5, device=DEVICE)
    ),
    "random_": lambda: _empty(dtype=torch.int64).random_(),
    "random_.to": lambda: _empty(dtype=torch.int64).random_(50),
    "random_.from": lambda: _empty(dtype=torch.int64).random_(10, 50),
    "log_normal_": lambda: _empty().log_normal_(),
    "cauchy_": lambda: _empty().cauchy_(),
    "geometric_": lambda: _empty().geometric_(0.5),
    "poisson": lambda: torch.poisson(torch.full((64,), 4.0, device=DEVICE)),
    "_standard_gamma": lambda: torch._standard_gamma(
        torch.full((64,), 2.0, device=DEVICE)
    ),
    "_sample_dirichlet": lambda: torch._sample_dirichlet(
        torch.full((8, 4), 2.0, device=DEVICE)
    ),
    "binomial": lambda: torch.binomial(
        torch.full((64,), 10.0, device=DEVICE), torch.full((64,), 0.5, device=DEVICE)
    ),
}

# Generator-LESS factory / *_like overloads. `torch.randint(...)` and
# `torch.randperm(...)` dispatch to these, NOT to their `.generator` siblings, so
# they were non-reproducible until the factory and functional-pure templates
# learned to thread the shared generator in explicitly.
_GENERATOR_LESS_OPS = {
    "randint": lambda: torch.randint(0, 100, (64,), device=DEVICE),
    "randint.low": lambda: torch.randint(10, 100, (64,), device=DEVICE),
    "randint_like": lambda: torch.randint_like(_empty(dtype=torch.int64), 0, 50),
    "randperm": lambda: torch.randperm(50, device=DEVICE),
}

# Generator-LESS out-variants -- the last template to be fixed. ATen places the
# injected `optional<Generator>` before the first of names/memory_format/out, so
# a mis-ordered injection would bind the wrong overload; a silent fallback to
# ATen's default generator shows up here as non-reproducibility.
_OUT_VARIANT_OPS = {
    "rand.out": lambda: torch.rand(64, out=_empty()),
    "randn.out": lambda: torch.randn(64, out=_empty()),
    "rand.names_out": lambda: aten.rand.names_out([64], names=None, out=_empty()),
    "randn.names_out": lambda: aten.randn.names_out([64], names=None, out=_empty()),
    "rand_like.out": lambda: aten.rand_like.out(_empty(), out=_empty()),
    "randn_like.out": lambda: aten.randn_like.out(_empty(), out=_empty()),
    "randint.out": lambda: torch.randint(0, 100, (64,), out=_empty(dtype=torch.int64)),
    "randint.low_out": lambda: torch.randint(
        10, 100, (64,), out=_empty(dtype=torch.int64)
    ),
    "randint_like.out": lambda: aten.randint_like.out(
        _empty(dtype=torch.int64), 50, out=_empty(dtype=torch.int64)
    ),
    "randint_like.low_dtype_out": lambda: aten.randint_like.low_dtype_out(
        _empty(dtype=torch.int64), 5, 50, out=_empty(dtype=torch.int64)
    ),
    "randperm.out": lambda: torch.randperm(50, out=_empty(50, torch.int64)),
}

_ALL_OPS = {
    **_GEMS_PATH_OPS,
    **_NATIVE_GENERATOR_OPS,
    **_GENERATOR_LESS_OPS,
    **_OUT_VARIANT_OPS,
}


class TestRngReproducible:
    """Same seed -> identical draw, for every routed RNG op."""

    @pytest.mark.anyplatform
    @pytest.mark.main_ops
    @pytest.mark.parametrize("name", list(_ALL_OPS))
    def test_same_seed_same_draw(self, name):
        op = _ALL_OPS[name]
        _skip_if_random_leaves_the_device(op)
        assert torch.equal(_draw(op, SEED), _draw(op, SEED)), (
            f"{name} is not reproducible under torch.manual_seed -- it is most "
            f"likely falling back to ATen's own default generator instead of the "
            f"shared flagos generator"
        )

    @pytest.mark.anyplatform
    @pytest.mark.main_ops
    @pytest.mark.parametrize("name", list(_ALL_OPS))
    def test_different_seed_differs(self, name):
        # Reproducibility alone is satisfied by a *constant* draw, which is how
        # the original philox monkeypatch bug hid: it seeded once and ignored
        # later manual_seed calls. Seed-sensitivity is what rules that out.
        op = _ALL_OPS[name]
        _skip_if_random_leaves_the_device(op)
        assert not torch.equal(_draw(op, SEED), _draw(op, OTHER_SEED)), (
            f"{name} ignores the seed -- same draw for two different seeds"
        )


class TestRngSeedSource:
    """The seed plumbing itself, independent of any single op."""

    @pytest.mark.anyplatform
    @pytest.mark.main_ops
    def test_flagos_default_generators_iterable(self):
        # The public flagos generator proxy must behave like the upstream
        # list-like default_generators object on every backend.
        gens = torch.flagos.default_generators
        n = len(gens)
        assert len(list(gens)) == n, "iteration does not stop at device_count"
        assert len(gens[:2]) == min(2, n), "slicing is not supported"
        with pytest.raises(IndexError):
            gens[n]

    @pytest.mark.anyplatform
    @pytest.mark.main_ops
    def test_manual_seed_reaches_flagos_module(self):
        # torch.random._seed_custom_device only seeds this device module when it
        # exposes BOTH manual_seed_all and _is_in_bad_fork; missing either one
        # makes torch.manual_seed warn "does not take effect" and skip it.
        torch.manual_seed(SEED)
        assert torch_fl.flagos.initial_seed() == SEED

    @pytest.mark.anyplatform
    @pytest.mark.main_ops
    def test_flagos_manual_seed_reproducible_across_reseeds(self):
        torch.flagos.manual_seed(SEED)
        first = torch.randn(64, device=DEVICE).cpu()
        torch.flagos.manual_seed(OTHER_SEED)
        other = torch.randn(64, device=DEVICE).cpu()
        torch.flagos.manual_seed(SEED)
        second = torch.randn(64, device=DEVICE).cpu()
        assert torch.equal(first, second)
        assert not torch.equal(first, other)

    @pytest.mark.anyplatform
    @pytest.mark.main_ops
    def test_default_generator_bounds_raise_clean_error(self):
        count = torch.flagos.device_count()
        for bad_idx in (count, count + 3, -2):
            with pytest.raises(RuntimeError):
                torch_fl._C._get_default_generator(bad_idx)
        with pytest.raises((RuntimeError, IndexError)):
            torch.flagos.get_rng_state(count)

    @pytest.mark.anyplatform
    @pytest.mark.main_ops
    def test_zero_visible_devices_manual_seed_subprocess(self):
        import os
        import subprocess
        import sys

        env = os.environ.copy()
        for var in (
            "ROCR_VISIBLE_DEVICES",
            "HIP_VISIBLE_DEVICES",
            "CUDA_VISIBLE_DEVICES",
            "ASCEND_RT_VISIBLE_DEVICES",
            "MUSA_VISIBLE_DEVICES",
            "XPURT_VISIBLE_DEVICES",
            "ZE_AFFINITY_MASK",
        ):
            env[var] = ""
        env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
        probe = (
            "import resource; resource.setrlimit(resource.RLIMIT_CORE, (0, 0))\n"
            "import torch_fl, torch\n"
            "if torch.flagos.device_count() != 0:\n"
            "    print('SKIP'); raise SystemExit(0)\n"
            "torch.flagos.manual_seed(1)\n"
            "torch.flagos.manual_seed_all(1)\n"
            "torch.flagos.seed()\n"
            "torch.flagos.seed_all()\n"
            "torch.manual_seed(1)\n"
            "for idx in (0, -1):\n"
            "    try:\n"
            "        torch_fl._C._get_default_generator(idx)\n"
            "    except RuntimeError:\n"
            "        pass\n"
            "    else:\n"
            "        raise AssertionError(f'expected RuntimeError for {idx}')\n"
            "print('OK')\n"
        )
        res = subprocess.run(
            [sys.executable, "-c", probe],
            env=env,
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )
        assert res.returncode == 0, res.stdout + res.stderr
        if res.stdout.strip() == "SKIP":
            pytest.skip("visibility masks do not hide devices on this runtime")
        assert res.stdout.strip().endswith("OK")

    @pytest.mark.anyplatform
    @pytest.mark.main_ops
    def test_manual_seed_emits_no_ineffective_warning(self):
        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            torch.manual_seed(SEED)
        assert not [w for w in caught if "does not take effect" in str(w.message)]

    @pytest.mark.anyplatform
    @pytest.mark.main_ops
    def test_flagos_generator_device_contract(self):
        gen = torch.Generator(device="flagos").manual_seed(99)
        if _supports_flagos_generator():
            first = _empty().normal_(generator=gen).cpu()
            gen.manual_seed(99)
            second = _empty().normal_(generator=gen).cpu()
            assert torch.equal(first, second)
        else:
            # Backends whose native kernels use a different generator contract
            # reject a flagos generator rather than silently changing streams.
            _skip_if_generator_is_accepted(_generator_device_rejection())
            with pytest.raises(RuntimeError, match="device type for generator"):
                _empty(8).normal_(generator=gen)

    @pytest.mark.anyplatform
    @pytest.mark.main_ops
    def test_flagos_generator_factory_contract(self):
        gen = torch.Generator(device="flagos").manual_seed(99)
        if _supports_flagos_generator():
            first = torch.rand(8, device=DEVICE, generator=gen).cpu()
            gen.manual_seed(99)
            second = torch.rand(8, device=DEVICE, generator=gen).cpu()
            assert torch.equal(first, second)
        elif _factory_rng_served_by_flag_gems():
            # The composite factory path is answered by a FlagGems kernel on
            # this wheel, so nothing here rejects the generator and the
            # assertion below has nothing to hold on.
            pytest.skip(
                "the factory RNG path is served by a FlagGems kernel on this "
                "wheel, which accepts an explicit generator instead of "
                "rejecting it"
            )
        else:
            _skip_if_generator_is_accepted(_generator_device_rejection())
            with pytest.raises(RuntimeError, match="device type for generator"):
                torch.rand(8, device=DEVICE, generator=gen)

    @pytest.mark.anyplatform
    @pytest.mark.main_ops
    def test_flagos_generator_does_not_reroute_cpu_op(self):
        # The tensor is CPU, so a PrivateUse1 redispatch here can only have come
        # from the generator.
        gen = torch.Generator(device="flagos")
        with pytest.raises(RuntimeError, match="device type for generator"):
            torch.empty(8).normal_(generator=gen)

    @pytest.mark.anyplatform
    @pytest.mark.main_ops
    def test_explicit_generator_is_honoured(self):
        # Injection must only fill an *absent* generator. A caller-supplied one
        # keeps its own stream, reproducible on its own terms.
        gen = _explicit_generator(99)
        a = _empty().normal_(generator=gen).cpu()
        gen.manual_seed(99)
        b = _empty().normal_(generator=gen).cpu()
        assert torch.equal(a, b)

    @pytest.mark.anyplatform
    @pytest.mark.main_ops
    def test_explicit_generator_isolated_from_manual_seed(self):
        gen = _explicit_generator(5)
        a = _empty().normal_(generator=gen).cpu()
        torch.manual_seed(SEED)  # must not perturb `gen`
        gen.manual_seed(5)
        b = _empty().normal_(generator=gen).cpu()
        assert torch.equal(a, b)

    @pytest.mark.anyplatform
    @pytest.mark.main_ops
    def test_interleaved_paths_share_one_seed(self):
        # A model mixes both paths freely (dropout next to init). One seed must
        # drive the whole interleaved sequence, so drawing from the FlagGems path
        # and the native path under one seed has to replay identically.
        def mixed():
            gems = torch.rand(32, device=DEVICE)
            native = _empty(32).normal_()
            return torch.cat([gems, native])

        assert torch.equal(_draw(mixed, SEED), _draw(mixed, SEED))
        assert not torch.equal(_draw(mixed, SEED), _draw(mixed, OTHER_SEED))

    @pytest.mark.anyplatform
    @pytest.mark.main_ops
    @_NEEDS_DEVICE_RNG
    def test_rng_state_round_trip(self):
        torch.flagos.manual_seed(SEED)
        state = torch.flagos.get_rng_state()
        first = _empty(17).normal_(0.5, 2.0)
        torch.flagos.set_rng_state(state)
        second = _empty(17).normal_(0.5, 2.0)
        assert torch.equal(first, second)

    @pytest.mark.anyplatform
    @pytest.mark.main_ops
    def test_integer_factory_and_out_contract(self):
        torch.flagos.manual_seed(SEED)
        values = torch.randint(-7, 13, (128,), device=DEVICE, dtype=torch.int32)
        assert values.dtype == torch.int32
        assert bool(torch.all(values >= -7)) and bool(torch.all(values < 13))

        out = torch.empty((128,), device=DEVICE, dtype=torch.int64)
        torch.randint(100, (128,), device=DEVICE, out=out)
        assert bool(torch.all(out >= 0)) and bool(torch.all(out < 100))

    @pytest.mark.anyplatform
    @pytest.mark.main_ops
    def test_like_preserves_shape_layout_and_device(self):
        ref = torch.empty((4, 9), device=DEVICE).transpose(0, 1)
        torch.flagos.manual_seed(SEED)
        sampled = torch.rand_like(ref)
        assert sampled.device == ref.device
        assert sampled.shape == ref.shape
        assert sampled.is_contiguous() == ref.is_contiguous()

    @pytest.mark.anyplatform
    @pytest.mark.main_ops
    def test_dropout_forward_backward_contract(self):
        torch.flagos.manual_seed(SEED)
        value = torch.ones((256,), device=DEVICE)
        # native_dropout draws from `random_`, so a wheel that does not serve
        # that draw cannot run the contract at all (CUDA lane, run
        # 40148742285); the probe is the same one the catalogue uses. It takes
        # the first element of the tuple: the probe warms up with `.float()`,
        # which a two-output op has no single answer for (Kunlunxin lane, run
        # 40149862610).
        _skip_if_random_leaves_the_device(
            lambda: torch.ops.aten.native_dropout(value, 0.25, True)[0]
        )
        output, mask = torch.ops.aten.native_dropout(value, 0.25, True)
        assert output.device == torch.device(DEVICE)
        assert mask.device == torch.device(DEVICE)
        assert mask.dtype == torch.bool
        assert torch.equal(output.ne(0), mask)
        # The draw can be served while the *comparison* is not: every unary here
        # leaves the device through cpu_fallback, and a CPU-only torch has no
        # float->float copy kernel for it -- "DispatchStub: missing kernel for
        # cuda" at `mask.to(output.dtype)` (NVIDIA lane, run 40154845873). The
        # contract is a device one, so skip where the wheel cannot express it.
        _require_device_arithmetic(
            lambda: mask.to(output.dtype) * (1.0 / 0.75),
            lambda: mask.to(torch.float32) * 4.0,
        )
        expected = mask.to(output.dtype) * (1.0 / 0.75)
        assert torch.allclose(output, expected)

        grad = torch.ops.aten.native_dropout_backward(
            torch.ones_like(output), mask, 4.0
        )
        assert torch.equal(grad, mask.to(grad.dtype) * 4.0)

    @pytest.mark.anyplatform
    @pytest.mark.main_ops
    def test_full_width_integer_ranges(self):
        torch.flagos.manual_seed(SEED)
        values = torch.empty((4096,), device=DEVICE, dtype=torch.int64).random_(
            -(1 << 63), (1 << 63) - 1
        )
        assert bool(torch.all(values >= -(1 << 63)))
        assert bool(torch.all(values < (1 << 63) - 1))
        assert bool(torch.any(values < 0)) and bool(torch.any(values >= 0))

        defaults = torch.empty((4096,), device=DEVICE, dtype=torch.int8).random_()
        assert bool(torch.all(defaults >= 0))
        assert bool(torch.all(defaults <= torch.iinfo(torch.int8).max))

    @pytest.mark.musa
    @pytest.mark.main_ops
    def test_musa_shared_reservation_order(self):
        torch.flagos.manual_seed(SEED)
        initial_state = torch.flagos.get_rng_state()
        bridge_seed = torch_fl._C._reserve_rng_seed(0)
        torch.rand((16,), device=DEVICE)
        mixed_state = torch.flagos.get_rng_state()

        torch.flagos.set_rng_state(initial_state)
        first_seed = torch_fl._C._reserve_rng_seed(0)
        torch_fl._C._reserve_rng_seed(0)
        reservation_state = torch.flagos.get_rng_state()
        assert bridge_seed == first_seed
        assert torch.equal(mixed_state, reservation_state)


class TestRngMultiDevice:
    """Per-device generators must be independently seeded and addressed.

    One rank per card is the normal distributed layout, so an RNG scheme that
    only works on device 0 is broken in exactly the setting that matters most.
    """

    @staticmethod
    def _second_device():
        if torch_fl.flagos.device_count() < 2:
            pytest.skip("needs at least 2 flagos devices")
        return "flagos:1"

    @pytest.mark.anyplatform
    @pytest.mark.main_ops
    @pytest.mark.parametrize("name", ["randn", "normal_", "randint", "randperm"])
    def test_reproducible_on_second_device(self, name):
        dev = self._second_device()
        ops = {
            "randn": lambda: torch.randn(64, device=dev),
            "normal_": lambda: _empty(64, device=dev).normal_(),
            "randint": lambda: torch.randint(0, 100, (64,), device=dev),
            "randperm": lambda: torch.randperm(50, device=dev),
        }
        op = ops[name]
        _skip_if_random_leaves_the_device(op)
        assert torch.equal(_draw(op, SEED), _draw(op, SEED))
        assert not torch.equal(_draw(op, SEED), _draw(op, OTHER_SEED))

    @pytest.mark.anyplatform
    @pytest.mark.main_ops
    def test_manual_seed_all_replays_each_device_independently(self):
        dev = self._second_device()
        torch.flagos.manual_seed_all(SEED)
        first0 = torch.rand(32, device=DEVICE)
        first1 = torch.rand(32, device=dev)

        torch.flagos.manual_seed_all(SEED)
        second1 = torch.rand(32, device=dev)
        second0 = torch.rand(32, device=DEVICE)
        assert first0.device.index == 0
        assert first1.device.index == 1
        assert torch.equal(first0, second0)
        assert torch.equal(first1, second1)

        torch.flagos.manual_seed_all(SEED)
        expected1 = torch.rand(32, device=dev)
        torch.flagos.manual_seed_all(SEED)
        torch.rand(32, device=DEVICE)
        actual1 = torch.rand(32, device=dev)
        assert torch.equal(expected1, actual1)

    @pytest.mark.anyplatform
    @pytest.mark.main_ops
    def test_each_device_has_its_own_generator(self):
        dev = self._second_device()
        gens = torch.flagos.default_generators
        assert len(gens) >= 2
        # Same seed on both devices: the draws are allowed to coincide (identical
        # inputs), but the generators must be distinct objects, or seeding one
        # device would silently reset another.
        assert gens[0] is not gens[1]
        torch.manual_seed(SEED)
        d0 = torch.randn(64, device=DEVICE).cpu()
        torch.manual_seed(SEED)
        d1 = torch.randn(64, device=dev).cpu()
        assert d0.shape == d1.shape

    @pytest.mark.flaggems
    @pytest.mark.main_ops
    @pytest.mark.xfail(
        reason="FlagGems multinomial launches its Triton kernel against the wrong "
        "device for any index != 0 -- 'Pointer argument (at 0) cannot be accessed "
        "from Triton (cpu tensor?)' on flagos:1..7, fine on flagos:0. A device-context "
        "bug on the FlagGems Triton path rather than an RNG one: the same op on the "
        "native path is reproducible on every device. xfail (non-strict) so the "
        "eventual fix surfaces as an xpass instead of being silently assumed. "
        "RuntimeError is in `raises` because an unrouted multinomial may fail "
        "earlier with 'DispatchStub: missing kernel for cuda'.",
        strict=False,
        raises=(ValueError, AssertionError, RuntimeError),
    )
    def test_multinomial_on_second_device(self):
        dev = self._second_device()

        def op():
            return torch.multinomial(torch.ones(10, device=dev), 5, replacement=True)

        assert torch.equal(_draw(op, SEED), _draw(op, SEED))


class TestRngDistribution:
    """Reproducibility is worthless if the numbers are wrong.

    A mis-threaded generator argument can bind a different overload and still
    look reproducible, so pin the actual distributions. Tolerances are loose
    enough for 100k samples at any seed.
    """

    @pytest.mark.anyplatform
    @pytest.mark.main_ops
    def test_rand_uniform_0_1(self):
        torch.manual_seed(42)
        r = torch.rand(100_000, device=DEVICE).cpu()
        assert abs(r.mean().item() - 0.5) < 0.02
        assert r.min().item() >= 0.0 and r.max().item() < 1.0

    @pytest.mark.anyplatform
    @pytest.mark.main_ops
    def test_randn_standard_normal(self):
        torch.manual_seed(42)
        n = torch.randn(100_000, device=DEVICE).cpu()
        assert abs(n.mean().item()) < 0.02
        assert abs(n.std().item() - 1.0) < 0.02

    @pytest.mark.anyplatform
    @pytest.mark.main_ops
    def test_normal_mean_std(self):
        torch.manual_seed(42)
        n = _empty(100_000).normal_(2.0, 3.0).cpu()
        assert abs(n.mean().item() - 2.0) < 0.05
        assert abs(n.std().item() - 3.0) < 0.05

    @pytest.mark.anyplatform
    @pytest.mark.main_ops
    def test_exponential_rate_1(self):
        torch.manual_seed(42)
        e = _empty(100_000).exponential_().cpu()
        assert abs(e.mean().item() - 1.0) < 0.05
        assert e.min().item() >= 0.0

    @pytest.mark.anyplatform
    @pytest.mark.main_ops
    def test_uniform_range(self):
        torch.manual_seed(42)
        u = _empty(100_000).uniform_(2.0, 5.0).cpu()
        assert abs(u.mean().item() - 3.5) < 0.05
        assert u.min().item() >= 2.0 and u.max().item() <= 5.0

    @pytest.mark.anyplatform
    @pytest.mark.main_ops
    def test_randint_range(self):
        torch.manual_seed(42)
        r = torch.randint(0, 10, (100_000,), device=DEVICE).cpu()
        assert r.min().item() == 0 and r.max().item() == 9
        assert abs(r.float().mean().item() - 4.5) < 0.05

    @pytest.mark.anyplatform
    @pytest.mark.main_ops
    def test_randperm_is_a_permutation(self):
        torch.manual_seed(42)
        p = torch.randperm(2000, device=DEVICE).cpu()
        assert sorted(p.tolist()) == list(range(2000))

    @pytest.mark.anyplatform
    @pytest.mark.main_ops
    def test_poisson_mean_equals_variance(self):
        torch.manual_seed(42)
        p = torch.poisson(torch.full((100_000,), 4.0, device=DEVICE)).cpu()
        assert abs(p.mean().item() - 4.0) < 0.06
        assert abs(p.var().item() - 4.0) < 0.15  # Poisson: var == mean

    @pytest.mark.anyplatform
    @pytest.mark.main_ops
    def test_bernoulli_probability(self):
        torch.manual_seed(42)
        b = _empty(100_000).bernoulli_(0.3).cpu()
        assert set(b.unique().tolist()) <= {0.0, 1.0}
        assert abs(b.mean().item() - 0.3) < 0.01

    @pytest.mark.anyplatform
    @pytest.mark.main_ops
    def test_multinomial_respects_weights(self):
        torch.manual_seed(42)
        weights = torch.tensor([0.1, 0.9], device=DEVICE)
        picks = torch.multinomial(weights, 20_000, replacement=True).cpu()
        assert set(picks.unique().tolist()) <= {0, 1}
        assert abs((picks == 1).float().mean().item() - 0.9) < 0.02


class TestRngDropout:
    """dropout is the RNG op models actually depend on for correctness.

    It is split out because its reproducibility is config-dependent: FlagGems
    routes `native_dropout` to its own Triton kernel (seeded from the shared
    generator), while the vendor config calls the ATen composite, whose schema
    exposes no `Generator?` at all -- there is no argument to inject, so the
    randomness comes from ATen's default generator and `torch.manual_seed` cannot
    reach it. Marked `flaggems` so it asserts the contract only where the
    contract currently holds; see test_dropout_masks_are_valid for the part that
    must hold everywhere.
    """

    @pytest.mark.anyplatform
    @pytest.mark.main_ops
    def test_dropout_masks_are_valid(self):
        # Config-independent: scaled-or-zeroed entries, roughly the right rate.
        torch.manual_seed(SEED)
        out = torch.nn.functional.dropout(
            torch.ones(100_000, device=DEVICE), p=0.5, training=True
        ).cpu()
        kept = out != 0
        assert abs(kept.float().mean().item() - 0.5) < 0.01
        # Survivors are scaled by 1/(1-p).
        assert torch.allclose(out[kept], torch.full_like(out[kept], 2.0), atol=1e-5)

    @staticmethod
    def _dropout():
        return torch.nn.functional.dropout(
            torch.ones(256, device=DEVICE), p=0.5, training=True
        )

    @pytest.mark.flaggems
    @pytest.mark.xfail(
        reason="FlagGems dropout is not reproducible with manual_seed. Same seed "
        "produces different masks across calls. Filed as FlagGems issue #6217.",
        strict=False,
    )
    def test_dropout_reproducible_on_flaggems_path(self):
        assert torch.equal(_draw(self._dropout, SEED), _draw(self._dropout, SEED))
        assert not torch.equal(
            _draw(self._dropout, SEED), _draw(self._dropout, OTHER_SEED)
        )

    @pytest.mark.anyplatform
    @pytest.mark.main_ops
    def test_dropout_reproducible(self):
        # No xfail: every platform's conf routes native_dropout to a
        # reproducible kernel (flaggems on cuda/dcu/metax/musa/ppu, ascend/gcu
        # native), so this holds wherever the test is collected. If it ever
        # fails on a platform, that is a real routing or reproducibility
        # regression -- do not re-add a conditional xfail without a measured
        # reason.
        assert torch.equal(_draw(self._dropout, SEED), _draw(self._dropout, SEED))
