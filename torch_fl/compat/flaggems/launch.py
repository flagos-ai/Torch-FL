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

"""Compatibility patches for FlagGems kernel launches."""

import torch

from ..._platform import platform_for
from . import runtime


# ---------------------------------------------------------------------------
# FlagGems per-launch host cost
#
# The routed ops are limited by their host side, not by their kernels: on a DCU
# bw1000 a bf16 ``where`` over ``(1, 4096, 1) x (1, 1, 4096)`` enqueues in
# 11.5 us on the vendor build and 262.4 us through FlagGems, while the device
# spends 124.2 and 74.0 us per call on the two -- so the vendor op is
# device-bound and the FlagGems op is Python-bound. The patches below take
# back the parts of that Python that are recomputed per launch without changing
# anything a caller can observe; none of them moves an op off FlagGems.
#
# Measured again on an H100 in the Qwen-Image-2.1 denoise loop, where the same
# reading holds and the kernels are not the difference either: the FlagGems
# ``mul`` kernel launched directly on (1, 4160, 3072) bf16 runs at 28.2 us and
# 2.72 TB/s, against the vendor's 28.7 us and 2.68 TB/s, while the *call*
# through ``torch.mul`` on that shape takes 85.4 us -- the kernel is at the
# memory roofline and 57 us of the call is Python around it. The patches below
# are therefore not a DCU-specific workaround; they are the fix for the shape
# of the FlagGems Python launch path, and every boxing build that routes to
# FlagGems has it.
# ---------------------------------------------------------------------------

# The builds whose routed ops are FlagGems Python/Triton launches reached through
# the generated bridge, which is the configuration these patches were written and
# measured on. ``dcu`` was the first; ``cuda`` is the CUDA-boxing build, where the
# same measurement was taken on an H100 (see the section header) -- an external
# libtorch_cuda.so behind a CPU torch changes how the vendor kernels are reached,
# not how FlagGems launches its own.
# Platform ids, not the record's vendor spelling: ``launch_patches_apply``
# resolves the record through ``platform_for`` before looking here.
_FLAGGEMS_LAUNCH_PATCH_BUILDS = ("dcu", "cuda")


def launch_patches_apply(accelerator: str):
    """Whether the per-launch host patches below apply to this build and conf.

    Both halves are load-bearing and are the two the individual functions used to
    spell out separately: the build has to be one whose routed ops go through the
    FlagGems Python launch path (a native or C++ vendor backend has its own, and
    patching FlagGems' ``LibEntry`` there would be dead weight), and the conf has
    to route to FlagGems at all -- a conf that does not has no launches to make
    cheaper.
    """
    return (
        platform_for(accelerator) in _FLAGGEMS_LAUNCH_PATCH_BUILDS
        and runtime.routes_to_flaggems()
    )


# The keyword arguments Triton's ``KernelInterface.__getitem__`` passes down to
# ``LibEntry.run``. A launch carrying anything else takes the stock path.
_FLAGGEMS_RUN_KWARGS = frozenset(("grid", "warmup"))

# Distinct from ``None``, so "this entry has no plan" and "no plan computed yet"
# do not collide in the per-entry plan cache.
_NO_PLAN = object()

# Enough for every shape a model repeats across a denoise loop, small enough
# that a cache keyed on shapes cannot grow without bound.
_BROADCAST_SHAPES_LIMIT = 4096


def _memoized_broadcast_shapes(original):
    """``torch.broadcast_shapes`` computed once per set of shapes.

    ``torch.broadcast_shapes`` is a Python wrapper: it asks
    ``torch.jit.is_tracing()`` and then delegates to
    ``torch._refs._broadcast_shapes``, which walks the shapes in Python.
    Measured on a DCU bw1000, three rank-3 shape tuples cost 21.9 us per call --
    and the same 21.9 us under this build's vendor torch, so it is a property of
    the DTK build rather than of the plugin. The vendor path pays none of it
    because its ``where`` never runs Python.

    FlagGems' ``where_self_out`` calls it once per ``where``
    (flag_gems/ops/where.py), and Qwen-Image-2.1 issues 130 ``where`` calls per
    denoise step on shapes that do not change between steps, so 2.8 ms of every
    step goes to recomputing one shape.

    The function is pure, so the memo cannot change an answer; the cache is
    dropped wholesale once it is large enough that a live model would have
    stopped repeating itself.
    """
    cache = {}
    limit = _BROADCAST_SHAPES_LIMIT

    def broadcast_shapes(*shapes):
        try:
            hit = cache.get(shapes)
        except TypeError:
            # A shape spelled as a list rather than a tuple: hashable callers
            # only, and this one is not.
            return original(*shapes)
        if hit is None:
            hit = original(*shapes)
            if len(cache) >= limit:
                cache.clear()
            cache[shapes] = hit
        return hit

    broadcast_shapes.__wrapped__ = original
    return broadcast_shapes


def _hoisted_descriptor_cache_key(descriptor_type):
    """``flag_gems.utils.libentry._descriptor_cache_key`` without the re-import.

    The stock function imports ``triton.tools.tensor_descriptor`` from inside
    its own body on every call, and ``LibEntry.key`` calls it once per kernel
    argument -- 37 times for a rank-3 pointwise kernel whose whole launch is
    ~350 us of host time. Resolving that import costs a ``sys.modules`` lookup,
    a ``getattr`` and a function frame each time, for a module that cannot have
    changed since the last launch.

    Behaviour is untouched: this is still a passthrough for everything that is
    not a ``TensorDescriptor``.
    """
    if descriptor_type is None:

        def descriptor_cache_key(arg):
            return arg

        return descriptor_cache_key

    def descriptor_cache_key(arg):
        if not isinstance(arg, descriptor_type):
            return arg
        return (
            "TensorDescriptor",
            tuple(arg.shape),
            tuple(arg.strides),
            tuple(arg.block_shape),
            getattr(arg, "padding", None),
        )

    return descriptor_cache_key


def _fast_libentry_key(entry, descriptor_cache_key, tensor_specialization):
    """``LibEntry.key`` with the per-call work it can do once, done once.

    The stock ``key`` defines ``spec_arg`` and ``dns_arg`` as nested functions,
    so every call builds two closures and pays two Python frames per argument,
    and ``spec_arg`` additionally re-resolves the vendor's
    ``get_tensor_specialization`` for every tensor argument. All of that is
    fixed when the kernel is constructed. What is left is one straight walk over
    the three argument lists, which is what this is.

    The key it produces is the stock key: same tuple, same order, same values.
    """
    divisibility = entry.divisibility
    specializes = tensor_specialization is not None

    def key(spec_args, dns_args, const_args):
        spec_key = []
        append = spec_key.append
        for arg in spec_args:
            arg = descriptor_cache_key(arg)
            if hasattr(arg, "data_ptr"):
                aligned = arg.data_ptr() % divisibility == 0
                if specializes:
                    append((arg.dtype, aligned, tensor_specialization(arg)))
                else:
                    append((arg.dtype, aligned))
            else:
                append((type(arg), arg))

        dns_key = []
        append = dns_key.append
        for arg in dns_args:
            arg = descriptor_cache_key(arg)
            if hasattr(arg, "data_ptr"):
                append(arg.dtype)
            elif not isinstance(arg, int):
                append(type(arg))
            elif -(2**31) <= arg <= 2**31 - 1:
                append("i32")
            elif 2**63 <= arg <= 2**64 - 1:
                append("u64")
            else:
                append("i64")

        const_key = [descriptor_cache_key(arg) for arg in const_args]
        return tuple(spec_key + dns_key + const_key)

    return key


def _libentry_run_plan(entry, n_args):
    """Which of ``args[0:n_args]`` are specialised, not, or constexpr.

    ``LibEntry.run`` decides this by walking ``signature.parameters`` and
    testing each index against ``specialize_indices`` and
    ``do_not_specialize_indices``, collecting the result into an
    ``OrderedDict`` keyed by parameter name, and then walking the whole
    signature a second time to read the arguments back out in order. Every input
    to that walk is fixed when the kernel is constructed except ``len(args)``,
    and every FlagGems pointwise wrapper passes all of its parameters
    positionally.

    Returns ``None`` for the call shapes this cannot express -- a launch that
    passes fewer arguments than the kernel has parameters, or a kernel whose
    constexpr values come from an ``Autotuner``/``Heuristics`` -- which sends
    the caller back to the stock ``run``.
    """
    if n_args != len(entry.jit_function.params) or entry._has_flagtune_tuner:
        return None
    specialized = frozenset(entry.specialize_indices)
    plain = frozenset(entry.do_not_specialize_indices)
    spec_idx, dns_idx, const_idx = [], [], []
    for i in range(n_args):
        if i in specialized:
            spec_idx.append(i)
        elif i in plain:
            dns_idx.append(i)
        else:
            const_idx.append(i)
    return (tuple(spec_idx), tuple(dns_idx), tuple(const_idx))


def _cached_libentry_key(entry, key_for_entry):
    """The entry's fast key function, built on first use and kept on the entry."""
    key_fn = entry.__dict__.get("_torch_fl_key_fn")
    if key_fn is None:
        key_fn = entry.__dict__["_torch_fl_key_fn"] = key_for_entry(entry)
    return key_fn


def _memoized_type_promotion(original):
    """Reuse promotion results without conflating scalar rank or default dtype.

    Torch gives zero-dimensional tensors different promotion precedence from
    dimensioned tensors. Python scalar types and the current default dtype also
    affect the result; equal values such as True, 1 and 1.0 must remain distinct.
    Unhashable arguments retain the original function's behavior.
    """
    cache = {}

    def type_promotion(*args, type_promotion):
        key = []
        for arg in args:
            key.append(
                (arg.dtype, arg.ndim == 0)
                if isinstance(arg, torch.Tensor)
                else (type(arg), arg)
            )
        try:
            lookup = (tuple(key), type_promotion, torch.get_default_dtype())
            hit = cache.get(lookup)
            if hit is None:
                hit = cache[lookup] = original(*args, type_promotion=type_promotion)
        except TypeError:
            return original(*args, type_promotion=type_promotion)
        return hit

    return type_promotion


def _flaggems_fast_run(key_for_entry, torch_device_fn, original_run):
    """``LibEntry.run`` for the case where it has everything it needs cached.

    See ``patch_pointwise_dispatch`` for why. The stock function
    rebuilds its per-argument classification and its argument list on every
    launch and then looks up a cache it will hit. This does the same lookup off
    a classification computed once, and hands the kernel the argument tuple it
    was given rather than a list reassembled from a name-keyed dict -- which for
    an all-positional launch is the same list.

    Anything unexpected -- a keyword launch, a callable grid, an entry whose
    launch constants came from a tuner, a cache miss -- falls through to the
    stock function, so the two cannot disagree about what a launch means.
    """

    def run(self, *args, **kwargs):
        if kwargs.keys() > _FLAGGEMS_RUN_KWARGS:
            return original_run(self, *args, **kwargs)

        plans = self.__dict__.get("_torch_fl_run_plans")
        if plans is None:
            plans = self.__dict__["_torch_fl_run_plans"] = {}
        n_args = len(args)
        plan = plans.get(n_args, _NO_PLAN)
        if plan is _NO_PLAN:
            plan = plans[n_args] = _libentry_run_plan(self, n_args)
        if plan is None:
            return original_run(self, *args, **kwargs)

        key_fn = _cached_libentry_key(self, key_for_entry)
        spec_idx, dns_idx, const_idx = plan
        entry_key = key_fn(
            [args[i] for i in spec_idx],
            [args[i] for i in dns_idx],
            [args[i] for i in const_idx],
        )
        device = torch_device_fn.current_device()
        # The stock function's two spellings of "which cache", kept because
        # ``current_device()`` answers "cpu" on a platform with a single
        # process-wide device.
        cache = self._cpu_cache if device == "cpu" else self.kernel_cache[device]
        cached = cache.get(entry_key)
        if cached is None:
            return original_run(self, *args, **kwargs)

        kernel, constexprs, tune_constexprs, heur_constexprs, launch_hooks = cached
        grid = kwargs["grid"]
        # A tuner-produced constant, a launch hook and a callable grid all need
        # the ``constexprs`` mapping this path does not rebuild, so any of them
        # means the stock function.
        if tune_constexprs or heur_constexprs or launch_hooks or callable(grid):
            return original_run(self, *args, **kwargs)

        kernel[(grid + (1, 1))[0:3]](*args)
        return kernel, constexprs

    return run


def _hygon_tensor_specialization(gems_device):
    """The vendor hook ``LibEntry.key`` folds into a tensor's key, or ``None``.

    ``spec_arg`` resolves ``triton.backends.hcu.compiler.HIPBackend.get_tensor_specialization``
    on every call on a hygon build and appends what it returns to that
    argument's key, which is what turns ``(dtype, aligned)`` into
    ``(dtype, aligned, 'S')``. The resolution is fixed for the life of the
    process, so it is done once here instead -- under the stock function's own
    conditions, the vendor test first so a non-hygon build keeps the
    two-element key.
    """
    if getattr(gems_device, "vendor_name", None) != "hygon":
        return None
    try:
        import triton

        if not hasattr(triton.backends, "hcu"):
            return None
        from triton.backends.hcu.compiler import HIPBackend
    except ImportError:
        return None
    specialization = getattr(HIPBackend, "get_tensor_specialization", None)
    return specialization if callable(specialization) else None


def patch_pointwise_dispatch(accelerator: str):
    """Take FlagGems' pointwise ops off the host and leave them on FlagGems.

    Every FlagGems elementwise op is a generated wrapper around a Triton kernel
    launched through ``flag_gems.utils.libentry.LibEntry``, and on a DCU bw1000
    the host side of that, not the kernel, is what a launch costs. In the
    Qwen-Image-2.1 denoise loop the FlagGems pointwise ops are 402 ms of the
    837 ms of exclusive host time across four steady steps -- ``where`` 196 ms
    over 520 calls, ``pow`` 61, ``tanh`` 58, ``rsqrt`` 56, ``silu`` 31 -- where
    the vendor build spends 34.94 us on each ``where`` against FlagGems'
    376.48. The kernels are not what differs: on that shape the FlagGems
    ``where`` kernel is 74.0 us of device time against 62.1 for the vendor's,
    while the enqueue is 262.4 us against 11.5 -- a launch cost, not a kernel
    cost.

    What is left is Python that recomputes per launch what the kernel fixed at
    construction: ``LibEntry.run`` re-derives which arguments are specialised,
    not-specialised and constexpr, builds a name-keyed ``OrderedDict`` from that
    walk and then walks the signature a second time to put the arguments back in
    order; ``LibEntry.key`` rebuilds two closures per call and re-resolves the
    vendor's tensor-specialisation hook per argument; and
    ``_descriptor_cache_key``, which ``key`` calls once per argument, re-imports
    a module each time. None of it can observe anything that changed between two
    launches of the same kernel on the same shapes.

    The saving is not confined to the ops named above, because every FlagGems
    launch in the loop goes through that same ``LibEntry``. The routed
    composites drop with the elementwise ops -- ``native_layer_norm`` 204.31 to
    169.88 us per call, ``tanh`` 226.51 to 212.80, ``rsqrt`` 216.23 to 202.41,
    ``silu`` 220.45 to 204.89 -- and the steady block's exclusive sum with them,
    0.837 s to 0.767 s.

    That sum is host time, and the steady step on this model is device-bound, so
    the four-step wall does not move with it (1.179 s to 1.178 s; the vendor
    build is at 1.098 s). This is a host-budget change, and the host budget is
    what bounds a launch-bound stretch of the run rather than one already
    hidden behind the device queue.

    Scoped to the builds and confs whose routed ops are FlagGems Python launches
    -- see ``_FLAGGEMS_LAUNCH_PATCH_BUILDS``; installed the way the patches
    around it are: by rebinding the functions in the FlagGems modules the C++
    bridge and the generated wrappers reach them through. No op changes route;
    the kernels, the grids and the cache keys stay the ones FlagGems computes.
    Upstream workaround rather than a design; remove it once FlagGems stops
    doing this work per launch. Best-effort, like its neighbours: an unpatched
    dispatch is slow here, not broken, so this must not be able to take down
    device init.
    """
    try:
        if not launch_patches_apply(accelerator):
            return

        import importlib

        libentry = importlib.import_module("flag_gems.utils.libentry")
        gems_runtime = importlib.import_module("flag_gems.runtime")
        gems_device = gems_runtime.device
        torch_device_fn = gems_runtime.torch_device_fn

        try:
            from triton.tools.tensor_descriptor import TensorDescriptor
        except ImportError:
            TensorDescriptor = None

        # Resolved once rather than per argument per launch; see the helper.
        tensor_specialization = _hygon_tensor_specialization(gems_device)

        descriptor_cache_key = _hoisted_descriptor_cache_key(TensorDescriptor)
        if not getattr(libentry._descriptor_cache_key, "_torch_fl_hoisted", False):
            descriptor_cache_key._torch_fl_hoisted = True
            descriptor_cache_key.__wrapped__ = libentry._descriptor_cache_key
            libentry._descriptor_cache_key = descriptor_cache_key

        def key_for_entry(entry):
            return _fast_libentry_key(
                entry, descriptor_cache_key, tensor_specialization
            )

        original_key = libentry.LibEntry.key
        if not getattr(original_key, "_torch_fl_fast", False):

            def key(self, spec_args, dns_args, const_args):
                return _cached_libentry_key(self, key_for_entry)(
                    spec_args, dns_args, const_args
                )

            key._torch_fl_fast = True
            key.__wrapped__ = original_key
            libentry.LibEntry.key = key

        # Reuse the complete promotion signature; every pointwise launch asks
        # for it, including native-host specialization misses.
        pointwise = importlib.import_module("flag_gems.utils.pointwise_dynamic")
        original_promotion = pointwise.type_promotion
        if not getattr(original_promotion, "_torch_fl_memo", False):
            memoized = _memoized_type_promotion(original_promotion)
            memoized._torch_fl_memo = True
            memoized.__wrapped__ = original_promotion
            pointwise.type_promotion = memoized

        original_run = libentry.LibEntry.run
        if not getattr(original_run, "_torch_fl_fast", False):
            fast_run = _flaggems_fast_run(key_for_entry, torch_device_fn, original_run)
            fast_run._torch_fl_fast = True
            fast_run.__wrapped__ = original_run
            libentry.LibEntry.run = fast_run

        # ``where_self_out`` asks for this through the ``torch`` module object,
        # so rebinding it there is what reaches the call site.
        if not getattr(torch.broadcast_shapes, "_torch_fl_memoized", False):
            memo = _memoized_broadcast_shapes(torch.broadcast_shapes)
            memo.__wrapped__ = torch.broadcast_shapes
            memo._torch_fl_memoized = True
            torch.broadcast_shapes = memo
    except Exception:
        pass
