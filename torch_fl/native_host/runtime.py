# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Shared host adapters, compiled along with the wheel-derived host source."""

import math
import operator

import torch
from torch_fl import _C
from fg_aot import _launcher

_resolve = None
_kernel_factory = None
_pointwise_helpers = None
_imports = None


def import_attribute(module, name):
    return getattr(_imports[module], name)


class Logger:
    def __init__(self, original):
        self.original = original

    def debug(self, message, *args, **kwargs):
        logger = self.original
        if logger.disabled or logger.manager._disable >= 10:
            return
        current = logger
        while current is not None:
            if current.level:
                if current.level <= 10:
                    logger.debug(message, *args, **kwargs)
                return
            current = current.parent
        logger.debug(message, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.original, name)


class DeviceScope:
    def __init__(self, device):
        self.index = device.index if isinstance(device, torch.device) else device
        self.previous = None

    def __enter__(self):
        if self.index is not None and self.index >= 0:
            self.previous = _C._exchangeDevice(self.index)
        return self

    def __exit__(self, kind, value, traceback):
        if self.previous is not None and _C._get_device() != self.previous:
            _C._set_device(self.previous)
        return False


class DeviceAPI:
    device = DeviceScope
    current_device = staticmethod(_C._get_device)
    device_count = staticmethod(_C._get_device_count)

    def __init__(self, original):
        self.original = original

    def is_current_stream_capturing(self):
        return _launcher.is_capturing(_C._get_device())

    def __getattr__(self, name):
        return getattr(self.original, name)


def argument_key(value):
    if hasattr(value, "data_ptr"):
        return (
            value.dtype,
            value.data_ptr() % 16 == 0,
            tuple(value.shape),
            tuple(value.stride()),
        )
    if hasattr(value, "base") and hasattr(value, "block_shape"):
        return (
            argument_key(value.base),
            tuple(value.shape),
            tuple(value.strides),
            tuple(value.block_shape),
            value.padding,
        )
    if isinstance(value, (tuple, list)):
        return tuple(argument_key(item) for item in value)
    return type(value), value


def _triton_dtype_constant(value):
    prefix = "__flagos_triton_dtype__:"
    if type(value) is not str or not value.startswith(prefix):
        return value
    from triton import language

    name = value[len(prefix) :]
    dtype = getattr(language, name, None)
    if not isinstance(dtype, language.dtype):
        raise ValueError("unknown Triton dtype in Native Host launch: " + name)
    return dtype


class BoundKernel:
    def __init__(self, entry, grid):
        self.entry = entry
        self.grid = grid

    def __call__(self, *args, **kwargs):
        return self.entry.run(*args, grid=self.grid, **kwargs)


class KernelEntry:
    """Cache device specializations; execute host control flow on every call."""

    def __init__(
        self,
        original,
        entry,
        reuse_runtime_floats=False,
        compile_before_enqueue=False,
        permit_single_config=False,
        eager_config=None,
        power_of_two_heuristics=(),
    ):
        self.original = original
        self.entry = entry
        self.returns_constants = original is entry
        self.arg_names = tuple(entry.arg_names)
        self.params = tuple(entry.jit_function.params)
        self.defaults = {
            param.name: param.default for param in self.params if param.has_default
        }
        self.cache = {}
        self.compile_before_enqueue = compile_before_enqueue
        self.permit_single_config = permit_single_config
        self.eager_config = eager_config
        self.power_of_two_heuristics = power_of_two_heuristics
        self.native_cache = _launcher.KernelCache(
            self.arg_names, self.defaults, self.returns_constants, reuse_runtime_floats
        )

    def __getitem__(self, grid):
        return BoundKernel(self, grid)

    def __getattr__(self, name):
        return getattr(self.original, name)

    def warmup(self, *args, **kwargs):
        return self.original.warmup(*args, **kwargs)

    def restore_constexpr_types(self, args, kwargs):
        constexpr = {p.name for p in self.params if p.is_constexpr}
        values = tuple(
            _triton_dtype_constant(value)
            if i < len(self.params) and self.params[i].is_constexpr
            else value
            for i, value in enumerate(args)
        )
        options = {
            name: _triton_dtype_constant(value) if name in constexpr else value
            for name, value in kwargs.items()
        }
        return values, options

    def autotuner_config(self, wrapper, args, options):
        from flag_gems.flagtune import cost_model
        from flag_gems.runtime import TuningMode
        from flag_gems.utils.libentry import LibTuner, LibTunerRunMode

        fixed = self.eager_config
        lib_tuner = isinstance(wrapper, LibTuner)
        if (lib_tuner and not self.permit_single_config) or (
            not lib_tuner and fixed is None
        ):
            return None
        if lib_tuner and self.entry._has_flagtune_tuner:
            self.entry._apply_flagtune()
        if (
            LibTunerRunMode(getattr(wrapper, "_run_mode", LibTunerRunMode.NORMAL))
            is not LibTunerRunMode.NORMAL
            or getattr(wrapper, "shared_config_pre_hook", None) is not None
        ):
            return None
        if fixed is not None and (
            getattr(wrapper, "user_defined_pre_hook", False)
            or getattr(wrapper, "user_defined_post_hook", False)
            or getattr(wrapper, "reset_to_zero", ())
            or getattr(wrapper, "restore_value", ())
            or getattr(wrapper, "early_config_prune", None) is not None
            or getattr(wrapper, "perf_model", None) is not None
        ):
            return None
        configs = wrapper.configs
        cost_model_mode = lib_tuner and wrapper._flagtune_mode is TuningMode.COST_MODEL
        if cost_model_mode:
            # Use the published tuner's legal runtime candidate domain.
            if getattr(wrapper, "early_config_prune", None) is not None:
                return None
            try:
                configs = cost_model.runtime_candidates(
                    wrapper, wrapper._flagtune_op_name, options
                )
            except RuntimeError:
                return None
        if fixed is not None:
            return next(
                (
                    config
                    for config in configs
                    if config.pre_hook is None and config.all_kwargs() == fixed
                ),
                None,
            )
        if cost_model_mode:
            # The cost model skips prediction for a singleton. Preserve
            # existing tuning when it could select another configuration.
            if len(configs) != 1:
                return None
            named = {**dict(zip(wrapper.arg_names, args)), **options}
            key = wrapper.get_key(
                {k: named[k] for k in wrapper.arg_names if k in named}
            )
            if key in wrapper.cache:
                cached = wrapper.cache[key]
                if cached.all_kwargs() != configs[0].all_kwargs():
                    return None
        if len(configs) != 1 or configs[0].pre_hook is not None:
            return None
        return configs[0]

    def compile_metadata(self, args, kwargs, grid):
        from triton.runtime import Autotuner, Heuristics

        wrapper = self.entry.fn
        options = dict(kwargs, grid=grid)
        tuned, heuristics = {}, {}
        if isinstance(wrapper, Autotuner):
            config = self.autotuner_config(wrapper, args, options)
            if config is None:
                return None
            config_options = config.all_kwargs()
            if options.keys() & config_options.keys():
                return None
            options.update(config_options)
            tuned = config.kwargs
            wrapper = wrapper.fn
        while type(wrapper) is Heuristics:
            named = dict(zip(wrapper.arg_names, args))
            for name, heuristic in wrapper.values.items():
                value = heuristic({**named, **options})
                if name in self.power_of_two_heuristics and value > 0:
                    value = 1 << (operator.index(value) - 1).bit_length()
                options[name] = heuristics[name] = value
            wrapper = wrapper.fn
        if wrapper is self.entry.jit_function:
            return options, tuned, heuristics
        return None

    def prepare_native(self, *args, grid, **kwargs):
        """Compile a specialization with known metadata before C++ submits it."""
        if not self.compile_before_enqueue:
            return False
        kwargs.setdefault("warmup", False)
        if kwargs["warmup"]:
            return False
        device = _C._get_device()
        with self.entry.lock:
            key = self.native_cache.missing_key(args, kwargs, device)
            if key is None:
                return True
            values, options = self.restore_constexpr_types(args, kwargs)
            metadata = self.compile_metadata(values, options, grid)
            if metadata is None:
                return False
            options, tuned, heuristics = metadata
            if callable(grid):
                # Some published compilers serialize the grid as an option.
                # Resolve it with the selected launch metadata before warmup.
                bound = {
                    **self.defaults,
                    **dict(zip(self.arg_names, values)),
                    **options,
                }
                options["grid"] = tuple(operator.index(size) for size in grid(bound))
            options["warmup"] = True
            kernel = self.entry.jit_function.run(*values, **options)
            constants = {
                p.name: p.default
                for p in self.params
                if p.is_constexpr and p.has_default
            }
            constants.update(tuned)
            constants.update(heuristics)
            for name in ("num_warps", "num_stages", "num_ctas"):
                if name in options:
                    constants[name] = options[name]
            variant = (
                kernel,
                kernel.run,
                kernel.function,
                kernel.packed_metadata,
                constants,
                tuned,
                heuristics,
                (),
            )
            self.native_cache.prepare(key, args, kwargs, variant)
        return True

    def run(self, *args, grid, **kwargs):
        kwargs.setdefault("warmup", False)
        if kwargs["warmup"]:
            values, options = self.restore_constexpr_types(args, kwargs)
            return self.original.run(*values, grid=grid, **options)
        device = _C._get_device()
        native = self.native_cache.run(args, kwargs, grid, device)
        if native is not None and not isinstance(native, bytes):
            return native
        key = (
            device,
            tuple(argument_key(a) for a in args),
            tuple((k, argument_key(v)) for k, v in sorted(kwargs.items())),
        )
        variant = self.cache.get(key)
        values, options = self.restore_constexpr_types(args, kwargs)
        if variant is None:
            # Only a specialization miss enters the original compiler/tuner.
            kernel, constants = self.entry.run(*values, grid=grid, **options)
            records = self.entry.kernel_cache[device].values()
            record = next(
                item for item in records if item[0] is kernel and item[1] is constants
            )
            hooks = tuple((_resolve(hook), values) for hook, values in record[4])
            variant = (
                kernel,
                kernel.run,
                kernel.function,
                kernel.packed_metadata,
                constants,
                record[2],
                record[3],
                hooks,
            )
            self.cache[key] = variant
            if native is not None:
                self.native_cache.prepare(native, args, kwargs, variant)
            return (kernel, constants) if self.returns_constants else kernel
        if native is not None:
            self.native_cache.prepare(native, args, kwargs, variant)
        kernel, launch, function, packed, constants, tuned, heuristics, hooks = variant
        named = dict(zip(self.arg_names, values))
        named.update(options)
        if callable(grid):
            grid = grid({**named, **constants})
        grid = tuple(grid) + (1, 1)
        for hook, values in hooks:
            hook({**named, **values})
        values = []
        for param in self.params:
            if param.name in named:
                values.append(named[param.name])
            elif param.name in tuned:
                values.append(tuned[param.name])
            elif param.name in heuristics:
                values.append(heuristics[param.name])
            elif param.name in constants:
                values.append(constants[param.name])
            elif param.name in self.defaults:
                values.append(self.defaults[param.name])
            else:
                raise RuntimeError(f"missing kernel argument: {param.name}")
        launch(
            grid[0],
            grid[1],
            grid[2],
            _launcher.current_stream(device),
            function,
            packed,
            None,
            None,
            None,
            *values,
        )
        return (kernel, constants) if self.returns_constants else kernel


class EnflamePointwiseOverload:
    """Rank-independent host adapter for the published GCU300 pointwise ABI.

    Ordinary tensors use the C++ preparation and schedule. The compiled vendor
    preparation and this Cython schedule preserve the general fallback. Only
    the unchanged device generator runs for a new rank; generated Python host
    wrappers are never loaded at runtime.
    """

    def __init__(self, owner, ndim, kernels):
        self.owner, self.ndim, self.kernels = owner, ndim, kernels
        self.helpers = _imports[owner._native_pointwise_helpers]
        if all(hasattr(kernel, "native_cache") for kernel in kernels):
            self.native = _launcher.PointwiseSchedule(
                owner,
                ndim,
                kernels,
                (
                    self.helpers.heuristics_for_tile_size,
                    self.helpers.heuristics_for_num_warps,
                    _imports["flag_gems.utils.shape_utils"].stride_order,
                    self.helpers.heuristics_for_tile_size_with_mmu_constraint,
                    self.helpers.ENFLAME_GCU300_4SIPS,
                    self.helpers.MMU_LIMIT,
                ),
                (_C._exchangeDevice, _C._set_device),
                True,
            )

    def __call__(self, *args, **kwargs):
        owner, ndim = self.owner, self.ndim
        schema, config = owner.fx, owner.config
        outputs = [kwargs[f"out{i}"] for i in range(schema.num_output_tensors())]
        inputs = [value for i, value in enumerate(args) if schema.is_tensor(i)]
        tensors = inputs + outputs
        shape, tasks = outputs[0].shape, outputs[0].numel()
        result = outputs[0] if len(outputs) == 1 else tuple(outputs)
        if any(value.shape != shape for value in tensors):
            raise ValueError("pointwise overload expects broadcasted inputs")
        if ndim and not tasks:
            return result
        parameters = list(args) + outputs
        options = {"ENABLE_I64": kwargs.get("enable_i64", False)}
        ctas, warps, no_dma = 1, 1, False
        if ndim:
            # This is the published WrapperGenerator's DMA selection predicate.
            for value in tensors:
                strides = value.stride()
                no_dma |= any(s <= 0 for s in strides)
                positive = [s for s in strides if s != 0]
                no_dma |= not all(
                    max(a, b) % min(a, b) == 0 and a != b
                    for i, a in enumerate(positive)
                    for b in positive[i + 1 :]
                )
            factor = 8 // max(inputs[0].element_size(), outputs[0].element_size())
            if no_dma:
                factor = max(1, factor // 4)
            budget = config.max_tile_size * factor
            if config.prefer_1d_tile:
                tiles = self.helpers.heuristics_for_tile_size(budget, tasks)
            elif no_dma:
                tiles = self.helpers.heuristics_for_tile_size_with_mmu_constraint(
                    budget, outputs[0].element_size(), outputs[0].stride(), *shape
                )
            else:
                tiles = self.helpers.heuristics_for_tile_size(budget, *shape)
            tile_size = math.prod(tiles)
            count = (
                (tasks + tile_size - 1) // tile_size
                if config.prefer_1d_tile
                else math.prod((s + t - 1) // t for s, t in zip(shape, tiles))
            )
            ctas = min(config.max_grid_size[0], count)
            per_cta = (count + ctas - 1) // ctas
            warps = self.helpers.heuristics_for_num_warps(tile_size)
            for value in tensors:
                strides = value.stride()
                parameters.extend(strides)
                if config.prefer_block_pointer and not config.prefer_1d_tile:
                    parameters.extend(
                        _imports["flag_gems.utils.shape_utils"].stride_order(strides)
                        if ndim >= 2
                        else (0,)
                    )
            parameters.extend(shape)
            parameters.extend((tasks, count))
            options.update(tiles_per_cta=per_cta, one_tile_per_cta=per_cta == 1)
            if config.prefer_1d_tile:
                options["tile_size"] = tile_size
            else:
                options.update((f"tile_size{i}", tile) for i, tile in enumerate(tiles))
                options["stages"] = 2
        options["num_warps"] = warps
        kernel = (
            self.kernels[1] if no_dma and len(self.kernels) > 1 else self.kernels[0]
        )
        with DeviceScope(inputs[0].device.index):
            kernel.run(*parameters, grid=(ctas, 1, 1), **options)
        return result


class PointwiseOverload:
    """Rank-independent adapter for FlagGems' original pointwise kernel ABI."""

    def __init__(self, owner, ndim, kernel):
        self.owner = owner
        self.ndim = ndim
        self.kernel = kernel
        self.native = _launcher.PointwiseSchedule(
            owner,
            ndim,
            kernel,
            _pointwise_helpers,
            (_C._exchangeDevice, _C._set_device),
        )

    def __call__(self, *args, **kwargs):
        native = self.native(args, kwargs)
        if native is not None:
            return native
        owner = self.owner
        schema, config, ndim = owner.fx, owner.config, self.ndim
        outputs = [kwargs[f"out{i}"] for i in range(schema.num_output_tensors())]
        inputs = [item for i, item in enumerate(args) if schema.is_tensor(i)]
        tensors = inputs + outputs
        shape = outputs[0].shape
        if any(item.shape != shape for item in tensors):
            raise ValueError("pointwise overload expects broadcasted inputs")
        num_tasks = outputs[0].numel()
        result = outputs[0] if len(outputs) == 1 else tuple(outputs)
        if ndim and num_tasks == 0:
            return result
        parameters = list(args) + outputs
        options = {}
        if ndim:
            (
                tile_heuristic,
                warp_heuristic,
                stride_order,
                balanced,
                all_complex,
                major,
            ) = _pointwise_helpers
            hygon = getattr(owner, "_native_pointwise_hygon", False)
            max_tile = config.max_tile_size // (
                2 if not hygon and all_complex(schema) else 1
            )
            fill = (
                "fill_scalar" in owner._scalar_fn.__name__
                and major >= 9
                and (not hygon or config.prefer_1d_tile)
            )
            task = (num_tasks,) if config.prefer_1d_tile else shape
            tile_sizes = (
                (1024 if config.prefer_1d_tile and not hygon else 64,)
                if fill
                else tile_heuristic(max_tile, *task)
            )
            tile_size = math.prod(tile_sizes)
            num_tiles = (
                (num_tasks + tile_size - 1) // tile_size
                if config.prefer_1d_tile
                else math.prod((s + t - 1) // t for s, t in zip(shape, tile_sizes))
            )
            if not hygon and getattr(config, "balance_grid", False):
                num_ctas, tiles_per_cta = balanced(num_tiles, config.max_grid_size[0])
            else:
                num_ctas = (
                    num_tiles if fill else min(config.max_grid_size[0], num_tiles)
                )
                tiles_per_cta = (num_tiles + num_ctas - 1) // num_ctas
            num_warps = warp_heuristic(tile_size)
            for tensor in tensors:
                strides = tensor.stride()
                parameters.extend(strides)
                if config.prefer_block_pointer and not config.prefer_1d_tile:
                    parameters.extend(stride_order(strides) if ndim >= 2 else (0,))
            parameters.extend(shape)
            parameters.append(num_tasks)
            options.update(
                tiles_per_cta=tiles_per_cta, one_tile_per_cta=tiles_per_cta == 1
            )
            if config.prefer_1d_tile:
                options["tile_size"] = tile_size
            else:
                options.update(
                    (f"tile_size{i}", tile) for i, tile in enumerate(tile_sizes)
                )
        else:
            num_ctas, num_warps = 1, 1
        with DeviceScope(inputs[0].device.index):
            self.kernel.run(
                *parameters, grid=(num_ctas, 1, 1), num_warps=num_warps, **options
            )
        return result


def pointwise_instantiate(owner, ndim):
    key = (
        ndim,
        owner.config.prefer_block_pointer,
        getattr(owner.config, "balance_grid", False),
    )
    if hasattr(owner, "_native_pointwise_module"):
        key += (
            getattr(owner.config, "max_tile_size", None),
            tuple(getattr(owner.config, "max_grid_size", ())),
            getattr(owner.config, "prefer_1d_tile", False),
        )
    overload = owner.overloads.get(key)
    if overload is None:
        kernels = _kernel_factory(owner, ndim)
        adapter = (
            EnflamePointwiseOverload
            if isinstance(kernels, tuple)
            else PointwiseOverload
        )
        overload = adapter(owner, ndim, kernels)
        owner.overloads[key] = overload
    return overload


def pointwise_prepare(owner, args, kwargs, execute=False):
    if hasattr(owner, "_native_pointwise_module") and any(
        key not in {f"out{i}" for i in range(owner.fx.num_output_tensors())}
        for key in kwargs
    ):
        return None
    native = owner.__dict__.get("_native_spec")
    if native is None:
        native = _launcher.PointwiseSpec(
            owner.fx,
            _imports["flag_gems.utils.pointwise_dynamic"].type_promotion,
            owner.config,
        )
        owner._native_spec = native
    return native.prepare(args, kwargs, _C._get_device(), owner if execute else None)


def configure(resolve, kernel_factory, helpers, imports):
    global _resolve, _kernel_factory, _pointwise_helpers, _imports
    _resolve, _kernel_factory, _pointwise_helpers = resolve, kernel_factory, helpers
    _imports = imports
