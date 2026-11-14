# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Shared host adapters, compiled along with the wheel-derived host source."""

import math

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
        if self.previous is not None:
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


class BoundKernel:
    def __init__(self, entry, grid):
        self.entry = entry
        self.grid = grid

    def __call__(self, *args, **kwargs):
        return self.entry.run(*args, grid=self.grid, **kwargs)


class KernelEntry:
    """Cache device specializations; execute host control flow on every call."""

    def __init__(self, original, entry):
        self.original = original
        self.entry = entry
        self.returns_constants = original is entry
        self.arg_names = tuple(entry.arg_names)
        self.params = tuple(entry.jit_function.params)
        self.defaults = {
            param.name: param.default for param in self.params if param.has_default
        }
        self.cache = {}

    def __getitem__(self, grid):
        return BoundKernel(self, grid)

    def __getattr__(self, name):
        return getattr(self.original, name)

    def warmup(self, *args, **kwargs):
        return self.original.warmup(*args, **kwargs)

    def run(self, *args, grid, **kwargs):
        kwargs.setdefault("warmup", False)
        if kwargs["warmup"]:
            return self.original.run(*args, grid=grid, **kwargs)
        device = _C._get_device()
        key = (
            device,
            tuple(argument_key(a) for a in args),
            tuple((k, argument_key(v)) for k, v in sorted(kwargs.items())),
        )
        variant = self.cache.get(key)
        if variant is None:
            # Only a specialization miss enters the original compiler/tuner.
            kernel, constants = self.entry.run(*args, grid=grid, **kwargs)
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
            return (kernel, constants) if self.returns_constants else kernel
        kernel, launch, function, packed, constants, tuned, heuristics, hooks = variant
        named = dict(zip(self.arg_names, args))
        named.update(kwargs)
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


class PointwiseOverload:
    """Rank-independent adapter for FlagGems' original pointwise kernel ABI."""

    def __init__(self, owner, ndim, kernel):
        self.owner = owner
        self.ndim = ndim
        self.kernel = kernel

    def __call__(self, *args, **kwargs):
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
            max_tile = config.max_tile_size // (2 if all_complex(schema) else 1)
            fill = "fill_scalar" in owner._scalar_fn.__name__ and major >= 9
            task = (num_tasks,) if config.prefer_1d_tile else shape
            tile_sizes = (
                (1024 if config.prefer_1d_tile else 64,)
                if fill
                else tile_heuristic(max_tile, *task)
            )
            tile_size = math.prod(tile_sizes)
            num_tiles = (
                (num_tasks + tile_size - 1) // tile_size
                if config.prefer_1d_tile
                else math.prod((s + t - 1) // t for s, t in zip(shape, tile_sizes))
            )
            if config.balance_grid:
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
    key = (ndim, owner.config.prefer_block_pointer, owner.config.balance_grid)
    overload = owner.overloads.get(key)
    if overload is None:
        overload = PointwiseOverload(owner, ndim, _kernel_factory(owner, ndim))
        owner.overloads[key] = overload
    return overload


def configure(resolve, kernel_factory, helpers, imports):
    global _resolve, _kernel_factory, _pointwise_helpers, _imports
    _resolve, _kernel_factory, _pointwise_helpers = resolve, kernel_factory, helpers
    _imports = imports
