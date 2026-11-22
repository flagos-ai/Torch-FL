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

"""PyTorch DDP and DataParallel adapters for PrivateUse1 models."""

import functools
import sys


def install_ddp():
    """Patch DDP to transparently support flagos (privateuseone) models.

    PyTorch's C++ Reducer has CUDA-specific assertions that fail for
    privateuseone tensors. This patch detects when the wrapped module lives on
    a flagos device and transparently:
      1. Forces ``python_reducer`` mode (bypasses the C++ Reducer).
      2. Replaces default accum-grad hooks with flagos-compatible ones that
         call ``dist.all_reduce`` (routed through ProcessGroupFlagOS).

    Users call standard DDP — no ``flagos_dist.DistributedDataParallel`` needed:
        model = torch.nn.parallel.DistributedDataParallel(model)
    """
    import torch.distributed as _dist
    from torch.nn.parallel import DistributedDataParallel as _DDP

    _orig_init = _DDP.__init__

    @functools.wraps(_orig_init)
    def _patched_init(self, module, **kwargs):
        # torch_fl renames PrivateUse1 to "flagos", so parameter tensors report
        # device.type == "flagos" (not the raw "privateuseone").
        device_types = {p.device.type for p in module.parameters()}
        if not device_types & {"flagos", "privateuseone"}:
            return _orig_init(self, module, **kwargs)

        # Force python_reducer to avoid C++ Reducer CUDA assertions
        import torch._dynamo.utils

        _orig_mode = torch._dynamo.utils.get_optimize_ddp_mode
        torch._dynamo.utils.get_optimize_ddp_mode = lambda: "python_reducer"
        try:
            kwargs.setdefault("gradient_as_bucket_view", True)
            kwargs.setdefault("broadcast_buffers", False)
            _orig_init(self, module, **kwargs)
        finally:
            torch._dynamo.utils.get_optimize_ddp_mode = _orig_mode

        # Replace DDP's python_reducer accum hooks. The stock hooks use
        # torch.distributed._functional_collectives (torch.ops._c10d_functional.*),
        # whose dispatcher path is not registered for privateuseone. We instead
        # go through dist.all_reduce on the group, which routes to
        # ProcessGroupFlagOS and does the privateuseone->cuda view conversion.
        # Mirrors DDP.compiled_accum_grad_hook, including _comm_hooks support.
        for h in self._accum_grad_hooks:
            h.remove()
        self._accum_grad_hooks.clear()

        def _accum_grad_hook(param, *, ddp_model=self):
            if not ddp_model.require_backward_grad_sync:
                return
            if param.grad is None:
                return
            pg = ddp_model.process_group
            if ddp_model._comm_hooks:
                for hook, state in ddp_model._comm_hooks:
                    hook(state, (param.grad, param))
            else:
                param.grad.div_(pg.size())
                _dist.all_reduce(param.grad, op=_dist.ReduceOp.SUM, group=pg)

        for param in self._module_parameters:
            if param.requires_grad:
                self._accum_grad_hooks.append(
                    param.register_post_accumulate_grad_hook(
                        functools.partial(_accum_grad_hook, ddp_model=self)
                    )
                )

    _DDP.__init__ = _patched_init


# ---------------------------------------------------------------------------
# DataParallel auto-patch: torch.nn.parallel.DataParallel and its comm layer
# ---------------------------------------------------------------------------

# Both spellings name the same device: the claim phase renames PrivateUse1 to
# "flagos", so tensors created afterwards report "flagos", while
# "privateuseone" is the raw name a tensor can still carry from before the
# rename (or from a wheel whose rename never ran).
_FLAGOS_DEVICE_TYPES = ("flagos", "privateuseone")


def _flagos_device_type_of_tensors(tensors):
    """Device type of the first flagos device among ``tensors``, else None."""
    for tensor in tensors:
        if tensor.device.type in _FLAGOS_DEVICE_TYPES:
            return tensor.device.type
    return None


def _flagos_module_device_type(module):
    """Device type a module is placed on if it is a flagos one, else None.

    Parameters and buffers both: DataParallel's own device guard walks the two
    together, and a module can hold buffers with no parameters.
    """
    for group in (module.parameters(), module.buffers()):
        device_type = _flagos_device_type_of_tensors(group)
        if device_type is not None:
            return device_type
    return None


def install_comm():
    """Hand ``torch.nn.parallel.comm``'s device moves to the flagos ops.

    ``comm.scatter``, ``comm.gather`` and ``comm.broadcast_coalesced`` -- and
    the ``_out`` forms DataParallel reaches through their ``out=`` argument --
    are thin validators over seven CUDA-only ops on ``torch._C`` (``_scatter``,
    ``_scatter_out``, ``_gather``, ``_gather_out``, ``_broadcast``,
    ``_broadcast_out``, ``_broadcast_coalesced``). Each reads the caller's
    device list as a list of *CUDA* indices whatever the tensors are, so on a
    flagos build it either mislabels the result or refuses outright -- measured
    on PPU:

        torch._C._scatter(flagos:0 tensor, [0, 1], ...) -> [flagos:0, cuda:1]
        torch._C._gather([flagos:0, flagos:1], 0, 0)    -> RuntimeError:
            "Expected all input tensors to be CUDA tensors, but tensor at
             index 0 has device flagos:0"

    The flagos implementations live in the extension
    (csrc/bindings/dataparallel.cc) and are published by rebinding those
    seven attributes on ``torch._C``, which is how torch_npu reaches the same
    symbols (its ``initCommMethods()``). They read and write flagos tensors the
    way the CUDA ones read and write CUDA ones, and each delegates to the
    original it replaced as soon as no flagos tensor is involved, so CUDA and
    CPU keep the stock code path. Nothing in comm's Python is replaced: the
    validation, the ``_handle_complex`` handling and the device index
    resolution DataParallel already relies on all stay torch's.

    Rebinding is idempotent on the extension side, so calling this more than
    once is harmless.
    """
    from torch_fl import _C

    _C._init_dataparallel_comm()


def install_dataparallel():
    """Make ``torch.nn.DataParallel`` place its replicas on flagos devices.

    Three device decisions in DataParallel are CUDA-only, and a flagos-placed
    module trips all three:

      * the device *type* comes from ``torch._utils._get_available_device_type()``,
        which answers "cuda" before it ever asks privateuse1. So on PPU and MetaX
        (where ``torch.cuda.is_available()`` is shimmed True) a module on
        ``flagos:0`` is given ``src_device_obj = torch.device("cuda", 0)`` and
        then fails DataParallel's own guard in forward with "module must have its
        parameters and buffers on device cuda:0 (device_ids[0]) but found one of
        them on device: flagos:0";
      * the default device list is ``_get_all_device_indices()``, i.e. torch.cuda's
        device count, which on a stock +cpu torch is 0;
      * ``_check_balance`` is CUDA-only, and on MetaX it is also what moves the
        current device: ``_query_metax_device_properties`` calls ``mcSetDevice``
        per device and never restores, so after the probe loop every later
        operation meant for device 0 runs on the last device probed.

    A flagos-placed module therefore builds its own state: devices counted and
    typed from the flagos device module, no balance probe. Everything after
    construction is DataParallel's own code -- scatter, replicate,
    parallel_apply, gather -- and works once ``torch.nn.parallel.comm`` has
    flagos branches, which is what ``install_comm`` installs.

    A module that is not on a flagos device goes to the original ``__init__``
    unchanged, so CUDA and CPU behavior is untouched.
    """
    import torch
    from torch._utils import _get_device_index
    from torch.nn.parallel import DataParallel as _DataParallel

    _orig_init = _DataParallel.__init__
    _orig_scatter = _DataParallel.scatter

    @functools.wraps(_orig_init)
    def _patched_init(self, module, device_ids=None, output_device=None, dim=0):
        device_type = _flagos_module_device_type(module)
        if device_type is None:
            return _orig_init(self, module, device_ids, output_device, dim)

        # Module.__init__ has to run before anything is assigned to self: it
        # installs the __setattr__ machinery that every `self.x = ...` below goes
        # through. The original __init__ is not called, so this replaces the
        # super().__init__() at its top.
        torch.nn.Module.__init__(self)
        torch._C._log_api_usage_once("torch.nn.parallel.DataParallel")

        if device_ids is None:
            # The flagos device module rather than torch.cuda: whatever
            # torch.cuda reports on this host is unrelated to the devices the
            # module's parameters live on. torch.flagos is the fallback for the
            # raw "privateuseone" spelling, whose own attribute does not exist.
            device_ids = range(getattr(torch, device_type, torch.flagos).device_count())
        if len(device_ids) == 0:
            raise RuntimeError("no available devices were found")
        if output_device is None:
            output_device = device_ids[0]

        self.dim = dim
        self.module = module
        self.device_ids = [_get_device_index(x, True) for x in device_ids]
        self.output_device = _get_device_index(output_device, True)
        self.src_device_obj = torch.device(device_type, self.device_ids[0])
        # The marker DataParallel.scatter reads below to decide whether the
        # device ids it hands to comm.scatter are flagos ones.
        self._flagos_device_type = device_type

        # No _check_balance: it is the CUDA memory/cores warning, and on MetaX
        # running it is a side effect on the current device.
        if len(self.device_ids) == 1:
            self.module.to(self.src_device_obj)

    @functools.wraps(_orig_scatter)
    def _patched_scatter(self, inputs, kwargs, device_ids):
        if getattr(self, "_flagos_device_type", None) is None:
            return _orig_scatter(self, inputs, kwargs, device_ids)
        # Tells torch._C._scatter that the integer device ids it is about to
        # receive name flagos devices. Only load-bearing for a CPU input, which
        # carries no device type of its own; see SetScatterScope in
        # csrc/bindings/dataparallel.h for why the ids alone are ambiguous
        # here.
        from torch_fl import _C

        previous = _C._set_scatter_scope(True)
        try:
            return _orig_scatter(self, inputs, kwargs, device_ids)
        finally:
            _C._set_scatter_scope(previous)

    _DataParallel.__init__ = _patched_init
    _DataParallel.scatter = _patched_scatter


def install_data_parallel():
    """Give the functional ``torch.nn.parallel.data_parallel()`` flagos support.

    It is the class's forward path with the device resolution restated inline,
    so it fails a flagos-placed module the same way (and additionally raises
    "device type could not be determined" on a build where nothing reports
    availability). For a flagos-placed module it is delegated to the patched
    class instead of restating torch's logic a second time, which keeps the two
    from drifting. Two consequences of that delegation, both deliberate:
    ``_check_balance`` is skipped where the class would also skip it, and a
    single-device call now ends with ``module.to(device_ids[0])``, which the
    class does in ``__init__`` and the function does not -- a no-op for a module
    already on the device it is being run on.
    """
    import torch
    from torch.nn.parallel import DataParallel as _DataParallel
    from torch.nn.parallel.data_parallel import data_parallel as _data_parallel

    @functools.wraps(_data_parallel)
    def _patched_data_parallel(
        module, inputs, device_ids=None, output_device=None, dim=0, module_kwargs=None
    ):
        if _flagos_module_device_type(module) is None:
            return _data_parallel(
                module, inputs, device_ids, output_device, dim, module_kwargs
            )
        if not isinstance(inputs, tuple):
            inputs = (inputs,) if inputs is not None else ()
        return _DataParallel(module, device_ids, output_device, dim)(
            *inputs, **(module_kwargs or {})
        )

    # Both spellings of the name: torch.nn.parallel re-exports the function, and
    # the module that defines it is still reachable by path.
    torch.nn.parallel.data_parallel = _patched_data_parallel
    sys.modules[
        "torch.nn.parallel.data_parallel"
    ].data_parallel = _patched_data_parallel
