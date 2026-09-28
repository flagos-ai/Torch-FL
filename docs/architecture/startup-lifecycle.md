# Torch-FL startup lifecycle

`import torch_fl` claims PyTorch's PrivateUse1 device. This process-wide action
must happen after selecting and preloading the vendor libtorch, but before loading
`torch_fl._C`. On vendors with a private libtorch overlay, import `torch_fl`
before `torch`; reversing that order is unsupported and fails with a diagnostic.

The import pipeline has five ordered phases:

| Phase | Work | Can be deferred? |
|---|---|---|
| `_phase_conf` | Choose the routing file and startup profile | No; precedes PyTorch import |
| `_phase_preload` | Select vendor libtorch and preload CUDA assets | No; precedes PyTorch import |
| `_phase_claim` | Import PyTorch, load the native extension, claim PrivateUse1 | No |
| `_phase_vendor_compat` | Install vendor shims and FlagGems codegen preparation | No |
| `_phase_ecosystem` | Set up FlagGems operators, CUDA alias and the FlagOS distributed backend | No |

FlagTree, FlagGems and FlagCX are required parts of a supported Torch-FL
environment. The startup profile does not make them optional, and the minimal
profile still performs FlagGems and distributed setup. It only controls the
framework compatibility hooks described below. A vendor-specific FlagTree wheel
provides the `triton` import package, and a vendor-specific FlagCX wheel is
required for the intended distributed path; see the
[wheel preflight](../reference/compatibility.md) and
[FlagCX architecture](distributed-flagcx.md) for environment compatibility.
FlagCX itself is loaded when a process group is created, after the `flagos`
distributed backend has been registered during import.

By default (`FLAGOS_STARTUP_PROFILE=full`), import activates the same framework
hooks as before: Apex, DDP, DataParallel, `torch.compile`, flex attention and the
BPU compile backend on BPU builds. Each hook reports an independent diagnostic
if it fails; unrelated hooks continue to activate.
`FLAGOS_STARTUP_PROFILE=minimal` skips
those hooks, so a caller can opt in after the mandatory bootstrap:

```python
import torch_fl

torch_fl.activate_optional_integrations("ddp", "compile")
print(torch_fl.optional_integration_status())
```

`activate_optional_integrations` accepts `apex`, `ddp`, `parallel_comm`,
`dataparallel`, `data_parallel`, `compile`, `flex_attention`, and `bpu`. With no
names it activates all of them. Dependencies between hooks are resolved first:
DataParallel needs `parallel_comm`, and flex attention needs `compile`. A second
activation does not patch PyTorch twice. An explicitly requested hook raises if
it fails; `strict=False` reports the failure on stderr and continues. A failed hook is not retried in
the same process because a third-party installation might have partially
modified global PyTorch state. Status values are `inactive`, `active`, or a
`failed: ...` reason. An optional package that is not installed remains
`inactive` and can be activated after installation.

Set the startup profile before importing `torch_fl`. Changing the environment
after import does not undo process-wide patches or replay the bootstrap.
