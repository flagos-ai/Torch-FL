"""``torch.backends`` entry point for processes that import torch first.

Registered in pyproject.toml under the ``torch.backends`` group, which is
PyTorch's official out-of-tree device plugin mechanism (RFC
https://github.com/pytorch/pytorch/issues/122468). ``torch/__init__.py`` calls
every entry point in that group at the end of its own import.

This exists for processes that torch_fl does not control the startup of, when
the installed build supports torch-first import. The motivating case is
Inductor's parallel compile workers: ``compile_threads > 1``
spawns ``python -m torch._inductor.compile_worker`` with
``worker_start_method="subprocess"``, a fresh interpreter that imports ``torch``
and ``triton`` but never ``torch_fl``. torch_fl is what points
``torch.cuda.is_available`` at the flagos device count, and Triton's nvidia/amd
drivers gate ``is_active()`` on exactly that probe, so an un-initialized worker
finds no active backend and every cold Triton compile dies in
``set_driver_to_gpu()`` with "Could not find an active GPU backend". The parent
process is unaffected, which is why this only shows up on cache misses.

Keeping the worker's environment identical to the parent's is required. The
switch ``TORCH_DEVICE_BACKEND_AUTOLOAD=0`` disables every backend entry point,
including this one; an opt-out process must import torch_fl explicitly.

This hook cannot select a vendor libtorch core before PyTorch imports itself.
Builds that require a private core overlay must import torch_fl first, or start
with the correctly preloaded vendor core. MUSA also disables autoload before its
own torch import to keep torch_musa from claiming PrivateUse1; its FlagTree
compile workers are pinned to one thread because MThreads cannot initialize in
a worker. These restrictions are documented in docs/architecture/startup-lifecycle.md.
"""


def init() -> None:
    """Import torch_fl for its device-registration side effects.

    Idempotent: Python caches the module, so a process that already imported
    torch_fl (the normal ``import torch; import torch_fl`` order) does no extra
    work here.
    """
    import torch_fl  # noqa: F401
