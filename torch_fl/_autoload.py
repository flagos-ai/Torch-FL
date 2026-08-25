"""``torch.backends`` entry point so a bare ``import torch`` initializes flagos.

Registered in pyproject.toml under the ``torch.backends`` group, which is
PyTorch's official out-of-tree device plugin mechanism (RFC
https://github.com/pytorch/pytorch/issues/122468). ``torch/__init__.py`` calls
every entry point in that group at the end of its own import.

This exists for processes that torch_fl does not control the startup of. The
motivating case is Inductor's parallel compile workers: ``compile_threads > 1``
spawns ``python -m torch._inductor.compile_worker`` with
``worker_start_method="subprocess"``, a fresh interpreter that imports ``torch``
and ``triton`` but never ``torch_fl``. torch_fl is what points
``torch.cuda.is_available`` at the flagos device count, and Triton's nvidia/amd
drivers gate ``is_active()`` on exactly that probe, so an un-initialized worker
finds no active backend and every cold Triton compile dies in
``set_driver_to_gpu()`` with "Could not find an active GPU backend". The parent
process is unaffected, which is why this only shows up on cache misses.

Keeping the worker's environment identical to the parent's is the fix; capping
``compile_threads`` to 1 would only avoid the worker.

MUSA is the one build where this does not fire: ``_disable_vendor_backend_autoload()``
sets ``TORCH_DEVICE_BACKEND_AUTOLOAD=0`` there to stop ``torch_musa`` from
claiming the PrivateUse1 key before flagos, and that switch is all-or-nothing --
it disables our entry point along with the vendor's. MUSA instead relies on
``_patch_vendor_flagtree_compile_workers()`` pinning ``compile_threads = 1``,
which it needs regardless because the MThreads driver is unsafe to initialize in
a worker at all.
"""


def init() -> None:
    """Import torch_fl for its device-registration side effects.

    Idempotent: Python caches the module, so a process that already imported
    torch_fl (the normal ``import torch; import torch_fl`` order) does no extra
    work here.
    """
    import torch_fl  # noqa: F401
