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

"""FlagGems autotuning compatibility and cache setup."""

import gc
import importlib
import os
import sys

import torch

from ..._platform import platform_for
from .launch import launch_patches_apply
from . import runtime


def align_flagtree_knobs(accelerator: str):
    """Give FlagGems back the FlagTree autotuning knob name 0.7.0 removed.

    flag_gems 5.4.0's small-sequence flash-attention backward
    (``flag_gems/ops/flash_attention_backward.py``,
    ``_disable_aabs_for_small_seqlen``) saves and restores a FlagTree knob it
    calls ``adjust_block_size``::

        previous = _autotuning_knobs.adjust_block_size          # <-- :52
        _autotuning_knobs.adjust_block_size = False
        try: yield
        finally: _autotuning_knobs.adjust_block_size = previous

    Its only guard is ``if _autotuning_knobs is None`` -- a *module* presence
    test, not an attribute one. FlagTree 0.7.0 still ships ``triton.knobs`` but
    renamed the knob away (``autotuning_knobs`` now declares only ``cache``,
    ``print`` and ``use_backend_autotuning``), so the import succeeds, the
    ``None`` guard does not fire, and the first line raises ``AttributeError:
    'autotuning_knobs' object has no attribute 'adjust_block_size'``. The
    consumer is FlagGems' decoder path -- ``_scaled_dot_product_efficient_
    attention_backward`` is routed to ``flaggems`` in backends_ascend.conf -- so
    every attention backward with ``min(seqlen_q, seqlen_k) < 16`` dies here.
    Measured on Ascend 910C (FlagTree 0.7.0+ascend3.5, flag_gems 5.4.0): a
    standalone ``scaled_dot_product_attention`` backward at seqlen 8 raises
    exactly this; the transformers BERT suite fails 24 tests on it.

    The restoration the compiler can act on is already in effect on 0.7.0. The
    knob FlagGems is reaching for is what FlagTree calls "adjust a tl.load-driven
    BLOCK to next_power_of_2(seqlen)"; 0.7.0 split that behaviour out into
    ``use_backend_autotuning`` (default False, i.e. FlagTree does not adjust)
    plus the ``auto_blockify_size`` compile meta-parameter, and gates the
    auto-blockify pass on the ``TRITON_ALL_BLOCKS_PARALLEL`` environment variable
    (``triton/backends/ascend/utils.py::_is_auto_map_parallel_blocks_enabled``,
    default false). FlagGems' intent -- do not let an adjustment run for a short
    sequence -- is therefore the state 0.7.0 is already in, and an attribute
    *alias* is a pure no-op that only stops the AttributeError. That was checked,
    not assumed: ``adjust_block_size`` has no reader anywhere in the installed
    triton (0 hits in ``libtriton.so``, 0 in its Python, absent from
    ``knob_descriptors`` and ``knobs``), and the backward passes even when the
    attribute is set to a non-boolean sentinel -- so the value is never consumed
    and the alias does not change the code path a reader would select.

    A knob-name restore is the minimal lever with a defined upstream owner. The
    alternative -- moving ``_scaled_dot_product_efficient_attention_backward``
    off the FlagGems route in the generated conf -- would trade a first-call
    AttributeError for the ``ascend`` aclnn path that
    ``docs/reference/operator-support.md`` records as never revalidated and
    known to rebuild a non-square causal mask as ``{S, S}`` (ret=161001), so it
    is a fresh correctness risk in place of a name-mediated crash.

    Scoped to the build whose knob set this tests for, read from the attribute
    (never a version string), and to an Ascend FlagTree with FlagGems routed.
    Best-effort like its neighbours: the attribute is only ever added when it is
    absent, so a build whose knob already exists is untouched.
    """
    try:
        # The record names either spelling; the platform is "ascend".
        if platform_for(accelerator) != "ascend":
            return
        if not runtime.routes_to_flaggems():
            return

        triton_knobs = importlib.import_module("triton.knobs")
        autotuning = getattr(triton_knobs, "autotuning", None)
        if autotuning is None or hasattr(autotuning, "adjust_block_size"):
            return

        # False is the value FlagGems itself assigns inside the context manager,
        # which it intends as "do not adjust". It matches 0.7.0's own default
        # state (use_backend_autotuning and the auto-blockify gate both false),
        # so making the restore a no-op leaves the code path FlagTree already
        # selected. Any value would stop the AttributeError -- it is unread --
        # but this is the one that would also be correct if a future FlagTree
        # reintroduced a reader.
        autotuning.adjust_block_size = False
    except Exception:
        pass


def _flaggems_autotune_knob(triton_knobs):
    """Whether the user turned ``TRITON_CACHE_AUTOTUNING`` off, parsed as Triton does.

    Returns True when the knob may be enabled (unset, or set to anything Triton
    reads as on) and False when the user explicitly declined -- that "no" is what
    the caller must respect. A build whose knobs module cannot answer at all
    returns True so the caller proceeds: this function exists to avoid
    overwriting a deliberate setting, not to veto enabling the cache.

    Parsed through the module-level ``getenv_bool`` -- the same parser
    ``env_bool.get`` calls -- rather than a hand-rolled truthiness test, so
    "on"/"yes"/"1" mean here exactly what they mean to Triton. Guarded by
    ``hasattr`` because that helper is not part of every Triton build (the
    Cambricon 3.4 line has no such name); a build without it proceeds rather
    than being silently vetoed.
    """
    getenv_bool = getattr(triton_knobs, "getenv_bool", None)
    if getenv_bool is None:
        return True
    return bool(getenv_bool("TRITON_CACHE_AUTOTUNING", True))


def enable_cache(accelerator: str):
    """Stop Triton from re-benchmarking the same autotune configs every process.

    FlagGems' ``layer_norm_persistent_kernel`` carries a three-way
    ``@triton.autotune`` (``num_warps`` 4/8/16 from the vendor's
    ``tune_configs.yaml``). Triton benchmarks all three on the first call for
    each tuning key and keeps the winner in ``Autotuner.cache``, which is in
    memory and therefore per process. The kernel itself is compiled once and
    cached on disk, so a fresh process re-pays for a decision that cannot have
    changed.

    The knob gates the disk cache in both directions, not just the lookup.
    ``triton/runtime/autotuner.py:262`` only reaches ``check_disk_cache`` when
    ``self.cache_results`` was set at construction from
    ``knobs.autotuning.cache``, and ``check_disk_cache`` is also what *writes*
    the entry -- with the knob off the benchmarks run and nothing at all is
    remembered between processes. This is therefore a prerequisite for the
    cache existing, not a preference for reading it.

    ``torch.native_layer_norm`` on ``(1, 4096, 4096)`` bf16, one fresh process
    per arm, on a DCU bw1000:

        arm 1  TRITON_CACHE_AUTOTUNING=0, no entry on disk
               first call                                       454.2 ms
               :236(run) -> :252(benchmark) -> :132(_bench) x3   0.452 s
               (no ``check_disk_cache`` frame -- it is never called)
        arm 2  cache enabled, still no entry on disk
               first call                                       462.5 ms
               :236(run) -> :194(check_disk_cache)               0.460 s
                         -> :252(benchmark) -> :132(_bench) x3   0.446 s
        arm 3  cache enabled, the entry arm 2 wrote now on disk
               first call                                        33.6 ms
               :236(run) 0.032 -> :194(check_disk_cache) 0.014
               (no benchmark frame)
        second call, every arm                                   0.3 ms

    The disk lookup hashes the Triton build, the backend target, the tuning key,
    the config list and the cache-invalidating environment variables, so a stale
    entry cannot survive a compiler or configuration change.

    Qwen-Image-2.1 issues exactly two tuning keys over a whole run --
    ``(4122, 4096)`` and ``(4096, 4096)``, measured at 0.38 s and 0.47 s -- so
    this is 0.85 s paid once per process, on a model whose steady-state denoise
    step is 1.098 s on the vendor build and 1.173 s on this one. It is a startup
    cost, not a per-step one: nothing here closes the 6.8% step gap, and the
    arms above are the whole of the claim.

    Set both ways on purpose. The environment variable is the knob Triton
    documents and is what a user would set, but ``triton.knobs`` reads it once,
    when the knob objects are constructed, which has already happened by the
    time this runs; the constructed knob is therefore written directly as well.

    An explicit ``TRITON_CACHE_AUTOTUNING=0`` -- a user who wants the benchmark
    back, or who is chasing an autotune that settled on a config they disagree
    with -- is left alone. It has to be read deliberately rather than left to
    ``setdefault``: ``env_bool`` is a data descriptor whose ``__set__`` calls
    ``knobs.setenv``, so assigning the knob *writes the variable*, and a
    ``setdefault`` followed by that assignment overwrites the user's ``0`` with
    the ``1`` it just wrote. The existing value is therefore parsed with the
    same ``getenv_bool`` that ``env_bool.get`` uses, so the two agree by
    construction -- and the value is only written when that parse says yes.

    Scoped to the builds and confs whose routed ops are FlagGems Python launches
    -- see ``_FLAGGEMS_LAUNCH_PATCH_BUILDS`` -- like the patches around it. No op
    changes route and no kernel changes: the config selected is the one
    FlagGems' own tuner would have picked. Best-effort, like its neighbours: an
    unset knob is slow here, not broken, so this must not be able to take down
    device init.
    """
    try:
        if not launch_patches_apply(accelerator):
            return

        triton_knobs = importlib.import_module("triton.knobs")

        # Default True on purpose: unset is the state this exists to change, and
        # anything a user did set that does not parse as truthy is a "no".
        # Never call ``triton_knobs.getenv_bool`` unguarded: the rest of this
        # function is under ``except Exception: pass``, so a Triton build
        # without that helper would turn into a silent no-op instead of a slow
        # run. ``_flaggems_autotune_knob`` owns that check.
        if not _flaggems_autotune_knob(triton_knobs):
            return

        os.environ.setdefault("TRITON_CACHE_AUTOTUNING", "1")
        triton_knobs.autotuning.cache = True
    except Exception:
        pass


def prewarm_triton_key(accelerator: str):
    """Hash Triton's own build on a thread instead of inside the first compile.

    ``triton.runtime.cache.triton_key`` is ``functools.lru_cache``d and is the
    first thing ``get_cache_key`` asks for, so it runs inside whichever kernel
    compiles first -- for Qwen-Image-2.1 that is FlagGems' embedding, in the
    text encoder, or the layer-norm autotune in the denoise loop, depending on
    which side is reached first. It sha256s every file under ``triton/compiler``,
    ``triton/backends`` and ``triton/language`` and then the whole of
    ``libtriton.<ext>`` in 1 MiB chunks.

    On the DTK 6.3 wheel that .so is 900,284,304 bytes. Measured with the venv's
    own python, importing only ``triton.runtime.cache``:

        import triton.runtime.cache    0.196 s
        triton_key()                   1.013 s   (1694 characters later)

    and the same second shows up inside the model: the loop's first step is
    3030.0 ms of enqueue against 254.0 ms on the second, and ``get_cache_key``
    accounts for 1.325 s of a first-step profile in which one ``triton_key``
    call is 1.085 s. The lru_cache makes it a once-per-process cost, so it is
    paid in full by every run.

    Nothing about the value depends on this process -- it is a function of the
    installed files and ``triton.__version__`` alone, with no environment term
    in it -- so it can be computed at any point, and there is nothing to
    invalidate if it is computed early. It is started here, at device init,
    because the caller still has tens of seconds of work in front of it: the
    pipeline materialises ~33 GB of weights before the first Triton kernel is
    reached. ``hashlib`` releases the GIL for buffers this size and so do the
    reads, so the thread overlaps that load rather than serialising with it.

    Scoped like its neighbours -- DCU, on a conf that routes to FlagGems -- so
    the only processes that start a thread are the ones that were measured.
    Best-effort: an un-warmed cache is slow here, not broken, so a failure to
    start the thread must not be able to take down device init.
    """
    try:
        # The build record names the vendor ("hygon"); the platform is "dcu".
        if platform_for(accelerator) != "dcu":
            return
        if not runtime.routes_to_flaggems():
            return

        import threading

        from triton.runtime.cache import triton_key

        def warm():
            try:
                triton_key()
            except Exception:
                pass

        threading.Thread(target=warm, name="torch_fl-triton-key", daemon=True).start()
    except Exception:
        pass


_TUNE_SPACE_PATCHED = "_flagos_tune_space_fits_device"

# Resolved on first use rather than at patch time: triton's driver only reports a
# target once the device is up, and this is first read from a kernel launch.
_MUSA_SMEM_LIMIT = None


def _entry_tuner(entry):
    """The policy tuner a ``LibEntry`` holds, if it holds one.

    ``@libentry()`` and ``@libtuner(...)`` are applied bottom-up, so for every
    tuned kernel the ``LibEntry`` is built *over* the tuner
    (``LibEntry(Tuner(jit_fn))``) and is what the kernel's ``[grid](...)``
    reaches -- ``LibEntry.run`` calls ``self.fn`` once compiled. A tuner patched
    as a class therefore still has to be asked for through its entry; on the
    MTT S5000 the float64 ``mm`` space is sized from ``LibEntry.fn.run``.
    """
    fn = getattr(entry, "fn", None)
    return fn if hasattr(fn, "run") and hasattr(fn, "configs") else None


def _musa_shared_memory_limit():
    """Static shared memory one MUSA kernel may ask for, in bytes, or None.

    Read from the backend's own table rather than from torch: on this backend
    ``torch.cuda.get_device_properties(0)`` leaves ``shared_memory_per_block``
    and ``shared_memory_per_block_optin`` as ``None``, and the number the compiler
    enforces its own check against is ``shared_memory_per_multiprocessor``
    (196608 on an MTT S5000), which triton's mthreads backend derives from the
    target's arch.
    """
    global _MUSA_SMEM_LIMIT
    if _MUSA_SMEM_LIMIT is None:
        try:
            from triton.backends.mthreads.compiler import (
                _max_static_shared_memory_from_arch,
            )
            from triton.runtime import driver as triton_driver

            arch = triton_driver.active.get_current_target().arch
            smem = _max_static_shared_memory_from_arch(arch)
            if not smem:
                return None
            _MUSA_SMEM_LIMIT = int(smem)
        except Exception:
            return None
    return _MUSA_SMEM_LIMIT


def _dot_tile(config):
    """``(BLOCK_M, BLOCK_N, BLOCK_K)`` of a ``tl.dot`` tune config, else None.

    The shared-memory model below is the one a matmul pipeline asks for, and its
    operand keys are what identifies such a config; anything else (a reduction,
    a pointwise block, a config keyed differently) is left exactly as declared.
    """
    tile = tuple(config.kwargs.get(key) for key in ("BLOCK_M", "BLOCK_N", "BLOCK_K"))
    if any(not isinstance(value, int) for value in tile):
        return None
    return tile


def _pipeline_shared_memory(num_stages, tile, element_size):
    """The shared memory a pipelined ``tl.dot`` config asks the compiler for.

    Measured on the MTT S5000 host by inverting the relation with a stage count
    large enough to trip the compiler's own check -- ``OutOfResources`` reports
    the exact ask -- over fp32 and fp64 tiles at 64x64x64 and 32x64x64::

        dtype  tile          stages  ask        outcome
        fp32   64x64x64      8       229376     OutOfResources (Required 229376)
        fp32   64x64x64      5       131072     runs, err 0.0e+00
        fp64   64x64x64      5       262144     OutOfResources (Required 262144)
        fp64   64x64x64      4       196608     Triton Error [MUSA]: invalid argument
        fp64   64x64x64      3       131072     runs, err 7.1e-15
        fp64   32x64x64      5       196608     Triton Error [MUSA]: invalid argument
        fp64   32x64x64      4       147456     runs, err 3.6e-15

    Two things follow, and both matter here. The ask is ``(num_stages - 1)`` times
    the two tile operands at the element size, so it is the *element size* that
    halves a tune space's budget on float64 -- nothing about the tile changes.
    And a config is launchable only while the ask stays *strictly* under the
    limit: an ask that lands exactly on it compiles and then fails at launch with
    a bare ``RuntimeError`` from the MUSA toolchain, which is why comparing
    against the limit with ``<=`` would keep handing out a config that cannot run.
    """
    block_m, block_n, block_k = tile
    return (num_stages - 1) * (block_m * block_k + block_k * block_n) * element_size


def _fitting_stages(config, tile, element_size, limit):
    """The deepest pipeline that fits, never deeper than the config declared.

    The ask is computed from the *declared* tile, which is the widest the tile can
    ever be: FlagTree's AABS narrows a declared tile down to the extent before the
    compiler sizes it, and the ask only shrinks with it. Sizing the declared tile
    is therefore the conservative reading, and it is the one that is sound -- a
    lower bound on the ask could let through a config that lands exactly on the
    limit the way the float64 4-stage row of the table above does. What keeps that
    conservatism cheap is *when* this is called: only after the device has refused
    the declared space at the shape in hand, where the tile it refused is the one
    the compiler saw (see ``patch_tune_space``). Sizing a space up front
    instead would shrink stages at shapes AABS was about to narrow below the limit
    on its own, and those shapes do not need it.
    """
    stages = config.num_stages
    while stages > 1 and _pipeline_shared_memory(stages, tile, element_size) >= limit:
        stages -= 1
    return stages


def _operand_element_size(args, kwargs):
    """Bytes per element of the widest operand this launch will be given."""
    sizes = [
        tensor.element_size()
        for tensor in list(args) + list(kwargs.values())
        if isinstance(tensor, torch.Tensor)
    ]
    return max(sizes) if sizes else None


def _fitted_tune_space(configs, args, kwargs, limit):
    """``configs`` with every config's pipeline cut back to what fits, else None.

    None means "nothing to change": either the space is not a ``tl.dot`` space,
    no operand dtype is known, or every config already fits -- in which case the
    caller hands the declared space to FlagGems untouched, so the platforms and
    the op families that were never affected keep the exact tune space they have
    today.
    """
    element_size = _operand_element_size(args, kwargs)
    if element_size is None:
        return None
    import triton

    fitted, changed, seen = [], False, set()
    for config in configs:
        tile = _dot_tile(config)
        if tile is None:
            return None
        stages = _fitting_stages(config, tile, element_size, limit)
        if stages == config.num_stages:
            candidate = config
        else:
            changed = True
            candidate = triton.Config(
                config.kwargs,
                num_warps=config.num_warps,
                num_stages=stages,
                num_ctas=config.num_ctas,
                maxnreg=config.maxnreg,
                pre_hook=config.pre_hook,
                ir_override=config.ir_override,
            )
        key = (
            tuple(sorted(candidate.kwargs.items())),
            candidate.num_warps,
            candidate.num_stages,
            candidate.num_ctas,
            candidate.maxnreg,
        )
        # Two declared configs differing only in num_stages collapse onto one
        # here -- on float64 both 64x64x64 entries of the mthreads `mm` space
        # become the same num_stages=3 config -- and a duplicate would only cost
        # a benchmark.
        if key in seen:
            continue
        seen.add(key)
        fitted.append(candidate)
    if not changed:
        return None
    return fitted or None


def _launch_in_fitted_space(original_run, tuner, configs, args, kwargs):
    """Run the tuner with a fitted space installed, and put FlagGems back.

    FlagGems keeps the space on the tuner for the life of the process, so the
    declared list has to be reinstated once the call returns. ``configs_hash`` and
    ``kernel_hash`` are ``cached_property`` values computed from whichever list is
    current, so they are dropped on both sides of the call: a value left behind by
    a fitted launch names a launcher nothing declares, and one left behind after
    the restore names the fitted space for the next call.
    """
    declared = tuner.configs
    tuner.configs = configs
    tuner.__dict__.pop("configs_hash", None)
    tuner.__dict__.pop("kernel_hash", None)
    try:
        return original_run(tuner, *args, **kwargs)
    finally:
        tuner.configs = declared
        tuner.__dict__.pop("configs_hash", None)
        tuner.__dict__.pop("kernel_hash", None)


def _tune_space_run(original_run):
    """``LibTuner.run`` with the refused-space repair in front of ``original_run``.

    The repair is applied to the class rather than to an instance, so a tuner
    built later (a kernel module imported after this patch ran) is covered too.
    See ``patch_tune_space`` for why the repair only runs after a
    refusal.
    """

    def run(self, *args, **kwargs):
        try:
            return original_run(self, *args, **kwargs)
        except Exception as refused:
            limit = _musa_shared_memory_limit()
            configs = (
                None
                if limit is None
                else _fitted_tune_space(
                    getattr(self, "configs", None) or [], args, kwargs, limit
                )
            )
            if configs is None:
                raise
            try:
                return _launch_in_fitted_space(
                    original_run, self, configs, args, kwargs
                )
            except Exception:
                raise refused from None

    run.__wrapped__ = original_run
    run.__name__ = getattr(original_run, "__name__", "run")
    return run


def _rewrap_flaggems_tuners(class_patch=None, only_in=None):
    """Point every FlagGems tuner policy subclass at ``LibTuner.run``, then patch it.

    FlagGems does not call ``LibTuner.run`` at a launch. ``@LibTuner.register``
    builds a policy subclass on every ``libtuner`` decoration, and that subclass
    gets its *own* ``run``: a copy of whatever ``LibTuner.run`` was when the
    subclass was created. ``flag_gems.ops`` is imported after ``flagos.init()``
    in a fresh process (dispatch imports it lazily), and much of it later still,
    so a patch written to ``LibTuner.run`` alone is never reached -- measured on
    the MTT S5000: with only ``LibTuner.run`` patched, float64 ``mm`` at
    33x33x33 kept ending on the refused 64x64x64 space.

    A subclass created before the patch therefore has to be re-pointed at
    ``LibTuner.run`` itself, which is what ``class_patch`` does; instances
    already built keep the subclass's previous ``run`` as an instance attribute
    and are re-wrapped in place. Subclasses created after the patch inherit the
    patched class method and need nothing.
    """
    libentry = importlib.import_module("flag_gems.utils.libentry")
    libtuner = getattr(libentry, "LibTuner", None)
    if libtuner is None:
        return False
    if only_in is not None and getattr(libtuner, _TUNING_MODULE_PATCHED, None) is (
        only_in
    ):
        # This FlagGems instance has already been repaired against this module;
        # see ``refresh_tune_space``.
        return getattr(libtuner.run, _TUNE_SPACE_PATCHED, False)
    patched_run = libtuner.run
    for subclass in _subclasses_in_order(libtuner):
        if subclass.__dict__.get("run") is not None and (
            class_patch is not None
            and not getattr(subclass.__dict__["run"], _TUNE_SPACE_PATCHED, False)
        ):
            class_patch(subclass)
        subclass_run = subclass.run
        if getattr(subclass_run, _TUNE_SPACE_PATCHED, False):
            continue
        for referrer in gc.get_referrers(subclass_run):
            # ``new`` wraps the function and runs the subclass body before
            # ``__init__``, so a tuner is seen here mid-construction. Its
            # ``run`` is then still the unpatched one to repair.
            if not isinstance(referrer, libtuner) or type(referrer) is not subclass:
                continue
            declared = referrer.__dict__.get("run", subclass_run)
            if getattr(declared, _TUNE_SPACE_PATCHED, False):
                continue
            repaired = _tune_space_run(declared)
            setattr(repaired, _TUNE_SPACE_PATCHED, True)
            referrer.run = repaired
    _rewrap_libentry_tuners(libtuner)
    if only_in is not None:
        setattr(libtuner, _TUNING_MODULE_PATCHED, only_in)
    return getattr(patched_run, _TUNE_SPACE_PATCHED, False)


def _rewrap_libentry_tuners(libtuner):
    """Patch the tuner a kernel reaches through its ``LibEntry``, in place.

    A kernel decorated ``@libentry()`` over ``@libtuner(...)`` is an instance
    whose ``fn`` is the tuner, and it calls ``self.fn`` rather than the class:
    ``LibEntry.run`` invokes the function object it was built with, and
    ``LibEntry`` instances are created while their module is imported, before
    ``flagos.init()`` can patch anything. So a tuner held this way keeps
    whatever ``run`` it had at construction however its class is later rebound,
    and the repairs above -- which write to classes and to instances of a
    policy subclass -- do not reach it. Measured through the kernel's own
    ``LibEntry`` on the MTT S5000, that is where the float64 ``mm`` space is
    sized from.
    """
    for referrer in gc.get_referrers(libtuner):
        if type(referrer) is not libtuner:
            continue
        if getattr(referrer, "run", None) is not libtuner.run:
            continue
        tuner = _entry_tuner(referrer.fn)
        if tuner is None:
            continue
        declared = tuner.__dict__.get("run", type(tuner).run)
        if getattr(declared, _TUNE_SPACE_PATCHED, False):
            continue
        repaired = _tune_space_run(declared)
        setattr(repaired, _TUNE_SPACE_PATCHED, True)
        tuner.run = repaired


def _subclasses_in_order(cls):
    for subclass in cls.__subclasses__():
        yield subclass
        yield from _subclasses_in_order(subclass)


#: Set on ``LibTuner`` once its ``__init_subclass__`` installs the repair on
#: every policy subclass it sees.
_SUBCLASS_PATCHED = "_flagos_tune_space_subclass_patch"

#: The ``flag_gems.runtime`` module instance a ``LibTuner`` was repaired for.
#: device_finder.py reloads the whole ``flag_gems`` package when
#: ``GEMS_VENDOR`` names another vendor, and a reload builds a new
#: ``LibTuner``; see ``refresh_tune_space``.
_TUNING_MODULE_PATCHED = "_flagos_tune_space_runtime"


def patch_tune_space(accelerator: str):
    """Repair a FlagGems tune space the MUSA device has refused, and only then.

    Also installed on every later device bring-up (``refresh_tune_space``),
    because the kernels that matter are imported after the first one -- dispatch
    imports ``flag_gems.ops`` lazily, and ``mm`` with it -- and their entries are
    built then.

    ``mm`` on float64 fails on this hardware for every shape whose M, K and N all
    exceed 32 -- ``OutOfResources: out of resource: shared memory, Required:
    262144, Hardware limit: 196608`` -- and the tune space is why: the mthreads
    backend ships one ``mm`` entry, two 64x64x64 tiles at ``num_stages`` 5 and 4,
    and at 8 bytes per element those two ask 262144 B and 196608 B against a
    196608 B limit. The element size is the whole difference; the same space on
    float32 asks 131072 B and 98304 B and runs (measured, err 0.0e+00).

    A shape with a dimension at or under 32 survives only because FlagTree's AABS
    narrows the tile to the extent first, which shrinks the ask below the limit;
    that is also why the defect looks shape-dependent rather than dtype-wide. A
    refusal is not something the tuner can absorb: ``LibTuner.run``'s ``bench``
    catches ``RuntimeError``, and the compiler's rejection is a ``TritonError``,
    which is an ``Exception`` and not a ``RuntimeError`` -- so the first config
    that cannot compile ends the whole tuning pass instead of being scored out.
    The launch-time half of the defect is caught, and that is why the exact-limit
    case is survivable at all: an ask that lands exactly on 196608 B compiles and
    is then rejected at launch with a bare ``RuntimeError: Triton Error [MUSA]:
    invalid argument``, which ``bench`` does catch, so at a narrowed shape the
    tuner quietly moves on to the next config. Neither half can be fixed from the
    error side, and both are the tune space asking for more than the device has.

    So the space *is* replaced, but only for the call that was refused. Doing it
    up front was measured to be wrong: AABS narrows a declared 64x64x64 tile to
    the operand extents, and a fit that runs before the first launch sizes the
    declared tile, takes the only remaining config down to ``num_stages`` 3, and
    leaves a one-config space -- which never reaches the benchmark at all, so the
    narrowing is lost with the stages. On float64 that costs 5.9x at
    ``2048x2048x16`` (9.251 ms against 1.579 ms), 6.2x at ``4096x4096x16``
    (32.212 against 5.211) and 1.8x at ``512x512x32`` (2.369 against 1.294), all
    measured on S5000 with a synchronized timer, against errors that stay at
    1e-13 either way. Repairing after the refusal leaves every one of those calls
    alone -- same winning configuration, same median -- and still fixes
    ``(64, 64, 64)`` float64, which raises on the unmodified tree and runs at
    0.324 ms, err 7.1e-15, with the repair.

    Fitting is arithmetic against the backend's own limit (see
    ``_pipeline_shared_memory``) rather than trial compilation, which keeps it
    deterministic and cache-stable. If the repaired space does not take either,
    the caller is shown the refusal of the declared space rather than a
    replacement's failure: the retry is a repair, not a redefinition of the call.

    Best-effort, like the patches around it: this must not be able to take down
    device init, and an unfitted tune space is the status quo, not a new failure.
    """
    # The record names the vendor ("mthreads"); the platform is "musa". The gate
    # sits outside the ``try`` so a patched import fails loudly instead of being
    # swallowed into "this build is not MUSA" -- the patches below are the ones
    # whose failures are best-effort, and a test holding a stub in
    # ``sys.modules`` has to see that they ran.
    if platform_for(accelerator) != "musa":
        return

    try:
        libentry = importlib.import_module("flag_gems.utils.libentry")
        libtuner = libentry.LibTuner

        # Every policy subclass gets its own copy of ``LibTuner.run`` at class
        # creation (``LibTuner.register`` builds one per ``libtuner`` call), and
        # that copy is what a launch calls. Wrapping the base class alone reaches
        # none of them; see ``_rewrap_flaggems_tuners``. Also, ``tune_speed`` and
        # ``LibEntry.run`` were already read from the module by kernels that were
        # imported before this ran, so the *class* has to be the binding, not the
        # module attribute this function could no longer change.
        if not getattr(libtuner, _SUBCLASS_PATCHED, False):

            def __init_subclass__(cls, **kwargs):
                super(libtuner, cls).__init_subclass__(**kwargs)
                _tune_space_subclass_patch(cls)

            libtuner.__init_subclass__ = __init_subclass__
            setattr(libtuner, _SUBCLASS_PATCHED, True)

        if not getattr(libtuner.run, _TUNE_SPACE_PATCHED, False):
            repaired = _tune_space_run(libtuner.run)
            setattr(repaired, _TUNE_SPACE_PATCHED, True)
            libtuner.run = repaired
        setattr(libtuner, _TUNE_SPACE_PATCHED, True)

        _rewrap_flaggems_tuners(
            class_patch=_tune_space_subclass_patch,
            only_in=importlib.import_module("flag_gems.runtime"),
        )
    except Exception:
        pass


def refresh_tune_space(accelerator: str) -> None:
    """Re-install the tune-space repair on every device bring-up after the first.

    ``flagos.init`` patches once, and by then only the FlagGems modules already
    imported are covered: ``flag_gems.ops`` is imported on the first routed op
    (and ``mm`` with it, building the entries that hold the tuner afterwards), so
    those entries are built after the one repair the import performs. Calling
    this from ``_lazy_init`` -- every bring-up repeats it -- re-applies the repair
    to whatever exists by then, and only once per ``flag_gems.runtime`` instance
    a reload may replace. It is a no-op until ``flag_gems.runtime`` and
    ``flag_gems.ops`` are both imported. See ``patch_tune_space``.
    """
    if platform_for(accelerator) != "musa":
        return
    if "flag_gems.utils.libentry" not in sys.modules:
        return
    if "flag_gems.ops" not in sys.modules:
        return
    patch_tune_space(accelerator)


def _tune_space_subclass_patch(subclass):
    """``class_patch`` for ``_rewrap_flaggems_tuners``: repair one stale subclass.

    A policy subclass made before the ``__init_subclass__`` arm was installed
    still holds its own copy of the unpatched ``run``; rebinding the method is
    what every instance -- and any code reaching ``Sub.run`` -- will then see.
    """
    repaired = _tune_space_run(subclass.__dict__["run"])
    setattr(repaired, _TUNE_SPACE_PATCHED, True)
    subclass.run = repaired
