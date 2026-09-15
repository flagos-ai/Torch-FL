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

"""The torch.compile backend: partition, compile, and splice BPU calls into the graph.

Each BPU partition is replaced by one call_function node invoking a BPURuntime,
so the surrounding CPU nodes keep running in eager mode and control flow that
Dynamo already split out is untouched.
"""

from __future__ import annotations

import contextvars
import logging
import operator
from typing import Any, Callable

import torch
from torch.fx import GraphModule, Node

from .compiler import CompileError, compile_partition, find_hbdk
from .decompose import decompose
from .partition import (
    Partition,
    extract_subgraph,
    partition_graph,
    runtime_inputs,
    summarize,
)
from .runtime import BPURuntime

log = logging.getLogger("torch_fl.bpu")

# Dynamo's own graph inputs, stashed so the inner aten compiler can tell which
# of them are parameters and buffers. By the time AOTAutograd calls us the
# names are gone and the tensors are fake, so the information has to be carried
# across rather than recovered. A ContextVar keeps concurrent compiles apart.
_OUTER_INPUTS: contextvars.ContextVar[tuple[list[str], list[torch.Tensor]] | None] = (
    contextvars.ContextVar("torch_fl_bpu_outer_inputs", default=None)
)


class _BPUCall(torch.nn.Module):
    """Holds a BPURuntime so it survives as a graph attribute.

    Also carries the weight check. The frozen weights are baked into the
    artifact, but Dynamo guards parameters on shape and dtype only unless
    `_weights_are_guarded()` holds -- so this module can be reached with a
    *different* model's weights and would otherwise return the first model's
    answer with no indication that anything was wrong.

    The spliced call therefore keeps the frozen weights as trailing arguments
    (`n_real` marks where they start). They are not passed to the artifact --
    they are compared against the tensors it was compiled from, and a mismatch
    falls back to running the original subgraph eagerly.
    """

    def __init__(
        self,
        rt: BPURuntime,
        n_outputs: int,
        n_real: int | None = None,
        frozen: list[torch.Tensor] | None = None,
        fallback: Callable[..., Any] | None = None,
    ):
        super().__init__()
        self.rt = rt
        self.n_outputs = n_outputs
        self.n_real = n_real
        self._frozen = list(frozen or [])
        self._fallback = fallback
        self._warned = False

    def _matches(self, guards: tuple[torch.Tensor, ...]) -> bool:
        if len(guards) != len(self._frozen):
            return False
        # Identity first: the common case is the very same parameter object.
        return all(
            g is f or (g.shape == f.shape and g.dtype == f.dtype and torch.equal(g, f))
            for g, f in zip(guards, self._frozen)
        )

    def forward(self, *args: torch.Tensor):
        if self.n_real is not None:
            real, guards = args[: self.n_real], args[self.n_real :]
            if not self._matches(guards):
                if not self._warned:
                    self._warned = True
                    log.warning(
                        "BPU: this graph was reached with different weights than "
                        "the artifact was compiled from, so the partition is "
                        "running on the CPU. Compile under torch.no_grad() with "
                        "torch._inductor.config.freezing = True to get one "
                        "artifact per model instead."
                    )
                if self._fallback is not None:
                    return self._fallback(*real, *guards)
            args = real
        outs = self.rt(*args)
        return outs[0] if self.n_outputs == 1 else tuple(outs)


def _example_inputs_for(
    p: Partition,
    example_inputs: list[torch.Tensor],
    gm: GraphModule,
    frozen: dict[str, torch.Tensor] | None = None,
) -> list[torch.Tensor] | None:
    """Materialize concrete tensors for a partition's boundary inputs.

    Uses the fake tensors Dynamo recorded in node.meta['val'] to synthesize
    real tensors of the right shape and dtype. Frozen weights use their real
    values, since the compiler bakes those into the artifact; everything else
    only needs the right shape.

    Returns None when a boundary input cannot be materialized, which
    `_unsupported_input` explains.
    """
    from torch._subclasses.fake_tensor import unset_fake_temporarily

    frozen = frozen or {}
    out = []
    # A backend runs under FakeTensorMode, so torch.zeros would itself produce
    # a fake tensor. The ONNX exporter needs real storage.
    with unset_fake_temporarily():
        for n in p.inputs:
            if n.name in frozen:
                continue
            val = n.meta.get("val")
            if not isinstance(val, torch.Tensor):
                return None
            if any(not isinstance(d, int) for d in val.shape):
                return None
            out.append(torch.zeros(tuple(val.shape), dtype=val.dtype))
    return out


def _unsupported_input(
    p: Partition, frozen: dict[str, torch.Tensor] | None = None
) -> str:
    """Why `_example_inputs_for` gave up, for the log.

    Worth the extra pass: this used to be reported as "dynamic shapes" whatever
    the cause, and the actual cause on ResNet was a boundary input whose
    `meta['val']` is a tuple -- a graph with no dynamic shapes at all.
    """
    frozen = frozen or {}
    for n in p.inputs:
        if n.name in frozen:
            continue
        val = n.meta.get("val")
        if val is None:
            return f"{n.name} ({n.target}) has no fake-tensor metadata"
        if not isinstance(val, torch.Tensor):
            return (
                f"{n.name} ({n.target}) is a {type(val).__name__}, not a tensor; "
                "a multi-output op cannot cross a partition boundary"
            )
        if any(not isinstance(d, int) for d in val.shape):
            return f"{n.name} has dynamic shape {tuple(val.shape)}"
    return "unknown"


# Dynamo names lifted module state after its access path, e.g.
# `l_self_modules_c1_parameters_weight_`. AOTAutograd renames these to
# `primals_N` but records the original index in node.meta['desc'].idx, which is
# how a weight is recognised two layers down from where it was named.
_STATE_MARKERS = ("_parameters_", "_buffers_")


def _outer_state_tensors() -> list[torch.Tensor]:
    """The real parameter and buffer tensors AOTAutograd lifted, in graph order.

    Dynamo has two ways of handing module state to a backend. Inlined state
    arrives as extra *inputs*, which the index lookup above resolves through
    `_OUTER_INPUTS`. Under `is_parameter_freezing()` -- inductor's
    `config.freezing` with grad disabled, which is what makes Dynamo guard on
    parameter identity so a second module instance recompiles -- the state stays
    on the module as `get_attr` nodes instead, and never appears in the input
    list at all. There the only real tensors are the tracing context's
    params_flat; `example_inputs` is entirely fake by that point.

    Getting this wrong is expensive rather than incorrect: every weight then
    crosses the boundary on each call, which measured ~2000x slower than the
    frozen path on a 6-layer conv stack.
    """
    ctx = torch._guards.TracingContext.try_get()
    flat = getattr(ctx, "params_flat", None) if ctx is not None else None
    if not flat:
        return []
    return [t for t in flat if isinstance(t, torch.Tensor)]


def _frozen_weights(
    gm: GraphModule, example_inputs: list[torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Map aten placeholder name -> constant tensor, for parameters and buffers.

    Returns {} when the mapping cannot be established, in which case weights
    stay as runtime inputs and the graph is still correct, just slower.
    """
    from torch._subclasses.fake_tensor import FakeTensor

    frozen: dict[str, torch.Tensor] = {}
    placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]

    outer = _OUTER_INPUTS.get()
    if outer:
        names, values = outer
        for node in placeholders:
            idx = getattr(node.meta.get("desc"), "idx", None)
            if idx is None or not (0 <= idx < len(values)):
                continue
            t, name = values[idx], names[idx]
            is_state = isinstance(t, torch.nn.Parameter) or any(
                m in name for m in _STATE_MARKERS
            )
            if (
                is_state
                and isinstance(t, torch.Tensor)
                and not isinstance(t, FakeTensor)
            ):
                frozen[node.name] = t

    if frozen:
        return frozen

    # Nothing matched by index: Dynamo kept the state on the module rather than
    # in its input list, so `_OUTER_INPUTS` holds only the real inputs and
    # `example_inputs` is entirely fake. The real tensors are in the tracing
    # context's params_flat, and AOTAutograd lifts them to the *leading*
    # placeholders -- `desc.idx` is set only on placeholders that came from an
    # actual graph input, so the leading run without one is exactly the lifted
    # parameters and buffers, in params_flat order.
    state = _outer_state_tensors()
    if not state or len(state) >= len(placeholders):
        return frozen
    for node, t in zip(placeholders[: len(state)], state):
        if getattr(node.meta.get("desc"), "idx", None) is not None:
            return {}
        if isinstance(t, torch.Tensor) and not isinstance(t, FakeTensor):
            frozen[node.name] = t
    return frozen


def _splice(
    gm: GraphModule,
    p: Partition,
    mod: _BPUCall,
    tag: str,
    call_inputs: list[Node] | None = None,
) -> None:
    """Replace a partition's nodes with a single call into `mod`.

    `call_inputs` must match the compiled artifact's input order — frozen
    weights are baked in and are not passed.
    """
    setattr(gm, tag, mod)
    args = tuple(p.inputs if call_inputs is None else call_inputs)

    with gm.graph.inserting_before(p.nodes[0]):
        call = gm.graph.call_module(tag, args=args)

    if len(p.outputs) == 1:
        p.outputs[0].replace_all_uses_with(call)
    else:
        for i, old in enumerate(p.outputs):
            with gm.graph.inserting_after(call):
                item = gm.graph.call_function(operator.getitem, (call, i))
            old.replace_all_uses_with(item)

    for node in reversed(p.nodes):
        if not node.users:
            gm.graph.erase_node(node)


def bpu_backend(
    gm: GraphModule,
    example_inputs: list[torch.Tensor],
    *,
    mode: str | None = None,
    options: dict[str, Any] | None = None,
    min_nodes: int = 3,
    strict: bool = False,
    act_scales: dict[str, float] | None = None,
) -> Callable[..., Any]:
    """Dynamo backend that offloads compilable subgraphs to the BPU.

    Dynamo hands us a graph of torch-level calls; AOTAutograd lowers it to
    aten ops with fake-tensor metadata, which is what the partitioner matches
    against.

    Partitions that cannot be compiled stay in the graph and run on CPU, so a
    missing or failing hbdk4 degrades performance but never correctness. Set
    `strict=True` to raise instead.

    `act_scales` maps ONNX tensor name to quantization scale (see
    `calibrate.calibrate_onnx`). Anything not listed falls back to
    `compiler.ACT_SCALE`.

    `mode` and `options` are accepted because torch.compile forwards them to any
    custom backend, so refusing them turns a routine
    `torch.compile(m, backend="bpu", mode="reduce-overhead")` into a hard
    failure. The modes describe inductor codegen strategies -- CUDA graphs,
    autotuning, kernel fusion -- and none of them apply to an artifact hbdk4
    already compiled ahead of time, so `mode` is ignored with a note rather than
    silently promising an effect it cannot have. `options` is the supported way
    to reach the real knobs above.
    """
    from torch._dynamo.backends.common import aot_autograd

    if mode is not None and mode != "default":
        log.info(
            "mode=%r has no effect on the BPU backend: it selects inductor "
            "codegen strategies, and the BPU runs a .hbm that hbdk4 compiled "
            "ahead of time. Use options={...} to set min_nodes/strict/act_scales.",
            mode,
        )
    if options:
        unknown = set(options) - {"min_nodes", "strict", "act_scales"}
        if unknown:
            raise TypeError(
                f"unknown BPU backend option(s): {sorted(unknown)}. "
                "Supported: min_nodes, strict, act_scales."
            )
        min_nodes = options.get("min_nodes", min_nodes)
        strict = options.get("strict", strict)
        act_scales = options.get("act_scales", act_scales)

    def _compile_aten(aten_gm: GraphModule, aten_inputs: list[torch.Tensor]):
        return _offload(
            aten_gm,
            aten_inputs,
            min_nodes=min_nodes,
            strict=strict,
            act_scales=act_scales,
        )

    names = [n.name for n in gm.graph.nodes if n.op == "placeholder"]
    token = _OUTER_INPUTS.set((names, list(example_inputs)))
    try:
        return aot_autograd(fw_compiler=_compile_aten)(gm, example_inputs)
    finally:
        _OUTER_INPUTS.reset(token)


def _weights_are_guarded() -> bool:
    """Does Dynamo guard on parameter *identity* for this compile?

    It does when `is_parameter_freezing()` holds -- inductor's `config.freezing`
    with grad disabled -- or when nn.Module inlining is off, which specializes on
    the instance instead. Otherwise parameters are guarded only on shape and
    dtype, so two instances of one architecture share a compiled graph.

    That sharing is safe for a backend that keeps weights as runtime inputs, and
    unsafe for this one: `_frozen_weights` bakes them into the .hbm, so a second
    model would otherwise be executed with the first model's weights. When this
    returns False the spliced call carries its weights as extra arguments and
    checks them, so the result stays correct either way -- the difference is
    whether a second model gets its own artifact or falls back to the CPU.
    """
    from torch._dynamo import config as dynamo_config

    if not dynamo_config.inline_inbuilt_nn_modules:
        return True
    return torch._inductor.config.freezing and not torch.is_grad_enabled()


def _subgraph_fallback(
    gm: GraphModule, p: Partition, frozen: dict[str, torch.Tensor]
) -> Callable[..., Any] | None:
    """An eager callable for a partition, taking its weights as arguments.

    Used when the artifact's baked-in weights do not match the ones actually
    flowing through the graph. The compiled subgraph cannot serve here: it has
    the original weights frozen in as attributes, which is exactly what is
    wrong. Rebuilding without a frozen set gives a graph whose placeholders are
    all of `p.inputs`, in that order -- real inputs first, then the weights the
    caller passed as guards.
    """
    try:
        sub = extract_subgraph(gm, p, None)
    except Exception:  # noqa: BLE001 - a missing fallback is not fatal
        return None
    order = [n.name for n in p.inputs]
    real = [n for n in order if n not in frozen]
    guards = [n for n in order if n in frozen]
    index = {name: i for i, name in enumerate(real + guards)}

    def run(*args: torch.Tensor):
        out = sub(*(args[index[name]] for name in order))
        return out[0] if isinstance(out, (tuple, list)) and len(out) == 1 else out

    return run


def _offload(
    gm: GraphModule,
    example_inputs: list[torch.Tensor],
    *,
    min_nodes: int = 3,
    strict: bool = False,
    act_scales: dict[str, float] | None = None,
) -> Callable[..., Any]:
    """Partition an aten graph and splice in BPU calls where possible."""
    # Before partitioning, not just before export: a tuple-returning op both
    # blocks its own partition and poisons the boundary of the next one, so
    # rewriting it to the single-output form is what keeps a network like
    # ResNet in one piece across its stem max-pool.
    decompose(gm)

    partitions = partition_graph(gm, min_nodes=min_nodes)

    if not partitions:
        log.info("no BPU-eligible partitions; running on CPU")
        return gm.forward

    log.info("%s", summarize(gm, partitions))

    if find_hbdk() is None:
        msg = (
            "no hbdk4 compiler reachable — %d partition(s) will run on CPU. "
            "hbdk4 ships x86_64-only wheels, so it runs on this board under "
            "box64. Run scripts/vendor/setup_bpu_hbdk4.sh once, then set "
            "FLAGOS_BPU_X86_PYTHON and FLAGOS_BPU_X86_EMULATOR. The stock "
            "64 KB-page kernel is fine (box64 0.4+); see docs/vendors/bpu/integration.md."
        )
        if strict:
            raise CompileError(msg % len(partitions))
        log.warning(msg, len(partitions))
        return gm.forward

    frozen = _frozen_weights(gm, example_inputs)
    if frozen:
        log.info("freezing %d weight tensor(s) into the artifact", len(frozen))
    else:
        log.warning(
            "could not identify parameters; weights will cross the boundary on "
            "every call, which usually costs more than the offload saves"
        )

    compiled = 0
    # Only worth guarding when Dynamo itself does not: with an identity guard in
    # place a mismatch cannot reach us, so the extra arguments and comparison
    # would be pure overhead on every call.
    guard_weights = bool(frozen) and not _weights_are_guarded()
    # Splice in reverse so earlier partitions' node references stay valid.
    for i, p in reversed(list(enumerate(partitions))):
        ex = _example_inputs_for(p, example_inputs, gm, frozen)
        if ex is None:
            log.warning(
                "partition %d: %s — keeping on CPU", i, _unsupported_input(p, frozen)
            )
            continue
        try:
            sub = extract_subgraph(gm, p, frozen)
            hbm, ins, outs = compile_partition(sub, ex, act_scales=act_scales)
            rt = BPURuntime(str(hbm), ins, outs)
            real_inputs = runtime_inputs(p, frozen)
            call_inputs, guard_nodes, guard_vals = real_inputs, [], []
            if guard_weights:
                # The frozen weights are still nodes in the outer graph, so they
                # can be passed alongside the real inputs and compared there.
                guard_nodes = [n for n in p.inputs if n.name in frozen]
                guard_vals = [frozen[n.name] for n in guard_nodes]
                call_inputs = real_inputs + guard_nodes
            _splice(
                gm,
                p,
                _BPUCall(
                    rt,
                    len(p.outputs),
                    n_real=len(real_inputs) if guard_weights else None,
                    frozen=guard_vals,
                    fallback=_subgraph_fallback(gm, p, frozen)
                    if guard_weights
                    else None,
                ),
                f"_bpu_{i}",
                call_inputs,
            )
            compiled += 1
        except CompileError as e:
            if strict:
                raise
            log.warning("partition %d: %s — keeping on CPU", i, e)
        except Exception as e:  # noqa: BLE001 - never break the user's model
            if strict:
                raise
            log.warning("partition %d: unexpected %s: %s", i, type(e).__name__, e)

    if compiled:
        gm.graph.lint()
        gm.recompile()
        log.info("offloaded %d/%d partition(s) to BPU", compiled, len(partitions))

    return gm.forward


def register() -> None:
    """Register the backend so torch.compile(backend="bpu") resolves."""
    from torch._dynamo import register_backend

    register_backend(name="bpu", compiler_fn=bpu_backend)
