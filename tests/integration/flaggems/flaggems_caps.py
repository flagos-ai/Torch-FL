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

"""Capability data for the hardware-agnostic FlagGems conformance suite.

This package is the FlagGems half of the same contract ``tests/integration/
runtime`` states for the device runtime: one suite, run unchanged on every
accelerator, deciding from what the wheel carries rather than from the name of
the machine it happens to be running on. Two facts are read, both of them files
inside the wheel:

    * the op-routing conf (``backends_<platform>.conf``). Unlike the runtime
      contract, a ``none`` route is *not* an error here: FlagGems is an optional
      layer whose per-operator coverage is expected to differ between chips, so
      the suite measures the operators this conf routes to a FlagGems backend
      and skips the rest. What it refuses to accept is a conf that claims a
      route the dispatcher does not take.
    * the build record (``torch_fl/_build_config.py``), which says whether the
      FlagGems kernel set was compiled into this wheel at all.

The operator lists, the pinned-off sets and the expected route counts below are
measurements, keyed by the accelerator the conf names -- they are data, not
branches. A new chip adds a row; it does not add a test file. That is the point
of this suite: one chip that needs a FlagGems change makes every chip run the
same cases, and an operator this chip cannot serve is a skip with the conf's own
route as the reason, never a silent pass.

Nothing here imports torch or torch_fl at module scope: the integration conftest
loads this module before torch_fl has preloaded its device assets, so the
import-visible surface is stdlib only and ``torch_fl`` is imported inside the
functions that need it.
"""

from __future__ import annotations

from pathlib import Path

#: ``[flagos dispatch]`` prints this device string for the flagos device.
DEVICE = "flagos:0"

#: The inputs every routing call below is written against. Device tensors, in
#: one block, so a call's shape requirements stay visible next to it: ``w`` has
#: three rows because ``idx`` only indexes 0..2, ``p``/``q`` are in [0, 1]
#: because ``binary_cross_entropy``'s CUDA kernel asserts on its input's domain,
#: and ``zero`` is a 0-d tensor because ``masked_fill``'s FlagGems kernel rejects
#: anything else.
PRELUDE = (
    "a = torch.randn(4, 4, device=DEVICE)\n"
    "b = torch.randn(4, 4, device=DEVICE)\n"
    "p = torch.rand(4, 4, device=DEVICE)\n"
    "q = torch.rand(4, 4, device=DEVICE)\n"
    "zero = torch.zeros((), device=DEVICE)\n"
    "w = torch.randn(3, 4, device=DEVICE)\n"
    "idx = torch.tensor([0, 2, 1], device=DEVICE)\n"
    "src = torch.randn(3, 4, device=DEVICE)\n"
)

#: conf op name -> the call that reaches it, on the device tensors in PRELUDE.
#:
#: The conf is the oracle for which backend each of these must reach, so the
#: call is the only thing stated here. These are the operators a FlagGems change
#: can move on any chip: every one of them is routed to a FlagGems backend by at
#: least one platform's conf, so the suite has something to measure wherever a
#: chip claims the route, and skips with the conf's route as the reason
#: elsewhere.
FLAGGEMS_ROUTING_CALLS = {
    "neg": "torch.neg(a)",
    "abs": "torch.abs(a)",
    "exp": "torch.exp(a)",
    "log": "torch.log(a)",
    "sqrt": "torch.sqrt(a)",
    "rsqrt": "torch.rsqrt(a)",
    "tanh": "torch.tanh(a)",
    "sigmoid": "torch.sigmoid(a)",
    "gelu": "torch.nn.functional.gelu(a)",
    "silu": "torch.nn.functional.silu(a)",
    "all": "torch.all(a)",
    "any": "torch.any(a)",
    "amax": "torch.amax(a, 1)",
    "amin": "torch.amin(a, 1)",
    "min.dim": "torch.min(a, 1)",
    "sub.Tensor": "a - b",
    "pow.Tensor_Tensor": "a ** b",
    "where.self": "torch.where(a > 0, a, b)",
    "masked_fill.Tensor": "a.masked_fill(a > 0, zero)",
    "embedding": "torch.nn.functional.embedding(idx, w)",
    "index_add_": "x = a.clone()\nx.index_add_(0, idx, src)",
    "linspace": "torch.linspace(0.0, 1.0, 8, device=DEVICE)",
    "full": "torch.full((4, 4), 3.0, device=DEVICE)",
    "ones": "torch.ones((4, 4), device=DEVICE)",
}

#: Operators each platform's conf pins OFF the FlagGems path, because a FlagGems
#: kernel was measured to answer wrongly, to raise, or to refuse a dtype its own
#: implementation does not need. Every entry is read from the conf: the test
#: asserts the route is not a FlagGems backend, so a re-measurement that flips
#: one is a deliberate conf edit and not a silent change.
#:
#: Keyed by the accelerator the conf names. ``metax`` carries the set the MetaX
#: re-measurement and its differential survey landed (see the ``BOXING_GAP_NOTES
#: ["metax"]`` note in ``scripts/codegen/gen_vendor_confs.py`` for the per-op
#: diagnosis); ``dcu`` carries the four shapes ``tests/integration/ops/
#: test_dcu_flaggems_pinned_routes.py`` reproduces on hardware.
FLAGGEMS_PINNED_OFF: dict[str, tuple[str, ...]] = {
    "metax": (
        "binary_cross_entropy",
        "binary_cross_entropy.out",
        "hardtanh_backward",
        "igammac",
        "igammac.out",
        "special_modified_bessel_i0",
        "special_modified_bessel_i0.out",
        "_fake_quantize_learnable_per_tensor_affine_backward",
        "upsample_bilinear2d",
        "mul_.Tensor",
        "igammac_",
        "max_unpool3d",
        "cudnn_batch_norm_backward",
        "special_bessel_j0",
        "special_i1e",
        "special_i1e.out",
        "special_chebyshev_polynomial_w.out",
        "nansum.out",
        "lu_unpack.out",
        "linalg_matrix_exp.out",
        "_cdist_forward",
        "sum.out",
        "_compute_linear_combination",
        "_compute_linear_combination.out",
        "_fused_rms_norm",
        "igamma",
        "igamma_",
        "logit_backward",
        "special_shifted_chebyshev_polynomial_t",
        "slice.Tensor",
    ),
    "dcu": (
        "scatter_add_",
        "argmin",
        "mean",
        "mean.dim",
        "native_batch_norm",
    ),
}

#: The two seeded-drawout operators: an op whose boxing kernel is ATen's own
#: ``native_dropout``, which draws from a generator ``torch.manual_seed`` cannot
#: write on a CPU-torch wheel. Every day-0 boxing variant routes them to FlagGems
#: for that reason (see ``BOXING_VARIANT_FLAGGEMS_OPS`` in
#: ``scripts/codegen/gen_vendor_confs.py``), and the default confs do too.
#:
#: They are the one place where a conf's ``flaggems`` route is about the kernel
#: rather than coverage, and not every conf routes them there -- NVIDIA's default
#: conf sends ``native_dropout_backward`` to its own kernel. A count that folded
#: them in would read that difference as coverage. ``flaggems_route_count`` still
#: counts them, because the conf does route them; ``flaggems_coverage_count`` --
#: the number ``EXPECTED_FLAGGEMS_ROUTES`` holds and the conformance test
#: compares -- does not. Nor could it: neither op is in
#: ``FLAGGEMS_ROUTING_CALLS``, so ``test_flaggems_routing.py``, the test that
#: proves a route on device, never visits one.
SEEDED_DRAWOPS = frozenset({"native_dropout", "native_dropout_backward"})

#: Number of ``flaggems`` routes each platform's conf is measured to hold --
#: as ``flaggems_coverage_count`` counts them: ``= flaggems`` less a ``flaggems``
#: route for a ``SEEDED_DRAWOPS`` operator. A row is a number CI measured on the
#: lane's own conf; the raw tally is two higher on the confs that route both.
#:
#: A regression guard rather than an exact contract: the count only moves when an
#: operator is added, removed or re-measured, and each of those changes the
#: support report too. It is what keeps a broad reroute from being split into
#: narrow ones while every operator still reaches some backend -- the failure
#: mode a per-operator route assertion cannot see.
EXPECTED_FLAGGEMS_ROUTES: dict[str, int] = {
    "ascend": 221,
    "dcu": 452,
    "gcu": 251,
    # The iluvatar lane's landing: the conf routes 585 ops to FlagGems, eleven
    # to the native slot (six ixdnn/ixblas kernels -- `_convolution`, `addmm`,
    # `bmm`, `convolution`, `mm`, `native_batch_norm` -- plus `_conj`,
    # `view_as_real`, `add_.Scalar`, `add_.Tensor` and `div_.Tensor` served
    # from register.cc) and the rest to `none`. Recorded because the lane runs
    # this suite on the platform and the count is what the conf states -- it is
    # not an on-device measurement yet. 583 is that tally less the two
    # ``SEEDED_DRAWOPS`` routes, which CI run 40341912173 measured on the lane.
    "iluvatar": 583,
    "kunlunxin": 0,
    "metax": 584,
    # 0: the apple conf routes no operator to FlagGems -- its native MPSGraph
    # kernels serve what it accelerates and the rest reaches cpu_fallback.
    "apple": 0,
    "musa": 464,
    #: The two confs the PPU lane can run. `ppu` is the default
    #: backends_thead.conf, which a FlagGems-primary wheel reads; `ppu_boxing`
    #: is the day-0 variant the boxing wheel selects (FLAGOS_BACKEND_CONFIG),
    #: identical to it except that the ops without a FlagGems route go to
    #: `boxing`. Both counts are the conf's text less the two ``SEEDED_DRAWOPS``
    #: routes -- 590, which both PPU lanes of CI run 40341912173 measured.
    "ppu": 590,
    "ppu_boxing": 590,
    # The CUDA lane's own conf, and its day-0 variant. The default conf keeps
    # FlagGems for the 549 operators the CUDA generation measured it on; the
    # day-0 variant routes everything else to `boxing` -- on NVIDIA the boxed
    # CUDA kernel *is* the covered route -- and keeps FlagGems for exactly the
    # two seeded-drawout ops (native_dropout, native_dropout_backward): the
    # boxing kernel there is ATen's own draw, which torch.manual_seed cannot
    # reach on a CPU-torch wheel (run 40255168276). It was 0 before those two
    # moved, and it is 0 again for this suite's purposes: those are the
    # variant's only two routes, neither op is in ``FLAGGEMS_ROUTING_CALLS``,
    # and ``flaggems_coverage_count`` leaves them out (see ``SEEDED_DRAWOPS``).
    # The suite's FlagGems cases skip by the conf's own route on this lane,
    # which is the designed behaviour.
    #
    # All counts are `flaggems` only: `backends_metax.conf` also carries 15
    # `flaggems_cpp` routes and `backends_metax_boxing.conf` 12, which neither
    # metax number includes.
    "nvidia": 548,
    "nvidia_boxing": 0,
    # metax's day-0 conf keeps the 12 C++ routes and moves everything else to
    # the boxed CUDA kernel the C550 build dispatches to; the DCU conf keeps its
    # FlagGems coverage and boxes the rest. Counted from the shipped confs the
    # same way ``flaggems_route_count`` counts (``= flaggems`` only) less the two
    # ``SEEDED_DRAWOPS`` routes (583 - 2 on metax, and the same - 2 on DCU's 453:
    # the seeded ops reach FlagGems in the day-0 tables of every platform).
    "metax_boxing": 581,
    "dcu_boxing": 451,
    # TsingMicro's conf is the other hand-maintained one: 81 routes, of which 65
    # are the plain `flaggems` key, 2 spell it `flagos_python` (a log name this
    # conf uses as a route value) and 14 are metadata-view routes on the vendor.
    # No CI lane runs this suite on TsingMicro yet, but the row is what the conf
    # states, so the suite is correct the first time one does. `bpu` is not here:
    # its conf is empty and an unlisted op defaults to FlagGems, so it has no
    # route count to record.
    "tsingmicro": 65,
}

#: The route ``scaled_dot_product_attention`` is expected to carry, per platform.
#:
#: ``None`` records that the conf does not name the operator at all, which the
#: wrapper in ``csrc/aten/sdp_choice_stub.cc`` reads as "keep the boxed CUDA
#: composite" -- a deliberate state, so it is spelled out rather than left to be
#: inferred from a missing key. The operator is a composite whose route the
#: dispatcher does not take on its own, which is why its text is pinned here as
#: well as the leaf operators above.
EXPECTED_SDPA_ROUTE: dict[str, str | None] = {
    "ascend": "none",
    "dcu": "flaggems",
    "gcu": "none",
    # The same `none` the other FlagGems-primary non-boxing platforms record:
    # the composite reaches the privateuseone kernel on every platform, and on
    # this conf it has no FlagGems route to branch to.
    "iluvatar": "none",
    "kunlunxin": "none",
    "metax": "flaggems",
    "apple": "none",
    "musa": "none",
    # `none` on PPU: backends_thead.conf leaves the composite to the boxed CUDA
    # composite (`none` is cpu_fallback, which serves it on a wheel that does
    # not register it), and the day-0 variant names `boxing` for the same
    # kernel. Which arm a call takes on either is the operator tests' property,
    # not this table's.
    "ppu": "none",
    "ppu_boxing": "boxing",
    # The day-0 confs name the boxed CUDA kernel for the composite, the arm the
    # packaging wheel actually dispatches to.
    "nvidia": None,
    "nvidia_boxing": "boxing",
    "metax_boxing": "boxing",
    "dcu_boxing": "boxing",
    # `backends_tsingmicro.conf` does not name the composite, like the CUDA conf.
    "tsingmicro": None,
}

#: conf value -> the name ``[flagos dispatch]`` prints for the same backend. The
#: two vocabularies are kept apart on purpose (the log names are frozen for test
#: compatibility); this is the one place the mapping is owned.
_LOG_NAME = {
    "flaggems": "flagos_python",
    "flaggems_python": "flagos_python",
    "flagos_python": "flagos_python",
    "flaggems_cpp": "flagos",
    # A conf row in the vendor spelling, against the id the log prints. The
    # NVIDIA rows print "boxing" (the slot they register into).
    "apple": "mps",
    "mthreads": "musa",
    "enflame": "gcu",
    "nvidia": "boxing",
}

#: The backend names that mean "this operator took a FlagGems route".
FLAGGEMS_BACKENDS = frozenset({"flaggems", "flaggems_cpp", "flaggems_python"})


def conf_path() -> Path:
    """Path of the op-routing conf this wheel resolved."""
    import torch_fl

    return Path(torch_fl.backend_config_path())


def load_routes() -> dict[str, str]:
    """Every ``op -> backend`` route in the conf, as written (``none`` included).

    Parsed the same way the C++ reader does: drop anything after ``#`` and split
    the remainder on the first ``=``. The value is the raw conf backend, before
    the log-name mapping ``route_reason`` applies.
    """
    routes: dict[str, str] = {}
    with open(conf_path()) as f:
        for line in f:
            name, sep, value = line.split("#")[0].partition("=")
            if sep:
                routes[name.strip()] = value.strip()
    return routes


def accel_label() -> str:
    """The accelerator the conf names, from its basename.

    Read from ``backends_<platform>.conf`` rather than from ``detect_platform()``
    (or any vendor probe): it is the wheel's own statement of what it was built
    for, and it is the key the measurement tables above are stored under. A conf
    with a user-chosen path has no row and the data-driven tests skip it.

    The label is the conf's basename, so the tables are keyed by the same string
    the conf ships under. Five confs are named after the vendor while the tables
    record the platform id (``backends_thead.conf`` is ``ppu``, ``hygon`` is
    ``dcu``, ``enflame`` is ``gcu``, ``mthreads`` is ``musa``, ``nvidia`` is
    ``cuda``), so a basename that is not already a row is looked up through
    ``torch_fl._platform.platform_for``. A day-0 variant
    (``backends_<platform>_boxing.conf``) is the same platform's other conf and
    is reported as ``<platform id>_boxing``.
    """
    name = conf_path().name
    if not (name.startswith("backends_") and name.endswith(".conf")):
        return ""
    label = name[len("backends_") : -len(".conf")]
    if label.endswith("_boxing"):
        platform = label[: -len("_boxing")]
        return f"{_platform_id(platform)}_boxing"
    return _platform_id(label)


def _platform_id(base: str) -> str:
    """``base`` as the platform id the tables are keyed by.

    Accepts either half of a platform's name: a conf basename that is also a
    platform id (``ascend``, ``metax``, ``iluvatar``, ...) is returned unchanged,
    and a vendor spelling (``thead``) resolves to its id (``ppu``). Unknown names
    come back as given, so the data-driven tests report them as missing rows.
    """
    if base in EXPECTED_FLAGGEMS_ROUTES:
        return base
    from torch_fl._platform import platform_for

    return platform_for(base) or base


def daily_variant_label(base: str) -> str | None:
    """``base``'s day-0 variant label when the tables carry one, else None."""
    label = f"{base}_boxing"
    return label if label in EXPECTED_FLAGGEMS_ROUTES else None


def route_reason(op: str) -> tuple[str | None, str]:
    """The backend ``op`` dispatches to, plus why it resolved that way.

    The backend is named the way ``[flagos dispatch]`` names it. The reason is
    ``conf_route_none`` when the conf hands the operator to cpu_fallback (no
    dispatch line is emitted for such an operator), ``unlisted`` when the conf
    does not name it at all, and ``routed`` otherwise.
    """
    raw = load_routes().get(op)
    if raw is None:
        return None, "unlisted"
    if raw == "none":
        return None, "conf_route_none"
    return _LOG_NAME.get(raw, raw), "routed"


def conf_route(op: str) -> str | None:
    """The raw conf value for ``op``, or ``None`` when the conf omits it."""
    return load_routes().get(op)


def routes_to_flaggems(op: str) -> bool:
    """Whether the conf sends ``op`` to a FlagGems backend."""
    return conf_route(op) in FLAGGEMS_BACKENDS


def build_kernels() -> frozenset[str]:
    """Kernel sets compiled into this wheel, from the build record.

    Names are setup.py's (``vendor``, ``flaggems``, ``boxing``, ``flaggems_cpp``,
    ``tileops``); empty for an unbuilt source checkout.
    """
    from torch_fl import _env

    return _env.build_kernels()


def flaggems_kernel_compiled() -> bool:
    """True when the FlagGems Python kernel set is in this wheel's build record."""
    return "flaggems" in build_kernels()


def flaggems_route_count() -> int:
    """How many operators this conf routes with the plain ``flaggems`` key."""
    return sum(1 for value in load_routes().values() if value == "flaggems")


def flaggems_coverage_count() -> int:
    """``flaggems_route_count()`` less the routes the call table cannot bill.

    A ``SEEDED_DRAWOPS`` route is a statement about the kernel (the boxing kernel
    cannot honour a seed), not about how much of the model FlagGems covers, and
    no call in ``FLAGGEMS_ROUTING_CALLS`` exercises a dropout -- so
    ``test_flaggems_routing.py``, which is what proves a route on device, never
    visits one. This is the set it can reach: the conformance guard compares
    this number, and the raw count stays available for the conf's own tally.
    """
    routes = load_routes()
    return sum(
        1
        for op, value in routes.items()
        if value == "flaggems" and op not in SEEDED_DRAWOPS
    )


def flaggems_runtime_unavailable() -> str:
    """Why a FlagGems kernel cannot run here, or ``""`` when it can.

    Three facts, none of them a platform name: the build record has to carry the
    kernel set, a flagos device has to exist, and the FlagGems package plus its
    Triton compiler have to import in this interpreter. Anything else -- a vendor
    Triton that raises on the first tile, an operator whose FlagGems entry point
    does not exist -- is what the execution cases are for and is reported as a
    failure, not hidden here.
    """
    import torch_fl

    if not flaggems_kernel_compiled():
        return "this wheel was not built with the FlagGems kernel set"
    if torch_fl.flagos.device_count() < 1:
        return "no flagos device is available"
    try:
        import triton  # noqa: F401
        import flag_gems  # noqa: F401
    except Exception as exc:  # noqa: BLE001 -- any import failure means skip
        return f"the FlagGems runtime is not importable: {exc}"
    return ""
