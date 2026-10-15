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

#: Number of ``flaggems`` routes each platform's conf is measured to hold.
#:
#: A regression guard rather than an exact contract: the count only moves when an
#: operator is added, removed or re-measured, and each of those changes the
#: support report too. It is what keeps a broad reroute from being split into
#: narrow ones while every operator still reaches some backend -- the failure
#: mode a per-operator route assertion cannot see.
EXPECTED_FLAGGEMS_ROUTES: dict[str, int] = {
    "ascend": 221,
    "cuda": 392,
    "dcu": 454,
    "gcu": 251,
    "kunlunxin": 0,
    "metax": 594,
    "musa": 466,
    "ppu": 592,
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
    "cuda": None,
    "dcu": "flaggems",
    "gcu": "none",
    "kunlunxin": "none",
    "metax": "flaggems",
    "musa": "none",
    "ppu": "cuda",
}

#: conf value -> the name ``[flagos dispatch]`` prints for the same backend. The
#: two vocabularies are kept apart on purpose (the log names are frozen for test
#: compatibility); this is the one place the mapping is owned.
_LOG_NAME = {
    "flaggems": "flagos_python",
    "flaggems_python": "flagos_python",
    "flagos_python": "flagos_python",
    "flaggems_cpp": "flagos",
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
    """
    name = conf_path().name
    if name.startswith("backends_") and name.endswith(".conf"):
        return name[len("backends_") : -len(".conf")]
    return ""


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
