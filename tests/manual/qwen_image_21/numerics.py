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

"""Per-operator check against a reference backend, for a chip that is being brought up.

``sweep.py`` tells you whether the model produces a picture. This tells you
*which operator* is wrong when it does not, and it does so before the model is
worth running: it is the cheapest way to find a mis-routed or numerically wrong
operator, and its output is a file you can diff.

Three checks per operator, because they fail independently:

**Numerics** -- run the op on this backend and on the reference, compare max
absolute and mean relative error. Catches a wrong kernel, a wrong dtype
promotion, or a route that reaches an implementation with different semantics.

**Aliasing** -- for the view/reshape family, whether the result shares storage
with its input. This is the check that caught ``_unsafe_view`` on torch_fl:
``flag_gems`` implements it as ``reshape``, which returns a materialised copy
where ATen returns a strided view or raises, and **the values are identical**, so
no numerics comparison can see it. A view that copies is also a silent 2x on
that tensor's memory.

**Cost** -- host microseconds per call on a tiny replica of the call, which is
what separates "this kernel is slower here" from "this call costs more to
issue". The second is the failure mode of a dispatch layer, and it does not show
up in any device-time measurement.

Inputs are built on the CPU from a fixed seed and moved to the device, never
drawn on the device: two backends do not share an RNG stream, so a device-side
``randn`` would give the two runs *different inputs* and every operator would
"differ" for the wrong reason. That mistake cost an afternoon once.

Run it on the reference (any working backend -- a vendor torch, or the CPU) and
on the chip, into separate files, then pass both:

    python tests/manual/qwen_image_21/numerics.py --device cuda   --out ref.pt
    python tests/manual/qwen_image_21/numerics.py --device flagos --out chip.pt
    python tests/manual/qwen_image_21/numerics.py --compare ref.pt chip.pt

Procedure and what a pass looks like: tests/manual/qwen_image_21/README.md
"""

import argparse
import sys

import common

# The value check runs at the shapes Qwen-Image-2.1 uses at 1024x1024, so a
# mismatch here is one the model would have hit. SEQ is the packed sequence the
# transformer sees (4096 = (128/2)^2), HIDDEN the text encoder's width, FFN its
# feed-forward width, and VAE_CHANNELS the VAE decoder's last base_dim x dim_mult.
REAL = {"seq": 4096, "hidden": 4096, "ffn": 12288, "vae": 1152, "spatial": 64}

# The cost check runs on a collapse of the same call at the same dtype, because
# it issues it hundreds of times and the device is not what is being measured:
# with the queue absorbing the work the loop is bounded by the host, and the
# device never becomes the limiter as long as the tensors are small. Repeating
# the *real* shape instead would spend minutes per operator -- a 1024x1024 3-D
# convolution is about half a second a call, and 1500 of them is 12 minutes.
TINY = {"seq": 8, "hidden": 64, "ffn": 128, "vae": 32, "spatial": 4}

# A relative error above this is not bf16 rounding. bf16 has ~3 decimal digits,
# and the ops below are one or two accumulation steps, so even a chained
# reduction lands well under 1e-2.
TOLERANCE = 5e-3


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Per-operator reference comparison for Qwen-Image-2.1",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--device", default=common.DEFAULT_DEVICE)
    parser.add_argument("--out", default=None, help="write the results to this file")
    parser.add_argument(
        "--reps", type=int, default=300, help="calls per host-cost sample"
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("REFERENCE", "CANDIDATE"),
        default=None,
        help="compare two result files instead of running anything; no device needed",
    )
    return parser.parse_args(argv)


def cases(torch, dev, scale):
    """(name, callable) for the operators the 2.1 pipeline reaches.

    Called twice: once at ``REAL`` scale for the value comparison, once at
    ``TINY`` for the host-cost one. Building both from one function is what keeps
    the two lists from drifting -- the cost of an operator and the value of an
    operator have to be the *same* call, or the cost column measures something
    the check above never looked at.

    Grouped by what a failure means, because that is what a bring-up needs: the
    elementwise family is a routing or dtype question, the views are a contract
    question, and ``mm`` / attention / the VAE convolution are the three kernels
    big enough for a performance difference to matter.
    """
    dt = torch.bfloat16
    seq, hidden, ffn, vae, spatial = (
        scale["seq"],
        scale["hidden"],
        scale["ffn"],
        scale["vae"],
        scale["spatial"],
    )
    # One generator, reset per call site, so both backends start from the same
    # bytes: a device-side randn would give the two runs different inputs.
    gen = torch.Generator().manual_seed(1234)

    def cpu(*shape, dtype=torch.float32):
        return torch.randn(*shape, dtype=dtype, generator=gen)

    def t(*shape, dtype=dt):
        return cpu(*shape).to(device=dev, dtype=dtype)

    x = t(1, seq, hidden)
    y = t(1, seq, hidden)
    ffn_x = t(1, seq, ffn)
    flat = t(seq, hidden)
    norm_w = t(hidden)
    norm_b = t(hidden)
    heads = t(1, seq, 2, min(128, hidden // 2))
    # bmm is the KV-cache matmul, which is genuinely 3-D (batched over heads),
    # unlike `heads` above which keeps the 4-D attention layout for `permute`.
    batch = t(2, max(1, seq // 64), min(64, hidden))
    batch_t = t(2, min(64, hidden), max(1, seq // 64))
    vae_act = t(1, vae, 1, spatial, spatial)
    vae_w = t(vae, vae, 3, 3, 3)
    small = t(1, seq // 4, min(64, hidden))
    # A mask that is genuinely both-valued. `randn` cast to bool is all True --
    # a float is False only when it is exactly 0.0 -- so `where` would return
    # the x branch everywhere and `masked_fill` would be a no-op, and a backend
    # with a broken false branch would agree with the reference by never
    # reaching it. Compared against zero, the generator's own reproducibility
    # still makes both backends start from the same bytes.
    mask = (cpu(1, seq, hidden) > 0).to(device=dev)
    idx = torch.arange(max(1, seq // 64), device=dev, dtype=torch.long)
    # The complex-dtype rotary embeddings: view_as_complex / view_as_real are
    # what the encoder's RoPE reaches, and the pair must round-trip exactly.
    complex_src = cpu(1, seq, 4, 8, 2).to(device=dev, dtype=torch.float32)

    return [
        # -- elementwise / reduction: the routing and dtype family --
        ("add.Tensor", lambda: x + y),
        ("mul.Tensor", lambda: x * y),
        ("div.Tensor", lambda: x / (y.abs() + 1)),
        ("silu", lambda: torch.nn.functional.silu(ffn_x)),
        ("gelu", lambda: torch.nn.functional.gelu(ffn_x)),
        ("tanh", lambda: torch.tanh(x)),
        ("sigmoid", lambda: torch.sigmoid(x)),
        ("exp", lambda: torch.exp(x)),
        ("neg", lambda: -x),
        ("rsqrt", lambda: torch.rsqrt(x.abs() + 1)),
        ("pow.Tensor_Scalar", lambda: x.pow(3)),
        ("mean.dim", lambda: torch.mean(x, dim=-1)),
        ("sum.dim_IntList", lambda: torch.sum(x, dim=-1)),
        ("linalg_vector_norm", lambda: torch.linalg.vector_norm(x, dim=-1)),
        ("where.self", lambda: torch.where(mask, x, y)),
        ("masked_fill", lambda: x.masked_fill(mask, 0.0)),
        ("cumsum", lambda: torch.cumsum(small, dim=1)),
        ("index_select", lambda: torch.index_select(small, 1, idx)),
        ("repeat_interleave.Tensor", lambda: torch.repeat_interleave(small, 2, dim=1)),
        ("cat", lambda: torch.cat([x, x], dim=1)),
        ("stack", lambda: torch.stack([x, x], dim=0)),
        # -- views: the contract family. The values match even when one of these
        #    is wrong, which is why aliasing is checked separately below.
        ("unsqueeze", lambda: x.unsqueeze(0)),
        ("squeeze.dim", lambda: x.unsqueeze(0).squeeze(0)),
        ("transpose.int", lambda: x.transpose(0, 1)),
        ("permute", lambda: heads.permute(0, 2, 1, 3)),
        ("_unsafe_view", lambda: torch.ops.aten._unsafe_view(x, [1, seq, hidden])),
        ("select.int", lambda: x.select(0, 0)),
        ("slice.Tensor", lambda: x[:, 1:]),
        ("split.Tensor", lambda: torch.split(x, max(1, seq // 4), dim=1)),
        ("view_as_complex", lambda: torch.view_as_complex(complex_src)),
        (
            "view_as_real",
            lambda: torch.view_as_real(torch.view_as_complex(complex_src)),
        ),
        # -- the kernels big enough for a performance difference to matter --
        ("mm", lambda: flat @ flat.T),
        ("bmm", lambda: torch.bmm(batch, batch_t)),
        ("softmax", lambda: torch.softmax(ffn_x, dim=-1)),
        # Returns (out, mean, rstd); the value check takes the first, which is
        # the one that reaches the next layer.
        (
            "native_layer_norm",
            lambda: torch.native_layer_norm(x, [hidden], norm_w, norm_b, 1e-6),
        ),
        ("conv3d-vae", lambda: torch.nn.functional.conv3d(vae_act, vae_w, padding=1)),
    ]


# The ops whose result must share storage with its input. Checked by comparing
# the underlying storage's data pointer, which is the only way to see it: the
# values are identical either way.
VIEW_OPS = {
    "unsqueeze": lambda torch, dev: (lambda a: (a, a.unsqueeze(0)))(
        torch.arange(32, device=dev).reshape(8, 4).t()
    ),
    "transpose.int": lambda torch, dev: (lambda a: (a, a.transpose(0, 1)))(
        torch.arange(32, device=dev).reshape(8, 4)
    ),
    "select.int": lambda torch, dev: (lambda a: (a, a.select(0, 0)))(
        torch.arange(32, device=dev).reshape(8, 4)
    ),
    "slice.Tensor": lambda torch, dev: (lambda a: (a, a[:, 1:]))(
        torch.arange(32, device=dev).reshape(8, 4)
    ),
    "squeeze.dim": lambda torch, dev: (lambda a: (a, a.unsqueeze(0).squeeze(0)))(
        torch.arange(32, device=dev).reshape(8, 4)
    ),
    "_unsafe_view.alias": lambda torch, dev: (
        lambda a: (a, torch.ops.aten._unsafe_view.default(a, [4, 2, 4]))
    )(torch.arange(32, device=dev).reshape(8, 4).t()),
    # ATen rejects flattening this transpose because no strided view can express
    # it. An implementation backed by reshape may instead materialise a copy;
    # comparing the error contract is what catches that semantic difference.
    "_unsafe_view.error": lambda torch, dev: (
        lambda a: (a, torch.ops.aten._unsafe_view.default(a, [32]))
    )(torch.arange(32, device=dev).reshape(8, 4).t()),
}


def host_cost(fn, reps):
    """Median host microseconds to *issue* one call, with no synchronisation.

    The device queue absorbs the work, so the host runs ahead and the loop is
    bounded by how fast the host can issue. For an operator the device is busy
    on, this is a floor rather than a total -- but it is exactly the number a
    dispatch layer changes, and the one that says whether a step is
    host-bound. Measured on A100: the FlagGems Python route costs 98 us to
    issue ``mul`` where CUDA boxing costs 9.7, against 102 us of device time --
    which is why that route is host-bound and the boxing one is not.
    """
    import time

    for _ in range(10):
        fn()
    samples = []
    for _ in range(5):
        start = time.perf_counter()
        for _ in range(reps):
            fn()
        samples.append((time.perf_counter() - start) / reps * 1e6)
    return round(sorted(samples)[len(samples) // 2], 2)


def run(args):
    torch = common.import_torch(args.device)
    dev = f"{args.device}:0"
    results = {
        "device": args.device,
        "torch": torch.__version__,
        "ops": {},
        "views": {},
    }

    real = cases(torch, dev, REAL)
    tiny = dict(cases(torch, dev, TINY))

    print(f"device: {args.device}   torch: {torch.__version__}")
    print(
        f"values at {REAL['seq']}x{REAL['hidden']}, cost at {TINY['seq']}x{TINY['hidden']}"
    )
    print(f"{'op':<26}{'cost us':>10}   note")
    for name, fn in real:
        try:
            with torch.no_grad():
                out = fn()
            if isinstance(out, (tuple, list)):
                out = out[0]
            cost_fn = tiny[name]
            with torch.no_grad():
                cost = host_cost(cost_fn, args.reps)
            results["ops"][name] = {
                "value": out.detach().float().cpu(),
                "shape": list(out.shape),
                "cost_us": cost,
            }
            print(f"{name:<26}{cost:>10.2f}")
        except Exception as exc:
            results["ops"][name] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"{name:<26}{'-':>10}   FAILED {type(exc).__name__}: {str(exc)[:60]}")

    # Aliasing, separately from the values.
    for name, build in VIEW_OPS.items():
        try:
            src, view = build(torch, dev)
            shared = (
                src.untyped_storage().data_ptr() == view.untyped_storage().data_ptr()
            )
            results["views"][name] = {"aliases": bool(shared)}
        except Exception as exc:
            results["views"][name] = {"error": f"{type(exc).__name__}: {exc}"}
            print(
                f"view {name:<21}{'':>10}   FAILED {type(exc).__name__}: "
                f"{str(exc)[:60]}"
            )

    # Counted separately, and both halves printed: a view that raised is not an
    # aliasing check that ran, and reporting it as one overstates the coverage
    # on this backend. `--compare` is what judges the candidate's errors.
    ops_ran = sum(1 for entry in results["ops"].values() if "error" not in entry)
    views_ran = sum(1 for entry in results["views"].values() if "aliases" in entry)
    print(
        f"\n{ops_ran}/{len(results['ops'])} ops ran, "
        f"{views_ran}/{len(results['views'])} aliasing checks ran"
    )
    if args.out:
        torch.save(results, args.out)
        print(f"wrote {args.out}")
    return 0


def compare(paths):
    """Diff two result files and say what disagreed."""
    import torch

    reference = torch.load(paths[0], weights_only=False)
    candidate = torch.load(paths[1], weights_only=False)

    print(f"reference: {reference['device']} (torch {reference['torch']})")
    print(f"candidate: {candidate['device']} (torch {candidate['torch']})")
    print()
    print(f"{'op':<26}{'max_abs':>12}{'rel':>12}   verdict")

    failed, errors = [], []
    for name, want in reference["ops"].items():
        got = candidate["ops"].get(name)
        if got is None:
            print(f"{name:<26}{'-':>12}{'-':>12}   MISSING")
            failed.append(name)
            continue
        if "error" in want or "error" in got:
            if "error" in got:
                print(f"{name:<26}{'-':>12}{'-':>12}   ERROR  {got['error'][:60]}")
                errors.append(name)
            else:
                print(f"{name:<26}{'-':>12}{'-':>12}   (reference errored, skipped)")
            continue
        if want["shape"] != got["shape"]:
            print(
                f"{name:<26}{'-':>12}{'-':>12}   SHAPE {want['shape']} vs {got['shape']}"
            )
            failed.append(name)
            continue
        diff = (got["value"] - want["value"]).abs()
        scale = want["value"].abs().mean().item() or 1.0
        rel = diff.mean().item() / scale
        ok = rel < TOLERANCE
        if not ok:
            failed.append(name)
        print(
            f"{name:<26}{diff.max().item():>12.5f}{rel:>12.2e}   "
            f"{'ok' if ok else 'DIFFERS'}"
        )

    print()
    view_bad = []
    view_skipped = 0
    view_compared = 0
    for name, want in reference["views"].items():
        got = candidate["views"].get(name)
        if got is None:
            view_bad.append(name)
            print(f"view {name:<21}{'':>12}{'':>12}   MISSING on candidate")
            continue

        want_error = want.get("error")
        got_error = got.get("error")
        if want_error or got_error:
            want_type = want_error.split(":", 1)[0] if want_error else None
            got_type = got_error.split(":", 1)[0] if got_error else None
            if want_type != got_type:
                view_bad.append(name)
                print(
                    f"view {name:<21}{'':>12}{'':>12}   ERROR CONTRACT "
                    f"{want_type} vs {got_type}"
                )
            continue

        if want.get("aliases") != got.get("aliases"):
            view_bad.append(name)
            print(
                f"view {name:<21}{'':>12}{'':>12}   ALIASING "
                f"{want.get('aliases')} vs {got.get('aliases')}"
            )
        else:
            view_compared += 1
    if not view_bad:
        print(
            f"view aliasing: all {view_compared} agree"
            + (f", {view_skipped} skipped" if view_skipped else "")
        )

    print()
    if errors:
        print(f"errors on the candidate: {errors}")
    print(f"numerically differing ops: {failed or 'none'}")
    return 1 if (failed or view_bad or errors) else 0


def main(argv=None):
    args = parse_args(argv)
    if args.compare:
        return compare(args.compare)
    if not args.out:
        raise SystemExit("--out is required unless --compare is given")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
