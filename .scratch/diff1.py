import torch

d = torch.device("flagos")
R = torch.randn


def cmp(name, fn, *args):
    try:
        host = fn(*[a for a in args])
        dev = fn(*[a.to(d) if isinstance(a, torch.Tensor) else a for a in args])
        dc = dev.cpu() if isinstance(dev, torch.Tensor) else None
        if dc is None:
            return (name, "skip", "")
        if dc.shape != host.shape:
            return (name, "SHAPE", f"{tuple(dc.shape)} vs {tuple(host.shape)}")
        if dc.dtype != host.dtype:
            return (name, "DTYPE", f"{dc.dtype} vs {host.dtype}")
        # bit-identical for integer/bool, tolerance for float
        if dc.dtype == torch.bool or dc.is_floating_point() is False:
            ok = torch.equal(dc, host)
        else:
            ok = torch.allclose(
                dc.float(), host.float(), rtol=2e-2, atol=2e-2, equal_nan=True
            )
        return (
            name,
            "OK" if ok else "MISMATCH",
            ""
            if ok
            else f"maxdiff={(dc.float() - host.float()).abs().max().item():.3e}",
        )
    except Exception as e:
        return (name, "ERROR", f"{type(e).__name__}: {str(e)[:60]}")


cases = []
T = torch.tensor
# reductions with keepdim variants
for kd in (False, True):
    for f, nm in (
        (torch.sum, "sum"),
        (torch.prod, "prod"),
        (torch.mean, "mean"),
        (torch.amax, "amax"),
        (torch.amin, "amin"),
        (torch.max, "max"),
        (torch.min, "min"),
    ):
        cases.append(
            (
                f"sum/mean dim kd={kd}",
                lambda x, f=f, kd=kd: f(x, 1, keepdim=kd),
                R(2, 5, 4),
            )
        )
# softmax / log_softmax over every dim
for dim in range(3):
    cases.append(
        (f"softmax dim={dim}", lambda x, dim=dim: torch.softmax(x, dim), R(2, 5, 4))
    )
    cases.append(
        (
            f"log_softmax dim={dim}",
            lambda x, dim=dim: torch.log_softmax(x, dim),
            R(2, 5, 4),
        )
    )
# argmax/argmin/all/any
for dim in (0, 1, None):
    cases.append(
        (
            f"argmax dim={dim}",
            lambda x, dim=dim: (
                torch.argmax(x, dim) if dim is not None else torch.argmax(x)
            ),
            R(3, 4),
        )
    )
    cases.append(
        (
            f"all dim={dim}",
            lambda x, dim=dim: (
                torch.all(x > 0, dim) if dim is not None else torch.all(x > 0)
            ),
            R(3, 4),
        )
    )
    cases.append(
        (
            f"any dim={dim}",
            lambda x, dim=dim: (
                torch.any(x > 0, dim) if dim is not None else torch.any(x > 0)
            ),
            R(3, 4),
        )
    )
# clamp family, which had the stale-plan bug
for mn, mx in [(0.0, 1.0), (-1.0, 1.0), (0.5, 0.5)]:
    cases.append(
        (f"clamp({mn},{mx})", lambda x, mn=mn, mx=mx: torch.clamp(x, mn, mx), R(3, 4))
    )
    cases.append(
        (
            f"clamp_int({mn},{mx})",
            lambda x, mn=mn, mx=mx: torch.clamp(x, mn, mx),
            torch.randint(-3, 3, (3, 4)),
        )
    )
cases.append(("clamp_min", lambda x: torch.clamp_min(x, 0.0), R(3, 4)))
cases.append(("clamp_max", lambda x: torch.clamp_max(x, 0.0), R(3, 4)))
cases.append(
    ("clamp nan", lambda x: torch.clamp(x, 0.0, 1.0), T([float("nan"), -1.0, 0.5, 2.0]))
)
# binary with broadcasting and scalars
for nm, f in (
    ("add", "add"),
    ("sub", "sub"),
    ("mul", "mul"),
    ("div", "div"),
    ("pow", "pow"),
    ("fmod", "fmod"),
    ("remainder", "remainder"),
):
    cases.append((f"{nm} .Scalar", lambda x, f=f: getattr(torch, f)(x, 2.5), R(3, 4)))
    cases.append(
        (f"{nm} .Tensor same", lambda x, f=f: getattr(torch, f)(x, x * 0.5), R(3, 4))
    )
    cases.append(
        (
            f"{nm} .Tensor bcast",
            lambda x, f=f: getattr(torch, f)(x, torch.randn(4) + 2.0),
            R(3, 4),
        )
    )
# comparisons
for nm, f in (
    ("eq", "eq"),
    ("ne", "ne"),
    ("lt", "lt"),
    ("le", "le"),
    ("gt", "gt"),
    ("ge", "ge"),
):
    cases.append((f"{nm}.Scalar", lambda x, f=f: getattr(torch, f)(x, 0.0), R(3, 4)))
    cases.append(
        (f"{nm}.Tensor", lambda x, f=f: getattr(torch, f)(x, R(3, 4)), R(3, 4))
    )
# max/min binary, which had a NaN bug
cases.append(
    (
        "maximum nan",
        lambda a, b: torch.maximum(a, b),
        T([float("nan"), 1.0]),
        T([0.0, 2.0]),
    )
)
cases.append(
    (
        "minimum nan",
        lambda a, b: torch.minimum(a, b),
        T([float("nan"), 1.0]),
        T([0.0, 2.0]),
    )
)
# unary sweep
U = [
    "abs",
    "neg",
    "sign",
    "sqrt",
    "rsqrt",
    "reciprocal",
    "exp",
    "log",
    "exp2",
    "log2",
    "log10",
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "sinh",
    "cosh",
    "tanh",
    "asinh",
    "acosh",
    "atanh",
    "erf",
    "ceil",
    "floor",
    "trunc",
    "round",
    "sigmoid",
    "logical_not",
    "relu",
    "silu",
    "gelu",
    "leaky_relu",
    "elu",
    "softplus",
    "hardsigmoid",
    "hardswish",
    "mish",
    "selu",
    "logsigmoid",
]
for u in U:
    f = getattr(torch, u, None)
    if f is None:
        continue
    x = (
        torch.rand(3, 4) + 0.5
        if u
        in (
            "log",
            "log2",
            "log10",
            "sqrt",
            "rsqrt",
            "reciprocal",
            "acosh",
            "atanh",
            "erf",
            "mish",
            "selu",
        )
        else R(3, 4)
    )
    cases.append((f"{u}", lambda x, f=f: f(x), x))

bad = []
for name, fn, *args in cases:
    r = cmp(name, fn, *args)
    if r[1] not in ("OK", "skip"):
        bad.append(r)
print(f"{len(cases)} cases, {len(bad)} not OK")
for name, st, det in bad:
    print(f"  {st:9s} {name:26s} {det}")
