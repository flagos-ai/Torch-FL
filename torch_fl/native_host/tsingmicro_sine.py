# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Stage the published sine with accurate small residuals on TsingMicro."""


def render(published, _root):
    """Reduce near integer multiples of pi before rounding the sine value."""
    text = published.decode("utf-8")
    original = "    return tl.sin(x.to(tl.float32))"
    if text.count(original) != 1:
        raise ValueError("unexpected published FlagGems sine body")
    # Splitting pi keeps the subtraction accurate near a multiple of pi.
    accurate = """    x = x.to(tl.float32)
    bounded = tl.where(tl.abs(x) <= 256.0, x, 0.0)
    magnitude = (tl.abs(bounded) * 0.3183098861837907 + 0.5).to(tl.int32)
    index = tl.where(bounded < 0.0, -magnitude, magnitude)
    k = index.to(tl.float32)
    residual = (bounded - k * 3.140625) - k * 0.0009676535897932385
    square = residual * residual
    small = residual * (1.0 + square * (-0.16666666666666666 + square * 0.008333333333333333))
    signed = tl.where((index & 1) == 0, small, -small)
    eligible = (tl.abs(x) <= 256.0) & (tl.abs(residual) <= 0.125)
    return tl.where(eligible, signed, tl.sin(x))"""
    return text.replace(original, accurate)
