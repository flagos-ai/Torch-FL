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

"""Compare two images produced by the Qwen-Image-2512 scripts.

PSNR and mean absolute error, in 8-bit units. Bitwise equality is not a
meaningful target: bf16 accumulation order differs between backends, and the
error compounds over 50 steps of iterative refinement.

What the numbers mean depends on how the two images were made, and the
distinction is not academic. Two runs that were each seeded independently have
different noise, so the comparison measures the RNG streams rather than the
backends -- expect a different picture and a PSNR under 10 dB even when both
backends are perfectly correct. The backend comparison injects the same latents
and prompt embeddings on both sides (`infer.py --save-inputs` / `--load-inputs`),
which removes the generator from the question; that is the run whose PSNR means
something.

Run, from the repository root; the script imports no torch, so any interpreter
will do:
    python tests/manual/qwen_image_2512/compare.py --a ref.png --b out.png

Procedure: tests/manual/qwen_image_2512/README.md
"""

import argparse
import sys

import numpy as np
from PIL import Image


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    return parser.parse_args(argv)


def load(path):
    with Image.open(path) as handle:
        return np.asarray(handle.convert("RGB"), dtype=np.float64)


def main(argv=None):
    args = parse_args(argv)
    a = load(args.a)
    b = load(args.b)

    if a.shape != b.shape:
        print(f"FAIL: shape mismatch {a.shape} vs {b.shape}")
        return 1

    diff = np.abs(a - b)
    mae = float(diff.mean())
    mse = float((diff**2).mean())
    if mse == 0.0:
        psnr = float("inf")
    else:
        psnr = float(10.0 * np.log10((255.0**2) / mse))

    print(f"a    : {args.a}")
    print(f"b    : {args.b}")
    print(f"shape: {a.shape}")
    print(f"MAE  : {mae:.4f}")
    print(f"PSNR : {psnr:.2f} dB")
    print(f"max  : {diff.max():.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
