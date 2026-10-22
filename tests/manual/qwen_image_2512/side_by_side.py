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

"""Lay two generations of the same prompt side by side, for the eye.

Each run writes its own directory -- ``<root>/<device>/`` -- so a sweep on the
vendor's own torch and a sweep on flagos never overwrite each other. This script
pairs the two directories image by image and writes ``<root>/compare/``, one
sheet per prompt with a timing bar, plus ``all.png`` holding every pair at once.

Timings and the device names come from each directory's ``manifest.json``; a
directory without one still pairs, it just has no timing to show. The prompt is
not drawn on the sheet: the model card's prompts are mostly CJK, PIL's built-in
font cannot render them, and the label is there to say which side is which.

Run:
    python tests/manual/qwen_image_2512/side_by_side.py \
        --root qwen-image-out --left cuda --right flagos

`--left` and `--right` take either a name under `--root` or a path, so a
reference directory kept somewhere else works too. The pairs are written in the
order of `--left`'s files.
"""

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

BAR = 26  # the caption strip above each image, in pixels
PAIR = (700, 390)  # one image in a pair sheet
CONTACT = (640, 186)  # one pair in all.png
CONTACT_COLUMNS = 2
GAP = 6


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Pair two generation directories into comparison sheets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--root", default=".", help="the directory to write compare/ into"
    )
    parser.add_argument("--left", required=True, help="directory or name under --root")
    parser.add_argument("--right", required=True, help="directory or name under --root")
    parser.add_argument("--output", default=None, help="default: <root>/compare")
    return parser.parse_args(argv)


def resolve(root, name):
    """A name under --root, or a path in its own right."""
    candidate = Path(name)
    if candidate.is_dir():
        return candidate
    candidate = Path(root) / name
    if candidate.is_dir():
        return candidate
    raise SystemExit(f"{name!r} is neither a directory nor a name under {root!r}")


def read_manifest(directory):
    """The per-image timings, or an empty map when the directory has no manifest."""
    path = directory / "manifest.json"
    if not path.is_file():
        return {}
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return {entry["file"]: entry for entry in manifest.get("images", [])}


def caption(directory, name, entry):
    device = directory.name
    if entry:
        return f"{name}  {device}  {entry['seconds']}s"
    return f"{name}  {device}"


def main(argv=None):
    args = parse_args(argv)
    left_dir = resolve(args.root, args.left)
    right_dir = resolve(args.root, args.right)

    left_manifest = read_manifest(left_dir)
    right_manifest = read_manifest(right_dir)

    names = sorted(p.name for p in left_dir.glob("*.png"))
    if not names:
        raise SystemExit(f"no PNGs in {left_dir}")

    compare_dir = Path(args.output) if args.output else Path(args.root) / "compare"
    compare_dir.mkdir(parents=True, exist_ok=True)

    pairs = []
    for name in names:
        right_path = right_dir / name
        if not right_path.is_file():
            print(f"  skip {name}: not in {right_dir}")
            continue

        sheet = Image.new("RGB", (PAIR[0] * 2 + GAP, PAIR[1] + BAR), "black")
        sheet.paste(Image.open(left_dir / name).resize(PAIR), (0, BAR))
        sheet.paste(Image.open(right_path).resize(PAIR), (PAIR[0] + GAP, BAR))

        draw = ImageDraw.Draw(sheet)
        draw.text(
            (6, 7), caption(left_dir, name, left_manifest.get(name)), fill="white"
        )
        draw.text(
            (PAIR[0] + GAP + 6, 7),
            caption(right_dir, name, right_manifest.get(name)),
            fill="white",
        )

        out = compare_dir / name
        sheet.save(out)
        pairs.append(out)
        print(f"  {out}")

    if not pairs:
        raise SystemExit("nothing to compare")

    # One sheet with every pair, so a whole sweep can be judged at a glance.
    rows = (len(pairs) + CONTACT_COLUMNS - 1) // CONTACT_COLUMNS
    contact = Image.new(
        "RGB", (CONTACT[0] * CONTACT_COLUMNS, CONTACT[1] * rows), "black"
    )
    for index, pair in enumerate(pairs):
        contact.paste(
            Image.open(pair).resize(CONTACT),
            (
                (index % CONTACT_COLUMNS) * CONTACT[0],
                (index // CONTACT_COLUMNS) * CONTACT[1],
            ),
        )
    contact.save(compare_dir / "all.png")

    print(
        f"{len(pairs)} pairs -> {compare_dir}, contact sheet {compare_dir / 'all.png'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
