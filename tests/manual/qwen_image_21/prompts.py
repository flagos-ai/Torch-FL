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

"""The prompt cohort Qwen-Image-2.1 is tested with, and where each prompt came from.

Unlike Qwen-Image-2512's card, **2.1's model card publishes no prompts**. Its
README on ModelScope is the auto-generated stub ("the contributors have not
provided a more detailed model description"), and the published model repository
carries no example snippet and no showcase section. There is therefore nothing
to read the cohort out of, the way ``qwen_image_2512/sweep.py`` reads the 2512
card -- the cohort has to be written down here, and its provenance stated, so
that nobody mistakes it for something the model authors published.

Provenance of each entry, in ``PROMPTS`` order:

``01``  the only prompt the 2.1 authors published anywhere: the example in
        ``QwenImage21Pipeline``'s own docstring, in the diffusers source the
        pipeline ships in. Copied verbatim, including its commas. This is also
        ``CANONICAL_ID`` -- the one prompt ``bench.py`` measures with.

``02``-``08``  written for this flow. They are not a random spread -- each one
        is the cheapest prompt that reaches an operator surface the others do
        not, so a failure names a component:

``02``  a photorealistic person at close range. Fine facial detail is where a
        numerically wrong reduction or a mis-scaled normalization shows up as a
        face that is recognizably *almost* right, which is the kind of failure a
        PSNR number cannot see.
``03``  text rendered in the image. Glyph shapes are high-frequency structure,
        so this is the prompt that exposes a resampler or a patch-embedding
        error that a smooth subject would hide.
``04``  structured geometry (a grid of identical objects). Countable, regular
        content makes a positional-encoding (RoPE) fault obvious -- the objects
        smear or the grid shears rather than merely looking different.
``05``  a wide landscape. The only prompt designed to be run at 1664x928 as well
        as 1024x1024, to exercise a second packed-sequence length.
``06``  a long, clause-heavy prompt. Long prompts are cheap activations for the
        text encoder only, so this is the one that separates a text-encoder
        fault from a transformer fault.
``07``  a dark, low-contrast scene. Compresses the dynamic range, which is where
        a bf16 accumulation-order difference grows largest.
``08``  the same content as ``01`` with a different style suffix, so a pair of
        prompts differs by one phrase. Comparing ``01`` against ``08`` shows
        whether a change is in the model or in the sampling.

Nothing here is a quality benchmark. These prompts exist to make a failure
attributable, and the readings a run should produce are in the README; the
images themselves are judged by eye against the vendor's own torch.

To test a different cohort -- a model card's, a product team's -- pass
``--prompts-file`` to ``sweep.py``: JSON of the same shape as ``PROMPTS``.
"""

# (id, prompt) pairs. Ids are stable: the sweep writes one PNG per id and the
# comparison sheets pair by file name, so changing an id renames an artifact.
PROMPTS = [
    (
        "01",
        "A capybara wearing a wizard hat, reading a book by candlelight, oil painting",
    ),
    (
        "02",
        "A 40-year-old fisherman mending a net on a wooden dock at dawn, "
        "weathered face in sharp focus, mist over the water behind him, "
        "photorealistic, 85mm lens, shallow depth of field",
    ),
    (
        "03",
        "A weathered enamel sign bolted to a brick wall reading OPEN 24 HOURS "
        "in white block letters, evening light raking across it from the left, "
        "damp brick texture, photorealistic",
    ),
    (
        "04",
        "Sixteen identical brass door handles arranged in a perfect four by four "
        "grid on a matte black panel, photographed straight on, even studio "
        "lighting, no perspective, product photography",
    ),
    (
        "05",
        "A long empty causeway crossing a tidal flat at low tide, distant "
        "mountains under a heavy overcast sky, wide landscape photograph, "
        "no people, natural colour",
    ),
    (
        "06",
        "An overgrown botanical greenhouse at dusk, where broken panes let in a "
        "low orange light that catches dust hanging in the air and throws long "
        "shadows across cracked terracotta pots, ferns and moss reclaiming the "
        "ironwork, a narrow gravel path leading the eye to a door left ajar at "
        "the far end, no people, moody and atmospheric, photorealistic",
    ),
    (
        "07",
        "A single lit candle in an otherwise dark stone room, the flame the only "
        "light source, deep shadows filling the corners, plain grey wall, "
        "photorealistic, low key lighting",
    ),
    (
        "08",
        "A capybara wearing a wizard hat, reading a book by candlelight, "
        "watercolour illustration",
    ),
]

# 2.1 is meant to be sampled without guidance: the pipeline's own docstring
# calls `true_cfg_scale=1.0` (its default) and says so in the __call__ help.
# The negative prompt exists for the runs that do enable guidance for
# comparison against 2512's settings.
NEGATIVE_PROMPT = " "

# The prompt every performance run uses. See `canonical`: this is the cohort's
# `01`, the only prompt the 2.1 authors published anywhere. The rest of the
# cohort exists to make a failure attributable and is not a benchmark.
CANONICAL_ID = "01"


def load(path=None):
    """The cohort, from ``path`` when given else the table above.

    A file is the escape hatch for testing a cohort this module does not carry;
    it must be a JSON list of ``{"id": ..., "prompt": ...}`` objects, which is
    also the shape ``sweep.py`` writes into its manifest.

    Every entry gains a ``sha256`` of its own prompt text, so an artifact can
    name the exact bytes behind a number instead of quoting the text and hoping
    it survived. See ``canonical``.
    """
    if path is None:
        entries = [{"id": pid, "prompt": prompt} for pid, prompt in PROMPTS]
    else:
        import json
        from pathlib import Path

        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, list) or not raw:
            raise SystemExit(f"{path}: expected a non-empty JSON list of prompts")
        for entry in raw:
            missing = {"id", "prompt"} - set(entry)
            if missing:
                raise SystemExit(
                    f"{path}: entry {entry!r} is missing {sorted(missing)}"
                )
        entries = [{"id": str(e["id"]), "prompt": e["prompt"]} for e in raw]

    for entry in entries:
        entry["sha256"] = sha256(entry["prompt"])
    return entries


def canonical():
    """The one prompt a performance run uses, as ``{id, prompt, sha256}``.

    ``bench.py`` has no ``--prompt`` flag and calls this, so a benchmark is
    always the same measurement: the prompt is fixed in the code rather than
    supplied by whoever runs it. ``infer.py`` and ``sweep.py`` take it as their
    default too, which is what removes the second copy of the string that used
    to live in ``infer.DEFAULT_PROMPT``.

    Which prompt is canonical, and why: ``CANONICAL_ID`` is the only prompt the
    2.1 authors published anywhere -- the example in ``QwenImage21Pipeline``'s
    own docstring -- and it is the same order of length as the captions MLPerf
    uses for text_to_image (10-15 words), so the text encoder's share of the
    latency is representative rather than a property of a prompt written here.
    """
    for entry in load():
        if entry["id"] == CANONICAL_ID:
            return entry
    raise SystemExit(f"CANONICAL_ID={CANONICAL_ID!r} is not in PROMPTS")


def sha256(text):
    """sha256 of a prompt's exact UTF-8 bytes, lowercase hex."""
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()
