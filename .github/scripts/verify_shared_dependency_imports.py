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

"""Map every explicitly pinned distribution to its import probe."""

MODULE_IMPORT_NAMES = {
    "onnx": "onnx",
    "onnxruntime": "onnxruntime",
    "jq": "jq",
    "transformers": "transformers",
    "huggingface-hub": "huggingface_hub",
    "sqlalchemy": "sqlalchemy",
    "greenlet": "greenlet",
    "hf-xet": "hf_xet",
    "tokenizers": "tokenizers",
    "safetensors": "safetensors",
    "protobuf": "google.protobuf",
    "ml_dtypes": "ml_dtypes",
    "coloredlogs": "coloredlogs",
    "humanfriendly": "humanfriendly",
    "flatbuffers": "flatbuffers",
    "regex": "regex",
    "numpy": "numpy",
    "requests": "requests",
    "urllib3": "urllib3",
    "charset-normalizer": "charset_normalizer",
    "certifi": "certifi",
    "idna": "idna",
    "httpx": "httpx",
    "httpcore": "httpcore",
    "h11": "h11",
    "anyio": "anyio",
    "diffusers": "diffusers",
    "pillow": "PIL",
    "pyyaml": "yaml",
    "typer": "typer",
    "shellingham": "shellingham",
    "annotated-doc": "annotated_doc",
    "rich": "rich",
    "markdown-it-py": "markdown_it",
    "pygments": "pygments",
    "mdurl": "mdurl",
    "tqdm": "tqdm",
    "filelock": "filelock",
    "packaging": "packaging",
    "importlib-metadata": "importlib_metadata",
    "zipp": "zipp",
    "click": "click",
    "fsspec": "fsspec",
    "typing-extensions": "typing_extensions",
    "sympy": "sympy",
    "mpmath": "mpmath",
}


def module_imports(packages, *, pinned=None):
    """Return probes and reject pins without an installed-import check."""

    if pinned is None:
        pinned = {name for name, version in packages.items() if version}
    missing = pinned - set(MODULE_IMPORT_NAMES)
    stale = set(MODULE_IMPORT_NAMES) - pinned
    if missing or stale:
        raise SystemExit(
            f"shared dependency probes are out of sync; missing: {sorted(missing)}; "
            f"stale: {sorted(stale)}"
        )
    if len(set(MODULE_IMPORT_NAMES.values())) != len(MODULE_IMPORT_NAMES):
        raise SystemExit("shared dependency probes must use distinct modules")
    # The backport deliberately exposes only a few top-level APIs and uninstalls
    # its module shim when the matching stdlib module exists. Its 46.5 Python
    # version has no stdlib importlib.metadata.
    assert packages["importlib-metadata"] <= "4.13.0"
    return MODULE_IMPORT_NAMES
