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

"""Unit coverage for Qwen-Image-2.1 prompt-mask elision."""

import importlib.util
from pathlib import Path

import pytest


FLOW = Path(__file__).parents[1] / "manual" / "qwen_image_21"
SPEC = importlib.util.spec_from_file_location(
    "qwen_image_21_common_under_test", FLOW / "common.py"
)
COMMON = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMMON)


class FakeMask:
    def __init__(self, all_valid):
        self.all_valid = all_valid

    def detach(self):
        return self

    def to(self, _device):
        return self

    def bool(self):
        return self

    def all(self):
        return self

    def item(self):
        return self.all_valid


@pytest.mark.parametrize("all_valid, expected_is_none", [(True, True), (False, False)])
def test_prompt_mask_elision_preserves_masks_with_padding(all_valid, expected_is_none):
    mask = FakeMask(all_valid)

    class PromptPipe:
        def encode_prompt(self):
            return "embeds", mask, "image-mask"

    pipe = PromptPipe()
    COMMON.enable_all_valid_prompt_mask_elision(pipe)
    COMMON.enable_all_valid_prompt_mask_elision(pipe)

    embeds, returned_mask, image_mask = pipe.encode_prompt()
    assert embeds == "embeds"
    assert (returned_mask is None) is expected_is_none
    assert returned_mask is None or returned_mask is mask
    assert image_mask == "image-mask"
