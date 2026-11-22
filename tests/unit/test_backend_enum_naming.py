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

"""The retired CUDA boxing route has no dispatcher slot or source file."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_retired_boxing_slots_are_absent():
    route = (REPO_ROOT / "csrc/aten/route.h").read_text(encoding="utf-8")
    assert "kBoxing" not in route
    assert "kCuda" not in route


def test_retired_boxing_registration_sources_are_absent():
    generated = REPO_ROOT / "csrc/aten/generated"
    assert not (generated / "cuda_kernels.cc").exists()
    assert not list(generated.glob("register_boxing_*.inc"))
