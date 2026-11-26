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

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def gcu_compat():
    path = (
        Path(__file__).resolve().parents[2]
        / "torch_fl/accelerator/enflame/_gcu_compat.py"
    )
    spec = importlib.util.spec_from_file_location("_gcu_compat_for_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("tops4", [False, True], ids=["legacy", "tops4"])
def test_flagtree_host_compile_and_cache_follow_the_sdk(gcu_compat, tmp_path, tops4):
    sdk_headers = tmp_path / "include"
    sdk_headers.mkdir()
    (sdk_headers / "tops_runtime_api.h").write_text(
        "#define TOPSRT_VERSION 4010\n" if tops4 else "/* Legacy TOPS */\n",
        encoding="utf-8",
    )
    src = tmp_path / "main.cpp"
    original = (
        "topsDeviceProp_t p; TOPS_SUCCESS; a.ThreadDim; TOPS_ERROR_INVALID_VALUE;"
    )
    src.write_text(original, encoding="utf-8")

    def build_extension(name, filename, srcdir, extra_objects):
        assert srcdir == tmp_path
        assert extra_objects == ["helper.o"]
        return Path(filename).read_text(encoding="utf-8")

    def get_cache(key):
        return key

    backend = SimpleNamespace(
        _get_topscc_root=lambda: tmp_path,
        build_gcu_ext=build_extension,
        get_cache_manager=get_cache,
    )
    gcu_compat._patch_flagtree_tops_jit(backend)
    compiler = backend.build_gcu_ext
    cache = backend.get_cache_manager
    gcu_compat._patch_flagtree_tops_jit(backend)
    assert backend.build_gcu_ext is compiler and backend.get_cache_manager is cache

    expected = (
        "topsDeviceProp p; topsSuccess; a.threadDim; topsErrorInvalidValue;"
        if tops4
        else original
    )
    for name in ("gcu_utils", "__triton_launcher"):
        assert compiler(name, src, tmp_path, ["helper.o"]) == expected
    assert compiler("other_extension", src, tmp_path, ["helper.o"]) == original
    assert src.read_text(encoding="utf-8") == original
    assert (cache("012345") != "012345") == tops4
    assert cache("012345") == cache("012345")
