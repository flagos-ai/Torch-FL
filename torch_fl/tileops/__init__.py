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

"""Python side of the TileOPs operator library.

TileOPs is registered on ``Backend::kTileOps`` from C++
(``csrc/aten/generated/tileops_python_kernels.cc``), so routing, per-op
``FLAGOS_OP_<op>`` overrides and ``FLAGOS_LOG_DISPATCH`` are all handled by the
dispatcher. What lives here is only what cannot: TileOPs ships no C++ API, so
the kernels themselves are Python, and the generated stubs call back into
:mod:`torch_fl.tileops.generated.shims` to reach them.

  - :mod:`~torch_fl.tileops.runtime`  instance cache, boxing, recipe builders
  - :mod:`~torch_fl.tileops.spec`     hand-maintained codegen input
  - :mod:`~torch_fl.tileops.generated`  products of ``scripts/codegen_tileops.py``

Nothing is imported eagerly: ``runtime`` pulls in TileOPs (and therefore
TileLang) only once a route is first exercised, so ``import torch_fl`` stays
cheap and works on hosts without TileOPs installed.

See ``docs/tileops_codegen_design.md``.
"""

__all__ = ["generated", "runtime", "spec"]
