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

"""Products of ``scripts/codegen_tileops.py``. Regenerate, do not hand-edit.

  - :mod:`~torch_fl.tileops.generated.routes`  the routing table (``ROUTES``)
  - :mod:`~torch_fl.tileops.generated.shims`   one free function per route, the
    form ``CallPythonOp_Generic`` resolves by qualname from the C++ stubs

``shims`` is imported by name from C++, never by Python code, so the module
names here are load-bearing: renaming one without regenerating the ``.cc``
breaks that route at first call rather than at import.
"""

__all__ = ["routes", "shims"]
