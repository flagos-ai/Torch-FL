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
"""Give the source tree's ``utils`` the name ``utils`` in a child process.

HuggingFace's test tree imports its own helper directory as top-level modules --
``tests/pipelines/test_pipelines_any_to_any.py`` does ``from
utils.fetch_hub_objects_for_ci import url_to_local_path`` -- because upstream
runs pytest from the repository root, where ``utils/`` is importable by name.
That directory holds no ``__init__.py``, so in the source tree it is only a
namespace portion.

The FlagGems backend meanwhile appends its own architecture directory to
``sys.path`` and never removes it (``flag_gems/runtime/backend/__init__.py``,
``get_arch_ops`` and ``get_vendor_module``), and that directory contains a
*regular* ``utils`` package. The import system keeps looking past a namespace
portion and a concrete package found further along the path wins, even though
the source tree is listed first: `utils` then resolves to the backend's helpers,
``utils.fetch_hub_objects_for_ci`` does not exist there, and every test module
that reaches the helper through ``test_pipeline_mixin`` fails at collection
rather than at run time. Measured on GCU: ``--model bert`` collects 56 nodes,
all of them offline tokenization tests, and the modeling suite is never collected
at all.

The runner installs this file as ``utils/__init__.py`` in the child's working
directory, which is ahead of both on ``sys.path``. That promotes the namespace
portion to a regular package and settles the name; merging the path with
``extend_path`` keeps the backend's own helpers importable, so the correction
cannot break the backend it is working around.

This is not a device shim and is not switched off by ``HF_TEST_NO_DEVICE_SHIMS``:
the collision is between two Python path entries and happens whatever
accelerator the child runs on.
"""

from pkgutil import extend_path

# Appends every other ``utils`` directory found on ``sys.path`` -- the source
# tree first, then the backend's -- instead of replacing the search path. The
# noqa is for ``__path__``, which the import system injects before this body
# runs; lint does not recognise it here because the file is only named
# ``__init__.py`` once the runner installs it.
__path__ = extend_path(__path__, __name__)  # noqa: F821
