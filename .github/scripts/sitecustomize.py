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

"""Keep a source checkout from replacing an installed native package.

The integration runner adds this directory to each child's PYTHONPATH. A unit
child also imports this module directly before pytest can add the checkout to
its path. Site initialization ignores this setting when no package was chosen.
"""

import sys
import sysconfig
from pathlib import Path

from _torch_fl_test_site import install


if install() is None:
    config = sysconfig.get_path("purelib")
    if config:
        sys.path[:] = [
            entry
            for entry in sys.path
            if str(Path(entry).resolve()) != str(Path(config).resolve())
        ]
