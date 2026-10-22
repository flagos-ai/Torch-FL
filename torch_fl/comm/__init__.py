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

"""
torch_fl.comm — native ProcessGroup backend for flagos (PrivateUse1).

Registering this package (done automatically by ``import torch_fl``) makes
``torch.distributed.init_process_group`` recognise the ``"flagos"`` backend
and route distributed ops on ``privateuseone`` tensors through it without any
monkeypatching.

After ``import torch_fl``:
    - ``torch.distributed.init_process_group("flagos")``  — explicit
    - ``torch.distributed.init_process_group(            ``  — auto-detect
          device_id=torch.device("privateuseone:0"))
"""

from torch_fl.comm.process_group import ProcessGroupFlagOS, register_flagos_backend

__all__ = ["ProcessGroupFlagOS", "register_flagos_backend"]
