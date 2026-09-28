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

"""The DCU wheel's native comm bridge, asserted against the installed package.

A DCU wheel front-ends the official ``torch==2.10.0+cpu`` core, which is built
without ``USE_C10D_NCCL``: ``torch.distributed.ProcessGroupNCCL`` does not exist
in it. `ProcessGroupFlagOS` builds its inner backend as FlagCX -> vendor native
-> host-staged gloo, and the vendor-native tier needs exactly that class, so
without the bridge the tier cannot be built at all and the group lands on
host-staged gloo -- a working tier, with nothing said about the one that is
missing. The bridge is ``_flagos_nccl``, a pybind factory built against the
bundled ``lib_dcu`` (``libc10_hip`` / ``libtorch_hip``) plus ``librccl``, and
issue #366 is about the wheel packaging it at ``torch_fl/comm/_nccl_ext/`` so an
installed wheel reaches RCCL with no source checkout.

Two things are asserted, in the order they can break:

1. **The packaging.** The extension must load from *inside the installed
   package* and export ``make_nccl_backend``. Both halves have failed
   independently: before #366 the wheel carried no ``.so`` at all, and a wheel
   built through ``setup.py`` without ``-DTORCH_EXTENSION_NAME=...`` packaged a
   valid-looking ``.so`` that exported ``PyInit_TORCH_EXTENSION_NAME`` and died
   at import with "dynamic module does not define module export function
   (PyInit__flagos_nccl)". Neither is a build error: both produce a wheel that
   installs cleanly, reports a compatibility manifest, and loses the tier
   silently. Asserting the extension loads is what catches the second one --
   that failure *is* an ImportError out of ``importlib``.

2. **The tier.** A world-size-1 ``flagos`` group must resolve its inner backend
   to ``nccl``, with neither the FlagCX nor the staged-gloo skip reason set, and
   complete a collective. The bridge being importable is not the same claim: the
   tier is reached through ``ProcessGroupFlagOS._load_nccl_extension()``, and a
   group that fell back would still leave the extension loaded.

This is the completion bar issue #366 states -- an installed wheel reaches RCCL
with no source checkout -- and it runs in CI's wheel-only workspace, where the
package under test is the installed one. It is a file rather than a snippet in
the DCU manifest because the manifest is a list of commands, and a command that
carries a program is a second, untested copy of the test tree.

Marked ``dcu``: ``torch_fl.comm._nccl_ext.build.wheel_extension()`` builds this
extension for DCU only, so on every other platform it is absent by design and
there is nothing to assert.

Usage:
    pytest tests/integration/test_comm_native_bridge.py -v
"""

from __future__ import annotations

import socket
from pathlib import Path

import pytest

# torch_fl MUST be imported before torch: it preloads the device assets the
# backend claims its device with.
import torch_fl
import torch
import torch.distributed as dist
from torch_fl.comm import _nccl_ext

from platform_support import detect_platform

pytestmark = pytest.mark.dcu


def _free_port() -> int:
    """An ephemeral port for the rendezvous, bound and released.

    The group is one process, but ``init_process_group`` still wants a
    rendezvous, and a fixed port would collide with whatever else is running on
    the shared DCU runners.
    """
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _skip_unless_dcu() -> None:
    platform = detect_platform()
    if platform != "dcu":
        pytest.skip(f"the native comm bridge is packaged for DCU, not {platform}")


def test_the_installed_package_carries_the_native_comm_bridge():
    _skip_unless_dcu()

    extension = _nccl_ext.get_extension()
    assert extension is not None, _nccl_ext.load_error()
    assert hasattr(extension, "make_nccl_backend"), dir(extension)

    package = Path(torch_fl.__file__).resolve().parent
    bridge = Path(extension.__file__).resolve()
    assert bridge.is_relative_to(package), (bridge, package)
    print(f"bridge: {bridge}")


def test_the_rccl_tier_resolves_to_that_bridge():
    _skip_unless_dcu()

    dist.init_process_group(
        "flagos",
        init_method=f"tcp://127.0.0.1:{_free_port()}",
        rank=0,
        world_size=1,
    )
    try:
        group = dist.distributed_c10d._get_default_group()
        inner = getattr(group, "_inner", None)
        assert inner is not None, dir(group)

        backend_name = inner._get_backend_name()
        assert backend_name == "nccl", backend_name

        for reason in ("_nccl_skip_reason", "_staged_skip_reason"):
            assert getattr(group, reason, None) is None, (
                reason,
                getattr(group, reason),
            )

        # The point of the tier, and the one thing the class name does not
        # prove: a collective goes through the bridge and comes back right.
        tensor = torch.ones(8, device="flagos:0")
        dist.all_reduce(tensor)
        assert tensor.cpu().tolist() == [1.0] * 8
    finally:
        dist.destroy_process_group()
