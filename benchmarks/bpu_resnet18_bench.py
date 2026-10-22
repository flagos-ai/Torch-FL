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

"""BPU ResNet-18: our compile path against D-Robotics' own artifact.

The vendor ships `/opt/hobot/model/<soc>/basic/resnet18_224x224_nv12.hbm` and
drives it from `/app/pydev_demo/classification_sample/resnet18/resnet18.py`.
That artifact is the honest upper bound for this board, so it is the number
worth measuring against -- eager CPU only says the offload happened.

The two are not the same workload and the table says so:

* the official artifact takes **NV12** (two uint8 planes, 74 KB); ours takes
  float32 NCHW (588 KB), because that is what an FX graph carries. The
  conversion to the artifact's int8 input is ours to pay.
* it was quantized by the vendor's toolchain with real calibration data; ours
  uses `FLAGOS_BPU_ACT_SCALE` unless `calibrate_module` was run.
* it is a full torchvision ResNet-18 with a 1000-class head, which is what this
  script rebuilds -- torchvision is not installed on the board.

Run:

    export FLAGOS_BPU_X86_PYTHON=~/hbdk4-x86/python/bin/python3.11
    export FLAGOS_BPU_X86_EMULATOR=~/hbdk4-x86/bin/box64
    python benchmarks/bpu_resnet18_bench.py

The first run compiles under box64 and takes ~20 minutes; afterwards the .hbm
is cached in `~/.cache/torch_fl_bpu` and startup is seconds.
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

DEFAULT_OFFICIAL = "/opt/hobot/model/s600/basic/resnet18_224x224_nv12.hbm"


def bench(fn, reps: int, warmup: int) -> tuple[float, float, float]:
    """Median, p10 and p90 wall time in ms."""
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return statistics.median(ts), ts[len(ts) // 10], ts[-max(1, len(ts) // 10)]


# -- the model ---------------------------------------------------------------


class BasicBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1, down=None) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.down = down

    def forward(self, x):
        idt = x if self.down is None else self.down(x)
        o = torch.relu(self.bn1(self.conv1(x)))
        return torch.relu(self.bn2(self.conv2(o)) + idt)


class ResNet18(nn.Module):
    """torchvision's resnet18, transcribed; torchvision is not on the board."""

    def __init__(self, num_classes: int = 1000) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self._inp = 64
        self.layer1 = self._layer(64, 2, 1)
        self.layer2 = self._layer(128, 2, 2)
        self.layer3 = self._layer(256, 2, 2)
        self.layer4 = self._layer(512, 2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def _layer(self, cout: int, blocks: int, stride: int) -> nn.Sequential:
        down = None
        if stride != 1 or self._inp != cout:
            down = nn.Sequential(
                nn.Conv2d(self._inp, cout, 1, stride, bias=False), nn.BatchNorm2d(cout)
            )
        layers = [BasicBlock(self._inp, cout, stride, down)]
        self._inp = cout
        layers += [BasicBlock(cout, cout) for _ in range(blocks - 1)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(torch.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.fc(torch.flatten(self.avgpool(x), 1))


# -- runs --------------------------------------------------------------------


def run_official(path: str, reps: int, warmup: int):
    """Median latency of the vendor artifact, or None if it is not installed."""
    if not Path(path).exists():
        return None
    from hbm_runtime import HB_HBMRuntime

    rt = HB_HBMRuntime(path)
    m = rt.model_names[0]
    y_name, uv_name = rt.input_names[m][:2]
    feed = {
        m: {
            y_name: np.random.randint(0, 256, (1, 224, 224, 1), dtype=np.uint8),
            uv_name: np.random.randint(0, 256, (1, 112, 112, 2), dtype=np.uint8),
        }
    }
    nbytes = 224 * 224 + 112 * 112 * 2
    return (*bench(lambda: rt.run(feed), reps, warmup), nbytes)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--official", default=DEFAULT_OFFICIAL)
    opt = ap.parse_args()

    import torch_fl  # noqa: F401
    from torch_fl.accelerator.bpu.compiler import find_hbdk

    print(f"hbdk4: {find_hbdk() or 'not reachable -- ours will run on CPU'}\n")

    torch._dynamo.reset()
    torch.manual_seed(0)
    model = ResNet18().eval()
    x = torch.randn(1, 3, 224, 224)
    in_kb = x.numel() * 4 / 1024

    with torch.no_grad():
        eager = bench(lambda: model(x), opt.reps, opt.warmup)

    compiled = torch.compile(model, backend="bpu")
    with torch.no_grad():
        t0 = time.perf_counter()
        out = compiled(x)
        first = time.perf_counter() - t0
        ours = bench(lambda: compiled(x), opt.reps, opt.warmup)

    official = run_official(opt.official, opt.reps, opt.warmup)

    print(f"{'path':36} {'median':>9} {'p10':>8} {'p90':>8}   input")
    if official:
        med, lo, hi, nb = official
        print(
            f"{'official resnet18_224x224_nv12':36} {med:8.3f}ms {lo:7.3f}ms "
            f"{hi:7.3f}ms   {nb / 1024:.0f} KB NV12 uint8"
        )
    else:
        print(f"{'official (not installed)':36} {'-':>9} {'-':>8} {'-':>8}")
    print(
        f"{'eager CPU float32':36} {eager[0]:8.3f}ms {eager[1]:7.3f}ms "
        f"{eager[2]:7.3f}ms   {in_kb:.0f} KB f32 NCHW"
    )
    print(
        f"{'ours torch.compile(backend=bpu)':36} {ours[0]:8.3f}ms {ours[1]:7.3f}ms "
        f"{ours[2]:7.3f}ms   {in_kb:.0f} KB f32 NCHW"
    )

    print(f"\noutput {tuple(out.shape)}, first call {first:.1f}s (compile included)")
    print(f"ours vs eager    : {eager[0] / ours[0]:.2f}x")
    if official:
        print(f"ours vs official : {ours[0] / official[0]:.2f}x slower")


if __name__ == "__main__":
    main()
