# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Copy the pinned base's complete FlagTree distribution into a clean venv."""

import importlib.metadata
from pathlib import Path
import shutil
import sys

distribution = importlib.metadata.distribution("flagtree")
assert distribution.version == sys.argv[1], distribution.version
root = Path("/opt/torch-fl-iluvatar-venv")
site = root / "lib/python3.12/site-packages"
for relative in distribution.files:
    source = Path(distribution.locate_file(relative))
    target = (site / relative).resolve()
    assert target.is_relative_to(root), target
    assert target.suffix != ".pth", "A source-path injection is not a wheel asset"
    if source.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
