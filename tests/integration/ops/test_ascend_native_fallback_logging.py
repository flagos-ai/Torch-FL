"""Native dtype fallback must remain observable during accelerator audits."""

import os
import subprocess
import sys

import pytest


@pytest.mark.ascend
def test_native_dtype_fallback_is_logged():
    environment = dict(os.environ, FLAGOS_LOG="fallback")
    code = """
import torch_fl
import torch
matrix = torch.tensor([[1, 2], [3, 4]], device="flagos:0", dtype=torch.int64)
result = matrix @ matrix
assert result.device.type == "flagos"
assert result.cpu().tolist() == [[7, 10], [15, 22]]
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        env=environment,
        text=True,
        capture_output=True,
        check=True,
        timeout=60,
    )
    assert "[flagos cpu_fallback] ascend native dtype fallback" in completed.stderr
