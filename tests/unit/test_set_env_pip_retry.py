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

"""pip_retry fails fast, and install_cpu_torch prefers a fast CPU-torch index.

Both behaviours come from run 38409565812 (Ascend build, 2026-10-13):

* ``pip_retry`` retried ``resource.flagos.net`` five times after name
  resolution had already failed, spending 21.5 of the job's 27.9 minutes on an
  error that cannot clear inside a pod. The index host is now checked before the
  first attempt, and an unresolvable host ends the call immediately -- unless
  the runner proxies that host, in which case the proxy does the lookup.
* The same run's runner carried ``NO_PROXY=...,resource.flagos.net,*.flagos.net``,
  exempting the index host from the proxy, and then could not resolve it
  directly. The exemption is what broke it -- the sibling Ascend runner, same
  proxy and no exemption, fetched the 535.9 MB FlagTree wheel through it -- so
  ``ensure_index_host_route`` drops the exemption and lets pip fall back to the
  proxy, rather than reporting a host the job can in fact reach.
* ``install_cpu_torch`` pulled a 146-190 MB wheel from an offshore index at
  150-480 kB/s. It now measures the pin against a mirror and installs from the
  faster one, keeping the pin when the mirror is not faster or the probe fails.

The library is sourced and its helpers are called directly. The one
network-facing piece, ``index_download_bps``, is pointed at a local HTTP server
so its listing parser is exercised without leaving the machine.
"""

import http.server
import os
import subprocess
import sys
import threading
from functools import partial
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LIB = REPO_ROOT / ".github" / "scripts" / "lib" / "set_env_common.sh"


def _run_bash(script: str, env: dict | None = None) -> subprocess.CompletedProcess:
    """Source the library and run ``script`` with its stdout/stderr captured."""
    # The library falls back to `python`, which is not on PATH on every host;
    # the tests that need a stub interpreter export PIP_RETRY_PYTHON themselves.
    merged = {
        **os.environ,
        "PIP_RETRY_PYTHON": sys.executable,
        "VENV_PYTHON": sys.executable,
    }
    if env:
        merged.update(env)
    return subprocess.run(
        ["bash", "-c", f'set -euo pipefail\nsource "{LIB}"\n{script}'],
        capture_output=True,
        text=True,
        env=merged,
        timeout=120,
    )


def _fake_interpreter(tmp_path: Path, pip_body: str) -> tuple[Path, Path]:
    """A python that answers ``-m pip`` with ``pip_body`` and delegates the rest.

    pip_retry uses one interpreter both to resolve the index host (a real
    getaddrinfo) and to run pip, so the stub records pip invocations and hands
    every other call to the real interpreter.
    """
    recorder = tmp_path / "pip-invocations"
    fake = tmp_path / "fake-python"
    fake.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "${1:-}" == "-m" && "${2:-}" == "pip" ]]; then\n'
        f'  echo "$*" >> "{recorder}"\n'
        f"{pip_body}"
        "fi\n"
        f'exec "{sys.executable}" "$@"\n',
        encoding="utf-8",
    )
    fake.chmod(0o755)
    return fake, recorder


def test_pip_retry_refuses_an_unresolvable_index_without_running_pip(tmp_path):
    """A DNS failure is permanent, so it must not reach the retry loop."""
    fake, recorder = _fake_interpreter(tmp_path, "exit 1\n")
    done = _run_bash(
        f'export PIP_RETRY_PYTHON="{fake}"\n'
        "if pip_retry --index-url https://no-such-host.invalid/simple somepkg; "
        "then echo UNEXPECTED-SUCCESS; fi\n",
        env={
            "HTTPS_PROXY": "",
            "HTTP_PROXY": "",
            "ALL_PROXY": "",
            "https_proxy": "",
            "http_proxy": "",
            "all_proxy": "",
            "NO_PROXY": "",
            "no_proxy": "",
        },
    )
    assert "::error::Cannot reach the package index host" in done.stdout
    assert "UNEXPECTED-SUCCESS" not in done.stdout
    assert not recorder.exists(), "pip ran for a host that does not resolve"


def test_pip_retry_installs_when_the_index_check_cannot_run(tmp_path):
    """Only a site that answers "does not resolve" may stop the install.

    The check runs in the venv interpreter, so it cannot run before that venv
    exists; its failure then says nothing about the index and must not replace a
    working install with a network diagnosis.
    """
    recorder = tmp_path / "pip-invocations"
    fake = tmp_path / "fake-python"
    fake.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "${1:-}" == "-m" && "${2:-}" == "pip" ]]; then\n'
        f'  echo "$*" >> "{recorder}"\n'
        "  exit 0\n"
        "fi\n"
        "exit 2\n",
        encoding="utf-8",
    )
    fake.chmod(0o755)
    done = _run_bash(
        f'export PIP_RETRY_PYTHON="{fake}"\n'
        "pip_retry --index-url https://example.invalid/simple somepkg && echo PIP-OK\n"
    )
    assert "::error::Cannot reach the package index host" not in done.stdout
    assert "::warning::Could not check how this pod reaches" in done.stdout
    assert "PIP-OK" in done.stdout
    assert recorder.read_text(encoding="utf-8").count("-m pip") == 1


def test_ensure_index_host_route_leaves_a_proxied_host_to_the_proxy():
    """The proxy looks the host up, so the pod's own resolution proves nothing."""
    done = _run_bash(
        "ensure_index_host_route no-such-host.invalid && echo NOT-CHECKED\n",
        env={"HTTPS_PROXY": "http://proxy.invalid:8080", "NO_PROXY": ""},
    )
    assert "NOT-CHECKED" in done.stdout
    assert "proxy" in done.stderr


def test_ensure_index_host_route_unexempts_a_host_the_runner_exempted():
    """An exemption that makes the host unreachable is undone, not reported.

    This is the Ascend runner: ``NO_PROXY`` names the index host, the pod cannot
    resolve it directly, and the proxy -- which the sibling runner uses for the
    same host -- can. Both the exact entry and the ``*.invalid`` wildcard the
    runner also lists are dropped, so pip falls back to the proxy.
    """
    done = _run_bash(
        "ensure_index_host_route no-such-host.invalid && echo NOT-CHECKED\n"
        'echo "NO_PROXY=${NO_PROXY:-}|no_proxy=${no_proxy:-}"\n',
        env={
            "HTTPS_PROXY": "http://proxy.invalid:8080",
            "NO_PROXY": "localhost,127.0.0.1,no-such-host.invalid,*.invalid",
            "no_proxy": "localhost,127.0.0.1,no-such-host.invalid,*.invalid",
        },
    )
    assert "NOT-CHECKED" in done.stdout
    assert "NO_PROXY=localhost,127.0.0.1|no_proxy=localhost,127.0.0.1" in done.stdout
    assert "exempted from the" in done.stderr


def test_ensure_index_host_route_still_fails_with_no_proxy_to_fall_back_on():
    """Without a proxy the pod itself must resolve the host, or nothing can."""
    blank = {
        name: ""
        for name in (
            "HTTPS_PROXY",
            "HTTP_PROXY",
            "ALL_PROXY",
            "https_proxy",
            "http_proxy",
            "all_proxy",
            "NO_PROXY",
            "no_proxy",
        )
    }
    done = _run_bash(
        "if ensure_index_host_route no-such-host.invalid; then echo RESOLVED; fi\n",
        env=blank,
    )
    assert "RESOLVED" not in done.stdout
    assert "does not resolve from this runner" in done.stderr


def test_ensure_index_host_route_reports_a_wildcard_exemption():
    """``NO_PROXY=*`` exempts the host whatever else is dropped."""
    done = _run_bash(
        "if ensure_index_host_route no-such-host.invalid; then echo RESOLVED; fi\n",
        env={"HTTPS_PROXY": "http://proxy.invalid:8080", "NO_PROXY": "*"},
    )
    assert "RESOLVED" not in done.stdout
    assert "does not resolve from this runner" in done.stderr


def test_ensure_index_host_route_leaves_a_resolvable_host_alone():
    done = _run_bash(
        "ensure_index_host_route 127.0.0.1 && echo OK\n",
        env={
            "HTTPS_PROXY": "http://proxy.invalid:8080",
            "NO_PROXY": "localhost,127.0.0.1",
        },
    )
    assert "OK" in done.stdout
    assert done.stderr.strip() == ""


def test_pip_retry_unexempts_the_index_host_before_installing(tmp_path):
    """The Ascend case end to end: a proxy-reachable host installs from the proxy."""
    fake, recorder = _fake_interpreter(tmp_path, "exit 0\n")
    done = _run_bash(
        f'export PIP_RETRY_PYTHON="{fake}"\n'
        'export HTTPS_PROXY="http://proxy.invalid:8080"\n'
        'export NO_PROXY="localhost,127.0.0.1,no-such-host.invalid,*.invalid"\n'
        "unset no_proxy\n"
        "pip_retry --index-url https://no-such-host.invalid/simple somepkg\n"
        'echo "NO_PROXY=${NO_PROXY}"\n'
    )
    assert "::error::Cannot reach the package index host" not in done.stdout
    assert "NO_PROXY=localhost,127.0.0.1" in done.stdout
    assert recorder.read_text(encoding="utf-8").count("-m pip") == 1


def test_pip_retry_stops_after_one_attempt_on_a_dns_failure(tmp_path):
    """A resolution failure mid-install is still permanent, not a slow network."""
    fake, recorder = _fake_interpreter(
        tmp_path,
        'echo "WARNING: Retrying after connection broken by "\n'
        "echo \"'NameResolutionError(...): Failed to resolve 'x' ([Errno -3])'\"\n"
        "exit 1\n",
    )
    done = _run_bash(
        f'export PIP_RETRY_PYTHON="{fake}"\n'
        "if pip_retry --index-url http://127.0.0.1/simple torch==2.10.0+cpu; "
        "then echo UNEXPECTED-SUCCESS; fi\n"
    )
    assert "::warning::pip install attempt" not in done.stdout
    assert "::error::pip install hit an error that retrying cannot fix" in done.stdout
    assert recorder.read_text(encoding="utf-8").count("-m pip") == 1


def test_pip_error_is_permanent_ignores_a_failure_that_can_recover(tmp_path):
    """A 502 prints ``Could not find a version`` too, and it does clear.

    Measured against an index answering 502: pip prints "Could not find a
    version that satisfies" and "No matching distribution found", the same two
    lines a DNS failure produces. Only the resolution wording is permanent.
    """
    permanent = tmp_path / "permanent.log"
    permanent.write_text(
        "ERROR: Could not find a version that satisfies the requirement torch\n"
        "WARNING: ... NameResolutionError(...): Failed to resolve 'x'\n",
        encoding="utf-8",
    )
    recoverable = tmp_path / "recoverable.log"
    recoverable.write_text(
        "ERROR: Could not find a version that satisfies the requirement torch\n"
        "ERROR: No matching distribution found for torch==2.10.0+cpu\n",
        encoding="utf-8",
    )
    done = _run_bash(
        f'pip_error_is_permanent "{permanent}" && echo PERMANENT\n'
        f'pip_error_is_permanent "{recoverable}" && echo RECOVERABLE-IS-PERMANENT '
        "|| true\n"
    )
    assert "PERMANENT" in done.stdout
    assert "RECOVERABLE-IS-PERMANENT" not in done.stdout


def test_faster_cpu_torch_index_returns_only_the_chosen_index():
    """The index is read from stdout, so the probe diagnostic goes to stderr."""
    probe = 'index_download_bps() { case "$1" in *mirror*) echo 9000000;; *) echo 400000;; esac; }\n'
    done = _run_bash(
        probe
        + "faster_cpu_torch_index https://pin.example/whl/cpu https://mirror.example/cpu\n"
    )
    assert done.stdout == "https://mirror.example/cpu"
    assert "CPU torch index probe:" in done.stderr


def test_faster_cpu_torch_index_keeps_the_pin_when_the_probe_fails():
    done = _run_bash(
        "index_download_bps() { echo 0; }\n"
        "faster_cpu_torch_index https://pin.example/whl/cpu https://mirror.example/cpu\n"
    )
    assert done.stdout == "https://pin.example/whl/cpu"


def test_install_cpu_torch_installs_from_the_mirror_when_it_is_faster():
    probe = 'index_download_bps() { case "$1" in *mirror*) echo 9000000;; *) echo 400000;; esac; }\n'
    done = _run_bash(
        "CPU_TORCH_INDEX_URL=https://pin.example/whl/cpu\n"
        "CPU_TORCH_VERSION=2.10.0\n"
        "TORCH_FL_CPU_TORCH_MIRROR_INDEX_URL=https://mirror.example/cpu\n"
        + probe
        + 'pip_retry() { printf "PIP %s\\n" "$*"; }\n'
        "install_cpu_torch\n"
    )
    assert done.stdout == "PIP --index-url https://mirror.example/cpu torch==2.10.0\n"


def test_install_cpu_torch_skips_the_probe_when_the_mirror_is_disabled():
    done = _run_bash(
        "CPU_TORCH_INDEX_URL=https://pin.example/whl/cpu\n"
        "CPU_TORCH_VERSION=2.10.0\n"
        "TORCH_FL_CPU_TORCH_MIRROR_INDEX_URL=\n"
        "index_download_bps() { echo PROBE-CALLED; }\n"
        'pip_retry() { printf "PIP %s\\n" "$*"; }\n'
        "install_cpu_torch\n"
    )
    assert done.stdout == "PIP --index-url https://pin.example/whl/cpu torch==2.10.0\n"


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args, **kwargs):
        """Silence the per-request line the probe would otherwise print."""


def _serve(docroot: Path) -> tuple[http.server.ThreadingHTTPServer, str]:
    server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), partial(_QuietHandler, directory=str(docroot))
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    host, port = server.server_address[:2]
    return server, f"http://{host}:{port}"


def test_index_download_bps_reads_a_listing_with_hash_fragments(tmp_path):
    """Both indexes put a ``#sha256=`` fragment after the wheel name.

    A first version of the probe matched a literal ``.whl"`` and found no wheel
    on either index, which silently disabled the mirror on every platform.
    """
    wheel = "torch-2.10.0%2Bcpu-cp310-cp310-manylinux_2_28_x86_64.whl"
    torch_dir = tmp_path / "cpu" / "torch"
    torch_dir.mkdir(parents=True)
    (torch_dir / "index.html").write_text(
        "<html><body>\n"
        f'<a href="/cpu/torch/{wheel}#sha256=deadbeef">one</a>\n'
        "</body></html>\n",
        encoding="utf-8",
    )
    (torch_dir / wheel.replace("%2B", "+")).write_bytes(b"0" * 400_000)

    server, base = _serve(tmp_path)
    try:
        done = _run_bash(f'index_download_bps "{base}/cpu" 10 200000\n')
    finally:
        server.shutdown()
        server.server_close()

    assert done.stdout.strip().isdigit(), done.stdout + done.stderr
    assert int(done.stdout) > 0, done.stderr


def test_index_download_bps_reports_zero_for_a_missing_index(tmp_path):
    server, base = _serve(tmp_path)
    try:
        done = _run_bash(f'index_download_bps "{base}/absent" 5 1000\n')
    finally:
        server.shutdown()
        server.server_close()
    assert done.stdout.strip() == "0"
