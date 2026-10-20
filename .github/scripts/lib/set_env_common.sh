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

# Shared helpers for the set_env_*.sh platform provisioning scripts.
#
# Sourced (not executed) by each script after REPO_ROOT and the version pins are
# set. These five functions were copied verbatim into every script -- `pip_retry`
# in five slightly different spellings, `flag_gems_installed`/`install_flag_gems`
# in six copies -- so a fix had to be applied seven times and was easy to miss.
#
# The interpreter is taken from `${VENV_PYTHON:-python}`: every script sets
# VENV_PYTHON, to its job-local venv except MetaX, which points it at the image's
# /opt/venv it provisions in place. `pip_retry` additionally honours
# PIP_RETRY_PYTHON (a per-call override, used by the CUDA script's several
# interpreters), PIP_RETRY_TIMEOUT (default 300s) and PIP_RETRY_NO_CACHE
# (default 1; the shared entrypoint enables caching for every platform).
#
# `install_cpu_torch` measures both CPU-torch indexes before it installs from
# one, and honours TORCH_FL_CPU_TORCH_MIRROR_INDEX_URL (an empty value disables
# the mirror and pins the install to CPU_TORCH_INDEX_URL).
# `pip_retry` refuses to retry an index host that nothing in the job can reach --
# the pod cannot resolve it and the runner does not proxy it -- because the retry
# loop otherwise turns a network error the job cannot recover from into what
# looks like a hang. A host that only fails to resolve because the runner
# exempted it from its proxy is un-exempted and installed from normally.
#
# The scripts must keep `set -euo pipefail` in effect; every variable read here
# is either set by the caller or has a default.

# Host of the first index URL in "$@", or empty when there is none. Only
# --index-url/-i is inspected: every caller here passes it as a separate
# argument, and --extra-index-url is not used anywhere in this directory.
pip_index_host() {
  local arg prev="" url="" host=""
  for arg in "$@"; do
    case "$arg" in
      --index-url=*|-i=*) url="${arg#*=}" ;;
      *)
        if [[ "$prev" == "--index-url" || "$prev" == "-i" ]]; then
          url="$arg"
        fi
        ;;
    esac
    if [[ -n "$url" ]]; then
      host="${url#*://}"
      host="${host%%/*}"
      printf '%s' "${host%%:*}"
      return 0
    fi
    prev="$arg"
  done
  return 0
}

# Make sure pip will be able to look up the index host $1, and leave the
# environment able to. Three cases, and only the last one stops the install.
#
#   * The runner proxies the host: the proxy does the lookup, so the pod's own
#     resolution proves nothing and is not consulted. Reading the proxy variables
#     directly rather than asking urllib keeps the answer the same on a Linux pod
#     and on a developer's laptop, where a system-wide proxy setting would
#     otherwise decide the outcome.
#   * The pod resolves the host for itself and can: nothing to do.
#   * The pod must resolve the host for itself and cannot, because the runner
#     exempted it from the proxy. The exemption is then the whole problem, and it
#     is worth undoing rather than reporting: the Ascend runner at
#     /root/flagcicd carries NO_PROXY=...,resource.flagos.net,*.flagos.net, and
#     resource.flagos.net does not resolve there at all (run 38478507163, and
#     again on this branch's run 38491926656), while the sibling Ascend runner --
#     same proxy, no exemption -- fetched the 535.9 MB FlagTree wheel through it
#     in 22 s. So print "re-route" with the un-exempted NO_PROXY/no_proxy for the
#     caller to adopt. Only a pod with no proxy to fall back on, or one whose list
#     is bypassed wholesale ("*"), is reported unreachable.
#
# A host that a retry cannot help must never reach pip_retry's loop: five attempts
# against the unresolvable resource.flagos.net in run 38409565812 spent 21.5
# minutes -- most of a 27.9-minute Ascend job -- before reporting it.
#
# Prints nothing when nothing has to change, and exits 1 for the unreachable case
# with the reason on stderr.
ensure_index_host_route() {
  local out="" status=0
  out="$("${PIP_RETRY_PYTHON:-${VENV_PYTHON:-python}}" - "$1" <<'PY'
import os
import socket
import sys

host = sys.argv[1]
SEP = "\x1f"


def first_env(*names):
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return ""


def covers(entry, host):
    """True when one NO_PROXY entry exempts `host` from the proxy.

    Accepts the spellings the runner lists use: a bare host, a ``.domain`` or
    ``*.domain`` suffix, and the list-wide ``*``. The Ascend runner carries both
    ``resource.flagos.net`` and ``*.flagos.net`` for the same host, so matching
    only the exact entry would leave the wildcard behind and pip would still
    dial it directly.
    """
    entry = entry.strip()
    if not entry:
        return False
    if entry == "*" or entry == host or entry == f"{host}:443":
        return True
    domain = entry.split(":")[0].lstrip(".")
    if domain.startswith("*."):
        domain = domain[2:]
    return bool(domain) and (host == domain or host.endswith(f".{domain}"))


def exempt(no_proxy, host):
    return any(covers(entry, host) for entry in no_proxy.split(","))


def without(no_proxy, host):
    """`no_proxy` with every host-specific entry that exempts `host` removed.

    A list-wide "*" stays: it bypasses the proxy for every host, and dropping it
    to route one host would send the whole job through the proxy. The caller
    reports that case instead of rewriting the list.
    """
    kept = []
    for entry in no_proxy.split(","):
        entry = entry.strip()
        if entry and (entry == "*" or not covers(entry, host)):
            kept.append(entry)
    return ",".join(kept)

proxy = first_env(
    "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy"
)
no_proxy = first_env("NO_PROXY", "no_proxy")
if proxy and not exempt(no_proxy, host):
    print(
        f"{host} is reached through the runner's HTTP proxy; not resolving it here",
        file=sys.stderr,
    )
    sys.exit(0)

try:
    socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
except OSError as exc:
    if not proxy:
        print(f"{host} does not resolve from this runner: {exc}", file=sys.stderr)
        sys.exit(1)
    upper, lower = os.environ.get("NO_PROXY", ""), os.environ.get("no_proxy", "")
    new_upper, new_lower = without(upper, host), without(lower, host)
    if exempt(new_upper, host) or exempt(new_lower, host):
        # A list-wide entry ("*") exempts the host whatever else is dropped, so
        # the proxy is not a route this check can open.
        print(f"{host} does not resolve from this runner: {exc}", file=sys.stderr)
        sys.exit(1)
    print(
        f"{host} does not resolve directly ({exc}); it is exempted from the "
        "runner's HTTP proxy, so the proxy is used for it instead",
        file=sys.stderr,
    )
    print(SEP.join(("re-route", new_upper, new_lower)))
PY
)" || status=$?
  if (( status != 0 )); then
    return "$status"
  fi
  [[ -n "$out" ]] || return 0
  # The two new lists are separator-delimited rather than quoted: an NO_PROXY
  # value is a comma-separated host list, so it holds neither a newline nor this
  # separator, and the caller has to print it back verbatim.
  local IFS=$'\x1f'
  local directive="" upper="" lower=""
  read -r directive upper lower <<< "$out"
  if [[ "$directive" == "re-route" ]]; then
    export NO_PROXY="$upper"
    export no_proxy="$lower"
  fi
  return 0
}

# True when pip's own output shows an error that a retry cannot fix: the pod
# cannot resolve the index host, and that condition does not clear while the job
# runs. Only name-resolution failures qualify.
#
# pip's "Could not find a version that satisfies" / "No matching distribution
# found" are deliberately *not* matched, even though a DNS failure prints them
# too. Measured: an index answering 502 prints exactly those two lines, and a
# 502 can clear, so treating them as permanent would turn a recoverable outage
# into a failed job. The host pre-check in pip_retry covers the static case; the
# match below covers a failure that starts after the pre-check passed.
pip_error_is_permanent() {
  grep -qE \
    -e 'Failed to resolve' \
    -e 'Temporary failure in name resolution' \
    -e 'NameResolutionError' \
    "$1"
}

# pip install with retries for large wheels on unstable networks.
pip_retry() {
  local python_exe="${PIP_RETRY_PYTHON:-${VENV_PYTHON:-python}}"
  local timeout="${PIP_RETRY_TIMEOUT:-300}"
  local -a extra=()
  if [[ "${PIP_RETRY_NO_CACHE:-1}" == "1" ]]; then
    extra+=(--no-cache-dir)
  fi
  local index_host="" log_file="" attempt=1 route_status=0
  index_host="$(pip_index_host "$@")"
  if [[ -n "$index_host" ]]; then
    # Exit 1 is the route check answering "nothing in this job can reach that
    # host"; anything else means the check itself did not run (no interpreter on
    # PATH, say), which says nothing about the index and must not stop a working
    # install. A zero exit may have un-exempted the host from the proxy, which is
    # why the check is asked before pip runs rather than after it fails.
    ensure_index_host_route "$index_host" || route_status=$?
    if (( route_status == 1 )); then
      echo "::error::Cannot reach the package index host '${index_host}': it does not resolve from this pod and no HTTP proxy is configured to look it up. Not retrying, because name resolution does not recover inside a job pod. If this host is only reachable through the runner's HTTP proxy, check that proxy's allowlist and the pod's NO_PROXY (see .github/scripts/lib/proxy_route.sh)."
      return 1
    fi
    if (( route_status != 0 )); then
      echo "::warning::Could not check how this pod reaches the package index host '${index_host}' (exit ${route_status}); installing without the check."
    fi
  fi
  # Tee the output so each failed attempt can be classified without losing the
  # live log a human reads while the job runs.
  log_file="$(mktemp)"
  while true; do
    # Raise pip's own retry limit and timeout for large wheels on unstable
    # networks: the FlagTree wheel is hundreds of MB, and pip's default timeout
    # (15s) and retries (5) are not enough when the mirror link drops
    # mid-download.
    if "$python_exe" -m pip install --retries 10 --timeout "$timeout" \
      ${extra[@]+"${extra[@]}"} "$@" 2>&1 | tee "$log_file"; then
      break
    fi
    if pip_error_is_permanent "$log_file"; then
      echo "::error::pip install hit an error that retrying cannot fix, and gave up after ${attempt} attempt(s): $*"
      rm -f "$log_file"
      return 1
    fi
    if (( attempt >= 5 )); then
      echo "::error::pip install failed after $attempt attempts: $*"
      rm -f "$log_file"
      return 1
    fi
    echo "::warning::pip install attempt $attempt failed; retrying: $*"
    attempt=$((attempt + 1))
    sleep 10
  done
  rm -f "$log_file"
  return 0
}

# A published wheel has a fixed version and an index-provided SHA-256. Check the
# prebuilt venv before downloading it, then verify the result so a stale image
# package cannot silently take precedence over the requested release.
flag_gems_installed() {
  "${VENV_PYTHON:-python}" - "$FLAGGEMS_VERSION" <<'PY'
import importlib.metadata as metadata
import importlib.util
import sys

try:
    assert metadata.version("flag_gems") == sys.argv[1]
    assert importlib.util.find_spec("flag_gems") is not None
except (AssertionError, metadata.PackageNotFoundError):
    raise SystemExit(1)
PY
}

# Install only the binary release, never a source distribution or Git checkout.
install_flag_gems() {
  if flag_gems_installed; then
    echo "Using preinstalled FlagGems $FLAGGEMS_VERSION"
    return 0
  fi
  pip_retry --no-deps --only-binary=:all: --index-url "$FLAGGEMS_INDEX_URL" \
    "flag-gems===$FLAGGEMS_VERSION"
  if ! flag_gems_installed; then
    echo "::error::FlagGems $FLAGGEMS_VERSION is not importable after wheel installation"
    return 1
  fi
}

# Drop a vendor-torch root from a colon-separated path list, so the isolated
# venv does not inherit the image's vendor torch.
strip_vendor_paths() {
  local value="${1:-}"
  local entry
  local -a entries=()
  local -a kept=()
  IFS=: read -ra entries <<< "$value"
  for entry in "${entries[@]}"; do
    [[ -z "$entry" ]] && continue
    case "$entry" in
      "$VENDOR_TORCH_ROOT"|"$VENDOR_TORCH_ROOT"/*) ;;
      *) kept+=("$entry") ;;
    esac
  done
  local joined=""
  for entry in "${kept[@]}"; do
    joined="${joined:+$joined:}$entry"
  done
  printf '%s' "$joined"
}

# True when the venv interpreter exists and has pip.
venv_is_usable() {
  [[ -x "$VENV_PYTHON" ]] || return 1
  "$VENV_PYTHON" -m pip --version >/dev/null 2>&1
}

# ---------------------------------------------------------------------------
# Shared provisioning phases.
#
# The seven scripts ran the same steps in the same order with only the SDK
# discovery, vendor-interpreter choice and asset staging differing. These five
# are the steps that were byte-for-byte the same modulo the pins, so they live
# here too; the platform scripts call them instead of repeating the command.

# Throughput $1 delivers for a large wheel, in bytes per second, as pip's own
# HTTP stack would see it: the probe inherits the probe process's proxy
# environment and reads the wheel the interpreter in this venv would actually
# install, so the number is comparable with the download it is choosing between.
# Bounded to ${2:-12}s and ${3:-4000000} bytes, and prints 0 when the index does
# not answer at all.
index_download_bps() {
  "${PIP_RETRY_PYTHON:-${VENV_PYTHON:-python}}" - "$1" "${2:-12}" "${3:-4000000}" <<'PY'
import re
import sys
import sysconfig
import time
import urllib.parse
import urllib.request

index, budget, target = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
listing_url = index.rstrip("/") + "/torch/"


def open_url(url, headers=None):
    request = urllib.request.Request(url, headers=headers or {})
    return urllib.request.urlopen(request, timeout=budget)


try:
    with open_url(listing_url) as response:
        listing = response.read().decode("utf-8", "replace")
    # A torch.org-style index appends a "#sha256=..." fragment and the mirror
    # uses root-relative hrefs, so match the wheel path and resolve it against
    # the listing URL rather than string-concatenating.
    hrefs = re.findall(r'href="([^"]*torch-[^"]*\.whl)(?:#[^"]*)?"', listing)
    wheels = [urllib.parse.urljoin(listing_url, href) for href in hrefs]
    if not wheels:
        print(f"{index} lists no torch wheel", file=sys.stderr)
        print(0)
        sys.exit(0)
    # Prefer the wheel for this interpreter and architecture; fall back to any
    # wheel, because the probe only needs a large file to read bytes from.
    cp_tag = "cp" + sysconfig.get_python_version().replace(".", "")
    machine = sysconfig.get_platform().split("-")[-1]
    preferred = [w for w in wheels if f"-{cp_tag}-" in w and machine in w]
    url = preferred[0] if preferred else wheels[0]
    started = time.monotonic()
    read = 0
    with open_url(url, {"Range": f"bytes=0-{target - 1}"}) as response:
        while read < target and time.monotonic() - started < budget:
            chunk = response.read(262144)
            if not chunk:
                break
            read += len(chunk)
    elapsed = time.monotonic() - started
    print(int(read / elapsed) if elapsed > 0 and read else 0)
except Exception as exc:  # any transport failure means "do not prefer this"
    print(f"{index} probe failed: {exc!r}", file=sys.stderr)
    print(0)
PY
}

# Print whichever of the two CPU-torch indexes measured faster. Both serve the
# same wheel, so this is not a correctness choice: CPU_TORCH_INDEX_URL is the
# pin and the mirror is a cache of it, and they differ by 10-100x in throughput
# from the vendor pods (see version-pins.env). Measuring rather than assuming
# keeps the slower one from being used on a pod whose route to the mirror is
# worse than its route offshore. Ties and probe failures keep the pin.
faster_cpu_torch_index() {
  local primary="$1" mirror="$2" primary_bps="" mirror_bps="" index=""
  primary_bps="$(index_download_bps "$primary")"
  mirror_bps="$(index_download_bps "$mirror")"
  # The caller captures this function's stdout as the chosen index, so the
  # diagnostic goes to stderr where it stays visible in the job log.
  echo "CPU torch index probe: ${primary} at ${primary_bps} B/s, ${mirror} at ${mirror_bps} B/s" >&2
  index="$primary"
  if [[ "$mirror_bps" =~ ^[0-9]+$ && "$primary_bps" =~ ^[0-9]+$ ]] && (( mirror_bps > primary_bps )); then
    index="$mirror"
  fi
  printf '%s' "$index"
}

# Install the CPU-only PyTorch the isolated venv is built on.
#
# The pinned index is authoritative, but it is offshore and the pods measure
# 150-480 kB/s there for this 146-190 MB wheel: 5 minutes on Ascend and 17-21
# minutes on the GCU and Kunlunxin build jobs. A domestic mirror of the same
# index was 10-100x faster. Either way the install is retried and either way an
# unreachable index now fails in seconds rather than in five retry rounds.
install_cpu_torch() {
  local mirror="${TORCH_FL_CPU_TORCH_MIRROR_INDEX_URL:-${CPU_TORCH_MIRROR_INDEX_URL_DEFAULT:-}}"
  local index="$CPU_TORCH_INDEX_URL"
  if [[ -n "$mirror" && "$mirror" != "$CPU_TORCH_INDEX_URL" ]]; then
    index="$(faster_cpu_torch_index "$CPU_TORCH_INDEX_URL" "$mirror")"
  fi
  pip_retry --index-url "$index" "$@" "torch==${CPU_TORCH_VERSION}"
}

# Install the published FlagTree wheel for this platform (its Triton build).
install_flagtree() {
  pip_retry --no-deps --only-binary=:all: --index-url "$FLAGTREE_INDEX_URL" \
    "flagtree===${FLAGTREE_VERSION}"
}

# Install the published FlagCX wheel and point torch at the flagos backend.
install_flagcx() {
  pip_retry --no-deps --only-binary=:all: --index-url "$FLAGGEMS_INDEX_URL" \
    "flagcx===${FLAGCX_VERSION}"
  export FLAGCX_TORCH_BACKEND=flagos
}

# Build the in-tree extension so package_data sees libtorch_fl.so before the
# common workflow runs `python -m build`.
build_flagos_inplace() {
  cd "$REPO_ROOT"
  python setup.py build_ext --inplace
}

# Isolate pytest from user-site and source-only imports. The hook checks this
# function on the job's bash 3.2; a capture is an assignment there, which would
# hide a failed write to GITHUB_PATH when sourced under `set -e`.
persist_python_import_isolation() {
  export PYTHONNOUSERSITE=1
  export PYTHONPATH=""
  if [[ -n "${GITHUB_PATH:-}" && -n "${VIRTUAL_ENV:-}" ]]; then
    printf '%s\n' "$VIRTUAL_ENV/bin" >> "$GITHUB_PATH"
  fi
  if [[ -n "${GITHUB_ENV:-}" ]]; then
    {
      echo "PYTHONNOUSERSITE=1"
      echo "PYTHONPATH="
      if [[ -n "${VIRTUAL_ENV:-}" ]]; then
        echo "VIRTUAL_ENV=$VIRTUAL_ENV"
      fi
    } >> "$GITHUB_ENV"
  fi
}

# Persist the named variables to $GITHUB_ENV for later workflow steps. Every
# name must be assigned before the call: they are read with eval under `set -u`.
export_ci_env() {
  [[ -n "${GITHUB_ENV:-}" ]] || return 0
  local name value=""
  # Indirection is a bash 4 feature. macOS still ships bash 3.2, so get each
  # value with printf -v. Its optional `-v name` is unambiguous; the format is
  # fixed, so a value such as --help cannot be read as printf's option.
  for name in "$@"; do
    eval "printf -v value '%s' \"\${$name}\""
    printf '%s=%s\n' "$name" "$value" >> "$GITHUB_ENV"
  done
}
