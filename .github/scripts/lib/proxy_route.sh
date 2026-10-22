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

# Reaching a host around the runner's HTTP proxy.
#
# A vendor runner pod injects HTTP(S)_PROXY into the job container along with its
# own allowlist and NO_PROXY list. resource.flagos.net is not on every one of
# those lists, and the failure mode reads like something else entirely: pip
# tunnels to it, the proxy refuses the CONNECT, and pip reports 'Tunnel
# connection failed: 500 Internal Server Error' as "Could not find a version that
# satisfies the requirement flagtree===" -- an unreachable index that looks like a
# missing wheel. The PPU pod is the known restricted one (its NO_PROXY list is
# localhost,127.0.0.1,10.1.12.192,10.1.12.38,harbor.baai.ac.cn,.example.com; two
# of those entries resolve to public addresses in the same /16 as
# resource.flagos.net, and the container is a NAT'd docker bridge off the runner
# host, so it does open public connections without the proxy). The sibling MUSA
# runner's proxy serves the same index fine, so the working route is a property of
# the pod rather than of the index.
#
# So probe the unproxied route and, when it answers, add just that host to
# NO_PROXY/no_proxy: pip's requests stack reads either case, git reads the
# lowercase one through libcurl. When the probe fails the environment is left
# exactly as the runner set it, so a pod that can only leave through the proxy
# keeps the previous behaviour instead of trading a 500 for a connect timeout.
# TORCH_FL_PROXY_ROUTE=direct|proxy skips the probe.
#
# Sourced rather than executed: these are definitions, with no top-level code, so
# a workflow step can `source` this file and call prefer_direct_route against the
# host it is about to use -- which is how the release upload reaches the vendor
# Nexus lanes from the platform that built the wheel.

add_no_proxy_host() {
  NO_PROXY="${NO_PROXY:+${NO_PROXY},}$1"
  no_proxy="${no_proxy:+${no_proxy},}$1"
  export NO_PROXY no_proxy
}

host_of_url() {
  local host="${1#*://}"
  host="${host%%/*}"
  printf '%s' "${host%%:*}"
}

# Succeeds if an unproxied GET of the URL gets an HTTP answer at all. The
# status is not the question: this probe only asks whether the pod can open the
# connection, and the index URL as configured has no trailing slash, so a 404
# here still proves the route works and a proxy-tunnelled request does not get
# that far. Only a transport failure -- DNS, connect, TLS, proxy refusal --
# counts as unreachable.
#
# VENV_PYTHON is not set in every caller (a workflow step that only wants the
# route has no venue to have created), so fall back to the interpreter on PATH.
direct_route_reachable() {
  "${VENV_PYTHON:-python}" - "$1" <<'PY'
import sys
import urllib.error
import urllib.request

# Trust whatever pip trusts -- certifi, not the system CA store -- so a store
# difference cannot make this probe fail where the real install would succeed.
handlers = [urllib.request.ProxyHandler({})]
try:
    import certifi
    import ssl

    handlers.append(
        urllib.request.HTTPSHandler(context=ssl.create_default_context(cafile=certifi.where()))
    )
except ImportError:
    pass
opener = urllib.request.build_opener(*handlers)
try:
    with opener.open(sys.argv[1], timeout=15) as response:
        status = response.status
except urllib.error.HTTPError as exc:
    status = exc.code
except Exception as exc:
    print(f"{sys.argv[1]} unreachable without the proxy: {exc!r}")
    sys.exit(1)
print(f"{sys.argv[1]} -> HTTP {status} without the proxy")
PY
}

# $1 = URL, $2 = what it serves, for the log line.
prefer_direct_route() {
  local url="$1" what="$2" host
  host="$(host_of_url "$url")"
  case "${TORCH_FL_PROXY_ROUTE:-auto}" in
    proxy)
      echo "Using the runner proxy for $host ($what): TORCH_FL_PROXY_ROUTE=proxy"
      return 0
      ;;
    direct)
      add_no_proxy_host "$host"
      echo "Bypassing the runner proxy for $host ($what): TORCH_FL_PROXY_ROUTE=direct"
      return 0
      ;;
  esac
  if direct_route_reachable "$url"; then
    add_no_proxy_host "$host"
    echo "Bypassing the runner proxy for $host ($what): it answers directly"
  else
    echo "::warning::Using the runner proxy for $host ($what), which did not answer without it. If the proxy then refuses the CONNECT with 500, this host has to be allowlisted on the runner's proxy or added to its NO_PROXY."
  fi
}
