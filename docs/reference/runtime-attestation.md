<!--
Copyright 2026 FlagOS Contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Runtime identity and FlagQuantum attestation

Torch-FL owns the mapping from the portable `flagos` device to the physical
accelerator. `torch_fl.runtime_identity()` exposes that mapping without making
a certification claim. Provisioning systems can additionally run Torch-FL's
device-route canary and issue a signed attestation accepted by FlagQuantum's
domestic single-card certification harness.

## Trust boundary

The attestation is intentionally fail-closed:

- Torch-FL derives the accelerator class from its build and platform signals.
  A normal CUDA build is classified as `cuda_reference`/NVIDIA and cannot issue
  a domestic-accelerator attestation. PPU is recognized only with Torch-FL's PPU
  platform signal because its libtorch is CUDA compatible.
- The probe requires `flagos:0`, checks the result remains a `flagos` device,
  and rejects any `[flagos cpu_fallback]` emitted by the C++ fallback handler.
- The Torch-FL revision and container image must use immutable full SHA values.
- The canonical JSON payload is authenticated with HMAC-SHA256. The provisioner
  must store and distribute the key through its secret-management system.

The no-fallback statement covers `torch_fl_device_route_canary_v1`, not every
PyTorch operator. FlagQuantum separately runs its P0-P5 workload matrix and
binds that evidence to this attestation's revision and image digest.

## Provisioner command

Set the fallback logger before starting Python so its native one-time setting
is active for the probe. Never put the signing key on the command line.

```bash
export FLAGOS_LOG_FALLBACK=1
export TORCH_FL_ATTESTATION_SIGNING_KEY="$(secret-tool read torch-fl-attestation)"

python -m torch_fl.provisioning.attestation \
  --provisioner flagos-lab \
  --device-model "Hygon DCU Z100" \
  --architecture gfx926 \
  --device-identifier DCU-serial-0001 \
  --driver-version 6.3.1 \
  --vendor-runtime-version DTK-25.04 \
  --torch-fl-revision 0123456789abcdef0123456789abcdef01234567 \
  --container-image-digest sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef \
  --key-id flagos-lab-2026-01 \
  --output /secure/evidence/torch-fl-attestation.json
```

The command exits without writing verified evidence if hardware classification,
device availability, route integrity, fallback auditing, immutable identifiers,
or signing prerequisites are not satisfied.
