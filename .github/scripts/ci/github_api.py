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

"""Small GitHub CLI wrapper used only by trusted CI control workflows."""

import json
import subprocess


class GitHubError(RuntimeError):
    pass


class GitHub:
    def __init__(self, repository):
        self.repository = repository

    def command(self, *args, payload=None):
        try:
            result = subprocess.run(
                ["gh", *args],
                input=json.dumps(payload) if payload is not None else None,
                capture_output=True,
                text=True,
                timeout=120,
            )
        except subprocess.TimeoutExpired as exc:
            raise GitHubError(
                "GitHub CLI request exceeded its 120-second budget"
            ) from exc
        if result.returncode:
            raise GitHubError(result.stderr.strip() or "GitHub CLI request failed")
        return result.stdout

    def api(self, path, method="GET", payload=None, paginate=False):
        args = ["api", path, "--method", method]
        if payload is not None:
            args += ["--input", "-"]
        if paginate:
            args += ["--paginate", "--slurp"]
        output = self.command(*args, payload=payload)
        return json.loads(output) if output.strip() else None

    def repo_api(self, suffix, **kwargs):
        return self.api(f"repos/{self.repository}/{suffix}", **kwargs)
