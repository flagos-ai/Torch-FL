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

import ast
import types
from torch._inductor.codegen import wrapper

original = wrapper.user_defined_triton_kernel_transitive_closure_source_code


def source_with_modules(kernel):
    source = original(kernel)
    used = {n.id for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Name)}
    imports = []
    for name in sorted(used):
        value = kernel.fn.__globals__.get(name)
        if isinstance(value, types.ModuleType):
            imports.append(f"import {value.__name__} as {name}")
    return source + "\n" + "\n".join(imports) + "\n"


wrapper.user_defined_triton_kernel_transitive_closure_source_code = source_with_modules
