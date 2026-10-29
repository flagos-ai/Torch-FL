"""Regeneration preserves shared metadata-view registrations."""

import importlib.util
from pathlib import Path


def test_shared_view_macros_keep_ascend_kernels(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "ascend_codegen_registration", root / "scripts/codegen/codegen_ascend.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / "views.cc").write_text(
        "#define VIEWS(device) \\\n"
        "  REGISTER_IMPL_TO_DISPATCHER(PermuteFn, permute_dispatcher, device, permute)\n"
        "VIEWS(Backend::kAscend)\n"
        "#define OTHER(device) \\\n"
        "  REGISTER_IMPL_TO_DISPATCHER(OtherFn, other_dispatcher, device, other)\n"
        "OTHER(Backend::kApple)\n"
        "REGISTER_IMPL_TO_DISPATCHER(SignFn, sign_dispatcher, Backend::kAscend, sign)\n"
    )
    monkeypatch.setattr(module, "HANDWRITTEN_DIRS", (tmp_path,))
    assert module._handwritten_dispatchers() == {"permute", "sign"}
