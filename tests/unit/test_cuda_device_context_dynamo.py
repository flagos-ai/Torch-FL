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

"""Unit tests for Bug #661: Dynamo CUDA device guard and multi-DCU FSDP compile.

Covers three invariants required for ``FSDP + torch.compile`` on ``flagos``:

1. ``torch_fl.compat.flaggems.device.prepare_cuda`` extends
   ``torch.cuda._utils._get_device_index`` (and ``torch.cuda._get_device_index``
   plus ``torch+cpu`` exchange stubs) so TorchDynamo's
   ``CUDADeviceVariable.create`` accepts ``flagos`` / ``privateuseone`` devices
   without raising ``ValueError: Expected a cuda device, but got: flagos:0`` or
   introducing graph breaks.
2. ``torch_fl.accelerator.hygon._dcu_compat._hip_total_memory`` restores the
   caller's active HIP device after calling ``hipSetDevice(device_index)`` for
   ``hipMemGetInfo``, so ``FlagOSDeviceInterface.Worker.get_device_properties``
   probing ``range(device_count())`` never leaves rank 0 bound to device N-1.
3. ``ProcessGroupFlagOS`` collective entrypoints and ``_HostStagedGloo._run``
   are marked ``@torch._dynamo.disable`` so skipped FSDP runtime hooks never
   trigger Dynamo tracing inside collective staging/view helpers.
"""

from __future__ import annotations

import ctypes
import importlib.util
from pathlib import Path
import sys
import types
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")

REPO_ROOT = Path(__file__).resolve().parents[2]
DCU_MODULE_PATH = REPO_ROOT / "torch_fl" / "accelerator" / "hygon" / "_dcu_compat.py"
DEVICE_MODULE_PATH = REPO_ROOT / "torch_fl" / "compat" / "flaggems" / "device.py"


def _ensure_flagos_backend_renamed():
    if torch._C._get_privateuse1_backend_name() != "flagos":
        torch.utils.rename_privateuse1_backend("flagos")


def _load_dcu_compat_isolated():
    spec = importlib.util.spec_from_file_location(
        "_test_dcu_compat_hip_restore", DCU_MODULE_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    fake_env = types.SimpleNamespace(flag=lambda *a, **k: False, value=lambda *a: "")
    fake_pkg = types.ModuleType("torch_fl")
    fake_pkg._env = fake_env
    with patch.dict(sys.modules, {"torch_fl": fake_pkg, "torch_fl._env": fake_env}):
        spec.loader.exec_module(mod)
    return mod


def _load_flaggems_device_module():
    try:
        from torch_fl.compat.flaggems import device as fg_device

        return fg_device
    except ImportError:
        pkg_root = types.ModuleType("torch_fl")
        pkg_root.__path__ = [str(REPO_ROOT / "torch_fl")]
        pkg_root._env = types.SimpleNamespace(flag=lambda *a, **k: False)
        pkg_compat = types.ModuleType("torch_fl.compat")
        pkg_compat.__path__ = [str(REPO_ROOT / "torch_fl" / "compat")]
        pkg_fg = types.ModuleType("torch_fl.compat.flaggems")
        pkg_fg.__path__ = [str(REPO_ROOT / "torch_fl" / "compat" / "flaggems")]
        pkg_fg.runtime = types.SimpleNamespace(routes_to_flaggems=lambda: True)
        spec = importlib.util.spec_from_file_location(
            "torch_fl.compat.flaggems.device", DEVICE_MODULE_PATH
        )
        mod = importlib.util.module_from_spec(spec)
        with patch.dict(
            sys.modules,
            {
                "torch_fl": pkg_root,
                "torch_fl.compat": pkg_compat,
                "torch_fl.compat.flaggems": pkg_fg,
                "torch_fl.compat.flaggems.runtime": pkg_fg.runtime,
            },
        ):
            spec.loader.exec_module(mod)
        return mod


@pytest.fixture()
def prepared_cuda_env(monkeypatch):
    _ensure_flagos_backend_renamed()
    fg_device = _load_flaggems_device_module()

    import torch.cuda._utils as cuda_utils

    state = {"current": 2, "count": 4}
    fake_flagos = types.SimpleNamespace(
        is_available=lambda: False,
        device_count=lambda: state["count"],
        current_device=lambda: state["current"],
        set_device=lambda idx: state.__setitem__("current", int(idx)),
    )
    real_flagos = getattr(torch, "flagos", None)
    if real_flagos is not None:
        monkeypatch.setattr(real_flagos, "device_count", fake_flagos.device_count)
        monkeypatch.setattr(real_flagos, "current_device", fake_flagos.current_device)
        monkeypatch.setattr(real_flagos, "set_device", fake_flagos.set_device)
    else:
        monkeypatch.setattr(torch, "flagos", fake_flagos, raising=False)

    import importlib

    graphs_mod = importlib.import_module("torch.cuda.graphs")
    saved_utils_gdi = cuda_utils._get_device_index
    saved_cuda_gdi = torch.cuda._get_device_index
    saved_init = torch.cuda.device.__init__
    saved_ex = torch.cuda._exchange_device
    saved_mex = torch.cuda._maybe_exchange_device
    saved_cuda_cap = getattr(torch.cuda, "is_current_stream_capturing", None)
    saved_graphs_cap = getattr(graphs_mod, "is_current_stream_capturing", None)
    saved_graphs_ccap = getattr(graphs_mod, "_cuda_isCurrentStreamCapturing", None)
    saved_subs = {}
    for sub_name in ("memory", "random", "streams"):
        sub = getattr(torch.cuda, sub_name, None)
        if sub is not None and hasattr(sub, "_get_device_index"):
            saved_subs[sub_name] = sub._get_device_index

    try:
        fg_device.prepare_cuda("hygon")
        yield types.SimpleNamespace(
            fg_device=fg_device,
            state=state,
            cuda_utils=cuda_utils,
            graphs_mod=graphs_mod,
        )
    finally:
        cuda_utils._get_device_index = saved_utils_gdi
        torch.cuda._get_device_index = saved_cuda_gdi
        torch.cuda.device.__init__ = saved_init
        torch.cuda._exchange_device = saved_ex
        torch.cuda._maybe_exchange_device = saved_mex
        if saved_cuda_cap is not None:
            torch.cuda.is_current_stream_capturing = saved_cuda_cap
        if saved_graphs_cap is not None:
            graphs_mod.is_current_stream_capturing = saved_graphs_cap
        if saved_graphs_ccap is not None:
            graphs_mod._cuda_isCurrentStreamCapturing = saved_graphs_ccap
        for sub_name, orig_fn in saved_subs.items():
            getattr(torch.cuda, sub_name)._get_device_index = orig_fn


class TestCudaGetDeviceIndexForFlagos:
    @pytest.mark.parametrize(
        "spec, expected",
        [
            (torch.device("privateuseone", 0), 0),
            (torch.device("privateuseone", 3), 3),
            ("privateuseone:1", 1),
            ("flagos:0", 0),
            ("flagos:3", 3),
            (0, 0),
            (5, 5),
        ],
    )
    def test_explicit_index_is_preserved(self, prepared_cuda_env, spec, expected):
        assert torch.cuda._get_device_index(spec, optional=True) == expected
        assert (
            prepared_cuda_env.cuda_utils._get_device_index(spec, optional=False)
            == expected
        )

    @pytest.mark.parametrize("spec", ["flagos", "privateuseone"])
    def test_indexless_flagos_uses_current_device_when_optional(
        self, prepared_cuda_env, spec
    ):
        prepared_cuda_env.state["current"] = 2
        assert torch.cuda._get_device_index(spec, optional=True) == 2
        assert torch.cuda._get_device_index(torch.device(spec), optional=True) == 2

    def test_indexless_flagos_returns_minus_one_when_zero_devices(
        self, prepared_cuda_env
    ):
        prepared_cuda_env.state["count"] = 0
        assert torch.cuda._get_device_index("flagos", optional=True) == -1

    @pytest.mark.parametrize("spec", ["flagos", "privateuseone"])
    def test_indexless_flagos_raises_when_not_optional(self, prepared_cuda_env, spec):
        with pytest.raises(ValueError, match="Expected a torch.device"):
            torch.cuda._get_device_index(spec, optional=False)
        with pytest.raises(ValueError, match="Expected a torch.device"):
            torch.cuda._get_device_index(torch.device(spec), optional=False)

    def test_cpu_and_invalid_devices_keep_upstream_semantics(self, prepared_cuda_env):
        assert torch.cuda._get_device_index("cpu", allow_cpu=True) == -1
        assert torch.cuda._get_device_index(torch.device("cpu"), allow_cpu=True) == -1
        with pytest.raises(ValueError, match="Expected a cuda"):
            torch.cuda._get_device_index("cpu", allow_cpu=False)

    def test_prepare_cuda_is_idempotent(self, prepared_cuda_env):
        first_gdi = torch.cuda._get_device_index
        first_init = torch.cuda.device.__init__
        prepared_cuda_env.fg_device.prepare_cuda("hygon")
        assert torch.cuda._get_device_index is first_gdi
        assert torch.cuda.device.__init__ is first_init

    def test_eager_and_dynamo_fullgraph_cuda_device_context(
        self, prepared_cuda_env, monkeypatch
    ):
        from torch._dynamo.device_interface import caching_worker_device_properties

        monkeypatch.setitem(
            caching_worker_device_properties,
            "flagos",
            [types.SimpleNamespace(major=8, minor=0) for _ in range(4)],
        )
        monkeypatch.setattr(
            torch.cuda, "get_device_capability", lambda device=None: (8, 0)
        )
        prepared_cuda_env.state["current"] = 0
        target_dev = torch.device("flagos", 1)

        if not hasattr(torch._C, "_cuda_exchangeDevice"):
            with torch.cuda.device(target_dev):
                assert prepared_cuda_env.state["current"] == 1
            assert prepared_cuda_env.state["current"] == 0

        torch._dynamo.reset()

        def scoped_fn(x):
            with torch.cuda.device(target_dev):
                return x * 2.0 + 1.0

        x = torch.ones(4)
        explanation = torch._dynamo.explain(scoped_fn)(x)
        assert explanation.graph_break_count == 0
        compiled = torch.compile(scoped_fn, backend="eager", fullgraph=True)
        torch.testing.assert_close(compiled(x), x * 2.0 + 1.0)

    def test_prepare_cuda_replaces_cpu_stream_capture_dummy_for_dynamo(
        self, prepared_cuda_env, monkeypatch
    ):
        graphs_mod = prepared_cuda_env.graphs_mod
        dummy = torch._utils._dummy_type("_cuda_isCurrentStreamCapturing")
        monkeypatch.setattr(graphs_mod, "_cuda_isCurrentStreamCapturing", dummy)

        prepared_cuda_env.fg_device.prepare_cuda("kunlunxin")
        assert torch.cuda.is_current_stream_capturing() is False
        assert graphs_mod.is_current_stream_capturing() is False

        monkeypatch.setattr(
            torch.flagos,
            "is_current_stream_capturing",
            lambda: True,
            raising=False,
        )
        assert torch.cuda.is_current_stream_capturing() is True
        monkeypatch.setattr(
            torch.flagos,
            "is_current_stream_capturing",
            lambda: False,
            raising=False,
        )

        torch._dynamo.reset()

        def pointwise_like_fn(x):
            if torch.cuda.is_current_stream_capturing():
                return x - 1.0
            return x * 3.0 + 2.0

        inp = torch.ones(4)
        explanation = torch._dynamo.explain(pointwise_like_fn)(inp)
        assert explanation.graph_break_count == 0
        compiled = torch.compile(pointwise_like_fn, backend="eager", fullgraph=True)
        torch.testing.assert_close(compiled(inp), inp * 3.0 + 2.0)

    def test_prepare_cuda_preserves_vendor_stream_capture_and_marks_constant(
        self, prepared_cuda_env, monkeypatch
    ):
        graphs_mod = prepared_cuda_env.graphs_mod
        capturing = {"active": False}

        def vendor_capture_query() -> bool:
            status = ctypes.c_int(1 if capturing["active"] else 0)
            _ = ctypes.byref(status)
            return status.value != 0

        monkeypatch.setattr(
            torch.cuda, "is_current_stream_capturing", vendor_capture_query
        )
        monkeypatch.setattr(
            graphs_mod, "is_current_stream_capturing", vendor_capture_query
        )
        monkeypatch.setattr(
            graphs_mod, "_cuda_isCurrentStreamCapturing", vendor_capture_query
        )

        prepared_cuda_env.fg_device.prepare_cuda("nvidia")
        assert torch.cuda.is_current_stream_capturing() is False
        capturing["active"] = True
        assert torch.cuda.is_current_stream_capturing() is True
        capturing["active"] = False

        torch._dynamo.reset()

        def vendor_compiled_fn(x):
            if torch.cuda.is_current_stream_capturing():
                return x - 1.0
            return x * 2.0

        inp = torch.ones(4)
        explanation = torch._dynamo.explain(vendor_compiled_fn)(inp)
        assert explanation.graph_break_count == 0
        compiled = torch.compile(vendor_compiled_fn, backend="eager", fullgraph=True)
        torch.testing.assert_close(compiled(inp), inp * 2.0)


class _FakeHipDriver:
    def __init__(
        self,
        current_device: int = 0,
        has_get_device: bool = True,
        get_device_ret: int = 0,
        meminfo_ret: int = 0,
        meminfo_raises: bool = False,
        total_bytes: int = 64 * 1024**3,
    ):
        self.current_device = current_device
        self.set_device_calls = []
        self._has_get_device = has_get_device
        self._get_device_ret = get_device_ret
        self._meminfo_ret = meminfo_ret
        self._meminfo_raises = meminfo_raises
        self._total_bytes = total_bytes
        if has_get_device:
            self.hipGetDevice = self._hip_get_device

    def _hip_get_device(self, dev_ptr):
        ctypes.cast(
            dev_ptr, ctypes.POINTER(ctypes.c_int)
        ).contents.value = self.current_device
        return self._get_device_ret

    def hipSetDevice(self, idx):
        self.set_device_calls.append(idx)
        self.current_device = idx
        return 0

    def hipMemGetInfo(self, free_ptr, total_ptr):
        if self._meminfo_raises:
            raise OSError("hipMemGetInfo failed")
        ctypes.cast(free_ptr, ctypes.POINTER(ctypes.c_size_t)).contents.value = (
            self._total_bytes // 2
        )
        ctypes.cast(
            total_ptr, ctypes.POINTER(ctypes.c_size_t)
        ).contents.value = self._total_bytes
        return self._meminfo_ret


class TestDcuHipTotalMemoryDeviceRestore:
    def test_probe_restores_previous_device_across_multi_device_scan(self):
        dcu = _load_dcu_compat_isolated()
        hip = _FakeHipDriver(current_device=0)
        for i in range(2):
            total = dcu._hip_total_memory(hip, i)
            assert total == 64 * 1024**3
        assert hip.current_device == 0
        assert hip.set_device_calls == [0, 0, 1, 0]

    def test_probe_restores_previous_device_when_meminfo_fails_or_raises(self):
        dcu = _load_dcu_compat_isolated()

        hip_err = _FakeHipDriver(current_device=0, meminfo_ret=1)
        assert dcu._hip_total_memory(hip_err, 1) == 0
        assert hip_err.current_device == 0
        assert hip_err.set_device_calls == [1, 0]

        hip_exc = _FakeHipDriver(current_device=0, meminfo_raises=True)
        assert dcu._hip_total_memory(hip_exc, 1) == 0
        assert hip_exc.current_device == 0
        assert hip_exc.set_device_calls == [1, 0]

    def test_probe_skips_restore_when_get_device_unavailable_or_fails(self):
        dcu = _load_dcu_compat_isolated()

        hip_no_getter = _FakeHipDriver(current_device=3, has_get_device=False)
        assert dcu._hip_total_memory(hip_no_getter, 1) == 64 * 1024**3
        assert hip_no_getter.set_device_calls == [1]

        hip_bad_getter = _FakeHipDriver(current_device=3, get_device_ret=1)
        assert dcu._hip_total_memory(hip_bad_getter, 1) == 64 * 1024**3
        assert hip_bad_getter.set_device_calls == [1]


def test_process_group_flagos_collectives_are_dynamo_disabled():
    pg = pytest.importorskip("torch_fl.comm.process_group")

    def _is_disabled(fn):
        return bool(getattr(fn, "_torchdynamo_disable", False))

    assert _is_disabled(pg._HostStagedGloo._run)
    for name in (
        "allreduce",
        "allreduce_coalesced",
        "allgather",
        "allgather_coalesced",
        "allgather_into_tensor_coalesced",
        "_allgather_base",
        "broadcast",
        "reduce",
        "reduce_scatter",
        "reduce_scatter_tensor_coalesced",
        "_reduce_scatter_base",
        "alltoall",
        "alltoall_base",
        "gather",
        "scatter",
        "send",
        "recv",
        "recv_anysource",
        "barrier",
        "monitored_barrier",
    ):
        assert _is_disabled(getattr(pg.ProcessGroupFlagOS, name)), name


class TestDcuLauncherDeviceInterfaceRegistration:
    @pytest.mark.parametrize("has_cuda_kernels", [False, True])
    def test_dcu_claims_cuda_launcher_interface_only_without_cuda_kernels(
        self, monkeypatch, has_cuda_kernels
    ):
        di = pytest.importorskip("torch_fl.compile.device_interface")
        pp = pytest.importorskip("torch_fl.compile.platform_profile")
        import torch._dynamo.device_interface as dyn_di

        for mod_name in ("platform_profile", "device_interface", "inductor_codegen"):
            monkeypatch.setattr(
                f"torch_fl.compile.{mod_name}.platform_profile",
                lambda: pp._DCU_PROFILE,
                raising=False,
            )
        monkeypatch.setattr(di, "_torch_has_cuda_kernels", lambda: has_cuda_kernels)
        for helper in (
            "_register_gpu_type",
            "_patch_device_properties",
            "_patch_inductor_benchmark_device",
            "_repair_stock_cuda_interface",
            "_patch_joint_graph_pattern_init",
            "_patch_benchmarker",
            "_patch_triton_warp_size",
            "_prime_has_triton",
        ):
            monkeypatch.setattr(di, helper, lambda: None, raising=False)
        monkeypatch.setattr(di, "is_native_accelerator", lambda: False)

        registered = {}
        monkeypatch.setattr(
            dyn_di,
            "register_interface_for_device",
            lambda name, iface: registered.__setitem__(name, iface),
        )

        di.register_flagos_device_interface()

        assert registered[di.DEVICE_TYPE] is di.FlagOSDeviceInterface
        assert registered["hip"] is di.FlagOSDeviceInterface
        if has_cuda_kernels:
            assert "cuda" not in registered
        else:
            assert registered["cuda"] is di.FlagOSDeviceInterface


class TestFlagosRecordStreamCompat:
    def test_coerce_privateuse1_stream_unwraps_vendor_and_flagos_facades(
        self, monkeypatch
    ):
        _ensure_flagos_backend_renamed()
        streams = pytest.importorskip("torch_fl.flagos._streams")
        streams._privateuse1_device_type.cache_clear()
        monkeypatch.setattr(
            torch.accelerator,
            "current_stream",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                RuntimeError("Ambiguous accelerator on dual-backend build")
            ),
        )

        class _FakeVendorStream:
            def __init__(self, handle, device_index=1):
                self.handle = handle
                self.device_index = device_index

            def synchronize(self):
                pass

        vendor_stream = _FakeVendorStream(0x7F00ABCD, device_index=1)
        coerced = streams.coerce_privateuse1_stream(
            vendor_stream, fallback_device_index=0
        )
        assert isinstance(coerced, torch.Stream)
        assert coerced.stream_id == 0x7F00ABCD
        assert coerced.device_index == 1
        assert coerced.device.type in ("flagos", "privateuseone")

        facade = types.SimpleNamespace(_stream=vendor_stream)
        coerced_facade = streams.coerce_privateuse1_stream(
            facade, fallback_device_index=0
        )
        assert isinstance(coerced_facade, torch.Stream)
        assert coerced_facade.stream_id == 0x7F00ABCD
        assert coerced_facade.device_index == 1

        null_handle_stream = _FakeVendorStream(None, device_index=2)
        coerced_null = streams.coerce_privateuse1_stream(
            null_handle_stream, fallback_device_index=0
        )
        assert isinstance(coerced_null, torch.Stream)
        assert coerced_null.stream_id == 0
        assert coerced_null.device_index == 2

    def test_coerce_privateuse1_stream_rejects_non_streams(self):
        streams = pytest.importorskip("torch_fl.flagos._streams")
        for invalid in ("not_a_stream", 123, None, object(), torch.ones(2)):
            assert streams.coerce_privateuse1_stream(invalid, 0) is None
        cpu_tensor = torch.ones(2)
        with pytest.raises(RuntimeError, match="unknown parameter type"):
            cpu_tensor.record_stream("not_a_stream")  # type: ignore[arg-type]

    def test_record_stream_is_dynamo_disabled_inside_no_dispatch(self):
        pytest.importorskip("torch_fl.flagos")
        assert getattr(torch.Tensor.record_stream, "_torchdynamo_disable", False)

        # Reproduce FSDP's `_no_dispatch_record_stream` called after a graph
        # break while Dynamo's C eval_frame callback remains active on thread.
        def fsdp_like_step(x):
            y = x * 2.0
            torch._dynamo.graph_break()
            with torch.utils._mode_utils.no_dispatch():
                with pytest.raises(RuntimeError, match="unknown parameter type"):
                    y.record_stream("not_a_stream")  # type: ignore[arg-type]
            return y + 1.0

        torch._dynamo.reset()
        compiled = torch.compile(fsdp_like_step, backend="eager")
        inp = torch.ones(4)
        torch.testing.assert_close(compiled(inp), inp * 2.0 + 1.0)

    def test_dcu_device_properties_exposes_max_threads_per_block(self):
        dcu = _load_dcu_compat_isolated()
        props = dcu._DcuDeviceProperties(name="Hygon DCU 0")
        assert props.max_threads_per_block == 1024
