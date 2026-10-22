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

"""Unit tests for the gloo-hostage breaks: staged gloo + gloo redirection.

Issue #263: on a flagos host ``init_process_group(backend="gloo")`` builds a
ProcessGroupGloo that rejects flagos tensors. Two mechanisms fix that, and both
are pure logic that can be driven with fakes here:

* ``_try_build_staged_gloo`` -- the last tier of ``_build_inner``, which wraps a
  real gloo group in the ``_HostStagedGloo`` facade.
* ``redirect_gloo_requests`` -- the interposition on ``init_process_group`` /
  ``new_group`` that answers a gloo request with the ``flagos`` backend.

No GPU, no NCCL and no real rendezvous: the gloo class and the accelerator query
are faked, so the decision table is asserted rather than the transport.

Run: pytest tests/unit/test_gloo_redirect.py
"""

import pytest

pg = pytest.importorskip("torch_fl.comm.process_group")
dist = pg.dist


# ---------------------------------------------------------------------------
# _gloo_requests_are_redirected
# ---------------------------------------------------------------------------


class _FakeAccelerator:
    def __init__(self, type_name):
        self.type = type_name


class _FakeC:
    """Stand-in for torch._C, only the accelerator query is read."""

    def __init__(self, type_name=None, raises=False):
        self._accelerator = _FakeAccelerator(type_name)
        self._raises = raises

    def _get_accelerator(self):
        if self._raises:
            raise RuntimeError("no accelerator")
        return self._accelerator


def _fake_accelerator(monkeypatch, type_name=None, raises=False):
    monkeypatch.setattr(pg.torch, "_C", _FakeC(type_name, raises))


def test_redirect_on_when_the_accelerator_is_the_flagos_device(monkeypatch):
    _fake_accelerator(monkeypatch, "flagos")
    assert pg._gloo_requests_are_redirected() is True


def test_redirect_on_for_the_privateuseone_spelling(monkeypatch):
    # Both spellings denote the same device; a vendor build may report either.
    _fake_accelerator(monkeypatch, "privateuseone")
    assert pg._gloo_requests_are_redirected() is True


def test_redirect_off_for_a_cuda_host(monkeypatch):
    # gloo is a legitimate backend for cpu/cuda tensors: leave it alone there.
    _fake_accelerator(monkeypatch, "cuda")
    assert pg._gloo_requests_are_redirected() is False


def test_redirect_off_when_there_is_no_accelerator_query(monkeypatch):
    _fake_accelerator(monkeypatch, raises=True)
    assert pg._gloo_requests_are_redirected() is False


def test_redirect_switch_disables_it_on_a_flagos_host(monkeypatch):
    _fake_accelerator(monkeypatch, "flagos")
    monkeypatch.setenv("FLAGOS_DIST_REDIRECT_GLOO", "0")
    assert pg._gloo_requests_are_redirected() is False


# ---------------------------------------------------------------------------
# _needs_gloo_redirect / _redirect_backend_argument
# ---------------------------------------------------------------------------


def test_only_a_gloo_request_is_redirected(monkeypatch):
    _fake_accelerator(monkeypatch, "flagos")
    assert pg._needs_gloo_redirect("gloo") is True
    assert pg._needs_gloo_redirect("GLOO") is True  # c10d compares case-folded
    assert pg._needs_gloo_redirect("nccl") is False
    assert pg._needs_gloo_redirect("flagos") is False
    assert pg._needs_gloo_redirect(None) is False  # backend=None inherits
    # Backend is a str subclass in this torch, so the enum spelling is caught too.
    assert pg._needs_gloo_redirect(dist.Backend("gloo")) is True
    assert pg._needs_gloo_redirect(dist.Backend("nccl")) is False


def test_needs_redirect_is_false_when_the_switch_is_off(monkeypatch):
    _fake_accelerator(monkeypatch, "flagos")
    monkeypatch.setenv("FLAGOS_DIST_REDIRECT_GLOO", "0")
    assert pg._needs_gloo_redirect("gloo") is False


def test_backend_rewritten_in_keyword_form(monkeypatch):
    _fake_accelerator(monkeypatch, "flagos")
    args, kwargs = pg._redirect_backend_argument(
        (), {"backend": "gloo", "rank": 0}, 0, "backend"
    )
    assert args == ()
    assert kwargs == {"backend": "flagos", "rank": 0}


def test_backend_rewritten_in_positional_form(monkeypatch):
    _fake_accelerator(monkeypatch, "flagos")
    args, kwargs = pg._redirect_backend_argument(("gloo", None, 2), {}, 0, "backend")
    assert args == ("flagos", None, 2)
    assert kwargs == {}


def test_positional_wins_over_keyword(monkeypatch):
    # init_process_group(backend, init_method, ...): once the positional slot is
    # filled the keyword cannot also have been passed, but if both somehow appear
    # the positional one is the value c10d will use.
    _fake_accelerator(monkeypatch, "flagos")
    args, kwargs = pg._redirect_backend_argument(
        ("gloo",), {"backend": "gloo"}, 0, "backend"
    )
    assert args == ("flagos",)
    assert kwargs == {"backend": "gloo"}


def test_non_gloo_backend_is_left_verbatim(monkeypatch):
    _fake_accelerator(monkeypatch, "flagos")
    args, kwargs = pg._redirect_backend_argument(("nccl",), {}, 0, "backend")
    assert args == ("nccl",)
    assert pg._redirect_backend_argument((), {"backend": "nccl"}, 0, "backend")[1] == {
        "backend": "nccl"
    }


def test_arguments_are_not_mutated_in_place(monkeypatch):
    # The wrappers are installed on torch.distributed for the whole process; a
    # caller's own kwargs dict must come back untouched.
    _fake_accelerator(monkeypatch, "flagos")
    kwargs = {"backend": "gloo"}
    _, out = pg._redirect_backend_argument((), kwargs, 0, "backend")
    assert kwargs == {"backend": "gloo"}
    assert out is not kwargs


# ---------------------------------------------------------------------------
# redirect_gloo_requests
# ---------------------------------------------------------------------------


@pytest.fixture
def restore_dist(monkeypatch):
    """Undo the interposition, which patches torch.distributed for the process."""
    saved = [
        (dist, "init_process_group"),
        (dist, "new_group"),
    ]
    c10d_module = getattr(dist, "distributed_c10d", None)
    if c10d_module is not None:
        saved += [(c10d_module, "init_process_group"), (c10d_module, "new_group")]
    originals = [(obj, name, getattr(obj, name)) for obj, name in saved]
    yield
    for obj, name, value in originals:
        setattr(obj, name, value)


def _record(monkeypatch):
    """Replace the two entry points with recorders and return the call log."""
    log = []

    def fake_init(*args, **kwargs):
        log.append(("init", args, kwargs))
        return "init-result"

    def fake_new(*args, **kwargs):
        log.append(("new", args, kwargs))
        return "new-result"

    monkeypatch.setattr(dist, "init_process_group", fake_init)
    monkeypatch.setattr(dist, "new_group", fake_new)
    return log


def test_redirect_serves_a_gloo_init_process_group(monkeypatch, restore_dist):
    _fake_accelerator(monkeypatch, "flagos")
    log = _record(monkeypatch)
    pg.redirect_gloo_requests()

    assert dist.init_process_group(backend="gloo", rank=0) == "init-result"
    assert log == [("init", (), {"backend": "flagos", "rank": 0})]


def test_redirect_serves_a_positional_gloo_new_group(monkeypatch, restore_dist):
    _fake_accelerator(monkeypatch, "flagos")
    log = _record(monkeypatch)
    pg.redirect_gloo_requests()

    assert dist.new_group([0, 1], None, "gloo") == "new-result"
    assert log == [("new", ([0, 1], None, "flagos"), {})]


def test_redirect_leaves_an_explicit_flagos_request_alone(monkeypatch, restore_dist):
    _fake_accelerator(monkeypatch, "flagos")
    log = _record(monkeypatch)
    pg.redirect_gloo_requests()

    dist.init_process_group(backend="flagos")
    assert log == [("init", (), {"backend": "flagos"})]


def test_redirect_is_inert_on_a_non_flagos_host(monkeypatch, restore_dist):
    _fake_accelerator(monkeypatch, "cuda")
    log = _record(monkeypatch)
    pg.redirect_gloo_requests()

    dist.init_process_group(backend="gloo")
    assert log == [("init", (), {"backend": "gloo"})]


def test_redirect_is_idempotent(monkeypatch, restore_dist):
    _fake_accelerator(monkeypatch, "flagos")
    log = _record(monkeypatch)
    pg.redirect_gloo_requests()
    once = dist.init_process_group
    pg.redirect_gloo_requests()
    assert dist.init_process_group is once  # not a second layer of wrapping

    dist.init_process_group(backend="gloo")
    assert log == [("init", (), {"backend": "flagos"})]


def test_redirect_marks_both_entry_points_and_the_defining_module(
    monkeypatch, restore_dist
):
    _fake_accelerator(monkeypatch, "flagos")
    _record(monkeypatch)
    pg.redirect_gloo_requests()

    assert getattr(dist.init_process_group, pg._REDIRECT_MARKER) is True
    assert getattr(dist.new_group, pg._REDIRECT_MARKER) is True
    c10d_module = dist.distributed_c10d
    # `from torch.distributed.distributed_c10d import init_process_group` has to
    # pick up the wrapper too, or the redirect only covers half the call sites.
    assert c10d_module.init_process_group is dist.init_process_group
    assert c10d_module.new_group is dist.new_group


# ---------------------------------------------------------------------------
# _try_build_staged_gloo
# ---------------------------------------------------------------------------


class _FakeGloo:
    """Records the rendezvous arguments the staged tier passes to gloo."""

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        _FakeGloo.instances.append(self)

    instances = []


@pytest.fixture
def fake_gloo(monkeypatch):
    _FakeGloo.instances = []
    monkeypatch.setattr(dist, "ProcessGroupGloo", _FakeGloo, raising=False)
    return _FakeGloo


@pytest.fixture(autouse=True)
def _reset_staged_warning(monkeypatch):
    """The staged-gloo warning is once-per-process; reset it per test."""
    monkeypatch.setattr(pg, "_STAGED_GLOO_WARNED", False)


def _bare_group():
    """A ProcessGroupFlagOS that never ran __init__ (no C++ base ctor)."""
    return pg.ProcessGroupFlagOS.__new__(pg.ProcessGroupFlagOS)


def test_staged_tier_wraps_a_gloo_group(monkeypatch, fake_gloo):
    monkeypatch.setenv("GEMS_VENDOR", "mthreads")
    obj = _bare_group()
    with pytest.warns(UserWarning, match="host-staged gloo"):
        assert obj._try_build_staged_gloo(object(), 0, 2, None) is True
    assert isinstance(obj._inner, pg._HostStagedGloo)
    # The facade must hand the real c10d Backend to _register_inner_backend.
    assert obj._inner.c10d_backend is fake_gloo.instances[0]


def test_staged_tier_forwards_the_timeout(monkeypatch, fake_gloo):
    obj = _bare_group()
    obj._try_build_staged_gloo(object(), 1, 4, "30s")
    assert fake_gloo.instances[0].args[1:3] == (1, 4)
    assert fake_gloo.instances[0].kwargs == {"timeout": "30s"}


def test_staged_tier_omits_the_timeout_when_there_is_none(monkeypatch, fake_gloo):
    obj = _bare_group()
    obj._try_build_staged_gloo(object(), 0, 1, None)
    assert fake_gloo.instances[0].kwargs == {}


def test_staged_switch_declines_the_tier(monkeypatch, fake_gloo):
    obj = _bare_group()
    monkeypatch.setenv("FLAGOS_DIST_STAGED_GLOO", "0")
    assert obj._try_build_staged_gloo(object(), 0, 1, None) is False
    assert obj._staged_skip_reason == "FLAGOS_DIST_STAGED_GLOO=0"
    assert fake_gloo.instances == []


def test_staged_tier_declines_without_a_store(monkeypatch, fake_gloo):
    # gloo rendezvouses through the store; without one there is nothing to build.
    obj = _bare_group()
    assert obj._try_build_staged_gloo(None, 0, 1, None) is False
    assert "store" in obj._staged_skip_reason
    assert fake_gloo.instances == []


def test_staged_tier_declines_without_processgroupgloo(monkeypatch):
    monkeypatch.delattr(dist, "ProcessGroupGloo")
    obj = _bare_group()
    assert obj._try_build_staged_gloo(object(), 0, 1, None) is False
    assert "without ProcessGroupGloo" in obj._staged_skip_reason


def test_staged_tier_reports_a_gloo_failure_instead_of_raising(monkeypatch):
    class _Refusing:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("incompatible function arguments\n" + "x" * 500)

    monkeypatch.setattr(dist, "ProcessGroupGloo", _Refusing, raising=False)
    obj = _bare_group()
    assert obj._try_build_staged_gloo(object(), 0, 1, None) is False
    # First line only: pybind's dump would otherwise bury the diagnostic.
    assert obj._staged_skip_reason.startswith("ProcessGroupGloo construction failed:")
    assert "x" * 10 not in obj._staged_skip_reason


def test_staged_warning_is_emitted_once(monkeypatch, fake_gloo):
    with pytest.warns(UserWarning, match="host-staged gloo"):
        _bare_group()._try_build_staged_gloo(object(), 0, 1, None)
    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter("error")
        assert _bare_group()._try_build_staged_gloo(object(), 0, 1, None) is True


def test_build_inner_returns_no_view_on_the_staged_tier(monkeypatch, fake_gloo):
    # The facade converts flagos operands itself, so a device view would be both
    # unnecessary and wrong (gloo wants cpu tensors, not cuda ones).
    monkeypatch.setenv("GEMS_VENDOR", "mthreads")
    monkeypatch.setattr(
        pg.ProcessGroupFlagOS, "_try_build_flagcx", lambda self, *a: False
    )
    monkeypatch.setattr(
        pg.ProcessGroupFlagOS, "_try_build_mccl", lambda self, *a: False
    )
    obj = _bare_group()
    with pytest.warns(UserWarning, match="host-staged gloo"):
        view = obj._build_inner(object(), 0, 2, None)
    assert view is None
    assert isinstance(obj._inner, pg._HostStagedGloo)


def test_staged_reason_appears_in_the_final_diagnostic(monkeypatch):
    monkeypatch.setenv("GEMS_VENDOR", "nvidia")
    monkeypatch.setenv("FLAGOS_DIST_STAGED_GLOO", "0")
    monkeypatch.setattr(
        pg.ProcessGroupFlagOS, "_try_build_flagcx", lambda self, *a: False
    )
    monkeypatch.setattr(
        pg.ProcessGroupFlagOS, "_try_build_nccl", lambda self, *a: False
    )
    obj = _bare_group()
    with pytest.raises(RuntimeError) as excinfo:
        obj._build_inner(object(), 0, 1, None)
    message = str(excinfo.value)
    assert "no suitable inner backend" in message
    assert "Host-staged gloo unavailable: FLAGOS_DIST_STAGED_GLOO=0." in message
