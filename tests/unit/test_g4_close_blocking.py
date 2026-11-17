# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The G4 storage worker blocked in query_memory or a file call: close and breaker."""

import atexit
import os
import threading
import time

import pytest
from _g4_test_utils import digest, fetching_g4, new_world, wait_for
from _kvcr_test_utils import _OPEN_KVCRS

from kvcr.backends import unmanaged_storage
from kvcr.config import G4Options
from kvcr.runtime import progress
from kvcr.types import BlockKey, QueryStatus

KEY = BlockKey(b"hung-head")


@pytest.fixture()
def short_bounds(monkeypatch):
    monkeypatch.setattr(progress, "_OP_CLEANUP_TIMEOUT_SECONDS", 0.1)
    monkeypatch.setattr(progress, "_JOIN_TIMEOUT_SECONDS", 0.1)


def _hung_check(w):
    w.storage.query_gate = threading.Event()
    assert w.kvcr.query([KEY], "r") == [fetching_g4()]
    wait_for(lambda: w.storage.query_blocked == 1)
    return w.storage.query_gate


@pytest.mark.parametrize("flavor", ["object", "file"])
def test_g4_close_waits_out_a_blocking_query_within_its_bound(
    g4env, tmp_path, flavor, short_bounds
):
    threads = threading.active_count()
    w = new_world(g4env, tmp_path, flavor=flavor)
    gate = _hung_check(w)
    # Past the cleanup and join bounds, inside close_timeout_ms.
    timer = threading.Timer(0.5, gate.set)
    timer.start()
    try:
        w.kvcr.close()
    finally:
        timer.join()
    assert w.kvcr._core.is_quiescent()
    assert w.storage.leaks() == []
    assert threading.active_count() == threads


def test_g4_close_timeout_ms_defaults_to_thirty_seconds():
    options = G4Options(flavor="object", key_prefix="p", key_layout=("",))
    assert options.close_timeout_ms == 30_000


@pytest.mark.parametrize("flavor", ["object", "file"])
def test_g4_close_gives_up_on_a_blocked_call_after_close_timeout_ms(
    g4env, tmp_path, flavor, short_bounds, monkeypatch
):
    hooks = []
    monkeypatch.setattr(
        atexit, "register", lambda func, *args, **kwargs: hooks.append((func, args))
    )
    w = new_world(g4env, tmp_path, flavor=flavor, close_timeout_ms=400)
    gate = _hung_check(w)
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="storage worker did not stop"):
        w.kvcr.close()
    assert time.monotonic() - started < 1.5
    assert len(hooks) == 1
    gate.set()
    func, args = hooks[0]
    func(*args)
    w.kvcr.close()
    assert w.kvcr._core.is_quiescent()
    _OPEN_KVCRS.remove(w.kvcr)


@pytest.mark.parametrize("flavor", ["object", "file"])
def test_g4_close_reports_no_error_when_the_worker_exits_as_the_block_clears(
    g4env, tmp_path, flavor, short_bounds, monkeypatch
):
    w = new_world(g4env, tmp_path, flavor=flavor)
    gate = _hung_check(w)
    # A second check waits behind the blocked one; the worker cancels it on
    # its way out, which keeps the thread alive just after the call returns.
    assert w.kvcr.query([BlockKey(b"queued")], "r2") == [fetching_g4()]
    cancel = unmanaged_storage._G4._try_cancel

    def slow_cancel(self, job):
        time.sleep(0.12)
        return cancel(self, job)

    monkeypatch.setattr(unmanaged_storage._G4, "_try_cancel", slow_cancel)
    # The call returns before the first join ends; the thread outlives it.
    timer = threading.Timer(0.02, gate.set)
    timer.start()
    try:
        w.kvcr.close()
    finally:
        timer.join()
    assert w.kvcr._core.is_quiescent()


@pytest.mark.parametrize("flavor", ["object", "file"])
def test_g4_opening_the_breaker_cancels_timed_out_checks_queued_behind_a_hung_call(
    g4env, tmp_path, flavor
):
    w = new_world(g4env, tmp_path, flavor=flavor, failure_threshold=2)
    keys = [BlockKey(b"queued-%d" % i) for i in range(3)]
    gate = _hung_check(w)
    assert w.kvcr.query(keys, "r") == [fetching_g4()] * 3
    # The hung check and the queued ones time out; two counts open the breaker.
    w.advance(1.5)
    assert w.kvcr.query(keys, "r") == [(QueryStatus.MISS, None)] * 3
    gate.set()
    w.settle()
    # Only the hung call reached storage; the queued checks were dropped.
    assert w.check_names() == [digest(KEY)]


def _blocked_fsync(w, monkeypatch):
    """Hold the storage worker inside the publish fsync of a spill."""
    gate = threading.Event()
    me = threading.get_ident()
    waiting = []
    real = os.fsync

    def blocked(fd):
        if threading.get_ident() != me:
            waiting.append(fd)
            gate.wait(10)
        return real(fd)

    monkeypatch.setattr(os, "fsync", blocked)
    assert w.deposit(KEY, 0).success
    w.deposit(BlockKey(b"next"), 1, wait=False)
    w.poll_until(lambda: bool(waiting))
    return gate


def test_g4_close_waits_out_a_blocked_file_call_within_close_timeout_ms(
    g4env, tmp_path, short_bounds, monkeypatch
):
    threads = threading.active_count()
    w = new_world(g4env, tmp_path, flavor="file")
    gate = _blocked_fsync(w, monkeypatch)
    timer = threading.Timer(0.5, gate.set)
    timer.start()
    try:
        w.kvcr.close()
    finally:
        timer.join()
    assert w.kvcr._core.is_quiescent()
    assert threading.active_count() == threads


def test_g4_close_gives_up_on_a_blocked_file_call_after_close_timeout_ms(
    g4env, tmp_path, short_bounds, monkeypatch
):
    hooks = []
    monkeypatch.setattr(
        atexit, "register", lambda func, *args, **kwargs: hooks.append((func, args))
    )
    w = new_world(g4env, tmp_path, flavor="file", close_timeout_ms=400)
    gate = _blocked_fsync(w, monkeypatch)
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="storage worker did not stop"):
        w.kvcr.close()
    assert time.monotonic() - started < 1.5
    assert len(hooks) == 1
    gate.set()
    func, args = hooks[0]
    func(*args)
    w.kvcr.close()
    assert w.kvcr._core.is_quiescent()
    _OPEN_KVCRS.remove(w.kvcr)


@pytest.mark.parametrize("flavor", ["object", "file"])
def test_g4_close_waits_no_longer_than_close_timeout_ms_in_total(
    g4env, tmp_path, flavor, short_bounds, monkeypatch
):
    monkeypatch.setattr(atexit, "register", lambda func, *args, **kwargs: None)
    w = new_world(g4env, tmp_path, flavor=flavor, close_timeout_ms=400)
    gate = _hung_check(w)
    thread = w.kvcr._core._g4._storage.thread
    join = thread.join
    waits = []

    def timed_join(timeout=None):
        waits.append(timeout)
        return join(timeout)

    monkeypatch.setattr(thread, "join", timed_join)
    with pytest.raises(RuntimeError, match="storage worker did not stop"):
        w.kvcr.close()
    assert None not in waits
    assert sum(waits) <= 0.4 + 1e-9
    monkeypatch.undo()
    gate.set()
    thread.join(5)
    w.kvcr.close()
    _OPEN_KVCRS.remove(w.kvcr)
