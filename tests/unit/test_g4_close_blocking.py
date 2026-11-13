# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The G4 storage worker blocked inside query_memory: close and breaker."""

import atexit
import threading

import pytest
from _g4_test_utils import digest, fetching_g4, new_world, wait_for
from _kvcr_test_utils import _OPEN_KVCRS

from kvcr.backends import unmanaged_storage
from kvcr.runtime import progress
from kvcr.types import BlockKey, QueryStatus

KEY = BlockKey(b"hung-head")


@pytest.fixture()
def short_bounds(monkeypatch):
    monkeypatch.setattr(progress, "_OP_CLEANUP_TIMEOUT_SECONDS", 0.1)
    monkeypatch.setattr(progress, "_JOIN_TIMEOUT_SECONDS", 0.1)
    monkeypatch.setattr(unmanaged_storage, "_QUERY_SECONDS_PER_ROW", 1.0)
    monkeypatch.setattr(unmanaged_storage, "_QUERY_MARGIN_SECONDS", 0.0)


def _hung_check(w):
    w.storage.query_gate = threading.Event()
    assert w.kvcr.query([KEY], "r") == [fetching_g4()]
    wait_for(lambda: w.storage.query_blocked == 1)
    return w.storage.query_gate


def test_g4_close_waits_out_a_blocking_query_within_its_bound(
    g4env, tmp_path, short_bounds
):
    threads = threading.active_count()
    w = new_world(g4env, tmp_path, flavor="object")
    gate = _hung_check(w)
    # Past the cleanup and join bounds, inside the per-row query bound.
    timer = threading.Timer(0.5, gate.set)
    timer.start()
    try:
        w.kvcr.close()
    finally:
        timer.join()
    assert w.kvcr._core.is_quiescent()
    assert w.storage.leaks() == []
    assert threading.active_count() == threads


def test_g4_close_past_the_query_bound_raises_and_joins_at_exit(
    g4env, tmp_path, short_bounds, monkeypatch
):
    hooks = []
    monkeypatch.setattr(
        atexit, "register", lambda func, *args, **kwargs: hooks.append((func, args))
    )
    w = new_world(g4env, tmp_path, flavor="object")
    gate = _hung_check(w)
    with pytest.raises(RuntimeError, match="storage worker did not stop"):
        w.kvcr.close()
    assert not w.kvcr._core.is_quiescent()
    assert len(hooks) == 1
    gate.set()
    func, args = hooks[0]
    func(*args)
    assert not w.kvcr._core._g4._storage.thread.is_alive()
    # Once the call returns, a second close releases everything.
    w.kvcr.close()
    assert w.kvcr._core.is_quiescent()
    assert w.storage.leaks() == []
    _OPEN_KVCRS.remove(w.kvcr)


def test_g4_opening_the_breaker_cancels_timed_out_checks_queued_behind_a_hung_call(
    g4env, tmp_path
):
    w = new_world(g4env, tmp_path, flavor="object", failure_threshold=2)
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
