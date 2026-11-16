# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""close() at every G4 operation stage leaves nothing behind; worker failure."""

import logging
import os
import threading

import pytest
from _g4_test_utils import (
    WorkerCrash,
    crash_on,
    digest,
    fetchable_g4,
    fetching_g4,
    new_world,
    tmp_files,
    wait_for,
)
from _kvcr_test_utils import _OPEN_KVCRS

from kvcr.runtime import progress
from kvcr.types import BlockKey, CacheTier, QueryStatus

A = BlockKey(b"block-a")
B = BlockKey(b"block-b")
C = BlockKey(b"block-c")


def _open_fds_under(root):
    found = []
    for name in os.listdir("/proc/self/fd"):
        try:
            target = os.readlink(f"/proc/self/fd/{name}")
        except OSError:
            continue
        if target.startswith(str(root)):
            found.append(target)
    return found


def _check_in_flight(w, mp):
    gate = w.hold_checks()
    assert w.kvcr.query([A], "r") == [fetching_g4()]
    wait_for(lambda: w.blocked_checks() == 1)
    return gate.set


def _check_queued(w, mp):
    release = _check_in_flight(w, mp)
    assert w.kvcr.query([B], "r") == [fetching_g4()]
    return release


def _pending_start(w, mp):
    return _check_queued(w, mp)


def _spill_queued_unstarted(w, mp):
    w.spill(C, 0)
    assert w.await_check(C, "r") == fetchable_g4()
    w.writes_before = len(w.storage.log("WRITE"))
    # A read blocked in storage keeps the worker busy, so the next spill queues.
    w.storage.init_gate = threading.Event()
    w.deliver(C, 4, "r")
    wait_for(lambda: w.storage.init_waiting == 1)
    release = w.storage.init_gate.set
    w.deposit(B, 1, wait=False)
    w.settle(5)
    assert len(w.storage.log("WRITE")) == w.writes_before
    return release


def _registered_not_submitted(w, mp):
    w.spill(A, 0)
    assert w.await_check(A) == fetchable_g4()
    w.storage.init_gate = threading.Event()
    w.deliver(A, 4)
    wait_for(lambda: w.storage.init_waiting == 1)
    assert w.storage.live_dev
    return w.storage.init_gate.set


def _read_in_flight(w, mp):
    w.spill(A, 0)
    assert w.await_check(A) == fetchable_g4()
    w.storage.outcome["READ"] = "HOLD"
    w.deliver(A, 4)
    wait_for(lambda: len(w.storage.held("READ")) == 1)
    return w.storage.finish_all


def _spill_in_flight(w, mp):
    w.storage.outcome["WRITE"] = "HOLD"
    assert w.deposit(A, 0).success
    w.deposit(B, 1, wait=False)
    w.poll_until(lambda: len(w.storage.held("WRITE")) == 1)
    return w.storage.finish_all


def _read_error_released(w, mp):
    w.spill(A, 0)
    assert w.await_check(A) == fetchable_g4()
    w.storage.outcome["READ"] = "ERR"
    w.deliver(A, 4)
    wait_for(lambda: bool(w.storage.released) and not w.storage.unreleased())
    return None


def _spill_done_released(w, mp):
    assert w.deposit(A, 0).success
    w.deposit(B, 1, wait=False)
    wait_for(lambda: bool(w.storage.released) and not w.storage.unreleased())
    return None


def _file_worker_busy(w, mp):
    gate = threading.Event()
    me = threading.get_ident()
    waiting = []
    real = os.fsync

    def blocked_fsync(fd):
        if threading.get_ident() != me:
            waiting.append(fd)
            gate.wait(10)
        return real(fd)

    mp.setattr(os, "fsync", blocked_fsync)
    assert w.deposit(A, 0).success
    w.deposit(B, 1, wait=False)
    w.poll_until(lambda: bool(waiting))
    return gate.set


def _fill_waiting(w, mp):
    w.spill(A, 0)
    assert w.await_check(A) == fetchable_g4()
    w.storage.outcome["WRITE"] = "HOLD"
    w.fetch(A)
    w.poll_until(lambda: len(w.storage.held("WRITE")) == 1)
    return w.storage.finish_all


STAGES = [
    ("check_in_flight", "object", {}, _check_in_flight),
    ("check_in_flight", "file", {}, _check_in_flight),
    ("check_queued", "object", {}, _check_queued),
    ("check_queued", "file", {}, _check_queued),
    ("pending_start", "object", {"max_inflight_checks": 1}, _pending_start),
    ("pending_start", "file", {"max_inflight_checks": 1}, _pending_start),
    ("spill_queued_unstarted", "object", {}, _spill_queued_unstarted),
    ("registered_not_submitted", "object", {}, _registered_not_submitted),
    ("read_in_flight", "object", {}, _read_in_flight),
    ("spill_in_flight", "file", {}, _spill_in_flight),
    ("read_error_released", "object", {}, _read_error_released),
    ("spill_done_released", "object", {}, _spill_done_released),
    ("file_worker_busy", "file", {}, _file_worker_busy),
    ("fill_waiting", "object", {}, _fill_waiting),
]


@pytest.mark.parametrize(
    ("stage", "flavor", "options", "setup"),
    STAGES,
    ids=[f"{stage}-{flavor}" for stage, flavor, _, _ in STAGES],
)
def test_g4_close_at_each_stage_leaves_no_leaks(
    g4env, tmp_path, monkeypatch, stage, flavor, options, setup
):
    threads = threading.active_count()
    w = new_world(g4env, tmp_path, flavor=flavor, **options)
    release = setup(w, monkeypatch)
    timer = threading.Timer(0.2, release) if release is not None else None
    if timer is not None:
        timer.start()
    try:
        w.kvcr.close()
    finally:
        if timer is not None:
            timer.join()
    assert w.kvcr._core.is_quiescent()
    assert w.storage.leaks() == []
    assert w.local_dram._pools._capacity_eviction_key is None
    # Both G4 workers and the progress thread are joined.
    assert threading.active_count() == threads
    if stage in ("check_queued", "pending_start"):
        assert len(w.check_names()) == 1
    if stage == "spill_queued_unstarted":
        assert len(w.storage.log("WRITE")) == w.writes_before
    if flavor == "file":
        assert tmp_files(w.root) == []
        assert _open_fds_under(w.root) == []


def test_g4_close_raises_while_a_transfer_never_terminates(
    g4env, tmp_path, monkeypatch
):
    monkeypatch.setattr(progress, "_OP_CLEANUP_TIMEOUT_SECONDS", 0.2)
    w = new_world(g4env, tmp_path, flavor="object")
    _read_in_flight(w, monkeypatch)
    with pytest.raises(RuntimeError):
        w.kvcr.close()
    assert w.storage.unreleased()
    assert w.storage.live_dev
    w.storage.finish_all("DONE")
    _OPEN_KVCRS.remove(w.kvcr)


# ---- File worker thread exists only for the file flavor ----


def _file_workers():
    return [
        t for t in threading.enumerate() if t.name == "kvcr-g4-file" and t.is_alive()
    ]


def test_g4_object_flavor_runs_no_file_worker_thread(g4env, tmp_path):
    w = new_world(g4env, tmp_path, flavor="object")
    w.spill(A, 0)
    assert w.await_check(A) == fetchable_g4()
    assert _file_workers() == []
    w.kvcr.close()
    assert _file_workers() == []


def test_g4_file_flavor_runs_one_file_worker_thread_until_close(g4env, tmp_path):
    threads = threading.active_count()
    w = new_world(g4env, tmp_path, flavor="file")
    assert len(_file_workers()) == 1
    w.kvcr.close()
    assert _file_workers() == []
    assert threading.active_count() == threads


def _check_workers():
    return [
        t for t in threading.enumerate() if t.name == "kvcr-g4-check" and t.is_alive()
    ]


@pytest.mark.parametrize("flavor", ["object", "file"])
def test_g4_check_worker_thread_runs_for_each_flavor_until_close(
    g4env, tmp_path, flavor
):
    threads = threading.active_count()
    w = new_world(g4env, tmp_path, flavor=flavor)
    assert len(_check_workers()) == 1
    w.kvcr.close()
    assert _check_workers() == []
    assert threading.active_count() == threads


# ---- A worker that dies, or fails on one job ----

D = BlockKey(b"block-d")
MISS = (QueryStatus.MISS, None)


def _g4_warnings(caplog):
    return [
        record
        for record in caplog.records
        if record.levelno >= logging.WARNING and "g4" in record.getMessage().lower()
    ]


def _held_read(w):
    w.spill(A, 0)
    assert w.await_check(A, "r") == fetchable_g4()
    w.storage.outcome["READ"] = "HOLD"
    handle = w.deliver(A, 4, "r")
    wait_for(lambda: len(w.storage.held("READ")) == 1)
    return handle


def _held_fill(w):
    w.spill(A, 0)
    assert w.await_check(A, "r") == fetchable_g4()
    w.storage.outcome["READ"] = "HOLD"
    handle = w.fetch(A, "r")
    w.poll_until(lambda: len(w.storage.held("READ")) == 1)
    return handle


def _held_spill(w):
    w.storage.outcome["WRITE"] = "HOLD"
    assert w.deposit(A, 0).success
    waiting = w.deposit(B, 1, wait=False)
    w.poll_until(lambda: len(w.storage.held("WRITE")) == 1)
    return waiting


def _deposit_cannot_land(w, key, index):
    handle = w.deposit(key, index, wait=False)
    w.advance(5.0)
    return not w.wait(handle)[key].success


# A dead storage worker cannot release its posted handles, so the memory those
# handles may still touch stays pinned until close.


@pytest.mark.parametrize("flavor", ["object", "file"])
def test_g4_storage_worker_crash_fails_the_open_read_and_poll_keeps_returning(
    g4env, tmp_path, flavor
):
    w = new_world(g4env, tmp_path, flavor=flavor)
    try:
        handle = _held_read(w)
        w.storage.fault = crash_on("check_xfer_state")
        assert not w.wait(handle)[A].success
        for _ in range(20):
            w.poll()
        assert w.storage.unreleased()
        assert not w.kvcr._core.is_quiescent()
    finally:
        w.abandon()


@pytest.mark.parametrize("flavor", ["object", "file"])
def test_g4_storage_worker_crash_disables_g4_with_one_warning(
    g4env, tmp_path, caplog, flavor
):
    w = new_world(g4env, tmp_path, flavor=flavor)
    caplog.set_level(logging.WARNING)
    try:
        handle = _held_read(w)
        w.storage.fault = crash_on("check_xfer_state")
        assert not w.wait(handle)[A].success
        assert w.kvcr.query([D], "q") == [MISS]
        writes = len(w.storage.log("WRITE"))
        # Other tiers go on: the deposit lands and its G4 spill is dropped.
        assert w.deposit(B, 1).success
        assert w.deposit(C, 2).success
        assert len(w.storage.log("WRITE")) == writes
        warned = len(_g4_warnings(caplog))
        assert warned >= 1
        for _ in range(30):
            w.poll()
            assert w.kvcr.query([D], "q") == [MISS]
        assert len(_g4_warnings(caplog)) == warned
    finally:
        w.abandon()


def test_g4_storage_worker_crash_keeps_a_posted_fill_slot_pinned(g4env, tmp_path):
    w = new_world(g4env, tmp_path, flavor="object")
    try:
        handle = _held_fill(w)
        w.storage.fault = crash_on("check_xfer_state")
        assert not w.wait(handle)[A].success
        assert w.kvcr.query([A])[0][0] is not QueryStatus.HIT
        # The slot may still be written by NIXL, so nothing reuses it.
        assert _deposit_cannot_land(w, B, 1)
    finally:
        w.abandon()


@pytest.mark.parametrize("flavor", ["object", "file"])
def test_g4_storage_worker_crash_keeps_a_posted_spill_source_pinned(
    g4env, tmp_path, flavor
):
    w = new_world(g4env, tmp_path, flavor=flavor)
    try:
        waiting = _held_spill(w)
        w.storage.fault = crash_on("check_xfer_state")
        w.settle(50)
        w.advance(5.0)
        assert not w.wait(waiting)[B].success
        assert not w.stored(A)
        assert not w.kvcr._core.is_quiescent()
    finally:
        w.abandon()


def test_g4_storage_worker_crash_before_the_post_frees_the_fill_slot(g4env, tmp_path):
    w = new_world(g4env, tmp_path, flavor="object")
    try:
        w.spill(A, 0)
        assert w.await_check(A, "r") == fetchable_g4()
        posts = []

        def crash_at_the_read(name):
            if name == "initialize_xfer":
                posts.append(name)
                # The first call is the spill of the victim; the read is second.
                if len(posts) == 2:
                    raise WorkerCrash("read never posted")

        w.storage.fault = crash_at_the_read
        assert not w.wait(w.fetch(A, "r"))[A].success
        assert w.kvcr.query([A])[0][0] is not QueryStatus.HIT
        assert w.deposit(B, 1).success
    finally:
        w.abandon()


@pytest.mark.parametrize("flavor", ["object", "file"])
def test_g4_check_worker_crash_lets_an_in_flight_read_finish(
    g4env, tmp_path, caplog, flavor
):
    w = new_world(g4env, tmp_path, flavor=flavor)
    caplog.set_level(logging.WARNING)
    try:
        handle = _held_read(w)
        w.storage.fault = crash_on("query_memory")
        assert w.kvcr.query([D], "x") == [fetching_g4()]
        w.poll_until(lambda: w.kvcr.query([D], "x") == [MISS])
        warned = len(_g4_warnings(caplog))
        assert warned >= 1
        w.storage.finish_all("DONE")
        assert w.wait(handle)[A].success
        for _ in range(30):
            w.poll()
            assert w.kvcr.query([D], "q") == [MISS]
        assert len(_g4_warnings(caplog)) == warned
        assert w.deposit(B, 1).success
    finally:
        w.abandon()


@pytest.mark.parametrize("flavor", ["object", "file"])
def test_g4_check_worker_crash_lets_an_in_flight_fill_complete(g4env, tmp_path, flavor):
    w = new_world(g4env, tmp_path, flavor=flavor)
    try:
        handle = _held_fill(w)
        w.storage.fault = crash_on("query_memory")
        assert w.kvcr.query([D], "x") == [fetching_g4()]
        w.poll_until(lambda: w.kvcr.query([D], "x") == [MISS])
        w.storage.finish_all("DONE")
        assert w.wait(handle)[A].success
        assert w.kvcr.query([A])[0] == (QueryStatus.HIT, CacheTier.LOCAL_G2)
    finally:
        w.abandon()


def _fail_file_open_once(monkeypatch, match, error=RuntimeError):
    """Make the file worker's open of a matching path raise something unexpected."""
    real = os.open
    me = threading.get_ident()
    fired = []

    def flaky(path, *args, **kwargs):
        if threading.get_ident() != me and match(str(path)) and not fired:
            fired.append(path)
            raise error("unexpected file worker fault")
        return real(path, *args, **kwargs)

    monkeypatch.setattr(os, "open", flaky)
    return fired


def test_g4_file_worker_fault_on_a_read_ends_that_op_and_the_worker_goes_on(
    g4env, tmp_path, monkeypatch
):
    w = new_world(g4env, tmp_path, flavor="file")
    w.spill(A, 0)
    w.spill(B, 1)
    assert w.await_check(A, "ra") == fetchable_g4()
    assert w.await_check(B, "rb") == fetchable_g4()
    fired = _fail_file_open_once(monkeypatch, lambda path: path.endswith(digest(A)))
    assert not w.wait(w.deliver(A, 4, "ra"), timeout=1.0)[A].success
    assert fired
    assert w.wait(w.deliver(B, 5, "rb"), timeout=1.0)[B].success


def test_g4_file_worker_fault_on_a_spill_ends_that_op_and_the_worker_goes_on(
    g4env, tmp_path, monkeypatch
):
    w = new_world(g4env, tmp_path, flavor="file")
    fired = _fail_file_open_once(monkeypatch, lambda path: ".tmp." in path)
    assert w.deposit(A, 0).success
    assert w.wait(w.deposit(B, 1, wait=False), timeout=1.0)[B].success
    assert fired
    assert not w.stored(A)
    assert tmp_files(w.root) == []
    w.spill(C, 2)
    assert w.stored(C)


def test_g4_file_worker_death_before_a_read_opens_fails_it_and_disables_g4(
    g4env, tmp_path, monkeypatch
):
    w = new_world(g4env, tmp_path, flavor="file")
    try:
        w.spill(A, 0)
        assert w.await_check(A, "r") == fetchable_g4()
        _fail_file_open_once(
            monkeypatch, lambda path: path.endswith(digest(A)), WorkerCrash
        )
        assert not w.wait(w.deliver(A, 4, "r"), timeout=2.0)[A].success
        w.poll_until(lambda: w.kvcr.query([D], "q") == [MISS])
        assert w.deposit(B, 1).success
    finally:
        w.abandon()


def test_g4_file_worker_death_before_a_spill_opens_frees_the_source(
    g4env, tmp_path, monkeypatch
):
    w = new_world(g4env, tmp_path, flavor="file")
    try:
        _fail_file_open_once(monkeypatch, lambda path: ".tmp." in path, WorkerCrash)
        assert w.deposit(A, 0).success
        assert w.wait(w.deposit(B, 1, wait=False), timeout=2.0)[B].success
        assert not w.stored(A)
        assert w.kvcr.query([D], "q") == [MISS]
    finally:
        w.abandon()
