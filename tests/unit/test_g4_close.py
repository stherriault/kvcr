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

from kvcr.types import BlockKey, QueryStatus

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


def _spill_in_flight(w, mp):
    w.storage.outcome["WRITE"] = "HOLD"
    assert w.deposit(A, 0).success
    w.deposit(B, 1, wait=False)
    w.poll_until(lambda: len(w.storage.held("WRITE")) == 1)
    return w.storage.finish_all


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


STAGES = [
    ("check_in_flight", "file", {}, _check_in_flight),
    ("check_queued", "file", {}, _check_queued),
    ("pending_start", "file", {"max_inflight_checks": 1}, _pending_start),
    ("spill_in_flight", "file", {}, _spill_in_flight),
    ("file_worker_busy", "file", {}, _file_worker_busy),
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


# ---- File worker thread exists only for the file flavor ----


def _file_workers():
    return [
        t for t in threading.enumerate() if t.name == "kvcr-g4-file" and t.is_alive()
    ]


def test_g4_file_flavor_runs_one_file_worker_thread_until_close(g4env, tmp_path):
    threads = threading.active_count()
    w = new_world(g4env, tmp_path, flavor="file")
    assert len(_file_workers()) == 1
    w.kvcr.close()
    assert _file_workers() == []
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


def _held_spill(w):
    w.storage.outcome["WRITE"] = "HOLD"
    assert w.deposit(A, 0).success
    waiting = w.deposit(B, 1, wait=False)
    w.poll_until(lambda: len(w.storage.held("WRITE")) == 1)
    return waiting


# A dead storage worker cannot release its posted handles, so the memory those
# handles may still touch stays pinned until close.


@pytest.mark.parametrize("flavor", ["file"])
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


@pytest.mark.parametrize("flavor", ["file"])
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


@pytest.mark.parametrize("flavor", ["file"])
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
