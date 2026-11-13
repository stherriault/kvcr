# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""close() at every G4 operation stage leaves nothing behind."""

import os
import threading

import pytest
from _g4_test_utils import (
    fetchable_g4,
    fetching_g4,
    new_world,
    tmp_files,
    wait_for,
)
from _kvcr_test_utils import _OPEN_KVCRS

from kvcr.runtime import progress
from kvcr.types import BlockKey

A = BlockKey(b"block-a")
B = BlockKey(b"block-b")


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
    w.storage.query_gate = threading.Event()
    assert w.kvcr.query([A], "r") == [fetching_g4()]
    wait_for(lambda: w.storage.query_blocked == 1)
    return w.storage.query_gate.set


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


STAGES = {
    "check_in_flight": ("object", _check_in_flight),
    "registered_not_submitted": ("object", _registered_not_submitted),
    "read_in_flight": ("object", _read_in_flight),
    "spill_in_flight": ("file", _spill_in_flight),
    "read_error_released": ("object", _read_error_released),
    "spill_done_released": ("object", _spill_done_released),
    "file_worker_busy": ("file", _file_worker_busy),
    "fill_waiting": ("object", _fill_waiting),
}


@pytest.mark.parametrize("stage", STAGES)
def test_g4_close_at_each_stage_leaves_no_leaks(g4env, tmp_path, monkeypatch, stage):
    flavor, setup = STAGES[stage]
    w = new_world(g4env, tmp_path, flavor=flavor)
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
