# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""G4 telemetry: transfers and existence checks show up like G3's."""

import errno
import os
import threading

import pytest
from _g4_test_utils import (
    PAGE,
    fetchable_g4,
    fetching_g4,
    new_world,
    wait_for,
)

from kvcr.runtime.core import (
    DURATION_METRIC,
    TRANSFER_BLOCKS_METRIC,
    TRANSFER_BYTES_METRIC,
)
from kvcr.types import BlockKey, QueryStatus

FLAVORS = ["file"]
A = BlockKey(b"block-a")
B = BlockKey(b"block-b")
MISS = (QueryStatus.MISS, None)

OP_COUNTERS = (
    "g4_fill_ok",
    "g4_fill_failed",
    "g4_deliver_ok",
    "g4_deliver_failed",
    "g4_spill_published",
)
CHECK_COUNTERS = ("g4_check_present", "g4_check_absent", "g4_check_error")


def _counters(w, names):
    state = w.g4_state()
    return {name: state.get(name) for name in names}


def _ops(**expected):
    return {name: expected.get(name, 0) for name in OP_COUNTERS}


def _checks(**expected):
    return {name: expected.get(name, 0) for name in CHECK_COUNTERS}


def _transfer(metrics, scope):
    return (
        metrics.get(("counter", TRANSFER_BLOCKS_METRIC, scope)),
        metrics.get(("counter", TRANSFER_BYTES_METRIC, scope)),
    )


def _durations(metrics, scope):
    return {
        result: metrics[key]
        for key in metrics
        if key[:3] == ("histogram", DURATION_METRIC, scope)
        for result in [key[3]]
    }


def _present(w, key=A):
    assert w.await_check(key) == fetchable_g4()


# ---- Spill ----


@pytest.mark.parametrize("flavor", FLAVORS)
def test_g4_published_spill_is_recorded_as_a_successful_transfer(
    g4env, tmp_path, flavor
):
    w = new_world(g4env, tmp_path, flavor=flavor, telemetry=True)
    w.spill(A, 0)
    metrics = w.metrics()
    assert _transfer(metrics, "g4_spill") == (1, PAGE)
    assert list(_durations(metrics, "g4_spill")) == ["success"]
    assert _counters(w, OP_COUNTERS) == _ops(g4_spill_published=1)


@pytest.mark.parametrize("flavor", FLAVORS)
def test_g4_failed_spill_is_recorded_as_failed_without_payload_counters(
    g4env, tmp_path, flavor
):
    w = new_world(g4env, tmp_path, flavor=flavor, telemetry=True)
    w.storage.outcome["WRITE"] = "ERR"
    assert w.deposit(A, 0).success
    assert w.deposit(B, 1).success
    metrics = w.metrics()
    assert list(_durations(metrics, "g4_spill")) == ["failed"]
    assert _transfer(metrics, "g4_spill") == (None, None)
    assert _counters(w, OP_COUNTERS) == _ops()


@pytest.mark.parametrize("flavor", FLAVORS)
def test_g4_spill_counts_the_payload_bytes_of_a_multi_page_block(
    g4env, tmp_path, flavor
):
    w = new_world(g4env, tmp_path, flavor=flavor, pages=2, telemetry=True)
    w.spill(A, 0)
    assert _transfer(w.metrics(), "g4_spill") == (1, 2 * PAGE)


def test_g4_file_spill_failing_at_publish_is_recorded_as_failed(
    g4env, tmp_path, monkeypatch
):
    w = new_world(g4env, tmp_path, flavor="file", failure_threshold=5, telemetry=True)
    me = threading.get_ident()
    real = os.fsync

    def failing(*args, **kwargs):
        if threading.get_ident() != me:
            raise OSError(errno.EIO, "publish failed")
        return real(*args, **kwargs)

    monkeypatch.setattr(os, "fsync", failing)
    assert w.deposit(A, 0).success
    assert w.deposit(B, 1).success
    assert not w.stored(A)
    metrics = w.metrics()
    assert list(_durations(metrics, "g4_spill")) == ["failed"]
    assert _transfer(metrics, "g4_spill") == (None, None)
    assert _counters(w, OP_COUNTERS) == _ops()


# ---- Fill (fetch into a G2 slot) ----


@pytest.mark.parametrize("flavor", FLAVORS)
def test_g4_fetch_is_recorded_as_a_successful_fill(g4env, tmp_path, flavor):
    w = new_world(g4env, tmp_path, flavor=flavor, telemetry=True)
    w.spill(A, 0)
    _present(w)
    assert w.wait(w.fetch(A))[A].success
    metrics = w.metrics()
    assert _transfer(metrics, "g4_fill") == (1, PAGE)
    assert list(_durations(metrics, "g4_fill")) == ["success"]
    assert _counters(w, OP_COUNTERS) == _ops(g4_spill_published=2, g4_fill_ok=1)


@pytest.mark.parametrize("flavor", FLAVORS)
def test_g4_failed_fetch_is_recorded_as_a_failed_fill(g4env, tmp_path, flavor):
    w = new_world(g4env, tmp_path, flavor=flavor, telemetry=True)
    w.spill(A, 0)
    _present(w)
    w.storage.outcome["READ"] = "ERR"
    assert not w.wait(w.fetch(A))[A].success
    metrics = w.metrics()
    assert list(_durations(metrics, "g4_fill")) == ["failed"]
    assert _transfer(metrics, "g4_fill") == (None, None)
    counters = _counters(w, OP_COUNTERS)
    assert (counters["g4_fill_ok"], counters["g4_fill_failed"]) == (0, 1)


@pytest.mark.parametrize("flavor", FLAVORS)
def test_g4_fill_counts_the_payload_bytes_of_a_multi_page_block(
    g4env, tmp_path, flavor
):
    w = new_world(g4env, tmp_path, flavor=flavor, pages=2, telemetry=True)
    w.spill(A, 0)
    _present(w)
    assert w.wait(w.fetch(A))[A].success
    assert _transfer(w.metrics(), "g4_fill") == (1, 2 * PAGE)


# ---- Deliver (read into a framework destination) ----


@pytest.mark.parametrize("flavor", FLAVORS)
def test_g4_deliver_is_recorded_as_a_successful_deliver(g4env, tmp_path, flavor):
    w = new_world(g4env, tmp_path, flavor=flavor, telemetry=True)
    w.spill(A, 0)
    _present(w)
    assert w.wait(w.deliver(A, 4))[A].success
    metrics = w.metrics()
    assert _transfer(metrics, "g4_deliver") == (1, PAGE)
    assert list(_durations(metrics, "g4_deliver")) == ["success"]
    assert _transfer(metrics, "g4_fill") == (None, None)
    assert _counters(w, OP_COUNTERS) == _ops(g4_spill_published=1, g4_deliver_ok=1)


@pytest.mark.parametrize("flavor", FLAVORS)
def test_g4_failed_deliver_is_recorded_as_a_failed_deliver(g4env, tmp_path, flavor):
    w = new_world(g4env, tmp_path, flavor=flavor, telemetry=True)
    w.spill(A, 0)
    _present(w)
    w.storage.outcome["READ"] = "ERR"
    assert not w.wait(w.deliver(A, 4))[A].success
    metrics = w.metrics()
    assert list(_durations(metrics, "g4_deliver")) == ["failed"]
    assert _transfer(metrics, "g4_deliver") == (None, None)
    assert _counters(w, OP_COUNTERS) == _ops(g4_spill_published=1, g4_deliver_failed=1)


@pytest.mark.parametrize("flavor", FLAVORS)
def test_g4_deliver_counts_the_payload_bytes_of_a_multi_page_block(
    g4env, tmp_path, flavor
):
    w = new_world(g4env, tmp_path, flavor=flavor, pages=2, telemetry=True)
    w.spill(A, 0)
    _present(w)
    assert w.wait(w.deliver(A, 4))[A].success
    assert _transfer(w.metrics(), "g4_deliver") == (1, 2 * PAGE)


# ---- Existence checks ----


@pytest.mark.parametrize("flavor", FLAVORS)
def test_g4_check_of_a_present_block_counts_as_present(g4env, tmp_path, flavor):
    w = new_world(g4env, tmp_path, flavor=flavor, telemetry=True)
    w.spill(A, 0)
    _present(w)
    assert _counters(w, CHECK_COUNTERS) == _checks(g4_check_present=1)


@pytest.mark.parametrize("flavor", FLAVORS)
def test_g4_check_of_an_absent_block_counts_as_absent(g4env, tmp_path, flavor):
    w = new_world(g4env, tmp_path, flavor=flavor, telemetry=True)
    assert w.await_check(A) == MISS
    assert _counters(w, CHECK_COUNTERS) == _checks(g4_check_absent=1)


@pytest.mark.parametrize("flavor", FLAVORS)
def test_g4_check_that_fails_counts_as_error_and_not_absent(g4env, tmp_path, flavor):
    w = new_world(g4env, tmp_path, flavor=flavor, telemetry=True)
    w.fail_checks()
    assert w.await_check(A) == MISS
    assert _counters(w, CHECK_COUNTERS) == _checks(g4_check_error=1)


@pytest.mark.parametrize("flavor", FLAVORS)
def test_g4_check_that_times_out_counts_as_error(g4env, tmp_path, flavor):
    w = new_world(g4env, tmp_path, flavor=flavor, check_timeout_ms=500, telemetry=True)
    w.hold_checks()
    assert w.kvcr.query([A], "r") == [fetching_g4()]
    wait_for(lambda: w.blocked_checks() == 1)
    w.settle(2)
    w.advance(0.6)
    assert w.kvcr.query([A], "r") == [MISS]
    assert _counters(w, CHECK_COUNTERS) == _checks(g4_check_error=1)


@pytest.mark.parametrize("flavor", FLAVORS)
def test_g4_checks_accumulate_across_requests(g4env, tmp_path, flavor):
    w = new_world(g4env, tmp_path, flavor=flavor, telemetry=True)
    w.spill(A, 0)
    _present(w)
    assert w.await_check(B, "r2") == MISS
    assert w.await_check(BlockKey(b"block-c"), "r3") == MISS
    assert _counters(w, CHECK_COUNTERS) == _checks(
        g4_check_present=1, g4_check_absent=2
    )


# ---- Telemetry off ----


@pytest.mark.parametrize("flavor", FLAVORS)
def test_g4_with_telemetry_off_records_nothing_and_still_works(g4env, tmp_path, flavor):
    w = new_world(g4env, tmp_path, flavor=flavor)
    w.spill(A, 0)
    _present(w)
    assert w.wait(w.deliver(A, 4))[A].success
    assert w.kvcr.get_stats() is None
    assert w.metrics() == {}
