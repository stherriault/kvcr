# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""G4 breaker: visible transitions, failure counting and outages."""

import logging
import re

import pytest
from _g4_test_utils import (
    CHECK_ERROR_TEXT,
    fetchable_g4,
    find_files,
    new_world,
    wait_for,
)

from kvcr.types import BlockKey, QueryStatus

FLAVORS = ["file"]
A = BlockKey(b"block-a")
B = BlockKey(b"block-b")
C = BlockKey(b"block-c")
D = BlockKey(b"block-d")
MISS = (QueryStatus.MISS, None)
BACKOFF_MS = 10_000


def _fresh(index: int) -> BlockKey:
    return BlockKey(b"fresh-%d" % index)


def _transitions(caplog) -> list[str]:
    """Messages of the WARNINGs that name a breaker state, in order."""
    return [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING
        and record.name.startswith("kvcr")
        and re.search(r"open|closed|half", record.getMessage(), re.IGNORECASE)
    ]


def _names_open(message: str) -> bool:
    return bool(re.search(r"\bopen", message, re.I)) and "half" not in message.lower()


def _names_half_open(message: str) -> bool:
    return bool(re.search(r"half[ _-]?open", message, re.I))


def _names_closed(message: str) -> bool:
    return bool(re.search(r"\bclosed\b", message, re.I))


# ---- Transitions are visible ----


@pytest.mark.parametrize("flavor", FLAVORS)
def test_g4_breaker_logs_one_warning_for_each_open_half_open_and_closed(
    g4env, tmp_path, caplog, flavor
):
    w = new_world(
        g4env, tmp_path, flavor=flavor, failure_threshold=2, backoff_ms=BACKOFF_MS
    )
    w.spill(A, 0)
    assert w.await_check(A, "trial") == fetchable_g4()
    caplog.set_level(logging.WARNING)
    w.fail_checks()
    assert w.await_check(_fresh(1)) == MISS
    assert _transitions(caplog) == []
    assert w.await_check(_fresh(2)) == MISS
    [opened] = _transitions(caplog)
    assert _names_open(opened)
    assert CHECK_ERROR_TEXT[flavor] in opened
    w.heal_checks()
    w.advance(BACKOFF_MS / 1000 + 0.5)
    assert w.await_check(_fresh(3)) == MISS
    assert w.g4_state()["g4_breaker_state"] == 2
    assert [_names_half_open(m) for m in _transitions(caplog)[1:]] == [True]
    assert w.wait(w.deliver(A, 4, "trial"))[A].success
    messages = _transitions(caplog)
    assert len(messages) == 3
    assert _names_closed(messages[2])


@pytest.mark.parametrize("flavor", FLAVORS)
def test_g4_breaker_logs_a_warning_when_the_half_open_trial_fails(
    g4env, tmp_path, caplog, flavor
):
    w = new_world(
        g4env, tmp_path, flavor=flavor, failure_threshold=1, backoff_ms=BACKOFF_MS
    )
    caplog.set_level(logging.WARNING)
    w.fail_checks()
    assert w.await_check(_fresh(1)) == MISS
    w.advance(BACKOFF_MS / 1000 + 0.5)
    # Half-open admits a check; it fails again, so the breaker reopens.
    assert w.kvcr.query([_fresh(2)], "r")[0][0] is QueryStatus.FETCHING
    w.poll_until(lambda: len(_transitions(caplog)) >= 3)
    first, half, again = _transitions(caplog)
    assert _names_open(first) and _names_half_open(half) and _names_open(again)
    assert CHECK_ERROR_TEXT[flavor] in again


# ---- Gauges ----


@pytest.mark.parametrize("flavor", FLAVORS)
def test_g4_telemetry_state_reports_breaker_state_and_failures(g4env, tmp_path, flavor):
    w = new_world(
        g4env, tmp_path, flavor=flavor, failure_threshold=3, backoff_ms=BACKOFF_MS
    )
    w.spill(A, 0)
    assert w.await_check(A, "trial") == fetchable_g4()
    state = w.g4_state()
    assert (state["g4_breaker_state"], state["g4_breaker_failures"]) == (0, 0)
    w.fail_checks()
    for index in range(1, 3):
        assert w.await_check(_fresh(index)) == MISS
        state = w.g4_state()
        assert (state["g4_breaker_state"], state["g4_breaker_failures"]) == (0, index)
    assert w.await_check(_fresh(3)) == MISS
    state = w.g4_state()
    assert (state["g4_breaker_state"], state["g4_breaker_failures"]) == (1, 3)
    w.heal_checks()
    w.advance(BACKOFF_MS / 1000 + 0.5)
    assert w.g4_state()["g4_breaker_state"] == 2
    assert w.wait(w.deliver(A, 4, "trial"))[A].success
    state = w.g4_state()
    assert (state["g4_breaker_state"], state["g4_breaker_failures"]) == (0, 0)


# ---- Consecutive failures ----


@pytest.mark.parametrize("flavor", FLAVORS)
def test_g4_clean_checks_between_errors_keep_the_breaker_closed(
    g4env, tmp_path, flavor
):
    w = new_world(g4env, tmp_path, flavor=flavor)
    w.spill(A, 0)
    w.fail_checks()
    for index in range(4):
        assert w.await_check(_fresh(index)) == MISS
    w.heal_checks()
    assert w.await_check(A, "present") == fetchable_g4()
    for index in range(200):
        assert w.await_check(_fresh(100 + index)) == MISS
    w.fail_checks()
    assert w.await_check(_fresh(999)) == MISS
    w.heal_checks()
    assert not w.breaker_open()


@pytest.mark.parametrize("flavor", FLAVORS)
def test_g4_five_check_errors_in_a_row_open_the_breaker(g4env, tmp_path, flavor):
    w = new_world(g4env, tmp_path, flavor=flavor)
    w.fail_checks()
    for index in range(4):
        assert w.await_check(_fresh(index)) == MISS
    w.heal_checks()
    assert not w.breaker_open()
    w.fail_checks()
    assert w.await_check(_fresh(4)) == MISS
    w.heal_checks()
    assert w.breaker_open()


def _remove(w, key) -> None:
    if w.flavor == "object":
        del w.store.objects[w.store.find(key, w.prefix)]
    else:
        find_files(w.root, key, w.prefix)[0].unlink()


def _read_error_on_a_vanished_block(w, key, rid) -> None:
    """Fail a held read with ERR after the block is gone; wait for its probe."""
    probes = len(w.check_names())
    w.storage.outcome["READ"] = "HOLD"
    handle = w.deliver(key, 4, rid)
    wait_for(lambda: len(w.storage.held("READ")) == 1)
    _remove(w, key)
    w.storage.finish_all("ERR")
    assert not w.wait(handle)[key].success
    wait_for(lambda: len(w.check_names()) > probes)
    w.settle(50)


@pytest.mark.parametrize("flavor", FLAVORS)
def test_g4_probe_that_finds_the_block_gone_resets_the_check_streak(
    g4env, tmp_path, flavor
):
    w = new_world(g4env, tmp_path, flavor=flavor)
    w.spill(B, 1)
    assert w.await_check(B, "b") == fetchable_g4()
    w.fail_checks()
    for index in range(4):
        assert w.await_check(_fresh(index)) == MISS
    w.heal_checks()
    # The probe answers "absent": storage is reachable, so the streak restarts.
    _read_error_on_a_vanished_block(w, B, "b")
    w.fail_checks()
    assert w.await_check(_fresh(99)) == MISS
    w.heal_checks()
    assert not w.breaker_open()


@pytest.mark.parametrize("flavor", FLAVORS)
def test_g4_failed_reads_open_the_breaker_although_every_check_answers(
    g4env, tmp_path, flavor
):
    w = new_world(g4env, tmp_path, flavor=flavor, failure_threshold=3)
    w.spill(A, 0)
    w.storage.outcome["READ"] = "ERR"
    for index in range(3):
        assert not w.breaker_open()
        assert w.await_check(A, f"r{index}") == fetchable_g4()
        assert not w.wait(w.deliver(A, 4, f"r{index}"))[A].success
        w.settle(30)
    assert w.breaker_open()


@pytest.mark.parametrize("flavor", FLAVORS)
def test_g4_failures_gauge_reports_the_larger_streak(g4env, tmp_path, flavor):
    w = new_world(g4env, tmp_path, flavor=flavor)
    w.spill(A, 0)
    for rid in ("r0", "r1"):
        assert w.await_check(A, rid) == fetchable_g4()
    w.fail_checks()
    assert w.await_check(_fresh(1)) == MISS
    w.heal_checks()
    assert w.g4_state()["g4_breaker_failures"] == 1
    w.storage.outcome["READ"] = "ERR"
    for rid in ("r0", "r1"):
        assert not w.wait(w.deliver(A, 4, rid))[A].success
    w.settle(30)
    assert w.g4_state()["g4_breaker_failures"] == 2


# ---- A spill that passes its deadline ----


def _capacity_key(w):
    return w.local_dram._pools._capacity_eviction_key


def _hung_spill(w, victim=A, waiting=B):
    """Start a spill of `victim` that never ends; `waiting` queues behind it."""
    w.storage.outcome["WRITE"] = "HOLD"
    assert w.deposit(victim, 0).success
    handle = w.deposit(waiting, 1, wait=False)
    w.poll_until(lambda: len(w.storage.held("WRITE")) == 1)
    return handle


@pytest.mark.parametrize("flavor", FLAVORS)
def test_g4_spill_past_its_deadline_counts_toward_the_breaker(g4env, tmp_path, flavor):
    w = new_world(g4env, tmp_path, flavor=flavor, failure_threshold=1)
    _hung_spill(w)
    assert not w.breaker_open()
    w.advance(2.0)
    w.poll_until(lambda: w.breaker_open())


@pytest.mark.parametrize("flavor", FLAVORS)
def test_g4_spill_past_its_deadline_frees_g2_admission(g4env, tmp_path, flavor):
    w = new_world(g4env, tmp_path, flavor=flavor, slots=2)
    w.storage.outcome["WRITE"] = "HOLD"
    for index, key in enumerate((A, B)):
        assert w.deposit(key, index).success
    w.deposit(C, 2, wait=False)
    w.poll_until(lambda: len(w.storage.held("WRITE")) == 1)
    assert _capacity_key(w) == A
    w.advance(2.0)
    w.poll_until(lambda: _capacity_key(w) is None)
    # The next eviction can pick another victim while the first write hangs.
    w.deposit(D, 3, wait=False)
    w.poll_until(lambda: len(w.storage.held("WRITE")) == 2)


@pytest.mark.parametrize("flavor", FLAVORS)
def test_g4_late_done_after_the_latch_publishes_without_counting_again(
    g4env, tmp_path, flavor
):
    w = new_world(g4env, tmp_path, flavor=flavor, failure_threshold=2)
    waiting = _hung_spill(w)
    w.advance(2.0)
    assert not w.wait(waiting)[B].success
    w.storage.finish_all("DONE")
    w.poll_until(lambda: w.g4_state()["g4_spill_late"] == 1)
    w.poll_until(lambda: w.stored(A) and _capacity_key(w) is None)
    assert not w.breaker_open()
    # The latch counted one data failure and the late DONE changed nothing: a
    # failed read is the second in a row.
    assert w.await_check(A, "r") == fetchable_g4()
    w.storage.outcome["READ"] = "ERR"
    assert not w.wait(w.deliver(A, 4, "r"))[A].success
    w.poll_until(lambda: w.breaker_open())


@pytest.mark.parametrize("flavor", FLAVORS)
def test_g4_spill_past_its_deadline_releases_the_half_open_trial(
    g4env, tmp_path, flavor
):
    w = new_world(
        g4env, tmp_path, flavor=flavor, failure_threshold=1, backoff_ms=BACKOFF_MS
    )
    w.spill(A, 0)
    assert w.await_check(A, "first") == fetchable_g4()
    assert w.await_check(A, "second") == fetchable_g4()
    w.storage.outcome["READ"] = "ERR"
    assert not w.wait(w.deliver(A, 4, "first"))[A].success
    w.poll_until(lambda: w.breaker_open())
    w.storage.outcome["READ"] = "DONE"
    w.storage.outcome["WRITE"] = "HOLD"
    w.advance(BACKOFF_MS / 1000 + 0.5)
    # The next eviction's spill takes the half-open trial and never ends.
    w.deposit(B, 1, wait=False)
    w.poll_until(lambda: len(w.storage.held("WRITE")) == 1)
    w.advance(BACKOFF_MS / 1000 + 1.6)
    w.settle(30)
    w.advance(2 * BACKOFF_MS / 1000 + 2.0)
    # Reads are admitted again instead of waiting for the hung write.
    assert w.wait(w.deliver(A, 5, "second"))[A].success


# ---- Outages ----


def _read_in_flight(w, key=A, rid="r"):
    """Spill `key`, check it, and start a read that stays in storage."""
    w.spill(key, 0)
    assert w.await_check(key, rid) == fetchable_g4()
    w.storage.outcome["READ"] = "HOLD"
    handle = w.deliver(key, 4, rid)
    wait_for(lambda: len(w.storage.held("READ")) == 1)
    return handle


@pytest.mark.parametrize("flavor", FLAVORS)
@pytest.mark.parametrize("phase", ["open", "half_open"])
def test_g4_no_read_error_probe_starts_while_the_breaker_is_not_closed(
    g4env, tmp_path, flavor, phase
):
    w = new_world(g4env, tmp_path, flavor=flavor, failure_threshold=1, backoff_ms=500)
    handle = _read_in_flight(w)
    w.fail_checks()
    assert w.await_check(_fresh(1)) == MISS
    if phase == "half_open":
        w.advance(0.7)
    checks = list(w.check_names())
    w.storage.finish_all("ERR")
    assert not w.wait(handle)[A].success
    w.settle(50)
    assert w.check_names() == checks


@pytest.mark.parametrize("flavor", FLAVORS)
def test_g4_failure_of_an_op_admitted_before_the_outage_does_not_extend_it(
    g4env, tmp_path, flavor
):
    w = new_world(
        g4env, tmp_path, flavor=flavor, failure_threshold=1, backoff_ms=BACKOFF_MS
    )
    handle = _read_in_flight(w)
    w.fail_checks()
    assert w.await_check(_fresh(1)) == MISS
    w.heal_checks()
    assert w.breaker_open()
    # The read passes its deadline and then fails, both while the breaker is open.
    w.advance(5.0)
    w.settle(30)
    w.storage.finish_all("ERR")
    assert not w.wait(handle)[A].success
    w.settle(50)
    w.advance(BACKOFF_MS / 1000 + 0.5)
    assert not w.breaker_open()
