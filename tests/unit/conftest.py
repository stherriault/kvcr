# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import inspect
import logging

import pytest
from _kvcr_test_utils import _OPEN_KVCRS

from kvcr.service import memmap


def pytest_addoption(parser):
    parser.addoption(
        "--hugetlbfs-dir",
        help="directory directly under a hugetlbfs mount, for the real-hugetlbfs test",
    )


@pytest.fixture(autouse=True)
def _close_open_kvcrs():
    yield
    while _OPEN_KVCRS:
        _OPEN_KVCRS.pop().close()


@pytest.fixture(autouse=True)
def _no_hugetlb(monkeypatch):
    monkeypatch.setattr(memmap, "_hugetlbfs_mounts", lambda: [])


@pytest.fixture()
def kvcr_caplog(caplog):
    kvcr_logger = logging.getLogger("kvcr.runtime.core")
    kvcr_logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.WARNING, logger=kvcr_logger.name):
            yield caplog
    finally:
        kvcr_logger.removeHandler(caplog.handler)


@pytest.fixture()
def g4env(monkeypatch):
    """Route KVCR's NIXL agents to fakes; release held fake transfers at teardown."""
    from _g4_test_utils import FakeNixlEnv

    env = FakeNixlEnv(monkeypatch)
    yield env
    env.release_all()


_DROP2_WORDS = ("obj", "aws", "nixl_module", "head", "probe", "devid")
# Tests that need miss backoff (D5, not in drop 1): run with -m pending_d5.
_PENDING_D5 = (
    "miss_backoff_suppresses_repeat_checks",
    "own_publish_clears_the_miss_backoff",
    "max_check_results_miss_does_not_arm_the_miss_backoff",
)


def pytest_configure(config):
    config.addinivalue_line("markers", "drop1: G4 file flavor and flavor-neutral core")
    config.addinivalue_line("markers", "drop2: G4 OBJ flavor")
    config.addinivalue_line("markers", "pending_d5: needs miss backoff (D5)")


def _is_object_test(item) -> bool:
    flavor = getattr(getattr(item, "callspec", None), "params", {}).get("flavor")
    if flavor is not None:
        return flavor == "object"
    name = item.originalname or item.name
    if any(word in name for word in _DROP2_WORDS):
        return True
    # A test that builds an OBJ world or options by literal is an OBJ test.
    return '"object"' in inspect.getsource(item.function)


def pytest_collection_modifyitems(items):
    """Tag G4 tests: drop 1 is file flavor plus flavor-neutral core, drop 2 is OBJ."""
    for item in items:
        if not item.fspath.basename.startswith("test_g4_"):
            continue
        name = item.originalname or item.name
        if any(name.endswith(pending) for pending in _PENDING_D5):
            item.add_marker(pytest.mark.pending_d5)
            item.add_marker(pytest.mark.skip(reason="D5: no miss backoff in drop 1"))
            continue
        item.add_marker(
            pytest.mark.drop2 if _is_object_test(item) else pytest.mark.drop1
        )
