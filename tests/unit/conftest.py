# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

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


_DROP2_WORDS = ("obj", "object", "aws", "nixl_module", "head", "probe", "devid")


def pytest_configure(config):
    config.addinivalue_line("markers", "drop1: G4 file flavor delivery drop")
    config.addinivalue_line("markers", "drop2: G4 OBJ flavor delivery drop")


def pytest_collection_modifyitems(items):
    """Tag G4 tests by delivery drop: file flavor is drop 1, OBJ is drop 2."""
    for item in items:
        if not item.fspath.basename.startswith("test_g4_"):
            continue
        flavor = getattr(getattr(item, "callspec", None), "params", {}).get("flavor")
        if flavor is None:
            name = item.originalname or item.name
            flavor = "object" if any(w in name for w in _DROP2_WORDS) else "file"
        item.add_marker(pytest.mark.drop2 if flavor == "object" else pytest.mark.drop1)
