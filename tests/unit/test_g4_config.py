# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""G4 configuration, startup rejections and environment handling."""

import ctypes
from pathlib import Path

import pytest
from _g4_test_utils import (
    FakeG4Agent,
    FakeMainAgent,
    build_kvcr,
    g4_options,
    new_hybrid_world,
    new_world,
)

from kvcr.config import KVCRConfig, KVCRGuardConfig, LocalDramOptions
from kvcr.types import CacheTier, RegionDescriptor

DOCS = Path(__file__).resolve().parents[2] / "docs" / "g4.md"


def test_cache_tier_g4_exists():
    assert CacheTier.G4.value == "G4"


# ---- Startup rejections (AC1) ----


def test_g4_object_key_layout_with_two_labels_is_rejected(g4env, tmp_path):
    with pytest.raises(ValueError):
        new_hybrid_world(g4env, tmp_path, flavor="object", key_layout=("full", "swa:a"))


def test_g4_object_flavor_rejects_a_root(g4env, tmp_path):
    with pytest.raises(ValueError):
        new_world(g4env, tmp_path, flavor="object", root=tmp_path)


def test_g4_file_flavor_requires_a_root(g4env, tmp_path):
    with pytest.raises(ValueError):
        new_world(g4env, tmp_path, flavor="file", options=g4_options("file", None))


def test_g4_file_key_layout_must_resolve_against_the_pools(g4env, tmp_path):
    with pytest.raises(ValueError):
        new_world(g4env, tmp_path, flavor="file", key_layout=("missing",))


def test_g4_file_root_must_be_a_directory(g4env, tmp_path):
    not_a_directory = tmp_path / "file"
    not_a_directory.write_text("x")
    with pytest.raises((ValueError, OSError, RuntimeError)):
        new_world(g4env, tmp_path, flavor="file", root=not_a_directory)


def test_g4_file_startup_probe_leaves_no_files_behind(g4env, tmp_path):
    w = new_world(g4env, tmp_path, flavor="file")
    assert [p for p in w.root.rglob("*") if p.is_file()] == []


def test_g4_requires_local_dram(g4env, tmp_path):
    with pytest.raises(ValueError):
        build_kvcr(
            g4env,
            main=FakeMainAgent(),
            storage=FakeG4Agent(),
            config=KVCRConfig(nixl_agent_name="target", pool_layouts=[("", 4096)]),
            framework_regions=[RegionDescriptor(addr=4096, size=4096, count=2)],
            local_dram=None,
            g4=g4_options("object", None),
        )


def test_g4_with_guard_recovery_is_rejected(g4env, tmp_path):
    guard = KVCRGuardConfig(
        kvcr_service_socket_path=str(tmp_path / "missing.sock"),
        guard_index=0,
        compatibility_digest="digest",
    )
    local = ctypes.create_string_buffer(4096)
    with pytest.raises(ValueError, match="(?i)guard"):
        build_kvcr(
            g4env,
            main=FakeMainAgent(),
            storage=FakeG4Agent(),
            config=KVCRConfig(nixl_agent_name="target", pool_layouts=[("", 4096)]),
            framework_regions=[RegionDescriptor(addr=4096, size=4096, count=2)],
            local_dram=LocalDramOptions([("", ctypes.addressof(local), 4096)]),
            g4=g4_options("object", None),
            guard_config=guard,
        )


# ---- Backend options (AC1) ----


def test_g4_file_flavor_forces_posix_aio(g4env, tmp_path):
    w = new_world(g4env, tmp_path, flavor="file")
    assert w.storage.backends["POSIX"]["use_aio"] == "true"


@pytest.mark.parametrize("option", ["use_uring", "use_posix_aio"])
def test_g4_file_flavor_rejects_uring_and_posix_aio(g4env, tmp_path, option):
    with pytest.raises(ValueError):
        new_world(
            g4env,
            tmp_path,
            flavor="file",
            backend_options={option: "true"},
        )


def test_g4_object_flavor_creates_the_obj_backend_with_its_options(g4env, tmp_path):
    w = new_world(g4env, tmp_path, flavor="object")
    options = w.storage.backends["OBJ"]
    assert options["bucket"] == "unit"
    assert options["endpoint_override"] == "127.0.0.1:1"
    assert options["scheme"] == "http"


def test_g4_options_repr_redacts_credentials():
    options = g4_options("object", None)
    assert "unit-secret" not in repr(options)
    assert "unit-access" not in repr(options)


# ---- Environment (AC9) ----


@pytest.mark.parametrize(
    ("credentials", "preset", "expected"),
    [
        (True, None, "true"),
        (True, "false", "false"),
        (False, None, None),
    ],
    ids=["static-credentials", "user-value-kept", "no-credentials"],
)
def test_g4_object_backend_disables_the_aws_metadata_probe_only_when_safe(
    g4env, tmp_path, monkeypatch, credentials, preset, expected
):
    if preset is not None:
        monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", preset)
    options = {"bucket": "unit", "endpoint_override": "127.0.0.1:1", "scheme": "http"}
    if credentials:
        options.update(access_key="a", secret_key="s")
    w = new_world(g4env, tmp_path, flavor="object", backend_options=options)
    assert w.storage.env_at_create["OBJ"] == expected


def test_g4_storage_agent_comes_from_the_kvcr_nixl_module(g4env, tmp_path, monkeypatch):
    options = g4_options("object", None)
    monkeypatch.setenv("KVCR_NIXL_MODULE", "kvcr_unit_no_such_nixl_module")
    with pytest.raises(Exception, match="kvcr_unit_no_such_nixl_module"):
        new_world(g4env, tmp_path, flavor="object", options=options)


def test_g4_backend_creation_failure_names_the_nixl_module_knob(
    g4env, tmp_path, monkeypatch
):
    def failing(self, backend, params=None):
        raise RuntimeError("plugin OBJ failed to load")

    options = g4_options("object", None)
    monkeypatch.setattr(FakeG4Agent, "create_backend", failing)
    with pytest.raises(Exception, match="KVCR_NIXL_MODULE"):
        new_world(g4env, tmp_path, flavor="object", options=options)


# ---- Documentation (AC10) ----


def test_g4_docs_cover_the_required_topics():
    text = DOCS.read_text().lower()
    topics = [
        "eviction model",
        "key_layout",
        "stall",
        "one label",
        "short read",
        "timeout",
        "uncertain",
        "quiesced",
        "lifecycle",
        "use_aio",
        "kvcr_nixl_module",
        "aws_ec2_metadata_disabled",
    ]
    missing = [topic for topic in topics if topic not in text]
    assert missing == []


def test_g4_options_defaults():
    from kvcr.config import G4Options

    options = G4Options(flavor="object", key_prefix="p", key_layout=("",))
    assert options.max_inflight_checks == 64
    assert options.max_check_results == 65_536
    assert options.check_timeout_ms is None
    assert options.check_batch_max == 32
    assert options.spill_timeout_ms is None
    assert options.miss_backoff_ms == 1_000
    assert options.direct_io is False
    assert options.failure_threshold == 5
    assert options.backoff_ms == 10_000
