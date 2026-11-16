# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""G4 configuration, startup rejections and environment handling."""

import ctypes
import errno
import os

import pytest
from _g4_test_utils import (
    PAGE,
    FakeG4Agent,
    FakeMainAgent,
    aligned,
    build_kvcr,
    g4_options,
    new_world,
)

from kvcr.config import KVCRConfig, LocalDramOptions
from kvcr.types import BlockKey, CacheTier, RegionDescriptor


def test_cache_tier_g4_exists():
    assert CacheTier.G4.value == "G4"


# ---- Startup rejections ----


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


# ---- Backend options ----


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


# ---- O_DIRECT default and startup check ----


def _refuse_o_direct(monkeypatch):
    real = os.open

    def refusing(path, flags, *args, **kwargs):
        if flags & os.O_DIRECT:
            raise OSError(errno.EINVAL, "Invalid argument", os.fspath(path))
        return real(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", refusing)


def test_g4_file_root_refusing_o_direct_fails_startup_by_default(
    g4env, tmp_path, monkeypatch
):
    from kvcr.config import G4Options

    root = tmp_path / "g4root"
    root.mkdir()
    _refuse_o_direct(monkeypatch)
    options = G4Options(flavor="file", key_prefix="p", key_layout=("",), root=root)
    with pytest.raises(Exception) as caught:
        new_world(g4env, tmp_path, flavor="file", root=root, options=options)
    message = str(caught.value)
    assert "direct_io" in message or "O_DIRECT" in message
    assert str(root) in message


def test_g4_file_root_refusing_o_direct_fails_startup_with_direct_io_true(
    g4env, tmp_path, monkeypatch
):
    root = tmp_path / "g4root"
    root.mkdir()
    _refuse_o_direct(monkeypatch)
    with pytest.raises(Exception) as caught:
        new_world(g4env, tmp_path, flavor="file", root=root, direct_io=True)
    message = str(caught.value)
    assert "direct_io" in message or "O_DIRECT" in message
    assert str(root) in message


def test_g4_file_root_refusing_o_direct_starts_with_direct_io_false(
    g4env, tmp_path, monkeypatch
):
    root = tmp_path / "g4root"
    root.mkdir()
    _refuse_o_direct(monkeypatch)
    w = new_world(g4env, tmp_path, flavor="file", root=root, direct_io=False)
    w.spill(BlockKey(b"a"), 0)
    assert w.stored(BlockKey(b"a"))


# ---- Page alignment of DRAM regions with direct_io (file flavor) ----


def _start_with_regions(
    g4env,
    tmp_path,
    *,
    flavor="file",
    direct_io=True,
    framework_offset=0,
    block=PAGE,
    stride=0,
    count=2,
    local_offset=0,
):
    """Start G4 over one framework region and one local DRAM region.

    Offsets and sizes are chosen by the caller so each test misaligns exactly
    one thing; the buffers themselves are page aligned mappings.
    """
    extent = max(block, stride) * count
    framework = aligned(extent + 2 * PAGE)
    local = aligned(2 * PAGE + 2 * block)
    local_size = max(PAGE, 2 * block)
    root = tmp_path / "g4root"
    root.mkdir(exist_ok=True)
    options = g4_options(
        flavor, root if flavor == "file" else None, direct_io=direct_io
    )
    return build_kvcr(
        g4env,
        main=FakeMainAgent(),
        storage=FakeG4Agent(),
        config=KVCRConfig(nixl_agent_name="target", pool_layouts=[("", block)]),
        framework_regions=[
            RegionDescriptor(
                addr=ctypes.addressof(framework) + framework_offset,
                size=block,
                stride=stride,
                count=count,
                label="",
            )
        ],
        local_dram=LocalDramOptions(
            [("", ctypes.addressof(local) + local_offset, local_size)]
        ),
        g4=options,
    )


def _names_address(message: str, address: int) -> bool:
    return str(address) in message or hex(address) in message


def test_g4_file_direct_io_rejects_an_unaligned_framework_base(g4env, tmp_path):
    with pytest.raises(ValueError) as caught:
        _start_with_regions(g4env, tmp_path, framework_offset=512)
    message = str(caught.value)
    assert "direct_io=False" in message


def test_g4_file_direct_io_names_the_unaligned_framework_region(g4env, tmp_path):
    base = aligned(4 * PAGE)
    address = ctypes.addressof(base) + 512
    local = aligned(PAGE)
    root = tmp_path / "g4root"
    root.mkdir()
    with pytest.raises(ValueError) as caught:
        build_kvcr(
            g4env,
            main=FakeMainAgent(),
            storage=FakeG4Agent(),
            config=KVCRConfig(nixl_agent_name="target", pool_layouts=[("", PAGE)]),
            framework_regions=[
                RegionDescriptor(addr=address, size=PAGE, count=2, label="")
            ],
            local_dram=LocalDramOptions([("", ctypes.addressof(local), PAGE)]),
            g4=g4_options("file", root, direct_io=True),
        )
    assert _names_address(str(caught.value), address)


def test_g4_file_direct_io_rejects_an_unaligned_local_dram_base(g4env, tmp_path):
    with pytest.raises(ValueError) as caught:
        _start_with_regions(g4env, tmp_path, local_offset=512)
    assert "direct_io=False" in str(caught.value)


def test_g4_file_direct_io_rejects_a_size_that_is_not_a_page_multiple(g4env, tmp_path):
    with pytest.raises(ValueError) as caught:
        _start_with_regions(g4env, tmp_path, block=PAGE // 2)
    assert "direct_io=False" in str(caught.value)


def test_g4_file_direct_io_rejects_a_stride_that_is_not_a_page_multiple(
    g4env, tmp_path
):
    with pytest.raises(ValueError) as caught:
        _start_with_regions(g4env, tmp_path, stride=PAGE + 512)
    assert "direct_io=False" in str(caught.value)


def test_g4_file_direct_io_starts_with_page_aligned_regions(g4env, tmp_path):
    kvcr = _start_with_regions(g4env, tmp_path, stride=2 * PAGE)
    assert kvcr is not None


@pytest.mark.parametrize(
    "layout",
    [
        {"framework_offset": 512},
        {"local_offset": 512},
        {"block": PAGE // 2},
        {"stride": PAGE + 512},
    ],
    ids=["base", "local-base", "size", "stride"],
)
def test_g4_file_flavor_with_direct_io_false_starts_with_unaligned_regions(
    g4env, tmp_path, layout
):
    kvcr = _start_with_regions(g4env, tmp_path, direct_io=False, **layout)
    assert kvcr is not None
