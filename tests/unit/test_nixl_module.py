# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""KVCR_NIXL_MODULE resolution against fake NIXL packages."""

import sys
import textwrap

import pytest

from kvcr.runtime.nixl_module import load_nixl


def _package(root, name, *, root_exports_agent):
    """A fake per-CUDA wheel: the error class lives only in _bindings."""
    package = root / name
    package.mkdir()
    (package / "_bindings.py").write_text(
        "class nixlRemoteDisconnectError(Exception):\n    pass\n"
    )
    (package / "_api.py").write_text(
        textwrap.dedent(
            """
            from ._bindings import nixlRemoteDisconnectError as _unused

            class nixl_agent:
                pass

            class nixl_agent_config:
                pass
            """
        )
    )
    init = "from ._api import nixl_agent, nixl_agent_config\n"
    (package / "__init__.py").write_text(init if root_exports_agent else "")


@pytest.fixture()
def fake_root(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(tmp_path))
    yield tmp_path
    for name in list(sys.modules):
        if name.startswith("kvcr_fake_nixl"):
            del sys.modules[name]


@pytest.mark.parametrize("root_exports_agent", [True, False])
@pytest.mark.parametrize("suffix", ["", "._api"])
def test_package_and_api_forms_resolve_to_one_build(
    fake_root, monkeypatch, root_exports_agent, suffix
):
    name = f"kvcr_fake_nixl_{int(root_exports_agent)}"
    _package(fake_root, name, root_exports_agent=root_exports_agent)
    monkeypatch.setenv("KVCR_NIXL_MODULE", name + suffix)
    api = load_nixl()
    bindings = sys.modules[f"{name}._bindings"]
    assert api.nixl_agent is sys.modules[f"{name}._api"].nixl_agent
    assert api.nixl_agent_config is sys.modules[f"{name}._api"].nixl_agent_config
    assert api.nixlRemoteDisconnectError is bindings.nixlRemoteDisconnectError
    assert not any(
        module.startswith("kvcr_fake_nixl") and not module.startswith(name)
        for module in sys.modules
    )


def test_missing_module_names_the_knob(monkeypatch):
    monkeypatch.setenv("KVCR_NIXL_MODULE", "kvcr_fake_nixl_missing")
    with pytest.raises(ImportError, match="KVCR_NIXL_MODULE"):
        load_nixl()


def test_module_without_the_api_is_rejected(fake_root, monkeypatch):
    (fake_root / "kvcr_fake_nixl_empty.py").write_text("")
    monkeypatch.setenv("KVCR_NIXL_MODULE", "kvcr_fake_nixl_empty")
    with pytest.raises(ImportError, match="nixl_agent"):
        load_nixl()


def test_default_is_the_nixl_package(monkeypatch):
    monkeypatch.delenv("KVCR_NIXL_MODULE", raising=False)
    import nixl

    api = load_nixl()
    assert api.name == "nixl"
    assert api.nixlRemoteDisconnectError is nixl.nixlRemoteDisconnectError
