# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Resolve the NIXL Python API named by KVCR_NIXL_MODULE.

The main agent and the G4 storage agent must use the same module: loading two
NIXL builds in one process fails when their bindings register the same types.
"""

import importlib
import os
from types import ModuleType, SimpleNamespace

NIXL_MODULE_ENV = "KVCR_NIXL_MODULE"
_NAMES = ("nixl_agent", "nixl_agent_config", "nixlRemoteDisconnectError")
# Per-CUDA wheels (nixl_cu12) export the agent from the package root but keep
# the rest of the API in these submodules.
_SUBMODULES = ("_api", "_bindings")


def nixl_module_name() -> str:
    return os.environ.get(NIXL_MODULE_ENV) or "nixl"


def load_nixl(name: str | None = None) -> SimpleNamespace:
    """Return ``nixl_agent``, ``nixl_agent_config`` and
    ``nixlRemoteDisconnectError`` from ``name`` (default: KVCR_NIXL_MODULE).

    ``name`` may be a package such as ``nixl_cu12`` or its ``nixl_cu12._api``.
    """
    return _load(name or nixl_module_name())


def _load(name: str) -> SimpleNamespace:
    try:
        module = importlib.import_module(name)
    except ImportError as error:
        raise ImportError(
            f"cannot import NIXL module {name!r} (set by {NIXL_MODULE_ENV}): {error}"
        ) from error
    package = name.partition(".")[0]
    sources: list[ModuleType] = [module]
    found: dict[str, object] = {}
    for attr in _NAMES:
        value = _find(attr, sources, package)
        if value is None:
            raise ImportError(
                f"NIXL module {name!r} (set by {NIXL_MODULE_ENV}) has no {attr}"
            )
        found[attr] = value
    return SimpleNamespace(name=name, **found)


def _find(attr: str, sources: list[ModuleType], package: str) -> object | None:
    for source in sources:
        if hasattr(source, attr):
            return getattr(source, attr)
    for suffix in _SUBMODULES:
        submodule_name = f"{package}.{suffix}"
        if any(source.__name__ == submodule_name for source in sources):
            continue
        try:
            submodule = importlib.import_module(submodule_name)
        except ImportError:
            continue
        sources.append(submodule)
        if hasattr(submodule, attr):
            return getattr(submodule, attr)
    return None
