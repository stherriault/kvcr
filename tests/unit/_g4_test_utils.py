# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Fake storage NIXL agent and KVCR builders for the G4 unit tests.

The storage agent mimics the NIXL 1.5.0 OBJ and POSIX plugins closely enough
to check the lifetime rules: one OBJ descriptor per handle, unique live OBJ
devIds, registrations released after the handle, and one owner thread.
File transfers run against real files, so the file flavor is exercised
end to end except for the NIXL transfer itself.
"""

import ctypes
import errno
import fcntl
import hashlib
import mmap
import os
import sys
import threading
import time
import types
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import msgspec
from _kvcr_test_utils import (
    _OPEN_KVCRS,
    FakeBytesControl,
    FakeNixlAgent,
    FakePrimaryPinning,
    _mem_descriptor,
)

from kvcr import KVCR, KVCRBindings
from kvcr.config import (
    KVCRBackendConfigs,
    KVCRConfig,
    LocalDramOptions,
    RemoteFWDramOptions,
)
from kvcr.policy import FIFOPolicy
from kvcr.types import (
    BlockKey,
    CacheTier,
    PlacementAction,
    QueryStatus,
    RegionDescriptor,
)

PAGE = os.sysconf("SC_PAGE_SIZE")
PREFIX = "kvcr/unit/model-a/base/bf16"
ERROR_XML = b"<Error><Code>NoSuchKey</Code></Error>"
FAKE_MODULE = "kvcr_unit_fake_nixl"
WAIT_SECONDS = 3.0


def fetching_g4():
    return (QueryStatus.FETCHING, CacheTier.G4)


def fetchable_g4():
    return (QueryStatus.FETCHABLE, CacheTier.G4)


def aligned(size: int):
    """Page-aligned ctypes buffer (O_DIRECT needs aligned memory)."""
    mapping = mmap.mmap(-1, size)
    buffer = (ctypes.c_char * size).from_buffer(mapping)
    buffer._mapping = mapping
    return buffer


def wait_for(predicate: Callable[[], bool], *, timeout: float = WAIT_SECONDS) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.001)
    raise AssertionError("condition was not reached")


def digest(key: BlockKey) -> str:
    return hashlib.sha256(bytes(key)).hexdigest()


def pattern(index: int, size: int = PAGE) -> bytes:
    return bytes([(index * 7 + 3) % 251 + 1]) * size


class FakeStorage:
    """Object store contents, shared by every agent that points at it."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def find(self, key: BlockKey, prefix: str = PREFIX) -> str | None:
        suffix = f"/{digest(key)[:2]}/{digest(key)}"
        for name in self.objects:
            if name.startswith(prefix + "/") and name.endswith(suffix):
                return name
        return None


def find_files(root: Path, key: BlockKey, prefix: str = PREFIX) -> list[Path]:
    d = digest(key)
    return sorted((root / prefix).glob(f"*/{d[:2]}/{d}"))


def tmp_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.tmp.*") if path.is_file())


@dataclass
class _Xfer:
    handle: int
    op: str
    kind: str
    local: list[tuple[int, int, int]]
    remote: list[tuple[Any, ...]]
    state: str = "NEW"
    snapshot: bytes | None = None
    released: bool = False
    deregistered_early: bool = False


class _Descs(list):
    mem_type: str | None = None


@dataclass
class _Registration:
    mem_type: str
    rows: list[tuple[Any, ...]]
    devs: list[int] = field(default_factory=list)
    fds: list[int] = field(default_factory=list)


class FakeG4Agent:
    """Storage-side NIXL agent for the OBJ and POSIX plugins."""

    def __init__(self, store: FakeStorage | None = None) -> None:
        self.store = store if store is not None else FakeStorage()
        self.name = ""
        self._lock = threading.RLock()
        self.backends: dict[str, dict[str, str]] = {}
        self.env_at_create: dict[str, str | None] = {}
        self.create_error: Exception | None = None
        self.calls: list[tuple[str, int]] = []
        self.dram_regs: list[_Registration] = []
        self.live_dram: list[_Registration] = []
        self.live_regs: list[_Registration] = []
        self.live_dev: dict[int, str] = {}
        self.live_fds: set[int] = set()
        self.devid_overwrites: list[int] = []
        self.violations: list[str] = []
        self.xfer_log: list[_Xfer] = []
        self.xfers: dict[int, _Xfer] = {}
        self.released: list[int] = []
        self.query_calls: list[tuple[str, ...]] = []
        self.query_blocked = 0
        self.query_gate: threading.Event | None = None
        self.query_error: Exception | None = None
        self.query_error_if: Callable[[tuple[str, ...]], bool] | None = None
        self.fd_direct: list[bool] = []
        self.init_gate: threading.Event | None = None
        self.init_waiting = 0
        self.init_error: Callable[[str], Exception | None] | None = None
        self.outcome = {"READ": "DONE", "WRITE": "DONE"}
        self.sequence: dict[str, list[str]] = {"READ": [], "WRITE": []}
        self.before_transfer: Callable[[_Xfer], None] | None = None
        self._next_handle = 1

    def _record(self, method: str) -> None:
        with self._lock:
            self.calls.append((method, threading.get_ident()))

    def threads(self) -> set[int]:
        with self._lock:
            return {ident for _, ident in self.calls}

    def get_plugin_list(self):
        self._record("get_plugin_list")
        return ["OBJ", "POSIX", "UCX", "GDS_MT"]

    def create_backend(self, backend, params=None):
        self._record("create_backend")
        if self.create_error is not None:
            raise self.create_error
        self.backends[backend] = dict(params or {})
        self.env_at_create[backend] = os.environ.get("AWS_EC2_METADATA_DISABLED")

    def get_backend_params(self, backend):
        self._record("get_backend_params")
        return self.backends[backend]

    def register_memory(self, descs, mem_type=None, backends=None):
        self._record("register_memory")
        rows = [tuple(row) for row in descs]
        reg = _Registration(str(mem_type), rows)
        with self._lock:
            if mem_type == "OBJ":
                for row in rows:
                    dev, name = row[2], row[3]
                    if dev in self.live_dev:
                        self.devid_overwrites.append(dev)
                    self.live_dev[dev] = name
                    reg.devs.append(dev)
                self.live_regs.append(reg)
            elif mem_type == "FILE":
                for row in rows:
                    self.live_fds.add(row[2])
                    reg.fds.append(row[2])
                    self.fd_direct.append(
                        bool(fcntl.fcntl(row[2], fcntl.F_GETFL) & os.O_DIRECT)
                    )
                self.live_regs.append(reg)
            else:
                self.dram_regs.append(reg)
                self.live_dram.append(reg)
        return reg

    def deregister_memory(self, handle, backends=None):
        self._record("deregister_memory")
        with self._lock:
            for xfer in self.xfers.values():
                if xfer.released:
                    continue
                refs = {row[2] for row in xfer.remote}
                if refs & (set(handle.devs) | set(handle.fds)):
                    self.violations.append("deregistered while its handle was live")
                    xfer.deregistered_early = True
            for dev in handle.devs:
                self.live_dev.pop(dev, None)
            for fd in handle.fds:
                self.live_fds.discard(fd)
            for registry in (self.live_regs, self.live_dram):
                if handle in registry:
                    registry.remove(handle)
        return True

    def query_memory(self, reg_list, backend, mem_type=None):
        self._record("query_memory")
        rows = [tuple(row) for row in reg_list]
        with self._lock:
            self.query_calls.append(tuple(str(row[3]) for row in rows))
        gate = self.query_gate
        if gate is not None and not gate.is_set():
            with self._lock:
                self.query_blocked += 1
            gate.wait(10)
            with self._lock:
                self.query_blocked -= 1
        if self.query_error is not None:
            raise self.query_error
        if self.query_error_if is not None and self.query_error_if(
            tuple(str(row[3]) for row in rows)
        ):
            raise RuntimeError("batch rejected")
        return [{} if row[3] in self.store.objects else None for row in rows]

    def get_xfer_descs(self, descs, mem_type=None):
        self._record("get_xfer_descs")
        result = _Descs(tuple(row) for row in descs)
        result.mem_type = mem_type
        return result

    def initialize_xfer(
        self, op, local_descs, remote_descs, remote_agent, notif_msg=b"", backends=None
    ):
        self._record("initialize_xfer")
        gate = self.init_gate
        if gate is not None and not gate.is_set():
            with self._lock:
                self.init_waiting += 1
            gate.wait(10)
            with self._lock:
                self.init_waiting -= 1
        if self.init_error is not None:
            error = self.init_error(op)
            if error is not None:
                raise error
        remote = [tuple(row) for row in remote_descs]
        local = [tuple(int(v) for v in row[:3]) for row in local_descs]
        kind = getattr(remote_descs, "mem_type", None)
        if kind not in ("OBJ", "FILE"):
            raise RuntimeError(f"unexpected remote memory type {kind!r}")
        if len(local) != len(remote) or any(
            low[1] != row[1] for low, row in zip(local, remote, strict=True)
        ):
            raise RuntimeError("NIXL rejected unaligned descriptors")
        if kind == "OBJ" and len(remote) != 1:
            raise RuntimeError("OBJ handle must carry exactly one descriptor")
        with self._lock:
            for row in remote:
                if kind == "OBJ" and row[2] not in self.live_dev:
                    raise RuntimeError("OBJ devId is not registered")
                if kind == "FILE" and row[2] not in self.live_fds:
                    raise RuntimeError("file descriptor is not registered")
            handle = self._next_handle
            self._next_handle += 1
            xfer = _Xfer(handle, op, kind, local, remote)
            self.xfers[handle] = xfer
            self.xfer_log.append(xfer)
        return handle

    def _next_outcome(self, op: str) -> str:
        with self._lock:
            queue = self.sequence[op]
            return queue.pop(0) if queue else self.outcome[op]

    def transfer(self, handle, notif_msg=b""):
        self._record("transfer")
        xfer = self.xfers[handle]
        if self.before_transfer is not None:
            self.before_transfer(xfer)
        outcome = self._next_outcome(xfer.op)
        if outcome == "HOLD":
            if xfer.kind == "OBJ" and xfer.op == "READ":
                name = self.live_dev[xfer.remote[0][2]]
                xfer.snapshot = self.store.objects.get(name)
            xfer.state = "PROC"
            return "PROC"
        self._complete(xfer, outcome)
        return xfer.state

    def finish(self, handle: int, state: str = "DONE") -> None:
        with self._lock:
            xfer = self.xfers[handle]
            if xfer.state == "PROC":
                self._complete(xfer, state)

    def finish_all(self, state: str = "DONE") -> None:
        with self._lock:
            for xfer in list(self.xfers.values()):
                if xfer.state == "PROC":
                    self._complete(xfer, state)

    def held(self, op: str | None = None) -> list[_Xfer]:
        with self._lock:
            return [
                x
                for x in self.xfer_log
                if x.state == "PROC" and (op is None or x.op == op)
            ]

    def log(self, op: str | None = None) -> list[_Xfer]:
        with self._lock:
            return [x for x in self.xfer_log if op is None or x.op == op]

    def _complete(self, xfer: _Xfer, state: str) -> None:
        if state == "DONE" and not self._move(xfer):
            state = "ERR"
        if state == "ERR" and xfer.op == "READ":
            addr, size, _ = xfer.local[0]
            ctypes.memmove(addr, ERROR_XML, min(size, len(ERROR_XML)))
        xfer.state = state

    def _move(self, xfer: _Xfer) -> bool:
        for (addr, size, _), row in zip(xfer.local, xfer.remote, strict=True):
            if xfer.kind == "OBJ":
                name = self.live_dev[row[2]]
                if xfer.op == "WRITE":
                    self.store.objects[name] = ctypes.string_at(addr, size)
                    continue
                data = xfer.snapshot or self.store.objects.get(name)
                if data is None:
                    return False
                # Silent short fill, as the OBJ plugin does.
                ctypes.memmove(addr, data, min(size, len(data)))
            else:
                offset = row[0]
                # Reopen so O_DIRECT alignment rules do not apply to the fake.
                plain = os.open(f"/proc/self/fd/{row[2]}", os.O_RDWR)
                try:
                    if xfer.op == "WRITE":
                        os.pwrite(plain, ctypes.string_at(addr, size), offset)
                        continue
                    data = os.pread(plain, size, offset)
                finally:
                    os.close(plain)
                if len(data) < size:
                    return False
                ctypes.memmove(addr, data, size)
        return True

    def check_xfer_state(self, handle):
        self._record("check_xfer_state")
        return self.xfers[handle].state

    def release_xfer_handle(self, handle):
        self._record("release_xfer_handle")
        with self._lock:
            xfer = self.xfers[handle]
            if xfer.state not in ("DONE", "ERR"):
                self.violations.append("released before terminal")
            if xfer.released:
                self.violations.append("released twice")
            xfer.released = True
            self.released.append(handle)

    def unreleased(self) -> list[int]:
        with self._lock:
            return [x.handle for x in self.xfer_log if not x.released]

    def open_gates(self) -> None:
        for gate in (self.query_gate, self.init_gate):
            if gate is not None:
                gate.set()
        self.finish_all("DONE")

    def leaks(self) -> list[str]:
        """Registrations or handles still live after close."""
        with self._lock:
            found = []
            if self.live_dev:
                found.append(f"OBJ devIds {sorted(self.live_dev)}")
            if self.live_fds:
                found.append(f"FILE fds {sorted(self.live_fds)}")
            if self.live_dram:
                found.append(f"{len(self.live_dram)} DRAM registrations")
            if self.unreleased():
                found.append(f"unreleased handles {self.unreleased()}")
        return found


class FakeMainAgent(FakeNixlAgent):
    """Main-agent fake that records the thread of every call."""

    def __init__(self) -> None:
        super().__init__()
        self.state = "DONE"
        self.calls: list[tuple[str, int]] = []
        self.backends: dict[str, dict[str, str]] = {}

    def _record(self, method: str) -> None:
        self.calls.append((method, threading.get_ident()))

    def get_plugin_list(self):
        return ["MOCK", "UCX"]

    def create_backend(self, backend, options):
        self.backends[backend] = dict(options)

    def get_backend_params(self, backend):
        return self.backends[backend]

    def register_memory(self, descs, mem_type="DRAM", backends=None):
        self._record("register_memory")
        return super().register_memory(descs, mem_type)

    def deregister_memory(self, handle, backends=None):
        self._record("deregister_memory")
        return super().deregister_memory(handle)

    def initialize_xfer(self, *args, **kwargs):
        self._record("initialize_xfer")
        return super().initialize_xfer(*args, **kwargs)

    def check_xfer_state(self, handle):
        self._record("check_xfer_state")
        return super().check_xfer_state(handle)

    def release_xfer_handle(self, handle):
        self._record("release_xfer_handle")
        return super().release_xfer_handle(handle)

    def threads(self) -> set[int]:
        return {ident for _, ident in self.calls}


class FakeNixlEnv:
    """Routes agent creation to the fake main agent or the fake storage agent.

    Both injection routes are covered: the progress module attributes and the
    module named by KVCR_NIXL_MODULE.
    """

    def __init__(self, monkeypatch) -> None:
        self.monkeypatch = monkeypatch
        self.gates: list[threading.Event] = []
        self.test_thread = threading.get_ident()
        self.agents: list[FakeG4Agent] = []
        self._main: FakeMainAgent | None = None
        self._storage: FakeG4Agent | None = None
        self._main_taken = False
        module = types.ModuleType(FAKE_MODULE)
        module.nixl_agent = self._create
        module.nixl_agent_config = lambda *args, **kwargs: kwargs
        module.nixlRemoteDisconnectError = type(
            "nixlRemoteDisconnectError", (Exception,), {}
        )
        monkeypatch.setitem(sys.modules, FAKE_MODULE, module)
        monkeypatch.setenv("KVCR_NIXL_MODULE", FAKE_MODULE)
        monkeypatch.delenv("AWS_EC2_METADATA_DISABLED", raising=False)
        from kvcr.runtime import progress

        monkeypatch.setattr(progress, "nixl_agent", self._create)
        monkeypatch.setattr(progress, "nixl_agent_config", module.nixl_agent_config)

    def _create(self, name, *args, **kwargs):
        if self._main is not None and not self._main_taken and name == self._main_name:
            self._main_taken = True
            agent: Any = self._main
        else:
            agent = self._storage
        agent.name = name
        return agent

    def activate(self, main: FakeMainAgent, storage: FakeG4Agent, name: str) -> None:
        self._main, self._storage, self._main_name = main, storage, name
        self._main_taken = False
        if storage not in self.agents:
            self.agents.append(storage)

    def release_all(self) -> None:
        for gate in self.gates:
            gate.set()
        for agent in self.agents:
            agent.open_gates()


class _MoveToG4Policy(FIFOPolicy):
    """FIFO eviction; every LOCAL_G2 victim is moved to G4."""

    def __init__(self) -> None:
        self.move = True
        self.failures: list[Any] = []
        self.ingested: list[Any] = []

    def on_ingest(self, meta, source):
        self.ingested.append((meta, source))

    def decide_eviction(self, meta, source):
        if self.move and source is CacheTier.LOCAL_G2:
            return (PlacementAction.MOVE_TO, CacheTier.G4)
        return super().decide_eviction(meta, source)

    def decide_recovery(self, meta, failure):
        self.failures.append((meta, failure))
        return super().decide_recovery(meta, failure)


def g4_options(flavor: str, root: Path | None, key_layout=None, **kwargs):
    from kvcr.config import G4Options

    # The suite runs on any tmp_path, which may refuse O_DIRECT.
    kwargs.setdefault("direct_io", False)
    if flavor == "object":
        kwargs.setdefault(
            "backend_options",
            {
                "bucket": "unit",
                "endpoint_override": "127.0.0.1:1",
                "scheme": "http",
                "access_key": "unit-access",
                "secret_key": "unit-secret",
            },
        )
        return G4Options(
            flavor="object",
            key_prefix=kwargs.pop("key_prefix", PREFIX),
            key_layout=tuple(key_layout if key_layout is not None else ("",)),
            root=root,
            **kwargs,
        )
    return G4Options(
        flavor="file",
        key_prefix=kwargs.pop("key_prefix", PREFIX),
        key_layout=tuple(key_layout if key_layout is not None else ("",)),
        root=root,
        **kwargs,
    )


class World:
    """One KVCR with G4 configured, its fake agents and its buffers."""

    def __init__(self) -> None:
        self.now = [0.0]
        self.results: dict[int, Any] = {}
        self.events: list[Any] = []
        self.fillers = 0

    def poll(self) -> None:
        for handle, result in self.kvcr.poll_completed():
            self.results[handle] = result

    def wait(self, handle, *, timeout: float = WAIT_SECONDS):
        deadline = time.monotonic() + timeout
        while handle not in self.results:
            if time.monotonic() > deadline:
                raise AssertionError("operation did not complete")
            self.poll()
            time.sleep(0.001)
        return self.results[handle]

    def settle(self, rounds: int = 20) -> None:
        """Poll a bounded number of times; used before a negative assertion."""
        for _ in range(rounds):
            self.poll()
            time.sleep(0.001)

    def poll_until(self, predicate, *, timeout: float = WAIT_SECONDS) -> None:
        deadline = time.monotonic() + timeout
        while not predicate():
            if time.monotonic() > deadline:
                raise AssertionError("condition was not reached")
            self.poll()
            time.sleep(0.001)

    def element(self, index: int) -> bytes:
        return bytes(self.engine_view[index * self.piece : (index + 1) * self.piece])

    def fill(self, index: int, data: bytes | None = None) -> None:
        data = data if data is not None else pattern(index, self.piece)
        self.engine_view[index * self.piece : (index + 1) * self.piece] = data

    def clear(self, index: int) -> None:
        self.fill(index, b"\0" * self.piece)

    def deposit(self, key: BlockKey, index: int, *, wait: bool = True):
        self.fill(index)
        handle = self.kvcr.deposit({key: [self.ref(index)]})
        return self.wait(handle)[key] if wait else handle

    def ref(self, index: int, label: str = ""):
        name = self.kvcr.config.nixl_agent_name
        return _mem_descriptor(index, label, end_point_name=name)

    def start_spill(self, key: BlockKey, index: int) -> int:
        """Deposit key, then a filler that evicts it; return the filler handle."""
        assert self.deposit(key, index).success
        self.fillers += 1
        self.filler = BlockKey(f"filler-{self.fillers}".encode())
        return self.deposit(self.filler, 7, wait=False)

    def spill(self, key: BlockKey, index: int) -> None:
        handle = self.start_spill(key, index)
        assert self.wait(handle)[self.filler].success
        assert self.stored(key)

    def stored(self, key: BlockKey) -> bool:
        if self.flavor == "object":
            return self.store.find(key, self.prefix) is not None
        return bool(find_files(self.root, key, self.prefix))

    def await_check(self, key: BlockKey, rid: str = "r"):
        deadline = time.monotonic() + WAIT_SECONDS
        while True:
            status = self.kvcr.query([key], rid)[0]
            if status[0] is not QueryStatus.FETCHING:
                return status
            if time.monotonic() > deadline:
                raise AssertionError("existence check did not finish")
            self.poll()
            time.sleep(0.001)

    def deliver(self, key: BlockKey, index: int, rid: str = "r") -> int:
        return self.kvcr.deliver({key: [self.ref(index)]}, rid)

    def fetch(self, key: BlockKey, rid: str = "r") -> int:
        return self.kvcr.fetch((key,), rid)

    def advance(self, seconds: float) -> None:
        self.now[0] = seconds

    def install_stat_spy(self) -> None:
        """File flavor: record, gate and fail the existence checks (stat calls)."""
        self.stat_gate = None
        self.stat_blocked = 0
        self.stat_log: list[str] = []
        self.stat_error = False
        real = os.stat
        root = str(self.root)

        def spy(path, *args, **kwargs):
            text = os.fspath(path) if isinstance(path, str | os.PathLike) else ""
            name = os.path.basename(text)
            mine = threading.get_ident() != self.env.test_thread
            if mine and text.startswith(root) and len(name) == 64:
                self.stat_log.append(name)
                gate = self.stat_gate
                if gate is not None and not gate.is_set():
                    self.stat_blocked += 1
                    gate.wait(10)
                    self.stat_blocked -= 1
                if self.stat_error:
                    raise OSError(errno.EIO, "check failed")
            return real(path, *args, **kwargs)

        self.env.monkeypatch.setattr(os, "stat", spy)

    def hold_checks(self) -> threading.Event:
        """Block every existence check until the returned event is set."""
        gate = threading.Event()
        self.env.gates.append(gate)
        if self.flavor == "object":
            self.storage.query_gate = gate
        else:
            self.stat_gate = gate
        return gate

    def blocked_checks(self) -> int:
        return (
            self.storage.query_blocked if self.flavor == "object" else self.stat_blocked
        )

    def check_names(self) -> list[str]:
        """Digests of the keys checked so far, in call order."""
        if self.flavor == "file":
            return list(self.stat_log)
        return [n.split("/")[-1] for call in self.storage.query_calls for n in call]

    def check_calls(self) -> int:
        """Storage calls made for checks (a batch is one call on OBJ)."""
        if self.flavor == "object":
            return len(self.storage.query_calls)
        return len(self.stat_log)

    def fail_checks(self) -> None:
        if self.flavor == "object":
            self.storage.query_error = RuntimeError("endpoint down")
        else:
            self.stat_error = True

    def query_only(self, key: BlockKey, rid: str = "r"):
        """Wait for a check result using query alone, never poll_completed."""
        deadline = time.monotonic() + WAIT_SECONDS
        while True:
            status = self.kvcr.query([key], rid)[0]
            if status[0] is not QueryStatus.FETCHING:
                return status
            if time.monotonic() > deadline:
                raise AssertionError("check result never reached query")
            time.sleep(0.001)

    @property
    def local_dram(self):
        return self.kvcr._core._local_dram


def build_kvcr(
    env: FakeNixlEnv,
    *,
    main: FakeMainAgent,
    storage: FakeG4Agent,
    config: KVCRConfig,
    framework_regions: list[RegionDescriptor],
    local_dram: LocalDramOptions | None,
    g4,
    policy=None,
    g3=None,
    on_resilience_event=None,
    guard_config=None,
) -> KVCR:
    pinning = FakePrimaryPinning()
    control = FakeBytesControl()
    config = msgspec.structs.replace(config, nixl_listen_port=1)
    pinning.agent_name = config.nixl_agent_name
    env.activate(main, storage, config.nixl_agent_name)
    kvcr = KVCR(
        config,
        KVCRBindings(
            request_pin=pinning.request_pin,
            poll_pin_results=pinning.poll_pin_results,
            release_pin=pinning.release_pin,
            framework_control=control,
            policy=policy,
            on_resilience_event=on_resilience_event,
        ),
        KVCRBackendConfigs(
            framework_regions=framework_regions,
            local_dram=local_dram,
            g3=g3,
            g4=g4,
            remote_fw_dram=RemoteFWDramOptions(),
        ),
        guard_config,
    )
    control._clock = lambda: kvcr._core._clock()
    _OPEN_KVCRS.append(kvcr)
    return kvcr


def new_world(
    env: FakeNixlEnv,
    tmp_path: Path,
    *,
    flavor: str = "object",
    slots: int = 1,
    store: FakeStorage | None = None,
    root: Path | None = None,
    policy="move",
    name: str = "target",
    g3=None,
    pages: int = 1,
    options=None,
    **g4_kwargs,
) -> World:
    """Single-pool world: one block is `pages` pages, `slots` G2 slots."""
    world = World()
    world.env = env
    world.flavor = flavor
    block = pages * PAGE
    world.piece = block
    world.prefix = g4_kwargs.get("key_prefix", PREFIX)
    world.store = store if store is not None else FakeStorage()
    world.root = root
    if flavor == "file" and root is None:
        world.root = tmp_path / "g4root"
    if world.root is not None and not world.root.exists():
        world.root.mkdir()
    world.engine = aligned(8 * block)
    world.engine_view = memoryview(world.engine).cast("B")
    world.local = aligned(slots * block)
    world.main = FakeMainAgent()
    world.storage = FakeG4Agent(world.store)
    world.policy = _MoveToG4Policy() if policy == "move" else policy
    world.options = (
        options if options is not None else g4_options(flavor, world.root, **g4_kwargs)
    )
    world.kvcr = build_kvcr(
        env,
        main=world.main,
        storage=world.storage,
        config=KVCRConfig(nixl_agent_name=name, pool_layouts=[("", block)]),
        framework_regions=[
            RegionDescriptor(
                addr=ctypes.addressof(world.engine), size=block, label="", count=8
            )
        ],
        local_dram=LocalDramOptions(
            [("", ctypes.addressof(world.local), len(world.local))]
        ),
        g4=world.options,
        policy=world.policy,
        on_resilience_event=world.events.append,
        g3=g3,
    )
    world.kvcr._core._clock = lambda: world.now[0]
    if flavor == "file" and world.root is not None:
        world.install_stat_spy()
    return world


def new_hybrid_world(
    env: FakeNixlEnv,
    tmp_path: Path,
    *,
    key_layout=("full", "swa:a", "swa:b"),
    flavor: str = "file",
    store: FakeStorage | None = None,
    **g4_kwargs,
) -> World:
    """One-block G2 with pools full (2 pages) and swa (1 page, 2 slots).

    Block i uses framework element i of `full` and elements 2i, 2i+1 of `swa`.
    """
    world = World()
    world.env = env
    world.flavor = flavor
    world.piece = PAGE
    world.prefix = PREFIX
    world.store = store if store is not None else FakeStorage()
    world.root = tmp_path / "g4root"
    world.root.mkdir(exist_ok=True)
    world.primary = {
        "full": ctypes.create_string_buffer(6 * PAGE),
        "swa": ctypes.create_string_buffer(6 * PAGE),
    }
    world.local_bufs = {
        "full": ctypes.create_string_buffer(2 * PAGE),
        "swa": ctypes.create_string_buffer(2 * PAGE),
    }
    for index in range(3):
        world.primary["full"][index * 2 * PAGE : (index + 1) * 2 * PAGE] = pattern(
            index, 2 * PAGE
        )
    for index in range(6):
        world.primary["swa"][index * PAGE : (index + 1) * PAGE] = pattern(
            index + 10, PAGE
        )
    named = ":" in key_layout[1] if len(key_layout) > 1 else False
    world.main = FakeMainAgent()
    world.storage = FakeG4Agent(world.store)
    world.policy = _MoveToG4Policy()
    world.options = g4_options(flavor, world.root, key_layout=key_layout, **g4_kwargs)
    world.kvcr = build_kvcr(
        env,
        main=world.main,
        storage=world.storage,
        config=KVCRConfig(
            nixl_agent_name="target",
            pool_layouts=[("full", 2 * PAGE), ("swa", PAGE)],
        ),
        framework_regions=[
            RegionDescriptor(
                addr=ctypes.addressof(world.primary["full"]),
                size=2 * PAGE,
                label="full",
                count=3,
            ),
            RegionDescriptor(
                addr=ctypes.addressof(world.primary["swa"]),
                size=PAGE,
                label="swa:*" if named else "swa",
                count=6,
            ),
        ],
        local_dram=LocalDramOptions(
            [
                (name, ctypes.addressof(buf), len(buf))
                for name, buf in world.local_bufs.items()
            ]
        ),
        g4=world.options,
        policy=world.policy,
        on_resilience_event=world.events.append,
    )
    world.kvcr._core._clock = lambda: world.now[0]
    world.layout = tuple(key_layout)
    if flavor == "file":
        world.install_stat_spy()
    return world


def hybrid_refs(index: int, layout=("full", "swa:a", "swa:b")):
    return [
        _mem_descriptor(index, layout[0]),
        _mem_descriptor(2 * index, layout[1]),
        _mem_descriptor(2 * index + 1, layout[2]),
    ]


def hybrid_block_bytes(world: World, index: int) -> bytes:
    """Payload of block `index` in key_layout order (full, swa, swa)."""
    full = bytes(world.primary["full"][index * 2 * PAGE : (index + 1) * 2 * PAGE])
    a = bytes(world.primary["swa"][2 * index * PAGE : (2 * index + 1) * PAGE])
    b = bytes(world.primary["swa"][(2 * index + 1) * PAGE : (2 * index + 2) * PAGE])
    return full + a + b


__all__ = [
    "ERROR_XML",
    "FakeG4Agent",
    "FakeMainAgent",
    "FakeNixlEnv",
    "FakeStorage",
    "PAGE",
    "PREFIX",
    "World",
    "build_kvcr",
    "fetchable_g4",
    "fetching_g4",
    "find_files",
    "g4_options",
    "hybrid_block_bytes",
    "hybrid_refs",
    "new_hybrid_world",
    "new_world",
    "pattern",
    "tmp_files",
    "wait_for",
]
