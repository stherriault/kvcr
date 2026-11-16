# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""G4: unmanaged remote storage (shared file system).

KVCR keeps no G4 index. ``query`` starts an existence check, ``fetch`` and
``deliver`` read one file per key straight into the destination, and a
LOCAL_G2 eviction writes the victim and publishes it atomically.

Threads: transfers run on the G4 storage worker, which owns the storage
agent. Every file system call runs on the G4 file worker. Core state changes
only on caller threads under ``_state_lock``; workers report through
completion queues that ``poll_completed`` (and ``query``, for checks) drain.

Lifetime rule: destinations, sources, registrations and fds are released only
after the storage worker released the transfer handle at DONE or ERR. This
proves quiescence because POSIX uses linux AIO only.
"""

import contextlib
import hashlib
import logging
import os
import queue
import secrets
import socket
import stat
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from ..config import G4Options
from ..runtime import progress as _progress
from ..runtime.nixl_module import load_nixl, nixl_module_name
from ..runtime.progress import (
    _FAILED_RESULT,
    _SUCCESS_RESULT,
    _Op,
    _OpId,
    _TransferRef,
)
from ..types import (
    BlockKey,
    BlockMeta,
    CacheTier,
    OpEntryResult,
    OpHandle,
    PlacementAction,
    PlacementDecision,
    PlacementFailure,
    QueryStatus,
)
from .local_dram_pools import _layout_indices, _LocalDramState

if TYPE_CHECKING:
    from ..runtime.core import _KVCRCore

logger = logging.getLogger(__name__)

_FORMAT_VERSION = 1
_IDLE_WAIT_SECONDS = 0.0005
_WAKE_SECONDS = 0.05
_STOP = object()
_FETCHING = (QueryStatus.FETCHING, CacheTier.G4)
_FETCHABLE = (QueryStatus.FETCHABLE, CacheTier.G4)
_MISS = (QueryStatus.MISS, None)
_FALSE_VALUES = ("", "0", "false", "no", "off")


@dataclass(eq=False)
class _CheckJob:
    """One existence check for the workers."""

    key: BlockKey
    name: str
    started: bool = False
    started_at: float | None = None
    cancelled: bool = False
    result: Literal["", "present", "absent", "error"] = ""


@dataclass(eq=False)
class _Check:
    job: _CheckJob | None
    subscribers: set[str] = field(default_factory=set)
    # Deadline while the job is queued; a started job's runs from its start.
    deadline: float = 0.0


@dataclass(eq=False)
class _Xfer:
    """One key's transfer; fields are written by whichever stage owns it."""

    op: "_G4Op"
    key: BlockKey
    name: str
    local: list[tuple[int, int, int]]
    remote: list[tuple[int, int]]
    refs: list[_TransferRef]
    tmp: str | None = None
    fd: int | None = None
    fd_closed: bool = False
    reg: Any = None
    handle: Any = None
    start_failed: bool = False
    outcome: Literal["", "DONE", "ERR", "ABSENT", "PUBLISH_ERR"] = ""
    published: bool = False
    finished: bool = False


@dataclass(eq=False)
class _G4Op(_Op):
    kind: Literal["fill", "deliver", "spill"] = "deliver"
    handle: OpHandle | None = None
    deadline: float = 0.0
    xfers: list[_Xfer] = field(default_factory=list)
    results: dict[BlockKey, OpEntryResult] = field(default_factory=dict)
    ended: bool = False
    latched: bool = False

    @property
    def open_xfers(self) -> bool:
        return any(not xfer.finished for xfer in self.xfers)


def _create_agent(g4: "_G4", name: str) -> Any:
    """Create a storage agent with the G4 backend; runs on its owner thread."""
    module_name = nixl_module_name()
    try:
        # Same resolution as the main agent, so one NIXL build is loaded.
        module = load_nixl(module_name)
    except Exception as error:
        raise RuntimeError(f"G4 storage agent: {error}") from error
    try:
        agent = module.nixl_agent(name, module.nixl_agent_config(backends=[]))
    except Exception as error:
        raise RuntimeError(
            f"G4 cannot create a storage agent with NIXL module "
            f"{module_name!r} (KVCR_NIXL_MODULE): {error}"
        ) from error
    backend = g4._backend
    try:
        plugins = agent.get_plugin_list()
        if backend not in plugins:
            raise RuntimeError(f"plugin {backend} is not available")
        agent.create_backend(backend, dict(g4._backend_params))
    except Exception as error:
        raise RuntimeError(
            f"G4 cannot create NIXL backend {backend} with module "
            f"{module_name!r}; KVCR_NIXL_MODULE selects another wheel: {error}"
        ) from error
    return agent


class _StorageWorker:
    """Owns the storage NIXL agent and runs every transfer on it."""

    def __init__(self, g4: "_G4") -> None:
        self._g4 = g4
        self.cond = threading.Condition()
        self.jobs: deque[object] = deque()
        self.stopping = False
        self.ready = threading.Event()
        self.error: BaseException | None = None
        self.failure: BaseException | None = None
        self.unreleased: set[_Xfer] = set()
        self.agent: Any = None
        self.agent_name = f"{g4._kvcr.nixl_agent_name}-g4"
        self._dram_regs: list[Any] = []
        self.thread = threading.Thread(
            target=self._run, daemon=True, name="kvcr-g4-storage"
        )

    def submit(self, job: object) -> None:
        with self.cond:
            self.jobs.append(job)
            self.cond.notify()

    def stop(self) -> None:
        with self.cond:
            self.stopping = True
            self.cond.notify()

    def _setup(self) -> None:
        g4 = self._g4
        agent = self.agent = _create_agent(g4, self.agent_name)
        backend = g4._backend
        for region in g4._dram_regions():
            extent = region.size + (region.count - 1) * (region.stride or region.size)
            self._dram_regs.append(
                agent.register_memory(
                    [(region.addr, extent, region.device_Id, "")],
                    mem_type="DRAM",
                    backends=[backend],
                )
            )

    def _run(self) -> None:
        try:
            self._setup()
        except BaseException as error:
            self.error = error
            self._deregister_dram()
            self.ready.set()
            return
        self.ready.set()
        active: list[_Xfer] = []
        close_deadline: float | None = None
        try:
            while True:
                with self.cond:
                    if not (self.jobs or active or self.stopping):
                        self.cond.wait(_WAKE_SECONDS)
                    jobs = list(self.jobs)
                    self.jobs.clear()
                    stopping = self.stopping
                if stopping:
                    # Close drops every transfer that has not started.
                    jobs = []
                    if close_deadline is None:
                        close_deadline = (
                            time.monotonic() + _progress._OP_CLEANUP_TIMEOUT_SECONDS
                        )
                for job in jobs:
                    self._start(job, active)
                worked = self._poll(active)
                if stopping:
                    if not active:
                        break
                    if time.monotonic() >= close_deadline:
                        raise RuntimeError("G4 storage transfers did not finish")
                    time.sleep(_IDLE_WAIT_SECONDS)
                    continue
                if active and not worked:
                    time.sleep(_IDLE_WAIT_SECONDS)
        except BaseException as error:
            self.failure = error
        if not self.unreleased:
            try:
                self._deregister_dram()
            except BaseException as error:
                if self.failure is None:
                    self.failure = error

    def _deregister_dram(self) -> None:
        agent = self.agent
        regs, self._dram_regs = self._dram_regs, []
        for reg in reversed(regs):
            agent.deregister_memory(reg, backends=[self._g4._backend])

    def _start(self, xfer: _Xfer, active: list[_Xfer]) -> None:
        g4 = self._g4
        agent = self.agent
        backend = g4._backend
        try:
            dev = xfer.fd
            xfer.reg = agent.register_memory(
                [(0, g4._payload_len, dev, "")], mem_type="FILE", backends=[backend]
            )
            local = agent.get_xfer_descs(xfer.local, mem_type="DRAM")
            remote = agent.get_xfer_descs(
                [(offset, size, dev) for offset, size in xfer.remote], mem_type="FILE"
            )
            xfer.handle = agent.initialize_xfer(
                "WRITE" if xfer.op.kind == "spill" else "READ",
                local,
                remote,
                self.agent_name,
                backends=[backend],
            )
            if xfer.handle is None:
                raise RuntimeError("NIXL transfer creation returned None")
        except Exception:
            logger.warning("KVCR G4 transfer could not start", exc_info=True)
            self._deregister(xfer)
            xfer.start_failed = True
            xfer.outcome = "ERR"
            g4._after_storage(xfer)
            return
        self.unreleased.add(xfer)
        active.append(xfer)
        try:
            agent.transfer(xfer.handle)
        except Exception:
            # The handle's own state decides; it is released only when terminal.
            logger.warning("KVCR G4 transfer post was ambiguous", exc_info=True)

    def _deregister(self, xfer: _Xfer) -> None:
        reg, xfer.reg = xfer.reg, None
        if reg is None:
            return
        try:
            self.agent.deregister_memory(reg, backends=[self._g4._backend])
        except Exception:
            logger.warning("KVCR G4 deregistration failed", exc_info=True)

    def _poll(self, active: list[_Xfer]) -> bool:
        worked = False
        agent = self.agent
        for xfer in tuple(active):
            try:
                state = agent.check_xfer_state(xfer.handle)
            except Exception:
                logger.warning("KVCR G4 transfer state failed", exc_info=True)
                state = "ERR"
            if state in ("PROC", "PEND"):
                continue
            try:
                agent.release_xfer_handle(xfer.handle)
            except Exception:
                logger.warning("KVCR G4 handle release failed", exc_info=True)
                continue
            active.remove(xfer)
            self.unreleased.discard(xfer)
            self._deregister(xfer)
            xfer.outcome = "DONE" if state == "DONE" else "ERR"
            self._g4._after_storage(xfer)
            worked = True
        return worked


class _FileWorker:
    """Runs every file system call of the file flavor, in queue order."""

    def __init__(self, g4: "_G4") -> None:
        self._g4 = g4
        self.queue: queue.SimpleQueue[object] = queue.SimpleQueue()
        self.stopping = False
        self.thread = threading.Thread(
            target=self._run, daemon=True, name="kvcr-g4-file"
        )

    def stop(self) -> None:
        self.stopping = True
        self.queue.put(_STOP)

    def _run(self) -> None:
        while True:
            job = self.queue.get()
            # Close drops every job still queued.
            if job is _STOP or self.stopping:
                return
            try:
                self._do(job)
            except Exception:
                logger.exception("KVCR G4 file worker job failed")

    def _do(self, job: object) -> None:
        if isinstance(job, _CheckJob):
            if self._g4._try_start(job):
                self._check(job)
        elif job.handle is None and not job.start_failed:
            self._open(job)
        else:
            self._finish(job)

    def _check(self, job: _CheckJob) -> None:
        try:
            info = os.stat(job.name)
        except FileNotFoundError:
            job.result = "absent"
        except OSError:
            job.result = "error"
        else:
            if stat.S_ISREG(info.st_mode) and info.st_size == self._g4._payload_len:
                job.result = "present"
            else:
                job.result = "absent"
        self._g4._check_done.put(job)

    def _open(self, xfer: _Xfer) -> None:
        g4 = self._g4
        direct = os.O_DIRECT if g4._options.direct_io else 0
        if xfer.op.kind == "spill":
            try:
                g4._make_dirs(os.path.dirname(xfer.name))
                xfer.fd = os.open(
                    xfer.tmp,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | direct,
                    0o644,
                )
                os.ftruncate(xfer.fd, g4._payload_len)
            except OSError:
                logger.warning("KVCR G4 spill file could not open", exc_info=True)
                xfer.start_failed = True
                xfer.outcome = "ERR"
                self._finish(xfer)
                return
            g4._storage.submit(xfer)
            return
        try:
            fd = os.open(xfer.name, os.O_RDONLY | os.O_CLOEXEC | direct)
        except FileNotFoundError:
            xfer.outcome = "ABSENT"
            g4._op_done.put(xfer)
            return
        except OSError:
            logger.warning("KVCR G4 block file could not open", exc_info=True)
            xfer.outcome = "ERR"
            g4._op_done.put(xfer)
            return
        xfer.fd = fd
        try:
            size = os.fstat(fd).st_size
        except OSError:
            xfer.outcome = "ERR"
            self._finish(xfer)
            return
        if size != g4._payload_len:
            xfer.outcome = "ABSENT"
            self._finish(xfer)
            return
        g4._storage.submit(xfer)

    def _close_fd(self, xfer: _Xfer) -> None:
        if xfer.fd is not None and not xfer.fd_closed:
            xfer.fd_closed = True
            os.close(xfer.fd)

    def _finish(self, xfer: _Xfer) -> None:
        if xfer.op.kind == "spill" and xfer.tmp is not None:
            if xfer.outcome == "DONE":
                try:
                    os.fsync(xfer.fd)
                    self._close_fd(xfer)
                    os.rename(xfer.tmp, xfer.name)
                    xfer.published = True
                except OSError:
                    logger.warning("KVCR G4 spill publish failed", exc_info=True)
                    xfer.outcome = "PUBLISH_ERR"
            if not xfer.published:
                with contextlib.suppress(OSError):
                    self._close_fd(xfer)
                with contextlib.suppress(OSError):
                    os.unlink(xfer.tmp)
        else:
            try:
                self._close_fd(xfer)
            except OSError:
                logger.warning("KVCR G4 block file close failed", exc_info=True)
        self._g4._op_done.put(xfer)


class _G4:
    """G4 tier state owned by caller threads under the core state lock."""

    def __init__(self, kvcr: "_KVCRCore", options: G4Options) -> None:
        self._kvcr = kvcr
        self._options = options
        config = kvcr.config
        if kvcr._local_dram is None:
            raise ValueError("G4 requires configured local DRAM")
        if options.flavor != "file":
            raise ValueError("G4 flavor must be 'file'")
        prefix = options.key_prefix
        if (
            not isinstance(prefix, str)
            or not prefix.strip("/")
            or any(part in ("", ".", "..") for part in prefix.strip("/").split("/"))
        ):
            raise ValueError("G4 key_prefix must be a non-empty relative path")
        self._prefix = prefix.strip("/")
        layout = tuple(options.key_layout)
        kvcr._validate_expected_layout(list(layout))
        self._layout = layout
        self._sizes = [kvcr._block_sizes[label.partition(":")[0]] for label in layout]
        self._offsets = [sum(self._sizes[:index]) for index in range(len(layout))]
        self._payload_len = sum(self._sizes)
        schema = "|".join(
            [f"g4v{_FORMAT_VERSION}"]
            + [f"{label}={size}" for label, size in zip(layout, self._sizes)]
        )
        self._layout_tag = hashlib.sha256(schema.encode()).hexdigest()[:8]
        for name in (
            "max_inflight_checks",
            "max_check_results",
        ):
            if type(getattr(options, name)) is not int or getattr(options, name) < 1:
                raise ValueError(f"G4 {name} must be a positive integer")
        timeout = options.check_timeout_ms
        if timeout is not None and (type(timeout) is not int or timeout <= 0):
            raise ValueError("G4 check_timeout_ms must be a positive integer or None")
        if not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in options.backend_options.items()
        ):
            raise TypeError("G4 backend options must map strings to strings")
        params = dict(options.backend_options)
        if options.root is None:
            raise ValueError("G4 file flavor requires a root directory")
        self._backend = options.backend or "POSIX"
        for name in ("use_uring", "use_posix_aio"):
            if params.get(name, "").lower() not in _FALSE_VALUES:
                raise ValueError(f"G4 file flavor rejects POSIX {name}")
        if params.get("use_aio", "true").lower() in _FALSE_VALUES:
            raise ValueError("G4 file flavor requires POSIX use_aio")
        params["use_aio"] = "true"
        self._root = Path(options.root)
        if not self._root.is_dir():
            raise ValueError(f"G4 root {self._root} is not a directory")
        self._probe_root()
        self._backend_params = params
        self._check_timeout = (
            options.check_timeout_ms or config.operation_timeout_ms
        ) / 1000
        self._job_lock = threading.Lock()
        self._check_done: queue.SimpleQueue[_CheckJob] = queue.SimpleQueue()
        self._op_done: queue.SimpleQueue[_Xfer] = queue.SimpleQueue()
        self._checks: dict[BlockKey, _Check] = {}
        self._pending_start: deque[BlockKey] = deque()
        self._outbox: list[_CheckJob] = []
        self._pending_count = 0
        self._inflight = 0
        self._sub_count = 0
        self._results: OrderedDict[tuple[str, BlockKey], bool] = OrderedDict()
        self._ops: dict[_OpId, _G4Op] = {}
        self._pending_spills: set[BlockKey] = set()
        self._closing = False
        self._storage = _StorageWorker(self)
        self._file = _FileWorker(self)

    # Startup and naming.

    def _probe_root(self) -> None:
        directory = self._root / self._prefix / self._layout_tag
        directory.mkdir(parents=True, exist_ok=True)
        tmp = directory / f".probe.tmp.{self._temp_suffix()}"
        final = directory / f".probe.{self._temp_suffix()}"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o644)
        try:
            try:
                os.write(fd, b"kvcr-g4")
                os.fsync(fd)
            finally:
                os.close(fd)
            os.rename(tmp, final)
            os.unlink(final)
        finally:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
        if self._options.direct_io:
            self._probe_direct_io(directory)

    def _probe_direct_io(self, directory: Path) -> None:
        # Opening a directory with O_DIRECT succeeds on file systems that
        # refuse it for files, so the check opens a file.
        path = directory / f".probe.direct.{self._temp_suffix()}"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_DIRECT
        try:
            os.close(os.open(path, flags, 0o644))
            os.close(os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECT))
        except OSError as error:
            raise ValueError(
                f"G4 root {self._root} does not accept O_DIRECT ({error}); "
                "set direct_io=False to use buffered I/O"
            ) from error
        finally:
            with contextlib.suppress(OSError):
                os.unlink(path)

    @staticmethod
    def _temp_suffix() -> str:
        return f"{socket.gethostname()}.{os.getpid()}.{secrets.randbits(64):016x}"

    def _name(self, key: BlockKey) -> str:
        digest = hashlib.sha256(bytes(key)).hexdigest()
        name = f"{self._prefix}/{self._layout_tag}/{digest[:2]}/{digest}"
        return str(self._root / name)

    def _make_dirs(self, path: str) -> None:
        try:
            os.mkdir(path)
        except FileExistsError:
            return
        except FileNotFoundError:
            self._make_dirs(os.path.dirname(path))
            with contextlib.suppress(FileExistsError):
                os.mkdir(path)

    def _dram_regions(self):
        return [
            region
            for regions in self._kvcr._memory_regions
            for region in regions.values()
            if region.mem_type == "DRAM"
        ]

    def check_dram_alignment(self) -> None:
        """Refuse DRAM regions that O_DIRECT transfers cannot use."""
        if not self._options.direct_io:
            return
        page = os.sysconf("SC_PAGE_SIZE")
        for region in self._dram_regions():
            fields = {"address": region.addr, "size": region.size}
            if region.stride:
                fields["stride"] = region.stride
            bad = [name for name, value in fields.items() if value % page]
            if bad:
                raise ValueError(
                    f"G4 direct_io needs page aligned DRAM regions: region "
                    f"{region.label!r} at {region.addr:#x} has {' and '.join(bad)} "
                    f"not a multiple of {page}; align it or set direct_io=False"
                )

    def start(self, timeout: float) -> None:
        self._storage.thread.start()
        self._file.thread.start()
        if not self._storage.ready.wait(timeout):
            raise RuntimeError(f"G4 storage worker did not start within {timeout:g}s")
        if self._storage.error is not None:
            raise self._storage.error

    # Worker hand-offs.

    def _try_start(self, job: _CheckJob) -> bool:
        with self._job_lock:
            if job.cancelled:
                return False
            job.started_at = self._kvcr._clock()
            job.started = True
            return True

    def _try_cancel(self, job: _CheckJob) -> bool:
        with self._job_lock:
            if job.started:
                return False
            job.cancelled = True
            return True

    def _after_storage(self, xfer: _Xfer) -> None:
        self._file.queue.put(xfer)

    def _launch(self, xfer: _Xfer) -> None:
        self._file.queue.put(xfer)

    # Existence checks.

    def refresh(self) -> None:
        """Apply finished checks and check deadlines; start queued checks."""
        while True:
            try:
                job = self._check_done.get_nowait()
            except queue.Empty:
                break
            self._apply_check(job)
        now = self._kvcr._clock()
        for key, check in tuple(self._checks.items()):
            job = check.job
            if self._checks.get(key) is not check or job is None:
                continue
            if job.started_at is None:
                if now < check.deadline:
                    continue
                if self._try_cancel(job):
                    # Queue wait says nothing about storage: just MISS.
                    self._resolve_check(key, check, False)
                    continue
            if now < job.started_at + self._check_timeout:
                continue
            self._resolve_check(key, check, False)
        self._start_pending(now)
        self.submit_checks()

    def _apply_check(self, job: _CheckJob) -> None:
        check = self._checks.get(job.key)
        if check is None or check.job is not job:
            return
        self._resolve_check(job.key, check, job.result == "present")

    def _resolve_check(self, key: BlockKey, check: _Check, present: bool) -> None:
        del self._checks[key]
        if check.job is not None:
            self._inflight -= 1
        else:
            self._pending_count -= 1
        for owner in check.subscribers:
            self._sub_count -= 1
            self._results[(owner, key)] = present

    def _start_pending(self, now: float) -> None:
        while (
            self._pending_start
            and self._inflight < self._options.max_inflight_checks
            and not self._closing
        ):
            key = self._pending_start.popleft()
            check = self._checks.get(key)
            if check is not None and check.job is None:
                self._pending_count -= 1
                self._start_check_job(key, check, now)

    def _start_check_job(self, key: BlockKey, check: _Check, now: float) -> None:
        job = _CheckJob(key, self._name(key))
        check.job = job
        check.deadline = now + self._check_timeout
        self._inflight += 1
        self._outbox.append(job)

    def submit_checks(self) -> None:
        """Hand the checks started since the last call to the worker at once."""
        jobs, self._outbox = self._outbox, []
        for job in jobs:
            self._file.queue.put(job)

    def lookup(
        self, key: BlockKey, owner: str | None
    ) -> tuple[QueryStatus, CacheTier | None]:
        if owner is None:
            return _MISS
        present = self._results.get((owner, key))
        if present is not None:
            if present:
                return _FETCHABLE
            self._pop_result(key, owner)
            return _MISS
        check = self._checks.get(key)
        if check is not None and owner in check.subscribers:
            return _FETCHING
        if self._closing:
            return _MISS
        held = self._sub_count + self._pending_count + len(self._results)
        if held >= self._options.max_check_results:
            return _MISS
        if check is None:
            check = self._checks[key] = _Check(None)
            if self._inflight < self._options.max_inflight_checks:
                self._start_check_job(key, check, self._kvcr._clock())
            else:
                self._pending_count += 1
                self._pending_start.append(key)
        check.subscribers.add(owner)
        self._sub_count += 1
        return _FETCHING

    def start_check(self, keys: Collection[BlockKey], owner: str) -> None:
        """Start or join checks for ``owner``; results are read through query."""
        with self._kvcr._state_lock:
            self.refresh()
            for key in keys:
                self.lookup(key, owner)
            self.submit_checks()

    def _pop_result(self, key: BlockKey, owner: str) -> bool | None:
        return self._results.pop((owner, key), None)

    # Reads.

    def take_read_source(
        self, key: BlockKey, owner: str | None, labels: list[str]
    ) -> bool:
        """Consume ``owner``'s present result when ``labels`` map to the schema."""
        if owner is None or not self._pop_result(key, owner):
            return False
        return _layout_indices(self._layout, labels) is not None

    def take_fill_source(
        self, key: BlockKey, owner: str | None, labels: list[str]
    ) -> bool:
        return self.take_read_source(key, owner, labels)

    def _make_read(
        self, op: _G4Op, key: BlockKey, refs: list[_TransferRef]
    ) -> _Xfer | None:
        indices = _layout_indices(self._layout, [ref.label for ref in refs])
        if indices is None:
            return None
        local = []
        for ref in refs:
            descriptor = self._kvcr._address_descriptor(ref)
            if descriptor.mem_type != "DRAM":
                return None
            local.append((descriptor.addr, descriptor.size, descriptor.device_Id))
        remote = [(self._offsets[index], self._sizes[index]) for index in indices]
        return _Xfer(op, key, self._name(key), local, remote, list(refs))

    def _start_reads(
        self,
        op: _G4Op,
        blocks: Mapping[BlockKey, list[_TransferRef]],
    ) -> list[BlockKey]:
        rejected = []
        for key, refs in blocks.items():
            xfer = None if self._closing else self._make_read(op, key, refs)
            if xfer is None:
                rejected.append(key)
                continue
            op.xfers.append(xfer)
        if op.xfers:
            self._ops[op.op_id] = op
            for xfer in op.xfers:
                self._launch(xfer)
        return rejected

    def start_fill(
        self,
        op_handle: OpHandle,
        blocks: Mapping[BlockKey, list[_TransferRef]],
        deadline: float,
    ) -> bool:
        op = _G4Op(
            op_id=("g4_fill", op_handle),
            keys=frozenset(blocks),
            kind="fill",
            handle=op_handle,
            deadline=deadline,
        )
        rejected = self._start_reads(op, blocks)
        if rejected:
            self._kvcr._local_dram.complete_fill(
                rejected, success=False, source=CacheTier.G4
            )
        return True

    def start_deliver(
        self,
        op_handle: OpHandle,
        blocks: Mapping[BlockKey, list[_TransferRef]],
        deadline: float,
    ) -> None:
        op = _G4Op(
            op_id=("g4_deliver", op_handle),
            keys=frozenset(blocks),
            kind="deliver",
            handle=op_handle,
            deadline=deadline,
        )
        for key in self._start_reads(op, blocks):
            op.results[key] = _FAILED_RESULT
        if not op.xfers:
            self._kvcr._complete(op_handle, op.results)

    # Spills.

    def resolve_eviction(
        self,
        meta: BlockMeta,
        source: CacheTier,
        decision: PlacementDecision,
        deadline: float,
    ) -> tuple[PlacementDecision, bool]:
        """Start a LOCAL_G2 -> G4 spill; the deposit deadline does not bound it."""
        if source is not CacheTier.LOCAL_G2 or decision != (
            PlacementAction.MOVE_TO,
            CacheTier.G4,
        ):
            return decision, False
        core = self._kvcr
        key = meta.block_key
        record = core._block_record_map.get(key)
        residency = record.local_dram if record is not None else None
        if residency is None or key in self._pending_spills:
            return (PlacementAction.KEEP, None), False
        complete = (
            residency.state is _LocalDramState.READY
            and not residency.pending_labels
            and residency.key_layout == self._layout
            and residency.layout == self._layout
        )
        op_id = ("g4_spill", key)
        op = _G4Op(
            op_id=op_id,
            keys=frozenset((key,)),
            kind="spill",
        )
        if self._closing or not complete:
            return (PlacementAction.DROP, None), False
        sources = core._claim_local_dram_sources(op_id, (key,))
        if key not in sources:
            return (PlacementAction.KEEP, None), False
        refs = list(
            core._local_dram._pools.source_refs(
                sources[key], range(len(sources[key].slots))
            )
        )
        local = []
        for ref in refs:
            descriptor = core._address_descriptor(ref)
            local.append((descriptor.addr, descriptor.size, descriptor.device_Id))
        name = self._name(key)
        xfer = _Xfer(
            op,
            key,
            name,
            local,
            list(zip(self._offsets, self._sizes)),
            refs,
            tmp=f"{name}.tmp.{self._temp_suffix()}",
        )
        op.xfers.append(xfer)
        self._ops[op_id] = op
        self._pending_spills.add(key)
        core._add_block_dependencies(op, new_operation=True)
        self._launch(xfer)
        return (PlacementAction.KEEP, None), True

    def _retire_source(self, op: _G4Op) -> None:
        core = self._kvcr
        sources = core._local_dram_sources_by_op.get(op.op_id)
        if sources:
            core._local_dram._pools.retire_sources(sources)
            core._release_local_dram_sources(op.op_id)

    def _recover_spill_failure(self, op: _G4Op, key: BlockKey, reason: str) -> None:
        core = self._kvcr
        record = core._block_record_map.get(key)
        if record is not None and record.local_dram is not None:
            core._policy.decide_recovery(
                core._block_meta(
                    key,
                    record,
                    core._local_dram._pools._size_bytes(record.local_dram),
                ),
                PlacementFailure(
                    attempted=(PlacementAction.MOVE_TO, CacheTier.G4),
                    source=CacheTier.LOCAL_G2,
                    reason=reason,
                    failure_count=1,
                ),
            )
        self._retire_source(op)

    # Completion pump.

    def poll_main(self) -> None:
        failure = self._storage.failure
        if failure is not None and not self._closing:
            raise RuntimeError("KVCR G4 storage worker failed") from failure
        self.refresh()
        while True:
            try:
                xfer = self._op_done.get_nowait()
            except queue.Empty:
                break
            self._finish_xfer(xfer)
        now = self._kvcr._clock()
        for op in tuple(self._ops.values()):
            # A spill has no deadline; a read past its own never reports SUCCESS.
            if op.kind != "spill" and not op.latched and now >= op.deadline:
                op.latched = True

    def _finish_xfer(self, xfer: _Xfer) -> None:
        op = xfer.op
        if xfer.finished or self._ops.get(op.op_id) is not op:
            return
        xfer.finished = True
        outcome = xfer.outcome
        if op.kind == "spill":
            self._finish_spill(op, xfer)
            return
        done = outcome == "DONE" and not op.latched
        if op.kind == "fill":
            self._kvcr._local_dram.complete_fill(
                (xfer.key,), success=done, source=CacheTier.G4
            )
        else:
            op.results[xfer.key] = _SUCCESS_RESULT if done else _FAILED_RESULT
        self._end_op_if_done(op)

    def _end_op_if_done(self, op: _G4Op) -> None:
        if op.open_xfers or self._ops.get(op.op_id) is not op:
            return
        if not op.ended:
            op.ended = True
            if op.kind == "deliver":
                self._kvcr._complete(op.handle, op.results)
        self._ops.pop(op.op_id)

    def _finish_spill(self, op: _G4Op, xfer: _Xfer) -> None:
        core = self._kvcr
        self._ops.pop(op.op_id)
        self._pending_spills.discard(xfer.key)
        try:
            if xfer.outcome == "DONE" and xfer.published:
                self._retire_source(op)
            else:
                if xfer.outcome == "PUBLISH_ERR":
                    reason = "publish failed"
                elif xfer.start_failed:
                    reason = "start failed"
                else:
                    reason = "transfer failed"
                self._recover_spill_failure(op, xfer.key, reason)
        finally:
            core._remove_block_dependencies(op)

    # Close.

    def close_workers(self) -> None:
        """Drop unstarted jobs, let started transfers end, and join both workers."""
        self._closing = True
        self._storage.stop()
        self._file.stop()
        timeout = _progress._OP_CLEANUP_TIMEOUT_SECONDS
        for worker, bound in (
            (self._storage, timeout + _progress._JOIN_TIMEOUT_SECONDS),
            (self._file, timeout),
        ):
            if worker.thread.ident is not None:
                worker.thread.join(timeout=bound)
        if self._storage.thread.is_alive():
            raise RuntimeError("G4 storage worker did not stop")
        if self._file.thread.is_alive():
            raise RuntimeError("G4 file worker did not stop")
        if self._storage.failure is not None:
            raise self._storage.failure

    def is_quiescent(self) -> bool:
        return (
            not self._storage.thread.is_alive()
            and not self._file.thread.is_alive()
            and not self._storage.unreleased
        )

    def close_main(self) -> None:
        while True:
            try:
                xfer = self._op_done.get_nowait()
            except queue.Empty:
                break
            self._finish_xfer(xfer)
        local = self._kvcr._local_dram
        for op in tuple(self._ops.values()):
            self._ops.pop(op.op_id, None)
            if op.ended:
                continue
            open_keys = []
            for xfer in op.xfers:
                if xfer.finished:
                    continue
                xfer.finished = True
                open_keys.append(xfer.key)
                if xfer.fd is not None and not xfer.fd_closed:
                    xfer.fd_closed = True
                    with contextlib.suppress(OSError):
                        os.close(xfer.fd)
                if xfer.tmp is not None and not xfer.published:
                    with contextlib.suppress(OSError):
                        os.unlink(xfer.tmp)
            if op.kind == "spill":
                self._pending_spills.difference_update(op.keys)
                self._kvcr._release_local_dram_sources(op.op_id)
                for key in op.keys:
                    local._pools.abandon_capacity_eviction(key)
                self._kvcr._remove_block_dependencies(op)
            elif op.kind == "fill":
                if open_keys:
                    local.complete_fill(open_keys, success=False, source=CacheTier.G4)
            else:
                for key in open_keys:
                    op.results[key] = _FAILED_RESULT
                self._kvcr._complete(op.handle, op.results)
        self._checks.clear()
        self._pending_start.clear()
        self._outbox.clear()
        self._results.clear()
