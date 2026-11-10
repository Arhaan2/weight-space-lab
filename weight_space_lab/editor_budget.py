"""Durable, reservation-first accounting for the development editor controller.

The controller, never the editor, owns this SQLite database. A reservation is
charged at its full bounds until a terminal result is durably reconciled.
Unknown usage stays null in observations and consumes its full reservation.
Deleting the database is not a supported restart: opening a missing ledger
fails unless the trusted operator explicitly creates a new run.

This is accounting, not a process sandbox. The controller must enforce the
returned deadline/token bounds and kill workers on cancellation or timeout.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import quote


LIMIT_FIELDS = frozenset({"editor_requests", "model_probes", "requested_sequences",
                          "generated_tokens", "elapsed_seconds", "billed_units"})
TERMINAL = frozenset({"completed", "failed", "cancelled", "timed_out", "interrupted"})
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_MAX_INT = 2**63 - 1


class BudgetError(ValueError):
    """A reservation, lifecycle transition, or frozen ledger is invalid."""


class BudgetExhausted(BudgetError):
    """The requested bounds do not fit within the remaining run budget."""


class BudgetCancelled(BudgetError):
    """The run was permanently cancelled; it cannot dispatch more work."""


def _integer(value, name, *, positive=False):
    if type(value) is not int or value < int(positive) or value > _MAX_INT:
        raise BudgetError(f"{name} must be a bounded {'positive' if positive else 'nonnegative'} integer")
    return value


def _seconds(value, name, *, positive=False):
    if (type(value) not in (int, float) or value < 0 or value > 1e12
            or not math.isfinite(value) or (positive and value == 0)):
        raise BudgetError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return float(value)


def _identifier(value, name):
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise BudgetError(f"{name} must be a bounded identifier")
    return value


def _json(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise BudgetError("ledger bindings must be finite JSON values") from error


def validate_limits(limits):
    if not isinstance(limits, dict) or set(limits) != LIMIT_FIELDS:
        raise BudgetError("budget limits have missing or unknown fields")
    checked = dict(limits)
    for key, value in checked.items():
        if key == "elapsed_seconds":
            checked[key] = _seconds(value, key)
        elif key != "billed_units" or value is not None:
            _integer(value, key)
    return checked


class BudgetLedger:
    """Single-run ledger with transactional reservations and immutable identity.

    ``billed_units`` is a nonnegative integer in the unit frozen in ``binding``
    (for example provider microcurrency units), never an inferred dollar cost.
    A null limit means billing is unavailable/not bounded here. Available
    reported billing is still recorded. A configured billing limit requires
    an explicit conservative billing bound on every reservation.
    """

    def __init__(self, path, *, run_id, limits, binding, create=False):
        self.path = Path(path)
        self.run_id = _identifier(run_id, "run_id")
        self.limits = validate_limits(limits)
        self.binding = json.loads(_json(binding))
        frozen = {"schema_version": 1, "run_id": run_id,
                  "limits": self.limits, "binding": self.binding}
        self._frozen_json = _json(frozen)
        self.configuration_sha256 = hashlib.sha256(self._frozen_json.encode()).hexdigest()
        if self.path.is_symlink():
            raise BudgetError("ledger may not be a symbolic link")
        if create:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            os.close(descriptor)
        if not self.path.is_file():
            raise BudgetError("ledger missing; restart cannot initialize a new budget")
        # mode=rw prevents sqlite from recreating a removed database on restart.
        self._db = sqlite3.connect(f"file:{quote(str(self.path.resolve()))}?mode=rw",
                                   uri=True, timeout=10, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA busy_timeout=10000")
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=FULL")
        self._db.execute("PRAGMA foreign_keys=ON")
        try:
            if create:
                self._initialize()
                directory_fd = os.open(self.path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            else:
                if self._db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                    raise BudgetError("ledger integrity check failed")
                row = self._db.execute("SELECT frozen_json FROM metadata WHERE singleton=1").fetchone()
                if row is None or row[0] != self._frozen_json:
                    raise BudgetError("frozen run identity, limits or binding changed")
        except BaseException:
            self._db.close()
            raise

    def _initialize(self):
        with self._transaction():
            self._db.execute("""CREATE TABLE metadata (
                singleton INTEGER PRIMARY KEY CHECK(singleton=1),
                frozen_json TEXT NOT NULL, created_at REAL NOT NULL,
                last_wall_time REAL NOT NULL, last_monotonic_time REAL NOT NULL,
                cancelled INTEGER NOT NULL DEFAULT 0 CHECK(cancelled IN (0,1)),
                cancellation_reason TEXT)""")
            self._db.execute("""CREATE TABLE attempts (
                request_id TEXT PRIMARY KEY, kind TEXT NOT NULL,
                retry_of TEXT REFERENCES attempts(request_id), attempt_index INTEGER NOT NULL,
                status TEXT NOT NULL, reserved_at REAL NOT NULL, dispatched_at REAL,
                reserved_monotonic REAL NOT NULL, deadline_monotonic REAL NOT NULL,
                deadline_at REAL NOT NULL, finished_at REAL,
                requested_sequences INTEGER NOT NULL, max_generated_tokens INTEGER NOT NULL,
                timeout_seconds REAL NOT NULL, max_billed_units INTEGER,
                completed_sequences INTEGER NOT NULL DEFAULT 0,
                failed_sequences INTEGER NOT NULL DEFAULT 0,
                observed_generated_tokens INTEGER, observed_elapsed_seconds REAL,
                observed_billed_units INTEGER,
                charged_generated_tokens INTEGER NOT NULL,
                charged_elapsed_seconds REAL NOT NULL, charged_billed_units INTEGER,
                conservative INTEGER NOT NULL DEFAULT 1,
                limit_violation INTEGER NOT NULL DEFAULT 0)""")
            self._db.execute("""CREATE TABLE events (
                event_index INTEGER PRIMARY KEY AUTOINCREMENT,
                occurred_at REAL NOT NULL, kind TEXT NOT NULL,
                request_id TEXT, detail_json TEXT NOT NULL)""")
            self._db.execute("INSERT INTO metadata(singleton,frozen_json,created_at,last_wall_time,last_monotonic_time) VALUES(1,?,?,?,?)",
                             (self._frozen_json, time.time(), time.time(), time.monotonic()))
            self._event("created", None, {"configuration_sha256": self.configuration_sha256})

    @contextmanager
    def _transaction(self):
        self._db.execute("BEGIN IMMEDIATE")
        try:
            yield
            self._db.execute("COMMIT")
        except BaseException:
            self._db.execute("ROLLBACK")
            raise

    def _event(self, kind, request_id, detail):
        self._db.execute("INSERT INTO events(occurred_at,kind,request_id,detail_json) VALUES(?,?,?,?)",
                         (time.time(), kind, request_id, _json(detail)))

    def _active(self):
        row = self._db.execute("SELECT cancelled,last_wall_time,last_monotonic_time FROM metadata WHERE singleton=1").fetchone()
        if row is None:
            raise BudgetError("ledger metadata missing")
        if row[0]:
            raise BudgetCancelled("run cancellation is persistent")
        if time.time() < row[1] or time.monotonic() < row[2]:
            raise BudgetExhausted("clock regressed; stop executors and reconcile interrupted requests")

    def _track_clocks(self):
        # Do not let a backwards wall clock or a reboot erase a prior time anchor.
        self._db.execute("UPDATE metadata SET last_wall_time=MAX(last_wall_time,?), last_monotonic_time=MAX(last_monotonic_time,?) WHERE singleton=1",
                         (time.time(), time.monotonic()))

    @staticmethod
    def _expired(row):
        wall, monotonic = time.time(), time.monotonic()
        return (wall < row["reserved_at"] or monotonic < row["reserved_monotonic"]
                or wall >= row["deadline_at"] or monotonic >= row["deadline_monotonic"])

    def _row(self, request_id):
        row = self._db.execute("SELECT * FROM attempts WHERE request_id=?", (request_id,)).fetchone()
        if row is None:
            raise BudgetError("unknown reservation")
        return dict(row)

    def _totals(self):
        row = self._db.execute("""SELECT
            SUM(kind='editor') AS editor_requests,
            SUM(kind='probe') AS model_probes,
            SUM(requested_sequences) AS requested_sequences,
            SUM(completed_sequences) AS completed_sequences,
            SUM(failed_sequences) AS failed_sequences,
            SUM(charged_generated_tokens) AS generated_tokens,
            SUM(charged_elapsed_seconds) AS elapsed_seconds,
            SUM(charged_billed_units) AS billed_units,
            SUM(observed_generated_tokens) AS observed_generated_tokens,
            SUM(observed_elapsed_seconds) AS observed_elapsed_seconds,
            SUM(observed_billed_units) AS observed_billed_units,
            SUM(retry_of IS NOT NULL) AS retries,
            SUM(status IN ('reserved','dispatched')) AS pending_requests,
            SUM(conservative) AS conservative_requests,
            SUM(limit_violation) AS limit_violations
            FROM attempts""").fetchone()
        # Observed unknowns are not falsely reported as measured zeros.
        nullable = {"billed_units", "observed_generated_tokens", "observed_elapsed_seconds", "observed_billed_units"}
        return {key: row[key] if row[key] is not None or key in nullable else 0 for key in row.keys()}

    def reserve(self, request_id, *, kind, sequences, max_generated_tokens,
                timeout_seconds, max_billed_units=None, retry_of=None):
        """Durably charge the maximum work before any provider dispatch.

        Request IDs cannot be reused. A retry is another fully charged attempt
        referencing a terminal unsuccessful attempt, with contiguous lineage.
        """
        _identifier(request_id, "request_id")
        if kind not in {"editor", "probe"}:
            raise BudgetError("reservation kind must be editor or probe")
        _integer(sequences, "sequences", positive=True)
        _integer(max_generated_tokens, "max_generated_tokens")
        timeout_seconds = _seconds(timeout_seconds, "timeout_seconds", positive=True)
        if max_billed_units is not None:
            _integer(max_billed_units, "max_billed_units")
        if self.limits["billed_units"] is not None and max_billed_units is None:
            raise BudgetError("a billing limit requires a conservative per-request billing bound")
        if retry_of is not None:
            _identifier(retry_of, "retry_of")
        with self._transaction():
            self._active()
            if self._db.execute("SELECT 1 FROM attempts WHERE request_id=?", (request_id,)).fetchone():
                raise BudgetError("request ID already reserved; no duplicate dispatch")
            attempt_index = 0
            if retry_of is not None:
                previous = self._row(retry_of)
                if previous["status"] not in TERMINAL - {"completed"} or previous["kind"] != kind:
                    raise BudgetError("retry must reference a terminal unsuccessful attempt of the same kind")
                if self._db.execute("SELECT 1 FROM attempts WHERE retry_of=?", (retry_of,)).fetchone():
                    raise BudgetError("retry lineage may not branch")
                attempt_index = previous["attempt_index"] + 1
            totals = self._totals()
            increments = {"editor_requests": int(kind == "editor"), "model_probes": int(kind == "probe"),
                          "requested_sequences": sequences, "generated_tokens": max_generated_tokens,
                          "elapsed_seconds": timeout_seconds, "billed_units": max_billed_units}
            for key, amount in increments.items():
                limit = self.limits[key]
                if limit is not None and (totals[key] or 0) + amount > limit:
                    raise BudgetExhausted(f"{key} budget exhausted")
            now, monotonic = time.time(), time.monotonic()
            self._track_clocks()
            self._db.execute("""INSERT INTO attempts(
                request_id,kind,retry_of,attempt_index,status,reserved_at,deadline_at,reserved_monotonic,deadline_monotonic,
                requested_sequences,max_generated_tokens,timeout_seconds,max_billed_units,
                charged_generated_tokens,charged_elapsed_seconds,charged_billed_units)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (request_id, kind, retry_of, attempt_index, "reserved", now, now + timeout_seconds,
                 monotonic, monotonic + timeout_seconds, sequences, max_generated_tokens, timeout_seconds, max_billed_units,
                 max_generated_tokens, timeout_seconds, max_billed_units))
            self._event("reserved", request_id, increments | {"retry_of": retry_of})
            return self._row(request_id)

    def mark_dispatched(self, request_id):
        """Record dispatch intent before launching. A crash is then chargeable."""
        with self._transaction():
            self._active()
            row = self._row(request_id)
            if row["status"] != "reserved":
                raise BudgetError("only an undispatched reservation may dispatch")
            if self._expired(row):
                raise BudgetExhausted("request deadline expired before dispatch")
            self._track_clocks()
            self._db.execute("UPDATE attempts SET status='dispatched',dispatched_at=? WHERE request_id=?",
                             (time.time(), request_id))
            self._event("dispatched", request_id, {})
            return self._row(request_id)

    def check_active(self, request_id=None):
        """Execution monitors call this before work and while work is running."""
        self._active()
        if request_id is not None:
            row = self._row(request_id)
            if row["status"] not in {"reserved", "dispatched"}:
                raise BudgetError("reservation is already terminal")
            if self._expired(row):
                raise BudgetExhausted("request deadline expired")
            return row
        return None

    def _finish(self, row, *, status, completed_sequences, failed_sequences,
                generated_tokens, elapsed_seconds, billed_units):
        charges = {
            "generated_tokens": row["max_generated_tokens"] if generated_tokens is None else generated_tokens,
            "elapsed_seconds": row["timeout_seconds"] if elapsed_seconds is None else elapsed_seconds,
            "billed_units": row["max_billed_units"] if billed_units is None else billed_units,
        }
        violation = any(charges[key] is not None and row[bound] is not None and charges[key] > row[bound]
                        for key, bound in (("generated_tokens", "max_generated_tokens"),
                                           ("elapsed_seconds", "timeout_seconds"),
                                           ("billed_units", "max_billed_units")))
        conservative = generated_tokens is None or elapsed_seconds is None or (
            row["max_billed_units"] is not None and billed_units is None)
        self._track_clocks()
        self._db.execute("""UPDATE attempts SET status=?,finished_at=?,
            completed_sequences=?,failed_sequences=?,observed_generated_tokens=?,
            observed_elapsed_seconds=?,observed_billed_units=?,charged_generated_tokens=?,
            charged_elapsed_seconds=?,charged_billed_units=?,conservative=?,limit_violation=?
            WHERE request_id=?""", (status, time.time(), completed_sequences, failed_sequences,
                                    generated_tokens, elapsed_seconds, billed_units,
                                    charges["generated_tokens"], charges["elapsed_seconds"], charges["billed_units"],
                                    int(conservative), int(violation), row["request_id"]))
        self._event(status, row["request_id"], {
            "completed_sequences": completed_sequences, "failed_sequences": failed_sequences,
            "observed": {"generated_tokens": generated_tokens, "elapsed_seconds": elapsed_seconds,
                         "billed_units": billed_units}, "charged": charges,
            "conservative": conservative, "limit_violation": violation})
        if violation:
            self._db.execute("UPDATE metadata SET cancelled=1,cancellation_reason='limit_violation' WHERE singleton=1")
            self._event("cancelled", row["request_id"], {"reason": "limit_violation"})
        return self._row(row["request_id"])

    def finish(self, request_id, *, status, completed_sequences, failed_sequences,
               generated_tokens=None, elapsed_seconds=None, billed_units=None):
        """Reconcile a stopped attempt; missing measurements spend their bounds.

        Overage is retained as evidence and permanently cancels the run. It is
        never silently clipped to a reservation. A terminal result is immutable.
        """
        if status not in TERMINAL - {"interrupted"}:
            raise BudgetError("invalid terminal status")
        _integer(completed_sequences, "completed_sequences")
        _integer(failed_sequences, "failed_sequences")
        if generated_tokens is not None:
            _integer(generated_tokens, "generated_tokens")
        if elapsed_seconds is not None:
            elapsed_seconds = _seconds(elapsed_seconds, "elapsed_seconds")
        if billed_units is not None:
            _integer(billed_units, "billed_units")
        with self._transaction():
            row = self._row(request_id)
            if row["status"] not in {"reserved", "dispatched"}:
                raise BudgetError("terminal accounting cannot be overwritten")
            if completed_sequences + failed_sequences != row["requested_sequences"]:
                raise BudgetError("every requested sequence must be completed or failed")
            if (status == "completed") != (failed_sequences == 0):
                raise BudgetError("terminal status and failed sequence count disagree")
            if row["status"] == "reserved" and completed_sequences:
                raise BudgetError("an undispatched reservation cannot have completed sequences")
            return self._finish(row, status=status, completed_sequences=completed_sequences,
                                failed_sequences=failed_sequences, generated_tokens=generated_tokens,
                                elapsed_seconds=elapsed_seconds, billed_units=billed_units)

    def recover_interrupted(self, *, executor_stopped=False):
        """Conservatively terminalize interrupted requests without re-dispatch.

        The controller must stop/confirm the old executor before passing True.
        Otherwise only expired/clock-regressed reservations are recoverable.
        Monotonic deadlines persist across processes; a reboot or backwards
        wall clock fails closed instead of granting a fresh timeout. No usage is
        refunded: full reserved tokens, time, and billing remain charged.
        """
        if type(executor_stopped) is not bool:
            raise BudgetError("executor_stopped must be boolean")
        with self._transaction():
            rows = self._db.execute("SELECT * FROM attempts WHERE status IN ('reserved','dispatched') ORDER BY reserved_at,request_id").fetchall()
            recovered = []
            for raw in rows:
                row = dict(raw)
                if executor_stopped or self._expired(row):
                    recovered.append(self._finish(row, status="interrupted", completed_sequences=0,
                                                  failed_sequences=row["requested_sequences"],
                                                  generated_tokens=None, elapsed_seconds=None, billed_units=None))
            return recovered

    def cancel(self, *, reason="operator_cancelled"):
        _identifier(reason, "cancellation reason")
        with self._transaction():
            self._db.execute("UPDATE metadata SET cancelled=1,cancellation_reason=COALESCE(cancellation_reason,?) WHERE singleton=1", (reason,))
            self._event("cancelled", None, {"reason": reason})
        return self.summary()

    def summary(self):
        # A read transaction makes metadata and totals one consistent snapshot.
        with self._transaction():
            metadata = self._db.execute("SELECT * FROM metadata WHERE singleton=1").fetchone()
            totals = self._totals()
            reservations = self._db.execute("""SELECT
                SUM(max_generated_tokens) AS generated_tokens,
                SUM(timeout_seconds) AS elapsed_seconds,
                SUM(max_billed_units) AS billed_units,
                SUM(CASE WHEN status IN ('reserved','dispatched') THEN max_generated_tokens ELSE 0 END) AS pending_generated_tokens,
                SUM(CASE WHEN status IN ('reserved','dispatched') THEN timeout_seconds ELSE 0 END) AS pending_elapsed_seconds,
                SUM(CASE WHEN status IN ('reserved','dispatched') THEN max_billed_units ELSE 0 END) AS pending_billed_units,
                SUM(observed_generated_tokens IS NOT NULL) AS token_measurements,
                SUM(observed_elapsed_seconds IS NOT NULL) AS elapsed_measurements,
                SUM(observed_billed_units IS NOT NULL) AS billing_measurements
                FROM attempts""").fetchone()
            reserved = {key: reservations[key] if reservations[key] is not None or key.endswith("billed_units") else 0
                        for key in reservations.keys() if not key.endswith("measurements")}
            observed = {"generated_tokens": totals["observed_generated_tokens"],
                        "elapsed_seconds": totals["observed_elapsed_seconds"],
                        "billed_units": totals["observed_billed_units"],
                        **{key: reservations[key] or 0 for key in reservations.keys() if key.endswith("measurements")}}
            remaining = {key: None if self.limits[key] is None else max(0, self.limits[key] - (totals[key] or 0))
                         for key in self.limits}
            return {"schema_version": 1, "run_id": self.run_id,
                    "configuration_sha256": self.configuration_sha256,
                    "limits": dict(self.limits), "reserved": reserved, "observed": observed,
                    "charged": totals, "remaining": remaining,
                    "cancelled": bool(metadata["cancelled"]),
                    "cancellation_reason": metadata["cancellation_reason"],
                    "billing_unit": self.binding.get("billing_unit") if isinstance(self.binding, dict) else None}

    def attempts(self):
        return [dict(row) for row in self._db.execute("SELECT * FROM attempts ORDER BY reserved_at,request_id")]

    def events(self):
        return [dict(row) | {"detail": json.loads(row["detail_json"])}
                for row in self._db.execute("SELECT * FROM events ORDER BY event_index")]

    def close(self):
        self._db.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
