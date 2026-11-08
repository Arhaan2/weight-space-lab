"""Fresh durable dispatch gates for browser-integration-v2; never runs inference.

The supervisor owns this ledger. It must durably grant these gates before
acknowledging the worker, enforce the returned absolute deadline, and terminate
owned processes on stop. V1 reservations are never opened by this module.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import sqlite3
import time
from urllib.parse import quote

from .editor_budget import BudgetLedger
from .inventory import file_sha256, read_json
from .native_runner import _write_json
from .protocol import canonical_hash

KIND = "browser_integration_v2"
ARMS = ("baseline", "candidate")
SETTINGS = {"temperature": 0, "topP": 1, "seed": 42, "maxTokens": 256,
            "contextWindow": 4096, "warmups": 0, "warmupTokens": 16,
            "requestLimitMs": 60000, "thinking": False}
LIMITS = {"engine_load_attempts": 4, "additional_load_attempts": 2,
          "generation_dispatches": 2, "generation_dispatches_per_arm": 1,
          "tokens_per_arm": 256, "tokens_total": 512, "wall_seconds": 1200,
          "generation_reservation_seconds": 70, "minimum_generation_remaining_seconds": 75,
          "warmups": 0, "new_proposals": 0, "experimental_frontier_requests": 0}
GEN_LIMITS = {"editor_requests": 0, "model_probes": 2, "requested_sequences": 2,
              "generated_tokens": 512, "elapsed_seconds": 1200, "billed_units": 0}
RETRYABLE = {"url", "asset_routing", "fetch", "cache"}
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_SHA = re.compile(r"[0-9a-f]{64}\Z")


class _ClockStop(ValueError):
    stop_reason = "clock_regressed"


class _DeadlineStop(TimeoutError):
    stop_reason = "deadline_expired"


def _save(path, value):
    _write_json(Path(path), value, immutable=True)


def _identifier(value):
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError("bounded attempt identifier required")
    return value


def _digest(value):
    if not isinstance(value, str) or not _SHA.fullmatch(value):
        raise ValueError("SHA256 required")
    return value


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def prepare(root, *, authorization_source, arm_configs, preserved_files=None,
            artifact_manifest="artifacts/browser-integration-v1/conversion/artifacts.json"):
    """Create a new manifest and ledgers; copy configs, never load an engine."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if root.is_symlink() or any((root / p).exists() for p in
            ("manifest.json", "ready.json", "loads.sqlite3", "generation-budget.sqlite3")):
        raise ValueError("v2 preparation requires unused manifest/ledger paths")
    if set(arm_configs) != set(ARMS):
        raise ValueError("both exact arms required")
    original = Path(authorization_source)
    source = root / "authorization-source.txt"
    if original.resolve() != source.resolve():
        with source.open("xb") as stream:
            stream.write(original.read_bytes())
            stream.flush()
            os.fsync(stream.fileno())
    source.chmod(0o600)
    if not source.read_bytes() or source.stat().st_size > 1_048_576:
        raise ValueError("bounded nonempty private authorization source required")
    pair = read_json(artifact_manifest)
    configs = {arm: read_json(path) for arm, path in arm_configs.items()}
    for arm, config in configs.items():
        if (config.get("arm") != arm or config.get("settings") != SETTINGS
                or len(config.get("items", [])) != 1
                or config.get("sourceCheckpointSha256") != pair["arms"][arm]["source_checkpoint_sha256"]
                or config.get("convertedArtifactSha256") != pair["arms"][arm]["artifact_sha256"]):
            raise ValueError("v2 must reuse exact verified artifacts, smoke schedule and settings")
        _digest(config["configurationSha256"])
        target = root / (arm + "-config.json")
        if Path(arm_configs[arm]).resolve() != target.resolve():
            _save(target, config)
    base, edit = configs["baseline"], configs["candidate"]
    if (base["items"] != edit["items"] or base["settings"] != edit["settings"]
            or base["configurationSha256"] != edit["configurationSha256"]
            or base["model"]["modelUrl"] == edit["model"]["modelUrl"]
            or base["model"]["id"] == edit["model"]["id"]
            or base["model"]["modelLib"] != edit["model"]["modelLib"]
            or base["convertedArtifactSha256"] == edit["convertedArtifactSha256"]):
        raise ValueError("matched settings and distinct model/cache identities required")
    protected = dict(preserved_files or {})
    for name, digest in protected.items():
        if file_sha256(name) != digest:
            raise ValueError("protected prior artifact differs")
    manifest = {"schema_version": 2, "kind": KIND, "author": "Arhaan2",
        "strict_ready": False, "editor_isolation": "unverified",
        "authorization_source_sha256": file_sha256(source), "limits": LIMITS,
        "configuration_sha256": base["configurationSha256"],
        "artifact_manifest": str(Path(artifact_manifest).resolve()),
        "artifact_manifest_sha256": file_sha256(artifact_manifest),
        "arm_config_files": {arm: {"path": str((root / (arm + "-config.json")).resolve()),
                                 "sha256": file_sha256(root / (arm + "-config.json"))} for arm in ARMS},
        "prior_artifacts": protected,
        "historical_capacity_transferred": False,
        "time_scope": "one wall deadline from first durable load grant through final cleanup; no reset",
        "generation_accounting_scope": "BudgetLedger observes generation time separately; global clock includes initialization, repairs and cleanup",
        "retry_policy": "confirmed pre-generation URL/asset-routing/fetch/cache failure only, changed verified code/correction, complete owned-resource cleanup; max two additional load attempts"}
    _save(root / "manifest.json", manifest)
    _save(root / "ready.json", {"manifest_sha256": canonical_hash(manifest)})
    with Controller(root, create=True) as controller:
        summary = controller.summary()
        _save(root / "accounting-before.json", summary)
    return {"prepared": True, "manifest_sha256": canonical_hash(manifest),
            "engine_loads": 0, "generation_dispatches": 0}


class Controller:
    """Fail-closed load state machine plus the existing generation BudgetLedger."""

    def __init__(self, root, *, create=False):
        self.root = Path(root)
        self.manifest = read_json(self.root / "manifest.json")
        self.digest = canonical_hash(self.manifest)
        if (self.manifest.get("kind") != KIND or self.manifest["limits"] != LIMITS
                or read_json(self.root / "ready.json") != {"manifest_sha256": self.digest}
                or file_sha256(self.root / "authorization-source.txt") != self.manifest["authorization_source_sha256"]):
            raise ValueError("v2 manifest/authorization identity changed")
        for arm, info in self.manifest["arm_config_files"].items():
            if file_sha256(info["path"]) != info["sha256"]:
                raise ValueError("frozen arm configuration changed")
        if file_sha256(self.manifest["artifact_manifest"]) != self.manifest["artifact_manifest_sha256"]:
            raise ValueError("verified conversion manifest changed")
        self.configs = {arm: read_json(info["path"]) for arm, info in self.manifest["arm_config_files"].items()}
        self.generations = BudgetLedger(self.root / "generation-budget.sqlite3",
            run_id="browser-integration-v2-generations", limits=GEN_LIMITS,
            binding={"manifest_sha256": self.digest, "billing_unit": "none_local_execution",
                     "time_scope": "generation only; global wall deadline in loads.sqlite3"}, create=create)
        path = self.root / "loads.sqlite3"
        if path.is_symlink():
            raise ValueError("load ledger must not be a symlink")
        if create:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.close(descriptor)
        if not path.is_file():
            raise ValueError("load ledger missing; do not recreate capacity")
        self.db = sqlite3.connect(f"file:{quote(str(path.resolve()))}?mode=rw", uri=True,
                                  isolation_level=None, timeout=10)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        if create:
            with self._transaction():
                self.db.execute("CREATE TABLE state (id INTEGER PRIMARY KEY CHECK(id=1), manifest_sha256 TEXT NOT NULL, started_wall REAL, started_mono REAL, deadline_wall REAL, deadline_mono REAL, last_wall REAL NOT NULL, last_mono REAL NOT NULL, stopped INTEGER NOT NULL DEFAULT 0, stop_reason TEXT)")
                self.db.execute("INSERT INTO state(id,manifest_sha256,last_wall,last_mono) VALUES(1,?,?,?)", (self.digest,time.time(),time.monotonic()))
                self.db.execute("CREATE TABLE versions (sha TEXT PRIMARY KEY, record TEXT NOT NULL)")
                self.db.execute("CREATE TABLE attempts (attempt_id TEXT PRIMARY KEY, arm TEXT NOT NULL, code_sha256 TEXT NOT NULL, status TEXT NOT NULL, began_wall REAL NOT NULL, began_mono REAL NOT NULL, generation_state TEXT NOT NULL DEFAULT 'not_dispatched', generation_wall REAL, generation_mono REAL, result TEXT)")
                self.db.execute("CREATE TABLE events (sequence INTEGER PRIMARY KEY AUTOINCREMENT, wall REAL NOT NULL, kind TEXT NOT NULL, detail TEXT NOT NULL)")
                self._event("created", {"manifest_sha256": self.digest})
            fd = os.open(self.root, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        elif (self.db.execute("PRAGMA quick_check").fetchone()[0] != "ok"
                or self._state()["manifest_sha256"] != self.digest):
            raise ValueError("load ledger integrity or binding changed")

    @contextmanager
    def _transaction(self):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            yield
            self.db.execute("COMMIT")
        except BaseException as exc:
            self.db.execute("ROLLBACK")
            if isinstance(exc, (_ClockStop, _DeadlineStop)):
                # A failed grant must still permanently record a safety stop.
                with self._transaction():
                    self.db.execute("UPDATE state SET stopped=1,stop_reason=? WHERE id=1", (exc.stop_reason,))
                    self._event("terminal_stop", {"reason": exc.stop_reason})
                self.generations.cancel(reason=exc.stop_reason)
            raise

    def _event(self, kind, detail):
        self.db.execute("INSERT INTO events(wall,kind,detail) VALUES(?,?,?)", (time.time(),kind,_json(detail)))

    def _state(self):
        return dict(self.db.execute("SELECT * FROM state WHERE id=1").fetchone())

    def _tick(self):
        state = self._state()
        if state["stopped"]:
            raise ValueError("v2 run is terminal: " + str(state["stop_reason"]))
        if time.time() < state["last_wall"] or time.monotonic() < state["last_mono"]:
            raise _ClockStop("clock regressed; unknown execution state, stop required")
        if state["deadline_wall"] is not None and self.remaining_seconds() <= 0:
            raise _DeadlineStop("one v2 wall deadline expired; it cannot be reset")
        self.db.execute("UPDATE state SET last_wall=?,last_mono=? WHERE id=1", (time.time(),time.monotonic()))

    def remaining_seconds(self):
        state = self._state()
        if state["started_wall"] is None:
            return LIMITS["wall_seconds"]
        return max(0.0, min(state["deadline_wall"]-time.time(), state["deadline_mono"]-time.monotonic()))

    def deadline(self):
        state = self._state()
        return {"started_epoch_ms": None if state["started_wall"] is None else state["started_wall"]*1000,
                "deadline_epoch_ms": None if state["deadline_wall"] is None else state["deadline_wall"]*1000,
                "deadline_monotonic": state["deadline_mono"], "remaining_seconds": self.remaining_seconds()}

    def register_code_version(self, files, *, preflight_path, correction=None):
        """Record a reviewed, fetch-only verified served build before a load gate."""
        if not files:
            raise ValueError("code version requires explicit source/build file hashes")
        for name, digest in files.items():
            if file_sha256(name) != _digest(digest):
                raise ValueError("code version bytes differ")
        proof = read_json(preflight_path)
        # Root may supply a normalized envelope around the preserved browser report.
        if (proof.get("status") != "passed" or proof.get("fetch_only") is not True
                or proof.get("engine_loads") != 0 or proof.get("generation_dispatches") != 0):
            raise ValueError("code registration requires successful fetch-only browser preflight")
        if correction is not None:
            required = {"previous_attempt_id", "failure_class", "diagnosis", "correction"}
            if not required <= correction.keys() or correction["failure_class"] not in RETRYABLE:
                raise ValueError("specific eligible correction required")
            if any(not isinstance(correction[k], str) or not correction[k].strip() for k in required):
                raise ValueError("specific nonempty correction evidence required")
        record = {"files": dict(files), "preflight_path": str(Path(preflight_path).resolve()),
                  "preflight_sha256": file_sha256(preflight_path), "correction": correction,
                  "content_sha256": canonical_hash(sorted(files.values()))}
        # Distinct evidence alone cannot manufacture a changed executable version.
        code_sha = canonical_hash(dict(files))
        with self._transaction():
            self._tick()
            if self.db.execute("SELECT 1 FROM versions WHERE sha=?", (code_sha,)).fetchone():
                existing = json.loads(self.db.execute("SELECT record FROM versions WHERE sha=?",(code_sha,)).fetchone()[0])
                if existing != record:
                    raise ValueError("code version is immutable; changed correction needs changed code")
                return code_sha
            self.db.execute("INSERT INTO versions VALUES(?,?)", (code_sha,_json(record)))
            self._event("code_version_registered", {"code_sha256": code_sha, **record})
        return code_sha

    def _version(self, code_sha):
        row = self.db.execute("SELECT record FROM versions WHERE sha=?", (_digest(code_sha),)).fetchone()
        if row is None:
            raise ValueError("unregistered served code version")
        value = json.loads(row[0])
        for path,digest in value["files"].items():
            if file_sha256(path) != digest:
                raise ValueError("served code changed since successful preflight")
        if file_sha256(value["preflight_path"]) != value["preflight_sha256"]:
            raise ValueError("fetch-only preflight evidence changed")
        return value

    def attempts(self):
        return [dict(row) | {"result": json.loads(row["result"]) if row["result"] else None}
                for row in self.db.execute("SELECT * FROM attempts ORDER BY began_mono,attempt_id")]

    def begin_load(self, arm, attempt_id, code_sha256):
        """Count a load and start the single clock durably before its ACK."""
        if arm not in ARMS:
            raise ValueError("invalid arm")
        _identifier(attempt_id)
        version = self._version(code_sha256)
        with self._transaction():
            self._tick()
            attempts = self.attempts()
            if any(a["status"] == "loading" for a in attempts):
                raise ValueError("previous owned attempt has not been released and finalized")
            if any(a["attempt_id"] == attempt_id for a in attempts):
                raise ValueError("load attempt already consumed; no duplicate ACK")
            if len(attempts) >= 4:
                raise ValueError("four engine-loading attempts exhausted")
            if any(a["arm"] == arm and a["generation_state"] != "not_dispatched" for a in attempts):
                raise ValueError("arm generation slot was dispatched or ambiguous; no more loads")
            same = [a for a in attempts if a["arm"] == arm]
            if arm == "candidate" and not any(a["arm"] == "baseline" and a["status"] == "completed" for a in attempts):
                raise ValueError("candidate must follow released successful baseline execution")
            if same:
                previous = same[-1]
                if not previous["result"].get("retry_eligible"):
                    raise ValueError("previous attempt is not eligible for a loading retry")
                correction = version["correction"]
                previous_version = json.loads(self.db.execute("SELECT record FROM versions WHERE sha=?",
                                                             (previous["code_sha256"],)).fetchone()[0])
                if (not correction or correction["previous_attempt_id"] != previous["attempt_id"]
                        or correction["failure_class"] != previous["result"]["failure_class"]
                        or code_sha256 == previous["code_sha256"]
                        or version["content_sha256"] == previous_version["content_sha256"]):
                    raise ValueError("retry requires diagnosed specific correction and changed code version")
                extra = sum(max(0, sum(a["arm"] == x for a in attempts)-1) for x in ARMS)
                if extra >= 2:
                    raise ValueError("two additional load attempts exhausted")
            state = self._state()
            now, mono = time.time(), time.monotonic()
            if state["started_wall"] is None:
                self.db.execute("UPDATE state SET started_wall=?,started_mono=?,deadline_wall=?,deadline_mono=? WHERE id=1",
                                (now,mono,now+1200,mono+1200))
            self.db.execute("INSERT INTO attempts(attempt_id,arm,code_sha256,status,began_wall,began_mono) VALUES(?,?,?,'loading',?,?)",
                            (attempt_id,arm,code_sha256,now,mono))
            self._event("load_dispatch_granted", {"arm": arm,"attempt_id":attempt_id,"code_sha256":code_sha256})
        return {"approved": True, "kind": "load", "arm": arm, "attempt_id": attempt_id,
                "code_version_sha256": code_sha256, **self.deadline()}

    def dispatch_generation(self, arm, attempt_id):
        """Irrevocably consume this arm before permission reaches the worker."""
        with self._transaction():
            self._tick()
            if self.remaining_seconds() < 75:
                raise TimeoutError("less than 75 seconds remain for response and cleanup")
            row = self.db.execute("SELECT * FROM attempts WHERE attempt_id=?", (attempt_id,)).fetchone()
            if row is None or row["arm"] != arm or row["status"] != "loading":
                raise ValueError("generation requires this active load attempt")
            if self.db.execute("SELECT 1 FROM attempts WHERE arm=? AND generation_state!='not_dispatched'",(arm,)).fetchone():
                raise ValueError("one generation slot already consumed or ambiguous")
            # Cross-database crash ordering is intentionally conservative: this
            # marker alone consumes the slot even if BudgetLedger is interrupted.
            self.db.execute("UPDATE attempts SET generation_state='intent',generation_wall=?,generation_mono=? WHERE attempt_id=?",
                            (time.time(),time.monotonic(),attempt_id))
            self._event("generation_intent_consumed", {"arm":arm,"attempt_id":attempt_id,"max_tokens":256})
        self.generations.reserve(arm, kind="probe", sequences=1, max_generated_tokens=256,
                                 timeout_seconds=70, max_billed_units=0)
        self.generations.mark_dispatched(arm)
        with self._transaction():
            self.db.execute("UPDATE attempts SET generation_state='dispatched' WHERE attempt_id=?",(attempt_id,))
            self._event("generation_dispatch_granted", {"arm":arm,"attempt_id":attempt_id})
        return {"approved": True,"kind":"generation","arm":arm,"attempt_id":attempt_id,
                "code_version_sha256": row["code_sha256"], "max_tokens":256,**self.deadline()}

    @staticmethod
    def cleanup_complete(cleanup):
        return (all(cleanup.get(key) is True for key in
                   ("worker_terminated", "chrome_exit_confirmed", "local_harness_exit_confirmed", "controller_exit_verified"))
                and not cleanup.get("retained_owned_pids_alive", [])
                and cleanup.get("chrome_os_exit_verified") is not False)

    def finish_attempt(self, attempt_id, *, outcome, failure_class=None, evidence=None,
                       cleanup=None, generated_tokens=None, generation_seconds=None):
        """Record every terminal attempt. Only confirmed safe load failures retry."""
        if outcome not in {"completed", "load_failed", "failed", "unknown"}:
            raise ValueError("explicit terminal outcome required")
        evidence, cleanup = dict(evidence or {}), dict(cleanup or {})
        rows = {x["attempt_id"]:x for x in self.attempts()}
        row = rows.get(attempt_id)
        if row is None or row["status"] != "loading":
            raise ValueError("attempt is missing or terminal; never overwrite it")
        dispatched = row["generation_state"] != "not_dispatched"
        clean = self.cleanup_complete(cleanup)
        success = outcome == "completed" and dispatched and clean and cleanup.get("unload_confirmed") is True
        if success and self.remaining_seconds() <= 0:
            success = False
            failure_class = "deadline_expired"
        if success and (generation_seconds is None or generation_seconds > 70):
            success = False
            failure_class = "generation_time_limit_or_unknown_usage"
        if success and (type(generated_tokens) is not int or generated_tokens < 0 or generated_tokens > 256):
            success = False
            failure_class = "token_limit_or_unknown_usage"
        if outcome == "completed" and not success:
            outcome = "failed"
            failure_class = failure_class or "cleanup_or_generation_evidence_unconfirmed"
        retry = (outcome == "load_failed" and failure_class in RETRYABLE and not dispatched and clean
                 and evidence.get("confirmed_pre_generation") is True
                 and evidence.get("generation_intent_absent") is True
                 and evidence.get("generation_ack_absent") is True
                 and evidence.get("unknown_execution_state") is False
                 and evidence.get("oom_or_device_loss") is False
                 and evidence.get("resource_stop") is False
                 and bool(evidence.get("diagnosis")) and self.remaining_seconds() > 0)
        result = {"outcome":outcome,"failure_class":failure_class,"evidence":evidence,
                  "cleanup":cleanup,"retry_eligible":retry,"generation_dispatched_or_ambiguous":dispatched,
                  "generated_tokens":generated_tokens,"generation_seconds":generation_seconds,
                  "finished_epoch_ms":time.time()*1000,"remaining_seconds":self.remaining_seconds()}
        if dispatched:
            gen_rows = {a["request_id"]:a for a in self.generations.attempts()}
            if row["arm"] not in gen_rows:
                # Interrupted between durable intent and reservation: retain
                # consumed marker; no fabricated zero or replacement dispatch.
                result["generation_budget_reconciliation"] = "intent_without_reservation; full arm slot and 256 tokens conservatively consumed"
            elif gen_rows[row["arm"]]["status"] in {"reserved","dispatched"}:
                self.generations.finish(row["arm"],status="completed" if success else "failed",
                    completed_sequences=int(success),failed_sequences=int(not success),
                    generated_tokens=generated_tokens,elapsed_seconds=generation_seconds,billed_units=0)
        with self._transaction():
            self.db.execute("UPDATE attempts SET status=?,result=? WHERE attempt_id=?",
                            ("completed" if success else outcome,_json(result),attempt_id))
            self._event("attempt_finished", {"attempt_id":attempt_id,**result})
            if not success and not retry:
                self.db.execute("UPDATE state SET stopped=1,stop_reason=? WHERE id=1",(failure_class or outcome,))
                self._event("terminal_stop", {"reason":failure_class or outcome})
        if not success and not retry:
            self.generations.cancel(reason="v2_stop_condition")
        return result

    def terminalize(self, reason, *, executor_stopped=False):
        """Permanently stop without renewing time or refunding ambiguous dispatch."""
        _identifier(reason)
        with self._transaction():
            self.db.execute("UPDATE state SET stopped=1,stop_reason=COALESCE(stop_reason,?) WHERE id=1",(reason,))
            self._event("terminal_stop", {"reason":reason,"executor_stopped":executor_stopped})
        self.generations.recover_interrupted(executor_stopped=executor_stopped)
        self.generations.cancel(reason="v2_terminal_stop")
        return self.summary()

    def validate_intent(self, intent, *, arm, attempt_id, code_sha256):
        expected = self.configs[arm]
        if (intent.get("schema") != "browser-v2-dispatch-intent"
                or intent.get("kind") not in {"load","generation"}
                or intent.get("arm") != arm or intent.get("attempt_id") != attempt_id
                or intent.get("code_version_sha256") != code_sha256
                or intent.get("configuration_sha256") != expected["configurationSha256"]
                or intent.get("artifact_sha256") != expected["convertedArtifactSha256"]
                or intent.get("run_id") != expected["runId"]):
            raise ValueError("worker dispatch intent differs from frozen arm/code identity")
        _identifier(intent.get("request_id"))
        return intent["kind"]

    def handle_intent(self, intent, *, arm, attempt_id, code_sha256):
        """Return an ACK only after its irrevocable durable ledger transition."""
        kind = self.validate_intent(intent,arm=arm,attempt_id=attempt_id,code_sha256=code_sha256)
        if kind == "generation":
            row = self.db.execute("SELECT code_sha256 FROM attempts WHERE attempt_id=?", (attempt_id,)).fetchone()
            if row is None or row["code_sha256"] != code_sha256:
                raise ValueError("generation code differs from durable load identity")
        result = (self.begin_load(arm,attempt_id,code_sha256) if kind == "load"
                  else self.dispatch_generation(arm,attempt_id))
        return {**result,"request_id":intent["request_id"]}

    def summary(self):
        state, attempts = self._state(), self.attempts()
        consumed = sum(a["generation_state"] != "not_dispatched" for a in attempts)
        gen = self.generations.summary()
        missing = [a["arm"] for a in attempts if a["generation_state"] != "not_dispatched"
                   and a["arm"] not in {r["request_id"] for r in self.generations.attempts()}]
        return {"schema_version":2,"manifest_sha256":self.digest,"limits":LIMITS,
            "engine_load_attempts":len(attempts),
            "additional_load_attempts":sum(max(0,sum(a["arm"]==arm for a in attempts)-1) for arm in ARMS),
            "generation_dispatches_consumed":consumed,
            "generation_slots_remaining":{arm:int(not any(a["arm"]==arm and a["generation_state"]!="not_dispatched" for a in attempts)) for arm in ARMS},
            "conservative_tokens_for_intents_without_reservation":256*len(missing),
            "clock":self.deadline(),"stopped":bool(state["stopped"]),"stop_reason":state["stop_reason"],
            "attempts":attempts,"generation_budget":gen,
            "generation_retries":0,"new_weight_proposals":0,"experimental_frontier_requests":0}

    get_summary = summary

    def close(self):
        self.db.close()
        self.generations.close()

    def __enter__(self):
        return self

    def __exit__(self,*_):
        self.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=["summary"])
    parser.add_argument("--root",required=True)
    args=parser.parse_args()
    with Controller(args.root) as controller:
        print(json.dumps(controller.summary(),indent=2))


if __name__ == "__main__":
    main()
