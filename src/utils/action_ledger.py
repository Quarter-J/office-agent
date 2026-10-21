"""Durable, atomic claims for task steps and shared write resources."""

from contextlib import contextmanager, closing
import json
from pathlib import Path
import sqlite3
import time

from src.config.settings import settings


class ActionLedger:
    def __init__(self, path=None):
        self.path = str(path or settings.action_ledger_path)

    @contextmanager
    def connect(self):
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path, timeout=2)) as db, db:
            db.row_factory = sqlite3.Row
            db.executescript("""
                CREATE TABLE IF NOT EXISTS actions (
                    task_id TEXT NOT NULL, step_id TEXT NOT NULL,
                    fingerprint TEXT NOT NULL, proposal TEXT NOT NULL,
                    owner TEXT NOT NULL, status TEXT NOT NULL,
                    result TEXT, resource TEXT, updated REAL NOT NULL,
                    PRIMARY KEY (task_id, step_id)
                );
                CREATE TABLE IF NOT EXISTS resource_claims (
                    resource TEXT PRIMARY KEY, task_id TEXT NOT NULL,
                    step_id TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS conflicts (
                    id INTEGER PRIMARY KEY, task_id TEXT NOT NULL,
                    kind TEXT NOT NULL, actor TEXT NOT NULL,
                    proposed TEXT NOT NULL, accepted TEXT NOT NULL,
                    reason TEXT NOT NULL, resolution TEXT NOT NULL,
                    created REAL NOT NULL
                );
            """)
            yield db

    def claim(self, task_id, step_id, fingerprint, proposal, owner, resource=None):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM actions WHERE task_id=? AND step_id=?",
                             (task_id, step_id)).fetchone()
            if row:
                return dict(row)
            resources = [resource] if isinstance(resource, str) else (resource or [])
            for item in resources:
                held = db.execute("SELECT * FROM resource_claims WHERE resource=?", (item,)).fetchone()
                if held:
                    return {"status": "resource_busy", "proposal": json.dumps(dict(held), ensure_ascii=False)}
            for item in resources:
                db.execute("INSERT INTO resource_claims VALUES (?, ?, ?)", (item, task_id, step_id))
            db.execute("INSERT INTO actions VALUES (?, ?, ?, ?, ?, 'running', NULL, ?, ?)",
                       (task_id, step_id, fingerprint, json.dumps(proposal, ensure_ascii=False),
                        owner, json.dumps(resources), time.time()))
            return None

    def finish(self, task_id, step_id, owner, status, result):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            changed = db.execute("UPDATE actions SET status=?, result=?, updated=? "
                                 "WHERE task_id=? AND step_id=? AND owner=? AND status='running'",
                                 (status, json.dumps(result, ensure_ascii=False), time.time(),
                                  task_id, step_id, owner)).rowcount
            if changed and status != "unknown":
                db.execute("DELETE FROM resource_claims WHERE task_id=? AND step_id=?", (task_id, step_id))

    def interrupt(self, task_id, owner):
        with self.connect() as db:
            db.execute("UPDATE actions SET status='unknown', updated=? "
                       "WHERE task_id=? AND owner=? AND status='running'", (time.time(), task_id, owner))

    def resolve(self, task_id, step_id, *, expected_updated, success, evidence, result):
        """Operator API after verification; never invoke automatically from a model.

        Only an unknown, unchanged record may be resolved. A failed receipt is
        still cached: another send requires a separate explicitly requested task.
        """
        if not evidence.strip() or not isinstance(result, dict) or type(success) is not bool:
            raise ValueError("必须提供核实依据、布尔结果和回执")
        status = "success" if success else "failed"
        receipt = {**result, "success": success, "status": status, "verification": evidence}
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            changed = db.execute("UPDATE actions SET status=?, result=?, updated=? "
                                 "WHERE task_id=? AND step_id=? AND status='unknown' AND updated=?",
                                 (status, json.dumps(receipt, ensure_ascii=False), time.time(),
                                  task_id, step_id, expected_updated)).rowcount
            if changed:
                db.execute("DELETE FROM resource_claims WHERE task_id=? AND step_id=?", (task_id, step_id))
            return bool(changed)

    def record_conflict(self, task_id, kind, actor, proposed, accepted, reason, resolution="blocked"):
        record = dict(kind=kind, actor=actor, proposed=proposed, accepted=accepted,
                      reason=reason, resolution=resolution)
        with self.connect() as db:
            db.execute("INSERT INTO conflicts (task_id, kind, actor, proposed, accepted, reason, resolution, created) "
                       "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                       (task_id, kind, actor, json.dumps(proposed, ensure_ascii=False),
                        json.dumps(accepted, ensure_ascii=False), reason, resolution, time.time()))
        return record

    def conflicts(self, task_id):
        with self.connect() as db:
            rows = db.execute("SELECT * FROM conflicts WHERE task_id=? ORDER BY id", (task_id,)).fetchall()
        return [{**dict(row), "proposed": json.loads(row["proposed"]),
                 "accepted": json.loads(row["accepted"])} for row in rows]

    def actions(self, task_id):
        with self.connect() as db:
            return [dict(row) for row in db.execute("SELECT * FROM actions WHERE task_id=?", (task_id,))]
