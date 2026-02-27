"""
SQLite-backed store for reinforcement learning hyperparameters.

Hyperparameters defined as constants in :class:`~eth_algo_trading.config.RLConfig`
are used as defaults.  This module lets them be read back and **updated at
runtime** via a database connection so that a running agent picks up new
values without a restart.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from typing import Any, Dict, List, Optional


class HyperparamDB:
    """
    Persistent key-value store for RL hyperparameters backed by SQLite.

    Thread-safe: all database operations are serialised with a
    :class:`threading.Lock`, and each thread obtains its own connection via
    :class:`threading.local`.

    Parameters
    ----------
    db_path:
        File path for the SQLite database, or ``':memory:'`` for an
        in-process ephemeral store (useful for tests).
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        self.db_path = db_path
        # RLock (reentrant) allows the same thread to re-acquire the lock,
        # which is necessary because _init_schema() → _get_conn() → lock.
        self._lock = threading.RLock()
        self._local = threading.local()
        # Track every thread-local connection so close_all() can clean up.
        self._all_conns: List[sqlite3.Connection] = []
        # Unique name used to build a per-instance shared-cache URI when
        # db_path is ':memory:', preventing collisions between instances.
        self._instance_id: str = uuid.uuid4().hex
        self._init_schema()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _get_conn(self) -> sqlite3.Connection:
        """Return a thread-local SQLite connection, creating one if needed.

        When ``db_path`` is ``':memory:'``, a per-instance shared-cache URI
        (``file:hyperparam_<id>?mode=memory&cache=shared``) is used so that
        all threads accessing the **same** ``HyperparamDB`` instance share one
        in-memory database.  ``check_same_thread`` is disabled because the
        :class:`threading.RLock` already serialises all operations and
        :meth:`close_all` must be able to close connections created in other
        threads.
        """
        if not hasattr(self._local, "conn"):
            if self.db_path == ":memory:":
                uri = f"file:hyperparam_{self._instance_id}?mode=memory&cache=shared"
                conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
            else:
                conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._local.conn = conn
            with self._lock:
                self._all_conns.append(conn)
        return self._local.conn

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        with self._lock:
            conn = self._get_conn()
            with conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS hyperparams (
                        key        TEXT PRIMARY KEY,
                        value      TEXT NOT NULL,
                        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def set(self, key: str, value: Any) -> None:
        """Persist *value* (JSON-serialisable) under *key*."""
        with self._lock:
            conn = self._get_conn()
            with conn:
                conn.execute(
                    """
                    INSERT INTO hyperparams (key, value, updated_at)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(key) DO UPDATE
                        SET value      = excluded.value,
                            updated_at = CURRENT_TIMESTAMP
                    """,
                    (key, json.dumps(value)),
                )

    def get(self, key: str, default: Any = None) -> Any:
        """Return the stored value for *key*, or *default* if not found."""
        with self._lock:
            row = self._get_conn().execute(
                "SELECT value FROM hyperparams WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            return default
        return json.loads(row[0])

    def load_all(self) -> Dict[str, Any]:
        """Return all stored hyperparameters as a ``{key: value}`` dict."""
        with self._lock:
            rows = self._get_conn().execute(
                "SELECT key, value FROM hyperparams"
            ).fetchall()
        return {k: json.loads(v) for k, v in rows}

    def delete(self, key: str) -> None:
        """Remove *key* from the store (no-op if it doesn't exist)."""
        with self._lock:
            conn = self._get_conn()
            with conn:
                conn.execute("DELETE FROM hyperparams WHERE key = ?", (key,))

    # ------------------------------------------------------------------
    # Model checkpoint helpers
    # ------------------------------------------------------------------

    def save_blob(self, key: str, data: bytes) -> None:
        """Store raw *bytes* (e.g. a serialised PyTorch state-dict)."""
        with self._lock:
            conn = self._get_conn()
            with conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS blobs (
                        key        TEXT PRIMARY KEY,
                        data       BLOB NOT NULL,
                        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
                conn.execute(
                    """
                    INSERT INTO blobs (key, data, updated_at)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(key) DO UPDATE
                        SET data       = excluded.data,
                            updated_at = CURRENT_TIMESTAMP
                    """,
                    (key, data),
                )

    def load_blob(self, key: str) -> Optional[bytes]:
        """Return the raw bytes stored under *key*, or ``None``."""
        with self._lock:
            try:
                row = self._get_conn().execute(
                    "SELECT data FROM blobs WHERE key = ?", (key,)
                ).fetchone()
            except sqlite3.OperationalError:
                return None
        if row is None:
            return None
        return bytes(row[0])

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the thread-local database connection, if open."""
        with self._lock:
            conn = getattr(self._local, "conn", None)
            if conn is not None:
                # Remove from global connection tracking to avoid retaining
                # references to closed connections in long-running processes.
                try:
                    self._all_conns.remove(conn)
                except ValueError:
                    # If the connection is not in the list, ignore.
                    pass
                conn.close()
                del self._local.conn

    def close_all(self) -> None:
        """
        Close all thread-local connections opened by any thread.

        Call this during application shutdown to ensure no connections leak.
        """
        with self._lock:
            for conn in self._all_conns:
                try:
                    conn.close()
                except Exception:
                    pass
            self._all_conns.clear()
        # Also clear the calling thread's local reference.
        if hasattr(self._local, "conn"):
            del self._local.conn

    def __enter__(self) -> "HyperparamDB":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
