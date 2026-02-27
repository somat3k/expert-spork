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
from typing import Any, Dict, Optional


class HyperparamDB:
    """
    Persistent key-value store for RL hyperparameters backed by SQLite.

    Parameters
    ----------
    db_path:
        File path for the SQLite database, or ``':memory:'`` for an
        in-process ephemeral store (useful for tests).
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._init_schema()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.execute(
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
        with self._conn:
            self._conn.execute(
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
        row = self._conn.execute(
            "SELECT value FROM hyperparams WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return default
        return json.loads(row[0])

    def load_all(self) -> Dict[str, Any]:
        """Return all stored hyperparameters as a ``{key: value}`` dict."""
        rows = self._conn.execute(
            "SELECT key, value FROM hyperparams"
        ).fetchall()
        return {k: json.loads(v) for k, v in rows}

    def delete(self, key: str) -> None:
        """Remove *key* from the store (no-op if it doesn't exist)."""
        with self._conn:
            self._conn.execute("DELETE FROM hyperparams WHERE key = ?", (key,))

    # ------------------------------------------------------------------
    # Model checkpoint helpers
    # ------------------------------------------------------------------

    def save_blob(self, key: str, data: bytes) -> None:
        """Store raw *bytes* (e.g. a serialised PyTorch state-dict)."""
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS blobs (
                    key        TEXT PRIMARY KEY,
                    data       BLOB NOT NULL,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            self._conn.execute(
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
        try:
            row = self._conn.execute(
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
        """Close the database connection."""
        self._conn.close()

    def __enter__(self) -> "HyperparamDB":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
