"""Datastore abstraction for the SafeEdge backend.

Two implementations behind one interface:
  • TablestoreStore  — Alibaba Cloud Tablestore (serverless NoSQL), used in prod.
  • InMemoryStore    — process-local dict store, used for local dev / tests.

Selection is automatic: if TABLESTORE_ENDPOINT is set we use Tablestore,
otherwise we fall back to in-memory. This lets us build and test the whole
backend locally with zero Alibaba dependency, then deploy unchanged.
"""
from __future__ import annotations

import logging
import os
import time
import uuid
from typing import Optional, Protocol

logger = logging.getLogger(__name__)


class Store(Protocol):
    def add_event(self, event: dict) -> str: ...
    def recent_events(self, limit: int = 100) -> list[dict]: ...
    def add_incident(self, incident: dict) -> str: ...
    def recent_incidents(self, limit: int = 50) -> list[dict]: ...
    def add_rho_sample(self, sample: dict) -> None: ...
    def set_forecast(self, forecast: dict) -> None: ...
    def get_forecast(self) -> Optional[dict]: ...
    def set_latest_state(self, state: dict) -> None: ...
    def get_latest_state(self) -> Optional[dict]: ...
    def set_kv(self, key: str, value: dict) -> None: ...
    def get_kv(self, key: str) -> Optional[dict]: ...


# ── In-memory (local dev) ─────────────────────────────────────────────────────

class InMemoryStore:
    def __init__(self, cap: int = 5000):
        self._events: list[dict] = []
        self._incidents: list[dict] = []
        self._rho: list[dict] = []
        self._forecast: Optional[dict] = None
        self._latest_state: Optional[dict] = None
        self._kv: dict = {}
        self._cap = cap
        logger.info("Store: in-memory (local dev mode)")

    def add_event(self, event: dict) -> str:
        eid = event.setdefault("id", uuid.uuid4().hex)
        self._events.append(event)
        self._events = self._events[-self._cap:]
        return eid

    def recent_events(self, limit: int = 100) -> list[dict]:
        return self._events[-limit:]

    def add_incident(self, incident: dict) -> str:
        iid = incident.setdefault("id", uuid.uuid4().hex)
        self._incidents.append(incident)
        self._incidents = self._incidents[-self._cap:]
        return iid

    def recent_incidents(self, limit: int = 50) -> list[dict]:
        return list(reversed(self._incidents[-limit:]))

    def add_rho_sample(self, sample: dict) -> None:
        self._rho.append(sample)
        self._rho = self._rho[-self._cap:]

    def set_forecast(self, forecast: dict) -> None:
        self._forecast = forecast

    def get_forecast(self) -> Optional[dict]:
        return self._forecast

    def set_latest_state(self, state: dict) -> None:
        self._latest_state = state

    def get_latest_state(self) -> Optional[dict]:
        return self._latest_state

    def set_kv(self, key: str, value: dict) -> None:
        self._kv[key] = value

    def get_kv(self, key: str) -> Optional[dict]:
        return self._kv.get(key)


# ── Tablestore (Alibaba Cloud, prod) ──────────────────────────────────────────

class TablestoreStore:
    """Thin wrapper over the Alibaba Cloud Tablestore SDK.

    Tables (auto-created on first run):
      safeedge_events    PK: (pk='ev', ts:INT desc)   cols: payload(JSON str)
      safeedge_incidents PK: (pk='in', ts:INT desc)   cols: payload(JSON str)
      safeedge_rho       PK: (pk='rho', ts:INT desc)  cols: payload(JSON str)
      safeedge_kv        PK: (key:STRING)             cols: value(JSON str)
    """

    def __init__(self):
        import json
        from tablestore import OTSClient
        self._json = json
        endpoint   = os.environ["TABLESTORE_ENDPOINT"]
        instance   = os.environ["TABLESTORE_INSTANCE"]
        access_id  = os.environ["ALIBABA_CLOUD_ACCESS_KEY_ID"]
        access_key = os.environ["ALIBABA_CLOUD_ACCESS_KEY_SECRET"]
        self._client = OTSClient(endpoint, access_id, access_key, instance)
        self._ensure_tables()
        logger.info("Store: Tablestore (instance=%s)", instance)

    # -- table bootstrap --
    def _ensure_tables(self) -> None:
        from tablestore import (TableMeta, TableOptions, ReservedThroughput,
                                 CapacityUnit)
        specs = {
            "safeedge_events":    [("pk", "STRING"), ("ts", "INTEGER")],
            "safeedge_incidents": [("pk", "STRING"), ("ts", "INTEGER")],
            "safeedge_rho":       [("pk", "STRING"), ("ts", "INTEGER")],
            "safeedge_kv":        [("key", "STRING")],
        }
        try:
            existing = set(self._client.list_table())
        except Exception as exc:  # noqa: BLE001
            logger.warning("list_table failed: %s", exc)
            existing = set()
        for name, pk in specs.items():
            if name in existing:
                continue
            meta = TableMeta(name, pk)
            opts = TableOptions(time_to_live=-1, max_version=1)
            self._client.create_table(
                meta, opts, ReservedThroughput(CapacityUnit(0, 0))
            )
            logger.info("Tablestore: created table %s", name)

    def _put(self, table: str, pk: list, payload: dict) -> None:
        from tablestore import Row
        row = Row(pk, [("payload", self._json.dumps(payload))])
        self._client.put_row(table, row)

    def _range(self, table: str, limit: int) -> list[dict]:
        from tablestore import Direction, INF_MIN, INF_MAX
        # get_range returns (consumed, next_start_pk, row_list, next_token)
        _, _, rows, _ = self._client.get_range(
            table, Direction.BACKWARD,
            [("pk", "ev" if "events" in table else "in" if "incidents" in table else "rho"),
             ("ts", INF_MAX)],
            [("pk", "ev" if "events" in table else "in" if "incidents" in table else "rho"),
             ("ts", INF_MIN)],
            limit=limit,
        )
        out = []
        for r in rows or []:
            cols = {c[0]: c[1] for c in r.attribute_columns}
            if "payload" in cols:
                out.append(self._json.loads(cols["payload"]))
        return out

    def add_event(self, event: dict) -> str:
        eid = event.setdefault("id", uuid.uuid4().hex)
        self._put("safeedge_events",
                  [("pk", "ev"), ("ts", int(event["timestamp"] * 1000))], event)
        return eid

    def recent_events(self, limit: int = 100) -> list[dict]:
        return list(reversed(self._range("safeedge_events", limit)))

    def add_incident(self, incident: dict) -> str:
        iid = incident.setdefault("id", uuid.uuid4().hex)
        self._put("safeedge_incidents",
                  [("pk", "in"), ("ts", int(incident["timestamp"] * 1000))], incident)
        return iid

    def recent_incidents(self, limit: int = 50) -> list[dict]:
        return self._range("safeedge_incidents", limit)

    def add_rho_sample(self, sample: dict) -> None:
        self._put("safeedge_rho",
                  [("pk", "rho"), ("ts", int(sample.get("timestamp", time.time()) * 1000))],
                  sample)

    def _kv_set(self, key: str, value: dict) -> None:
        from tablestore import Row
        row = Row([("key", key)], [("value", self._json.dumps(value))])
        self._client.put_row("safeedge_kv", row)

    def _kv_get(self, key: str) -> Optional[dict]:
        try:
            _, row, _ = self._client.get_row("safeedge_kv", [("key", key)])
            if row is None:
                return None
            cols = {c[0]: c[1] for c in row.attribute_columns}
            return self._json.loads(cols["value"]) if "value" in cols else None
        except Exception as exc:  # noqa: BLE001
            logger.warning("kv_get(%s) failed: %s", key, exc)
            return None

    def set_forecast(self, forecast: dict) -> None:
        self._kv_set("forecast", forecast)

    def get_forecast(self) -> Optional[dict]:
        return self._kv_get("forecast")

    def set_latest_state(self, state: dict) -> None:
        self._kv_set("latest_state", state)

    def get_latest_state(self) -> Optional[dict]:
        return self._kv_get("latest_state")

    def set_kv(self, key: str, value: dict) -> None:
        self._kv_set(key, value)

    def get_kv(self, key: str) -> Optional[dict]:
        return self._kv_get(key)


# ── factory ───────────────────────────────────────────────────────────────────

_store: Optional[Store] = None


def get_store() -> Store:
    global _store
    if _store is not None:
        return _store
    if os.environ.get("TABLESTORE_ENDPOINT"):
        try:
            _store = TablestoreStore()
            return _store
        except Exception as exc:  # noqa: BLE001
            logger.error("Tablestore init failed (%s) — falling back to in-memory", exc)
    _store = InMemoryStore()
    return _store
