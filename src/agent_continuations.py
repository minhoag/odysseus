"""Hidden agent continuation records.

These are not user-visible ScheduledTask rows. They let the app remember that
an agent turn is waiting on an external subsystem such as Cookbook, then resume
the same chat when that subsystem reaches a terminal state.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

from core.atomic_io import atomic_write_json
from src.constants import DATA_DIR

import json

_STORE = Path(DATA_DIR) / "agent_continuations.json"


def _load() -> Dict[str, Dict[str, Any]]:
    try:
        if _STORE.exists():
            data = json.loads(_STORE.read_text(encoding="utf-8")) or {}
            return data if isinstance(data, dict) else {}
    except Exception:
        pass
    return {}


def _save(rows: Dict[str, Dict[str, Any]]) -> None:
    atomic_write_json(str(_STORE), rows, indent=2)


def add_cookbook_wait(
    *,
    session_id: str,
    owner: str,
    cookbook_session_id: str,
    cookbook_type: str,
    next_hint: str = "",
) -> Dict[str, Any]:
    rows = _load()
    now = time.time()
    for rec in rows.values():
        if (
            rec.get("status") == "waiting"
            and rec.get("kind") == "cookbook"
            and rec.get("session_id") == session_id
            and rec.get("cookbook_session_id") == cookbook_session_id
        ):
            rec["updated_at"] = now
            rec["next_hint"] = next_hint or rec.get("next_hint", "")
            _save(rows)
            return rec
    cid = uuid.uuid4().hex[:12]
    rec = {
        "id": cid,
        "kind": "cookbook",
        "status": "waiting",
        "session_id": session_id,
        "owner": owner,
        "cookbook_session_id": cookbook_session_id,
        "cookbook_type": cookbook_type,
        "next_hint": next_hint,
        "created_at": now,
        "updated_at": now,
        "followed_up": False,
    }
    rows[cid] = rec
    _save(rows)
    return rec


def add_timer_wait(
    *,
    session_id: str,
    owner: str,
    delay_seconds: int,
    next_hint: str = "",
    checklist: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    rows = _load()
    now = time.time()
    due_at = now + max(1, int(delay_seconds or 1))
    for rec in rows.values():
        if (
            rec.get("status") == "waiting"
            and rec.get("kind") == "timer"
            and rec.get("session_id") == session_id
            and not rec.get("followed_up")
        ):
            # If the existing rec's due_at is already in the PAST (its wake
            # is in-flight but hasn't marked followed_up yet), reusing it
            # would either fire the new wait immediately (min picked the
            # past value) OR silently swallow the new wait (we wouldn't
            # extend it). Either way wrong. Leave the in-flight rec alone
            # and create a fresh one so the new agent_wait actually waits.
            existing_due = float(rec.get("due_at") or 0)
            if existing_due <= now:
                break  # fall through to create a fresh rec
            rec["updated_at"] = now
            # New due_at always wins when reusing — the agent explicitly
            # asked for a new duration; don't second-guess by min()-ing
            # against an older (likely stale) value.
            rec["due_at"] = due_at
            rec["next_hint"] = next_hint or rec.get("next_hint", "")
            if checklist:
                rec["checklist"] = checklist
            _save(rows)
            return rec
    cid = uuid.uuid4().hex[:12]
    rec = {
        "id": cid,
        "kind": "timer",
        "status": "waiting",
        "session_id": session_id,
        "owner": owner,
        "due_at": due_at,
        "next_hint": next_hint,
        "checklist": checklist or [],
        "created_at": now,
        "updated_at": now,
        "followed_up": False,
    }
    rows[cid] = rec
    _save(rows)
    return rec


def pending_cookbook() -> List[Dict[str, Any]]:
    rows = _load()
    return [
        dict(rec)
        for rec in rows.values()
        if rec.get("kind") == "cookbook"
        and rec.get("status") == "waiting"
        and not rec.get("followed_up")
    ]


def due_timers() -> List[Dict[str, Any]]:
    now = time.time()
    rows = _load()
    return [
        dict(rec)
        for rec in rows.values()
        if rec.get("kind") == "timer"
        and rec.get("status") == "waiting"
        and not rec.get("followed_up")
        and float(rec.get("due_at") or 0) <= now
    ]


def mark_followed_up(cont_id: str) -> None:
    rows = _load()
    rec = rows.get(cont_id)
    if rec:
        rec["followed_up"] = True
        rec["status"] = "done"
        rec["updated_at"] = time.time()
        _save(rows)


def fire_now_for_session(session_id: str) -> int:
    """Push all waiting continuations for this session to due_at = now.

    Called by the "Continue now" button on the countdown chip — the user
    doesn't want to wait the remaining N minutes. Returns the count of
    recs fast-tracked. bg_monitor picks them up on its next poll (~2 s).
    """
    if not session_id:
        return 0
    rows = _load()
    n = 0
    now = time.time()
    for rec in rows.values():
        if (rec.get("session_id") == session_id
                and rec.get("status") == "waiting"
                and not rec.get("followed_up")):
            rec["due_at"] = now
            rec["updated_at"] = now
            n += 1
    if n:
        _save(rows)
    return n


def cancel_for_session(session_id: str) -> int:
    """Cancel every waiting continuation for this session.

    Called when a new user message arrives on the session — the user's new
    intent supersedes any pending wake. Without this, a bg_monitor wake for
    an old download/timer fires concurrently with the chat reply and the
    two get tangled in the same SSE stream ("Yo Felix!" followed by
    download-status prose). Returns the count of recs cancelled.
    """
    if not session_id:
        return 0
    rows = _load()
    n = 0
    now = time.time()
    for rec in rows.values():
        if (rec.get("session_id") == session_id
                and rec.get("status") == "waiting"
                and not rec.get("followed_up")):
            rec["status"] = "cancelled"
            rec["followed_up"] = True
            rec["updated_at"] = now
            n += 1
    if n:
        _save(rows)
    return n
