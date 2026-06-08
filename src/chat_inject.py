"""Mid-stream message injection for the agent loop.

When the agent is mid-stream (mid-tool-call, mid-wait, whatever), the user
may want to slip in extra context or change direction without waiting for
the current turn to finish — terminal-agent style ("oh wait, also do X").

This module keeps a small in-memory queue per session. The HTTP route
appends; the agent loop drains at the start of each round, turning each
queued string into a user-role message in the live context. Ephemeral: no
disk persistence, so the queue clears on server restart (acceptable — at
restart there's no live stream to inject into anyway).

Thread-safety: a single asyncio lock guards mutations. agent_loop and the
inject route both run on the same uvicorn event loop, so cross-thread
coordination isn't needed.
"""
from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from typing import Dict, List


_QUEUE: Dict[str, List[Dict]] = defaultdict(list)
_LOCK = asyncio.Lock()
_MAX_PER_SESSION = 20
_MAX_MSG_CHARS = 4000


async def push(session_id: str, message: str) -> Dict:
    """Queue a message for the next round of session_id's agent loop.

    Returns the queued record so the caller can echo it back. Trims to
    MAX_MSG_CHARS and caps queue depth so a flood can't blow up memory.
    """
    if not session_id or not message:
        return {}
    text = (message or "").strip()[:_MAX_MSG_CHARS]
    if not text:
        return {}
    rec = {"text": text, "ts": time.time()}
    async with _LOCK:
        q = _QUEUE[session_id]
        q.append(rec)
        if len(q) > _MAX_PER_SESSION:
            del q[: len(q) - _MAX_PER_SESSION]
    return rec


async def drain(session_id: str) -> List[Dict]:
    """Atomically pop all queued messages for session_id."""
    if not session_id:
        return []
    async with _LOCK:
        msgs = list(_QUEUE.get(session_id) or [])
        _QUEUE[session_id] = []
    return msgs


async def peek(session_id: str) -> List[Dict]:
    """Snapshot of queued messages without draining (for UI polling)."""
    if not session_id:
        return []
    async with _LOCK:
        return list(_QUEUE.get(session_id) or [])
