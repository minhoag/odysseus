"""Live background-task progress publisher.

When the agent kicks off a download_model/serve_model, the work runs on a
remote machine and the chat stream goes silent until the cookbook task
reaches a terminal status (could be 20+ minutes for a large download). The
user is left watching a "Resumed agent stream live" chip with no actual
activity — the "gap between background and front" problem.

This module fixes that by spawning an asyncio task that polls the cookbook
status endpoint on a short cadence and pushes `bg_task_progress` SSE events
into the same `agent_runs._Run` buffer the chat client is subscribed to.
The publisher self-terminates when:
  - the cookbook task reaches a terminal status (the wake will fire next),
  - the wake-run has already started for this session (status=running),
  - a hard max-duration is reached (safety net so a stuck task doesn't
    leak a polling task forever).

The publisher is fire-and-forget — agent_loop spawns it via asyncio.create_task
and moves on. It deduplicates by session+cookbook_session_id so a rapid
double-launch can't spawn two pollers for the same task.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Active publishers, keyed by (chat_session_id, cookbook_session_id) so a
# duplicate spawn is a no-op rather than a parallel poller.
_ACTIVE: Dict[tuple, asyncio.Task] = {}

# Cookbook task statuses that mean "no more progress, wake will fire."
_TERMINAL = {"completed", "done", "ready", "error", "failed", "stopped",
             "crashed", "killed", "cancelled"}

# Poll cadence (s). Slow enough to not hammer the cookbook endpoint; fast
# enough that the user sees meaningful updates on a long download.
_POLL_S = 8
# Hard cap on how long a single publisher can live, in case the task is
# stuck pre-terminal (e.g. tmux hung). 30 min = ample for a big download.
_MAX_LIFE_S = 30 * 60


def _make_progress_event(task: dict, cookbook_session_id: str) -> str:
    """Build the SSE `data:` line for a single progress snapshot."""
    tail = task.get("output_tail") or ""
    # Only the last 4 lines of pane output — enough to convey progress
    # (e.g. an hf download bar) without flooding the chat with traceback noise.
    lines = [ln for ln in str(tail).splitlines() if ln.strip()]
    tail_short = "\n".join(lines[-4:])[-600:]
    payload = {
        "type": "bg_task_progress",
        "session": cookbook_session_id,
        "tool": task.get("type") or "",
        "status": task.get("status") or "",
        "model": task.get("name") or task.get("payload", {}).get("repo_id") or "",
        "tail": tail_short,
        "host": task.get("remoteHost") or task.get("host") or "",
        "ts": time.time(),
    }
    return f"data: {json.dumps(payload)}\n\n"


async def _poll_and_publish(chat_session_id: str, cookbook_session_id: str) -> None:
    from src import agent_runs
    from src.bg_monitor import _fetch_cookbook_task_status
    started = time.time()
    last_status = ""
    last_tail = ""
    last_emit_ts = 0.0
    # Emit an initial "starting" event immediately so the user sees the
    # publisher pick up the work even if the cookbook task finishes
    # before the first poll cadence elapses.
    try:
        run = agent_runs._RUNS.get(chat_session_id)
        if run is not None:
            ev0 = (
                f"data: " + json.dumps({
                    "type": "bg_task_progress",
                    "session": cookbook_session_id,
                    "status": "starting",
                    "tail": "",
                    "ts": time.time(),
                }) + "\n\n"
            )
            agent_runs._publish(run, ev0)
    except Exception as e:
        logger.warning(f"[bg-progress] initial emit failed: {e!r}")
    # First real poll on a short delay (~1.5 s) so we catch a fast-finishing
    # task quickly without DDoSing the cookbook endpoint.
    first = True
    while time.time() - started < _MAX_LIFE_S:
        try:
            await asyncio.sleep(1.5 if first else _POLL_S)
            first = False
            # If a wake-run has started for this session, stop publishing —
            # the wake-run is now the source of truth for live events.
            if agent_runs.is_active(chat_session_id):
                logger.info(
                    "[bg-progress] wake-run took over for %s — publisher exit",
                    cookbook_session_id,
                )
                return
            try:
                tasks = await _fetch_cookbook_task_status()
            except Exception as e:
                logger.warning(f"[bg-progress] status fetch failed: {e!r}")
                continue
            task = next(
                (t for t in (tasks or [])
                 if isinstance(t, dict) and t.get("session_id") == cookbook_session_id),
                None,
            )
            if not task:
                continue
            status = (task.get("status") or "").lower()
            tail = task.get("output_tail") or ""
            now_t = time.time()
            # Throttle re-emissions: only push when status changes OR tail
            # changes OR 30 s since last emit (so a busy-but-quiet task still
            # registers a heartbeat in the chat).
            unchanged = status == last_status and tail == last_tail
            if unchanged and (now_t - last_emit_ts) < 30:
                if status in _TERMINAL:
                    break
                continue
            run = agent_runs._RUNS.get(chat_session_id)
            if run is None:
                # Original run got evicted (long gap). Stop polling — nothing
                # to push to. The cookbook continuation in bg_monitor will
                # still fire when the task reaches terminal status.
                logger.info("[bg-progress] no run for %s; publisher exit",
                            chat_session_id)
                return
            ev_str = _make_progress_event(task, cookbook_session_id)
            try:
                agent_runs._publish(run, ev_str)
            except Exception as e:
                logger.warning(f"[bg-progress] publish failed: {e!r}")
            last_status = status
            last_tail = tail
            last_emit_ts = now_t
            if status in _TERMINAL:
                logger.info(
                    "[bg-progress] task %s reached terminal=%s — publisher exit",
                    cookbook_session_id, status,
                )
                return
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(f"[bg-progress] unexpected: {e!r}")
    logger.info("[bg-progress] max-life timeout for %s", cookbook_session_id)


def start_publisher(chat_session_id: str, cookbook_session_id: str) -> None:
    """Spawn (or no-op) a progress publisher for this (chat, cookbook) pair."""
    if not chat_session_id or not cookbook_session_id:
        return
    key = (chat_session_id, cookbook_session_id)
    existing = _ACTIVE.get(key)
    if existing and not existing.done():
        return
    task = asyncio.create_task(_poll_and_publish(chat_session_id, cookbook_session_id))
    _ACTIVE[key] = task

    def _cleanup(_t: asyncio.Task) -> None:
        cur = _ACTIVE.get(key)
        if cur is _t:
            _ACTIVE.pop(key, None)
    task.add_done_callback(_cleanup)
