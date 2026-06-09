"""google_calendar_sync.py — Pull calendars + events from Google Calendar v3 REST.

Upserts into CalendarCal (source="google") and CalendarEvent (origin="google").
Uses httpx.AsyncClient directly against the Google Calendar REST API.
"""

import hashlib
import logging
from datetime import datetime, timedelta, timezone

import httpx

from core.database import SessionLocal, CalendarCal, CalendarEvent
from src.google_token_service import get_access_token, TokenLoadError

logger = logging.getLogger(__name__)

CALENDAR_LIST_URL = "https://www.googleapis.com/calendar/v3/users/me/calendarList"
EVENTS_URL = "https://www.googleapis.com/calendar/v3/calendars/{calendarId}/events"

SYNC_LOOKBACK_DAYS = 90
SYNC_LOOKAHEAD_DAYS = 365


def _calendar_local_id(owner: str, google_calendar_id: str) -> str:
    digest = hashlib.sha256(f"{owner}{google_calendar_id}".encode()).hexdigest()[:24]
    return f"google-{digest}"


def _event_local_uid(google_event_id: str) -> str:
    return f"google-{google_event_id}"


def _parse_google_dt(dt_obj: dict) -> tuple[datetime, bool, bool]:
    """Return (naive_dt, all_day, is_utc) from a Google dateTime object.

    All-day: {"date": "2026-05-15"}
    Timed:   {"dateTime": "2026-05-15T10:00:00Z"} or with offset
    """
    if "date" in dt_obj:
        d = datetime.strptime(dt_obj["date"], "%Y-%m-%d")
        return d, True, False
    raw = dt_obj["dateTime"]
    # Remove trailing Z and treat as UTC.
    if raw.endswith("Z"):
        dt = datetime.fromisoformat(raw[:-1])
        return dt, False, True
    # Has timezone offset like +09:00 or -05:00 — convert to UTC then strip.
    if "+" in raw[10:] or raw.count("-") > 2:
        dt_aware = datetime.fromisoformat(raw)
        dt_utc = dt_aware.astimezone(timezone.utc).replace(tzinfo=None)
        return dt_utc, False, True
    # No tz info — treat as naive local.
    return datetime.fromisoformat(raw), False, False


def _parse_recurrence(recurrence: list) -> str:
    """Join recurrence array into rrule string, stripping RRULE: prefix."""
    if not recurrence:
        return ""
    parts = []
    for item in recurrence:
        s = str(item).strip()
        if s.upper().startswith("RRULE:"):
            s = s[6:]
        if s:
            parts.append(s)
    return ";".join(parts)


def _headers(access_token: str) -> dict:
    return {"Authorization": f"Bearer {access_token}"}


async def _fetch_calendar_list(client: httpx.AsyncClient, access_token: str) -> list:
    calendars = []
    page_token = None
    while True:
        params = {}
        if page_token:
            params["pageToken"] = page_token
        r = await client.get(CALENDAR_LIST_URL, headers=_headers(access_token), params=params)
        r.raise_for_status()
        data = r.json()
        for item in data.get("items", []):
            calendars.append({
                "id": item["id"],
                "summary": item.get("summary") or item.get("summaryOverride") or item["id"],
                "backgroundColor": item.get("backgroundColor") or "#5b8abf",
            })
        page_token = data.get("nextPageToken")
        if not page_token:
            break
    return calendars


async def _fetch_events(
    client: httpx.AsyncClient,
    access_token: str,
    google_calendar_id: str,
    time_min: str,
    time_max: str,
) -> list:
    events = []
    page_token = None
    while True:
        params = {
            "singleEvents": "false",
            "showDeleted": "true",
            "timeMin": time_min,
            "timeMax": time_max,
            "maxResults": "2500",
        }
        if page_token:
            params["pageToken"] = page_token
        url = EVENTS_URL.format(calendarId=google_calendar_id)
        r = await client.get(url, headers=_headers(access_token), params=params)
        r.raise_for_status()
        data = r.json()
        events.extend(data.get("items", []))
        page_token = data.get("nextPageToken")
        if not page_token:
            break
    return events


async def sync_google_calendar(owner: str) -> dict:
    """Pull calendars + events from Google Calendar v3, upsert into local DB.

    Returns: {calendars: int, events: int, deleted: int, errors: list}
    """
    try:
        access_token = get_access_token()
    except TokenLoadError as e:
        logger.warning("Google token unavailable, skipping sync: %s", e)
        return {"calendars": 0, "events": 0, "deleted": 0, "errors": [str(e)]}

    now = datetime.utcnow()
    time_min = (now - timedelta(days=SYNC_LOOKBACK_DAYS)).strftime("%Y-%m-%dT00:00:00Z")
    time_max = (now + timedelta(days=SYNC_LOOKAHEAD_DAYS)).strftime("%Y-%m-%dT23:59:59Z")

    db = SessionLocal()
    result = {"calendars": 0, "events": 0, "deleted": 0, "errors": []}

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            try:
                g_cals = await _fetch_calendar_list(client, access_token)
            except httpx.HTTPStatusError as e:
                msg = f"calendar list fetch failed: {e.response.status_code}"
                logger.warning(msg)
                result["errors"].append(msg)
                return result
            except Exception as e:
                msg = f"calendar list fetch failed: {e}"
                logger.warning(msg)
                result["errors"].append(msg)
                return result

            # Build set of local calendar IDs from this sync so we can prune
            # calendars that Google no longer returns.
            synced_cal_ids: set[str] = set()

            for g_cal in g_cals:
                google_cal_id = g_cal["id"]
                local_cal_id = _calendar_local_id(owner, google_cal_id)
                synced_cal_ids.add(local_cal_id)

                # Upsert CalendarCal
                cal = db.query(CalendarCal).filter(CalendarCal.id == local_cal_id).first()
                if cal:
                    cal.name = g_cal["summary"]
                    cal.color = g_cal["backgroundColor"]
                    cal.account_id = google_cal_id
                    cal.source = "google"
                    # Preserve existing sync_enabled — don't reset on re-sync.
                else:
                    cal = CalendarCal(
                        id=local_cal_id,
                        owner=owner,
                        name=g_cal["summary"],
                        color=g_cal["backgroundColor"],
                        source="google",
                        account_id=google_cal_id,
                        sync_enabled=True,
                    )
                    db.add(cal)
                result["calendars"] += 1

                # Only fetch events when sync is enabled for this calendar.
                if not cal.sync_enabled:
                    db.flush()
                    continue

                # Fetch events for this calendar
                try:
                    g_events = await _fetch_events(
                        client, access_token, google_cal_id, time_min, time_max
                    )
                except Exception as e:
                    msg = f"{g_cal['summary']}: events fetch failed: {e}"
                    logger.warning(msg)
                    result["errors"].append(msg)
                    continue

                synced_uids: set[str] = set()

                for g_ev in g_events:
                    g_id = g_ev.get("id")
                    if not g_id:
                        continue

                    # Skip cancelled events beyond recording them as cancelled.
                    status = (g_ev.get("status") or "confirmed").lower()

                    local_uid = _event_local_uid(g_id)
                    synced_uids.add(local_uid)

                    # Parse start/end
                    start_obj = g_ev.get("start") or {}
                    end_obj = g_ev.get("end") or {}
                    dtstart, all_day_s, is_utc_s = _parse_google_dt(start_obj)
                    dtend, all_day_e, is_utc_e = _parse_google_dt(end_obj)
                    all_day = all_day_s
                    is_utc = is_utc_s and not all_day

                    # Recurrence
                    rrule = _parse_recurrence(g_ev.get("recurrence") or [])

                    existing = db.query(CalendarEvent).filter(CalendarEvent.uid == local_uid).first()
                    if existing:
                        existing.summary = g_ev.get("summary") or ""
                        existing.description = g_ev.get("description") or ""
                        existing.location = g_ev.get("location") or ""
                        existing.status = status
                        existing.dtstart = dtstart
                        existing.dtend = dtend
                        existing.all_day = all_day
                        existing.is_utc = is_utc
                        existing.rrule = rrule
                        existing.origin = "google"
                    else:
                        ev = CalendarEvent(
                            uid=local_uid,
                            calendar_id=local_cal_id,
                            summary=g_ev.get("summary") or "",
                            description=g_ev.get("description") or "",
                            location=g_ev.get("location") or "",
                            status=status,
                            dtstart=dtstart,
                            dtend=dtend,
                            all_day=all_day,
                            is_utc=is_utc,
                            rrule=rrule,
                            origin="google",
                        )
                        db.add(ev)
                    result["events"] += 1

                # Prune stale Google-origin events in the sync window that no
                # longer appear in the Google response.
                stale = (
                    db.query(CalendarEvent)
                    .filter(
                        CalendarEvent.calendar_id == local_cal_id,
                        CalendarEvent.origin == "google",
                        CalendarEvent.dtstart >= datetime.strptime(time_min, "%Y-%m-%dT%H:%M:%SZ"),
                        CalendarEvent.dtstart <= datetime.strptime(time_max, "%Y-%m-%dT%H:%M:%SZ"),
                    )
                    .all()
                )
                for ev in stale:
                    if ev.uid not in synced_uids:
                        db.delete(ev)
                        result["deleted"] += 1

                db.flush()

            db.commit()
    except Exception as e:
        db.rollback()
        msg = f"Google calendar sync error: {e}"
        logger.exception(msg)
        result["errors"].append(msg)
    finally:
        db.close()

    return result
