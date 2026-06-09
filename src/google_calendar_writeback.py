"""google_calendar_writeback.py — Push create/update/delete to Google Calendar v3 REST.

Used by calendar routes and agent tools when a CalendarEvent belongs to a
calendar with source="google". Uses httpx.AsyncClient directly.
"""

import logging

import httpx

from core.database import CalendarEvent
from src.google_token_service import get_access_token, TokenLoadError

logger = logging.getLogger(__name__)

EVENTS_URL = "https://www.googleapis.com/calendar/v3/calendars/{calendarId}/events"
EVENT_URL = "https://www.googleapis.com/calendar/v3/calendars/{calendarId}/events/{eventId}"


def _headers(access_token: str) -> dict:
    return {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }


def _strip_google_prefix(uid: str) -> str:
    if uid.startswith("google-"):
        return uid[7:]
    return uid


def _event_to_google(event: CalendarEvent) -> dict:
    """Convert a local CalendarEvent to a Google Calendar event JSON body."""
    body: dict = {
        "summary": event.summary or "",
        "description": event.description or "",
        "location": event.location or "",
    }

    if event.all_day:
        body["start"] = {"date": event.dtstart.strftime("%Y-%m-%d")}
        body["end"] = {"date": event.dtend.strftime("%Y-%m-%d")}
    else:
        if event.is_utc:
            body["start"] = {"dateTime": event.dtstart.isoformat() + "Z"}
            body["end"] = {"dateTime": event.dtend.isoformat() + "Z"}
        else:
            body["start"] = {"dateTime": event.dtstart.isoformat()}
            body["end"] = {"dateTime": event.dtend.isoformat()}

    # Recurrence: local stores joined RRULE parts; Google wants a list with
    # each entry prefixed with "RRULE:" if not already present.
    if event.rrule:
        parts = [p.strip() for p in event.rrule.split(";") if p.strip()]
        recurrence = []
        for part in parts:
            upper = part.upper()
            if upper.startswith("RRULE:") or upper.startswith("EXRULE:") or \
               upper.startswith("EXDATE:") or upper.startswith("RDATE:"):
                recurrence.append(part)
            else:
                recurrence.append(f"RRULE:{part}")
        body["recurrence"] = recurrence

    return body


async def create_google_event(calendar_id: str, event: CalendarEvent) -> str:
    """Create event on Google and return the Google event ID.

    Caller is responsible for prefixing the returned ID with 'google-' when
    setting the local uid.
    """
    try:
        access_token = get_access_token()
    except TokenLoadError as e:
        raise RuntimeError(f"Google token unavailable: {e}")

    body = _event_to_google(event)
    url = EVENTS_URL.format(calendarId=calendar_id)

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(url, headers=_headers(access_token), json=body)
        r.raise_for_status()
        data = r.json()

    return data["id"]


async def update_google_event(calendar_id: str, event_uid: str, event: CalendarEvent) -> None:
    """PATCH an existing event on Google. Strips the 'google-' prefix from uid."""
    try:
        access_token = get_access_token()
    except TokenLoadError as e:
        raise RuntimeError(f"Google token unavailable: {e}")

    google_event_id = _strip_google_prefix(event_uid)
    body = _event_to_google(event)
    url = EVENT_URL.format(calendarId=calendar_id, eventId=google_event_id)

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.patch(url, headers=_headers(access_token), json=body)
        r.raise_for_status()


async def delete_google_event(calendar_id: str, event_uid: str) -> None:
    """DELETE an event from Google. Strips the 'google-' prefix from uid."""
    try:
        access_token = get_access_token()
    except TokenLoadError as e:
        raise RuntimeError(f"Google token unavailable: {e}")

    google_event_id = _strip_google_prefix(event_uid)
    url = EVENT_URL.format(calendarId=calendar_id, eventId=google_event_id)

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.delete(url, headers=_headers(access_token))
        r.raise_for_status()
