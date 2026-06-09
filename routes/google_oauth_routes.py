"""
google_oauth_routes.py — Web OAuth flow for Google Gmail / Calendar.

Three endpoints implement the standard authorization-code grant against
Google's OAuth 2.0 server, writing the resulting token to
``data/google_oauth/token.json`` where ``src/google_token_service.py``
can load it for XOAUTH2 IMAP/SMTP.

* ``GET /api/google-oauth/status``  — lightweight status dict (no secrets).
* ``GET /api/google-oauth/start``   — builds the consent URL and redirects
  the browser to Google. Requires authentication.
* ``GET /api/google-oauth/callback`` — receives the authorization code
  back from Google, exchanges it, and returns a self-contained HTML page
  that tells the user they can close the window.

The redirect URI ``/api/google-oauth/callback`` must be pre-registered in
Google Cloud Console. If the browser opens the app from a URL different
from the one registered there, Google will reject the request with a
"redirect_uri mismatch" error.
"""

import logging
import secrets
import time
from html import escape as _html_escape

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from routes.email_helpers import require_user

logger = logging.getLogger(__name__)

# Single-flight CSRF state for the /start → /callback round-trip.
# Replaced on every /start call. A personal server runs one flow at a
# time, so a single slot is sufficient.
_PENDING_STATE: dict = {"token": None, "redirect_uri": None, "created_at": 0.0}

# Maximum age (seconds) of a pending OAuth state before /callback rejects it.
_STATE_MAX_AGE_SECONDS = 600


def _callback_html(*, title: str, body: str, accent_css: str, auto_close: bool = True) -> str:
    """Return a self-contained HTML page for the OAuth callback response.

    Kept inline so no static assets are required. The script closes the
    window after a short delay on success so the user lands back in the
    settings page without an extra click.
    """
    close_script = (
        "<script>setTimeout(function(){ "
        "if (window.opener && !window.opener.closed) { try { window.opener.location.reload(); } catch(_) {} } "
        "window.close(); "
        "}, 1800);</script>"
        if auto_close else ""
    )
    return (
        "<!DOCTYPE html>"
        "<html><head><meta charset='utf-8'>"
        f"<title>{_html_escape(title)}</title>"
        "<style>"
        "body{font:14px system-ui,sans-serif;background:#1a1a1a;color:#f0f0f0;"
        "display:flex;align-items:center;justify-content:center;min-height:100vh;margin:0;}"
        ".card{max-width:420px;padding:24px;border-radius:10px;"
        f"border:1px solid {accent_css};box-shadow:0 0 0 1px {accent_css}33;background:#222;}}"
        "h1{margin:0 0 8px;font-size:16px;}p{margin:0;opacity:0.8;line-height:1.5;}"
        "</style>"
        f"</head><body><div class='card'><h1>{_html_escape(title)}</h1>"
        f"<p>{_html_escape(body)}</p></div>{close_script}</body></html>"
    )


def _build_redirect_uri(request: Request) -> str:
    base = str(request.base_url).rstrip("/")
    return f"{base}/api/google-oauth/callback"


def setup_google_oauth_routes() -> APIRouter:
    router = APIRouter(prefix="/api/google-oauth", tags=["google-oauth"])

    @router.get("/status")
    async def google_oauth_status(_owner: str = Depends(require_user)):
        """Lightweight status dict. Never includes token values."""
        from src.google_token_service import get_status, TokenLoadError
        try:
            return get_status()
        except TokenLoadError as e:
            return {
                "has_credentials": False,
                "has_token": False,
                "state": "error",
                "detail": str(e),
            }

    @router.get("/start")
    async def google_oauth_start(request: Request, _owner: str = Depends(require_user)):
        """Build the Google consent URL and 302-redirect the browser to it.

        Stores a random ``state`` token so ``/callback`` can verify the
        response belongs to this flow (CSRF). If the registered redirect
        URI doesn't match what the browser is using, Google will reject
        the request and the user will see a mismatch error — there is no
        way around that other than accessing the app from the exact URL
        pre-registered in Cloud Console.
        """
        from src.google_token_service import build_auth_url, TokenLoadError

        redirect_uri = _build_redirect_uri(request)
        try:
            auth_url = build_auth_url(redirect_uri)
        except TokenLoadError as e:
            return HTMLResponse(
                _callback_html(
                    title="Google OAuth not configured",
                    body=f"Cannot start OAuth flow: {e}",
                    accent_css="var(--red, #ff5555)",
                    auto_close=True,
                ),
                status_code=400,
            )

        # Stash state for /callback's CSRF check.
        _PENDING_STATE["token"] = secrets.token_urlsafe(32)
        _PENDING_STATE["redirect_uri"] = redirect_uri
        _PENDING_STATE["created_at"] = time.monotonic()
        auth_url_with_state = f"{auth_url}&state={_PENDING_STATE['token']}"
        return RedirectResponse(auth_url_with_state, status_code=302)

    @router.get("/callback")
    async def google_oauth_callback(request: Request):
        """Receive the authorization code from Google, exchange it,
        and persist the resulting tokens to token.json.

        Not behind ``require_user`` — this request arrives directly from
        Google's redirect and carries no session cookies. We rely on the
        ``state`` parameter (checked against what ``/start`` stored) to
        verify the request belongs to our initiated flow.
        """
        from src.google_token_service import exchange_code, TokenLoadError

        params = request.query_params
        error = params.get("error")
        code = params.get("code")
        state = params.get("state")

        # Google reported a problem (user denied consent, invalid scope, …).
        if error:
            return HTMLResponse(
                _callback_html(
                    title="Authorization denied",
                    body=f"Google reported: {error}",
                    accent_css="var(--red, #ff5555)",
                    auto_close=True,
                ),
                status_code=400,
            )

        # CSRF check.
        if (
            not state
            or state != _PENDING_STATE["token"]
            or not _PENDING_STATE["token"]
            or (time.monotonic() - _PENDING_STATE["created_at"]) > _STATE_MAX_AGE_SECONDS
        ):
            return HTMLResponse(
                _callback_html(
                    title="Invalid or expired state",
                    body="The OAuth state did not match, or the flow timed out. Please try again.",
                    accent_css="var(--red, #ff5555)",
                    auto_close=True,
                ),
                status_code=400,
            )
        # Consume the state so it can't be reused.
        _PENDING_STATE["token"] = None

        if not code:
            return HTMLResponse(
                _callback_html(
                    title="Missing authorization code",
                    body="Google did not return an authorization code.",
                    accent_css="var(--red, #ff5555)",
                    auto_close=True,
                ),
                status_code=400,
            )

        redirect_uri = _PENDING_STATE["redirect_uri"] or _build_redirect_uri(request)
        try:
            exchange_code(code=code, redirect_uri=redirect_uri)
        except TokenLoadError as e:
            return HTMLResponse(
                _callback_html(
                    title="Token exchange failed",
                    body=str(e),
                    accent_css="var(--red, #ff5555)",
                    auto_close=True,
                ),
                status_code=502,
            )
        except Exception as e:
            logger.exception("Google OAuth token exchange raised")
            return HTMLResponse(
                _callback_html(
                    title="Token exchange failed",
                    body=f"Unexpected error: {e}",
                    accent_css="var(--red, #ff5555)",
                    auto_close=True,
                ),
                status_code=502,
            )

        return HTMLResponse(
            _callback_html(
                title="Authorized",
                body="Google account authorized. You can close this window.",
                accent_css="var(--green, #50fa7b)",
                auto_close=True,
            )
        )

    return router
