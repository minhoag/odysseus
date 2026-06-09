"""google_token_service.py — file-backed Google OAuth token service.

Reads credentials.json (client metadata) and token.json (authorized-user
credentials) from data/google_oauth/. Provides credential loading, token
refresh, a compact status dict, and the web-OAuth helpers used by
``routes/google_oauth_routes.py`` to exchange an authorization code for a
persisted token. Never logs or exposes token contents.
"""
import json
import logging
from pathlib import Path

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import Flow

from src.constants import DATA_DIR

logger = logging.getLogger(__name__)

GOOGLE_OAUTH_DIR = Path(DATA_DIR) / "google_oauth"
CREDENTIALS_FILE = GOOGLE_OAUTH_DIR / "credentials.json"
TOKEN_FILE = GOOGLE_OAUTH_DIR / "token.json"

# Scopes requested in the web OAuth consent screen. Matches the scopes used
# by the manual generate_google_token.py script.
GOOGLE_OAUTH_SCOPES = [
    "https://mail.google.com/",
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/calendar.events",
]


class TokenLoadError(Exception):
    """Raised when token loading encounters an unrecoverable problem."""


def _read_json_file(path: Path, label: str) -> dict:
    if not path.is_file():
        raise TokenLoadError(f"{label} not found at {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise TokenLoadError(f"{label} is empty")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise TokenLoadError(f"{label} contains invalid JSON: {e}")
    if not isinstance(data, dict):
        raise TokenLoadError(f"{label} must contain a JSON object")
    return data


def _unwrap_credentials_container(data: dict) -> dict:
    """Return a flat credentials dict from Google's various JSON shapes.

    Google Cloud Console exports credentials in one of these layouts:
      * top-level keys (standalone export)
      * keys nested under ``"web"`` (Web client)
      * keys nested under ``"installed"`` (Desktop client)
    """
    if data.get("client_id"):
        return data
    for bucket in ("web", "installed"):
        inner = data.get(bucket)
        if isinstance(inner, dict) and inner.get("client_id"):
            return inner
    return data


def _read_credentials() -> dict:
    data = _read_json_file(CREDENTIALS_FILE, "credentials.json")
    data = _unwrap_credentials_container(data)
    if "client_id" not in data or not data["client_id"]:
        raise TokenLoadError("credentials.json missing or empty 'client_id'")
    return data


def _read_token() -> dict:
    data = _read_json_file(TOKEN_FILE, "token.json")
    if "refresh_token" not in data:
        raise TokenLoadError("token.json missing 'refresh_token'")
    if "access_token" in data and "token" not in data:
        raise TokenLoadError(
            "token.json uses non-standard 'access_token' key; "
            "standard authorized-user JSON uses 'token'"
        )
    return data


def _load_google_credentials(credentials_info: dict, token_data: dict) -> Credentials:
    """Build authorized-user info from token + credentials metadata and load.

    Merges token_data with client metadata from credentials_info (client_id,
    client_secret, token_uri) if those fields are missing from token_data.
    Always uses GOOGLE_OAUTH_SCOPES for scopes.
    """
    # Validate scopes in token_data if present - must be a list of strings
    if "scopes" in token_data and not isinstance(token_data["scopes"], list):
        raise TokenLoadError(
            "token.json has malformed 'scopes' field (expected list of strings); "
            "delete token.json and re-authorize"
        )

    # Build the authorized-user info dict
    info = dict(token_data)

    # Merge in client metadata from credentials_info if missing
    for key in ("client_id", "client_secret", "token_uri"):
        if key not in info and key in credentials_info:
            info[key] = credentials_info[key]

    try:
        return Credentials.from_authorized_user_info(info, scopes=GOOGLE_OAUTH_SCOPES)
    except (ValueError, KeyError) as e:
        raise TokenLoadError(f"failed to load Google credentials: {e}")


def _ensure_valid(creds: Credentials) -> Credentials:
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
    return creds


def load_credentials() -> dict:
    """Load Google authorized-user credentials from disk.

    Returns a status dict::

        {
            "state": "missing" | "invalid" | "valid",
            "detail": "...",       # human-readable reason
            "credentials": Credentials(...),  # only on state="valid", else None
        }

    Never includes token values in ``detail`` or log output.
    """
    try:
        credentials_info = _read_credentials()
    except TokenLoadError as e:
        return {"state": "missing", "detail": str(e), "credentials": None}

    try:
        token_data = _read_token()
    except TokenLoadError as e:
        return {"state": "invalid", "detail": str(e), "credentials": None}

    try:
        creds = _load_google_credentials(credentials_info, token_data)
    except TokenLoadError as e:
        return {"state": "invalid", "detail": str(e), "credentials": None}

    if creds.expired and creds.refresh_token:
        try:
            creds = _ensure_valid(creds)
        except Exception as e:
            return {"state": "invalid", "detail": f"token refresh failed: {e}", "credentials": None}

    return {"state": "valid", "detail": "OK", "credentials": creds}


def save_token(creds: Credentials) -> None:
    """Persist the current token state back to token.json."""
    TOKEN_FILE.write_text(creds.to_json(), encoding="utf-8")


def get_access_token() -> str:
    """Return a valid Google access token, refreshing if needed.

    Raises TokenLoadError when credentials are missing, invalid, or
    refresh fails. Never logs token contents.
    """
    result = load_credentials()
    if result["state"] != "valid":
        raise TokenLoadError(f"Google credentials not available: {result['detail']}")
    creds = result["credentials"]
    # Save refreshed token back to disk so next load doesn't re-refresh.
    save_token(creds)
    return creds.token


# ---------------------------------------------------------------------------
# Web OAuth helpers — used by routes/google_oauth_routes.py.
# ---------------------------------------------------------------------------

def _extract_client_info() -> dict:
    """Read client_id and client_secret from credentials.json.

    Supports three layouts that Google Cloud Console can export:
      * top-level keys (standalone export)
      * keys nested under ``"installed"`` (Desktop client)
      * keys nested under ``"web"`` (Web client)
    """
    data = _read_json_file(CREDENTIALS_FILE, "credentials.json")
    info = _unwrap_credentials_container(data)
    if info.get("client_id") and info.get("client_secret"):
        return {"client_id": info["client_id"], "client_secret": info["client_secret"]}
    raise TokenLoadError(
        "credentials.json must contain client_id and client_secret "
        "(top-level or under 'installed'/'web')"
    )


def get_status() -> dict:
    """Return a UI-safe status dict. Never includes token values."""
    has_credentials = CREDENTIALS_FILE.is_file()
    has_token = TOKEN_FILE.is_file()
    if not has_credentials:
        return {
            "has_credentials": False,
            "has_token": False,
            "state": "no_credentials",
            "detail": "OAuth client credentials not configured",
        }
    if not has_token:
        return {
            "has_credentials": True,
            "has_token": False,
            "state": "no_token",
            "detail": "No token yet — authorize through the web flow",
        }
    result = load_credentials()
    state = result["state"]
    return {
        "has_credentials": True,
        "has_token": state == "valid",
        "state": state,
        "detail": result["detail"],
    }


def build_auth_url(redirect_uri: str) -> str:
    """Build the Google authorization URL for the web OAuth flow.

    ``redirect_uri`` must exactly match one of the URIs registered in
    Google Cloud Console, otherwise Google rejects the request.
    """
    client = _extract_client_info()
    from urllib.parse import urlencode
    params = {
        "client_id": client["client_id"],
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "access_type": "offline",
        "prompt": "consent",
        "scope": " ".join(GOOGLE_OAUTH_SCOPES),
    }
    return "https://accounts.google.com/o/oauth2/v2/auth?" + urlencode(params)


def exchange_code(code: str, redirect_uri: str) -> Credentials:
    """Exchange an authorization code for tokens and persist to token.json.

    Uses ``google_auth_oauthlib.flow.Flow`` which handles the protocol,
    including the ``installed``/``web`` credentials-file layout. Returns
    the resulting ``Credentials`` object (also written to ``token.json``).
    """
    # _extract_client_info validates that the file is present and holds
    # the required client_id/client_secret before we delegate to Flow.
    _extract_client_info()
    flow = Flow.from_client_secrets_file(
        str(CREDENTIALS_FILE),
        scopes=GOOGLE_OAUTH_SCOPES,
        redirect_uri=redirect_uri,
    )
    flow.fetch_token(code=code)
    creds = flow.credentials
    GOOGLE_OAUTH_DIR.mkdir(parents=True, exist_ok=True)
    TOKEN_FILE.write_text(creds.to_json(), encoding="utf-8")
    logger.info("Google OAuth token saved to %s", TOKEN_FILE)
    return creds
