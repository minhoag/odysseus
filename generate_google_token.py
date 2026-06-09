from pathlib import Path

from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = [
    "https://mail.google.com/",
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/calendar.events",
]

BASE = Path("data/google_oauth")
credentials_path = BASE / "credentials.json"
token_path = BASE / "token.json"

flow = InstalledAppFlow.from_client_secrets_file(
    str(credentials_path),
    scopes=SCOPES,
)

creds = flow.run_local_server(
    host="127.0.0.1",
    port=8088,
    access_type="offline",
    prompt="consent",
)

token_path.write_text(creds.to_json(), encoding="utf-8")
print(f"Saved token to {token_path}")
