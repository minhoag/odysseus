"""Tests for the web-OAuth helpers added to ``src/google_token_service.py``.

Covers: ``get_status``, ``build_auth_url``, ``_extract_client_info``, and
``exchange_code`` (mocked HTTP exchange). Tests never touch real tokens.
"""
import json
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src import google_token_service as gts


# ---------------------------------------------------------------------------
# Fixtures — build a synthetic google_oauth dir per test and redirect the
# module's module-level path constants to it. No real disk writes.
# ---------------------------------------------------------------------------

@pytest.fixture()
def oauth_dir(tmp_path, monkeypatch):
    """Redirect CREDENTIALS_FILE and TOKEN_FILE to tmp_path, return the dir."""
    monkeypatch.setattr(gts, "GOOGLE_OAUTH_DIR", tmp_path)
    monkeypatch.setattr(gts, "CREDENTIALS_FILE", tmp_path / "credentials.json")
    monkeypatch.setattr(gts, "TOKEN_FILE", tmp_path / "token.json")
    return tmp_path


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


# ---------------------------------------------------------------------------
# get_status
# ---------------------------------------------------------------------------

class TestGetStatus:
    def test_no_credentials_file(self, oauth_dir):
        status = gts.get_status()
        assert status["has_credentials"] is False
        assert status["has_token"] is False
        assert status["state"] == "no_credentials"
        # Must not leak path secrets or token values in detail.
        assert "token" not in status["detail"].lower() or "credentials" in status["detail"].lower()

    def test_no_token_file(self, oauth_dir):
        _write(gts.CREDENTIALS_FILE, {"client_id": "c", "client_secret": "s"})
        status = gts.get_status()
        assert status["has_credentials"] is True
        assert status["has_token"] is False
        assert status["state"] == "no_token"

    def test_valid_token(self, oauth_dir, monkeypatch):
        _write(gts.CREDENTIALS_FILE, {"client_id": "c", "client_secret": "s"})
        _write(gts.TOKEN_FILE, {
            "token": "access",
            "refresh_token": "refresh",
            "token_uri": "https://oauth2.googleapis.com/token",
            "client_id": "c",
            "client_secret": "s",
        })
        # load_credentials goes through network for refresh when expired.
        # Stub it to report "valid" directly.
        monkeypatch.setattr(gts, "load_credentials", lambda: {
            "state": "valid", "detail": "OK", "credentials": None,
        })
        status = gts.get_status()
        assert status["has_credentials"] is True
        assert status["has_token"] is True
        assert status["state"] == "valid"
        # Must not include secret keys in the response.
        assert "access" not in json.dumps(status)
        assert "refresh" not in json.dumps(status)

    def test_invalid_token_state(self, oauth_dir, monkeypatch):
        _write(gts.CREDENTIALS_FILE, {"client_id": "c", "client_secret": "s"})
        _write(gts.TOKEN_FILE, {"token": "access", "refresh_token": "refresh"})
        monkeypatch.setattr(gts, "load_credentials", lambda: {
            "state": "invalid", "detail": "refresh failed", "credentials": None,
        })
        status = gts.get_status()
        assert status["has_credentials"] is True
        assert status["has_token"] is False
        assert status["state"] == "invalid"


# ---------------------------------------------------------------------------
# _extract_client_info
# ---------------------------------------------------------------------------

class TestExtractClientInfo:
    def test_top_level(self, oauth_dir):
        _write(gts.CREDENTIALS_FILE, {"client_id": "cid", "client_secret": "cs"})
        info = gts._extract_client_info()
        assert info["client_id"] == "cid"
        assert info["client_secret"] == "cs"

    def test_nested_under_installed(self, oauth_dir):
        _write(gts.CREDENTIALS_FILE, {
            "installed": {"client_id": "i_cid", "client_secret": "i_cs"},
        })
        info = gts._extract_client_info()
        assert info["client_id"] == "i_cid"
        assert info["client_secret"] == "i_cs"

    def test_nested_under_web(self, oauth_dir):
        _write(gts.CREDENTIALS_FILE, {
            "web": {"client_id": "w_cid", "client_secret": "w_cs"},
        })
        info = gts._extract_client_info()
        assert info["client_id"] == "w_cid"
        assert info["client_secret"] == "w_cs"

    def test_missing_client_secret(self, oauth_dir):
        _write(gts.CREDENTIALS_FILE, {"client_id": "cid"})
        with pytest.raises(gts.TokenLoadError):
            gts._extract_client_info()


# ---------------------------------------------------------------------------
# build_auth_url
# ---------------------------------------------------------------------------

class TestBuildAuthUrl:
    def test_contains_required_params(self, oauth_dir):
        _write(gts.CREDENTIALS_FILE, {"client_id": "cid", "client_secret": "cs"})
        url = gts.build_auth_url("http://example.test/api/google-oauth/callback")
        assert url.startswith("https://accounts.google.com/o/oauth2/v2/auth?")
        assert "client_id=cid" in url
        assert "redirect_uri=http" in url
        assert "response_type=code" in url
        assert "access_type=offline" in url
        assert "prompt=consent" in url
        assert "scope=" in url


# ---------------------------------------------------------------------------
# exchange_code
# ---------------------------------------------------------------------------

class TestExchangeCode:
    def test_exchange_persists_token(self, oauth_dir):
        _write(gts.CREDENTIALS_FILE, {"client_id": "cid", "client_secret": "cs"})

        fake_creds = MagicMock()
        fake_creds.to_json.return_value = json.dumps({
            "token": "access_xyz",
            "refresh_token": "refresh_xyz",
        })

        fake_flow = MagicMock()
        fake_flow.credentials = fake_creds
        fake_flow_cls = MagicMock(return_value=fake_flow)
        fake_flow_cls.from_client_secrets_file = MagicMock(return_value=fake_flow)

        with patch.object(gts, "Flow", fake_flow_cls):
            creds = gts.exchange_code(
                code="4/auth_code_xyz",
                redirect_uri="http://example.test/api/google-oauth/callback",
            )

        # fetch_token received the code.
        fake_flow.fetch_token.assert_called_once_with(code="4/auth_code_xyz")
        # We got back the credentials object.
        assert creds is fake_creds
        # token.json was written.
        assert gts.TOKEN_FILE.is_file()
        text = gts.TOKEN_FILE.read_text()
        assert "access_xyz" in text
        assert "refresh_xyz" in text

    def test_exchange_missing_credentials_file(self, oauth_dir):
        with pytest.raises(gts.TokenLoadError):
            gts.exchange_code("code", "http://x.test/cb")


# ---------------------------------------------------------------------------
# _read_credentials — must accept web/installed/top-level shapes
# ---------------------------------------------------------------------------

class TestReadCredentials:
    def test_top_level(self, oauth_dir):
        _write(gts.CREDENTIALS_FILE, {
            "client_id": "cid",
            "client_secret": "cs",
            "token_uri": "https://oauth2.googleapis.com/token",
        })
        data = gts._read_credentials()
        assert data["client_id"] == "cid"

    def test_web_shaped(self, oauth_dir):
        _write(gts.CREDENTIALS_FILE, {
            "web": {
                "client_id": "w_cid",
                "client_secret": "w_cs",
                "token_uri": "https://oauth2.googleapis.com/token",
            },
        })
        data = gts._read_credentials()
        assert data["client_id"] == "w_cid"
        assert data["client_secret"] == "w_cs"

    def test_installed_shaped(self, oauth_dir):
        _write(gts.CREDENTIALS_FILE, {
            "installed": {
                "client_id": "i_cid",
                "client_secret": "i_cs",
                "token_uri": "https://oauth2.googleapis.com/token",
            },
        })
        data = gts._read_credentials()
        assert data["client_id"] == "i_cid"

    def test_no_client_id_raises(self, oauth_dir):
        _write(gts.CREDENTIALS_FILE, {"irrelevant": "data"})
        with pytest.raises(gts.TokenLoadError, match="client_id"):
            gts._read_credentials()


# ---------------------------------------------------------------------------
# get_access_token / load_credentials — must work with web-shaped credentials
# This is the exact scenario that failed before the fix.
# ---------------------------------------------------------------------------

class TestLoadWithWebCredentials:
    def _setup_web(self, oauth_dir):
        _write(gts.CREDENTIALS_FILE, {
            "web": {
                "client_id": "w_cid.webapp",
                "client_secret": "w_secret",
                "token_uri": "https://oauth2.googleapis.com/token",
            },
        })
        _write(gts.TOKEN_FILE, {
            "token": "access_tok",
            "refresh_token": "refresh_tok",
            "token_uri": "https://oauth2.googleapis.com/token",
            "client_id": "w_cid.webapp",
            "client_secret": "w_secret",
        })

    @patch.object(gts, "_load_google_credentials")
    @patch.object(gts, "_ensure_valid")
    def test_load_credentials_web_shape_valid(self, mock_ensure, mock_load, oauth_dir):
        self._setup_web(oauth_dir)
        fake_creds = MagicMock()
        fake_creds.expired = False
        mock_load.return_value = fake_creds
        result = gts.load_credentials()
        assert result["state"] == "valid", result["detail"]
        assert result["detail"] == "OK"

    @patch.object(gts, "_load_google_credentials")
    @patch.object(gts, "_ensure_valid")
    @patch.object(gts, "save_token")
    def test_get_access_token_web_shape(self, mock_save, mock_ensure, mock_load, oauth_dir):
        """get_access_token must succeed with web-shaped credentials.json."""
        self._setup_web(oauth_dir)
        fake_creds = MagicMock()
        fake_creds.token = "valid_access_token"
        fake_creds.expired = False
        mock_load.return_value = fake_creds
        token = gts.get_access_token()
        assert token == "valid_access_token"


# ---------------------------------------------------------------------------
# _load_google_credentials — must handle scopes correctly
# ---------------------------------------------------------------------------

class TestLoadGoogleCredentials:
    def test_malformed_scopes_object_rejected(self, oauth_dir):
        """Token with scopes as object (not list) must be rejected."""
        credentials_info = {
            "client_id": "cid",
            "client_secret": "cs",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
        token_data = {
            "token": "access",
            "refresh_token": "refresh",
            "scopes": {  # Bad: object instead of list
                "client_id": "cid",
                "project_id": "proj",
                "redirect_uris": ["http://localhost"],
            },
        }
        with pytest.raises(gts.TokenLoadError, match="malformed.*scopes"):
            gts._load_google_credentials(credentials_info, token_data)

    def test_merges_client_metadata_from_credentials(self, oauth_dir):
        """Token missing client_id/client_secret gets them from credentials.json."""
        credentials_info = {
            "client_id": "from_creds",
            "client_secret": "secret_from_creds",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
        token_data = {
            "token": "access",
            "refresh_token": "refresh",
        }
        # Should not raise - the merge should work
        creds = gts._load_google_credentials(credentials_info, token_data)
        # Verify the credentials object was created with merged info
        assert creds.client_id == "from_creds"
        assert creds.client_secret == "secret_from_creds"

    def test_token_with_valid_scopes_list(self, oauth_dir):
        """Token with scopes as list of strings must be accepted."""
        credentials_info = {
            "client_id": "cid",
            "client_secret": "cs",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
        token_data = {
            "token": "access",
            "refresh_token": "refresh",
            "scopes": ["https://www.googleapis.com/auth/calendar"],
        }
        # Should not raise
        creds = gts._load_google_credentials(credentials_info, token_data)
        assert creds is not None
