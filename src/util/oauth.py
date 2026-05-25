import requests
from urllib.parse import urlencode


class OAuthClient:
    """
    Handles OAuth 2.0 / OpenID Connect workflows for Google, GitHub, and Developer Mock Login.
    """

    def __init__(self, provider: str, client_id: str = None, client_secret: str = None, redirect_uri: str = None):
        self.provider = provider.lower()
        self.client_id = client_id
        self.client_secret = client_secret
        if not redirect_uri:
            try:
                import streamlit as st
                redirect_uri = st.context.url.rstrip("/")
            except Exception:
                pass
        self.redirect_uri = redirect_uri or "http://localhost:8501"

    def get_authorization_url(self, state: str = "state123") -> str:
        """
        Generates the authorization redirect URL for the chosen provider.
        """
        if self.provider == "google":
            params = {
                "client_id": self.client_id,
                "redirect_uri": self.redirect_uri,
                "response_type": "code",
                "scope": "openid email profile",
                "state": state,
                "access_type": "online",
            }
            return f"https://accounts.google.com/o/oauth2/v2/auth?{urlencode(params)}"
        elif self.provider == "github":
            params = {
                "client_id": self.client_id,
                "redirect_uri": self.redirect_uri,
                "scope": "read:user user:email",
                "state": state,
            }
            return f"https://github.com/login/oauth/authorize?{urlencode(params)}"
        else:
            return f"{self.redirect_uri}?code=mock_code_dev_user&state={state}"

    def exchange_code_for_user(self, code: str) -> dict:
        """
        Exchanges the authorization code for user details.
        Supports both real OAuth API calls and Developer Mock login code.
        """
        if code.startswith("mock_code_"):
            return self._get_mock_user(code)

        if not self.client_id or not self.client_secret:
            raise ValueError("Client ID and Client Secret must be configured for real OAuth flow.")

        if self.provider == "google":
            return self._exchange_google(code)
        elif self.provider == "github":
            return self._exchange_github(code)
        else:
            return self._get_mock_user(code)

    def _exchange_google(self, code: str) -> dict:
        token_url = "https://oauth2.googleapis.com/token"
        payload = {
            "code": code,
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "redirect_uri": self.redirect_uri,
            "grant_type": "authorization_code",
        }
        res = requests.post(token_url, data=payload, timeout=30)
        res.raise_for_status()
        tokens = res.json()
        access_token = tokens.get("access_token")

        userinfo_url = "https://openidconnect.googleapis.com/v1/userinfo"
        headers = {"Authorization": f"Bearer {access_token}"}
        user_res = requests.get(userinfo_url, headers=headers, timeout=30)
        user_res.raise_for_status()
        userinfo = user_res.json()

        return {
            "name": userinfo.get("name", "Google User"),
            "email": userinfo.get("email"),
            "avatar": userinfo.get("picture"),
            "provider": "google",
        }

    def _exchange_github(self, code: str) -> dict:
        token_url = "https://github.com/login/oauth/access_token"
        payload = {
            "code": code,
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "redirect_uri": self.redirect_uri,
        }
        headers = {"Accept": "application/json"}
        res = requests.post(token_url, json=payload, headers=headers, timeout=30)
        res.raise_for_status()
        tokens = res.json()
        access_token = tokens.get("access_token")

        user_url = "https://api.github.com/user"
        user_headers = {"Authorization": f"Bearer {access_token}", "Accept": "application/vnd.github.v3+json"}
        user_res = requests.get(user_url, headers=user_headers, timeout=30)
        user_res.raise_for_status()
        userinfo = user_res.json()

        email = userinfo.get("email")
        if not email:
            email_res = requests.get("https://api.github.com/user/emails", headers=user_headers, timeout=30)
            if email_res.status_code == 200:
                emails = email_res.json()
                primary_emails = [e.get("email") for e in emails if e.get("primary")]
                if primary_emails:
                    email = primary_emails[0]
                elif emails:
                    email = emails[0].get("email")

        return {
            "name": userinfo.get("name") or userinfo.get("login", "GitHub User"),
            "email": email,
            "avatar": userinfo.get("avatar_url"),
            "provider": "github",
        }

    def _get_mock_user(self, code: str) -> dict:
        username = "Developer"
        email = "developer@example.com"
        avatar = "data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='%2378909C'><path d='M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z'/></svg>"
        if "google" in code:
            username = "Google Developer"
            email = "developer.google@example.com"
        elif "github" in code:
            username = "GitHub Developer"
            email = "developer.github@example.com"

        return {
            "name": username,
            "email": email,
            "avatar": avatar,
            "provider": "mock",
        }
