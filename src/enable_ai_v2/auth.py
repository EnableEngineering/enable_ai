"""
Authentication handlers for API calls.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class Auth(ABC):
    """Base class for authentication."""

    @abstractmethod
    def get_headers(self) -> dict[str, str]:
        """Return headers to add to requests."""
        pass

    @abstractmethod
    def refresh(self) -> bool:
        """Refresh credentials if needed. Returns True if refreshed."""
        pass


class JWTAuth(Auth):
    """JWT Bearer token authentication."""

    def __init__(
        self,
        token: str,
        refresh_token: Optional[str] = None,
        refresh_callback: Optional[callable] = None,
    ):
        """
        Args:
            token: JWT access token
            refresh_token: Optional refresh token
            refresh_callback: Optional callback(refresh_token) -> new_token
        """
        self.token = token
        self.refresh_token = refresh_token
        self._refresh_callback = refresh_callback

    def get_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"}

    def refresh(self) -> bool:
        if self._refresh_callback and self.refresh_token:
            new_token = self._refresh_callback(self.refresh_token)
            if new_token:
                self.token = new_token
                return True
        return False

    def update_token(self, token: str) -> None:
        """Update the token (e.g., after external refresh)."""
        self.token = token


class APIKeyAuth(Auth):
    """API Key authentication (header or query param)."""

    def __init__(
        self,
        api_key: str,
        header_name: str = "X-API-Key",
        as_query_param: bool = False,
        query_param_name: str = "api_key",
    ):
        """
        Args:
            api_key: The API key
            header_name: Header name for key (default: X-API-Key)
            as_query_param: If True, add as query param instead of header
            query_param_name: Query param name if as_query_param=True
        """
        self.api_key = api_key
        self.header_name = header_name
        self.as_query_param = as_query_param
        self.query_param_name = query_param_name

    def get_headers(self) -> dict[str, str]:
        if self.as_query_param:
            return {}
        return {self.header_name: self.api_key}

    def get_query_params(self) -> dict[str, str]:
        if self.as_query_param:
            return {self.query_param_name: self.api_key}
        return {}

    def refresh(self) -> bool:
        return False  # API keys don't refresh


class BasicAuth(Auth):
    """HTTP Basic authentication."""

    def __init__(self, username: str, password: str):
        self.username = username
        self.password = password

    def get_headers(self) -> dict[str, str]:
        import base64
        credentials = f"{self.username}:{self.password}"
        encoded = base64.b64encode(credentials.encode()).decode()
        return {"Authorization": f"Basic {encoded}"}

    def refresh(self) -> bool:
        return False


class NoAuth(Auth):
    """No authentication (for public APIs)."""

    def get_headers(self) -> dict[str, str]:
        return {}

    def refresh(self) -> bool:
        return False
