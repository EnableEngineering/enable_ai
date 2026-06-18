"""
HTTP client for executing API calls.
"""

from __future__ import annotations

import re
import time
from typing import Any, Optional
from urllib.parse import urljoin

import httpx

from .auth import Auth, NoAuth, APIKeyAuth
from .config import Config
from .types import APICallTrace, Endpoint, ToolCall


class APIClient:
    """Execute API calls against a base URL."""

    def __init__(
        self,
        base_url: str,
        auth: Optional[Auth] = None,
        config: Optional[Config] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.auth = auth or NoAuth()
        self.config = config or Config()

        self._client = httpx.Client(
            timeout=self.config.request_timeout,
            follow_redirects=True,
        )

    def execute(
        self,
        tool_call: ToolCall,
        endpoint: Endpoint,
    ) -> tuple[dict, APICallTrace]:
        """
        Execute a single tool call against its endpoint with retry for 5xx errors.

        Args:
            tool_call: The tool call with arguments
            endpoint: The endpoint definition

        Returns:
            Tuple of (response_data, trace)
        """
        last_result = None
        last_trace = None

        for attempt in range(self.config.max_retries + 1):
            result, trace = self._execute_once(tool_call, endpoint)
            last_result = result
            last_trace = trace

            # Success or client error (4xx) - don't retry
            if trace.success or (trace.status_code >= 400 and trace.status_code < 500):
                return result, trace

            # Server error (5xx) - retry with backoff
            if trace.status_code >= 500 and attempt < self.config.max_retries:
                delay = self.config.retry_delay_seconds * (2 ** attempt)
                time.sleep(delay)
                continue

            # Timeout or other error - don't retry
            break

        return last_result, last_trace

    def _execute_once(
        self,
        tool_call: ToolCall,
        endpoint: Endpoint,
    ) -> tuple[dict, APICallTrace]:
        """Execute a single API call (no retry)."""
        start_time = time.time()

        # Build URL with path parameters
        url = self._build_url(endpoint.path, tool_call.arguments)

        # Separate query params from body params
        query_params, body_params = self._split_params(
            tool_call.arguments,
            endpoint,
        )

        # Add auth query params if needed
        if isinstance(self.auth, APIKeyAuth):
            query_params.update(self.auth.get_query_params())

        # Build headers
        headers = {"Content-Type": "application/json"}
        headers.update(self.auth.get_headers())

        # Execute request
        method = endpoint.method.upper()

        try:
            if method == "GET":
                response = self._client.get(url, params=query_params, headers=headers)
            elif method == "POST":
                response = self._client.post(
                    url, params=query_params, json=body_params or None, headers=headers
                )
            elif method == "PUT":
                response = self._client.put(
                    url, params=query_params, json=body_params or None, headers=headers
                )
            elif method == "PATCH":
                response = self._client.patch(
                    url, params=query_params, json=body_params or None, headers=headers
                )
            elif method == "DELETE":
                response = self._client.delete(url, params=query_params, headers=headers)
            else:
                raise ValueError(f"Unsupported method: {method}")

            elapsed_ms = int((time.time() - start_time) * 1000)

            # Parse response
            try:
                data = response.json()
            except Exception:
                data = {"raw": response.text}

            trace = APICallTrace(
                endpoint=endpoint.path,
                method=method,
                params=tool_call.arguments,
                status_code=response.status_code,
                response_time_ms=elapsed_ms,
                success=response.is_success,
            )

            if response.is_success:
                return {"data": data, "status": response.status_code}, trace
            else:
                return {
                    "error": f"HTTP {response.status_code}",
                    "detail": data,
                    "status": response.status_code,
                }, trace

        except httpx.TimeoutException:
            elapsed_ms = int((time.time() - start_time) * 1000)
            trace = APICallTrace(
                endpoint=endpoint.path,
                method=method,
                params=tool_call.arguments,
                status_code=0,
                response_time_ms=elapsed_ms,
                success=False,
            )
            return {"error": "Request timed out"}, trace

        except Exception as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            trace = APICallTrace(
                endpoint=endpoint.path,
                method=method,
                params=tool_call.arguments,
                status_code=0,
                response_time_ms=elapsed_ms,
                success=False,
            )
            return {"error": str(e)}, trace

    def _build_url(self, path: str, arguments: dict) -> str:
        """Build URL with path parameters substituted."""
        url = path

        # Replace path parameters: /users/{id} -> /users/123
        for match in re.finditer(r"\{(\w+)\}", path):
            param_name = match.group(1)
            if param_name in arguments:
                url = url.replace(
                    match.group(0),
                    str(arguments[param_name])
                )

        return urljoin(self.base_url + "/", url.lstrip("/"))

    def _split_params(
        self,
        arguments: dict,
        endpoint: Endpoint,
    ) -> tuple[dict, dict]:
        """Split arguments into query params and body params."""
        query_params = {}
        body_params = {}

        # Get path parameter names
        path_params = set(re.findall(r"\{(\w+)\}", endpoint.path))

        # Get query parameter names from endpoint definition
        query_param_names = {
            p["name"] for p in endpoint.parameters
            if p.get("in") == "query"
        }

        for name, value in arguments.items():
            if name in path_params:
                continue  # Already handled in URL
            elif name in query_param_names:
                query_params[name] = value
            else:
                body_params[name] = value

        return query_params, body_params

    def close(self):
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
