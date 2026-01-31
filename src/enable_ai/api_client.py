import time
import requests

from .types import APIRequest, APIResponse, APIError
from .utils import setup_logger
from . import constants


class APIClient:
    def __init__(self, api_base_url, access_token=None):
        """
        Initialize the API client with a base URL and optional access token.
        
        Args:
            api_base_url: Base URL for all API calls
            access_token: JWT access token for authentication (optional)
        """
        self.api_base_url = api_base_url.rstrip('/')
        self.access_token = access_token
        self.session = requests.Session()
        self.logger = setup_logger('enable_ai.client')
        
        # Set authorization header if token provided
        if access_token:
            self.session.headers.update({
                'Authorization': f'Bearer {access_token}',
                'Content-Type': 'application/json'
            })
            self.logger.debug("API client initialized with access token")
        else:
            self.logger.debug(f"API client initialized: base_url={api_base_url}")
    
    def _build_url(self, base_url: str, path: str) -> str:
        """
        Safely concatenate base_url and path, avoiding duplicate path segments.
        
        This fixes Issue #3 where base_url="http://localhost/api" + path="/api/users/"
        would result in "http://localhost/api/api/users/" (doubled /api).
        
        Args:
            base_url: Base URL (e.g., "http://localhost:8002/api")
            path: Endpoint path (e.g., "/api/users/" or "/users/")
            
        Returns:
            Correctly concatenated URL
            
        Examples:
            >>> _build_url("http://localhost/api", "/api/users/")
            "http://localhost/api/users/"
            
            >>> _build_url("http://localhost/api", "/users/")
            "http://localhost/api/users/"
            
            >>> _build_url("http://localhost", "/api/users/")
            "http://localhost/api/users/"
        """
        from urllib.parse import urlparse, urlunparse
        
        # Remove trailing slash from base_url
        base_url = base_url.rstrip('/')
        
        # Ensure path starts with /
        if not path.startswith('/'):
            path = '/' + path
        
        # Parse base_url to extract its path component
        parsed_base = urlparse(base_url)
        base_path = parsed_base.path.rstrip('/')
        
        # Check if the endpoint path already contains the base path
        if base_path and path.startswith(base_path):
            # Path already includes base path - use directly
            final_url = urlunparse((
                parsed_base.scheme,
                parsed_base.netloc,
                path,
                '',
                '',
                ''
            ))
        else:
            # Simple concatenation
            final_url = f"{base_url}{path}"
        
        self.logger.debug(f"Built URL: {base_url} + {path} → {final_url}")
        return final_url

    def call_api(self, api_request):
        """
        Make an API call using the APIRequest object.
        
        Args:
            api_request: APIRequest or APIError object
            
        Returns:
            APIResponse: If the API call succeeds
            APIError: If the API call fails or input was an error
        """
        # If input is already an error, return it
        if isinstance(api_request, APIError):
            return api_request
        
        if not isinstance(api_request, APIRequest):
            return APIError("Invalid request type - expected APIRequest")
        
        # Check if authentication is required
        if api_request.authentication_required and not self.access_token:
            return APIError("Authentication required but no access token provided")
        
        url = self._build_url(self.api_base_url, api_request.endpoint)
        method = api_request.method.upper()
        if method not in ('GET', 'POST', 'PUT', 'PATCH', 'DELETE'):
            return APIError(f"Unsupported HTTP method: {method}")
        
        last_error = None
        timeout_seconds = constants.REQUEST_TIMEOUT
        for attempt in range(constants.REQUEST_RETRY_ATTEMPTS):
            try:
                if method == 'GET':
                    response = self.session.get(url, params=api_request.params, timeout=timeout_seconds)
                elif method == 'POST':
                    response = self.session.post(url, json=api_request.params, timeout=timeout_seconds)
                elif method == 'PUT':
                    response = self.session.put(url, json=api_request.params, timeout=timeout_seconds)
                elif method == 'PATCH':
                    response = self.session.patch(url, json=api_request.params, timeout=timeout_seconds)
                else:
                    response = self.session.delete(url, params=api_request.params, timeout=timeout_seconds)
                
                # Success: return immediately
                if response.status_code in [200, 201]:
                    try:
                        data = response.json() if response.content else {}
                    except ValueError:
                        data = {"response": response.text}
                    return APIResponse(status_code=response.status_code, data=data)
                if response.status_code == 204:
                    return APIResponse(status_code=response.status_code, data={'detail': 'Operation completed successfully'})
                
                # Non-retryable client errors: return immediately
                if response.status_code == 401:
                    return APIError("Authentication failed - token may be invalid or expired")
                if response.status_code == 403:
                    return APIError("Insufficient permissions to access this resource")
                if response.status_code == 404:
                    return APIError(f"Resource not found: {api_request.endpoint}")
                if response.status_code == 400:
                    try:
                        error_data = response.json()
                        error_message = error_data.get('detail', str(error_data))
                    except ValueError:
                        error_message = response.text
                    return APIError(f"Bad request: {error_message}")
                
                # Retryable status: 429, 502, 503, 504
                if response.status_code in (429, 502, 503, 504):
                    try:
                        err_data = response.json()
                        msg = err_data.get('detail', str(err_data))
                    except ValueError:
                        msg = f"API error: {response.status_code} - {response.text}"
                    last_error = APIError(msg)
                    if attempt < constants.REQUEST_RETRY_ATTEMPTS - 1:
                        delay = constants.REQUEST_RETRY_BACKOFF_SECONDS * (2 ** attempt)
                        self.logger.warning(
                            "API %s %s (attempt %s/%s), retrying in %.1fs: %s",
                            response.status_code, api_request.endpoint,
                            attempt + 1, constants.REQUEST_RETRY_ATTEMPTS, delay, msg
                        )
                        time.sleep(delay)
                        continue
                    return last_error
                
                # Other server/client errors: do not retry
                try:
                    error_data = response.json()
                    error_message = error_data.get('detail', f"API error: {response.status_code}")
                except ValueError:
                    error_message = f"API error: {response.status_code} - {response.text}"
                return APIError(error_message)
            
            except requests.Timeout:
                last_error = APIError(constants.API_TIMEOUT_MESSAGE.format(timeout=timeout_seconds))
                if attempt < constants.REQUEST_RETRY_ATTEMPTS - 1:
                    delay = constants.REQUEST_RETRY_BACKOFF_SECONDS * (2 ** attempt)
                    self.logger.warning("API timeout (attempt %s/%s), retrying in %.1fs", attempt + 1, constants.REQUEST_RETRY_ATTEMPTS, delay)
                    time.sleep(delay)
                else:
                    return last_error
            except requests.ConnectionError:
                last_error = APIError(constants.API_FAILED_CONNECT.format(base_url=self.api_base_url))
                if attempt < constants.REQUEST_RETRY_ATTEMPTS - 1:
                    delay = constants.REQUEST_RETRY_BACKOFF_SECONDS * (2 ** attempt)
                    self.logger.warning("API connection error (attempt %s/%s), retrying in %.1fs", attempt + 1, constants.REQUEST_RETRY_ATTEMPTS, delay)
                    time.sleep(delay)
                else:
                    return last_error
            except requests.RequestException as e:
                return APIError(f"API request failed: {str(e)}")
            except Exception as e:
                return APIError(f"Unexpected error during API call: {str(e)}")
        
        return last_error or APIError("API call failed after retries")

    def get_full_url(self, url: str, timeout: int = None):
        """
        Perform GET request to an absolute URL (e.g. pagination 'next' link).
        Uses same session and auth headers as other calls. Retries on timeout,
        connection error, and 429/502/503/504 (see REQUEST_RETRY_* in constants).

        Args:
            url: Full URL (e.g. https://api.example.com/users/?page=2)
            timeout: Request timeout in seconds (default: constants.REQUEST_TIMEOUT)

        Returns:
            APIResponse with parsed JSON data, or APIError on failure.
        """
        if not url or not url.startswith(("http://", "https://")):
            return APIError("Invalid or missing URL for get_full_url")
        timeout_seconds = timeout if timeout is not None else constants.REQUEST_TIMEOUT
        last_error = None
        for attempt in range(constants.REQUEST_RETRY_ATTEMPTS):
            try:
                response = self.session.get(url, timeout=timeout_seconds)
                if response.status_code in [200, 201]:
                    try:
                        data = response.json() if response.content else {}
                    except ValueError:
                        data = {"response": response.text}
                    return APIResponse(status_code=response.status_code, data=data)
                if response.status_code == 404:
                    return APIError("Resource not found")
                if response.status_code in (429, 502, 503, 504):
                    try:
                        err = response.json()
                        msg = err.get("detail", str(err))
                    except ValueError:
                        msg = response.text or f"API error: {response.status_code}"
                    last_error = APIError(msg)
                    if attempt < constants.REQUEST_RETRY_ATTEMPTS - 1:
                        delay = constants.REQUEST_RETRY_BACKOFF_SECONDS * (2 ** attempt)
                        self.logger.warning(
                            "get_full_url %s (attempt %s/%s), retrying in %.1fs: %s",
                            response.status_code, attempt + 1, constants.REQUEST_RETRY_ATTEMPTS, delay, msg
                        )
                        time.sleep(delay)
                        continue
                    return last_error
                try:
                    err = response.json()
                    msg = err.get("detail", str(err))
                except ValueError:
                    msg = response.text or f"API error: {response.status_code}"
                return APIError(msg)
            except requests.Timeout:
                last_error = APIError(constants.API_TIMEOUT_MESSAGE.format(timeout=timeout_seconds))
                if attempt < constants.REQUEST_RETRY_ATTEMPTS - 1:
                    delay = constants.REQUEST_RETRY_BACKOFF_SECONDS * (2 ** attempt)
                    self.logger.warning("get_full_url timeout (attempt %s/%s), retrying in %.1fs", attempt + 1, constants.REQUEST_RETRY_ATTEMPTS, delay)
                    time.sleep(delay)
                else:
                    return last_error
            except requests.ConnectionError:
                last_error = APIError(constants.API_FAILED_CONNECT.format(base_url=url))
                if attempt < constants.REQUEST_RETRY_ATTEMPTS - 1:
                    delay = constants.REQUEST_RETRY_BACKOFF_SECONDS * (2 ** attempt)
                    self.logger.warning("get_full_url connection error (attempt %s/%s), retrying in %.1fs", attempt + 1, constants.REQUEST_RETRY_ATTEMPTS, delay)
                    time.sleep(delay)
                else:
                    return last_error
            except requests.RequestException as e:
                return APIError(f"Request failed: {str(e)}")
        return last_error or APIError("Request failed after retries")