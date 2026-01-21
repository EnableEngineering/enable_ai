import requests
from .types import APIRequest, APIResponse, APIError


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
        
        # Set authorization header if token provided
        if access_token:
            self.session.headers.update({
                'Authorization': f'Bearer {access_token}',
                'Content-Type': 'application/json'
            })

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
        
        try:
            url = f"{self.api_base_url}{api_request.endpoint}"
            method = api_request.method.upper()
            
            # Make the API call based on HTTP method
            if method == 'GET':
                response = self.session.get(url, params=api_request.params, timeout=30)
            elif method == 'POST':
                response = self.session.post(url, json=api_request.params, timeout=30)
            elif method == 'PUT':
                response = self.session.put(url, json=api_request.params, timeout=30)
            elif method == 'PATCH':
                response = self.session.patch(url, json=api_request.params, timeout=30)
            elif method == 'DELETE':
                response = self.session.delete(url, params=api_request.params, timeout=30)
            else:
                return APIError(f"Unsupported HTTP method: {method}")
            
            # Handle different response status codes
            if response.status_code in [200, 201]:
                # Successful response with data
                try:
                    data = response.json() if response.content else {}
                except ValueError:
                    data = {"response": response.text}
                
                return APIResponse(
                    status_code=response.status_code,
                    data=data
                )
            elif response.status_code == 204:
                # Successful response with no content
                return APIResponse(
                    status_code=response.status_code,
                    data={'detail': 'Operation completed successfully'}
                )
            elif response.status_code == 401:
                return APIError("Authentication failed - token may be invalid or expired")
            elif response.status_code == 403:
                return APIError("Insufficient permissions to access this resource")
            elif response.status_code == 404:
                return APIError(f"Resource not found: {api_request.endpoint}")
            elif response.status_code == 400:
                try:
                    error_data = response.json()
                    error_message = error_data.get('detail', str(error_data))
                except ValueError:
                    error_message = response.text
                return APIError(f"Bad request: {error_message}")
            else:
                # Other error responses
                try:
                    error_data = response.json()
                    error_message = error_data.get('detail', f"API error: {response.status_code}")
                except ValueError:
                    error_message = f"API error: {response.status_code} - {response.text}"
                return APIError(error_message)
            
        except requests.Timeout:
            return APIError("API request timed out after 30 seconds")
        except requests.ConnectionError:
            return APIError(f"Failed to connect to API at {self.api_base_url}")
        except requests.RequestException as e:
            return APIError(f"API request failed: {str(e)}")
        except Exception as e:
            return APIError(f"Unexpected error during API call: {str(e)}")