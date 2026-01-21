class APIRequest:
    def __init__(self, endpoint: str, params: dict, method: str = 'GET', authentication_required: bool = True):
        self.endpoint = endpoint
        self.params = params
        self.method = method
        self.authentication_required = authentication_required

class APIResponse:
    def __init__(self, status_code: int, data: dict):
        self.status_code = status_code
        self.data = data

class APIError:
    def __init__(self, message: str):
        self.message = message

class MissingInformation:
    """
    Represents a request for additional information from the user.
    Used when the user's input is incomplete and we need clarification.
    """
    def __init__(self, message: str, missing_fields: list, matched_endpoint: dict = None, context: dict = None):
        """
        Args:
            message: Human-readable question to ask the user
            missing_fields: List of field names that are required but missing
            matched_endpoint: The endpoint that was matched (for context)
            context: Current parsed context (intent, entities extracted so far)
        """
        self.message = message
        self.missing_fields = missing_fields
        self.matched_endpoint = matched_endpoint
        self.context = context or {}
    
    def __repr__(self):
        return f"MissingInformation(message='{self.message}', missing_fields={self.missing_fields})"