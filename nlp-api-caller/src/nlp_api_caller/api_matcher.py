from .types import APIRequest, APIError, MissingInformation


class APIMatcher:
    def __init__(self):
        """
        Initialize the API matcher.
        
        Note: API schema is passed to match_api() method, not during initialization.
        This ensures the matcher only works with schema data, not the original data source.
        """
        pass

    def match_api(self, parsed_input, api_schema):
        """
        Match the parsed input against the API schema.
        Validates if all required information is present, otherwise returns MissingInformation.
        
        Args:
            parsed_input: Structured input dict with 'intent' and 'entities'
            api_schema: API schema dictionary with 'resources' format
            
        Returns:
            APIRequest: If a matching API is found and all required info is present
            MissingInformation: If matched but missing required parameters
            APIError: If no match is found or matching fails
        """
        if isinstance(parsed_input, APIError):
            return parsed_input
        
        try:
            intent = parsed_input.get('intent', '').lower()
            resource = parsed_input.get('resource', '').lower()
            entities = parsed_input.get('entities', {})
            
            # Get resources from schema
            resources = api_schema.get('resources', {})
            
            if not resources:
                return APIError("No resources found in API schema")
            
            # Search through all resources and endpoints
            matched_endpoint = None
            matched_resource_name = None
            
            for resource_name, resource_data in resources.items():
                # Check if resource name matches
                if resource and resource_name.lower() != resource:
                    continue
                    
                endpoints = resource_data.get('endpoints', [])
                
                for endpoint_data in endpoints:
                    if self._is_match(endpoint_data, intent):
                        matched_endpoint = endpoint_data
                        matched_resource_name = resource_name
                        break
                
                if matched_endpoint:
                    break
            
            if not matched_endpoint:
                return APIError(f"No matching API found for intent: {intent}, resource: {resource}")
            
            # Validate if all required information is present
            validation_result = self._validate_required_fields(matched_endpoint, entities, intent)
            
            if isinstance(validation_result, MissingInformation):
                return validation_result
            
            # Build and return the API request
            return self._build_api_request(matched_endpoint, entities)
            
        except Exception as e:
            return APIError(f"API matching failed: {str(e)}")

    def _is_match(self, endpoint_data, intent):
        """
        Check if an endpoint matches the user intent.
        
        Args:
            endpoint_data: Endpoint definition from API schema
            intent: User intent string (e.g., "read", "create", "update", "delete")
            
        Returns:
            bool: True if the endpoint matches the intent
        """
        # Check direct intent field (e.g., endpoint has "intent": "read")
        endpoint_intent = endpoint_data.get('intent', '').lower()
        if endpoint_intent and endpoint_intent == intent.lower():
            return True
        
        # Check against intent keywords (legacy support)
        intent_keywords = endpoint_data.get('intent_keywords', [])
        
        for keyword in intent_keywords:
            if keyword.lower() in intent:
                return True
        
        # Check against endpoint description
        description = endpoint_data.get('description', '').lower()
        intent_words = intent.split()
        
        # Check if any significant words from intent appear in description
        for word in intent_words:
            if len(word) > 3 and word in description:  # Skip short words like "the", "and"
                return True
        
        return False
    
    def _validate_required_fields(self, endpoint_data, entities, intent):
        """
        Validate if all required fields are present for the matched endpoint.
        
        Args:
            endpoint_data: Matched endpoint definition
            entities: Extracted entities from user input
            intent: User's intent string
            
        Returns:
            None: If all required fields are present
            MissingInformation: If some required fields are missing
        """
        missing_fields = []
        missing_descriptions = []
        method = endpoint_data.get('method', 'GET')
        
        # Check required path parameters
        path_params = endpoint_data.get('path_parameters', {})
        for param_name, param_info in path_params.items():
            param_desc = param_info if isinstance(param_info, str) else param_info.get('description', param_name)
            if 'required' in param_desc.lower() or method in ['GET', 'PUT', 'PATCH', 'DELETE']:
                # Path parameters in these methods are typically required
                if param_name not in entities:
                    missing_fields.append(param_name)
                    missing_descriptions.append(self._generate_question_for_field(param_name, param_desc))
        
        # Check required query parameters (for GET/DELETE)
        if method in ['GET', 'DELETE']:
            query_params = endpoint_data.get('query_parameters', {})
            for param_name, param_info in query_params.items():
                param_desc = param_info if isinstance(param_info, str) else param_info.get('description', param_name)
                if 'required' in param_desc.lower():
                    if param_name not in entities:
                        missing_fields.append(param_name)
                        missing_descriptions.append(self._generate_question_for_field(param_name, param_desc))
        
        # Check required request body fields (for POST/PUT/PATCH)
        if method in ['POST', 'PUT', 'PATCH']:
            request_body = endpoint_data.get('request_body', {})
            if isinstance(request_body, dict):
                for field_name, field_info in request_body.items():
                    field_desc = field_info if isinstance(field_info, str) else field_info.get('description', field_name)
                    if 'required' in field_desc.lower():
                        if field_name not in entities:
                            missing_fields.append(field_name)
                            missing_descriptions.append(self._generate_question_for_field(field_name, field_desc))
        
        # If there are missing fields, return MissingInformation
        if missing_fields:
            if len(missing_fields) == 1:
                message = missing_descriptions[0]
            else:
                message = "I need some additional information:\n" + "\n".join(f"- {desc}" for desc in missing_descriptions)
            
            return MissingInformation(
                message=message,
                missing_fields=missing_fields,
                matched_endpoint=endpoint_data,
                context={'intent': intent, 'entities': entities}
            )
        
        return None
    
    def _generate_question_for_field(self, field_name, field_description):
        """
        Generate a natural language question for a missing field.
        
        Args:
            field_name: Name of the missing field
            field_description: Description of the field
            
        Returns:
            str: Natural language question
        """
        # Map common field names to natural questions
        question_templates = {
            'id': "Which {entity} ID would you like to work with?",
            'user_id': "Which user ID should I use?",
            'customer_id': "Which customer ID should I use?",
            'product_id': "Which product ID should I use?",
            'order_id': "Which order ID should I use?",
            'name': "What name should I use?",
            'email': "What email address should I use?",
            'status': "What status should I set? (e.g., active, inactive, pending)",
            'date': "What date should I use? (format: YYYY-MM-DD)",
            'start_date': "What is the start date? (format: YYYY-MM-DD)",
            'end_date': "What is the end date? (format: YYYY-MM-DD)",
            'description': "Please provide a description.",
            'quantity': "What quantity should I use?",
            'price': "What price should I set?",
        }
        
        # Try to find a matching template
        field_lower = field_name.lower()
        if field_lower in question_templates:
            return question_templates[field_lower].format(entity=field_name.replace('_', ' '))
        
        # Generate a generic question based on the field description
        if 'id' in field_lower:
            return f"What is the {field_name.replace('_', ' ')}?"
        else:
            return f"Please provide the {field_name.replace('_', ' ')}."
    
    def _build_api_request(self, endpoint_data, entities):
        """
        Build APIRequest from matched endpoint and extracted entities.
        
        Args:
            endpoint_data: Matched endpoint definition
            entities: Extracted entities from user input
            
        Returns:
            APIRequest: Constructed API request
        """
        method = endpoint_data.get('method', 'GET')
        path = endpoint_data.get('path', '')
        
        # Replace path parameters with actual values from entities
        path_params = endpoint_data.get('path_parameters', {})
        for param_name in path_params.keys():
            if param_name in entities:
                path = path.replace(f'{{{param_name}}}', str(entities[param_name]))
        
        # Build query parameters or request body
        params = {}
        
        if method in ['GET', 'DELETE']:
            # For GET/DELETE, use query_parameters
            query_params = endpoint_data.get('query_parameters', {})
            for param_name in query_params.keys():
                if param_name in entities:
                    params[param_name] = entities[param_name]
        else:
            # For POST/PUT/PATCH, use request_body
            request_body = endpoint_data.get('request_body', {})
            if isinstance(request_body, dict):
                for field_name in request_body.keys():
                    if field_name in entities:
                        params[field_name] = entities[field_name]
            else:
                # If request_body is a string description, use all entities
                params = entities.copy()
        
        return APIRequest(
            endpoint=path,
            params=params,
            method=method,
            authentication_required=endpoint_data.get('authentication_required', True)
        )