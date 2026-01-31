from .types import APIRequest, APIError, MissingInformation
from .utils import setup_logger
from . import constants


class APIMatcher:
    def __init__(self):
        """
        Initialize the API matcher.
        
        Note: API schema is passed to match_api() method, not during initialization.
        This ensures the matcher only works with schema data, not the original data source.
        """
        self.logger = setup_logger('enable_ai.matcher')
        self.logger.info("APIMatcher initialized")

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
            filters = parsed_input.get('filters', {}) or {}
            original_input = (parsed_input.get('original_input') or "").lower()
            
            self.logger.debug(
                f"Matching: intent={intent}, resource={resource}, "
                f"entities={list(entities.keys())}, filters={list(filters.keys())}"
            )
            
            # Get resources and optional resource-level hints from schema
            resources = api_schema.get('resources', {})
            resource_hints = api_schema.get('resource_hints', {}) or {}
            
            if not resources:
                self.logger.error(constants.ERROR_NO_RESOURCES_IN_SCHEMA)
                return APIError(constants.ERROR_NO_RESOURCES_IN_SCHEMA)

            # If the parser returned a generic/high-level resource (e.g. "inventory"),
            # try to refine it to a more specific one using resource_hints and
            # __resource_synonyms__ (e.g. "inventory-equipment" vs "inventory-consumables").
            #
            # This is fully config-driven: you control which child resources exist
            # and which nouns map to them in config.json. We simply look for a
            # child resource whose synonyms appear in the original input.
            if resource and resource_hints and resources:
                parsed_res_l = resource.lower()
                original_tokens = set(
                    t for t in original_input.replace('/', ' ').replace('-', ' ').split() if t
                )
                best_child = None
                best_hits = 0

                for rh_name, rh_data in resource_hints.items():
                    rh_name_l = str(rh_name or "").lower()
                    # Look for "namespace-child" style resources such as
                    # "inventory-equipment" when parsed_res_l == "inventory".
                    if not rh_name_l.startswith(parsed_res_l + "-"):
                        continue

                    syns = rh_data.get("__resource_synonyms__") or []
                    if not isinstance(syns, (list, tuple)):
                        syns = [syns]

                    hits = 0
                    for s in syns:
                        s_norm = str(s).lower()
                        if not s_norm:
                            continue
                        variants = {s_norm}
                        if s_norm.endswith('s'):
                            variants.add(s_norm[:-1])
                        else:
                            variants.add(s_norm + 's')

                        if any(v in original_tokens for v in variants):
                            hits += 1

                    if hits > best_hits:
                        best_hits = hits
                        best_child = rh_name

                # Only refine if we found a clearly better child resource whose
                # synonyms actually appear in the query.
                if best_child and best_hits > 0:
                    self.logger.debug(
                        f"Refining parsed resource from '{resource}' to '{best_child}' "
                        f"based on __resource_synonyms__ and user input."
                    )
                    resource = best_child
            
            # Search through all resources and endpoints
            # When a resource has multiple read endpoints (e.g. service-orders list vs
            # service-orders/customer-workflow-configs), prefer the one whose path is
            # "list this resource" — i.e. last path segment equals resource name.
            candidates = []
            for resource_name, resource_data in resources.items():
                # Check if resource name matches (allow service_orders <-> service-orders)
                rn = (resource_name or '').lower().replace('-', '_')
                rr = (resource or '').lower().replace('-', '_')
                if resource and rn != rr:
                    continue

                endpoints = resource_data.get('endpoints', [])
                for endpoint_data in endpoints:
                    if self._is_match(endpoint_data, intent):
                        candidates.append((resource_name, endpoint_data))

            matched_resource_name = None
            matched_endpoint = None
            if candidates:
                # Prefer endpoint whose path ends with /{resource}/ (the "list self" endpoint)
                def path_is_list_self(res_name, ep):
                    p = (ep.get('path') or '').rstrip('/')
                    if not p:
                        return False
                    last = p.split('/')[-1].lower().replace('-', '_')
                    r = (res_name or '').lower().replace('-', '_')
                    return last == r

                # Heuristic scoring to choose best endpoint when multiple match:
                # - Use resource_hints["__resource_synonyms__"] (configured in config.json)
                #   to bias endpoints whose path matches domain nouns from the original query.
                # - Give a small bonus when endpoint query params match parsed filters.
                # - Prefer "list self" paths.
                # - Finally fall back to the first candidate.
                #
                # NOTE: We deliberately support simple plural/singular matching for
                # resource synonyms (e.g. "equipments" in the user query should
                # still match an `/equipment/` endpoint path). This keeps the
                # behaviour dynamic and configuration‑driven instead of hardcoding
                # specific domain nouns like "equipment" or "consumables".
                def endpoint_score(res_name, ep):
                    score = 0
                    path = (ep.get('path') or '').lower()
                    # Pre-tokenize for smarter synonym matching
                    original_tokens = set(
                        t for t in original_input.replace('/', ' ').replace('-', ' ').split() if t
                    )
                    path_segments = [seg for seg in path.strip('/').split('/') if seg]

                    # Determine a simple "namespace" for resource hints, so that
                    # hints like "inventory-equipment" are only used when the
                    # parsed resource is "inventory" (and similar patterns).
                    parsed_resource = (resource or "").lower()

                    # Domain-word hints from config-driven resource_hints.
                    # This keeps the matcher generic: you configure which nouns
                    # correspond to which resources/sub-resources in config.json.
                    for rh_name, rh_data in resource_hints.items():
                        rh_name_l = str(rh_name or "").lower()

                        # If the parsed resource is present, prefer hints that
                        # are clearly "under" that namespace, e.g.
                        #   resource="inventory" → use "inventory-equipment", "inventory-consumables"
                        # This keeps cross-domain hints (like service orders vs
                        # inventory) from interfering with each other.
                        if parsed_resource and not (
                            rh_name_l == parsed_resource
                            or rh_name_l.startswith(parsed_resource + "-")
                        ):
                            # Still allow a tiny weight for global hints, but the
                            # main bias will come from the namespace‑aligned ones.
                            namespace_weight = 0.5
                        else:
                            namespace_weight = 1.0

                        syns = rh_data.get("__resource_synonyms__") or []
                        if not isinstance(syns, (list, tuple)):
                            syns = [syns]
                        for s in syns:
                            s_norm = str(s).lower()
                            if not s_norm:
                                continue
                            # Build simple plural/singular variants, so that
                            # "equipments" in the query still matches an
                            # `/equipment/` path segment and vice versa.
                            variants = {s_norm}
                            if s_norm.endswith('s'):
                                variants.add(s_norm[:-1])
                            else:
                                variants.add(s_norm + 's')

                            # Check if any variant is explicitly mentioned in the
                            # user input (token based, to avoid substring noise).
                            if not any(v in original_tokens for v in variants):
                                continue

                            # Now check if any variant appears in a path segment
                            # (again allowing plural/singular variants).
                            path_hit = False
                            for seg in path_segments:
                                seg_l = seg.lower()
                                if any(v == seg_l or v in seg_l for v in variants):
                                    path_hit = True
                                    break

                            if path_hit:
                                # Strong bias towards endpoints whose path
                                # aligns with the domain noun used in the query.
                                score += int(10 * namespace_weight)

                    # If filters reference fields that appear as query params on the endpoint,
                    # give a small bonus (helps differentiate similar inventory endpoints).
                    endpoint_params = (ep.get('parameters', {}) or {}).get('query', [])
                    endpoint_param_names = set()
                    for p in endpoint_params:
                        if isinstance(p, dict) and p.get('name'):
                            endpoint_param_names.add(p['name'])
                        elif isinstance(p, str):
                            endpoint_param_names.add(p)
                    for f_name in filters.keys():
                        if f_name in endpoint_param_names:
                            score += 2

                    # Existing "list self" preference
                    if path_is_list_self(res_name, ep):
                        score += 5

                    return score

                # Pick the candidate with the highest score
                best = None
                best_score = float("-inf")
                for rn, ep in candidates:
                    s = endpoint_score(rn, ep)
                    self.logger.debug(
                        f"Candidate endpoint score: resource={rn}, path={ep.get('path')}, score={s}"
                    )
                    if s > best_score:
                        best_score = s
                        best = (rn, ep)

                if best is not None:
                    matched_resource_name, matched_endpoint = best
                else:
                    matched_resource_name, matched_endpoint = candidates[0]

            if not matched_endpoint:
                return APIError(constants.ERROR_NO_MATCHING_API.format(intent=intent, resource=resource))
            
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

    def _get_path_params(self, endpoint_data):
        """Return path params as dict name -> info. Supports converter format (parameters.path list)."""
        out = endpoint_data.get('path_parameters')
        if isinstance(out, dict):
            return out
        params = endpoint_data.get('parameters', {})
        path_list = params.get('path', []) if isinstance(params, dict) else []
        return {p.get('name'): p for p in path_list if isinstance(p, dict) and p.get('name')} or {}

    def _get_query_params(self, endpoint_data):
        """Return query params as dict name -> info. Supports converter format (parameters.query list)."""
        out = endpoint_data.get('query_parameters')
        if isinstance(out, dict):
            return out
        params = endpoint_data.get('parameters', {})
        query_list = params.get('query', []) if isinstance(params, dict) else []
        return {p.get('name'): p for p in query_list if isinstance(p, dict) and p.get('name')} or {}
    
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
        path_params = self._get_path_params(endpoint_data)
        for param_name, param_info in path_params.items():
            param_desc = param_info if isinstance(param_info, str) else param_info.get('description', param_name)
            if 'required' in param_desc.lower() or method in ['GET', 'PUT', 'PATCH', 'DELETE']:
                # Path parameters in these methods are typically required
                if param_name not in entities:
                    missing_fields.append(param_name)
                    missing_descriptions.append(self._generate_question_for_field(param_name, param_desc))
        
        # Check required query parameters (for GET/DELETE)
        if method in ['GET', 'DELETE']:
            query_params = self._get_query_params(endpoint_data)
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
                message = constants.API_MATCHER_NEED_INFO_PREFIX + "\n".join(f"- {desc}" for desc in missing_descriptions)
            
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
        path_params = self._get_path_params(endpoint_data)
        for param_name in path_params.keys():
            if param_name in entities:
                path = path.replace(f'{{{param_name}}}', str(entities[param_name]))
        
        # Build query parameters or request body
        params = {}
        
        if method in ['GET', 'DELETE']:
            # For GET/DELETE, use query_parameters (or parameters.query from converter)
            query_params = self._get_query_params(endpoint_data)
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