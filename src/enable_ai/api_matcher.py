from typing import Dict, Any, Optional, List
from .types import APIRequest, APIError, MissingInformation
from .utils import setup_logger
from .query_execution import (
    dedupe_fk_lookup_filters,
    format_sort_param,
    split_filters_for_endpoint,
)
from .semantic_filters import inject_semantic_filters
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
            question_type = (parsed_input.get('question_type') or '').lower()
            
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

            # v0.3.41: Use resource_hints to resolve resource name via synonyms
            # This handles cases where parser returns "company" but schema has "companies"
            # or parser returns "flash report" but schema has "flash-reports"
            self.logger.info(
                f"v0.3.42 debug: resource='{resource}', resource_hints={bool(resource_hints)}, "
                f"hints_count={len(resource_hints) if resource_hints else 0}"
            )
            if resource and resource_hints:
                parsed_res_l = resource.lower().replace('-', ' ').replace('_', ' ')
                original_tokens = set(
                    t for t in original_input.replace('/', ' ').replace('-', ' ').split() if t
                )

                # First, try to find exact match or synonym match for the parsed resource
                best_match = None
                best_score = 0

                for rh_name, rh_data in resource_hints.items():
                    if not isinstance(rh_data, dict):
                        continue
                    rh_name_l = str(rh_name or "").lower()
                    rh_name_normalized = rh_name_l.replace('-', '_')
                    parsed_normalized = parsed_res_l.replace('-', '_').replace(' ', '_')

                    # Check if this resource_hint matches the parsed resource
                    score = 0

                    # Exact match (with normalization)
                    if rh_name_normalized == parsed_normalized:
                        score = 100

                    # Check __resource_synonyms__
                    syns = rh_data.get("__resource_synonyms__") or []
                    if not isinstance(syns, (list, tuple)):
                        syns = [syns]

                    for s in syns:
                        s_norm = str(s).lower().replace('-', ' ').replace('_', ' ')
                        if not s_norm:
                            continue

                        # Build variants (singular/plural)
                        variants = {s_norm, s_norm.replace(' ', '_'), s_norm.replace(' ', '-')}
                        if s_norm.endswith('s'):
                            singular = s_norm[:-1]
                            variants.add(singular)
                            variants.add(singular.replace(' ', '_'))
                        else:
                            plural = s_norm + 's'
                            variants.add(plural)
                            variants.add(plural.replace(' ', '_'))

                        # Check if parsed resource matches any variant
                        if parsed_res_l in variants or parsed_normalized in variants:
                            score = max(score, 90)

                        # Check if variant appears in original input
                        if any(v in original_tokens for v in variants):
                            score = max(score, 80)

                    if score > best_score:
                        best_score = score
                        best_match = rh_name

                # If we found a better match, use it
                if best_match and best_score >= 80:
                    self.logger.info(
                        f"Resolved resource '{resource}' to '{best_match}' "
                        f"via __resource_synonyms__ (score={best_score})"
                    )
                    resource = best_match

            # Also handle "namespace-child" style resources
            # (e.g. "inventory" + "equipment" in query → "inventory-equipment")
            if resource and resource_hints and resources:
                parsed_res_l = resource.lower()
                original_tokens = set(
                    t for t in original_input.replace('/', ' ').replace('-', ' ').split() if t
                )
                best_child = None
                best_hits = 0

                for rh_name, rh_data in resource_hints.items():
                    if not isinstance(rh_data, dict):
                        continue
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
            
            # v0.3.42: Log resource matching for debugging
            self.logger.info(
                f"Looking for resource '{resource}' in schema with {len(resources)} resources. "
                f"resource_hints available: {bool(resource_hints)}, hint_keys: {list(resource_hints.keys())[:5]}"
            )

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
                    # v0.3.42: Log skipped resources for debugging
                    self.logger.debug(f"Skipping resource '{resource_name}' (rn={rn}, rr={rr})")
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
                    ep_path_params = self._get_path_params(ep)
                    if ep_path_params and entities:
                        required = list(ep_path_params.keys())
                        if required and all(p in entities for p in required):
                            score += 25
                        elif any(p in entities for p in required):
                            score -= 10
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
                        if not isinstance(rh_data, dict):
                            continue
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

                    # Endpoint role routing (list vs summary/dashboard)
                    from .hint_utils import get_endpoint_role
                    role = get_endpoint_role(res_name, resource_hints)
                    if role in ("summary", "dashboard", "metrics"):
                        if question_type in ("count", "list"):
                            score -= 30
                        elif question_type in ("summary", "aggregate_metric"):
                            score += 20
                    elif role == "list":
                        if question_type in ("count", "list"):
                            score += 15
                        elif question_type in ("summary", "aggregate_metric"):
                            score -= 10
                    elif question_type in ("count", "list"):
                        if "dashboard" in path or path.rstrip("/").endswith("-dashboard"):
                            score -= 25
                        elif "/invoices" in path or path.rstrip("/").endswith("/invoices"):
                            score += 10

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
                # v0.3.44: Enhanced error handling with helpful suggestions
                available_resources = list(resources.keys())
                self.logger.warning(
                    f"No endpoint found for resource='{resource}', intent='{intent}'. "
                    f"Available resources: {available_resources[:10]}"
                )
                # Generate helpful error message with suggestions
                error_msg = self._generate_helpful_error(
                    resource, intent, available_resources, resource_hints
                )
                return APIError(error_msg)
            
            # Validate if all required information is present
            validation_result = self._validate_required_fields(matched_endpoint, entities, intent)
            
            if isinstance(validation_result, MissingInformation):
                return validation_result

            # v0.3.37: Collect filter warnings to pass to user
            filter_warnings = []

            filters = inject_semantic_filters(
                filters or {}, resource, original_input, resource_hints,
            )

            # Validate filter values against schema
            if filters:
                validated = self._validate_filter_values(filters, resource, resource_hints)
                if validated.get('filters'):
                    filters = dedupe_fk_lookup_filters(validated['filters'])
                if validated.get('warnings'):
                    for warning in validated['warnings']:
                        self.logger.warning(warning)
                        filter_warnings.append(warning)

                # Split filters: server-side params vs client-side post-filter
                server_filters, client_filters, split_warnings = split_filters_for_endpoint(
                    filters, matched_endpoint, resource=resource, resource_hints=resource_hints,
                )
                filter_warnings.extend(split_warnings)
                filters = server_filters
            else:
                client_filters = {}

            # Build and return the API request with validated filters and warnings
            return self._build_api_request(
                matched_endpoint, entities,
                filters=filters,
                schema=api_schema,
                resource=resource,
                warnings=filter_warnings,
                parsed_input=parsed_input,
                client_side_filters=client_filters,
            )

        except Exception as e:
            return APIError(f"API matching failed: {str(e)}")

    def build_query_params(
        self,
        endpoint_data: Dict[str, Any],
        parsed: Dict[str, Any],
        schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build server-side filter query parameters from parsed filters.

        Maps parsed filter names to actual API query parameter names,
        validates values against schema constraints, and applies synonyms.

        Args:
            endpoint_data: Matched endpoint definition
            parsed: Parsed query with filters
            schema: Full API schema

        Returns:
            Dict of query parameters to send to the API
        """
        params = {}
        filters = parsed.get('filters', {})
        resource = parsed.get('resource', '')
        resource_hints = schema.get('resource_hints', {}).get(resource, {})

        # Get endpoint's supported query params
        query_params = self._get_query_params(endpoint_data)
        available_params = set(query_params.keys())

        # 1. Handle search/q parameter
        search_term = parsed.get('search_term') or parsed.get('q')
        if search_term:
            if 'search' in available_params:
                params['search'] = search_term
            elif 'q' in available_params:
                params['q'] = search_term

        # 2. Map each filter to best matching API param
        for field, filter_val in filters.items():
            # Extract value and operator
            if isinstance(filter_val, dict):
                value = filter_val.get('value')
                operator = filter_val.get('operator', 'equals')
            else:
                value = filter_val
                operator = 'equals'

            # v0.3.41: Build the full param name WITH operator for Django-style lookups
            # e.g., {current_quantity: {operator: "lt", value: 10}} → "current_quantity__lt=10"
            if operator and operator != 'equals':
                # Django lookup operators: lt, lte, gt, gte, contains, icontains, in, etc.
                full_field = f"{field}__{operator}"
            else:
                full_field = field

            # Check if the full field (with operator) exists in available params
            if full_field in available_params:
                params[full_field] = value
                self.logger.debug(f"Using exact param match: {full_field}={value}")
            else:
                # Try to find best match for base field
                param_name = self._find_best_param_match(field, available_params)
                if param_name:
                    # If operator is not 'equals', try to append it
                    if operator and operator != 'equals':
                        param_with_op = f"{param_name}__{operator}"
                        if param_with_op in available_params:
                            params[param_with_op] = value
                        else:
                            # Fallback: use param with operator even if not in schema
                            # (API might support it even if not documented)
                            params[param_with_op] = value
                            self.logger.debug(
                                f"Using operator param (not in schema): {param_with_op}={value}"
                            )
                    else:
                        params[param_name] = value
                else:
                    # Use field name as-is (with operator if applicable)
                    params[full_field] = value
                    self.logger.debug(f"Using filter as-is: {full_field}={value}")

        # 3. Handle field selection (if API supports it)
        requested_fields = parsed.get('fields', [])
        if requested_fields and 'fields' in available_params:
            params['fields'] = ','.join(requested_fields)

        # 4. Handle sorting (dict -> API ordering string, e.g. "-created_at")
        sort_value = format_sort_param(parsed.get('sort'))
        if sort_value:
            if 'ordering' in available_params:
                params['ordering'] = sort_value
            elif 'sort' in available_params:
                params['sort'] = sort_value
            elif 'order_by' in available_params:
                params['order_by'] = sort_value

        # 5. Handle limit (don't add arbitrary limits - only if user specified)
        limit = parsed.get('limit')
        if limit:
            for limit_param in constants.LIMIT_PARAM_NAMES:
                if limit_param in available_params:
                    params[limit_param] = limit
                    break

        return params

    def _find_best_param_match(
        self,
        field: str,
        available_params: set
    ) -> Optional[str]:
        """
        Find the best matching API parameter for a filter field.

        Tries multiple patterns:
        1. Exact match
        2. Django-style lookups (field__name, field__id, etc.)
        3. Case-insensitive match

        Args:
            field: Filter field name
            available_params: Set of available query parameter names

        Returns:
            Best matching parameter name or None
        """
        # 1. Exact match
        if field in available_params:
            return field

        # 2. Try Django-style lookups
        for suffix in constants.DJANGO_LOOKUP_SUFFIXES:
            pattern = f"{field}{suffix}"
            if pattern in available_params:
                return pattern

        # 3. Try without suffix if field has one
        if '__' in field:
            base_field = field.split('__')[0]
            if base_field in available_params:
                return base_field
            # Try with different suffixes
            for suffix in constants.DJANGO_LOOKUP_SUFFIXES:
                pattern = f"{base_field}{suffix}"
                if pattern in available_params:
                    return pattern

        # 4. Case-insensitive match
        field_lower = field.lower()
        for param in available_params:
            if param.lower() == field_lower:
                return param

        return None

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

        result = {}
        for p in query_list:
            if isinstance(p, dict) and p.get('name'):
                result[p['name']] = p
            elif isinstance(p, str):
                # Handle string params like "status__name"
                result[p] = {"name": p, "type": "string"}
        return result
    
    def _validate_filter_values(self, filters: dict, resource: str, resource_hints: dict) -> dict:
        """
        Validate filter values against schema-defined allowed values.
        Apply synonyms if needed, including semantic filter mappings.

        Supports semantic filter mappings like:
            "low stock" -> "current_quantity__lt=10"
        Which transforms the filter from {stock_level: "low"} to {current_quantity: {operator: "lt", value: 10}}

        Args:
            filters: Dict of field -> filter_val (may be {"operator": ..., "value": ...} or raw value)
            resource: Current resource name
            resource_hints: Schema resource hints with allowed values and synonyms

        Returns:
            Dict with validated/transformed filters and any warnings
        """
        validated = {}
        warnings = []
        hints = resource_hints.get(resource, {})

        for field, filter_val in filters.items():
            value = filter_val.get("value") if isinstance(filter_val, dict) else filter_val
            operator = filter_val.get("operator", "equals") if isinstance(filter_val, dict) else "equals"
            field_hints = hints.get(field, {}) if isinstance(hints, dict) else {}

            # v0.3.47: Track whether we applied a semantic filter mapping for THIS field
            applied_semantic_mapping = False

            if isinstance(field_hints, dict) and field_hints:
                allowed_values = field_hints.get("values", [])
                synonyms = field_hints.get("synonyms", {})

                # Check synonyms first (case-insensitive)
                # v0.3.38: Support semantic filter mappings like "low" -> "current_quantity__lt=10"
                if isinstance(synonyms, dict):
                    value_lower = str(value).lower()
                    for syn_key, syn_val in synonyms.items():
                        if value_lower == str(syn_key).lower():
                            self.logger.debug(f"Found synonym mapping: {value} -> {syn_val}")

                            # v0.3.38: Check if this is a semantic filter mapping (field__operator=value)
                            if isinstance(syn_val, str) and '=' in syn_val and '__' in syn_val.split('=')[0]:
                                # Parse semantic filter: "current_quantity__lt=10"
                                new_field_op, new_value = syn_val.split('=', 1)
                                parts = new_field_op.rsplit('__', 1)
                                if len(parts) == 2:
                                    new_field, new_operator = parts
                                    # Try to convert numeric values
                                    try:
                                        new_value = int(new_value)
                                    except ValueError:
                                        try:
                                            new_value = float(new_value)
                                        except ValueError:
                                            pass  # Keep as string

                                    self.logger.info(
                                        f"Applied semantic filter: {field}={value} -> "
                                        f"{new_field} {new_operator} {new_value}"
                                    )
                                    # Replace the field entirely with the new semantic filter
                                    validated[new_field] = {"operator": new_operator, "value": new_value}
                                    # v0.3.47: Mark that we applied semantic mapping
                                    applied_semantic_mapping = True
                                    break
                            else:
                                # Simple value synonym
                                value = syn_val
                            break

                # v0.3.47: If we applied a semantic filter mapping, skip adding original field
                if applied_semantic_mapping:
                    continue

                # Validate against allowed values (case-insensitive check for strings)
                if allowed_values:
                    value_matches = False
                    for av in allowed_values:
                        if isinstance(av, str) and isinstance(value, str):
                            if av.lower() == value.lower():
                                value_matches = True
                                value = av  # Use canonical case
                                break
                        elif av == value:
                            value_matches = True
                            break

                    if not value_matches:
                        warnings.append(f"Value '{value}' for {field} not in allowed values: {allowed_values}")

            validated[field] = {"operator": operator, "value": value}

        return {"filters": validated, "warnings": warnings}

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
            if 'required' in str(param_desc).lower() or method in ['GET', 'PUT', 'PATCH', 'DELETE']:
                # Path parameters in these methods are typically required
                if param_name not in entities:
                    missing_fields.append(param_name)
                    missing_descriptions.append(self._generate_question_for_field(param_name, param_desc))
        
        # Check required query parameters (for GET/DELETE)
        if method in ['GET', 'DELETE']:
            query_params = self._get_query_params(endpoint_data)
            for param_name, param_info in query_params.items():
                param_desc = param_info if isinstance(param_info, str) else param_info.get('description', param_name)
                if 'required' in str(param_desc).lower():
                    if param_name not in entities:
                        missing_fields.append(param_name)
                        missing_descriptions.append(self._generate_question_for_field(param_name, param_desc))
        
        # Check required request body fields (for POST/PUT/PATCH)
        if method in ['POST', 'PUT', 'PATCH']:
            request_body = endpoint_data.get('request_body', {})
            if isinstance(request_body, dict):
                for field_name, field_info in request_body.items():
                    field_desc = field_info if isinstance(field_info, str) else field_info.get('description', field_name)
                    if 'required' in str(field_desc).lower():
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
    
    def _check_unmatched_filters(
        self,
        filters: Dict[str, Any],
        endpoint_data: Dict[str, Any],
        resource: str
    ) -> List[str]:
        """
        Check if any filters don't match available endpoint parameters (v0.3.37).

        This helps users understand when their filter won't be applied.

        Args:
            filters: Parsed filters
            endpoint_data: Matched endpoint definition
            resource: Resource name

        Returns:
            List of warning messages for unmatched filters
        """
        warnings = []

        # Get available query parameters from endpoint
        available_params = set()
        params_def = endpoint_data.get('parameters', {})
        query_params = params_def.get('query', []) if isinstance(params_def, dict) else []

        for param in query_params:
            if isinstance(param, dict):
                name = param.get('name', '')
                if name:
                    available_params.add(name)
                    # Also add base name for Django lookups
                    base = name.split('__')[0]
                    available_params.add(base)
            elif isinstance(param, str):
                available_params.add(param)
                available_params.add(param.split('__')[0])

        # Check each filter
        for field in filters.keys():
            base_field = field.split('__')[0]
            # Check if filter matches any available param
            if field not in available_params and base_field not in available_params:
                # Check if any available param starts with this field
                has_match = any(
                    p.startswith(base_field + '__') or p == base_field
                    for p in available_params
                )
                if not has_match and available_params:
                    # Generate helpful warning
                    sample_params = sorted(list(available_params))[:5]
                    warnings.append(
                        f"Filter '{field}' may not be available for {resource}. "
                        f"Results may not be filtered. Available filters include: {', '.join(sample_params)}"
                    )
                    self.logger.warning(f"Unmatched filter '{field}' for {resource}")

        return warnings

    def _build_api_request(
        self,
        endpoint_data: Dict[str, Any],
        entities: Dict[str, Any],
        filters: Optional[Dict[str, Any]] = None,
        schema: Optional[Dict[str, Any]] = None,
        resource: Optional[str] = None,
        warnings: Optional[List[str]] = None,
        parsed_input: Optional[Dict[str, Any]] = None,
        client_side_filters: Optional[Dict[str, Any]] = None,
    ) -> APIRequest:
        """
        Build APIRequest from matched endpoint and extracted entities/filters.

        Args:
            endpoint_data: Matched endpoint definition
            entities: Extracted entities from user input
            filters: Validated filters (optional, for server-side filtering)
            schema: Full API schema (optional, for build_query_params)
            resource: Resource name (optional, for context)
            warnings: Filter validation warnings to include in request (v0.3.37)

        Returns:
            APIRequest: Constructed API request with warnings
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
            # For GET/DELETE, use build_query_params for server-side filtering
            if schema:
                parsed_for_params = {
                    'filters': filters or {},
                    'resource': resource or '',
                }
                if parsed_input:
                    if parsed_input.get('sort'):
                        parsed_for_params['sort'] = parsed_input['sort']
                    if parsed_input.get('limit') is not None:
                        parsed_for_params['limit'] = parsed_input['limit']
                params = self.build_query_params(endpoint_data, parsed_for_params, schema)

            # Also add direct entity matches not already set by build_query_params
            query_params = self._get_query_params(endpoint_data)
            for param_name in query_params.keys():
                if param_name not in entities or param_name in params:
                    continue
                # Skip base FK when lookup param already set (status vs status__name)
                if "__" not in param_name and any(
                    k.startswith(f"{param_name}__") for k in params
                ):
                    continue
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
            authentication_required=endpoint_data.get('authentication_required', True),
            warnings=warnings or [],
            client_side_filters=client_side_filters or {},
        )

    def _generate_helpful_error(
        self,
        resource: str,
        intent: str,
        available_resources: List[str],
        resource_hints: Dict[str, Any]
    ) -> str:
        """
        Generate a helpful error message when endpoint is not found (v0.3.44).

        Instead of a generic error, suggests similar resources and provides
        examples of valid queries.

        Args:
            resource: The resource that was not found
            intent: User's intent
            available_resources: List of available resource names
            resource_hints: Resource hints from schema

        Returns:
            Helpful error message string
        """
        resource_lower = (resource or '').lower()

        # Find similar resources (substring match or Levenshtein-like)
        similar = []
        for r in available_resources:
            r_lower = r.lower()
            # Check substring match
            if resource_lower in r_lower or r_lower in resource_lower:
                similar.append(r)
                continue
            # Check word overlap
            resource_words = set(resource_lower.replace('-', '_').replace('_', ' ').split())
            r_words = set(r_lower.replace('-', '_').replace('_', ' ').split())
            if resource_words & r_words:
                similar.append(r)

        # Also check resource_hints synonyms for similar resources
        for hint_resource, hints in resource_hints.items():
            if not isinstance(hints, dict):
                continue
            synonyms = hints.get('__resource_synonyms__', [])
            if not isinstance(synonyms, (list, tuple)):
                synonyms = [synonyms]
            for syn in synonyms:
                syn_lower = str(syn).lower()
                if resource_lower in syn_lower or syn_lower in resource_lower:
                    if hint_resource not in similar and hint_resource in available_resources:
                        similar.append(hint_resource)
                        break

        # Build helpful message
        msg_parts = [f"I couldn't find a '{resource}' endpoint in the API."]

        if similar:
            msg_parts.append(f"\nDid you mean: {', '.join(similar[:3])}?")

        # Add examples of valid queries
        msg_parts.append("\n\nAvailable resources you can query:")
        for r in available_resources[:8]:
            # Get synonyms for this resource to show example
            hint = resource_hints.get(r, {})
            synonyms = hint.get('__resource_synonyms__', [])
            if synonyms and isinstance(synonyms, list):
                example_term = synonyms[0] if synonyms else r
            else:
                example_term = r.replace('_', ' ').replace('-', ' ')
            msg_parts.append(f"  • {r} - e.g., \"list {example_term}\"")

        if len(available_resources) > 8:
            msg_parts.append(f"  ...and {len(available_resources) - 8} more")

        # Add tip about checking schema
        msg_parts.append(
            "\n\nTip: The resource may exist under a different name or "
            "may need to be added to your API schema configuration."
        )

        return ''.join(msg_parts)