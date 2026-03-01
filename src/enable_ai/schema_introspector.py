"""
Schema Introspector - Discover available filters and values from API schema.

This module analyzes the API schema to:
1. Discover filterable fields for each resource
2. Extract allowed values from resource_hints
3. Provide synonym mappings for filter values
4. Validate filter values against schema constraints

Domain-agnostic: works with any schema following the Enable AI format.
"""

from typing import Dict, Any, List, Optional
from .types import FilterFieldInfo, ValidationResult
from .utils import setup_logger
from . import constants


class SchemaIntrospector:
    """
    Analyze OpenAPI/Enable AI schema to discover filterable fields and valid values.

    Used by:
    - api_matcher.py: To build correct query parameters
    - query_parser.py: To validate extracted filters
    - conversation_loop.py: To suggest valid values on empty results
    """

    def __init__(self, schema: Dict[str, Any]):
        """
        Initialize the introspector with an API schema.

        Args:
            schema: Enable AI format schema with 'resources' and 'resource_hints'
        """
        self.schema = schema
        self.resource_hints = schema.get('resource_hints', {}) or {}
        self.resources = schema.get('resources', {}) or {}
        self.logger = setup_logger('enable_ai.introspector')

        # Cache for computed filter info
        self._filter_cache: Dict[str, Dict[str, FilterFieldInfo]] = {}

        self.logger.info(f"SchemaIntrospector initialized with {len(self.resources)} resources")

    def get_filterable_fields(self, resource: str) -> Dict[str, FilterFieldInfo]:
        """
        Get all filterable fields for a resource.

        Combines:
        1. Query parameters from endpoint definitions
        2. Values and synonyms from resource_hints

        Args:
            resource: Resource name (e.g., 'users', 'service-orders')

        Returns:
            Dict mapping field name to FilterFieldInfo
        """
        # Check cache first
        if resource in self._filter_cache:
            return self._filter_cache[resource]

        filters: Dict[str, FilterFieldInfo] = {}

        # 1. Extract from endpoint query parameters
        resource_schema = self._get_resource_schema(resource)
        if resource_schema:
            for endpoint in resource_schema.get('endpoints', []):
                # Only GET endpoints have filter params
                if endpoint.get('method', '').upper() != 'GET':
                    continue

                params = endpoint.get('parameters', {})
                query_params = params.get('query', []) if isinstance(params, dict) else []

                for param in query_params:
                    if isinstance(param, dict):
                        name = param.get('name', '')
                        if not name:
                            continue

                        filters[name] = FilterFieldInfo(
                            name=name,
                            type=param.get('schema', {}).get('type', 'string') if isinstance(param.get('schema'), dict) else 'string',
                            description=param.get('description', ''),
                            operators=list(constants.DEFAULT_FILTER_OPERATORS),
                            required=param.get('required', False),
                            allowed_values=param.get('schema', {}).get('enum', []) if isinstance(param.get('schema'), dict) else None,
                        )
                    elif isinstance(param, str):
                        # Simple string param name
                        filters[param] = FilterFieldInfo(
                            name=param,
                            type='string',
                            operators=list(constants.DEFAULT_FILTER_OPERATORS),
                        )

        # 2. Enrich with resource_hints values and synonyms
        hints = self.resource_hints.get(resource, {})
        if isinstance(hints, dict):
            for field, hint_config in hints.items():
                # Skip special keys like __resource_synonyms__
                if field.startswith('__'):
                    continue

                if not isinstance(hint_config, dict):
                    continue

                values = hint_config.get('values', [])
                synonyms = hint_config.get('synonyms', {})

                if field in filters:
                    # Update existing field with values/synonyms
                    if values:
                        filters[field].allowed_values = values
                    if synonyms:
                        filters[field].synonyms = synonyms
                else:
                    # Add new field from hints
                    filters[field] = FilterFieldInfo(
                        name=field,
                        type='string',
                        allowed_values=values if values else None,
                        synonyms=synonyms if synonyms else {},
                        operators=list(constants.DEFAULT_FILTER_OPERATORS),
                    )

        # Cache and return
        self._filter_cache[resource] = filters
        return filters

    def get_allowed_values(self, resource: str, field: str) -> Optional[List[str]]:
        """
        Get allowed values for a specific field.

        Args:
            resource: Resource name
            field: Field name (e.g., 'status__name', 'role')

        Returns:
            List of allowed values or None if unrestricted
        """
        filters = self.get_filterable_fields(resource)

        if field in filters:
            return filters[field].allowed_values

        # Try stripping Django-style suffixes
        base_field = field.split('__')[0] if '__' in field else field
        if base_field in filters:
            return filters[base_field].allowed_values

        return None

    def is_valid_filter_field(self, resource: str, field: str) -> bool:
        """
        Check if a field is a valid filter for the resource.

        Useful for preventing hallucinated filter fields.

        Args:
            resource: Resource name
            field: Field name to check (can include Django lookups like status__name)

        Returns:
            True if field is a valid filter, False otherwise
        """
        filters = self.get_filterable_fields(resource)

        # Direct match
        if field in filters:
            return True

        # Check base field (strip Django lookups)
        base_field = field.split('__')[0] if '__' in field else field
        if base_field in filters:
            return True

        # Check if any filter starts with this field (e.g., 'status' matches 'status__name')
        for filter_name in filters.keys():
            if filter_name.startswith(base_field + '__') or filter_name == base_field:
                return True

        return False

    def get_field_synonyms(self, resource: str, field: str) -> Dict[str, str]:
        """
        Get synonym mappings for a field.

        Args:
            resource: Resource name
            field: Field name

        Returns:
            Dict mapping user terms to canonical values
            e.g., {"tech": "Technician", "admin": "Admin"}
        """
        filters = self.get_filterable_fields(resource)

        if field in filters:
            return filters[field].synonyms

        # Try stripping Django-style suffixes
        base_field = field.split('__')[0] if '__' in field else field
        if base_field in filters:
            return filters[base_field].synonyms

        return {}

    def validate_filter_value(
        self,
        resource: str,
        field: str,
        value: Any
    ) -> ValidationResult:
        """
        Validate a filter value against schema constraints.

        Checks:
        1. If value is in allowed_values
        2. If value matches a synonym (returns corrected value)
        3. Type coercion (e.g., "true" -> True for boolean fields)

        Args:
            resource: Resource name
            field: Field name
            value: Value to validate

        Returns:
            ValidationResult with is_valid, corrected_value, warnings
        """
        filters = self.get_filterable_fields(resource)

        # Find the field (or base field)
        field_info = filters.get(field)
        if not field_info:
            base_field = field.split('__')[0] if '__' in field else field
            field_info = filters.get(base_field)

        if not field_info:
            # Unknown field - allow but warn
            return ValidationResult(
                is_valid=True,
                corrected_value=value,
                original_value=value,
                warnings=[f"Field '{field}' not found in schema for resource '{resource}'"]
            )

        # Check synonyms first (case-insensitive)
        if field_info.synonyms:
            value_lower = str(value).lower()
            for syn_key, syn_value in field_info.synonyms.items():
                if value_lower == str(syn_key).lower():
                    self.logger.debug(f"Applied synonym: {value} -> {syn_value}")
                    return ValidationResult.corrected(
                        original=value,
                        corrected=syn_value,
                        warning=f"Applied synonym mapping: '{value}' -> '{syn_value}'"
                    )

        # Check allowed values (case-insensitive for strings)
        if field_info.allowed_values:
            for allowed in field_info.allowed_values:
                if isinstance(allowed, str) and isinstance(value, str):
                    if allowed.lower() == value.lower():
                        # Use canonical case
                        return ValidationResult.valid(allowed)
                elif allowed == value:
                    return ValidationResult.valid(value)

            # Value not in allowed list
            return ValidationResult.invalid(
                value=value,
                warning=f"Value '{value}' not in allowed values for '{field}': {field_info.allowed_values}"
            )

        # Type coercion
        if field_info.type == 'boolean':
            if isinstance(value, str):
                if value.lower() in ('true', 'yes', '1'):
                    return ValidationResult.corrected(value, True, "Coerced string to boolean")
                elif value.lower() in ('false', 'no', '0'):
                    return ValidationResult.corrected(value, False, "Coerced string to boolean")

        elif field_info.type == 'integer':
            if isinstance(value, str) and value.isdigit():
                return ValidationResult.corrected(value, int(value), "Coerced string to integer")

        # No validation issues
        return ValidationResult.valid(value)

    def suggest_filters_for_query(
        self,
        resource: str,
        user_query: str
    ) -> List[Dict[str, Any]]:
        """
        Suggest relevant filters based on user's natural language query.

        Analyzes the query for:
        1. Exact matches to allowed values
        2. Matches to synonyms
        3. Common filter patterns

        Args:
            resource: Resource name
            user_query: User's natural language query

        Returns:
            List of suggested filters with confidence scores
        """
        filters = self.get_filterable_fields(resource)
        suggestions = []
        query_lower = user_query.lower()

        for field, field_info in filters.items():
            # Check for exact value matches
            if field_info.allowed_values:
                for value in field_info.allowed_values:
                    value_str = str(value).lower()
                    if value_str in query_lower:
                        suggestions.append({
                            'field': field,
                            'value': value,
                            'confidence': 'high',
                            'match_type': 'exact_value',
                        })

            # Check for synonym matches
            if field_info.synonyms:
                for synonym, mapped_value in field_info.synonyms.items():
                    if synonym.lower() in query_lower:
                        suggestions.append({
                            'field': field,
                            'value': mapped_value,
                            'confidence': 'medium',
                            'match_type': 'synonym',
                            'matched_synonym': synonym,
                        })

        # Sort by confidence (high first)
        confidence_order = {'high': 0, 'medium': 1, 'low': 2}
        suggestions.sort(key=lambda x: confidence_order.get(x['confidence'], 99))

        return suggestions

    def get_resource_synonyms(self, resource: str) -> List[str]:
        """
        Get synonym names for a resource.

        Args:
            resource: Resource name

        Returns:
            List of synonym strings (e.g., ['user', 'users', 'staff', 'employees'])
        """
        hints = self.resource_hints.get(resource, {})
        if isinstance(hints, dict):
            synonyms = hints.get('__resource_synonyms__', [])
            if isinstance(synonyms, list):
                return synonyms
        return []

    def find_resource_by_synonym(self, synonym: str) -> Optional[str]:
        """
        Find a resource by its synonym.

        Args:
            synonym: A word that might refer to a resource

        Returns:
            Resource name if found, None otherwise
        """
        synonym_lower = synonym.lower()

        for resource in self.resources.keys():
            # Check resource name itself
            resource_variants = {
                resource.lower(),
                resource.lower().replace('-', '_'),
                resource.lower().replace('_', '-'),
                resource.lower().replace('-', ''),
                resource.lower().replace('_', ''),
            }

            if synonym_lower in resource_variants:
                return resource

            # Check synonyms from hints
            resource_synonyms = self.get_resource_synonyms(resource)
            for rs in resource_synonyms:
                if rs.lower() == synonym_lower:
                    return resource

        return None

    def get_all_resources(self) -> List[str]:
        """Get list of all resource names."""
        return list(self.resources.keys())

    def _get_resource_schema(self, resource: str) -> Optional[Dict[str, Any]]:
        """Get schema definition for a resource."""
        # Try exact match first
        if resource in self.resources:
            return self.resources[resource]

        # Try normalized versions
        normalized = resource.lower().replace('-', '_')
        for res_name, res_data in self.resources.items():
            if res_name.lower().replace('-', '_') == normalized:
                return res_data

        return None

    def clear_cache(self):
        """Clear the filter cache (call after schema changes)."""
        self._filter_cache.clear()
