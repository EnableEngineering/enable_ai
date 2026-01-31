"""
Schema Validator - Validates API schemas to catch format issues early

This module validates schemas against the Enable AI schema format to ensure:
- Required fields are present
- Structure is correct
- Intent fields are defined for API matching
- No common mistakes
"""

from typing import Dict, Any, List, Tuple
from .utils import setup_logger

logger = setup_logger('enable_ai.schema_validator')


class SchemaValidationError(Exception):
    """Raised when schema validation fails."""
    pass


class SchemaValidator:
    """
    Validates API schemas to ensure they match Enable AI format.
    
    Provides detailed error messages and warnings to help fix schema issues.
    """
    
    def __init__(self):
        """Initialize the schema validator."""
        self.logger = setup_logger('enable_ai.schema_validator')
    
    def validate_schema(self, schema: Dict[str, Any], strict: bool = False) -> Tuple[bool, List[str], List[str]]:
        """
        Validate a schema and return detailed results.
        
        Args:
            schema: Schema dictionary to validate
            strict: If True, warnings become errors
            
        Returns:
            Tuple of (is_valid, errors, warnings)
            - is_valid: True if no errors (warnings allowed)
            - errors: List of error messages (critical issues)
            - warnings: List of warning messages (best practices)
        """
        errors = []
        warnings = []
        
        if not schema or not isinstance(schema, dict):
            errors.append("Schema must be a non-empty dictionary")
            return False, errors, warnings
        
        # Validate schema type
        schema_type = schema.get('type')
        if not schema_type:
            errors.append("Schema missing 'type' field. Expected: 'api', 'database', or 'knowledge_graph'")
        elif schema_type not in ['api', 'api_schema', 'database', 'knowledge_graph']:
            errors.append(f"Invalid schema type: '{schema_type}'. Expected: 'api', 'database', or 'knowledge_graph'")
        
        # Validate based on schema type
        if schema_type in ['api', 'api_schema']:
            api_errors, api_warnings = self._validate_api_schema(schema)
            errors.extend(api_errors)
            warnings.extend(api_warnings)
        elif schema_type == 'database':
            db_errors, db_warnings = self._validate_database_schema(schema)
            errors.extend(db_errors)
            warnings.extend(db_warnings)
        elif schema_type == 'knowledge_graph':
            kg_errors, kg_warnings = self._validate_knowledge_graph_schema(schema)
            errors.extend(kg_errors)
            warnings.extend(kg_warnings)
        
        # If strict mode, treat warnings as errors
        if strict and warnings:
            errors.extend(warnings)
            warnings = []
        
        is_valid = len(errors) == 0
        
        return is_valid, errors, warnings
    
    def _validate_api_schema(self, schema: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """
        Validate API schema structure.
        
        Returns:
            Tuple of (errors, warnings)
        """
        errors = []
        warnings = []
        
        # Check for resources
        resources = schema.get('resources')
        if not resources:
            errors.append("API schema missing 'resources' field. Expected: {'resources': {...}}")
            return errors, warnings
        
        if not isinstance(resources, dict):
            errors.append(f"'resources' must be a dictionary, got: {type(resources).__name__}")
            return errors, warnings
        
        if len(resources) == 0:
            warnings.append("API schema has no resources defined (empty 'resources' dict)")
        
        # Validate each resource
        for resource_name, resource_data in resources.items():
            if not isinstance(resource_data, dict):
                errors.append(f"Resource '{resource_name}' must be a dictionary, got: {type(resource_data).__name__}")
                continue
            
            # Check for endpoints
            endpoints = resource_data.get('endpoints')
            if not endpoints:
                errors.append(f"Resource '{resource_name}' missing 'endpoints' field")
                continue
            
            if not isinstance(endpoints, list):
                errors.append(f"Resource '{resource_name}' endpoints must be a list, got: {type(endpoints).__name__}")
                continue
            
            if len(endpoints) == 0:
                warnings.append(f"Resource '{resource_name}' has no endpoints defined (empty list)")
                continue
            
            # Validate each endpoint
            for idx, endpoint in enumerate(endpoints):
                if not isinstance(endpoint, dict):
                    errors.append(f"Resource '{resource_name}' endpoint {idx} must be a dictionary")
                    continue
                
                endpoint_path = endpoint.get('path', f'<unnamed #{idx}>')
                
                # Check required fields
                if 'path' not in endpoint:
                    errors.append(f"Resource '{resource_name}' endpoint {idx} missing 'path' field")
                
                if 'method' not in endpoint:
                    errors.append(f"Resource '{resource_name}' endpoint '{endpoint_path}' missing 'method' field")
                else:
                    method = endpoint.get('method', '').upper()
                    if method not in ['GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'HEAD', 'OPTIONS']:
                        errors.append(
                            f"Resource '{resource_name}' endpoint '{endpoint_path}' has invalid method: '{method}'. "
                            f"Expected: GET, POST, PUT, PATCH, DELETE"
                        )
                
                # Check for intent or intent_keywords (important for matching)
                has_intent = 'intent' in endpoint
                has_intent_keywords = 'intent_keywords' in endpoint
                has_description = 'description' in endpoint
                
                if not has_intent and not has_intent_keywords:
                    if has_description:
                        warnings.append(
                            f"Resource '{resource_name}' endpoint '{endpoint_path}' missing 'intent' field. "
                            f"Has 'description' (fallback matching), but adding 'intent' improves accuracy. "
                            f"Example: \"intent\": \"read\" for GET, \"create\" for POST"
                        )
                    else:
                        errors.append(
                            f"Resource '{resource_name}' endpoint '{endpoint_path}' missing 'intent', "
                            f"'intent_keywords', and 'description'. At least one is required for API matching."
                        )
                
                # Validate intent value if present
                if has_intent:
                    intent = endpoint.get('intent', '').lower()
                    valid_intents = ['read', 'create', 'update', 'delete', 'list', 'search']
                    if intent not in valid_intents:
                        warnings.append(
                            f"Resource '{resource_name}' endpoint '{endpoint_path}' has non-standard intent: '{intent}'. "
                            f"Recommended: {', '.join(valid_intents)}"
                        )
                
                # Check for description (helpful but not required)
                if not has_description:
                    warnings.append(
                        f"Resource '{resource_name}' endpoint '{endpoint_path}' missing 'description'. "
                        f"Descriptions improve LLM understanding and API matching."
                    )
        
        return errors, warnings
    
    def _validate_database_schema(self, schema: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """
        Validate database schema structure.
        
        Returns:
            Tuple of (errors, warnings)
        """
        errors = []
        warnings = []
        
        # Check for tables
        tables = schema.get('tables')
        if not tables:
            errors.append("Database schema missing 'tables' field. Expected: {'tables': {...}}")
            return errors, warnings
        
        if not isinstance(tables, dict):
            errors.append(f"'tables' must be a dictionary, got: {type(tables).__name__}")
            return errors, warnings
        
        if len(tables) == 0:
            warnings.append("Database schema has no tables defined (empty 'tables' dict)")
        
        # Validate each table
        for table_name, table_data in tables.items():
            if not isinstance(table_data, dict):
                errors.append(f"Table '{table_name}' must be a dictionary, got: {type(table_data).__name__}")
                continue
            
            # Check for columns
            columns = table_data.get('columns')
            if not columns:
                errors.append(f"Table '{table_name}' missing 'columns' field")
                continue
            
            if not isinstance(columns, dict):
                errors.append(f"Table '{table_name}' columns must be a dictionary, got: {type(columns).__name__}")
                continue
            
            if len(columns) == 0:
                warnings.append(f"Table '{table_name}' has no columns defined (empty dict)")
        
        return errors, warnings
    
    def _validate_knowledge_graph_schema(self, schema: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """
        Validate knowledge graph schema structure.
        
        Returns:
            Tuple of (errors, warnings)
        """
        errors = []
        warnings = []
        
        # Check for entities
        entities = schema.get('entities')
        if not entities:
            errors.append("Knowledge graph schema missing 'entities' field. Expected: {'entities': {...}}")
            return errors, warnings
        
        if not isinstance(entities, dict):
            errors.append(f"'entities' must be a dictionary, got: {type(entities).__name__}")
            return errors, warnings
        
        if len(entities) == 0:
            warnings.append("Knowledge graph schema has no entities defined (empty 'entities' dict)")
        
        # Validate each entity
        for entity_name, entity_data in entities.items():
            if not isinstance(entity_data, dict):
                errors.append(f"Entity '{entity_name}' must be a dictionary, got: {type(entity_data).__name__}")
                continue
            
            # Check for properties
            if 'properties' not in entity_data:
                warnings.append(f"Entity '{entity_name}' missing 'properties' field")
        
        return errors, warnings
    
    def validate_and_raise(self, schema: Dict[str, Any], strict: bool = False):
        """
        Validate schema and raise exception if invalid.
        
        Args:
            schema: Schema to validate
            strict: If True, warnings become errors
            
        Raises:
            SchemaValidationError: If schema is invalid
        """
        is_valid, errors, warnings = self.validate_schema(schema, strict=strict)
        
        if not is_valid:
            error_msg = "Schema validation failed:\n\n"
            error_msg += "ERRORS:\n"
            for idx, error in enumerate(errors, 1):
                error_msg += f"  {idx}. {error}\n"
            
            if warnings:
                error_msg += "\nWARNINGS:\n"
                for idx, warning in enumerate(warnings, 1):
                    error_msg += f"  {idx}. {warning}\n"
            
            raise SchemaValidationError(error_msg)
        
        # Log warnings even if valid
        if warnings:
            self.logger.warning(f"Schema validation passed with {len(warnings)} warning(s):")
            for warning in warnings:
                self.logger.warning(f"  - {warning}")
    
    def validate_and_log(self, schema: Dict[str, Any], strict: bool = False) -> bool:
        """
        Validate schema and log results (doesn't raise exception).
        
        Args:
            schema: Schema to validate
            strict: If True, warnings become errors
            
        Returns:
            True if valid, False otherwise
        """
        is_valid, errors, warnings = self.validate_schema(schema, strict=strict)
        
        if errors:
            self.logger.error(f"Schema validation failed with {len(errors)} error(s):")
            for error in errors:
                self.logger.error(f"  ❌ {error}")
        
        if warnings:
            self.logger.warning(f"Schema validation passed with {len(warnings)} warning(s):")
            for warning in warnings:
                self.logger.warning(f"  ⚠️  {warning}")
        
        if is_valid and not warnings:
            self.logger.info("✅ Schema validation passed with no issues")
        
        return is_valid


# Convenience function for quick validation
def validate_schema(schema: Dict[str, Any], strict: bool = False) -> Tuple[bool, List[str], List[str]]:
    """
    Validate a schema (convenience function).
    
    Args:
        schema: Schema dictionary to validate
        strict: If True, warnings become errors
        
    Returns:
        Tuple of (is_valid, errors, warnings)
    """
    validator = SchemaValidator()
    return validator.validate_schema(schema, strict=strict)
