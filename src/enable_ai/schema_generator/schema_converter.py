"""
OpenAPI/Swagger to API Schema Converter

Converts OpenAPI 3.0/Swagger 2.0 specifications into Enable AI API schemas.
Handles 80% of use cases - most REST APIs have OpenAPI docs.
"""

from typing import Dict, Any, List, Optional
from .base import SchemaGenerator


class OpenAPIConverter(SchemaGenerator):
    """
    Convert OpenAPI/Swagger specs to API schema.
    
    Usage:
        converter = OpenAPIConverter()
        
        # From file
        schema = converter.generate('swagger.json')
        
        # From dict
        schema = converter.generate(openapi_dict)
        
        # Save to file
        converter.save_schema(schema, 'api_schema.json')
    """
    
    def get_schema_type(self) -> str:
        return 'api'
    
    def generate(self, source: Any, **kwargs) -> Dict[str, Any]:
        """
        Generate API schema from OpenAPI spec.
        
        Args:
            source: OpenAPI spec (file path or dict)
            **kwargs: Options:
                - base_url: Override base URL from spec
                - include_deprecated: Include deprecated endpoints (default: False)
                - resource_filter: List of resources to include (default: all)
        
        Returns:
            API schema dict
        """
        # Load OpenAPI spec
        if isinstance(source, str):
            spec = self.load_json_file(source)
        elif isinstance(source, dict):
            spec = source
        else:
            raise ValueError("Source must be file path (str) or OpenAPI dict")
        
        # Detect version
        version = self._detect_version(spec)
        
        if version == 3:
            return self._convert_openapi_v3(spec, **kwargs)
        elif version == 2:
            return self._convert_swagger_v2(spec, **kwargs)
        else:
            raise ValueError(f"Unsupported OpenAPI version: {version}")
    
    def _detect_version(self, spec: Dict[str, Any]) -> int:
        """Detect OpenAPI version."""
        if 'openapi' in spec:
            version = spec['openapi']
            if version.startswith('3.'):
                return 3
        elif 'swagger' in spec:
            version = spec['swagger']
            if version.startswith('2.'):
                return 2
        
        raise ValueError("Could not detect OpenAPI/Swagger version")
    
    def _convert_openapi_v3(self, spec: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Convert OpenAPI 3.0 spec to API schema."""
        # Extract base URL
        base_url = kwargs.get('base_url')
        if not base_url:
            servers = spec.get('servers', [])
            if servers:
                base_url = servers[0].get('url', 'http://localhost:8000')
            else:
                base_url = 'http://localhost:8000'
        
        # Extract API info
        info = spec.get('info', {})
        api_name = info.get('title', 'API')
        api_version = info.get('version', '1.0.0')
        
        # Convert paths to resources
        paths = spec.get('paths', {})
        resources = self._convert_paths_to_resources(
            paths,
            spec.get('components', {}),
            version=3,
            **kwargs
        )
        
        # Build schema
        schema = {
            "type": "api",
            "version": "1.0.0",
            "metadata": {
                "name": api_name,
                "description": info.get('description', ''),
                "api_version": api_version,
                "generated_from": "openapi_3.0"
            },
            "base_url": base_url,
            "resources": resources
        }
        
        return schema
    
    def _convert_swagger_v2(self, spec: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Convert Swagger 2.0 spec to API schema."""
        # Extract base URL
        base_url = kwargs.get('base_url')
        if not base_url:
            host = spec.get('host', 'localhost:8000')
            base_path = spec.get('basePath', '')
            schemes = spec.get('schemes', ['http'])
            base_url = f"{schemes[0]}://{host}{base_path}"
        
        # Extract API info
        info = spec.get('info', {})
        api_name = info.get('title', 'API')
        api_version = info.get('version', '1.0.0')
        
        # Convert paths to resources
        paths = spec.get('paths', {})
        resources = self._convert_paths_to_resources(
            paths,
            spec.get('definitions', {}),
            version=2,
            **kwargs
        )
        
        # Build schema
        schema = {
            "type": "api",
            "version": "1.0.0",
            "metadata": {
                "name": api_name,
                "description": info.get('description', ''),
                "api_version": api_version,
                "generated_from": "swagger_2.0"
            },
            "base_url": base_url,
            "resources": resources
        }
        
        return schema
    
    def _convert_paths_to_resources(
        self,
        paths: Dict[str, Any],
        components: Dict[str, Any],
        version: int,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Convert OpenAPI paths to resource definitions.
        
        Groups endpoints by resource (e.g., /users, /users/{id} → users resource)
        """
        include_deprecated = kwargs.get('include_deprecated', False)
        resource_filter = kwargs.get('resource_filter', None)
        
        resources: Dict[str, Any] = {}
        # Collect inferred fields per resource from response schemas
        resource_fields: Dict[str, set] = {}
        
        for path, path_item in paths.items():
            # Extract resource name from path
            resource_name = self._extract_resource_name(path)
            
            # Filter resources if specified
            if resource_filter and resource_name not in resource_filter:
                continue
            
            # Initialize resource if not exists
            if resource_name not in resources:
                resources[resource_name] = {
                    "name": resource_name,
                    "description": path_item.get('description', f'{resource_name.title()} resource'),
                    "endpoints": []
                }
                resource_fields[resource_name] = set()
            
            # Convert each HTTP method
            for method in ['get', 'post', 'put', 'patch', 'delete']:
                if method not in path_item:
                    continue
                
                operation = path_item[method]
                
                # Skip deprecated endpoints unless explicitly included
                if operation.get('deprecated') and not include_deprecated:
                    continue
                
                # Convert operation to endpoint
                endpoint = self._convert_operation_to_endpoint(
                    path,
                    method,
                    operation,
                    components,
                    version
                )
                
                # Infer response fields for this resource from the operation's response schema
                inferred_fields = self._extract_response_fields(
                    operation,
                    components,
                    version
                )
                if inferred_fields:
                    resource_fields[resource_name].update(inferred_fields)
                
                resources[resource_name]['endpoints'].append(endpoint)
        
        # Attach inferred fields and display_field metadata to each resource (if any)
        for resource_name, fields in resource_fields.items():
            if not fields:
                continue
            
            # Only add fields list if not already provided by some external enhancer
            field_list = resources[resource_name].setdefault("fields", sorted(fields))
            
            # Infer a reasonable display field for this resource (used for naming)
            display_field = self._infer_display_field(resource_name, field_list)
            if display_field:
                resources[resource_name].setdefault("display_field", display_field)
        
        return resources
    
    def _extract_resource_name(self, path: str) -> str:
        """
        Extract resource name from path.
        
        Examples:
            /users → users
            /users/{id} → users
            /api/v1/customers → customers
            /service_orders/{id}/items → service_orders
        """
        # Remove leading/trailing slashes
        path = path.strip('/')
        
        # Split by slash
        parts = path.split('/')
        
        # Find first non-parameter part
        for part in parts:
            if not part.startswith('{') and not part.startswith(':'):
                # Skip common prefixes
                if part.lower() not in ['api', 'v1', 'v2', 'v3']:
                    return part
        
        return parts[0] if parts else 'unknown'
    
    def _convert_operation_to_endpoint(
        self,
        path: str,
        method: str,
        operation: Dict[str, Any],
        components: Dict[str, Any],
        version: int
    ) -> Dict[str, Any]:
        """Convert OpenAPI operation to endpoint definition."""
        # Map HTTP method to CRUD intent
        intent_map = {
            'get': 'read',
            'post': 'create',
            'put': 'update',
            'patch': 'update',
            'delete': 'delete'
        }
        
        # Extract parameters
        parameters = self._extract_parameters(
            operation.get('parameters', []),
            path,
            method,
            components,
            version
        )
        
        # Build endpoint
        endpoint = {
            "path": path,
            "method": method.upper(),
            "intent": intent_map.get(method, 'read'),
            "description": operation.get('summary') or operation.get('description', ''),
            "parameters": parameters,
            "authentication_required": self._requires_auth(operation),
            "response_type": self._extract_response_type(operation, components, version)
        }
        
        return endpoint
    
    def _extract_parameters(
        self,
        params: List[Dict[str, Any]],
        path: str,
        method: str,
        components: Dict[str, Any],
        version: int
    ) -> Dict[str, Any]:
        """Extract and categorize parameters."""
        result = {
            "required": [],
            "optional": [],
            "path": [],
            "query": [],
            "body": []
        }
        
        for param in params:
            # Handle $ref
            if '$ref' in param:
                param = self._resolve_ref(param['$ref'], components)
            
            param_name = param.get('name')
            param_in = param.get('in', 'query')
            param_required = param.get('required', False)
            param_type = self._get_param_type(param, version)
            param_desc = param.get('description', '')
            
            param_def = {
                "name": param_name,
                "type": param_type,
                "description": param_desc
            }
            
            # Add to required/optional
            if param_required:
                result['required'].append(param_name)
            else:
                result['optional'].append(param_name)
            
            # Add to location-specific list
            if param_in == 'path':
                result['path'].append(param_def)
            elif param_in == 'query':
                result['query'].append(param_def)
            elif param_in == 'body':
                result['body'].append(param_def)
        
        # Handle request body (OpenAPI 3.0)
        if version == 3 and method in ['post', 'put', 'patch']:
            # Request body is handled separately in OpenAPI 3.0
            # We'll add a generic 'body' parameter
            result['body'].append({
                "name": "body",
                "type": "object",
                "description": "Request body"
            })
        
        return result
    
    def _get_param_type(self, param: Dict[str, Any], version: int) -> str:
        """Extract parameter type."""
        if version == 3:
            schema = param.get('schema', {})
            return schema.get('type', 'string')
        else:
            return param.get('type', 'string')
    
    def _requires_auth(self, operation: Dict[str, Any]) -> bool:
        """Check if operation requires authentication."""
        security = operation.get('security', [])
        return len(security) > 0
    
    def _extract_response_type(
        self,
        operation: Dict[str, Any],
        components: Dict[str, Any],
        version: int
    ) -> str:
        """Extract response type from operation."""
        responses = operation.get('responses', {})
        
        # Look for successful response (200, 201)
        for status_code in ['200', '201', '202']:
            if status_code in responses:
                response = responses[status_code]
                
                if version == 3:
                    content = response.get('content', {})
                    if 'application/json' in content:
                        schema = content['application/json'].get('schema', {})
                        return self._get_type_from_schema(schema, components)
                else:
                    schema = response.get('schema', {})
                    return self._get_type_from_schema(schema, components)
        
        return 'object'

    def _extract_response_fields(
        self,
        operation: Dict[str, Any],
        components: Dict[str, Any],
        version: int
    ) -> List[str]:
        """
        Infer field names from a successful JSON response schema.
        
        This is used to populate resource-level "fields" metadata so that
        downstream components (parser, summariser) can reason about which
        keys are present in list/detail responses.
        """
        responses = operation.get('responses', {})
        
        for status_code in ['200', '201', '202']:
            if status_code not in responses:
                continue
            
            response = responses[status_code]
            
            # OpenAPI 3: look under content -> application/json -> schema
            if version == 3:
                content = response.get('content', {})
                if 'application/json' not in content:
                    continue
                schema = content['application/json'].get('schema', {})
            else:
                # Swagger 2: response.schema
                schema = response.get('schema', {})
            
            if not isinstance(schema, dict):
                continue
            
            fields = self._collect_fields_from_schema(schema, components)
            if fields:
                return fields
        
        return []
    
    def _collect_fields_from_schema(
        self,
        schema: Dict[str, Any],
        components: Dict[str, Any]
    ) -> List[str]:
        """
        Recursively collect field names from an object/array schema.
        
        Handles common patterns:
        - Plain object with properties
        - Array of objects
        - Paginated responses with "results" array of objects
        """
        if '$ref' in schema:
            # Resolve reference then continue
            ref_schema = self._resolve_ref(schema['$ref'], components)
            if not isinstance(ref_schema, dict):
                return []
            return self._collect_fields_from_schema(ref_schema, components)
        
        schema_type = schema.get('type')
        
        # Array: inspect items
        if schema_type == 'array':
            items = schema.get('items', {})
            if isinstance(items, dict):
                return self._collect_fields_from_schema(items, components)
            return []
        
        # Object or unspecified: inspect properties
        if schema_type in (None, 'object'):
            props = schema.get('properties', {})
            if not isinstance(props, dict) or not props:
                return []
            
            # Handle paginated structures: look for "results" array of objects
            if 'results' in props and isinstance(props['results'], dict):
                results_schema = props['results']
                # Sometimes "results" is array with inner object schema
                if results_schema.get('type') == 'array':
                    inner = results_schema.get('items', {})
                    if isinstance(inner, dict):
                        inner_fields = self._collect_fields_from_schema(inner, components)
                        if inner_fields:
                            return inner_fields
            
            # Fallback: use top-level property names as fields
            return list(props.keys())
        
        return []

    def _infer_display_field(self, resource_name: str, fields: List[str]) -> Optional[str]:
        """
        Infer a primary display field for a resource from its field list.
        
        Generic heuristics only – no project-specific knowledge:
        1. Exact "name" if present.
        2. Field ending with "_name".
        3. Fields containing "name" (excluding obvious unit-like fields).
        4. Common identifiers like "title", "label", "display_name", "code".
        """
        if not fields:
            return None
        
        lower_fields = {f: f.lower() for f in fields}
        
        # Helper to filter out unit-like fields
        def is_unit_like(fname: str) -> bool:
            lf = fname.lower()
            return any(
                key in lf
                for key in ["unit", "uom", "measurement", "measure", "qty_per", "per_unit"]
            )
        
        # 1) Exact "name"
        for f, lf in lower_fields.items():
            if lf == "name" and not is_unit_like(f):
                return f
        
        # 2) Ends with "_name"
        for f, lf in lower_fields.items():
            if lf.endswith("_name") and not is_unit_like(f):
                return f
        
        # 3) Contains "name" (but avoid unit-like)
        for f, lf in lower_fields.items():
            if "name" in lf and not is_unit_like(f):
                return f
        
        # 4) Common identifier fields
        preferred = ["title", "label", "display_name", "short_name", "code", "sku"]
        for pref in preferred:
            for f, lf in lower_fields.items():
                if lf == pref and not is_unit_like(f):
                    return f
        
        # Fallback: first non unit-like field
        for f in fields:
            if not is_unit_like(f):
                return f
        
        # As a last resort, just return the first field
        return fields[0]
    
    def _get_type_from_schema(
        self,
        schema: Dict[str, Any],
        components: Dict[str, Any]
    ) -> str:
        """Get type from schema definition."""
        if '$ref' in schema:
            # Resolve reference
            ref_schema = self._resolve_ref(schema['$ref'], components)
            return self._get_type_from_schema(ref_schema, components)
        
        schema_type = schema.get('type', 'object')
        
        if schema_type == 'array':
            return 'array'
        else:
            return 'object'
    
    def _resolve_ref(
        self,
        ref: str,
        components: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Resolve $ref to actual definition."""
        # ref format: #/components/schemas/User or #/definitions/User
        parts = ref.split('/')
        
        if len(parts) < 2:
            return {}
        
        # Navigate through components
        current = components
        for part in parts[1:]:  # Skip '#'
            if part in current:
                current = current[part]
            else:
                return {}
        
        return current if isinstance(current, dict) else {}


# Convenience function
def convert_openapi(source: Any, output_path: Optional[str] = None, auto_save: bool = True, **kwargs) -> Dict[str, Any]:
    """
    Quick function to convert OpenAPI spec to API schema.
    
    Args:
        source: OpenAPI spec (file path or dict)
        output_path: Output path (optional, defaults to schemas/api_schema.json)
        auto_save: Automatically save to schemas/ directory (default: True)
        **kwargs: Additional conversion options
    
    Usage:
        # Generate and auto-save to schemas/
        schema = convert_openapi('swagger.json')
        
        # Generate and save to custom location
        schema = convert_openapi('swagger.json', output_path='custom/api_schema.json')
        
        # Generate without saving
        schema = convert_openapi('swagger.json', auto_save=False)
    """
    converter = OpenAPIConverter()
    schema = converter.generate(source, **kwargs)
    
    if auto_save or output_path:
        saved_path = converter.save_schema(schema, output_path)
        schema['_saved_path'] = saved_path
    
    return schema
