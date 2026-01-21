"""Generate API schema from api_spec.json"""
import json
import sys
from pathlib import Path
sys.path.insert(0, 'src')

from nlp_api_caller.schema_generator import OpenAPIConverter

# Try to find api_spec.json (legacy location or in schemas directory)
api_spec_paths = [
    Path('src/nlp_api_caller/api_spec.json'),  # Legacy location
    Path('enable_api_schema.json'),  # Root directory
]

api_spec = None
for spec_path in api_spec_paths:
    if spec_path.exists():
        with open(spec_path, 'r') as f:
            api_spec = json.load(f)
        print(f"✓ Loaded API spec from: {spec_path}")
        break

if not api_spec:
    print("⚠️  Warning: api_spec.json not found. Cannot generate schema.")
    print("   Checked locations:")
    for path in api_spec_paths:
        print(f"   - {path}")
    sys.exit(1)

# Convert to schema format
converter = OpenAPIConverter()

# Extract the modules and convert them
schema = {
    "type": "api",
    "version": api_spec.get("version", "1.0.0"),
    "metadata": {
        "name": api_spec.get("api_name", "API"),
        "description": "Generated from Enable ERP API specification",
        "api_version": api_spec.get("version", "1.0.0"),
        "generated_from": "enable_erp_api_spec"
    },
    "base_url": api_spec.get("base_url", ""),
    "resources": {}
}

# Extract resources from modules
modules = api_spec.get("modules", {})
for module_name, module_data in modules.items():
    if isinstance(module_data, dict):
        endpoints = module_data.get("endpoints", {})
        
        for endpoint_key, endpoint_data in endpoints.items():
            # Determine resource name from endpoint path
            path = endpoint_data.get("path", "")
            method = endpoint_data.get("method", "GET")
            
            # Extract resource name from path (e.g., /users → users, /service_orders/{id} → service_orders)
            parts = path.strip('/').split('/')
            if parts:
                resource_name = parts[0]
                
                # Initialize resource if not exists
                if resource_name not in schema["resources"]:
                    schema["resources"][resource_name] = {
                        "name": resource_name,
                        "description": f"{resource_name.replace('_', ' ').title()} resource",
                        "endpoints": []
                    }
                
                # Map method to intent
                method_to_intent = {
                    "GET": "read",
                    "POST": "create",
                    "PUT": "update",
                    "PATCH": "update",
                    "DELETE": "delete"
                }
                intent = method_to_intent.get(method.upper(), "read")
                
                # Extract parameters
                params = endpoint_data.get("parameters", {})
                query_params = params.get("query", [])
                path_params = params.get("path", [])
                body_params = params.get("body", {})
                
                # Build parameter lists
                required_params = []
                optional_params = []
                
                # Path parameters (always required)
                for param in path_params:
                    if isinstance(param, dict):
                        param_name = param.get("name", "")
                    else:
                        param_name = str(param)
                    if param_name:
                        required_params.append(param_name)
                
                # Query parameters
                for param in query_params:
                    if isinstance(param, dict):
                        param_name = param.get("name", "")
                        required = param.get("required", False)
                    else:
                        param_name = str(param)
                        required = False
                    
                    if param_name:
                        if required:
                            required_params.append(param_name)
                        else:
                            optional_params.append(param_name)
                
                # Body parameters
                if body_params:
                    for field_name, field_info in body_params.items():
                        if isinstance(field_info, dict):
                            required = field_info.get("required", False)
                        else:
                            required = False
                        
                        if required:
                            required_params.append(field_name)
                        else:
                            optional_params.append(field_name)
                
                # Add endpoint to resource
                endpoint = {
                    "path": path,
                    "method": method.upper(),
                    "intent": intent,
                    "description": endpoint_data.get("description", ""),
                    "parameters": {
                        "required": required_params,
                        "optional": optional_params,
                        "path": [p.get("name", p) if isinstance(p, dict) else str(p) for p in path_params],
                        "query": [p.get("name", p) if isinstance(p, dict) else str(p) for p in query_params],
                        "body": list(body_params.keys()) if body_params else []
                    },
                    "authentication_required": endpoint_data.get("authentication_required", True),
                    "response_type": "object"
                }
                
                schema["resources"][resource_name]["endpoints"].append(endpoint)

# Determine output path - use environment.py if available
try:
    import environment
    schemas_dir = environment.get_schemas_dir()
    output_path = f'{schemas_dir}/api_schema.json'
except (ImportError, AttributeError):
    # Fallback to default path
    output_path = 'schemas/api_schema.json'

# Save schema
with open(output_path, 'w') as f:
    json.dump(schema, f, indent=2)

print(f"✓ Generated API schema with {len(schema['resources'])} resources")
print(f"✓ Saved to {output_path}")
print(f"\nResources:")
for resource_name, resource_data in schema["resources"].items():
    endpoint_count = len(resource_data.get("endpoints", []))
    print(f"  - {resource_name} ({endpoint_count} endpoints)")
