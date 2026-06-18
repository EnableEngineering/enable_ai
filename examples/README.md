# Enable AI v2 Examples

## simple_usage.py

Basic usage demonstrating:
- Config with OpenAPI spec
- Resource hints for accuracy
- Status synonyms
- Processing natural language queries

```bash
export ANTHROPIC_API_KEY='your-key'
python simple_usage.py
```

## Integration with Your Backend

```python
from enable_ai_v2 import (
    Orchestrator,
    Config,
    JWTAuth,
    ResourceHint,
    UserContext,
    build_resource_hints_from_api,
    build_status_synonyms,
)

# Fetch data from your APIs
openapi_schema = fetch_openapi_schema()
service_order_statuses = fetch_service_order_statuses()
service_order_priorities = fetch_service_order_priorities()

# Build config with API data (no hardcoding)
config = Config(
    openapi_schema=openapi_schema,
    base_url="https://api.example.com",
    resource_hints=build_resource_hints_from_api(
        openapi_schema,
        service_order_statuses,
        service_order_priorities,
    ),
    status_synonyms=build_status_synonyms(service_order_statuses),
)

# Initialize with auth
ai = Orchestrator(
    config=config,
    auth=JWTAuth(token="user-jwt-token"),
)

# Process with user context
user_ctx = UserContext(
    user_id=123,
    role="Technician",
    is_admin=False,
)

result = ai.process("show my open orders", user_context=user_ctx)
print(result.message)
```

## Key Points

1. **Parent module provides all config** - enable_ai_v2 doesn't fetch anything
2. **Server handles data scoping** - don't add user filters unless explicitly requested
3. **Resource hints from API** - use `build_resource_hints_from_api()` helper
4. **Status synonyms from API** - use `build_status_synonyms()` helper
