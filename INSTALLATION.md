# Enable AI Installation Guide

## Installation

### From PyPI (Coming Soon)
```bash
pip install enable-ai
```

### From Source (Current)
```bash
# Clone the repository
git clone https://github.com/EnableEngineering/enable_ai.git
cd enable_ai

# Install in development mode
pip install -e .

# Or install from wheel
python3 -m build
pip install dist/enable_ai-0.1.0-py3-none-any.whl
```

## Quick Start

### Basic Usage

```python
from enable_ai import APIOrchestrator, SchemaLoader
import os

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "your-api-key-here"

# Load your API schema
schema_loader = SchemaLoader()
schema = schema_loader.load_schema("path/to/your/openapi.json", schema_type="api")

# Or load from URL
# schema = schema_loader.load_schema("https://api.example.com/swagger.json", schema_type="api")

# Create orchestrator
orchestrator = APIOrchestrator(api_schema=schema)

# Process a natural language query
result = orchestrator.process(
    "Get all users created in the last week",
    access_token="your-api-token-if-needed"
)

print(result)
```

### Using the Schema Generator CLI

Convert OpenAPI/Swagger specs to Enable AI format:

```bash
# Basic conversion
enable-schema generate --input swagger.json --output api_schema.json

# With custom base URL
enable-schema generate --input swagger.json \
  --base-url https://api.example.com \
  --output api_schema.json

# Include deprecated endpoints
enable-schema generate --input swagger.json \
  --include-deprecated \
  --output api_schema.json
```

### Integration with FastAPI Backend

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from enable_ai import APIOrchestrator, SchemaLoader
import os

app = FastAPI()

# Initialize once at startup
schema_loader = SchemaLoader()
api_schema = schema_loader.load_schema("path/to/api_schema.json")
orchestrator = APIOrchestrator(api_schema=api_schema)

class QueryRequest(BaseModel):
    query: str
    access_token: str = None

@app.post("/query")
async def process_query(request: QueryRequest):
    try:
        result = orchestrator.process(
            request.query,
            access_token=request.access_token
        )
        return {"success": True, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

## Dependencies

Core dependencies (automatically installed):
- `openai>=1.0.0` - OpenAI API client
- `langgraph>=0.2.0` - State management and workflow orchestration
- `langgraph-checkpoint>=1.0.0` - Checkpointing for state persistence
- `requests>=2.25.0` - HTTP client
- `python-dotenv>=0.19.0` - Environment variable management

## Optional Features

### Development Tools
```bash
pip install enable-ai[dev]
```

Includes:
- pytest
- pytest-asyncio
- black
- flake8
- mypy

### MCP Server Support
Note: MCP SDK is not yet available on PyPI. If you need MCP server functionality, install it manually.

## Configuration

Enable AI looks for configuration in the following order:
1. `config.json` in the current directory
2. Default configuration (embedded)

Create a `config.json` file to customize behavior:

```json
{
  "nlp": {
    "model": "gpt-4o",
    "max_tokens": 2000,
    "temperature": 0.0
  },
  "api": {
    "timeout": 30,
    "retry_count": 3
  }
}
```

## Environment Variables

Required:
- `OPENAI_API_KEY` - Your OpenAI API key

Optional:
- `ENABLE_AI_CONFIG_PATH` - Custom path to config.json
- `ENABLE_AI_LOG_LEVEL` - Logging level (DEBUG, INFO, WARNING, ERROR)

## Troubleshooting

### Import Errors
If you encounter import errors, ensure you're using Python 3.8+:
```bash
python --version
```

### OpenSSL Warnings
If you see urllib3 OpenSSL warnings, they're informational and won't affect functionality. To resolve:
```bash
# macOS
brew install openssl
pip install --upgrade urllib3
```

### Config Not Found Warnings
These are informational. Enable AI will use default configuration if `config.json` is not found.

## Next Steps

- Check the [documentation](./docs/README.md) for detailed guides
- Review [examples](./examples/) for integration patterns
- See [context.md](./docs/context.md) for architecture details

## Support

For issues, feature requests, or questions:
- GitHub Issues: https://github.com/EnableEngineering/enable_ai/issues
- Email: engineering@enableyou.co
