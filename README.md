# Enable AI

> Natural language interface for REST APIs with MCP server support

Transform natural language queries into API calls using a LangGraph-powered workflow.

**Works with any API documentation:** Feed OpenAPI/Swagger (file, URL, or dict) and a base URL — the module converts the spec, matches user questions to endpoints, calls the right APIs, and returns answers. No hardcoded endpoints; it generalizes to any documented API.

---

## 🎯 Overview

`enable-ai` is a Python library that understands natural language and automatically:
- **Matches queries to APIs** – "list all users" → GET /users/
- **Authenticates automatically** – JWT, OAuth, API keys from config
- **Extracts parameters** – "get user 5" → GET /users/5/
- **Returns structured data** – JSON with summary, data, pagination, suggested actions
- **Multi-step execution** – Runs 3–4+ API calls in sequence when the query implies related resources
- **Automatic pagination** – Fetches next pages when the API returns `has_more`/`next` (single-step and multi-step)
- **API retry** – Retries on timeout, connection errors, and 429/502/503/504 with exponential backoff
- **LangGraph workflow** – Stateful pipeline; optional `process_stream()` for state after each node
- **Exposes MCP server** – Integrate with AI assistants like Claude Desktop

**Use Cases:**
- Build natural language interfaces for your APIs
- Create AI-powered chatbots for customer support
- Integrate with SaaS platforms for AI-driven workflows

**Current scope:** API-only. Database and document features are planned and documented as future extensions.

---

## 🚀 Installation & Setup

### **Step 1: Install the Package**

```bash
pip install enable-ai
```

Or for development:
```bash
git clone https://github.com/EnableEngineering/enable_ai.git
cd enable_ai
pip install -e .
```

### **Step 2: Create Configuration Files**

The module automatically detects `config.json` and `.env` from your working directory.

#### **config.json** - Define your data sources

```json
{
  "data_sources": {
    "api": {
      "type": "api",
      "enabled": true,
      "base_url": "http://localhost:8002/api",
      "schema_path": "schemas/api_schema.json"
    }
  },
  "security_credentials": {
    "api": {
      "jwt": {
        "enabled": true,
        "token_endpoint": "/token/",
        "username_field": "email",
        "password_field": "password",
        "env": {
          "username": "API_EMAIL",
          "password": "API_PASSWORD"
        }
      }
    }
  }
}
```

#### **.env** - Store credentials securely

```bash
OPENAI_API_KEY=sk-proj-your-key-here
API_EMAIL=admin@example.com
API_PASSWORD=your_password
```

**Important:** Add `.env` to your `.gitignore`!

### **Environment setup**

- **Python:** Use Python 3.9+ and a virtual environment (recommended):
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate   # Windows: .venv\Scripts\activate
  pip install enable-ai
  ```
- **OpenAI:** Set `OPENAI_API_KEY` in `.env` or export it. The module uses it for query parsing, planning, and summarization.
- **API credentials:** For JWT, set `API_EMAIL` and `API_PASSWORD` in `.env` (or the keys defined in `config.json` under `security_credentials.api.jwt.env`).
- **Working directory:** Run your app (or MCP server) from the directory that contains `config.json` and `.env`, or set `NLP_CONFIG_PATH` / paths explicitly.

### **Minimal configuration (no config.json)**

You can run with just a schema and base URL—no `config.json` required:

```python
# Schema path or dict; base_url in schema or pass at init
orchestrator = APIOrchestrator(schemas={"api": "path/to/api_schema.json"})
# Set base_url in the schema file under "base_url", or in config.json under data_sources.api.base_url
```

Required: **OPENAI_API_KEY** (env). **Schema:** pass at init `APIOrchestrator(schemas={"api": path_or_dict})`. **base_url:** from `config.json` (`data_sources.api.base_url`) or from the schema (`base_url`). Optional: `config.json` for auth (JWT/OAuth/API keys) and schema paths; `.env` for credentials.

### **Configurable limits**

Key limits are in `enable_ai.constants` and can be overridden by **environment variables** (set before import or in `.env`):

| Env var | Default | Effect |
|--------|--------|--------|
| `ENABLE_AI_SAFETY_MAX_PAGES` | 500 | Max pages when auto-paginating (stops and logs warning if API has more). |
| `ENABLE_AI_PAGE_SIZE_CAP` | 100 | Max `page_size` per request (e.g. "show 200 items" → 100 per page). |
| `ENABLE_AI_CONVERSATION_HISTORY_LIMIT` | 10 | Messages loaded per session for context (long chats may lose earlier filters). |
| `ENABLE_AI_IN_MEMORY_MAX_MESSAGES` | 10 | Max messages kept per session (InMemoryConversationStore). |
| `ENABLE_AI_REDIS_MAX_MESSAGES` | 20 | Max messages kept per session (RedisConversationStore). |
| `ENABLE_AI_REQUEST_TIMEOUT` | 30 | HTTP timeout in seconds (api_client, schema fetch, auth). |
| `ENABLE_AI_PROGRESS_MIN_DISPLAY_MS` | 0 | Minimum ms to show each progress stage (e.g. 400 so frontend can display each stage before the next). |

Example: `export ENABLE_AI_SAFETY_MAX_PAGES=1000` or in `.env`: `ENABLE_AI_REQUEST_TIMEOUT=60`.

### **Multi-step and pagination**

- **Multi-step:** The workflow runs 3–4+ API calls in sequence when your query implies related resources (e.g. “get users and their orders”). Conversation history and planner `extract` pass data between steps.
- **Pagination:** Responses with `has_more`/`next` are detected; pagination info and “show more” suggestions are returned. **Automatic pagination** fetches and merges next page(s) until no `next` link (single-step and multi-step; cap `ENABLE_AI_SAFETY_MAX_PAGES`).

### **Progress and streaming**

- **Progress:** Pass `progress_callback=(stage, message, progress, metadata)` to `process()` for real-time stage updates. All stages are emitted in order: STARTED → PARSING_QUERY → INTENT_DETECTED → MATCHING_API → PLANNING → API_MATCHED → PLAN_READY → EXECUTING_API → API_COMPLETED → SUMMARIZING → COMPLETED (or ERROR). If stages flash by too quickly, set `ENABLE_AI_PROGRESS_MIN_DISPLAY_MS=400` (or similar) so each stage is shown for at least that many ms. The **final response is returned once** at the end (no token-level streaming of the summary).
- **Optional stream:** Use `process_stream()` to receive state updates after each workflow node (e.g. for richer frontend progress or partial results). See `examples/streaming_backend.py` for SSE.

---

## 📖 Usage Guide

### **1. Python Library Usage**

```python
from enable_ai import APIOrchestrator

# Initialize (auto-detects config.json and .env from current directory)
orchestrator = APIOrchestrator()

# Process natural language queries
result = orchestrator.process("list all users")

print(result['summary'])  # Natural language summary
print(result['data'])     # Structured data from API
```

#### **Use with any API documentation (OpenAPI/Swagger)**

```python
from enable_ai import APIOrchestrator, load_schema

# Option A: Pass OpenAPI file path — auto-converted to Enable AI format
orchestrator = APIOrchestrator(schemas={
    "api": "/path/to/your/openapi.json"  # or swagger.yaml
})
# Set base_url in config.json under data_sources.api.base_url, or in the schema

# Option B: Pass OpenAPI dict (e.g. from URL or in-memory)
import requests
openapi_spec = requests.get("https://api.example.com/openapi.json").json()
orchestrator = APIOrchestrator(schemas={"api": openapi_spec})
# base_url can be in the spec (servers[0].url) or set via config

# Option C: Pre-convert with CLI, then use schema file
# $ enable-schema generate --input openapi.json --base-url https://api.example.com --output api_schema.json
orchestrator = APIOrchestrator(schemas={"api": "api_schema.json"})

result = orchestrator.process("list all users")
```

#### **Advanced Usage - Custom Config**

```python
# Use specific config path
orchestrator = APIOrchestrator(config_path="/path/to/config.json")

# Pass schema override (file path or dict; OpenAPI auto-converted)
orchestrator = APIOrchestrator(schemas={"api": "schemas/api_schema.json"})

# Override authentication token
result = orchestrator.process(
    "list all users",
    access_token="your_jwt_token_here"
)
```

### **2. MCP Server Usage**

Run as a Model Context Protocol (MCP) server for AI assistants:

```bash
# Start MCP server (auto-detects config from current directory)
python3 -m enable_ai.mcp_server
```

#### **Test with MCP Inspector**

```bash
# Install MCP inspector
npm install -g @modelcontextprotocol/inspector

# Launch inspector
cd /path/to/your-backend
npx @modelcontextprotocol/inspector python3 -m enable_ai.mcp_server
```

#### **Integrate with Claude Desktop**

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "enable_ai": {
      "command": "python3",
      "args": ["-m", "enable_ai.mcp_server"],
      "cwd": "/path/to/your-backend"
    }
  }
}
```

Now Claude can process natural language queries against your APIs!

### **3. Command Line Usage**

```bash
# Quick test from command line
cd /path/to/your-backend
python3 -c "
from enable_ai import APIOrchestrator
orch = APIOrchestrator()
result = orch.process('list all users')
print(result['summary'])
"
```

---

## 🏗️ Architecture

```
User Query: "list all users"
         ↓
    LangGraph Workflow
         ↓
    Parser (LLM-powered)
         ↓
    Intent + Parameters
         ↓
    Matcher (API only)
         ↓
    Execution Plan
         ↓
    Authentication (JWT/OAuth/API Key)
         ↓
    Execute Query
         ↓
    Results + Summary
```

---

## 📦 Module Components

### **Core Modules**

#### **`orchestrator.py`** - Main orchestrator
The central processing engine that coordinates all operations via a LangGraph workflow. Handles query parsing, authentication, execution planning, and response summarization. This is your main entry point via `APIOrchestrator` class.

#### **`workflow.py`** - LangGraph pipeline
Defines the stateful workflow for parsing, planning, executing, and summarizing API calls. Enables step re-runs and back-and-forth when required.

#### **`query_parser.py`** - Natural language understanding
Converts user queries into structured intents using OpenAI GPT-4. Extracts entities (IDs, names, dates), determines actions (list, get, create, update, delete), and identifies target resources.

#### **`types.py`** - Type definitions and data structures
Defines type-safe classes for requests, responses, and errors. Includes `APIRequest`, `APIResponse`, `APIError`, and authentication credential structures.

### **Data Source Matchers**

#### **`api_matcher.py`** - REST API matching
Matches parsed queries to REST API endpoints from OpenAPI/custom schemas. Handles path parameters, query strings, request bodies, and HTTP methods (GET, POST, PUT, DELETE, PATCH).

#### **`database_matcher.py`** - Database query generation (planned)
Database support is planned for a future release; the current pipeline focuses on APIs only.

#### **`knowledge_graph_matcher.py`** - Document/RAG search (planned)
Knowledge graph support is planned for a future release; the current pipeline focuses on APIs only.

### **Utilities**

#### **`api_client.py`** – HTTP request handler
Executes REST API calls with retry on timeout, connection errors, and 429/502/503/504 (exponential backoff; see `enable_ai.constants.REQUEST_RETRY_*`). Timeout and limits are in `enable_ai.constants`. Supports all HTTP methods and authentication schemes.

#### **`config_loader.py`** - Configuration management
Loads and validates configuration from JSON files or dictionaries. Handles environment variable substitution and schema path resolution.

#### **`mcp_server.py`** - MCP protocol server
Exposes the NLP processor through Model Context Protocol for integration with AI assistants like Claude Desktop. Provides 4 tools: process_query, get_schema_resources, authenticate, get_config_info.

### **Schema Generation**

#### **`schema_generator/`** - Automatic schema creation
Tools to automatically generate schemas from various sources:
- **`schema_converter.py`** - Convert OpenAPI specs to internal format (supported)
- **`database_inspector.py`** - Introspect database schemas (planned)
- **`pdf_analyzer.py`** - Extract structure from PDF documents (planned)
- **`json_analyzer.py`** - Analyze JSON APIs automatically (planned)
- **`cli.py`** - Command-line interface for schema generation

---

## 🔍 Auto-Detection

The module automatically finds configuration files from your working directory:

### **Priority Order**

1. **Current working directory** - `./config.json`, `./.env` (highest priority)
2. **Environment variables** - `$NLP_CONFIG_PATH`
3. **User home directory** - `~/.enable_ai/config.json`
4. **Package defaults** - Bundled examples

### **Verification**

```bash
# Test auto-detection
cd /path/to/your-backend
python3 << 'EOF'
import sys; sys.stderr = sys.stdout
from enable_ai.mcp_server import DEFAULT_CONFIG_PATH, DEFAULT_ENV_PATH
print(f"Config: {DEFAULT_CONFIG_PATH}")
print(f"Env: {DEFAULT_ENV_PATH}")
EOF
```

Expected output:
```
✓ Loaded .env from: /path/to/your-backend/.env
✓ Found config.json at: /path/to/your-backend/config.json
```

---

## 🔐 Authentication Support

### **JWT (JSON Web Tokens)**
Automatically obtains and refreshes JWT tokens using credentials from `.env`.

### **OAuth 2.0**
Supports client credentials and authorization code flows.

### **API Keys**
Loads API keys from environment variables and includes them in request headers.

### **Manual Tokens**
Pass tokens explicitly: `orchestrator.process("query", access_token="token")`

---

## 🗃️ Schema Examples

### **API Schema** (OpenAPI format)

```json
{
  "type": "api",
  "resources": {
    "users": {
      "description": "User management endpoints",
      "endpoints": [
        {
          "path": "/users/",
          "method": "GET",
          "description": "List all users"
        },
        {
          "path": "/users/{id}/",
          "method": "GET",
          "description": "Get user by ID"
        }
      ]
    }
  }
}
```

### **Database Schema (planned)**

```json
{
  "type": "database",
  "tables": {
    "users": {
      "description": "User accounts table",
      "columns": {
        "id": {"type": "INTEGER", "primary_key": true},
        "email": {"type": "VARCHAR"},
        "name": {"type": "VARCHAR"}
      }
    }
  }
}
```

---

## 🧪 Testing

```bash
# From project root (install in dev mode first: pip install -e .)
pytest tests/ -v

# Run specific test file
python3 -m pytest tests/test_v034_fixes_issues.py -v

# With OpenAI key (for parser/LLM tests)
OPENAI_API_KEY=sk-... pytest tests/ -v
```

---

## 📊 Example Queries

| Natural Language | Result |
|-----------------|--------|
| "list all users" | GET /users/ → Returns user list |
| "get user 5" | GET /users/5/ → Returns user details |
| "show me service orders with high priority" | Filters service orders by priority |
| "create a new user with email test@example.com" | POST /users/ → Creates user |
| "find documents about machine learning" | Semantic search in knowledge base |

---

## 🛠️ Development

### **Generate Schemas Automatically**

```bash
# From OpenAPI spec
python -m enable_ai.schema_generator.cli \
  --source openapi \
  --input swagger.json \
  --output schemas/api_schema.json

# From database (planned)
python -m enable_ai.schema_generator.cli \
  --source database \
  --connection-string "postgresql://localhost/db" \
  --output schemas/db_schema.json

# From PDFs (planned)
python -m enable_ai.schema_generator.cli \
  --source pdf \
  --input documents/ \
  --output schemas/knowledge_graph.json
```

---

## 🌐 Use Cases

### **1. Customer Support Chatbot**
```python
orchestrator = APIOrchestrator()
user_query = "Show me my recent orders"
result = orchestrator.process(user_query, access_token=user_token)
# Returns order history automatically
```

### **2. Internal Tools (planned)**
```python
# Let employees query APIs naturally
result = orchestrator.process("How many users signed up this month?")
```

### **3. API Documentation Assistant**
```python
# Feed any OpenAPI/Swagger; users ask in natural language
result = orchestrator.process("What user endpoints are available?")
```

### **4. SaaS Integration**
```python
# Deploy as MCP server for AI assistant integration
# Claude Desktop, custom agents, etc.
```

---

## 🔧 Troubleshooting

| Issue | What to check |
|-------|----------------|
| **"OPENAI_API_KEY not set"** | Add `OPENAI_API_KEY=sk-...` to `.env` in your project directory, or `export OPENAI_API_KEY=...` before running. |
| **"No schema provided"** | Pass a schema at init: `APIOrchestrator(schemas={"api": path_or_dict})`, or set `schemas.api` and `data_sources.api.enabled: true` in `config.json`. |
| **"No base_url configured"** | Set `data_sources.api.base_url` in `config.json`, or include `base_url` in your schema. |
| **Authentication failed** | For JWT: ensure `API_EMAIL` and `API_PASSWORD` (or your config’s env keys) are in `.env`; for OAuth, check `client_id` / `client_secret` in config. |
| **"Could not understand your question"** | Usually an LLM/network failure. Check connectivity and API key; try rephrasing the query. |
| **Request timed out** | Default timeout is 30s (`enable_ai.constants.REQUEST_TIMEOUT`). Failed requests are retried automatically (see `REQUEST_RETRY_ATTEMPTS`). For very slow APIs, adjust constants or install from source and change `constants.py`. |
| **MCP server not found** | Run from the directory with `config.json` and schema, or set `cwd` in Claude config to that directory. |

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

---

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details

---

## 🔗 Resources

- **Repository**: https://github.com/EnableEngineering/enable_ai
- **Issues**: https://github.com/EnableEngineering/enable_ai/issues
- **PyPI**: https://pypi.org/project/enable-ai/

## 📤 Publishing to PyPI

From the project root:

1. **Bump version** in `pyproject.toml`, `src/enable_ai/__init__.py`, and `setup.py`.
2. **Build** (no Poetry required):
   ```bash
   python3 setup.py sdist bdist_wheel
   ```
3. **Upload** (requires PyPI account and token):
   ```bash
   pip install twine
   twine upload dist/*
   ```
   You will be prompted for your PyPI username and password (use an API token as password for 2FA accounts).

Alternatively with [Poetry](https://python-poetry.org/): `poetry build` then `poetry publish`. For TestPyPI first: `twine upload --repository testpypi dist/*`.

---

## 💡 Quick Start Summary

```bash
# 1. Install
pip install enable-ai

# 2. Create config.json and .env in your project

# 3. Use it (config.json + schema, or pass schemas= and base_url)
python3 -c "
from enable_ai import APIOrchestrator
orch = APIOrchestrator()
print(orch.process('list all users')['summary'])
"
```

That's it! The module handles authentication, API matching, and execution automatically. 🚀
