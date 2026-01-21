# NLP API Caller

> Natural language interface for REST APIs, databases, and knowledge graphs with MCP server support

Transform natural language queries into actual API calls, database queries, or document searches using AI-powered processing.

---

## 🎯 Overview

`nlp-api-caller` is a Python library that understands natural language and automatically:
- **Matches queries to APIs** - "list all users" → GET /users/
- **Authenticates automatically** - Handles JWT, OAuth, API keys
- **Extracts parameters** - "get user 5" → GET /users/5/
- **Returns structured data** - Clean JSON responses
- **Works with multiple sources** - APIs, databases (PostgreSQL, MySQL, MongoDB), documents (PDFs)
- **Exposes MCP server** - Integrate with AI assistants like Claude Desktop

**Use Cases:**
- Build natural language interfaces for your APIs
- Create AI-powered chatbots for customer support
- Enable database queries via conversational AI
- Integrate with SaaS platforms for AI-driven workflows

---

## 🚀 Installation & Setup

### **Step 1: Install the Package**

```bash
pip install nlp-api-caller
```

Or for development:
```bash
git clone https://github.com/yourusername/nlp-api-caller.git
cd nlp-api-caller
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
    },
    "database": {
      "type": "database",
      "enabled": false,
      "connection_string": "postgresql://user:pass@localhost/db",
      "schema_path": "schemas/database_schema.json"
    },
    "knowledge_graph": {
      "type": "knowledge_graph",
      "enabled": false,
      "schema_path": "schemas/knowledge_graph_schema.json"
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

---

## 📖 Usage Guide

### **1. Python Library Usage**

```python
from nlp_api_caller import NLPProcessor

# Initialize (auto-detects config.json and .env from current directory)
processor = NLPProcessor()

# Process natural language queries
result = processor.process("list all users")

print(result['summary'])  # Natural language summary
print(result['data'])     # Structured data from API
```

#### **Advanced Usage - Custom Config**

```python
# Use specific config path
processor = NLPProcessor(config_path="/path/to/config.json")

# Pass custom config dictionary
config = {
    "data_sources": {
        "api": {
            "type": "api",
            "enabled": True,
            "base_url": "http://api.example.com"
        }
    }
}
processor = NLPProcessor(config=config)

# Override authentication token
result = processor.process(
    "list all users",
    access_token="your_jwt_token_here"
)
```

### **2. MCP Server Usage**

Run as a Model Context Protocol (MCP) server for AI assistants:

```bash
# Start MCP server (auto-detects config from current directory)
python3 -m nlp_api_caller.mcp_server
```

#### **Test with MCP Inspector**

```bash
# Install MCP inspector
npm install -g @modelcontextprotocol/inspector

# Launch inspector
cd /path/to/your-backend
npx @modelcontextprotocol/inspector python3 -m nlp_api_caller.mcp_server
```

#### **Integrate with Claude Desktop**

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "nlp-api-caller": {
      "command": "python3",
      "args": ["-m", "nlp_api_caller.mcp_server"],
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
from nlp_api_caller import NLPProcessor
proc = NLPProcessor()
result = proc.process('list all users')
print(result['summary'])
"
```

---

## 🏗️ Architecture

```
User Query: "list all users"
         ↓
    Parser (LLM-powered)
         ↓
    Intent + Parameters
         ↓
    Matcher (API/DB/KG)
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

#### **`processor.py`** - Main orchestrator
The central processing engine that coordinates all operations. Handles query parsing, authentication, execution planning, and response summarization. This is your main entry point via `NLPProcessor` class.

#### **`parser.py`** - Natural language understanding
Converts user queries into structured intents using OpenAI GPT-4. Extracts entities (IDs, names, dates), determines actions (list, get, create, update, delete), and identifies target resources.

#### **`types.py`** - Type definitions and data structures
Defines type-safe classes for requests, responses, and errors. Includes `APIRequest`, `APIResponse`, `APIError`, and authentication credential structures.

### **Data Source Matchers**

#### **`api_matcher.py`** - REST API matching
Matches parsed queries to REST API endpoints from OpenAPI/custom schemas. Handles path parameters, query strings, request bodies, and HTTP methods (GET, POST, PUT, DELETE, PATCH).

#### **`database_matcher.py`** - Database query generation
Converts natural language to SQL queries for PostgreSQL, MySQL, and MongoDB. Supports SELECT, INSERT, UPDATE, DELETE operations with automatic query construction.

#### **`knowledge_graph_matcher.py`** - Document/RAG search
Performs semantic search across documents and knowledge graphs using embeddings. Ideal for finding information in PDFs, text documents, or structured knowledge bases.

### **Utilities**

#### **`api_client.py`** - HTTP request handler
Executes REST API calls with automatic retry logic, timeout handling, and error management. Supports all HTTP methods and authentication schemes.

#### **`config_loader.py`** - Configuration management
Loads and validates configuration from JSON files or dictionaries. Handles environment variable substitution and schema path resolution.

#### **`mcp_server.py`** - MCP protocol server
Exposes the NLP processor through Model Context Protocol for integration with AI assistants like Claude Desktop. Provides 4 tools: process_query, get_schema_resources, authenticate, get_config_info.

### **Schema Generation**

#### **`schema_generator/`** - Automatic schema creation
Tools to automatically generate schemas from various sources:
- **`openapi_converter.py`** - Convert OpenAPI specs to internal format
- **`database_inspector.py`** - Introspect database schemas (PostgreSQL, MySQL, MongoDB)
- **`pdf_analyzer.py`** - Extract structure from PDF documents
- **`json_analyzer.py`** - Analyze JSON APIs automatically
- **`cli.py`** - Command-line interface for schema generation

---

## 🔍 Auto-Detection

The module automatically finds configuration files from your working directory:

### **Priority Order**

1. **Current working directory** - `./config.json`, `./.env` (highest priority)
2. **Environment variables** - `$NLP_CONFIG_PATH`
3. **User home directory** - `~/.nlp-api-caller/config.json`
4. **Package defaults** - Bundled examples

### **Verification**

```bash
# Test auto-detection
cd /path/to/your-backend
python3 << 'EOF'
import sys; sys.stderr = sys.stdout
from nlp_api_caller.mcp_server import DEFAULT_CONFIG_PATH, DEFAULT_ENV_PATH
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
Pass tokens explicitly: `processor.process("query", access_token="token")`

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

### **Database Schema**

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
# Run tests
pytest tests/

# Run specific test
python tests/test_processor_query.py

# Test with real API
python tests/test_api_endpoint.py
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
python -m nlp_api_caller.schema_generator.cli \
  --source openapi \
  --input swagger.json \
  --output schemas/api_schema.json

# From database
python -m nlp_api_caller.schema_generator.cli \
  --source database \
  --connection-string "postgresql://localhost/db" \
  --output schemas/db_schema.json

# From PDFs
python -m nlp_api_caller.schema_generator.cli \
  --source pdf \
  --input documents/ \
  --output schemas/knowledge_graph.json
```

---

## 🌐 Use Cases

### **1. Customer Support Chatbot**
```python
processor = NLPProcessor()
user_query = "Show me my recent orders"
result = processor.process(user_query, access_token=user_token)
# Returns order history automatically
```

### **2. Internal Tools**
```python
# Let employees query databases naturally
result = processor.process("How many users signed up this month?")
```

### **3. API Documentation Assistant**
```python
# Help developers discover APIs
result = processor.process("What user endpoints are available?")
```

### **4. SaaS Integration**
```python
# Deploy as MCP server for AI assistant integration
# Claude Desktop, custom agents, etc.
```

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

- **Repository**: https://github.com/yourusername/nlp-api-caller
- **Issues**: https://github.com/yourusername/nlp-api-caller/issues
- **PyPI**: https://pypi.org/project/nlp-api-caller/

---

## 💡 Quick Start Summary

```bash
# 1. Install
pip install nlp-api-caller

# 2. Create config.json and .env in your project

# 3. Use it
python3 -c "
from nlp_api_caller import NLPProcessor
proc = NLPProcessor()
print(proc.process('list all users')['summary'])
"
```

That's it! The module handles authentication, API matching, and execution automatically. 🚀
