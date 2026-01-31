"""
MCP Server for Enable AI

Exposes the NLP processor functionality through Model Context Protocol (MCP)
so that AI applications can process natural language queries against APIs.
"""

import asyncio
import json
import os
import sys
from typing import Any, Optional
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
    LoggingLevel
)

from .orchestrator import APIOrchestrator
from .types import APIResponse, APIError
from . import constants


def find_env_file():
    """
    Find and load .env file using priority order:
    1. environment.py configuration (if exists)
    2. CWD and parent directories
    3. Explicit search paths
    """
    try:
        from dotenv import load_dotenv, find_dotenv
        
        # Priority 1: Check environment.py for DEV/PROD configuration
        try:
            # Import from package
            from . import environment
            env_path = Path(environment.get_env_path())
            
            if env_path.exists():
                load_dotenv(env_path)
                print(f"✓ Loaded {environment.ACTIVE_ENVIRONMENT} .env from environment.py: {env_path}", file=sys.stderr)
                return str(env_path)
            else:
                print(f"⚠️  environment.py configured but .env not found: {env_path}", file=sys.stderr)
        except (ImportError, AttributeError, FileNotFoundError):
            # environment.py doesn't exist or is invalid, fall back to auto-detection
            pass
        
        # Priority 2: Search for .env in CWD and parent directories
        env_file = find_dotenv(usecwd=True)
        
        if env_file:
            load_dotenv(env_file)
            print(f"✓ Loaded .env from: {env_file}", file=sys.stderr)
            return env_file
        else:
            # Priority 3: Try explicit locations
            search_paths = [
                Path.cwd() / ".env",
                Path.home() / ".enable_ai" / ".env"
            ]
            
            for path in search_paths:
                if path.exists():
                    load_dotenv(path)
                    print(f"✓ Loaded .env from: {path}", file=sys.stderr)
                    return str(path)
            
            print("⚠ No .env file found - using system environment variables", file=sys.stderr)
            return None
    except ImportError:
        print("⚠ python-dotenv not installed - using system environment variables", file=sys.stderr)
        return None


def find_config_json():
    """
    Find config.json using priority order:
    1. environment.py configuration (if exists)
    2. Current working directory
    3. Environment variable
    4. User home directory
    5. Package directory
    """
    # Priority 1: Check environment.py for DEV/PROD configuration
    try:
        # Import from package
        from . import environment
        config_path = Path(environment.get_config_path())
        
        if config_path.exists():
            resolved_path = str(config_path.resolve())
            print(f"✓ Using {environment.ACTIVE_ENVIRONMENT} config from environment.py: {resolved_path}", file=sys.stderr)
            return resolved_path
        else:
            print(f"⚠️  environment.py configured but config not found: {config_path}", file=sys.stderr)
    except (ImportError, AttributeError, FileNotFoundError):
        # environment.py doesn't exist or is invalid, fall back to auto-detection
        pass
    
    # Priority 2-5: Auto-detection
    search_paths = [
        # 2. Current working directory (highest priority)
        Path.cwd() / "config.json",
        
        # 3. Environment variable
        os.getenv('NLP_CONFIG_PATH'),
        
        # 4. User home directory
        Path.home() / ".enable_ai" / "config.json",
        
        # 5. Package directory (fallback)
        Path(__file__).parent.parent.parent / "config.json"
    ]
    
    for path in search_paths:
        if path and Path(path).exists():
            config_path = str(Path(path).resolve())
            print(f"✓ Found config.json at: {config_path}", file=sys.stderr)
            return config_path
    
    print("⚠ No config.json found - processor will use defaults", file=sys.stderr)
    return None


# Auto-detect environment and config on module import
DEFAULT_ENV_PATH = find_env_file()
DEFAULT_CONFIG_PATH = find_config_json()


# Initialize the MCP server
app = Server("enable_ai")

# Global orchestrator instance
orchestrator: Optional[APIOrchestrator] = None


def get_orchestrator(config_path: Optional[str] = None) -> APIOrchestrator:
    """Get or create the API Orchestrator instance. Conversation state is via conversation_store (no LangGraph checkpointer)."""
    global orchestrator
    
    if orchestrator is None:
        # Use provided config_path, or auto-detected default
        if not config_path:
            config_path = DEFAULT_CONFIG_PATH
        
        # Redirect stdout to stderr to prevent debug prints from interfering with JSON-RPC
        old_stdout = sys.stdout
        sys.stdout = sys.stderr
        
        try:
            if config_path:
                orchestrator = APIOrchestrator(config_path=config_path)
            else:
                orchestrator = APIOrchestrator()
        finally:
            sys.stdout = old_stdout
    
    return orchestrator


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools for the MCP client."""
    return [
        Tool(
            name="process_query",
            description="""
Process a natural language query against configured API data sources.
This is the main tool that understands natural language, matches it to the appropriate endpoint,
authenticates if needed, and executes the query.

Examples:
- "list all users"
- "get user with id 5"
- "create a new service order for customer ABC"
- "find documents about machine learning"
- "show me service orders with high priority"
            """.strip(),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language query to process"
                    },
                    "access_token": {
                        "type": "string",
                        "description": "Optional JWT access token for authentication. If not provided, will authenticate automatically."
                    },
                    "session_id": {
                        "type": "string",
                        "description": "Optional session ID for multi-turn conversations. Use the same session_id to maintain context across queries."
                    },
                    "config_path": {
                        "type": "string",
                        "description": "Optional path to config.json file. If not provided, will use default config."
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="get_schema_resources",
            description="""
Get a list of all available resources (endpoints) from the loaded schemas.
Useful to know what queries you can make.

Returns: List of resource names and descriptions for API, database tables, or knowledge graph entities.
            """.strip(),
            inputSchema={
                "type": "object",
                "properties": {
                    "schema_type": {
                        "type": "string",
                        "description": "Type of schema to query: 'api' (API-only for now)",
                        "enum": ["api"]
                    },
                    "config_path": {
                        "type": "string",
                        "description": "Optional path to config.json file"
                    }
                },
                "required": ["schema_type"]
            }
        ),
        Tool(
            name="authenticate",
            description="""
Authenticate with the API and get an access token.
This is useful if you want to get a token first and reuse it for multiple queries.

Returns: JWT access token that can be used with process_query.
            """.strip(),
            inputSchema={
                "type": "object",
                "properties": {
                    "config_path": {
                        "type": "string",
                        "description": "Optional path to config.json file"
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="get_config_info",
            description="""
Get information about the current configuration including:
- Enabled data sources
- Loaded schemas
- Authentication methods
- Base URLs

Useful for debugging or understanding the current setup.
            """.strip(),
            inputSchema={
                "type": "object",
                "properties": {
                    "config_path": {
                        "type": "string",
                        "description": "Optional path to config.json file"
                    }
                },
                "required": []
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """Handle tool calls from the MCP client."""
    
    try:
        if name == "process_query":
            return await handle_process_query(arguments)
        elif name == "get_schema_resources":
            return await handle_get_schema_resources(arguments)
        elif name == "authenticate":
            return await handle_authenticate(arguments)
        elif name == "get_config_info":
            return await handle_get_config_info(arguments)
        else:
            return [TextContent(
                type="text",
                text=f"Unknown tool: {name}"
            )]
    
    except Exception as e:
        return [TextContent(
            type="text",
            text=constants.MCP_ERROR_TOOL_EXEC.format(name=name, error=str(e))
        )]


async def handle_process_query(arguments: dict) -> list[TextContent]:
    """Handle the process_query tool call."""
    query = arguments.get("query")
    access_token = arguments.get("access_token")
    config_path = arguments.get("config_path")
    session_id = arguments.get("session_id")
    
    if not query:
        return [TextContent(
            type="text",
            text=constants.ERROR_QUERY_REQUIRED
        )]
    
    # Get orchestrator
    orch = get_orchestrator(config_path)
    
    # Process the query
    result = orch.process(query, access_token=access_token, session_id=session_id)
    
    # Format the response
    response = {
        "success": result.get("success"),
        "query": result.get("query"),
        "summary": result.get("summary"),
        "data": result.get("data"),
        "endpoint": result.get("endpoint"),
        "method": result.get("method"),
        "schema_type": result.get("schema_type")
    }
    
    # Add error if present
    if result.get("error"):
        response["error"] = result.get("error")
    
    # Add needs_info if present
    if result.get("needs_info"):
        response["needs_info"] = result.get("needs_info")
        response["message"] = result.get("message")
        response["missing_fields"] = result.get("missing_fields")
    
    return [TextContent(
        type="text",
        text=json.dumps(response, indent=2)
    )]


async def handle_get_schema_resources(arguments: dict) -> list[TextContent]:
    """Handle the get_schema_resources tool call."""
    schema_type = arguments.get("schema_type")
    config_path = arguments.get("config_path")
    
    if not schema_type:
        return [TextContent(
            type="text",
            text=constants.ERROR_SCHEMA_TYPE_REQUIRED
        )]

    if schema_type != "api":
        return [TextContent(
            type="text",
            text=constants.ERROR_ONLY_API_SCHEMA_SUPPORTED
        )]
    
    # Get orchestrator
    orch = get_orchestrator(config_path)
    
    # Get the schema
    schema = orch.schemas.get(schema_type)
    
    if not schema:
        return [TextContent(
            type="text",
            text=f"Schema type '{schema_type}' not loaded. Available schemas: {list(orch.schemas.keys())}"
        )]
    
    # Extract resources based on schema type
    resources = {}
    
    if schema_type == "api":
        for resource_name, resource_data in schema.get("resources", {}).items():
            endpoints = resource_data.get("endpoints", [])
            resources[resource_name] = {
                "description": resource_data.get("description", ""),
                "endpoint_count": len(endpoints),
                "methods": list(set(ep.get("method") for ep in endpoints if isinstance(endpoints, list)))
            }
    
    elif schema_type == "database":
        for table_name, table_data in schema.get("tables", {}).items():
            columns = table_data.get("columns", {})
            resources[table_name] = {
                "description": table_data.get("description", ""),
                "column_count": len(columns),
                "columns": list(columns.keys())
            }
    
    elif schema_type == "knowledge_graph":
        for entity_name, entity_data in schema.get("entities", {}).items():
            resources[entity_name] = {
                "description": entity_data.get("description", ""),
                "properties": entity_data.get("properties", [])
            }
    
    response = {
        "schema_type": schema_type,
        "resource_count": len(resources),
        "resources": resources
    }
    
    return [TextContent(
        type="text",
        text=json.dumps(response, indent=2)
    )]


async def handle_authenticate(arguments: dict) -> list[TextContent]:
    """Handle the authenticate tool call."""
    config_path = arguments.get("config_path")
    
    # Get orchestrator
    orch = get_orchestrator(config_path)
    
    # Get schema to detect auth type
    schema = orch.schemas.get("api")
    
    if not schema:
        return [TextContent(
            type="text",
            text=constants.ERROR_NO_API_SCHEMA_LOADED
        )]
    
    # Detect auth type
    auth_type = orch._detect_auth_type(schema)
    
    if not auth_type:
        return [TextContent(
            type="text",
            text=constants.ERROR_NO_AUTH_CONFIGURED
        )]
    
    # Get base URL
    api_config = orch.config.get('data_sources', {}).get('api', {})
    base_url = api_config.get('base_url') or schema.get('base_url')
    
    if not base_url:
        return [TextContent(
            type="text",
            text=constants.ERROR_NO_BASE_URL
        )]
    
    # Authenticate
    auth_result = orch._authenticate(base_url, auth_type)
    
    if auth_result.get('error'):
        return [TextContent(
            type="text",
            text=f"Authentication failed: {auth_result['error']}"
        )]
    
    token = auth_result.get('token')
    
    response = {
        "success": True,
        "auth_type": auth_type,
        "token": token,
        "message": constants.MCP_AUTH_SUCCESS_MESSAGE
    }
    
    return [TextContent(
        type="text",
        text=json.dumps(response, indent=2)
    )]


async def handle_get_config_info(arguments: dict) -> list[TextContent]:
    """Handle the get_config_info tool call."""
    config_path = arguments.get("config_path")
    
    # Get orchestrator
    orch = get_orchestrator(config_path)
    
    # Gather config info
    enabled_sources = []
    data_sources = orch.config.get('data_sources', {})
    
    for source_name, source_config in data_sources.items():
        if isinstance(source_config, dict) and source_config.get('enabled'):
            enabled_sources.append({
                "name": source_name,
                "type": source_config.get("type"),
                "base_url": source_config.get("base_url")
            })
    
    # Get loaded schemas
    loaded_schemas = {}
    for schema_type, schema in orch.schemas.items():
        loaded_schemas[schema_type] = {
            "type": schema.get("type"),
            "resource_count": len(schema.get("resources", schema.get("tables", schema.get("entities", {}))))
        }
    
    # Get auth info
    security = orch.config.get('security_credentials', {}).get('api', {})
    auth_methods = []
    
    if security.get('jwt', {}).get('enabled'):
        auth_methods.append("JWT")
    if security.get('oauth', {}).get('enabled'):
        auth_methods.append("OAuth 2.0")
    if security.get('api_keys', {}).get('enabled'):
        auth_methods.append("API Key")
    
    response = {
        "enabled_data_sources": enabled_sources,
        "loaded_schemas": loaded_schemas,
        "authentication_methods": auth_methods,
        "config_path": config_path or "default"
    }
    
    return [TextContent(
        type="text",
        text=json.dumps(response, indent=2)
    )]


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
