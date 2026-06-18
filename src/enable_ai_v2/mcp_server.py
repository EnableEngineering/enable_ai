"""
MCP Server for Enable AI v2.

Exposes the NL-to-API functionality through Model Context Protocol (MCP)
for integration with AI applications like Claude Desktop.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

# MCP imports (optional dependency)
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

from .orchestrator import Orchestrator
from .auth import JWTAuth, APIKeyAuth, NoAuth
from .config import Config


def create_mcp_server(
    openapi_spec: str | Path | dict,
    base_url: str,
    auth_token: Optional[str] = None,
    api_key: Optional[str] = None,
    config: Optional[Config] = None,
) -> Any:
    """
    Create an MCP server instance.

    Args:
        openapi_spec: Path to OpenAPI spec or spec dict
        base_url: Base URL for API calls
        auth_token: Optional JWT token for authentication
        api_key: Optional API key for authentication
        config: Optional configuration

    Returns:
        MCP Server instance

    Raises:
        ImportError: If MCP SDK is not installed
    """
    if not MCP_AVAILABLE:
        raise ImportError(
            "MCP SDK not installed. Install with: pip install mcp"
        )

    # Set up authentication
    if auth_token:
        auth = JWTAuth(token=auth_token)
    elif api_key:
        auth = APIKeyAuth(api_key=api_key)
    else:
        auth = NoAuth()

    # Create orchestrator
    orchestrator = Orchestrator(
        openapi_spec=openapi_spec,
        base_url=base_url,
        auth=auth,
        config=config or Config(),
    )

    # Create MCP server
    server = Server("enable-ai")

    @server.list_tools()
    async def list_tools():
        """List available tools."""
        return [
            Tool(
                name="process_query",
                description=(
                    "Process a natural language query against a REST API. "
                    "Converts natural language to API calls and returns results."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language query (e.g., 'show pending orders')",
                        },
                        "session_id": {
                            "type": "string",
                            "description": "Optional session ID for conversation context",
                        },
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="list_endpoints",
                description="List all available API endpoints and their descriptions.",
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
            Tool(
                name="clear_cache",
                description="Clear the query cache.",
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list:
        """Handle tool calls."""
        if name == "process_query":
            query = arguments.get("query", "")
            session_id = arguments.get("session_id")

            try:
                result = orchestrator.process(query, session_id=session_id)

                response_text = result.message
                if result.data:
                    response_text += f"\n\nData:\n```json\n{json.dumps(result.data, indent=2, default=str)}\n```"

                if result.suggestions:
                    response_text += "\n\nSuggested follow-ups:\n"
                    for suggestion in result.suggestions:
                        response_text += f"- {suggestion}\n"

                return [TextContent(type="text", text=response_text)]

            except Exception as e:
                return [TextContent(type="text", text=f"Error: {str(e)}")]

        elif name == "list_endpoints":
            endpoints = []
            for ep in orchestrator.endpoints:
                endpoints.append({
                    "name": ep.tool_name,
                    "method": ep.method,
                    "path": ep.path,
                    "summary": ep.summary,
                })

            return [
                TextContent(
                    type="text",
                    text=f"Available endpoints:\n```json\n{json.dumps(endpoints, indent=2)}\n```",
                )
            ]

        elif name == "clear_cache":
            orchestrator.clear_cache()
            return [TextContent(type="text", text="Cache cleared.")]

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

    return server


async def run_mcp_server(
    openapi_spec: str | Path | dict,
    base_url: str,
    auth_token: Optional[str] = None,
    api_key: Optional[str] = None,
):
    """
    Run the MCP server (stdio mode).

    Args:
        openapi_spec: Path to OpenAPI spec or spec dict
        base_url: Base URL for API calls
        auth_token: Optional JWT token
        api_key: Optional API key
    """
    if not MCP_AVAILABLE:
        print("Error: MCP SDK not installed. Install with: pip install mcp", file=sys.stderr)
        sys.exit(1)

    server = create_mcp_server(
        openapi_spec=openapi_spec,
        base_url=base_url,
        auth_token=auth_token,
        api_key=api_key,
    )

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream)


def main():
    """
    CLI entry point for MCP server.

    Environment variables:
        OPENAPI_SPEC_PATH: Path to OpenAPI spec file
        API_BASE_URL: Base URL for API
        API_AUTH_TOKEN: JWT token (optional)
        API_KEY: API key (optional)
    """
    spec_path = os.environ.get("OPENAPI_SPEC_PATH")
    base_url = os.environ.get("API_BASE_URL")
    auth_token = os.environ.get("API_AUTH_TOKEN")
    api_key = os.environ.get("API_KEY")

    if not spec_path or not base_url:
        print(
            "Error: Set OPENAPI_SPEC_PATH and API_BASE_URL environment variables",
            file=sys.stderr,
        )
        sys.exit(1)

    asyncio.run(
        run_mcp_server(
            openapi_spec=spec_path,
            base_url=base_url,
            auth_token=auth_token,
            api_key=api_key,
        )
    )


if __name__ == "__main__":
    main()
