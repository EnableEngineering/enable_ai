#!/usr/bin/env python3
"""
Enable AI v2 - Basic Usage Example

Demonstrates:
1. Creating Config with OpenAPI spec
2. Processing natural language queries with Claude's native tool calling
3. Using resource hints for better accuracy
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports when running locally
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from enable_ai_v2 import (
    Orchestrator,
    Config,
    JWTAuth,
    ResourceHint,
    UserContext,
)


def main():
    print("Enable AI v2 - Basic Usage Example")
    print("=" * 40)
    print()

    # Check for Anthropic API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Warning: ANTHROPIC_API_KEY not set")
        print("Set it with: export ANTHROPIC_API_KEY='your-key'")
        return

    # Sample OpenAPI spec (inline for demo)
    sample_spec = {
        "openapi": "3.0.0",
        "info": {"title": "Demo API", "version": "1.0.0"},
        "paths": {
            "/users": {
                "get": {
                    "operationId": "list_users",
                    "summary": "List all users",
                    "parameters": [
                        {
                            "name": "status",
                            "in": "query",
                            "schema": {"type": "string", "enum": ["active", "inactive"]},
                        },
                    ],
                    "responses": {"200": {"description": "OK"}},
                },
            },
            "/users/{id}": {
                "get": {
                    "operationId": "get_user",
                    "summary": "Get a user by ID",
                    "parameters": [
                        {
                            "name": "id",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "integer"},
                        },
                    ],
                    "responses": {"200": {"description": "OK"}},
                },
            },
            "/orders": {
                "get": {
                    "operationId": "list_orders",
                    "summary": "List orders",
                    "parameters": [
                        {
                            "name": "status",
                            "in": "query",
                            "schema": {"type": "string", "enum": ["pending", "completed", "cancelled"]},
                        },
                    ],
                    "responses": {"200": {"description": "OK"}},
                },
            },
        },
    }

    # Config - parent module provides everything
    config = Config(
        openapi_schema=sample_spec,
        base_url="https://api.example.com",  # Replace with real API

        # Resource hints improve accuracy
        resource_hints={
            "orders": ResourceHint(
                status_field="status",
                status_values=["pending", "completed", "cancelled"],
            ),
            "users": ResourceHint(
                status_field="status",
                status_values=["active", "inactive"],
            ),
        },

        # Status synonyms map natural language to actual values
        status_synonyms={
            "open": "pending",
            "done": "completed",
            "closed": "completed",
        },

        # Enable tracing for debugging
        include_trace=True,
    )

    # For real API, use actual auth:
    # auth = JWTAuth(token="your-jwt-token")

    print("Initializing...")
    ai = Orchestrator(config=config)
    print("Ready")
    print()

    # Example queries
    queries = [
        "Show me all active users",
        "Get user 12345",
        "What pending orders do we have?",
    ]

    for query in queries:
        print("-" * 40)
        print(f"Query: {query}")

        try:
            # Optionally provide user context
            user_ctx = UserContext(
                user_id=1,
                role="Admin",
                is_admin=True,
            )

            result = ai.process(query, user_context=user_ctx)

            print(f"Response: {result.message}")
            print(f"Success: {result.success}")

            if result.trace:
                print(f"Tool calls: {[tc.name for tc in result.trace.tool_calls]}")
                print(f"Response time: {result.trace.response_time_ms}ms")

        except Exception as e:
            print(f"Error: {e}")

        print()

    ai.close()
    print("Done!")


if __name__ == "__main__":
    main()
