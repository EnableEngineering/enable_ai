#!/usr/bin/env python3.11
"""
Test script for MCP server - sends test messages via stdio
"""

import json
import subprocess
import sys

def send_mcp_request(method: str, params: dict = None):
    """Send a JSON-RPC request to the MCP server."""
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": method,
        "params": params or {}
    }
    
    # Start the MCP server
    proc = subprocess.Popen(
        [sys.executable, "-m", "enable_ai.mcp_server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Send initialization
    init_request = {
        "jsonrpc": "2.0",
        "id": 0,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "test-client",
                "version": "1.0.0"
            }
        }
    }
    
    proc.stdin.write(json.dumps(init_request) + "\n")
    proc.stdin.flush()
    
    # Wait for initialization response
    init_response = proc.stdout.readline()
    if not init_response:
        proc.stdin.close()
        output = proc.stdout.read()
        errors = proc.stderr.read()
        print("Failed to get init response")
        if errors:
            print("STDERR:", errors)
        return []
    
    # Send initialized notification
    initialized_notification = {
        "jsonrpc": "2.0",
        "method": "notifications/initialized"
    }
    proc.stdin.write(json.dumps(initialized_notification) + "\n")
    proc.stdin.flush()
    
    # Send the actual request
    proc.stdin.write(json.dumps(request) + "\n")
    proc.stdin.flush()
    proc.stdin.close()
    
    # Read response
    output = proc.stdout.read()
    errors = proc.stderr.read()
    
    if errors:
        print("STDERR:", errors)
    
    # Parse JSON-RPC responses
    responses = []
    for line in output.strip().split("\n"):
        if line.strip():
            try:
                responses.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Failed to parse: {line}")
    
    return responses


def test_list_tools():
    """Test listing available tools."""
    print("=" * 60)
    print("TEST: List Available Tools")
    print("=" * 60)
    
    responses = send_mcp_request("tools/list")
    
    for resp in responses:
        if resp.get("method") == "tools/list":
            continue
        print(json.dumps(resp, indent=2))
    print()


def test_config_info():
    """Test getting configuration info."""
    print("=" * 60)
    print("TEST: Get Configuration Info")
    print("=" * 60)
    
    responses = send_mcp_request("tools/call", {
        "name": "get_config_info",
        "arguments": {}
    })
    
    for resp in responses:
        print(json.dumps(resp, indent=2))
    print()


def test_schema_resources():
    """Test getting schema resources."""
    print("=" * 60)
    print("TEST: Get Schema Resources")
    print("=" * 60)
    
    responses = send_mcp_request("tools/call", {
        "name": "get_schema_resources",
        "arguments": {
            "schema_type": "api"
        }
    })
    
    for resp in responses:
        print(json.dumps(resp, indent=2))
    print()


def test_query_processing():
    """Test processing a natural language query."""
    print("=" * 60)
    print("TEST: Process Query - 'list all users'")
    print("=" * 60)
    
    responses = send_mcp_request("tools/call", {
        "name": "process_query",
        "arguments": {
            "query": "list all users",
            "source_preference": "api"
        }
    })
    
    for resp in responses:
        print(json.dumps(resp, indent=2))
    print()


if __name__ == "__main__":
    print("\n🧪 Testing Enable AI MCP Server\n")
    
    try:
        test_list_tools()
        test_config_info()
        test_schema_resources()
        # test_query_processing()  # Uncomment to test actual query processing
        
        print("✅ MCP Server tests completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
