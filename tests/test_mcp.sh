#!/bin/bash

# Simple MCP Server Test Script
# Tests the MCP server by sending JSON-RPC messages via stdio

echo "🧪 Testing Enable AI MCP Server"
echo "======================================"
echo ""

# Test 1: List available tools
echo "📋 Test 1: List Available Tools"
echo '{"jsonrpc":"2.0","id":0,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}' | python3.11 -m enable_ai.mcp_server 2>&1 | grep -v "^$"
echo ""

# Test 2: Simple echo test
echo "📋 Test 2: Start MCP Server (press Ctrl+C to stop)"
echo "You can now send JSON-RPC messages manually..."
echo ""
python3.11 -m enable_ai.mcp_server
