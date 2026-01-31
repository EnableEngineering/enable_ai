#!/usr/bin/env python3
"""
Enable AI - Basic Usage Example

Demonstrates:
1. Loading API schema
2. Processing natural language queries
3. Smart response formatting (auto, table, grouped, etc.)
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports when running locally
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from enable_ai import APIOrchestrator, ResponseFormatter


def main():
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║         Enable AI - Basic Usage Example                  ║")
    print("╚═══════════════════════════════════════════════════════════╝")
    print()
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not set in environment")
        print("   Set it with: export OPENAI_API_KEY='your-key'")
        print()
        return
    
    # Initialize components
    print("📦 Initializing Enable AI components...")
    orchestrator = APIOrchestrator()
    formatter = ResponseFormatter()
    print("✓ Initialized")
    print()
    
    # Example 1: Basic query processing
    print("=" * 60)
    print("Example 1: Basic Query Processing")
    print("=" * 60)
    
    query = "Get all users"
    print(f"Query: {query}")
    print()
    
    try:
        result = orchestrator.process(query)
        print("✓ Query processed successfully")
        print(f"Summary: {result.get('summary', 'No summary')}")
        print()
    except Exception as e:
        print(f"✗ Error: {e}")
        print()
    
    # Example 2: Smart formatting - Auto format
    print("=" * 60)
    print("Example 2: Smart Formatting (Auto)")
    print("=" * 60)
    
    # Sample inventory data (low stock items)
    sample_data = [
        {
            "name": "STATIONARY MACHINE -8000 AMP",
            "type": "Equipment",
            "serial": "98568",
            "quantity": 1,
            "location": "Main Branch - Pune"
        },
        {
            "name": "STATIONARY MACHINE -3000 AMP",
            "type": "Equipment",
            "serial": "98555",
            "quantity": 1,
            "location": "Main Branch - Pune"
        },
        {
            "name": "MAGNAFLUX SKL-SP1",
            "type": "Consumable",
            "quantity_ml": 1.0,
            "unit_cost": 1.60,
            "expiry": "2028-05-30",
            "location": "Main Branch - Pune"
        },
        {
            "name": "MAGNAFLUX MG-2410",
            "type": "Consumable",
            "quantity_ml": 2.0,
            "unit_cost": 1.80,
            "expiry": "2028-09-30",
            "location": "Main Branch - Pune"
        }
    ]
    
    query = "Show items that are low in stock grouped by type"
    print(f"Query: {query}")
    print()
    
    try:
        result = formatter.format_response(
            data=sample_data,
            query=query,
            format_type="auto"  # LLM chooses best format
        )
        
        print(f"✓ LLM chose format: {result['format']}")
        print()
        print("Formatted Output:")
        print("-" * 60)
        print(result['formatted'])
        print()
    except Exception as e:
        print(f"✗ Error: {e}")
        print()
    
    # Example 3: Force table format
    print("=" * 60)
    print("Example 3: Force Table Format")
    print("=" * 60)
    
    try:
        result = formatter.format_response(
            data=sample_data,
            query=query,
            format_type="table"  # Force table format
        )
        
        print("Formatted as Table:")
        print("-" * 60)
        print(result['formatted'])
        print()
    except Exception as e:
        print(f"✗ Error: {e}")
        print()
    
    # Example 4: Concise summary
    print("=" * 60)
    print("Example 4: Concise Summary")
    print("=" * 60)
    
    try:
        result = formatter.format_response(
            data=sample_data,
            query=query,
            format_type="concise"  # Brief summary
        )
        
        print("Concise Summary:")
        print("-" * 60)
        print(result['formatted'])
        print()
    except Exception as e:
        print(f"✗ Error: {e}")
        print()
    
    # Example 5: Chart-ready format
    print("=" * 60)
    print("Example 5: Chart-Ready JSON")
    print("=" * 60)
    
    try:
        result = formatter.format_response(
            data=sample_data,
            query=query,
            format_type="chart"  # Chart data
        )
        
        print("Chart-Ready JSON:")
        print("-" * 60)
        print(result['formatted'])
        print()
    except Exception as e:
        print(f"✗ Error: {e}")
        print()
    
    print("=" * 60)
    print("✅ All examples completed!")
    print()
    print("Next Steps:")
    print("  1. Try with your own API schema in config.json")
    print("  2. Test different queries")
    print("  3. Try streaming_backend.py for real-time progress")
    print("=" * 60)


if __name__ == "__main__":
    main()
