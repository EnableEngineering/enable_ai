#!/usr/bin/env python3
"""
Schema Generator CLI - API-only

Command-line interface for generating API schemas from OpenAPI/Swagger specifications.

Usage:
    # Generate API schema from OpenAPI/Swagger
    enable-schema generate --input swagger.json --output api_schema.json
    
    # With custom base URL
    enable-schema generate --input swagger.json --base-url https://api.example.com --output api_schema.json
    
    # Include deprecated endpoints
    enable-schema generate --input swagger.json --include-deprecated --output api_schema.json
"""

import argparse
import sys
from pathlib import Path

# Import converter
try:
    from enable_ai.schema_generator import OpenAPIConverter
except ImportError:
    # Development mode - adjust import path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from schema_generator.schema_converter import OpenAPIConverter


def main():
    parser = argparse.ArgumentParser(
        description='Generate Enable AI API schemas from OpenAPI/Swagger specifications',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic conversion
  %(prog)s generate --input swagger.json --output api_schema.json
  
  # Override base URL
  %(prog)s generate --input swagger.json --base-url https://api.example.com --output api_schema.json
  
  # Include deprecated endpoints
  %(prog)s generate --input swagger.json --include-deprecated --output api_schema.json

Note:
  This tool focuses on API schema generation only. Database and document analysis
  are planned for future extensions.
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate API schema from OpenAPI spec')
    gen_parser.add_argument(
        '--input',
        dest='input_path',
        required=True,
        help='Input OpenAPI/Swagger file path (JSON or YAML)'
    )
    gen_parser.add_argument(
        '--output',
        dest='output_path',
        help='Output schema file path (optional, defaults to schemas/api_schema.json)'
    )
    
    # OpenAPI options
    gen_parser.add_argument(
        '--base-url',
        dest='base_url',
        help='Override base URL from OpenAPI spec'
    )
    gen_parser.add_argument(
        '--include-deprecated',
        dest='include_deprecated',
        action='store_true',
        help='Include deprecated endpoints'
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    if args.command == 'generate':
        return generate_schema(args)
    
    return 0


def generate_schema(args) -> int:
    """Generate API schema from OpenAPI specification."""
    try:
        if not args.input_path:
            print("❌ --input is required")
            return 1
        
        print(f"🔄 Converting OpenAPI spec: {args.input_path}")
        
        converter = OpenAPIConverter()
        
        # Build kwargs
        kwargs = {}
        if args.base_url:
            kwargs['base_url'] = args.base_url
        if args.include_deprecated:
            kwargs['include_deprecated'] = True
        
        # Generate schema
        schema = converter.generate(args.input_path, **kwargs)
        
        # Save schema (uses default schemas/api_schema.json if no output_path)
        saved_path = converter.save_schema(schema, args.output_path)
        
        print(f"✅ API schema generated successfully!")
        print(f"   📄 Output: {saved_path}")
        print(f"   📊 Resources: {len(schema.get('resources', {}))}")
        print(f"   🌐 Base URL: {schema.get('base_url', 'N/A')}")
        
        return 0
            
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        return 1
    except Exception as e:
        print(f"❌ Error generating schema: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
