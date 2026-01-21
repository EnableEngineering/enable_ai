#!/usr/bin/env python3
"""
Schema Generator CLI

Command-line interface for generating schemas from various sources.

Usage:
    # OpenAPI/Swagger
    nlp-schema generate --from openapi --input swagger.json --output api_schema.json
    
    # Database
    nlp-schema generate --from database --connection postgresql://user:pass@localhost/db --output db_schema.json
    
    # JSON files
    nlp-schema generate --from json --input data/ --recursive --output kg_schema.json
    
    # PDF documents
    nlp-schema generate --from pdf --input docs/ --use-ner --output kg_schema.json
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Import converters
try:
    from nlp_api_caller.schema_generator import (
        OpenAPIConverter,
        DatabaseInspector,
        JSONAnalyzer,
        PDFAnalyzer
    )
except ImportError:
    # Development mode - adjust import path
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from schema_generator import (
        OpenAPIConverter,
        DatabaseInspector,
        JSONAnalyzer,
        PDFAnalyzer
    )


def main():
    parser = argparse.ArgumentParser(
        description='Generate NLP API Caller schemas from various sources',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # OpenAPI/Swagger to API schema
  %(prog)s generate --from openapi --input swagger.json --output api_schema.json
  
  # Database introspection to database schema
  %(prog)s generate --from database --connection "postgresql://user:pass@localhost/mydb" --output db_schema.json
  
  # JSON files to knowledge graph schema
  %(prog)s generate --from json --input data/ --recursive --output kg_schema.json
  
  # PDF documents to knowledge graph schema
  %(prog)s generate --from pdf --input docs/ --use-ner --output kg_schema.json
  
  # Override base URL for OpenAPI
  %(prog)s generate --from openapi --input swagger.json --base-url https://api.example.com --output api_schema.json
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate schema from source')
    gen_parser.add_argument(
        '--from',
        dest='source_type',
        required=True,
        choices=['openapi', 'database', 'json', 'pdf'],
        help='Source type to convert from'
    )
    gen_parser.add_argument(
        '--input',
        dest='input_path',
        help='Input file/directory path (for openapi, json, pdf)'
    )
    gen_parser.add_argument(
        '--connection',
        dest='connection_string',
        help='Database connection string (for database)'
    )
    gen_parser.add_argument(
        '--output',
        dest='output_path',
        help='Output schema file path (optional, defaults to schemas/ directory)'
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
        help='Include deprecated endpoints (OpenAPI)'
    )
    
    # Database options
    gen_parser.add_argument(
        '--tables',
        dest='table_filter',
        nargs='+',
        help='Filter specific tables (database)'
    )
    gen_parser.add_argument(
        '--include-views',
        dest='include_views',
        action='store_true',
        help='Include database views (database)'
    )
    
    # JSON/PDF options
    gen_parser.add_argument(
        '--recursive',
        dest='recursive',
        action='store_true',
        help='Scan directories recursively (json, pdf)'
    )
    
    # PDF options
    gen_parser.add_argument(
        '--use-ner',
        dest='use_ner',
        action='store_true',
        default=True,
        help='Use NER for entity extraction (pdf)'
    )
    gen_parser.add_argument(
        '--no-ner',
        dest='use_ner',
        action='store_false',
        help='Disable NER, use pattern matching (pdf)'
    )
    
    # Common options
    gen_parser.add_argument(
        '--entity-threshold',
        dest='entity_threshold',
        type=int,
        default=5,
        help='Min occurrences for entity type (json, pdf)'
    )
    gen_parser.add_argument(
        '--sample-limit',
        dest='sample_limit',
        type=int,
        help='Max files to analyze (json, pdf)'
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    if args.command == 'generate':
        return generate_schema(args)
    
    return 0


def generate_schema(args) -> int:
    """Generate schema based on arguments."""
    source_type = args.source_type
    output_path = args.output_path
    
    try:
        if source_type == 'openapi':
            return generate_from_openapi(args)
        elif source_type == 'database':
            return generate_from_database(args)
        elif source_type == 'json':
            return generate_from_json(args)
        elif source_type == 'pdf':
            return generate_from_pdf(args)
        else:
            print(f"❌ Unknown source type: {source_type}")
            return 1
            
    except Exception as e:
        print(f"❌ Error generating schema: {e}")
        import traceback
        traceback.print_exc()
        return 1


def generate_from_openapi(args) -> int:
    """Generate API schema from OpenAPI spec."""
    if not args.input_path:
        print("❌ --input is required for OpenAPI conversion")
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
    
    # Save schema (uses default schemas/ if no output_path)
    saved_path = converter.save_schema(schema, args.output_path)
    
    print(f"✅ API schema generated successfully!")
    print(f"   📄 Output: {saved_path}")
    print(f"   📊 Resources: {len(schema.get('resources', {}))}")
    
    return 0


def generate_from_database(args) -> int:
    """Generate database schema from live database."""
    if not args.connection_string:
        print("❌ --connection is required for database introspection")
        return 1
    
    print(f"🔄 Introspecting database...")
    
    inspector = DatabaseInspector()
    
    # Build kwargs
    kwargs = {}
    if args.table_filter:
        kwargs['table_filter'] = args.table_filter
    if args.include_views:
        kwargs['include_views'] = True
    
    # Generate schema
    schema = inspector.generate(args.connection_string, **kwargs)
    
    # Save schema (uses default schemas/ if no output_path)
    saved_path = inspector.save_schema(schema, args.output_path)
    
    print(f"✅ Database schema generated successfully!")
    print(f"   📄 Output: {saved_path}")
    print(f"   📊 Tables: {len(schema.get('tables', {}))}")
    
    return 0


def generate_from_json(args) -> int:
    """Generate knowledge graph schema from JSON files."""
    if not args.input_path:
        print("❌ --input is required for JSON analysis")
        return 1
    
    print(f"🔄 Analyzing JSON files: {args.input_path}")
    
    analyzer = JSONAnalyzer()
    
    # Build kwargs
    kwargs = {}
    if args.recursive:
        kwargs['recursive'] = True
    if args.entity_threshold:
        kwargs['entity_threshold'] = args.entity_threshold
    if args.sample_limit:
        kwargs['sample_limit'] = args.sample_limit
    
    # Generate schema
    schema = analyzer.generate(args.input_path, **kwargs)
    
    # Save schema (uses default schemas/ if no output_path)
    saved_path = analyzer.save_schema(schema, args.output_path)
    
    print(f"✅ Knowledge graph schema generated successfully!")
    print(f"   📄 Output: {saved_path}")
    print(f"   📊 Entity types: {len(schema.get('entities', {}))}")
    print(f"   🔗 Relationships: {len(schema.get('relationships', []))}")
    
    return 0


def generate_from_pdf(args) -> int:
    """Generate knowledge graph schema from PDF documents."""
    if not args.input_path:
        print("❌ --input is required for PDF analysis")
        return 1
    
    print(f"🔄 Analyzing PDF documents: {args.input_path}")
    
    analyzer = PDFAnalyzer()
    
    # Build kwargs
    kwargs = {}
    if args.recursive:
        kwargs['recursive'] = True
    if hasattr(args, 'use_ner'):
        kwargs['use_ner'] = args.use_ner
    if args.entity_threshold:
        kwargs['entity_threshold'] = args.entity_threshold
    if args.sample_limit:
        kwargs['sample_limit'] = args.sample_limit
    
    # Generate schema
    schema = analyzer.generate(args.input_path, **kwargs)
    
    # Save schema (uses default schemas/ if no output_path)
    saved_path = analyzer.save_schema(schema, args.output_path)
    
    print(f"✅ Knowledge graph schema generated successfully!")
    print(f"   📄 Output: {saved_path}")
    print(f"   📊 Entity types: {len(schema.get('entities', {}))}")
    print(f"   🔗 Relationships: {len(schema.get('relationships', []))}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
