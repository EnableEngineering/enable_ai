"""
Schema Generator - Auto-generate schemas from various sources

Converts existing documentation/data sources into NLP API Caller schemas:
- OpenAPI/Swagger → API Schema
- Database → Database Schema
- JSON Files → Knowledge Graph Schema
- PDF Documents → Knowledge Graph Schema
"""

from .openapi_converter import OpenAPIConverter, convert_openapi
from .database_inspector import DatabaseInspector, inspect_database
from .json_analyzer import JSONAnalyzer, analyze_json
from .pdf_analyzer import PDFAnalyzer, analyze_pdfs

__all__ = [
    'OpenAPIConverter',
    'DatabaseInspector',
    'JSONAnalyzer',
    'PDFAnalyzer',
    'convert_openapi',
    'inspect_database',
    'analyze_json',
    'analyze_pdfs'
]
