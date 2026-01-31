"""
Schema Generator - Auto-generate API schemas from OpenAPI/Swagger

Converts OpenAPI/Swagger specifications into Enable AI API schemas.

API-only focus - Database and document analysis planned for future extensions.
"""

from .schema_converter import OpenAPIConverter, convert_openapi

__all__ = [
    'OpenAPIConverter',
    'convert_openapi',
]
