"""
Schema Loader - Auto-load API schemas from multiple sources

Supports:
- File paths (local JSON files)
- URLs (fetch from Swagger/OpenAPI endpoints)
- Auto-conversion (OpenAPI → Enable AI schema format)
- Multiple formats (OpenAPI, Swagger, Postman collections)

Usage in config.json:
{
  "schemas": {
    "api": "schemas/api_schema.json",           // File path
    "api": "https://api.example.com/swagger",   // URL
    "api": {                                     // Advanced config
      "source": "https://api.example.com/swagger.json",
      "type": "openapi",
      "auto_refresh": true,
      "refresh_interval": 3600
    }
  }
}
"""

import json
import requests
from pathlib import Path
from typing import Dict, Any, Optional, Union
from urllib.parse import urlparse

from . import constants

from .schema_generator import OpenAPIConverter
from .utils import setup_logger


class SchemaLoader:
    """
    Intelligent schema loader that supports multiple sources and formats.
    
    Auto-detects schema type and converts if needed.
    """
    
    def __init__(self):
        """Initialize schema loader."""
        self.logger = setup_logger('enable_ai.schema_loader')
        self.converter = OpenAPIConverter()
        self.cache = {}
    
    def load_schema(self, schema_config: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Load schema from various sources.
        
        Args:
            schema_config: Can be:
                - String: File path or URL
                - Dict: Advanced config with source, type, etc.
        
        Returns:
            Loaded schema in Enable AI format
        
        Examples:
            # Simple file path
            schema = loader.load_schema("schemas/api_schema.json")
            
            # URL
            schema = loader.load_schema("https://api.example.com/swagger.json")
            
            # Advanced config
            schema = loader.load_schema({
                "source": "https://api.example.com/swagger.json",
                "type": "openapi",
                "auto_refresh": true
            })
        """
        # Simple string config (file or URL)
        if isinstance(schema_config, str):
            return self._load_from_source(schema_config)
        
        # Advanced dict config
        elif isinstance(schema_config, dict):
            source = schema_config.get('source')
            if not source:
                raise ValueError("Schema config missing 'source' field")
            
            schema_type = schema_config.get('type', 'auto')
            auto_refresh = schema_config.get('auto_refresh', False)
            
            # Load schema
            schema = self._load_from_source(source, schema_type)
            
            # TODO: Implement auto-refresh if enabled
            if auto_refresh:
                self.logger.info(f"Auto-refresh enabled for {source}")
            
            return schema
        
        else:
            raise ValueError(f"Invalid schema config type: {type(schema_config)}")
    
    def _load_from_source(self, source: str, schema_type: str = 'auto') -> Dict[str, Any]:
        """
        Load schema from file or URL.
        
        Args:
            source: File path or URL
            schema_type: 'auto', 'enable_ai', 'openapi', 'swagger', 'postman'
        
        Returns:
            Schema in Enable AI format
        """
        # Check cache
        if source in self.cache:
            self.logger.debug(f"Using cached schema for {source}")
            return self.cache[source]
        
        # Determine if source is URL or file
        if self._is_url(source):
            self.logger.info(f"Fetching schema from URL: {source}")
            schema = self._load_from_url(source)
        else:
            self.logger.info(f"Loading schema from file: {source}")
            schema = self._load_from_file(source)
        
        # Auto-detect and convert if needed
        if schema_type == 'auto':
            schema_type = self._detect_schema_type(schema)
            self.logger.info(f"Detected schema type: {schema_type}")
        
        # Convert to Enable AI format if needed
        if schema_type in ['openapi', 'swagger']:
            self.logger.info("Converting OpenAPI/Swagger to Enable AI schema format")
            schema = self._convert_openapi_schema(schema)
        elif schema_type == 'postman':
            self.logger.warning("Postman collection conversion not yet implemented")
            # TODO: Implement Postman conversion
        elif schema_type != 'enable_ai':
            self.logger.warning(f"Unknown schema type: {schema_type}, assuming Enable AI format")
        
        # Validate schema
        if not self._validate_schema(schema):
            raise ValueError(f"Invalid schema from {source}")
        
        # Cache and return
        self.cache[source] = schema
        return schema
    
    def _is_url(self, source: str) -> bool:
        """Check if source is a URL."""
        try:
            result = urlparse(source)
            return all([result.scheme, result.netloc])
        except:
            return False
    
    def _load_from_url(self, url: str) -> Dict[str, Any]:
        """
        Fetch schema from URL.
        
        Supports:
        - https://api.example.com/swagger.json
        - https://api.example.com/openapi.yaml (TODO)
        - https://www.getpostman.com/collections/xxx (TODO)
        """
        try:
            response = requests.get(url, timeout=constants.REQUEST_TIMEOUT)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            self.logger.error(f"Failed to fetch schema from {url}: {e}")
            raise
    
    def _load_from_file(self, file_path: str) -> Dict[str, Any]:
        """Load schema from local file."""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Schema file not found: {file_path}")
        
        with open(path, 'r') as f:
            return json.load(f)
    
    def _detect_schema_type(self, schema: Dict[str, Any]) -> str:
        """
        Auto-detect schema type.
        
        Returns: 'enable_ai', 'openapi', 'swagger', 'postman', or 'unknown'
        """
        # Enable AI schema
        if schema.get('type') == 'api' and 'resources' in schema:
            return 'enable_ai'
        
        # OpenAPI 3.x
        if 'openapi' in schema and schema['openapi'].startswith('3'):
            return 'openapi'
        
        # Swagger 2.0
        if 'swagger' in schema and schema['swagger'] == '2.0':
            return 'swagger'
        
        # Postman collection
        if 'info' in schema and 'item' in schema:
            # Postman has 'info' and 'item' at root
            return 'postman'
        
        return 'unknown'
    
    def _convert_openapi_schema(self, openapi_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Convert OpenAPI/Swagger to Enable AI format."""
        # Use the existing converter (would need to refactor it to accept dict instead of file)
        # For now, save to temp file and convert
        # TODO: Refactor OpenAPIConverter to accept dict directly
        
        # Quick conversion for now
        enable_ai_schema = {
            'type': 'api',
            'version': '1.0.0',
            'base_url': self._extract_base_url(openapi_schema),
            'resources': {}
        }
        
        # TODO: Full conversion logic
        # For now, return minimal schema
        self.logger.warning("Using minimal OpenAPI conversion - full converter integration needed")
        
        return enable_ai_schema
    
    def _extract_base_url(self, openapi_schema: Dict[str, Any]) -> str:
        """Extract base URL from OpenAPI schema."""
        # OpenAPI 3.x
        if 'servers' in openapi_schema and openapi_schema['servers']:
            return openapi_schema['servers'][0].get('url', '')
        
        # Swagger 2.0
        if 'host' in openapi_schema:
            scheme = openapi_schema.get('schemes', ['https'])[0]
            host = openapi_schema['host']
            base_path = openapi_schema.get('basePath', '')
            return f"{scheme}://{host}{base_path}"
        
        return ''
    
    def _validate_schema(self, schema: Dict[str, Any]) -> bool:
        """Validate that schema is in Enable AI format."""
        if not isinstance(schema, dict):
            return False
        
        # Must have type = 'api'
        if schema.get('type') != 'api':
            return False
        
        # Must have resources or base_url
        if 'resources' not in schema and 'base_url' not in schema:
            return False
        
        return True
    
    def clear_cache(self):
        """Clear the schema cache."""
        self.cache.clear()
        self.logger.info("Schema cache cleared")


# Singleton instance
_schema_loader = None


def get_schema_loader() -> SchemaLoader:
    """Get the singleton schema loader instance."""
    global _schema_loader
    if _schema_loader is None:
        _schema_loader = SchemaLoader()
    return _schema_loader


def load_schema(source: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Convenience function to load a schema.
    
    Args:
        source: File path, URL, or config dict
    
    Returns:
        Schema in Enable AI format
    
    Examples:
        # From file
        schema = load_schema("schemas/api_schema.json")
        
        # From URL
        schema = load_schema("https://api.example.com/swagger.json")
        
        # With config
        schema = load_schema({
            "source": "https://api.example.com/swagger.json",
            "type": "openapi",
            "auto_refresh": true
        })
    """
    loader = get_schema_loader()
    return loader.load_schema(source)
