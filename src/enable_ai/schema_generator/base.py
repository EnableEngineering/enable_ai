"""
Base classes for schema generators
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import json
from pathlib import Path
from datetime import datetime


class SchemaGenerator(ABC):
    """Base class for all schema generators."""
    
    # Default schemas directory (relative to project root)
    DEFAULT_SCHEMAS_DIR = "schemas"
    
    @abstractmethod
    def generate(self, source: Any, **kwargs) -> Dict[str, Any]:
        """
        Generate schema from source.
        
        Args:
            source: Input source (file path, connection string, etc.)
            **kwargs: Additional generation options
            
        Returns:
            Generated schema dict
        """
        pass
    
    @abstractmethod
    def get_schema_type(self) -> str:
        """
        Get the type of schema this generator creates.
        
        Returns:
            Schema type ('api')
        """
        pass
    
    def validate_schema(self, schema: Dict[str, Any]) -> bool:
        """
        Validate generated schema structure.
        
        Args:
            schema: Schema to validate
            
        Returns:
            True if valid, raises ValueError if invalid
        """
        if 'type' not in schema:
            raise ValueError("Schema missing 'type' field")
        
        if schema['type'] != self.get_schema_type():
            raise ValueError(
                f"Schema type mismatch. Expected '{self.get_schema_type()}', "
                f"got '{schema['type']}'"
            )
        
        return True
    
    def get_default_output_path(self, source_name: Optional[str] = None) -> str:
        """
        Get default output path for schema in schemas/ directory.
        
        Args:
            source_name: Optional source name to include in filename
            
        Returns:
            Default output path
        """
        schema_type = self.get_schema_type()
        
        # Build filename based on schema type
        if source_name:
            # e.g., api_schema_enableerp.json
            filename = f"{schema_type}_schema_{source_name}.json"
        else:
            # e.g., api_schema.json, database_schema.json
            filename = f"{schema_type}_schema.json"
        
        return str(Path(self.DEFAULT_SCHEMAS_DIR) / filename)
    
    def save_schema(self, schema: Dict[str, Any], output_path: Optional[str] = None) -> str:
        """
        Save schema to JSON file.
        
        Args:
            schema: Schema to save
            output_path: Output file path (optional, defaults to schemas/ directory)
            
        Returns:
            Path where schema was saved
        """
        self.validate_schema(schema)
        
        # Use default path if none provided
        if output_path is None:
            output_path = self.get_default_output_path()
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(schema, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Schema saved to {output_path}")
        return output_path
    
    def load_json_file(self, file_path: str) -> Dict[str, Any]:
        """
        Load JSON file.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Parsed JSON data
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
