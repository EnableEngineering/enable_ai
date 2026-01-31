"""
Configuration Loader for Enable AI

Loads and validates configuration from config.json, handling
data source connections, entity recognition rules, and system settings.

Priority for finding config.json:
1. environment.py (if exists) - DEV/PROD configuration
2. Provided config_path parameter
3. Current working directory
4. Script directory
5. Parent directories
"""

import json
import os
from typing import Dict, Any, Optional
from pathlib import Path
import sys


class ConfigLoader:
    """Loads and manages configuration settings."""
    
    _instance = None
    _config = None
    
    def __new__(cls):
        """Singleton pattern to ensure one config instance."""
        if cls._instance is None:
            cls._instance = super(ConfigLoader, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize config loader."""
        if self._config is None:
            self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from config.json."""
        config_path = self._find_config_file()
        
        if not config_path.exists():
            print(f"Warning: config.json not found at {config_path}")
            return self._get_default_config()
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            return config
        except json.JSONDecodeError as e:
            print(f"Error parsing config.json: {e}")
            return self._get_default_config()
        except Exception as e:
            print(f"Error loading config: {e}")
            return self._get_default_config()
    
    def _find_config_file(self) -> Path:
        """
        Find config.json with priority order:
        1. environment.py configuration (if exists)
        2. Current working directory
        3. Script directory
        4. Parent directories
        """
        # Priority 1: Check environment.py for DEV/PROD configuration
        try:
            # Import from package
            from . import environment
            config_path = Path(environment.get_config_path())
            
            if config_path.exists():
                print(f"✓ Using {environment.ACTIVE_ENVIRONMENT} config from environment.py", file=sys.stderr)
                return config_path
            else:
                print(f"⚠️  environment.py configured but config not found: {config_path}", file=sys.stderr)
        except (ImportError, AttributeError, FileNotFoundError):
            # environment.py doesn't exist or is invalid, fall back to auto-detection
            pass
        
        # Priority 2: Try current directory
        current = Path.cwd() / "config.json"
        if current.exists():
            return current
        
        # Priority 3: Try script directory
        script_dir = Path(__file__).parent.parent.parent / "config.json"
        if script_dir.exists():
            return script_dir
        
        # Priority 4: Try parent directories
        for parent in Path.cwd().parents:
            config_file = parent / "config.json"
            if config_file.exists():
                return config_file
        
        # Default to current directory
        return Path.cwd() / "config.json"
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return minimal default configuration."""
        return {
            "data_sources": {
                "primary_api": {
                    "type": "rest_api",
                    "base_url": "http://localhost:8000",
                    "authentication": {
                        "method": "jwt",
                        "header_format": "Bearer {token}"
                    },
                    "endpoints_spec": "api_spec.json"
                }
            },
            "entity_recognition": {
                "custom_entities": {}
            },
            "query_understanding": {
                "intent_classifier": {
                    "type": "keyword_based",
                    "confidence_threshold": 0.6
                },
                "synonym_expansion": {
                    "enabled": False,
                    "synonyms": {}
                }
            },
            "logging": {
                "level": "INFO",
                "log_queries": False
            },
            "features": {}
        }
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-separated key path.
        
        Args:
            key_path: Dot-separated path (e.g., "data_sources.primary_api.base_url")
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self._config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def get_data_source_config(self, source_name: str = "primary_api") -> Dict[str, Any]:
        """Get data source configuration."""
        return self.get(f"data_sources.{source_name}", {})
    
    def get_entity_patterns(self, entity_name: str) -> list:
        """Get regex patterns for entity recognition."""
        return self.get(f"entity_recognition.custom_entities.{entity_name}.patterns", [])
    
    def get_synonyms(self) -> Dict[str, list]:
        """Get synonym mappings for query understanding."""
        return self.get("query_understanding.synonym_expansion.synonyms", {})
    
    def is_feature_enabled(self, feature_path: str) -> bool:
        """Check if a feature is enabled."""
        return self.get(feature_path, False)
    
    def reload(self):
        """Reload configuration from file."""
        self._config = self._load_config()


# Global instance
_config_loader = ConfigLoader()


def get_config(key_path: str = None, default: Any = None) -> Any:
    """
    Get configuration value.
    
    Args:
        key_path: Dot-separated path to config value
        default: Default value if not found
        
    Returns:
        Configuration value or entire config if key_path is None
    """
    if key_path is None:
        return _config_loader._config
    return _config_loader.get(key_path, default)


def reload_config():
    """Reload configuration from file."""
    _config_loader.reload()
