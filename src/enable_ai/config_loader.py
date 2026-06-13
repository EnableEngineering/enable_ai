"""
Configuration Loader for Enable AI

Loads and validates configuration from config.json, handling
data source connections, entity recognition rules, and system settings.

Priority for finding config.json:
1. environment.py (if exists) - DEV/PROD configuration
2. Current working directory
3. Script directory
4. Parent directories
"""

import json
from typing import Dict, Any, Optional
from pathlib import Path
import sys

from . import constants

# Module-level config state
_config: Optional[Dict[str, Any]] = None


def _find_config_file() -> Path:
    """
    Find config.json with priority order:
    1. environment.py configuration (if exists)
    2. Current working directory
    3. Script directory
    4. Parent directories
    """
    try:
        from . import environment
        config_path = Path(environment.get_config_path())

        if config_path.exists():
            print(f"✓ Using {environment.ACTIVE_ENVIRONMENT} config from environment.py", file=sys.stderr)
            return config_path
        else:
            print(f"⚠️  environment.py configured but config not found: {config_path}", file=sys.stderr)
    except (ImportError, AttributeError, FileNotFoundError):
        pass

    current = Path.cwd() / "config.json"
    if current.exists():
        return current

    script_dir = Path(__file__).parent.parent.parent / "config.json"
    if script_dir.exists():
        return script_dir

    for parent in Path.cwd().parents:
        config_file = parent / "config.json"
        if config_file.exists():
            return config_file

    return Path.cwd() / "config.json"


def _get_default_config() -> Dict[str, Any]:
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
            "default_page_size": constants.DEFAULT_PAGE_SIZE,
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


def _load_config() -> Dict[str, Any]:
    """Load configuration from config.json."""
    config_path = _find_config_file()

    if not config_path.exists():
        print(f"Warning: config.json not found at {config_path}")
        return _get_default_config()

    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error parsing config.json: {e}")
        return _get_default_config()
    except Exception as e:
        print(f"Error loading config: {e}")
        return _get_default_config()


def _ensure_loaded() -> Dict[str, Any]:
    """Ensure config is loaded, return it."""
    global _config
    if _config is None:
        _config = _load_config()
    return _config


def get_config(key_path: str = None, default: Any = None) -> Any:
    """
    Get configuration value.

    Args:
        key_path: Dot-separated path to config value
        default: Default value if not found

    Returns:
        Configuration value or entire config if key_path is None
    """
    config = _ensure_loaded()
    if key_path is None:
        return config

    keys = key_path.split('.')
    value = config

    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default

    return value


def get_data_source_config(source_name: str = "primary_api") -> Dict[str, Any]:
    """Get data source configuration."""
    return get_config(f"data_sources.{source_name}", {})


def get_entity_patterns(entity_name: str) -> list:
    """Get regex patterns for entity recognition."""
    return get_config(f"entity_recognition.custom_entities.{entity_name}.patterns", [])


def get_synonyms() -> Dict[str, list]:
    """Get synonym mappings for query understanding."""
    return get_config("query_understanding.synonym_expansion.synonyms", {})


def is_feature_enabled(feature_path: str) -> bool:
    """Check if a feature is enabled."""
    return get_config(feature_path, False)


def get_default_page_size() -> int:
    """
    Get default page_size for list queries when the user does not specify a limit.

    Reads query_understanding.default_page_size from config.json, falling back to
    constants.DEFAULT_PAGE_SIZE (overridable via ENABLE_AI_DEFAULT_PAGE_SIZE).
    """
    raw = get_config("query_understanding.default_page_size", constants.DEFAULT_PAGE_SIZE)
    try:
        size = int(raw)
    except (TypeError, ValueError):
        return constants.DEFAULT_PAGE_SIZE
    return max(1, min(size, constants.PAGE_SIZE_CAP))


def reload_config():
    """Reload configuration from file."""
    global _config
    _config = _load_config()
