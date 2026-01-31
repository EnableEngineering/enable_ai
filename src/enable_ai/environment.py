"""
Environment Configuration for enable_ai

This file defines paths to config.json and .env files for different environments.
Switch between DEV and PROD by commenting/uncommenting the ACTIVE_ENVIRONMENT line.

Directory Structure:
    enable_ai/              # Root directory
    ├── src/                     # Library source code (pip module)
    │   └── enable_ai/      # Main package
    ├── examples/                # Example configurations and schemas
    │   └── backend_simulator/   # Development testing environment (simulates production)
    │   ├── config.json          # API configuration
    │   ├── .env                 # Environment variables (credentials, API keys)
    │   └── schemas/             # API/Database/KG schemas
    ├── tests/                   # Unit and integration tests
    └── environment.py           # This file (environment configuration)

Usage:
    1. Set paths for DEV and PROD environments below
    2. Comment/uncomment ACTIVE_ENVIRONMENT to switch
    3. All modules will automatically use the active environment
    
Note:
    DEV environment uses examples/backend_simulator/ for testing
    PROD environment should point to actual production backend paths
"""

from pathlib import Path

# ============================================================================
# ENVIRONMENT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent
DEV_ENV_DIR = PROJECT_ROOT / "examples" / "backend_simulator"

DEV_CONFIG = {
    'config_path': str(DEV_ENV_DIR / "config.json"),
    'env_path': str(DEV_ENV_DIR / ".env"),
    'schemas_dir': str(DEV_ENV_DIR / "schemas"),
    'description': 'Development environment (simulates production backend)'
}

PROD_CONFIG = {
    'config_path': '/path/to/production/backend/config.json',
    'env_path': '/path/to/production/backend/.env',
    'schemas_dir': '/path/to/production/backend/schemas',
    'description': 'Production environment (actual SaaS backend)'
}

# ============================================================================
# ACTIVE ENVIRONMENT (comment one, uncomment the other)
# ============================================================================

ACTIVE_ENVIRONMENT = 'DEV'  # Development
# ACTIVE_ENVIRONMENT = 'PROD'  # Production

# ============================================================================
# DO NOT MODIFY BELOW THIS LINE
# ============================================================================

def get_active_config():
    """
    Get configuration for the active environment.
    
    Returns:
        dict: Active configuration with 'config_path', 'env_path', 'schemas_dir', 'description'
    """
    if ACTIVE_ENVIRONMENT == 'DEV':
        return DEV_CONFIG
    elif ACTIVE_ENVIRONMENT == 'PROD':
        return PROD_CONFIG
    else:
        raise ValueError(f"Invalid ACTIVE_ENVIRONMENT: {ACTIVE_ENVIRONMENT}. Must be 'DEV' or 'PROD'")


def get_config_path():
    """Get path to config.json for active environment."""
    return get_active_config()['config_path']


def get_env_path():
    """Get path to .env file for active environment."""
    return get_active_config()['env_path']


def get_schemas_dir():
    """Get path to schemas directory for active environment."""
    return get_active_config()['schemas_dir']


def get_environment_info():
    """Get information about the active environment."""
    config = get_active_config()
    return {
        'environment': ACTIVE_ENVIRONMENT,
        'config_path': config['config_path'],
        'env_path': config['env_path'],
        'schemas_dir': config['schemas_dir'],
        'description': config['description']
    }


if __name__ == '__main__':
    # Print active environment info
    info = get_environment_info()
    print(f"\n{'='*80}")
    print(f"ACTIVE ENVIRONMENT: {info['environment']}")
    print(f"{'='*80}")
    print(f"Description: {info['description']}")
    print(f"Config Path: {info['config_path']}")
    print(f"Env Path:    {info['env_path']}")
    print(f"Schemas Dir: {info['schemas_dir']}")
    print(f"{'='*80}\n")
