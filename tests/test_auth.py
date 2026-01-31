#!/usr/bin/env python3
"""Test authentication flow"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env
env_path = Path('.env')
load_dotenv(env_path)

print('='*80)
print('AUTHENTICATION TEST')
print('='*80)

print('\n1. Environment variables:')
print(f'   API_EMAIL: {os.getenv("API_EMAIL")}')
print(f'   API_PASSWORD: {"***" if os.getenv("API_PASSWORD") else "NOT SET"}')

# Test processor
from enable_ai import APIOrchestrator
processor = APIOrchestrator()

# Check config
print('\n2. JWT Config from config.json:')
security = processor.config.get('security_credentials', {}).get('api', {})
jwt_config = security.get('jwt', {})
print(f'   Enabled: {jwt_config.get("enabled")}')
print(f'   Token endpoint: {jwt_config.get("token_endpoint")}')

# Test authentication
print('\n3. API Configuration:')
base_url = processor.config.get('data_sources', {}).get('api', {}).get('base_url')
print(f'   Base URL: {base_url}')
print(f'   Full token URL: {base_url.rstrip("/") + jwt_config.get("token_endpoint", "/token/")}')

# Detect auth type
print('\n4. Detecting authentication type:')
api_schema = processor.schemas.get('api', {})
auth_type = processor._detect_auth_type(api_schema)
print(f'   Detected auth type: {auth_type}')

# Try authentication
print('\n5. Attempting authentication...')
auth_result = processor._authenticate(base_url, 'jwt')
if 'error' in auth_result:
    print(f'   ❌ Authentication failed: {auth_result["error"]}')
else:
    print(f'   ✅ Authentication successful')
    token = auth_result.get('token', '')
    print(f'   Token (first 50 chars): {token[:50]}...')

print('\n' + '='*80)
