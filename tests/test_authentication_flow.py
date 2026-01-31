"""
Test script to demonstrate the authentication flow.

This shows how the processor:
1. Detects authentication type from config
2. Calls the authentication endpoint
3. Gets the token
4. Uses the token for the actual API call
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from enable_ai import APIOrchestrator


def test_auth_flow_with_credentials():
    """Test authentication flow with provided credentials."""
    print("\n" + "="*80)
    print("AUTHENTICATION FLOW TEST")
    print("="*80)
    
    # Initialize processor
    config_path = Path(__file__).parent.parent / 'config.json'
    processor = APIOrchestrator(config_path=str(config_path))
    
    print("\n1. Processor initialized")
    print(f"   Config path: {config_path}")
    
    # Test query
    query = "list all users"
    print(f"\n2. Query: {query}")
    
    # Get active schema
    schema = processor.schemas.get('api')
    if not schema:
        print("❌ No API schema loaded")
        return False
    
    print(f"   Schema loaded: {schema.get('type')}")
    
    # Detect authentication type
    auth_type = processor._detect_auth_type(schema)
    print(f"\n3. Authentication type detected: {auth_type or 'None'}")
    
    if auth_type:
        # Show JWT config
        jwt_config = processor.config.get('security_credentials', {}).get('api', {}).get('jwt', {})
        print(f"   JWT enabled: {jwt_config.get('enabled')}")
        print(f"   Token endpoint: {jwt_config.get('token_endpoint')}")
        
        # Attempt authentication (without actual credentials, will fail gracefully)
        print("\n4. Authentication attempt:")
        print("   Note: Set API_EMAIL and API_PASSWORD environment variables")
        print("         OR configure test_email/test_password in config.json")
    
    # Process query (without token - will try to authenticate)
    print("\n5. Processing query...")
    result = processor.process(query)
    
    # Show result
    print("\n" + "="*80)
    print("RESULT")
    print("="*80)
    print(f"Success: {result.get('success')}")
    print(f"Summary: {result.get('summary')}")
    
    if result.get('error'):
        print(f"\nError details: {result.get('error')}")
        print("\nTo fix:")
        print("  1. Set environment variables:")
        print("     export API_EMAIL='your_email@example.com'")
        print("     export API_PASSWORD='your_password'")
        print("\n  2. OR add to config.json under security_credentials.api.jwt:")
        print("     \"test_email\": \"admin@example.com\",")
        print("     \"test_password\": \"admin123\"")
    
    return result.get('success', False)


def test_auth_flow_with_token():
    """Test authentication flow with pre-provided token."""
    print("\n" + "="*80)
    print("AUTHENTICATION FLOW TEST (WITH TOKEN)")
    print("="*80)
    
    config_path = Path(__file__).parent.parent / 'config.json'
    processor = APIOrchestrator(config_path=str(config_path))
    
    query = "list all users"
    token = "test_jwt_token_12345"
    
    print(f"\nQuery: {query}")
    print(f"Token provided: {token[:20]}...")
    print("\nWhen a token is provided, authentication endpoint is skipped.")
    
    # Process with token
    result = processor.process(query, access_token=token)
    
    print("\n" + "="*80)
    print("RESULT")
    print("="*80)
    print(f"Success: {result.get('success')}")
    print(f"Summary: {result.get('summary')}")
    
    return result.get('success', False)


if __name__ == '__main__':
    # Test 1: Without token (will authenticate)
    print("\n" + "╔"+"═"*78+"╗")
    print("║" + " TEST 1: Auto-authentication (no token provided)".center(78) + "║")
    print("╚"+"═"*78+"╝")
    test_auth_flow_with_credentials()
    
    # Test 2: With token (skip authentication)
    print("\n\n" + "╔"+"═"*78+"╗")
    print("║" + " TEST 2: Token provided (skip authentication)".center(78) + "║")
    print("╚"+"═"*78+"╝")
    test_auth_flow_with_token()
