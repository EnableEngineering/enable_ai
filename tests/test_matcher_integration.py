"""
Integration test for API Matcher with full NLP pipeline.

Flow:
1. Initialize processor with config.json
2. Accept user query via command line
3. Check and generate schemas if needed
4. Parse client query using LLM
5. Match parsed query to API endpoints
6. Display matched API details
"""

import sys
import unittest
import json
import subprocess
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent))  # For environment.py

# Try to from enable_ai import environment module for config/env/schemas path detection
try:
    from enable_ai import environment
    HAS_ENVIRONMENT_MODULE = True
except ImportError:
    HAS_ENVIRONMENT_MODULE = False

from enable_ai import APIOrchestrator


class TestMatcherIntegration(unittest.TestCase):
    """Test full NLP pipeline: processor → schema → parser → matcher."""
    
    # Class variable to store query (set only once)
    user_query = None
    
    @classmethod
    def setUpClass(cls):
        """Set up class fixtures - runs once for all tests."""
        # Get user query from command line or prompt user
        cls.user_query = cls._get_user_query()
    
    @classmethod
    def _get_user_query(cls):
        """Get user query from command line arguments or prompt user."""
        if len(sys.argv) > 1 and not sys.argv[1].startswith('test_'):
            # Query provided as command line argument
            query = ' '.join(sys.argv[1:])
            print(f"\n📝 Using provided query: '{query}'")
            return query
        else:
            # Prompt user for query
            print("\n" + "="*80)
            print("API MATCHER INTEGRATION TEST")
            print("="*80)
            print("\nThis test will:")
            print("  1. Initialize the processor with config.json")
            print("  2. Generate schemas if needed")
            print("  3. Parse your query using GPT-4o LLM")
            print("  4. Match the parsed query to API endpoints")
            print("  5. Display matched API endpoint details")
            print("\nExamples:")
            print("  - get user with id 5")
            print("  - show me service orders for company ABC")
            print("  - find equipment with quantity less than 10")
            print("  - list all urgent service orders")
            print("="*80)
            
            query = input("\n🤖 Enter your query: ").strip()
            
            if not query:
                print("📝 No query entered, using default: 'get user with id 5'")
                return "get user with id 5"
            
            return query
    
    def setUp(self):
        """Set up test fixtures for each test."""
        # Get paths from environment.py if available
        if HAS_ENVIRONMENT_MODULE:
            self.config_path = Path(environment.get_config_path())
            self.schemas_dir = Path(environment.get_schemas_dir())
        else:
            self.config_path = Path(__file__).parent.parent / 'config.json'
            self.schemas_dir = Path(__file__).parent.parent / 'schemas'
        
        self.api_schema_path = self.schemas_dir / 'api_schema.json'
        
        # Verify config.json exists
        self.assertTrue(
            self.config_path.exists(),
            f"config.json not found at {self.config_path}"
        )
    
    def test_01_initialize_processor_with_config(self):
        """Step 1: Initialize processor with config.json."""
        print(f"\n{'='*80}")
        print(f"STEP 1: PROCESSOR INITIALIZATION")
        print(f"{'='*80}")
        
        # Initialize processor with config.json
        self.processor = APIOrchestrator(config_path=str(self.config_path))
        
        # Verify initialization
        self.assertIsNotNone(self.processor, "Processor should initialize")
        
        # Display initialization details
        print(f"\n✓ Processor initialized with config.json")
        print(f"  Config path: {self.config_path}")
        print(f"  User query: '{self.user_query}'")
    
    def test_02_check_and_generate_schemas(self):
        """Step 2: Check schemas and generate if missing."""
        print(f"\n{'='*80}")
        print(f"STEP 2: SCHEMA VALIDATION & GENERATION")
        print(f"{'='*80}")
        
        # Initialize processor
        self.processor = APIOrchestrator(config_path=str(self.config_path))
        
        # Load config to check enabled data sources
        with open(self.config_path, 'r') as f:
            config = json.load(f)
        
        # Get enabled data sources
        enabled_sources = [
            source for source, is_enabled 
            in config.get('data_sources', {}).items() 
            if is_enabled
        ]
        
        print(f"\nEnabled data sources: {', '.join(enabled_sources)}")
        
        # Map data sources to schema types
        schema_mapping = {
            'api': 'api',
            'database': 'database',
            'knowledge_graph': 'knowledge_graph'
        }
        
        required_schemas = [
            schema_mapping[source] 
            for source in enabled_sources 
            if source in schema_mapping
        ]
        
        print(f"Required schema types: {', '.join(required_schemas)}")
        
        # Check if schemas exist
        missing_schemas = []
        for schema_type in required_schemas:
            schema_file = self.schemas_dir / f"{schema_type}_schema.json"
            if not schema_file.exists():
                missing_schemas.append(schema_type)
        
        # Generate missing schemas
        if missing_schemas:
            print(f"\n⚠️  Missing schemas: {', '.join(missing_schemas)}")
            print(f"Generating schemas...")
            
            # Run schema generation script
            generate_script = Path(__file__).parent.parent / 'generate_schema.py'
            if generate_script.exists():
                result = subprocess.run(
                    ['python', str(generate_script)],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode == 0:
                    print(f"✓ Schemas generated successfully")
                    
                    # Reload processor with new schemas
                    self.processor = APIOrchestrator(config_path=str(self.config_path))
                else:
                    print(f"✗ Schema generation failed: {result.stderr}")
        else:
            print(f"\n✓ All required schemas exist")
        
        # Verify loaded schemas
        loaded_schemas = list(self.processor.schemas.keys())
        print(f"\nLoaded schemas: {', '.join(loaded_schemas)}")
        
        self.assertTrue(
            len(loaded_schemas) > 0,
            "At least one schema should be loaded"
        )
    
    def test_03_parse_query_with_llm(self):
        """Step 3: Parse client query using LLM."""
        print(f"\n{'='*80}")
        print(f"STEP 3: LLM QUERY PARSING")
        print(f"{'='*80}")
        
        # Initialize processor
        self.processor = APIOrchestrator(config_path=str(self.config_path))
        
        print(f"\nQuery to parse: '{self.user_query}'")
        
        # Get active schema
        schema_type = list(self.processor.schemas.keys())[0]
        schema = self.processor.schemas[schema_type]
        
        print(f"Schema type: {schema_type}")
        
        if schema_type == 'api':
            resources = list(schema.get('resources', {}).keys())
            print(f"Schema resources: {len(resources)} available")
        
        # Parse query using LLM
        print(f"\n🤖 Parsing query with LLM (GPT-4o)...")
        
        parsed_result = self.processor._understand_query(
            self.user_query,
            schema=schema
        )
        
        # Store for next test
        self.__class__.parsed_result = parsed_result
        
        # Verify parsing
        self.assertIsNotNone(parsed_result, "Parsed result should not be None")
        self.assertIn('intent', parsed_result, "Parsed result should have 'intent'")
        self.assertIn('resource', parsed_result, "Parsed result should have 'resource'")
        
        # Display parser output
        print(f"\n{'='*80}")
        print(f"PARSER OUTPUT")
        print(f"{'='*80}")
        print(json.dumps(parsed_result, indent=2))
        
        # Display extracted information
        print(f"\n{'='*80}")
        print(f"EXTRACTED INFORMATION")
        print(f"{'='*80}")
        print(f"  Intent: {parsed_result.get('intent')}")
        print(f"  Resource: {parsed_result.get('resource')}")
        print(f"  Schema Type: {parsed_result.get('schema_type')}")
        
        if parsed_result.get('filters'):
            print(f"\n  Filters:")
            for field, filter_obj in parsed_result['filters'].items():
                operator = filter_obj.get('operator', 'equals')
                value = filter_obj.get('value')
                print(f"    - {field}: {operator} {value}")
        
        if parsed_result.get('entities'):
            print(f"\n  Entities: {parsed_result['entities']}")
        
        if parsed_result.get('relationships'):
            print(f"\n  Relationships: {len(parsed_result['relationships'])} detected")
            for rel in parsed_result['relationships']:
                print(f"    - {rel.get('type')}: {rel.get('target_entity')}")
        
        print(f"\n✓ Query parsed successfully using LLM")
    
    def test_04_match_api_endpoint(self):
        """Step 4: Match parsed query to API endpoint."""
        print(f"\n{'='*80}")
        print(f"STEP 4: API ENDPOINT MATCHING")
        print(f"{'='*80}")
        
        # Initialize processor
        self.processor = APIOrchestrator(config_path=str(self.config_path))
        
        # Get parsed result from previous test
        parsed_result = getattr(self.__class__, 'parsed_result', None)
        
        if not parsed_result:
            # Parse again if not available
            print(f"\n🤖 Parsing query: '{self.user_query}'")
            schema = self.processor.schemas.get('api')
            parsed_result = self.processor._understand_query(
                self.user_query,
                schema=schema
            )
        
        print(f"\nMatching query to API endpoint...")
        print(f"  Resource: {parsed_result.get('resource')}")
        print(f"  Intent: {parsed_result.get('intent')}")
        
        # Load API schema (new resources format)
        with open(self.api_schema_path, 'r') as f:
            api_schema = json.load(f)
        
        # Find matching endpoint
        resource = parsed_result.get('resource')
        intent = parsed_result.get('intent')
        
        # Normalize resource name (handle hyphen vs underscore)
        resource_normalized = resource.replace('-', '_') if resource else None
        
        # Map intent to HTTP methods
        intent_to_method = {
            'read': ['GET'],
            'create': ['POST'],
            'update': ['PUT', 'PATCH'],
            'delete': ['DELETE']
        }
        
        expected_methods = intent_to_method.get(intent, ['GET'])
        
        # Search for matching endpoint in resources
        matched_endpoint = None
        matched_method = None
        matched_resource = None
        endpoint_key = None
        
        # Check if resource exists in resources
        resources = api_schema.get('resources', {})
        
        # Try both original and normalized resource names
        resource_to_check = resource_normalized if resource_normalized in resources else resource
        
        if resource_to_check in resources:
            resource_data = resources[resource_to_check]
            endpoints = resource_data.get('endpoints', [])
            
            # Search endpoints in this resource
            for ep_data in endpoints:
                method = ep_data.get('method')
                
                # Check if method matches intent
                if method in expected_methods:
                    matched_endpoint = ep_data.get('path')
                    matched_method = method
                    matched_resource = resource_to_check
                    endpoint_key = ep_data.get('name', 'unnamed')
                    break
        
        # Display match results
        print(f"\n{'='*80}")
        print(f"MATCH RESULTS")
        print(f"{'='*80}")
        
        if matched_endpoint:
            print(f"\n✓ API Endpoint Matched!")
            print(f"\n  Resource: {matched_resource}")
            print(f"  Endpoint: {matched_method} {matched_endpoint}")
            
            # Get endpoint details
            resource_data = api_schema['resources'][matched_resource]
            # Find the endpoint data in the list
            endpoint_details = None
            for ep in resource_data['endpoints']:
                if ep.get('path') == matched_endpoint and ep.get('method') == matched_method:
                    endpoint_details = ep
                    break
            
            print(f"  Description: {endpoint_details.get('description', 'N/A')}")
            
            # ========================================================================
            # VALIDATE REQUIRED PARAMETERS
            # ========================================================================
            missing_params = []
            path_params = endpoint_details.get('path_parameters', {})
            
            if path_params:
                for param_name in path_params.keys():
                    # Check if parameter exists in entities
                    if param_name not in parsed_result.get('entities', {}):
                        missing_params.append(param_name)
            
            # If missing required parameters, return informative response
            if missing_params:
                print(f"\n⚠️  INCOMPLETE QUERY - Missing Required Information")
                print(f"\n{'='*80}")
                print(f"MISSING PARAMETERS")
                print(f"{'='*80}")
                
                print(f"\nThe following required parameters are missing:")
                for param in missing_params:
                    param_desc = path_params.get(param, 'N/A')
                    print(f"  ❌ {param}: {param_desc}")
                
                print(f"\nCurrent query: '{parsed_result.get('original_input')}'")
                print(f"\nTo complete this request, please provide:")
                
                # Generate helpful suggestions
                suggestions = []
                for param in missing_params:
                    if param == 'id':
                        suggestions.append(f"  • The specific {resource} ID (e.g., 'service order 123')")
                    else:
                        suggestions.append(f"  • The {param} value")
                
                for suggestion in suggestions:
                    print(suggestion)
                
                # Show example queries
                print(f"\nExample queries:")
                if 'id' in missing_params:
                    original_query = parsed_result.get('original_input', '')
                    if resource:
                        print(f"  • '{original_query.replace('the ', '').replace(resource.replace('_', ' '), resource.replace('_', ' ') + ' 123')}'")
                        print(f"  • 'update {resource.replace('_', ' ')} 456 with status done'")
                
                print(f"\n{'='*80}")
                print(f"CLIENT RESPONSE")
                print(f"{'='*80}")
                
                # Return structured response for client
                client_response = {
                    "status": "incomplete_query",
                    "message": "Missing required information to complete this request",
                    "missing_parameters": [
                        {
                            "name": param,
                            "description": path_params.get(param, 'N/A'),
                            "required": True
                        }
                        for param in missing_params
                    ],
                    "matched_endpoint": {
                        "method": matched_method,
                        "path": matched_endpoint,
                        "description": endpoint_details.get('description')
                    },
                    "suggestions": suggestions,
                    "original_query": parsed_result.get('original_input')
                }
                
                print(json.dumps(client_response, indent=2))
                
                # Don't fail the test, but indicate incomplete query
                self.assertTrue(
                    len(missing_params) > 0,
                    "Should detect missing parameters"
                )
                print(f"\n⚠️  Query incomplete - client should be prompted for missing information")
                return
            
            # ========================================================================
            # COMPLETE QUERY - PROCEED WITH FULL DETAILS
            # ========================================================================
            
            # Build full URL with filters
            base_url = api_schema.get('base_url', 'https://api.example.com')
            full_url = f"{base_url}{matched_endpoint}"
            
            # Replace path parameters with actual values
            if parsed_result.get('entities'):
                for key, value in parsed_result['entities'].items():
                    if f"{{{key}}}" in full_url:
                        full_url = full_url.replace(f"{{{key}}}", str(value))
            
            # Add query parameters for GET requests
            if matched_method == 'GET' and parsed_result.get('entities'):
                params = []
                for key, value in parsed_result['entities'].items():
                    # Only add as query param if not already used in path
                    if f"{{{key}}}" not in matched_endpoint:
                        params.append(f"{key}={value}")
                
                if params:
                    full_url += "?" + "&".join(params)
            
            print(f"  Full URL: {full_url}")
            
            # Display path parameters
            if endpoint_details.get('path_parameters'):
                print(f"\n  Path Parameters:")
                for param_name, param_desc in endpoint_details['path_parameters'].items():
                    value = parsed_result.get('entities', {}).get(param_name, 'N/A')
                    print(f"    - {param_name}: {value} ({param_desc})")
            
            # Display query parameters
            if matched_method == 'GET' and endpoint_details.get('query_parameters'):
                print(f"\n  Available Query Parameters:")
                for param_name, param_desc in endpoint_details['query_parameters'].items():
                    print(f"    - {param_name}: {param_desc}")
            
            # Display request body for POST/PUT
            if matched_method in ['POST', 'PUT', 'PATCH'] and endpoint_details.get('request_body'):
                request_body = endpoint_details['request_body']
                
                print(f"\n  Request Body Schema:")
                
                # Handle both dict and string request_body
                if isinstance(request_body, dict):
                    for field_name, field_desc in request_body.items():
                        print(f"    - {field_name}: {field_desc}")
                else:
                    # String description
                    print(f"    {request_body}")
                
                if parsed_result.get('entities'):
                    print(f"\n  Request Body (from query):")
                    print(f"  {json.dumps(parsed_result['entities'], indent=4)}")
            
            # Display expected response
            response_type = endpoint_details.get('response_type', 'object')
            response_schema = endpoint_details.get('response_schema', 'N/A')
            status_code = endpoint_details.get('status_code', 200)
            
            print(f"\n  Expected Response:")
            print(f"    Status: {status_code}")
            print(f"    Type: {response_type}")
            print(f"    Schema: {response_schema}")
            
            # Display authentication info
            if endpoint_details.get('authentication_required'):
                permissions = endpoint_details.get('permissions', [])
                print(f"\n  Authentication:")
                print(f"    Required: Yes")
                print(f"    Permissions: {', '.join(permissions)}")
            
            self.assertIsNotNone(matched_endpoint, "Should match an API endpoint")
            print(f"\n✓ API matching successful!")
            
        else:
            print(f"\n✗ No matching API endpoint found")
            print(f"\nAvailable endpoints for resource '{resource}':")
            
            # Show available endpoints
            if resource_to_check in modules:
                for ep_name, ep_data in modules[resource_to_check].get('endpoints', {}).items():
                    method = ep_data.get('method')
                    path = ep_data.get('path')
                    print(f"  - {method} {path} ({ep_name})")
            else:
                print(f"  Resource '{resource}' (normalized: '{resource_normalized}') not found in API spec modules")
                print(f"\n  Available modules: {', '.join(modules.keys())}")
            
            self.fail(f"No matching endpoint found for resource '{resource}' with intent '{intent}'")


def main():
    """Run the integration test."""
    # Configure unittest to run in verbose mode
    unittest.main(verbosity=2, argv=['test_matcher_integration.py'])


if __name__ == '__main__':
    main()
