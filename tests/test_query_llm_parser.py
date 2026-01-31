"""
Unit test for LLM Parser with manual client query input.

Flow:
1. Initialize processor with config.json
2. Accept user query as argument
3. Generate schema based on enabled data sources
4. Parse client query using LLM with schema
5. Display parser output
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


class TestLLMParserWithQuery(unittest.TestCase):
    """Test LLM parser with manual query input."""
    
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
            print("LLM PARSER INTERACTIVE TEST")
            print("="*80)
            print("\nThis test will:")
            print("  1. Initialize the processor with config.json")
            print("  2. Generate schemas if needed")
            print("  3. Parse your query using GPT-4o LLM")
            print("\nExamples:")
            print("  - get user with id 5")
            print("  - show me urgent service orders assigned to John")
            print("  - find equipment with quantity less than 10")
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
        self.assertIsNotNone(self.processor.parser, "Parser should be initialized")
        
        print(f"\n✓ Processor initialized with config.json")
        print(f"  Config path: {self.config_path}")
        print(f"  User query: '{self.user_query}'")
    
    def test_02_check_and_generate_schemas(self):
        """Step 2: Check schemas and generate if missing."""
        print(f"\n{'='*80}")
        print(f"STEP 2: SCHEMA GENERATION")
        print(f"{'='*80}")
        
        # Initialize processor
        self.processor = APIOrchestrator(config_path=str(self.config_path))
        
        # Get enabled data sources and required schemas
        enabled_sources = self.processor._get_enabled_data_sources_list()
        required_schemas = self.processor._get_required_schema_types(enabled_sources)
        
        print(f"\nEnabled data sources: {', '.join(enabled_sources)}")
        print(f"Required schema types: {', '.join(sorted(required_schemas))}")
        
        # Check for missing schemas
        with open(self.config_path, 'r') as f:
            config = json.load(f)
        
        schema_paths = config.get('schemas', {})
        missing_schemas = []
        
        for schema_type in required_schemas:
            schema_path = schema_paths.get(schema_type)
            if schema_path:
                full_path = self.schemas_dir.parent / schema_path
                if not full_path.exists():
                    missing_schemas.append(schema_type)
        
        if missing_schemas:
            print(f"\n⚠️  Missing schemas: {', '.join(missing_schemas)}")
            
            # Generate API schema if missing
            if 'api' in missing_schemas:
                generator_script = Path(__file__).parent.parent / 'generate_schema.py'
                
                if generator_script.exists():
                    print(f"\n⚙️  Generating API schema...")
                    
                    result = subprocess.run(
                        [sys.executable, str(generator_script)],
                        cwd=str(generator_script.parent),
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    
                    if result.returncode == 0:
                        print(f"✓ Schema generated successfully")
                        print(result.stdout)
                        
                        # Reload processor to pick up new schema
                        self.processor = APIOrchestrator(config_path=str(self.config_path))
                    else:
                        self.fail(f"Schema generation failed: {result.stderr}")
        else:
            print(f"\n✓ All required schemas exist")
        
        # Verify schemas are loaded
        loaded_schemas = set(self.processor.schemas.keys())
        print(f"\nLoaded schemas: {', '.join(sorted(loaded_schemas))}")
        
        for schema_type in required_schemas:
            self.assertIn(
                schema_type,
                loaded_schemas,
                f"Required schema '{schema_type}' should be loaded"
            )
    
    def test_03_parse_query_with_llm(self):
        """Step 3: Parse client query using LLM parser with schema."""
        print(f"\n{'='*80}")
        print(f"STEP 3: LLM QUERY PARSING")
        print(f"{'='*80}")
        
        # Initialize processor
        self.processor = APIOrchestrator(config_path=str(self.config_path))
        
        # Get active schema
        active_schema = self.processor._get_active_schema()
        
        self.assertIsNotNone(active_schema, "Active schema should be available")
        
        print(f"\nQuery to parse: '{self.user_query}'")
        print(f"Schema type: {active_schema.get('type')}")
        print(f"Schema resources: {len(active_schema.get('resources', {}))} available")
        
        # Parse the query using LLM
        print(f"\n🤖 Parsing query with LLM (GPT-4o)...")
        parsed_output = self.processor._understand_query(self.user_query, active_schema)
        
        # Verify parsed output structure
        self.assertIsInstance(parsed_output, dict, "Parsed output should be a dictionary")
        self.assertIn('intent', parsed_output, "Parsed output should have 'intent'")
        self.assertIn('resource', parsed_output, "Parsed output should have 'resource'")
        
        # Display parser output
        print(f"\n{'='*80}")
        print(f"PARSER OUTPUT")
        print(f"{'='*80}")
        print(json.dumps(parsed_output, indent=2))
        
        # Display key extracted information
        print(f"\n{'='*80}")
        print(f"EXTRACTED INFORMATION")
        print(f"{'='*80}")
        print(f"  Intent: {parsed_output.get('intent')}")
        print(f"  Resource: {parsed_output.get('resource')}")
        print(f"  Schema Type: {parsed_output.get('schema_type')}")
        
        if 'filters' in parsed_output and parsed_output['filters']:
            print(f"\n  Filters:")
            for field, filter_info in parsed_output['filters'].items():
                if isinstance(filter_info, dict):
                    operator = filter_info.get('operator', 'N/A')
                    value = filter_info.get('value', 'N/A')
                    print(f"    - {field}: {operator} {value}")
                else:
                    print(f"    - {field}: {filter_info}")
        
        if 'entities' in parsed_output and parsed_output['entities']:
            print(f"\n  Entities: {parsed_output['entities']}")
        
        if 'relationships' in parsed_output and parsed_output['relationships']:
            print(f"\n  Relationships: {len(parsed_output['relationships'])} detected")
            for rel in parsed_output['relationships']:
                print(f"    - {rel.get('type')}: {rel.get('target_entity')}")
        
        if 'sort' in parsed_output:
            print(f"\n  Sort: {parsed_output['sort']}")
        
        if 'limit' in parsed_output:
            print(f"  Limit: {parsed_output['limit']}")
        
        print(f"\n✓ Query parsed successfully using LLM")


def run_tests_with_query(query=None):
    """Run tests with optional custom query."""
    print("\n" + "="*80)
    print("LLM PARSER TEST WITH CLIENT QUERY")
    print("="*80)
    
    if query:
        # Inject query into sys.argv for the test
        sys.argv = [sys.argv[0]] + query.split()
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add tests in order
    suite.addTests(loader.loadTestsFromTestCase(TestLLMParserWithQuery))
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*80)
    if result.wasSuccessful():
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*80 + "\n")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    # Check if query provided as command line argument
    if len(sys.argv) > 1 and not sys.argv[1].startswith('test_'):
        # Query provided - run with that query
        query = ' '.join(sys.argv[1:])
        print(f"\n📝 Using provided query: '{query}'")
        success = run_tests_with_query(query)
    else:
        # No query - use default
        print(f"\n📝 No query provided, using default test query")
        print(f"   To provide a query: python test_query_llm_parser.py 'your query here'")
        success = run_tests_with_query()
    
    sys.exit(0 if success else 1)
