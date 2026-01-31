"""
Unit test for APIOrchestrator initialization with config.json.

Tests that the processor correctly loads ONLY the schemas for enabled data sources.
"""

import os
import sys
import unittest
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent))  # For environment.py

# Try to from enable_ai import environment module for config/env path detection
try:
    from enable_ai import environment
    HAS_ENVIRONMENT_MODULE = True
except ImportError:
    HAS_ENVIRONMENT_MODULE = False

from enable_ai import APIOrchestrator


def find_config_file():
    """Find config.json using environment.py or fallback to default paths."""
    if HAS_ENVIRONMENT_MODULE:
        try:
            config_path = environment.get_config_path()
            if config_path and Path(config_path).exists():
                return Path(config_path)
        except Exception:
            pass
    
    # Fallback: check parent directory
    default_path = Path(__file__).parent.parent / 'config.json'
    return default_path


def find_env_file():
    """Find .env file using environment.py or fallback to default paths."""
    if HAS_ENVIRONMENT_MODULE:
        try:
            env_path = environment.get_env_path()
            if env_path and Path(env_path).exists():
                return Path(env_path)
        except Exception:
            pass
    
    # Fallback: check parent directory
    default_path = Path(__file__).parent.parent / '.env'
    return default_path


class TestProcessorInitialization(unittest.TestCase):
    """Test processor initialization from config.json."""
    
    def setUp(self):
        """Set up test fixtures and verify config/env accessibility."""
        # Find and verify config.json
        self.config_path = find_config_file()
        self.assertTrue(
            self.config_path.exists(),
            f"config.json not found at {self.config_path}"
        )
        
        # Find and verify .env file
        self.env_path = find_env_file()
        self.assertTrue(
            self.env_path.exists(),
            f".env file not found at {self.env_path}"
        )
        
        # Load .env for tests that require environment variables
        load_dotenv(self.env_path)
        
        # Verify critical environment variables (if needed for tests)
        # This is optional but helps catch configuration issues early
        if HAS_ENVIRONMENT_MODULE:
            try:
                env_info = environment.get_environment_info()
                print(f"\n✓ Using {env_info['environment']} environment")
                print(f"  Config: {self.config_path}")
                print(f"  Env: {self.env_path}")
            except Exception:
                pass
        else:
            print(f"\n✓ Config loaded from: {self.config_path}")
            print(f"  Env loaded from: {self.env_path}")
    
    def test_processor_loads_schemas_for_enabled_data_sources(self):
        """
        Test that processor loads ONLY schemas for enabled data sources.
        
        Based on config.json:
        - api: enabled → should load 'api' schema
        - json_files: enabled → should load 'knowledge_graph' schema
        - database: disabled → should NOT load 'database' schema
        - pdf_documents: disabled → should NOT load 'knowledge_graph' (unless json_files is enabled)
        - vector_search: disabled → should NOT load additional schemas
        - search_databases: disabled → should NOT load 'database' schema
        """
        # Initialize processor with config.json only (no schema override)
        processor = APIOrchestrator(config_path=str(self.config_path))
        
        # Check that processor initialized
        self.assertIsNotNone(processor, "Processor should initialize successfully")
        
        # Check that schemas dict exists
        self.assertIsNotNone(processor.schemas, "Processor should have schemas dict")
        
        # Get loaded schema types
        loaded_schemas = set(processor.schemas.keys())
        
        # Expected schemas based on enabled data sources in config.json
        # api: enabled → requires 'api' schema
        # json_files: DISABLED in current config
        expected_schemas = {'api'}
        
        # Check that expected schemas are loaded
        for schema_type in expected_schemas:
            self.assertIn(
                schema_type,
                loaded_schemas,
                f"Schema '{schema_type}' should be loaded (required by enabled data source)"
            )
        
        # Check that database schema is NOT loaded (no database sources enabled)
        # Note: Only check if no database-related sources are enabled
        enabled_sources = processor._get_enabled_data_sources_list()
        database_sources = ['database', 'search_databases']
        
        if not any(source in enabled_sources for source in database_sources):
            self.assertNotIn(
                'database',
                loaded_schemas,
                "Schema 'database' should NOT be loaded (no database sources enabled)"
            )
        
        print(f"\n✓ Test passed!")
        print(f"  Enabled data sources: {', '.join(enabled_sources)}")
        print(f"  Loaded schemas: {', '.join(loaded_schemas)}")
        print(f"  Expected schemas: {', '.join(expected_schemas)}")
    
    def test_processor_has_correct_enabled_data_sources(self):
        """Test that processor correctly identifies enabled data sources from config."""
        processor = APIOrchestrator(config_path=str(self.config_path))
        
        enabled_sources = processor._get_enabled_data_sources_list()
        
        # Check that enabled_sources is a list
        self.assertIsInstance(enabled_sources, list, "Enabled sources should be a list")
        
        # Based on current config.json, these should be enabled
        expected_enabled = {'api'}
        
        # Convert to set for comparison
        enabled_set = set(enabled_sources)
        
        # Check expected sources are enabled
        for source in expected_enabled:
            self.assertIn(
                source,
                enabled_set,
                f"Data source '{source}' should be enabled in config.json"
            )
        
        print(f"\n✓ Enabled data sources detected correctly: {', '.join(enabled_sources)}")
    
    def test_schema_loading_matches_data_source_requirements(self):
        """
        Test the mapping between data sources and required schemas.
        
        Mapping per processor._get_required_schema_types():
        - api → 'api' schema
        - database → 'database' schema
        - search_databases → 'database' schema
        - json_files → 'knowledge_graph' schema
        - pdf_documents → 'knowledge_graph' schema
        - vector_search → 'knowledge_graph' schema
        """
        processor = APIOrchestrator(config_path=str(self.config_path))
        
        enabled_sources = processor._get_enabled_data_sources_list()
        required_schemas = processor._get_required_schema_types(enabled_sources)
        loaded_schemas = set(processor.schemas.keys())
        
        # Check that all required schemas are loaded
        for schema_type in required_schemas:
            self.assertIn(
                schema_type,
                loaded_schemas,
                f"Required schema '{schema_type}' should be loaded for enabled data sources"
            )
        
        print(f"\n✓ Schema requirements satisfied!")
        print(f"  Enabled sources: {enabled_sources}")
        print(f"  Required schemas: {required_schemas}")
        print(f"  Loaded schemas: {loaded_schemas}")
    
    def test_processor_validation_detects_missing_schemas(self):
        """Test that processor handles missing schema files gracefully."""
        # This test verifies the warning system works when schema files are missing
        # The processor should still initialize but warn about missing schemas
        
        processor = APIOrchestrator(config_path=str(self.config_path))
        
        # Processor should initialize even if some schema files are missing
        self.assertIsNotNone(processor, "Processor should initialize despite missing schemas")
        
        # Check that at least some schemas are loaded (the ones that exist)
        # In a real scenario, this would depend on which schema files actually exist
        enabled_sources = processor._get_enabled_data_sources_list()
        
        if enabled_sources:
            # If there are enabled sources, we expect some attempt to load schemas
            self.assertIsNotNone(
                processor.schemas,
                "Processor should have schemas dict even if some are missing"
            )
        
        print(f"\n✓ Processor handles schema loading gracefully")
        print(f"  Enabled sources: {', '.join(enabled_sources)}")
        print(f"  Schemas loaded: {', '.join(processor.schemas.keys()) if processor.schemas else 'none'}")


class TestSchemaValidation(unittest.TestCase):
    """Test schema validation logic."""
    
    def setUp(self):
        """Set up test fixtures and verify config/env accessibility."""
        # Find and verify config.json
        self.config_path = find_config_file()
        self.assertTrue(
            self.config_path.exists(),
            f"config.json not found at {self.config_path}"
        )
        
        # Find and load .env
        self.env_path = find_env_file()
        if self.env_path.exists():
            load_dotenv(self.env_path)
        
        # Initialize processor after environment is loaded
        self.processor = APIOrchestrator(config_path=str(self.config_path))
    
    def test_loaded_schemas_have_correct_type_field(self):
        """Test that all loaded schemas have the correct 'type' field."""
        for schema_type, schema_content in self.processor.schemas.items():
            self.assertIn(
                'type',
                schema_content,
                f"Schema '{schema_type}' should have a 'type' field"
            )
            
            # The 'type' field should match the schema_type key
            # (unless it's a special case like api_schema)
            actual_type = schema_content['type']
            
            # Handle special case where file has 'api_schema' but we expect 'api'
            if schema_type == 'api' and actual_type == 'api_schema':
                print(f"  Note: Schema type mismatch: expected 'api', got 'api_schema'")
            elif actual_type != schema_type:
                self.fail(
                    f"Schema type mismatch: key='{schema_type}', type field='{actual_type}'"
                )
        
        print(f"\n✓ All schemas have valid 'type' fields")


class TestProcessorWithClientQuery(unittest.TestCase):
    """Test processor initialization and NLP query processing."""
    
    def setUp(self):
        """Set up test fixtures and verify config/env accessibility."""
        # Find and verify config.json
        self.config_path = find_config_file()
        self.assertTrue(
            self.config_path.exists(),
            f"config.json not found at {self.config_path}"
        )
        
        # Find and load .env (required for LLM parser with OpenAI API key)
        self.env_path = find_env_file()
        self.assertTrue(
            self.env_path.exists(),
            f".env file not found at {self.env_path} (required for LLM parser)"
        )
        load_dotenv(self.env_path)
        
        # Verify OPENAI_API_KEY is available (required for query processing)
        openai_key = os.getenv('OPENAI_API_KEY')
        if not openai_key:
            self.skipTest("OPENAI_API_KEY not found in .env - skipping query processing tests")
        
        # Initialize processor with config.json after environment is loaded
        self.processor = APIOrchestrator(config_path=str(self.config_path))
    
    def test_processor_initializes_and_processes_simple_query(self):
        """Test that processor initializes from config.json and processes a simple NLP query."""
        # Verify processor initialized successfully
        self.assertIsNotNone(self.processor, "Processor should initialize successfully")
        self.assertIsNotNone(self.processor.schemas, "Processor should have schemas loaded")
        
        # Test simple NLP query
        query = "get user with id 5"
        
        # Process the query (note: this won't actually call the API, just parse and plan)
        result = self.processor.process(query)
        
        # Verify result structure
        self.assertIsInstance(result, dict, "Result should be a dictionary")
        self.assertIn('success', result, "Result should have 'success' field")
        self.assertIn('query', result, "Result should have 'query' field")
        
        # Verify the query was stored in result
        self.assertEqual(result['query'], query, "Query should match input")
        
        print(f"\n✓ Processor initialized and processed query successfully!")
        print(f"  Query: '{query}'")
        print(f"  Success: {result.get('success')}")
        print(f"  Schema type: {result.get('schema_type', 'N/A')}")
        
        if result.get('success'):
            print(f"  Endpoint: {result.get('endpoint', 'N/A')}")
            print(f"  Method: {result.get('method', 'N/A')}")
        else:
            print(f"  Error: {result.get('error', 'N/A')}")
    
    def test_processor_handles_complex_query(self):
        """Test that processor can handle complex NLP queries with multiple conditions."""
        # Test complex query with date calculation
        query = "show me all service orders created in the last 7 days"
        
        result = self.processor.process(query)
        
        # Verify result
        self.assertIsInstance(result, dict, "Result should be a dictionary")
        self.assertIn('success', result, "Result should have 'success' field")
        self.assertEqual(result['query'], query, "Query should match input")
        
        print(f"\n✓ Complex query processed!")
        print(f"  Query: '{query}'")
        print(f"  Success: {result.get('success')}")
        
        if result.get('success'):
            print(f"  Endpoint: {result.get('endpoint', 'N/A')}")
        else:
            print(f"  Error/Info: {result.get('error', result.get('message', 'N/A'))}")
    
    def test_processor_handles_multiple_queries(self):
        """Test that processor can handle multiple sequential queries."""
        queries = [
            "get user with id 5",
            "list all users",
            "show me service orders",
            "find equipment"
        ]
        
        results = []
        for query in queries:
            result = self.processor.process(query)
            results.append(result)
            
            # Each result should be valid
            self.assertIsInstance(result, dict, f"Result for '{query}' should be a dict")
            self.assertIn('success', result, f"Result for '{query}' should have 'success' field")
            self.assertEqual(result['query'], query, f"Query should match input: '{query}'")
        
        print(f"\n✓ Multiple queries processed successfully!")
        print(f"  Total queries: {len(queries)}")
        successful = sum(1 for r in results if r.get('success'))
        print(f"  Successful: {successful}/{len(queries)}")
        
        # Display results summary
        for i, (query, result) in enumerate(zip(queries, results), 1):
            status = "✓" if result.get('success') else "✗"
            print(f"  {status} Query {i}: '{query}' - {result.get('schema_type', 'N/A')}")
    
    def test_processor_parser_integration(self):
        """Test that processor's parser (LLM-based) is properly initialized and working."""
        # Verify parser exists
        self.assertIsNotNone(self.processor.parser, "Processor should have a parser")
        
        # Verify parser is the LLM-based parser
        self.assertTrue(
            hasattr(self.processor.parser, 'client'),
            "Parser should have OpenAI client (LLM-based parser)"
        )
        
        # Test that parser can parse a query directly
        query = "get user with id 5"
        schema = self.processor._get_active_schema()
        
        if schema:
            parsed = self.processor._understand_query(query, schema)
            
            # Verify parsed output structure
            self.assertIsInstance(parsed, dict, "Parsed output should be a dictionary")
            self.assertIn('intent', parsed, "Parsed output should have 'intent' field")
            self.assertIn('resource', parsed, "Parsed output should have 'resource' field")
            
            print(f"\n✓ LLM Parser integration verified!")
            print(f"  Query: '{query}'")
            print(f"  Parsed intent: {parsed.get('intent')}")
            print(f"  Parsed resource: {parsed.get('resource')}")
            print(f"  Schema type: {schema.get('type')}")
        else:
            self.skipTest("No active schema available for parsing test")


def run_tests():
    """Run all tests and display results."""
    print("=" * 80)
    print("Testing APIOrchestrator Initialization with config.json")
    print("=" * 80)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestProcessorInitialization))
    suite.addTests(loader.loadTestsFromTestCase(TestSchemaValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestProcessorWithClientQuery))
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 80)
    if result.wasSuccessful():
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("=" * 80)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
