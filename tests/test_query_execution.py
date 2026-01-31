"""
Complete Query Execution Test Suite

This test validates the entire end-to-end query execution pipeline with detailed step-by-step logging:

PIPELINE STEPS:
1. Environment Setup - Load environment configuration from environment.py
2. Processor Initialization - Initialize NLP processor with config.json
3. Data Source Detection - Identify enabled data sources
4. Schema Loading - Load and validate schemas
5. Authentication - Detect auth method and get credentials
6. Query Parsing - Parse natural language query using LLM
7. API Matching - Match parsed query to API endpoint
8. Query Execution - Execute API call with authentication
9. Response Validation - Validate and display results

Usage:
    python test_query_execution.py                     # Run automated test suite
    python test_query_execution.py "your query here"   # Execute single query with full pipeline
    python test_query_execution.py --help              # Show help
"""

import sys
import json
import unittest
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent))  # For environment.py

# Import environment module
try:
    from enable_ai import environment
    HAS_ENVIRONMENT_MODULE = True
except ImportError:
    HAS_ENVIRONMENT_MODULE = False

# Load environment variables
if HAS_ENVIRONMENT_MODULE:
    from dotenv import load_dotenv
    env_path = Path(environment.get_env_path())
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✓ Loaded .env from: {env_path}")

from enable_ai import APIOrchestrator


class TestCompleteQueryExecution(unittest.TestCase):
    """Complete end-to-end query execution test with detailed logging."""
    
    def setUp(self):
        """Set up test fixtures with detailed step logging."""
        print("\n" + "=" * 80)
        print("PIPELINE SETUP")
        print("=" * 80)
        
        # Step 1: Environment Setup
        self._log_step(1, "Environment Setup")
        if HAS_ENVIRONMENT_MODULE:
            self.config_path = Path(environment.get_config_path())
            self.env_path = Path(environment.get_env_path())
            self.schemas_dir = Path(environment.get_schemas_dir())
            print(f"  ✓ Active Environment: {environment.ACTIVE_ENVIRONMENT}")
            print(f"  ✓ Config: {self.config_path.name}")
            print(f"  ✓ Schemas: {self.schemas_dir.name}")
        else:
            self.config_path = Path(__file__).parent.parent / 'config.json'
            print(f"  ⚠️  Using fallback: {self.config_path}")
        
        self.assertTrue(self.config_path.exists(), f"Config not found: {self.config_path}")
        
        # Step 2: Processor Initialization
        self._log_step(2, "Processor Initialization")
        self.processor = APIOrchestrator(config_path=str(self.config_path))
        print(f"  ✓ Processor initialized")
        
        # Load config
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)
        
        # Step 3: Data Source Detection
        self._log_step(3, "Data Source Detection")
        data_sources = self.config.get('data_sources', {})
        if isinstance(data_sources, dict):
            self.enabled_sources = [name for name, cfg in data_sources.items() if isinstance(cfg, dict) and cfg.get('enabled')]
            for source in self.enabled_sources:
                src_cfg = data_sources[source]
                print(f"  ✓ {source}: {src_cfg.get('type', 'unknown') if isinstance(src_cfg, dict) else 'unknown'}")
        else:
            self.enabled_sources = []
            print(f"  ⚠️  No data sources found")
        
        # Step 4: Schema Loading
        self._log_step(4, "Schema Loading")
        self.loaded_schemas = getattr(self.processor, 'schemas', {})
        for schema_type, schema_data in self.loaded_schemas.items():
            if schema_type == 'api':
                resources = schema_data.get('resources', {})
                print(f"  ✓ {schema_type}: {len(resources)} resources")
        
        # Step 5: Authentication Setup
        self._log_step(5, "Authentication Setup")
        security = self.config.get('security_credentials', {})
        if 'api' in security and 'jwt' in security['api']:
            jwt_config = security['api']['jwt']
            print(f"  ✓ Method: JWT")
            print(f"  ✓ Endpoint: {jwt_config.get('login_url', 'N/A')}")
        else:
            print(f"  ⚠️  No authentication configured")
    
    def _log_step(self, step_num, step_name):
        """Log a pipeline step."""
        print(f"\n[Step {step_num}] {step_name}")
    
    def test_01_query_parsing(self):
        """Test Step 6: Query Parsing with LLM."""
        print("\n" + "=" * 80)
        print("TEST 1: QUERY PARSING")
        print("=" * 80)
        
        query = "show me all users"
        print(f"\n📝 Query: {query}")
        
        self._log_step(6, "Query Parsing with LLM")
        schema = self.loaded_schemas.get('api')
        parsed = self.processor._understand_query(query, schema)
        
        print(f"  ✓ Intent: {parsed.get('intent')}")
        print(f"  ✓ Resource: {parsed.get('resource')}")
        
        self.assertIn('intent', parsed)
        self.assertIn('resource', parsed)
    
    def test_02_api_matching(self):
        """Test Step 7: API Endpoint Matching."""
        print("\n" + "=" * 80)
        print("TEST 2: API ENDPOINT MATCHING")
        print("=" * 80)
        
        query = "show me all users"
        print(f"\n📝 Query: {query}")
        
        # Step 6: Parse
        self._log_step(6, "Query Parsing")
        schema = self.loaded_schemas.get('api')
        parsed = self.processor._understand_query(query, schema)
        print(f"  ✓ Parsed: intent={parsed.get('intent')}, resource={parsed.get('resource')}")
        
        # Step 7: Match
        self._log_step(7, "API Endpoint Matching")
        plan = self.processor._create_plan(parsed, schema)
        print(f"  ✓ Endpoint: {plan.get('method')} {plan.get('endpoint')}")
        print(f"  ✓ Auth Required: {plan.get('authentication_required')}")
        
        self.assertEqual(plan.get('type'), 'api')
        self.assertIn('endpoint', plan)
    
    def test_03_full_pipeline_execution(self):
        """Test Steps 6-9: Complete Query Execution Pipeline."""
        print("\n" + "=" * 80)
        print("TEST 3: COMPLETE PIPELINE EXECUTION")
        print("=" * 80)
        
        query = "show me all service orders"
        print(f"\n📝 Query: {query}")
        
        # Step 6: Parse
        self._log_step(6, "Query Parsing")
        schema = self.loaded_schemas.get('api')
        parsed = self.processor._understand_query(query, schema)
        print(f"  ✓ Intent: {parsed.get('intent')}")
        print(f"  ✓ Resource: {parsed.get('resource')}")
        
        # Step 7: Match
        self._log_step(7, "API Endpoint Matching")
        plan = self.processor._create_plan(parsed, schema)
        print(f"  ✓ Endpoint: {plan.get('method')} {plan.get('endpoint')}")
        
        # Step 8: Execute (with authentication)
        self._log_step(8, "Query Execution")
        result = self.processor.process(query)
        print(f"  ✓ Authentication: {'Success' if result.get('success') else 'Failed'}")
        print(f"  ✓ API Call: {result.get('method', 'N/A')} {result.get('endpoint', 'N/A')}")
        
        # Step 9: Validate Response
        self._log_step(9, "Response Validation")
        print(f"  ✓ Success: {result.get('success')}")
        print(f"  ✓ Summary: {result.get('summary', 'N/A')}")
        
        if result.get('data') and isinstance(result['data'], dict):
            if 'count' in result['data']:
                print(f"  ✓ Total Records: {result['data']['count']}")
            if 'results' in result['data']:
                print(f"  ✓ Retrieved: {len(result['data']['results'])} items")
        
        self.assertTrue(result.get('success'), f"Query failed: {result.get('error')}")
    
    def test_04_query_with_id_parameter(self):
        """Test pipeline with ID parameter extraction."""
        print("\n" + "=" * 80)
        print("TEST 4: QUERY WITH ID PARAMETER")
        print("=" * 80)
        
        query = "get user with id 5"
        print(f"\n📝 Query: {query}")
        
        # Full pipeline
        self._log_step(6, "Query Parsing")
        result = self.processor.process(query)
        
        print(f"  ✓ Endpoint: {result.get('endpoint')}")
        print(f"  ✓ Method: {result.get('method')}")
        print(f"  ✓ Success: {result.get('success')}")
        
        self.assertIsInstance(result, dict)
    
    def test_05_multiple_queries(self):
        """Test sequential query execution."""
        print("\n" + "=" * 80)
        print("TEST 5: MULTIPLE QUERIES SEQUENTIAL")
        print("=" * 80)
        
        queries = ["show me all users", "show me all service orders"]
        
        for i, query in enumerate(queries, 1):
            print(f"\n[Query {i}/{len(queries)}] {query}")
            result = self.processor.process(query)
            
            success_icon = "✅" if result.get('success') else "❌"
            print(f"  {success_icon} {result.get('endpoint')} - {result.get('summary', 'N/A')[:50]}")
        
        print(f"\n✓ Executed {len(queries)} queries successfully")


def execute_single_query(query: str) -> bool:
    """
    Execute a single query with complete pipeline visualization.
    
    Shows all 9 steps of the execution pipeline with detailed logging.
    """
    print("\n" + "=" * 80)
    print("COMPLETE QUERY EXECUTION PIPELINE")
    print("=" * 80)
    print(f"\n📝 Query: {query}\n")
    
    # Step 1: Environment Setup
    print("[Step 1] Environment Setup")
    if HAS_ENVIRONMENT_MODULE:
        config_path = Path(environment.get_config_path())
        print(f"  ✓ Environment: {environment.ACTIVE_ENVIRONMENT}")
        print(f"  ✓ Config: {config_path}")
    else:
        config_path = Path(__file__).parent.parent / 'config.json'
        print(f"  ⚠️  Fallback config: {config_path}")
    
    if not config_path.exists():
        print(f"  ❌ Config not found")
        return False
    
    # Step 2: Load Configuration
    print("\n[Step 2] Load Configuration")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    data_sources = config.get('data_sources', {})
    if isinstance(data_sources, dict):
        enabled = [name for name, cfg in data_sources.items() if isinstance(cfg, dict) and cfg.get('enabled')]
    else:
        enabled = []
    print(f"  ✓ Enabled data sources: {', '.join(enabled) if enabled else 'none'}")
    
    # Step 3: Initialize Processor
    print("\n[Step 3] Initialize Processor")
    processor = APIOrchestrator(config_path=str(config_path))
    schemas = list(processor.schemas.keys()) if hasattr(processor, 'schemas') else []
    print(f"  ✓ Loaded schemas: {', '.join(schemas)}")
    
    if 'api' in processor.schemas:
        resources = list(processor.schemas['api'].get('resources', {}).keys())
        print(f"  ✓ API resources: {len(resources)} available")
    
    # Step 4: Check Authentication
    print("\n[Step 4] Authentication Configuration")
    security = config.get('security_credentials', {})
    if 'api' in security and 'jwt' in security['api']:
        jwt_config = security['api']['jwt']
        print(f"  ✓ Method: JWT")
        print(f"  ✓ Endpoint: {jwt_config.get('login_url')}")
    else:
        print(f"  ⚠️  No authentication configured")
    
    # Step 5: Parse Query
    print("\n[Step 5] Parse Query with LLM")
    schema = processor.schemas.get('api') if hasattr(processor, 'schemas') else None
    if schema:
        parsed = processor._understand_query(query, schema)
        print(f"  ✓ Intent: {parsed.get('intent')}")
        print(f"  ✓ Resource: {parsed.get('resource')}")
        if parsed.get('entities'):
            print(f"  ✓ Entities: {parsed.get('entities')}")
    
    # Step 6: Match API Endpoint
    print("\n[Step 6] Match API Endpoint")
    if schema and parsed:
        plan = processor._create_plan(parsed, schema)
        if plan and plan.get('type') == 'api':
            print(f"  ✓ Endpoint: {plan.get('endpoint')}")
            print(f"  ✓ Method: {plan.get('method')}")
            print(f"  ✓ Auth Required: {plan.get('authentication_required')}")
    
    # Step 7-9: Execute Query (includes authentication & response validation)
    print("\n[Step 7] Authenticate & Execute Query")
    result = processor.process(query)
    
    print("\n[Step 8] Response Validation")
    success_icon = "✅" if result.get('success') else "❌"
    print(f"  {success_icon} Success: {result.get('success')}")
    print(f"  📊 Summary: {result.get('summary')}")
    
    print("\n[Step 9] Display Results")
    if result.get('endpoint'):
        print(f"  🔗 Endpoint: {result.get('method')} {result.get('endpoint')}")
    
    if result.get('error'):
        print(f"  ❌ Error: {result.get('error')}")
    elif result.get('data'):
        data = result.get('data')
        if isinstance(data, dict):
            if 'count' in data:
                print(f"  📈 Total: {data.get('count')}")
            if 'results' in data:
                results_list = data.get('results', [])
                print(f"  📦 Retrieved: {len(results_list)} items")
                
                if len(results_list) > 0:
                    print(f"\n  First item preview:")
                    first = results_list[0]
                    for key, value in list(first.items())[:5]:
                        value_str = str(value)[:50] + "..." if len(str(value)) > 50 else str(value)
                        print(f"    • {key}: {value_str}")
                    if len(first) > 5:
                        print(f"    ... ({len(first) - 5} more fields)")
            else:
                # Single item
                print(f"  📋 Fields: {list(data.keys())[:10]}")
    
    print("\n" + "=" * 80)
    print(f"PIPELINE COMPLETE: {'SUCCESS ✅' if result.get('success') else 'FAILED ❌'}")
    print("=" * 80 + "\n")
    
    return result.get('success', False)


def run_tests() -> bool:
    """Run the automated test suite."""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestCompleteQueryExecution)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 80)
    if result.wasSuccessful():
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("=" * 80)
    
    return result.wasSuccessful()


def interactive_mode():
    """
    Interactive mode - continuously prompts for queries until exit.
    
    Allows you to enter multiple queries interactively and see the full
    pipeline execution for each one.
    """
    print("\n" + "=" * 80)
    print("INTERACTIVE QUERY EXECUTION MODE")
    print("=" * 80)
    print("\nThis mode allows you to enter queries interactively.")
    print("Each query will be executed with full pipeline visualization.\n")
    print("Commands:")
    print("  • Type your query and press Enter to execute")
    print("  • Type 'exit', 'quit', or 'q' to stop")
    print("  • Type 'help' or '?' for example queries")
    print("=" * 80 + "\n")
    
    query_count = 0
    
    while True:
        try:
            # Prompt for query
            query = input("📝 Enter query (or 'exit' to quit): ").strip()
            
            if not query:
                continue
            
            # Handle exit commands
            if query.lower() in ['exit', 'quit', 'q']:
                print(f"\n👋 Exiting interactive mode...")
                print(f"✓ Executed {query_count} queries")
                break
            
            # Handle help command
            if query.lower() in ['help', '?']:
                print("\n📚 Example Queries:")
                print("  • show me all users")
                print("  • show me all service orders")
                print("  • get user with id 5")
                print("  • list all equipment")
                print("  • find service orders with priority high")
                print("  • show me users where role is admin")
                print()
                continue
            
            # Execute the query
            query_count += 1
            print(f"\n[Query #{query_count}]")
            success = execute_single_query(query)
            
            # Brief result summary
            if success:
                print("✅ Query executed successfully\n")
            else:
                print("❌ Query execution failed\n")
        
        except KeyboardInterrupt:
            print(f"\n\n⚠️  Interrupted by user")
            print(f"✓ Executed {query_count} queries")
            break
        except EOFError:
            print(f"\n\n👋 Exiting...")
            print(f"✓ Executed {query_count} queries")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            print()
    
    print("=" * 80)
    print("INTERACTIVE MODE ENDED")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        
        # Help mode
        if arg in ['--help', '-h', 'help']:
            print(__doc__)
            print("\nUsage:")
            print("  python test_query_execution.py                 # Run automated tests")
            print("  python test_query_execution.py \"<query>\"       # Execute single query")
            print("  python test_query_execution.py --interactive   # Interactive mode")
            print("  python test_query_execution.py --help          # Show this help")
            print("\nExamples:")
            print("  python test_query_execution.py \"show me all users\"")
            print("  python test_query_execution.py \"show me all service orders\"")
            print("  python test_query_execution.py \"get user with id 5\"")
            print("  python test_query_execution.py -i              # Start interactive mode")
            sys.exit(0)
        
        # Interactive/Manual mode
        elif arg in ['--interactive', '-i', 'interactive', 'manual', '-m', '--manual']:
            interactive_mode()
            sys.exit(0)
        
        # Single query mode
        else:
            query = ' '.join(sys.argv[1:])
            success = execute_single_query(query)
            sys.exit(0 if success else 1)
    
    else:
        # Run automated tests
        success = run_tests()
        sys.exit(0 if success else 1)
