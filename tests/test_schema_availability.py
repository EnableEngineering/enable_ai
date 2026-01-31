"""
Unit test to verify enabled data sources and schema availability.

This comprehensive test suite validates:

1. DATA SOURCE DETECTION (TestEnabledDataSources):
   - Identifies which data sources are enabled in config.json
   - Maps enabled data sources to required schema types
   - Validates the mapping logic

2. SCHEMA AVAILABILITY (TestSchemaAvailability):
   - Verifies schemas directory exists
   - Checks that schema files exist for all enabled data sources
   - Validates schema files are valid JSON
   - Verifies schema data references match config.json

3. SCHEMA GENERATION (TestSchemaGeneration):
   - Checks API schema generation capability (api_spec.json → api_schema.json)
   - Verifies schema generator script exists
   - Validates all required schemas are loaded by processor

Usage:
    python test_schema_availability.py
    
Expected Output:
    - Shows enabled/disabled data sources from config.json
    - Displays data source → schema type mapping
    - Lists available vs missing schema files
    - Validates schema references to data sources
    - Confirms schema generation availability
"""

import sys
import unittest
import json
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


class TestEnabledDataSources(unittest.TestCase):
    """Test enabled data sources detection from config.json."""
    
    def setUp(self):
        """Set up test fixtures."""
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
        
        # Load config
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)
    
    def test_identify_enabled_data_sources(self):
        """Test identification of enabled data sources from config.json."""
        data_sources = self.config.get('data_sources', {})
        
        enabled_sources = []
        disabled_sources = []
        
        for source_name, source_config in data_sources.items():
            if isinstance(source_config, dict):
                if source_config.get('enabled', False):
                    enabled_sources.append(source_name)
                else:
                    disabled_sources.append(source_name)
        
        # Display results
        print(f"\n{'='*80}")
        print(f"DATA SOURCES IN CONFIG.JSON")
        print(f"{'='*80}")
        print(f"\n✓ Enabled Data Sources ({len(enabled_sources)}):")
        for source in enabled_sources:
            source_config = data_sources[source]
            print(f"  • {source}")
            print(f"    - Name: {source_config.get('name', 'N/A')}")
            print(f"    - Description: {source_config.get('description', 'N/A')[:60]}...")
        
        print(f"\n✗ Disabled Data Sources ({len(disabled_sources)}):")
        for source in disabled_sources:
            print(f"  • {source}")
        
        # Store for other tests
        self.enabled_sources = enabled_sources
        self.disabled_sources = disabled_sources
        
        # Assertions
        self.assertIsInstance(enabled_sources, list, "Enabled sources should be a list")
        self.assertGreater(len(data_sources), 0, "Config should have data sources defined")
        
        return enabled_sources, disabled_sources
    
    def test_map_data_sources_to_schema_types(self):
        """Test mapping of enabled data sources to required schema types."""
        # Get enabled sources
        enabled_sources, _ = self.test_identify_enabled_data_sources()
        
        # Mapping per processor logic
        source_to_schema = {
            'api': 'api',
            'database': 'database',
            'search_databases': 'database',
            'json_files': 'knowledge_graph',
            'pdf_documents': 'knowledge_graph',
            'vector_search': 'knowledge_graph'
        }
        
        required_schemas = set()
        source_schema_mapping = {}
        
        for source in enabled_sources:
            schema_type = source_to_schema.get(source)
            if schema_type:
                required_schemas.add(schema_type)
                source_schema_mapping[source] = schema_type
        
        print(f"\n{'='*80}")
        print(f"DATA SOURCE → SCHEMA MAPPING")
        print(f"{'='*80}")
        print(f"\nRequired Schema Types: {', '.join(sorted(required_schemas))}")
        print(f"\nDetailed Mapping:")
        for source, schema_type in source_schema_mapping.items():
            print(f"  • {source} → {schema_type}")
        
        # Assertions
        self.assertGreater(len(required_schemas), 0, "Should have at least one required schema")
        
        return required_schemas, source_schema_mapping


class TestSchemaAvailability(unittest.TestCase):
    """Test schema file availability in schemas directory."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Get paths from environment.py if available
        if HAS_ENVIRONMENT_MODULE:
            self.config_path = Path(environment.get_config_path())
            self.schemas_dir = Path(environment.get_schemas_dir())
        else:
            self.config_path = Path(__file__).parent.parent / 'config.json'
            self.schemas_dir = Path(__file__).parent.parent / 'schemas'
        
        # Load config
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)
        
        # Get schema paths from config
        self.schema_paths = self.config.get('schemas', {})
        
        # Initialize processor once for all tests in this class
        self.processor = APIOrchestrator(config_path=str(self.config_path))
    
    def test_schemas_directory_exists(self):
        """Test that schemas directory exists."""
        self.assertTrue(
            self.schemas_dir.exists(),
            f"Schemas directory not found at {self.schemas_dir}"
        )
        self.assertTrue(
            self.schemas_dir.is_dir(),
            f"Schemas path exists but is not a directory: {self.schemas_dir}"
        )
        
        print(f"\n✓ Schemas directory exists: {self.schemas_dir}")
    
    def test_required_schema_files_exist(self):
        """Test that schema files for enabled data sources exist."""
        # Get required schemas from processor
        enabled_sources = self.processor._get_enabled_data_sources_list()
        required_schemas = self.processor._get_required_schema_types(enabled_sources)
        
        print(f"\n{'='*80}")
        print(f"SCHEMA FILE AVAILABILITY")
        print(f"{'='*80}")
        
        existing_schemas = {}
        missing_schemas = {}
        
        for schema_type in required_schemas:
            # Get schema path from config
            schema_path = self.schema_paths.get(schema_type)
            
            if schema_path:
                # Resolve path relative to config directory (parent of schemas_dir)
                # schema_path is like "schemas/api_schema.json"
                config_dir = self.config_path.parent
                full_path = config_dir / schema_path
                exists = full_path.exists()
                
                if exists:
                    # Check if it's valid JSON
                    try:
                        with open(full_path, 'r') as f:
                            schema_content = json.load(f)
                        existing_schemas[schema_type] = {
                            'path': str(schema_path),
                            'exists': True,
                            'valid': True,
                            'type': schema_content.get('type', 'N/A')
                        }
                    except json.JSONDecodeError as e:
                        existing_schemas[schema_type] = {
                            'path': str(schema_path),
                            'exists': True,
                            'valid': False,
                            'error': str(e)
                        }
                else:
                    missing_schemas[schema_type] = {
                        'path': str(schema_path),
                        'exists': False
                    }
            else:
                missing_schemas[schema_type] = {
                    'path': None,
                    'exists': False,
                    'error': 'No path defined in config.json'
                }
        
        # Display results
        if existing_schemas:
            print(f"\n✓ Available Schemas ({len(existing_schemas)}):")
            for schema_type, info in existing_schemas.items():
                status = "✓ Valid" if info.get('valid') else "✗ Invalid"
                print(f"  • {schema_type}")
                print(f"    - Path: {info['path']}")
                print(f"    - Status: {status}")
                if info.get('valid'):
                    print(f"    - Type field: {info.get('type')}")
                else:
                    print(f"    - Error: {info.get('error', 'Unknown')}")
        
        if missing_schemas:
            print(f"\n✗ Missing Schemas ({len(missing_schemas)}):")
            for schema_type, info in missing_schemas.items():
                print(f"  • {schema_type}")
                if info.get('path'):
                    print(f"    - Expected path: {info['path']}")
                else:
                    print(f"    - Error: {info.get('error', 'Path not configured')}")
        
        # Assertions
        for schema_type in required_schemas:
            self.assertIn(
                schema_type,
                existing_schemas,
                f"Required schema '{schema_type}' should exist in schemas directory"
            )
            if schema_type in existing_schemas:
                self.assertTrue(
                    existing_schemas[schema_type]['valid'],
                    f"Schema '{schema_type}' should be valid JSON"
                )
        
        return existing_schemas, missing_schemas
    
    def test_schema_data_references_in_config(self):
        """Test that schemas reference the correct data sources from config."""
        # Get schema paths from config
        schema_paths = self.config.get('schemas', {})
        data_sources = self.config.get('data_sources', {})
        
        print(f"\n{'='*80}")
        print(f"SCHEMA DATA REFERENCES")
        print(f"{'='*80}")
        
        # For each schema type, check what data it should reference
        schema_data_refs = {}
        
        # API schema should reference API data sources
        if 'api' in schema_paths:
            api_source = data_sources.get('api', {})
            if isinstance(api_source, dict) and api_source.get('enabled'):
                schema_data_refs['api'] = {
                    'data_source': 'api',
                    'reference': api_source.get('endpoints_spec', 'API specification')
                }
        
        # Database schema should reference database sources
        db_sources = []
        
        # Check if database source exists and is enabled
        database_source = data_sources.get('database', {})
        if isinstance(database_source, dict) and database_source.get('enabled'):
            db_sources.append('database')
        
        # Check search_databases (can be list or dict)
        search_dbs = data_sources.get('search_databases', [])
        if isinstance(search_dbs, list):
            # Count enabled items in list
            for db in search_dbs:
                if isinstance(db, dict) and db.get('enabled'):
                    db_sources.append(f"search_databases[{db.get('name', 'unnamed')}]")
        elif isinstance(search_dbs, dict) and search_dbs.get('enabled'):
            db_sources.append('search_databases')
        
        if db_sources and 'database' in schema_paths:
            schema_data_refs['database'] = {
                'data_sources': db_sources,
                'reference': 'Database connections'
            }
        
        # Knowledge graph schema should reference document sources
        kg_sources = []
        
        json_files = data_sources.get('json_files', {})
        if isinstance(json_files, dict) and json_files.get('enabled'):
            kg_sources.append('json_files')
        
        pdf_docs = data_sources.get('pdf_documents', {})
        if isinstance(pdf_docs, dict) and pdf_docs.get('enabled'):
            kg_sources.append('pdf_documents')
        
        vector_search = data_sources.get('vector_search', {})
        if isinstance(vector_search, dict) and vector_search.get('enabled'):
            kg_sources.append('vector_search')
        
        if kg_sources and 'knowledge_graph' in schema_paths:
            schema_data_refs['knowledge_graph'] = {
                'data_sources': kg_sources,
                'reference': 'Document knowledge graphs'
            }
        
        # Display
        print(f"\nSchema → Data Source References:")
        for schema_type, refs in schema_data_refs.items():
            print(f"\n  • {schema_type}")
            if 'data_source' in refs:
                print(f"    - Data Source: {refs['data_source']}")
                print(f"    - Reference: {refs['reference']}")
            elif 'data_sources' in refs:
                print(f"    - Data Sources: {', '.join(refs['data_sources'])}")
                print(f"    - Reference: {refs['reference']}")
        
        if not schema_data_refs:
            print(f"\n  (No schema data references found for enabled sources)")
        
        self.assertIsInstance(schema_data_refs, dict)
        return schema_data_refs


class TestSchemaGeneration(unittest.TestCase):
    """Test schema generation for enabled data sources."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Get paths from environment.py if available
        if HAS_ENVIRONMENT_MODULE:
            self.config_path = Path(environment.get_config_path())
            self.schemas_dir = Path(environment.get_schemas_dir())
        else:
            self.config_path = Path(__file__).parent.parent / 'config.json'
            self.schemas_dir = Path(__file__).parent.parent / 'schemas'
        
        self.api_spec_path = Path(__file__).parent.parent / 'src' / 'enable_ai' / 'api_spec.json'
        
        # Load config
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)
        
        # Initialize processor once for all tests in this class
        self.processor = APIOrchestrator(config_path=str(self.config_path))
    
    def test_api_schema_generation_available(self):
        """Test that API schema can be generated from api_spec.json."""
        data_sources = self.config.get('data_sources', {})
        api_source = data_sources.get('api', {})
        
        if not api_source.get('enabled'):
            self.skipTest("API data source is not enabled")
        
        print(f"\n{'='*80}")
        print(f"API SCHEMA GENERATION CHECK")
        print(f"{'='*80}")
        
        # Check if api_spec.json exists
        api_spec_exists = self.api_spec_path.exists()
        
        print(f"\n✓ API Specification File:")
        print(f"  - Path: {self.api_spec_path}")
        print(f"  - Exists: {api_spec_exists}")
        
        if api_spec_exists:
            # Check if it's valid JSON
            try:
                with open(self.api_spec_path, 'r') as f:
                    api_spec = json.load(f)
                
                print(f"  - Valid: True")
                print(f"  - API Name: {api_spec.get('api_name', 'N/A')}")
                print(f"  - Version: {api_spec.get('version', 'N/A')}")
                print(f"  - Modules: {len(api_spec.get('modules', {}))}")
                
                # Check if schema generator script exists
                generator_script = Path(__file__).parent.parent / 'generate_schema.py'
                print(f"\n✓ Schema Generator Script:")
                print(f"  - Path: {generator_script}")
                print(f"  - Exists: {generator_script.exists()}")
                
                self.assertTrue(api_spec_exists, "api_spec.json should exist")
                
            except json.JSONDecodeError as e:
                print(f"  - Valid: False")
                print(f"  - Error: {e}")
                self.fail(f"api_spec.json is not valid JSON: {e}")
    
    def test_generate_missing_schemas(self):
        """Test that missing schemas can be generated automatically."""
        import subprocess
        import sys
        
        enabled_sources = self.processor._get_enabled_data_sources_list()
        required_schemas = self.processor._get_required_schema_types(enabled_sources)
        
        # Check which schemas are missing
        schema_paths = self.config.get('schemas', {})
        missing_schemas = []
        
        for schema_type in required_schemas:
            schema_path = schema_paths.get(schema_type)
            if schema_path:
                full_path = self.schemas_dir.parent / schema_path
                if not full_path.exists():
                    missing_schemas.append(schema_type)
        
        print(f"\n{'='*80}")
        print(f"SCHEMA GENERATION")
        print(f"{'='*80}")
        print(f"\nRequired schemas: {', '.join(sorted(required_schemas))}")
        
        if missing_schemas:
            print(f"Missing schemas: {', '.join(missing_schemas)}")
            
            # Only generate API schema for now (if missing)
            if 'api' in missing_schemas:
                generator_script = Path(__file__).parent.parent / 'generate_schema.py'
                
                if generator_script.exists():
                    print(f"\n⚙️  Generating API schema...")
                    
                    try:
                        # Run the generator script
                        result = subprocess.run(
                            [sys.executable, str(generator_script)],
                            cwd=str(generator_script.parent),
                            capture_output=True,
                            text=True,
                            timeout=30
                        )
                        
                        if result.returncode == 0:
                            print(f"✓ Schema generation successful")
                            print(result.stdout)
                            
                            # Verify the schema was created
                            api_schema_path = self.schemas_dir / 'api_schema.json'
                            self.assertTrue(
                                api_schema_path.exists(),
                                "API schema should exist after generation"
                            )
                            
                            # Validate it's valid JSON
                            with open(api_schema_path, 'r') as f:
                                schema = json.load(f)
                            
                            self.assertIn('type', schema, "Generated schema should have 'type' field")
                            self.assertIn('resources', schema, "Generated schema should have 'resources' field")
                            
                            print(f"✓ Generated schema validated")
                            print(f"  - Type: {schema.get('type')}")
                            print(f"  - Resources: {len(schema.get('resources', {}))}")
                        else:
                            print(f"✗ Schema generation failed")
                            print(f"Error: {result.stderr}")
                            self.fail(f"Schema generation failed: {result.stderr}")
                    
                    except subprocess.TimeoutExpired:
                        self.fail("Schema generation timed out")
                    except Exception as e:
                        self.fail(f"Schema generation error: {e}")
                else:
                    print(f"⚠️  Generator script not found: {generator_script}")
                    self.skipTest("Generator script not available")
        else:
            print(f"✓ All required schemas already exist")
            print(f"  No generation needed")
    
    def test_schema_files_match_enabled_sources(self):
        """Test that schema files exist for all enabled data sources."""
        enabled_sources = self.processor._get_enabled_data_sources_list()
        required_schemas = self.processor._get_required_schema_types(enabled_sources)
        loaded_schemas = set(self.processor.schemas.keys())
        
        print(f"\n{'='*80}")
        print(f"SCHEMA LOADING VERIFICATION")
        print(f"{'='*80}")
        print(f"\nEnabled Data Sources: {', '.join(enabled_sources)}")
        print(f"Required Schema Types: {', '.join(sorted(required_schemas))}")
        print(f"Loaded Schema Types: {', '.join(sorted(loaded_schemas))}")
        
        missing = required_schemas - loaded_schemas
        extra = loaded_schemas - required_schemas
        
        if missing:
            print(f"\n✗ Missing Schemas: {', '.join(missing)}")
        else:
            print(f"\n✓ All required schemas loaded successfully")
        
        if extra:
            print(f"\nℹ️ Extra Schemas (not required by enabled sources): {', '.join(extra)}")
        
        # Assertions
        for schema_type in required_schemas:
            self.assertIn(
                schema_type,
                loaded_schemas,
                f"Required schema '{schema_type}' should be loaded"
            )


def run_tests():
    """Run all tests and display results."""
    print("\n" + "="*80)
    print("TESTING: ENABLED DATA SOURCES AND SCHEMA AVAILABILITY")
    print("="*80)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes in order
    suite.addTests(loader.loadTestsFromTestCase(TestEnabledDataSources))
    suite.addTests(loader.loadTestsFromTestCase(TestSchemaAvailability))
    suite.addTests(loader.loadTestsFromTestCase(TestSchemaGeneration))
    
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
    success = run_tests()
    sys.exit(0 if success else 1)
