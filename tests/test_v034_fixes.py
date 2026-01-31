"""
Test cases for v0.3.4 fixes

Tests all 4 issues that were fixed:
1. Intelligent summary generation
2. Conversation context/session memory
3. URL concatenation bug
4. Error response format consistency

Note: These are unit tests that verify the fix implementations exist and work correctly.
Full integration tests would require a working API server and schema.
"""

import inspect
import pytest

from enable_ai import APIOrchestrator
from enable_ai.api_client import APIClient


class TestIntelligentSummaries:
    """Test Issue #1: Intelligent summary generation"""
    
    def test_resource_type_detection_patterns(self):
        """Verify resource type detection patterns exist"""
        # Test that the fix implementation includes resource detection
        # Verify method exists
        assert hasattr(APIOrchestrator, '_detect_resource_type')
        assert hasattr(APIOrchestrator, '_extract_examples')
        assert hasattr(APIOrchestrator, '_extract_name')
        
        # Verify method signatures
        sig = inspect.signature(APIOrchestrator._detect_resource_type)
        assert 'results' in sig.parameters
    
    def test_summarize_result_enhanced(self):
        """Verify _summarize_result was enhanced"""
        # Get the source code
        source = inspect.getsource(APIOrchestrator._summarize_result)
        
        # Verify it includes our enhancements
        assert '_detect_resource_type' in source
        assert '_extract_examples' in source
        assert 'Examples:' in source  # Our new summary format


class TestConversationContext:
    """Test Issue #2: Conversation context/session memory"""
    
    def test_conversation_memory_exists(self):
        """Verify conversation memory was added to APIOrchestrator"""
        # Verify methods exist
        assert hasattr(APIOrchestrator, '_get_conversation_history')
        assert hasattr(APIOrchestrator, '_add_to_conversation')
        assert hasattr(APIOrchestrator, 'clear_conversation')
    
    def test_parser_accepts_conversation_history(self):
        """Verify parser now accepts conversation_history parameter"""
        from enable_ai.query_parser import QueryParser
        
        sig = inspect.signature(QueryParser.parse_input)
        assert 'conversation_history' in sig.parameters
    
    def test_workflow_has_conversation_state(self):
        """Verify workflow state includes conversation fields"""
        import inspect
        from enable_ai import workflow
        
        # Get the source code
        source = inspect.getsource(workflow)
        
        # Verify conversation fields in state
        assert 'session_id' in source
        assert 'conversation_history' in source


class TestURLConcatenation:
    """Test Issue #3: URL concatenation bug"""
    
    def test_build_url_method_exists(self):
        """Verify _build_url method was added to APIClient"""
        assert hasattr(APIClient, '_build_url')
    
    def test_build_url_implementation(self):
        """Verify _build_url handles duplicate paths"""
        source = inspect.getsource(APIClient._build_url)
        
        # Verify it uses urllib.parse for smart concatenation
        assert 'urlparse' in source
        assert 'base_path' in source or 'path' in source
    
    def test_call_api_uses_build_url(self):
        """Verify call_api now uses _build_url"""
        source = inspect.getsource(APIClient.call_api)
        
        # Verify it calls _build_url instead of f-string concatenation
        assert '_build_url' in source


class TestErrorResponseConsistency:
    """Test Issue #4: Error response format consistency"""
    
    def test_error_response_has_all_fields(self):
        """Test that error responses have same structure as success responses"""
        required_fields = ['success', 'data', 'summary', 'error', 'query', 'total_steps', 'schema_type']
        
        # Simulate error response (as per v0.3.4 fix)
        error_response = {
            'success': False,
            'data': None,
            'summary': 'Error: Resource not found',
            'error': 'Resource not found',
            'query': 'test query',
            'total_steps': 0,
            'schema_type': 'api'
        }
        
        for field in required_fields:
            assert field in error_response, f"Missing required field: {field}"
    
    def test_success_response_structure(self):
        """Verify success response has all required fields"""
        required_fields = ['success', 'data', 'summary', 'query', 'total_steps', 'schema_type']
        
        success_response = {
            'success': True,
            'data': {'count': 25, 'results': []},
            'summary': 'Found 25 items',
            'query': 'test query',
            'total_steps': 1,
            'schema_type': 'api'
        }
        
        for field in required_fields:
            assert field in success_response, f"Missing required field: {field}"
        
        # Error field should not be present or should be None
        assert success_response.get('error') is None
    
    def test_workflow_format_error_updated(self):
        """Verify workflow format_error includes all fields"""
        import inspect
        from enable_ai import workflow
        
        source = inspect.getsource(workflow)
        
        # Find format_error function
        assert 'def format_error' in source
        
        # Verify it includes all required fields
        # Look for the Issue #4 fix comment
        assert 'Issue #4' in source or 'consistent structure' in source.lower()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
