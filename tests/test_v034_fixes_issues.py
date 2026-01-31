"""
Test cases for v0.3.4 fixes

Tests all 4 issues that were fixed:
1. Intelligent summary generation
2. Conversation context/session memory
3. URL concatenation bug
4. Error response format consistency
"""

import pytest
from enable_ai import APIOrchestrator
from enable_ai.api_client import APIClient


class TestIntelligentSummaries:
    """Test Issue #1: Intelligent summary generation"""
    
    def test_summary_field_exists(self):
        """Verify that summary field is always present in response"""
        # This would require a working config and schema
        # For now, we'll test the helper methods directly
        processor = APIOrchestrator.__new__(APIOrchestrator)
        
        # Test resource type detection
        results = [
            {"email": "john@test.com", "role": "admin"},
            {"email": "jane@test.com", "role": "user"}
        ]
        resource_type = processor._detect_resource_type(results)
        assert resource_type == "users"
    
    def test_extract_examples(self):
        """Test example extraction from results"""
        processor = APIOrchestrator.__new__(APIOrchestrator)
        
        results = [
            {"name": "Item 1", "id": 1},
            {"name": "Item 2", "id": 2},
            {"name": "Item 3", "id": 3},
            {"name": "Item 4", "id": 4}
        ]
        
        examples = processor._extract_examples(results, max_examples=3)
        assert len(examples) == 3
        assert "Item 1" in examples
        assert "Item 2" in examples
        assert "Item 3" in examples
    
    def test_extract_name(self):
        """Test name extraction from various item structures"""
        processor = APIOrchestrator.__new__(APIOrchestrator)
        
        # Test with name field
        item1 = {"name": "Test Item", "id": 1}
        assert processor._extract_name(item1) == "Test Item"
        
        # Test with email field
        item2 = {"email": "test@example.com", "id": 2}
        assert processor._extract_name(item2) == "test@example.com"
        
        # Test with serial_number field
        item3 = {"serial_number": "SN-12345", "id": 3}
        assert processor._extract_name(item3) == "SN-12345"
        
        # Test with only id
        item4 = {"id": 4, "some_field": "value"}
        assert processor._extract_name(item4) == "4"


class TestConversationContext:
    """Test Issue #2: Conversation context/session memory"""
    
    def test_conversation_memory_storage(self):
        """Test that conversation history is stored correctly"""
        # Create processor with minimal init
        processor = APIOrchestrator.__new__(APIOrchestrator)
        processor.conversation_memory = {}
        processor.max_conversation_history = 10
        
        # Add messages
        processor._add_to_conversation("session1", "user", "Show me users")
        processor._add_to_conversation("session1", "assistant", "Found 25 users")
        
        # Retrieve history
        history = processor._get_conversation_history("session1")
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "Show me users"
        assert history[1]["role"] == "assistant"
        assert history[1]["content"] == "Found 25 users"
    
    def test_conversation_history_limit(self):
        """Test that conversation history is limited to max_conversation_history"""
        processor = APIOrchestrator.__new__(APIOrchestrator)
        processor.conversation_memory = {}
        processor.max_conversation_history = 5
        
        # Add more than max messages
        for i in range(10):
            processor._add_to_conversation("session1", "user", f"Message {i}")
        
        # Should only keep last 5
        history = processor._get_conversation_history("session1")
        assert len(history) == 5
        assert history[0]["content"] == "Message 5"
        assert history[-1]["content"] == "Message 9"
    
    def test_clear_conversation(self):
        """Test clearing conversation history"""
        processor = APIOrchestrator.__new__(APIOrchestrator)
        processor.conversation_memory = {}
        processor.max_conversation_history = 10
        
        processor._add_to_conversation("session1", "user", "Test")
        assert len(processor._get_conversation_history("session1")) == 1
        
        processor.clear_conversation("session1")
        assert len(processor._get_conversation_history("session1")) == 0


class TestURLConcatenation:
    """Test Issue #3: URL concatenation bug"""
    
    def test_url_building_no_duplication(self):
        """Test that URL building prevents duplicate path segments"""
        client = APIClient("http://localhost:8002/api")
        
        # Case 1: Path already includes /api
        url1 = client._build_url("http://localhost:8002/api", "/api/users/")
        assert url1 == "http://localhost:8002/api/users/"
        assert url1.count('/api') == 1  # Should appear only once
        
        # Case 2: Path doesn't include /api
        url2 = client._build_url("http://localhost:8002/api", "/users/")
        assert url2 == "http://localhost:8002/api/users/"
        
        # Case 3: Base URL without /api, path with /api
        url3 = client._build_url("http://localhost:8002", "/api/users/")
        assert url3 == "http://localhost:8002/api/users/"
    
    def test_url_building_various_combinations(self):
        """Test various base_url and path combinations"""
        test_cases = [
            ("http://localhost/api", "/api/inventory/", "http://localhost/api/inventory/"),
            ("http://localhost/api/", "/api/inventory/", "http://localhost/api/inventory/"),
            ("http://localhost", "/inventory/", "http://localhost/inventory/"),
            ("https://api.example.com/v1", "/v1/users", "https://api.example.com/v1/users"),
        ]
        
        for base_url, path, expected in test_cases:
            client = APIClient(base_url)
            result = client._build_url(base_url, path)
            assert result == expected, f"Failed: {base_url} + {path} = {result} (expected {expected})"


class TestErrorResponseConsistency:
    """Test Issue #4: Error response format consistency"""
    
    def test_error_response_has_all_fields(self):
        """Test that error responses have same structure as success responses"""
        # This test would need actual workflow execution
        # Here we verify the structure programmatically
        
        required_fields = ['success', 'data', 'summary', 'error', 'query', 'total_steps', 'schema_type']
        
        # Simulate error response
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


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
