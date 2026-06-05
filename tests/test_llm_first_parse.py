from unittest.mock import patch

from enable_ai.query_parser import QueryParser


@patch("enable_ai.query_parser.get_openai_client")
def test_build_prompt_includes_classification_hint(mock_client):
    parser = QueryParser()
    prompt = parser._build_prompt(
        "list new service orders",
        {"type": "api", "resources": {"service-orders": {}}},
        classification_hint={
            "intent": "read",
            "resource": "service-orders",
            "confidence": 0.9,
        },
    )
    assert "CLASSIFICATION_HINT" in prompt
    assert "service-orders" in prompt
    assert "LLM decision wins" in prompt


@patch("enable_ai.query_parser.get_openai_client")
def test_build_prompt_no_preprocess_note(mock_client):
    parser = QueryParser()
    prompt = parser._build_prompt("show low stock items", {"type": "api", "resources": {}})
    assert "pre-processed" not in prompt.lower()
