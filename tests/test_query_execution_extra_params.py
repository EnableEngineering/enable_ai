from enable_ai.query_execution import split_filters_for_endpoint


def test_extra_query_params_sent_server_side_without_warning():
    endpoint = {
        "parameters": {
            "query": [
                {"name": "ordering"},
                {"name": "page_size"},
                {"name": "search"},
            ]
        }
    }
    hints = {"users": {"__extra_query_params__": ["role", "is_active"]}}
    filters = {
        "role": {"operator": "equals", "value": "Technician"},
        "search": {"operator": "contains", "value": "tech"},
    }
    server, client, warnings = split_filters_for_endpoint(
        filters, endpoint, resource="users", resource_hints=hints,
    )
    assert "role" in server
    assert "search" in server
    assert not client
    assert not warnings


def test_client_side_filter_fields_no_warning():
    endpoint = {
        "parameters": {
            "query": [{"name": "search"}],
        }
    }
    hints = {"users": {"__client_side_filters__": ["role"]}}
    filters = {"role": {"operator": "equals", "value": "Technician"}}
    server, client, warnings = split_filters_for_endpoint(
        filters, endpoint, resource="users", resource_hints=hints,
    )
    assert "role" in client
    assert not warnings
