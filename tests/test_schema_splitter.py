"""Tests for schema resource splitting."""

from enable_ai.schema_splitter import split_grouped_resources


def test_split_grouped_resources_from_hints():
    schema = {
        "resources": {
            "master-data": {
                "endpoints": [
                    {"path": "/api/master-data/companies/", "method": "GET"},
                    {"path": "/api/master-data/locations/", "method": "GET"},
                ],
                "fields": ["id", "name"],
            }
        },
        "resource_hints": {
            "companies": {"__resource_synonyms__": ["company"]},
        },
    }
    result = split_grouped_resources(schema)
    assert "companies" in result["resources"]
    assert len(result["resources"]["companies"]["endpoints"]) == 1
    assert "companies" in result["resources"]["companies"]["endpoints"][0]["path"]


def test_explicit_split_rules():
    schema = {
        "resource_split_rules": {
            "inventory": {
                "inventory-consumables": "/consumables",
            }
        },
        "resources": {
            "inventory": {
                "endpoints": [
                    {"path": "/api/inventory/consumables/", "method": "GET"},
                    {"path": "/api/inventory/equipment/", "method": "GET"},
                ],
                "fields": ["id"],
            }
        },
    }
    result = split_grouped_resources(schema)
    assert "inventory-consumables" in result["resources"]
    assert "inventory" in result["resources"]  # equipment endpoint remains in parent
