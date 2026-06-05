from enable_ai.response_envelope import enrich_api_response


def test_enrich_adds_parsed_step_results_and_api_call():
    response = {"success": True, "data": {"results": []}}
    state = {
        "parsed": {
            "intent": "read",
            "resource": "service-orders",
            "filters": {"status__name": {"operator": "equals", "value": "New"}},
            "_internal": "hidden",
        },
        "step_results": [
            {
                "step_id": 1,
                "api_call": {
                    "endpoint": "/api/service-orders/",
                    "method": "GET",
                    "params": {"status__name": "New"},
                },
            }
        ],
        "filter_warnings": ["example warning"],
        "result": {
            "endpoint": "/api/service-orders/",
            "method": "GET",
            "params": {"status__name": "New"},
        },
    }

    out = enrich_api_response(response, state)

    assert out["parsed"]["resource"] == "service-orders"
    assert "_internal" not in out["parsed"]
    assert out["step_results"][0]["api_call"]["params"]["status__name"] == "New"
    assert out["filter_warnings"] == ["example warning"]
    assert out["api_call"]["endpoint"] == "/api/service-orders/"
