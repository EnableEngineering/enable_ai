from enable_ai.semantic_filters import inject_semantic_filters

HINTS = {
    "inventory-consumables": {
        "stock_level": {"synonyms": {"low in stock": "low", "low stock": "low"}},
    },
    "users": {"role": {"synonyms": {"customer": "Client", "customer type": "Client"}}},
}


def test_low_stock_injection():
    filters = inject_semantic_filters(
        {}, "inventory-consumables", "items low in stock", HINTS,
    )
    assert filters["stock_level"]["value"] == "low"


def test_role_injection():
    filters = inject_semantic_filters(
        {}, "users", "show customer type users", HINTS,
    )
    assert filters["role"]["value"] == "Client"


def test_no_hints_no_injection():
    filters = inject_semantic_filters({}, "users", "low stock items", {})
    assert filters == {}


def test_idempotent():
    once = inject_semantic_filters({}, "users", "show customers", HINTS)
    twice = inject_semantic_filters(once, "users", "show customers", HINTS)
    assert once == twice


def test_new_status_word_boundary():
    """Single-token synonym 'new' must not match inside other words."""
    hints = {
        "service-orders": {
            "status__name": {"synonyms": {"new": "New", "quoted": "Quoted"}},
        },
    }
    filters = inject_semantic_filters(
        {}, "service-orders", "list new service orders", hints,
    )
    assert filters["status__name"]["value"] == "New"

    # 'renew' should not trigger 'new'
    no_match = inject_semantic_filters(
        {}, "service-orders", "renew service orders", hints,
    )
    assert "status__name" not in no_match

    # plural form of synonym value
    plural = inject_semantic_filters(
        {}, "users", "which technicians are available", {
            "users": {"role": {"synonyms": {"technician": "Technician"}}},
        },
    )
    assert plural["role"]["value"] == "Technician"
