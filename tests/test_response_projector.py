from enable_ai.follow_up_detection import build_session_metadata, try_advance_chat_window
from enable_ai.response_projector import (
    ResponseProjector,
    apply_chat_window,
    format_projected_table,
    project_row,
)


def test_project_row_with_nested_role():
    item = {
        "id": 1,
        "username": "tech@example.com",
        "role": {"name": "Technician"},
        "is_active": True,
    }
    fields = ["username", "role.name", "is_active"]
    projected = project_row(item, fields)
    assert projected["username"] == "tech@example.com"
    assert projected["role.name"] == "Technician"
    assert projected["is_active"] is True


def test_apply_chat_window():
    items = [{"id": i} for i in range(25)]
    window, offset, has_more = apply_chat_window(items, 0, 10)
    assert len(window) == 10
    assert offset == 0
    assert has_more is True

    window2, offset2, has_more2 = apply_chat_window(items, 10, 10)
    assert len(window2) == 10
    assert offset2 == 10
    assert has_more2 is True


def test_response_projector_prepare_list_display():
    schema = {
        "resource_hints": {
            "service-orders": {
                "__list_display_fields__": ["number", "status__name"],
                "__chat_window_size__": 5,
            }
        }
    }
    data = {
        "count": 12,
        "results": [
            {"id": i, "number": f"SO-{i}", "status__name": "New"}
            for i in range(12)
        ],
    }
    projector = ResponseProjector(schema)
    result = projector.prepare_list_display(data, "service-orders", display_mode="summary")
    assert len(result["window_items"]) == 5
    assert len(result["list_cache"]) == 12
    assert result["has_more_in_chat"] is True
    assert result["list_display_fields"] == ["number", "status__name"]


def test_format_projected_table():
    rows = [
        {"number": "SO-1", "status__name": "New"},
        {"number": "SO-2", "status__name": "Closed"},
    ]
    table = format_projected_table(rows, ["number", "status__name"], resource="service-orders")
    assert "SO-1" in table
    assert "Number" in table or "number" in table.lower()


def test_build_session_metadata_includes_chat_cache():
    schema = {
        "resource_hints": {
            "users": {
                "__list_display_fields__": ["username"],
                "__chat_window_size__": 3,
            }
        }
    }
    projection = {
        "list_cache": [{"username": f"u{i}"} for i in range(5)],
        "chat_offset": 0,
        "chat_window_size": 3,
        "list_display_fields": ["username"],
        "has_more_in_chat": True,
        "total_cached": 5,
    }
    meta = build_session_metadata(
        {"resource": "users", "question_type": "list"},
        {"data": {"results": projection["list_cache"][:3]}, "pagination": {"total_count": 5}},
        schema=schema,
        projection=projection,
    )
    assert len(meta["list_cache"]) == 5
    assert meta["chat_offset"] == 0
    assert meta["has_more_in_chat"] is True


def test_try_advance_chat_window_without_api():
    last_metadata = {
        "resource": "users",
        "list_cache": [{"username": f"u{i}"} for i in range(15)],
        "chat_offset": 0,
        "chat_window_size": 10,
        "list_display_fields": ["username"],
        "count": 15,
        "filters": {},
    }
    advance = try_advance_chat_window(last_metadata, 10)
    assert advance is not None
    assert len(advance["window_items"]) == 5
    assert advance["chat_offset"] == 10
    assert "u10" in advance["summary"] or "u11" in advance["summary"]

    last_metadata["chat_offset"] = 10
    exhausted = try_advance_chat_window(last_metadata, 10)
    assert exhausted is None
