"""
Split grouped OpenAPI resources into individual resources for APIMatcher.

OpenAPI conversion groups endpoints by first path segment (e.g. master-data, documents).
Consumer config resource_hints often use individual names (companies, flash-reports).
"""

import copy
from typing import Any, Dict, List, Optional

from .hint_utils import get_aggregate_resources
from .utils import setup_logger

logger = setup_logger("enable_ai.schema_splitter")

# Default rules for common API layouts (overridable via schema.resource_split_rules)
DEFAULT_SPLIT_RULES: Dict[str, Dict[str, str]] = {
    "inventory": {
        "inventory-equipment": "/equipment",
        "inventory-consumables": "/consumables",
    },
    "master-data": {
        "companies": "/companies",
        "locations": "/locations",
        "service-types": "/service-types",
        "service-categories": "/service-categories",
        "service-rate-cards": "/service-rate-cards",
        "skills": "/skills",
    },
    "documents": {
        "components": "/components",
        "machines": "/machines",
        "products": "/products",
        "document-types": "/document-types",
        "documents": "/documents/documents",
    },
    "service-orders": {
        "flash-reports": "/flash-reports",
        "details-reports": "/details-reports",
    },
    "invoicing": {
        "invoicing-invoices": "/invoices",
        "invoicing-ar-summary": "/ar-dashboard",
    },
}


def _path_matches(path: str, pattern: str) -> bool:
    return pattern.lower() in (path or "").lower()


def _build_auto_split_rules(
    resources: Dict[str, Any],
    resource_hints: Dict[str, Any],
) -> Dict[str, Dict[str, str]]:
    """
    Auto-detect split rules when resource_hints reference names missing from resources.
    """
    auto_rules: Dict[str, Dict[str, str]] = {}
    missing_hints = [
        name for name in resource_hints
        if name not in resources and isinstance(resource_hints.get(name), dict)
    ]

    for hint_name in missing_hints:
        hint_norm = hint_name.replace("_", "-")
        parent_prefix = hint_norm.split("-")[0] if "-" in hint_norm else hint_norm
        patterns = [f"/{hint_norm}"]
        # invoicing-invoices → /invoices under parent invoicing
        for parent_name, parent_data in resources.items():
            parent_norm = parent_name.replace("_", "-")
            if hint_norm.startswith(parent_norm + "-"):
                suffix = hint_norm[len(parent_norm) + 1:]
                if suffix:
                    patterns.insert(0, f"/{suffix}")
            elif parent_norm == parent_prefix:
                remainder = hint_norm[len(parent_norm) + 1:] if hint_norm.startswith(parent_norm + "-") else ""
                if remainder:
                    patterns.insert(0, f"/{remainder}")
        for parent_name, parent_data in resources.items():
            endpoints = parent_data.get("endpoints") or []
            for pattern in patterns:
                if any(_path_matches(ep.get("path", ""), pattern) for ep in endpoints):
                    auto_rules.setdefault(parent_name, {})[hint_name] = pattern
                    break
            if hint_name in auto_rules.get(parent_name, {}):
                break

    return auto_rules


def split_grouped_resources(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Split grouped resources into individual sub-resources.

    Uses schema.resource_split_rules when present, else DEFAULT_SPLIT_RULES,
    plus auto-detection from resource_hints.
    """
    if not schema or not schema.get("resources"):
        return schema

    schema = copy.deepcopy(schema)
    resources = schema["resources"]
    resource_hints = schema.get("resource_hints") or {}

    explicit_rules = schema.get("resource_split_rules") or {}
    split_rules = {**DEFAULT_SPLIT_RULES, **explicit_rules}
    auto_rules = _build_auto_split_rules(resources, resource_hints)

    for parent_name, sub_rules in auto_rules.items():
        merged = dict(split_rules.get(parent_name, {}))
        merged.update(sub_rules)
        split_rules[parent_name] = merged

    for parent_name, sub_resources in split_rules.items():
        parent = resources.get(parent_name)
        if not parent or not parent.get("endpoints"):
            continue

        parent_fields = parent.get("fields", [])
        parent_display = parent.get("display_field")
        created: List[str] = []

        for sub_name, path_pattern in sub_resources.items():
            if sub_name in resources and sub_name != parent_name:
                continue

            matching = [
                ep for ep in parent.get("endpoints", [])
                if _path_matches(ep.get("path", ""), path_pattern)
            ]
            if not matching:
                continue

            resources[sub_name] = {
                "name": sub_name,
                "description": sub_name.replace("-", " ").title(),
                "endpoints": matching,
                "fields": parent_fields,
                "display_field": parent_display,
            }
            created.append(f"{sub_name}({len(matching)})")

        if not created:
            continue

        matched_paths = set()
        for sub_name in sub_resources:
            if sub_name in resources:
                for ep in resources[sub_name].get("endpoints", []):
                    matched_paths.add(ep.get("path"))

        remaining = [
            ep for ep in parent.get("endpoints", [])
            if ep.get("path") not in matched_paths
        ]
        parent_is_also_sub = parent_name in sub_resources

        if remaining and not parent_is_also_sub:
            resources[parent_name]["endpoints"] = remaining
            logger.info(
                "Split '%s' into %s (kept %d endpoints in parent)",
                parent_name, ", ".join(created), len(remaining),
            )
        elif not remaining and not parent_is_also_sub:
            del resources[parent_name]
            logger.info("Split '%s' into %s", parent_name, ", ".join(created))
        else:
            logger.info(
                "Split '%s' into %s (parent preserved — sub shares parent name)",
                parent_name, ", ".join(created),
            )

    schema["resources"] = resources
    return _strip_virtual_aggregate_resources(schema)


def _strip_virtual_aggregate_resources(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove virtual aggregate parents from matchable resources.

    Parents with __aggregate_resources__ (e.g. inventory) may keep utility
    endpoints like /low-stock/ after split — they must not win over children.
    """
    if not schema or not schema.get("resources"):
        return schema

    resources = schema["resources"]
    hints = schema.get("resource_hints") or {}
    removed: List[str] = []

    for name, hint_data in hints.items():
        if not isinstance(hint_data, dict):
            continue
        if not get_aggregate_resources(hint_data):
            continue
        if name in resources:
            del resources[name]
            removed.append(name)

    if removed:
        logger.info(
            "Stripped virtual aggregate resources from matcher: %s",
            ", ".join(removed),
        )
    schema["resources"] = resources
    return schema
