"""
Attach workflow debug/inspection fields to the public API response.
"""

from typing import Any, Dict, List, Optional


def _public_parsed(parsed: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Strip internal keys (prefixed with _) from parsed query."""
    if not parsed or not isinstance(parsed, dict):
        return None
    return {k: v for k, v in parsed.items() if not str(k).startswith("_")}


def enrich_api_response(
    response: Dict[str, Any],
    state: Dict[str, Any],
) -> Dict[str, Any]:
    """Merge parsed, step_results, filter_warnings, and api_call into response."""
    out = dict(response)

    parsed = _public_parsed(state.get("parsed"))
    if parsed is not None:
        out["parsed"] = parsed

    step_results = state.get("step_results")
    if step_results:
        out["step_results"] = step_results

    filter_warnings = state.get("filter_warnings") or out.get("filter_warnings")
    if filter_warnings:
        out["filter_warnings"] = filter_warnings

    exec_result = state.get("result") or {}
    if isinstance(exec_result, dict) and exec_result.get("endpoint"):
        out["api_call"] = {
            "endpoint": exec_result.get("endpoint"),
            "method": exec_result.get("method"),
            "params": exec_result.get("params"),
        }

    return out
