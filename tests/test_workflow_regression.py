"""Regression tests for 0.3.82 outage (missing merge_execution_context import)."""

import importlib


def test_workflow_imports_merge_execution_context():
    workflow = importlib.import_module("enable_ai.workflow")
    assert hasattr(workflow, "merge_execution_context")


def test_merge_execution_context_callable_from_workflow_module():
    from enable_ai.query_execution import merge_execution_context
    from enable_ai.workflow import merge_execution_context as wf_merge

    assert merge_execution_context is wf_merge
    step = merge_execution_context({"resource": "service-orders"}, {"limit": 1})
    assert step.get("limit") == 1
