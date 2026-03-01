"""
Tracing module for enable-ai debugging and vetting.

Stores all processing steps to a log file for later analysis.
This helps identify exactly where queries fail in the pipeline.

Usage:
    tracer = QueryTracer("show me users", session_id="123")
    tracer.log_stage("parse", {"intent": "read", "resource": "users"})
    tracer.log_stage("api_call", {"endpoint": "/users/", "params": {}})
    TracingStore().save_trace(tracer)
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from .utils import setup_logger

logger = setup_logger('enable_ai.tracing')


class QueryTracer:
    """Trace a single query through all processing stages."""

    def __init__(self, query: str, session_id: str = None):
        """
        Initialize tracer for a query.

        Args:
            query: The user's natural language query
            session_id: Optional session identifier for multi-turn context
        """
        self.query = query
        self.session_id = session_id
        self.trace_id = datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')
        self.stages: List[Dict[str, Any]] = []
        self.start_time = datetime.utcnow()
        self.warnings: List[str] = []
        self.errors: List[str] = []

    def log_stage(
        self,
        stage: str,
        data: Dict[str, Any],
        success: bool = True,
        error: str = None
    ):
        """
        Log a processing stage.

        Args:
            stage: Stage name (parse, plan, api_call, api_response, summarize)
            data: Stage-specific data to log
            success: Whether this stage succeeded
            error: Error message if failed
        """
        entry = {
            'stage': stage,
            'timestamp': datetime.utcnow().isoformat(),
            'elapsed_ms': (datetime.utcnow() - self.start_time).total_seconds() * 1000,
            'success': success,
            'data': self._sanitize_data(data),
        }

        if error:
            entry['error'] = error
            self.errors.append(f"{stage}: {error}")

        self.stages.append(entry)
        logger.debug(f"Trace [{self.trace_id}] {stage}: success={success}")

    def add_warning(self, warning: str):
        """Add a warning message to the trace."""
        self.warnings.append(warning)
        logger.warning(f"Trace [{self.trace_id}] Warning: {warning}")

    def _sanitize_data(self, data: Any, max_depth: int = 3) -> Any:
        """Sanitize data for logging (truncate large values, remove sensitive data)."""
        if max_depth <= 0:
            return "<truncated>"

        if isinstance(data, dict):
            sanitized = {}
            for key, value in data.items():
                # Skip sensitive fields
                if key.lower() in ('password', 'token', 'secret', 'api_key', 'access_token'):
                    sanitized[key] = '<redacted>'
                elif key == 'results' and isinstance(value, list) and len(value) > 5:
                    # Truncate large result lists
                    sanitized[key] = f"[{len(value)} items - showing first 3]"
                    sanitized[f'{key}_sample'] = [self._sanitize_data(v, max_depth - 1) for v in value[:3]]
                else:
                    sanitized[key] = self._sanitize_data(value, max_depth - 1)
            return sanitized

        if isinstance(data, list):
            if len(data) > 10:
                return f"[{len(data)} items]"
            return [self._sanitize_data(item, max_depth - 1) for item in data]

        if isinstance(data, str) and len(data) > 500:
            return data[:500] + "..."

        return data

    def to_dict(self) -> Dict[str, Any]:
        """Convert trace to dictionary for logging."""
        return {
            'trace_id': self.trace_id,
            'query': self.query,
            'session_id': self.session_id,
            'start_time': self.start_time.isoformat(),
            'total_duration_ms': (datetime.utcnow() - self.start_time).total_seconds() * 1000,
            'stages': self.stages,
            'warnings': self.warnings,
            'errors': self.errors,
            'final_success': len(self.errors) == 0 and all(s.get('success', True) for s in self.stages),
        }

    def get_summary(self) -> str:
        """Get a human-readable summary of the trace."""
        lines = [
            f"Query: {self.query}",
            f"Trace ID: {self.trace_id}",
            f"Duration: {(datetime.utcnow() - self.start_time).total_seconds() * 1000:.1f}ms",
            f"Stages: {len(self.stages)}",
        ]

        for stage in self.stages:
            status = "✅" if stage.get('success') else "❌"
            lines.append(f"  {status} {stage['stage']}: {stage.get('elapsed_ms', 0):.1f}ms")

        if self.warnings:
            lines.append(f"Warnings: {len(self.warnings)}")
            for w in self.warnings:
                lines.append(f"  ⚠️ {w}")

        if self.errors:
            lines.append(f"Errors: {len(self.errors)}")
            for e in self.errors:
                lines.append(f"  ❌ {e}")

        return "\n".join(lines)


class TracingStore:
    """Persist traces to file for later vetting and debugging."""

    def __init__(self, log_dir: str = None, enabled: bool = None):
        """
        Initialize the tracing store.

        Args:
            log_dir: Directory to store trace files. Defaults to ENABLE_AI_TRACE_DIR env var
                     or /tmp/enable_ai_traces
            enabled: Whether tracing is enabled. Defaults to ENABLE_AI_TRACING env var or True
        """
        self.enabled = enabled if enabled is not None else os.getenv('ENABLE_AI_TRACING', 'true').lower() == 'true'

        if self.enabled:
            self.log_dir = Path(log_dir or os.getenv('ENABLE_AI_TRACE_DIR', '/tmp/enable_ai_traces'))
            try:
                self.log_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"TracingStore initialized: {self.log_dir}")
            except Exception as e:
                logger.warning(f"Could not create trace directory: {e}. Tracing disabled.")
                self.enabled = False
        else:
            self.log_dir = None
            logger.info("TracingStore disabled")

    def save_trace(self, tracer: QueryTracer):
        """
        Save trace to file.

        Creates two outputs:
        1. Individual trace file: trace_{trace_id}.json
        2. Appends to daily log: traces_YYYYMMDD.jsonl
        """
        if not self.enabled:
            return

        try:
            trace_data = tracer.to_dict()

            # Individual trace file (for detailed inspection)
            filename = f"trace_{tracer.trace_id}.json"
            filepath = self.log_dir / filename

            with open(filepath, 'w') as f:
                json.dump(trace_data, f, indent=2, default=str)

            # Daily log (for aggregate analysis)
            daily_log = self.log_dir / f"traces_{datetime.utcnow().strftime('%Y%m%d')}.jsonl"
            with open(daily_log, 'a') as f:
                f.write(json.dumps(trace_data, default=str) + '\n')

            logger.debug(f"Saved trace: {filepath}")

        except Exception as e:
            logger.error(f"Failed to save trace: {e}")

    def get_recent_traces(self, limit: int = 50, include_successful: bool = True) -> List[Dict[str, Any]]:
        """
        Get recent traces for debugging.

        Args:
            limit: Maximum number of traces to return
            include_successful: Whether to include successful traces

        Returns:
            List of trace dictionaries, most recent first
        """
        if not self.enabled:
            return []

        try:
            today = datetime.utcnow().strftime('%Y%m%d')
            daily_log = self.log_dir / f"traces_{today}.jsonl"

            if not daily_log.exists():
                return []

            traces = []
            with open(daily_log, 'r') as f:
                for line in f:
                    if line.strip():
                        trace = json.loads(line)
                        if include_successful or not trace.get('final_success', True):
                            traces.append(trace)

            # Return most recent first
            return list(reversed(traces[-limit:]))

        except Exception as e:
            logger.error(f"Failed to read traces: {e}")
            return []

    def get_failed_traces(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent failed traces for debugging."""
        return self.get_recent_traces(limit=limit, include_successful=False)

    def get_trace_by_id(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific trace by ID."""
        if not self.enabled:
            return None

        try:
            filepath = self.log_dir / f"trace_{trace_id}.json"
            if filepath.exists():
                with open(filepath, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Failed to read trace {trace_id}: {e}")

        return None


# Global tracer store instance
_tracer_store: Optional[TracingStore] = None


def get_tracer_store() -> TracingStore:
    """Get the global tracer store instance."""
    global _tracer_store
    if _tracer_store is None:
        _tracer_store = TracingStore()
    return _tracer_store


def create_tracer(query: str, session_id: str = None) -> QueryTracer:
    """Create a new query tracer."""
    return QueryTracer(query, session_id)
