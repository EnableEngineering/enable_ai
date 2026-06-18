"""
Progress tracking for real-time streaming updates.

Enables frontend to show progress during query processing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


class ProgressStage(str, Enum):
    """Progress stages for query processing."""
    STARTED = "started"
    CHECKING_CACHE = "checking_cache"
    CACHE_HIT = "cache_hit"
    CALLING_LLM = "calling_llm"
    CALLING_CLAUDE = "calling_llm"  # deprecated alias — use CALLING_LLM
    TOOLS_SELECTED = "tools_selected"
    VALIDATING = "validating"
    EXECUTING_API = "executing_api"
    API_COMPLETED = "api_completed"
    FORMATTING = "formatting"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class ProgressUpdate:
    """Progress update message."""
    stage: ProgressStage
    message: str
    progress: float = 0.0  # 0.0 to 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "stage": self.stage.value,
            "message": self.message,
            "progress": self.progress,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }


# Default progress percentages per stage
STAGE_PROGRESS = {
    ProgressStage.STARTED: 0.0,
    ProgressStage.CHECKING_CACHE: 0.1,
    ProgressStage.CACHE_HIT: 0.9,
    ProgressStage.CALLING_LLM: 0.2,
    ProgressStage.TOOLS_SELECTED: 0.4,
    ProgressStage.VALIDATING: 0.5,
    ProgressStage.EXECUTING_API: 0.6,
    ProgressStage.API_COMPLETED: 0.8,
    ProgressStage.FORMATTING: 0.9,
    ProgressStage.COMPLETED: 1.0,
    ProgressStage.ERROR: 0.0,
}

# Default messages per stage
STAGE_MESSAGES = {
    ProgressStage.STARTED: "Processing your request...",
    ProgressStage.CHECKING_CACHE: "Checking for cached results...",
    ProgressStage.CACHE_HIT: "Found cached result!",
    ProgressStage.CALLING_LLM: "Understanding your request...",
    ProgressStage.TOOLS_SELECTED: "Identified the right API calls...",
    ProgressStage.VALIDATING: "Validating parameters...",
    ProgressStage.EXECUTING_API: "Calling the API...",
    ProgressStage.API_COMPLETED: "Got the data!",
    ProgressStage.FORMATTING: "Preparing your response...",
    ProgressStage.COMPLETED: "Done!",
    ProgressStage.ERROR: "Something went wrong.",
}


class ProgressTracker:
    """
    Track and stream progress updates during query processing.

    Usage:
        def send_to_frontend(update):
            websocket.send(update.to_dict())

        tracker = ProgressTracker(callback=send_to_frontend)
        tracker.update(ProgressStage.STARTED)
        # ... processing ...
        tracker.update(ProgressStage.COMPLETED)
    """

    def __init__(
        self,
        callback: Optional[Callable[[ProgressUpdate], None]] = None,
        min_display_ms: int = 0,
    ):
        """
        Initialize progress tracker.

        Args:
            callback: Function called with each progress update
            min_display_ms: Minimum time between updates (for UI smoothness)
        """
        self.callback = callback
        self.min_display_ms = min_display_ms
        self.updates: list[ProgressUpdate] = []
        self.start_time = time.time()
        self._last_update_time: float = 0.0
        self._current_stage = ProgressStage.STARTED

    def update(
        self,
        stage: ProgressStage,
        message: Optional[str] = None,
        progress: Optional[float] = None,
        **metadata: Any,
    ) -> None:
        """
        Send a progress update.

        Args:
            stage: Current processing stage
            message: Custom message (uses default if not provided)
            progress: Progress percentage 0.0-1.0 (auto-calculated if not provided)
            **metadata: Additional metadata (e.g., tool_name, endpoint)
        """
        # Enforce minimum display time
        if self.min_display_ms > 0 and self._last_update_time > 0:
            elapsed_ms = (time.time() - self._last_update_time) * 1000
            if elapsed_ms < self.min_display_ms:
                time.sleep((self.min_display_ms - elapsed_ms) / 1000.0)

        # Use defaults if not provided
        if message is None:
            message = STAGE_MESSAGES.get(stage, "Processing...")
        if progress is None:
            progress = STAGE_PROGRESS.get(stage, 0.0)

        update = ProgressUpdate(
            stage=stage,
            message=message,
            progress=progress,
            metadata=metadata,
        )

        self.updates.append(update)
        self._current_stage = stage
        self._last_update_time = time.time()

        # Call callback if provided
        if self.callback:
            try:
                self.callback(update)
            except Exception:
                pass  # Don't let callback errors break processing

    def get_elapsed_time(self) -> float:
        """Get elapsed time in seconds since start."""
        return time.time() - self.start_time

    def get_all_updates(self) -> list[dict[str, Any]]:
        """Get all progress updates as dictionaries."""
        return [u.to_dict() for u in self.updates]

    def get_current_stage(self) -> ProgressStage:
        """Get current processing stage."""
        return self._current_stage
