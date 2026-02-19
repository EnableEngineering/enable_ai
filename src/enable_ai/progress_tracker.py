"""
Progress Tracker - Real-time progress updates for frontend streaming

Enables streaming progress updates to frontend:
- Intent detection progress
- API matching status
- API execution steps
- Summarization progress
"""

from typing import Callable, Optional, Dict, Any, List
from enum import Enum
import time
from .utils import setup_logger
from . import constants

logger = setup_logger(__name__)


class ProgressStage(str, Enum):
    """Progress stages for query processing."""
    STARTED = "started"
    PARSING_QUERY = "parsing_query"
    INTENT_DETECTED = "intent_detected"
    MATCHING_API = "matching_api"
    API_MATCHED = "api_matched"
    PLANNING = "planning"
    PLAN_READY = "plan_ready"
    EXECUTING_API = "executing_api"
    API_COMPLETED = "api_completed"
    SUMMARIZING = "summarizing"
    COMPLETED = "completed"
    ERROR = "error"


class ProgressUpdate:
    """Progress update message."""
    
    def __init__(
        self,
        stage: ProgressStage,
        message: str,
        progress: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.stage = stage
        self.message = message
        self.progress = progress  # 0.0 to 1.0
        self.metadata = metadata or {}
        self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "stage": self.stage.value,
            "message": self.message,
            "progress": self.progress,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }


class ProgressTracker:
    """
    Track and stream progress updates during query processing.
    
    Usage:
        tracker = ProgressTracker(callback=send_to_frontend)
        tracker.update(ProgressStage.PARSING_QUERY, "Analyzing your query...")
    """
    
    def __init__(self, callback: Optional[Callable[[ProgressUpdate], None]] = None):
        """
        Initialize progress tracker.
        
        Args:
            callback: Function to call with each progress update
                     Signature: callback(ProgressUpdate) -> None
        """
        self.callback = callback
        self.updates: List[ProgressUpdate] = []
        self.start_time = time.time()
        self._last_update_time: float = 0.0
        self._current_stage = ProgressStage.STARTED
        logger.info("ProgressTracker initialized")
    
    def update(
        self,
        stage: ProgressStage,
        message: str,
        progress: Optional[float] = None,
        **metadata
    ):
        """
        Send a progress update.
        
        Args:
            stage: Current processing stage
            message: Human-readable progress message
            progress: Optional progress percentage (0.0 to 1.0)
            **metadata: Additional metadata (e.g., api_name, endpoint)
        """
        # Optional: ensure minimum display time so frontend can show each stage (avoids stages flashing by)
        min_ms = getattr(constants, "PROGRESS_MIN_DISPLAY_MS", 0) or 0
        if min_ms > 0 and self._last_update_time > 0:
            elapsed_ms = (time.time() - self._last_update_time) * 1000
            if elapsed_ms < min_ms:
                time.sleep((min_ms - elapsed_ms) / 1000.0)

        # Auto-calculate progress if not provided
        if progress is None:
            progress = self._calculate_progress(stage)
        
        update = ProgressUpdate(stage, message, progress, metadata)
        self.updates.append(update)
        self._current_stage = stage
        self._last_update_time = time.time()
        
        logger.info(f"Progress: {stage.value} - {message} ({progress*100:.0f}%)")
        
        # Call callback if provided
        if self.callback:
            try:
                self.callback(update)
            except Exception as e:
                logger.error(f"Error in progress callback: {e}")
    
    def _calculate_progress(self, stage: ProgressStage) -> float:
        """Auto-calculate progress based on stage."""
        stage_progress = {
            ProgressStage.STARTED: 0.0,
            ProgressStage.PARSING_QUERY: 0.1,
            ProgressStage.INTENT_DETECTED: 0.2,
            ProgressStage.MATCHING_API: 0.3,
            ProgressStage.API_MATCHED: 0.4,
            ProgressStage.PLANNING: 0.5,
            ProgressStage.PLAN_READY: 0.6,
            ProgressStage.EXECUTING_API: 0.7,
            ProgressStage.API_COMPLETED: 0.8,
            ProgressStage.SUMMARIZING: 0.9,
            ProgressStage.COMPLETED: 1.0,
            ProgressStage.ERROR: 0.0
        }
        return stage_progress.get(stage, 0.0)
    
    def get_elapsed_time(self) -> float:
        """Get elapsed time since start."""
        return time.time() - self.start_time
    
    def get_all_updates(self) -> List[Dict[str, Any]]:
        """Get all progress updates as dictionaries."""
        return [update.to_dict() for update in self.updates]
    
    def get_current_stage(self) -> ProgressStage:
        """Get current processing stage."""
        return self._current_stage


# Convenience function for creating friendly messages
def create_progress_message(stage: ProgressStage, **kwargs) -> str:
    """Create user-friendly progress messages."""
    from . import constants
    messages = {
        ProgressStage.STARTED: constants.PROGRESS_STARTED,
        ProgressStage.PARSING_QUERY: constants.PROGRESS_PARSING_QUERY,
        ProgressStage.INTENT_DETECTED: constants.PROGRESS_INTENT_DETECTED,
        ProgressStage.MATCHING_API: constants.PROGRESS_MATCHING_API,
        ProgressStage.API_MATCHED: constants.PROGRESS_API_MATCHED,
        ProgressStage.PLANNING: constants.PROGRESS_PLANNING,
        ProgressStage.PLAN_READY: constants.PROGRESS_PLAN_READY,
        ProgressStage.EXECUTING_API: constants.PROGRESS_EXECUTING_API,
        ProgressStage.API_COMPLETED: constants.PROGRESS_API_COMPLETED,
        ProgressStage.SUMMARIZING: constants.PROGRESS_SUMMARIZING,
        ProgressStage.COMPLETED: constants.PROGRESS_COMPLETED,
        ProgressStage.ERROR: constants.PROGRESS_ERROR,
    }
    
    template = messages.get(stage, constants.PROGRESS_DEFAULT)
    try:
        return template.format(**kwargs)
    except KeyError:
        return template
