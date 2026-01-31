"""
Utility functions for Enable AI

Contains:
- OpenAI client management and calls
- Logging configuration
- Common helper functions
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from openai import OpenAI

# Check for debug prompts environment variable
DEBUG_OPENAI_PROMPTS = os.getenv('ENABLE_AI_DEBUG_PROMPTS', 'false').lower() == 'true'


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Set up a logger with consistent formatting.
    
    Args:
        name: Logger name (usually module name)
        level: Logging level (default: INFO)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Only add handler if logger doesn't have one
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    logger.setLevel(level)
    return logger


# ============================================================================
# OPENAI CLIENT MANAGEMENT
# ============================================================================

class OpenAIClient:
    """
    Centralized OpenAI client for consistent API calls across the application.
    
    Benefits:
    - Single source of truth for model configuration
    - Easy to update model versions
    - Consistent error handling
    - Request/response logging
    """
    
    _instance = None
    _client = None
    
    def __new__(cls):
        """Singleton pattern to ensure single OpenAI client instance."""
        if cls._instance is None:
            cls._instance = super(OpenAIClient, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize OpenAI client if not already initialized."""
        if self._client is None:
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                from . import constants
                raise ValueError(constants.ERROR_OPENAI_KEY_MISSING)
            
            self._client = OpenAI(api_key=api_key)
            self.logger = setup_logger('enable_ai.openai')
            self.logger.info("OpenAI client initialized")
    
    @property
    def client(self) -> OpenAI:
        """Get the OpenAI client instance."""
        return self._client
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-4o",
        temperature: float = 0.1,
        response_format: Optional[Dict[str, str]] = None,
        max_tokens: Optional[int] = None,
        log_request: bool = True,
        debug_prompts: Optional[bool] = None
    ) -> str:
        """
        Make a chat completion request to OpenAI.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model to use (default: gpt-4o)
            temperature: Temperature for sampling (default: 0.1 for deterministic)
            response_format: Optional response format (e.g., {"type": "json_object"})
            max_tokens: Optional max tokens in response
            log_request: Whether to log the request (default: True)
            debug_prompts: Whether to log full prompts and responses. If None, uses ENABLE_AI_DEBUG_PROMPTS env var.
            
        Returns:
            The completion text from OpenAI
            
        Raises:
            Exception: If OpenAI API call fails
        """
        # Use environment variable if not explicitly set
        if debug_prompts is None:
            debug_prompts = DEBUG_OPENAI_PROMPTS
        
        if log_request:
            self.logger.info(f"OpenAI request: model={model}, temp={temperature}, messages={len(messages)}")
        
        # Debug logging: log full prompts (can be verbose!)
        if debug_prompts:
            from . import constants
            self.logger.debug("=" * 80)
            self.logger.debug("OpenAI Request Prompts:")
            for idx, msg in enumerate(messages):
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')
                # Truncate very long content
                if len(content) > constants.LOG_PREVIEW_LENGTH:
                    content = content[: constants.UTILS_CONTENT_TRUNCATE] + "\n...[truncated]...\n" + content[-constants.UTILS_CONTENT_TRUNCATE:]
                self.logger.debug(f"Message {idx} ({role}):\n{content}")
            self.logger.debug("=" * 80)
        
        try:
            kwargs = {
                "model": model,
                "messages": messages,
                "temperature": temperature
            }
            
            if response_format:
                kwargs["response_format"] = response_format
            
            if max_tokens:
                kwargs["max_tokens"] = max_tokens
            
            response = self._client.chat.completions.create(**kwargs)
            
            content = response.choices[0].message.content
            
            if log_request:
                usage = response.usage
                self.logger.info(
                    f"OpenAI response: tokens={usage.total_tokens} "
                    f"(prompt={usage.prompt_tokens}, completion={usage.completion_tokens})"
                )
            
            # Debug logging: log full response
            if debug_prompts:
                self.logger.debug("=" * 80)
                self.logger.debug("OpenAI Response:")
                # Truncate very long responses
                from . import constants
                if content and len(content) > constants.LOG_PREVIEW_LENGTH:
                    truncated = content[: constants.UTILS_CONTENT_TRUNCATE] + "\n...[truncated]...\n" + content[-constants.UTILS_CONTENT_TRUNCATE:]
                    self.logger.debug(truncated)
                else:
                    self.logger.debug(content)
                self.logger.debug("=" * 80)
            
            return content
            
        except Exception as e:
            self.logger.error(f"OpenAI API call failed: {str(e)}")
            raise
    
    def parse_json_response(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-4o",
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        debug_prompts: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Make a chat completion request expecting JSON response.
        
        Args:
            messages: List of message dicts
            model: Model to use
            temperature: Temperature for sampling
            max_tokens: Optional max tokens
            debug_prompts: Whether to log full prompts and responses
            
        Returns:
            Parsed JSON response as dict
            
        Raises:
            json.JSONDecodeError: If response is not valid JSON
            Exception: If OpenAI API call fails
        """
        content = self.chat_completion(
            messages=messages,
            model=model,
            temperature=temperature,
            response_format={"type": "json_object"},
            max_tokens=max_tokens,
            debug_prompts=debug_prompts
        )
        
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON response: {content}")
            raise
    
    def chat_completion_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        tool_choice: str = "auto",
        model: str = "gpt-4o",
        temperature: float = 0.1,
        log_request: bool = True,
        debug_prompts: Optional[bool] = None
    ) -> Any:
        """
        Make a chat completion request with tool/function calling support.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            tools: List of tool definitions (OpenAI function calling format)
            tool_choice: Tool choice strategy ("auto", "none", or specific tool)
            model: Model to use (default: gpt-4o)
            temperature: Temperature for sampling
            log_request: Whether to log the request
            debug_prompts: Whether to log full prompts, tools, and responses. If None, uses ENABLE_AI_DEBUG_PROMPTS env var.
            
        Returns:
            Full OpenAI response object (to access tool_calls if any)
            
        Raises:
            Exception: If OpenAI API call fails
        """
        # Use environment variable if not explicitly set
        if debug_prompts is None:
            debug_prompts = DEBUG_OPENAI_PROMPTS
        
        if log_request:
            self.logger.info(f"OpenAI request with tools: model={model}, tools={len(tools)}, messages={len(messages)}")
        
        # Debug logging: log tools
        if debug_prompts:
            self.logger.debug("=" * 80)
            self.logger.debug("OpenAI Function Calling Request:")
            self.logger.debug(f"Tools defined ({len(tools)}):")
            for idx, tool in enumerate(tools):
                tool_name = tool.get('function', {}).get('name', 'unknown')
                tool_desc = tool.get('function', {}).get('description', 'No description')
                self.logger.debug(f"  Tool {idx}: {tool_name}")
                from . import constants
                self.logger.debug(f"    Description: {tool_desc[: constants.UTILS_TOOL_DESC_PREVIEW]}...")
            self.logger.debug("\nMessages:")
            for idx, msg in enumerate(messages):
                role = msg.get('role', 'unknown')
                content = str(msg.get('content', ''))
                if len(content) > constants.LOG_CONTENT_PREVIEW:
                    content = content[: constants.TRUNCATE_MAX_LENGTH] + "...[truncated]..." + content[-constants.TRUNCATE_MAX_LENGTH:]
                self.logger.debug(f"  Message {idx} ({role}): {content}")
            self.logger.debug("=" * 80)
        
        try:
            response = self._client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                temperature=temperature
            )
            
            if log_request:
                usage = response.usage
                self.logger.info(
                    f"OpenAI response: tokens={usage.total_tokens} "
                    f"(prompt={usage.prompt_tokens}, completion={usage.completion_tokens})"
                )
            
            # Debug logging: log tool calls
            if debug_prompts:
                self.logger.debug("=" * 80)
                self.logger.debug("OpenAI Function Calling Response:")
                message = response.choices[0].message
                if message.tool_calls:
                    self.logger.debug(f"Tool calls made: {len(message.tool_calls)}")
                    for tc in message.tool_calls:
                        self.logger.debug(f"  - Function: {tc.function.name}")
                        self.logger.debug(f"    Arguments: {tc.function.arguments}")
                else:
                    self.logger.debug("No tool calls made")
                    if message.content:
                        content = message.content
                        if len(content) > constants.UTILS_CONTENT_TRUNCATE:
                            content = content[: constants.UTILS_CONTENT_DEBUG] + "...[truncated]..." + content[-constants.UTILS_CONTENT_DEBUG:]
                        self.logger.debug(f"Content: {content}")
                self.logger.debug("=" * 80)
            
            return response
            
        except Exception as e:
            self.logger.error(f"OpenAI API call with tools failed: {str(e)}")
            raise


# Singleton instance
_openai_client = None


def get_openai_client() -> OpenAIClient:
    """
    Get the singleton OpenAI client instance.
    
    Returns:
        OpenAIClient instance
    """
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAIClient()
    return _openai_client


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def truncate_string(s: str, max_length: int = None) -> str:
    """
    Truncate a string to max_length, adding ellipsis if needed.

    Args:
        s: String to truncate
        max_length: Maximum length (default: constants.TRUNCATE_MAX_LENGTH)

    Returns:
        Truncated string
    """
    from . import constants
    if max_length is None:
        max_length = constants.TRUNCATE_MAX_LENGTH
    if len(s) <= max_length:
        return s
    return s[:max_length - 3] + "..."


def safe_json_dumps(obj: Any, indent: int = 2) -> str:
    """
    Safely convert object to JSON string, handling non-serializable objects.
    
    Args:
        obj: Object to serialize
        indent: JSON indentation
        
    Returns:
        JSON string
    """
    try:
        return json.dumps(obj, indent=indent)
    except (TypeError, ValueError):
        return str(obj)


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON object from text that may contain additional content.
    
    Useful for parsing LLM responses that may include explanations.
    
    Args:
        text: Text potentially containing JSON
        
    Returns:
        Parsed JSON dict or None if not found
    """
    # Try to find JSON object in the text
    start_idx = text.find('{')
    end_idx = text.rfind('}')
    
    if start_idx == -1 or end_idx == -1:
        return None
    
    try:
        json_str = text[start_idx:end_idx + 1]
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None


# ============================================================================
# CONFIGURATION
# ============================================================================

# Model configurations - easy to update in one place
DEFAULT_MODEL = "gpt-4o"
FAST_MODEL = "gpt-4o-mini"  # For simple tasks
COMPLEX_MODEL = "gpt-4o"     # For complex planning

# Temperature settings
DETERMINISTIC_TEMP = 0.1     # For parsing/planning
CREATIVE_TEMP = 0.7          # For summarization


def get_model_for_task(task: str) -> str:
    """
    Get the appropriate model for a given task.
    
    Args:
        task: Task type ('parse', 'plan', 'summarize')
        
    Returns:
        Model name to use
    """
    task_models = {
        'parse': DEFAULT_MODEL,
        'plan': COMPLEX_MODEL,
        'summarize': DEFAULT_MODEL,
        'simple': FAST_MODEL
    }
    return task_models.get(task, DEFAULT_MODEL)
