"""
Unified LLM client with provider swap support.

Supports: openai, anthropic
Default: openai (cost-effective)
"""

from __future__ import annotations

import json
import os
from typing import Optional, TYPE_CHECKING

from .config import Config
from .tool_filter import filter_tools_for_query, get_max_tools_for_provider
from .types import ToolCall

if TYPE_CHECKING:
    from openai import OpenAI  # type: ignore[import-not-found]
    from anthropic import Anthropic  # type: ignore[import-not-found]


class LLMClient:
    """
    Unified LLM client supporting multiple providers.

    Usage:
        # OpenAI (default, cheaper)
        client = LLMClient(provider="openai")

        # Anthropic
        client = LLMClient(provider="anthropic")
    """

    PROVIDERS = {"openai", "anthropic"}

    # Model mappings per provider
    DEFAULT_MODELS = {
        "openai": "gpt-4o",
        "anthropic": "claude-sonnet-4-20250514",
    }

    _client: "OpenAI | Anthropic"

    def __init__(
        self,
        provider: str = "openai",
        api_key: Optional[str] = None,
        config: Optional[Config] = None,
    ):
        if provider not in self.PROVIDERS:
            raise ValueError(f"Provider must be one of: {self.PROVIDERS}")

        self.provider = provider
        self.config = config or Config()

        # Get API key
        if provider == "openai":
            self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
            env_var = "OPENAI_API_KEY"
        else:  # anthropic
            self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            env_var = "ANTHROPIC_API_KEY"

        if not self.api_key:
            raise ValueError(f"{env_var} required. Set env var or pass api_key.")

        # Initialize client
        self._init_client()

    def _init_client(self) -> None:
        """Initialize the provider-specific client."""
        if self.provider == "openai":
            from openai import OpenAI  # type: ignore[import-not-found]
            self._client = OpenAI(api_key=self.api_key)
        else:  # anthropic
            from anthropic import Anthropic  # type: ignore[import-not-found]
            self._client = Anthropic(api_key=self.api_key)

    def _get_model(self) -> str:
        """Get model name, using config or default for provider."""
        # If config has a provider-specific model, use it
        # Otherwise use default for this provider
        config_model = self.config.model

        # Check if config model matches current provider
        if self.provider == "openai" and config_model.startswith("gpt"):
            return config_model
        elif self.provider == "anthropic" and "claude" in config_model:
            return config_model

        # Use default for provider
        return self.DEFAULT_MODELS[self.provider]

    def _convert_tools_for_provider(self, tools: list[dict]) -> list[dict]:
        """Convert tool definitions to provider-specific format."""
        if self.provider == "openai":
            # OpenAI uses 'parameters' instead of 'input_schema'
            # and wraps in {"type": "function", "function": {...}}
            converted = []
            for tool in tools:
                converted.append({
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
                    }
                })
            return converted
        else:  # anthropic - already in correct format
            return tools

    def process_query(
        self,
        query: str,
        tools: list[dict],
        system_prompt: str,
        conversation_history: Optional[list[dict]] = None,
    ) -> tuple[list[ToolCall], str, str]:
        """
        Process a natural language query with tool calling.

        Args:
            query: User's natural language query
            tools: List of tool definitions (LLM tool schema format)
            system_prompt: System prompt with context
            conversation_history: Optional prior messages

        Returns:
            Tuple of (tool_calls, reasoning, stop_reason)
        """
        # Shared query-time filtering (all providers)
        if self.config.tool_filter_enabled:
            max_tools = get_max_tools_for_provider(
                self.provider,
                self.config.max_tools_per_query,
            )
            filtered_tools = filter_tools_for_query(
                query,
                tools,
                max_tools=max_tools,
                conversation_history=conversation_history,
                id_code_patterns=self.config.id_code_patterns,
            )
        else:
            filtered_tools = tools

        if self.provider == "openai":
            return self._process_openai(query, filtered_tools, system_prompt, conversation_history)
        else:
            return self._process_anthropic(query, filtered_tools, system_prompt, conversation_history)

    def _process_openai(
        self,
        query: str,
        tools: list[dict],
        system_prompt: str,
        conversation_history: Optional[list[dict]] = None,
    ) -> tuple[list[ToolCall], str, str]:
        """Process query using OpenAI."""
        messages = [{"role": "system", "content": system_prompt}]

        if conversation_history:
            messages.extend(conversation_history)

        messages.append({"role": "user", "content": query})

        openai_tools = self._convert_tools_for_provider(tools)

        # Use higher max_tokens to avoid truncation
        max_tokens = max(self.config.max_tokens, 8192)

        response = self._client.chat.completions.create(
            model=self._get_model(),
            messages=messages,
            tools=openai_tools if openai_tools else None,
            tool_choice="auto" if openai_tools else None,
            temperature=self.config.temperature,
            max_tokens=max_tokens,
        )

        message = response.choices[0].message
        tool_calls = []
        reasoning = message.content or ""
        finish_reason = response.choices[0].finish_reason

        # Handle truncation - continue if needed
        if finish_reason == "length" and not message.tool_calls:
            # Response was truncated, try to continue
            messages.append({"role": "assistant", "content": reasoning})
            messages.append({"role": "user", "content": "Please continue."})
            continuation = self._client.chat.completions.create(
                model=self._get_model(),
                messages=messages,
                tools=openai_tools if openai_tools else None,
                tool_choice="auto" if openai_tools else None,
                temperature=self.config.temperature,
                max_tokens=max_tokens,
            )
            cont_message = continuation.choices[0].message
            reasoning += cont_message.content or ""
            message = cont_message
            finish_reason = continuation.choices[0].finish_reason

        if message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=json.loads(tc.function.arguments),
                    )
                )

        return tool_calls, reasoning, finish_reason

    def _process_anthropic(
        self,
        query: str,
        tools: list[dict],
        system_prompt: str,
        conversation_history: Optional[list[dict]] = None,
    ) -> tuple[list[ToolCall], str, str]:
        """Process query using Anthropic."""
        messages = conversation_history or []
        messages = messages + [{"role": "user", "content": query}]

        # Use higher max_tokens to avoid truncation
        max_tokens = max(self.config.max_tokens, 8192)

        response = self._client.messages.create(
            model=self._get_model(),
            max_tokens=max_tokens,
            temperature=self.config.temperature,
            system=system_prompt,
            tools=tools,
            messages=messages,
        )

        tool_calls = []
        reasoning = ""

        for block in response.content:
            if block.type == "text":
                reasoning = block.text
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=block.input,
                    )
                )

        # Handle truncation - continue if needed
        if response.stop_reason == "max_tokens" and not tool_calls:
            messages.append({"role": "assistant", "content": reasoning})
            messages.append({"role": "user", "content": "Please continue."})
            continuation = self._client.messages.create(
                model=self._get_model(),
                max_tokens=max_tokens,
                temperature=self.config.temperature,
                system=system_prompt,
                tools=tools,
                messages=messages,
            )
            for block in continuation.content:
                if block.type == "text":
                    reasoning += block.text
                elif block.type == "tool_use":
                    tool_calls.append(
                        ToolCall(
                            id=block.id,
                            name=block.name,
                            arguments=block.input,
                        )
                    )

        return tool_calls, reasoning, response.stop_reason

    def generate_response(
        self,
        query: str,
        api_results: list[dict],
        tool_calls: list[ToolCall],
        system_prompt: str,
    ) -> str:
        """
        Generate a chat-friendly response from API results.

        Args:
            query: Original user query
            api_results: Results from API calls
            tool_calls: Tool calls that were executed
            system_prompt: System prompt for response generation

        Returns:
            Chat-friendly response string
        """
        # Build context message
        context = f"User asked: {query}\n\n"
        context += "API Results:\n"
        for i, (tc, result) in enumerate(zip(tool_calls, api_results)):
            context += f"\n{i+1}. {tc.name}:\n{_format_result(result)}\n"

        user_content = context + "\n\nGenerate a helpful, conversational response."

        if self.provider == "openai":
            response = self._client.chat.completions.create(
                model=self._get_model(),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            return response.choices[0].message.content or "I processed your request."

        else:  # anthropic
            response = self._client.messages.create(
                model=self._get_model(),
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_content}],
            )
            for block in response.content:
                if block.type == "text":
                    return block.text
            return "I processed your request."


def _format_result(result: dict) -> str:
    """Format API result for context."""
    if "error" in result:
        return f"Error: {result['error']}"

    data = result.get("data", result)

    # Truncate large responses
    text = json.dumps(data, indent=2, default=str)
    if len(text) > 2000:
        text = text[:2000] + "\n... (truncated)"

    return text


# Backward compatibility alias
ClaudeClient = LLMClient
