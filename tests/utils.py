"""FunctionModel test helpers for testing pydantic-ai agents.

Provides utilities for creating test doubles that simulate agent responses
without making real API calls, following the mining-report testing pattern.
"""

from typing import Any

from pydantic_ai import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel


def create_simple_model(response: str) -> FunctionModel:
    """Create a FunctionModel that always returns the given text response.

    Args:
        response: The text response to return.

    Returns:
        A FunctionModel instance for use with agent.override().
    """

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content=response)])

    return FunctionModel(model_fn)


def create_tool_calling_model(
    tool_name: str,
    tool_args: dict[str, Any],
    final_response: str,
) -> FunctionModel:
    """Create a FunctionModel that first calls a tool, then returns a final response.

    Args:
        tool_name: Name of the tool to call.
        tool_args: Arguments for the tool call.
        final_response: The text response after the tool call.

    Returns:
        A FunctionModel instance for use with agent.override().
    """
    from pydantic_ai import ToolCallPart
    from pydantic_ai.models.function import FunctionModel

    call_count = 0

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ModelResponse(parts=[ToolCallPart(tool_name, tool_args)])
        return ModelResponse(parts=[TextPart(content=final_response)])

    return FunctionModel(model_fn)


def create_sequential_model(responses: list[str]) -> FunctionModel:
    """Create a FunctionModel that returns responses in sequence.

    Args:
        responses: List of text responses to return in order.

    Returns:
        A FunctionModel instance for use with agent.override().
    """
    call_count = 0

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        idx = min(call_count, len(responses) - 1)
        call_count += 1
        return ModelResponse(parts=[TextPart(content=responses[idx])])

    return FunctionModel(model_fn)