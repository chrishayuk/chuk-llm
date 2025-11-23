"""
Automatic tool execution for function calling
"""

import asyncio
import inspect
import json
import logging
from collections.abc import Callable
from typing import Any

from chuk_llm.core.constants import ResponseKey, ToolParam
from chuk_llm.core.enums import MessageRole

logger = logging.getLogger(__name__)


class ToolExecutor:
    """Handles automatic execution of tool calls"""

    def __init__(self):
        self.tools: dict[str, Callable] = {}

    def register(self, name: str, func: Callable) -> None:
        """Register a function that can be called"""
        self.tools[name] = func

    def register_multiple(self, tools: list[dict[str, Any]]) -> None:
        """Register multiple tools from OpenAI format with functions"""
        for tool in tools:
            if (
                isinstance(tool, dict)
                and tool.get(ToolParam.TYPE.value) == ToolParam.FUNCTION.value
            ):
                func_def = tool.get(ToolParam.FUNCTION.value, {})
                name = func_def.get(ToolParam.NAME.value)
                # Check if there's an associated callable
                if hasattr(tool, "_func"):
                    self.tools[name] = tool._func

    async def execute(self, tool_call: dict[str, Any]) -> Any:
        """Execute a single tool call (async-native)"""
        try:
            # Extract tool information
            if tool_call.get(ToolParam.TYPE.value) == ToolParam.FUNCTION.value:
                func_info = tool_call.get(ToolParam.FUNCTION.value, {})
            else:
                func_info = tool_call

            name = func_info.get(ToolParam.NAME.value)
            arguments = func_info.get(ToolParam.ARGUMENTS.value, {})

            # Parse arguments if they're a string
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse arguments for {name}: {arguments}")
                    arguments = {}

            # Execute the function (async-native)
            if name in self.tools:
                func = self.tools[name]

                # Check if function is async
                if inspect.iscoroutinefunction(func):
                    result = await func(**arguments)
                else:
                    # Run sync functions in thread pool to avoid blocking
                    result = await asyncio.to_thread(func, **arguments)

                return {
                    ToolParam.TOOL_CALL_ID.value: tool_call.get(
                        ToolParam.ID.value, "unknown"
                    ),
                    ToolParam.NAME.value: name,
                    ResponseKey.RESULT.value: result,
                }
            else:
                logger.warning(f"Tool {name} not found in executor")
                return {
                    ToolParam.TOOL_CALL_ID.value: tool_call.get(
                        ToolParam.ID.value, "unknown"
                    ),
                    ToolParam.NAME.value: name,
                    ResponseKey.ERROR.value: f"Tool {name} not registered",
                }

        except Exception as e:
            logger.error(f"Error executing tool {tool_call}: {e}")
            return {
                ToolParam.TOOL_CALL_ID.value: tool_call.get(
                    ToolParam.ID.value, "unknown"
                ),
                ResponseKey.ERROR.value: str(e),
            }

    async def execute_all(
        self, tool_calls: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Execute multiple tool calls in parallel (async-native)"""
        # Execute all tool calls concurrently
        tasks = [self.execute(tool_call) for tool_call in tool_calls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to error results
        formatted_results: list[dict[str, Any]] = []
        for i, result in enumerate(results):
            if isinstance(result, BaseException):
                formatted_results.append(
                    {
                        ToolParam.TOOL_CALL_ID.value: tool_calls[i].get(
                            ToolParam.ID.value, "unknown"
                        ),
                        ResponseKey.ERROR.value: str(result),
                    }
                )
            else:
                formatted_results.append(result)

        return formatted_results


async def execute_tool_calls(
    response: dict[str, Any],
    tools: list[dict[str, Any]],
    tool_functions: dict[str, Callable] | None = None,
    tool_objects: dict[str, Any] | None = None,
) -> str:
    """
    Execute tool calls from an LLM response and format the results.

    Args:
        response: The response from the LLM containing tool_calls
        tools: List of tool definitions in OpenAI format
        tool_functions: Optional mapping of tool names to functions

    Returns:
        Formatted string with tool execution results
    """
    if not response.get(ResponseKey.TOOL_CALLS.value):
        return response.get("response", "")

    # Create executor and register functions
    executor = ToolExecutor()

    # Register provided functions
    if tool_functions:
        for name, func in tool_functions.items():
            executor.register(name, func)

    # Try to extract functions from tool definitions
    for tool in tools:
        if hasattr(tool, "_func") and hasattr(tool, ToolParam.NAME.value):
            executor.register(tool.name, tool._func)

    # Execute all tool calls
    tool_calls = response.get(ResponseKey.TOOL_CALLS.value, [])
    results = await executor.execute_all(tool_calls)

    # Format results
    if not results:
        return response.get("response", "")

    # Build response with tool results
    response_parts = []

    # Add any initial response
    if response.get("response"):
        response_parts.append(response["response"])

    # Add tool results
    for result in results:
        if ResponseKey.ERROR.value in result:
            response_parts.append(
                f"Error calling {result.get(ToolParam.NAME.value, MessageRole.TOOL.value)}: {result[ResponseKey.ERROR.value]}"
            )
        else:
            tool_result = result.get(ResponseKey.RESULT.value, "")
            if tool_result:
                response_parts.append(f"Result: {tool_result}")

    return "\n".join(response_parts) if response_parts else "Tool calls executed"
