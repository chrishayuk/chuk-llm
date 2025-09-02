"""
Automatic tool execution for function calling
"""

import json
import logging
from collections.abc import Callable
from typing import Any

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
            if isinstance(tool, dict) and tool.get("type") == "function":
                func_def = tool.get("function", {})
                name = func_def.get("name")
                # Check if there's an associated callable
                if hasattr(tool, "_func"):
                    self.tools[name] = tool._func

    def execute(self, tool_call: dict[str, Any]) -> Any:
        """Execute a single tool call"""
        try:
            # Extract tool information
            if tool_call.get("type") == "function":
                func_info = tool_call.get("function", {})
            else:
                func_info = tool_call

            name = func_info.get("name")
            arguments = func_info.get("arguments", {})

            # Parse arguments if they're a string
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse arguments for {name}: {arguments}")
                    arguments = {}

            # Execute the function
            if name in self.tools:
                func = self.tools[name]
                result = func(**arguments)
                return {
                    "tool_call_id": tool_call.get("id", "unknown"),
                    "name": name,
                    "result": result,
                }
            else:
                logger.warning(f"Tool {name} not found in executor")
                return {
                    "tool_call_id": tool_call.get("id", "unknown"),
                    "name": name,
                    "error": f"Tool {name} not registered",
                }

        except Exception as e:
            logger.error(f"Error executing tool {tool_call}: {e}")
            return {"tool_call_id": tool_call.get("id", "unknown"), "error": str(e)}

    def execute_all(self, tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Execute multiple tool calls"""
        results = []
        for tool_call in tool_calls:
            result = self.execute(tool_call)
            results.append(result)
        return results


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
    if not response.get("tool_calls"):
        return response.get("response", "")

    # Create executor and register functions
    executor = ToolExecutor()

    # Register provided functions
    if tool_functions:
        for name, func in tool_functions.items():
            executor.register(name, func)

    # Try to extract functions from tool definitions
    for tool in tools:
        if hasattr(tool, "_func") and hasattr(tool, "name"):
            executor.register(tool.name, tool._func)

    # Execute all tool calls
    tool_calls = response.get("tool_calls", [])
    results = executor.execute_all(tool_calls)

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
        if "error" in result:
            response_parts.append(
                f"Error calling {result.get('name', 'tool')}: {result['error']}"
            )
        else:
            tool_result = result.get("result", "")
            if tool_result:
                response_parts.append(f"Result: {tool_result}")

    return "\n".join(response_parts) if response_parts else "Tool calls executed"
