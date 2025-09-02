"""
Enhanced tool execution with multi-step conversation support
"""

import json
import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


class ToolExecutor:
    """Handles automatic execution of tool calls with conversation continuation"""

    def __init__(self):
        self.tools: dict[str, Callable] = {}
        self.execution_log = []

    def register(self, name: str, func: Callable) -> None:
        """Register a function that can be called"""
        self.tools[name] = func
        logger.debug(f"Registered tool: {name}")

    def register_from_definitions(
        self, tools: list[dict[str, Any]], functions: dict[str, Callable] | None = None
    ) -> None:
        """Register tools from OpenAI-format definitions with optional function mappings"""
        functions = functions or {}

        for tool in tools:
            if tool.get("type") == "function":
                func_def = tool.get("function", {})
                name = func_def.get("name")

                # Try to find the function implementation
                if name in functions:
                    self.register(name, functions[name])
                    logger.debug(f"Registered tool {name} with implementation")
                else:
                    logger.warning(
                        f"Tool {name} defined but no implementation provided"
                    )

    def execute(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        """Execute a single tool call and return results"""
        try:
            # Extract tool information based on format
            if "function" in tool_call:
                # OpenAI format
                func_info = tool_call["function"]
                name = func_info.get("name")
                arguments = func_info.get("arguments", {})
                tool_id = tool_call.get("id", f"call_{name}")
            else:
                # Simple format
                name = tool_call.get("name")
                arguments = tool_call.get("arguments", {})
                tool_id = tool_call.get("id", f"call_{name}")

            # Parse arguments if they're a JSON string
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse arguments for {name}: {arguments}")
                    arguments = {}

            # Execute the function
            if name in self.tools:
                func = self.tools[name]
                logger.debug(f"Executing tool {name} with args: {arguments}")

                try:
                    result = func(**arguments)

                    # Convert result to string if needed
                    if not isinstance(result, str):
                        result = (
                            json.dumps(result)
                            if isinstance(result, dict | list)
                            else str(result)
                        )

                    execution_result = {
                        "tool_call_id": tool_id,
                        "role": "tool",
                        "name": name,
                        "content": result,
                    }

                    self.execution_log.append(
                        {
                            "tool": name,
                            "args": arguments,
                            "result": result,
                            "success": True,
                        }
                    )

                    return execution_result

                except Exception as e:
                    error_msg = f"Error executing {name}: {str(e)}"
                    logger.error(error_msg)

                    self.execution_log.append(
                        {
                            "tool": name,
                            "args": arguments,
                            "error": str(e),
                            "success": False,
                        }
                    )

                    return {
                        "tool_call_id": tool_id,
                        "role": "tool",
                        "name": name,
                        "content": error_msg,
                    }
            else:
                error_msg = f"Tool {name} not found in executor"
                logger.warning(error_msg)
                return {
                    "tool_call_id": tool_id,
                    "role": "tool",
                    "name": name,
                    "content": error_msg,
                }

        except Exception as e:
            logger.error(f"Error processing tool call {tool_call}: {e}")
            return {
                "tool_call_id": "error",
                "role": "tool",
                "content": f"Error: {str(e)}",
            }

    def execute_all(self, tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Execute multiple tool calls and return results"""
        results = []
        for tool_call in tool_calls:
            result = self.execute(tool_call)
            results.append(result)
        return results

    def get_execution_summary(self) -> str:
        """Get a summary of all tool executions"""
        if not self.execution_log:
            return "No tools executed"

        summary_parts = []
        for entry in self.execution_log:
            if entry.get("success"):
                summary_parts.append(
                    f"✓ {entry['tool']}: {entry.get('result', 'Success')[:100]}"
                )
            else:
                summary_parts.append(
                    f"✗ {entry['tool']}: {entry.get('error', 'Failed')}"
                )

        return "\n".join(summary_parts)


async def execute_with_tools(
    ask_func: Callable,
    prompt: str,
    tools: list[dict[str, Any]],
    tool_functions: dict[str, Callable],
    max_rounds: int = 3,
    **kwargs,
) -> str:
    """
    Execute a prompt with tools using multi-step conversation pattern.

    This follows the standard OpenAI pattern:
    1. Initial call with tools -> get tool_calls
    2. Execute tools locally
    3. Send results back to model
    4. Get final response

    Args:
        ask_func: The async ask function to use
        prompt: Initial prompt
        tools: Tool definitions in OpenAI format
        tool_functions: Mapping of tool names to callable functions
        max_rounds: Maximum conversation rounds for tool execution
        **kwargs: Additional arguments for ask_func

    Returns:
        Final response after tool execution
    """
    from .core import ask

    executor = ToolExecutor()
    executor.register_from_definitions(tools, tool_functions)

    messages: list[dict[str, Any]] = []
    current_prompt = prompt

    for round_num in range(max_rounds):
        logger.debug(f"Tool execution round {round_num + 1}/{max_rounds}")

        # Call with tools - automatically includes tool calls in response
        response = await ask(current_prompt, tools=tools, **kwargs)

        # Check if we got tool calls
        if isinstance(response, dict) and response.get("tool_calls"):
            tool_calls = response["tool_calls"]
            logger.debug(f"Got {len(tool_calls)} tool calls")

            # Execute tools
            tool_results = executor.execute_all(tool_calls)

            # Format results for next message
            results_text = []
            for result in tool_results:
                tool_name = result.get("name", "tool")
                tool_content = result.get("content", "")
                results_text.append(f"{tool_name}: {tool_content}")

            # Continue conversation with tool results
            if results_text:
                current_prompt = (
                    "Tool execution results:\n"
                    + "\n".join(results_text)
                    + "\n\nBased on these results, please provide the final answer."
                )

                # Add to message history for context
                messages.append(
                    {
                        "role": "assistant",
                        "content": response.get("response", ""),
                        "tool_calls": tool_calls,
                    }
                )
                messages.append({"role": "user", "content": current_prompt})
            else:
                # No results, try to get a text response
                break

        else:
            # Got a text response or no tool calls
            if isinstance(response, dict):
                final_response = response.get("response", "")
            else:
                final_response = str(response) if response else ""

            # If we have a response, return it
            if final_response:
                return final_response

            # If no response and we have execution history, summarize
            if executor.execution_log:
                summary = executor.get_execution_summary()
                return f"Executed tools:\n{summary}"

            # Otherwise, break to avoid infinite loop
            break

    # If we exhausted rounds, return what we have
    if executor.execution_log:
        summary = executor.get_execution_summary()
        return f"Tool execution completed:\n{summary}"

    return "No response generated"


def execute_with_tools_sync(
    ask_func: Callable,
    prompt: str,
    tools: list[dict[str, Any]],
    tool_functions: dict[str, Callable],
    max_rounds: int = 3,
    **kwargs,
) -> str:
    """
    Synchronous version of execute_with_tools.
    """
    from .sync import ask_sync

    executor = ToolExecutor()
    executor.register_from_definitions(tools, tool_functions)

    messages: list[dict[str, Any]] = []
    current_prompt = prompt

    for round_num in range(max_rounds):
        logger.debug(f"Tool execution round {round_num + 1}/{max_rounds}")

        # Call with tools - automatically includes tool calls in response
        response = ask_sync(current_prompt, tools=tools, **kwargs)

        # Check if we got tool calls
        if isinstance(response, dict) and response.get("tool_calls"):
            tool_calls = response["tool_calls"]
            logger.debug(f"Got {len(tool_calls)} tool calls")

            # Execute tools
            tool_results = executor.execute_all(tool_calls)

            # Format results for next message
            results_text = []
            for result in tool_results:
                tool_name = result.get("name", "tool")
                tool_content = result.get("content", "")
                results_text.append(f"{tool_name}: {tool_content}")

            # Continue conversation with tool results
            if results_text:
                current_prompt = (
                    "Tool execution results:\n"
                    + "\n".join(results_text)
                    + "\n\nBased on these results, please provide the final answer."
                )

                # Add to message history for context
                messages.append(
                    {
                        "role": "assistant",
                        "content": response.get("response", ""),
                        "tool_calls": tool_calls,
                    }
                )
                messages.append({"role": "user", "content": current_prompt})
            else:
                # No results, try to get a text response
                break

        else:
            # Got a text response or no tool calls
            if isinstance(response, dict):
                final_response = response.get("response", "")
            else:
                final_response = str(response) if response else ""

            # If we have a response, return it
            if final_response:
                return final_response

            # If no response and we have execution history, summarize
            if executor.execution_log:
                summary = executor.get_execution_summary()
                return f"Executed tools:\n{summary}"

            # Otherwise, break to avoid infinite loop
            break

    # If we exhausted rounds, return what we have
    if executor.execution_log:
        summary = executor.get_execution_summary()
        return f"Tool execution completed:\n{summary}"

    return "No response generated"
