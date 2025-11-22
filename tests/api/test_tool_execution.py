"""
Comprehensive tests for tool_execution module
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from chuk_llm.api.tool_execution import (
    ToolExecutor,
    execute_with_tools,
    execute_with_tools_sync,
)


class TestToolExecutor:
    """Test cases for ToolExecutor class"""

    def test_init(self):
        """Test ToolExecutor initialization"""
        executor = ToolExecutor()
        assert isinstance(executor.tools, dict)
        assert len(executor.tools) == 0
        assert isinstance(executor.execution_log, list)
        assert len(executor.execution_log) == 0

    def test_register(self):
        """Test registering a tool"""

        def sample_tool():
            return "result"

        executor = ToolExecutor()
        executor.register("sample", sample_tool)

        assert "sample" in executor.tools
        assert executor.tools["sample"] == sample_tool

    def test_register_from_definitions_with_functions(self):
        """Test registering from OpenAI format definitions"""

        def weather_func(location: str):
            return f"Weather in {location}"

        tools = [
            {
                "type": "function",
                "function": {"name": "get_weather", "description": "Get weather"},
            }
        ]

        executor = ToolExecutor()
        executor.register_from_definitions(tools, {"get_weather": weather_func})

        assert "get_weather" in executor.tools

    def test_register_from_definitions_without_implementation(self):
        """Test registering without implementation (should warn)"""
        tools = [
            {
                "type": "function",
                "function": {"name": "missing_func", "description": "Test"},
            }
        ]

        executor = ToolExecutor()
        executor.register_from_definitions(tools, {})

        # Should not be registered without implementation
        assert "missing_func" not in executor.tools

    def test_register_from_definitions_non_function_type(self):
        """Test registering non-function type tools"""
        tools = [{"type": "other", "function": {"name": "test"}}]

        executor = ToolExecutor()
        executor.register_from_definitions(tools, {})

        assert len(executor.tools) == 0

    def test_execute_openai_format(self):
        """Test executing tool call in OpenAI format"""

        def add(a: int, b: int):
            return a + b

        executor = ToolExecutor()
        executor.register("add", add)

        tool_call = {
            "id": "call_123",
            "function": {"name": "add", "arguments": {"a": 5, "b": 3}},
        }

        result = executor.execute(tool_call)

        assert result["tool_call_id"] == "call_123"
        assert result["role"] == "tool"
        assert result["name"] == "add"
        assert "8" in result["content"]

    def test_execute_simple_format(self):
        """Test executing tool call in simple format"""

        def multiply(x: int, y: int):
            return x * y

        executor = ToolExecutor()
        executor.register("multiply", multiply)

        tool_call = {
            "id": "call_456",
            "name": "multiply",
            "arguments": {"x": 4, "y": 5},
        }

        result = executor.execute(tool_call)

        assert result["tool_call_id"] == "call_456"
        assert "20" in result["content"]

    def test_execute_with_json_string_arguments(self):
        """Test executing with arguments as JSON string"""

        def func(param: str):
            return f"Got {param}"

        executor = ToolExecutor()
        executor.register("func", func)

        tool_call = {
            "function": {"name": "func", "arguments": '{"param": "value"}'}
        }

        result = executor.execute(tool_call)

        assert "Got value" in result["content"]

    def test_execute_with_invalid_json_arguments(self):
        """Test executing with invalid JSON arguments"""

        def no_args():
            return "ok"

        executor = ToolExecutor()
        executor.register("no_args", no_args)

        tool_call = {"function": {"name": "no_args", "arguments": "invalid{json"}}

        result = executor.execute(tool_call)

        assert "ok" in result["content"]

    def test_execute_tool_not_found(self):
        """Test executing non-existent tool"""
        executor = ToolExecutor()

        tool_call = {"function": {"name": "nonexistent", "arguments": {}}}

        result = executor.execute(tool_call)

        # Message format is: "Tool nonexistent not found in executor"
        assert "not found" in result["content"].lower()
        assert "nonexistent" in result["content"].lower()

    def test_execute_tool_raises_exception(self):
        """Test executing tool that raises exception"""

        def failing_tool():
            raise ValueError("Tool failed")

        executor = ToolExecutor()
        executor.register("failing", failing_tool)

        tool_call = {"function": {"name": "failing", "arguments": {}}}

        result = executor.execute(tool_call)

        assert "error" in result["content"].lower()
        assert "failed" in result["content"].lower()

    def test_execute_non_string_result(self):
        """Test executing tool that returns non-string"""

        def dict_tool():
            return {"key": "value", "number": 42}

        executor = ToolExecutor()
        executor.register("dict_tool", dict_tool)

        tool_call = {"function": {"name": "dict_tool", "arguments": {}}}

        result = executor.execute(tool_call)

        # Result should be JSON-serialized
        assert '"key": "value"' in result["content"]

    def test_execute_with_default_id(self):
        """Test executing without ID generates default"""

        def tool():
            return "ok"

        executor = ToolExecutor()
        executor.register("tool", tool)

        tool_call = {"name": "tool", "arguments": {}}

        result = executor.execute(tool_call)

        assert "call_tool" in result["tool_call_id"]

    def test_execution_log(self):
        """Test that executions are logged"""

        def tool1():
            return "result1"

        def tool2():
            return "result2"

        executor = ToolExecutor()
        executor.register("tool1", tool1)
        executor.register("tool2", tool2)

        executor.execute({"function": {"name": "tool1", "arguments": {}}})
        executor.execute({"function": {"name": "tool2", "arguments": {}}})

        assert len(executor.execution_log) == 2

    def test_execute_with_malformed_tool_call(self):
        """Test executing with malformed tool call raises exception"""
        executor = ToolExecutor()

        def tool():
            return "ok"

        executor.register("tool", tool)

        # Malformed tool call that will cause an exception during processing
        tool_call = {"function": {"name": "tool", "arguments": "not_valid_json"}}

        result = executor.execute(tool_call)

        # Should handle the exception gracefully
        assert result["role"] == "tool"
        assert "tool_call_id" in result

    def test_get_execution_summary_empty(self):
        """Test execution summary with no executions"""
        executor = ToolExecutor()

        summary = executor.get_execution_summary()

        assert summary == "No tools executed"

    def test_get_execution_summary_with_failures(self):
        """Test execution summary with both success and failures"""
        executor = ToolExecutor()

        def success_tool():
            return "Success result"

        def failing_tool():
            raise ValueError("Tool failed")

        executor.register("success", success_tool)
        executor.register("failing", failing_tool)

        # Execute both
        executor.execute({"function": {"name": "success", "arguments": {}}})
        executor.execute({"function": {"name": "failing", "arguments": {}}})

        summary = executor.get_execution_summary()

        assert "✓ success" in summary
        assert "✗ failing" in summary


class TestExecuteWithTools:
    """Test cases for execute_with_tools function"""

    @pytest.mark.asyncio
    async def test_execute_with_tools_basic(self):
        """Test basic tool execution flow"""

        def get_weather(location: str):
            return f"Sunny in {location}"

        # Mock ask function
        async def mock_ask(prompt, **kwargs):
            if "tool_calls" not in kwargs.get("messages", [{}])[-1]:
                # First call - return tool calls
                return {
                    "response": "Let me check",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {
                                "name": "get_weather",
                                "arguments": {"location": "Paris"},
                            },
                        }
                    ],
                }
            else:
                # Second call - return final response
                return {"response": "The weather is sunny!", "tool_calls": []}

        tools = [{"type": "function", "function": {"name": "get_weather"}}]
        functions = {"get_weather": get_weather}

        result = await execute_with_tools(
            mock_ask, "What's the weather?", tools, functions
        )

        assert "sunny" in result.lower()

    @pytest.mark.asyncio
    async def test_execute_with_tools_max_iterations(self):
        """Test max iterations limit"""

        async def mock_ask_infinite(prompt, **kwargs):
            # Always return tool calls
            return {
                "response": "More tools",
                "tool_calls": [
                    {"id": "call_1", "function": {"name": "tool", "arguments": {}}}
                ],
            }

        def dummy_tool():
            return "result"

        tools = [{"type": "function", "function": {"name": "tool"}}]
        functions = {"tool": dummy_tool}

        result = await execute_with_tools(
            mock_ask_infinite, "prompt", tools, functions, max_iterations=2
        )

        # Should stop after max iterations
        assert result is not None

    @pytest.mark.asyncio
    async def test_execute_with_tools_no_tool_calls(self):
        """Test when no tool calls are needed"""

        async def mock_ask(prompt, **kwargs):
            return {"response": "Direct answer", "tool_calls": []}

        result = await execute_with_tools(mock_ask, "Simple question", [], {})

        assert result == "Direct answer"

    @pytest.mark.asyncio
    async def test_execute_with_tools_string_response(self):
        """Test when response is a string instead of dict"""

        async def mock_ask(prompt, **kwargs):
            return "String response"

        result = await execute_with_tools(mock_ask, "Question", [], {})

        assert result == "String response"

    @pytest.mark.asyncio
    async def test_execute_with_tools_no_results_from_tools(self):
        """Test when tools execute but produce no results"""

        async def mock_ask(prompt, **kwargs):
            # Return empty tool results
            return {"response": "", "tool_calls": []}

        def dummy_tool():
            return ""

        tools = [{"type": "function", "function": {"name": "dummy"}}]
        functions = {"dummy": dummy_tool}

        result = await execute_with_tools(mock_ask, "Test", tools, functions)

        # Should handle empty results gracefully
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_execute_with_tools_exhausted_rounds(self):
        """Test when max rounds are exhausted with tool executions"""
        call_count = {"count": 0}

        async def mock_ask_always_tools(prompt, **kwargs):
            call_count["count"] += 1
            return {
                "response": "",
                "tool_calls": [
                    {"id": f"call_{call_count['count']}", "function": {"name": "tool", "arguments": {}}}
                ],
            }

        def tool():
            return "result"

        tools = [{"type": "function", "function": {"name": "tool"}}]
        functions = {"tool": tool}

        result = await execute_with_tools(
            mock_ask_always_tools, "Test", tools, functions, max_rounds=2
        )

        # Should return summary after exhausting rounds
        assert "Tool execution completed" in result or "Executed tools" in result


class TestExecuteWithToolsSync:
    """Test cases for execute_with_tools_sync function"""

    def test_execute_with_tools_sync_basic(self):
        """Test synchronous tool execution"""

        def get_data():
            return "data"

        # Mock ask_sync function
        def mock_ask_sync(prompt, **kwargs):
            if "tool_calls" not in kwargs.get("messages", [{}])[-1]:
                return {
                    "response": "Getting data",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {"name": "get_data", "arguments": {}},
                        }
                    ],
                }
            else:
                return {"response": "Here's the data!", "tool_calls": []}

        tools = [{"type": "function", "function": {"name": "get_data"}}]
        functions = {"get_data": get_data}

        result = execute_with_tools_sync(
            mock_ask_sync, "Get me data", tools, functions
        )

        assert "data" in result.lower()

    def test_execute_with_tools_sync_no_tools(self):
        """Test sync execution without tools"""

        def mock_ask_sync(prompt, **kwargs):
            return {"response": "No tools needed", "tool_calls": []}

        result = execute_with_tools_sync(mock_ask_sync, "Question", [], {})

        assert result == "No tools needed"

    def test_execute_with_tools_sync_max_iterations(self):
        """Test sync max iterations"""

        def mock_ask_sync_infinite(prompt, **kwargs):
            return {
                "response": "More",
                "tool_calls": [
                    {"id": "call_1", "function": {"name": "tool", "arguments": {}}}
                ],
            }

        def dummy():
            return "ok"

        tools = [{"type": "function", "function": {"name": "tool"}}]

        result = execute_with_tools_sync(
            mock_ask_sync_infinite, "Q", tools, {"tool": dummy}, max_iterations=3
        )

        assert result is not None
