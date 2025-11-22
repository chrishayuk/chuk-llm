"""
Comprehensive tests for tool_executor module
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from chuk_llm.api.tool_executor import ToolExecutor, execute_tool_calls
from chuk_llm.core.constants import ResponseKey, ToolParam
from chuk_llm.core.enums import MessageRole


class TestToolExecutor:
    """Test cases for ToolExecutor class"""

    def test_init(self):
        """Test ToolExecutor initialization"""
        executor = ToolExecutor()
        assert isinstance(executor.tools, dict)
        assert len(executor.tools) == 0

    def test_register(self):
        """Test registering a single tool"""
        executor = ToolExecutor()

        def sample_func():
            return "result"

        executor.register("sample", sample_func)
        assert "sample" in executor.tools
        assert executor.tools["sample"] == sample_func

    def test_register_multiple_with_functions(self):
        """Test registering multiple tools with _func attribute"""
        executor = ToolExecutor()

        # Create mock tools with _func attribute
        def func1():
            return "result1"

        def func2():
            return "result2"

        # Create a custom class that acts like a dict but has _func
        class ToolWithFunc(dict):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._func = None

        tool1 = ToolWithFunc({
            ToolParam.TYPE.value: ToolParam.FUNCTION.value,
            ToolParam.FUNCTION.value: {ToolParam.NAME.value: "tool1"},
        })
        tool1._func = func1

        tool2 = ToolWithFunc({
            ToolParam.TYPE.value: ToolParam.FUNCTION.value,
            ToolParam.FUNCTION.value: {ToolParam.NAME.value: "tool2"},
        })
        tool2._func = func2

        executor.register_multiple([tool1, tool2])

        assert "tool1" in executor.tools
        assert "tool2" in executor.tools

    def test_register_multiple_without_func(self):
        """Test registering tools without _func attribute (should not register)"""
        executor = ToolExecutor()

        tools = [
            {
                ToolParam.TYPE.value: ToolParam.FUNCTION.value,
                ToolParam.FUNCTION.value: {ToolParam.NAME.value: "tool1"},
            }
        ]

        executor.register_multiple(tools)
        # Should not register tools without _func attribute
        assert len(executor.tools) == 0

    def test_register_multiple_non_function_type(self):
        """Test registering tools with wrong type (should skip)"""
        executor = ToolExecutor()

        tools = [
            {
                ToolParam.TYPE.value: "not_function",
                ToolParam.FUNCTION.value: {ToolParam.NAME.value: "tool1"},
            }
        ]

        executor.register_multiple(tools)
        assert len(executor.tools) == 0

    @pytest.mark.asyncio
    async def test_execute_sync_function(self):
        """Test executing a synchronous function"""
        executor = ToolExecutor()

        def add(a: int, b: int) -> int:
            return a + b

        executor.register("add", add)

        tool_call = {
            ToolParam.ID.value: "call_123",
            ToolParam.TYPE.value: ToolParam.FUNCTION.value,
            ToolParam.FUNCTION.value: {
                ToolParam.NAME.value: "add",
                ToolParam.ARGUMENTS.value: {"a": 5, "b": 3},
            },
        }

        result = await executor.execute(tool_call)

        assert result[ToolParam.TOOL_CALL_ID.value] == "call_123"
        assert result[ToolParam.NAME.value] == "add"
        assert result[ResponseKey.RESULT.value] == 8

    @pytest.mark.asyncio
    async def test_execute_async_function(self):
        """Test executing an async function"""
        executor = ToolExecutor()

        async def async_add(a: int, b: int) -> int:
            await asyncio.sleep(0.01)
            return a + b

        executor.register("async_add", async_add)

        tool_call = {
            ToolParam.ID.value: "call_456",
            ToolParam.TYPE.value: ToolParam.FUNCTION.value,
            ToolParam.FUNCTION.value: {
                ToolParam.NAME.value: "async_add",
                ToolParam.ARGUMENTS.value: {"a": 10, "b": 20},
            },
        }

        result = await executor.execute(tool_call)

        assert result[ToolParam.TOOL_CALL_ID.value] == "call_456"
        assert result[ToolParam.NAME.value] == "async_add"
        assert result[ResponseKey.RESULT.value] == 30

    @pytest.mark.asyncio
    async def test_execute_with_string_arguments(self):
        """Test executing with JSON string arguments"""
        executor = ToolExecutor()

        def multiply(x: int, y: int) -> int:
            return x * y

        executor.register("multiply", multiply)

        tool_call = {
            ToolParam.ID.value: "call_789",
            ToolParam.TYPE.value: ToolParam.FUNCTION.value,
            ToolParam.FUNCTION.value: {
                ToolParam.NAME.value: "multiply",
                ToolParam.ARGUMENTS.value: json.dumps({"x": 4, "y": 5}),
            },
        }

        result = await executor.execute(tool_call)

        assert result[ResponseKey.RESULT.value] == 20

    @pytest.mark.asyncio
    async def test_execute_with_invalid_json_arguments(self):
        """Test executing with invalid JSON arguments"""
        executor = ToolExecutor()

        def no_args_func() -> str:
            return "success"

        executor.register("no_args", no_args_func)

        tool_call = {
            ToolParam.ID.value: "call_invalid",
            ToolParam.TYPE.value: ToolParam.FUNCTION.value,
            ToolParam.FUNCTION.value: {
                ToolParam.NAME.value: "no_args",
                ToolParam.ARGUMENTS.value: "invalid json {",
            },
        }

        result = await executor.execute(tool_call)

        # Should use empty dict for invalid JSON and succeed if func takes no args
        assert result[ResponseKey.RESULT.value] == "success"

    @pytest.mark.asyncio
    async def test_execute_tool_not_found(self):
        """Test executing a tool that doesn't exist"""
        executor = ToolExecutor()

        tool_call = {
            ToolParam.ID.value: "call_notfound",
            ToolParam.TYPE.value: ToolParam.FUNCTION.value,
            ToolParam.FUNCTION.value: {
                ToolParam.NAME.value: "nonexistent",
                ToolParam.ARGUMENTS.value: {},
            },
        }

        result = await executor.execute(tool_call)

        assert result[ToolParam.TOOL_CALL_ID.value] == "call_notfound"
        assert result[ToolParam.NAME.value] == "nonexistent"
        assert ResponseKey.ERROR.value in result
        assert "not registered" in result[ResponseKey.ERROR.value]

    @pytest.mark.asyncio
    async def test_execute_with_exception(self):
        """Test executing a function that raises an exception"""
        executor = ToolExecutor()

        def failing_func():
            raise ValueError("Test error")

        executor.register("failing", failing_func)

        tool_call = {
            ToolParam.ID.value: "call_fail",
            ToolParam.TYPE.value: ToolParam.FUNCTION.value,
            ToolParam.FUNCTION.value: {
                ToolParam.NAME.value: "failing",
                ToolParam.ARGUMENTS.value: {},
            },
        }

        result = await executor.execute(tool_call)

        assert result[ToolParam.TOOL_CALL_ID.value] == "call_fail"
        assert ResponseKey.ERROR.value in result
        assert "Test error" in result[ResponseKey.ERROR.value]

    @pytest.mark.asyncio
    async def test_execute_without_type_field(self):
        """Test executing tool call without type field (direct function format)"""
        executor = ToolExecutor()

        def simple_func() -> str:
            return "direct call"

        executor.register("simple", simple_func)

        # Tool call without type field
        tool_call = {
            ToolParam.ID.value: "call_direct",
            ToolParam.NAME.value: "simple",
            ToolParam.ARGUMENTS.value: {},
        }

        result = await executor.execute(tool_call)

        assert result[ResponseKey.RESULT.value] == "direct call"

    @pytest.mark.asyncio
    async def test_execute_with_missing_id(self):
        """Test executing tool call without ID"""
        executor = ToolExecutor()

        def func():
            return "result"

        executor.register("func", func)

        tool_call = {
            ToolParam.TYPE.value: ToolParam.FUNCTION.value,
            ToolParam.FUNCTION.value: {
                ToolParam.NAME.value: "func",
                ToolParam.ARGUMENTS.value: {},
            },
        }

        result = await executor.execute(tool_call)

        assert result[ToolParam.TOOL_CALL_ID.value] == "unknown"
        assert result[ResponseKey.RESULT.value] == "result"

    @pytest.mark.asyncio
    async def test_execute_all(self):
        """Test executing multiple tool calls in parallel"""
        executor = ToolExecutor()

        def add(a: int, b: int) -> int:
            return a + b

        def multiply(x: int, y: int) -> int:
            return x * y

        executor.register("add", add)
        executor.register("multiply", multiply)

        tool_calls = [
            {
                ToolParam.ID.value: "call_1",
                ToolParam.TYPE.value: ToolParam.FUNCTION.value,
                ToolParam.FUNCTION.value: {
                    ToolParam.NAME.value: "add",
                    ToolParam.ARGUMENTS.value: {"a": 1, "b": 2},
                },
            },
            {
                ToolParam.ID.value: "call_2",
                ToolParam.TYPE.value: ToolParam.FUNCTION.value,
                ToolParam.FUNCTION.value: {
                    ToolParam.NAME.value: "multiply",
                    ToolParam.ARGUMENTS.value: {"x": 3, "y": 4},
                },
            },
        ]

        results = await executor.execute_all(tool_calls)

        assert len(results) == 2
        assert results[0][ResponseKey.RESULT.value] == 3
        assert results[1][ResponseKey.RESULT.value] == 12

    @pytest.mark.asyncio
    async def test_execute_all_with_exception(self):
        """Test execute_all with one call raising exception"""
        executor = ToolExecutor()

        def success_func():
            return "success"

        def fail_func():
            raise RuntimeError("Failed")

        executor.register("success", success_func)
        executor.register("fail", fail_func)

        tool_calls = [
            {
                ToolParam.ID.value: "call_success",
                ToolParam.TYPE.value: ToolParam.FUNCTION.value,
                ToolParam.FUNCTION.value: {
                    ToolParam.NAME.value: "success",
                    ToolParam.ARGUMENTS.value: {},
                },
            },
            {
                ToolParam.ID.value: "call_fail",
                ToolParam.TYPE.value: ToolParam.FUNCTION.value,
                ToolParam.FUNCTION.value: {
                    ToolParam.NAME.value: "fail",
                    ToolParam.ARGUMENTS.value: {},
                },
            },
        ]

        results = await executor.execute_all(tool_calls)

        assert len(results) == 2
        assert results[0][ResponseKey.RESULT.value] == "success"
        assert ResponseKey.ERROR.value in results[1]
        assert "Failed" in results[1][ResponseKey.ERROR.value]

    @pytest.mark.asyncio
    async def test_execute_all_empty_list(self):
        """Test execute_all with empty tool calls list"""
        executor = ToolExecutor()
        results = await executor.execute_all([])
        assert results == []


class TestExecuteToolCalls:
    """Test cases for execute_tool_calls helper function"""

    @pytest.mark.asyncio
    async def test_execute_tool_calls_no_tools(self):
        """Test when response has no tool calls"""
        response = {"response": "Hello", ResponseKey.TOOL_CALLS.value: []}
        result = await execute_tool_calls(response, [])
        assert result == "Hello"

    @pytest.mark.asyncio
    async def test_execute_tool_calls_with_tool_functions(self):
        """Test executing tool calls with provided functions"""

        def get_weather(location: str) -> str:
            return f"Weather in {location}: Sunny"

        response = {
            "response": "Let me check the weather",
            ResponseKey.TOOL_CALLS.value: [
                {
                    ToolParam.ID.value: "call_weather",
                    ToolParam.TYPE.value: ToolParam.FUNCTION.value,
                    ToolParam.FUNCTION.value: {
                        ToolParam.NAME.value: "get_weather",
                        ToolParam.ARGUMENTS.value: {"location": "London"},
                    },
                }
            ],
        }

        tools = [
            {
                ToolParam.TYPE.value: ToolParam.FUNCTION.value,
                ToolParam.FUNCTION.value: {ToolParam.NAME.value: "get_weather"},
            }
        ]

        result = await execute_tool_calls(
            response, tools, tool_functions={"get_weather": get_weather}
        )

        assert "Let me check the weather" in result
        assert "Weather in London: Sunny" in result

    @pytest.mark.asyncio
    async def test_execute_tool_calls_with_tool_objects(self):
        """Test executing tool calls with tool objects having _func"""

        def calculate(x: int, y: int) -> int:
            return x + y

        class MockTool:
            def __init__(self):
                self.name = "calculate"
                self._func = calculate

        tool_obj = MockTool()

        response = {
            ResponseKey.TOOL_CALLS.value: [
                {
                    ToolParam.ID.value: "call_calc",
                    ToolParam.TYPE.value: ToolParam.FUNCTION.value,
                    ToolParam.FUNCTION.value: {
                        ToolParam.NAME.value: "calculate",
                        ToolParam.ARGUMENTS.value: {"x": 5, "y": 3},
                    },
                }
            ]
        }

        result = await execute_tool_calls(response, [tool_obj])

        assert "Result: 8" in result

    @pytest.mark.asyncio
    async def test_execute_tool_calls_with_error(self):
        """Test executing tool calls when tool returns error"""

        def failing_tool():
            raise ValueError("Tool failed")

        response = {
            ResponseKey.TOOL_CALLS.value: [
                {
                    ToolParam.ID.value: "call_fail",
                    ToolParam.TYPE.value: ToolParam.FUNCTION.value,
                    ToolParam.FUNCTION.value: {
                        ToolParam.NAME.value: "failing_tool",
                        ToolParam.ARGUMENTS.value: {},
                    },
                }
            ]
        }

        tools = []

        result = await execute_tool_calls(
            response, tools, tool_functions={"failing_tool": failing_tool}
        )

        assert "Error calling" in result
        assert "Tool failed" in result

    @pytest.mark.asyncio
    async def test_execute_tool_calls_no_response_field(self):
        """Test when response dict has no 'response' field"""
        response = {
            ResponseKey.TOOL_CALLS.value: [
                {
                    ToolParam.ID.value: "call_test",
                    ToolParam.TYPE.value: ToolParam.FUNCTION.value,
                    ToolParam.FUNCTION.value: {
                        ToolParam.NAME.value: "test_func",
                        ToolParam.ARGUMENTS.value: {},
                    },
                }
            ]
        }

        def test_func():
            return "test result"

        result = await execute_tool_calls(
            response, [], tool_functions={"test_func": test_func}
        )

        assert "Result: test result" in result

    @pytest.mark.asyncio
    async def test_execute_tool_calls_empty_results(self):
        """Test when tool execution returns empty results"""
        response = {"response": "Initial response"}

        result = await execute_tool_calls(response, [])

        assert result == "Initial response"

    @pytest.mark.asyncio
    async def test_execute_tool_calls_multiple_tools(self):
        """Test executing multiple tool calls"""

        def tool1():
            return "result1"

        def tool2():
            return "result2"

        response = {
            "response": "Starting",
            ResponseKey.TOOL_CALLS.value: [
                {
                    ToolParam.ID.value: "call_1",
                    ToolParam.TYPE.value: ToolParam.FUNCTION.value,
                    ToolParam.FUNCTION.value: {
                        ToolParam.NAME.value: "tool1",
                        ToolParam.ARGUMENTS.value: {},
                    },
                },
                {
                    ToolParam.ID.value: "call_2",
                    ToolParam.TYPE.value: ToolParam.FUNCTION.value,
                    ToolParam.FUNCTION.value: {
                        ToolParam.NAME.value: "tool2",
                        ToolParam.ARGUMENTS.value: {},
                    },
                },
            ],
        }

        result = await execute_tool_calls(
            response, [], tool_functions={"tool1": tool1, "tool2": tool2}
        )

        assert "Starting" in result
        assert "result1" in result
        assert "result2" in result

    @pytest.mark.asyncio
    async def test_execute_tool_calls_no_initial_response(self):
        """Test when there's no initial response, only tool results"""
        response = {
            ResponseKey.TOOL_CALLS.value: [
                {
                    ToolParam.ID.value: "call_1",
                    ToolParam.TYPE.value: ToolParam.FUNCTION.value,
                    ToolParam.FUNCTION.value: {
                        ToolParam.NAME.value: "get_data",
                        ToolParam.ARGUMENTS.value: {},
                    },
                }
            ]
        }

        def get_data():
            return "data"

        result = await execute_tool_calls(
            response, [], tool_functions={"get_data": get_data}
        )

        assert "Result: data" in result

    @pytest.mark.asyncio
    async def test_execute_tool_calls_empty_result_value(self):
        """Test when tool result is empty/falsy"""

        def empty_tool():
            return ""

        response = {
            ResponseKey.TOOL_CALLS.value: [
                {
                    ToolParam.ID.value: "call_empty",
                    ToolParam.TYPE.value: ToolParam.FUNCTION.value,
                    ToolParam.FUNCTION.value: {
                        ToolParam.NAME.value: "empty_tool",
                        ToolParam.ARGUMENTS.value: {},
                    },
                }
            ]
        }

        result = await execute_tool_calls(
            response, [], tool_functions={"empty_tool": empty_tool}
        )

        # Should not include empty results
        assert result == "" or "Tool calls executed" in result
