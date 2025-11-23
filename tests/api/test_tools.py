"""
Comprehensive tests for tools module
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from chuk_llm.api.tools import (
    Tool,
    ToolKit,
    Tools,
    create_tool,
    tool,
    tools_from_functions,
    _python_type_to_json_schema,
)


class TestTool:
    """Test cases for Tool class"""

    def test_tool_creation(self):
        """Test creating a Tool"""
        t = Tool(
            name="test_tool",
            description="A test tool",
            parameters={"type": "object"},
        )

        assert t.name == "test_tool"
        assert t.description == "A test tool"
        assert t.parameters == {"type": "object"}
        assert t.func is None

    def test_tool_with_function(self):
        """Test Tool with function"""

        def my_func():
            return "result"

        t = Tool(name="func_tool", description="Tool with func", func=my_func)

        assert t.func == my_func

    def test_to_openai_format(self):
        """Test converting Tool to OpenAI format"""
        t = Tool(
            name="weather",
            description="Get weather",
            parameters={"type": "object", "properties": {"location": {"type": "string"}}},
        )

        openai_format = t.to_openai_format()

        assert openai_format["type"] == "function"
        assert openai_format["function"]["name"] == "weather"
        assert openai_format["function"]["description"] == "Get weather"
        assert openai_format["function"]["parameters"]["type"] == "object"

    def test_execute_with_function(self):
        """Test executing a tool with function"""

        def add(a: int, b: int):
            return a + b

        t = Tool(name="add", description="Add numbers", func=add)

        result = t.execute(a=5, b=3)

        assert result == 8

    def test_execute_without_function(self):
        """Test executing tool without function raises error"""
        t = Tool(name="no_func", description="No implementation")

        with pytest.raises(NotImplementedError):
            t.execute()


class TestToolKit:
    """Test cases for ToolKit class"""

    def test_toolkit_creation(self):
        """Test creating a ToolKit"""
        tk = ToolKit(name="my_tools")

        assert tk.name == "my_tools"
        assert isinstance(tk.tools, dict)
        assert len(tk.tools) == 0

    def test_toolkit_default_name(self):
        """Test ToolKit with default name"""
        tk = ToolKit()

        assert tk.name == "default"

    def test_add_tool(self):
        """Test adding a tool to toolkit"""
        tk = ToolKit()
        tool = Tool(name="test", description="Test tool")

        tk.add(tool)

        assert "test" in tk.tools
        assert tk.tools["test"] == tool

    def test_add_function(self):
        """Test adding function as tool"""

        def multiply(x: int, y: int) -> int:
            """Multiply two numbers"""
            return x * y

        tk = ToolKit()
        tk.add_function(multiply)

        assert "multiply" in tk.tools
        assert tk.tools["multiply"].func == multiply
        assert "Multiply two numbers" in tk.tools["multiply"].description

    def test_add_function_with_custom_name(self):
        """Test adding function with custom name"""

        def func():
            return "ok"

        tk = ToolKit()
        tk.add_function(func, name="custom_name")

        assert "custom_name" in tk.tools
        assert "func" not in tk.tools

    def test_add_function_with_custom_description(self):
        """Test adding function with custom description"""

        def func():
            pass

        tk = ToolKit()
        tk.add_function(func, description="Custom description")

        assert tk.tools["func"].description == "Custom description"

    def test_add_function_generates_parameters(self):
        """Test that add_function generates parameters from signature"""

        def process(name: str, age: int, active: bool = True):
            return f"{name} is {age}"

        tk = ToolKit()
        tk.add_function(process)

        tool = tk.tools["process"]
        params = tool.parameters

        assert params["type"] == "object"
        assert "name" in params["properties"]
        assert "age" in params["properties"]
        assert "active" in params["properties"]

        assert params["properties"]["name"]["type"] == "string"
        assert params["properties"]["age"]["type"] == "integer"
        assert params["properties"]["active"]["type"] == "boolean"

        # name and age are required (no defaults), active is optional
        assert "name" in params.get("required", [])
        assert "age" in params.get("required", [])
        assert "active" not in params.get("required", [])

    def test_add_function_without_docstring(self):
        """Test adding function without docstring"""

        def no_doc():
            return "ok"

        tk = ToolKit()
        tk.add_function(no_doc)

        assert "Function no_doc" in tk.tools["no_doc"].description

    def test_to_openai_format(self):
        """Test converting toolkit to OpenAI format"""
        tk = ToolKit()

        def tool1():
            return "1"

        def tool2():
            return "2"

        tk.add_function(tool1)
        tk.add_function(tool2)

        openai_format = tk.to_openai_format()

        assert len(openai_format) == 2
        assert all(t["type"] == "function" for t in openai_format)

    def test_execute_tool(self):
        """Test executing a tool by name"""

        def calculator(operation: str, a: int, b: int):
            if operation == "add":
                return a + b
            return 0

        tk = ToolKit()
        tk.add_function(calculator)

        result = tk.execute("calculator", operation="add", a=10, b=5)

        assert result == 15

    def test_execute_nonexistent_tool(self):
        """Test executing non-existent tool raises error"""
        tk = ToolKit()

        with pytest.raises(ValueError, match="not found"):
            tk.execute("nonexistent")

    @pytest.mark.asyncio
    async def test_ask_with_auto_execute(self):
        """Test ask with auto_execute=True"""
        tk = ToolKit()

        def get_answer():
            return "42"

        tk.add_function(get_answer)

        with patch("chuk_llm.api.tool_execution.execute_with_tools") as mock_execute:
            mock_execute.return_value = "The answer is 42"

            result = await tk.ask("What is the answer?", auto_execute=True)

            assert result == "The answer is 42"
            mock_execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_ask_without_auto_execute(self):
        """Test ask with auto_execute=False"""
        tk = ToolKit()

        def dummy():
            return "ok"

        tk.add_function(dummy)

        with patch("chuk_llm.api.tools.ask") as mock_ask:
            mock_ask.return_value = {
                "response": "Use the tool",
                "tool_calls": [{"name": "dummy"}],
            }

            result = await tk.ask("Question", auto_execute=False)

            assert result == "Use the tool"

    @pytest.mark.asyncio
    async def test_ask_returns_string_response(self):
        """Test ask when result is already a string"""
        tk = ToolKit()

        with patch("chuk_llm.api.tools.ask") as mock_ask:
            mock_ask.return_value = "Direct string response"

            result = await tk.ask("Question", auto_execute=False)

            assert result == "Direct string response"

    def test_ask_sync_with_auto_execute(self):
        """Test ask_sync with auto_execute=True"""
        tk = ToolKit()

        def tool():
            return "result"

        tk.add_function(tool)

        with patch("chuk_llm.api.tool_execution.execute_with_tools_sync") as mock_execute:
            mock_execute.return_value = "Executed result"

            result = tk.ask_sync("Prompt", auto_execute=True)

            assert result == "Executed result"

    def test_ask_sync_without_auto_execute(self):
        """Test ask_sync with auto_execute=False"""
        tk = ToolKit()

        with patch("chuk_llm.api.tools.ask_sync") as mock_ask:
            mock_ask.return_value = {"response": "Response", "tool_calls": []}

            result = tk.ask_sync("Prompt", auto_execute=False)

            assert result == "Response"


class TestToolDecorator:
    """Test cases for @tool decorator"""

    def test_tool_decorator_basic(self):
        """Test basic tool decorator"""

        @tool(description="Get weather info")
        def get_weather(location: str):
            return f"Weather in {location}"

        assert hasattr(get_weather, "_is_tool")
        assert get_weather._is_tool is True
        assert get_weather._tool_name == "get_weather"
        assert get_weather._tool_description == "Get weather info"

    def test_tool_decorator_with_custom_name(self):
        """Test tool decorator with custom name"""

        @tool(name="custom_name", description="Custom tool")
        def func():
            pass

        assert func._tool_name == "custom_name"

    def test_tool_decorator_without_description(self):
        """Test tool decorator uses docstring"""

        @tool()
        def documented():
            """This is the docstring"""
            pass

        assert "This is the docstring" in documented._tool_description

    def test_tool_decorator_without_docstring(self):
        """Test tool decorator fallback without docstring"""

        @tool()
        def no_docs():
            pass

        assert "Function no_docs" in no_docs._tool_description

    def test_decorated_function_still_callable(self):
        """Test that decorated function still works"""

        @tool(description="Add numbers")
        def add(a: int, b: int):
            return a + b

        result = add(3, 4)

        assert result == 7


class TestToolsClass:
    """Test cases for Tools base class"""

    def test_tools_class_registration(self):
        """Test that Tools class auto-registers decorated methods"""

        class MyTools(Tools):
            @tool(description="Get data")
            def get_data(self):
                return "data"

            @tool()
            def process(self, value: str):
                return f"Processed: {value}"

        tools = MyTools()

        assert "get_data" in tools.toolkit.tools
        assert "process" in tools.toolkit.tools

    def test_tools_class_ignores_non_decorated(self):
        """Test that non-decorated methods are ignored"""

        class MyTools(Tools):
            @tool()
            def decorated(self):
                return "ok"

            def not_decorated(self):
                return "skip"

        tools = MyTools()

        assert "decorated" in tools.toolkit.tools
        assert "not_decorated" not in tools.toolkit.tools

    @pytest.mark.asyncio
    async def test_tools_ask(self):
        """Test Tools.ask method"""

        class MyTools(Tools):
            @tool()
            def helper(self):
                return "help"

        tools = MyTools()

        with patch.object(tools.toolkit, "ask") as mock_ask:
            mock_ask.return_value = "Result"

            result = await tools.ask("Question")

            assert result == "Result"

    def test_tools_ask_sync(self):
        """Test Tools.ask_sync method"""

        class MyTools(Tools):
            @tool()
            def helper(self):
                return "help"

        tools = MyTools()

        with patch.object(tools.toolkit, "ask_sync") as mock_ask_sync:
            mock_ask_sync.return_value = "Sync result"

            result = tools.ask_sync("Question")

            assert result == "Sync result"

    def test_tools_property(self):
        """Test Tools.tools property"""

        class MyTools(Tools):
            @tool()
            def tool1(self):
                pass

        tools = MyTools()
        tools_list = tools.tools

        assert isinstance(tools_list, list)
        assert len(tools_list) > 0
        assert all(t["type"] == "function" for t in tools_list)


class TestHelperFunctions:
    """Test cases for helper functions"""

    def test_create_tool(self):
        """Test create_tool helper"""
        tool = create_tool(
            name="test",
            description="Test tool",
            parameters={"type": "object", "properties": {}},
        )

        assert isinstance(tool, Tool)
        assert tool.name == "test"
        assert tool.description == "Test tool"

    def test_create_tool_with_function(self):
        """Test create_tool with function"""

        def func():
            return "ok"

        tool = create_tool(
            name="func_tool", description="Tool", parameters={}, func=func
        )

        assert tool.func == func

    def test_tools_from_functions(self):
        """Test tools_from_functions helper"""

        def func1(a: int):
            return a * 2

        def func2(b: str):
            return b.upper()

        toolkit = tools_from_functions(func1, func2)

        assert isinstance(toolkit, ToolKit)
        assert "func1" in toolkit.tools
        assert "func2" in toolkit.tools

    def test_tools_from_functions_empty(self):
        """Test tools_from_functions with no functions"""
        toolkit = tools_from_functions()

        assert isinstance(toolkit, ToolKit)
        assert len(toolkit.tools) == 0

    def test_python_type_to_json_schema_basic_types(self):
        """Test type conversion for basic types"""
        assert _python_type_to_json_schema(str) == "string"
        assert _python_type_to_json_schema(int) == "integer"
        assert _python_type_to_json_schema(float) == "number"
        assert _python_type_to_json_schema(bool) == "boolean"
        assert _python_type_to_json_schema(dict) == "object"
        assert _python_type_to_json_schema(list) == "array"

    def test_python_type_to_json_schema_any_type(self):
        """Test type conversion for Any"""
        from typing import Any

        assert _python_type_to_json_schema(Any) == "string"

    def test_python_type_to_json_schema_optional(self):
        """Test type conversion for Optional types"""
        from typing import Optional

        # Optional[str] should return 'string'
        result = _python_type_to_json_schema(Optional[str])
        assert result == "string"

    def test_python_type_to_json_schema_unknown_type(self):
        """Test type conversion for unknown types defaults to string"""

        class CustomType:
            pass

        assert _python_type_to_json_schema(CustomType) == "string"
