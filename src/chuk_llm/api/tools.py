"""
Clean, developer-friendly API for function/tool calling
"""

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Optional, get_type_hints

from .core import ask
from .sync import ask_sync


@dataclass
class Tool:
    """Clean tool definition"""

    name: str
    description: str
    func: Callable | None = None
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_openai_format(self) -> dict[str, Any]:
        """Convert to OpenAI tool format"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def execute(self, **kwargs) -> Any:
        """Execute the tool function if available"""
        if self.func:
            return self.func(**kwargs)
        raise NotImplementedError(f"Tool {self.name} has no implementation")


class ToolKit:
    """Collection of tools for easy management"""

    def __init__(self, name: str = "default"):
        self.name = name
        self.tools: dict[str, Tool] = {}

    def add(self, tool: Tool) -> None:
        """Add a tool to the kit"""
        self.tools[tool.name] = tool

    def add_function(
        self, func: Callable, name: str | None = None, description: str | None = None
    ) -> None:
        """Add a function as a tool with auto-detection"""
        tool_name = name or func.__name__
        tool_desc = description or func.__doc__ or f"Function {tool_name}"

        # Auto-generate parameters from function signature
        sig = inspect.signature(func)
        type_hints = get_type_hints(func)

        properties = {}
        required = []

        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue

            param_type = type_hints.get(param_name, Any)
            param_info = {"type": _python_type_to_json_schema(param_type)}

            # Add description from docstring if available
            if func.__doc__:
                # Simple docstring parsing (could be enhanced)
                param_info["description"] = f"Parameter {param_name}"

            properties[param_name] = param_info

            if param.default == inspect.Parameter.empty:
                required.append(param_name)

        parameters = {"type": "object", "properties": properties}
        if required:
            parameters["required"] = required

        tool = Tool(
            name=tool_name,
            description=tool_desc.strip(),
            func=func,
            parameters=parameters,
        )
        self.add(tool)

    def to_openai_format(self) -> list[dict[str, Any]]:
        """Convert all tools to OpenAI format"""
        return [tool.to_openai_format() for tool in self.tools.values()]

    def execute(self, tool_name: str, **kwargs) -> Any:
        """Execute a tool by name"""
        if tool_name not in self.tools:
            raise ValueError(f"Tool {tool_name} not found")
        return self.tools[tool_name].execute(**kwargs)

    async def ask(self, prompt: str, auto_execute: bool = True, **kwargs) -> str:
        """Ask with these tools available"""
        tools_format = self.to_openai_format()

        if auto_execute:
            from .tool_execution import execute_with_tools

            # Build function mapping from our Tool objects
            tool_functions = {
                name: tool.func for name, tool in self.tools.items() if tool.func
            }

            # Use multi-step execution
            return await execute_with_tools(
                ask, prompt, tools_format, tool_functions, **kwargs
            )
        else:
            # Get response with tools (includes tool calls automatically)
            result = await ask(prompt, tools=tools_format, **kwargs)

            # Extract just the response string if result is a dict
            if isinstance(result, dict):
                return result.get("response", str(result))
            return str(result)

    def ask_sync(self, prompt: str, auto_execute: bool = True, **kwargs) -> str:
        """Sync version of ask with tools"""

        tools_format = self.to_openai_format()

        if auto_execute:
            from .tool_execution import execute_with_tools_sync

            # Build function mapping from our Tool objects
            tool_functions = {
                name: tool.func for name, tool in self.tools.items() if tool.func
            }

            # Use multi-step execution
            return execute_with_tools_sync(
                ask_sync, prompt, tools_format, tool_functions, **kwargs
            )
        else:
            # Get response with tools (includes tool calls automatically)
            result = ask_sync(prompt, tools=tools_format, **kwargs)

            # Extract just the response string if result is a dict
            if isinstance(result, dict):
                return result.get("response", str(result))
            return str(result)


def tool(name: str | None = None, description: str | None = None) -> Callable:
    """
    Decorator to mark a function as a tool.

    Usage:
        @tool(description="Get current weather for a location")
        def get_weather(location: str, unit: str = "celsius") -> dict:
            return {"temp": 22, "unit": unit}
    """

    def decorator(func: Callable) -> Callable:
        # Store tool metadata on the function
        func._tool_name = name or func.__name__  # type: ignore[attr-defined]
        func._tool_description = (  # type: ignore[attr-defined]
            description or func.__doc__ or f"Function {func.__name__}"
        )
        func._is_tool = True  # type: ignore[attr-defined]
        return func

    return decorator


class Tools:
    """
    Simple class-based tool definition.

    Usage:
        class MyTools(Tools):
            @tool(description="Get weather information")
            def get_weather(self, location: str) -> dict:
                return {"temp": 22, "location": location}

            @tool
            def calculate(self, expression: str) -> float:
                "Evaluate a mathematical expression"
                return eval(expression)

        tools = MyTools()
        response = await tools.ask("What's the weather in Paris?")
    """

    def __init__(self):
        self.toolkit = ToolKit(name=self.__class__.__name__)
        self._register_tools()

    def _register_tools(self):
        """Auto-register all methods marked with @tool"""
        for _name, method in inspect.getmembers(self):
            if hasattr(method, "_is_tool"):
                self.toolkit.add_function(
                    method, name=method._tool_name, description=method._tool_description
                )

    async def ask(self, prompt: str, auto_execute: bool = True, **kwargs) -> str:
        """Ask with available tools"""
        result = await self.toolkit.ask(prompt, auto_execute=auto_execute, **kwargs)
        return result  # ToolKit.ask already handles extraction and execution

    def ask_sync(self, prompt: str, auto_execute: bool = True, **kwargs) -> str:
        """Sync version of ask"""
        result = self.toolkit.ask_sync(prompt, auto_execute=auto_execute, **kwargs)
        return result  # ToolKit.ask_sync already handles extraction and execution

    @property
    def tools(self) -> list[dict[str, Any]]:
        """Get tools in OpenAI format"""
        return self.toolkit.to_openai_format()


def _python_type_to_json_schema(python_type: type) -> str:
    """Convert Python type to JSON schema type"""
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        dict: "object",
        list: "array",
        Any: "string",  # Default fallback
    }

    # Handle Optional types
    if hasattr(python_type, "__origin__"):
        if python_type.__origin__ is type(Optional):
            # Get the inner type of Optional[X]
            inner_type = python_type.__args__[0] if python_type.__args__ else Any  # type: ignore[attr-defined]
            return _python_type_to_json_schema(inner_type)

    return type_map.get(python_type, "string")


# Convenience functions for quick tool creation


def create_tool(
    name: str,
    description: str,
    parameters: dict[str, Any],
    func: Callable | None = None,
) -> Tool:
    """
    Create a tool with explicit parameters.

    Example:
        weather_tool = create_tool(
            name="get_weather",
            description="Get weather for a location",
            parameters={
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }
        )
    """
    return Tool(name=name, description=description, parameters=parameters, func=func)


def tools_from_functions(*funcs: Callable) -> ToolKit:
    """
    Create a toolkit from multiple functions.

    Example:
        def get_weather(location: str) -> dict:
            return {"temp": 22}

        def calculate(expr: str) -> float:
            return eval(expr)

        toolkit = tools_from_functions(get_weather, calculate)
        response = await toolkit.ask("What's 2+2?")
    """
    toolkit = ToolKit()
    for func in funcs:
        toolkit.add_function(func)
    return toolkit
