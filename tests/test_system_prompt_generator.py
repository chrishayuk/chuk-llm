# tests/test_system_prompt_generator.py
import json
from chuk_llm.system_prompt_generator import SystemPromptGenerator

def test_system_prompt_generator_initialization():
    """Test the initialization of SystemPromptGenerator."""
    generator = SystemPromptGenerator()
    
    # Check default values
    assert generator.template is not None
    assert "{{ TOOL DEFINITIONS IN JSON SCHEMA }}" in generator.template
    assert "{{ USER SYSTEM PROMPT }}" in generator.template
    assert "{{ TOOL CONFIGURATION }}" in generator.template
    
    assert generator.default_user_system_prompt == "You are an intelligent assistant capable of using tools to solve user queries effectively."
    assert generator.default_tool_config == "No additional configuration is required."

def test_generate_prompt_with_defaults():
    """Test generate_prompt with default values."""
    generator = SystemPromptGenerator()
    tools = {"weather": {"description": "Get weather information", "parameters": {"type": "object"}}}
    
    prompt = generator.generate_prompt(tools)
    
    # Check that the prompt contains the tools JSON
    assert json.dumps(tools, indent=2) in prompt
    
    # Check that it contains the default user system prompt
    assert generator.default_user_system_prompt in prompt
    
    # Check that it contains the default tool config
    assert generator.default_tool_config in prompt

def test_generate_prompt_with_custom_user_prompt():
    """Test generate_prompt with custom user prompt."""
    generator = SystemPromptGenerator()
    tools = {"calculator": {"description": "Perform calculations", "parameters": {"type": "object"}}}
    custom_prompt = "You are a math assistant that helps with calculations."
    
    prompt = generator.generate_prompt(tools, user_system_prompt=custom_prompt)
    
    # Check that the prompt contains the tools JSON
    assert json.dumps(tools, indent=2) in prompt
    
    # Check that it contains the custom user system prompt
    assert custom_prompt in prompt
    
    # Check that it does NOT contain the default user system prompt
    assert generator.default_user_system_prompt not in prompt
    
    # Check that it contains the default tool config
    assert generator.default_tool_config in prompt

def test_generate_prompt_with_custom_tool_config():
    """Test generate_prompt with custom tool configuration."""
    generator = SystemPromptGenerator()
    tools = {"search": {"description": "Search the web", "parameters": {"type": "object"}}}
    custom_config = "Use the search tool for factual queries only."
    
    prompt = generator.generate_prompt(tools, tool_config=custom_config)
    
    # Check that the prompt contains the tools JSON
    assert json.dumps(tools, indent=2) in prompt
    
    # Check that it contains the default user system prompt
    assert generator.default_user_system_prompt in prompt
    
    # Check that it contains the custom tool config
    assert custom_config in prompt
    
    # Check that it does NOT contain the default tool config
    assert generator.default_tool_config not in prompt

def test_generate_prompt_with_all_custom_parameters():
    """Test generate_prompt with all custom parameters."""
    generator = SystemPromptGenerator()
    tools = {"database": {"description": "Query a database", "parameters": {"type": "object"}}}
    custom_prompt = "You are a database assistant."
    custom_config = "Database queries should be secure and optimized."
    
    prompt = generator.generate_prompt(tools, user_system_prompt=custom_prompt, tool_config=custom_config)
    
    # Check that the prompt contains the tools JSON
    assert json.dumps(tools, indent=2) in prompt
    
    # Check that it contains the custom user system prompt
    assert custom_prompt in prompt
    
    # Check that it contains the custom tool config
    assert custom_config in prompt
    
    # Check that it does NOT contain any defaults
    assert generator.default_user_system_prompt not in prompt
    assert generator.default_tool_config not in prompt

def test_generate_prompt_with_complex_tools():
    """Test generate_prompt with complex nested tool definitions."""
    generator = SystemPromptGenerator()
    complex_tools = {
        "search": {
            "description": "Search for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "options": {
                        "type": "object",
                        "properties": {
                            "limit": {"type": "integer"},
                            "filter": {"type": "string"}
                        }
                    }
                },
                "required": ["query"]
            }
        }
    }
    
    prompt = generator.generate_prompt(complex_tools)
    
    # Check that the complex tools JSON is included correctly
    assert json.dumps(complex_tools, indent=2) in prompt

def test_generate_prompt_with_empty_tools():
    """Test generate_prompt with empty tools object."""
    generator = SystemPromptGenerator()
    
    prompt = generator.generate_prompt({})
    
    # Check that an empty tools JSON is included
    assert json.dumps({}, indent=2) in prompt
    
    # Check that the default values are included
    assert generator.default_user_system_prompt in prompt
    assert generator.default_tool_config in prompt