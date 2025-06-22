# tests/core/providers/test_gemini_client.py
import sys
import types
import asyncio
import json
import uuid
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

# ---------------------------------------------------------------------------
# Build a stub for the ``google.genai`` SDK *before* importing the client, so
# that the real heavy package is never needed and no network calls are made.
# ---------------------------------------------------------------------------

google_mod = sys.modules.get("google") or types.ModuleType("google")
if "google" not in sys.modules:
    sys.modules["google"] = google_mod

# --- sub-module ``google.genai`` -------------------------------------------

genai_mod = types.ModuleType("google.genai")
sys.modules["google.genai"] = genai_mod
setattr(google_mod, "genai", genai_mod)

# --- sub-module ``google.genai.types`` -------------------------------------

types_mod = types.ModuleType("google.genai.types")
sys.modules["google.genai.types"] = types_mod
setattr(genai_mod, "types", types_mod)

# Provide *minimal* class stubs used by the adapter's helper code. We keep
# them extremely simple - they only need to accept the constructor args.

class _Simple:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __repr__(self):
        return f"<_Simple {self.__dict__}>"


# Updated to match actual implementation usage
class Tool(_Simple):
    def __init__(self, function_declarations=None, **kwargs):
        super().__init__(**kwargs)
        self.function_declarations = function_declarations or []

class GenerateContentConfig(_Simple):
    def __init__(self, tools=None, system_instruction=None, max_output_tokens=None, 
                 temperature=None, top_p=None, top_k=None, **kwargs):
        super().__init__(**kwargs)
        self.tools = tools
        self.system_instruction = system_instruction
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k

types_mod.Tool = Tool
types_mod.GenerateContentConfig = GenerateContentConfig

# ---------------------------------------------------------------------------
# Fake client that matches the actual implementation
# ---------------------------------------------------------------------------

class _MockAIOModels:
    async def generate_content(self, **kwargs):
        # Return a mock response with text
        mock_response = MagicMock()
        mock_response.text = "Mock response from Gemini"
        mock_response.candidates = None
        return mock_response

    async def generate_content_stream(self, **kwargs):
        # Return an async generator
        async def mock_stream():
            chunk1 = MagicMock()
            chunk1.text = "chunk1"
            yield chunk1
            
            chunk2 = MagicMock()
            chunk2.text = "chunk2"
            yield chunk2
        
        return mock_stream()

class _MockAIO:
    def __init__(self):
        self.models = _MockAIOModels()

class _DummyModels:
    def generate_content(self, *a, **k):
        return None

    def generate_content_stream(self, *a, **k):
        return []

class DummyGenAIClient:
    def __init__(self, *args, **kwargs):
        self.models = _DummyModels()
        self.aio = _MockAIO()

genai_mod.Client = DummyGenAIClient

# ---------------------------------------------------------------------------
# Mock dotenv
# ---------------------------------------------------------------------------
sys.modules["dotenv"] = types.ModuleType("dotenv")
sys.modules["dotenv"].load_dotenv = lambda: None

# ---------------------------------------------------------------------------
# Now import the adapter under test (it will pick up the stubs).
# ---------------------------------------------------------------------------

from chuk_llm.llm.providers.gemini_client import (
    GeminiLLMClient, 
    _convert_tools_to_gemini_format,
    _parse_gemini_response,
    validate_and_map_model
)  # noqa: E402  pylint: disable=wrong-import-position

# ---------------------------------------------------------------------------
# Configuration Mock Classes
# ---------------------------------------------------------------------------

class MockFeature:
    TEXT = "text"
    STREAMING = "streaming"
    TOOLS = "tools"
    VISION = "vision"
    JSON_MODE = "json_mode"
    SYSTEM_MESSAGES = "system_messages"
    PARALLEL_CALLS = "parallel_calls"
    MULTIMODAL = "multimodal"
    REASONING = "reasoning"
    
    @classmethod
    def from_string(cls, feature_str):
        return getattr(cls, feature_str.upper(), None)

class MockModelCapabilities:
    def __init__(self, features=None, max_context_length=8192, max_output_tokens=4096):
        self.features = features or {
            MockFeature.TEXT, MockFeature.STREAMING, MockFeature.TOOLS, 
            MockFeature.VISION, MockFeature.SYSTEM_MESSAGES, MockFeature.MULTIMODAL
        }
        self.max_context_length = max_context_length
        self.max_output_tokens = max_output_tokens

class MockProviderConfig:
    def __init__(self, name="gemini", client_class="GeminiLLMClient"):
        self.name = name
        self.client_class = client_class
        self.api_base = "https://generativelanguage.googleapis.com"
        self.models = ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash", "gemini-1.5-pro"]
        self.model_aliases = {}
        self.rate_limits = {"requests_per_minute": 60}
    
    def get_model_capabilities(self, model):
        # Gemini models typically have comprehensive features
        features = {
            MockFeature.TEXT, MockFeature.STREAMING, MockFeature.TOOLS, 
            MockFeature.VISION, MockFeature.SYSTEM_MESSAGES, MockFeature.MULTIMODAL
        }
        
        # Pro models have additional capabilities
        if "pro" in model.lower():
            features.add(MockFeature.JSON_MODE)
            features.add(MockFeature.REASONING)
        
        return MockModelCapabilities(features=features)

class MockConfig:
    def __init__(self):
        self.gemini_provider = MockProviderConfig()
    
    def get_provider(self, provider_name):
        if provider_name == "gemini":
            return self.gemini_provider
        return None

# ---------------------------------------------------------------------------
# Fixtures with Configuration Mocking
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_configuration():
    """Mock the configuration system"""
    mock_config = MockConfig()
    
    with patch('chuk_llm.configuration.get_config', return_value=mock_config):
        with patch('chuk_llm.configuration.Feature', MockFeature):
            yield mock_config

@pytest.fixture
def client(mock_configuration, monkeypatch):
    """Gemini client with configuration properly mocked"""
    cl = GeminiLLMClient(model="gemini-2.5-flash", api_key="fake-key")
    
    # Ensure configuration methods are properly mocked
    monkeypatch.setattr(cl, "supports_feature", lambda feature: feature in [
        "text", "streaming", "tools", "vision", "system_messages", "multimodal"
    ])
    
    monkeypatch.setattr(cl, "get_model_info", lambda: {
        "provider": "gemini",
        "model": "gemini-2.5-flash",
        "client_class": "GeminiLLMClient",
        "api_base": "https://generativelanguage.googleapis.com",
        "features": ["text", "streaming", "tools", "vision", "system_messages", "multimodal"],
        "supports_text": True,
        "supports_streaming": True,
        "supports_tools": True,
        "supports_vision": True,
        "supports_system_messages": True,
        "supports_json_mode": False,
        "supports_parallel_calls": False,
        "supports_multimodal": True,
        "supports_reasoning": False,
        "max_context_length": 8192,
        "max_output_tokens": 4096,
        "gemini_specific": {
            "model_family": cl._detect_model_family(),
            "supports_function_calling": True,
            "supports_vision": True,
            "supports_multimodal": True,
            "max_input_tokens": 1000000,
        }
    })
    
    # Mock token limits
    monkeypatch.setattr(cl, "get_max_tokens_limit", lambda: 4096)
    monkeypatch.setattr(cl, "get_context_length_limit", lambda: 8192)
    
    # Mock parameter validation
    def mock_validate_parameters(**kwargs):
        result = kwargs.copy()
        if 'max_tokens' in result and result['max_tokens'] > 4096:
            result['max_tokens'] = 4096
        return result
    monkeypatch.setattr(cl, "validate_parameters", mock_validate_parameters)
    
    return cl

@pytest.fixture
def pro_client(mock_configuration, monkeypatch):
    """Gemini Pro client with advanced features"""
    cl = GeminiLLMClient(model="gemini-2.5-pro", api_key="fake-key")
    
    # Pro model has additional features
    monkeypatch.setattr(cl, "supports_feature", lambda feature: feature in [
        "text", "streaming", "tools", "vision", "system_messages", "multimodal", "json_mode", "reasoning"
    ])
    
    monkeypatch.setattr(cl, "get_model_info", lambda: {
        "provider": "gemini",
        "model": "gemini-2.5-pro",
        "client_class": "GeminiLLMClient",
        "api_base": "https://generativelanguage.googleapis.com",
        "features": ["text", "streaming", "tools", "vision", "system_messages", "multimodal", "json_mode", "reasoning"],
        "supports_text": True,
        "supports_streaming": True,
        "supports_tools": True,
        "supports_vision": True,
        "supports_system_messages": True,
        "supports_json_mode": True,
        "supports_parallel_calls": False,
        "supports_multimodal": True,
        "supports_reasoning": True,
        "max_context_length": 8192,
        "max_output_tokens": 4096,
        "gemini_specific": {
            "model_family": cl._detect_model_family(),
            "supports_function_calling": True,
            "supports_vision": True,
            "supports_multimodal": True,
            "max_input_tokens": 1000000,
        }
    })
    
    return cl

# ---------------------------------------------------------------------------
# Model validation tests
# ---------------------------------------------------------------------------

def test_validate_and_map_model_valid():
    """Test model validation with valid models."""
    assert validate_and_map_model("gemini-2.5-pro") == "gemini-2.5-pro"
    assert validate_and_map_model("gemini-2.0-flash") == "gemini-2.0-flash"
    assert validate_and_map_model("gemini-1.5-pro") == "gemini-1.5-pro"

def test_validate_and_map_model_invalid():
    """Test model validation with invalid models."""
    with pytest.raises(ValueError) as exc_info:
        validate_and_map_model("invalid-model")
    
    assert "not available for provider 'gemini'" in str(exc_info.value)
    assert "gemini-2.5-pro" in str(exc_info.value)

# ---------------------------------------------------------------------------
# Helper conversion functions tests
# ---------------------------------------------------------------------------

def test_convert_tools_basic():
    """Test basic tool conversion."""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    },
                    "required": ["location"]
                }
            }
        }
    ]
    
    gem_tools = _convert_tools_to_gemini_format(tools)
    
    # Check tools list
    assert len(gem_tools) == 1
    assert len(gem_tools[0].function_declarations) == 1
    assert gem_tools[0].function_declarations[0]["name"] == "get_weather"
    assert gem_tools[0].function_declarations[0]["description"] == "Get the current weather"

def test_convert_tools_empty():
    """Test conversion with no tools."""
    gem_tools = _convert_tools_to_gemini_format(None)
    assert gem_tools is None
    
    gem_tools = _convert_tools_to_gemini_format([])
    assert gem_tools is None

def test_convert_tools_invalid():
    """Test conversion with invalid tools."""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "",  # Invalid empty name
                "description": "Invalid tool",
                "parameters": {}
            }
        }
    ]
    
    gem_tools = _convert_tools_to_gemini_format(tools)
    # Should return None or empty list due to invalid tool
    assert gem_tools is None

# ---------------------------------------------------------------------------
# Response parsing tests
# ---------------------------------------------------------------------------

def test_parse_gemini_response_text_only():
    """Test parsing response with text only."""
    mock_response = MagicMock()
    mock_response.text = "Hello world"
    mock_response.candidates = None
    
    result = _parse_gemini_response(mock_response)
    
    assert result["response"] == "Hello world"
    assert result["tool_calls"] == []

def test_parse_gemini_response_candidates_structure():
    """Test parsing response with candidates structure."""
    # Create mock candidate with content and parts
    mock_part = MagicMock()
    mock_part.text = "Response from candidates"
    
    mock_content = MagicMock()
    mock_content.parts = [mock_part]
    
    mock_candidate = MagicMock()
    mock_candidate.content = mock_content
    
    mock_response = MagicMock()
    mock_response.text = None
    mock_response.candidates = [mock_candidate]
    
    result = _parse_gemini_response(mock_response)
    
    assert result["response"] == "Response from candidates"
    assert result["tool_calls"] == []

def test_parse_gemini_response_max_tokens():
    """Test parsing response when max tokens reached."""
    mock_candidate = MagicMock()
    mock_candidate.finish_reason = "FinishReason.MAX_TOKENS"
    mock_candidate.content = None
    
    mock_response = MagicMock()
    mock_response.text = None
    mock_response.candidates = [mock_candidate]
    
    result = _parse_gemini_response(mock_response)
    
    assert "token limit" in result["response"]
    assert result["tool_calls"] == []

def test_parse_gemini_response_function_calls():
    """Test parsing response with function calls."""
    # Create mock function call
    mock_function_call = MagicMock()
    mock_function_call.name = "get_weather"
    mock_function_call.args = {"city": "NYC"}
    
    mock_part = MagicMock()
    mock_part.text = None
    mock_part.function_call = mock_function_call
    
    mock_content = MagicMock()
    mock_content.parts = [mock_part]
    
    mock_candidate = MagicMock()
    mock_candidate.content = mock_content
    
    mock_response = MagicMock()
    mock_response.text = None
    mock_response.candidates = [mock_candidate]
    
    result = _parse_gemini_response(mock_response)
    
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["function"]["name"] == "get_weather"
    assert "NYC" in result["tool_calls"][0]["function"]["arguments"]

# ---------------------------------------------------------------------------
# Client method tests
# ---------------------------------------------------------------------------

def test_client_initialization(mock_configuration):
    """Test client initialization."""
    client = GeminiLLMClient(model="gemini-2.5-pro", api_key="test-key")
    assert client.model == "gemini-2.5-pro"

def test_client_initialization_invalid_model():
    """Test client initialization with invalid model."""
    with pytest.raises(ValueError):
        GeminiLLMClient(model="invalid-model", api_key="test-key")

def test_get_model_info(client):
    """Test get_model_info method."""
    info = client.get_model_info()
    
    assert "model" in info
    assert "provider" in info
    assert "gemini_specific" in info
    assert info["gemini_specific"]["model_family"] == "gemini-2.5"

def test_detect_model_family(client):
    """Test model family detection."""
    assert client._detect_model_family() == "gemini-2.5"
    
    client.model = "gemini-2.0-flash"
    assert client._detect_model_family() == "gemini-2.0"
    
    client.model = "gemini-1.5-pro"
    assert client._detect_model_family() == "gemini-1.5"

def test_filter_gemini_params(client):
    """Test parameter filtering."""
    params = {
        "temperature": 0.8,
        "max_tokens": 1000,
        "frequency_penalty": 0.5,  # Unsupported
        "top_p": 0.9,
        "user": "test_user",  # Unsupported
        "top_k": 40
    }
    
    filtered = client._filter_gemini_params(params)
    
    assert "temperature" in filtered
    assert "max_output_tokens" in filtered  # Mapped from max_tokens
    assert "top_p" in filtered
    assert "top_k" in filtered
    assert "frequency_penalty" not in filtered
    assert "user" not in filtered

def test_check_json_mode(pro_client):
    """Test JSON mode detection with Pro client that supports JSON mode."""
    # Test OpenAI-style response_format
    kwargs = {"response_format": {"type": "json_object"}}
    instruction = pro_client._check_json_mode(kwargs)
    assert instruction is not None
    assert "JSON" in instruction
    
    # Test custom json mode instruction
    kwargs = {"_json_mode_instruction": "Custom JSON instruction"}
    instruction = pro_client._check_json_mode(kwargs)
    assert instruction == "Custom JSON instruction"
    
    # Test no JSON mode
    kwargs = {}
    instruction = pro_client._check_json_mode(kwargs)
    assert instruction is None

def test_check_json_mode_not_supported(client):
    """Test JSON mode when not supported by model."""
    # Flash model doesn't support JSON mode
    kwargs = {"response_format": {"type": "json_object"}}
    instruction = client._check_json_mode(kwargs)
    assert instruction is None  # Should return None when not supported

# ---------------------------------------------------------------------------
# Async message splitting tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_split_for_gemini_async_basic(client):
    """Test basic message splitting."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello Gemini"},
        {"role": "assistant", "content": "Hello! How can I help you today?"}
    ]
    
    system_txt, contents = await client._split_for_gemini_async(messages)
    
    assert system_txt == "You are a helpful assistant."
    assert len(contents) == 2
    assert "Hello Gemini" in contents
    assert "Hello! How can I help you today?" in contents

@pytest.mark.asyncio
async def test_split_for_gemini_async_multimodal(client):
    """Test message splitting with multimodal content when vision is supported."""
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
        ]}
    ]
    
    system_txt, contents = await client._split_for_gemini_async(messages)
    
    assert system_txt == ""
    assert len(contents) == 1
    assert "What's in this image?" in contents[0]
    assert "[Image content provided]" in contents[0]

@pytest.mark.asyncio
async def test_split_for_gemini_async_multimodal_not_supported(client, monkeypatch):
    """Test message splitting with multimodal content when vision is not supported."""
    # Mock vision as not supported
    monkeypatch.setattr(client, "supports_feature", lambda feature: feature != "vision")
    
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
        ]}
    ]
    
    system_txt, contents = await client._split_for_gemini_async(messages)
    
    assert system_txt == ""
    assert len(contents) == 1
    # Should only contain text when vision not supported
    assert "What's in this image?" in contents[0]
    assert "[Image content provided]" not in contents[0]

@pytest.mark.asyncio
async def test_split_for_gemini_async_tool_calls(client):
    """Test message splitting with tool calls."""
    messages = [
        {"role": "user", "content": "Get weather"},
        {"role": "assistant", "tool_calls": [
            {"function": {"name": "get_weather", "arguments": '{"city": "NYC"}'}}
        ]},
        {"role": "tool", "name": "get_weather", "content": "Sunny, 75°F"}
    ]
    
    system_txt, contents = await client._split_for_gemini_async(messages)
    
    assert system_txt == ""
    assert len(contents) == 3
    assert "Get weather" in contents[0]
    assert "get_weather" in contents[1]
    assert "Tool get_weather returned: Sunny, 75°F" in contents[2]

# ---------------------------------------------------------------------------
# Non-streaming completion tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_regular_completion_async(client):
    """Test regular (non-streaming) completion."""
    messages = [{"role": "user", "content": "Hello"}]
    
    # Mock the client's generate_content method
    mock_response = MagicMock()
    mock_response.text = "Hello! How can I help you?"
    mock_response.candidates = None
    
    async def mock_generate_content(**kwargs):
        return mock_response
    
    client.client.aio.models.generate_content = mock_generate_content
    
    result = await client._regular_completion_async(
        system=None,
        json_instruction=None,
        messages=messages,
        gemini_tools=None,
        filtered_params={}
    )
    
    assert result["response"] == "Hello! How can I help you?"
    assert result["tool_calls"] == []

@pytest.mark.asyncio
async def test_regular_completion_async_with_system(client):
    """Test regular completion with system instruction."""
    messages = [{"role": "user", "content": "Hello"}]
    system = "You are a helpful assistant."
    
    # Mock the client's generate_content method
    mock_response = MagicMock()
    mock_response.text = "Hello! I'm here to help."
    mock_response.candidates = None
    
    async def mock_generate_content(**kwargs):
        # Verify system instruction is included
        config = kwargs.get('config')
        if config:
            assert config.system_instruction == system
        return mock_response
    
    client.client.aio.models.generate_content = mock_generate_content
    
    result = await client._regular_completion_async(
        system=system,
        json_instruction=None,
        messages=messages,
        gemini_tools=None,
        filtered_params={}
    )
    
    assert result["response"] == "Hello! I'm here to help."

@pytest.mark.asyncio
async def test_regular_completion_async_error_handling(client):
    """Test error handling in regular completion."""
    messages = [{"role": "user", "content": "Hello"}]
    
    # Mock the client to raise an exception
    async def mock_generate_content_error(**kwargs):
        raise Exception("API Error")
    
    client.client.aio.models.generate_content = mock_generate_content_error
    
    result = await client._regular_completion_async(
        system=None,
        json_instruction=None,
        messages=messages,
        gemini_tools=None,
        filtered_params={}
    )
    
    assert "error" in result
    assert result["error"] is True
    assert "API Error" in result["response"]

# ---------------------------------------------------------------------------
# Streaming completion tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stream_completion_async(client):
    """Test streaming completion."""
    messages = [{"role": "user", "content": "Hello"}]
    
    # Mock the streaming response
    async def mock_stream():
        chunk1 = MagicMock()
        chunk1.text = "Hello"
        yield chunk1
        
        chunk2 = MagicMock()
        chunk2.text = " world!"
        yield chunk2
    
    async def mock_generate_content_stream(**kwargs):
        return mock_stream()
    
    client.client.aio.models.generate_content_stream = mock_generate_content_stream
    
    # Collect streaming results
    chunks = []
    async for chunk in client._stream_completion_async(
        system=None,
        json_instruction=None,
        messages=messages,
        gemini_tools=None,
        filtered_params={}
    ):
        chunks.append(chunk)
    
    assert len(chunks) == 2
    assert chunks[0]["response"] == "Hello"
    assert chunks[1]["response"] == " world!"

@pytest.mark.asyncio
async def test_stream_completion_async_error_handling(client):
    """Test error handling in streaming completion."""
    messages = [{"role": "user", "content": "Hello"}]
    
    # Mock the streaming to raise an error
    async def mock_generate_content_stream_error(**kwargs):
        raise Exception("Streaming error")
    
    client.client.aio.models.generate_content_stream = mock_generate_content_stream_error
    
    # Collect streaming results
    chunks = []
    async for chunk in client._stream_completion_async(
        system=None,
        json_instruction=None,
        messages=messages,
        gemini_tools=None,
        filtered_params={}
    ):
        chunks.append(chunk)
    
    # Should yield an error chunk
    assert len(chunks) == 1
    assert "error" in chunks[0]
    assert chunks[0]["error"] is True
    assert "Streaming error" in chunks[0]["response"]

# ---------------------------------------------------------------------------
# Main interface tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_create_completion_non_streaming(client):
    """Test create_completion with non-streaming."""
    messages = [{"role": "user", "content": "Hello"}]
    
    # Mock the regular completion method
    expected_result = {"response": "Hello!", "tool_calls": []}
    
    async def mock_regular_completion_async(system, json_instruction, messages, gemini_tools, filtered_params):
        return expected_result
    
    client._regular_completion_async = mock_regular_completion_async
    
    result = client.create_completion(messages, stream=False)
    
    # Should return an awaitable
    assert hasattr(result, '__await__')
    
    final_result = await result
    assert final_result == expected_result

@pytest.mark.asyncio
async def test_create_completion_streaming(client):
    """Test create_completion with streaming."""
    messages = [{"role": "user", "content": "Hello"}]
    
    # Mock the streaming method
    async def mock_stream_completion_async(system, json_instruction, messages, gemini_tools, filtered_params):
        yield {"response": "chunk1", "tool_calls": []}
        yield {"response": "chunk2", "tool_calls": []}
    
    client._stream_completion_async = mock_stream_completion_async
    
    result = client.create_completion(messages, stream=True)
    
    # Should return an async generator
    assert hasattr(result, '__aiter__')
    
    chunks = []
    async for chunk in result:
        chunks.append(chunk)
    
    assert len(chunks) == 2
    assert chunks[0]["response"] == "chunk1"
    assert chunks[1]["response"] == "chunk2"

@pytest.mark.asyncio
async def test_create_completion_streaming_not_supported(client, monkeypatch):
    """Test create_completion with streaming when not supported."""
    messages = [{"role": "user", "content": "Hello"}]
    
    # Mock streaming as not supported
    monkeypatch.setattr(client, "supports_feature", lambda feature: feature != "streaming")
    
    # Mock the regular completion method (should be called instead of streaming)
    expected_result = {"response": "Hello!", "tool_calls": []}
    
    async def mock_regular_completion_async(system, json_instruction, messages, gemini_tools, filtered_params):
        return expected_result
    
    client._regular_completion_async = mock_regular_completion_async
    
    result = client.create_completion(messages, stream=True)
    
    # Should return an awaitable (not async iterator) when streaming not supported
    assert hasattr(result, '__await__')
    assert not hasattr(result, '__aiter__')
    
    final_result = await result
    assert final_result == expected_result

@pytest.mark.asyncio
async def test_create_completion_with_tools(client):
    """Test create_completion with tools."""
    messages = [{"role": "user", "content": "What's the weather?"}]
    tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]
    
    # Mock regular completion
    expected_result = {
        "response": "I'll check the weather for you.",
        "tool_calls": [
            {"id": "call_123", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}
        ]
    }
    
    async def mock_regular_completion_async(system, json_instruction, messages, gemini_tools, filtered_params):
        # Verify tools were converted
        assert gemini_tools is not None
        assert len(gemini_tools) == 1
        return expected_result
    
    client._regular_completion_async = mock_regular_completion_async
    
    result = await client.create_completion(messages, tools=tools, stream=False)
    
    assert result == expected_result
    assert len(result["tool_calls"]) == 1

@pytest.mark.asyncio
async def test_create_completion_with_tools_not_supported(client, monkeypatch):
    """Test create_completion with tools when not supported."""
    messages = [{"role": "user", "content": "What's the weather?"}]
    tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]
    
    # Mock tools as not supported
    monkeypatch.setattr(client, "supports_feature", lambda feature: feature != "tools")
    
    # Mock regular completion
    expected_result = {"response": "I cannot use tools.", "tool_calls": []}
    
    async def mock_regular_completion_async(system, json_instruction, messages, gemini_tools, filtered_params):
        # Verify tools were not passed
        assert gemini_tools is None
        return expected_result
    
    client._regular_completion_async = mock_regular_completion_async
    
    result = await client.create_completion(messages, tools=tools, stream=False)
    
    assert result == expected_result

@pytest.mark.asyncio
async def test_create_completion_with_max_tokens(client):
    """Test create_completion with max_tokens parameter."""
    messages = [{"role": "user", "content": "Hello"}]
    
    # Mock regular completion to check parameters
    async def mock_regular_completion_async(system, json_instruction, messages, gemini_tools, filtered_params):
        # Verify max_tokens was converted to max_output_tokens
        assert "max_output_tokens" in filtered_params
        assert filtered_params["max_output_tokens"] == 500
        return {"response": "Hello!", "tool_calls": []}
    
    client._regular_completion_async = mock_regular_completion_async
    
    result = await client.create_completion(messages, max_tokens=500, stream=False)
    
    assert result["response"] == "Hello!"

# ---------------------------------------------------------------------------
# Tool call extraction tests
# ---------------------------------------------------------------------------

def test_extract_tool_calls_from_response(client):
    """Test extracting tool calls from response."""
    # Create mock response with function calls
    mock_function_call = MagicMock()
    mock_function_call.name = "get_weather"
    mock_function_call.args = {"city": "NYC"}
    
    mock_part = MagicMock()
    mock_part.function_call = mock_function_call
    
    mock_content = MagicMock()
    mock_content.parts = [mock_part]
    
    mock_candidate = MagicMock()
    mock_candidate.content = mock_content
    
    mock_response = MagicMock()
    mock_response.candidates = [mock_candidate]
    
    tool_calls = client._extract_tool_calls_from_response(mock_response)
    
    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "get_weather"
    assert "NYC" in tool_calls[0]["function"]["arguments"]

def test_extract_tool_calls_empty(client):
    """Test extracting tool calls from response with no function calls."""
    mock_response = MagicMock()
    mock_response.candidates = None
    
    tool_calls = client._extract_tool_calls_from_response(mock_response)
    
    assert tool_calls == []

def test_extract_tool_calls_no_function_call_attribute(client):
    """Test extracting tool calls when parts don't have function_call."""
    mock_part = MagicMock()
    # Mock part without function_call attribute
    if hasattr(mock_part, 'function_call'):
        del mock_part.function_call
    
    mock_content = MagicMock()
    mock_content.parts = [mock_part]
    
    mock_candidate = MagicMock()
    mock_candidate.content = mock_content
    
    mock_response = MagicMock()
    mock_response.candidates = [mock_candidate]
    
    tool_calls = client._extract_tool_calls_from_response(mock_response)
    
    assert tool_calls == []

# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_full_integration_non_streaming(client):
    """Test full integration for non-streaming completion."""
    messages = [
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "Hello"}
    ]
    
    # Mock the actual Gemini API call
    mock_response = MagicMock()
    mock_response.text = "Hello! How can I help you today?"
    mock_response.candidates = None
    
    async def mock_generate_content(**kwargs):
        # Verify request structure
        assert "model" in kwargs
        assert "contents" in kwargs
        assert kwargs["model"] == "gemini-2.5-flash"
        return mock_response
    
    client.client.aio.models.generate_content = mock_generate_content
    
    result = await client.create_completion(messages, stream=False)
    
    assert result["response"] == "Hello! How can I help you today?"
    assert result["tool_calls"] == []

@pytest.mark.asyncio
async def test_full_integration_streaming(client):
    """Test full integration for streaming completion."""
    messages = [{"role": "user", "content": "Tell me a story"}]
    
    # Mock streaming response
    async def mock_stream():
        for text in ["Once", " upon", " a", " time..."]:
            chunk = MagicMock()
            chunk.text = text
            yield chunk
    
    async def mock_generate_content_stream(**kwargs):
        return mock_stream()
    
    client.client.aio.models.generate_content_stream = mock_generate_content_stream
    
    # Collect all chunks
    story_parts = []
    async for chunk in client.create_completion(messages, stream=True):
        story_parts.append(chunk["response"])
    
    # Verify we got all parts
    assert len(story_parts) == 4
    assert story_parts == ["Once", " upon", " a", " time..."]

@pytest.mark.asyncio
async def test_full_integration_streaming_not_supported(client, monkeypatch):
    """Test full integration for streaming when not supported."""
    messages = [{"role": "user", "content": "Tell me a story"}]
    
    # Mock streaming as not supported
    monkeypatch.setattr(client, "supports_feature", lambda feature: feature != "streaming")
    
    # Mock regular completion
    async def mock_generate_content(**kwargs):
        mock_response = MagicMock()
        mock_response.text = "Once upon a time..."
        mock_response.candidates = None
        return mock_response
    
    client.client.aio.models.generate_content = mock_generate_content
    
    # Should return regular completion instead of streaming
    result = await client.create_completion(messages, stream=True)
    
    assert result["response"] == "Once upon a time..."
    assert result["tool_calls"] == []

# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_error_handling_comprehensive(client):
    """Test comprehensive error handling."""
    messages = [{"role": "user", "content": "Hello"}]
    
    # Test various error scenarios
    error_scenarios = [
        "Network error",
        "Rate limit exceeded",
        "Invalid request",
        "Timeout error"
    ]
    
    for error_msg in error_scenarios:
        async def mock_generate_content_error(**kwargs):
            raise Exception(error_msg)
        
        client.client.aio.models.generate_content = mock_generate_content_error
        
        result = await client._regular_completion_async(
            system=None,
            json_instruction=None,
            messages=messages,
            gemini_tools=None,
            filtered_params={}
        )
        
        assert "error" in result
        assert result["error"] is True
        assert error_msg in result["response"]

# ---------------------------------------------------------------------------
# Feature support validation tests
# ---------------------------------------------------------------------------

def test_feature_support_validation(client, monkeypatch):
    """Test that feature support is properly validated."""
    # Test supported features (from fixture)
    supported_features = ["text", "streaming", "tools", "vision", "system_messages", "multimodal"]
    
    for feature in supported_features:
        assert client.supports_feature(feature) is True
    
    # Test unsupported features
    unsupported_features = ["json_mode", "reasoning", "parallel_calls"]
    
    for feature in unsupported_features:
        assert client.supports_feature(feature) is False
    
    # Test individual feature isolation
    for feature in supported_features:
        # Mock only this feature as supported
        monkeypatch.setattr(client, "supports_feature", lambda f, target_feature=feature: f == target_feature)
        
        # Test that only the target feature is supported
        assert client.supports_feature(feature) is True
        
        # Test that other features are not supported
        other_features = [f for f in supported_features if f != feature]
        for other_feature in other_features:
            assert client.supports_feature(other_feature) is False

def test_pro_model_features(pro_client):
    """Test that Pro model has additional features."""
    info = pro_client.get_model_info()
    
    # Pro model should have additional capabilities
    assert info["supports_json_mode"] is True
    assert info["supports_reasoning"] is True
    assert info["supports_tools"] is True
    assert info["supports_vision"] is True