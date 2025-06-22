# tests/providers/test_openai_client.py
import asyncio
import sys
import types
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# ---------------------------------------------------------------------------
# Mock Configuration Classes (similar to WatsonX)
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
            MockFeature.SYSTEM_MESSAGES, MockFeature.JSON_MODE, MockFeature.VISION
        }
        self.max_context_length = max_context_length
        self.max_output_tokens = max_output_tokens

class MockProviderConfig:
    def __init__(self, name="openai", client_class="OpenAILLMClient"):
        self.name = name
        self.client_class = client_class
        self.api_base = "https://api.openai.com/v1"
        self.models = ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"]
        self.model_aliases = {}
        self.rate_limits = {"requests_per_minute": 10000}
    
    def get_model_capabilities(self, model):
        # Different capabilities based on model
        features = {MockFeature.TEXT, MockFeature.STREAMING, MockFeature.TOOLS, MockFeature.SYSTEM_MESSAGES}
        
        # All OpenAI models support JSON mode
        features.add(MockFeature.JSON_MODE)
        
        # Vision models
        if "vision" in model.lower() or "gpt-4" in model.lower():
            features.add(MockFeature.VISION)
            features.add(MockFeature.MULTIMODAL)
        
        return MockModelCapabilities(features=features)

class MockConfig:
    def __init__(self):
        self.openai_provider = MockProviderConfig()
        self.deepseek_provider = MockProviderConfig("deepseek", "OpenAILLMClient")
        self.groq_provider = MockProviderConfig("groq", "OpenAILLMClient")
    
    def get_provider(self, provider_name):
        if provider_name == "openai":
            return self.openai_provider
        elif provider_name == "deepseek":
            return self.deepseek_provider
        elif provider_name == "groq":
            return self.groq_provider
        return None

# ---------------------------------------------------------------------------
# OpenAI SDK Mock Classes - FIXED
# ---------------------------------------------------------------------------

class MockChatCompletionChunk:
    def __init__(self, content=None, tool_calls=None):
        self.choices = [types.SimpleNamespace(
            delta=types.SimpleNamespace(
                content=content,
                tool_calls=tool_calls
            )
        )]
        self.model = "gpt-4o-mini"
        self.id = "chatcmpl-test"

class MockChatCompletion:
    def __init__(self, content="Test response", tool_calls=None):
        self.choices = [types.SimpleNamespace(
            message=types.SimpleNamespace(
                content=content,
                tool_calls=tool_calls
            )
        )]
        self.model = "gpt-4o-mini"
        self.id = "chatcmpl-test"

class MockChatCompletions:
    def __init__(self):
        # FIXED: Use AsyncMock for async methods
        self.create = AsyncMock()

class MockChat:
    def __init__(self):
        self.completions = MockChatCompletions()

class MockAsyncOpenAI:
    def __init__(self, api_key=None, base_url=None, **kwargs):
        self.api_key = api_key
        self.base_url = base_url
        self.chat = MockChat()

class MockOpenAI:
    def __init__(self, api_key=None, base_url=None, **kwargs):
        self.api_key = api_key
        self.base_url = base_url
        self.chat = MockChat()

# ---------------------------------------------------------------------------
# Fixtures - FIXED
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_configuration():
    """Mock the configuration system"""
    mock_config = MockConfig()
    
    with patch('chuk_llm.configuration.get_config', return_value=mock_config):
        with patch('chuk_llm.configuration.Feature', MockFeature):
            yield mock_config

@pytest.fixture
def mock_openai_sdk():
    """Mock the OpenAI SDK at fixture level"""
    # Create mock module
    openai_mod = types.ModuleType("openai")
    openai_mod.AsyncOpenAI = MockAsyncOpenAI
    openai_mod.OpenAI = MockOpenAI
    
    # Patch the module
    with patch.dict('sys.modules', {'openai': openai_mod}):
        yield openai_mod

@pytest.fixture
def client(mock_configuration, mock_openai_sdk, monkeypatch):
    """OpenAI client with configuration and SDK properly mocked"""
    # Import after mocking
    from chuk_llm.llm.providers.openai_client import OpenAILLMClient
    
    cl = OpenAILLMClient(model="gpt-4o-mini", api_key="fake-key")
    
    # FIXED: Properly mock the async client with correct structure
    cl.async_client = MagicMock()
    cl.async_client.chat = MagicMock()
    cl.async_client.chat.completions = MagicMock()
    cl.async_client.chat.completions.create = AsyncMock()
    
    # Mock configuration methods
    monkeypatch.setattr(cl, "supports_feature", lambda feature: feature in [
        "text", "streaming", "tools", "system_messages", "json_mode", "vision"
    ])
    
    monkeypatch.setattr(cl, "get_model_info", lambda: {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "client_class": "OpenAILLMClient",
        "api_base": "https://api.openai.com/v1",
        "detected_provider": "openai",
        "openai_compatible": True,
        "features": ["text", "streaming", "tools", "system_messages", "json_mode", "vision"],
        "supports_text": True,
        "supports_streaming": True,
        "supports_tools": True,
        "supports_vision": True,
        "supports_system_messages": True,
        "supports_json_mode": True,
        "supports_parallel_calls": False,
        "supports_multimodal": True,
        "supports_reasoning": False,
        "max_context_length": 128000,
        "max_output_tokens": 4096,
        "parameter_mapping": {
            "temperature": "temperature",
            "max_tokens": "max_tokens",
            "top_p": "top_p",
            "frequency_penalty": "frequency_penalty",
            "presence_penalty": "presence_penalty",
            "stop": "stop",
            "stream": "stream"
        }
    })
    
    # Mock token limits
    monkeypatch.setattr(cl, "get_max_tokens_limit", lambda: 4096)
    monkeypatch.setattr(cl, "get_context_length_limit", lambda: 128000)
    
    # Mock parameter validation
    def mock_validate_parameters(**kwargs):
        result = kwargs.copy()
        if 'max_tokens' in result and result['max_tokens'] > 4096:
            result['max_tokens'] = 4096
        return result
    monkeypatch.setattr(cl, "validate_parameters", mock_validate_parameters)
    
    return cl

@pytest.fixture
def deepseek_client(mock_configuration, mock_openai_sdk, monkeypatch):
    """DeepSeek client using OpenAI-compatible interface"""
    # Import after mocking
    from chuk_llm.llm.providers.openai_client import OpenAILLMClient
    
    cl = OpenAILLMClient(
        model="deepseek-chat",
        api_key="fake-key",
        api_base="https://api.deepseek.com/v1"
    )
    
    # FIXED: Properly mock the async client
    cl.async_client = MagicMock()
    cl.async_client.chat = MagicMock()
    cl.async_client.chat.completions = MagicMock()
    cl.async_client.chat.completions.create = AsyncMock()
    
    # Mock configuration methods for DeepSeek
    monkeypatch.setattr(cl, "supports_feature", lambda feature: feature in [
        "text", "streaming", "tools", "system_messages", "json_mode"
    ])
    
    monkeypatch.setattr(cl, "get_model_info", lambda: {
        "provider": "deepseek",
        "model": "deepseek-chat",
        "client_class": "OpenAILLMClient",
        "api_base": "https://api.deepseek.com/v1",
        "detected_provider": "deepseek",
        "openai_compatible": True,
        "features": ["text", "streaming", "tools", "system_messages", "json_mode"],
        "supports_text": True,
        "supports_streaming": True,
        "supports_tools": True,
        "supports_vision": False,
        "supports_system_messages": True,
        "supports_json_mode": True,
        "max_context_length": 32768,
        "max_output_tokens": 4096
    })
    
    return cl

# ---------------------------------------------------------------------------
# Client initialization tests
# ---------------------------------------------------------------------------

def test_client_initialization(mock_configuration, mock_openai_sdk):
    """Test client initialization with different parameters."""
    from chuk_llm.llm.providers.openai_client import OpenAILLMClient
    
    # Test with default model
    client1 = OpenAILLMClient()
    assert client1.model == "gpt-4o-mini"
    assert client1.detected_provider == "openai"
    
    # Test with custom model and API key
    client2 = OpenAILLMClient(model="gpt-4", api_key="test-key")
    assert client2.model == "gpt-4"
    
    # Test with custom API base (should detect provider)
    client3 = OpenAILLMClient(
        model="gpt-4", 
        api_base="https://api.deepseek.com/v1"
    )
    assert client3.model == "gpt-4"
    assert client3.detected_provider == "deepseek"

def test_detect_provider_name(client):
    """Test provider detection from API base URL."""
    assert client._detect_provider_name(None) == "openai"
    assert client._detect_provider_name("https://api.deepseek.com/v1") == "deepseek"
    assert client._detect_provider_name("https://api.groq.com/openai/v1") == "groq"
    assert client._detect_provider_name("https://api.together.xyz/v1") == "together"
    assert client._detect_provider_name("https://api.perplexity.ai") == "perplexity"
    assert client._detect_provider_name("https://api.anyscale.com/v1") == "anyscale"
    assert client._detect_provider_name("https://custom-api.com/v1") == "openai_compatible"

def test_get_model_info(client):
    """Test model info method."""
    info = client.get_model_info()
    
    assert info["provider"] == "openai"
    assert info["model"] == "gpt-4o-mini"
    assert info["detected_provider"] == "openai"
    assert info["openai_compatible"] is True
    assert "parameter_mapping" in info

# ---------------------------------------------------------------------------
# Request validation tests
# ---------------------------------------------------------------------------

def test_validate_request_with_config(client):
    """Test request validation against configuration."""
    messages = [{"role": "user", "content": "Hello"}]
    tools = [{"type": "function", "function": {"name": "test_tool"}}]
    
    # Mock configuration support
    client.supports_feature = lambda feature: feature in ["streaming", "tools", "json_mode"]
    
    validated_messages, validated_tools, validated_stream, validated_kwargs = client._validate_request_with_config(
        messages, tools, stream=True, temperature=0.7, response_format={"type": "json_object"}
    )
    
    assert validated_messages == messages
    assert validated_tools == tools
    assert validated_stream is True
    assert "response_format" in validated_kwargs

def test_validate_request_unsupported_features(client):
    """Test request validation when features are not supported."""
    messages = [{"role": "user", "content": "Hello"}]
    tools = [{"type": "function", "function": {"name": "test_tool"}}]
    
    # Mock configuration to not support some features
    client.supports_feature = lambda feature: feature == "streaming"
    
    validated_messages, validated_tools, validated_stream, validated_kwargs = client._validate_request_with_config(
        messages, tools, stream=True, response_format={"type": "json_object"}
    )
    
    assert validated_messages == messages
    assert validated_tools is None  # Should be None when not supported
    assert validated_stream is True
    assert "response_format" not in validated_kwargs  # Should be removed

# ---------------------------------------------------------------------------
# Message normalization tests
# ---------------------------------------------------------------------------

def test_normalize_message_text_only(client):
    """Test normalizing message with text content only."""
    mock_message = types.SimpleNamespace(
        content="Hello from OpenAI",
        tool_calls=None
    )
    
    result = client._normalize_message(mock_message)
    
    assert result == {"response": "Hello from OpenAI", "tool_calls": []}

def test_normalize_message_with_tool_calls(client):
    """Test normalizing message with tool calls."""
    # Mock tool call structure
    mock_tool_call = types.SimpleNamespace(
        id="call_123",
        function=types.SimpleNamespace(
            name="test_tool",
            arguments='{"arg": "value"}'
        )
    )
    
    mock_message = types.SimpleNamespace(
        content="",
        tool_calls=[mock_tool_call]
    )
    
    result = client._normalize_message(mock_message)
    
    assert result["response"] is None  # Should be None when tool calls present
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["function"]["name"] == "test_tool"
    assert "value" in result["tool_calls"][0]["function"]["arguments"]

def test_normalize_message_dict_format(client):
    """Test normalizing message in dict format."""
    mock_message = {
        "content": "Hello from dict",
        "tool_calls": []
    }
    
    result = client._normalize_message(mock_message)
    
    assert result == {"response": "Hello from dict", "tool_calls": []}

def test_normalize_message_nested_structure(client):
    """Test normalizing message with nested message structure."""
    mock_message = types.SimpleNamespace(
        message=types.SimpleNamespace(
            content="Nested content",
            tool_calls=None
        )
    )
    
    result = client._normalize_message(mock_message)
    
    assert result == {"response": "Nested content", "tool_calls": []}

def test_normalize_message_alternative_fields(client):
    """Test normalizing message with alternative content fields."""
    # Mock message with alternative field
    mock_message = types.SimpleNamespace(
        content="",  # Empty content
        text="Alternative text field",
        tool_calls=None
    )
    
    result = client._normalize_message(mock_message)
    
    assert result["response"] == "Alternative text field"
    assert result["tool_calls"] == []

def test_normalize_message_invalid_tool_call_json(client):
    """Test normalizing message with invalid JSON in tool calls."""
    mock_tool_call = types.SimpleNamespace(
        id="call_123",
        function=types.SimpleNamespace(
            name="test_tool",
            arguments='{"invalid": json}'  # Invalid JSON
        )
    )
    
    mock_message = types.SimpleNamespace(
        content="",
        tool_calls=[mock_tool_call]
    )
    
    result = client._normalize_message(mock_message)
    
    assert result["response"] is None
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["function"]["arguments"] == "{}"  # Should fallback to empty JSON

# ---------------------------------------------------------------------------
# Parameter adjustment tests
# ---------------------------------------------------------------------------

def test_adjust_parameters_for_provider(client):
    """Test parameter adjustment using configuration."""
    params = {
        "temperature": 0.7,
        "max_tokens": 2000,
        "top_p": 0.9
    }
    
    # Mock parameter validation
    def mock_validate_parameters(**kwargs):
        # Simulate max_tokens adjustment
        if kwargs.get("max_tokens", 0) > 1500:
            kwargs["max_tokens"] = 1500
        return kwargs
    
    client.validate_parameters = mock_validate_parameters
    
    adjusted = client._adjust_parameters_for_provider(params)
    
    assert adjusted["temperature"] == 0.7
    assert adjusted["max_tokens"] == 1500  # Should be adjusted
    assert adjusted["top_p"] == 0.9

def test_adjust_parameters_fallback(client):
    """Test parameter adjustment fallback when config fails."""
    params = {"temperature": 0.5}
    
    # Mock validation to raise an exception
    def mock_validate_parameters(**kwargs):
        raise Exception("Config error")
    
    client.validate_parameters = mock_validate_parameters
    
    adjusted = client._adjust_parameters_for_provider(params)
    
    assert adjusted["temperature"] == 0.5
    assert adjusted["max_tokens"] == 4096  # Should add default max_tokens

# ---------------------------------------------------------------------------
# Streaming tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stream_from_async_basic(client):
    """Test basic streaming from async generator."""
    # Mock async stream
    async def mock_stream():
        yield MockChatCompletionChunk(content="Hello")
        yield MockChatCompletionChunk(content=" world")
        yield MockChatCompletionChunk(content="!")
    
    # Collect streaming results
    chunks = []
    async for chunk in client._stream_from_async(mock_stream()):
        chunks.append(chunk)
    
    assert len(chunks) == 3
    assert chunks[0]["response"] == "Hello"
    assert chunks[1]["response"] == " world"
    assert chunks[2]["response"] == "!"

@pytest.mark.asyncio
async def test_stream_from_async_with_tool_calls(client):
    """Test streaming with tool calls."""
    # Mock tool call in streaming
    mock_tool_call = types.SimpleNamespace(
        id="call_123",
        function=types.SimpleNamespace(
            name="test_tool",
            arguments='{"arg": "value"}'
        )
    )
    
    async def mock_stream():
        yield MockChatCompletionChunk(content="Let me")
        yield MockChatCompletionChunk(content="", tool_calls=[mock_tool_call])
        yield MockChatCompletionChunk(content=" help you")
    
    # Collect streaming results
    chunks = []
    async for chunk in client._stream_from_async(mock_stream()):
        chunks.append(chunk)
    
    assert len(chunks) == 3
    assert chunks[0]["response"] == "Let me"
    assert chunks[1]["response"] == ""
    assert len(chunks[1]["tool_calls"]) == 1
    assert chunks[2]["response"] == " help you"

@pytest.mark.asyncio
async def test_stream_from_async_error_handling(client):
    """Test error handling in streaming."""
    async def mock_error_stream():
        yield MockChatCompletionChunk(content="Start")
        raise Exception("Stream error")
    
    # Collect streaming results
    chunks = []
    async for chunk in client._stream_from_async(mock_error_stream()):
        chunks.append(chunk)
    
    # Should have at least one chunk plus error chunk
    assert len(chunks) >= 1
    error_chunk = chunks[-1]
    assert "error" in error_chunk
    assert "Stream error" in error_chunk["response"]

@pytest.mark.asyncio
async def test_stream_from_async_custom_normalization(client):
    """Test streaming with custom chunk normalization."""
    def custom_normalize(result, chunk):
        # Add custom field to result
        result["custom_field"] = "added"
        return result
    
    async def mock_stream():
        yield MockChatCompletionChunk(content="Test")
    
    # Collect streaming results with custom normalization
    chunks = []
    async for chunk in client._stream_from_async(mock_stream(), normalize_chunk=custom_normalize):
        chunks.append(chunk)
    
    assert len(chunks) == 1
    assert chunks[0]["response"] == "Test"
    assert chunks[0]["custom_field"] == "added"

# ---------------------------------------------------------------------------
# Main interface tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_create_completion_non_streaming(client):
    """Test create_completion with non-streaming."""
    messages = [{"role": "user", "content": "Hello"}]
    
    # Mock the regular completion method
    expected_result = {"response": "Hello!", "tool_calls": []}
    
    async def mock_regular_completion(messages, tools, **kwargs):
        return expected_result
    
    client._regular_completion = mock_regular_completion
    
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
    async def mock_stream_completion_async(messages, tools, **kwargs):
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
async def test_create_completion_with_tools(client):
    """Test create_completion with tools."""
    messages = [{"role": "user", "content": "What's the weather?"}]
    tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]
    
    # Mock regular completion
    expected_result = {
        "response": None,
        "tool_calls": [
            {"id": "call_123", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}
        ]
    }
    
    async def mock_regular_completion(messages, tools, **kwargs):
        # Verify tools were passed
        assert tools is not None
        assert len(tools) == 1
        return expected_result
    
    client._regular_completion = mock_regular_completion
    
    result = await client.create_completion(messages, tools=tools, stream=False)
    
    assert result == expected_result
    assert len(result["tool_calls"]) == 1

@pytest.mark.asyncio
async def test_create_completion_parameter_validation(client):
    """Test that parameters are validated through configuration."""
    messages = [{"role": "user", "content": "Hello"}]
    
    # Mock parameter validation
    validated_params = {}
    def mock_validate_parameters(**kwargs):
        validated_params.update(kwargs)
        return kwargs
    
    client.validate_parameters = mock_validate_parameters
    
    async def mock_regular_completion(messages, tools, **kwargs):
        return {"response": "Hello!", "tool_calls": []}
    
    client._regular_completion = mock_regular_completion
    
    result = await client.create_completion(
        messages, 
        temperature=0.7, 
        max_tokens=100,
        stream=False
    )
    
    # Verify parameters were validated
    assert "temperature" in validated_params
    assert "max_tokens" in validated_params

# ---------------------------------------------------------------------------
# Regular completion tests - FIXED
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_regular_completion(client):
    """Test regular (non-streaming) completion."""
    messages = [{"role": "user", "content": "Hello"}]
    
    # Mock the async client response
    mock_response = MockChatCompletion(content="Hello! How can I help you?")
    client.async_client.chat.completions.create.return_value = mock_response
    
    result = await client._regular_completion(messages)
    
    assert result["response"] == "Hello! How can I help you?"
    assert result["tool_calls"] == []
    
    # Verify API call
    client.async_client.chat.completions.create.assert_called_once()
    call_kwargs = client.async_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "gpt-4o-mini"
    assert call_kwargs["stream"] is False

@pytest.mark.asyncio
async def test_regular_completion_with_tools(client):
    """Test regular completion with tools."""
    messages = [{"role": "user", "content": "Use a tool"}]
    tools = [{"type": "function", "function": {"name": "test_tool"}}]
    
    # Mock tool call response
    mock_tool_call = types.SimpleNamespace(
        id="call_123",
        function=types.SimpleNamespace(
            name="test_tool",
            arguments='{"arg": "value"}'
        )
    )
    
    mock_response = MockChatCompletion(content="", tool_calls=[mock_tool_call])
    client.async_client.chat.completions.create.return_value = mock_response
    
    result = await client._regular_completion(messages, tools)
    
    assert result["response"] is None
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["function"]["name"] == "test_tool"
    
    # Verify tools were passed
    call_kwargs = client.async_client.chat.completions.create.call_args.kwargs
    assert "tools" in call_kwargs
    assert call_kwargs["tools"] == tools

@pytest.mark.asyncio
async def test_regular_completion_error_handling(client):
    """Test error handling in regular completion."""
    messages = [{"role": "user", "content": "Hello"}]
    
    # Mock the client to raise an exception
    client.async_client.chat.completions.create.side_effect = Exception("API Error")
    
    result = await client._regular_completion(messages)
    
    assert "error" in result
    assert result["error"] is True
    assert "API Error" in result["response"]

# ---------------------------------------------------------------------------
# Stream completion tests - FIXED
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stream_completion_async(client):
    """Test streaming completion."""
    messages = [{"role": "user", "content": "Tell a story"}]
    
    # Mock streaming response
    async def mock_stream():
        yield MockChatCompletionChunk(content="Once")
        yield MockChatCompletionChunk(content=" upon")
        yield MockChatCompletionChunk(content=" a time")
    
    client.async_client.chat.completions.create.return_value = mock_stream()
    
    # Collect streaming results
    chunks = []
    async for chunk in client._stream_completion_async(messages):
        chunks.append(chunk)
    
    assert len(chunks) == 3
    assert chunks[0]["response"] == "Once"
    assert chunks[1]["response"] == " upon"
    assert chunks[2]["response"] == " a time"

@pytest.mark.asyncio
async def test_stream_completion_async_with_retry(client):
    """Test streaming completion with retry logic."""
    messages = [{"role": "user", "content": "Hello"}]
    
    # Mock first call to fail, second to succeed
    call_count = 0
    async def mock_create(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise Exception("timeout error")  # Retryable error
        else:
            async def success_stream():
                yield MockChatCompletionChunk(content="Success")
            return success_stream()
    
    client.async_client.chat.completions.create.side_effect = mock_create
    
    # Collect streaming results
    chunks = []
    async for chunk in client._stream_completion_async(messages):
        chunks.append(chunk)
    
    # Should succeed on retry
    assert len(chunks) == 1
    assert chunks[0]["response"] == "Success"

@pytest.mark.asyncio
async def test_stream_completion_async_error_handling(client):
    """Test error handling in streaming completion."""
    messages = [{"role": "user", "content": "Hello"}]
    
    # Mock the client to raise a non-retryable error
    client.async_client.chat.completions.create.side_effect = Exception("Authentication failed")
    
    # Collect streaming results
    chunks = []
    async for chunk in client._stream_completion_async(messages):
        chunks.append(chunk)
    
    # Should yield an error chunk
    assert len(chunks) == 1
    assert "error" in chunks[0]
    assert chunks[0]["error"] is True
    assert "Authentication failed" in chunks[0]["response"]

# ---------------------------------------------------------------------------
# Provider-specific tests
# ---------------------------------------------------------------------------

def test_deepseek_provider_detection(mock_configuration, mock_openai_sdk):
    """Test DeepSeek provider detection."""
    from chuk_llm.llm.providers.openai_client import OpenAILLMClient
    
    client = OpenAILLMClient(
        model="deepseek-chat",
        api_base="https://api.deepseek.com/v1"
    )
    
    assert client.detected_provider == "deepseek"
    assert client.api_base == "https://api.deepseek.com/v1"

def test_groq_provider_detection(mock_configuration, mock_openai_sdk):
    """Test Groq provider detection."""
    from chuk_llm.llm.providers.openai_client import OpenAILLMClient
    
    client = OpenAILLMClient(
        model="llama-3-8b-8192",
        api_base="https://api.groq.com/openai/v1"
    )
    
    assert client.detected_provider == "groq"

def test_together_provider_detection(mock_configuration, mock_openai_sdk):
    """Test Together AI provider detection."""
    from chuk_llm.llm.providers.openai_client import OpenAILLMClient
    
    client = OpenAILLMClient(
        model="meta-llama/Llama-2-7b-chat-hf",
        api_base="https://api.together.xyz/v1"
    )
    
    assert client.detected_provider == "together"

# ---------------------------------------------------------------------------
# Interface compliance tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_interface_compliance(client):
    """Test that create_completion follows the correct interface."""
    # Test non-streaming - should return awaitable
    messages = [{"role": "user", "content": "Test"}]
    
    # Mock the completion
    async def mock_completion(messages, tools, **kwargs):
        return {"response": "Test response", "tool_calls": []}
    
    client._regular_completion = mock_completion
    
    # Non-streaming should return awaitable
    result_coro = client.create_completion(messages, stream=False)
    assert asyncio.iscoroutine(result_coro)
    
    result = await result_coro
    assert isinstance(result, dict)
    assert "response" in result
    
    # Streaming should return async iterator
    async def mock_stream(messages, tools, **kwargs):
        yield {"response": "chunk1", "tool_calls": []}
        yield {"response": "chunk2", "tool_calls": []}
    
    client._stream_completion_async = mock_stream
    
    stream_result = client.create_completion(messages, stream=True)
    assert hasattr(stream_result, "__aiter__")
    
    chunks = []
    async for chunk in stream_result:
        chunks.append(chunk)
    
    assert len(chunks) == 2

# ---------------------------------------------------------------------------
# Error handling and edge cases
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_streaming_error_handling(client, monkeypatch):
    """Test error handling in streaming mode."""
    messages = [{"role": "user", "content": "test"}]

    # Mock streaming with error
    async def error_stream(messages, tools, **kwargs):
        yield {"response": "Starting...", "tool_calls": []}
        yield {"response": "Streaming error: Test error", "tool_calls": [], "error": True}

    monkeypatch.setattr(client, "_stream_completion_async", error_stream)

    stream_result = client.create_completion(messages, stream=True)
    chunks = []
    async for chunk in stream_result:
        chunks.append(chunk)

    assert len(chunks) == 2
    assert chunks[0]["response"] == "Starting..."
    assert chunks[1]["error"] is True

@pytest.mark.asyncio
async def test_non_streaming_error_handling(client, monkeypatch):
    """Test error handling in non-streaming mode."""
    messages = [{"role": "user", "content": "test"}]

    # Mock error in regular completion
    async def error_completion(messages, tools, **kwargs):
        return {"response": "Error: Test error", "tool_calls": [], "error": True}

    monkeypatch.setattr(client, "_regular_completion", error_completion)

    result = await client.create_completion(messages, stream=False)

    assert result["error"] is True
    assert "Test error" in result["response"]

# ---------------------------------------------------------------------------
# Integration tests - FIXED
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_full_integration_non_streaming(client):
    """Test full integration for non-streaming completion."""
    messages = [
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "Hello"}
    ]
    
    # Mock the actual OpenAI API call
    mock_response = MockChatCompletion(content="Hello! How can I help you today?")
    
    captured_params = {}
    async def mock_create(**kwargs):
        captured_params.update(kwargs)
        return mock_response
    
    client.async_client.chat.completions.create.side_effect = mock_create
    
    result = await client.create_completion(messages, stream=False)
    
    assert result["response"] == "Hello! How can I help you today?"
    assert result["tool_calls"] == []
    
    # Verify request structure
    assert captured_params["model"] == "gpt-4o-mini"
    assert len(captured_params["messages"]) == 2
    assert captured_params["messages"][0]["role"] == "system"

@pytest.mark.asyncio
async def test_full_integration_streaming(client):
    """Test full integration for streaming completion."""
    messages = [{"role": "user", "content": "Tell me a story"}]
    
    # Mock streaming response
    async def mock_stream():
        story_parts = ["Once", " upon", " a", " time..."]
        for part in story_parts:
            yield MockChatCompletionChunk(content=part)
    
    client.async_client.chat.completions.create.return_value = mock_stream()
    
    # Collect all chunks
    story_parts = []
    async for chunk in client.create_completion(messages, stream=True):
        story_parts.append(chunk["response"])
    
    # Verify we got all parts
    assert len(story_parts) == 4
    assert story_parts == ["Once", " upon", " a", " time..."]

# ---------------------------------------------------------------------------
# Configuration integration tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_configuration_feature_validation(client):
    """Test that configuration properly validates features."""
    messages = [{"role": "user", "content": "Hello"}]
    tools = [{"function": {"name": "test_tool"}}]
    
    # Mock configuration to support only streaming (not tools)
    client.supports_feature = lambda feature: feature == "streaming"
    
    # Test with tools - should be filtered out
    result = client.create_completion(messages, tools=tools, stream=True)
    assert hasattr(result, "__aiter__")  # Should still return stream

@pytest.mark.asyncio
async def test_provider_specific_adjustments(deepseek_client):
    """Test provider-specific parameter adjustments."""
    assert deepseek_client.detected_provider == "deepseek"
    
    # Test parameter adjustment
    params = {"temperature": 0.7, "max_tokens": 1000}
    adjusted = deepseek_client._adjust_parameters_for_provider(params)
    
    # Should include the parameters
    assert "temperature" in adjusted
    assert "max_tokens" in adjusted

def test_tool_name_sanitization(client):
    """Test tool name sanitization."""
    tools = [{"function": {"name": "invalid@name"}}]
    sanitized = client._sanitize_tool_names(tools)
    assert sanitized[0]["function"]["name"] == "invalid_name"

# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_normalize_message_no_attributes(client):
    """Test normalizing message with no recognizable attributes."""
    mock_message = types.SimpleNamespace()  # Empty object
    
    result = client._normalize_message(mock_message)
    
    assert result["response"] == ""
    assert result["tool_calls"] == []

def test_normalize_message_tool_call_missing_function(client):
    """Test normalizing message with malformed tool call."""
    mock_tool_call = types.SimpleNamespace(
        id="call_123"
        # Missing function attribute
    )
    
    mock_message = types.SimpleNamespace(
        content="",
        tool_calls=[mock_tool_call]
    )
    
    result = client._normalize_message(mock_message)
    
    # Should handle gracefully and not include malformed tool call
    assert result["response"] == ""
    assert result["tool_calls"] == []

@pytest.mark.asyncio
async def test_complex_conversation_flow(client):
    """Test a complex conversation with multiple message types."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi! How can I help?"},
        {"role": "user", "content": [
            {"type": "text", "text": "Look at this image"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
        ]}
    ]
    
    tools = [{"function": {"name": "analyze_image", "parameters": {}}}]

    # Mock completion
    async def mock_completion(messages, tools, **kwargs):
        return {"response": "Complex conversation complete", "tool_calls": []}

    client._regular_completion = mock_completion

    result = await client.create_completion(messages, tools=tools, stream=False)
    
    assert result["response"] == "Complex conversation complete"
    assert result["tool_calls"] == []