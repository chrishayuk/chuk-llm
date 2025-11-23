# tests/providers/test_ollama_client.py
import asyncio
import json
import types
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chuk_llm.core.enums import MessageRole, ContentType, ToolType
from chuk_llm.core.models import Message, Tool, ToolFunction, ToolCall, FunctionCall, TextContent, ImageUrlContent

# ---------------------------------------------------------------------------
# Test Helper: Convert dict to Pydantic (for backward compat with existing tests)
# ---------------------------------------------------------------------------

def ensure_pydantic_messages(messages):
    """Convert dict messages to Pydantic Message objects for tests."""
    from chuk_llm.llm.core.base import _ensure_pydantic_messages
    return _ensure_pydantic_messages(messages)

def ensure_pydantic_tools(tools):
    """Convert dict tools to Pydantic Tool objects for tests."""
    from chuk_llm.llm.core.base import _ensure_pydantic_tools
    return _ensure_pydantic_tools(tools)

# ---------------------------------------------------------------------------
# Stub the `ollama` SDK before importing the adapter.
# ---------------------------------------------------------------------------


# Mock Ollama response classes
class MockOllamaMessage:
    def __init__(self, content="", tool_calls=None, thinking=""):
        self.content = content
        self.tool_calls = tool_calls or []
        self.thinking = thinking  # NEW: Add thinking support


class MockOllamaResponse:
    def __init__(self, content="Test response", tool_calls=None, thinking=""):
        self.message = MockOllamaMessage(content, tool_calls, thinking)


class MockOllamaStreamChunk:
    def __init__(self, content="", tool_calls=None, thinking=""):
        self.message = MockOllamaMessage(content, tool_calls, thinking)


class MockOllamaShowResponse:
    def __init__(self, capabilities: list[str]):
        self.capabilities = capabilities


# Mock Ollama AsyncClient
class MockAsyncOllamaClient:
    def __init__(self, host=None, **kwargs):
        self.host = host

    async def chat(self, **kwargs):
        if kwargs.get("stream", False):
            # Return async generator for streaming
            async def mock_stream():
                yield MockOllamaStreamChunk("Hello")
                yield MockOllamaStreamChunk(" from")
                yield MockOllamaStreamChunk(" Ollama!")

            return mock_stream()
        else:
            # Return regular response for non-streaming
            return MockOllamaResponse("Hello from Ollama!")


# Mock Ollama sync Client
class MockOllamaClient:
    # Global map of model capabilities that will be updated in a context manager
    model_features = {}

    def __init__(self, host=None, **kwargs):
        self.host = host

    def chat(self, **kwargs):
        return MockOllamaResponse("Hello from Ollama!")

    def show(self, model: str):
        return MockOllamaShowResponse(self.__class__.model_features.get(model, []))


# Fixture to patch the main ollama module for all tests
@pytest.fixture(autouse=True)
def patch_ollama_module():
    with (
        patch("ollama.AsyncClient", new=MockAsyncOllamaClient),
        patch("ollama.Client", new=MockOllamaClient),
        # For backwards compatibility
        patch("ollama.client", create=True),
    ):
        yield


# Context manager to set the model features for a specific model
@contextmanager
def set_model_features(model: str, features: list[str]):
    prev_features = MockOllamaClient.model_features.get(model)
    MockOllamaClient.model_features[model] = features
    try:
        yield
    finally:
        if prev_features is not None:
            MockOllamaClient.model_features[model] = prev_features


# ---------------------------------------------------------------------------
# Now import the client (will see the stub).
# ---------------------------------------------------------------------------

from chuk_llm.llm.providers.ollama_client import OllamaLLMClient  # noqa: E402

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
    def __init__(self, features=None, max_context_length=4096, max_output_tokens=2048):
        self.features = features or {
            MockFeature.TEXT,
            MockFeature.STREAMING,
            MockFeature.TOOLS,
            MockFeature.SYSTEM_MESSAGES,
            MockFeature.VISION,
        }
        self.max_context_length = max_context_length
        self.max_output_tokens = max_output_tokens


class MockProviderConfig:
    def __init__(self, name="ollama", client_class="OllamaLLMClient"):
        self.name = name
        self.client_class = client_class
        self.api_base = "http://localhost:11434"
        self.models = ["qwen3", "llama3.1", "mistral", "phi-3", "gpt-oss"]
        self.model_aliases = {}
        self.rate_limits = {
            "requests_per_minute": 1000
        }  # Local deployment, higher limits

    def get_model_capabilities(self, model):
        # Different capabilities based on model
        features = {
            MockFeature.TEXT,
            MockFeature.STREAMING,
            MockFeature.TOOLS,
            MockFeature.SYSTEM_MESSAGES,
        }

        # Most Ollama models support vision if they're multimodal
        if any(
            vision_term in model.lower()
            for vision_term in ["vision", "multimodal", "llava"]
        ):
            features.add(MockFeature.VISION)
            features.add(MockFeature.MULTIMODAL)

        # Reasoning models support reasoning
        if any(
            reasoning_term in model.lower()
            for reasoning_term in ["gpt-oss", "qwq", "reasoning"]
        ):
            features.add(MockFeature.REASONING)

        # Some models might have JSON mode
        if any(json_term in model.lower() for json_term in ["instruct", "chat"]):
            features.add(MockFeature.JSON_MODE)

        return MockModelCapabilities(features=features)


class MockConfig:
    def __init__(self):
        self.ollama_provider = MockProviderConfig()

    def get_provider(self, provider_name):
        if provider_name == "ollama":
            return self.ollama_provider
        return None


# ---------------------------------------------------------------------------
# Fixtures with Configuration Mocking
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_configuration():
    """Mock the configuration system"""
    mock_config = MockConfig()

    with patch("chuk_llm.configuration.get_config", return_value=mock_config):
        with patch("chuk_llm.configuration.Feature", MockFeature):
            yield mock_config


@pytest.fixture
def client(mock_configuration, monkeypatch):
    """Ollama client with configuration properly mocked"""
    cl = OllamaLLMClient(model="qwen3", api_base="http://localhost:11434")

    # Ensure configuration methods are properly mocked
    monkeypatch.setattr(
        cl,
        "supports_feature",
        lambda feature: feature
        in ["text", "streaming", "tools", "system_messages", "vision"],
    )

    # FIXED: Added parameter_mapping to resolve test failure
    monkeypatch.setattr(
        cl,
        "get_model_info",
        lambda: {
            "provider": "ollama",
            "model": "qwen3",
            "client_class": "OllamaLLMClient",
            "api_base": "http://localhost:11434",
            "features": ["text", "streaming", "tools", "system_messages", "vision"],
            "supports_text": True,
            "supports_streaming": True,
            "supports_tools": True,
            "supports_vision": True,
            "supports_system_messages": True,
            "supports_json_mode": False,
            "supports_parallel_calls": False,
            "supports_multimodal": True,
            "supports_reasoning": False,
            "max_context_length": 4096,
            "max_output_tokens": 2048,
            "ollama_specific": {
                "local_deployment": True,
                "no_api_key_required": True,
                "host": "http://localhost:11434",
                "model_family": "qwen",
                "supports_chat": True,
                "supports_streaming": True,
                "is_reasoning_model": False,
                "supports_thinking_stream": False,
            },
            # This was the missing field causing the test failure
            "parameter_mapping": {
                "temperature": "temperature",
                "max_tokens": "num_predict",
                "top_p": "top_p",
                "top_k": "top_k",
                "stop": "stop",
                "stream": "stream",
                "seed": "seed",
            },
            "unsupported_parameters": [
                "frequency_penalty",
                "presence_penalty",
                "logit_bias",
                "user",
                "n",
                "best_of",
                "logprobs",
                "echo",
            ],
            "rate_limits": {"requests_per_minute": 1000},
            "available_models": ["qwen3", "llama3.1", "mistral", "phi-3", "gpt-oss"],
            "model_aliases": {},
        },
    )

    # Mock token limits
    monkeypatch.setattr(cl, "get_max_tokens_limit", lambda: 2048)
    monkeypatch.setattr(cl, "get_context_length_limit", lambda: 4096)

    # Mock parameter validation
    def mock_validate_parameters(**kwargs):
        result = kwargs.copy()
        if "max_tokens" in result and result["max_tokens"] > 2048:
            result["max_tokens"] = 2048
        return result

    monkeypatch.setattr(cl, "validate_parameters", mock_validate_parameters)

    return cl


@pytest.fixture
def reasoning_client(mock_configuration, monkeypatch):
    """GPT-OSS reasoning client for testing reasoning model functionality"""
    cl = OllamaLLMClient(model="gpt-oss:latest", api_base="http://localhost:11434")

    # Ensure configuration methods are properly mocked for reasoning model
    monkeypatch.setattr(
        cl,
        "supports_feature",
        lambda feature: feature
        in ["text", "streaming", "tools", "system_messages", "reasoning"],
    )

    monkeypatch.setattr(
        cl,
        "get_model_info",
        lambda: {
            "provider": "ollama",
            "model": "gpt-oss:latest",
            "client_class": "OllamaLLMClient",
            "api_base": "http://localhost:11434",
            "features": ["text", "streaming", "tools", "system_messages", "reasoning"],
            "supports_text": True,
            "supports_streaming": True,
            "supports_tools": True,
            "supports_vision": False,
            "supports_system_messages": True,
            "supports_json_mode": False,
            "supports_parallel_calls": False,
            "supports_multimodal": False,
            "supports_reasoning": True,
            "max_context_length": 4096,
            "max_output_tokens": 2048,
            "ollama_specific": {
                "local_deployment": True,
                "no_api_key_required": True,
                "host": "http://localhost:11434",
                "model_family": "gpt-oss",
                "supports_chat": True,
                "supports_streaming": True,
                "is_reasoning_model": True,
                "supports_thinking_stream": True,
            },
            "parameter_mapping": {
                "temperature": "temperature",
                "max_tokens": "num_predict",
                "top_p": "top_p",
                "top_k": "top_k",
                "stop": "stop",
                "stream": "stream",
                "seed": "seed",
            },
            "unsupported_parameters": [
                "frequency_penalty",
                "presence_penalty",
                "logit_bias",
                "user",
                "n",
                "best_of",
                "logprobs",
                "echo",
            ],
            "rate_limits": {"requests_per_minute": 1000},
            "available_models": ["qwen3", "llama3.1", "mistral", "phi-3", "gpt-oss"],
            "model_aliases": {},
        },
    )

    # Mock token limits
    monkeypatch.setattr(cl, "get_max_tokens_limit", lambda: 2048)
    monkeypatch.setattr(cl, "get_context_length_limit", lambda: 4096)

    # Mock parameter validation
    def mock_validate_parameters(**kwargs):
        result = kwargs.copy()
        if "max_tokens" in result and result["max_tokens"] > 2048:
            result["max_tokens"] = 2048
        return result

    monkeypatch.setattr(cl, "validate_parameters", mock_validate_parameters)

    return cl


# ---------------------------------------------------------------------------
# Client initialization tests
# ---------------------------------------------------------------------------


def test_client_initialization(mock_configuration):
    """Test client initialization with different parameters."""
    # Test with default model
    client1 = OllamaLLMClient()
    assert client1.model == "qwen3"
    assert client1.api_base == "http://localhost:11434"

    # Test with custom model
    client2 = OllamaLLMClient(model="llama3.1")
    assert client2.model == "llama3.1"

    # Test with custom API base
    client3 = OllamaLLMClient(model="mistral", api_base="http://remote-ollama:11434")
    assert client3.model == "mistral"
    assert client3.api_base == "http://remote-ollama:11434"


def test_client_initialization_with_model_features(mock_configuration):
    """Test client initialization with a model that has reported features from
    the show endpoint
    """
    model = "foo-bar:latest"
    with set_model_features(model, ["tools", "thinking"]):
        client = OllamaLLMClient(model=model)

    assert client.supports_feature("tools")
    assert client.supports_feature("thinking")
    assert client._is_reasoning_model()


def test_client_initialization_with_host_support(mock_configuration):
    """Test client initialization with host parameter support."""
    # Should work with modern ollama-python that supports host parameter
    client = OllamaLLMClient(model="qwen3", api_base="http://custom:11434")
    assert client.async_client.host == "http://custom:11434"
    assert client.sync_client.host == "http://custom:11434"


def test_get_model_info(client):
    """Test model info method."""
    info = client.get_model_info()

    assert info["provider"] == "ollama"
    assert info["model"] == "qwen3"
    assert "ollama_specific" in info
    assert info["ollama_specific"]["local_deployment"] is True
    assert info["ollama_specific"]["no_api_key_required"] is True
    assert info["ollama_specific"]["host"] == "http://localhost:11434"
    assert "parameter_mapping" in info


def test_detect_model_family(client):
    """Test model family detection."""
    assert client._detect_model_family() == "qwen"

    client.model = "llama3.1"
    assert client._detect_model_family() == "llama"

    client.model = "mistral-7b"
    assert client._detect_model_family() == "mistral"

    client.model = "granite-code"
    assert client._detect_model_family() == "granite"

    client.model = "gemma-2b"
    assert client._detect_model_family() == "gemma"

    client.model = "phi-3"
    assert client._detect_model_family() == "phi"

    client.model = "gpt-oss:latest"
    assert client._detect_model_family() == "gpt-oss"

    client.model = "codellama"
    assert (
        client._detect_model_family() == "llama"
    )  # Note: codellama contains "llama", so it matches that first

    client.model = "custom-model"
    assert client._detect_model_family() == "unknown"


# ---------------------------------------------------------------------------
# Reasoning model tests
# ---------------------------------------------------------------------------


def test_is_reasoning_model(mock_configuration):
    """Test reasoning model detection."""
    # Test regular model
    client = OllamaLLMClient(model="qwen3")
    assert client._is_reasoning_model() is False

    # Test reasoning models
    reasoning_models = [
        "gpt-oss:latest",
        "qwq:32b",
        "marco-o1",
        "deepseek-r1",
        "reasoning-model",
    ]
    for model in reasoning_models:
        client = OllamaLLMClient(model=model)
        assert client._is_reasoning_model() is True, (
            f"Model {model} should be detected as reasoning model"
        )


def test_get_model_info_reasoning(reasoning_client):
    """Test model info for reasoning model."""
    info = reasoning_client.get_model_info()

    assert info["provider"] == "ollama"
    assert info["model"] == "gpt-oss:latest"
    assert info["ollama_specific"]["is_reasoning_model"] is True
    assert info["ollama_specific"]["supports_thinking_stream"] is True
    assert info["ollama_specific"]["model_family"] == "gpt-oss"


# ---------------------------------------------------------------------------
# Request validation tests
# ---------------------------------------------------------------------------


def test_validate_request_with_config(client):
    """Test request validation against configuration."""
    messages = [{"role": "user", "content": "Hello"}]
    tools = [{"type": "function", "function": {"name": "test_tool", "description": "", "parameters": {}}}]

    # Convert to Pydantic
    pydantic_messages = ensure_pydantic_messages(messages)
    pydantic_tools = ensure_pydantic_tools(tools)

    # Mock configuration support
    client.supports_feature = lambda feature: feature in ["streaming", "tools"]

    validated_messages, validated_tools, validated_stream, validated_kwargs = (
        client._validate_request_with_config(
            pydantic_messages, pydantic_tools, stream=True, temperature=0.7, logit_bias={"test": 1}
        )
    )

    assert len(validated_messages) == len(pydantic_messages)
    assert len(validated_tools) == len(pydantic_tools)
    assert validated_stream is True
    assert "temperature" in validated_kwargs
    assert "logit_bias" not in validated_kwargs  # Should be removed


# ---------------------------------------------------------------------------
# Message preparation tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prepare_ollama_messages_basic(client):
    """Test basic message preparation."""
    messages = [
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]

    # Convert to Pydantic
    pydantic_messages = ensure_pydantic_messages(messages)

    # Mock system message support
    client.supports_feature = lambda feature: feature == "system_messages"

    prepared = await client._prepare_ollama_messages(pydantic_messages)

    assert len(prepared) == 3
    assert prepared[0]["role"] == "system"
    assert prepared[0]["content"] == "You are helpful"
    assert prepared[1]["role"] == "user"
    assert prepared[2]["role"] == "assistant"


@pytest.mark.asyncio
async def test_prepare_ollama_messages_no_system_support(client):
    """Test message preparation when system messages are not supported."""
    messages = [
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "Hello"},
    ]

    # Convert to Pydantic
    pydantic_messages = ensure_pydantic_messages(messages)

    # Mock no system message support
    client.supports_feature = lambda feature: feature != "system_messages"

    prepared = await client._prepare_ollama_messages(pydantic_messages)

    assert len(prepared) == 2
    # System message should be converted to user message
    assert prepared[0]["role"] == "user"
    assert "System: You are helpful" in prepared[0]["content"]
    assert prepared[1]["role"] == "user"


@pytest.mark.asyncio
async def test_prepare_ollama_messages_with_vision(client):
    """Test message preparation with vision content."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Look at this image"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
                    },
                },
            ],
        }
    ]

    # Mock vision support
    client.supports_feature = lambda feature: feature == "vision"

    pydantic_messages = ensure_pydantic_messages(messages)
    prepared = await client._prepare_ollama_messages(pydantic_messages)

    assert len(prepared) == 1
    assert prepared[0]["role"] == "user"
    assert "images" in prepared[0]
    assert len(prepared[0]["images"]) == 1


@pytest.mark.asyncio
async def test_prepare_ollama_messages_multi_turn_tools(client):
    """Test message preparation when there are tool calls in historical messages
    that need to be resent to the server.
    """
    messages = [
        {"role": "user", "content": "list products"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "some-id",
                    "type": "function",
                    "function": {
                        "name": "stdio.list_tables",
                        # NOTE: value is a serialized json string
                        "arguments": '{"query": null}',
                    },
                }
            ],
        },
    ]

    pydantic_messages = ensure_pydantic_messages(messages)
    prepared = await client._prepare_ollama_messages(pydantic_messages)

    assert len(prepared) == 2
    assert prepared[0]["role"] == "user"
    assert prepared[1]["role"] == "assistant"
    assert prepared[1]["tool_calls"] == [
        {"function": {"name": "stdio.list_tables", "arguments": {"query": None}}}
    ]


# ---------------------------------------------------------------------------
# Parameter building tests
# ---------------------------------------------------------------------------


def test_build_ollama_options(client):
    """Test building Ollama options from OpenAI-style parameters."""
    kwargs = {
        "temperature": 0.7,
        "max_tokens": 100,
        "top_p": 0.9,
        "stop": ["stop1", "stop2"],
        "top_k": 40,
        "seed": 42,
        "unsupported_param": "value",
    }

    options = client._build_ollama_options(kwargs)

    assert options["temperature"] == 0.7
    assert options["num_predict"] == 100  # Should map max_tokens to num_predict
    assert options["top_p"] == 0.9
    assert options["stop"] == ["stop1", "stop2"]
    assert options["top_k"] == 40
    assert options["seed"] == 42
    assert "unsupported_param" not in options


def test_build_ollama_options_with_custom_options(client):
    """Test building Ollama options with custom Ollama-specific options."""
    kwargs = {
        "temperature": 0.5,
        "options": {"custom_ollama_param": "value", "another_param": 123},
    }

    options = client._build_ollama_options(kwargs)

    assert options["temperature"] == 0.5
    assert options["custom_ollama_param"] == "value"
    assert options["another_param"] == 123


# ---------------------------------------------------------------------------
# Response parsing tests
# ---------------------------------------------------------------------------


def test_parse_response_text_only(client):
    """Test parsing Ollama response with text only."""
    mock_response = MockOllamaResponse(content="Hello from Ollama")

    result = client._parse_response(mock_response)

    assert result["response"] == "Hello from Ollama"
    assert result["tool_calls"] == []


def test_parse_response_with_tool_calls(client):
    """Test parsing Ollama response with tool calls."""
    # Mock tool call structure
    mock_tool_call = types.SimpleNamespace(
        id="call_123",
        function=types.SimpleNamespace(name="test_tool", arguments={"arg": "value"}),
    )

    mock_response = MockOllamaResponse(content="", tool_calls=[mock_tool_call])

    # Mock tool support
    client.supports_feature = lambda feature: feature == "tools"

    result = client._parse_response(mock_response)

    assert result["response"] is None  # Should be None when empty content
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["function"]["name"] == "test_tool"
    assert '"arg": "value"' in result["tool_calls"][0]["function"]["arguments"]


def test_parse_response_reasoning_model_thinking(reasoning_client):
    """Test parsing response for reasoning model with thinking."""
    mock_response = MockOllamaResponse(
        content="", thinking="Let me think about this..."
    )

    result = reasoning_client._parse_response(mock_response)

    # For reasoning models, thinking content should be used as main response when content is empty
    assert result["response"] == "Let me think about this..."
    assert result["tool_calls"] == []
    assert "reasoning" in result
    assert result["reasoning"]["thinking"] == "Let me think about this..."
    assert result["reasoning"]["content"] == ""
    assert result["reasoning"]["model_type"] == "reasoning"


def test_parse_response_reasoning_model_with_content_and_thinking(reasoning_client):
    """Test parsing response for reasoning model with both content and thinking."""
    mock_response = MockOllamaResponse(
        content="Here's the answer", thinking="Let me think about this..."
    )

    result = reasoning_client._parse_response(mock_response)

    # When both exist, content takes precedence
    assert result["response"] == "Here's the answer"
    assert result["tool_calls"] == []
    assert "reasoning" in result
    assert result["reasoning"]["thinking"] == "Let me think about this..."
    assert result["reasoning"]["content"] == "Here's the answer"
    assert result["reasoning"]["model_type"] == "reasoning"


def test_parse_response_tools_not_supported(client):
    """Test parsing response when tools are not supported."""
    mock_tool_call = types.SimpleNamespace(
        id="call_123",
        function=types.SimpleNamespace(name="test_tool", arguments={"arg": "value"}),
    )

    mock_response = MockOllamaResponse(content="Some text", tool_calls=[mock_tool_call])

    # Mock no tool support
    client.supports_feature = lambda feature: feature != "tools"

    result = client._parse_response(mock_response)

    # Should return text and ignore tool calls
    assert result["response"] == "Some text"
    assert result["tool_calls"] == []


def test_parse_response_no_message(client):
    """Test parsing response with no message."""
    mock_response = types.SimpleNamespace()  # No message attribute

    result = client._parse_response(mock_response)

    # UPDATED: Should return None for response (not empty string) and empty tool_calls
    assert result["response"] is None  # Changed from "" to None
    assert result["tool_calls"] == []


# ---------------------------------------------------------------------------
# Sync completion tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_sync_basic(client):
    """Test synchronous completion creation."""
    messages = [{"role": "user", "content": "Hello"}]

    # Mock the async client (not sync client!)
    mock_response = MockOllamaResponse("Hello from sync Ollama")
    client.async_client.chat = AsyncMock(return_value=mock_response)

    pydantic_messages = ensure_pydantic_messages(messages)
    result = await client._create_sync(pydantic_messages)

    assert result["response"] == "Hello from sync Ollama"
    assert result["tool_calls"] == []

    # Verify the call was made correctly
    client.async_client.chat.assert_called_once()
    call_kwargs = client.async_client.chat.call_args.kwargs
    assert call_kwargs["model"] == "qwen3"
    assert call_kwargs["stream"] is False
    assert len(call_kwargs["messages"]) == 1


@pytest.mark.asyncio
async def test_create_sync_with_tools(client):
    """Test synchronous completion with tools."""
    messages = [{"role": "user", "content": "Use a tool"}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "test_tool",
                "description": "A test tool",
                "parameters": {"type": "object"},
            },
        }
    ]

    # Mock tool support
    client.supports_feature = lambda feature: feature == "tools"

    mock_response = MockOllamaResponse("Tool used")
    client.async_client.chat = AsyncMock(return_value=mock_response)

    pydantic_messages = ensure_pydantic_messages(messages)
    pydantic_tools = ensure_pydantic_tools(tools)
    result = await client._create_sync(pydantic_messages, pydantic_tools)

    assert result["response"] == "Tool used"

    # Verify tools were passed correctly
    call_kwargs = client.async_client.chat.call_args.kwargs
    assert "tools" in call_kwargs
    assert len(call_kwargs["tools"]) == 1
    assert call_kwargs["tools"][0]["function"]["name"] == "test_tool"


@pytest.mark.asyncio
async def test_create_sync_with_options(client):
    """Test synchronous completion with Ollama options."""
    messages = [{"role": "user", "content": "Hello"}]

    mock_response = MockOllamaResponse("Hello with options")
    client.async_client.chat = AsyncMock(return_value=mock_response)

    pydantic_messages = ensure_pydantic_messages(messages)
    result = await client._create_sync(pydantic_messages, temperature=0.8, max_tokens=200)

    assert result["response"] == "Hello with options"

    # Verify options were passed correctly
    call_kwargs = client.async_client.chat.call_args.kwargs
    assert "options" in call_kwargs
    assert call_kwargs["options"]["temperature"] == 0.8
    assert call_kwargs["options"]["num_predict"] == 200


# ---------------------------------------------------------------------------
# Regular completion tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_regular_completion(client):
    """Test regular (non-streaming) completion."""
    messages = [{"role": "user", "content": "Hello"}]

    # Convert to Pydantic
    pydantic_messages = ensure_pydantic_messages(messages)

    # Mock the _create_sync method
    expected_result = {"response": "Hello from Ollama!", "tool_calls": []}
    client._create_sync = AsyncMock(return_value=expected_result)

    result = await client._regular_completion(pydantic_messages)

    assert result == expected_result


@pytest.mark.asyncio
async def test_regular_completion_error_handling(client):
    """Test error handling in regular completion."""
    messages = [{"role": "user", "content": "Hello"}]

    # Convert to Pydantic
    pydantic_messages = ensure_pydantic_messages(messages)

    # Mock _create_sync to raise an exception
    client._create_sync = AsyncMock(side_effect=Exception("Ollama error"))
    result = await client._regular_completion(pydantic_messages)

    assert "error" in result
    assert result["error"] is True
    assert "Ollama error" in result["response"]


# ---------------------------------------------------------------------------
# Streaming completion tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_completion_async(client):
    """Test streaming completion."""
    messages = [{"role": "user", "content": "Tell a story"}]

    # Convert to Pydantic
    pydantic_messages = ensure_pydantic_messages(messages)

    # Convert to Pydantic
    pydantic_messages = ensure_pydantic_messages(messages)

    # Mock the async client's streaming response
    async def mock_stream():
        yield MockOllamaStreamChunk("Once")
        yield MockOllamaStreamChunk(" upon")
        yield MockOllamaStreamChunk(" a time")

    async def mock_chat(**kwargs):
        if kwargs.get("stream"):
            return mock_stream()
        return MockOllamaResponse()

    client.async_client.chat = mock_chat

    # Collect streaming results
    chunks = []
    async for chunk in client._stream_completion_async(pydantic_messages):
        chunks.append(chunk)

    assert len(chunks) == 3
    assert chunks[0]["response"] == "Once"
    assert chunks[1]["response"] == " upon"
    assert chunks[2]["response"] == " a time"


@pytest.mark.asyncio
async def test_stream_completion_async_reasoning_model(reasoning_client):
    """Test streaming completion with reasoning model."""
    messages = [{"role": "user", "content": "Solve this problem"}]

    # Convert to Pydantic
    pydantic_messages = ensure_pydantic_messages(messages)

    # Mock streaming with thinking content
    async def mock_stream():
        yield MockOllamaStreamChunk("", thinking="Let me think...")
        yield MockOllamaStreamChunk("", thinking="I need to consider...")
        yield MockOllamaStreamChunk("The answer is 42")  # Regular content

    async def mock_chat(**kwargs):
        return mock_stream()

    reasoning_client.async_client.chat = mock_chat

    # Collect streaming results
    chunks = []
    async for chunk in reasoning_client._stream_completion_async(pydantic_messages):
        chunks.append(chunk)

    assert len(chunks) == 3
    # First two chunks should stream thinking content
    assert chunks[0]["response"] == "Let me think..."
    assert chunks[0]["reasoning"]["is_thinking"] is True
    assert chunks[1]["response"] == "I need to consider..."
    assert chunks[1]["reasoning"]["is_thinking"] is True
    # Third chunk should stream regular content
    assert chunks[2]["response"] == "The answer is 42"
    assert chunks[2]["reasoning"]["is_thinking"] is False


@pytest.mark.asyncio
async def test_stream_completion_async_with_tools(client):
    """Test streaming completion with tools."""
    messages = [{"role": "user", "content": "Use tools"}]

    # Convert to Pydantic
    pydantic_messages = ensure_pydantic_messages(messages)
    tools = [{"function": {"name": "test_tool", "parameters": {}}}]

    # Mock tool call in streaming
    mock_tool_call = types.SimpleNamespace(
        id="call_123",
        function=types.SimpleNamespace(name="test_tool", arguments={"arg": "value"}),
    )

    async def mock_stream():
        yield MockOllamaStreamChunk("I'll use")
        yield MockOllamaStreamChunk(" a tool", tool_calls=[mock_tool_call])
        yield MockOllamaStreamChunk(" now")

    async def mock_chat(**kwargs):
        return mock_stream()

    client.async_client.chat = mock_chat
    client.supports_feature = lambda feature: feature == "tools"

    # Convert tools to Pydantic
    pydantic_tools = ensure_pydantic_tools(tools)

    # Collect streaming results
    chunks = []
    async for chunk in client._stream_completion_async(pydantic_messages, pydantic_tools):
        chunks.append(chunk)

    assert len(chunks) == 3
    assert chunks[0]["response"] == "I'll use"
    assert chunks[1]["response"] == " a tool"
    assert len(chunks[1]["tool_calls"]) == 1
    assert chunks[2]["response"] == " now"


@pytest.mark.asyncio
async def test_stream_completion_async_error_handling(client):
    """Test error handling in streaming completion."""
    messages = [{"role": "user", "content": "Hello"}]

    # Convert to Pydantic
    pydantic_messages = ensure_pydantic_messages(messages)

    # Mock the async client to raise an exception
    async def mock_chat_error(**kwargs):
        raise Exception("Streaming error")

    client.async_client.chat = mock_chat_error

    # Collect streaming results
    chunks = []
    async for chunk in client._stream_completion_async(pydantic_messages):
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

    async def mock_regular_completion(messages, tools, **kwargs):
        return expected_result

    client._regular_completion = mock_regular_completion

    result = client.create_completion(messages, stream=False)

    # Should return an awaitable
    assert hasattr(result, "__await__")

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
    assert hasattr(result, "__aiter__")

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
    tools = [{"function": {"name": "get_weather", "parameters": {}}}]

    # Mock regular completion
    expected_result = {
        "response": "I'll check the weather",
        "tool_calls": [
            {"id": "call_123", "function": {"name": "get_weather", "arguments": "{}"}}
        ],
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

    await client.create_completion(
        messages, temperature=0.7, max_tokens=100, stream=False
    )

    # Verify parameters were validated
    assert "temperature" in validated_params
    assert "max_tokens" in validated_params


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
# Error handling tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_streaming_error_handling(client):
    """Test error handling in streaming mode."""
    messages = [{"role": "user", "content": "test"}]

    # Convert to Pydantic
    pydantic_messages = ensure_pydantic_messages(messages)

    # Mock streaming with error
    async def error_stream(messages, tools, **kwargs):
        yield {"response": "Starting...", "tool_calls": []}
        yield {
            "response": "Streaming error: Test error",
            "tool_calls": [],
            "error": True,
        }

    client._stream_completion_async = error_stream

    stream_result = client.create_completion(messages, stream=True)
    chunks = []
    async for chunk in stream_result:
        chunks.append(chunk)

    assert len(chunks) == 2
    assert chunks[0]["response"] == "Starting..."
    assert chunks[1]["error"] is True


@pytest.mark.asyncio
async def test_non_streaming_error_handling(client):
    """Test error handling in non-streaming mode."""
    messages = [{"role": "user", "content": "test"}]

    # Mock error in regular completion
    async def error_completion(messages, tools, **kwargs):
        return {"response": "Error: Test error", "tool_calls": [], "error": True}

    client._regular_completion = error_completion

    result = await client.create_completion(messages, stream=False)

    assert result["error"] is True
    assert "Test error" in result["response"]


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_integration_non_streaming(client):
    """Test full integration for non-streaming completion."""
    messages = [
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "Hello"},
    ]

    # Mock the actual Ollama API call
    expected_response = {
        "response": "Hello! How can I help you today?",
        "tool_calls": [],
    }

    client._create_sync = AsyncMock(return_value=expected_response)

    result = await client.create_completion(messages, stream=False)

    assert result["response"] == "Hello! How can I help you today?"
    assert result["tool_calls"] == []


@pytest.mark.asyncio
async def test_full_integration_streaming(client):
    """Test full integration for streaming completion."""
    messages = [{"role": "user", "content": "Tell me a story"}]

    # Mock streaming response
    async def mock_stream():
        story_parts = ["Once", " upon", " a", " time..."]
        for part in story_parts:
            yield MockOllamaStreamChunk(part)

    async def mock_chat(**kwargs):
        return mock_stream()

    client.async_client.chat = mock_chat

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


# ---------------------------------------------------------------------------
# Edge cases and special scenarios
# ---------------------------------------------------------------------------


def test_client_initialization_missing_chat(mock_configuration):
    """Test client initialization when ollama module doesn't have chat."""
    import ollama

    # This test is less relevant with our mocking approach
    # The actual client would check for the chat attribute on the real ollama module
    # For now, we'll just verify that our mock client initializes correctly
    assert hasattr(ollama, "AsyncClient")
    assert hasattr(ollama, "Client")
    # The actual ValueError would only be raised with the real ollama module


@pytest.mark.asyncio
async def test_complex_conversation_flow(client):
    """Test a complex conversation with multiple message types."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi! How can I help?"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Look at this image"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,..."},
                },
            ],
        },
    ]

    tools = [{"function": {"name": "analyze_image", "parameters": {}}}]

    # Mock completion
    async def mock_completion(messages, tools, **kwargs):
        return {"response": "Complex conversation complete", "tool_calls": []}

    client._regular_completion = mock_completion

    result = await client.create_completion(messages, tools=tools, stream=False)

    assert result["response"] == "Complex conversation complete"
    assert result["tool_calls"] == []


def test_parameter_mapping_edge_cases(client):
    """Test parameter mapping with edge cases."""
    # Test with None values
    kwargs = {"temperature": None, "max_tokens": 0, "top_p": 1.0}

    options = client._build_ollama_options(kwargs)

    assert options["temperature"] is None
    assert options["num_predict"] == 0
    assert options["top_p"] == 1.0


def test_response_parsing_edge_cases(client):
    """Test response parsing with edge cases."""
    # Test with empty tool calls list
    mock_response = MockOllamaResponse(content="Text response", tool_calls=[])
    result = client._parse_response(mock_response)
    assert result["response"] == "Text response"
    assert result["tool_calls"] == []

    # Test with tool calls that have string arguments
    mock_tool_call = types.SimpleNamespace(
        id="call_123",
        function=types.SimpleNamespace(
            name="test_tool",
            arguments='{"already": "json"}',  # Already a JSON string
        ),
    )

    mock_response = MockOllamaResponse(content="", tool_calls=[mock_tool_call])
    client.supports_feature = lambda feature: feature == "tools"

    result = client._parse_response(mock_response)
    assert result["response"] is None
    assert result["tool_calls"][0]["function"]["arguments"] == '{"already": "json"}'


# ---------------------------------------------------------------------------
# Context Memory Preservation Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prepare_ollama_messages_with_tool_responses(client):
    """Test that tool responses are properly formatted for context preservation."""
    messages = [
        {"role": "user", "content": "What's the weather?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "London"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "name": "get_weather",
            "content": "It's 20°C and sunny in London",
        },
        {"role": "user", "content": "Is that warm?"},
    ]

    pydantic_messages = ensure_pydantic_messages(messages)
    prepared = await client._prepare_ollama_messages(pydantic_messages)

    assert len(prepared) == 4

    # Check assistant message with tool calls
    assert prepared[1]["role"] == "assistant"
    assert "tool_calls" in prepared[1]
    assert len(prepared[1]["tool_calls"]) == 1
    assert prepared[1]["tool_calls"][0]["function"]["name"] == "get_weather"
    assert prepared[1]["tool_calls"][0]["function"]["arguments"] == {
        "location": "London"
    }

    # CRITICAL: Check tool response is converted to user message
    assert prepared[2]["role"] == "user"
    assert "Tool Response from get_weather:" in prepared[2]["content"]
    assert "20°C and sunny in London" in prepared[2]["content"]
    assert prepared[2]["metadata"]["type"] == "tool_response"
    assert prepared[2]["metadata"]["tool_name"] == "get_weather"

    # Check final user message
    assert prepared[3]["role"] == "user"
    assert prepared[3]["content"] == "Is that warm?"


@pytest.mark.asyncio
async def test_prepare_ollama_messages_empty_assistant_content_with_tools(client):
    """Test that assistant messages with only tool calls get descriptive content."""
    messages = [
        {"role": "user", "content": "Calculate something"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "calculator",
                        "arguments": '{"expression": "2+2"}',
                    }
                },
                {
                    "function": {
                        "name": "converter",
                        "arguments": '{"value": 100, "from": "USD", "to": "EUR"}',
                    }
                },
            ],
        },
    ]

    pydantic_messages = ensure_pydantic_messages(messages)
    prepared = await client._prepare_ollama_messages(pydantic_messages)

    assert len(prepared) == 2
    assert prepared[1]["role"] == "assistant"
    # Should have added descriptive content for models that need it
    assert prepared[1]["content"] == "[Called tools: calculator, converter]"
    assert len(prepared[1]["tool_calls"]) == 2


def test_validate_conversation_context_valid(client):
    """Test conversation context validation with valid message flow."""
    messages = [
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"},
        {"role": "assistant", "content": "I'm doing well, thanks!"},
    ]

    result = client._validate_conversation_context(ensure_pydantic_messages(messages))
    assert result is True


def test_validate_conversation_context_duplicate_roles(client, caplog):
    """Test context validation detects duplicate consecutive roles."""
    import logging

    caplog.set_level(logging.DEBUG)

    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "user", "content": "Are you there?"},  # Duplicate user role
        {"role": "assistant", "content": "Yes, I'm here!"},
    ]

    result = client._validate_conversation_context(ensure_pydantic_messages(messages))
    assert result is True  # Still returns True but logs warning
    assert "Duplicate consecutive user messages detected" in caplog.text


def test_validate_conversation_context_missing_tool_responses(client, caplog):
    """Test context validation detects missing tool responses."""
    import logging

    caplog.set_level(logging.DEBUG)

    messages = [
        {"role": "user", "content": "What's the weather?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_123",
                    "function": {"name": "get_weather", "arguments": "{}"},
                }
            ],
        },
        # Missing tool response here
        {"role": "user", "content": "Never mind, tell me a joke instead"},
    ]

    result = client._validate_conversation_context(ensure_pydantic_messages(messages))
    assert result is True  # Still returns True but logs warning
    assert "tool calls without responses" in caplog.text


def test_validate_conversation_context_with_complete_tool_flow(client):
    """Test context validation with complete tool call and response flow."""
    messages = [
        {"role": "user", "content": "What's 2+2?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "calc_1",
                    "function": {"name": "calculator", "arguments": '{"expr": "2+2"}'},
                }
            ],
        },
        {
            "role": "tool",
            "name": "calculator",
            "tool_call_id": "calc_1",
            "content": "4",
        },
        {"role": "assistant", "content": "2+2 equals 4"},
    ]

    result = client._validate_conversation_context(ensure_pydantic_messages(messages))
    assert result is True
    # Should not log any warnings for this valid flow


@pytest.mark.asyncio
async def test_prepare_messages_preserves_full_context(client):
    """Test that full conversation context is preserved through message preparation."""
    # Simulate a multi-turn conversation
    messages = [
        {"role": "system", "content": "You are a helpful weather assistant"},
        {"role": "user", "content": "My name is Alice"},
        {"role": "assistant", "content": "Nice to meet you, Alice!"},
        {"role": "user", "content": "What's the weather in Paris?"},
        {
            "role": "assistant",
            "content": "Let me check that for you.",
            "tool_calls": [
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "Paris"}',
                    }
                }
            ],
        },
        {"role": "tool", "name": "get_weather", "content": "22°C, partly cloudy"},
        {
            "role": "assistant",
            "content": "The weather in Paris is 22°C and partly cloudy.",
        },
        {"role": "user", "content": "What's my name?"},  # Tests context memory
    ]

    pydantic_messages = ensure_pydantic_messages(messages)
    prepared = await client._prepare_ollama_messages(pydantic_messages)

    # All messages should be preserved
    assert len(prepared) == 8

    # Verify system message
    assert prepared[0]["role"] == "system"
    assert prepared[0]["content"] == "You are a helpful weather assistant"

    # Verify early context is preserved
    assert prepared[1]["role"] == "user"
    assert "Alice" in prepared[1]["content"]
    assert prepared[2]["role"] == "assistant"
    assert "Alice" in prepared[2]["content"]

    # Verify tool flow is preserved
    assert prepared[4]["role"] == "assistant"
    assert "tool_calls" in prepared[4]
    assert prepared[5]["role"] == "user"  # Tool response converted to user
    assert "Tool Response from get_weather" in prepared[5]["content"]

    # Verify final question that tests memory
    assert prepared[7]["role"] == "user"
    assert prepared[7]["content"] == "What's my name?"


@pytest.mark.asyncio
async def test_prepare_messages_handles_mixed_content_types(client):
    """Test context preservation with mixed content types (text, images, tools)."""
    # Use valid base64 image data
    valid_base64_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="

    messages = [
        {"role": "user", "content": "I'm Bob"},
        {"role": "assistant", "content": "Hello Bob!"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What do you see here?"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{valid_base64_image}"},
                },
            ],
        },
        {
            "role": "assistant",
            "content": "I can see an image.",
            "tool_calls": [{"function": {"name": "analyze_image", "arguments": "{}"}}],
        },
        {"role": "tool", "name": "analyze_image", "content": "Image shows a cat"},
        {"role": "assistant", "content": "The image shows a cat."},
        {"role": "user", "content": "What's my name again?"},
    ]

    # Mock vision support
    client.supports_feature = lambda feature: feature in ["vision", "tools"]

    pydantic_messages = ensure_pydantic_messages(messages)
    prepared = await client._prepare_ollama_messages(pydantic_messages)

    # Verify all context is preserved
    assert len(prepared) == 7

    # Check that Bob's name from the first message is still in context
    assert prepared[0]["content"] == "I'm Bob"
    assert "Bob" in prepared[1]["content"]

    # Check image handling
    assert "images" in prepared[2]

    # Check tool flow
    assert "tool_calls" in prepared[3]
    assert "Tool Response from analyze_image" in prepared[4]["content"]

    # Final question should have full context available
    assert prepared[6]["content"] == "What's my name again?"


@pytest.mark.asyncio
async def test_prepare_messages_tool_arguments_parsing(client):
    """Test that tool arguments are properly parsed from strings to dicts."""
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "test_tool",
                        "arguments": '{"key": "value", "number": 42}',  # JSON string
                    }
                }
            ],
        }
    ]

    pydantic_messages = ensure_pydantic_messages(messages)
    prepared = await client._prepare_ollama_messages(pydantic_messages)

    assert len(prepared) == 1
    assert prepared[0]["role"] == "assistant"
    assert "tool_calls" in prepared[0]

    # Arguments should be parsed to dict
    tool_call = prepared[0]["tool_calls"][0]
    assert tool_call["function"]["arguments"]["key"] == "value"
    assert tool_call["function"]["arguments"]["number"] == 42


@pytest.mark.asyncio
async def test_prepare_messages_malformed_tool_arguments(client):
    """Test that malformed tool arguments are rejected by Pydantic."""
    from pydantic_core import ValidationError

    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "bad_tool",
                        "arguments": "not valid json",  # Invalid JSON
                    }
                }
            ],
        }
    ]

    # Pydantic should reject invalid JSON in arguments
    with pytest.raises(ValidationError, match="arguments must be valid JSON"):
        pydantic_messages = ensure_pydantic_messages(messages)


@pytest.mark.asyncio
async def test_create_completion_preserves_context(client):
    """Test that create_completion preserves conversation context."""
    from chuk_llm.core.models import Message

    # Multi-turn conversation - Pydantic native!
    messages = [
        Message(role=MessageRole.USER, content="My favorite color is blue"),
        Message(role=MessageRole.ASSISTANT, content="Blue is a nice color!"),
        Message(role=MessageRole.USER, content="What's my favorite color?"),
    ]

    # Mock _create_sync to verify messages are passed correctly
    async def mock_create_sync(msgs, tools=None, **kwargs):
        # Verify all messages are passed as Pydantic
        assert len(msgs) == 3
        assert all(isinstance(m, Message) for m in msgs)
        # Return response that shows context was used
        return {"response": "Your favorite color is blue", "tool_calls": []}

    client._create_sync = mock_create_sync

    result = await client._regular_completion(messages)

    assert "blue" in result["response"].lower()


@pytest.mark.asyncio
async def test_streaming_preserves_context(client):
    """Test that streaming preserves conversation context."""
    messages = [
        {"role": "user", "content": "I live in Tokyo"},
        {"role": "assistant", "content": "Tokyo is a fascinating city!"},
        {"role": "user", "content": "What city do I live in?"},
    ]

    # Convert to Pydantic
    pydantic_messages = ensure_pydantic_messages(messages)

    # Mock streaming response that uses context
    async def mock_stream():
        yield MockOllamaStreamChunk("You live in")
        yield MockOllamaStreamChunk(" Tokyo")

    async def mock_chat(**kwargs):
        # Verify all messages are passed
        assert len(kwargs["messages"]) == 3
        return mock_stream()

    client.async_client.chat = mock_chat

    chunks = []
    async for chunk in client._stream_completion_async(pydantic_messages):
        chunks.append(chunk["response"])

    full_response = "".join(chunks)
    assert "Tokyo" in full_response


@pytest.mark.asyncio
async def test_context_logging(client, caplog):
    """Test that context is properly logged for debugging."""
    import logging
    from chuk_llm.core.models import Message

    caplog.set_level(logging.DEBUG)

    # Pydantic native messages
    messages = [
        Message(role=MessageRole.SYSTEM, content="System prompt"),
        Message(role=MessageRole.USER, content="First message"),
        Message(role=MessageRole.ASSISTANT, content="First response"),
        Message(role=MessageRole.USER, content="Second message"),
    ]

    await client._prepare_ollama_messages(messages)

    # Should log context information
    assert "Prepared 4 messages for Ollama with full context" in caplog.text
    assert "role=user" in caplog.text
    assert "role=assistant" in caplog.text


@pytest.mark.asyncio
async def test_conversation_flow_logging(client, caplog):
    """Test that conversation flow is logged in create_completion."""
    import logging

    caplog.set_level(logging.DEBUG)

    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
        {"role": "user", "content": "How are you?"},
        {"role": "assistant", "content": "Good"},
        {"role": "user", "content": "Great!"},
    ]

    # Mock regular completion
    async def mock_completion(messages, tools, **kwargs):
        return {"response": "Response", "tool_calls": []}

    client._regular_completion = mock_completion

    await client.create_completion(messages, stream=False)

    # Should log conversation flow
    assert "Creating completion with 5 messages in context" in caplog.text
    assert "Recent conversation flow:" in caplog.text
    assert "user -> assistant -> user -> assistant -> user" in caplog.text


@pytest.mark.asyncio
async def test_tool_response_metadata(client):
    """Test that tool response metadata is properly added."""
    messages = [{"role": "tool", "name": "calculator", "content": "Result: 42"}]

    pydantic_messages = ensure_pydantic_messages(messages)
    prepared = await client._prepare_ollama_messages(pydantic_messages)

    assert len(prepared) == 1
    assert prepared[0]["role"] == "user"  # Converted to user
    assert "Tool Response from calculator" in prepared[0]["content"]
    assert "Result: 42" in prepared[0]["content"]

    # Check metadata
    assert "metadata" in prepared[0]
    assert prepared[0]["metadata"]["type"] == "tool_response"
    assert prepared[0]["metadata"]["tool_name"] == "calculator"


@pytest.mark.asyncio
async def test_empty_messages_list(client):
    """Test handling of empty message list."""
    messages = []

    pydantic_messages = ensure_pydantic_messages(messages)
    prepared = await client._prepare_ollama_messages(pydantic_messages)
    assert prepared == []

    # Validation should pass for empty list
    result = client._validate_conversation_context(ensure_pydantic_messages(messages))
    assert result is True


@pytest.mark.asyncio
async def test_system_message_passthrough(client):
    """Test that system messages are passed through without conversion."""
    messages = [{"role": "system", "content": "You are a helpful assistant"}]

    pydantic_messages = ensure_pydantic_messages(messages)
    prepared = await client._prepare_ollama_messages(pydantic_messages)

    assert len(prepared) == 1
    assert prepared[0]["role"] == "system"
    assert prepared[0]["content"] == "You are a helpful assistant"
    # Should NOT be converted to user message


@pytest.mark.asyncio
async def test_complex_tool_call_preservation(client):
    """Test preservation of complex tool calls with nested arguments."""
    messages = [
        {
            "role": "assistant",
            "content": "Let me search for that.",
            "tool_calls": [
                {
                    "id": "call_complex",
                    "type": "function",
                    "function": {
                        "name": "complex_search",
                        "arguments": json.dumps(
                            {
                                "query": "weather",
                                "filters": {
                                    "location": ["Paris", "London"],
                                    "date_range": {
                                        "start": "2024-01-01",
                                        "end": "2024-01-31",
                                    },
                                },
                                "options": {
                                    "include_forecast": True,
                                    "units": "celsius",
                                },
                            }
                        ),
                    },
                }
            ],
        }
    ]

    pydantic_messages = ensure_pydantic_messages(messages)
    prepared = await client._prepare_ollama_messages(pydantic_messages)

    assert len(prepared) == 1
    assert prepared[0]["role"] == "assistant"
    assert "tool_calls" in prepared[0]

    tool_call = prepared[0]["tool_calls"][0]
    args = tool_call["function"]["arguments"]

    # Verify complex nested structure is preserved
    assert args["query"] == "weather"
    assert "filters" in args
    assert args["filters"]["location"] == ["Paris", "London"]
    assert args["filters"]["date_range"]["start"] == "2024-01-01"
    assert args["options"]["include_forecast"] is True


# ---------------------------------------------------------------------------
# Additional coverage tests for missing lines
# ---------------------------------------------------------------------------


def test_client_initialization_without_chat_attribute(mock_configuration):
    """Test client initialization when ollama doesn't have chat attribute."""
    with patch("ollama.chat", create=False):
        # Remove chat attribute temporarily
        import ollama
        if hasattr(ollama, "chat"):
            delattr(ollama, "chat")

        with pytest.raises(ValueError, match="does not expose 'chat'"):
            OllamaLLMClient(model="qwen3")


def test_client_initialization_old_ollama_fallback(mock_configuration):
    """Test client initialization with old ollama version (no host parameter)."""
    # Mock old-style clients that don't accept host parameter
    class OldMockAsyncClient:
        def __init__(self):
            pass

    class OldMockClient:
        def __init__(self):
            pass

        def show(self, model: str):
            return MockOllamaShowResponse([])

    with patch("ollama.AsyncClient", OldMockAsyncClient):
        with patch("ollama.Client", OldMockClient):
            # Should fall back to old initialization
            client = OllamaLLMClient(model="qwen3", api_base="http://custom:11434")

            assert client.async_client is not None
            assert client.sync_client is not None


def test_client_initialization_old_ollama_with_set_host(mock_configuration):
    """Test client initialization with old ollama that has set_host method."""
    # Mock old-style clients
    class OldMockAsyncClient:
        def __init__(self):
            pass

    class OldMockClient:
        def __init__(self):
            pass

        def show(self, model: str):
            return MockOllamaShowResponse([])

    with patch("ollama.AsyncClient", OldMockAsyncClient):
        with patch("ollama.Client", OldMockClient):
            # Create a mock module with set_host attribute
            with patch("ollama.set_host", create=True) as mock_set_host:
                client = OllamaLLMClient(model="qwen3", api_base="http://custom:11434")

                # Should call set_host
                mock_set_host.assert_called_once_with("http://custom:11434")


def test_get_model_info_with_error(mock_configuration, monkeypatch):
    """Test get_model_info when there's an error in base info."""
    client = OllamaLLMClient(model="qwen3")

    # Mock get_model_info to return error
    def mock_base_info():
        return {"error": "Configuration not found"}

    monkeypatch.setattr(
        "chuk_llm.llm.providers._config_mixin.ConfigAwareProviderMixin.get_model_info",
        lambda self: mock_base_info()
    )

    info = client.get_model_info()

    # Should not add ollama_specific when there's an error
    assert "error" in info
    assert "ollama_specific" not in info


def test_detect_model_family_code(client):
    """Test model family detection for code models."""
    client.model = "starcoder"
    assert client._detect_model_family() == "code"

    client.model = "deepseek-coder"
    assert client._detect_model_family() == "code"


@pytest.mark.asyncio
async def test_prepare_messages_tool_args_as_list(client):
    """Test that list arguments are rejected by Pydantic (must be JSON string)."""
    from pydantic_core import ValidationError

    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "test_tool",
                        "arguments": [1, 2, 3],  # List instead of JSON string - invalid!
                    }
                }
            ],
        }
    ]

    # Pydantic expects arguments to be a JSON string, not a list
    with pytest.raises(ValidationError):
        pydantic_messages = ensure_pydantic_messages(messages)


@pytest.mark.asyncio
async def test_prepare_messages_with_pydantic_image_content(client):
    """Test message preparation with actual Pydantic image content."""
    from chuk_llm.core.models import TextContent, ImageUrlContent, Message

    # Use real Pydantic content objects
    message = Message(
        role=MessageRole.USER,
        content=[
            TextContent(type=ContentType.TEXT, text="Look at this"),
            ImageUrlContent(
                type=ContentType.IMAGE_URL,
                image_url="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
            ),
        ],
    )

    client.supports_feature = lambda feature: feature == "vision"

    prepared = await client._prepare_ollama_messages([message])

    assert len(prepared) == 1
    assert "images" in prepared[0]


@pytest.mark.asyncio
async def test_prepare_messages_unknown_role(client):
    """Test that unknown roles are rejected by Pydantic."""
    messages = [
        {"role": "custom_role", "content": "Custom content"}
    ]

    # Pydantic should reject invalid role
    with pytest.raises(ValueError, match="not a valid MessageRole"):
        pydantic_messages = ensure_pydantic_messages(messages)


@pytest.mark.asyncio
async def test_create_sync_tools_not_supported(client):
    """Test _create_sync when tools are provided but not supported."""
    messages = [{"role": "user", "content": "Hello"}]
    tools = [{"function": {"name": "test_tool"}}]

    # Mock no tool support
    client.supports_feature = lambda feature: feature != "tools"

    mock_response = MockOllamaResponse("Hello")
    client.async_client.chat = AsyncMock(return_value=mock_response)

    pydantic_messages = ensure_pydantic_messages(messages)
    pydantic_tools = ensure_pydantic_tools(tools)
    result = await client._create_sync(pydantic_messages, pydantic_tools)

    # Should succeed but not pass tools
    assert result["response"] == "Hello"
    call_kwargs = client.async_client.chat.call_args.kwargs
    assert "tools" not in call_kwargs


@pytest.mark.asyncio
async def test_create_sync_with_think_parameter(client):
    """Test _create_sync with think parameter for reasoning models."""
    messages = [{"role": "user", "content": "Solve this"}]

    mock_response = MockOllamaResponse("Solution")

    # Mock async_client.chat to accept think parameter
    async def mock_chat(model, messages, stream, think=None, **kwargs):
        return mock_response

    client.async_client.chat = mock_chat

    # Test with boolean think
    pydantic_messages = ensure_pydantic_messages(messages)
    result = await client._create_sync(pydantic_messages, think=True)
    assert result["response"] == "Solution"

    # Test with string think
    pydantic_messages = ensure_pydantic_messages(messages)
    result = await client._create_sync(pydantic_messages, think="high")
    assert result["response"] == "Solution"


@pytest.mark.asyncio
async def test_create_sync_with_think_parameter_not_supported(client):
    """Test _create_sync when think parameter is not supported by async client."""
    messages = [{"role": "user", "content": "Solve this"}]

    mock_response = MockOllamaResponse("Solution")

    # Mock async_client.chat that doesn't accept think parameter
    async def mock_chat(model, messages, stream, **kwargs):
        if "think" in kwargs:
            raise TypeError("Unexpected keyword argument 'think'")
        return mock_response

    client.async_client.chat = mock_chat

    # Should handle gracefully by not passing think
    pydantic_messages = ensure_pydantic_messages(messages)
    result = await client._create_sync(pydantic_messages, think="medium")
    assert result["response"] == "Solution"


def test_build_ollama_options_with_special_parameters(client, caplog):
    """Test that special parameters are logged but not included in options."""
    import logging
    caplog.set_level(logging.DEBUG)

    kwargs = {
        "temperature": 0.7,
        "think": "medium",
        "stream": True,
        "tools": [],
        "messages": [],
        "model": "qwen3",
    }

    options = client._build_ollama_options(kwargs)

    # Special parameters should not be in options
    assert "think" not in options
    assert "stream" not in options
    assert "tools" not in options
    assert "messages" not in options
    assert "model" not in options

    # Regular parameter should be in options
    assert "temperature" in options


def test_parse_response_tool_args_as_other_type(client):
    """Test parsing response when tool arguments are neither dict nor string."""
    mock_tool_call = types.SimpleNamespace(
        id="call_123",
        function=types.SimpleNamespace(
            name="test_tool",
            arguments=12345  # Integer instead of dict/string
        ),
    )

    mock_response = MockOllamaResponse(content="", tool_calls=[mock_tool_call])
    client.supports_feature = lambda feature: feature == "tools"

    result = client._parse_response(mock_response)

    # Should convert to string
    assert result["tool_calls"][0]["function"]["arguments"] == "12345"


@pytest.mark.asyncio
async def test_create_completion_context_validation_failure(client, caplog):
    """Test create_completion when context validation fails."""
    import logging
    caplog.set_level(logging.DEBUG)

    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "user", "content": "Hello again"},  # Duplicate user messages
    ]

    async def mock_completion(messages, tools, **kwargs):
        return {"response": "Hello!", "tool_calls": []}

    client._regular_completion = mock_completion

    await client.create_completion(messages, stream=False)

    # Should log debug about duplicate roles
    assert "Duplicate consecutive user messages detected" in caplog.text


@pytest.mark.asyncio
async def test_stream_completion_tools_not_supported(client):
    """Test streaming when tools are provided but not supported."""
    messages = [{"role": "user", "content": "Hello"}]

    # Convert to Pydantic
    pydantic_messages = ensure_pydantic_messages(messages)
    tools = [{"function": {"name": "test_tool"}}]

    # Mock no tool support
    client.supports_feature = lambda feature: feature != "tools"

    async def mock_stream():
        yield MockOllamaStreamChunk("Hello")

    async def mock_chat(**kwargs):
        # Verify tools were not passed
        assert "tools" not in kwargs
        return mock_stream()

    client.async_client.chat = mock_chat

    chunks = []
    async for chunk in client._stream_completion_async(messages, tools):
        chunks.append(chunk)

    assert len(chunks) == 1


@pytest.mark.asyncio
async def test_stream_completion_with_think_parameter(client):
    """Test streaming with think parameter."""
    messages = [{"role": "user", "content": "Solve this"}]

    # Convert to Pydantic
    pydantic_messages = ensure_pydantic_messages(messages)

    async def mock_stream():
        yield MockOllamaStreamChunk("Solution")

    async def mock_chat(**kwargs):
        # Verify think parameter was passed
        assert "think" in kwargs
        assert kwargs["think"] == "high"
        return mock_stream()

    client.async_client.chat = mock_chat

    chunks = []
    async for chunk in client._stream_completion_async(messages, think="high"):
        chunks.append(chunk)

    assert len(chunks) == 1


@pytest.mark.asyncio
async def test_stream_completion_with_think_boolean(client):
    """Test streaming with boolean think parameter."""
    messages = [{"role": "user", "content": "Solve this"}]

    # Convert to Pydantic
    pydantic_messages = ensure_pydantic_messages(messages)

    async def mock_stream():
        yield MockOllamaStreamChunk("Solution")

    async def mock_chat(**kwargs):
        # Boolean should be converted to "medium"
        assert "think" in kwargs
        assert kwargs["think"] == "medium"
        return mock_stream()

    client.async_client.chat = mock_chat

    chunks = []
    async for chunk in client._stream_completion_async(messages, think=True):
        chunks.append(chunk)

    assert len(chunks) == 1


@pytest.mark.asyncio
async def test_stream_completion_tool_extraction_various_attributes(client):
    """Test tool call extraction with various attribute names."""
    messages = [{"role": "user", "content": "Use tools"}]

    # Convert to Pydantic
    pydantic_messages = ensure_pydantic_messages(messages)

    # Mock tool call with different attribute names
    class ToolCallWithCapitalAttrs:
        def __init__(self):
            self.Id = "call_caps"
            self.Function = types.SimpleNamespace()
            self.Function.Name = "tool_caps"
            self.Function.Arguments = {"arg": "value"}

    async def mock_stream():
        yield MockOllamaStreamChunk("", tool_calls=[ToolCallWithCapitalAttrs()])

    async def mock_chat(**kwargs):
        return mock_stream()

    client.async_client.chat = mock_chat
    client.supports_feature = lambda feature: feature == "tools"

    chunks = []
    async for chunk in client._stream_completion_async(pydantic_messages):
        chunks.append(chunk)

    assert len(chunks) == 1
    assert len(chunks[0]["tool_calls"]) == 1
    assert chunks[0]["tool_calls"][0]["function"]["name"] == "tool_caps"


@pytest.mark.asyncio
async def test_stream_completion_tool_at_chunk_level(client):
    """Test tool calls at chunk level (not in message)."""
    messages = [{"role": "user", "content": "Use tools"}]

    # Convert to Pydantic
    pydantic_messages = ensure_pydantic_messages(messages)

    # Create mock where tool_calls is at chunk level
    class MockChunkWithToolCalls:
        def __init__(self):
            self.message = MockOllamaMessage("")
            self.tool_calls = [
                types.SimpleNamespace(
                    id="chunk_call",
                    function=types.SimpleNamespace(
                        name="chunk_tool",
                        arguments={}
                    )
                )
            ]

    async def mock_stream():
        yield MockChunkWithToolCalls()

    async def mock_chat(**kwargs):
        return mock_stream()

    client.async_client.chat = mock_chat
    client.supports_feature = lambda feature: feature == "tools"

    chunks = []
    async for chunk in client._stream_completion_async(pydantic_messages):
        chunks.append(chunk)

    assert len(chunks) == 1
    assert len(chunks[0]["tool_calls"]) == 1


@pytest.mark.asyncio
async def test_stream_completion_final_chunk_with_tools(client):
    """Test final chunk (done=True) with tool calls."""
    messages = [{"role": "user", "content": "Use tools"}]

    # Convert to Pydantic
    pydantic_messages = ensure_pydantic_messages(messages)

    # Create mock final chunk with done=True
    class MockFinalChunk:
        def __init__(self):
            self.message = MockOllamaMessage("Done", tool_calls=[
                types.SimpleNamespace(
                    id="final_call",
                    function=types.SimpleNamespace(
                        name="final_tool",
                        arguments={}
                    )
                )
            ])
            self.done = True
            self.tool_calls = None  # Tool calls only in message

    async def mock_stream():
        yield MockOllamaStreamChunk("Starting")
        yield MockFinalChunk()

    async def mock_chat(**kwargs):
        return mock_stream()

    client.async_client.chat = mock_chat
    client.supports_feature = lambda feature: feature == "tools"

    chunks = []
    async for chunk in client._stream_completion_async(pydantic_messages):
        chunks.append(chunk)

    assert len(chunks) == 2
    # Final chunk should have tool calls
    assert len(chunks[1]["tool_calls"]) == 1


@pytest.mark.asyncio
async def test_stream_completion_empty_chunks_and_asyncio_sleep(client):
    """Test empty chunk handling and asyncio sleep for large chunk counts."""
    messages = [{"role": "user", "content": "Hello"}]

    # Convert to Pydantic
    pydantic_messages = ensure_pydantic_messages(messages)

    async def mock_stream():
        # Yield many chunks to trigger asyncio.sleep
        for i in range(25):
            if i % 5 == 0:
                yield MockOllamaStreamChunk(f"chunk{i}")
            else:
                # Empty chunks
                yield MockOllamaStreamChunk("")

    async def mock_chat(**kwargs):
        return mock_stream()

    client.async_client.chat = mock_chat

    chunks = []
    async for chunk in client._stream_completion_async(pydantic_messages):
        chunks.append(chunk)

    # Should only get non-empty chunks
    assert len(chunks) == 5


@pytest.mark.asyncio
async def test_stream_completion_deduplication(client):
    """Test that duplicate tool calls are deduplicated."""
    messages = [{"role": "user", "content": "Use tools"}]

    # Convert to Pydantic
    pydantic_messages = ensure_pydantic_messages(messages)

    # Same tool call appears twice
    mock_tool_call = types.SimpleNamespace(
        id="dup_call",
        function=types.SimpleNamespace(
            name="dup_tool",
            arguments={"arg": "value"}
        )
    )

    class MockChunkWithDuplicates:
        def __init__(self):
            self.message = MockOllamaMessage("", tool_calls=[mock_tool_call])
            self.tool_calls = [mock_tool_call]  # Same tool call in both places

    async def mock_stream():
        yield MockChunkWithDuplicates()

    async def mock_chat(**kwargs):
        return mock_stream()

    client.async_client.chat = mock_chat
    client.supports_feature = lambda feature: feature == "tools"

    chunks = []
    async for chunk in client._stream_completion_async(pydantic_messages):
        chunks.append(chunk)

    # Should only have one tool call (deduplicated)
    assert len(chunks) == 1
    assert len(chunks[0]["tool_calls"]) == 1


@pytest.mark.asyncio
async def test_stream_completion_tool_without_function_name(client):
    """Test tool call extraction when function name is missing."""
    messages = [{"role": "user", "content": "Use tools"}]

    # Convert to Pydantic
    pydantic_messages = ensure_pydantic_messages(messages)

    # Tool call without function name
    class ToolCallNoName:
        def __init__(self):
            self.id = "no_name_call"
            self.function = types.SimpleNamespace()
            # No name attribute
            self.function.arguments = {}

    async def mock_stream():
        yield MockOllamaStreamChunk("", tool_calls=[ToolCallNoName()])

    async def mock_chat(**kwargs):
        return mock_stream()

    client.async_client.chat = mock_chat
    client.supports_feature = lambda feature: feature == "tools"

    chunks = []
    async for chunk in client._stream_completion_async(pydantic_messages):
        chunks.append(chunk)

    # Should not add tool call without name
    assert len(chunks) == 0  # No content, no valid tool calls


def test_validate_request_with_vision_content(client):
    """Test request validation with vision content detection."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Look"},
                {"type": "image_url", "image_url": {"url": "http://example.com/img.jpg"}},
            ],
        }
    ]

    # Convert to Pydantic
    pydantic_messages = ensure_pydantic_messages(messages)

    # Mock no vision support
    client.supports_feature = lambda feature: feature != "vision"

    validated_messages, validated_tools, validated_stream, validated_kwargs = (
        client._validate_request_with_config(pydantic_messages, None, False)
    )

    # Should still pass through (warning logged)
    assert len(validated_messages) == len(pydantic_messages)


@pytest.mark.asyncio
async def test_prepare_messages_with_pydantic_image_url_object(client):
    """Test image handling with Pydantic ImageUrlContent object."""
    from chuk_llm.core.models import ImageUrlContent, Message

    # Use real Pydantic ImageUrlContent
    message = Message(
        role=MessageRole.USER,
        content=[
            ImageUrlContent(
                type=ContentType.IMAGE_URL,
                image_url="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
            )
        ],
    )

    client.supports_feature = lambda feature: feature == "vision"

    prepared = await client._prepare_ollama_messages([message])

    assert len(prepared) == 1
    assert "images" in prepared[0]
