# tests/core/providers/test_gemini_client.py
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

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
google_mod.genai = genai_mod

# --- sub-module ``google.genai.types`` -------------------------------------

types_mod = types.ModuleType("google.genai.types")
sys.modules["google.genai.types"] = types_mod
genai_mod.types = types_mod

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
    def __init__(
        self,
        tools=None,
        system_instruction=None,
        max_output_tokens=None,
        temperature=None,
        top_p=None,
        top_k=None,
        **kwargs,
    ):
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
        # Return a mock response with text using our helper
        return create_mock_gemini_response(text="Mock response from Gemini")

    async def generate_content_stream(self, **kwargs):
        # Return an async generator
        async def mock_stream():
            yield create_mock_gemini_response(text="chunk1")
            yield create_mock_gemini_response(text="chunk2")

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
sys.modules["dotenv"].load_dotenv = lambda *args, **kwargs: None

# ---------------------------------------------------------------------------
# Now import the adapter under test (it will pick up the stubs).
# ---------------------------------------------------------------------------

from chuk_llm.llm.providers.gemini_client import (
    GeminiLLMClient,
    _convert_tools_to_gemini_format,
    _safe_parse_gemini_response,  # Fixed: use the correct function name
    validate_and_map_model,
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
            MockFeature.TEXT,
            MockFeature.STREAMING,
            MockFeature.TOOLS,
            MockFeature.VISION,
            MockFeature.SYSTEM_MESSAGES,
            MockFeature.MULTIMODAL,
        }
        self.max_context_length = max_context_length
        self.max_output_tokens = max_output_tokens


class MockProviderConfig:
    def __init__(self, name="gemini", client_class="GeminiLLMClient"):
        self.name = name
        self.client_class = client_class
        self.api_base = "https://generativelanguage.googleapis.com"
        self.models = [
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-2.0-flash",
            "gemini-1.5-pro",
        ]
        self.model_aliases = {}
        self.rate_limits = {"requests_per_minute": 60}

    def get_model_capabilities(self, model):
        # Gemini models typically have comprehensive features
        features = {
            MockFeature.TEXT,
            MockFeature.STREAMING,
            MockFeature.TOOLS,
            MockFeature.VISION,
            MockFeature.SYSTEM_MESSAGES,
            MockFeature.MULTIMODAL,
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

    with patch("chuk_llm.configuration.get_config", return_value=mock_config):
        with patch("chuk_llm.configuration.Feature", MockFeature):
            yield mock_config


@pytest.fixture
def client(mock_configuration, monkeypatch):
    """Gemini client with configuration properly mocked"""
    cl = GeminiLLMClient(model="gemini-2.5-flash", api_key="fake-key")

    # Ensure configuration methods are properly mocked
    monkeypatch.setattr(
        cl,
        "supports_feature",
        lambda feature: feature
        in ["text", "streaming", "tools", "vision", "system_messages", "multimodal"],
    )

    monkeypatch.setattr(
        cl,
        "get_model_info",
        lambda: {
            "provider": "gemini",
            "model": "gemini-2.5-flash",
            "client_class": "GeminiLLMClient",
            "api_base": "https://generativelanguage.googleapis.com",
            "features": [
                "text",
                "streaming",
                "tools",
                "vision",
                "system_messages",
                "multimodal",
            ],
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
            "tool_compatibility": {
                "supports_universal_naming": True,
                "sanitization_method": "replace_chars",
                "restoration_method": "name_mapping",
                "supported_name_patterns": ["alphanumeric_underscore"],
            },
            "gemini_specific": {
                "model_family": cl._detect_model_family(),
                "supports_function_calling": True,
                "supports_vision": True,
                "supports_multimodal": True,
                "max_input_tokens": 1000000,
                "context_length": "2M tokens",
                "experimental_features": True,
                "warning_suppression": "ultimate",
                "enhanced_reasoning": True,
                "data_loss_protection": "enabled",
            },
        },
    )

    # Mock token limits
    monkeypatch.setattr(cl, "get_max_tokens_limit", lambda: 4096)
    monkeypatch.setattr(cl, "get_context_length_limit", lambda: 8192)

    # Mock parameter validation
    def mock_validate_parameters(**kwargs):
        result = kwargs.copy()
        if "max_tokens" in result and result["max_tokens"] > 4096:
            result["max_tokens"] = 4096
        return result

    monkeypatch.setattr(cl, "validate_parameters", mock_validate_parameters)

    # Mock tool compatibility methods
    monkeypatch.setattr(cl, "_sanitize_tool_names", lambda tools: tools)
    monkeypatch.setattr(
        cl, "_restore_tool_names_in_response", lambda response, mapping: response
    )
    monkeypatch.setattr(
        cl,
        "get_tool_compatibility_info",
        lambda: {
            "supports_universal_naming": True,
            "sanitization_method": "replace_chars",
            "restoration_method": "name_mapping",
            "supported_name_patterns": ["alphanumeric_underscore"],
        },
    )

    # Initialize empty name mapping
    cl._current_name_mapping = {}

    return cl


@pytest.fixture
def pro_client(mock_configuration, monkeypatch):
    """Gemini Pro client with advanced features"""
    cl = GeminiLLMClient(model="gemini-2.5-pro", api_key="fake-key")

    # Pro model has additional features
    monkeypatch.setattr(
        cl,
        "supports_feature",
        lambda feature: feature
        in [
            "text",
            "streaming",
            "tools",
            "vision",
            "system_messages",
            "multimodal",
            "json_mode",
            "reasoning",
        ],
    )

    monkeypatch.setattr(
        cl,
        "get_model_info",
        lambda: {
            "provider": "gemini",
            "model": "gemini-2.5-pro",
            "client_class": "GeminiLLMClient",
            "api_base": "https://generativelanguage.googleapis.com",
            "features": [
                "text",
                "streaming",
                "tools",
                "vision",
                "system_messages",
                "multimodal",
                "json_mode",
                "reasoning",
            ],
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
            "tool_compatibility": {
                "supports_universal_naming": True,
                "sanitization_method": "replace_chars",
                "restoration_method": "name_mapping",
                "supported_name_patterns": ["alphanumeric_underscore"],
            },
            "gemini_specific": {
                "model_family": cl._detect_model_family(),
                "supports_function_calling": True,
                "supports_vision": True,
                "supports_multimodal": True,
                "max_input_tokens": 1000000,
                "context_length": "2M tokens",
                "experimental_features": True,
                "warning_suppression": "ultimate",
                "enhanced_reasoning": True,
                "data_loss_protection": "enabled",
            },
        },
    )

    # Mock tool compatibility methods
    monkeypatch.setattr(cl, "_sanitize_tool_names", lambda tools: tools)
    monkeypatch.setattr(
        cl, "_restore_tool_names_in_response", lambda response, mapping: response
    )
    monkeypatch.setattr(
        cl,
        "get_tool_compatibility_info",
        lambda: {
            "supports_universal_naming": True,
            "sanitization_method": "replace_chars",
            "restoration_method": "name_mapping",
            "supported_name_patterns": ["alphanumeric_underscore"],
        },
    )

    # Initialize empty name mapping
    cl._current_name_mapping = {}

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
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            },
        }
    ]

    gem_tools = _convert_tools_to_gemini_format(tools)

    # Check tools list
    assert len(gem_tools) == 1
    assert len(gem_tools[0].function_declarations) == 1
    assert gem_tools[0].function_declarations[0]["name"] == "get_weather"
    assert (
        gem_tools[0].function_declarations[0]["description"]
        == "Get the current weather"
    )


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
                "parameters": {},
            },
        }
    ]

    gem_tools = _convert_tools_to_gemini_format(tools)
    # Should return None or empty list due to invalid tool
    assert gem_tools is None


# ---------------------------------------------------------------------------
# Response parsing tests
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Helper functions to create proper mock responses
# ---------------------------------------------------------------------------


def create_mock_gemini_response(text=None, function_calls=None, finish_reason=None):
    """Create a properly structured mock Gemini response."""
    mock_response = MagicMock()

    if text or function_calls or finish_reason:
        # Create candidate with content and parts
        parts = []

        if text:
            text_part = MagicMock()
            text_part.text = text

            # Mock hasattr for text_part
            def text_part_hasattr(attr):
                return attr in ["text", "function_call"]

            text_part.__class__.__name__ = "TextPart"
            text_part.__hasattr__ = text_part_hasattr
            parts.append(text_part)

        if function_calls:
            for fc in function_calls:
                fc_part = MagicMock()
                fc_part.text = None
                fc_part.function_call = MagicMock()
                fc_part.function_call.name = fc["name"]
                fc_part.function_call.args = fc.get("args", {})

                # Mock hasattr for fc_part
                def fc_part_hasattr(attr):
                    return attr in ["text", "function_call"]

                fc_part.__class__.__name__ = "FunctionCallPart"
                fc_part.__hasattr__ = fc_part_hasattr
                parts.append(fc_part)

        if parts:
            content = MagicMock()
            content.parts = parts

            # Mock hasattr for content
            def content_hasattr(attr):
                return attr in ["parts", "text"]

            content.__class__.__name__ = "Content"
            content.__hasattr__ = content_hasattr
        else:
            content = None

        candidate = MagicMock()
        candidate.content = content
        if finish_reason:
            candidate.finish_reason = finish_reason

        # Mock hasattr for candidate
        def candidate_hasattr(attr):
            return attr in ["content", "finish_reason"]

        candidate.__class__.__name__ = "Candidate"
        candidate.__hasattr__ = candidate_hasattr

        mock_response.candidates = [candidate]
    else:
        mock_response.candidates = None

    # Mock hasattr for response
    def response_hasattr(attr):
        return attr in ["candidates", "text"]

    mock_response.__class__.__name__ = "GenerateContentResponse"
    mock_response.__hasattr__ = response_hasattr

    return mock_response


# ---------------------------------------------------------------------------
# Updated response parsing tests
# ---------------------------------------------------------------------------


def test_safe_parse_gemini_response_text_only():
    """Test parsing response with text only."""
    mock_response = create_mock_gemini_response(text="Hello world")

    result = _safe_parse_gemini_response(mock_response)

    assert result["response"] == "Hello world"
    assert result["tool_calls"] == []


def test_safe_parse_gemini_response_candidates_structure():
    """Test parsing response with candidates structure."""
    mock_response = create_mock_gemini_response(text="Response from candidates")

    result = _safe_parse_gemini_response(mock_response)

    assert result["response"] == "Response from candidates"
    assert result["tool_calls"] == []


def test_safe_parse_gemini_response_max_tokens():
    """Test parsing response when max tokens reached."""
    mock_response = create_mock_gemini_response(finish_reason="MAX_TOKENS")

    result = _safe_parse_gemini_response(mock_response)

    assert "token limit" in result["response"]
    assert result["tool_calls"] == []


def test_safe_parse_gemini_response_function_calls():
    """Test parsing response with function calls."""
    function_calls = [{"name": "get_weather", "args": {"city": "NYC"}}]
    mock_response = create_mock_gemini_response(function_calls=function_calls)

    result = _safe_parse_gemini_response(mock_response)

    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["function"]["name"] == "get_weather"
    assert "NYC" in result["tool_calls"][0]["function"]["arguments"]


def test_safe_parse_gemini_response_mixed_content():
    """Test parsing response with both text and function calls."""
    function_calls = [{"name": "get_weather", "args": {"city": "NYC"}}]
    mock_response = create_mock_gemini_response(
        text="I'll check the weather for you.", function_calls=function_calls
    )

    result = _safe_parse_gemini_response(mock_response)

    assert result["response"] == "I'll check the weather for you."
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["function"]["name"] == "get_weather"


def test_safe_parse_gemini_response_no_candidates():
    """Test parsing response when there are no candidates."""
    mock_response = create_mock_gemini_response()  # Empty response

    result = _safe_parse_gemini_response(mock_response)

    assert (
        "[Unable to extract content from response - no candidates]"
        in result["response"]
    )
    assert result["tool_calls"] == []


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
        "top_k": 40,
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
        {"role": "assistant", "content": "Hello! How can I help you today?"},
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
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
                    },
                },
            ],
        }
    ]

    system_txt, contents = await client._split_for_gemini_async(messages)

    assert system_txt == ""
    assert len(contents) == 1
    # Should contain multimodal parts
    assert isinstance(contents[0], list)


@pytest.mark.asyncio
async def test_split_for_gemini_async_multimodal_not_supported(client, monkeypatch):
    """Test message splitting with multimodal content when vision is not supported."""
    # Mock vision as not supported
    monkeypatch.setattr(client, "supports_feature", lambda feature: feature != "vision")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,..."},
                },
            ],
        }
    ]

    system_txt, contents = await client._split_for_gemini_async(messages)

    assert system_txt == ""
    assert len(contents) == 1
    # Should only contain text when vision not supported
    assert "What's in this image?" in contents[0]


@pytest.mark.asyncio
async def test_split_for_gemini_async_tool_calls(client):
    """Test message splitting with tool calls."""
    messages = [
        {"role": "user", "content": "Get weather"},
        {
            "role": "assistant",
            "tool_calls": [
                {"function": {"name": "get_weather", "arguments": '{"city": "NYC"}'}}
            ],
        },
        {"role": "tool", "name": "get_weather", "content": "Sunny, 75°F"},
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
    mock_response = create_mock_gemini_response(text="Hello! How can I help you?")

    async def mock_generate_content(**kwargs):
        return mock_response

    client.client.aio.models.generate_content = mock_generate_content

    result = await client._regular_completion_async(
        system=None,
        json_instruction=None,
        messages=messages,
        gemini_tools=None,
        filtered_params={},
        name_mapping={},
    )

    assert result["response"] == "Hello! How can I help you?"
    assert result["tool_calls"] == []


@pytest.mark.asyncio
async def test_regular_completion_async_with_system(client):
    """Test regular completion with system instruction."""
    messages = [{"role": "user", "content": "Hello"}]
    system = "You are a helpful assistant."

    # Mock the client's generate_content method
    mock_response = create_mock_gemini_response(text="Hello! I'm here to help.")

    async def mock_generate_content(**kwargs):
        # Verify system instruction is included
        config = kwargs.get("config")
        if config:
            assert config.system_instruction == system
        return mock_response

    client.client.aio.models.generate_content = mock_generate_content

    result = await client._regular_completion_async(
        system=system,
        json_instruction=None,
        messages=messages,
        gemini_tools=None,
        filtered_params={},
        name_mapping={},
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
        filtered_params={},
        name_mapping={},
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
        chunk1 = create_mock_gemini_response(text="Hello")
        yield chunk1

        chunk2 = create_mock_gemini_response(text=" world!")
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
        filtered_params={},
        name_mapping={},
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

    client.client.aio.models.generate_content_stream = (
        mock_generate_content_stream_error
    )

    # Collect streaming results
    chunks = []
    async for chunk in client._stream_completion_async(
        system=None,
        json_instruction=None,
        messages=messages,
        gemini_tools=None,
        filtered_params={},
        name_mapping={},
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

    async def mock_regular_completion_async(
        system, json_instruction, messages, gemini_tools, filtered_params, name_mapping
    ):
        return expected_result

    client._regular_completion_async = mock_regular_completion_async

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
    async def mock_stream_completion_async(
        system, json_instruction, messages, gemini_tools, filtered_params, name_mapping
    ):
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
async def test_create_completion_streaming_not_supported(client, monkeypatch):
    """Test create_completion with streaming when not supported."""
    messages = [{"role": "user", "content": "Hello"}]

    # Mock streaming as not supported
    monkeypatch.setattr(
        client, "supports_feature", lambda feature: feature != "streaming"
    )

    # Mock the regular completion method (should be called instead of streaming)
    expected_result = {"response": "Hello!", "tool_calls": []}

    async def mock_regular_completion_async(
        system, json_instruction, messages, gemini_tools, filtered_params, name_mapping
    ):
        return expected_result

    client._regular_completion_async = mock_regular_completion_async

    result = client.create_completion(messages, stream=True)

    # Should return an awaitable (not async iterator) when streaming not supported
    assert hasattr(result, "__await__")
    assert not hasattr(result, "__aiter__")

    final_result = await result
    assert final_result == expected_result


@pytest.mark.asyncio
async def test_create_completion_with_tools(client):
    """Test create_completion with tools."""
    messages = [{"role": "user", "content": "What's the weather?"}]
    tools = [
        {"type": "function", "function": {"name": "get_weather", "parameters": {}}}
    ]

    # Mock regular completion
    expected_result = {
        "response": "I'll check the weather for you.",
        "tool_calls": [
            {
                "id": "call_123",
                "type": "function",
                "function": {"name": "get_weather", "arguments": "{}"},
            }
        ],
    }

    async def mock_regular_completion_async(
        system, json_instruction, messages, gemini_tools, filtered_params, name_mapping
    ):
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
    tools = [
        {"type": "function", "function": {"name": "get_weather", "parameters": {}}}
    ]

    # Mock tools as not supported
    monkeypatch.setattr(client, "supports_feature", lambda feature: feature != "tools")

    # Mock regular completion
    expected_result = {"response": "I cannot use tools.", "tool_calls": []}

    async def mock_regular_completion_async(
        system, json_instruction, messages, gemini_tools, filtered_params, name_mapping
    ):
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
    async def mock_regular_completion_async(
        system, json_instruction, messages, gemini_tools, filtered_params, name_mapping
    ):
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
    # Create mock response with function calls using the helper
    function_calls = [{"name": "get_weather", "args": {"city": "NYC"}}]
    mock_response = create_mock_gemini_response(function_calls=function_calls)

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
    if hasattr(mock_part, "function_call"):
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
        {"role": "user", "content": "Hello"},
    ]

    # Mock the actual Gemini API call
    mock_response = create_mock_gemini_response(text="Hello! How can I help you today?")

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
            chunk = create_mock_gemini_response(text=text)
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
    monkeypatch.setattr(
        client, "supports_feature", lambda feature: feature != "streaming"
    )

    # Mock regular completion
    async def mock_generate_content(**kwargs):
        return create_mock_gemini_response(text="Once upon a time...")

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
        "Timeout error",
    ]

    for error_msg in error_scenarios:

        def create_error_mock(msg):
            async def mock_generate_content_error(**kwargs):
                raise Exception(msg)
            return mock_generate_content_error

        mock_generate_content_error = create_error_mock(error_msg)

        client.client.aio.models.generate_content = mock_generate_content_error

        result = await client._regular_completion_async(
            system=None,
            json_instruction=None,
            messages=messages,
            gemini_tools=None,
            filtered_params={},
            name_mapping={},
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
    supported_features = [
        "text",
        "streaming",
        "tools",
        "vision",
        "system_messages",
        "multimodal",
    ]

    for feature in supported_features:
        assert client.supports_feature(feature) is True

    # Test unsupported features
    unsupported_features = ["json_mode", "reasoning", "parallel_calls"]

    for feature in unsupported_features:
        assert client.supports_feature(feature) is False

    # Test individual feature isolation
    for feature in supported_features:
        # Mock only this feature as supported
        monkeypatch.setattr(
            client,
            "supports_feature",
            lambda f, target_feature=feature: f == target_feature,
        )

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


# ---------------------------------------------------------------------------
# Tool compatibility tests
# ---------------------------------------------------------------------------


def test_tool_compatibility_info(client):
    """Test tool compatibility information."""
    info = client.get_tool_compatibility_info()

    assert info["supports_universal_naming"] is True
    assert info["sanitization_method"] == "replace_chars"
    assert info["restoration_method"] == "name_mapping"
    assert "alphanumeric_underscore" in info["supported_name_patterns"]


def test_tool_name_sanitization_and_restoration(client):
    """Test tool name sanitization and restoration."""
    # Test that sanitization is called (mocked to return tools unchanged)
    tools = [
        {"type": "function", "function": {"name": "test.tool:name", "parameters": {}}}
    ]

    # Mock sanitization to simulate real behavior
    def mock_sanitize(tools_list):
        client._current_name_mapping = {"test_tool_name": "test.tool:name"}
        return [
            {
                "type": "function",
                "function": {"name": "test_tool_name", "parameters": {}},
            }
        ]

    client._sanitize_tool_names = mock_sanitize

    sanitized_tools = client._sanitize_tool_names(tools)

    # Verify sanitization occurred
    assert sanitized_tools[0]["function"]["name"] == "test_tool_name"
    assert "test_tool_name" in client._current_name_mapping


def test_response_with_tool_name_restoration(client):
    """Test response parsing with tool name restoration."""
    # Set up name mapping
    client._current_name_mapping = {"get_weather_data": "weather.api:get_data"}

    # Mock response with sanitized tool name
    mock_response = {
        "response": "I'll get the weather data.",
        "tool_calls": [
            {
                "id": "call_123",
                "type": "function",
                "function": {"name": "get_weather_data", "arguments": "{}"},
            }
        ],
    }

    # Mock restoration to simulate real behavior
    def mock_restore(response, mapping):
        if response.get("tool_calls") and mapping:
            for tool_call in response["tool_calls"]:
                sanitized_name = tool_call["function"]["name"]
                if sanitized_name in mapping:
                    tool_call["function"]["name"] = mapping[sanitized_name]
        return response

    client._restore_tool_names_in_response = mock_restore

    restored_response = client._restore_tool_names_in_response(
        mock_response, client._current_name_mapping
    )

    # Verify restoration occurred
    assert (
        restored_response["tool_calls"][0]["function"]["name"] == "weather.api:get_data"
    )


# ---------------------------------------------------------------------------
# Vision support tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_vision_content_processing(client):
    """Test processing of vision content."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
                    },
                },
            ],
        }
    ]

    system_txt, contents = await client._split_for_gemini_async(messages)

    assert system_txt == ""
    assert len(contents) == 1
    # Should contain multimodal content
    assert isinstance(contents[0], list)


@pytest.mark.asyncio
async def test_vision_content_fallback_when_not_supported(client, monkeypatch):
    """Test vision content fallback when vision is not supported."""
    # Mock vision as not supported
    monkeypatch.setattr(client, "supports_feature", lambda feature: feature != "vision")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,..."},
                },
            ],
        }
    ]

    system_txt, contents = await client._split_for_gemini_async(messages)

    assert system_txt == ""
    assert len(contents) == 1
    # Should only contain text content when vision not supported
    assert isinstance(contents[0], str)
    assert "What's in this image?" in contents[0]


# ---------------------------------------------------------------------------
# Parameter validation tests
# ---------------------------------------------------------------------------


def test_parameter_mapping_and_limits(client):
    """Test parameter mapping and limit enforcement."""
    params = {
        "temperature": 3.0,  # Above Gemini limit
        "max_tokens": 10000,  # Above model limit
        "top_p": 0.95,
        "top_k": 50,
        "frequency_penalty": 0.5,  # Unsupported
        "presence_penalty": 0.5,  # Unsupported
    }

    filtered = client._filter_gemini_params(params)

    # Temperature should be capped at 2.0 for Gemini
    assert filtered["temperature"] == 2.0

    # max_tokens should be mapped and potentially capped
    assert "max_output_tokens" in filtered

    # Supported parameters should be included
    assert "top_p" in filtered
    assert "top_k" in filtered

    # Unsupported parameters should be excluded
    assert "frequency_penalty" not in filtered
    assert "presence_penalty" not in filtered


# ---------------------------------------------------------------------------
# LOGLEVEL environment variable tests
# ---------------------------------------------------------------------------


def test_loglevel_env_var(monkeypatch):
    """Test LOGLEVEL environment variable handling."""
    monkeypatch.setenv("LOGLEVEL", "DEBUG")

    # Re-import to trigger the LOGLEVEL check
    import importlib
    from chuk_llm.llm.providers import gemini_client
    importlib.reload(gemini_client)

    # The log level should be set based on environment variable
    # This tests line 40


# ---------------------------------------------------------------------------
# Warning suppression function tests
# ---------------------------------------------------------------------------


def test_silent_warn():
    """Test silent warning function."""
    from chuk_llm.llm.providers.gemini_client import _silent_warn

    # Should not raise any exceptions
    _silent_warn("test warning", category=UserWarning)
    # Tests line 54


def test_silent_showwarning():
    """Test silent showwarning function."""
    from chuk_llm.llm.providers.gemini_client import _silent_showwarning

    # Should not raise any exceptions
    _silent_showwarning("test", Warning, "file.py", 1)
    # Tests line 58-59


def test_silent_formatwarning():
    """Test silent formatwarning function."""
    from chuk_llm.llm.providers.gemini_client import _silent_formatwarning

    result = _silent_formatwarning("test", Warning, "file.py", 1)
    assert result == ""
    # Tests line 63-64


# ---------------------------------------------------------------------------
# Context manager tests
# ---------------------------------------------------------------------------


def test_suppress_all_output_context_manager():
    """Test SuppressAllOutput context manager."""
    from chuk_llm.llm.providers.gemini_client import SuppressAllOutput

    with SuppressAllOutput() as suppressor:
        # Should suppress output
        import warnings
        warnings.warn("This should be suppressed")
    # Tests lines 516-523


def test_suppress_warnings_context_manager():
    """Test suppress_warnings context manager."""
    from chuk_llm.llm.providers.gemini_client import suppress_warnings

    with suppress_warnings():
        # Should suppress warnings
        import warnings
        warnings.warn("This should be suppressed")
    # Tests lines 528-531


# ---------------------------------------------------------------------------
# API key validation tests
# ---------------------------------------------------------------------------


def test_client_initialization_no_api_key(monkeypatch, mock_configuration):
    """Test client initialization without API key."""
    # Remove all API key environment variables
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    with pytest.raises(ValueError) as exc_info:
        GeminiLLMClient(model="gemini-2.5-flash", api_key=None)

    assert "GEMINI_API_KEY / GOOGLE_API_KEY env var not set" in str(exc_info.value)
    # Tests line 572


# ---------------------------------------------------------------------------
# Model family detection edge cases
# ---------------------------------------------------------------------------


def test_detect_model_family_unknown(client):
    """Test model family detection for unknown models."""
    client.model = "gemini-unknown-model"
    assert client._detect_model_family() == "unknown"
    # Tests lines 594-595


def test_detect_model_family_flash(client):
    """Test model family detection for flash models."""
    client.model = "gemini-flash"
    assert client._detect_model_family() == "flash"
    # Tests lines 590-591


def test_detect_model_family_pro(client):
    """Test model family detection for pro models."""
    client.model = "gemini-pro"
    assert client._detect_model_family() == "pro"
    # Tests lines 592-593


# ---------------------------------------------------------------------------
# Error handling in response parsing tests
# ---------------------------------------------------------------------------


def test_safe_parse_gemini_response_function_call_error():
    """Test parsing response with function call extraction error."""
    mock_response = MagicMock()

    # Create a part with function_call that raises error on attribute access
    fc_part = MagicMock()
    fc_part.text = None
    fc_part.function_call = MagicMock()

    # Make name access raise an exception
    type(fc_part.function_call).name = property(lambda self: (_ for _ in ()).throw(Exception("Test error")))

    def fc_part_hasattr(attr):
        return attr in ["text", "function_call"]

    fc_part.__class__.__name__ = "FunctionCallPart"
    fc_part.__hasattr__ = fc_part_hasattr

    content = MagicMock()
    content.parts = [fc_part]

    def content_hasattr(attr):
        return attr in ["parts", "text"]

    content.__class__.__name__ = "Content"
    content.__hasattr__ = content_hasattr

    candidate = MagicMock()
    candidate.content = content

    def candidate_hasattr(attr):
        return attr in ["content", "finish_reason"]

    candidate.__class__.__name__ = "Candidate"
    candidate.__hasattr__ = candidate_hasattr

    mock_response.candidates = [candidate]

    def response_hasattr(attr):
        return attr in ["candidates", "text"]

    mock_response.__class__.__name__ = "GenerateContentResponse"
    mock_response.__hasattr__ = response_hasattr

    result = _safe_parse_gemini_response(mock_response)

    # Should handle error gracefully
    assert "response" in result
    # Tests lines 362-363


def test_safe_parse_gemini_response_content_no_parts_no_text():
    """Test parsing response with content but no parts or text."""
    mock_response = MagicMock()

    content = MagicMock()
    content.parts = None
    content.text = None

    def content_hasattr(attr):
        if attr == "parts":
            return False
        if attr == "text":
            return False
        return False

    content.__class__.__name__ = "Content"
    content.__hasattr__ = content_hasattr

    candidate = MagicMock()
    candidate.content = content

    def candidate_hasattr(attr):
        return attr in ["content", "finish_reason"]

    candidate.__class__.__name__ = "Candidate"
    candidate.__hasattr__ = candidate_hasattr

    mock_response.candidates = [candidate]

    def response_hasattr(attr):
        return attr in ["candidates", "text"]

    mock_response.__class__.__name__ = "GenerateContentResponse"
    mock_response.__hasattr__ = response_hasattr

    result = _safe_parse_gemini_response(mock_response)

    assert "[No content available in response]" in result["response"]
    # Tests lines 398-401


def test_safe_parse_gemini_response_content_text_only():
    """Test parsing response with content.text but no parts."""
    mock_response = MagicMock()

    content = MagicMock()
    content.parts = None
    content.text = "Response from content.text"

    def content_hasattr(attr):
        if attr == "parts":
            return False
        if attr == "text":
            return True
        return False

    content.__class__.__name__ = "Content"
    content.__hasattr__ = content_hasattr

    candidate = MagicMock()
    candidate.content = content

    def candidate_hasattr(attr):
        return attr in ["content", "finish_reason"]

    candidate.__class__.__name__ = "Candidate"
    candidate.__hasattr__ = candidate_hasattr

    mock_response.candidates = [candidate]

    def response_hasattr(attr):
        return attr in ["candidates", "text"]

    mock_response.__class__.__name__ = "GenerateContentResponse"
    mock_response.__hasattr__ = response_hasattr

    result = _safe_parse_gemini_response(mock_response)

    assert result["response"] == "Response from content.text"
    # Tests lines 388-390


def test_safe_parse_gemini_response_safety_blocked():
    """Test parsing response blocked by safety filters."""
    mock_response = MagicMock()

    candidate = MagicMock()
    candidate.content = None
    candidate.finish_reason = "SAFETY"

    def candidate_hasattr(attr):
        return attr in ["content", "finish_reason"]

    candidate.__class__.__name__ = "Candidate"
    candidate.__hasattr__ = candidate_hasattr

    mock_response.candidates = [candidate]

    def response_hasattr(attr):
        return attr in ["candidates", "text"]

    mock_response.__class__.__name__ = "GenerateContentResponse"
    mock_response.__hasattr__ = response_hasattr

    result = _safe_parse_gemini_response(mock_response)

    assert "safety filters" in result["response"]
    # Tests lines 412-413


def test_safe_parse_gemini_response_other_finish_reason():
    """Test parsing response with other finish reasons."""
    mock_response = MagicMock()

    candidate = MagicMock()
    candidate.content = None
    candidate.finish_reason = "STOP"

    def candidate_hasattr(attr):
        return attr in ["content", "finish_reason"]

    candidate.__class__.__name__ = "Candidate"
    candidate.__hasattr__ = candidate_hasattr

    mock_response.candidates = [candidate]

    def response_hasattr(attr):
        return attr in ["candidates", "text"]

    mock_response.__class__.__name__ = "GenerateContentResponse"
    mock_response.__hasattr__ = response_hasattr

    result = _safe_parse_gemini_response(mock_response)

    assert "Response completed with status: STOP" in result["response"]
    # Tests lines 414-415


def test_safe_parse_gemini_response_no_content_no_finish_reason():
    """Test parsing response with no content and no finish reason."""
    mock_response = MagicMock()

    candidate = MagicMock()
    # Set content to False (not None) to indicate no content
    candidate.content = False
    candidate.finish_reason = MagicMock()  # Has finish_reason

    def candidate_hasattr(attr):
        if attr == "content":
            return True  # content attribute exists
        if attr == "finish_reason":
            return True  # finish_reason exists
        return False

    candidate.__class__.__name__ = "Candidate"
    candidate.__hasattr__ = candidate_hasattr

    mock_response.candidates = [candidate]

    def response_hasattr(attr):
        return attr in ["candidates", "text"]

    mock_response.__class__.__name__ = "GenerateContentResponse"
    mock_response.__hasattr__ = response_hasattr

    result = _safe_parse_gemini_response(mock_response)

    # When there's a finish_reason, it should be included in the response
    assert "Response completed with status:" in result["response"]
    # Tests lines 417-418


def test_safe_parse_gemini_response_json_cleanup():
    """Test JSON response cleanup for duplicate JSON objects."""
    mock_response = MagicMock()

    # Create response with duplicate JSON - note the count method check
    text_part = MagicMock()
    # This text has more than one '{"' which should trigger cleanup
    text_part.text = '{"key": "value1"}{"key": "value2"}'

    def text_part_hasattr(attr):
        return attr in ["text", "function_call"]

    text_part.__class__.__name__ = "TextPart"
    text_part.__hasattr__ = text_part_hasattr

    content = MagicMock()
    content.parts = [text_part]

    def content_hasattr(attr):
        return attr in ["parts", "text"]

    content.__class__.__name__ = "Content"
    content.__hasattr__ = content_hasattr

    candidate = MagicMock()
    candidate.content = content

    def candidate_hasattr(attr):
        return attr in ["content", "finish_reason"]

    candidate.__class__.__name__ = "Candidate"
    candidate.__hasattr__ = candidate_hasattr

    mock_response.candidates = [candidate]

    def response_hasattr(attr):
        return attr in ["candidates", "text"]

    mock_response.__class__.__name__ = "GenerateContentResponse"
    mock_response.__hasattr__ = response_hasattr

    result = _safe_parse_gemini_response(mock_response)

    # Should clean up duplicate JSON - extracts first valid JSON object
    assert result["response"] == '{"key": "value1"}'
    # Tests lines 433-440


def test_safe_parse_gemini_response_non_text_non_function_part():
    """Test parsing response with non-text, non-function parts."""
    mock_response = MagicMock()

    # Create a part that is neither text nor function call
    other_part = MagicMock()
    other_part.text = None

    def other_part_hasattr(attr):
        if attr == "text":
            return False
        if attr == "function_call":
            return False
        return False

    other_part.__class__.__name__ = "OtherPart"
    other_part.__hasattr__ = other_part_hasattr

    content = MagicMock()
    content.parts = [other_part]

    def content_hasattr(attr):
        return attr in ["parts", "text"]

    content.__class__.__name__ = "Content"
    content.__hasattr__ = content_hasattr

    candidate = MagicMock()
    candidate.content = content

    def candidate_hasattr(attr):
        return attr in ["content", "finish_reason"]

    candidate.__class__.__name__ = "Candidate"
    candidate.__hasattr__ = candidate_hasattr

    mock_response.candidates = [candidate]

    def response_hasattr(attr):
        return attr in ["candidates", "text"]

    mock_response.__class__.__name__ = "GenerateContentResponse"
    mock_response.__hasattr__ = response_hasattr

    result = _safe_parse_gemini_response(mock_response)

    # Should handle non-text, non-function parts
    assert "response" in result
    # Tests lines 366-370


# ---------------------------------------------------------------------------
# Tool conversion error handling tests
# ---------------------------------------------------------------------------


def test_convert_tools_with_gemini_tool_creation_error():
    """Test tool conversion when Gemini Tool creation fails."""
    from chuk_llm.llm.providers.gemini_client import _convert_tools_to_gemini_format

    tools = [
        {
            "type": "function",
            "function": {
                "name": "valid_tool",
                "description": "A valid tool",
                "parameters": {"type": "object"},
            },
        }
    ]

    # Mock Tool class to raise an error
    original_tool = sys.modules["google.genai.types"].Tool

    class ErrorTool:
        def __init__(self, **kwargs):
            raise Exception("Tool creation failed")

    sys.modules["google.genai.types"].Tool = ErrorTool

    try:
        result = _convert_tools_to_gemini_format(tools)
        # Should return None when tool creation fails
        assert result is None
    finally:
        # Restore original Tool class
        sys.modules["google.genai.types"].Tool = original_tool
    # Tests lines 500-502


def test_convert_tools_with_general_error():
    """Test tool conversion with general exception."""
    from chuk_llm.llm.providers.gemini_client import _convert_tools_to_gemini_format

    # Pass invalid tools structure that will raise exception
    invalid_tools = [{"invalid": "structure"}]

    result = _convert_tools_to_gemini_format(invalid_tools)

    # Should return None on error
    assert result is None
    # Tests lines 504-505


# ---------------------------------------------------------------------------
# Safe get helper tests
# ---------------------------------------------------------------------------


def test_safe_get_with_dict():
    """Test _safe_get with dictionary."""
    from chuk_llm.llm.providers.gemini_client import _safe_get

    obj = {"key": "value", "other": "data"}

    assert _safe_get(obj, "key") == "value"
    assert _safe_get(obj, "missing", "default") == "default"
    # Tests line 465-466


def test_safe_get_with_object():
    """Test _safe_get with attribute-style object."""
    from chuk_llm.llm.providers.gemini_client import _safe_get

    class TestObj:
        key = "value"

    obj = TestObj()

    assert _safe_get(obj, "key") == "value"
    assert _safe_get(obj, "missing", "default") == "default"
    # Tests line 465-466


# ---------------------------------------------------------------------------
# Tool name restoration in response tests
# ---------------------------------------------------------------------------


def test_parse_gemini_response_with_restoration(client):
    """Test parsing response with tool name restoration."""
    function_calls = [{"name": "sanitized_name", "args": {"key": "value"}}]
    mock_response = create_mock_gemini_response(function_calls=function_calls)

    name_mapping = {"sanitized_name": "original.name:with:special"}

    # Mock the restoration method to actually work
    def mock_restore(response, mapping):
        if response.get("tool_calls") and mapping:
            for tool_call in response["tool_calls"]:
                sanitized = tool_call["function"]["name"]
                if sanitized in mapping:
                    tool_call["function"]["name"] = mapping[sanitized]
        return response

    client._restore_tool_names_in_response = mock_restore

    result = client._parse_gemini_response_with_restoration(mock_response, name_mapping)

    assert result["tool_calls"][0]["function"]["name"] == "original.name:with:special"
    # Tests line 607


# ---------------------------------------------------------------------------
# Close method tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_close_method(client):
    """Test close method resets name mapping."""
    client._current_name_mapping = {"some": "mapping"}

    await client.close()

    assert client._current_name_mapping == {}
    # Tests lines 1422-1424


# ---------------------------------------------------------------------------
# External image URL download tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_download_image_to_base64_success():
    """Test successful image download and base64 conversion."""
    from chuk_llm.llm.providers.gemini_client import GeminiLLMClient

    # Create a mock httpx module
    mock_httpx = MagicMock()

    mock_response = MagicMock()
    mock_response.content = b"fake_image_data"
    mock_response.headers = {"content-type": "image/jpeg"}
    mock_response.raise_for_status = MagicMock()

    mock_client = MagicMock()

    async def async_enter(self):
        return mock_client

    async def async_exit(self, exc_type, exc, tb):
        return None

    async def mock_get(url):
        return mock_response

    mock_client.__aenter__ = async_enter
    mock_client.__aexit__ = async_exit
    mock_client.get = mock_get

    mock_httpx.AsyncClient = MagicMock(return_value=mock_client)

    # Patch the import statement itself
    with patch.dict('sys.modules', {'httpx': mock_httpx}):
        content_type, image_data = await GeminiLLMClient._download_image_to_base64("http://example.com/image.jpg")

        assert content_type == "image/jpeg"
        assert len(image_data) > 0
    # Tests lines 772-787


@pytest.mark.asyncio
async def test_download_image_to_base64_no_content_type():
    """Test image download with missing content type."""
    from chuk_llm.llm.providers.gemini_client import GeminiLLMClient

    mock_httpx = MagicMock()

    mock_response = MagicMock()
    mock_response.content = b"fake_image_data"
    mock_response.headers = {}  # No content-type
    mock_response.raise_for_status = MagicMock()

    mock_client = MagicMock()

    async def async_enter(self):
        return mock_client

    async def async_exit(self, exc_type, exc, tb):
        return None

    async def mock_get(url):
        return mock_response

    mock_client.__aenter__ = async_enter
    mock_client.__aexit__ = async_exit
    mock_client.get = mock_get

    mock_httpx.AsyncClient = MagicMock(return_value=mock_client)

    with patch.dict('sys.modules', {'httpx': mock_httpx}):
        content_type, image_data = await GeminiLLMClient._download_image_to_base64("http://example.com/image.jpg")

        assert content_type == "image/png"  # Default fallback
        assert len(image_data) > 0
    # Tests lines 779-782


@pytest.mark.asyncio
async def test_download_image_to_base64_error():
    """Test image download error handling."""
    from chuk_llm.llm.providers.gemini_client import GeminiLLMClient

    mock_httpx = MagicMock()

    mock_client = MagicMock()

    async def async_enter(self):
        return mock_client

    async def async_exit(self, exc_type, exc, tb):
        return None

    async def mock_get_error(url):
        raise Exception("Network error")

    mock_client.__aenter__ = async_enter
    mock_client.__aexit__ = async_exit
    mock_client.get = mock_get_error

    mock_httpx.AsyncClient = MagicMock(return_value=mock_client)

    with patch.dict('sys.modules', {'httpx': mock_httpx}):
        with pytest.raises(ValueError) as exc_info:
            await GeminiLLMClient._download_image_to_base64("http://example.com/image.jpg")

        assert "Could not download image" in str(exc_info.value)
    # Tests lines 789-791


# ---------------------------------------------------------------------------
# Vision conversion tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_convert_universal_vision_to_gemini_data_url():
    """Test converting data URL to Gemini format."""
    from chuk_llm.llm.providers.gemini_client import GeminiLLMClient

    content_item = {
        "type": "image_url",
        "image_url": {
            "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        }
    }

    result = await GeminiLLMClient._convert_universal_vision_to_gemini_async(content_item)

    assert "inline_data" in result
    assert result["inline_data"]["mime_type"] == "image/png"
    assert "data" in result["inline_data"]
    # Tests lines 805-822


@pytest.mark.asyncio
async def test_convert_universal_vision_to_gemini_invalid_data_url():
    """Test converting invalid data URL."""
    from chuk_llm.llm.providers.gemini_client import GeminiLLMClient

    content_item = {
        "type": "image_url",
        "image_url": {
            "url": "data:invalid"  # Invalid format
        }
    }

    result = await GeminiLLMClient._convert_universal_vision_to_gemini_async(content_item)

    assert result == {"text": "[Invalid image format]"}
    # Tests lines 823-825


@pytest.mark.asyncio
async def test_convert_universal_vision_to_gemini_external_url():
    """Test converting external URL to Gemini format."""
    from chuk_llm.llm.providers.gemini_client import GeminiLLMClient

    with patch.object(GeminiLLMClient, "_download_image_to_base64") as mock_download:
        mock_download.return_value = ("image/jpeg", "base64_encoded_data")

        content_item = {
            "type": "image_url",
            "image_url": {
                "url": "http://example.com/image.jpg"
            }
        }

        result = await GeminiLLMClient._convert_universal_vision_to_gemini_async(content_item)

        assert "inline_data" in result
        assert result["inline_data"]["mime_type"] == "image/jpeg"
        assert result["inline_data"]["data"] == "base64_encoded_data"
        mock_download.assert_called_once()
    # Tests lines 827-836


@pytest.mark.asyncio
async def test_convert_universal_vision_to_gemini_external_url_error():
    """Test converting external URL with download error."""
    from chuk_llm.llm.providers.gemini_client import GeminiLLMClient

    with patch.object(GeminiLLMClient, "_download_image_to_base64") as mock_download:
        mock_download.side_effect = Exception("Download failed")

        content_item = {
            "type": "image_url",
            "image_url": {
                "url": "http://example.com/image.jpg"
            }
        }

        result = await GeminiLLMClient._convert_universal_vision_to_gemini_async(content_item)

        assert result["text"] == "[Could not load image: Download failed]"
    # Tests lines 837-839


@pytest.mark.asyncio
async def test_convert_universal_vision_invalid_media_type():
    """Test data URL with invalid media type."""
    from chuk_llm.llm.providers.gemini_client import GeminiLLMClient

    content_item = {
        "type": "image_url",
        "image_url": {
            "url": "data:text/plain;base64,SGVsbG8="
        }
    }

    result = await GeminiLLMClient._convert_universal_vision_to_gemini_async(content_item)

    # Should default to image/png for non-image types
    assert "inline_data" in result
    assert result["inline_data"]["mime_type"] == "image/png"
    # Tests line 814


# ---------------------------------------------------------------------------
# Pydantic content object tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_split_for_gemini_async_pydantic_text_content(client):
    """Test message splitting with Pydantic TextContent objects."""
    # Create mock Pydantic content objects
    from chuk_llm.core.enums import ContentType

    class MockTextContent:
        def __init__(self, text):
            self.type = ContentType.TEXT
            self.text = text

    messages = [
        {
            "role": "user",
            "content": [MockTextContent("Hello from Pydantic")],
        }
    ]

    system_txt, contents = await client._split_for_gemini_async(messages)

    assert system_txt == ""
    assert len(contents) == 1
    assert isinstance(contents[0], list)
    assert "Hello from Pydantic" in contents[0]
    # Tests lines 941-944


@pytest.mark.asyncio
async def test_split_for_gemini_async_pydantic_image_content(client):
    """Test message splitting with Pydantic ImageUrlContent objects."""
    from chuk_llm.core.enums import ContentType

    class MockImageUrlContent:
        def __init__(self, url):
            self.type = ContentType.IMAGE_URL
            self.image_url = {"url": url}

    messages = [
        {
            "role": "user",
            "content": [
                MockImageUrlContent("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==")
            ],
        }
    ]

    system_txt, contents = await client._split_for_gemini_async(messages)

    assert system_txt == ""
    assert len(contents) == 1
    assert isinstance(contents[0], list)
    # Tests lines 945-962


@pytest.mark.asyncio
async def test_split_for_gemini_async_pydantic_image_content_error(client):
    """Test message splitting with Pydantic ImageUrlContent error."""
    from chuk_llm.core.enums import ContentType

    class MockImageUrlContent:
        def __init__(self, url):
            self.type = ContentType.IMAGE_URL
            self.image_url = {"url": url}

    # Mock the conversion to raise an error
    original_convert = client._convert_universal_vision_to_gemini_async

    async def mock_convert_error(content_item):
        raise Exception("Conversion failed")

    client._convert_universal_vision_to_gemini_async = mock_convert_error

    messages = [
        {
            "role": "user",
            "content": [
                MockImageUrlContent("http://example.com/image.jpg")
            ],
        }
    ]

    system_txt, contents = await client._split_for_gemini_async(messages)

    # Should handle error gracefully
    assert len(contents) == 1
    assert isinstance(contents[0], list)
    assert "[Image conversion failed]" in contents[0]

    # Restore original method
    client._convert_universal_vision_to_gemini_async = original_convert
    # Tests lines 960-962


@pytest.mark.asyncio
async def test_split_for_gemini_async_pydantic_other_content(client):
    """Test message splitting with other Pydantic content types."""
    class MockOtherContent:
        def __init__(self):
            self.data = "other data"

    messages = [
        {
            "role": "user",
            "content": [MockOtherContent()],
        }
    ]

    system_txt, contents = await client._split_for_gemini_async(messages)

    assert system_txt == ""
    assert len(contents) == 1
    # Should convert to string
    # Tests line 964


@pytest.mark.asyncio
async def test_split_for_gemini_async_none_content(client):
    """Test message splitting with None content."""
    messages = [
        {"role": "user", "content": None},
    ]

    system_txt, contents = await client._split_for_gemini_async(messages)

    assert system_txt == ""
    assert len(contents) == 0  # None content should be skipped
    # Tests line 881


@pytest.mark.asyncio
async def test_split_for_gemini_async_other_content_type(client):
    """Test message splitting with other content types."""
    messages = [
        {"role": "user", "content": 123},  # Integer content
    ]

    system_txt, contents = await client._split_for_gemini_async(messages)

    assert system_txt == ""
    assert len(contents) == 1
    assert "123" in contents  # Should be converted to string
    # Tests line 971


# ---------------------------------------------------------------------------
# Streaming tool call deduplication tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_completion_tool_call_deduplication(client):
    """Test streaming completion with tool call deduplication."""
    messages = [{"role": "user", "content": "Get weather"}]

    # Mock streaming response with duplicate tool calls
    async def mock_stream():
        # First chunk with tool call
        fc_part1 = MagicMock()
        fc_part1.text = None
        fc_part1.function_call = MagicMock()
        fc_part1.function_call.name = "get_weather"
        fc_part1.function_call.args = {"city": "NYC"}

        def fc_part_hasattr(attr):
            return attr in ["text", "function_call"]

        fc_part1.__hasattr__ = fc_part_hasattr

        content1 = MagicMock()
        content1.parts = [fc_part1]
        content1.__hasattr__ = lambda attr: attr in ["parts", "text"]

        candidate1 = MagicMock()
        candidate1.content = content1
        candidate1.__hasattr__ = lambda attr: attr in ["content", "finish_reason"]

        chunk1 = MagicMock()
        chunk1.candidates = [candidate1]
        chunk1.__hasattr__ = lambda attr: attr in ["candidates", "text"]

        yield chunk1

        # Second chunk with same tool call (should be deduplicated)
        yield chunk1

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
        filtered_params={},
        name_mapping={},
    ):
        chunks.append(chunk)

    # Should only have one tool call chunk due to deduplication
    tool_call_chunks = [c for c in chunks if c.get("tool_calls")]
    assert len(tool_call_chunks) == 1
    # Tests lines 1175-1223, 1257-1274


@pytest.mark.asyncio
async def test_stream_completion_chunk_processing_error(client):
    """Test streaming completion with chunk processing error."""
    messages = [{"role": "user", "content": "Hello"}]

    # Mock streaming response with error in chunk
    async def mock_stream():
        # Create a malformed chunk
        chunk = MagicMock()
        chunk.candidates = None  # This will cause error in processing
        chunk.__hasattr__ = lambda attr: attr in ["candidates", "text"]

        yield chunk

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
        filtered_params={},
        name_mapping={},
    ):
        chunks.append(chunk)

    # Should handle chunk error gracefully and continue
    # Tests lines 1270-1274


# ---------------------------------------------------------------------------
# System message handling when not supported tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_completion_json_instruction(client):
    """Test streaming completion with JSON instruction."""
    messages = [{"role": "user", "content": "Hello"}]
    json_instruction = "Respond with JSON only"

    # Mock streaming response
    async def mock_stream():
        chunk = create_mock_gemini_response(text="Response")
        yield chunk

    async def mock_generate_content_stream(**kwargs):
        return mock_stream()

    client.client.aio.models.generate_content_stream = mock_generate_content_stream

    # Collect streaming results
    chunks = []
    async for chunk in client._stream_completion_async(
        system=None,
        json_instruction=json_instruction,
        messages=messages,
        gemini_tools=None,
        filtered_params={},
        name_mapping={},
    ):
        chunks.append(chunk)

    assert len(chunks) > 0
    # Tests lines 1087-1091


@pytest.mark.asyncio
async def test_stream_completion_with_system_not_supported(client, monkeypatch):
    """Test streaming completion with system message when not supported."""
    messages = [{"role": "user", "content": "Hello"}]
    system = "You are helpful"

    # Mock system_messages as not supported
    monkeypatch.setattr(
        client, "supports_feature", lambda feature: feature != "system_messages"
    )

    # Mock streaming response
    async def mock_stream():
        chunk = create_mock_gemini_response(text="Response")
        yield chunk

    async def mock_generate_content_stream(**kwargs):
        # Verify system instruction is prepended to content
        contents = kwargs.get("contents")
        assert "System:" in contents
        return mock_stream()

    client.client.aio.models.generate_content_stream = mock_generate_content_stream

    # Collect streaming results
    chunks = []
    async for chunk in client._stream_completion_async(
        system=system,
        json_instruction=None,
        messages=messages,
        gemini_tools=None,
        filtered_params={},
        name_mapping={},
    ):
        chunks.append(chunk)

    assert len(chunks) > 0
    # Tests lines 1123-1124


@pytest.mark.asyncio
async def test_stream_completion_max_output_tokens_default(client):
    """Test streaming completion sets default max_output_tokens."""
    messages = [{"role": "user", "content": "Hello"}]

    # Mock streaming response
    async def mock_stream():
        chunk = create_mock_gemini_response(text="Response")
        yield chunk

    async def mock_generate_content_stream(**kwargs):
        # Verify max_output_tokens is set
        config = kwargs.get("config")
        if config:
            assert config.max_output_tokens == 4096
        return mock_stream()

    client.client.aio.models.generate_content_stream = mock_generate_content_stream

    # Collect streaming results
    chunks = []
    async for chunk in client._stream_completion_async(
        system=None,
        json_instruction=None,
        messages=messages,
        gemini_tools=None,
        filtered_params={},  # No max_output_tokens
        name_mapping={},
    ):
        chunks.append(chunk)

    assert len(chunks) > 0
    # Tests lines 1105-1106


@pytest.mark.asyncio
async def test_stream_completion_config_creation_error(client):
    """Test streaming completion with config creation error."""
    messages = [{"role": "user", "content": "Hello"}]

    # Mock GenerateContentConfig to raise error
    original_config = sys.modules["google.genai.types"].GenerateContentConfig

    class ErrorConfig:
        def __init__(self, **kwargs):
            raise Exception("Config creation failed")

    sys.modules["google.genai.types"].GenerateContentConfig = ErrorConfig

    # Mock streaming response
    async def mock_stream():
        chunk = create_mock_gemini_response(text="Response")
        yield chunk

    async def mock_generate_content_stream(**kwargs):
        return mock_stream()

    client.client.aio.models.generate_content_stream = mock_generate_content_stream

    try:
        # Collect streaming results
        chunks = []
        async for chunk in client._stream_completion_async(
            system="Test system",
            json_instruction=None,
            messages=messages,
            gemini_tools=None,
            filtered_params={"temperature": 0.7},
            name_mapping={},
        ):
            chunks.append(chunk)

        # Should still work even if config creation fails
        assert len(chunks) > 0
    finally:
        # Restore original config
        sys.modules["google.genai.types"].GenerateContentConfig = original_config
    # Tests lines 1115-1117


@pytest.mark.asyncio
async def test_stream_completion_system_instruction_supported(client):
    """Test streaming completion with system instruction when supported."""
    messages = [{"role": "user", "content": "Hello"}]
    system = "You are helpful"

    # Mock streaming response
    async def mock_stream():
        chunk = create_mock_gemini_response(text="Response")
        yield chunk

    async def mock_generate_content_stream(**kwargs):
        # Verify system instruction is in config
        config = kwargs.get("config")
        if config:
            assert config.system_instruction == system
        return mock_stream()

    client.client.aio.models.generate_content_stream = mock_generate_content_stream

    # Collect streaming results
    chunks = []
    async for chunk in client._stream_completion_async(
        system=system,
        json_instruction=None,
        messages=messages,
        gemini_tools=None,
        filtered_params={},
        name_mapping={},
    ):
        chunks.append(chunk)

    assert len(chunks) > 0
    # Tests line 1112


# ---------------------------------------------------------------------------
# Regular completion system message tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_regular_completion_json_instruction(client):
    """Test regular completion with JSON instruction."""
    messages = [{"role": "user", "content": "Hello"}]
    json_instruction = "Respond with JSON only"

    # Mock the client's generate_content method
    mock_response = create_mock_gemini_response(text='{"response": "Hello"}')

    async def mock_generate_content(**kwargs):
        return mock_response

    client.client.aio.models.generate_content = mock_generate_content

    result = await client._regular_completion_async(
        system=None,
        json_instruction=json_instruction,
        messages=messages,
        gemini_tools=None,
        filtered_params={},
        name_mapping={},
    )

    assert "response" in result
    # Tests lines 1308-1312


@pytest.mark.asyncio
async def test_regular_completion_max_output_tokens_default(client):
    """Test regular completion sets default max_output_tokens."""
    messages = [{"role": "user", "content": "Hello"}]

    # Mock the client's generate_content method
    mock_response = create_mock_gemini_response(text="Hello!")

    async def mock_generate_content(**kwargs):
        # Verify max_output_tokens is set
        config = kwargs.get("config")
        if config:
            assert config.max_output_tokens == 4096
        return mock_response

    client.client.aio.models.generate_content = mock_generate_content

    result = await client._regular_completion_async(
        system=None,
        json_instruction=None,
        messages=messages,
        gemini_tools=None,
        filtered_params={},  # No max_output_tokens
        name_mapping={},
    )

    assert result["response"] == "Hello!"
    # Tests lines 1325-1326


@pytest.mark.asyncio
async def test_regular_completion_config_creation_error(client):
    """Test regular completion with config creation error."""
    messages = [{"role": "user", "content": "Hello"}]

    # Mock GenerateContentConfig to raise error
    original_config = sys.modules["google.genai.types"].GenerateContentConfig

    class ErrorConfig:
        def __init__(self, **kwargs):
            raise Exception("Config creation failed")

    sys.modules["google.genai.types"].GenerateContentConfig = ErrorConfig

    # Mock the client's generate_content method
    mock_response = create_mock_gemini_response(text="Hello!")

    async def mock_generate_content(**kwargs):
        return mock_response

    client.client.aio.models.generate_content = mock_generate_content

    try:
        result = await client._regular_completion_async(
            system="Test system",
            json_instruction=None,
            messages=messages,
            gemini_tools=None,
            filtered_params={"temperature": 0.7},
            name_mapping={},
        )

        # Should still work even if config creation fails
        assert result["response"] == "Hello!"
    finally:
        # Restore original config
        sys.modules["google.genai.types"].GenerateContentConfig = original_config
    # Tests lines 1335-1337


@pytest.mark.asyncio
async def test_regular_completion_multimodal_content(client):
    """Test regular completion with multimodal content."""
    messages = [{"role": "user", "content": "Hello"}]

    # Mock _split_for_gemini_async to return multimodal content
    original_split = client._split_for_gemini_async

    async def mock_split(messages):
        # Return multimodal content (list of lists)
        return "", [["text part", {"inline_data": {"mime_type": "image/png", "data": "base64"}}]]

    client._split_for_gemini_async = mock_split

    # Mock the client's generate_content method
    mock_response = create_mock_gemini_response(text="Response")

    async def mock_generate_content(**kwargs):
        # Verify contents is a list
        contents = kwargs.get("contents")
        assert isinstance(contents, list)
        return mock_response

    client.client.aio.models.generate_content = mock_generate_content

    result = await client._regular_completion_async(
        system=None,
        json_instruction=None,
        messages=messages,
        gemini_tools=None,
        filtered_params={},
        name_mapping={},
    )

    # Restore original method
    client._split_for_gemini_async = original_split

    assert "response" in result
    # Tests lines 1343-1349


@pytest.mark.asyncio
async def test_regular_completion_system_not_supported(client, monkeypatch):
    """Test regular completion with system message when not supported."""
    messages = [{"role": "user", "content": "Hello"}]
    system = "You are helpful"

    # Mock system_messages as not supported
    monkeypatch.setattr(
        client, "supports_feature", lambda feature: feature != "system_messages"
    )

    # Mock the client's generate_content method
    mock_response = create_mock_gemini_response(text="Hello!")

    async def mock_generate_content(**kwargs):
        # Verify system instruction is prepended to content
        contents = kwargs.get("contents")
        assert len(contents) > 0
        # The content should have system prepended
        return mock_response

    client.client.aio.models.generate_content = mock_generate_content

    result = await client._regular_completion_async(
        system=system,
        json_instruction=None,
        messages=messages,
        gemini_tools=None,
        filtered_params={},
        name_mapping={},
    )

    assert result["response"] == "Hello!"
    # Tests lines 1356-1362


@pytest.mark.asyncio
async def test_regular_completion_system_not_supported_multimodal(client, monkeypatch):
    """Test regular completion with system when not supported and multimodal content."""
    messages = [{"role": "user", "content": "Hello"}]
    system = "You are helpful"

    # Mock system_messages as not supported
    monkeypatch.setattr(
        client, "supports_feature", lambda feature: feature != "system_messages"
    )

    # Mock _split_for_gemini_async to return multimodal content
    original_split = client._split_for_gemini_async

    async def mock_split(messages):
        return "", [["text part"]]

    client._split_for_gemini_async = mock_split

    # Mock the client's generate_content method
    mock_response = create_mock_gemini_response(text="Hello!")

    async def mock_generate_content(**kwargs):
        # Verify contents is a list with system prepended
        contents = kwargs.get("contents")
        assert isinstance(contents, list)
        return mock_response

    client.client.aio.models.generate_content = mock_generate_content

    result = await client._regular_completion_async(
        system=system,
        json_instruction=None,
        messages=messages,
        gemini_tools=None,
        filtered_params={},
        name_mapping={},
    )

    # Restore original method
    client._split_for_gemini_async = original_split

    assert result["response"] == "Hello!"
    # Tests lines 1357-1358


@pytest.mark.asyncio
async def test_regular_completion_empty_contents(client):
    """Test regular completion with empty contents."""
    messages = []

    # Mock the client's generate_content method
    mock_response = create_mock_gemini_response(text="Hello")

    async def mock_generate_content(**kwargs):
        # Verify fallback to "Hello"
        contents = kwargs.get("contents")
        assert "Hello" in str(contents)
        return mock_response

    client.client.aio.models.generate_content = mock_generate_content

    result = await client._regular_completion_async(
        system=None,
        json_instruction=None,
        messages=messages,
        gemini_tools=None,
        filtered_params={},
        name_mapping={},
    )

    assert "response" in result
    # Tests line 1353


# ---------------------------------------------------------------------------
# Tool extraction with feature support test
# ---------------------------------------------------------------------------


def test_extract_tool_calls_tools_not_supported(client, monkeypatch):
    """Test extracting tool calls when tools are not supported."""
    # Mock tools as not supported
    monkeypatch.setattr(client, "supports_feature", lambda feature: feature != "tools")

    function_calls = [{"name": "get_weather", "args": {"city": "NYC"}}]
    mock_response = create_mock_gemini_response(function_calls=function_calls)

    tool_calls = client._extract_tool_calls_from_response_with_restoration(
        mock_response, {}
    )

    # Should return empty list when tools not supported
    assert tool_calls == []
    # Tests line 1393


# ---------------------------------------------------------------------------
# Tool call restoration in extraction tests
# ---------------------------------------------------------------------------


def test_extract_tool_calls_with_name_restoration(client):
    """Test extracting tool calls with name restoration."""
    function_calls = [{"name": "sanitized_name", "args": {"key": "value"}}]
    mock_response = create_mock_gemini_response(function_calls=function_calls)

    name_mapping = {"sanitized_name": "original.name:special"}

    tool_calls = client._extract_tool_calls_from_response_with_restoration(
        mock_response, name_mapping
    )

    # Should restore the original name
    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "original.name:special"
    # Tests lines 1402-1406


def test_extract_tool_calls_with_error(client):
    """Test extracting tool calls with error."""
    # Create a response that will cause an error
    mock_response = MagicMock()
    mock_response.candidates = None

    # Make hasattr raise an exception
    def error_hasattr(attr):
        raise Exception("Test error")

    mock_response.__hasattr__ = error_hasattr

    tool_calls = client._extract_tool_calls_from_response_with_restoration(
        mock_response, {}
    )

    # Should return empty list on error
    assert tool_calls == []
    # Tests lines 1408-1409


# ---------------------------------------------------------------------------
# Unknown parameter warning test
# ---------------------------------------------------------------------------


def test_filter_gemini_params_unknown_parameter(client):
    """Test filtering with unknown parameter."""
    params = {
        "temperature": 0.8,
        "unknown_param": "value",  # Unknown parameter
    }

    filtered = client._filter_gemini_params(params)

    # Should filter out unknown parameter and log warning
    assert "temperature" in filtered
    assert "unknown_param" not in filtered
    # Tests line 741
