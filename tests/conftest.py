# tests/conftest.py
"""
Comprehensive test configuration for chuk-llm provider tests.
Provides mocks for all external dependencies and common test utilities.
"""

import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Provider SDK Mocks
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_openai():
    """Mock OpenAI module properly"""
    with patch("chuk_llm.llm.providers.openai_client.openai") as mock_openai_module:
        # Create mock classes
        mock_async_client = AsyncMock()
        mock_sync_client = MagicMock()

        # Mock the AsyncOpenAI class
        mock_openai_module.AsyncOpenAI = MagicMock(return_value=mock_async_client)
        mock_openai_module.OpenAI = MagicMock(return_value=mock_sync_client)

        # Mock chat completions
        mock_async_client.chat.completions.create = AsyncMock()
        mock_sync_client.chat.completions.create = MagicMock()

        yield {
            "module": mock_openai_module,
            "async_client": mock_async_client,
            "sync_client": mock_sync_client,
        }


@pytest.fixture
def mock_anthropic():
    """Mock Anthropic SDK"""
    with patch(
        "chuk_llm.llm.providers.anthropic_client.anthropic"
    ) as mock_anthropic_module:
        mock_async_client = AsyncMock()
        mock_sync_client = MagicMock()

        mock_anthropic_module.AsyncAnthropic = MagicMock(return_value=mock_async_client)
        mock_anthropic_module.Anthropic = MagicMock(return_value=mock_sync_client)

        # Mock messages API
        mock_async_client.messages.create = AsyncMock()
        mock_sync_client.messages.create = MagicMock()

        yield {
            "module": mock_anthropic_module,
            "async_client": mock_async_client,
            "sync_client": mock_sync_client,
        }


@pytest.fixture
def mock_groq():
    """Mock Groq SDK"""
    # Create the groq module if it doesn't exist
    if "groq" not in sys.modules:
        groq_module = types.ModuleType("groq")
        sys.modules["groq"] = groq_module

    with patch("chuk_llm.llm.providers.groq_client.groq") as mock_groq_module:
        mock_async_client = AsyncMock()
        mock_sync_client = MagicMock()

        mock_groq_module.AsyncGroq = MagicMock(return_value=mock_async_client)
        mock_groq_module.Groq = MagicMock(return_value=mock_sync_client)

        # Mock chat completions
        mock_async_client.chat.completions.create = AsyncMock()
        mock_sync_client.chat.completions.create = MagicMock()

        yield {
            "module": mock_groq_module,
            "async_client": mock_async_client,
            "sync_client": mock_sync_client,
        }


@pytest.fixture
def mock_mistral():
    """Mock Mistral SDK"""
    if "mistralai" not in sys.modules:
        mistral_module = types.ModuleType("mistralai")
        sys.modules["mistralai"] = mistral_module

    with patch(
        "chuk_llm.llm.providers.mistral_client.mistralai"
    ) as mock_mistral_module:
        mock_async_client = AsyncMock()
        mock_sync_client = MagicMock()

        mock_mistral_module.Mistral = MagicMock(return_value=mock_async_client)

        # Mock chat completions
        mock_async_client.chat.complete = AsyncMock()

        yield {
            "module": mock_mistral_module,
            "async_client": mock_async_client,
            "sync_client": mock_sync_client,
        }


@pytest.fixture
def mock_gemini():
    """Mock Google Gemini SDK"""
    if "google.generativeai" not in sys.modules:
        google_module = types.ModuleType("google")
        genai_module = types.ModuleType("google.generativeai")
        sys.modules["google"] = google_module
        sys.modules["google.generativeai"] = genai_module

    with patch("chuk_llm.llm.providers.gemini_client.genai") as mock_genai:
        mock_model = MagicMock()
        mock_genai.GenerativeModel = MagicMock(return_value=mock_model)
        mock_genai.configure = MagicMock()

        # Mock generation methods
        mock_model.generate_content = MagicMock()
        mock_model.generate_content_async = AsyncMock()

        yield {"module": mock_genai, "model": mock_model}


@pytest.fixture
def mock_ollama():
    """Mock Ollama SDK"""
    if "ollama" not in sys.modules:
        ollama_module = types.ModuleType("ollama")
        sys.modules["ollama"] = ollama_module

    with patch("chuk_llm.llm.providers.ollama_client.ollama") as mock_ollama_module:
        mock_async_client = AsyncMock()
        mock_sync_client = MagicMock()

        mock_ollama_module.AsyncClient = MagicMock(return_value=mock_async_client)
        mock_ollama_module.Client = MagicMock(return_value=mock_sync_client)

        # Mock chat methods
        mock_async_client.chat = AsyncMock()
        mock_sync_client.chat = MagicMock()

        yield {
            "module": mock_ollama_module,
            "async_client": mock_async_client,
            "sync_client": mock_sync_client,
        }


@pytest.fixture
def mock_watsonx():
    """Mock IBM Watson SDK"""
    if "ibm_watsonx_ai" not in sys.modules:
        watsonx_module = types.ModuleType("ibm_watsonx_ai")
        sys.modules["ibm_watsonx_ai"] = watsonx_module

    with patch("chuk_llm.llm.providers.watsonx_client.watsonx") as mock_watsonx_module:
        mock_client = MagicMock()
        mock_watsonx_module.APIClient = MagicMock(return_value=mock_client)

        # Mock generation methods
        mock_client.generate = MagicMock()

        yield {"module": mock_watsonx_module, "client": mock_client}


# ---------------------------------------------------------------------------
# Configuration Mocks
# ---------------------------------------------------------------------------


class MockFeature:
    """Mock Feature enum"""

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
    def from_string(cls, feature_str: str):
        return getattr(cls, feature_str.upper(), None)


class MockModelCapabilities:
    """Mock model capabilities"""

    def __init__(self, features=None, max_context_length=4096, max_output_tokens=2048):
        self.features = features or {MockFeature.TEXT, MockFeature.STREAMING}
        self.max_context_length = max_context_length
        self.max_output_tokens = max_output_tokens


class MockProviderConfig:
    """Mock provider configuration"""

    def __init__(self, name, client_class="MockClient", api_base=None, **kwargs):
        self.name = name
        self.client_class = client_class
        self.api_base = api_base
        self.models = kwargs.get("models", [f"{name}-model-1"])
        self.model_aliases = kwargs.get("model_aliases", {})
        self.rate_limits = kwargs.get("rate_limits", {"requests_per_minute": 60})
        self._capabilities = {}

    def get_model_capabilities(self, model):
        if model not in self._capabilities:
            features = {MockFeature.TEXT, MockFeature.STREAMING}
            if "advanced" in model or "gpt-4" in model:
                features.update({MockFeature.TOOLS, MockFeature.VISION})
            self._capabilities[model] = MockModelCapabilities(features=features)
        return self._capabilities[model]


class MockConfig:
    """Mock configuration object"""

    def __init__(self):
        self._providers = {}

    def get_provider(self, provider_name):
        return self._providers.get(provider_name)

    def add_provider(self, provider_config):
        self._providers[provider_config.name] = provider_config


@pytest.fixture
def mock_config():
    """Mock configuration with common providers"""
    config = MockConfig()

    providers = [
        MockProviderConfig(
            "openai",
            "OpenAILLMClient",
            "https://api.openai.com/v1",
            models=["gpt-4", "gpt-3.5-turbo"],
        ),
        MockProviderConfig(
            "anthropic",
            "AnthropicLLMClient",
            "https://api.anthropic.com",
            models=["claude-3-opus", "claude-3-sonnet"],
        ),
        MockProviderConfig(
            "groq",
            "GroqAILLMClient",
            "https://api.groq.com/openai/v1",
            models=["llama-3.3-70b-versatile"],
        ),
        MockProviderConfig(
            "mistral",
            "MistralLLMClient",
            "https://api.mistral.ai",
            models=["mistral-large"],
        ),
        MockProviderConfig(
            "gemini",
            "GeminiLLMClient",
            "https://generativelanguage.googleapis.com",
            models=["gemini-pro"],
        ),
        MockProviderConfig(
            "ollama", "OllamaLLMClient", "http://localhost:11434", models=["llama2"]
        ),
        MockProviderConfig(
            "watsonx",
            "WatsonxLLMClient",
            "https://us-south.ml.cloud.ibm.com",
            models=["llama2-70b-chat"],
        ),
    ]

    for provider in providers:
        config.add_provider(provider)

    return config


@pytest.fixture
def mock_configuration_system(mock_config):
    """Mock the entire configuration system"""
    with patch("chuk_llm.configuration.get_config", return_value=mock_config):
        with patch("chuk_llm.configuration.Feature", MockFeature):
            yield mock_config


# ---------------------------------------------------------------------------
# Common Test Utilities
# ---------------------------------------------------------------------------


class MockStreamChunk:
    """Mock streaming chunk for testing"""

    def __init__(self, content="", tool_calls=None, finish_reason=None):
        self.choices = [MockChoice(content, tool_calls, finish_reason)]


class MockChoice:
    """Mock choice for streaming chunks"""

    def __init__(self, content="", tool_calls=None, finish_reason=None):
        self.delta = MockDelta(content, tool_calls)
        self.finish_reason = finish_reason


class MockDelta:
    """Mock delta for streaming"""

    def __init__(self, content="", tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []


class MockMessage:
    """Mock message for API responses"""

    def __init__(self, content="", tool_calls=None, role="assistant"):
        self.content = content
        self.tool_calls = tool_calls or []
        self.role = role


class MockToolCall:
    """Mock tool call"""

    def __init__(self, id="call_123", name="test_function", arguments="{}"):
        self.id = id
        self.function = MockFunction(name, arguments)


class MockFunction:
    """Mock function in tool call"""

    def __init__(self, name="test_function", arguments="{}"):
        self.name = name
        self.arguments = arguments


@pytest.fixture
def mock_stream_chunks():
    """Provide common mock streaming chunks"""
    return [
        MockStreamChunk("Hello"),
        MockStreamChunk(" world"),
        MockStreamChunk("!", finish_reason="stop"),
    ]


@pytest.fixture
def mock_message_with_tools():
    """Provide mock message with tool calls"""
    tool_call = MockToolCall(name="get_weather", arguments='{"location": "NYC"}')
    return MockMessage("I'll check the weather", [tool_call])


@pytest.fixture
def sample_messages():
    """Provide sample conversation messages"""
    return [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you!"},
        {"role": "user", "content": "Can you help me with something?"},
    ]


@pytest.fixture
def sample_tools():
    """Provide sample tool definitions"""
    return [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"}
                    },
                    "required": ["location"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "Perform calculation",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Math expression",
                        }
                    },
                    "required": ["expression"],
                },
            },
        },
    ]


# ---------------------------------------------------------------------------
# Provider Client Mock Helpers
# ---------------------------------------------------------------------------


def mock_provider_client_methods(client, provider_name="test_provider"):
    """Helper to mock common provider client methods"""
    client.supports_feature = lambda feature: feature in [
        "streaming",
        "tools",
        "system_messages",
        "text",
    ]

    client.get_model_info = lambda: {
        "provider": provider_name,
        "model": getattr(client, "model", "test-model"),
        "client_class": f"{provider_name.title()}LLMClient",
        "api_base": f"https://api.{provider_name}.com",
        "features": ["text", "streaming", "tools"],
        "supports_text": True,
        "supports_streaming": True,
        "supports_tools": True,
        "supports_vision": False,
        "supports_system_messages": True,
        "max_context_length": 4096,
        "max_output_tokens": 2048,
        f"{provider_name}_specific": {"feature_1": True, "feature_2": "enabled"},
    }

    # Mock token limits
    client.get_max_tokens_limit = lambda: 2048
    client.get_context_length_limit = lambda: 4096

    # Mock parameter validation
    original_validate = getattr(client, "validate_parameters", lambda **kwargs: kwargs)
    client.validate_parameters = lambda **kwargs: original_validate(**kwargs)

    return client


@pytest.fixture
def mock_async_stream():
    """Mock async stream for testing"""

    async def stream_generator(chunks):
        for chunk in chunks:
            yield chunk

    return stream_generator


# ---------------------------------------------------------------------------
# Error Simulation Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def simulate_api_error():
    """Helper to simulate API errors"""

    class APIError(Exception):
        def __init__(self, message, status_code=500):
            self.message = message
            self.status_code = status_code
            super().__init__(message)

    return APIError


@pytest.fixture
def simulate_timeout():
    """Helper to simulate timeout errors"""

    return TimeoutError("Request timed out")


# ---------------------------------------------------------------------------
# Test Environment Setup
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Automatically setup test environment for all tests"""
    # Ensure all required modules are available
    required_modules = [
        "openai",
        "anthropic",
        "groq",
        "mistralai",
        "google.generativeai",
        "ollama",
        "ibm_watsonx_ai",
    ]

    for module_name in required_modules:
        if module_name not in sys.modules:
            # Create minimal module structure
            parts = module_name.split(".")
            for i in range(len(parts)):
                partial_name = ".".join(parts[: i + 1])
                if partial_name not in sys.modules:
                    sys.modules[partial_name] = types.ModuleType(partial_name)

    yield

    # Cleanup is handled by pytest automatically


# ---------------------------------------------------------------------------
# Pytest Configuration
# ---------------------------------------------------------------------------


def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "provider: mark test as provider-specific")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers"""
    for item in items:
        # Add provider marker based on test file location
        if "providers" in str(item.fspath):
            item.add_marker(pytest.mark.provider)

        # Add slow marker for tests that might be slow
        if "integration" in item.name or "full_workflow" in item.name:
            item.add_marker(pytest.mark.slow)
