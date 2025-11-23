"""
Comprehensive tests for all capability resolvers.
"""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from chuk_llm.registry.models import ModelCapabilities, ModelSpec, QualityTier


class TestBaseCapabilityResolver:
    """Test base resolver functionality."""

    def test_empty_capabilities(self):
        """Test _empty_capabilities helper."""
        from chuk_llm.registry.resolvers.base import BaseCapabilityResolver

        class TestResolver(BaseCapabilityResolver):
            async def get_capabilities(self, spec: ModelSpec) -> ModelCapabilities:
                return self._empty_capabilities()

        resolver = TestResolver()
        caps = resolver._empty_capabilities()
        assert isinstance(caps, ModelCapabilities)
        assert caps.max_context is None

    def test_partial_capabilities(self):
        """Test _partial_capabilities helper."""
        from chuk_llm.registry.resolvers.base import BaseCapabilityResolver

        class TestResolver(BaseCapabilityResolver):
            async def get_capabilities(self, spec: ModelSpec) -> ModelCapabilities:
                return self._partial_capabilities(max_context=100_000, supports_tools=True)

        resolver = TestResolver()
        caps = resolver._partial_capabilities(max_context=100_000, supports_tools=True)
        assert caps.max_context == 100_000
        assert caps.supports_tools is True


class TestHeuristicCapabilityResolver:
    """Test heuristic-based capability resolver."""

    @pytest.mark.asyncio
    async def test_best_tier_gpt4o(self):
        """Test quality tier inference for GPT-4o."""
        from chuk_llm.registry.resolvers.heuristic import HeuristicCapabilityResolver

        resolver = HeuristicCapabilityResolver()
        spec = ModelSpec(provider="openai", name="gpt-4o")
        caps = await resolver.get_capabilities(spec)

        assert caps.quality_tier == QualityTier.BEST
        assert caps.max_context == 128_000

    @pytest.mark.asyncio
    async def test_best_tier_claude_opus(self):
        """Test quality tier inference for Claude Opus."""
        from chuk_llm.registry.resolvers.heuristic import HeuristicCapabilityResolver

        resolver = HeuristicCapabilityResolver()
        spec = ModelSpec(provider="anthropic", name="claude-3-opus-20240229")
        caps = await resolver.get_capabilities(spec)

        assert caps.quality_tier == QualityTier.BEST
        assert caps.max_context == 200_000

    @pytest.mark.asyncio
    async def test_best_tier_gemini_pro(self):
        """Test quality tier for Gemini Pro."""
        from chuk_llm.registry.resolvers.heuristic import HeuristicCapabilityResolver

        resolver = HeuristicCapabilityResolver()
        spec = ModelSpec(provider="gemini", name="gemini-1.5-pro")
        caps = await resolver.get_capabilities(spec)

        assert caps.quality_tier == QualityTier.BEST
        assert caps.max_context == 2_000_000

    @pytest.mark.asyncio
    async def test_cheap_tier_mini(self):
        """Test quality tier inference for mini models."""
        from chuk_llm.registry.resolvers.heuristic import HeuristicCapabilityResolver

        resolver = HeuristicCapabilityResolver()
        # CHEAP patterns are checked first, so "mini" should win over "gpt-4o"
        spec = ModelSpec(provider="openai", name="gpt-4o-mini")
        caps = await resolver.get_capabilities(spec)

        assert caps.quality_tier == QualityTier.CHEAP

    @pytest.mark.asyncio
    async def test_cheap_tier_haiku(self):
        """Test quality tier for Claude Haiku."""
        from chuk_llm.registry.resolvers.heuristic import HeuristicCapabilityResolver

        resolver = HeuristicCapabilityResolver()
        spec = ModelSpec(provider="anthropic", name="claude-3-haiku-20240307")
        caps = await resolver.get_capabilities(spec)

        assert caps.quality_tier == QualityTier.CHEAP

    @pytest.mark.asyncio
    async def test_cheap_tier_flash(self):
        """Test quality tier for Gemini Flash."""
        from chuk_llm.registry.resolvers.heuristic import HeuristicCapabilityResolver

        resolver = HeuristicCapabilityResolver()
        spec = ModelSpec(provider="gemini", name="gemini-1.5-flash")
        caps = await resolver.get_capabilities(spec)

        assert caps.quality_tier == QualityTier.CHEAP

    @pytest.mark.asyncio
    async def test_balanced_tier_gpt4(self):
        """Test quality tier for GPT-4."""
        from chuk_llm.registry.resolvers.heuristic import HeuristicCapabilityResolver

        resolver = HeuristicCapabilityResolver()
        spec = ModelSpec(provider="openai", name="gpt-4-turbo")
        caps = await resolver.get_capabilities(spec)

        assert caps.quality_tier == QualityTier.BALANCED

    @pytest.mark.asyncio
    async def test_balanced_tier_claude_sonnet(self):
        """Test quality tier for Claude Sonnet."""
        from chuk_llm.registry.resolvers.heuristic import HeuristicCapabilityResolver

        resolver = HeuristicCapabilityResolver()
        spec = ModelSpec(provider="anthropic", name="claude-3-sonnet-20240229")
        caps = await resolver.get_capabilities(spec)

        assert caps.quality_tier == QualityTier.BALANCED

    @pytest.mark.asyncio
    async def test_balanced_tier_deepseek(self):
        """Test quality tier for DeepSeek."""
        from chuk_llm.registry.resolvers.heuristic import HeuristicCapabilityResolver

        resolver = HeuristicCapabilityResolver()
        spec = ModelSpec(provider="deepseek", name="deepseek-chat")
        caps = await resolver.get_capabilities(spec)

        assert caps.quality_tier == QualityTier.BALANCED

    @pytest.mark.asyncio
    async def test_unknown_tier(self):
        """Test unknown quality tier for unrecognized models."""
        from chuk_llm.registry.resolvers.heuristic import HeuristicCapabilityResolver

        resolver = HeuristicCapabilityResolver()
        spec = ModelSpec(provider="unknown", name="random-model-xyz")
        caps = await resolver.get_capabilities(spec)

        assert caps.quality_tier == QualityTier.UNKNOWN

    @pytest.mark.asyncio
    async def test_context_from_name_128k(self):
        """Test context inference from model name (128k)."""
        from chuk_llm.registry.resolvers.heuristic import HeuristicCapabilityResolver

        resolver = HeuristicCapabilityResolver()
        spec = ModelSpec(provider="test", name="model-128k")
        caps = await resolver.get_capabilities(spec)

        assert caps.max_context == 128_000

    @pytest.mark.asyncio
    async def test_context_from_name_200k(self):
        """Test context inference from model name (200k)."""
        from chuk_llm.registry.resolvers.heuristic import HeuristicCapabilityResolver

        resolver = HeuristicCapabilityResolver()
        spec = ModelSpec(provider="test", name="model-200k")
        caps = await resolver.get_capabilities(spec)

        assert caps.max_context == 200_000

    @pytest.mark.asyncio
    async def test_context_from_name_1m(self):
        """Test context inference from model name (1M)."""
        from chuk_llm.registry.resolvers.heuristic import HeuristicCapabilityResolver

        resolver = HeuristicCapabilityResolver()
        spec = ModelSpec(provider="test", name="model-1m")
        caps = await resolver.get_capabilities(spec)

        assert caps.max_context == 1_000_000

    @pytest.mark.asyncio
    async def test_context_from_name_2m(self):
        """Test context inference from model name (2M)."""
        from chuk_llm.registry.resolvers.heuristic import HeuristicCapabilityResolver

        resolver = HeuristicCapabilityResolver()
        spec = ModelSpec(provider="test", name="model-2m")
        caps = await resolver.get_capabilities(spec)

        assert caps.max_context == 2_000_000

    @pytest.mark.asyncio
    async def test_openai_gpt35_context(self):
        """Test OpenAI GPT-3.5 context."""
        from chuk_llm.registry.resolvers.heuristic import HeuristicCapabilityResolver

        resolver = HeuristicCapabilityResolver()
        spec = ModelSpec(provider="openai", name="gpt-3.5-turbo")
        caps = await resolver.get_capabilities(spec)

        assert caps.max_context == 16_385

    @pytest.mark.asyncio
    async def test_groq_default_context(self):
        """Test Groq default context."""
        from chuk_llm.registry.resolvers.heuristic import HeuristicCapabilityResolver

        resolver = HeuristicCapabilityResolver()
        spec = ModelSpec(provider="groq", name="llama-3-70b")
        caps = await resolver.get_capabilities(spec)

        assert caps.max_context == 128_000

    @pytest.mark.asyncio
    async def test_gemini_2_context(self):
        """Test Gemini 2 context."""
        from chuk_llm.registry.resolvers.heuristic import HeuristicCapabilityResolver

        resolver = HeuristicCapabilityResolver()
        spec = ModelSpec(provider="gemini", name="gemini-2.0-flash")
        caps = await resolver.get_capabilities(spec)

        assert caps.max_context == 1_000_000

    @pytest.mark.asyncio
    async def test_default_capabilities(self):
        """Test default capabilities are conservative."""
        from chuk_llm.registry.resolvers.heuristic import HeuristicCapabilityResolver

        resolver = HeuristicCapabilityResolver()
        spec = ModelSpec(provider="test", name="test-model")
        caps = await resolver.get_capabilities(spec)

        # Conservative defaults
        assert caps.supports_tools is False
        assert caps.supports_vision is False
        assert caps.supports_json_mode is False
        assert caps.supports_structured_outputs is False
        # Safe assumptions
        assert caps.supports_streaming is True
        assert caps.supports_system_messages is True


class TestYamlCapabilityResolver:
    """Test YAML cache resolver."""

    @pytest.fixture
    def temp_cache_dir(self, tmp_path):
        """Create temporary cache directory with test YAML files."""
        cache_dir = tmp_path / "capabilities"
        cache_dir.mkdir()

        # Create test YAML files
        openai_cache = {
            "models": {
                "gpt-4o": {
                    "max_context": 128_000,
                    "supports_tools": True,
                    "supports_vision": True,
                    "quality_tier": "best",
                },
                "gpt-4o-mini": {
                    "max_context": 128_000,
                    "supports_tools": True,
                    "quality_tier": "cheap",
                },
            },
            "families": {
                "gpt-4o": {
                    "max_context": 128_000,
                    "supports_tools": True,
                }
            }
        }

        anthropic_cache = {
            "models": {
                "claude-3-5-sonnet-20241022": {
                    "max_context": 200_000,
                    "supports_tools": True,
                    "supports_vision": True,
                    "quality_tier": "best",
                    "known_params": ["temperature", "max_tokens", "top_p"],
                }
            }
        }

        import yaml
        with open(cache_dir / "openai.yaml", "w") as f:
            yaml.dump(openai_cache, f)

        with open(cache_dir / "anthropic.yaml", "w") as f:
            yaml.dump(anthropic_cache, f)

        return cache_dir

    @pytest.mark.asyncio
    async def test_load_from_yaml_cache(self, temp_cache_dir):
        """Test loading capabilities from YAML cache."""
        from chuk_llm.registry.resolvers.yaml_config import YamlCapabilityResolver

        resolver = YamlCapabilityResolver(cache_dir=temp_cache_dir)
        spec = ModelSpec(provider="openai", name="gpt-4o")
        caps = await resolver.get_capabilities(spec)

        assert caps.max_context == 128_000
        assert caps.supports_tools is True
        assert caps.supports_vision is True
        assert caps.quality_tier == QualityTier.BEST

    @pytest.mark.asyncio
    async def test_known_params_as_set(self, temp_cache_dir):
        """Test that known_params is converted from list to set."""
        from chuk_llm.registry.resolvers.yaml_config import YamlCapabilityResolver

        resolver = YamlCapabilityResolver(cache_dir=temp_cache_dir)
        spec = ModelSpec(provider="anthropic", name="claude-3-5-sonnet-20241022")
        caps = await resolver.get_capabilities(spec)

        assert isinstance(caps.known_params, set)
        assert "temperature" in caps.known_params
        assert len(caps.known_params) == 3

    @pytest.mark.asyncio
    async def test_provider_not_in_cache(self, temp_cache_dir):
        """Test provider not in cache returns empty."""
        from chuk_llm.registry.resolvers.yaml_config import YamlCapabilityResolver

        resolver = YamlCapabilityResolver(cache_dir=temp_cache_dir)
        spec = ModelSpec(provider="unknown", name="test-model")
        caps = await resolver.get_capabilities(spec)

        assert caps.max_context is None

    @pytest.mark.asyncio
    async def test_model_not_in_cache(self, temp_cache_dir):
        """Test model not in cache returns empty."""
        from chuk_llm.registry.resolvers.yaml_config import YamlCapabilityResolver

        resolver = YamlCapabilityResolver(cache_dir=temp_cache_dir)
        spec = ModelSpec(provider="openai", name="nonexistent-model")
        caps = await resolver.get_capabilities(spec)

        assert caps.max_context is None

    @pytest.mark.asyncio
    async def test_family_fallback(self, temp_cache_dir):
        """Test fallback to family defaults."""
        from chuk_llm.registry.resolvers.yaml_config import YamlCapabilityResolver

        resolver = YamlCapabilityResolver(cache_dir=temp_cache_dir)
        spec = ModelSpec(provider="openai", name="gpt-4o-unknown", family="gpt-4o")
        caps = await resolver.get_capabilities(spec)

        assert caps.max_context == 128_000
        assert caps.supports_tools is True

    @pytest.mark.asyncio
    async def test_nonexistent_cache_dir(self):
        """Test resolver with nonexistent cache directory."""
        from chuk_llm.registry.resolvers.yaml_config import YamlCapabilityResolver

        resolver = YamlCapabilityResolver(cache_dir=Path("/nonexistent/path"))
        spec = ModelSpec(provider="openai", name="gpt-4o")
        caps = await resolver.get_capabilities(spec)

        # Should return empty capabilities
        assert caps.max_context is None

    @pytest.mark.asyncio
    async def test_invalid_yaml_file(self, tmp_path):
        """Test handling of invalid YAML files."""
        from chuk_llm.registry.resolvers.yaml_config import YamlCapabilityResolver

        cache_dir = tmp_path / "capabilities"
        cache_dir.mkdir()

        # Create invalid YAML file
        with open(cache_dir / "bad.yaml", "w") as f:
            f.write("invalid: yaml: content: {{")

        resolver = YamlCapabilityResolver(cache_dir=cache_dir)
        spec = ModelSpec(provider="bad", name="test-model")
        caps = await resolver.get_capabilities(spec)

        # Should return empty capabilities
        assert caps.max_context is None

    @pytest.mark.asyncio
    async def test_invalid_capability_data(self, tmp_path):
        """Test handling of invalid capability data in YAML."""
        from chuk_llm.registry.resolvers.yaml_config import YamlCapabilityResolver
        import yaml

        cache_dir = tmp_path / "capabilities"
        cache_dir.mkdir()

        # Create YAML with invalid data type
        bad_cache = {
            "models": {
                "test-model": {
                    "max_context": "not-a-number",  # Invalid type
                }
            }
        }

        with open(cache_dir / "test.yaml", "w") as f:
            yaml.dump(bad_cache, f)

        resolver = YamlCapabilityResolver(cache_dir=cache_dir)
        spec = ModelSpec(provider="test", name="test-model")
        caps = await resolver.get_capabilities(spec)

        # Should return empty capabilities due to validation error
        assert caps.max_context is None

    @pytest.mark.asyncio
    async def test_default_cache_dir(self):
        """Test using default package cache directory."""
        from chuk_llm.registry.resolvers.yaml_config import YamlCapabilityResolver

        # Create resolver with default cache_dir (None)
        resolver = YamlCapabilityResolver(cache_dir=None)

        # Should use package capabilities directory
        assert "capabilities" in str(resolver.cache_dir)

    @pytest.mark.asyncio
    async def test_model_inheritance(self, tmp_path):
        """Test model inheritance from family."""
        from chuk_llm.registry.resolvers.yaml_config import YamlCapabilityResolver
        import yaml

        cache_dir = tmp_path / "capabilities"
        cache_dir.mkdir()

        # Create YAML with inheritance
        cache_data = {
            "models": {
                "base-model": {
                    "max_context": 100_000,
                    "supports_tools": True,
                    "inherits_from": "test-family",
                }
            }
        }

        with open(cache_dir / "test.yaml", "w") as f:
            yaml.dump(cache_data, f)

        resolver = YamlCapabilityResolver(cache_dir=cache_dir)

        # Look for a model in the same family
        spec = ModelSpec(provider="test", name="new-model", family="test-family")
        caps = await resolver.get_capabilities(spec)

        # Should inherit from base-model
        assert caps.max_context == 100_000
        assert caps.supports_tools is True


class TestRuntimeTestingResolver:
    """Test runtime testing resolver."""

    @pytest.mark.asyncio
    async def test_disabled_by_default(self):
        """Test that runtime testing is disabled by default."""
        from chuk_llm.registry.resolvers.runtime import RuntimeTestingResolver

        resolver = RuntimeTestingResolver()
        assert resolver.enabled is False

        spec = ModelSpec(provider="openai", name="gpt-4o")
        caps = await resolver.get_capabilities(spec)

        assert caps.source == "runtime_resolver_disabled"

    @pytest.mark.asyncio
    async def test_enabled_explicitly(self):
        """Test enabling runtime testing explicitly."""
        from chuk_llm.registry.resolvers.runtime import RuntimeTestingResolver

        resolver = RuntimeTestingResolver(enabled=True)
        assert resolver.enabled is True

    @pytest.mark.asyncio
    async def test_env_var_enable(self, monkeypatch):
        """Test enabling via environment variable."""
        from chuk_llm.registry.resolvers.runtime import RuntimeTestingResolver

        monkeypatch.setenv("CHUK_LLM_RUNTIME_TESTING", "true")
        resolver = RuntimeTestingResolver()
        assert resolver.enabled is True

        monkeypatch.setenv("CHUK_LLM_RUNTIME_TESTING", "1")
        resolver = RuntimeTestingResolver()
        assert resolver.enabled is True

        monkeypatch.setenv("CHUK_LLM_RUNTIME_TESTING", "yes")
        resolver = RuntimeTestingResolver()
        assert resolver.enabled is True

    @pytest.mark.asyncio
    async def test_env_var_disable(self, monkeypatch):
        """Test disabling via environment variable."""
        from chuk_llm.registry.resolvers.runtime import RuntimeTestingResolver

        monkeypatch.setenv("CHUK_LLM_RUNTIME_TESTING", "false")
        resolver = RuntimeTestingResolver()
        assert resolver.enabled is False

    @pytest.mark.asyncio
    async def test_runtime_test_failure(self):
        """Test handling of runtime test failures."""
        from chuk_llm.registry.resolvers.runtime import RuntimeTestingResolver
        from unittest.mock import AsyncMock, patch

        resolver = RuntimeTestingResolver(enabled=True)

        # Mock RuntimeCapabilityTester to raise an exception
        with patch("chuk_llm.registry.resolvers.runtime.RuntimeCapabilityTester") as MockTester:
            mock_tester = AsyncMock()
            mock_tester.test_model = AsyncMock(side_effect=Exception("Test failed"))
            MockTester.return_value = mock_tester

            spec = ModelSpec(provider="openai", name="gpt-4o")
            caps = await resolver.get_capabilities(spec)

            assert caps.source == "runtime_test_failed"

    @pytest.mark.asyncio
    async def test_runtime_test_success(self):
        """Test successful runtime testing."""
        from chuk_llm.registry.resolvers.runtime import RuntimeTestingResolver
        from unittest.mock import AsyncMock, patch

        resolver = RuntimeTestingResolver(enabled=True)

        # Mock RuntimeCapabilityTester to return test capabilities
        test_caps = ModelCapabilities(
            max_context=128_000,
            supports_tools=True,
            supports_vision=True,
        )

        with patch("chuk_llm.registry.resolvers.runtime.RuntimeCapabilityTester") as MockTester:
            mock_tester = AsyncMock()
            mock_tester.test_model = AsyncMock(return_value=test_caps)
            MockTester.return_value = mock_tester

            spec = ModelSpec(provider="openai", name="gpt-4o")
            caps = await resolver.get_capabilities(spec)

            assert caps.max_context == 128_000
            assert caps.supports_tools is True
            assert caps.supports_vision is True


class TestOllamaCapabilityResolver:
    """Test Ollama capability resolver."""

    @pytest.mark.asyncio
    async def test_non_ollama_provider(self):
        """Test that non-Ollama providers return empty capabilities."""
        from chuk_llm.registry.resolvers.ollama import OllamaCapabilityResolver

        resolver = OllamaCapabilityResolver()
        spec = ModelSpec(provider="openai", name="gpt-4o")
        caps = await resolver.get_capabilities(spec)

        assert caps.max_context is None

    @pytest.mark.asyncio
    async def test_ollama_api_unavailable(self):
        """Test handling when Ollama API is unavailable."""
        from chuk_llm.registry.resolvers.ollama import OllamaCapabilityResolver

        resolver = OllamaCapabilityResolver(base_url="http://localhost:99999")
        spec = ModelSpec(provider="ollama", name="llama3.2")
        caps = await resolver.get_capabilities(spec)

        # Should return empty capabilities
        assert caps.max_context is None

    @pytest.mark.asyncio
    async def test_ollama_successful_parse(self):
        """Test successful parsing of Ollama metadata."""
        from chuk_llm.registry.resolvers.ollama import OllamaCapabilityResolver
        from unittest.mock import AsyncMock, MagicMock, patch

        resolver = OllamaCapabilityResolver()

        # Mock response from Ollama API
        mock_response_data = {
            "model_info": {
                "llama.context_length": 8192,
            },
            "details": {
                "families": ["llava"],
                "parameter_size": "70B",
            },
            "template": "{{.Tools}}",
        }

        async def mock_post(*args, **kwargs):
            """Mock async post method."""
            mock_resp = MagicMock()
            mock_resp.json = MagicMock(return_value=mock_response_data)
            mock_resp.raise_for_status = MagicMock()
            return mock_resp

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = MagicMock()
            mock_client.post = mock_post
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client

            spec = ModelSpec(provider="ollama", name="llama3.2-vision:70b")
            caps = await resolver.get_capabilities(spec)

            assert caps.max_context == 8192
            assert caps.supports_vision is True
            assert caps.supports_tools is True
            assert caps.quality_tier == QualityTier.BEST

    @pytest.mark.asyncio
    async def test_custom_base_url(self, monkeypatch):
        """Test custom base URL configuration."""
        from chuk_llm.registry.resolvers.ollama import OllamaCapabilityResolver

        monkeypatch.setenv("OLLAMA_BASE_URL", "http://custom:11434")
        resolver = OllamaCapabilityResolver()
        assert resolver.base_url == "http://custom:11434"

    @pytest.mark.asyncio
    async def test_parse_context_from_parameters(self):
        """Test parsing context length from parameters."""
        from chuk_llm.registry.resolvers.ollama import OllamaCapabilityResolver

        resolver = OllamaCapabilityResolver()

        # Mock data with num_ctx in parameters
        mock_data = {
            "model_info": {},
            "details": {},
            "parameters": {"num_ctx": 4096},
            "template": "",
        }

        caps = resolver._parse_ollama_metadata(mock_data, ModelSpec(provider="ollama", name="test"))
        assert caps.max_context == 4096

    @pytest.mark.asyncio
    async def test_parse_quality_tier_from_size(self):
        """Test quality tier inference from model size."""
        from chuk_llm.registry.resolvers.ollama import OllamaCapabilityResolver

        resolver = OllamaCapabilityResolver()

        # Test 70B model (BEST tier)
        mock_data = {
            "model_info": {},
            "details": {"parameter_size": "70B"},
            "template": "",
        }
        caps = resolver._parse_ollama_metadata(mock_data, ModelSpec(provider="ollama", name="test"))
        assert caps.quality_tier == QualityTier.BEST

        # Test 7B model (CHEAP tier)
        mock_data["details"]["parameter_size"] = "7B"
        caps = resolver._parse_ollama_metadata(mock_data, ModelSpec(provider="ollama", name="test"))
        assert caps.quality_tier == QualityTier.CHEAP

        # Test 34B model (BALANCED tier)
        mock_data["details"]["parameter_size"] = "34B"
        caps = resolver._parse_ollama_metadata(mock_data, ModelSpec(provider="ollama", name="test"))
        assert caps.quality_tier == QualityTier.BALANCED


class TestGeminiCapabilityResolver:
    """Test Gemini capability resolver."""

    @pytest.mark.asyncio
    async def test_non_gemini_provider(self):
        """Test that non-Gemini providers return empty capabilities."""
        from chuk_llm.registry.resolvers.gemini import GeminiCapabilityResolver

        resolver = GeminiCapabilityResolver(api_key="test-key")
        spec = ModelSpec(provider="openai", name="gpt-4o")
        caps = await resolver.get_capabilities(spec)

        assert caps.max_context is None

    @pytest.mark.asyncio
    async def test_no_api_key(self):
        """Test behavior when API key is missing."""
        from chuk_llm.registry.resolvers.gemini import GeminiCapabilityResolver

        resolver = GeminiCapabilityResolver(api_key=None)
        spec = ModelSpec(provider="gemini", name="gemini-1.5-pro")
        caps = await resolver.get_capabilities(spec)

        assert caps.max_context is None

    @pytest.mark.asyncio
    async def test_api_key_from_env(self, monkeypatch):
        """Test API key loading from environment."""
        from chuk_llm.registry.resolvers.gemini import GeminiCapabilityResolver

        monkeypatch.setenv("GEMINI_API_KEY", "test-key-123")
        resolver = GeminiCapabilityResolver()
        assert resolver.api_key == "test-key-123"

        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setenv("GOOGLE_API_KEY", "google-key-456")
        resolver = GeminiCapabilityResolver()
        assert resolver.api_key == "google-key-456"

    @pytest.mark.asyncio
    async def test_successful_gemini_parse(self):
        """Test successful parsing of Gemini metadata."""
        from chuk_llm.registry.resolvers.gemini import GeminiCapabilityResolver
        from unittest.mock import AsyncMock, MagicMock, patch

        resolver = GeminiCapabilityResolver(api_key="test-key")

        # Mock response from Gemini API
        mock_response_data = {
            "supportedGenerationMethods": ["generateContent", "streamGenerateContent"],
            "inputTokenLimit": 2_000_000,
            "outputTokenLimit": 8_192,
        }

        async def mock_get(*args, **kwargs):
            """Mock async get method."""
            mock_resp = MagicMock()
            mock_resp.json = MagicMock(return_value=mock_response_data)
            mock_resp.raise_for_status = MagicMock()
            return mock_resp

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = MagicMock()
            mock_client.get = mock_get
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client

            spec = ModelSpec(provider="gemini", name="gemini-1.5-pro")
            caps = await resolver.get_capabilities(spec)

            assert caps.max_context == 2_000_000
            assert caps.max_output_tokens == 8_192
            assert caps.supports_tools is True
            assert caps.supports_vision is True
            assert caps.quality_tier == QualityTier.BEST

    @pytest.mark.asyncio
    async def test_gemini_caching(self):
        """Test that Gemini responses are cached."""
        from chuk_llm.registry.resolvers.gemini import GeminiCapabilityResolver
        from unittest.mock import AsyncMock, MagicMock, patch

        resolver = GeminiCapabilityResolver(api_key="test-key")

        mock_response_data = {
            "supportedGenerationMethods": ["generateContent"],
            "inputTokenLimit": 1_000_000,
        }

        call_count = 0

        async def mock_get(*args, **kwargs):
            """Mock async get method that tracks calls."""
            nonlocal call_count
            call_count += 1
            mock_resp = MagicMock()
            mock_resp.json = MagicMock(return_value=mock_response_data)
            mock_resp.raise_for_status = MagicMock()
            return mock_resp

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = MagicMock()
            mock_client.get = mock_get
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client

            spec = ModelSpec(provider="gemini", name="gemini-2.0-flash")

            # First call
            caps1 = await resolver.get_capabilities(spec)
            # Second call (should be cached)
            caps2 = await resolver.get_capabilities(spec)

            assert caps1 == caps2
            # API should only be called once
            assert call_count == 1

    @pytest.mark.asyncio
    async def test_quality_tier_inference(self):
        """Test quality tier inference for Gemini models."""
        from chuk_llm.registry.resolvers.gemini import GeminiCapabilityResolver

        resolver = GeminiCapabilityResolver(api_key="test-key")

        # Pro models are BEST
        assert resolver._infer_quality_tier("gemini-1.5-pro") == QualityTier.BEST
        # Flash models are CHEAP
        assert resolver._infer_quality_tier("gemini-2.0-flash") == QualityTier.CHEAP
        # Others are BALANCED
        assert resolver._infer_quality_tier("gemini-unknown") == QualityTier.BALANCED

    @pytest.mark.asyncio
    async def test_gemini_api_failure(self):
        """Test handling of API failures."""
        from chuk_llm.registry.resolvers.gemini import GeminiCapabilityResolver
        from unittest.mock import AsyncMock, MagicMock, patch
        import httpx

        resolver = GeminiCapabilityResolver(api_key="test-key")

        async def mock_get_error(*args, **kwargs):
            """Mock async get that raises an httpx error."""
            raise httpx.HTTPError("API Error")

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = MagicMock()
            mock_client.get = mock_get_error
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client

            spec = ModelSpec(provider="gemini", name="gemini-1.5-pro")
            caps = await resolver.get_capabilities(spec)

            # Should return empty capabilities
            assert caps.max_context is None
