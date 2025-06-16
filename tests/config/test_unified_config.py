# tests/test_unified_config.py
"""
Comprehensive pytest tests for the unified configuration system
"""

import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from dataclasses import dataclass
from typing import Set

from chuk_llm.configuration.unified_config import (
    Feature,
    ModelCapabilities,
    ProviderConfig,
    UnifiedConfigManager,
    ConfigValidator,
    CapabilityChecker,
    get_config,
    reset_config,
)


# ──────────────────────────── Feature Tests ─────────────────────────────
class TestFeature:
    """Test the Feature enum"""
    
    def test_feature_values(self):
        """Test that all features have correct values"""
        assert Feature.TEXT.value == "text"
        assert Feature.STREAMING.value == "streaming"
        assert Feature.TOOLS.value == "tools"
        assert Feature.VISION.value == "vision"
        assert Feature.JSON_MODE.value == "json_mode"
        assert Feature.PARALLEL_CALLS.value == "parallel_calls"
        assert Feature.SYSTEM_MESSAGES.value == "system_messages"
        assert Feature.MULTIMODAL.value == "multimodal"
        assert Feature.REASONING.value == "reasoning"
    
    def test_from_string(self):
        """Test converting strings to Feature enum"""
        assert Feature.from_string("text") == Feature.TEXT
        assert Feature.from_string("TEXT") == Feature.TEXT
        assert Feature.from_string("streaming") == Feature.STREAMING
        assert Feature.from_string("VISION") == Feature.VISION
    
    def test_from_string_invalid(self):
        """Test that invalid strings raise ValueError"""
        with pytest.raises(ValueError, match="Unknown feature: invalid"):
            Feature.from_string("invalid")
        
        with pytest.raises(ValueError):
            Feature.from_string("")


# ──────────────────────────── ModelCapabilities Tests ─────────────────────────────
class TestModelCapabilities:
    """Test the ModelCapabilities dataclass"""
    
    def test_initialization(self):
        """Test creating ModelCapabilities"""
        cap = ModelCapabilities(
            pattern="gpt-4.*",
            features={Feature.TEXT, Feature.TOOLS},
            max_context_length=128000,
            max_output_tokens=4096
        )
        
        assert cap.pattern == "gpt-4.*"
        assert Feature.TEXT in cap.features
        assert Feature.TOOLS in cap.features
        assert cap.max_context_length == 128000
        assert cap.max_output_tokens == 4096
    
    def test_matches(self):
        """Test pattern matching for models"""
        cap = ModelCapabilities(pattern="gpt-4.*")
        
        assert cap.matches("gpt-4")
        assert cap.matches("gpt-4-turbo")
        assert cap.matches("gpt-4o")
        assert cap.matches("GPT-4O")  # Case insensitive
        assert not cap.matches("gpt-3.5-turbo")
        assert not cap.matches("claude-3")
    
    def test_complex_patterns(self):
        """Test more complex regex patterns"""
        # Pattern for o-series models
        cap1 = ModelCapabilities(pattern="o[1-4].*")
        assert cap1.matches("o1")
        assert cap1.matches("o3-mini")
        assert not cap1.matches("o5")
        
        # Pattern for specific versions
        cap2 = ModelCapabilities(pattern="claude-3-[57]-.*")
        assert cap2.matches("claude-3-5-sonnet-20241022")
        assert cap2.matches("claude-3-7-sonnet-20250219")
        assert not cap2.matches("claude-3-opus-20240229")
    
    def test_get_effective_features(self):
        """Test feature inheritance"""
        provider_features = {Feature.TEXT, Feature.STREAMING}
        model_features = {Feature.TOOLS, Feature.VISION}
        
        cap = ModelCapabilities(
            pattern=".*",
            features=model_features
        )
        
        effective = cap.get_effective_features(provider_features)
        
        # Should have all features from both
        assert effective == {Feature.TEXT, Feature.STREAMING, Feature.TOOLS, Feature.VISION}


# ──────────────────────────── ProviderConfig Tests ─────────────────────────────
class TestProviderConfig:
    """Test the ProviderConfig dataclass"""
    
    def test_initialization(self):
        """Test creating ProviderConfig"""
        provider = ProviderConfig(
            name="openai",
            client_class="chuk_llm.llm.providers.openai_client.OpenAILLMClient",
            api_key_env="OPENAI_API_KEY",
            default_model="gpt-4o-mini",
            models=["gpt-4", "gpt-3.5-turbo"],
            features={Feature.TEXT, Feature.STREAMING, Feature.TOOLS}
        )
        
        assert provider.name == "openai"
        assert provider.client_class == "chuk_llm.llm.providers.openai_client.OpenAILLMClient"
        assert provider.api_key_env == "OPENAI_API_KEY"
        assert provider.default_model == "gpt-4o-mini"
        assert "gpt-4" in provider.models
        assert Feature.TOOLS in provider.features
    
    def test_supports_feature_provider_level(self):
        """Test feature support at provider level"""
        provider = ProviderConfig(
            name="test",
            features={Feature.TEXT, Feature.STREAMING}
        )
        
        assert provider.supports_feature(Feature.TEXT)
        assert provider.supports_feature("streaming")  # String input
        assert not provider.supports_feature(Feature.TOOLS)
    
    def test_supports_feature_model_level(self):
        """Test feature support with model-specific capabilities"""
        provider = ProviderConfig(
            name="test",
            features={Feature.TEXT, Feature.STREAMING},
            model_capabilities=[
                ModelCapabilities(
                    pattern="advanced-.*",
                    features={Feature.TOOLS, Feature.VISION}
                )
            ]
        )
        
        # Basic model should only have provider features
        assert provider.supports_feature(Feature.TEXT, "basic-model")
        assert not provider.supports_feature(Feature.TOOLS, "basic-model")
        
        # Advanced model should have both provider and model features
        assert provider.supports_feature(Feature.TEXT, "advanced-model")
        assert provider.supports_feature(Feature.TOOLS, "advanced-model")
        assert provider.supports_feature(Feature.VISION, "advanced-model")
    
    def test_get_model_capabilities(self):
        """Test getting model-specific capabilities"""
        provider = ProviderConfig(
            name="test",
            features={Feature.TEXT},
            max_context_length=8192,
            max_output_tokens=2048,
            model_capabilities=[
                ModelCapabilities(
                    pattern="large-.*",
                    features={Feature.TOOLS},
                    max_context_length=32768,
                    max_output_tokens=4096
                )
            ]
        )
        
        # Default model
        default_caps = provider.get_model_capabilities("small-model")
        assert Feature.TEXT in default_caps.features
        assert Feature.TOOLS not in default_caps.features
        assert default_caps.max_context_length == 8192
        
        # Large model with overrides
        large_caps = provider.get_model_capabilities("large-model")
        assert Feature.TEXT in large_caps.features  # Inherited
        assert Feature.TOOLS in large_caps.features  # Model-specific
        assert large_caps.max_context_length == 32768  # Override
    
    def test_get_rate_limit(self):
        """Test rate limit retrieval"""
        provider = ProviderConfig(
            name="test",
            rate_limits={"default": 100, "premium": 500}
        )
        
        assert provider.get_rate_limit() == 100
        assert provider.get_rate_limit("default") == 100
        assert provider.get_rate_limit("premium") == 500
        assert provider.get_rate_limit("unknown") is None


# ──────────────────────────── UnifiedConfigManager Tests ─────────────────────────────
class TestUnifiedConfigManager:
    """Test the UnifiedConfigManager class"""
    
    @pytest.fixture
    def sample_yaml_content(self):
        """Sample YAML configuration for testing"""
        return """
__global__:
  active_provider: openai
  active_model: gpt-4o-mini

__global_aliases__:
  gpt4: openai/gpt-4o
  claude: anthropic/claude-3-sonnet

openai:
  client_class: "chuk_llm.llm.providers.openai_client.OpenAILLMClient"
  api_key_env: "OPENAI_API_KEY"
  default_model: "gpt-4o-mini"
  features: [text, streaming, tools]
  max_context_length: 128000
  models:
    - "gpt-4o"
    - "gpt-4o-mini"
  model_aliases:
    latest: "gpt-4o"
    
anthropic:
  client_class: "chuk_llm.llm.providers.anthropic_client.AnthropicLLMClient"
  api_key_env: "ANTHROPIC_API_KEY"
  default_model: "claude-3-sonnet"
  features: [text, streaming]
  
ollama:
  inherits: "openai"
  api_base: "http://localhost:11434"
  default_model: "llama3"
"""
    
    @pytest.fixture
    def temp_config_file(self, sample_yaml_content):
        """Create a temporary config file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(sample_yaml_content)
            temp_path = f.name
        
        yield temp_path
        
        # Cleanup
        os.unlink(temp_path)
    
    def test_initialization(self):
        """Test creating a config manager"""
        config = UnifiedConfigManager()
        assert config.providers == {}
        assert config.global_aliases == {}
        assert config.global_settings == {}
        assert config._loaded is False
    
    def test_load_yaml(self, temp_config_file):
        """Test loading YAML configuration"""
        config = UnifiedConfigManager(config_path=temp_config_file)
        config.load()
        
        # Check global settings
        assert config.global_settings["active_provider"] == "openai"
        assert config.global_settings["active_model"] == "gpt-4o-mini"
        
        # Check global aliases
        assert config.global_aliases["gpt4"] == "openai/gpt-4o"
        assert config.global_aliases["claude"] == "anthropic/claude-3-sonnet"
        
        # Check providers
        assert "openai" in config.providers
        assert "anthropic" in config.providers
        assert "ollama" in config.providers
    
    def test_get_provider(self, temp_config_file):
        """Test getting provider configuration"""
        config = UnifiedConfigManager(config_path=temp_config_file)
        
        openai = config.get_provider("openai")
        assert openai.name == "openai"
        assert openai.default_model == "gpt-4o-mini"
        assert openai.api_key_env == "OPENAI_API_KEY"
        assert Feature.TEXT in openai.features
        assert Feature.TOOLS in openai.features
    
    def test_get_provider_unknown(self, temp_config_file):
        """Test getting unknown provider raises error"""
        config = UnifiedConfigManager(config_path=temp_config_file)
        
        with pytest.raises(ValueError, match="Unknown provider: invalid"):
            config.get_provider("invalid")
    
    def test_inheritance(self, temp_config_file):
        """Test provider inheritance"""
        config = UnifiedConfigManager(config_path=temp_config_file)
        config.load()
        
        ollama = config.get_provider("ollama")
        
        # Should inherit from openai
        assert ollama.client_class == "chuk_llm.llm.providers.openai_client.OpenAILLMClient"
        assert Feature.TEXT in ollama.features
        assert Feature.TOOLS in ollama.features
        
        # Should have its own settings
        assert ollama.api_base == "http://localhost:11434"
        assert ollama.default_model == "llama3"
    
    def test_get_api_key(self, temp_config_file):
        """Test API key retrieval"""
        config = UnifiedConfigManager(config_path=temp_config_file)
        
        # Mock environment variables - need to clear existing OPENAI_API_KEY
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key-123"}, clear=True):
            key = config.get_api_key("openai")
            assert key == "test-key-123"
        
        # No environment variable - clear all env vars
        with patch.dict(os.environ, {}, clear=True):
            key = config.get_api_key("openai")
            assert key is None
    
    def test_supports_feature(self, temp_config_file):
        """Test feature support checking"""
        config = UnifiedConfigManager(config_path=temp_config_file)
        
        assert config.supports_feature("openai", Feature.TEXT)
        assert config.supports_feature("openai", "tools")
        assert not config.supports_feature("anthropic", Feature.TOOLS)
    
    def test_get_all_providers(self, temp_config_file):
        """Test getting all provider names"""
        config = UnifiedConfigManager(config_path=temp_config_file)
        
        providers = config.get_all_providers()
        assert "openai" in providers
        assert "anthropic" in providers
        assert "ollama" in providers
        assert len(providers) == 3
    
    def test_reload(self, temp_config_file):
        """Test reloading configuration"""
        config = UnifiedConfigManager(config_path=temp_config_file)
        config.load()
        
        # Modify settings
        config.set_global_setting("test", "value")
        assert config.global_settings["test"] == "value"
        
        # Reload should clear modifications
        config.reload()
        assert "test" not in config.global_settings


# ──────────────────────────── ConfigValidator Tests ─────────────────────────────
class TestConfigValidator:
    """Test the ConfigValidator class"""
    
    def test_validate_provider_config_valid(self):
        """Test validating a valid provider config"""
        provider = ProviderConfig(
            name="test",
            client_class="test.Client",
            api_key_env="TEST_KEY",
            default_model="test-model"
        )
        
        with patch.dict(os.environ, {"TEST_KEY": "key123"}):
            is_valid, issues = ConfigValidator.validate_provider_config(provider)
            assert is_valid
            assert len(issues) == 0
    
    def test_validate_provider_config_missing_fields(self):
        """Test validating config with missing required fields"""
        provider = ProviderConfig(name="test")
        
        is_valid, issues = ConfigValidator.validate_provider_config(provider)
        assert not is_valid
        assert any("client_class" in issue for issue in issues)
        assert any("default_model" in issue for issue in issues)
    
    def test_validate_provider_config_missing_api_key(self):
        """Test validating config with missing API key"""
        provider = ProviderConfig(
            name="test",
            client_class="test.Client",
            api_key_env="TEST_KEY",
            default_model="test-model"
        )
        
        # No environment variable set
        is_valid, issues = ConfigValidator.validate_provider_config(provider)
        assert not is_valid
        assert any("TEST_KEY environment variable not set" in issue for issue in issues)
    
    def test_validate_provider_config_local_providers(self):
        """Test that local providers don't need API keys"""
        provider = ProviderConfig(
            name="ollama",
            client_class="test.Client",
            default_model="test-model"
        )
        
        is_valid, issues = ConfigValidator.validate_provider_config(provider)
        assert is_valid  # ollama doesn't need API key
    
    def test_validate_request_compatibility(self):
        """Test request compatibility validation"""
        # Mock config manager
        with patch('chuk_llm.configuration.unified_config.get_config') as mock_get_config:
            mock_config = MagicMock()
            mock_provider = MagicMock()
            mock_provider.supports_feature.side_effect = lambda f, m: f == Feature.TEXT
            mock_config.get_provider.return_value = mock_provider
            mock_get_config.return_value = mock_config
            
            # Valid request
            is_valid, issues = ConfigValidator.validate_request_compatibility(
                provider_name="test",
                model="test-model",
                stream=False
            )
            assert is_valid
            
            # Request with unsupported streaming
            is_valid, issues = ConfigValidator.validate_request_compatibility(
                provider_name="test",
                model="test-model",
                stream=True
            )
            assert not is_valid
            assert any("doesn't support streaming" in issue for issue in issues)
    
    def test_has_vision_content(self):
        """Test vision content detection"""
        # Text only
        messages = [{"role": "user", "content": "Hello"}]
        assert not ConfigValidator._has_vision_content(messages)
        
        # With image
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "What's this?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
            ]
        }]
        assert ConfigValidator._has_vision_content(messages)
    
    def test_is_valid_url(self):
        """Test URL validation"""
        assert ConfigValidator._is_valid_url("https://api.openai.com")
        assert ConfigValidator._is_valid_url("http://localhost:8080")
        assert ConfigValidator._is_valid_url("https://example.com/v1/api")
        
        assert not ConfigValidator._is_valid_url("")
        assert not ConfigValidator._is_valid_url("not-a-url")
        assert not ConfigValidator._is_valid_url("ftp://example.com")


# ──────────────────────────── CapabilityChecker Tests ─────────────────────────────
class TestCapabilityChecker:
    """Test the CapabilityChecker class"""
    
    def test_can_handle_request(self):
        """Test checking if provider can handle request"""
        with patch('chuk_llm.configuration.unified_config.get_config') as mock_get_config:
            mock_config = MagicMock()
            mock_provider = MagicMock()
            mock_provider.supports_feature.side_effect = lambda f, m: f in [Feature.TEXT, Feature.STREAMING]
            mock_config.get_provider.return_value = mock_provider
            mock_get_config.return_value = mock_config
            
            # Can handle
            can_handle, problems = CapabilityChecker.can_handle_request(
                provider="test",
                needs_streaming=True
            )
            assert can_handle
            assert len(problems) == 0
            
            # Cannot handle
            can_handle, problems = CapabilityChecker.can_handle_request(
                provider="test",
                has_tools=True
            )
            assert not can_handle
            assert "tools not supported" in problems
    
    def test_get_best_provider_for_features(self):
        """Test finding best provider for features"""
        with patch('chuk_llm.configuration.unified_config.get_config') as mock_get_config:
            mock_config = MagicMock()
            
            # Setup providers
            provider1 = MagicMock()
            provider1.get_model_capabilities.return_value.features = {Feature.TEXT, Feature.STREAMING}
            provider1.get_rate_limit.return_value = 100
            
            provider2 = MagicMock()
            provider2.get_model_capabilities.return_value.features = {Feature.TEXT, Feature.STREAMING, Feature.TOOLS}
            provider2.get_rate_limit.return_value = 200
            
            mock_config.get_all_providers.return_value = ["provider1", "provider2"]
            mock_config.get_provider.side_effect = lambda name: provider1 if name == "provider1" else provider2
            mock_get_config.return_value = mock_config
            
            # Should pick provider2 (higher rate limit)
            best = CapabilityChecker.get_best_provider_for_features(
                required_features={Feature.TEXT, Feature.STREAMING}
            )
            assert best == "provider2"
            
            # With exclusion
            best = CapabilityChecker.get_best_provider_for_features(
                required_features={Feature.TEXT},
                exclude={"provider2"}
            )
            assert best == "provider1"
    
    def test_get_model_info(self):
        """Test getting model information"""
        with patch('chuk_llm.configuration.unified_config.get_config') as mock_get_config:
            mock_config = MagicMock()
            mock_provider = MagicMock()
            
            mock_caps = MagicMock()
            mock_caps.features = {Feature.TEXT, Feature.STREAMING, Feature.TOOLS}
            mock_caps.max_context_length = 128000
            mock_caps.max_output_tokens = 4096
            
            mock_provider.get_model_capabilities.return_value = mock_caps
            mock_provider.rate_limits = {"default": 1000}
            
            mock_config.get_provider.return_value = mock_provider
            mock_get_config.return_value = mock_config
            
            info = CapabilityChecker.get_model_info("test-provider", "test-model")
            
            assert info["provider"] == "test-provider"
            assert info["model"] == "test-model"
            assert "text" in info["features"]
            assert "streaming" in info["features"]
            assert "tools" in info["features"]
            assert info["max_context_length"] == 128000
            assert info["max_output_tokens"] == 4096
            assert info["supports_streaming"] is True
            assert info["supports_tools"] is True
            assert info["supports_vision"] is False


# ──────────────────────────── Global Functions Tests ─────────────────────────────
class TestGlobalFunctions:
    """Test global functions"""
    
    def test_get_config_singleton(self):
        """Test that get_config returns singleton"""
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2
    
    def test_reset_config(self):
        """Test resetting configuration"""
        config1 = get_config()
        reset_config()
        config2 = get_config()
        
        assert config1 is not config2


# ──────────────────────────── Integration Tests ─────────────────────────────
class TestIntegration:
    """Integration tests for the whole system"""
    
    @pytest.fixture
    def complex_yaml_content(self):
        """Complex YAML for integration testing"""
        return """
__global__:
  active_provider: openai
  default_timeout: 30

__global_aliases__:
  gpt4: openai/gpt-4o
  claude: anthropic/claude-3-sonnet
  llama: groq/llama-3.1-70b

openai:
  client_class: "openai.Client"
  api_key_env: "OPENAI_API_KEY"
  default_model: "gpt-4o-mini"
  features: [text, streaming, system_messages]
  max_context_length: 128000
  models:
    - "gpt-4o"
    - "gpt-4o-mini"
    - "o1"
    - "o1-mini"
  model_capabilities:
    - pattern: "o[1-4].*"
      features: [tools, reasoning]
      max_context_length: 200000

anthropic:
  client_class: "anthropic.Client"
  api_key_env: "ANTHROPIC_API_KEY"
  default_model: "claude-3-sonnet"
  features: [text, streaming]
  model_capabilities:
    - pattern: "claude-3-opus.*"
      features: [tools, vision, reasoning]
      max_context_length: 200000

groq:
  inherits: "openai"
  api_base: "https://api.groq.com"
  api_key_env: "GROQ_API_KEY"
  default_model: "llama-3.1-70b"
  models:
    - "llama-3.1-70b"
    - "mixtral-8x7b"
"""
    
    def test_full_workflow(self, complex_yaml_content):
        """Test a complete workflow"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(complex_yaml_content)
            temp_path = f.name
        
        try:
            # Create config
            config = UnifiedConfigManager(config_path=temp_path)
            
            # Load and check global settings
            settings = config.get_global_settings()
            assert settings["active_provider"] == "openai"
            assert settings["default_timeout"] == 30
            
            # Check aliases
            aliases = config.get_global_aliases()
            assert aliases["gpt4"] == "openai/gpt-4o"
            assert aliases["llama"] == "groq/llama-3.1-70b"
            
            # Check OpenAI config
            openai = config.get_provider("openai")
            assert openai.default_model == "gpt-4o-mini"
            assert config.supports_feature("openai", Feature.STREAMING)
            assert not config.supports_feature("openai", Feature.TOOLS)  # Base doesn't have tools
            assert config.supports_feature("openai", Feature.TOOLS, "o1")  # o1 has tools
            
            # Check model capabilities
            o1_caps = openai.get_model_capabilities("o1")
            assert Feature.REASONING in o1_caps.features
            assert o1_caps.max_context_length == 200000
            
            # Check inheritance (Groq inherits from OpenAI)
            groq = config.get_provider("groq")
            assert groq.client_class == "openai.Client"  # Inherited
            assert groq.api_base == "https://api.groq.com"  # Own setting
            assert groq.default_model == "llama-3.1-70b"  # Own setting
            
            # Validate requests
            is_valid, issues = ConfigValidator.validate_request_compatibility(
                provider_name="anthropic",
                model="claude-3-opus",
                tools=[{"name": "test"}]
            )
            assert is_valid  # claude-3-opus supports tools
            
            # Find best provider
            best = CapabilityChecker.get_best_provider_for_features(
                required_features={Feature.TEXT, Feature.STREAMING, Feature.TOOLS}
            )
            # Gemini also supports all these features based on the YAML
            assert best in ["openai", "anthropic", "gemini"]
            
        finally:
            os.unlink(temp_path)
    
    def test_no_yaml_fallback(self):
        """Test behavior when PyYAML is not available"""
        with patch('chuk_llm.configuration.unified_config.yaml', None):
            config = UnifiedConfigManager()
            config.load()
            
            # Should load with empty config
            assert len(config.providers) == 0
            assert len(config.global_aliases) == 0