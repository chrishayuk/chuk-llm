# tests/test_discovery/test_engine.py
"""
Tests for chuk_llm.llm.discovery.engine module
"""

import pytest
import time
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

from chuk_llm.llm.discovery.engine import (
    DiscoveredModel,
    BaseModelDiscoverer, 
    ConfigDrivenInferenceEngine,
    UniversalModelDiscoveryManager
)
from chuk_llm.configuration import Feature, ModelCapabilities


class TestDiscoveredModelEngine:
    """Test DiscoveredModel in engine context"""
    
    def test_discovered_model_post_init(self):
        """Test __post_init__ method sets defaults"""
        model = DiscoveredModel("test-model", "test-provider")
        
        assert model.capabilities == set()
        assert model.metadata == {}
    
    def test_discovered_model_with_none_values(self):
        """Test handling of None values in __post_init__"""
        model = DiscoveredModel(
            "test-model", 
            "test-provider",
            capabilities=None,
            metadata=None
        )
        
        assert model.capabilities == set()
        assert model.metadata == {}
    
    def test_to_dict_with_feature_objects(self):
        """Test to_dict conversion with Feature objects"""
        model = DiscoveredModel(
            "test-model", 
            "test-provider",
            capabilities={Feature.TEXT, Feature.VISION}
        )
        
        result = model.to_dict()
        capability_values = set(result["capabilities"])
        
        assert "text" in capability_values
        assert "vision" in capability_values
    
    def test_to_dict_with_string_capabilities(self):
        """Test to_dict with mixed capability types"""
        mock_feature = Mock()
        mock_feature.value = "custom_feature"
        
        plain_string = "plain_string"
        
        model = DiscoveredModel(
            "test-model", 
            "test-provider",
            capabilities={mock_feature, plain_string}
        )
        
        result = model.to_dict()
        capability_values = set(result["capabilities"])
        
        assert "custom_feature" in capability_values
        assert "plain_string" in capability_values


class TestConfigDrivenInferenceEngine:
    """Test ConfigDrivenInferenceEngine"""
    
    def setup_method(self):
        """Setup test inference config"""
        self.test_config = {
            "default_features": ["text", "streaming"],
            "default_context_length": 8192,
            "default_max_output_tokens": 4096,
            "family_rules": {
                "llama": {
                    "patterns": [r"llama"],
                    "features": ["text", "streaming", "tools"],
                    "base_context_length": 4096,
                    "context_rules": {
                        r"llama.*3\.1": 128000,  # More specific pattern first
                        r"llama-3": 8192  # Less specific pattern second
                    }
                }
            },
            "pattern_rules": {
                "vision_models": {
                    "patterns": [r".*vision.*", r"gpt-4o"],
                    "add_features": ["vision", "multimodal"],
                    "context_length": 16384
                }
            },
            "size_rules": {
                "large_models": {
                    "min_size_bytes": 10000000,
                    "max_size_bytes": 100000000,
                    "add_features": ["reasoning"],
                    "context_length": 32768
                }
            },
            "model_overrides": {
                "gpt-4-turbo": {
                    "features": ["text", "vision", "tools"],
                    "context_length": 128000,
                    "family": "gpt4"
                }
            },
            "universal_patterns": {
                "reasoning": {
                    "patterns": [r"o1", r"reasoning"],
                    "add_features": ["reasoning"],
                    "family": "reasoning"
                }
            },
            "universal_size_rules": {
                "massive": {
                    "min_size_bytes": 100000000,
                    "add_features": ["parallel_calls"]
                }
            }
        }
        
        self.engine = ConfigDrivenInferenceEngine("test-provider", self.test_config)
    
    def test_engine_initialization(self):
        """Test engine initialization with config"""
        assert self.engine.provider_name == "test-provider"
        assert self.engine.default_context == 8192
        assert self.engine.default_max_output == 4096
        assert Feature.TEXT in self.engine.default_features
        assert Feature.STREAMING in self.engine.default_features
    
    def test_infer_capabilities_basic_model(self):
        """Test capability inference for basic model"""
        model = DiscoveredModel("basic-model", "test-provider")
        
        result = self.engine.infer_capabilities(model)
        
        assert Feature.TEXT in result.capabilities
        assert Feature.STREAMING in result.capabilities
        assert result.context_length == 8192
        assert result.max_output_tokens == 4096
    
    def test_model_override_takes_precedence(self):
        """Test that model overrides have highest priority"""
        model = DiscoveredModel("gpt-4-turbo", "test-provider")
        
        result = self.engine.infer_capabilities(model)
        
        assert result.family == "gpt4"
        assert result.context_length == 128000
        assert Feature.VISION in result.capabilities
        assert Feature.TOOLS in result.capabilities
    
    def test_family_rules_application(self):
        """Test family rules application"""
        model = DiscoveredModel("llama-2-7b", "test-provider")
        
        result = self.engine.infer_capabilities(model)
        
        assert result.family == "llama"
        assert Feature.TOOLS in result.capabilities
        assert result.context_length == 4096  # Base context for llama family
    
    def test_family_context_rules(self):
        """Test family-specific context rules"""
        model1 = DiscoveredModel("llama-3-8b", "test-provider")
        model2 = DiscoveredModel("llama-3.1-70b", "test-provider")
        
        result1 = self.engine.infer_capabilities(model1)
        result2 = self.engine.infer_capabilities(model2)
        
        # Model1 matches "llama-3" pattern -> 8192
        assert result1.context_length == 8192  # Matches llama-3 pattern
        # Model2 matches "llama.*3\.1" pattern -> 128000
        assert result2.context_length == 128000  # Matches llama.*3\.1 pattern
    
    def test_context_rules_with_matching_pattern(self):
        """Test context rules when pattern actually matches"""
        # Create a model name that would match the llama.*3\.1 pattern
        model = DiscoveredModel("llama3.1-70b", "test-provider")  # Different format
        
        result = self.engine.infer_capabilities(model)
        
        # This matches the "llama.*3\.1" pattern -> 128000
        assert result.context_length == 128000  # Matches llama.*3\.1 pattern
        assert result.family == "llama"
    
    def test_universal_patterns_application(self):
        """Test universal patterns are applied"""
        model = DiscoveredModel("o1-preview", "test-provider")
        
        result = self.engine.infer_capabilities(model)
        
        assert result.family == "reasoning"
        assert Feature.REASONING in result.capabilities
    
    def test_pattern_rules_application(self):
        """Test pattern-based rules"""
        model = DiscoveredModel("gpt-4o-vision", "test-provider")
        
        result = self.engine.infer_capabilities(model)
        
        assert Feature.VISION in result.capabilities
        assert Feature.MULTIMODAL in result.capabilities
        assert result.context_length == 16384
    
    def test_size_rules_application(self):
        """Test size-based rules"""
        model = DiscoveredModel("large-model", "test-provider", size_bytes=50000000)
        
        result = self.engine.infer_capabilities(model)
        
        assert Feature.REASONING in result.capabilities
        assert result.context_length == 32768
    
    def test_universal_size_rules_application(self):
        """Test universal size rules"""
        model = DiscoveredModel("massive-model", "test-provider", size_bytes=150000000)
        
        result = self.engine.infer_capabilities(model)
        
        assert Feature.PARALLEL_CALLS in result.capabilities
    
    def test_parameter_extraction(self):
        """Test parameter extraction from model name"""
        model1 = DiscoveredModel("llama-7b", "test-provider")
        model2 = DiscoveredModel("gpt-175b", "test-provider")
        
        result1 = self.engine.infer_capabilities(model1)
        result2 = self.engine.infer_capabilities(model2)
        
        assert result1.parameters == "7B"
        assert result2.parameters == "175B"
    
    def test_validation_and_cleanup(self):
        """Test validation and cleanup of model data"""
        model = DiscoveredModel("test-model", "test-provider")
        model.capabilities = set()  # Empty capabilities
        model.context_length = -1  # Invalid context length
        
        result = self.engine.infer_capabilities(model)
        
        assert Feature.TEXT in result.capabilities  # At least TEXT added
        assert result.context_length == 8192  # Default applied
        assert result.max_output_tokens == 4096  # Default applied
    
    def test_family_inference_from_capabilities(self):
        """Test family inference from capabilities when family is unknown"""
        model = DiscoveredModel("unknown-model", "test-provider")
        model.capabilities = {Feature.REASONING}
        model.family = "unknown"
        
        result = self.engine._validate_and_cleanup(model)
        
        # The _validate_and_cleanup method DOES infer family from capabilities
        assert Feature.REASONING in result.capabilities
        assert result.family == "reasoning"  # It does infer reasoning family from capabilities
    
    def test_complex_inference_chain(self):
        """Test complex inference with multiple rules applying"""
        # Model that matches multiple patterns
        model = DiscoveredModel("llama-3.1-70b-vision", "test-provider", size_bytes=50000000)
        
        result = self.engine.infer_capabilities(model)
        
        # Should have features from family, pattern, and size rules
        assert result.family == "llama"
        assert Feature.TOOLS in result.capabilities  # From family
        assert Feature.VISION in result.capabilities  # From pattern
        assert Feature.REASONING in result.capabilities  # From size
        # Context should be from size rules since large_models rule applies (50MB > 30MB)
        assert result.context_length == 32768  # From size rule for large models


class MockDiscoverer(BaseModelDiscoverer):
    """Mock discoverer for testing UniversalModelDiscoveryManager"""
    
    def __init__(self, provider_name: str, **config):
        super().__init__(provider_name, **config)
        self.mock_models = config.get('mock_models', [])
        self.should_fail = config.get('should_fail', False)
    
    async def discover_models(self) -> List[Dict[str, Any]]:
        if self.should_fail:
            raise Exception("Mock discovery failure")
        return self.mock_models.copy()


class TestUniversalModelDiscoveryManager:
    """Test UniversalModelDiscoveryManager"""
    
    def setup_method(self):
        """Setup test manager"""
        self.mock_models = [
            {"name": "llama-3-8b", "size": 8000000000},
            {"name": "gpt-4o", "size": 100000000000},
            {"name": "basic-model", "size": 1000000000}
        ]
        
        self.discoverer = MockDiscoverer("test-provider", mock_models=self.mock_models)
        
        self.inference_config = {
            "default_features": ["text"],
            "default_context_length": 8192,
            "default_max_output_tokens": 4096,
            "family_rules": {
                "llama": {
                    "patterns": [r"llama"],
                    "features": ["text", "tools"]
                }
            }
        }
        
        self.manager = UniversalModelDiscoveryManager(
            "test-provider", 
            self.discoverer, 
            self.inference_config
        )
    
    def test_manager_initialization(self):
        """Test manager initialization"""
        assert self.manager.provider_name == "test-provider"
        assert self.manager.discoverer == self.discoverer
        assert self.manager.inference_config == self.inference_config
        assert isinstance(self.manager.inference_engine, ConfigDrivenInferenceEngine)
    
    def test_load_default_inference_config(self):
        """Test loading default config when none provided"""
        discoverer = MockDiscoverer("test-provider")
        manager = UniversalModelDiscoveryManager("test-provider", discoverer)
        
        # Should have minimal config
        assert "default_features" in manager.inference_config
        assert manager.inference_config["default_context_length"] == 8192
    
    @pytest.mark.asyncio
    async def test_discover_models_success(self):
        """Test successful model discovery"""
        models = await self.manager.discover_models()
        
        assert len(models) == 3
        assert all(isinstance(model, DiscoveredModel) for model in models)
        assert models[0].provider == "test-provider"
    
    @pytest.mark.asyncio
    async def test_discover_models_with_inference(self):
        """Test discovery applies inference rules"""
        models = await self.manager.discover_models()
        
        # Find llama model
        llama_model = next(m for m in models if "llama" in m.name.lower())
        
        assert llama_model.family == "llama"
        assert Feature.TOOLS in llama_model.capabilities
    
    @pytest.mark.asyncio
    async def test_discover_models_caching(self):
        """Test caching behavior"""
        # First discovery
        models1 = await self.manager.discover_models()
        
        # Second discovery should use cache
        models2 = await self.manager.discover_models()
        
        assert models1 == models2
        assert len(models1) == 3
    
    @pytest.mark.asyncio
    async def test_discover_models_force_refresh(self):
        """Test force refresh bypasses cache"""
        # First discovery
        await self.manager.discover_models()
        
        # Modify mock data
        self.discoverer.mock_models.append({"name": "new-model", "size": 500000000})
        
        # Force refresh should get new data
        models = await self.manager.discover_models(force_refresh=True)
        
        assert len(models) == 4
        assert any(m.name == "new-model" for m in models)
    
    @pytest.mark.asyncio
    async def test_discover_models_failure_handling(self):
        """Test handling of discovery failures"""
        failing_discoverer = MockDiscoverer("test-provider", should_fail=True)
        manager = UniversalModelDiscoveryManager("test-provider", failing_discoverer)
        
        models = await manager.discover_models()
        
        assert models == []  # Should return empty list on failure
    
    def test_update_inference_config(self):
        """Test updating inference configuration"""
        new_config = {
            "default_features": ["text", "vision"],
            "default_context_length": 16384
        }
        
        self.manager.update_inference_config(new_config)
        
        assert self.manager.inference_config == new_config
        assert self.manager._cached_models is None  # Cache should be cleared
    
    @pytest.mark.asyncio
    async def test_get_model_capabilities(self):
        """Test getting capabilities for specific model"""
        await self.manager.discover_models()  # Populate cache
        
        capabilities = self.manager.get_model_capabilities("llama-3-8b")
        
        assert capabilities is not None
        assert isinstance(capabilities, ModelCapabilities)
        assert Feature.TEXT in capabilities.features
    
    @pytest.mark.asyncio
    async def test_get_model_capabilities_fuzzy_match(self):
        """Test fuzzy matching for model capabilities"""
        await self.manager.discover_models()
        
        # Try partial match
        capabilities = self.manager.get_model_capabilities("llama-3")
        
        assert capabilities is not None
    
    @pytest.mark.asyncio
    async def test_get_model_capabilities_not_found(self):
        """Test getting capabilities for non-existent model"""
        await self.manager.discover_models()
        
        capabilities = self.manager.get_model_capabilities("nonexistent-model")
        
        assert capabilities is None
    
    @pytest.mark.asyncio
    async def test_get_available_models(self):
        """Test getting list of available models"""
        await self.manager.discover_models()
        
        models = self.manager.get_available_models()
        
        assert len(models) == 3
        assert "llama-3-8b" in models
        assert "gpt-4o" in models
    
    @pytest.mark.asyncio
    async def test_generate_config_yaml(self):
        """Test YAML configuration generation"""
        await self.manager.discover_models()
        
        yaml_config = self.manager.generate_config_yaml()
        
        assert "# Dynamically discovered test-provider models" in yaml_config
        assert "models:" in yaml_config
        assert "model_capabilities:" in yaml_config
        assert "llama-3-8b" in yaml_config
    
    @pytest.mark.asyncio
    async def test_get_discovery_stats(self):
        """Test discovery statistics generation"""
        await self.manager.discover_models()
        
        stats = self.manager.get_discovery_stats()
        
        assert stats["total"] == 3
        assert stats["provider"] == "test-provider"
        assert "families" in stats
        assert "features" in stats
        assert stats["cache_age_seconds"] >= 0
    
    @pytest.mark.asyncio
    async def test_get_discovery_stats_no_cache(self):
        """Test stats when no models cached"""
        stats = self.manager.get_discovery_stats()
        
        assert stats["total"] == 0


class TestIntegration:
    """Integration tests for engine components"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_discovery_and_inference(self):
        """Test complete discovery and inference workflow"""
        # Setup complex inference rules
        inference_config = {
            "default_features": ["text"],
            "family_rules": {
                "llama": {
                    "patterns": [r"llama"],
                    "features": ["text", "tools", "reasoning"],
                    "base_context_length": 4096,
                    "context_rules": {r"llama.*3\.1": 128000}
                },
                "gpt4": {
                    "patterns": [r"gpt-4"],
                    "features": ["text", "tools", "vision"],
                    "base_context_length": 8192
                }
            },
            "pattern_rules": {
                "vision": {
                    "patterns": [r"vision", r"4o"],
                    "add_features": ["vision", "multimodal"]
                }
            }
        }
        
        # Mock models with variety
        mock_models = [
            {"name": "llama-3.1-70b", "size": 70000000000},
            {"name": "gpt-4o-vision", "size": 50000000000},
            {"name": "basic-chat", "size": 1000000000}
        ]
        
        discoverer = MockDiscoverer("test-provider", mock_models=mock_models)
        manager = UniversalModelDiscoveryManager("test-provider", discoverer, inference_config)
        
        # Discover models
        models = await manager.discover_models()
        
        assert len(models) == 3
        
        # Check llama model
        llama_model = next(m for m in models if "llama" in m.name)
        assert llama_model.family == "llama"
        assert llama_model.context_length == 128000  # Matches llama.*3\.1 pattern
        
        # Check gpt4 model
        gpt_model = next(m for m in models if "gpt-4" in m.name)
        assert gpt_model.family == "gpt4"
        assert Feature.VISION in gpt_model.capabilities
        assert Feature.MULTIMODAL in gpt_model.capabilities  # From pattern rules
        
        # Test YAML generation
        yaml_config = manager.generate_config_yaml()
        assert "llama-3.1-70b" in yaml_config
        assert "gpt-4o-vision" in yaml_config
        
        # Test model capabilities
        llama_caps = manager.get_model_capabilities("llama-3.1-70b")
        assert llama_caps is not None
        assert llama_caps.max_context_length == 128000  # Matches llama.*3\.1 pattern
    
    @patch('chuk_llm.configuration.get_config')
    def test_manager_with_config_integration(self, mock_get_config):
        """Test manager integration with configuration system"""
        # Mock configuration
        mock_config_manager = Mock()
        mock_provider_config = Mock()
        mock_provider_config.extra = {
            "dynamic_discovery": {
                "inference_config": {
                    "default_features": ["text", "streaming"]
                }
            },
            "model_inference": {
                "family_rules": {"test": {"patterns": [r"test"]}}
            }
        }
        
        mock_config_manager.get_provider.return_value = mock_provider_config
        mock_get_config.return_value = mock_config_manager
        
        discoverer = MockDiscoverer("test-provider")
        manager = UniversalModelDiscoveryManager("test-provider", discoverer)
        
        # Should have merged configuration
        assert "default_features" in manager.inference_config
        assert "family_rules" in manager.inference_config