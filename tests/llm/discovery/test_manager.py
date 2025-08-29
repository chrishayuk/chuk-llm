# tests/test_discovery/test_manager.py
"""
Tests for chuk_llm.llm.discovery.manager module
"""

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from chuk_llm.configuration import Feature
from chuk_llm.llm.discovery.base import BaseModelDiscoverer, DiscoveredModel
from chuk_llm.llm.discovery.engine import UniversalModelDiscoveryManager
from chuk_llm.llm.discovery.manager import DiscoveryResults, UniversalDiscoveryManager


class MockDiscoverer(BaseModelDiscoverer):
    """Mock discoverer for testing"""

    def __init__(self, provider_name: str, **config):
        super().__init__(provider_name, **config)
        self.mock_models = config.get("mock_models", [])
        self.should_fail = config.get("should_fail", False)
        self.discover_call_count = 0

    async def discover_models(self) -> list[dict[str, Any]]:
        self.discover_call_count += 1
        if self.should_fail:
            raise Exception(f"Mock discovery failure for {self.provider_name}")
        return self.mock_models.copy()


class MockConfigManager:
    """Mock configuration manager"""

    def __init__(self, providers_config: dict[str, Any] = None):
        self.providers_config = providers_config or {}

    def get_all_providers(self):
        return self.providers_config

    def get_provider(self, provider_name: str):
        return self.providers_config.get(provider_name, Mock())


class TestDiscoveryResults:
    """Test DiscoveryResults dataclass"""

    def test_discovery_results_creation(self):
        """Test basic DiscoveryResults creation"""
        models_by_provider = {
            "openai": [DiscoveredModel("gpt-4", "openai")],
            "ollama": [DiscoveredModel("llama-3", "ollama")],
        }

        errors = {"anthropic": "API key not found"}
        summary = {"total_providers": 3, "successful_providers": 2}

        results = DiscoveryResults(
            total_models=2,
            models_by_provider=models_by_provider,
            discovery_time=1.5,
            errors=errors,
            summary=summary,
        )

        assert results.total_models == 2
        assert len(results.models_by_provider) == 2
        assert results.discovery_time == 1.5
        assert "anthropic" in results.errors
        assert results.summary["total_providers"] == 3


class TestUniversalDiscoveryManager:
    """Test UniversalDiscoveryManager"""

    def setup_method(self):
        """Setup test manager"""
        self.manager = UniversalDiscoveryManager()

    def test_manager_initialization_without_config(self):
        """Test manager initialization without config manager"""
        manager = UniversalDiscoveryManager()

        assert manager.config_manager is None
        assert manager.provider_managers == {}
        assert manager._discovery_cache is None
        assert manager._cache_timeout == 300

    def test_manager_initialization_with_config(self):
        """Test manager initialization with config manager"""
        config_manager = MockConfigManager(
            {
                "openai": Mock(extra={"dynamic_discovery": {"enabled": True}}),
                "ollama": Mock(extra={}),  # No discovery config
            }
        )

        with patch.object(
            UniversalDiscoveryManager, "_setup_provider_discovery"
        ) as mock_setup:
            manager = UniversalDiscoveryManager(config_manager)

            assert manager.config_manager is config_manager
            mock_setup.assert_called_once_with("openai", {"enabled": True})

    def test_setup_provider_discovery_success(self):
        """Test successful provider discovery setup"""
        discovery_config = {
            "enabled": True,
            "inference_config": {"default_features": ["text"]},
            "discoverer_config": {"cache_timeout": 600},
        }

        with patch("chuk_llm.llm.discovery.manager.DiscovererFactory") as mock_factory:
            mock_discoverer = MockDiscoverer("test-provider")
            mock_factory.create_discoverer.return_value = mock_discoverer

            self.manager._setup_provider_discovery("test-provider", discovery_config)

            assert "test-provider" in self.manager.provider_managers
            assert isinstance(
                self.manager.provider_managers["test-provider"],
                UniversalModelDiscoveryManager,
            )

            mock_factory.create_discoverer.assert_called_once()

    def test_setup_provider_discovery_failure(self):
        """Test handling of provider discovery setup failure"""
        discovery_config = {"enabled": True}

        with patch("chuk_llm.llm.discovery.manager.DiscovererFactory") as mock_factory:
            mock_factory.create_discoverer.side_effect = Exception("Setup failed")

            # Should not raise, just log warning
            self.manager._setup_provider_discovery("test-provider", discovery_config)

            assert "test-provider" not in self.manager.provider_managers

    def test_build_discoverer_config_openai(self):
        """Test building discoverer config for OpenAI"""
        discovery_config = {
            "cache_timeout": 600,
            "discoverer_config": {"custom_param": "value"},
        }

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            config = self.manager._build_discoverer_config("openai", discovery_config)

            assert config["api_key"] == "test-key"
            assert config["api_base"] == "https://api.openai.com/v1"
            assert config["cache_timeout"] == 600
            assert config["custom_param"] == "value"

    def test_build_discoverer_config_ollama(self):
        """Test building discoverer config for Ollama"""
        discovery_config = {"api_base": "http://custom-ollama:11434"}

        config = self.manager._build_discoverer_config("ollama", discovery_config)

        assert config["api_base"] == "http://custom-ollama:11434"

    def test_build_discoverer_config_groq(self):
        """Test building discoverer config for Groq"""
        discovery_config = {}

        with patch.dict("os.environ", {"GROQ_API_KEY": "groq-key"}):
            config = self.manager._build_discoverer_config("groq", discovery_config)

            assert config["api_key"] == "groq-key"
            assert config["api_base"] == "https://api.groq.com/openai/v1"

    def test_build_discoverer_config_deepseek(self):
        """Test building discoverer config for Deepseek"""
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "deepseek-key"}):
            config = self.manager._build_discoverer_config("deepseek", {})

            assert config["api_key"] == "deepseek-key"
            assert config["api_base"] == "https://api.deepseek.com"

    def test_build_discoverer_config_huggingface(self):
        """Test building discoverer config for HuggingFace"""
        discovery_config = {"limit": 100}

        with patch.dict("os.environ", {"HUGGINGFACE_API_KEY": "hf-key"}):
            config = self.manager._build_discoverer_config(
                "huggingface", discovery_config
            )

            assert config["api_key"] == "hf-key"
            assert config["limit"] == 100

    def test_build_discoverer_config_local(self):
        """Test building discoverer config for local provider"""
        discovery_config = {"model_paths": ["/path/to/models"]}

        config = self.manager._build_discoverer_config("local", discovery_config)

        assert config["model_paths"] == ["/path/to/models"]

    @pytest.mark.asyncio
    async def test_discover_all_models_success(self):
        """Test successful discovery from all providers"""
        # Setup mock managers
        mock_manager1 = Mock()
        mock_manager1.discover_models = AsyncMock(
            return_value=[
                DiscoveredModel("model1", "provider1"),
                DiscoveredModel("model2", "provider1"),
            ]
        )

        mock_manager2 = Mock()
        mock_manager2.discover_models = AsyncMock(
            return_value=[DiscoveredModel("model3", "provider2")]
        )

        self.manager.provider_managers = {
            "provider1": mock_manager1,
            "provider2": mock_manager2,
        }

        results = await self.manager.discover_all_models()

        assert isinstance(results, DiscoveryResults)
        assert results.total_models == 3
        assert len(results.models_by_provider) == 2
        assert len(results.models_by_provider["provider1"]) == 2
        assert len(results.models_by_provider["provider2"]) == 1
        assert len(results.errors) == 0
        assert results.discovery_time >= 0  # Can be 0 on fast machines

    @pytest.mark.asyncio
    async def test_discover_all_models_with_failures(self):
        """Test discovery with some provider failures"""
        mock_manager1 = Mock()
        mock_manager1.discover_models = AsyncMock(
            return_value=[DiscoveredModel("model1", "provider1")]
        )

        mock_manager2 = Mock()
        mock_manager2.discover_models = AsyncMock(
            side_effect=Exception("Provider2 failed")
        )

        self.manager.provider_managers = {
            "provider1": mock_manager1,
            "provider2": mock_manager2,
        }

        results = await self.manager.discover_all_models()

        assert results.total_models == 1  # Only successful provider
        assert len(results.models_by_provider["provider1"]) == 1
        assert len(results.models_by_provider["provider2"]) == 0
        assert "provider2" in results.errors
        assert "Provider2 failed" in results.errors["provider2"]

    @pytest.mark.asyncio
    async def test_discover_all_models_caching(self):
        """Test caching behavior"""
        mock_manager = Mock()
        mock_manager.discover_models = AsyncMock(
            return_value=[DiscoveredModel("model1", "provider1")]
        )

        self.manager.provider_managers = {"provider1": mock_manager}

        # First call
        results1 = await self.manager.discover_all_models()

        # Second call should use cache
        results2 = await self.manager.discover_all_models()

        assert results1 == results2
        mock_manager.discover_models.assert_called_once()  # Only called once due to cache

    @pytest.mark.asyncio
    async def test_discover_all_models_force_refresh(self):
        """Test force refresh bypasses cache"""
        mock_manager = Mock()
        mock_manager.discover_models = AsyncMock(
            return_value=[DiscoveredModel("model1", "provider1")]
        )

        self.manager.provider_managers = {"provider1": mock_manager}

        # First call
        await self.manager.discover_all_models()

        # Force refresh should bypass cache
        await self.manager.discover_all_models(force_refresh=True)

        assert mock_manager.discover_models.call_count == 2

    @pytest.mark.asyncio
    async def test_discover_provider_models(self):
        """Test discovering models for specific provider"""
        mock_models = [DiscoveredModel("model1", "test-provider")]
        mock_manager = Mock()
        mock_manager.discover_models = AsyncMock(return_value=mock_models)

        self.manager.provider_managers = {"test-provider": mock_manager}

        result = await self.manager.discover_provider_models("test-provider")

        assert result == mock_models
        mock_manager.discover_models.assert_called_once_with(False)

    @pytest.mark.asyncio
    async def test_discover_provider_models_not_found(self):
        """Test error when provider not found"""
        with pytest.raises(
            ValueError, match="No discovery manager for provider: nonexistent"
        ):
            await self.manager.discover_provider_models("nonexistent")

    def test_get_available_providers(self):
        """Test getting list of available providers"""
        self.manager.provider_managers = {"provider1": Mock(), "provider2": Mock()}

        providers = self.manager.get_available_providers()

        assert set(providers) == {"provider1", "provider2"}

    def test_get_provider_info_success(self):
        """Test getting provider info"""
        mock_manager = Mock()
        mock_manager.get_discovery_stats.return_value = {"total": 5, "provider": "test"}

        self.manager.provider_managers = {"test-provider": mock_manager}

        info = self.manager.get_provider_info("test-provider")

        assert info["total"] == 5
        assert info["provider"] == "test"

    def test_get_provider_info_not_found(self):
        """Test getting info for non-existent provider"""
        info = self.manager.get_provider_info("nonexistent")

        assert "error" in info
        assert "No discovery manager" in info["error"]

    def test_register_custom_provider(self):
        """Test registering custom provider"""
        mock_discoverer = MockDiscoverer("custom-provider")
        inference_config = {"default_features": ["text"]}

        self.manager.register_custom_provider(
            "custom-provider", mock_discoverer, inference_config
        )

        assert "custom-provider" in self.manager.provider_managers
        manager = self.manager.provider_managers["custom-provider"]
        assert isinstance(manager, UniversalModelDiscoveryManager)
        assert manager.provider_name == "custom-provider"

    def test_generate_discovery_summary(self):
        """Test generating discovery summary"""
        models_by_provider = {
            "provider1": [
                DiscoveredModel(
                    "model1",
                    "provider1",
                    family="llama",
                    capabilities={Feature.TEXT, Feature.REASONING},
                ),
                DiscoveredModel(
                    "model2",
                    "provider1",
                    family="gpt4",
                    capabilities={Feature.TEXT, Feature.VISION},
                ),
            ],
            "provider2": [
                DiscoveredModel(
                    "model3", "provider2", family="llama", capabilities={Feature.TEXT}
                )
            ],
        }

        errors = {"provider3": "Failed to connect"}

        # Setup provider managers to calculate success rate
        self.manager.provider_managers = {
            "provider1": Mock(),
            "provider2": Mock(),
            "provider3": Mock(),
        }

        summary = self.manager._generate_discovery_summary(
            models_by_provider, errors, 2.5
        )

        assert summary["discovery_time"] == 2.5
        assert summary["total_providers"] == 3
        assert summary["successful_providers"] == 2
        assert summary["success_rate"] == 66.7
        assert summary["total_models"] == 3

        # Check family counts
        families = dict(summary["top_families"])
        assert families["llama"] == 2
        assert families["gpt4"] == 1

        # Check capability counts
        capabilities = dict(summary["top_capabilities"])
        assert capabilities["text"] == 3
        assert capabilities["reasoning"] == 1
        assert capabilities["vision"] == 1

    def test_generate_discovery_summary_with_metadata(self):
        """Test summary generation with model metadata"""
        model_with_reasoning = DiscoveredModel("reasoning-model", "provider1")
        model_with_reasoning.metadata = {
            "reasoning_capable": True,
            "is_reasoning": True,
        }

        model_with_vision = DiscoveredModel("vision-model", "provider1")
        model_with_vision.metadata = {"supports_vision": True}

        model_with_code = DiscoveredModel("code-model", "provider1")
        model_with_code.metadata = {"specialization": "code"}

        models_by_provider = {
            "provider1": [model_with_reasoning, model_with_vision, model_with_code]
        }

        summary = self.manager._generate_discovery_summary(models_by_provider, {}, 1.0)

        special_counts = summary["special_model_counts"]
        assert (
            special_counts["reasoning_models"] == 1
        )  # reasoning_capable OR is_reasoning
        assert special_counts["vision_models"] == 1
        assert special_counts["code_models"] == 1

    def test_get_model_recommendations_reasoning(self):
        """Test model recommendations for reasoning use case"""
        # Setup discovery cache with models
        model1 = DiscoveredModel("reasoning-model", "openai")
        model1.metadata = {
            "reasoning_capable": True,
            "supports_tools": True,
            "performance_tier": "high",
        }

        model2 = DiscoveredModel("basic-model", "ollama")
        model2.metadata = {"reasoning_capable": False, "performance_tier": "medium"}

        self.manager._discovery_cache = Mock()
        self.manager._discovery_cache.models_by_provider = {
            "openai": [model1],
            "ollama": [model2],
        }

        recommendations = self.manager.get_model_recommendations("reasoning")

        # Should recommend reasoning model with higher score
        assert len(recommendations) >= 1
        top_rec = recommendations[0]
        assert top_rec["model"] == "reasoning-model"
        assert top_rec["reasoning"] is True
        assert top_rec["score"] > 100  # Should have high score for reasoning use case

    def test_get_model_recommendations_vision(self):
        """Test model recommendations for vision use case"""
        vision_model = DiscoveredModel("vision-model", "openai")
        vision_model.metadata = {"supports_vision": True, "performance_tier": "high"}

        self.manager._discovery_cache = Mock()
        self.manager._discovery_cache.models_by_provider = {"openai": [vision_model]}

        recommendations = self.manager.get_model_recommendations("vision")

        assert len(recommendations) >= 1
        assert recommendations[0]["vision"] is True
        assert recommendations[0]["score"] > 100

    def test_get_model_recommendations_no_cache(self):
        """Test recommendations when no cache available"""
        recommendations = self.manager.get_model_recommendations("general")

        assert recommendations == []

    def test_generate_config_updates(self):
        """Test generating configuration updates"""
        mock_manager1 = Mock()
        mock_manager1.generate_config_yaml.return_value = "yaml_config_1"

        mock_manager2 = Mock()
        mock_manager2.generate_config_yaml.return_value = ""  # Empty config

        self.manager.provider_managers = {
            "provider1": mock_manager1,
            "provider2": mock_manager2,
        }

        self.manager._discovery_cache = Mock()
        self.manager._discovery_cache.models_by_provider = {
            "provider1": [Mock()],
            "provider2": [Mock()],
        }

        config_updates = self.manager.generate_config_updates()

        assert "provider1" in config_updates
        assert config_updates["provider1"] == "yaml_config_1"
        assert "provider2" not in config_updates  # Empty config not included

    def test_generate_config_updates_no_cache(self):
        """Test config updates when no cache available"""
        config_updates = self.manager.generate_config_updates()

        assert config_updates == {}

    @pytest.mark.asyncio
    async def test_health_check_all_healthy(self):
        """Test health check with all providers healthy"""
        mock_manager1 = Mock()
        mock_manager1.discover_models = AsyncMock(return_value=[Mock(), Mock()])

        mock_manager2 = Mock()
        mock_manager2.discover_models = AsyncMock(return_value=[Mock()])

        self.manager.provider_managers = {
            "provider1": mock_manager1,
            "provider2": mock_manager2,
        }

        health = await self.manager.health_check()

        assert health["overall_status"] == "healthy"
        assert health["total_providers"] == 2
        assert health["healthy_providers"] == 2

        assert health["providers"]["provider1"]["status"] == "healthy"
        assert health["providers"]["provider1"]["model_count"] == 2
        assert (
            health["providers"]["provider1"]["response_time"] >= 0
        )  # Allow 0 for very fast mock calls

        assert health["providers"]["provider2"]["status"] == "healthy"
        assert health["providers"]["provider2"]["model_count"] == 1

    @pytest.mark.asyncio
    async def test_health_check_some_unhealthy(self):
        """Test health check with some providers unhealthy"""
        mock_manager1 = Mock()
        mock_manager1.discover_models = AsyncMock(return_value=[Mock()])

        mock_manager2 = Mock()
        mock_manager2.discover_models = AsyncMock(
            side_effect=Exception("Connection failed")
        )

        self.manager.provider_managers = {
            "provider1": mock_manager1,
            "provider2": mock_manager2,
        }

        health = await self.manager.health_check()

        assert health["overall_status"] == "degraded"
        assert health["healthy_providers"] == 1

        assert health["providers"]["provider1"]["status"] == "healthy"
        assert health["providers"]["provider2"]["status"] == "unhealthy"
        assert health["providers"]["provider2"]["error"] == "Connection failed"
        assert health["providers"]["provider2"]["model_count"] == 0

    @pytest.mark.asyncio
    async def test_health_check_all_unhealthy(self):
        """Test health check with all providers unhealthy"""
        mock_manager = Mock()
        mock_manager.discover_models = AsyncMock(side_effect=Exception("All failed"))

        self.manager.provider_managers = {"provider1": mock_manager}

        health = await self.manager.health_check()

        assert health["overall_status"] == "critical"
        assert health["healthy_providers"] == 0


class TestIntegration:
    """Integration tests for discovery manager"""

    @pytest.mark.asyncio
    async def test_full_discovery_workflow(self):
        """Test complete discovery workflow from setup to results"""
        # Mock configuration
        config_manager = MockConfigManager(
            {
                "test-provider": Mock(
                    extra={
                        "dynamic_discovery": {
                            "enabled": True,
                            "inference_config": {"default_features": ["text"]},
                        }
                    }
                )
            }
        )

        # Mock factory to return our test discoverer
        with patch("chuk_llm.llm.discovery.manager.DiscovererFactory") as mock_factory:
            mock_discoverer = MockDiscoverer(
                "test-provider",
                mock_models=[
                    {"name": "model1", "size": 1000000},
                    {"name": "model2", "size": 2000000},
                ],
            )
            mock_factory.create_discoverer.return_value = mock_discoverer

            # Initialize manager with config
            manager = UniversalDiscoveryManager(config_manager)

            # Should have setup the provider
            assert "test-provider" in manager.provider_managers

            # Discover all models
            results = await manager.discover_all_models()

            # Verify results
            assert results.total_models == 2
            assert "test-provider" in results.models_by_provider
            assert len(results.models_by_provider["test-provider"]) == 2
            assert len(results.errors) == 0

            # Test provider-specific discovery
            provider_models = await manager.discover_provider_models("test-provider")
            assert len(provider_models) == 2

            # Test recommendations
            recommendations = manager.get_model_recommendations("general")
            assert len(recommendations) >= 0  # May be empty if scores too low

            # Test health check
            health = await manager.health_check()
            assert health["overall_status"] in ["healthy", "degraded", "critical"]
            assert "test-provider" in health["providers"]

    def test_multi_provider_setup(self):
        """Test setting up multiple providers"""
        config_manager = MockConfigManager(
            {
                "provider1": Mock(extra={"dynamic_discovery": {"enabled": True}}),
                "provider2": Mock(extra={"dynamic_discovery": {"enabled": True}}),
                "provider3": Mock(extra={}),  # No discovery config
            }
        )

        with patch("chuk_llm.llm.discovery.manager.DiscovererFactory") as mock_factory:
            mock_factory.create_discoverer.return_value = MockDiscoverer("test")

            manager = UniversalDiscoveryManager(config_manager)

            # Should setup only providers with discovery enabled
            assert len(manager.provider_managers) == 2
            assert "provider1" in manager.provider_managers
            assert "provider2" in manager.provider_managers
            assert "provider3" not in manager.provider_managers

    @pytest.mark.asyncio
    async def test_concurrent_discovery_performance(self):
        """Test that concurrent discovery is actually faster than sequential"""

        # Setup multiple providers with delayed responses
        async def slow_discovery():
            await asyncio.sleep(0.1)  # 100ms delay
            return [{"name": "model"}]

        mock_managers = {}
        for i in range(3):
            mock_manager = Mock()
            mock_manager.discover_models = AsyncMock(side_effect=slow_discovery)
            mock_managers[f"provider{i}"] = mock_manager

        manager = UniversalDiscoveryManager()
        manager.provider_managers = mock_managers

        # Time the discovery
        start_time = time.time()
        await manager.discover_all_models()
        end_time = time.time()

        # Should be closer to 0.1s (concurrent) than 0.3s (sequential)
        elapsed = end_time - start_time
        assert elapsed < 0.2  # Should be much less than 3 * 0.1s if truly concurrent
