"""
Comprehensive tests for Gemini model discoverer
Target coverage: 100%
"""

from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from chuk_llm.llm.discovery.gemini_discoverer import GeminiModelDiscoverer


class TestGeminiModelDiscoverer:
    """Test GeminiModelDiscoverer initialization"""

    def test_discoverer_initialization_defaults(self):
        """Test initialization with defaults"""
        discoverer = GeminiModelDiscoverer(api_key="test-key")
        assert discoverer.provider_name == "gemini"
        assert discoverer.api_key == "test-key"
        assert discoverer.api_base == "https://generativelanguage.googleapis.com/v1beta"

    def test_discoverer_initialization_custom(self):
        """Test initialization with custom values"""
        discoverer = GeminiModelDiscoverer(
            provider_name="custom-gemini",
            api_key="test-key",
            api_base="https://custom.api/v1",
        )
        assert discoverer.provider_name == "custom-gemini"
        assert discoverer.api_base == "https://custom.api/v1"

    def test_discoverer_initialization_no_api_key(self):
        """Test initialization without API key"""
        discoverer = GeminiModelDiscoverer()
        assert discoverer.api_key is None


class TestGeminiDiscoverModels:
    """Test model discovery"""

    @pytest.mark.asyncio
    async def test_discover_models_success(self):
        """Test successful model discovery"""
        mock_response_data = {
            "models": [
                {
                    "name": "models/gemini-2.0-flash",
                    "displayName": "Gemini 2.0 Flash",
                    "description": "Fast model",
                    "supportedGenerationMethods": ["generateContent", "streamGenerateContent"],
                    "inputTokenLimit": 1000000,
                    "outputTokenLimit": 8192,
                },
                {
                    "name": "models/gemini-1.5-pro",
                    "displayName": "Gemini 1.5 Pro",
                    "supportedGenerationMethods": ["generateContent"],
                    "inputTokenLimit": 2000000,
                    "outputTokenLimit": 8192,
                },
            ]
        }

        mock_response = Mock()
        mock_response.json.return_value = mock_response_data
        mock_response.raise_for_status = Mock()

        discoverer = GeminiModelDiscoverer(api_key="test-key")

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__ = AsyncMock(
                return_value=mock_client.return_value
            )
            mock_client.return_value.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value.get = AsyncMock(return_value=mock_response)

            models = await discoverer.discover_models()

        assert len(models) == 2
        flash = next(m for m in models if "flash" in m["name"])
        assert flash["display_name"] == "Gemini 2.0 Flash"
        assert flash["version"] == "2.0"
        assert flash["tier"] == "fast"

    @pytest.mark.asyncio
    async def test_discover_models_no_api_key(self):
        """Test fallback when no API key"""
        discoverer = GeminiModelDiscoverer()
        models = await discoverer.discover_models()
        
        # Should return fallback models
        assert len(models) > 0
        assert any("gemini" in m["name"] for m in models)

    @pytest.mark.asyncio
    async def test_discover_models_filters_non_generative(self):
        """Test that non-generative models are filtered"""
        mock_response_data = {
            "models": [
                {
                    "name": "models/embedding-001",
                    "supportedGenerationMethods": ["embedContent"],  # Not generative
                },
                {
                    "name": "models/gemini-2.0-flash",
                    "supportedGenerationMethods": ["generateContent"],
                },
            ]
        }

        mock_response = Mock()
        mock_response.json.return_value = mock_response_data
        mock_response.raise_for_status = Mock()

        discoverer = GeminiModelDiscoverer(api_key="test-key")

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__ = AsyncMock(
                return_value=mock_client.return_value
            )
            mock_client.return_value.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value.get = AsyncMock(return_value=mock_response)

            models = await discoverer.discover_models()

        # Should only include generative model
        assert len(models) == 1
        assert "gemini" in models[0]["name"]

    @pytest.mark.asyncio
    async def test_discover_models_http_error(self):
        """Test error handling returns fallback"""
        discoverer = GeminiModelDiscoverer(api_key="test-key")

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__ = AsyncMock(
                return_value=mock_client.return_value
            )
            mock_client.return_value.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value.get = AsyncMock(
                side_effect=httpx.HTTPStatusError("Error", request=Mock(), response=Mock())
            )

            models = await discoverer.discover_models()

        # Should return fallback models
        assert len(models) > 0

    @pytest.mark.asyncio
    async def test_discover_models_network_error(self):
        """Test network error returns fallback"""
        discoverer = GeminiModelDiscoverer(api_key="test-key")

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__ = AsyncMock(
                return_value=mock_client.return_value
            )
            mock_client.return_value.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value.get = AsyncMock(
                side_effect=httpx.ConnectError("Connection failed")
            )

            models = await discoverer.discover_models()

        assert len(models) > 0


class TestGeminiCategorizeModel:
    """Test model categorization"""

    def setup_method(self):
        self.discoverer = GeminiModelDiscoverer(api_key="test-key")

    def test_categorize_gemini_2_flash(self):
        """Test Gemini 2.0 Flash categorization"""
        model_data = {
            "displayName": "Gemini 2.0 Flash",
            "description": "Fast model",
            "supportedGenerationMethods": ["generateContent", "streamGenerateContent"],
            "inputTokenLimit": 1000000,
            "outputTokenLimit": 8192,
        }

        info = self.discoverer._categorize_model("gemini-2.0-flash", model_data)

        assert info["name"] == "gemini-2.0-flash"
        assert info["version"] == "2.0"
        assert info["tier"] == "fast"
        assert info["is_vision"] is True
        assert info["supports_streaming"] is True
        assert info["model_family"] == "gemini"

    def test_categorize_gemini_1_5_pro(self):
        """Test Gemini 1.5 Pro categorization"""
        model_data = {
            "displayName": "Gemini 1.5 Pro",
            "supportedGenerationMethods": ["generateContent"],
            "inputTokenLimit": 2000000,
            "outputTokenLimit": 8192,
        }

        info = self.discoverer._categorize_model("gemini-1.5-pro", model_data)

        assert info["version"] == "1.5"
        assert info["tier"] == "advanced"
        assert info["is_vision"] is True

    def test_categorize_gemini_1_0(self):
        """Test Gemini 1.0 categorization"""
        model_data = {
            "supportedGenerationMethods": ["generateContent"],
        }

        info = self.discoverer._categorize_model("gemini-1.0-pro", model_data)

        assert info["version"] == "1.0"
        assert info["tier"] == "advanced"

    def test_categorize_code_model(self):
        """Test code model detection"""
        model_data = {"supportedGenerationMethods": ["generateContent"]}

        info = self.discoverer._categorize_model("gemini-code-pro", model_data)

        assert info["is_code"] is True

    def test_categorize_streaming_support(self):
        """Test streaming detection"""
        model_data = {
            "supportedGenerationMethods": ["streamGenerateContent"],
        }

        info = self.discoverer._categorize_model("gemini-test", model_data)

        assert info["supports_streaming"] is True

    def test_categorize_no_streaming(self):
        """Test no streaming support"""
        model_data = {
            "supportedGenerationMethods": ["generateContent"],
        }

        info = self.discoverer._categorize_model("gemini-test", model_data)

        assert info["supports_streaming"] is False


class TestGeminiFallbackModels:
    """Test fallback models"""

    def test_fallback_models_returned(self):
        """Test that fallback models are well-formed"""
        discoverer = GeminiModelDiscoverer()
        fallback = discoverer._get_fallback_models()

        assert len(fallback) > 0
        
        # Check each model has required fields
        for model in fallback:
            assert "name" in model
            assert "display_name" in model
            assert "model_family" in model
            assert model["model_family"] == "gemini"
            assert "supports_tools" in model
            assert "version" in model

    def test_fallback_includes_latest(self):
        """Test fallback includes latest models"""
        discoverer = GeminiModelDiscoverer()
        fallback = discoverer._get_fallback_models()

        names = [m["name"] for m in fallback]
        assert "gemini-2.0-flash" in names or "gemini-2.5-flash" in names
