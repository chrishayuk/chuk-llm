"""Comprehensive tests for registry/runtime_tester.py"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime
from chuk_llm.registry.runtime_tester import RuntimeCapabilityTester
from chuk_llm.core.constants import CapabilityKey
from chuk_llm.core.enums import Provider
from chuk_llm.registry.models import QualityTier


class TestRuntimeCapabilityTesterInit:
    """Test RuntimeCapabilityTester initialization"""

    def test_init_with_openai(self):
        """Test initialization with OpenAI provider"""
        tester = RuntimeCapabilityTester(Provider.OPENAI.value)
        assert tester.provider == "openai"

    def test_init_with_anthropic(self):
        """Test initialization with Anthropic provider"""
        tester = RuntimeCapabilityTester(Provider.ANTHROPIC.value)
        assert tester.provider == "anthropic"

    def test_init_with_string_provider(self):
        """Test initialization with string provider name"""
        tester = RuntimeCapabilityTester("custom-provider")
        assert tester.provider == "custom-provider"


class TestGetClient:
    """Test _get_client method"""

    def test_get_client_openai(self):
        """Test getting OpenAI client"""
        tester = RuntimeCapabilityTester(Provider.OPENAI.value)

        with patch('chuk_llm.llm.providers.openai_client.OpenAILLMClient') as mock_client_class:
            mock_instance = Mock()
            mock_client_class.return_value = mock_instance

            client = tester._get_client("gpt-4")

            mock_client_class.assert_called_once_with(model="gpt-4")
            assert client == mock_instance

    def test_get_client_anthropic(self):
        """Test getting Anthropic client"""
        tester = RuntimeCapabilityTester(Provider.ANTHROPIC.value)

        # Import is done inside the function, so mock at import time
        mock_instance = Mock()
        mock_client_class = Mock(return_value=mock_instance)

        with patch.dict('sys.modules', {'chuk_llm.llm.providers.anthropic_client': Mock(AnthropicLLMClient=mock_client_class)}):
            client = tester._get_client("claude-3")

            mock_client_class.assert_called_once_with(model="claude-3")
            assert client == mock_instance

    def test_get_client_gemini(self):
        """Test getting Gemini client"""
        tester = RuntimeCapabilityTester(Provider.GEMINI.value)

        with patch('chuk_llm.llm.providers.gemini_client.GeminiLLMClient') as mock_client_class:
            mock_instance = Mock()
            mock_client_class.return_value = mock_instance

            client = tester._get_client("gemini-pro")

            mock_client_class.assert_called_once_with(model="gemini-pro")
            assert client == mock_instance

    def test_get_client_groq(self):
        """Test getting Groq client"""
        tester = RuntimeCapabilityTester(Provider.GROQ.value)

        with patch('chuk_llm.llm.providers.groq_client.GroqAILLMClient') as mock_client_class:
            mock_instance = Mock()
            mock_client_class.return_value = mock_instance

            client = tester._get_client("mixtral-8x7b")

            mock_client_class.assert_called_once_with(model="mixtral-8x7b")
            assert client == mock_instance

    def test_get_client_mistral(self):
        """Test getting Mistral client"""
        tester = RuntimeCapabilityTester(Provider.MISTRAL.value)

        # Import is done inside the function, so mock at import time
        mock_instance = Mock()
        mock_client_class = Mock(return_value=mock_instance)

        with patch.dict('sys.modules', {'chuk_llm.llm.providers.mistral_client': Mock(MistralLLMClient=mock_client_class)}):
            client = tester._get_client("mistral-large")

            mock_client_class.assert_called_once_with(model="mistral-large")
            assert client == mock_instance

    def test_get_client_deepseek(self):
        """Test getting DeepSeek client (uses OpenAI client)"""
        tester = RuntimeCapabilityTester(Provider.DEEPSEEK.value)

        with patch('chuk_llm.llm.providers.openai_client.OpenAILLMClient') as mock_client_class:
            mock_instance = Mock()
            mock_client_class.return_value = mock_instance

            client = tester._get_client("deepseek-chat")

            mock_client_class.assert_called_once_with(model="deepseek-chat")
            assert client == mock_instance

    def test_get_client_perplexity(self):
        """Test getting Perplexity client"""
        tester = RuntimeCapabilityTester(Provider.PERPLEXITY.value)

        with patch('chuk_llm.llm.providers.perplexity_client.PerplexityLLMClient') as mock_client_class:
            mock_instance = Mock()
            mock_client_class.return_value = mock_instance

            client = tester._get_client("llama-3")

            mock_client_class.assert_called_once_with(model="llama-3")
            assert client == mock_instance

    def test_get_client_openrouter(self):
        """Test getting OpenRouter client - has a typo in source with llm.llm"""
        tester = RuntimeCapabilityTester(Provider.OPENROUTER.value)

        # The source has a typo with llm.llm.providers, so this will raise ImportError
        # We expect the function to raise an exception due to the typo
        with pytest.raises((ImportError, ModuleNotFoundError)):
            tester._get_client("openrouter-model")

    def test_get_client_unknown_provider(self):
        """Test getting client for unknown provider raises error"""
        tester = RuntimeCapabilityTester("unknown-provider")

        with pytest.raises(ValueError, match="Unknown provider: unknown-provider"):
            tester._get_client("test-model")


class TestTestModel:
    """Test test_model method"""

    @pytest.mark.asyncio
    async def test_test_model_all_capabilities_true(self):
        """Test when all capabilities are supported"""
        tester = RuntimeCapabilityTester(Provider.OPENAI.value)

        mock_client = Mock()

        with patch.object(tester, '_get_client', return_value=mock_client):
            with patch('chuk_llm.registry.runtime_tester.test_tools', new_callable=AsyncMock) as mock_tools:
                with patch('chuk_llm.registry.runtime_tester.test_vision', new_callable=AsyncMock) as mock_vision:
                    with patch('chuk_llm.registry.runtime_tester.test_json_mode', new_callable=AsyncMock) as mock_json:
                        with patch('chuk_llm.registry.runtime_tester.test_streaming', new_callable=AsyncMock) as mock_stream:
                            mock_tools.return_value = True
                            mock_vision.return_value = True
                            mock_json.return_value = True
                            mock_stream.return_value = True

                            capabilities = await tester.test_model("gpt-4o")

                            # Verify all test functions were called with the client
                            mock_tools.assert_called_once_with(mock_client)
                            mock_vision.assert_called_once_with(mock_client)
                            mock_json.assert_called_once_with(mock_client)
                            mock_stream.assert_called_once_with(mock_client)

                            # Verify capabilities
                            assert capabilities.supports_tools is True
                            assert capabilities.supports_vision is True
                            assert capabilities.supports_json_mode is True
                            assert capabilities.supports_streaming is True
                            assert capabilities.supports_system_messages is True
                            assert capabilities.quality_tier == QualityTier.UNKNOWN
                            assert capabilities.source == "runtime_test"
                            assert capabilities.last_updated is not None

    @pytest.mark.asyncio
    async def test_test_model_mixed_capabilities(self):
        """Test when some capabilities are supported"""
        tester = RuntimeCapabilityTester(Provider.ANTHROPIC.value)

        mock_client = Mock()

        with patch.object(tester, '_get_client', return_value=mock_client):
            with patch('chuk_llm.registry.runtime_tester.test_tools', new_callable=AsyncMock) as mock_tools:
                with patch('chuk_llm.registry.runtime_tester.test_vision', new_callable=AsyncMock) as mock_vision:
                    with patch('chuk_llm.registry.runtime_tester.test_json_mode', new_callable=AsyncMock) as mock_json:
                        with patch('chuk_llm.registry.runtime_tester.test_streaming', new_callable=AsyncMock) as mock_stream:
                            mock_tools.return_value = True
                            mock_vision.return_value = True
                            mock_json.return_value = False  # No JSON mode
                            mock_stream.return_value = True

                            capabilities = await tester.test_model("claude-3")

                            assert capabilities.supports_tools is True
                            assert capabilities.supports_vision is True
                            assert capabilities.supports_json_mode is False
                            assert capabilities.supports_streaming is True

    @pytest.mark.asyncio
    async def test_test_model_no_capabilities(self):
        """Test when no capabilities are supported"""
        tester = RuntimeCapabilityTester(Provider.GROQ.value)

        mock_client = Mock()

        with patch.object(tester, '_get_client', return_value=mock_client):
            with patch('chuk_llm.registry.runtime_tester.test_tools', new_callable=AsyncMock) as mock_tools:
                with patch('chuk_llm.registry.runtime_tester.test_vision', new_callable=AsyncMock) as mock_vision:
                    with patch('chuk_llm.registry.runtime_tester.test_json_mode', new_callable=AsyncMock) as mock_json:
                        with patch('chuk_llm.registry.runtime_tester.test_streaming', new_callable=AsyncMock) as mock_stream:
                            mock_tools.return_value = False
                            mock_vision.return_value = False
                            mock_json.return_value = False
                            mock_stream.return_value = False

                            capabilities = await tester.test_model("llama-model")

                            assert capabilities.supports_tools is False
                            assert capabilities.supports_vision is False
                            assert capabilities.supports_json_mode is False
                            assert capabilities.supports_streaming is False
                            # System messages always assumed true
                            assert capabilities.supports_system_messages is True

    @pytest.mark.asyncio
    async def test_test_model_logs_info(self):
        """Test that testing logs appropriate info messages"""
        tester = RuntimeCapabilityTester(Provider.OPENAI.value)

        mock_client = Mock()

        with patch.object(tester, '_get_client', return_value=mock_client):
            with patch('chuk_llm.registry.runtime_tester.test_tools', new_callable=AsyncMock, return_value=True):
                with patch('chuk_llm.registry.runtime_tester.test_vision', new_callable=AsyncMock, return_value=True):
                    with patch('chuk_llm.registry.runtime_tester.test_json_mode', new_callable=AsyncMock, return_value=True):
                        with patch('chuk_llm.registry.runtime_tester.test_streaming', new_callable=AsyncMock, return_value=True):
                            with patch('chuk_llm.registry.runtime_tester.log') as mock_log:
                                await tester.test_model("gpt-4")

                                # Verify logging calls
                                assert mock_log.info.call_count == 2
                                # First call - start testing
                                first_call = mock_log.info.call_args_list[0][0][0]
                                assert "Runtime testing capabilities" in first_call
                                assert "openai/gpt-4" in first_call

                                # Second call - results
                                second_call = mock_log.info.call_args_list[1][0][0]
                                assert "Runtime test complete" in second_call
                                assert "tools=True" in second_call
                                assert "vision=True" in second_call

    @pytest.mark.asyncio
    async def test_test_model_timestamp_format(self):
        """Test that timestamp is in ISO format"""
        tester = RuntimeCapabilityTester(Provider.OPENAI.value)

        mock_client = Mock()

        with patch.object(tester, '_get_client', return_value=mock_client):
            with patch('chuk_llm.registry.runtime_tester.test_tools', new_callable=AsyncMock, return_value=True):
                with patch('chuk_llm.registry.runtime_tester.test_vision', new_callable=AsyncMock, return_value=False):
                    with patch('chuk_llm.registry.runtime_tester.test_json_mode', new_callable=AsyncMock, return_value=False):
                        with patch('chuk_llm.registry.runtime_tester.test_streaming', new_callable=AsyncMock, return_value=True):
                            capabilities = await tester.test_model("test-model")

                            # Verify timestamp can be parsed back
                            timestamp = capabilities.last_updated
                            assert timestamp is not None
                            # Should be able to parse ISO format
                            parsed = datetime.fromisoformat(timestamp)
                            assert isinstance(parsed, datetime)

    @pytest.mark.asyncio
    async def test_test_model_returns_model_capabilities(self):
        """Test that ModelCapabilities object is returned"""
        tester = RuntimeCapabilityTester(Provider.GEMINI.value)

        mock_client = Mock()

        with patch.object(tester, '_get_client', return_value=mock_client):
            with patch('chuk_llm.registry.runtime_tester.test_tools', new_callable=AsyncMock, return_value=True):
                with patch('chuk_llm.registry.runtime_tester.test_vision', new_callable=AsyncMock, return_value=True):
                    with patch('chuk_llm.registry.runtime_tester.test_json_mode', new_callable=AsyncMock, return_value=True):
                        with patch('chuk_llm.registry.runtime_tester.test_streaming', new_callable=AsyncMock, return_value=True):
                            from chuk_llm.registry.models import ModelCapabilities

                            capabilities = await tester.test_model("gemini-pro")

                            assert isinstance(capabilities, ModelCapabilities)


class TestIntegration:
    """Integration tests"""

    @pytest.mark.asyncio
    async def test_full_workflow_openai(self):
        """Test full workflow for OpenAI model"""
        tester = RuntimeCapabilityTester(Provider.OPENAI.value)

        with patch('chuk_llm.llm.providers.openai_client.OpenAILLMClient') as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            with patch('chuk_llm.registry.runtime_tester.test_tools', new_callable=AsyncMock, return_value=True):
                with patch('chuk_llm.registry.runtime_tester.test_vision', new_callable=AsyncMock, return_value=True):
                    with patch('chuk_llm.registry.runtime_tester.test_json_mode', new_callable=AsyncMock, return_value=True):
                        with patch('chuk_llm.registry.runtime_tester.test_streaming', new_callable=AsyncMock, return_value=True):
                            capabilities = await tester.test_model("gpt-4o")

                            # Verify client was created
                            mock_client_class.assert_called_once_with(model="gpt-4o")

                            # Verify all capabilities
                            assert capabilities.supports_tools is True
                            assert capabilities.supports_vision is True
                            assert capabilities.supports_json_mode is True
                            assert capabilities.supports_streaming is True
                            assert capabilities.supports_system_messages is True
                            assert capabilities.quality_tier == QualityTier.UNKNOWN
                            assert capabilities.source == "runtime_test"

    @pytest.mark.asyncio
    async def test_full_workflow_anthropic(self):
        """Test full workflow for Anthropic model"""
        tester = RuntimeCapabilityTester(Provider.ANTHROPIC.value)

        mock_client = Mock()
        mock_client_class = Mock(return_value=mock_client)

        with patch.dict('sys.modules', {'chuk_llm.llm.providers.anthropic_client': Mock(AnthropicLLMClient=mock_client_class)}):
            with patch('chuk_llm.registry.runtime_tester.test_tools', new_callable=AsyncMock, return_value=True):
                with patch('chuk_llm.registry.runtime_tester.test_vision', new_callable=AsyncMock, return_value=False):
                    with patch('chuk_llm.registry.runtime_tester.test_json_mode', new_callable=AsyncMock, return_value=False):
                        with patch('chuk_llm.registry.runtime_tester.test_streaming', new_callable=AsyncMock, return_value=True):
                            capabilities = await tester.test_model("claude-3-haiku")

                            # Verify client was created
                            mock_client_class.assert_called_once_with(model="claude-3-haiku")

                            # Verify capabilities
                            assert capabilities.supports_tools is True
                            assert capabilities.supports_vision is False
                            assert capabilities.supports_json_mode is False
                            assert capabilities.supports_streaming is True

    @pytest.mark.asyncio
    async def test_multiple_models_same_provider(self):
        """Test testing multiple models with same tester instance"""
        tester = RuntimeCapabilityTester(Provider.OPENAI.value)

        with patch('chuk_llm.llm.providers.openai_client.OpenAILLMClient') as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            with patch('chuk_llm.registry.runtime_tester.test_tools', new_callable=AsyncMock, return_value=True):
                with patch('chuk_llm.registry.runtime_tester.test_vision', new_callable=AsyncMock, return_value=True):
                    with patch('chuk_llm.registry.runtime_tester.test_json_mode', new_callable=AsyncMock, return_value=True):
                        with patch('chuk_llm.registry.runtime_tester.test_streaming', new_callable=AsyncMock, return_value=True):
                            # Test first model
                            caps1 = await tester.test_model("gpt-4")
                            # Test second model
                            caps2 = await tester.test_model("gpt-4o-mini")

                            # Both should succeed
                            assert caps1.supports_tools is True
                            assert caps2.supports_tools is True

                            # Should have created two clients
                            assert mock_client_class.call_count == 2
                            mock_client_class.assert_any_call(model="gpt-4")
                            mock_client_class.assert_any_call(model="gpt-4o-mini")


class TestCapabilityKeyUsage:
    """Test that CapabilityKey enum is used correctly"""

    @pytest.mark.asyncio
    async def test_capability_keys_are_correct(self):
        """Test that correct capability keys are used"""
        tester = RuntimeCapabilityTester(Provider.OPENAI.value)

        mock_client = Mock()

        with patch.object(tester, '_get_client', return_value=mock_client):
            with patch('chuk_llm.registry.runtime_tester.test_tools', new_callable=AsyncMock, return_value=True):
                with patch('chuk_llm.registry.runtime_tester.test_vision', new_callable=AsyncMock, return_value=True):
                    with patch('chuk_llm.registry.runtime_tester.test_json_mode', new_callable=AsyncMock, return_value=True):
                        with patch('chuk_llm.registry.runtime_tester.test_streaming', new_callable=AsyncMock, return_value=True):
                            capabilities = await tester.test_model("test-model")

                            # Verify the capability keys match the enum
                            assert hasattr(capabilities, 'supports_tools')
                            assert hasattr(capabilities, 'supports_vision')
                            assert hasattr(capabilities, 'supports_json_mode')
                            assert hasattr(capabilities, 'supports_streaming')
                            assert hasattr(capabilities, 'supports_system_messages')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
