"""
Comprehensive tests for sync API wrappers
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from chuk_llm.api.sync import (
    ask_sync,
    compare_providers,
    quick_question,
    stream_sync,
    stream_sync_iter,
)


class TestAskSync:
    """Test cases for ask_sync function"""

    @patch("chuk_llm.api.sync.run_sync")
    @patch("chuk_llm.api.sync.ask")
    def test_ask_sync_basic(self, mock_ask, mock_run_sync):
        """Test basic ask_sync call"""
        mock_ask.return_value = AsyncMock()
        mock_run_sync.return_value = "Test response"

        result = ask_sync("Hello")

        assert result == "Test response"
        mock_ask.assert_called_once_with("Hello")
        mock_run_sync.assert_called_once()

    @patch("chuk_llm.api.sync.run_sync")
    @patch("chuk_llm.api.sync.ask")
    def test_ask_sync_with_kwargs(self, mock_ask, mock_run_sync):
        """Test ask_sync with additional arguments"""
        mock_ask.return_value = AsyncMock()
        mock_run_sync.return_value = {"response": "test", "tool_calls": []}

        result = ask_sync("Test", provider="openai", temperature=0.7)

        mock_ask.assert_called_once_with("Test", provider="openai", temperature=0.7)
        assert result == {"response": "test", "tool_calls": []}

    @patch("chuk_llm.api.sync.run_sync")
    @patch("chuk_llm.api.sync.ask")
    def test_ask_sync_with_tools(self, mock_ask, mock_run_sync):
        """Test ask_sync returns dict when tools are involved"""
        mock_ask.return_value = AsyncMock()
        mock_run_sync.return_value = {
            "response": "Using tools",
            "tool_calls": [{"name": "test_tool"}],
        }

        result = ask_sync("Test", tools=[{"name": "test_tool"}])

        assert isinstance(result, dict)
        assert "tool_calls" in result


class TestStreamSync:
    """Test cases for stream_sync function"""

    @patch("chuk_llm.api.sync.run_sync")
    def test_stream_sync_basic(self, mock_run_sync):
        """Test basic stream_sync call"""
        mock_run_sync.return_value = ["chunk1", "chunk2", "chunk3"]

        result = stream_sync("Tell me a story")

        assert result == ["chunk1", "chunk2", "chunk3"]
        # Verify run_sync was called
        assert mock_run_sync.called

    @patch("chuk_llm.api.sync.run_sync")
    def test_stream_sync_with_kwargs(self, mock_run_sync):
        """Test stream_sync with additional arguments"""
        mock_run_sync.return_value = ["test"]

        result = stream_sync("Prompt", provider="anthropic", max_tokens=100)

        assert result == ["test"]
        assert mock_run_sync.called

    @patch("chuk_llm.api.sync.run_sync")
    @patch("chuk_llm.api.sync.stream")
    def test_stream_sync_empty_stream(self, mock_stream, mock_run_sync):
        """Test stream_sync with no chunks"""

        async def mock_async_generator():
            return
            yield  # Make it a generator

        mock_stream.return_value = mock_async_generator()
        mock_run_sync.return_value = []

        result = stream_sync("Empty")

        assert result == []


class TestStreamSyncIter:
    """Test cases for stream_sync_iter function"""

    @patch("chuk_llm.api.sync.run_sync")
    @patch("chuk_llm.api.sync.stream")
    def test_stream_sync_iter_basic(self, mock_stream, mock_run_sync):
        """Test basic stream_sync_iter call"""

        async def mock_async_generator():
            yield "chunk1"
            yield "chunk2"

        mock_stream.return_value = mock_async_generator()

        # Mock run_sync to actually execute the async function
        def mock_run_sync_impl(coro):
            import asyncio

            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(coro)
            finally:
                loop.close()

        mock_run_sync.side_effect = mock_run_sync_impl

        chunks = list(stream_sync_iter("Test prompt"))

        assert "chunk1" in chunks
        assert "chunk2" in chunks

    def test_stream_sync_iter_with_exception(self):
        """Test stream_sync_iter handles exceptions in run_sync"""
        # This test verifies that exceptions from run_sync are captured and raised
        # We need to actually run the code to test exception handling

        with patch("chuk_llm.api.sync.run_sync") as mock_run_sync:
            # Make run_sync raise an exception
            mock_run_sync.side_effect = RuntimeError("Sync error")

            chunks = []
            with pytest.raises(RuntimeError, match="Sync error"):
                for chunk in stream_sync_iter("Test"):
                    chunks.append(chunk)


class TestCompareProviders:
    """Test cases for compare_providers function"""

    @patch("chuk_llm.api.sync.ask_sync")
    def test_compare_providers_with_list(self, mock_ask_sync):
        """Test comparing specific providers"""
        mock_ask_sync.side_effect = ["Response from openai", "Response from anthropic"]

        result = compare_providers("What is AI?", providers=["openai", "anthropic"])

        assert "openai" in result
        assert "anthropic" in result
        assert result["openai"] == "Response from openai"
        assert result["anthropic"] == "Response from anthropic"
        assert mock_ask_sync.call_count == 2

    @patch("chuk_llm.api.sync.ask_sync")
    @patch("chuk_llm.configuration.get_config")
    def test_compare_providers_auto_detect(self, mock_get_config, mock_ask_sync):
        """Test comparing with auto-detected providers"""
        mock_config = MagicMock()
        mock_config.get_all_providers.return_value = [
            "provider1",
            "provider2",
            "provider3",
            "provider4",
        ]
        mock_get_config.return_value = mock_config

        mock_ask_sync.side_effect = ["Response 1", "Response 2", "Response 3"]

        result = compare_providers("Test question")

        # Should use first 3 providers
        assert len(result) == 3
        assert "provider1" in result
        assert "provider2" in result
        assert "provider3" in result

    @patch("chuk_llm.api.sync.ask_sync")
    @patch("chuk_llm.configuration.get_config")
    def test_compare_providers_fewer_than_three(self, mock_get_config, mock_ask_sync):
        """Test comparing when fewer than 3 providers available"""
        mock_config = MagicMock()
        mock_config.get_all_providers.return_value = ["provider1", "provider2"]
        mock_get_config.return_value = mock_config

        mock_ask_sync.side_effect = ["Response 1", "Response 2"]

        result = compare_providers("Test question")

        assert len(result) == 2
        assert "provider1" in result
        assert "provider2" in result

    @patch("chuk_llm.api.sync.ask_sync")
    def test_compare_providers_with_error(self, mock_ask_sync):
        """Test compare_providers handles errors gracefully"""
        mock_ask_sync.side_effect = [
            "Success",
            Exception("Provider error"),
            "Another success",
        ]

        result = compare_providers(
            "Test", providers=["provider1", "provider2", "provider3"]
        )

        assert result["provider1"] == "Success"
        assert "Error:" in result["provider2"]
        assert "Provider error" in result["provider2"]
        assert result["provider3"] == "Another success"

    @patch("chuk_llm.api.sync.ask_sync")
    def test_compare_providers_empty_list(self, mock_ask_sync):
        """Test compare_providers with empty provider list"""
        result = compare_providers("Test", providers=[])

        assert result == {}
        mock_ask_sync.assert_not_called()


class TestQuickQuestion:
    """Test cases for quick_question function"""

    @patch("chuk_llm.api.sync.ask_sync")
    def test_quick_question_with_provider(self, mock_ask_sync):
        """Test quick_question with specified provider"""
        mock_ask_sync.return_value = "Quick answer"

        result = quick_question("What is 2+2?", provider="openai")

        assert result == "Quick answer"
        mock_ask_sync.assert_called_once_with("What is 2+2?", provider="openai")

    @patch("chuk_llm.api.sync.ask_sync")
    @patch("chuk_llm.configuration.get_config")
    def test_quick_question_default_provider(self, mock_get_config, mock_ask_sync):
        """Test quick_question uses default provider from config"""
        mock_config = MagicMock()
        mock_config.global_settings = {"active_provider": "anthropic"}
        mock_get_config.return_value = mock_config

        mock_ask_sync.return_value = "Default answer"

        result = quick_question("Test question")

        assert result == "Default answer"
        mock_ask_sync.assert_called_once_with("Test question", provider="anthropic")

    @patch("chuk_llm.api.sync.ask_sync")
    @patch("chuk_llm.configuration.get_config")
    def test_quick_question_fallback_to_openai(self, mock_get_config, mock_ask_sync):
        """Test quick_question falls back to openai if no active_provider in config"""
        mock_config = MagicMock()
        mock_config.global_settings = {}
        mock_get_config.return_value = mock_config

        mock_ask_sync.return_value = "Fallback answer"

        result = quick_question("Test")

        mock_ask_sync.assert_called_once_with("Test", provider="openai")

    @patch("chuk_llm.api.sync.ask_sync")
    def test_quick_question_returns_dict(self, mock_ask_sync):
        """Test quick_question can return dict for tool responses"""
        mock_ask_sync.return_value = {"response": "test", "tool_calls": []}

        result = quick_question("Test with tools", provider="openai")

        assert isinstance(result, dict)
        assert "response" in result

    @patch("chuk_llm.api.sync.ask_sync")
    @patch("chuk_llm.configuration.get_config")
    def test_quick_question_none_provider_uses_config(
        self, mock_get_config, mock_ask_sync
    ):
        """Test passing None for provider uses config"""
        mock_config = MagicMock()
        mock_config.global_settings = {"active_provider": "gemini"}
        mock_get_config.return_value = mock_config

        mock_ask_sync.return_value = "Config provider answer"

        result = quick_question("Test", provider=None)

        mock_ask_sync.assert_called_once_with("Test", provider="gemini")
