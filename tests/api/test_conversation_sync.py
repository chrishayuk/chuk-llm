"""
Comprehensive tests for conversation_sync module
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call
from contextlib import contextmanager

from chuk_llm.api.conversation_sync import (
    ConversationContextSync,
    conversation_sync,
)


class TestConversationContextSync:
    """Test cases for ConversationContextSync class"""

    @pytest.fixture
    def mock_async_context(self):
        """Create a mock async context"""
        mock = MagicMock()
        mock.ask = AsyncMock(return_value="async response")
        mock.stream = AsyncMock()
        mock.save = AsyncMock(return_value="conv_123")
        mock.summarize = AsyncMock(return_value="Summary text")
        mock.extract_key_points = AsyncMock(return_value=["point1", "point2"])
        mock.get_session_stats = AsyncMock(return_value={"stat": "value"})
        mock.messages = [{"role": "user", "content": "test"}]
        mock.conversation_id = "conv_456"
        mock.session_id = "session_789"
        mock.has_session = True
        return mock

    @pytest.fixture
    def mock_loop_thread(self):
        """Create a mock event loop thread"""
        import asyncio

        def run_coro_sync(coro):
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(coro)
            finally:
                loop.close()

        mock = MagicMock()
        mock.run_coro = MagicMock(side_effect=run_coro_sync)
        return mock

    def test_init(self, mock_async_context, mock_loop_thread):
        """Test ConversationContextSync initialization"""
        sync_ctx = ConversationContextSync(mock_async_context, mock_loop_thread)

        assert sync_ctx._async_context == mock_async_context
        assert sync_ctx._loop_thread == mock_loop_thread

    def test_ask(self, mock_async_context, mock_loop_thread):
        """Test synchronous ask method"""
        sync_ctx = ConversationContextSync(mock_async_context, mock_loop_thread)

        result = sync_ctx.ask("Hello")

        assert result == "async response"
        mock_async_context.ask.assert_called_once()

    def test_ask_with_image(self, mock_async_context, mock_loop_thread):
        """Test ask with image parameter"""
        sync_ctx = ConversationContextSync(mock_async_context, mock_loop_thread)

        result = sync_ctx.ask("Describe this", image="image.jpg")

        mock_async_context.ask.assert_called_once_with(
            "Describe this", image="image.jpg"
        )

    def test_ask_with_kwargs(self, mock_async_context, mock_loop_thread):
        """Test ask with additional kwargs"""
        sync_ctx = ConversationContextSync(mock_async_context, mock_loop_thread)

        result = sync_ctx.ask("Question", temperature=0.7, max_tokens=100)

        call_args = mock_async_context.ask.call_args
        assert call_args.kwargs["temperature"] == 0.7
        assert call_args.kwargs["max_tokens"] == 100

    def test_stream(self, mock_async_context, mock_loop_thread):
        """Test synchronous stream method"""

        async def mock_stream_gen(*args, **kwargs):
            for chunk in ["Hello", " ", "World"]:
                yield chunk

        mock_async_context.stream = mock_stream_gen

        # Mock run_coro to return the collected chunks
        def run_coro_impl(coro):
            import asyncio

            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(coro)
            finally:
                loop.close()

        mock_loop_thread.run_coro = run_coro_impl

        sync_ctx = ConversationContextSync(mock_async_context, mock_loop_thread)

        chunks = list(sync_ctx.stream("Tell me a story"))

        assert chunks == ["Hello", " ", "World"]

    def test_stream_with_image(self, mock_async_context, mock_loop_thread):
        """Test stream with image"""

        async def mock_stream_gen(*args, **kwargs):
            yield "chunk"

        mock_async_context.stream = mock_stream_gen

        def run_coro_impl(coro):
            import asyncio

            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(coro)
            finally:
                loop.close()

        mock_loop_thread.run_coro = run_coro_impl

        sync_ctx = ConversationContextSync(mock_async_context, mock_loop_thread)

        chunks = list(sync_ctx.stream("Describe", image=b"image_bytes"))

        assert chunks == ["chunk"]

    def test_save(self, mock_async_context, mock_loop_thread):
        """Test save method"""
        sync_ctx = ConversationContextSync(mock_async_context, mock_loop_thread)

        result = sync_ctx.save()

        assert result == "conv_123"
        mock_async_context.save.assert_called_once()

    def test_load(self, mock_async_context, mock_loop_thread):
        """Test load method"""
        # Mock the load class method
        new_async_ctx = MagicMock()

        with patch(
            "chuk_llm.api.conversation_sync.ConversationContext.load",
            new=AsyncMock(return_value=new_async_ctx),
        ):
            sync_ctx = ConversationContextSync(mock_async_context, mock_loop_thread)

            loaded_ctx = sync_ctx.load("conv_999")

            assert isinstance(loaded_ctx, ConversationContextSync)
            assert loaded_ctx._async_context == new_async_ctx

    def test_summarize(self, mock_async_context, mock_loop_thread):
        """Test summarize method"""
        sync_ctx = ConversationContextSync(mock_async_context, mock_loop_thread)

        result = sync_ctx.summarize(max_length=300)

        assert result == "Summary text"
        mock_async_context.summarize.assert_called_once_with(300)

    def test_summarize_default_length(self, mock_async_context, mock_loop_thread):
        """Test summarize with default length"""
        sync_ctx = ConversationContextSync(mock_async_context, mock_loop_thread)

        result = sync_ctx.summarize()

        mock_async_context.summarize.assert_called_once_with(500)

    def test_extract_key_points(self, mock_async_context, mock_loop_thread):
        """Test extract_key_points method"""
        sync_ctx = ConversationContextSync(mock_async_context, mock_loop_thread)

        result = sync_ctx.extract_key_points()

        assert result == ["point1", "point2"]
        mock_async_context.extract_key_points.assert_called_once()

    def test_clear(self, mock_async_context, mock_loop_thread):
        """Test clear method"""
        sync_ctx = ConversationContextSync(mock_async_context, mock_loop_thread)

        sync_ctx.clear()

        mock_async_context.clear.assert_called_once()

    def test_get_history(self, mock_async_context, mock_loop_thread):
        """Test get_history method"""
        sync_ctx = ConversationContextSync(mock_async_context, mock_loop_thread)
        mock_async_context.get_history = MagicMock(
            return_value=[{"role": "user", "content": "test"}]
        )

        result = sync_ctx.get_history()

        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_pop_last(self, mock_async_context, mock_loop_thread):
        """Test pop_last method"""
        sync_ctx = ConversationContextSync(mock_async_context, mock_loop_thread)

        sync_ctx.pop_last()

        mock_async_context.pop_last.assert_called_once()

    def test_get_stats(self, mock_async_context, mock_loop_thread):
        """Test get_stats method"""
        sync_ctx = ConversationContextSync(mock_async_context, mock_loop_thread)
        mock_async_context.get_stats = MagicMock(
            return_value={"messages": 5, "tokens": 100}
        )

        result = sync_ctx.get_stats()

        assert result["messages"] == 5
        assert result["tokens"] == 100

    def test_get_session_stats(self, mock_async_context, mock_loop_thread):
        """Test get_session_stats method"""
        sync_ctx = ConversationContextSync(mock_async_context, mock_loop_thread)

        result = sync_ctx.get_session_stats()

        assert result == {"stat": "value"}
        mock_async_context.get_session_stats.assert_called_once()

    def test_set_system_prompt(self, mock_async_context, mock_loop_thread):
        """Test set_system_prompt method"""
        sync_ctx = ConversationContextSync(mock_async_context, mock_loop_thread)

        sync_ctx.set_system_prompt("You are a helpful assistant")

        mock_async_context.set_system_prompt.assert_called_once_with(
            "You are a helpful assistant"
        )

    def test_messages_property(self, mock_async_context, mock_loop_thread):
        """Test messages property"""
        sync_ctx = ConversationContextSync(mock_async_context, mock_loop_thread)

        messages = sync_ctx.messages

        assert messages == [{"role": "user", "content": "test"}]

    def test_conversation_id_property(self, mock_async_context, mock_loop_thread):
        """Test conversation_id property"""
        sync_ctx = ConversationContextSync(mock_async_context, mock_loop_thread)

        conv_id = sync_ctx.conversation_id

        assert conv_id == "conv_456"

    def test_session_id_property(self, mock_async_context, mock_loop_thread):
        """Test session_id property"""
        sync_ctx = ConversationContextSync(mock_async_context, mock_loop_thread)

        session_id = sync_ctx.session_id

        assert session_id == "session_789"

    def test_has_session_property(self, mock_async_context, mock_loop_thread):
        """Test has_session property"""
        sync_ctx = ConversationContextSync(mock_async_context, mock_loop_thread)

        has_session = sync_ctx.has_session

        assert has_session is True

    def test_branch(self, mock_async_context, mock_loop_thread):
        """Test branch context manager"""
        # Create a mock branch context
        branch_ctx = MagicMock()
        branch_cm = MagicMock()
        branch_cm.__aenter__ = AsyncMock(return_value=branch_ctx)
        branch_cm.__aexit__ = AsyncMock(return_value=None)

        mock_async_context.branch = MagicMock(return_value=branch_cm)

        sync_ctx = ConversationContextSync(mock_async_context, mock_loop_thread)

        with sync_ctx.branch() as branch:
            assert isinstance(branch, ConversationContextSync)
            assert branch._async_context == branch_ctx

        # Verify __aexit__ was called
        branch_cm.__aexit__.assert_called_once()


class TestConversationSyncContextManager:
    """Test cases for conversation_sync context manager"""

    @patch("chuk_llm.api.conversation_sync.EventLoopThread")
    @patch("chuk_llm.api.conversation_sync.async_conversation")
    def test_conversation_sync_basic(self, mock_async_conv, mock_loop_thread_class):
        """Test basic conversation_sync usage"""
        # Setup mocks
        mock_loop = MagicMock()
        mock_loop_thread_class.return_value = mock_loop

        mock_async_ctx = MagicMock()
        mock_async_cm = MagicMock()
        mock_async_cm.__aenter__ = AsyncMock(return_value=mock_async_ctx)
        mock_async_cm.__aexit__ = AsyncMock(return_value=None)

        mock_loop.run_coro = MagicMock(side_effect=[mock_async_cm, mock_async_ctx, None])

        mock_async_conv.return_value = mock_async_cm

        # Test
        with conversation_sync(provider="openai") as chat:
            assert isinstance(chat, ConversationContextSync)
            assert chat._async_context == mock_async_ctx

        # Verify cleanup
        mock_loop.stop.assert_called_once()

    @patch("chuk_llm.api.conversation_sync.EventLoopThread")
    @patch("chuk_llm.api.conversation_sync.async_conversation")
    def test_conversation_sync_with_all_parameters(
        self, mock_async_conv, mock_loop_thread_class
    ):
        """Test conversation_sync with all parameters"""
        import asyncio

        def run_coro_sync(coro):
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(coro)
            finally:
                loop.close()

        mock_loop = MagicMock()
        mock_loop.run_coro = MagicMock(side_effect=run_coro_sync)
        mock_loop_thread_class.return_value = mock_loop

        mock_async_ctx = MagicMock()
        mock_async_cm = MagicMock()
        mock_async_cm.__aenter__ = AsyncMock(return_value=mock_async_ctx)
        mock_async_cm.__aexit__ = AsyncMock(return_value=None)

        mock_async_conv.return_value = mock_async_cm

        with conversation_sync(
            provider="anthropic",
            model="claude-3",
            system_prompt="You are helpful",
            session_id="session_123",
            infinite_context=False,
            token_threshold=3000,
            resume_from="conv_456",
            temperature=0.8,
        ) as chat:
            pass

        # Verify async_conversation was called
        assert mock_async_conv.called
        call_kwargs = mock_async_conv.call_args[1] if mock_async_conv.call_args else {}
        assert call_kwargs.get("provider") == "anthropic"

    @patch("chuk_llm.api.conversation_sync.EventLoopThread")
    @patch("chuk_llm.api.conversation_sync.async_conversation")
    def test_conversation_sync_cleanup_on_exception(
        self, mock_async_conv, mock_loop_thread_class
    ):
        """Test that cleanup happens even on exception"""
        import asyncio

        def run_coro_sync(coro):
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(coro)
            finally:
                loop.close()

        mock_loop = MagicMock()
        mock_loop.run_coro = MagicMock(side_effect=run_coro_sync)
        mock_loop_thread_class.return_value = mock_loop

        mock_async_ctx = MagicMock()
        mock_async_cm = MagicMock()
        mock_async_cm.__aenter__ = AsyncMock(return_value=mock_async_ctx)
        mock_async_cm.__aexit__ = AsyncMock(return_value=None)

        mock_async_conv.return_value = mock_async_cm

        with pytest.raises(ValueError):
            with conversation_sync(provider="openai") as chat:
                raise ValueError("Test error")

        # Verify cleanup still happened
        mock_loop.stop.assert_called_once()

    @patch("chuk_llm.api.conversation_sync.EventLoopThread")
    @patch("chuk_llm.api.conversation_sync.async_conversation")
    def test_conversation_sync_default_parameters(
        self, mock_async_conv, mock_loop_thread_class
    ):
        """Test conversation_sync with default parameters"""
        import asyncio

        def run_coro_sync(coro):
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(coro)
            finally:
                loop.close()

        mock_loop = MagicMock()
        mock_loop.run_coro = MagicMock(side_effect=run_coro_sync)
        mock_loop_thread_class.return_value = mock_loop

        mock_async_ctx = MagicMock()
        mock_async_cm = MagicMock()
        mock_async_cm.__aenter__ = AsyncMock(return_value=mock_async_ctx)
        mock_async_cm.__aexit__ = AsyncMock(return_value=None)

        mock_async_conv.return_value = mock_async_cm

        with conversation_sync() as chat:
            pass

        # Verify conversation was created
        assert mock_async_conv.called


class TestConversationSyncIntegration:
    """Integration tests for conversation_sync"""

    @patch("chuk_llm.api.conversation_sync.EventLoopThread")
    @patch("chuk_llm.api.conversation_sync.async_conversation")
    def test_full_conversation_workflow(
        self, mock_async_conv, mock_loop_thread_class
    ):
        """Test a complete conversation workflow"""
        mock_loop = MagicMock()
        mock_loop_thread_class.return_value = mock_loop

        mock_async_ctx = MagicMock()
        mock_async_ctx.ask = AsyncMock(return_value="Response 1")
        mock_async_ctx.save = AsyncMock(return_value="conv_saved")
        mock_async_ctx.conversation_id = "conv_123"

        mock_async_cm = MagicMock()
        mock_async_cm.__aenter__ = AsyncMock(return_value=mock_async_ctx)
        mock_async_cm.__aexit__ = AsyncMock(return_value=None)

        mock_loop.run_coro = MagicMock(
            side_effect=[mock_async_cm, mock_async_ctx, "Response 1", "conv_saved", None]
        )

        mock_async_conv.return_value = mock_async_cm

        with conversation_sync(provider="openai") as chat:
            # Ask a question
            response = chat.ask("Hello")
            assert response == "Response 1"

            # Save conversation
            conv_id = chat.save()
            assert conv_id == "conv_saved"

            # Access properties
            assert chat.conversation_id == "conv_123"
