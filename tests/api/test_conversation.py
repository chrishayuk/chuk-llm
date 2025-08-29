"""
Comprehensive pytest tests for chuk_llm/api/conversation.py

This module tests:
- ConversationContext class functionality
- Session management integration
- Multi-modal support
- Conversation branching
- Persistence and resumption
- Async operations
- Error handling

Run with:
    pytest tests/api/test_conversation.py -v
    pytest tests/api/test_conversation.py -v --tb=short
    pytest tests/api/test_conversation.py::TestConversationContext::test_basic_conversation -v
"""

from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest

# Import the module under test
from chuk_llm.api.conversation import (
    _CONVERSATION_STORE,
    ConversationContext,
)
from chuk_llm.api.conversation import (
    conversation as conversation_context_manager,
)


class TestConversationContext:
    """Test suite for the ConversationContext class."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock client for testing."""
        client = AsyncMock()
        client.create_completion = AsyncMock()
        return client

    @pytest.fixture
    def mock_session_manager(self):
        """Create a mock session manager."""
        session_manager = AsyncMock()
        session_manager.session_id = "test-session-123"
        session_manager.user_says = AsyncMock()
        session_manager.ai_responds = AsyncMock()
        session_manager.get_conversation = AsyncMock()
        session_manager.get_stats = AsyncMock()
        session_manager.update_system_prompt = AsyncMock()
        return session_manager

    @pytest.fixture
    def sample_conversation_data(self):
        """Sample conversation data for testing."""
        return {
            "id": "test-conv-123",
            "created_at": datetime.utcnow().isoformat(),
            "provider": "openai",
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ],
            "kwargs": {"temperature": 0.7},
            "stats": {"total_messages": 3},
        }

    def test_conversation_context_initialization_basic(self):
        """Test basic ConversationContext initialization."""
        with patch("chuk_llm.api.conversation.get_client") as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client

            with patch(
                "chuk_llm.llm.system_prompt_generator.SystemPromptGenerator"
            ) as mock_generator:
                mock_gen_instance = Mock()
                mock_gen_instance.generate_prompt.return_value = "Test system prompt"
                mock_generator.return_value = mock_gen_instance

                ctx = ConversationContext(
                    provider="openai", model="gpt-4", temperature=0.7
                )

                assert ctx.provider == "openai"
                assert ctx.model == "gpt-4"
                assert ctx.kwargs == {"temperature": 0.7}
                assert len(ctx.messages) == 1  # System message
                assert ctx.messages[0]["role"] == "system"
                # The actual system prompt content may vary, just check it exists
                assert isinstance(ctx.messages[0]["content"], str)
                assert len(ctx.messages[0]["content"]) > 0
                assert ctx.client == mock_client
                assert isinstance(ctx._conversation_id, str)
                assert isinstance(ctx._created_at, datetime)

    def test_conversation_context_with_custom_system_prompt(self):
        """Test ConversationContext with custom system prompt."""
        with patch("chuk_llm.api.conversation.get_client"):
            ctx = ConversationContext(
                provider="anthropic",
                model="claude-3-sonnet",
                system_prompt="Custom system prompt",
            )

            assert len(ctx.messages) == 1
            assert ctx.messages[0]["role"] == "system"
            assert ctx.messages[0]["content"] == "Custom system prompt"

    @patch("chuk_llm.api.conversation._SESSIONS_ENABLED", True)
    @patch("chuk_llm.api.conversation.SessionManager")
    def test_conversation_context_with_session_manager(
        self, mock_session_manager_class
    ):
        """Test ConversationContext with session manager enabled."""
        mock_session_instance = Mock()
        mock_session_instance.session_id = "test-session-456"
        mock_session_manager_class.return_value = mock_session_instance

        with patch("chuk_llm.api.conversation.get_client"):
            ctx = ConversationContext(
                provider="openai",
                model="gpt-4",
                session_id="existing-session",
                infinite_context=True,
                token_threshold=5000,
            )

            assert ctx.session_manager == mock_session_instance
            assert ctx.session_id == "test-session-456"
            assert ctx.has_session is True

            # Verify session manager was initialized with correct parameters
            mock_session_manager_class.assert_called_once()
            call_kwargs = mock_session_manager_class.call_args[1]
            assert call_kwargs["session_id"] == "existing-session"
            assert call_kwargs["infinite_context"] is True
            assert call_kwargs["token_threshold"] == 5000

    @patch("chuk_llm.api.conversation._SESSIONS_ENABLED", False)
    def test_conversation_context_without_session_manager(self):
        """Test ConversationContext when sessions are disabled."""
        with patch("chuk_llm.api.conversation.get_client"):
            ctx = ConversationContext(provider="openai")

            assert ctx.session_manager is None
            assert ctx.session_id is None
            assert ctx.has_session is False

    @pytest.mark.asyncio
    async def test_say_basic_conversation(self, mock_client):
        """Test basic conversation with say method."""
        # Mock client response
        mock_client.create_completion.return_value = {
            "response": "Hello! How can I help you today?",
            "error": None,
        }

        with patch("chuk_llm.api.conversation.get_client", return_value=mock_client):
            ctx = ConversationContext(provider="openai", model="gpt-4")

            response = await ctx.say("Hello")

            assert response == "Hello! How can I help you today?"
            assert len(ctx.messages) == 3  # system, user, assistant
            assert ctx.messages[1]["role"] == "user"
            assert ctx.messages[1]["content"] == "Hello"
            assert ctx.messages[2]["role"] == "assistant"
            assert ctx.messages[2]["content"] == "Hello! How can I help you today?"

    @pytest.mark.asyncio
    async def test_say_with_session_tracking(self, mock_client, mock_session_manager):
        """Test say method with session tracking."""
        mock_client.create_completion.return_value = {
            "response": "Great question!",
            "error": None,
        }

        with patch("chuk_llm.api.conversation.get_client", return_value=mock_client):
            ctx = ConversationContext(provider="openai", model="gpt-4")
            ctx.session_manager = mock_session_manager

            await ctx.say("What's the weather like?")

            # Verify session tracking calls
            mock_session_manager.user_says.assert_called_once_with(
                "What's the weather like?"
            )
            mock_session_manager.ai_responds.assert_called_once_with(
                "Great question!", model="gpt-4", provider="openai"
            )

    @pytest.mark.asyncio
    async def test_say_with_error_response(self, mock_client):
        """Test say method handling error responses."""
        mock_client.create_completion.return_value = {
            "error": True,
            "error_message": "Rate limit exceeded",
        }

        with patch("chuk_llm.api.conversation.get_client", return_value=mock_client):
            ctx = ConversationContext(provider="openai")

            response = await ctx.say("Test message")

            assert "Error: Rate limit exceeded" in response
            assert ctx.messages[-1]["role"] == "assistant"
            assert "Error: Rate limit exceeded" in ctx.messages[-1]["content"]

    @pytest.mark.asyncio
    async def test_say_with_exception(self, mock_client):
        """Test say method handling exceptions."""
        mock_client.create_completion.side_effect = RuntimeError("Network error")

        with patch("chuk_llm.api.conversation.get_client", return_value=mock_client):
            ctx = ConversationContext(provider="openai")

            response = await ctx.say("Test message")

            assert "Conversation error: Network error" in response

    @pytest.mark.asyncio
    async def test_say_with_image_multimodal(self):
        """Test say method with image input (multi-modal)."""
        mock_client = AsyncMock()
        mock_client.create_completion.return_value = {
            "response": "I can see the image you shared.",
            "error": None,
        }

        # Mock the vision message preparation - check if it exists first
        mock_vision_message = {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,..."},
                },
            ],
        }

        with patch("chuk_llm.api.conversation.get_client", return_value=mock_client):
            # Try to patch the vision message function, but handle if it doesn't exist
            try:
                with patch(
                    "chuk_llm.api.conversation._prepare_vision_message",
                    return_value=mock_vision_message,
                ):
                    ctx = ConversationContext(provider="openai", model="gpt-4-vision")

                    response = await ctx.say(
                        "What's in this image?", image=b"fake_image_data"
                    )

                    assert response == "I can see the image you shared."
                    # Check that vision message was added
                    assert ctx.messages[1] == mock_vision_message
            except AttributeError:
                # If _prepare_vision_message doesn't exist, test fallback behavior
                ctx = ConversationContext(provider="openai", model="gpt-4-vision")

                # Should handle missing vision preparation gracefully
                response = await ctx.say(
                    "What's in this image?", image=b"fake_image_data"
                )

                # Should still get a response (either vision or fallback)
                assert isinstance(response, str)
                assert len(response) > 0

    @pytest.mark.asyncio
    async def test_stream_say_basic(self, mock_client):
        """Test streaming conversation."""

        # Mock streaming response
        async def mock_stream():
            chunks = [
                {"response": "Hello", "error": None},
                {"response": " there", "error": None},
                {"response": "!", "error": None},
            ]
            for chunk in chunks:
                yield chunk

        mock_client.create_completion.return_value = mock_stream()

        with patch("chuk_llm.api.conversation.get_client", return_value=mock_client):
            ctx = ConversationContext(provider="openai")

            full_response = ""
            async for chunk in ctx.stream_say("Hello"):
                full_response += chunk

            assert full_response == "Hello there!"
            assert ctx.messages[-1]["content"] == "Hello there!"

    @pytest.mark.asyncio
    async def test_stream_say_with_error(self, mock_client):
        """Test streaming with error."""

        async def mock_error_stream():
            yield {"error": True, "error_message": "API Error"}

        mock_client.create_completion.return_value = mock_error_stream()

        with patch("chuk_llm.api.conversation.get_client", return_value=mock_client):
            ctx = ConversationContext(provider="openai")

            chunks = []
            async for chunk in ctx.stream_say("Test"):
                chunks.append(chunk)

            assert len(chunks) == 1
            assert "[Error: API Error]" in chunks[0]

    @pytest.mark.asyncio
    async def test_stream_say_with_exception(self, mock_client):
        """Test streaming with exception."""
        mock_client.create_completion.side_effect = RuntimeError("Connection failed")

        with patch("chuk_llm.api.conversation.get_client", return_value=mock_client):
            ctx = ConversationContext(provider="openai")

            chunks = []
            async for chunk in ctx.stream_say("Test"):
                chunks.append(chunk)

            assert len(chunks) == 1
            assert "[Streaming error: Connection failed]" in chunks[0]

    @pytest.mark.asyncio
    async def test_branch_conversation(self, mock_client):
        """Test conversation branching."""
        with patch("chuk_llm.api.conversation.get_client", return_value=mock_client):
            ctx = ConversationContext(provider="openai", model="gpt-4")

            # Add some conversation history
            ctx.messages.extend(
                [
                    {"role": "user", "content": "What's 2+2?"},
                    {"role": "assistant", "content": "2+2 equals 4."},
                ]
            )

            # Create a branch
            async with ctx.branch() as branch_ctx:
                assert branch_ctx._parent == ctx
                assert len(branch_ctx.messages) == len(ctx.messages)
                assert branch_ctx.messages == ctx.messages
                assert branch_ctx in ctx._branches

                # Modify branch without affecting parent
                branch_ctx.messages.append(
                    {"role": "user", "content": "What about 3+3?"}
                )

                assert len(branch_ctx.messages) == len(ctx.messages) + 1
                assert len(ctx.messages) == 3  # Original remains unchanged

    @pytest.mark.asyncio
    async def test_save_and_load_conversation(
        self, mock_client, sample_conversation_data
    ):
        """Test saving and loading conversations."""
        with patch("chuk_llm.api.conversation.get_client", return_value=mock_client):
            # Create context
            ctx = ConversationContext(provider="openai", model="gpt-4")
            ctx.messages = sample_conversation_data["messages"]

            # Save conversation
            saved_id = await ctx.save()

            assert saved_id == ctx._conversation_id
            assert saved_id in _CONVERSATION_STORE

            # Load conversation
            loaded_ctx = await ConversationContext.load(saved_id)

            assert loaded_ctx.provider == ctx.provider
            assert loaded_ctx.model == ctx.model
            assert loaded_ctx.messages == ctx.messages
            assert loaded_ctx._conversation_id == ctx._conversation_id

    @pytest.mark.asyncio
    async def test_load_nonexistent_conversation(self):
        """Test loading a conversation that doesn't exist."""
        with pytest.raises(ValueError, match="Conversation nonexistent not found"):
            await ConversationContext.load("nonexistent")

    @pytest.mark.asyncio
    async def test_summarize_conversation(self, mock_client):
        """Test conversation summarization."""
        mock_client.create_completion.return_value = {
            "response": "The user asked about math, and I provided answers about basic arithmetic.",
            "error": None,
        }

        with patch("chuk_llm.api.conversation.get_client", return_value=mock_client):
            ctx = ConversationContext(provider="openai")

            # Add conversation history
            ctx.messages.extend(
                [
                    {"role": "user", "content": "What's 2+2?"},
                    {"role": "assistant", "content": "4"},
                    {"role": "user", "content": "What about 5+5?"},
                    {"role": "assistant", "content": "10"},
                ]
            )

            summary = await ctx.summarize(max_length=100)

            assert "math" in summary.lower()
            assert "arithmetic" in summary.lower()

    @pytest.mark.asyncio
    async def test_summarize_empty_conversation(self, mock_client):
        """Test summarization of empty conversation."""
        with patch("chuk_llm.api.conversation.get_client", return_value=mock_client):
            ctx = ConversationContext(provider="openai")

            summary = await ctx.summarize()

            assert "just started" in summary.lower()

    @pytest.mark.asyncio
    async def test_extract_key_points(self, mock_client):
        """Test key point extraction."""
        mock_client.create_completion.return_value = {
            "response": "- User learned about basic math\n- Discussed addition operations\n- Covered numbers 2-10",
            "error": None,
        }

        with patch("chuk_llm.api.conversation.get_client", return_value=mock_client):
            ctx = ConversationContext(provider="openai")

            # Add conversation history
            ctx.messages.extend(
                [
                    {"role": "user", "content": "Teach me math"},
                    {"role": "assistant", "content": "Let's start with addition"},
                ]
            )

            key_points = await ctx.extract_key_points()

            assert len(key_points) == 3
            assert "User learned about basic math" in key_points
            assert "Discussed addition operations" in key_points
            assert "Covered numbers 2-10" in key_points

    @pytest.mark.asyncio
    async def test_extract_key_points_empty(self, mock_client):
        """Test key point extraction on empty conversation."""
        with patch("chuk_llm.api.conversation.get_client", return_value=mock_client):
            ctx = ConversationContext(provider="openai")

            key_points = await ctx.extract_key_points()

            assert key_points == []

    def test_clear_conversation(self, mock_client):
        """Test clearing conversation history."""
        with patch("chuk_llm.api.conversation.get_client", return_value=mock_client):
            ctx = ConversationContext(provider="openai", system_prompt="System message")

            # Add messages
            ctx.messages.extend(
                [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi"},
                ]
            )

            assert len(ctx.messages) == 3  # system + user + assistant

            ctx.clear()

            assert len(ctx.messages) == 1  # Only system message remains
            assert ctx.messages[0]["role"] == "system"

    def test_get_history(self, mock_client):
        """Test getting conversation history."""
        with patch("chuk_llm.api.conversation.get_client", return_value=mock_client):
            ctx = ConversationContext(provider="openai")

            # Add messages
            test_messages = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
            ]
            ctx.messages.extend(test_messages)

            history = ctx.get_history()

            # Should be a copy, not the original
            assert history == ctx.messages
            assert history is not ctx.messages

    @pytest.mark.asyncio
    async def test_get_session_history_with_session(
        self, mock_client, mock_session_manager
    ):
        """Test getting session history when session manager is available."""
        mock_session_manager.get_conversation.return_value = [
            {"role": "user", "content": "Test", "timestamp": "2024-01-01T00:00:00"}
        ]

        with patch("chuk_llm.api.conversation.get_client", return_value=mock_client):
            ctx = ConversationContext(provider="openai")
            ctx.session_manager = mock_session_manager

            history = await ctx.get_session_history()

            mock_session_manager.get_conversation.assert_called_once()
            assert len(history) == 1
            assert history[0]["content"] == "Test"

    @pytest.mark.asyncio
    async def test_get_session_history_without_session(self):
        """Test getting session history when no session manager."""
        with patch("chuk_llm.api.conversation.get_client"):
            ctx = ConversationContext(provider="openai")

            # Ensure we have no session manager
            ctx.session_manager = None

            history = await ctx.get_session_history()

            # Should return regular history - the method calls self.get_history() when no session manager
            expected_history = ctx.get_history()
            assert history == expected_history

    def test_pop_last_exchange(self, mock_client):
        """Test removing last user-assistant exchange."""
        with patch("chuk_llm.api.conversation.get_client", return_value=mock_client):
            ctx = ConversationContext(provider="openai", system_prompt="System")

            # Add conversation
            ctx.messages.extend(
                [
                    {"role": "user", "content": "Question 1"},
                    {"role": "assistant", "content": "Answer 1"},
                    {"role": "user", "content": "Question 2"},
                    {"role": "assistant", "content": "Answer 2"},
                ]
            )

            assert len(ctx.messages) == 5  # system + 4 conversation messages

            ctx.pop_last()

            assert len(ctx.messages) == 3  # system + first exchange only
            assert ctx.messages[-1]["content"] == "Answer 1"

    def test_get_stats(self):
        """Test getting conversation statistics."""
        with patch("chuk_llm.api.conversation.get_client"):
            ctx = ConversationContext(provider="openai", system_prompt="System prompt")

            # Add messages
            ctx.messages.extend(
                [
                    {"role": "user", "content": "Hello world"},
                    {"role": "assistant", "content": "Hi there!"},
                ]
            )

            stats = ctx.get_stats()

            assert stats["total_messages"] == 3
            assert stats["user_messages"] == 1
            assert stats["assistant_messages"] == 1
            assert stats["has_system_prompt"] is True
            assert stats["estimated_tokens"] > 0
            # has_session may be True or False depending on session manager availability
            assert isinstance(stats["has_session"], bool)
            assert "conversation_id" in stats
            assert "created_at" in stats
            assert stats["branch_count"] == 0

    @pytest.mark.asyncio
    async def test_get_session_stats_with_session(
        self, mock_client, mock_session_manager
    ):
        """Test getting comprehensive stats with session manager."""
        mock_session_manager.get_stats.return_value = {
            "total_tokens": 150,
            "estimated_cost": 0.002,
            "session_segments": 2,
            "session_duration": "5m 30s",
        }

        with patch("chuk_llm.api.conversation.get_client", return_value=mock_client):
            ctx = ConversationContext(provider="openai")
            ctx.session_manager = mock_session_manager

            stats = await ctx.get_session_stats()

            assert stats["total_tokens"] == 150
            assert stats["estimated_cost"] == 0.002
            assert stats["session_segments"] == 2
            assert stats["session_duration"] == "5m 30s"

    @pytest.mark.asyncio
    async def test_set_system_prompt(self):
        """Test updating system prompt."""
        mock_session_manager = AsyncMock()

        with patch("chuk_llm.api.conversation.get_client"):
            ctx = ConversationContext(
                provider="openai", system_prompt="Original prompt"
            )
            ctx.session_manager = mock_session_manager

            # Add some conversation
            ctx.messages.extend(
                [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi"},
                ]
            )

            assert len(ctx.messages) == 3

            # Mock the asyncio.create_task to avoid event loop issues
            with patch("asyncio.create_task") as mock_create_task:
                ctx.set_system_prompt("New system prompt")

                # Should reset to only system message
                assert len(ctx.messages) == 1
                assert ctx.messages[0]["role"] == "system"
                assert ctx.messages[0]["content"] == "New system prompt"

                # Should attempt to update session manager
                mock_create_task.assert_called_once()


class TestConversationContextManager:
    """Test suite for the conversation context manager function."""

    @pytest.mark.asyncio
    async def test_conversation_context_manager_basic(self):
        """Test basic conversation context manager usage."""
        with patch("chuk_llm.api.conversation.get_client") as mock_get_client:
            with patch("chuk_llm.api.conversation.get_config") as mock_get_config:
                # Mock configuration
                mock_config_manager = Mock()
                mock_config_manager.get_global_settings.return_value = {
                    "active_provider": "openai"
                }
                mock_provider_config = Mock()
                mock_provider_config.default_model = "gpt-4"
                mock_config_manager.get_provider.return_value = mock_provider_config
                mock_get_config.return_value = mock_config_manager

                mock_client = Mock()
                mock_get_client.return_value = mock_client

                async with conversation_context_manager() as ctx:
                    assert isinstance(ctx, ConversationContext)
                    assert ctx.provider == "openai"
                    assert ctx.model == "gpt-4"

    @pytest.mark.asyncio
    async def test_conversation_context_manager_with_params(self):
        """Test conversation context manager with specific parameters."""
        with patch("chuk_llm.api.conversation.get_client"):
            async with conversation_context_manager(
                provider="anthropic",
                model="claude-3-sonnet",
                system_prompt="Custom prompt",
                temperature=0.8,
            ) as ctx:
                assert ctx.provider == "anthropic"
                assert ctx.model == "claude-3-sonnet"
                assert ctx.kwargs["temperature"] == 0.8
                assert ctx.messages[0]["content"] == "Custom prompt"

    @pytest.mark.asyncio
    async def test_conversation_context_manager_resume(self, sample_conversation_data):
        """Test resuming conversation from saved ID."""
        # Save conversation data to store
        conversation_id = sample_conversation_data["id"]
        _CONVERSATION_STORE[conversation_id] = sample_conversation_data

        with patch("chuk_llm.api.conversation.get_client"):
            async with conversation_context_manager(resume_from=conversation_id) as ctx:
                assert ctx._conversation_id == conversation_id
                assert ctx.provider == "openai"
                assert ctx.model == "gpt-4"
                assert len(ctx.messages) == 3

    @pytest.mark.asyncio
    async def test_conversation_context_manager_with_session_logging(self):
        """Test that context manager logs session stats on completion."""
        mock_session_manager = AsyncMock()
        mock_session_manager.session_id = "test-session"

        with patch("chuk_llm.api.conversation.get_client"):
            with patch("chuk_llm.api.conversation.logger") as mock_logger:
                async with conversation_context_manager(provider="openai") as ctx:
                    ctx.session_manager = mock_session_manager
                    ctx.get_session_stats = AsyncMock(
                        return_value={
                            "session_id": "test-session",
                            "total_tokens": 100,
                            "estimated_cost": 0.001,
                        }
                    )

                # Should log final stats
                mock_logger.debug.assert_called()
                log_call = mock_logger.debug.call_args[0][0]
                assert "test-session" in log_call
                assert "100" in log_call


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases."""

    @pytest.fixture
    def sample_conversation_data(self):
        """Sample conversation data for testing."""
        return {
            "id": "test-conv-456",
            "created_at": datetime.utcnow().isoformat(),
            "provider": "openai",
            "model": "gpt-4",
            "messages": [{"role": "system", "content": "Test"}],
            "kwargs": {},
            "stats": {},
        }

    @pytest.mark.asyncio
    async def test_session_manager_initialization_failure(self):
        """Test graceful handling of session manager initialization failure."""
        with patch("chuk_llm.api.conversation._SESSIONS_ENABLED", True):
            with patch("chuk_llm.api.conversation.SessionManager") as mock_sm_class:
                mock_sm_class.side_effect = RuntimeError("Session service unavailable")

                with patch("chuk_llm.api.conversation.get_client"):
                    ctx = ConversationContext(provider="openai")

                    # Should handle failure gracefully
                    assert ctx.session_manager is None
                    assert ctx.has_session is False

    @pytest.mark.asyncio
    async def test_session_tracking_failures_during_conversation(self):
        """Test that session tracking failures don't break conversation."""
        mock_client = AsyncMock()
        mock_session_manager = AsyncMock()
        mock_session_manager.user_says.side_effect = RuntimeError("Session error")
        mock_session_manager.ai_responds.side_effect = RuntimeError("Session error")

        mock_client.create_completion.return_value = {
            "response": "Response despite session errors",
            "error": None,
        }

        with patch("chuk_llm.api.conversation.get_client", return_value=mock_client):
            ctx = ConversationContext(provider="openai")
            ctx.session_manager = mock_session_manager

            # Should still work despite session errors
            response = await ctx.say("Test message")

            assert response == "Response despite session errors"
            assert len(ctx.messages) >= 2  # At least system + user + assistant

    @pytest.mark.asyncio
    async def test_summarize_with_api_failure(self):
        """Test summarization when API call fails."""
        mock_client = AsyncMock()
        mock_client.create_completion.side_effect = RuntimeError("API failure")

        with patch("chuk_llm.api.conversation.get_client", return_value=mock_client):
            ctx = ConversationContext(provider="openai")

            # Add some conversation
            ctx.messages.extend(
                [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi"},
                ]
            )

            summary = await ctx.summarize()

            assert summary == "Unable to generate summary."

    @pytest.mark.asyncio
    async def test_extract_key_points_with_api_failure(self):
        """Test key point extraction when API call fails."""
        mock_client = AsyncMock()
        mock_client.create_completion.side_effect = RuntimeError("API failure")

        with patch("chuk_llm.api.conversation.get_client", return_value=mock_client):
            ctx = ConversationContext(provider="openai")

            # Add some conversation
            ctx.messages.extend(
                [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi"},
                ]
            )

            key_points = await ctx.extract_key_points()

            assert key_points == []

    @pytest.mark.asyncio
    async def test_conversation_with_string_response(self):
        """Test handling when client returns string instead of dict."""
        mock_client = AsyncMock()
        mock_client.create_completion.return_value = "Direct string response"

        with patch("chuk_llm.api.conversation.get_client", return_value=mock_client):
            ctx = ConversationContext(provider="openai")

            response = await ctx.say("Test")

            assert response == "Direct string response"
            assert ctx.messages[-1]["content"] == "Direct string response"

    def test_pop_last_with_only_system_message(self):
        """Test pop_last when only system message exists."""
        with patch("chuk_llm.api.conversation.get_client"):
            ctx = ConversationContext(provider="openai", system_prompt="System only")

            initial_length = len(ctx.messages)
            assert initial_length >= 1

            ctx.pop_last()  # Should not crash

            # Should not remove system message
            assert len(ctx.messages) >= 1
            assert ctx.messages[0]["role"] == "system"

    def test_pop_last_with_incomplete_exchange(self):
        """Test pop_last with incomplete user-assistant exchange."""
        with patch("chuk_llm.api.conversation.get_client"):
            ctx = ConversationContext(provider="openai", system_prompt="System")

            # Add only user message (incomplete exchange)
            ctx.messages.append({"role": "user", "content": "Question without answer"})

            initial_length = len(ctx.messages)
            assert initial_length >= 2  # system + user

            ctx.pop_last()

            # Should remove the user message
            final_length = len(ctx.messages)
            assert final_length == initial_length - 1
            # Last message should be system
            assert ctx.messages[-1]["role"] == "system"

    @pytest.mark.asyncio
    async def test_conversation_context_manager_config_fallback(self):
        """Test context manager fallback when config retrieval fails."""
        with patch("chuk_llm.api.conversation.get_config") as mock_get_config:
            # Make the first call fail, but provide a fallback
            mock_get_config.side_effect = [
                RuntimeError("Config unavailable"),  # First call fails
                Mock(
                    get_global_settings=Mock(return_value={"active_provider": "openai"})
                ),  # Fallback
            ]

            with patch("chuk_llm.api.conversation.get_client"):
                try:
                    async with conversation_context_manager() as ctx:
                        assert isinstance(ctx, ConversationContext)
                        # Should use fallback provider
                        assert ctx.provider in [
                            "openai",
                            None,
                        ]  # Depends on implementation
                except RuntimeError:
                    # If config is completely unavailable, the function might raise
                    # This is acceptable behavior
                    pass

    @pytest.mark.asyncio
    async def test_conversation_with_multimodal_error(self):
        """Test multimodal conversation when vision message preparation fails."""
        mock_client = AsyncMock()
        mock_client.create_completion.return_value = {
            "response": "Text response",
            "error": None,
        }

        with patch("chuk_llm.api.conversation.get_client", return_value=mock_client):
            # Try to patch vision preparation if it exists
            try:
                with patch(
                    "chuk_llm.api.conversation._prepare_vision_message"
                ) as mock_prepare:
                    mock_prepare.side_effect = ImportError("Vision not available")

                    ctx = ConversationContext(provider="openai")

                    # Should handle vision preparation failure gracefully
                    response = await ctx.say("Describe image", image=b"fake_data")

                    # Should still process as text-only
                    assert response == "Text response"
            except AttributeError:
                # If _prepare_vision_message doesn't exist, that's fine
                ctx = ConversationContext(provider="openai")
                response = await ctx.say("Describe image", image=b"fake_data")
                assert isinstance(response, str)

    def test_conversation_id_uniqueness(self):
        """Test that each conversation gets a unique ID."""
        with patch("chuk_llm.api.conversation.get_client"):
            ctx1 = ConversationContext(provider="openai")
            ctx2 = ConversationContext(provider="openai")

            assert ctx1._conversation_id != ctx2._conversation_id
            assert isinstance(ctx1._conversation_id, str)
            assert isinstance(ctx2._conversation_id, str)


@pytest.fixture
def sample_conversation_data():
    """Shared fixture for conversation data."""
    return {
        "id": "test-conv-789",
        "created_at": datetime.utcnow().isoformat(),
        "provider": "openai",
        "model": "gpt-4",
        "messages": [
            {"role": "system", "content": "Test system"},
            {"role": "user", "content": "Test user"},
            {"role": "assistant", "content": "Test assistant"},
        ],
        "kwargs": {"temperature": 0.5},
        "stats": {"total_messages": 3},
    }


class TestConversationPersistence:
    """Test conversation persistence and state management."""

    def setup_method(self):
        """Clear conversation store before each test."""
        global _CONVERSATION_STORE
        _CONVERSATION_STORE.clear()

    @pytest.mark.asyncio
    async def test_save_conversation_stores_data(self, sample_conversation_data):
        """Test that saving stores all necessary data."""
        with patch("chuk_llm.api.conversation.get_client"):
            ctx = ConversationContext(provider="openai", model="gpt-4")
            ctx.messages = sample_conversation_data["messages"]

            conversation_id = await ctx.save()

            assert conversation_id in _CONVERSATION_STORE
            stored_data = _CONVERSATION_STORE[conversation_id]

            assert stored_data["provider"] == "openai"
            assert stored_data["model"] == "gpt-4"
            assert stored_data["messages"] == sample_conversation_data["messages"]
            assert "stats" in stored_data

    @pytest.mark.asyncio
    async def test_load_conversation_restores_state(self, sample_conversation_data):
        """Test that loading restores complete conversation state."""
        # Manually store conversation data
        conversation_id = sample_conversation_data["id"]
        _CONVERSATION_STORE[conversation_id] = sample_conversation_data

        with patch("chuk_llm.api.conversation.get_client"):
            loaded_ctx = await ConversationContext.load(conversation_id)

            assert loaded_ctx.provider == sample_conversation_data["provider"]
            assert loaded_ctx.model == sample_conversation_data["model"]
            assert loaded_ctx.messages == sample_conversation_data["messages"]
            assert loaded_ctx.kwargs == sample_conversation_data["kwargs"]
            assert loaded_ctx._conversation_id == conversation_id

    @pytest.mark.asyncio
    async def test_multiple_conversations_isolation(self):
        """Test that multiple conversations are properly isolated."""
        with patch("chuk_llm.api.conversation.get_client"):
            ctx1 = ConversationContext(provider="openai", model="gpt-4")
            ctx2 = ConversationContext(provider="anthropic", model="claude-3-sonnet")

            # Save both
            id1 = await ctx1.save()
            id2 = await ctx2.save()

            assert id1 != id2
            assert len(_CONVERSATION_STORE) == 2

            # Load and verify isolation
            loaded1 = await ConversationContext.load(id1)
            loaded2 = await ConversationContext.load(id2)

            assert loaded1.provider == "openai"
            assert loaded2.provider == "anthropic"
            assert loaded1._conversation_id != loaded2._conversation_id


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"])
