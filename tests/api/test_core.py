"""
Comprehensive pytest tests for chuk_llm/api/core.py

This module tests:
- ask() function with all parameters and configurations
- stream() function with streaming responses
- Session management integration
- Multi-provider support
- Tool usage and JSON mode
- Configuration resolution and validation
- Error handling and fallbacks
- Utility functions

Run with:
    pytest tests/api/test_core.py -v
    pytest tests/api/test_core.py -v --tb=short
    pytest tests/api/test_core.py::TestAskFunction::test_basic_ask -v
"""

from unittest.mock import AsyncMock, MagicMock, Mock, call, patch

import pytest

from chuk_llm.core.enums import MessageRole

# Import the module under test
from chuk_llm.api import core
from chuk_llm.api.core import (
    _add_json_instruction_to_messages,
    _build_messages,
    _get_session_manager,
    ask,
    ask_json,
    disable_sessions,
    enable_sessions,
    get_current_session_id,
    get_session_history,
    get_session_stats,
    multi_provider_ask,
    quick_ask,
    reset_session,
    stream,
    validate_request,
)


class TestAskFunction:
    """Test suite for the ask() function."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock client for testing."""
        client = AsyncMock()
        client.create_completion = AsyncMock()
        return client

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        return {
            "provider": "openai",
            "model": "gpt-4",
            "api_key": "sk-test123",
            "api_base": "https://api.openai.com/v1",
            "system_prompt": "You are a helpful assistant",
            "temperature": 0.7,
            "max_tokens": 1000,
        }

    @pytest.fixture
    def mock_session_manager(self):
        """Create a mock session manager."""
        session_manager = AsyncMock()
        session_manager.session_id = "test-session-123"
        session_manager.user_says = AsyncMock()
        session_manager.ai_responds = AsyncMock()
        session_manager.update_system_prompt = AsyncMock()
        session_manager.tool_used = AsyncMock()
        return session_manager

    @pytest.fixture
    def setup_mocks(self, mock_client, mock_config):
        """Setup common mocks for testing."""
        with patch("chuk_llm.api.core.get_current_config", return_value=mock_config):
            with patch("chuk_llm.api.core.get_client", return_value=mock_client):
                with patch("chuk_llm.api.core.get_config") as mock_get_config:
                    # Mock config manager
                    mock_config_manager = Mock()
                    mock_config_manager.get_provider.return_value = Mock(
                        default_model="gpt-4", api_base="https://api.openai.com/v1"
                    )
                    mock_config_manager.get_api_key.return_value = "sk-test123"
                    mock_config_manager.supports_feature.return_value = True
                    mock_config_manager.get_global_aliases.return_value = {}
                    # Mock _ensure_model_available to return the model unchanged
                    mock_config_manager._ensure_model_available.return_value = "gpt-4"
                    mock_get_config.return_value = mock_config_manager

                    with patch("chuk_llm.api.core.ConfigValidator") as mock_validator:
                        mock_validator.validate_request_compatibility.return_value = (
                            True,
                            [],
                        )
                        yield mock_client, mock_config_manager, mock_validator

    @pytest.mark.asyncio
    async def test_basic_ask(self, setup_mocks):
        """Test basic ask functionality."""
        mock_client, mock_config_manager, mock_validator = setup_mocks

        # Mock response
        mock_client.create_completion.return_value = {
            "response": "Hello! How can I help you today?",
            "error": None,
        }

        response = await ask("Hello")

        assert response == "Hello! How can I help you today?"
        mock_client.create_completion.assert_called_once()

        # Verify call arguments
        call_args = mock_client.create_completion.call_args[1]
        assert "messages" in call_args
        assert len(call_args["messages"]) == 2  # system + user
        # Messages are Pydantic objects, use attribute access
        assert call_args["messages"][0].role == "system"
        assert call_args["messages"][1].role == "user"
        assert call_args["messages"][1].content == "Hello"

    @pytest.mark.asyncio
    async def test_ask_with_provider_override(self, mock_client, mock_config):
        """Test ask with provider override."""
        with patch("chuk_llm.api.core.get_current_config", return_value=mock_config):
            with patch("chuk_llm.api.core.get_client", return_value=mock_client):
                with patch("chuk_llm.api.core.get_config") as mock_get_config:
                    # Mock different provider config
                    mock_config_manager = Mock()
                    anthropic_provider = Mock()
                    anthropic_provider.default_model = "claude-3-sonnet"
                    anthropic_provider.api_base = "https://api.anthropic.com"
                    mock_config_manager.get_provider.return_value = anthropic_provider
                    mock_config_manager.get_api_key.return_value = "sk-ant-test"
                    mock_config_manager.supports_feature.return_value = True
                    mock_get_config.return_value = mock_config_manager

                    with patch("chuk_llm.api.core.ConfigValidator") as mock_validator:
                        mock_validator.validate_request_compatibility.return_value = (
                            True,
                            [],
                        )

                        mock_client.create_completion.return_value = {
                            "response": "Anthropic response"
                        }

                        response = await ask("Test", provider="anthropic")

                        assert response == "Anthropic response"

                        # Verify client was called with anthropic settings
                        patch("chuk_llm.api.core.get_client").start()
                        # The get_client call should use anthropic provider

    @pytest.mark.asyncio
    async def test_ask_with_all_parameters(self, setup_mocks):
        """Test ask with all possible parameters."""
        mock_client, mock_config_manager, mock_validator = setup_mocks

        mock_client.create_completion.return_value = {
            "response": "Full parameter response"
        }

        tools = [{"type": "function", "function": {"name": "test_tool"}}]

        response = await ask(
            "Test prompt",
            provider="openai",
            model="gpt-4",
            system_prompt="Custom system prompt",
            temperature=0.9,
            max_tokens=2000,
            tools=tools,
            json_mode=True,
            context="Additional context",
            previous_messages=[{"role": "user", "content": "Previous question"}],
            custom_param="test",
        )

        # With tools provided, response should be a dict
        assert isinstance(response, dict)
        assert response["response"] == "Full parameter response"
        assert "tool_calls" in response

        # Verify call arguments contain all parameters
        call_args = mock_client.create_completion.call_args[1]
        assert call_args["temperature"] == 0.9
        assert call_args["max_tokens"] == 2000
        assert "tools" in call_args
        assert call_args["custom_param"] == "test"

    @pytest.mark.asyncio
    async def test_ask_with_session_tracking(self, setup_mocks, mock_session_manager):
        """Test ask with session tracking enabled."""
        mock_client, _, _ = setup_mocks

        mock_client.create_completion.return_value = {
            "response": "Session tracked response"
        }

        with patch(
            "chuk_llm.api.core._get_session_manager", return_value=mock_session_manager
        ):
            response = await ask("Test with session")

            assert response == "Session tracked response"

            # Verify session tracking calls
            mock_session_manager.user_says.assert_called_once_with("Test with session")
            mock_session_manager.ai_responds.assert_called_once_with(
                "Session tracked response", model="gpt-4", provider="openai"
            )

    @pytest.mark.asyncio
    async def test_ask_with_tools_parameter(
        self, setup_mocks, mock_session_manager
    ):
        """Test ask with tools that get called."""
        mock_client, _, _ = setup_mocks

        # Mock response with tool calls
        mock_client.create_completion.return_value = {
            "response": "I used a tool to help you",
            "tool_calls": [
                {
                    "name": "search_function",
                    "arguments": {"query": "test"},
                    "result": {"status": "success"},
                }
            ],
        }

        tools = [{"type": "function", "function": {"name": "search_function"}}]

        with patch(
            "chuk_llm.api.core._get_session_manager", return_value=mock_session_manager
        ):
            response = await ask("Search for something", tools=tools)

            # With tools provided, response should be a dict
            assert isinstance(response, dict)
            assert response["response"] == "I used a tool to help you"
            assert "tool_calls" in response

            # Verify tool usage was tracked
            mock_session_manager.tool_used.assert_called_once_with(
                tool_name="search_function",
                arguments={"query": "test"},
                result={"status": "success"},
            )

    @pytest.mark.asyncio
    async def test_ask_with_json_mode_openai(self, setup_mocks):
        """Test ask with JSON mode for OpenAI."""
        mock_client, mock_config_manager, _ = setup_mocks

        mock_client.create_completion.return_value = {
            "response": '{"result": "json response"}'
        }

        response = await ask("Give me JSON", json_mode=True, provider="openai")

        assert response == '{"result": "json response"}'

        # Verify JSON mode was set for OpenAI
        call_args = mock_client.create_completion.call_args[1]
        assert call_args.get("response_format") == {"type": "json_object"}

    @pytest.mark.asyncio
    async def test_ask_with_json_mode_gemini(self, setup_mocks):
        """Test ask with JSON mode for Gemini."""
        mock_client, mock_config_manager, _ = setup_mocks

        mock_client.create_completion.return_value = {
            "response": '{"result": "gemini json"}'
        }

        response = await ask("Give me JSON", json_mode=True, provider="gemini")

        assert response == '{"result": "gemini json"}'

        # Verify JSON mode was set for Gemini
        call_args = mock_client.create_completion.call_args[1]
        assert "generation_config" in call_args
        assert (
            call_args["generation_config"]["response_mime_type"] == "application/json"
        )

    @pytest.mark.asyncio
    async def test_ask_with_json_mode_unsupported_provider(self, setup_mocks):
        """Test ask with JSON mode for provider that doesn't support it."""
        mock_client, mock_config_manager, _ = setup_mocks

        # Mock unsupported JSON mode
        mock_config_manager.supports_feature.return_value = False

        mock_client.create_completion.return_value = {
            "response": "JSON instruction added"
        }

        await ask("Give me JSON", json_mode=True, provider="unsupported")

        # Should add JSON instruction to system message
        call_args = mock_client.create_completion.call_args[1]
        system_message = call_args["messages"][0].content
        assert "JSON" in system_message
        assert "valid JSON only" in system_message

    @pytest.mark.asyncio
    async def test_ask_with_error_response(self, setup_mocks):
        """Test ask handling error responses."""
        mock_client, _, _ = setup_mocks

        mock_client.create_completion.return_value = {
            "error": True,
            "error_message": "Rate limit exceeded",
        }

        with pytest.raises(Exception, match="LLM Error: Rate limit exceeded"):
            await ask("Test")

    @pytest.mark.asyncio
    async def test_ask_with_exception(self, setup_mocks):
        """Test ask handling exceptions."""
        mock_client, _, _ = setup_mocks

        mock_client.create_completion.side_effect = RuntimeError("Network error")

        with pytest.raises(RuntimeError, match="Network error"):
            await ask("Test")

    @pytest.mark.asyncio
    async def test_ask_with_string_response(self, setup_mocks):
        """Test ask handling string response instead of dict."""
        mock_client, _, _ = setup_mocks

        mock_client.create_completion.return_value = "Direct string response"

        response = await ask("Test")

        assert response == "Direct string response"

    @pytest.mark.asyncio
    async def test_ask_with_context_and_previous_messages(self, setup_mocks):
        """Test ask with additional context and previous messages."""
        mock_client, _, _ = setup_mocks

        mock_client.create_completion.return_value = {
            "response": "Context aware response"
        }

        previous_messages = [
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"},
        ]

        response = await ask(
            "Follow up question",
            context="Important context information",
            previous_messages=previous_messages,
        )

        assert response == "Context aware response"

        # Verify messages structure
        call_args = mock_client.create_completion.call_args[1]
        messages = call_args["messages"]

        # Should have system (with context), previous messages, and current user message
        # Messages from ask() are Message objects
        assert len(messages) >= 4
        assert (
            "Important context information" in messages[0].content
        )  # Context in system
        assert any(msg.content == "Previous question" for msg in messages)
        assert any(msg.content == "Follow up question" for msg in messages)

    @pytest.mark.asyncio
    async def test_ask_validation_warnings(self, setup_mocks):
        """Test ask with validation warnings but continued execution."""
        mock_client, _, mock_validator = setup_mocks

        # Mock validation issues
        mock_validator.validate_request_compatibility.return_value = (
            False,
            ["Model doesn't support tools", "Temperature too high"],
        )

        mock_client.create_completion.return_value = {
            "response": "Response despite warnings"
        }

        with patch("chuk_llm.api.core.logger") as mock_logger:
            response = await ask("Test")

            assert response == "Response despite warnings"

            # Should log warnings
            mock_logger.warning.assert_has_calls(
                [
                    call("Request compatibility issue: Model doesn't support tools"),
                    call("Request compatibility issue: Temperature too high"),
                ]
            )


class TestStreamFunction:
    """Test suite for the stream() function."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock client for testing."""
        client = AsyncMock()
        client.create_completion = Mock()
        return client

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        return {
            "provider": "openai",
            "model": "gpt-4",
            "api_key": "sk-test123",
            "api_base": "https://api.openai.com/v1",
            "temperature": 0.7,
        }

    @pytest.mark.asyncio
    async def test_basic_streaming(self, mock_client, mock_config):
        """Test basic streaming functionality."""

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

        with patch("chuk_llm.api.core.get_current_config", return_value=mock_config):
            with patch("chuk_llm.api.core.get_client", return_value=mock_client):
                with patch("chuk_llm.api.core.get_config") as mock_get_config:
                    mock_config_manager = Mock()
                    mock_config_manager.supports_feature.return_value = True
                    mock_get_config.return_value = mock_config_manager

                    chunks = []
                    async for chunk in stream("Hello"):
                        chunks.append(chunk)

                    assert chunks == ["Hello", " there", "!"]

    @pytest.mark.asyncio
    async def test_streaming_with_session_tracking(self, mock_client, mock_config):
        """Test streaming with session tracking."""

        async def mock_stream():
            yield {"response": "Streamed response", "error": None}

        mock_client.create_completion.return_value = mock_stream()
        mock_session_manager = AsyncMock()
        mock_session_manager.user_says = AsyncMock()
        mock_session_manager.ai_responds = AsyncMock()

        with patch("chuk_llm.api.core.get_current_config", return_value=mock_config):
            with patch("chuk_llm.api.core.get_client", return_value=mock_client):
                with patch("chuk_llm.api.core.get_config") as mock_get_config:
                    mock_config_manager = Mock()
                    mock_config_manager.supports_feature.return_value = True
                    mock_config_manager.get_global_aliases.return_value = {}
                    mock_config_manager._ensure_model_available.return_value = "gpt-4"
                    mock_get_config.return_value = mock_config_manager

                    with patch(
                        "chuk_llm.api.core._get_session_manager",
                        return_value=mock_session_manager,
                    ):
                        chunks = []
                        async for chunk in stream("Test streaming"):
                            chunks.append(chunk)

                        # Verify session tracking
                        mock_session_manager.user_says.assert_called_once_with(
                            "Test streaming"
                        )
                        mock_session_manager.ai_responds.assert_called_once_with(
                            "Streamed response", model="gpt-4", provider="openai"
                        )

    @pytest.mark.asyncio
    async def test_streaming_fallback_to_non_streaming(self, mock_client, mock_config):
        """Test streaming fallback when provider doesn't support streaming."""
        with patch("chuk_llm.api.core.get_current_config", return_value=mock_config):
            with patch("chuk_llm.api.core.get_client", return_value=mock_client):
                with patch("chuk_llm.api.core.get_config") as mock_get_config:
                    mock_config_manager = Mock()
                    mock_config_manager.supports_feature.return_value = (
                        False  # No streaming support
                    )
                    mock_get_config.return_value = mock_config_manager

                    with patch("chuk_llm.api.core.ask") as mock_ask:
                        mock_ask.return_value = "Non-streaming response"

                        chunks = []
                        async for chunk in stream("Test"):
                            chunks.append(chunk)

                        assert chunks == ["Non-streaming response"]
                        mock_ask.assert_called_once()

    @pytest.mark.asyncio
    async def test_streaming_with_error_chunk(self, mock_client, mock_config):
        """Test streaming with error in chunk."""

        async def mock_error_stream():
            yield {"error": True, "error_message": "API Error"}

        mock_client.create_completion.return_value = mock_error_stream()

        with patch("chuk_llm.api.core.get_current_config", return_value=mock_config):
            with patch("chuk_llm.api.core.get_client", return_value=mock_client):
                with patch("chuk_llm.api.core.get_config") as mock_get_config:
                    mock_config_manager = Mock()
                    mock_config_manager.supports_feature.return_value = True
                    mock_get_config.return_value = mock_config_manager

                    chunks = []
                    async for chunk in stream("Test"):
                        chunks.append(chunk)

                    assert len(chunks) == 1
                    assert "[Error: API Error]" in chunks[0]

    @pytest.mark.asyncio
    async def test_streaming_with_exception(self, mock_client, mock_config):
        """Test streaming with exception."""
        mock_client.create_completion.side_effect = RuntimeError("Connection failed")

        with patch("chuk_llm.api.core.get_current_config", return_value=mock_config):
            with patch("chuk_llm.api.core.get_client", return_value=mock_client):
                with patch("chuk_llm.api.core.get_config") as mock_get_config:
                    mock_config_manager = Mock()
                    mock_config_manager.supports_feature.return_value = True
                    mock_config_manager.get_global_aliases.return_value = {}
                    mock_config_manager._ensure_model_available.return_value = "gpt-4"
                    mock_get_config.return_value = mock_config_manager

                    with patch("chuk_llm.api.core.ask") as mock_ask:
                        # Mock the fallback ask function to return a fallback response
                        mock_ask.return_value = "Fallback response"

                        chunks = []
                        async for chunk in stream("Test"):
                            chunks.append(chunk)

                        # The stream function should fall back to non-streaming mode
                        # when streaming fails, so we expect the fallback response
                        assert len(chunks) == 1
                        assert chunks[0] == "Fallback response"

                        # Verify fallback was called
                        mock_ask.assert_called_once()

    @pytest.mark.asyncio
    async def test_streaming_non_dict_chunks(self, mock_client, mock_config):
        """Test streaming with non-dict chunks."""

        async def mock_string_stream():
            yield "First"
            yield " chunk"
            yield "!"

        mock_client.create_completion.return_value = mock_string_stream()

        with patch("chuk_llm.api.core.get_current_config", return_value=mock_config):
            with patch("chuk_llm.api.core.get_client", return_value=mock_client):
                with patch("chuk_llm.api.core.get_config") as mock_get_config:
                    mock_config_manager = Mock()
                    mock_config_manager.supports_feature.return_value = True
                    mock_get_config.return_value = mock_config_manager

                    chunks = []
                    async for chunk in stream("Test"):
                        chunks.append(chunk)

                    assert chunks == ["First", " chunk", "!"]


class TestSessionManagement:
    """Test session management functions."""

    @pytest.fixture
    def mock_session_manager(self):
        """Create a mock session manager."""
        session_manager = AsyncMock()
        session_manager.session_id = "test-session-456"
        session_manager.get_stats = AsyncMock()
        session_manager.get_conversation = AsyncMock()
        return session_manager

    def test_get_session_manager_creates_instance(self):
        """Test that _get_session_manager creates a session manager."""
        with patch("chuk_llm.api.core._SESSIONS_ENABLED", True):
            with patch("chuk_llm.api.core.SessionManager") as mock_sm_class:
                with patch("chuk_llm.api.core.get_current_config") as mock_config:
                    mock_config.return_value = {"system_prompt": "Test prompt"}

                    mock_instance = Mock()
                    mock_instance.session_id = "new-session"
                    mock_sm_class.return_value = mock_instance

                    # Clear global session manager
                    core._global_session_manager = None

                    result = _get_session_manager()

                    assert result == mock_instance
                    mock_sm_class.assert_called_once_with(
                        system_prompt="Test prompt",
                        infinite_context=True,
                        token_threshold=4000,
                    )

    def test_get_session_manager_when_disabled(self):
        """Test _get_session_manager when sessions are disabled."""
        with patch("chuk_llm.api.core._SESSIONS_ENABLED", False):
            result = _get_session_manager()
            assert result is None

    def test_get_session_manager_handles_exception(self):
        """Test _get_session_manager handles exceptions gracefully."""
        with patch("chuk_llm.api.core._SESSIONS_ENABLED", True):
            with patch("chuk_llm.api.core.SessionManager") as mock_sm_class:
                mock_sm_class.side_effect = RuntimeError("Session service down")

                # Clear global session manager
                core._global_session_manager = None

                result = _get_session_manager()

                assert result is None

    @pytest.mark.asyncio
    async def test_get_session_stats_with_session(self, mock_session_manager):
        """Test get_session_stats with active session."""
        mock_session_manager.get_stats.return_value = {
            "session_id": "test-session-456",
            "total_tokens": 150,
            "estimated_cost": 0.002,
        }

        with patch(
            "chuk_llm.api.core._get_session_manager", return_value=mock_session_manager
        ):
            stats = await get_session_stats(include_all_segments=True)

            assert stats["session_id"] == "test-session-456"
            assert stats["total_tokens"] == 150
            mock_session_manager.get_stats.assert_called_once_with(
                include_all_segments=True
            )

    @pytest.mark.asyncio
    async def test_get_session_stats_without_session(self):
        """Test get_session_stats without active session."""
        with patch("chuk_llm.api.core._get_session_manager", return_value=None):
            stats = await get_session_stats()

            assert stats["session_available"] is False
            assert "No active session" in stats["message"]

    @pytest.mark.asyncio
    async def test_get_session_history_with_session(self, mock_session_manager):
        """Test get_session_history with active session."""
        mock_session_manager.get_conversation.return_value = [
            {"role": "user", "content": "Test", "timestamp": "2024-01-01T00:00:00"}
        ]

        with patch(
            "chuk_llm.api.core._get_session_manager", return_value=mock_session_manager
        ):
            history = await get_session_history(include_all_segments=False)

            assert len(history) == 1
            assert history[0]["content"] == "Test"
            mock_session_manager.get_conversation.assert_called_once_with(
                include_all_segments=False
            )

    @pytest.mark.asyncio
    async def test_get_session_history_without_session(self):
        """Test get_session_history without active session."""
        with patch("chuk_llm.api.core._get_session_manager", return_value=None):
            history = await get_session_history()

            assert history == []

    def test_get_current_session_id_with_session(self, mock_session_manager):
        """Test get_current_session_id with active session."""
        with patch(
            "chuk_llm.api.core._get_session_manager", return_value=mock_session_manager
        ):
            session_id = get_current_session_id()

            assert session_id == "test-session-456"

    def test_get_current_session_id_without_session(self):
        """Test get_current_session_id without active session."""
        with patch("chuk_llm.api.core._get_session_manager", return_value=None):
            session_id = get_current_session_id()

            assert session_id is None

    def test_reset_session(self):
        """Test reset_session function."""
        # Set a global session manager
        core._global_session_manager = Mock()

        reset_session()

        assert core._global_session_manager is None

    def test_disable_sessions(self):
        """Test disable_sessions function."""
        # Set initial state
        core._SESSIONS_ENABLED = True
        core._global_session_manager = Mock()

        disable_sessions()

        assert core._SESSIONS_ENABLED is False
        assert core._global_session_manager is None

    def test_enable_sessions_when_available(self):
        """Test enable_sessions when session manager is available."""
        with patch("chuk_llm.api.core._SESSION_AVAILABLE", True):
            core._SESSIONS_ENABLED = False

            enable_sessions()

            assert core._SESSIONS_ENABLED is True

    def test_enable_sessions_when_not_available(self):
        """Test enable_sessions when session manager is not available."""
        with patch("chuk_llm.api.core._SESSION_AVAILABLE", False):
            core._SESSIONS_ENABLED = False

            with patch("chuk_llm.api.core.logger") as mock_logger:
                enable_sessions()

                assert core._SESSIONS_ENABLED is False
                mock_logger.warning.assert_called_once()


class TestUtilityFunctions:
    """Test utility and convenience functions."""

    @pytest.fixture
    def mock_config(self):
        """Mock configuration."""
        return {
            "provider": "openai",
            "model": "gpt-4o",  # Use gpt-4o which definitely supports tools
            "api_key": "sk-test123",
            "api_base": "https://api.openai.com/v1"
        }

    @pytest.mark.asyncio
    async def test_ask_with_tools_parameter(self, mock_config):
        """Test ask function with tools parameter."""
        tools = [{"type": "function", "function": {"name": "test_tool"}}]

        with (
            patch("chuk_llm.api.core.get_client") as mock_get_client,
            patch("chuk_llm.api.core.get_current_config", return_value=mock_config),
            patch(
                "chuk_llm.api.core.get_current_session_id",
                return_value="session-123",
            ),
        ):
            mock_client = MagicMock()
            mock_client.create_completion = AsyncMock(return_value={"response": "Tool response", "tool_calls": []})
            mock_client.supports_feature = MagicMock(return_value=True)  # Mock to support all features including tools
            mock_get_client.return_value = mock_client

            result = await ask("Use a tool", tools=tools, temperature=0.8)

            # When tools are provided, ask returns a dict
            assert isinstance(result, dict)
            assert result["response"] == "Tool response"
            assert "tool_calls" in result
            mock_client.create_completion.assert_called_once()
            call_args = mock_client.create_completion.call_args[1]
            # Tools may be converted to Pydantic objects, check structure
            assert len(call_args["tools"]) == 1
            assert call_args["tools"][0].function.name == "test_tool"
            assert call_args["temperature"] == 0.8

    @pytest.mark.asyncio
    async def test_ask_json(self):
        """Test ask_json function."""
        with patch("chuk_llm.api.core.ask") as mock_ask:
            mock_ask.return_value = '{"result": "json"}'

            result = await ask_json("Give me JSON", temperature=0.5)

            assert result == '{"result": "json"}'
            mock_ask.assert_called_once_with(
                "Give me JSON", json_mode=True, temperature=0.5
            )

    @pytest.mark.asyncio
    async def test_quick_ask(self):
        """Test quick_ask function."""
        with patch("chuk_llm.api.core.ask") as mock_ask:
            mock_ask.return_value = "Quick response"

            result = await quick_ask("Quick question", provider="anthropic")

            assert result == "Quick response"
            mock_ask.assert_called_once_with("Quick question", provider="anthropic")

    @pytest.mark.asyncio
    async def test_multi_provider_ask(self):
        """Test multi_provider_ask function."""
        with patch("chuk_llm.api.core.ask") as mock_ask:
            # Mock different responses for different providers
            async def mock_ask_func(prompt, provider=None):
                if provider == "openai":
                    return "OpenAI response"
                elif provider == "anthropic":
                    return "Anthropic response"
                else:
                    return "Default response"

            mock_ask.side_effect = mock_ask_func

            result = await multi_provider_ask("Test question", ["openai", "anthropic"])

            assert result["openai"] == "OpenAI response"
            assert result["anthropic"] == "Anthropic response"
            assert len(result) == 2

    @pytest.mark.asyncio
    async def test_multi_provider_ask_with_error(self):
        """Test multi_provider_ask handling errors."""
        with patch("chuk_llm.api.core.ask") as mock_ask:

            async def mock_ask_func(prompt, provider=None):
                if provider == "openai":
                    return "OpenAI response"
                elif provider == "failing":
                    raise RuntimeError("Provider failed")
                else:
                    return "Default response"

            mock_ask.side_effect = mock_ask_func

            result = await multi_provider_ask("Test", ["openai", "failing"])

            assert result["openai"] == "OpenAI response"
            assert "Error:" in result["failing"]

    def test_validate_request(self, mock_config):
        """Test validate_request function."""
        with patch("chuk_llm.api.core.get_current_config", return_value=mock_config):
            with patch("chuk_llm.api.core.ConfigValidator") as mock_validator:
                mock_validator.validate_request_compatibility.return_value = (True, [])

                result = validate_request(
                    "Test prompt",
                    provider="anthropic",
                    model="claude-3-sonnet",
                    tools=[{"type": "function"}],
                )

                assert result["valid"] is True
                assert result["issues"] == []
                assert result["provider"] == "anthropic"
                assert result["model"] == "claude-3-sonnet"

                # Verify the validator was called correctly
                mock_validator.validate_request_compatibility.assert_called_once()

    def test_validate_request_with_issues(self, mock_config):
        """Test validate_request with validation issues."""
        with patch("chuk_llm.api.core.get_current_config", return_value=mock_config):
            with patch("chuk_llm.api.core.ConfigValidator") as mock_validator:
                mock_validator.validate_request_compatibility.return_value = (
                    False,
                    ["Tools not supported", "Streaming unavailable"],
                )

                result = validate_request("Test prompt")

                assert result["valid"] is False
                assert len(result["issues"]) == 2
                assert "Tools not supported" in result["issues"]


class TestMessageBuilding:
    """Test message building and formatting functions."""

    def test_build_messages_basic(self):
        """Test basic message building."""
        with patch("chuk_llm.api.core.get_config") as mock_get_config:
            mock_config_manager = Mock()
            mock_config_manager.supports_feature.return_value = True
            mock_get_config.return_value = mock_config_manager

            messages = _build_messages(
                prompt="Hello",
                system_prompt="You are helpful",
                tools=None,
                provider="openai",
                model="gpt-4",
            )

            assert len(messages) == 2
            assert messages[0]["role"] == "system"
            assert messages[0]["content"] == "You are helpful"
            assert messages[1]["role"] == "user"
            assert messages[1]["content"] == "Hello"

    def test_build_messages_with_tools(self):
        """Test message building with tools."""
        tools = [{"type": "function", "function": {"name": "test_tool"}}]

        with patch("chuk_llm.api.core.get_config") as mock_get_config:
            mock_config_manager = Mock()
            mock_config_manager.supports_feature.return_value = True
            mock_get_config.return_value = mock_config_manager

            with patch(
                "chuk_llm.llm.system_prompt_generator.SystemPromptGenerator"
            ) as mock_gen:
                mock_generator = Mock()
                mock_generator.generate_prompt.return_value = "Tool-aware system prompt"
                mock_gen.return_value = mock_generator

                messages = _build_messages(
                    prompt="Use tools",
                    system_prompt=None,
                    tools=tools,
                    provider="openai",
                    model="gpt-4",
                )

                assert messages[0]["content"] == "Tool-aware system prompt"
                mock_generator.generate_prompt.assert_called_once_with(tools)

    def test_build_messages_with_context(self):
        """Test message building with context."""
        with patch("chuk_llm.api.core.get_config") as mock_get_config:
            mock_config_manager = Mock()
            mock_config_manager.supports_feature.return_value = True
            mock_get_config.return_value = mock_config_manager

            messages = _build_messages(
                prompt="Question",
                system_prompt="Base prompt",
                tools=None,
                provider="openai",
                model="gpt-4",
                context="Important context",
            )

            system_content = messages[0]["content"]
            assert "Base prompt" in system_content
            assert "Important context" in system_content

    def test_build_messages_with_previous_messages(self):
        """Test message building with previous messages."""
        previous_messages = [
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"},
        ]

        with patch("chuk_llm.api.core.get_config") as mock_get_config:
            mock_config_manager = Mock()
            mock_config_manager.supports_feature.return_value = True
            mock_get_config.return_value = mock_config_manager

            messages = _build_messages(
                prompt="Follow up",
                system_prompt="System",
                tools=None,
                provider="openai",
                model="gpt-4",
                previous_messages=previous_messages,
            )

            # Should have system + previous + current user
            assert len(messages) == 4
            assert messages[1]["content"] == "Previous question"
            assert messages[2]["content"] == "Previous answer"
            assert messages[3]["content"] == "Follow up"

    def test_build_messages_no_system_support(self):
        """Test message building for providers without system message support."""
        with patch("chuk_llm.api.core.get_config") as mock_get_config:
            mock_config_manager = Mock()
            mock_config_manager.supports_feature.return_value = (
                False  # No system support
            )
            mock_get_config.return_value = mock_config_manager

            messages = _build_messages(
                prompt="Hello",
                system_prompt="System prompt",
                tools=None,
                provider="no_system",
                model="basic",
            )

            # Should only have user message with system content prepended
            # _build_messages returns dicts
            assert len(messages) == 1
            assert messages[0]["role"] == "user"
            assert "System: System prompt" in messages[0]["content"]
            assert "User: Hello" in messages[0]["content"]

    def test_add_json_instruction_to_messages_with_system(self):
        """Test adding JSON instruction to existing system message."""
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Give me JSON"},
        ]

        _add_json_instruction_to_messages(messages)

        # messages are dicts, not Message objects here
        system_content = messages[0]["content"]
        assert "You are helpful" in system_content
        assert "valid JSON only" in system_content

    def test_add_json_instruction_to_messages_no_system(self):
        """Test adding JSON instruction when no system message exists."""
        messages = [{"role": "user", "content": "Give me JSON"}]

        _add_json_instruction_to_messages(messages)

        # Should add system message at the beginning
        # messages are dicts, not Message objects here
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert "valid JSON only" in messages[0]["content"]


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_ask_with_provider_resolution_failure(self):
        """Test ask when provider resolution fails."""
        mock_config = {
            "provider": "openai",
            "model": "gpt-4",
            "api_key": "sk-test",
            "api_base": None,
        }

        with patch("chuk_llm.api.core.get_current_config", return_value=mock_config):
            with patch("chuk_llm.api.core.get_config") as mock_get_config:
                mock_config_manager = Mock()
                mock_config_manager.get_provider.side_effect = ValueError(
                    "Unknown provider"
                )
                mock_get_config.return_value = mock_config_manager

                with patch("chuk_llm.api.core.ConfigValidator") as mock_validator:
                    mock_validator.validate_request_compatibility.return_value = (
                        True,
                        [],
                    )

                    with patch("chuk_llm.api.core.get_client") as mock_get_client:
                        mock_client = AsyncMock()
                        mock_client.create_completion.return_value = {
                            "response": "Fallback response"
                        }
                        mock_get_client.return_value = mock_client

                        # Should still work with fallback values
                        response = await ask("Test", provider="unknown")

                        assert response == "Fallback response"

    @pytest.mark.asyncio
    async def test_session_tracking_errors_dont_break_functionality(self):
        """Test that session tracking errors don't break core functionality."""
        mock_session_manager = AsyncMock()
        mock_session_manager.user_says.side_effect = RuntimeError("Session error")
        mock_session_manager.ai_responds.side_effect = RuntimeError("Session error")

        mock_config = {
            "provider": "openai",
            "model": "gpt-4",
            "api_key": "sk-test",
            "api_base": None,
        }

        with patch("chuk_llm.api.core.get_current_config", return_value=mock_config):
            with patch(
                "chuk_llm.api.core._get_session_manager",
                return_value=mock_session_manager,
            ):
                with patch("chuk_llm.api.core.get_config") as mock_get_config:
                    mock_config_manager = Mock()
                    mock_config_manager.supports_feature.return_value = True
                    mock_get_config.return_value = mock_config_manager

                    with patch("chuk_llm.api.core.ConfigValidator") as mock_validator:
                        mock_validator.validate_request_compatibility.return_value = (
                            True,
                            [],
                        )

                        with patch("chuk_llm.api.core.get_client") as mock_get_client:
                            mock_client = AsyncMock()
                            mock_client.create_completion.return_value = {
                                "response": "Success despite session errors"
                            }
                            mock_get_client.return_value = mock_client

                            # Should still work despite session errors
                            response = await ask("Test")

                            assert response == "Success despite session errors"

    def test_build_messages_with_tool_import_error(self):
        """Test message building when system prompt generator import fails."""
        tools = [{"type": "function"}]

        with patch("chuk_llm.api.core.get_config") as mock_get_config:
            mock_config_manager = Mock()
            mock_config_manager.supports_feature.return_value = True
            mock_get_config.return_value = mock_config_manager

            # Mock import error
            with patch(
                "chuk_llm.llm.system_prompt_generator.SystemPromptGenerator"
            ) as mock_gen:
                mock_gen.side_effect = ImportError("Module not found")

                messages = _build_messages(
                    prompt="Test",
                    system_prompt=None,
                    tools=tools,
                    provider="openai",
                    model="gpt-4",
                )

                # Should use fallback system prompt
                # _build_messages returns dicts
                assert "function calling tools" in messages[0]["content"].lower()

    def test_get_session_manager_reuses_existing_instance(self):
        """Test that _get_session_manager reuses existing instance."""
        with patch("chuk_llm.api.core._SESSIONS_ENABLED", True):
            # Set up existing session manager
            existing_manager = Mock()
            existing_manager.session_id = "existing-session"
            core._global_session_manager = existing_manager

            result = _get_session_manager()

            # Should return existing instance
            assert result == existing_manager


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"])
