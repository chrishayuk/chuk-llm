"""
Comprehensive tests for LlamaCppLLMClient.

Tests cover:
- Initialization with auto-port finding
- Server lifecycle management
- Health checking
- Completion requests (inherited from OpenAI)
- Context manager usage
- Error handling
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chuk_llm.llm.providers.llamacpp_client import (
    LlamaCppLLMClient,
    _find_available_port,
)
from chuk_llm.llm.providers.llamacpp_server import (
    LlamaCppServerConfig,
    LlamaCppServerManager,
)


class TestFindAvailablePort:
    """Test port finding utility function."""

    def test_find_available_port_default(self):
        """Test finding available port from default start."""
        port = _find_available_port()
        assert isinstance(port, int)
        assert 8080 <= port < 8180

    def test_find_available_port_custom_start(self):
        """Test finding available port from custom start."""
        port = _find_available_port(start_port=9000)
        assert isinstance(port, int)
        assert 9000 <= port < 9100

    @patch("socket.socket")
    def test_find_available_port_all_taken(self, mock_socket):
        """Test error when no ports available."""
        mock_socket.return_value.__enter__.return_value.bind.side_effect = OSError

        with pytest.raises(RuntimeError, match="No available ports found"):
            _find_available_port(start_port=8080, max_attempts=5)


class TestLlamaCppClientInit:
    """Test LlamaCppLLMClient initialization."""

    def test_init_with_valid_model_path(self, tmp_path):
        """Test initialization with valid model file."""
        # Create a temporary model file
        model_file = tmp_path / "test_model.gguf"
        model_file.write_text("fake model data")

        with patch("chuk_llm.llm.providers.llamacpp_client._find_available_port", return_value=8080):
            client = LlamaCppLLMClient(model=model_file)

        assert client.model == str(model_file)
        assert client.api_base == "http://127.0.0.1:8080"
        assert client._server_config.model_path == model_file
        assert client._server_config.port == 8080
        assert not client._server_started

    def test_init_with_custom_port(self, tmp_path):
        """Test initialization with custom port."""
        model_file = tmp_path / "test_model.gguf"
        model_file.write_text("fake model data")

        client = LlamaCppLLMClient(model=model_file, port=9000)

        assert client._server_config.port == 9000
        assert client.api_base == "http://127.0.0.1:9000"

    def test_init_with_custom_host(self, tmp_path):
        """Test initialization with custom host."""
        model_file = tmp_path / "test_model.gguf"
        model_file.write_text("fake model data")

        client = LlamaCppLLMClient(model=model_file, host="0.0.0.0", port=8080)

        assert client._server_config.host == "0.0.0.0"
        assert client.api_base == "http://0.0.0.0:8080"

    def test_init_with_gpu_layers(self, tmp_path):
        """Test initialization with custom GPU layers."""
        model_file = tmp_path / "test_model.gguf"
        model_file.write_text("fake model data")

        client = LlamaCppLLMClient(model=model_file, n_gpu_layers=35, port=8080)

        assert client._server_config.n_gpu_layers == 35

    def test_init_with_context_size(self, tmp_path):
        """Test initialization with custom context size."""
        model_file = tmp_path / "test_model.gguf"
        model_file.write_text("fake model data")

        client = LlamaCppLLMClient(model=model_file, ctx_size=16384, port=8080)

        assert client._server_config.ctx_size == 16384

    def test_init_with_nonexistent_model(self):
        """Test initialization with nonexistent model file."""
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            LlamaCppLLMClient(model="/nonexistent/model.gguf")

    def test_init_with_auto_start_false(self, tmp_path):
        """Test initialization with auto_start=False."""
        model_file = tmp_path / "test_model.gguf"
        model_file.write_text("fake model data")

        client = LlamaCppLLMClient(model=model_file, auto_start=False, port=8080)

        assert not client._auto_start

    def test_detect_provider_name_override(self, tmp_path):
        """Test provider name detection always returns llamacpp."""
        model_file = tmp_path / "test_model.gguf"
        model_file.write_text("fake model data")

        client = LlamaCppLLMClient(model=model_file, port=8080)

        # Should always return llamacpp, even if api_base looks like OpenAI
        assert client._detect_provider_name("https://api.openai.com") == "llamacpp"
        assert client._detect_provider_name(None) == "llamacpp"


class TestLlamaCppClientServerManagement:
    """Test server lifecycle management."""

    @pytest.mark.asyncio
    async def test_ensure_server_started_with_auto_start(self, tmp_path):
        """Test server auto-start on first request."""
        model_file = tmp_path / "test_model.gguf"
        model_file.write_text("fake model data")

        client = LlamaCppLLMClient(model=model_file, auto_start=True, port=8080)

        # Mock the server manager
        client._server_manager.start = AsyncMock()
        # base_url is a property - access it through mock
        with patch.object(type(client._server_manager), 'base_url', new_callable=lambda: property(lambda self: "http://127.0.0.1:8080")):
            # Mock the model info request
            with patch("httpx.AsyncClient") as mock_client:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = {"data": [{"id": "test-model"}]}
                mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)

                await client._ensure_server_started()

        assert client._server_started
        client._server_manager.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_ensure_server_started_already_started(self, tmp_path):
        """Test that server start is not called if already started."""
        model_file = tmp_path / "test_model.gguf"
        model_file.write_text("fake model data")

        client = LlamaCppLLMClient(model=model_file, auto_start=True, port=8080)
        client._server_started = True
        client._server_manager.start = AsyncMock()

        await client._ensure_server_started()

        # Should not call start again
        client._server_manager.start.assert_not_called()

    @pytest.mark.asyncio
    async def test_ensure_server_started_without_auto_start(self, tmp_path):
        """Test error when auto_start=False and server not started."""
        model_file = tmp_path / "test_model.gguf"
        model_file.write_text("fake model data")

        client = LlamaCppLLMClient(model=model_file, auto_start=False, port=8080)

        with pytest.raises(RuntimeError, match="Server not started and auto_start=False"):
            await client._ensure_server_started()

    @pytest.mark.asyncio
    async def test_start_server_manually(self, tmp_path):
        """Test manual server start."""
        model_file = tmp_path / "test_model.gguf"
        model_file.write_text("fake model data")

        client = LlamaCppLLMClient(model=model_file, auto_start=True, port=8080)  # Use auto_start=True for start_server()
        client._server_manager.start = AsyncMock()

        with patch.object(type(client._server_manager), 'base_url', new_callable=lambda: property(lambda self: "http://127.0.0.1:8080")):
            with patch("httpx.AsyncClient") as mock_client:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = {"data": [{"id": "test-model"}]}
                mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)

                await client.start_server()

        assert client._server_started
        client._server_manager.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_server(self, tmp_path):
        """Test stopping server."""
        model_file = tmp_path / "test_model.gguf"
        model_file.write_text("fake model data")

        client = LlamaCppLLMClient(model=model_file, port=8080)
        client._server_started = True
        client._server_manager.stop = AsyncMock()

        await client.stop_server()

        assert not client._server_started
        client._server_manager.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_server_when_not_started(self, tmp_path):
        """Test stopping server when not started (no-op)."""
        model_file = tmp_path / "test_model.gguf"
        model_file.write_text("fake model data")

        client = LlamaCppLLMClient(model=model_file, port=8080)
        client._server_manager.stop = AsyncMock()

        await client.stop_server()

        # Should NOT call stop if not started
        client._server_manager.stop.assert_not_called()

    @pytest.mark.asyncio
    async def test_is_server_healthy_not_started(self, tmp_path):
        """Test health check when server not started."""
        model_file = tmp_path / "test_model.gguf"
        model_file.write_text("fake model data")

        client = LlamaCppLLMClient(model=model_file, port=8080)

        healthy = await client.is_server_healthy()

        assert not healthy

    @pytest.mark.asyncio
    async def test_is_server_healthy_started(self, tmp_path):
        """Test health check when server started."""
        model_file = tmp_path / "test_model.gguf"
        model_file.write_text("fake model data")

        client = LlamaCppLLMClient(model=model_file, port=8080)
        client._server_started = True
        client._server_manager.is_healthy = AsyncMock(return_value=True)

        healthy = await client.is_server_healthy()

        assert healthy
        client._server_manager.is_healthy.assert_called_once()


class TestLlamaCppClientCompletions:
    """Test completion methods (with server auto-start)."""

    @pytest.mark.asyncio
    async def test_create_completion_non_streaming(self, tmp_path):
        """Test non-streaming completion auto-starts server."""
        model_file = tmp_path / "test_model.gguf"
        model_file.write_text("fake model data")

        client = LlamaCppLLMClient(model=model_file, port=8080)

        # Mock server start
        client._ensure_server_started = AsyncMock()

        # Mock parent create_completion
        mock_response = {"response": "Hello!", "tool_calls": []}
        with patch.object(client.__class__.__bases__[0], 'create_completion', new_callable=AsyncMock, return_value=mock_response):
            result = await client.create_completion(
                messages=[{"role": "user", "content": "Hi"}],
                stream=False
            )

        # Should have started server
        client._ensure_server_started.assert_called_once()
        assert result == mock_response

    @pytest.mark.asyncio
    async def test_create_completion_streaming(self, tmp_path):
        """Test streaming completion auto-starts server."""
        model_file = tmp_path / "test_model.gguf"
        model_file.write_text("fake model data")

        client = LlamaCppLLMClient(model=model_file, port=8080)

        # Mock server start
        client._ensure_server_started = AsyncMock()

        # Mock parent streaming
        async def mock_stream():
            yield {"response": "Hello", "tool_calls": None}
            yield {"response": "!", "tool_calls": None}

        with patch.object(client.__class__.__bases__[0], 'create_completion', return_value=mock_stream()):
            result_stream = client.create_completion(
                messages=[{"role": "user", "content": "Hi"}],
                stream=True
            )

            chunks = []
            async for chunk in result_stream:
                chunks.append(chunk)

        # Should have started server before streaming
        client._ensure_server_started.assert_called_once()
        assert len(chunks) == 2
        assert chunks[0]["response"] == "Hello"



class TestLlamaCppClientContextManager:
    """Test async context manager functionality."""

    @pytest.mark.asyncio
    async def test_context_manager_starts_and_stops_server(self, tmp_path):
        """Test context manager starts server on enter and stops on exit."""
        model_file = tmp_path / "test_model.gguf"
        model_file.write_text("fake model data")

        client = LlamaCppLLMClient(model=model_file, port=8080)
        client._ensure_server_started = AsyncMock()
        client.stop_server = AsyncMock()

        async with client:
            # Server should be started
            client._ensure_server_started.assert_called_once()

        # Server should be stopped after exit
        client.stop_server.assert_called_once()


class TestLlamaCppClientCleanup:
    """Test cleanup and resource management."""

    @pytest.mark.asyncio
    async def test_close_stops_server(self, tmp_path):
        """Test close method stops server."""
        model_file = tmp_path / "test_model.gguf"
        model_file.write_text("fake model data")

        client = LlamaCppLLMClient(model=model_file, port=8080)

        # Mock the server manager's stop method instead
        client._server_manager.stop = AsyncMock()
        client._server_started = True

        await client.close()

        # Should call stop_server which calls manager.stop
        client._server_manager.stop.assert_called_once()

    def test_del_attempts_cleanup(self, tmp_path):
        """Test __del__ attempts to stop server."""
        model_file = tmp_path / "test_model.gguf"
        model_file.write_text("fake model data")

        client = LlamaCppLLMClient(model=model_file, port=8080)
        client._server_started = True

        # Mock the event loop
        with patch("asyncio.get_event_loop") as mock_get_loop:
            mock_loop = MagicMock()
            mock_loop.is_running.return_value = True
            mock_get_loop.return_value = mock_loop

            # Trigger __del__
            client.__del__()

            # Should have created a task to stop server
            mock_loop.create_task.assert_called_once()


class TestLlamaCppClientAPIBaseUpdate:
    """Test API base URL updates when server starts."""

    @pytest.mark.asyncio
    async def test_api_base_updated_after_server_start(self, tmp_path):
        """Test that api_base is updated if server returns different URL."""
        model_file = tmp_path / "test_model.gguf"
        model_file.write_text("fake model data")

        client = LlamaCppLLMClient(model=model_file, port=8080)
        client._server_manager.start = AsyncMock()

        # Server returns different base URL via property
        with patch.object(type(client._server_manager), 'base_url', new_callable=lambda: property(lambda self: "http://127.0.0.1:8081")):
            with patch("httpx.AsyncClient") as mock_client:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = {"data": [{"id": "actual-model-name"}]}
                mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)

                await client._ensure_server_started()

        # API base should be updated
        assert client.api_base == "http://127.0.0.1:8081"
        # Model name should be updated
        assert client.model == "actual-model-name"

    @pytest.mark.asyncio
    async def test_model_name_fetch_failure_keeps_original(self, tmp_path):
        """Test that original model name is kept if fetching actual name fails."""
        model_file = tmp_path / "test_model.gguf"
        model_file.write_text("fake model data")

        client = LlamaCppLLMClient(model=model_file, port=8080)
        original_model = client.model
        client._server_manager.start = AsyncMock()

        with patch.object(type(client._server_manager), 'base_url', new_callable=lambda: property(lambda self: "http://127.0.0.1:8080")):
            with patch("httpx.AsyncClient") as mock_client:
                # Simulate failure to get model name
                mock_client.return_value.__aenter__.return_value.get = AsyncMock(side_effect=Exception("Network error"))

                await client._ensure_server_started()

        # Model name should remain original
        assert client.model == original_model
