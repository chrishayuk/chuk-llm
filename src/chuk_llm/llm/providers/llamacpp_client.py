# chuk_llm/llm/providers/llamacpp_client.py
"""
llama.cpp client with auto-managed server
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Transparent server lifecycle management - just provide a model path!
"""

from __future__ import annotations

import logging
import socket
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from chuk_llm.llm.providers.llamacpp_server import (
    LlamaCppServerConfig,
    LlamaCppServerManager,
)
from chuk_llm.llm.providers.openai_client import OpenAILLMClient

log = logging.getLogger(__name__)


def _find_available_port(start_port: int = 8080, max_attempts: int = 100) -> int:
    """Find an available port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", port))
                return port
        except OSError:
            continue
    raise RuntimeError(
        f"No available ports found in range {start_port}-{start_port + max_attempts}"
    )


class LlamaCppLLMClient(OpenAILLMClient):
    """
    llama.cpp client with automatic server management.

    Transparently starts and manages llama-server process. Just provide
    a model path and use like any other LLM client!

    Features:
    - Auto-starts llama-server on first request
    - Auto-stops server on cleanup
    - Reuses existing server if already running
    - OpenAI-compatible API

    Examples:
        >>> # Simple usage - server auto-managed
        >>> client = LlamaCppLLMClient(model="/path/to/model.gguf")
        >>> response = await client.create_completion([{"role": "user", "content": "Hello!"}])

        >>> # Advanced: Custom server config
        >>> client = LlamaCppLLMClient(
        ...     model="/path/to/model.gguf",
        ...     ctx_size=16384,
        ...     n_gpu_layers=-1,
        ...     port=8080
        ... )

        >>> # Cleanup (automatic on context manager exit or del)
        >>> await client.close()
    """

    def __init__(
        self,
        model: str | Path,
        host: str = "127.0.0.1",
        port: int | None = None,
        ctx_size: int = 8192,
        n_gpu_layers: int = -1,
        server_binary: str | None = None,
        timeout: float = 120.0,
        api_key: str = "not-needed",
        extra_args: list[str] | None = None,
        auto_start: bool = True,
        **kwargs,
    ):
        """
        Initialize llama.cpp client with auto-managed server.

        Args:
            model: Path to GGUF model file
            host: Host to bind server to (default: 127.0.0.1)
            port: Port for server (default: auto-find from 8080)
            ctx_size: Context size (default: 8192)
            n_gpu_layers: GPU layers to use, -1 = all (default: -1)
            server_binary: Path to llama-server binary (default: auto-find in PATH)
            timeout: Server startup timeout in seconds (default: 120.0)
            api_key: API key (not needed for local server, default: "not-needed")
            extra_args: Additional llama-server arguments
            auto_start: Auto-start server on first request (default: True)
            **kwargs: Additional arguments passed to OpenAILLMClient
        """
        # Convert model to Path
        model_path = Path(model)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Find available port if not specified
        if port is None:
            port = _find_available_port(start_port=8080)
            log.debug(f"Auto-selected port {port} for llama-server")

        # Create server config
        self._server_config = LlamaCppServerConfig(
            model_path=model_path,
            host=host,
            port=port,
            ctx_size=ctx_size,
            n_gpu_layers=n_gpu_layers,
            server_binary=server_binary,
            timeout=timeout,
            extra_args=extra_args or [],
        )

        # Create server manager
        self._server_manager = LlamaCppServerManager(self._server_config)
        self._auto_start = auto_start
        self._server_started = False

        # Initialize parent with server URL
        base_url = f"http://{host}:{port}"

        # We'll get the actual model name after server starts
        # For now, use the path as placeholder
        super().__init__(
            model=str(model_path),
            api_base=base_url,
            api_key=api_key,
            **kwargs,
        )

        log.info(f"LlamaCppLLMClient initialized for {model_path.name}")
        log.debug(f"Server will be available at {base_url}")

    def _detect_provider_name(self, api_base: str | None) -> str:
        """Override provider detection to always return 'llamacpp'."""
        return "llamacpp"

    async def _ensure_server_started(self) -> None:
        """Ensure server is started before making requests."""
        if self._server_started:
            return

        if not self._auto_start:
            raise RuntimeError(
                "Server not started and auto_start=False. Call start_server() manually."
            )

        log.info("Starting llama-server...")
        await self._server_manager.start()
        self._server_started = True

        # Update api_base and client to point to actual server URL
        actual_base_url = self._server_manager.base_url
        if actual_base_url != self.api_base:
            log.debug(f"Updating API base from {self.api_base} to {actual_base_url}")
            self.api_base = actual_base_url
            # Update the OpenAI client's base_url
            self.client._base_url = actual_base_url

        # Update model name to actual name from server
        try:
            import httpx

            async with httpx.AsyncClient() as http_client:
                response = await http_client.get(f"{actual_base_url}/v1/models")
                if response.status_code == 200:
                    models_data = response.json()
                    if models_data.get("data"):
                        actual_model_name = models_data["data"][0]["id"]
                        self.model = actual_model_name
                        log.debug(f"Updated model name to: {actual_model_name}")
        except Exception as e:
            log.warning(f"Failed to get actual model name from server: {e}")

        log.info(f"llama-server ready at {self._server_manager.base_url}")

    async def start_server(self) -> None:
        """Manually start the server (useful if auto_start=False)."""
        await self._ensure_server_started()

    async def stop_server(self) -> None:
        """Stop the llama-server process."""
        if self._server_started:
            log.info("Stopping llama-server...")
            try:
                await self._server_manager.stop()
            except Exception as e:
                # Log at debug level to avoid CLI pollution
                log.debug(f"Error stopping server: {e}")
            finally:
                self._server_started = False

    async def is_server_healthy(self) -> bool:
        """Check if server is healthy."""
        if not self._server_started:
            return False
        return await self._server_manager.is_healthy()

    # Override parent methods to ensure server is started

    def create_completion(self, *args, **kwargs) -> Any:
        """Create completion (auto-starts server if needed)."""
        # Check if streaming
        is_streaming = kwargs.get("stream", False)

        if is_streaming:
            # For streaming, return an async generator that starts server first
            return self._create_completion_streaming(*args, **kwargs)
        else:
            # For non-streaming, return a coroutine that starts server first
            return self._create_completion_non_streaming(*args, **kwargs)

    async def _create_completion_non_streaming(self, *args, **kwargs) -> Any:
        """Non-streaming completion with server start."""
        await self._ensure_server_started()
        return await super().create_completion(*args, **kwargs)

    async def _create_completion_streaming(
        self, *args, **kwargs
    ) -> AsyncIterator[dict[str, Any]]:
        """Streaming completion with server start."""
        # CRITICAL: Ensure server is started BEFORE creating the async generator
        # If we call super().create_completion() first, it returns an async generator
        # that will immediately try to connect when we start iterating it
        await self._ensure_server_started()

        # Now get the async generator from parent - server is guaranteed to be ready
        async_gen = super().create_completion(*args, **kwargs)

        # Yield from it
        async for chunk in async_gen:
            yield chunk

    async def list_models(self) -> list[str]:
        """List available models (auto-starts server if needed)."""
        await self._ensure_server_started()
        return await super().list_models()

    # Lifecycle management

    async def close(self) -> None:
        """Close client and stop server."""
        await self.stop_server()
        # Close parent client if it has a close method
        if hasattr(super(), "close"):
            await super().close()

    async def __aenter__(self):
        """Async context manager entry - starts server."""
        await self._ensure_server_started()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - stops server."""
        await self.stop_server()

    def __del__(self):
        """Cleanup on deletion - best effort server stop."""
        if self._server_started:
            # Try to stop server in sync context (not ideal but necessary for __del__)
            import asyncio

            try:
                # If there's a running event loop, schedule cleanup
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Create task but don't wait for it - this can cause warnings
                    # but it's better than blocking __del__
                    task = loop.create_task(self.stop_server())
                    # Suppress "Task exception was never retrieved" by adding done callback
                    task.add_done_callback(
                        lambda t: None if t.exception() is None else None
                    )
                else:
                    # Run in new loop if no loop is running
                    asyncio.run(self.stop_server())
            except Exception:
                # Silently ignore all exceptions in __del__
                pass
