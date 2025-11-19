"""
Async-Native Base Client
=========================

Modern, async-first base implementation using httpx for connection pooling
and proper async/await patterns.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

try:
    import httpx

    _HTTPX_AVAILABLE = True
except ImportError:
    _HTTPX_AVAILABLE = False

from chuk_llm.core import (
    CompletionRequest,
    CompletionResponse,
    ContentTypeValue,
    Default,
    ErrorType,
    HttpHeader,
    HttpMethod,
    HttpStatus,
    LLMError,
    StreamChunk,
)

logger = logging.getLogger(__name__)


class AsyncLLMClient(ABC):
    """
    Async-native base client with connection pooling.

    All provider clients should inherit from this for consistent behavior.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        timeout: float = Default.TIMEOUT,
        max_connections: int = Default.MAX_CONNECTIONS,
        max_keepalive: int = Default.MAX_KEEPALIVE,
    ):
        """
        Initialize async client with connection pool.

        Args:
            api_key: API authentication key
            base_url: Base URL for API
            timeout: Request timeout in seconds
            max_connections: Maximum number of connections in pool
            max_keepalive: Maximum number of keep-alive connections
        """
        if not _HTTPX_AVAILABLE:
            raise ImportError(
                "httpx is required for async clients. Install with: pip install httpx"
            )

        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

        # Create async HTTP client with connection pooling
        limits = httpx.Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive,
        )

        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout),
            limits=limits,
            headers=self._get_default_headers(),
        )

        self._closed = False
        logger.debug(
            f"Initialized async client: base_url={base_url}, "
            f"max_connections={max_connections}"
        )

    def _get_default_headers(self) -> dict[str, str]:
        """Get default headers for all requests."""
        return {
            HttpHeader.AUTHORIZATION.value: f"Bearer {self.api_key}",
            HttpHeader.CONTENT_TYPE.value: ContentTypeValue.JSON.value,
        }

    @abstractmethod
    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """
        Create a non-streaming completion.

        Args:
            request: Validated completion request

        Returns:
            Validated completion response

        Raises:
            LLMError: On API errors
        """
        ...

    @abstractmethod
    async def stream(self, request: CompletionRequest) -> AsyncIterator[StreamChunk]:
        """
        Create a streaming completion.

        Args:
            request: Validated completion request

        Yields:
            Stream chunks with incremental content

        Raises:
            LLMError: On API errors
        """
        ...

    async def _post_json(
        self, endpoint: str, data: dict[str, Any], **kwargs: Any
    ) -> dict[str, Any]:
        """
        Helper for JSON POST requests.

        Args:
            endpoint: API endpoint (relative to base_url)
            data: JSON request body
            **kwargs: Additional httpx request options

        Returns:
            JSON response

        Raises:
            LLMError: On API errors
        """
        try:
            response = await self._client.post(endpoint, json=data, **kwargs)
            response.raise_for_status()
            return response.json()

        except httpx.HTTPStatusError as e:
            raise self._handle_http_error(e) from e
        except httpx.RequestError as e:
            raise LLMError(
                error_type=ErrorType.NETWORK_ERROR.value,
                error_message=f"Network error: {str(e)}",
            ) from e

    async def _stream_post(
        self, endpoint: str, data: dict[str, Any], **kwargs: Any
    ) -> AsyncIterator[bytes]:
        """
        Helper for streaming POST requests.

        Args:
            endpoint: API endpoint (relative to base_url)
            data: JSON request body
            **kwargs: Additional httpx request options

        Yields:
            Response bytes chunks

        Raises:
            LLMError: On API errors
        """
        try:
            async with self._client.stream(
                HttpMethod.POST.value, endpoint, json=data, **kwargs
            ) as response:
                response.raise_for_status()

                async for chunk in response.aiter_bytes():
                    yield chunk

        except httpx.HTTPStatusError as e:
            raise self._handle_http_error(e) from e
        except httpx.RequestError as e:
            raise LLMError(
                error_type=ErrorType.NETWORK_ERROR.value,
                error_message=f"Network error: {str(e)}",
            ) from e

    def _handle_http_error(self, error: httpx.HTTPStatusError) -> LLMError:
        """
        Convert HTTP errors to LLMError.

        Args:
            error: httpx HTTP status error

        Returns:
            Structured LLM error
        """
        status = error.response.status_code

        # Try to extract error message from response
        try:
            error_data = error.response.json()
            error_msg = (
                error_data.get("error", {}).get("message")
                or error_data.get("message")
                or str(error)
            )
        except Exception:
            error_msg = str(error)

        # Categorize errors
        if status == HttpStatus.UNAUTHORIZED:
            error_type = ErrorType.AUTHENTICATION_ERROR
        elif status == HttpStatus.FORBIDDEN:
            error_type = ErrorType.PERMISSION_ERROR
        elif status == HttpStatus.NOT_FOUND:
            error_type = ErrorType.NOT_FOUND_ERROR
        elif status == HttpStatus.RATE_LIMIT:
            error_type = ErrorType.RATE_LIMIT_ERROR
            # Try to extract retry-after
            retry_after = error.response.headers.get(HttpHeader.RETRY_AFTER.value)
            return LLMError(
                error_type=error_type.value,
                error_message=error_msg,
                retry_after=float(retry_after) if retry_after else None,
            )
        elif HttpStatus.SERVER_ERROR <= status < 600:
            error_type = ErrorType.SERVER_ERROR
        else:
            error_type = ErrorType.API_ERROR

        return LLMError(error_type=error_type.value, error_message=error_msg)

    async def close(self) -> None:
        """Close the HTTP client and cleanup resources."""
        if not self._closed:
            await self._client.aclose()
            self._closed = True
            logger.debug("Closed async client")

    async def __aenter__(self) -> AsyncLLMClient:
        """Context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        await self.close()

    def __del__(self) -> None:
        """Cleanup on deletion."""
        if not self._closed and hasattr(self, "_client"):
            # Log warning if client not properly closed
            logger.warning(
                "AsyncLLMClient was not properly closed. Use 'async with' or call close()"
            )
