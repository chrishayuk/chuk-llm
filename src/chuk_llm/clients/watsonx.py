"""
Modern Watsonx Client
======================

Type-safe async client for IBM Watsonx using Pydantic V2.
Wraps IBM SDK for authentication while maintaining modern patterns.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

from chuk_llm.clients.base import AsyncLLMClient
from chuk_llm.core import (
    CompletionRequest,
    CompletionResponse,
    ContentType,
    ErrorType,
    FinishReason,
    LLMError,
    MessageRole,
    StreamChunk,
)

logger = logging.getLogger(__name__)


class WatsonxClient(AsyncLLMClient):
    """
    Modern async client for IBM Watsonx.

    Wraps IBM's watsonx.ai SDK with modern Pydantic patterns.

    Args:
        model: Model name (e.g., "ibm/granite-3-8b-instruct")
        api_key: IBM Cloud API key
        project_id: Watsonx project ID
        space_id: Optional space ID (alternative to project_id)
        watsonx_ai_url: Watsonx API URL (default: us-south)
        **kwargs: Additional client options
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        project_id: str | None = None,
        space_id: str | None = None,
        watsonx_ai_url: str = "https://us-south.ml.cloud.ibm.com",
        **kwargs: Any,
    ):
        """
        Initialize Watsonx client.

        Args:
            model: Model name
            api_key: IBM Cloud API key
            project_id: Watsonx project ID
            space_id: Optional space ID
            watsonx_ai_url: Watsonx API URL
            **kwargs: Additional options
        """
        # Import IBM SDK here to make it optional
        try:
            from ibm_watsonx_ai import APIClient, Credentials
            from ibm_watsonx_ai.foundation_models import ModelInference
        except ImportError as e:
            raise ImportError(
                "ibm-watsonx-ai package required for Watsonx. "
                "Install with: pip install ibm-watsonx-ai"
            ) from e

        # We don't use httpx for Watsonx since IBM SDK handles HTTP
        # Just initialize with dummy values
        super().__init__(
            api_key=api_key,
            base_url=watsonx_ai_url,
            **kwargs,
        )

        self.model = model
        self.project_id = project_id
        self.space_id = space_id
        self.watsonx_ai_url = watsonx_ai_url

        # Initialize IBM SDK client
        credentials = Credentials(url=watsonx_ai_url, api_key=api_key)
        self._ibm_client = APIClient(credentials)
        self._ModelInference = ModelInference

        logger.info(f"Initialized Watsonx client: model={model}")

    def _is_granite_model(self) -> bool:
        """Check if model is Granite."""
        return "granite" in self.model.lower()

    def _prepare_watsonx_params(self, request: CompletionRequest) -> dict[str, Any]:
        """
        Prepare parameters for Watsonx API.

        Args:
            request: Pydantic completion request

        Returns:
            Dict of Watsonx parameters
        """
        params: dict[str, Any] = {}

        if request.temperature is not None:
            params["temperature"] = request.temperature

        if request.max_tokens is not None:
            # Watsonx uses max_new_tokens
            params["max_new_tokens"] = request.max_tokens

        if request.top_p is not None:
            params["top_p"] = request.top_p

        # Add time limit (IBM SDK requirement)
        params["time_limit"] = 60000  # 60 seconds

        return params

    def _format_messages_for_watsonx(self, request: CompletionRequest) -> str:
        """
        Format messages into prompt string for Watsonx.

        Watsonx SDK expects a prompt string, not message array.

        Args:
            request: Completion request

        Returns:
            Formatted prompt string
        """
        prompt_parts = []

        for msg in request.messages:
            # Extract text content
            if isinstance(msg.content, list):
                text_parts = [p.text for p in msg.content if p.type == ContentType.TEXT]
                content = " ".join(text_parts)
            else:
                content = msg.content or ""

            # Format based on role
            if msg.role == MessageRole.SYSTEM:
                prompt_parts.append(f"System: {content}")
            elif msg.role == MessageRole.USER:
                prompt_parts.append(f"User: {content}")
            elif msg.role == MessageRole.ASSISTANT:
                prompt_parts.append(f"Assistant: {content}")

        # Add final prompt for assistant response
        prompt_parts.append("Assistant:")

        return "\n\n".join(prompt_parts)

    def _parse_watsonx_response(self, response: dict[str, Any]) -> CompletionResponse:
        """
        Parse Watsonx response to CompletionResponse.

        Args:
            response: Raw Watsonx response

        Returns:
            Pydantic CompletionResponse

        Raises:
            LLMError: On parsing errors
        """
        try:
            # Watsonx response structure
            results = response.get("results", [])
            if not results:
                raise LLMError(
                    error_type=ErrorType.API_ERROR.value,
                    error_message="No results in Watsonx response",
                )

            result = results[0]
            content = result.get("generated_text", "").strip()

            # Check for stop reason
            stop_reason = result.get("stop_reason", "")
            if stop_reason == "max_tokens":
                finish_reason = FinishReason.LENGTH
            elif stop_reason == "eos_token":
                finish_reason = FinishReason.STOP
            else:
                finish_reason = FinishReason.STOP

            # Build response
            return CompletionResponse(
                content=content,
                finish_reason=finish_reason,
                tool_calls=[],  # Watsonx tool calling would need special parsing
            )

        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(
                error_type=ErrorType.API_ERROR.value,
                error_message=f"Failed to parse Watsonx response: {e}",
            ) from e

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """
        Complete a request using Watsonx.

        Args:
            request: Pydantic completion request

        Returns:
            Pydantic completion response

        Raises:
            LLMError: On API errors
        """
        try:
            # Prepare parameters
            params = self._prepare_watsonx_params(request)

            # Format prompt
            prompt = self._format_messages_for_watsonx(request)

            # Create model inference instance
            model_kwargs = {
                "model_id": self.model,
                "params": params,
                "project_id": self.project_id,
            }

            if self.space_id:
                model_kwargs["space_id"] = self.space_id

            model = self._ModelInference(
                **model_kwargs,
                api_client=self._ibm_client,
            )

            # Generate (IBM SDK is synchronous, run in executor)
            import asyncio

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, model.generate, prompt)

            # Parse response
            return self._parse_watsonx_response(response)

        except Exception as e:
            if isinstance(e, LLMError):
                raise

            raise LLMError(
                error_type=ErrorType.API_ERROR.value,
                error_message=f"Watsonx completion failed: {e}",
            ) from e

    async def stream(  # type: ignore[misc,override]
        self, request: CompletionRequest
    ) -> AsyncIterator[StreamChunk]:
        """
        Stream a request using Watsonx.

        Note: Watsonx streaming works differently than other providers.

        Args:
            request: Pydantic completion request

        Yields:
            Stream chunks

        Raises:
            LLMError: On API errors
        """
        try:
            # Prepare parameters
            params = self._prepare_watsonx_params(request)
            params["stream"] = True

            # Format prompt
            prompt = self._format_messages_for_watsonx(request)

            # Create model inference instance
            model_kwargs = {
                "model_id": self.model,
                "params": params,
                "project_id": self.project_id,
            }

            if self.space_id:
                model_kwargs["space_id"] = self.space_id

            model = self._ModelInference(
                **model_kwargs,
                api_client=self._ibm_client,
            )

            # Generate stream (IBM SDK returns iterator)
            import asyncio

            loop = asyncio.get_event_loop()
            stream_iter = await loop.run_in_executor(
                None, model.generate_text_stream, prompt
            )

            # Yield chunks
            for chunk_text in stream_iter:
                if chunk_text:
                    yield StreamChunk(content=chunk_text)

        except Exception as e:
            if isinstance(e, LLMError):
                raise

            raise LLMError(
                error_type=ErrorType.STREAMING_ERROR.value,
                error_message=f"Watsonx streaming failed: {e}",
            ) from e
