"""
Modern Azure OpenAI Client
===========================

Type-safe async client for Azure OpenAI using Pydantic V2.
"""

from __future__ import annotations

import contextlib
import logging
import os
from typing import Any

import httpx

from chuk_llm.clients.openai import OpenAIClient
from chuk_llm.core import (
    EnvVar,
    ErrorType,
    LLMError,
)

logger = logging.getLogger(__name__)


class AzureOpenAIClient(OpenAIClient):
    """
    Modern async client for Azure OpenAI.

    Extends OpenAIClient with Azure-specific authentication and endpoints.

    Args:
        model: Model/deployment name
        api_key: Azure OpenAI API key
        azure_endpoint: Azure endpoint (e.g., https://my-resource.openai.azure.com)
        api_version: Azure API version (default: 2024-02-01)
        azure_deployment: Optional deployment name (defaults to model)
        azure_ad_token: Optional Azure AD token for auth
        timeout: Request timeout in seconds
        max_retries: Maximum number of retries
        **kwargs: Additional httpx client options
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        azure_endpoint: str | None = None,
        api_version: str = "2024-02-01",
        azure_deployment: str | None = None,
        azure_ad_token: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 2,
        **kwargs: Any,
    ):
        # Get Azure endpoint
        endpoint = azure_endpoint or os.getenv(EnvVar.AZURE_OPENAI_ENDPOINT.value)
        if not endpoint:
            raise ValueError(
                "azure_endpoint required for Azure OpenAI. "
                "Set AZURE_OPENAI_ENDPOINT environment variable."
            )

        # Get authentication
        if not api_key and not azure_ad_token:
            api_key = os.getenv(EnvVar.AZURE_OPENAI_API_KEY.value)

        if not api_key and not azure_ad_token:
            raise ValueError(
                "Authentication required for Azure OpenAI. "
                "Set AZURE_OPENAI_API_KEY or provide azure_ad_token."
            )

        # Azure deployment name (defaults to model name)
        self.azure_deployment = azure_deployment or model
        self.api_version = api_version
        self.azure_ad_token = azure_ad_token

        # Build Azure-specific base URL
        # Format: https://{resource}.openai.azure.com/openai/deployments/{deployment}
        base_url = f"{endpoint.rstrip('/')}/openai/deployments/{self.azure_deployment}"

        # Initialize parent OpenAIClient with Azure URL
        super().__init__(
            model=model,
            api_key=api_key or "azure-ad-auth",  # Dummy key if using AD token
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            **kwargs,
        )

        # Override headers for Azure AD token if provided
        if azure_ad_token:
            self._client.headers["Authorization"] = f"Bearer {azure_ad_token}"
            # Remove api-key header if using AD token
            if "api-key" in self._client.headers:
                del self._client.headers["api-key"]

        logger.info(
            f"AzureOpenAIClient initialized: deployment={self.azure_deployment}, "
            f"api_version={self.api_version}"
        )

    async def _post_json(
        self, endpoint: str, data: dict[str, Any], **kwargs: Any
    ) -> dict[str, Any]:
        """
        Override to add Azure-specific query parameters.

        Args:
            endpoint: API endpoint
            data: Request payload

        Returns:
            Response dict

        Raises:
            LLMError: On API errors
        """
        # Add api-version query parameter
        url = f"{self.base_url}/{endpoint}"
        if "?" in url:
            url = f"{url}&api-version={self.api_version}"
        else:
            url = f"{url}?api-version={self.api_version}"

        try:
            response = await self._client.post(url, json=data)
            response.raise_for_status()
            return response.json()

        except httpx.HTTPStatusError as e:
            error_data = {}
            with contextlib.suppress(Exception):
                error_data = e.response.json()

            # Azure-specific error handling
            error_message = error_data.get("error", {}).get("message", str(e))
            error_code = error_data.get("error", {}).get("code", "unknown")

            # Map Azure error codes to our error types
            if e.response.status_code == 401:
                error_type_str = ErrorType.AUTHENTICATION_ERROR.value
            elif e.response.status_code == 429:
                error_type_str = ErrorType.RATE_LIMIT_ERROR.value
            elif e.response.status_code >= 500:
                error_type_str = ErrorType.SERVER_ERROR.value
            elif error_code == "content_filter":
                error_type_str = ErrorType.VALIDATION_ERROR.value
                error_message = f"Content filtered by Azure: {error_message}"
            else:
                error_type_str = ErrorType.API_ERROR.value

            raise LLMError(
                error_type=error_type_str,
                error_message=f"Azure OpenAI error (HTTP {e.response.status_code}): {error_message}",
            ) from e

        except httpx.RequestError as e:
            raise LLMError(
                error_type=ErrorType.NETWORK_ERROR.value,
                error_message=f"Azure OpenAI request failed: {e}",
            ) from e

    async def _post_stream(self, endpoint: str, data: dict[str, Any]) -> httpx.Response:
        """
        Override to add Azure-specific query parameters for streaming.

        Args:
            endpoint: API endpoint
            data: Request payload

        Returns:
            httpx Response with streaming content
        """
        # Add api-version query parameter
        url = f"{self.base_url}/{endpoint}"
        if "?" in url:
            url = f"{url}&api-version={self.api_version}"
        else:
            url = f"{url}?api-version={self.api_version}"

        try:
            response = await self._client.post(url, json=data)
            response.raise_for_status()
            return response

        except httpx.HTTPStatusError as e:
            error_data = {}
            with contextlib.suppress(Exception):
                error_data = e.response.json()

            error_message = error_data.get("error", {}).get("message", str(e))

            raise LLMError(
                error_type=ErrorType.API_ERROR.value,
                error_message=f"Azure OpenAI streaming error (HTTP {e.response.status_code}): {error_message}",
            ) from e
