# chuk_llm/providers/base.py
"""
Common abstract interface for every LLM adapter.
"""

from __future__ import annotations

import abc
from typing import Any, Dict, List, Optional


class BaseLLMClient(abc.ABC):
    """Abstract base class for LLM chat clients."""

    @abc.abstractmethod
    async def create_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Generate (or continue) a chat conversation.

        Parameters
        ----------
        messages
            List of ChatML-style message dicts.
        tools
            Optional list of OpenAI-function-tool schemas.

        Returns
        -------
        Standardised payload with keys ``response`` and ``tool_calls``.
        """
        ...
