# tests/conftest.py
# Add this fixture to your conftest.py file (create if it doesn't exist)

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

@pytest.fixture
def mock_openai():
    """Mock OpenAI module properly"""
    with patch('chuk_llm.llm.providers.openai_client.openai') as mock_openai_module:
        # Create mock classes
        mock_async_client = AsyncMock()
        mock_sync_client = MagicMock()
        
        # Mock the AsyncOpenAI class
        mock_openai_module.AsyncOpenAI = MagicMock(return_value=mock_async_client)
        mock_openai_module.OpenAI = MagicMock(return_value=mock_sync_client)
        
        # Mock chat completions
        mock_async_client.chat.completions.create = AsyncMock()
        mock_sync_client.chat.completions.create = MagicMock()
        
        yield {
            'module': mock_openai_module,
            'async_client': mock_async_client,
            'sync_client': mock_sync_client
        }