"""
Comprehensive pytest tests for chuk_llm/api/utils.py

Run with:
    pytest tests/api/test_utils.py -v
    pytest tests/api/test_utils.py -v --tb=short
    pytest tests/api/test_utils.py::TestGetMetrics::test_get_metrics_no_cached_client -v
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Dict, Any
import sys
import os

# Import the module under test
from chuk_llm.api import utils
from chuk_llm.api.config import get_current_config
from chuk_llm.api.utils import (
    get_metrics,
    health_check,
    health_check_sync,
    get_current_client_info,
    test_connection,
    test_connection_sync,
    test_all_providers,
    test_all_providers_sync,
    print_diagnostics,
    cleanup,
    cleanup_sync
)


class TestGetMetrics:
    """Test suite for get_metrics function."""

    def test_get_metrics_no_cached_client(self):
        """Test get_metrics when no client is cached."""
        with patch('chuk_llm.api.utils._cached_client', None):
            result = get_metrics()
            assert result == {}

    def test_get_metrics_client_without_middleware(self):
        """Test get_metrics when client has no middleware stack."""
        mock_client = Mock()
        # Client without middleware_stack attribute
        del mock_client.middleware_stack
        
        with patch('chuk_llm.api.utils._cached_client', mock_client):
            result = get_metrics()
            assert result == {}

    def test_get_metrics_with_middleware_no_metrics(self):
        """Test get_metrics when middleware exists but has no get_metrics method."""
        mock_middleware = Mock()
        del mock_middleware.get_metrics  # Remove get_metrics method
        
        mock_client = Mock()
        mock_client.middleware_stack.middlewares = [mock_middleware]
        
        with patch('chuk_llm.api.utils._cached_client', mock_client):
            result = get_metrics()
            assert result == {}

    def test_get_metrics_with_valid_middleware(self):
        """Test get_metrics when middleware has valid metrics."""
        expected_metrics = {
            "total_requests": 42,
            "average_duration": 1.5,
            "error_count": 3
        }
        
        mock_middleware = Mock()
        mock_middleware.get_metrics.return_value = expected_metrics
        
        mock_client = Mock()
        mock_client.middleware_stack.middlewares = [mock_middleware]
        
        with patch('chuk_llm.api.utils._cached_client', mock_client):
            result = get_metrics()
            assert result == expected_metrics
            mock_middleware.get_metrics.assert_called_once()

    def test_get_metrics_multiple_middleware_first_has_metrics(self):
        """Test get_metrics when multiple middleware exist, first has metrics."""
        expected_metrics = {"requests": 10}
        
        mock_middleware1 = Mock()
        mock_middleware1.get_metrics.return_value = expected_metrics
        
        mock_middleware2 = Mock()
        mock_middleware2.get_metrics.return_value = {"other": "data"}
        
        mock_client = Mock()
        mock_client.middleware_stack.middlewares = [mock_middleware1, mock_middleware2]
        
        with patch('chuk_llm.api.utils._cached_client', mock_client):
            result = get_metrics()
            assert result == expected_metrics
            mock_middleware1.get_metrics.assert_called_once()
            mock_middleware2.get_metrics.assert_not_called()


class TestHealthCheck:
    """Test suite for health check functions."""

    @pytest.mark.asyncio
    async def test_health_check_success(self):
        """Test successful health check."""
        expected_health = {
            "status": "healthy",
            "total_clients": 5,
            "active_connections": 3
        }
        
        with patch('chuk_llm.llm.connection_pool.get_llm_health_status', new_callable=AsyncMock) as mock_health:
            mock_health.return_value = expected_health
            
            result = await health_check()
            assert result == expected_health
            mock_health.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_check_import_error(self):
        """Test health check when connection pool module is not available."""
        # Mock the import inside the health_check function
        with patch.dict('sys.modules', {'chuk_llm.llm.connection_pool': None}):
            with patch('builtins.__import__', side_effect=ImportError("Module not found")):
                result = await health_check()
                
                expected = {
                    "status": "unknown",
                    "error": "Health check not available - connection pool not found"
                }
                assert result == expected

    @pytest.mark.asyncio
    async def test_health_check_other_exception(self):
        """Test health check when other exceptions occur."""
        with patch('chuk_llm.llm.connection_pool.get_llm_health_status', new_callable=AsyncMock) as mock_health:
            mock_health.side_effect = RuntimeError("Connection failed")
            
            with pytest.raises(RuntimeError, match="Connection failed"):
                await health_check()

    def test_health_check_sync(self):
        """Test synchronous health check wrapper."""
        expected_health = {"status": "healthy"}
        
        with patch('chuk_llm.api.utils.health_check', new_callable=AsyncMock) as mock_async_health:
            mock_async_health.return_value = expected_health
            
            with patch('asyncio.run') as mock_run:
                mock_run.return_value = expected_health
                
                result = health_check_sync()
                assert result == expected_health
                mock_run.assert_called_once()


class TestGetCurrentClientInfo:
    """Test suite for get_current_client_info function."""

    def test_get_current_client_info_no_client(self):
        """Test get_current_client_info when no client is cached."""
        with patch('chuk_llm.api.utils._cached_client', None):
            result = get_current_client_info()
            
            expected = {
                "status": "no_client",
                "message": "No client currently cached"
            }
            assert result == expected

    def test_get_current_client_info_with_client_no_middleware(self):
        """Test get_current_client_info with client but no middleware."""
        mock_client = Mock()
        mock_client.__class__.__name__ = "TestClient"
        # Remove middleware_stack attribute
        del mock_client.middleware_stack
        
        mock_config = {
            "provider": "openai",
            "model": "gpt-4",
            "api_key": "test_key"
        }
        
        with patch('chuk_llm.api.utils._cached_client', mock_client):
            with patch('chuk_llm.api.utils.get_current_config', return_value=mock_config):
                result = get_current_client_info()
                
                expected = {
                    "status": "active",
                    "provider": "openai",
                    "model": "gpt-4",
                    "client_type": "TestClient",
                    "has_middleware": False,
                }
                assert result == expected

    def test_get_current_client_info_with_middleware(self):
        """Test get_current_client_info with client and middleware."""
        mock_middleware1 = Mock()
        mock_middleware1.__class__.__name__ = "MetricsMiddleware"
        mock_middleware2 = Mock()
        mock_middleware2.__class__.__name__ = "LoggingMiddleware"
        
        mock_client = Mock()
        mock_client.__class__.__name__ = "TestClient"
        mock_client.middleware_stack.middlewares = [mock_middleware1, mock_middleware2]
        
        mock_config = {
            "provider": "anthropic",
            "model": "claude-3-sonnet",
        }
        
        with patch('chuk_llm.api.utils._cached_client', mock_client):
            with patch('chuk_llm.api.utils.get_current_config', return_value=mock_config):
                result = get_current_client_info()
                
                expected = {
                    "status": "active",
                    "provider": "anthropic",
                    "model": "claude-3-sonnet",
                    "client_type": "TestClient",
                    "has_middleware": True,
                    "middleware": ["MetricsMiddleware", "LoggingMiddleware"]
                }
                assert result == expected

    def test_get_current_client_info_missing_config_keys(self):
        """Test get_current_client_info when config is missing keys."""
        mock_client = Mock()
        mock_client.__class__.__name__ = "TestClient"
        del mock_client.middleware_stack
        
        mock_config = {}  # Empty config
        
        with patch('chuk_llm.api.utils._cached_client', mock_client):
            with patch('chuk_llm.api.utils.get_current_config', return_value=mock_config):
                result = get_current_client_info()
                
                expected = {
                    "status": "active",
                    "provider": "unknown",
                    "model": "unknown",
                    "client_type": "TestClient",
                    "has_middleware": False,
                }
                assert result == expected


class TestTestConnection:
    """Test suite for test_connection functions."""

    @pytest.mark.asyncio
    async def test_test_connection_success(self):
        """Test successful connection test."""
        mock_response = "Hello! This is a test response."
        
        # Mock the ask function from core module
        with patch('chuk_llm.api.core.ask', new_callable=AsyncMock) as mock_ask:
            mock_ask.return_value = mock_response
            
            with patch('chuk_llm.api.utils.get_current_config') as mock_config:
                mock_config.return_value = {"provider": "openai", "model": "gpt-4"}
                
                # Mock the event loop time
                with patch('asyncio.get_event_loop') as mock_loop:
                    mock_loop.return_value.time.side_effect = [1000.0, 1001.5]  # 1.5 second duration
                    
                    result = await test_connection()
                    
                    expected = {
                        "success": True,
                        "provider": "openai",
                        "model": "gpt-4",
                        "duration": 1.5,
                        "response_length": len(mock_response),
                        "response_preview": mock_response
                    }
                    assert result == expected
                    
                    mock_ask.assert_called_once_with(
                        "Hello, this is a connection test.",
                        provider="openai",
                        model="gpt-4",
                        max_tokens=50
                    )

    @pytest.mark.asyncio
    async def test_test_connection_with_overrides(self):
        """Test connection test with provider and model overrides."""
        mock_response = "Test response from Anthropic"
        
        with patch('chuk_llm.api.core.ask', new_callable=AsyncMock) as mock_ask:
            mock_ask.return_value = mock_response
            
            with patch('chuk_llm.api.utils.get_current_config') as mock_config:
                mock_config.return_value = {"provider": "openai", "model": "gpt-4"}
                
                with patch('asyncio.get_event_loop') as mock_loop:
                    mock_loop.return_value.time.side_effect = [1000.0, 1002.0]
                    
                    result = await test_connection(
                        provider="anthropic",
                        model="claude-3-sonnet",
                        test_prompt="Custom test prompt"
                    )
                    
                    expected = {
                        "success": True,
                        "provider": "anthropic",
                        "model": "claude-3-sonnet",
                        "duration": 2.0,
                        "response_length": len(mock_response),
                        "response_preview": mock_response
                    }
                    assert result == expected
                    
                    mock_ask.assert_called_once_with(
                        "Custom test prompt",
                        provider="anthropic",
                        model="claude-3-sonnet",
                        max_tokens=50
                    )

    @pytest.mark.asyncio
    async def test_test_connection_failure(self):
        """Test connection test when an error occurs."""
        with patch('chuk_llm.api.core.ask', new_callable=AsyncMock) as mock_ask:
            mock_ask.side_effect = ValueError("API key invalid")
            
            with patch('chuk_llm.api.utils.get_current_config') as mock_config:
                mock_config.return_value = {"provider": "openai", "model": "gpt-4"}
                
                with patch('asyncio.get_event_loop') as mock_loop:
                    mock_loop.return_value.time.side_effect = [1000.0, 1001.0]
                    
                    result = await test_connection()
                    
                    expected = {
                        "success": False,
                        "provider": "openai",
                        "model": "gpt-4",
                        "duration": 1.0,
                        "error": "API key invalid",
                        "error_type": "ValueError"
                    }
                    assert result == expected

    @pytest.mark.asyncio
    async def test_test_connection_long_response(self):
        """Test connection test with long response that gets truncated."""
        long_response = "A" * 200  # 200 character response
        
        with patch('chuk_llm.api.core.ask', new_callable=AsyncMock) as mock_ask:
            mock_ask.return_value = long_response
            
            with patch('chuk_llm.api.utils.get_current_config') as mock_config:
                mock_config.return_value = {"provider": "openai", "model": "gpt-4"}
                
                with patch('asyncio.get_event_loop') as mock_loop:
                    mock_loop.return_value.time.side_effect = [1000.0, 1001.0]
                    
                    result = await test_connection()
                    
                    assert result["success"] is True
                    assert result["response_length"] == 200
                    assert result["response_preview"] == "A" * 100 + "..."  # Truncated to 100 chars + "..."

    def test_test_connection_sync(self):
        """Test synchronous test_connection wrapper."""
        expected_result = {"success": True, "provider": "openai"}
        
        with patch('chuk_llm.api.utils.test_connection', new_callable=AsyncMock) as mock_async:
            mock_async.return_value = expected_result
            
            with patch('asyncio.run') as mock_run:
                mock_run.return_value = expected_result
                
                result = test_connection_sync("anthropic", "claude-3-sonnet", "test prompt")
                assert result == expected_result
                mock_run.assert_called_once()


class TestTestAllProviders:
    """Test suite for test_all_providers functions."""

    @pytest.mark.asyncio
    async def test_test_all_providers_success(self):
        """Test successful testing of all providers."""
        mock_results = {
            "openai": {"success": True, "provider": "openai", "duration": 1.0},
            "anthropic": {"success": True, "provider": "anthropic", "duration": 1.5},
            "google": {"success": True, "provider": "google", "duration": 2.0}
        }
        
        async def mock_test_connection(provider=None, test_prompt=None):
            return mock_results[provider]
        
        with patch('chuk_llm.api.utils.test_connection', side_effect=mock_test_connection):
            result = await test_all_providers()
            
            assert result == mock_results
            assert len(result) == 3
            assert all(result[provider]["success"] for provider in result)

    @pytest.mark.asyncio
    async def test_test_all_providers_with_custom_list(self):
        """Test testing specific providers."""
        custom_providers = ["openai", "anthropic"]
        
        mock_results = {
            "openai": {"success": True, "provider": "openai"},
            "anthropic": {"success": False, "provider": "anthropic", "error": "API key missing"}
        }
        
        async def mock_test_connection(provider=None, test_prompt=None):
            return mock_results[provider]
        
        with patch('chuk_llm.api.utils.test_connection', side_effect=mock_test_connection):
            result = await test_all_providers(providers=custom_providers)
            
            assert result == mock_results
            assert len(result) == 2

    @pytest.mark.asyncio
    async def test_test_all_providers_with_exceptions(self):
        """Test handling of exceptions during provider testing."""
        async def mock_test_connection(provider=None, test_prompt=None):
            if provider == "openai":
                return {"success": True, "provider": "openai"}
            elif provider == "anthropic":
                raise ConnectionError("Network error")
            else:
                raise ValueError("Invalid provider")
        
        with patch('chuk_llm.api.utils.test_connection', side_effect=mock_test_connection):
            result = await test_all_providers()
            
            assert result["openai"]["success"] is True
            assert result["anthropic"]["success"] is False
            assert "Network error" in result["anthropic"]["error"]
            assert result["anthropic"]["error_type"] == "ConnectionError"
            assert result["google"]["success"] is False
            assert "Invalid provider" in result["google"]["error"]

    @pytest.mark.asyncio
    async def test_test_all_providers_custom_prompt(self):
        """Test testing providers with custom prompt."""
        custom_prompt = "Custom test message"
        
        async def mock_test_connection(provider=None, test_prompt=None):
            assert test_prompt == custom_prompt
            return {"success": True, "provider": provider}
        
        with patch('chuk_llm.api.utils.test_connection', side_effect=mock_test_connection):
            await test_all_providers(test_prompt=custom_prompt)

    def test_test_all_providers_sync(self):
        """Test synchronous test_all_providers wrapper."""
        expected_result = {"openai": {"success": True}}
        
        with patch('chuk_llm.api.utils.test_all_providers', new_callable=AsyncMock) as mock_async:
            mock_async.return_value = expected_result
            
            with patch('asyncio.run') as mock_run:
                mock_run.return_value = expected_result
                
                result = test_all_providers_sync(["openai"], "test prompt")
                assert result == expected_result
                mock_run.assert_called_once()


class TestPrintDiagnostics:
    """Test suite for print_diagnostics function."""

    def test_print_diagnostics_full_info(self, capsys):
        """Test print_diagnostics with full information available."""
        mock_config = {
            "provider": "openai",
            "model": "gpt-4",
            "api_key": "sk-1234567890abcdef",
            "temperature": 0.7
        }
        
        mock_client_info = {
            "status": "active",
            "provider": "openai",
            "client_type": "OpenAIClient",
            "has_middleware": True,
            "middleware": ["MetricsMiddleware"]
        }
        
        mock_metrics = {
            "total_requests": 42,
            "average_duration": 1.23,
            "error_count": 2
        }
        
        mock_health = {
            "status": "healthy",
            "active_clients": 3
        }
        
        with patch('chuk_llm.api.utils.get_current_config', return_value=mock_config):
            with patch('chuk_llm.api.utils.get_current_client_info', return_value=mock_client_info):
                with patch('chuk_llm.api.utils.get_metrics', return_value=mock_metrics):
                    with patch('chuk_llm.api.utils.health_check_sync', return_value=mock_health):
                        print_diagnostics()
        
        captured = capsys.readouterr()
        output = captured.out
        
        # Check that key information is present
        assert "ChukLLM Diagnostics" in output
        assert "Current Configuration:" in output
        assert "provider: openai" in output
        assert "model: gpt-4" in output
        assert "api_key: sk-12345..." in output  # Should be masked
        assert "Client Information:" in output
        assert "status: active" in output
        assert "Metrics:" in output
        assert "total_requests: 42" in output
        assert "average_duration: 1.23" in output
        assert "Health Check:" in output
        assert "status: healthy" in output

    def test_print_diagnostics_no_metrics(self, capsys):
        """Test print_diagnostics when no metrics are available."""
        mock_config = {"provider": "openai"}
        mock_client_info = {"status": "active"}
        mock_health = {"status": "healthy"}
        
        with patch('chuk_llm.api.utils.get_current_config', return_value=mock_config):
            with patch('chuk_llm.api.utils.get_current_client_info', return_value=mock_client_info):
                with patch('chuk_llm.api.utils.get_metrics', return_value={}):
                    with patch('chuk_llm.api.utils.health_check_sync', return_value=mock_health):
                        print_diagnostics()
        
        captured = capsys.readouterr()
        output = captured.out
        
        assert "No metrics available (enable_metrics=False)" in output

    def test_print_diagnostics_health_check_error(self, capsys):
        """Test print_diagnostics when health check fails."""
        mock_config = {"provider": "openai"}
        mock_client_info = {"status": "active"}
        
        with patch('chuk_llm.api.utils.get_current_config', return_value=mock_config):
            with patch('chuk_llm.api.utils.get_current_client_info', return_value=mock_client_info):
                with patch('chuk_llm.api.utils.get_metrics', return_value={}):
                    with patch('chuk_llm.api.utils.health_check_sync', side_effect=Exception("Connection failed")):
                        print_diagnostics()
        
        captured = capsys.readouterr()
        output = captured.out
        
        assert "Error: Connection failed" in output

    def test_print_diagnostics_short_api_key(self, capsys):
        """Test print_diagnostics with short API key (should not be truncated)."""
        mock_config = {"api_key": "short"}
        
        with patch('chuk_llm.api.utils.get_current_config', return_value=mock_config):
            with patch('chuk_llm.api.utils.get_current_client_info', return_value={}):
                with patch('chuk_llm.api.utils.get_metrics', return_value={}):
                    with patch('chuk_llm.api.utils.health_check_sync', return_value={}):
                        print_diagnostics()
        
        captured = capsys.readouterr()
        output = captured.out
        
        assert "api_key: ***" in output


class TestCleanup:
    """Test suite for cleanup functions."""

    @pytest.mark.asyncio
    async def test_cleanup_success(self):
        """Test successful cleanup."""
        mock_client = Mock()
        mock_client.close = AsyncMock()
        
        with patch('chuk_llm.api.utils._cached_client', mock_client):
            with patch('chuk_llm.llm.connection_pool.cleanup_llm_resources', new_callable=AsyncMock) as mock_cleanup:
                await cleanup()
                
                mock_cleanup.assert_called_once()
                mock_client.close.assert_called_once()
                
                # Check that cached client is cleared
                assert utils._cached_client is None

    @pytest.mark.asyncio
    async def test_cleanup_import_error(self):
        """Test cleanup when connection pool module is not available."""
        mock_client = AsyncMock()
        mock_client.close = AsyncMock()
        
        with patch('chuk_llm.api.utils._cached_client', mock_client):
            # Mock the import error inside cleanup function
            with patch.dict('sys.modules', {'chuk_llm.llm.connection_pool': None}):
                with patch('builtins.__import__', side_effect=ImportError):
                    await cleanup()
                    
                    # Should still close client and clear cache
                    mock_client.close.assert_called_once()
                    assert utils._cached_client is None

    @pytest.mark.asyncio
    async def test_cleanup_no_cached_client(self):
        """Test cleanup when no client is cached."""
        with patch('chuk_llm.api.utils._cached_client', None):
            with patch('chuk_llm.llm.connection_pool.cleanup_llm_resources', new_callable=AsyncMock) as mock_cleanup:
                await cleanup()
                
                mock_cleanup.assert_called_once()
                assert utils._cached_client is None

    @pytest.mark.asyncio
    async def test_cleanup_client_without_close_method(self):
        """Test cleanup when client doesn't have close method."""
        mock_client = Mock()
        # Remove close method
        del mock_client.close
        
        with patch('chuk_llm.api.utils._cached_client', mock_client):
            with patch('chuk_llm.llm.connection_pool.cleanup_llm_resources', new_callable=AsyncMock) as mock_cleanup:
                await cleanup()
                
                mock_cleanup.assert_called_once()
                assert utils._cached_client is None

    def test_cleanup_sync(self):
        """Test synchronous cleanup wrapper."""
        with patch('chuk_llm.api.utils.cleanup', new_callable=AsyncMock) as mock_async_cleanup:
            with patch('asyncio.run') as mock_run:
                cleanup_sync()
                mock_run.assert_called_once()


class TestFixtures:
    """Test fixtures and setup helpers."""

    @pytest.fixture
    def mock_config(self):
        """Fixture providing a mock configuration."""
        return {
            "provider": "openai",
            "model": "gpt-4",
            "api_key": "test_key",
            "temperature": 0.7,
            "max_tokens": 1000
        }

    @pytest.fixture
    def mock_client_with_middleware(self):
        """Fixture providing a mock client with middleware."""
        mock_middleware = Mock()
        mock_middleware.get_metrics.return_value = {"requests": 10}
        mock_middleware.__class__.__name__ = "TestMiddleware"
        
        mock_client = Mock()
        mock_client.__class__.__name__ = "TestClient"
        mock_client.middleware_stack.middlewares = [mock_middleware]
        
        return mock_client


# Integration test example
class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_full_diagnostic_workflow(self, capsys):
        """Test a complete diagnostic workflow."""
        # Create proper dictionary objects (not Mock objects)
        mock_config = {
            "provider": "openai", 
            "model": "gpt-4",
            "api_key": "sk-test123",
            "temperature": 0.7
        }
        mock_client_info = {
            "status": "active", 
            "provider": "openai",
            "client_type": "TestClient",
            "has_middleware": False
        }
        # Return empty dict for metrics to match what real function returns when no middleware
        mock_metrics = {}  # Changed to empty dict to avoid KeyError
        mock_health = {"status": "healthy", "active_clients": 2}
        
        # Mock the _cached_client to avoid "no_client" status
        mock_client = Mock()
        mock_client.__class__.__name__ = "TestClient"
        # Remove middleware_stack to avoid hasattr issues
        if hasattr(mock_client, 'middleware_stack'):
            delattr(mock_client, 'middleware_stack')
        
        with patch.object(utils, '_cached_client', mock_client):
            with patch('chuk_llm.api.utils.get_current_config', return_value=mock_config):
                with patch('chuk_llm.api.utils.get_current_client_info', return_value=mock_client_info):
                    with patch('chuk_llm.api.utils.get_metrics', return_value=mock_metrics):
                        with patch('chuk_llm.api.utils.health_check_sync', return_value=mock_health):
                            # Get individual components - these should now return our mocked values
                            config = get_current_config()
                            client_info = get_current_client_info()
                            metrics = get_metrics()
                            
                            # Verify components work with the actual return values
                            assert config["provider"] == "openai"
                            assert client_info["status"] == "active"
                            assert metrics == {}  # Updated assertion
                            
                            # Test print diagnostics - this should work with empty metrics
                            print_diagnostics()
                            
                            captured = capsys.readouterr()
                            assert "openai" in captured.out
                            assert "active" in captured.out
                            assert "healthy" in captured.out
                            assert "No metrics available" in captured.out  # Should show this message


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])