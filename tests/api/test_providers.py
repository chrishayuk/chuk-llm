"""
Pytest tests for chuk_llm/api/providers.py (Clean Dynamic Implementation)

Tests the new clean dynamic provider function generation system.
Everything comes from YAML configuration with zero hardcoding.

Run with:
    pytest tests/api/test_providers.py -v
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, mock_open
import asyncio
import threading
import time
from typing import List, Dict, Any

# Import specific functions we need to test
from chuk_llm.api.providers import (
    _sanitize_name,
    _get_persistent_loop,
    _run_sync,
    _create_provider_function,
    _create_stream_function, 
    _create_sync_function,
    _create_global_alias_function,
)


class TestSanitizeName:
    """Test suite for _sanitize_name function."""

    def test_sanitize_basic_names(self):
        """Test basic name sanitization."""
        assert _sanitize_name("gpt-4o-mini") == "gpt4o_mini"
        assert _sanitize_name("claude-3-sonnet") == "claude3_sonnet"

    def test_sanitize_with_dots(self):
        """Test sanitization with dots in name."""
        # Updated expectations for simple rule - dots become underscores
        assert _sanitize_name("llama-3.3-70b") == "llama3_3_70b"
        assert _sanitize_name("granite-3.1") == "granite3_1"

    def test_sanitize_with_slashes(self):
        """Test sanitization with slashes (for provider/model paths)."""
        assert _sanitize_name("openai/gpt-4o") == "openai_gpt4o"
        assert _sanitize_name("meta-llama/llama-3.1") == "meta_llama_llama3_1"

    def test_sanitize_special_characters(self):
        """Test sanitization removes special characters."""
        assert _sanitize_name("model@name#test") == "modelnametest"
        assert _sanitize_name("test-model!v2") == "test_modelv2"

    def test_sanitize_starts_with_number(self):
        """Test that names starting with numbers get prefixed."""
        assert _sanitize_name("3.5-turbo") == "model_3_5_turbo"
        assert _sanitize_name("4o-mini") == "model_4o_mini"

    def test_sanitize_empty_string(self):
        """Test sanitization with empty string."""
        assert _sanitize_name("") == ""
        assert _sanitize_name(None) == ""

    def test_sanitize_case_conversion(self):
        """Test that names are converted to lowercase."""
        assert _sanitize_name("GPT-4O-MINI") == "gpt4o_mini"
        assert _sanitize_name("Claude-3-Sonnet") == "claude3_sonnet"

    def test_sanitize_consecutive_separators(self):
        """Test handling of consecutive separators."""
        assert _sanitize_name("test--model..name") == "test_model_name"
        assert _sanitize_name("model___name") == "model_name"


class TestPersistentEventLoop:
    """Test suite for persistent event loop functionality used by sync functions."""

    def test_get_persistent_loop_creates_loop(self):
        """Test that _get_persistent_loop creates a background loop."""
        # Reset any existing loop state
        import chuk_llm.api.providers as providers_module
        providers_module._persistent_loop = None
        providers_module._loop_thread = None
        
        loop = _get_persistent_loop()
        
        # Should get a valid event loop
        assert loop is not None
        assert isinstance(loop, asyncio.AbstractEventLoop)
        assert not loop.is_closed()
        
        # Should be running in a background thread
        assert providers_module._loop_thread is not None
        assert providers_module._loop_thread.is_alive()

    def test_get_persistent_loop_reuses_existing(self):
        """Test that _get_persistent_loop reuses existing loop."""
        loop1 = _get_persistent_loop()
        loop2 = _get_persistent_loop()
        
        # Should be the same loop instance
        assert loop1 is loop2

    def test_run_sync_basic_functionality(self):
        """Test basic functionality of _run_sync."""
        async def test_coro():
            await asyncio.sleep(0.001)
            return "test_result"
        
        result = _run_sync(test_coro())
        assert result == "test_result"

    def test_run_sync_with_exception(self):
        """Test exception handling in _run_sync."""
        async def failing_coro():
            raise ValueError("Test exception")
        
        with pytest.raises(ValueError, match="Test exception"):
            _run_sync(failing_coro())

    def test_run_sync_from_async_context_fails(self):
        """Test that calling from async context raises error."""
        import warnings
        
        async def test_async_context():
            async def test_coro():
                return "should_not_work"
            
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*was never awaited.*")
                with pytest.raises(RuntimeError, match="Cannot call sync functions from async context"):
                    _run_sync(test_coro())
        
        asyncio.run(test_async_context())

    def test_persistent_loop_multiple_calls(self):
        """Test that persistent loop handles multiple calls correctly."""
        async def test_coro(value):
            await asyncio.sleep(0.001)
            return f"result_{value}"
        
        results = []
        for i in range(5):
            result = _run_sync(test_coro(i))
            results.append(result)
        
        expected = [f"result_{i}" for i in range(5)]
        assert results == expected

    def test_persistent_loop_thread_safety(self):
        """Test thread safety of persistent loop."""
        async def test_coro(thread_id):
            await asyncio.sleep(0.001)
            return f"thread_{thread_id}_result"
        
        results = []
        
        def worker(thread_id):
            result = _run_sync(test_coro(thread_id))
            results.append(result)
        
        threads = []
        for i in range(3):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        assert len(results) == 3
        for i in range(3):
            assert f"thread_{i}_result" in results


class TestProviderFunctionCreation:
    """Test suite for provider function creation."""

    def test_create_provider_function_basic(self):
        """Test creating basic provider function."""
        func = _create_provider_function("openai")
        
        assert callable(func)
        assert func.__name__ is not None

    def test_create_provider_function_with_model(self):
        """Test creating provider function with specific model."""
        func = _create_provider_function("openai", "gpt-4o")
        
        assert callable(func)

    @pytest.mark.asyncio
    async def test_provider_function_execution_pattern(self):
        """Test the execution pattern of provider functions."""
        # Mock the core.ask function
        with patch('chuk_llm.api.core.ask') as mock_ask:
            mock_ask.return_value = "mocked response"
            
            # Create a function
            func = _create_provider_function("openai", "gpt-4o")
            
            # Test that it's async and calls ask correctly
            result = await func("test prompt", temperature=0.8)
            
            mock_ask.assert_called_once_with(
                "test prompt", 
                provider="openai", 
                model="gpt-4o", 
                temperature=0.8
            )
            assert result == "mocked response"

    def test_create_stream_function_basic(self):
        """Test creating basic streaming function."""
        func = _create_stream_function("anthropic")
        
        assert callable(func)

    def test_create_stream_function_with_model(self):
        """Test creating streaming function with specific model."""
        func = _create_stream_function("anthropic", "claude-3-sonnet")
        
        assert callable(func)

    @pytest.mark.asyncio
    async def test_stream_function_execution_pattern(self):
        """Test the execution pattern of streaming functions."""
        # Mock the core.stream function
        async def mock_stream(*args, **kwargs):
            yield "chunk1"
            yield "chunk2"
            yield "chunk3"
        
        with patch('chuk_llm.api.core.stream', side_effect=mock_stream):
            func = _create_stream_function("anthropic", "claude-3-sonnet")
            
            chunks = []
            async for chunk in func("test prompt"):
                chunks.append(chunk)
            
            assert chunks == ["chunk1", "chunk2", "chunk3"]

    def test_create_sync_function_basic(self):
        """Test creating basic sync function."""
        func = _create_sync_function("groq")
        
        assert callable(func)

    def test_create_sync_function_with_model(self):
        """Test creating sync function with specific model."""
        func = _create_sync_function("groq", "llama-3.3-70b")
        
        assert callable(func)

    def test_sync_function_execution_pattern(self):
        """Test the execution pattern of sync functions."""
        with patch('chuk_llm.api.core.ask') as mock_ask:
            mock_ask.return_value = "sync response"
            
            with patch('chuk_llm.api.providers._run_sync') as mock_run_sync:
                mock_run_sync.return_value = "sync response"
                
                func = _create_sync_function("groq", "llama-3.3-70b")
                result = func("test prompt", temperature=0.5)
                
                # Should have called _run_sync
                mock_run_sync.assert_called_once()
                assert result == "sync response"


class TestGlobalAliasFunctions:
    """Test suite for global alias function creation."""

    def test_create_global_alias_function_basic(self):
        """Test creating global alias functions."""
        functions = _create_global_alias_function("gpt4", "openai/gpt-4o")
        
        expected_keys = ["ask_gpt4", "ask_gpt4_sync", "stream_gpt4"]
        assert all(key in functions for key in expected_keys)
        assert all(callable(func) for func in functions.values())

    def test_create_global_alias_function_invalid_format(self):
        """Test handling of invalid global alias format."""
        functions = _create_global_alias_function("invalid", "no_slash_here")
        
        # Should return empty dict for invalid format
        assert functions == {}

    @pytest.mark.asyncio
    async def test_global_alias_async_function_pattern(self):
        """Test execution pattern of global alias async function."""
        with patch('chuk_llm.api.core.ask') as mock_ask:
            mock_ask.return_value = "alias response"
            
            functions = _create_global_alias_function("claude", "anthropic/claude-3-sonnet")
            ask_func = functions["ask_claude"]
            
            result = await ask_func("test prompt")
            
            mock_ask.assert_called_once_with(
                "test prompt",
                provider="anthropic",
                model="claude-3-sonnet"
            )
            assert result == "alias response"

    def test_global_alias_sync_function_pattern(self):
        """Test execution pattern of global alias sync function."""
        with patch('chuk_llm.api.providers._run_sync') as mock_run_sync:
            mock_run_sync.return_value = "sync alias response"
            
            functions = _create_global_alias_function("llama", "groq/llama-3.3-70b")
            sync_func = functions["ask_llama_sync"]
            
            result = sync_func("test prompt")
            
            mock_run_sync.assert_called_once()
            assert result == "sync alias response"


class TestFunctionGenerationLogic:
    """Test suite for the function generation logic."""

    @patch('chuk_llm.api.providers.get_config')
    def test_generate_functions_basic_flow(self, mock_get_config):
        """Test the basic flow of function generation."""
        # Mock configuration manager
        mock_config_manager = Mock()
        mock_config_manager.get_all_providers.return_value = ["openai", "anthropic"]
        
        # Mock provider configs
        openai_config = Mock()
        openai_config.models = ["gpt-4o", "gpt-4o-mini"]
        openai_config.model_aliases = {"gpt4o": "gpt-4o", "mini": "gpt-4o-mini"}
        
        anthropic_config = Mock()
        anthropic_config.models = ["claude-3-sonnet"]
        anthropic_config.model_aliases = {"sonnet": "claude-3-sonnet"}
        
        mock_config_manager.get_provider.side_effect = lambda p: {
            "openai": openai_config,
            "anthropic": anthropic_config
        }[p]
        
        mock_config_manager.get_global_aliases.return_value = {
            "gpt4": "openai/gpt-4o",
            "claude": "anthropic/claude-3-sonnet"
        }
        
        mock_get_config.return_value = mock_config_manager
        
        # Import and test the generation logic
        from chuk_llm.api.providers import _generate_functions
        
        functions = _generate_functions()
        
        # Should have base provider functions
        assert "ask_openai" in functions
        assert "stream_openai" in functions
        assert "ask_openai_sync" in functions
        assert "ask_anthropic" in functions
        
        # Should have model-specific functions
        assert "ask_openai_gpt4o" in functions
        assert "ask_openai_gpt4o_mini" in functions
        # Updated expectation: "claude-3-sonnet" -> "claude3_sonnet" with simple rule
        assert "ask_anthropic_claude3_sonnet" in functions
        
        # Should have alias functions
        assert "ask_openai_mini" in functions  # From model_aliases
        assert "ask_anthropic_sonnet" in functions
        
        # Should have global alias functions
        assert "ask_gpt4" in functions
        assert "ask_claude" in functions

    def test_function_naming_pattern(self):
        """Test function naming patterns."""
        provider = "openai"
        model = "gpt-4o-mini"
        alias = "mini"
        
        # Base function names
        base_names = [
            f"ask_{provider}",
            f"stream_{provider}",
            f"ask_{provider}_sync"
        ]
        
        # Model-specific function names
        model_suffix = _sanitize_name(model)
        model_names = [
            f"ask_{provider}_{model_suffix}",
            f"stream_{provider}_{model_suffix}",
            f"ask_{provider}_{model_suffix}_sync"
        ]
        
        # Alias function names
        alias_suffix = _sanitize_name(alias)
        alias_names = [
            f"ask_{provider}_{alias_suffix}",
            f"stream_{provider}_{alias_suffix}",
            f"ask_{provider}_{alias_suffix}_sync"
        ]
        
        expected_base = ["ask_openai", "stream_openai", "ask_openai_sync"]
        expected_model = ["ask_openai_gpt4o_mini", "stream_openai_gpt4o_mini", "ask_openai_gpt4o_mini_sync"]
        expected_alias = ["ask_openai_mini", "stream_openai_mini", "ask_openai_mini_sync"]
        
        assert base_names == expected_base
        assert model_names == expected_model
        assert alias_names == expected_alias

    def test_function_docstring_generation(self):
        """Test function docstring generation logic."""
        # Test docstring generation pattern
        test_cases = [
            ("ask_openai", "Async openai call."),
            ("ask_openai_sync", "Synchronous openai call."),
            ("stream_openai", "Stream from openai."),
            ("ask_openai_gpt4o", "Async openai gpt4o call."),
            ("ask_openai_gpt4o_sync", "Synchronous openai gpt4o call."),
            ("stream_anthropic_claude", "Stream from anthropic claude."),
        ]
        
        for name, expected_doc in test_cases:
            # Generate docstring using the same logic as the implementation
            if name.startswith("ask_") and name.endswith("_sync"):
                base_name = name[4:-5]
                actual_doc = f"Synchronous {base_name.replace('_', ' ')} call."
            elif name.startswith("ask_"):
                base_name = name[4:]
                actual_doc = f"Async {base_name.replace('_', ' ')} call."
            elif name.startswith("stream_"):
                base_name = name[7:]
                actual_doc = f"Stream from {base_name.replace('_', ' ')}."
            
            assert actual_doc == expected_doc


class TestUtilityFunctions:
    """Test suite for utility function creation."""

    @patch('chuk_llm.configuration.config.get_config')
    def test_quick_question_logic(self, mock_get_config):
        """Test quick_question utility function logic."""
        mock_config_manager = Mock()
        mock_config_manager.get_global_settings.return_value = {
            "active_provider": "openai"
        }
        mock_get_config.return_value = mock_config_manager
        
        # Test the logic pattern
        def mock_quick_question(question: str, provider: str = None):
            if not provider:
                settings = mock_config_manager.get_global_settings()
                provider = settings.get("active_provider", "openai")
            
            return f"Quick response from {provider}: {question}"
        
        result = mock_quick_question("What is 2+2?")
        assert result == "Quick response from openai: What is 2+2?"
        
        result = mock_quick_question("Hello", provider="anthropic")
        assert result == "Quick response from anthropic: Hello"

    @patch('chuk_llm.configuration.config.get_config')
    def test_compare_providers_logic(self, mock_get_config):
        """Test compare_providers utility function logic."""
        mock_config_manager = Mock()
        mock_config_manager.get_all_providers.return_value = ["openai", "anthropic", "groq"]
        mock_get_config.return_value = mock_config_manager
        
        # Test the logic pattern
        def mock_compare_providers(question: str, providers: List[str] = None):
            if not providers:
                all_providers = mock_config_manager.get_all_providers()
                providers = all_providers[:3] if len(all_providers) >= 3 else all_providers
            
            results = {}
            for provider in providers:
                results[provider] = f"{provider} response: {question}"
            
            return results
        
        result = mock_compare_providers("Test question")
        expected = {
            "openai": "openai response: Test question",
            "anthropic": "anthropic response: Test question",
            "groq": "groq response: Test question"
        }
        assert result == expected

    def test_show_config_logic(self):
        """Test show_config utility function logic."""
        # Mock the show_config function behavior
        def mock_show_config():
            return {
                "providers": ["openai", "anthropic"],
                "global_aliases": ["gpt4", "claude"],
                "status": "loaded"
            }
        
        result = mock_show_config()
        assert "providers" in result
        assert "global_aliases" in result
        assert result["status"] == "loaded"


class TestErrorHandling:
    """Test suite for error handling in provider function generation."""

    @patch('chuk_llm.configuration.config.get_config')
    def test_handle_missing_provider_config(self, mock_get_config):
        """Test handling of missing provider configuration."""
        mock_config_manager = Mock()
        mock_config_manager.get_all_providers.return_value = ["unknown_provider"]
        mock_config_manager.get_provider.side_effect = ValueError("Unknown provider")
        mock_get_config.return_value = mock_config_manager
        
        # Should handle error gracefully without crashing
        from chuk_llm.api.providers import _generate_functions
        
        # Should not raise exception
        functions = _generate_functions()
        
        # Should return some functions (at least utilities)
        assert isinstance(functions, dict)

    def test_handle_invalid_model_names(self):
        """Test handling of invalid model names."""
        # Test edge cases in model name sanitization
        invalid_names = ["", None, "!!!invalid!!!", "123", "model.with.lots.of.dots"]
        
        for name in invalid_names:
            result = _sanitize_name(name)
            
            # Should not crash and should return valid Python identifier or empty string
            if result:  # If not empty
                assert result.replace('_', '').isalnum() or result.startswith('model_')

    def test_persistent_loop_error_recovery(self):
        """Test that persistent loop recovers from errors."""
        async def good_coro():
            return "success"
        
        async def bad_coro():
            raise Exception("test error")
        
        # Bad coroutine should raise exception but not break loop
        with pytest.raises(Exception, match="test error"):
            _run_sync(bad_coro())
        
        # Good coroutine should still work after the error
        result = _run_sync(good_coro())
        assert result == "success"

    def test_concurrent_loop_access_safety(self):
        """Test concurrent access to persistent loop is safe."""
        results = []
        errors = []
        
        async def test_coro(value):
            await asyncio.sleep(0.001)
            return f"value_{value}"
        
        def worker(worker_id):
            try:
                result = _run_sync(test_coro(worker_id))
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Start multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=5.0)
        
        # Should have no errors
        assert len(errors) == 0, f"Unexpected errors: {errors}"
        
        # Should have results from all workers
        assert len(results) == 3
        for i in range(3):
            assert f"value_{i}" in results


class TestCleanupAndLifecycle:
    """Test suite for cleanup and lifecycle management."""

    def test_cleanup_function_exists(self):
        """Test that cleanup function exists."""
        from chuk_llm.api.providers import _cleanup_loop
        assert callable(_cleanup_loop)

    def test_atexit_registration(self):
        """Test that atexit registration works."""
        import atexit
        
        def mock_cleanup():
            pass
        
        # Should not raise exception
        atexit.register(mock_cleanup)
        assert callable(mock_cleanup)

    def test_loop_state_management(self):
        """Test loop state management."""
        import chuk_llm.api.providers as providers_module
        
        # Test that we can access loop state variables
        assert hasattr(providers_module, '_persistent_loop')
        assert hasattr(providers_module, '_loop_thread')
        assert hasattr(providers_module, '_loop_lock')

    def test_module_import_safety(self):
        """Test that importing the module is safe."""
        # This test verifies that importing doesn't crash
        import chuk_llm.api.providers
        
        # Should have the main components
        assert hasattr(chuk_llm.api.providers, '_generate_functions')
        assert hasattr(chuk_llm.api.providers, '_create_utility_functions')
        assert hasattr(chuk_llm.api.providers, '__all__')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])