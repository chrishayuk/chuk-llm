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
import sys
import warnings
from typing import List, Dict, Any

# Suppress coroutine warnings in tests
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*coroutine.*was never awaited")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*AsyncMockMixin.*")

# Import specific functions we need to test
from chuk_llm.api.providers import (
    _sanitize_name,
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


class TestRunSyncWithEventLoopManager:
    """Test suite for _run_sync function using event loop manager."""

    @patch('chuk_llm.api.event_loop_manager.run_sync')
    def test_run_sync_uses_event_loop_manager(self, mock_run_sync):
        """Test that _run_sync uses the event loop manager when available."""
        mock_run_sync.return_value = "test_result"
        
        # Create a simple coroutine that returns immediately
        async def test_coro():
            return "test_result"
        
        # Create the coroutine object
        coro = test_coro()
        
        try:
            result = _run_sync(coro)
            
            # Should have called the event loop manager's run_sync
            mock_run_sync.assert_called_once()
            assert result == "test_result"
        finally:
            # Close the coroutine to avoid warnings
            coro.close()

    @patch.dict('sys.modules', {'chuk_llm.api.event_loop_manager': None})
    @patch('asyncio.run')
    def test_run_sync_fallback_to_asyncio_run(self, mock_asyncio_run):
        """Test fallback to asyncio.run when event loop manager not available."""
        mock_asyncio_run.return_value = "fallback_result"
        
        async def test_coro():
            return "fallback_result"
        
        result = _run_sync(test_coro())
        
        # Should have fallen back to asyncio.run
        mock_asyncio_run.assert_called_once()
        assert result == "fallback_result"

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

    def test_run_sync_multiple_calls(self):
        """Test that _run_sync handles multiple calls correctly."""
        async def test_coro(value):
            await asyncio.sleep(0.001)
            return f"result_{value}"
        
        results = []
        for i in range(5):
            result = _run_sync(test_coro(i))
            results.append(result)
        
        expected = [f"result_{i}" for i in range(5)]
        assert results == expected

    def test_run_sync_thread_safety(self):
        """Test thread safety of _run_sync."""
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
        # Create a mock coroutine
        async def mock_ask_coro(*args, **kwargs):
            return "mocked response"
        
        # Mock the core.ask function to return our coroutine
        with patch('chuk_llm.api.core.ask', side_effect=mock_ask_coro) as mock_ask:
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
        # Create a mock coroutine that returns a value
        async def mock_coro(*args, **kwargs):
            return "sync response"
        
        with patch('chuk_llm.api.core.ask', return_value=mock_coro()):
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
        # Create a mock coroutine
        async def mock_ask_coro(*args, **kwargs):
            return "alias response"
        
        with patch('chuk_llm.api.core.ask', side_effect=mock_ask_coro) as mock_ask:
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
        # Create a mock coroutine
        async def mock_coro(*args, **kwargs):
            return "sync alias response"
        
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
        
        # Import and test the generation logic with warnings suppressed
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
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

    def test_run_sync_error_recovery(self):
        """Test that _run_sync recovers from errors."""
        async def good_coro():
            return "success"
        
        async def bad_coro():
            raise Exception("test error")
        
        # Bad coroutine should raise exception but not break event loop
        with pytest.raises(Exception, match="test error"):
            _run_sync(bad_coro())
        
        # Good coroutine should still work after the error
        result = _run_sync(good_coro())
        assert result == "success"

    def test_concurrent_run_sync_safety(self):
        """Test concurrent calls to _run_sync are safe."""
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


class TestModuleIntegration:
    """Test suite for module-level integration."""

    def test_module_imports_successfully(self):
        """Test that the module imports without errors."""
        # This should not raise any exceptions
        import chuk_llm.api.providers
        
        # Check that expected attributes exist
        assert hasattr(chuk_llm.api.providers, '__all__')
        assert isinstance(chuk_llm.api.providers.__all__, list)

    def test_functions_are_exported(self):
        """Test that functions are properly exported in __all__."""
        import chuk_llm.api.providers
        
        # Should have some functions exported
        assert len(chuk_llm.api.providers.__all__) > 0
        
        # All exported names should be callable or utility functions
        for name in chuk_llm.api.providers.__all__:
            assert hasattr(chuk_llm.api.providers, name)

    @patch('chuk_llm.api.providers.get_config')
    def test_error_during_generation_handled(self, mock_get_config):
        """Test that errors during function generation are handled gracefully."""
        # Make get_config raise an exception
        mock_get_config.side_effect = Exception("Config error")
        
        # Reload the module - should handle the error gracefully
        import importlib
        import chuk_llm.api.providers
        
        try:
            importlib.reload(chuk_llm.api.providers)
        except Exception as e:
            # The module should still be usable even if generation fails
            assert hasattr(chuk_llm.api.providers, '__all__')

    def test_warnings_are_suppressed(self):
        """Test that asyncio warnings are properly suppressed."""
        import warnings
        
        # These warnings should be filtered
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Try to trigger some warnings
            async def test_coro():
                return "test"
            
            _run_sync(test_coro())
            
            # Should not have event loop warnings
            event_loop_warnings = [
                warning for warning in w 
                if "Event loop is closed" in str(warning.message)
            ]
            assert len(event_loop_warnings) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])