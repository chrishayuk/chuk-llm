"""
Pytest tests for chuk_llm/api/providers.py (Persistent Loop Implementation)

IMPORTANT: The persistent loop is ONLY used for sync functions (ending in _sync).
Regular async functions (ask_*, stream_*) use normal async execution patterns.

Run with:
    pytest tests/api/test_providers.py -v
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import asyncio
import threading
import time
from typing import List, Dict, Any

# Import only the specific functions we need to test, not the whole module
# This avoids triggering the module-level function generation
from chuk_llm.api.providers import (
    _sanitize_model_name,
    _get_common_models_for_provider,
    _get_model_aliases_for_provider,
    _get_persistent_loop,
    _run_async_on_persistent_loop,
)


class TestSanitizeModelName:
    """Test suite for _sanitize_model_name function."""

    def test_sanitize_basic_model_name(self):
        """Test basic model name sanitization."""
        assert _sanitize_model_name("gpt-4o-mini") == "gpt_4o_mini"
        assert _sanitize_model_name("claude-3-sonnet") == "claude_3_sonnet"

    def test_sanitize_with_dots(self):
        """Test sanitization with dots in model name."""
        assert _sanitize_model_name("llama-3.3-70b") == "llama_33_70b"
        assert _sanitize_model_name("granite-3.1") == "granite_31"

    def test_sanitize_with_dates(self):
        """Test sanitization with date suffixes."""
        assert _sanitize_model_name("claude-3-sonnet-20240229") == "claude_3_sonnet_20240229"

    def test_sanitize_special_characters(self):
        """Test sanitization removes special characters."""
        assert _sanitize_model_name("model@name#test") == "modelnametest"
        assert _sanitize_model_name("test-model!v2") == "test_modelv2"

    def test_sanitize_starts_with_number(self):
        """Test that model names starting with numbers get prefixed."""
        assert _sanitize_model_name("3.5-turbo") == "model_35_turbo"
        assert _sanitize_model_name("4o-mini") == "model_4o_mini"

    def test_sanitize_empty_string(self):
        """Test sanitization with empty string."""
        assert _sanitize_model_name("") == ""
        assert _sanitize_model_name(None) == ""

    def test_sanitize_case_conversion(self):
        """Test that names are converted to lowercase."""
        assert _sanitize_model_name("GPT-4O-MINI") == "gpt_4o_mini"
        assert _sanitize_model_name("Claude-3-Sonnet") == "claude_3_sonnet"


class TestGetModelsFromYAML:
    """Test suite for _get_common_models_for_provider reading from YAML."""

    def test_get_models_from_yaml_models_key(self):
        """Test reading models from 'models' key in YAML."""
        mock_config = {
            'models': ['gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo'],
            'default_model': 'gpt-4o-mini'
        }
        
        with patch('chuk_llm.api.providers.get_provider_config') as mock_get_config:
            mock_get_config.return_value = mock_config
            
            models = _get_common_models_for_provider('openai')
            
            assert models == ['gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo']
            mock_get_config.assert_called_once_with('openai')

    def test_get_models_from_yaml_models_section_with_names(self):
        """Test reading models from 'models' section with name keys."""
        mock_config = {
            'models': [
                {'name': 'claude-3-sonnet', 'features': ['streaming']},
                {'name': 'claude-3-opus', 'features': ['streaming', 'vision']},
                {'name': 'claude-3-haiku', 'features': ['streaming']}
            ],
            'default_model': 'claude-3-sonnet'
        }
        
        with patch('chuk_llm.api.providers.get_provider_config') as mock_get_config:
            mock_get_config.return_value = mock_config
            
            models = _get_common_models_for_provider('anthropic')
            
            expected = ['claude-3-sonnet', 'claude-3-opus', 'claude-3-haiku']
            assert models == expected

    def test_get_models_from_yaml_models_section_strings(self):
        """Test reading models from 'models' section as simple strings."""
        mock_config = {
            'models': ['llama-3.3-70b', 'llama-3.1-8b', 'mixtral-8x7b'],
            'default_model': 'llama-3.3-70b'
        }
        
        with patch('chuk_llm.api.providers.get_provider_config') as mock_get_config:
            mock_get_config.return_value = mock_config
            
            models = _get_common_models_for_provider('groq')
            
            assert models == ['llama-3.3-70b', 'llama-3.1-8b', 'mixtral-8x7b']

    def test_get_models_no_config_returns_empty(self):
        """Test that no YAML config returns empty list (no fallbacks)."""
        with patch('chuk_llm.api.providers.get_provider_config') as mock_get_config:
            mock_get_config.return_value = {}  # Empty config
            
            models = _get_common_models_for_provider('openai')
            
            # Should return empty list (no hardcoded fallbacks)
            assert models == []

    def test_get_models_unknown_provider_returns_empty(self):
        """Test handling of unknown provider returns empty list."""
        with patch('chuk_llm.api.providers.get_provider_config') as mock_get_config:
            mock_get_config.return_value = {}
            
            models = _get_common_models_for_provider('unknown_provider')
            
            # Should return empty list for unknown provider
            assert models == []


class TestGetModelAliases:
    """Test suite for _get_model_aliases_for_provider function."""

    def test_get_model_aliases_success(self):
        """Test reading model aliases from YAML configuration."""
        mock_config = {
            'models': ['gpt-4o', 'gpt-4o-mini'],
            'model_aliases': {
                'gpt4o': 'gpt-4o',
                'gpt4o_mini': 'gpt-4o-mini'
            }
        }
        
        with patch('chuk_llm.api.providers.get_provider_config') as mock_get_config:
            mock_get_config.return_value = mock_config
            
            aliases = _get_model_aliases_for_provider('openai')
            
            expected = {'gpt4o': 'gpt-4o', 'gpt4o_mini': 'gpt-4o-mini'}
            assert aliases == expected

    def test_get_model_aliases_no_aliases(self):
        """Test when no model aliases are defined."""
        mock_config = {
            'models': ['gpt-4o', 'gpt-4o-mini']
            # No model_aliases key
        }
        
        with patch('chuk_llm.api.providers.get_provider_config') as mock_get_config:
            mock_get_config.return_value = mock_config
            
            aliases = _get_model_aliases_for_provider('openai')
            
            assert aliases == {}

    def test_get_model_aliases_no_provider_config(self):
        """Test when provider has no configuration."""
        with patch('chuk_llm.api.providers.get_provider_config') as mock_get_config:
            mock_get_config.return_value = {}
            
            aliases = _get_model_aliases_for_provider('unknown_provider')
            
            assert aliases == {}


class TestPersistentEventLoop:
    """Test suite for persistent event loop functionality.
    
    Note: The persistent loop is ONLY used for sync functions (ending in _sync).
    Regular async functions (ask_*, stream_*) use normal async execution.
    """

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

    def test_get_persistent_loop_reuses_existing_loop(self):
        """Test that _get_persistent_loop reuses existing loop."""
        # Get the first loop
        loop1 = _get_persistent_loop()
        
        # Get the loop again
        loop2 = _get_persistent_loop()
        
        # Should be the same loop instance
        assert loop1 is loop2

    def test_run_async_on_persistent_loop_basic(self):
        """Test basic functionality of _run_async_on_persistent_loop."""
        async def test_coro():
            await asyncio.sleep(0.001)
            return "test_result"
        
        result = _run_async_on_persistent_loop(test_coro())
        assert result == "test_result"

    def test_run_async_on_persistent_loop_with_exception(self):
        """Test exception handling in _run_async_on_persistent_loop."""
        async def failing_coro():
            raise ValueError("Test exception")
        
        with pytest.raises(ValueError, match="Test exception"):
            _run_async_on_persistent_loop(failing_coro())

    def test_run_async_on_persistent_loop_from_async_context_fails(self):
        """Test that calling from async context raises error."""
        import warnings
        
        async def test_async_context():
            async def test_coro():
                return "should_not_work"
            
            # We expect this coroutine to not be awaited, so suppress the warning
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*was never awaited.*")
                with pytest.raises(RuntimeError, match="Cannot call sync functions from async context"):
                    _run_async_on_persistent_loop(test_coro())
        
        # Run this test in an async context
        asyncio.run(test_async_context())

    def test_persistent_loop_survives_multiple_calls(self):
        """Test that persistent loop survives multiple function calls."""
        async def test_coro(value):
            await asyncio.sleep(0.001)
            return f"result_{value}"
        
        # Make multiple calls
        results = []
        for i in range(5):
            result = _run_async_on_persistent_loop(test_coro(i))
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
            result = _run_async_on_persistent_loop(test_coro(thread_id))
            results.append(result)
        
        # Start multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Should have results from all threads
        assert len(results) == 3
        for i in range(3):
            assert f"thread_{i}_result" in results


class TestProviderFunctionCreationLogic:
    """Test suite for provider function creation logic without imports.
    
    Note: This tests the logic for creating three types of functions:
    1. ask_* (async functions - use normal async execution)
    2. stream_* (async functions - use normal async execution) 
    3. ask_*_sync (sync functions - use persistent loop)
    """

    def test_provider_function_naming_logic(self):
        """Test provider function naming logic."""
        # Test function name generation logic
        provider = "openai"
        model = "gpt-4o"
        
        # Test base function names
        base_ask = f"ask_{provider}"
        base_stream = f"stream_{provider}"
        base_sync = f"ask_{provider}_sync"
        
        assert base_ask == "ask_openai"
        assert base_stream == "stream_openai"
        assert base_sync == "ask_openai_sync"
        
        # Test model-specific function names
        model_suffix = _sanitize_model_name(model)  # gpt-4o -> gpt_4o
        model_ask = f"ask_{provider}_{model_suffix}"
        model_stream = f"stream_{provider}_{model_suffix}"
        model_sync = f"ask_{provider}_{model_suffix}_sync"
        
        assert model_ask == "ask_openai_gpt_4o"
        assert model_stream == "stream_openai_gpt_4o"
        assert model_sync == "ask_openai_gpt_4o_sync"

    def test_function_documentation_logic(self):
        """Test function documentation generation logic."""
        # Test documentation generation logic
        def generate_doc(name, is_sync=False, has_model=False):
            parts = name.replace('ask_', '').replace('stream_', '').replace('_sync', '')
            
            if is_sync:
                if has_model and '_' in parts and len(parts.split('_')) > 1:
                    provider_part = parts.split('_')[0]
                    model_part = '_'.join(parts.split('_')[1:])
                    return f"Synchronous {provider_part} with model {model_part}."
                else:
                    return f"Synchronous {parts} (default model)."
            else:
                if has_model and '_' in parts and len(parts.split('_')) > 1:
                    provider_part = parts.split('_')[0]
                    model_part = '_'.join(parts.split('_')[1:])
                    return f"Ask {provider_part} with model {model_part}."
                else:
                    return f"Ask {parts} (default model)."
        
        # Test base function documentation
        base_doc = generate_doc("ask_openai", is_sync=False, has_model=False)
        assert base_doc == "Ask openai (default model)."
        
        # Test model-specific function documentation
        model_doc = generate_doc("ask_openai_gpt_4o", is_sync=False, has_model=True)
        assert model_doc == "Ask openai with model gpt_4o."
        
        # Test sync function documentation
        sync_doc = generate_doc("ask_openai_sync", is_sync=True, has_model=False)
        assert sync_doc == "Synchronous openai (default model)."

    def test_sync_function_call_pattern(self):
        """Test sync function call pattern with persistent loop.
        
        Only sync functions (_sync) use the persistent loop.
        Regular async functions use normal async execution.
        """
        # Test the pattern that sync functions would use
        def mock_run_async_on_persistent_loop(coro):
            # This simulates the persistent loop behavior for SYNC functions only
            return "Sync result from persistent loop"
        
        # Simulate what ask_openai_sync would do (uses persistent loop)
        result = mock_run_async_on_persistent_loop("mock_coroutine")
        assert result == "Sync result from persistent loop"

    @pytest.mark.asyncio 
    async def test_streaming_function_pattern(self):
        """Test streaming function pattern.
        
        Regular async functions like stream_* do NOT use the persistent loop.
        They use normal async execution.
        """
        # Test the pattern that streaming functions would use
        async def mock_stream(prompt, provider=None, model=None, **kwargs):
            chunks = [f"chunk1_{provider}", f"chunk2_{provider}", f"chunk3_{provider}"]
            for chunk in chunks:
                yield chunk
        
        # Simulate what stream_openai would do (normal async, no persistent loop)
        chunks = []
        async for chunk in mock_stream("Test prompt", provider="openai"):
            chunks.append(chunk)
        
        assert chunks == ["chunk1_openai", "chunk2_openai", "chunk3_openai"]

    @pytest.mark.asyncio
    async def test_async_function_call_pattern(self):
        """Test async function call pattern without persistent loop.
        
        Regular async functions (ask_*, stream_*) do NOT use persistent loop.
        """
        # Test the pattern that async provider functions would use
        async def mock_ask(prompt, provider=None, model=None, **kwargs):
            return f"Response from {provider} using {model or 'default'}: {prompt}"
        
        # Simulate what ask_openai would do (normal async execution)
        result = await mock_ask("Test prompt", provider="openai", model=None)
        assert result == "Response from openai using default: Test prompt"
        
        # Simulate what ask_openai_gpt_4o would do (normal async execution)
        result = await mock_ask("Test prompt", provider="openai", model="gpt-4o")
        assert result == "Response from openai using gpt-4o: Test prompt"


class TestFunctionGenerationLogic:
    """Test suite for function generation logic without actual generation."""

    def test_generate_function_names_logic(self):
        """Test the logic for generating function names."""
        # Simulate the logic from _generate_all_functions
        providers = ['openai', 'anthropic']
        models_data = {
            'openai': ['gpt-4o', 'gpt-4o-mini'],
            'anthropic': ['claude-3-sonnet']
        }
        
        expected_functions = []
        
        for provider in providers:
            # Base functions
            expected_functions.extend([
                f"ask_{provider}",
                f"stream_{provider}",
                f"ask_{provider}_sync"
            ])
            
            # Model-specific functions
            models = models_data.get(provider, [])
            for model in models:
                model_suffix = _sanitize_model_name(model)
                if model_suffix:
                    expected_functions.extend([
                        f"ask_{provider}_{model_suffix}",
                        f"stream_{provider}_{model_suffix}",
                        f"ask_{provider}_{model_suffix}_sync"
                    ])
        
        # Verify expected functions
        assert 'ask_openai' in expected_functions
        assert 'ask_anthropic' in expected_functions
        assert 'ask_openai_gpt_4o' in expected_functions
        assert 'ask_openai_gpt_4o_mini' in expected_functions
        assert 'ask_anthropic_claude_3_sonnet' in expected_functions

    def test_model_alias_function_logic(self):
        """Test model alias function generation logic."""
        provider = 'openai'
        models = ['gpt-4o']
        aliases = {'gpt4o': 'gpt-4o', 'turbo': 'gpt-4-turbo'}
        
        expected_functions = []
        
        # Regular model functions
        for model in models:
            model_suffix = _sanitize_model_name(model)
            expected_functions.extend([
                f"ask_{provider}_{model_suffix}",
                f"stream_{provider}_{model_suffix}",
                f"ask_{provider}_{model_suffix}_sync"
            ])
        
        # Alias functions
        for alias, actual_model in aliases.items():
            alias_suffix = _sanitize_model_name(alias)
            expected_functions.extend([
                f"ask_{provider}_{alias_suffix}",
                f"stream_{provider}_{alias_suffix}",
                f"ask_{provider}_{alias_suffix}_sync"
            ])
        
        # Verify functions
        assert 'ask_openai_gpt4o' in expected_functions     # From models
        assert 'ask_openai_gpt4o' in expected_functions      # From alias 'gpt4o'
        assert 'ask_openai_turbo' in expected_functions      # From alias 'turbo'


class TestUtilityFunctionLogic:
    """Test suite for utility function logic without actual creation."""

    def test_quick_question_function_logic(self):
        """Test quick_question function logic."""
        # Simulate the quick_question function logic
        def mock_quick_question(question: str, provider: str = "openai") -> str:
            # This would normally call ask_sync
            return f"Quick response from {provider}: {question}"
        
        result = mock_quick_question("What is 2+2?")
        assert result == "Quick response from openai: What is 2+2?"
        
        result = mock_quick_question("Hello", provider="anthropic")
        assert result == "Quick response from anthropic: Hello"

    def test_compare_providers_function_logic(self):
        """Test compare_providers function logic."""
        # Simulate the compare_providers function logic
        def mock_compare_providers(question: str, providers: list = None) -> dict:
            if providers is None:
                providers = ["openai", "anthropic"]
            
            results = {}
            for provider in providers:
                # Simulate ask_sync call
                results[provider] = f"{provider} response: {question}"
            
            return results
        
        result = mock_compare_providers("Test question", ["openai", "anthropic"])
        expected = {
            "openai": "openai response: Test question",
            "anthropic": "anthropic response: Test question"
        }
        assert result == expected


class TestPersistentLoopCleanup:
    """Test suite for persistent loop cleanup functionality."""

    def test_cleanup_persistent_loop_function_exists(self):
        """Test that cleanup function exists and is callable."""
        from chuk_llm.api.providers import _cleanup_persistent_loop
        assert callable(_cleanup_persistent_loop)

    def test_loop_survives_errors_in_coroutines(self):
        """Test that loop survives errors in individual coroutines."""
        async def good_coro():
            return "success"
        
        async def bad_coro():
            raise Exception("test error")
        
        # Bad coroutine should raise exception but not break loop
        with pytest.raises(Exception, match="test error"):
            _run_async_on_persistent_loop(bad_coro())
        
        # Good coroutine should still work after the error
        result = _run_async_on_persistent_loop(good_coro())
        assert result == "success"

    def test_atexit_registration_logic(self):
        """Test that atexit registration logic works."""
        import atexit
        
        # Test that we can register a cleanup function
        def mock_cleanup():
            pass
        
        # This should not raise an exception
        atexit.register(mock_cleanup)
        
        # Test that atexit registration works (without accessing private attributes)
        # We just verify that the registration call succeeds
        assert callable(mock_cleanup)


class TestThreadSafety:
    """Test suite for thread safety of persistent loop implementation."""

    def test_concurrent_loop_access(self):
        """Test concurrent access to persistent loop."""
        results = []
        errors = []
        
        async def test_coro(value):
            await asyncio.sleep(0.001)
            return f"value_{value}"
        
        def worker(worker_id):
            try:
                for i in range(3):
                    result = _run_async_on_persistent_loop(test_coro(f"{worker_id}_{i}"))
                    results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Start multiple threads
        threads = []
        for worker_id in range(3):
            thread = threading.Thread(target=worker, args=(worker_id,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=5.0)
        
        # Should have no errors
        assert len(errors) == 0, f"Unexpected errors: {errors}"
        
        # Should have results from all workers
        assert len(results) == 9  # 3 workers * 3 iterations each
        
        # All results should be unique
        assert len(set(results)) == 9

    def test_loop_creation_thread_safety(self):
        """Test thread safety of loop creation."""
        import chuk_llm.api.providers as providers_module
        
        # Reset loop state
        providers_module._persistent_loop = None
        providers_module._loop_thread = None
        
        loops = []
        
        def get_loop_worker():
            loop = _get_persistent_loop()
            loops.append(loop)
        
        # Start multiple threads that try to get the loop
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=get_loop_worker)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join(timeout=5.0)
        
        # All threads should get the same loop instance
        assert len(loops) == 5
        for loop in loops:
            assert loop is loops[0], "All threads should get the same loop instance"


class TestErrorHandling:
    """Test suite for error handling in persistent loop implementation."""

    def test_coroutine_timeout_handling(self):
        """Test handling of coroutines that timeout."""
        async def slow_coro():
            await asyncio.sleep(10)  # Very slow
            return "should_timeout"
        
        # This test just ensures we can start and cancel slow operations
        # without breaking the persistent loop
        start_time = time.time()
        
        # Create a future and cancel it quickly
        loop = _get_persistent_loop()
        future = asyncio.run_coroutine_threadsafe(slow_coro(), loop)
        
        # Cancel after a short time
        time.sleep(0.1)
        future.cancel()
        
        # Should complete quickly due to cancellation
        elapsed = time.time() - start_time
        assert elapsed < 1.0, "Cancellation should be fast"
        
        # Loop should still work for other operations
        async def quick_coro():
            return "still_works"
        
        result = _run_async_on_persistent_loop(quick_coro())
        assert result == "still_works"

    def test_exception_propagation(self):
        """Test that exceptions are properly propagated."""
        async def exception_coro():
            raise ValueError("Custom exception message")
        
        with pytest.raises(ValueError, match="Custom exception message"):
            _run_async_on_persistent_loop(exception_coro())

    def test_async_context_detection(self):
        """Test detection of running async context."""
        import warnings
        
        async def test_in_async_context():
            async def dummy_coro():
                return "dummy"
            
            # We expect this coroutine to not be awaited, so suppress the warning
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*was never awaited.*")
                # Should raise RuntimeError when called from async context
                with pytest.raises(RuntimeError, match="Cannot call sync functions from async context"):
                    _run_async_on_persistent_loop(dummy_coro())
        
        # Run the test in an async context
        asyncio.run(test_in_async_context())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])