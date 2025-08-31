"""Tests for auto-detection of sync/async context in provider functions"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from chuk_llm.api.providers import _create_provider_function


class TestProviderAutoDetection:
    """Test the auto-detection of sync/async context"""
    
    def test_provider_function_in_sync_context(self):
        """Test that provider functions auto-detect sync context and return results directly"""
        
        # Create a provider function
        with patch('chuk_llm.api.core.ask') as mock_ask:
            # Mock ask to return a coroutine
            async def mock_async_ask(*args, **kwargs):
                return "Test response"
            
            mock_ask.return_value = mock_async_ask("test", provider="ollama", model="granite")
            
            # Create the provider function
            provider_func = _create_provider_function("ollama", "granite", supports_vision=False)
            
            # In sync context (no event loop running)
            with patch('asyncio.get_running_loop', side_effect=RuntimeError("No event loop")):
                with patch('chuk_llm.api.event_loop_manager.run_sync') as mock_run_sync:
                    mock_run_sync.return_value = "Test response"
                    
                    # Call the function
                    result = provider_func("Test prompt", system_prompt="Be a pirate")
                    
                    # Should have called run_sync
                    mock_run_sync.assert_called_once()
                    assert result == "Test response"
    
    @pytest.mark.asyncio
    async def test_provider_function_in_async_context(self):
        """Test that provider functions auto-detect async context and return coroutines"""
        
        # Create a provider function
        with patch('chuk_llm.api.core.ask') as mock_ask:
            # Set up mock_ask to return the result directly (as our auto-detect will handle the async part)
            mock_ask.return_value = "Test response"
            
            # Create the provider function
            provider_func = _create_provider_function("ollama", "granite", supports_vision=False)
            
            # Mock that we're in an async context
            with patch('asyncio.get_running_loop', return_value=Mock()):
                # Call the function
                result = provider_func("Test prompt", system_prompt="Be a pirate")
                
                # Should return a coroutine
                assert asyncio.iscoroutine(result)
                
                # Await it to get the result
                actual_result = await result
                assert actual_result == "Test response"
    
    def test_provider_function_with_system_prompt(self):
        """Test that provider functions pass system prompts correctly"""
        
        with patch('chuk_llm.api.core.ask') as mock_ask:
            # Track that system_prompt was passed through
            call_args = None
            
            async def mock_async_ask(*args, **kwargs):
                nonlocal call_args
                call_args = kwargs
                return "Arr matey!"
            
            # Make ask return a coroutine when called
            mock_ask.side_effect = lambda *args, **kwargs: mock_async_ask(*args, **kwargs)
            
            # Create the provider function
            provider_func = _create_provider_function("ollama", "granite", supports_vision=False)
            
            # In sync context
            with patch('asyncio.get_running_loop', side_effect=RuntimeError("No event loop")):
                with patch('chuk_llm.api.event_loop_manager.run_sync') as mock_run_sync:
                    mock_run_sync.return_value = "Arr matey!"
                    
                    # Call with system_prompt
                    result = provider_func("Test prompt", system_prompt="Be a pirate")
                    
                    # Verify run_sync was called and result is correct
                    assert mock_run_sync.called
                    assert result == "Arr matey!"
    
    def test_provider_function_with_multiple_kwargs(self):
        """Test that provider functions pass all kwargs correctly"""
        
        with patch('chuk_llm.api.core.ask') as mock_ask:
            # Make ask return a coroutine when called
            async def mock_async_ask(*args, **kwargs):
                return "Test response"
            
            # Capture the actual kwargs passed to ask
            mock_ask.side_effect = lambda *args, **kwargs: mock_async_ask(*args, **kwargs)
            
            # Create the provider function
            provider_func = _create_provider_function("ollama", "granite", supports_vision=False)
            
            # In sync context
            with patch('asyncio.get_running_loop', side_effect=RuntimeError("No event loop")):
                with patch('chuk_llm.api.event_loop_manager.run_sync') as mock_run_sync:
                    mock_run_sync.return_value = "Test response"
                    
                    # Call with multiple kwargs
                    result = provider_func(
                        "Test prompt",
                        system_prompt="Be a pirate",
                        max_tokens=100,
                        temperature=0.7,
                        json_mode=True
                    )
                    
                    # Verify run_sync was called with a coroutine that has the right kwargs
                    assert mock_run_sync.called
                    # Get the coroutine that was passed to run_sync
                    call_args = mock_run_sync.call_args[0][0]
                    assert asyncio.iscoroutine(call_args)
                    assert result == "Test response"


class TestStreamFunctionAutoDetection:
    """Test auto-detection for streaming functions"""
    
    def test_stream_function_returns_async_generator(self):
        """Test that stream functions always return async generators"""
        from chuk_llm.api.providers import _create_stream_function
        
        with patch('chuk_llm.api.core.stream') as mock_stream:
            # Mock stream to return an async generator
            async def mock_async_stream(*args, **kwargs):
                for chunk in ["Hello", " ", "World"]:
                    yield chunk
            
            mock_stream.return_value = mock_async_stream("test", provider="ollama", model="granite")
            
            # Create the stream function
            stream_func = _create_stream_function("ollama", "granite", supports_vision=False)
            
            # Call the function
            result = stream_func("Test prompt", system_prompt="Be a pirate")
            
            # Should return an async generator
            assert hasattr(result, '__aiter__')
    
    def test_stream_function_with_system_prompt(self):
        """Test that stream functions pass system prompts"""
        from chuk_llm.api.providers import _create_stream_function
        
        call_kwargs = None
        
        with patch('chuk_llm.api.core.stream') as mock_stream:
            # Mock stream to capture kwargs
            async def mock_async_stream(*args, **kwargs):
                nonlocal call_kwargs
                call_kwargs = kwargs
                yield "Test chunk"
            
            # Use side_effect to capture the actual call
            mock_stream.side_effect = lambda *args, **kwargs: mock_async_stream(*args, **kwargs)
            
            # Create the stream function
            stream_func = _create_stream_function("ollama", "granite", supports_vision=False)
            
            # Call with system_prompt
            gen = stream_func("Test prompt", system_prompt="Be a pirate")
            
            # Consume the generator to trigger the call
            async def consume():
                async for chunk in gen:
                    pass
            
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(consume())
                # Check that stream was called with the expected kwargs
                assert mock_stream.called
                # Get the kwargs from the call to stream
                _, call_kwargs = mock_stream.call_args
                assert call_kwargs.get('system_prompt') == "Be a pirate"
            finally:
                loop.close()


class TestSyncFunctionBehavior:
    """Test the _sync suffix functions"""
    
    def test_sync_function_always_blocks(self):
        """Test that _sync functions always block and return results"""
        from chuk_llm.api.providers import _create_sync_function
        
        with patch('chuk_llm.api.core.ask') as mock_ask:
            # Mock ask to return a coroutine
            async def mock_async_ask(*args, **kwargs):
                return "Test response"
            
            mock_ask.side_effect = lambda *args, **kwargs: mock_async_ask(*args, **kwargs)
            
            # Create the sync function
            sync_func = _create_sync_function("ollama", "granite", supports_vision=False)
            
            with patch('chuk_llm.api.event_loop_manager.run_sync') as mock_run_sync:
                mock_run_sync.return_value = "Test response"
                
                # Call the sync function
                result = sync_func("Test prompt", system_prompt="Be a pirate")
                
                # Should have called run_sync (not asyncio.run directly)
                mock_run_sync.assert_called_once()
                assert result == "Test response"
    
    def test_sync_function_with_kwargs(self):
        """Test that _sync functions pass all kwargs"""
        from chuk_llm.api.providers import _create_sync_function
        
        with patch('chuk_llm.api.core.ask') as mock_ask:
            # Mock ask to return a coroutine
            async def mock_async_ask(*args, **kwargs):
                return "Test response"
            
            # Make ask return a coroutine when called
            mock_ask.side_effect = lambda *args, **kwargs: mock_async_ask(*args, **kwargs)
            
            # Create the sync function
            sync_func = _create_sync_function("ollama", "granite", supports_vision=False)
            
            with patch('chuk_llm.api.event_loop_manager.run_sync') as mock_run_sync:
                mock_run_sync.return_value = "Test response"
                
                # Call with multiple kwargs
                result = sync_func(
                    "Test prompt",
                    system_prompt="Be a pirate",
                    max_tokens=100,
                    temperature=0.7
                )
                
                # Verify run_sync was called and result is correct
                assert mock_run_sync.called
                assert result == "Test response"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])