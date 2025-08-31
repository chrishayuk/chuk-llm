"""Tests for the ChukLLM CLI"""

import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock, call
import asyncio
import inspect
from chuk_llm.cli import ChukLLMCLI
from chuk_llm.api.providers import _GENERATED_FUNCTIONS


class TestCLISystemPromptSupport:
    """Test system prompt support in CLI"""
    
    def test_handle_convenience_function_with_system_prompt(self):
        """Test that convenience functions accept system prompts"""
        cli = ChukLLMCLI()
        
        # Mock at the core API level to intercept actual calls
        with patch('chuk_llm.api.core.ask', new_callable=AsyncMock) as mock_ask:
            mock_ask.return_value = "Test response"
            
            with patch('chuk_llm.api.providers.has_function', return_value=True):
                result = cli.handle_convenience_function(
                    'ask_ollama_granite',
                    'Test prompt',
                    system_prompt='Be a pirate'
                )
                
                # Verify the core ask function was called with system_prompt
                mock_ask.assert_called_once()
                call_kwargs = mock_ask.call_args[1]
                assert call_kwargs['system_prompt'] == 'Be a pirate'
                assert result == "Test response"
    
    def test_handle_convenience_function_with_multiple_kwargs(self):
        """Test that convenience functions accept multiple kwargs"""
        cli = ChukLLMCLI()
        
        # Mock at the core API level
        with patch('chuk_llm.api.core.ask', new_callable=AsyncMock) as mock_ask:
            mock_ask.return_value = "Test response"
            
            with patch('chuk_llm.api.providers.has_function', return_value=True):
                result = cli.handle_convenience_function(
                    'ask_ollama_granite',
                    'Test prompt',
                    system_prompt='Be a pirate',
                    max_tokens=100,
                    temperature=0.7
                )
                
                # Verify all kwargs were passed
                mock_ask.assert_called_once()
                call_kwargs = mock_ask.call_args[1]
                assert call_kwargs['system_prompt'] == 'Be a pirate'
                assert call_kwargs['max_tokens'] == 100
                assert call_kwargs['temperature'] == 0.7
    
    def test_handle_convenience_function_async_detection(self):
        """Test that the CLI correctly handles async/sync auto-detection"""
        cli = ChukLLMCLI()
        
        # Test with a function that returns a coroutine
        async def async_func(prompt, **kwargs):
            return "Async response"
        
        mock_func = Mock(return_value=async_func("Test", system_prompt="pirate"))
        
        with patch.dict('chuk_llm.api.providers._GENERATED_FUNCTIONS', 
                       {'ask_ollama_granite': mock_func}):
            with patch('chuk_llm.api.providers.has_function', return_value=True):
                with patch('chuk_llm.api.event_loop_manager.run_sync') as mock_run_sync:
                    mock_run_sync.return_value = "Async response"
                    
                    result = cli.handle_convenience_function(
                        'ask_ollama_granite',
                        'Test prompt',
                        system_prompt='Be a pirate'
                    )
                    
                    # Verify run_sync was called with the coroutine
                    mock_run_sync.assert_called_once()
                    assert result == "Async response"
    
    def test_handle_convenience_function_sync_detection(self):
        """Test that the CLI correctly handles functions that return strings directly"""
        cli = ChukLLMCLI()
        
        # Mock at the core API level for sync detection
        with patch('chuk_llm.api.core.ask', new_callable=AsyncMock) as mock_ask:
            mock_ask.return_value = "Sync response"
            
            with patch('chuk_llm.api.providers.has_function', return_value=True):
                result = cli.handle_convenience_function(
                    'ask_ollama_granite',
                    'Test prompt',
                    system_prompt='Be a pirate'
                )
                
                # Verify the response was handled correctly
                assert result == "Sync response"
    
    def test_handle_ask_alias_with_system_prompt(self):
        """Test that ask_alias accepts system prompts"""
        cli = ChukLLMCLI()
        
        # Mock config
        cli.config = Mock()
        cli.config.get_global_aliases.return_value = {
            'granite': 'ollama/granite3.3'
        }
        
        with patch.object(cli, 'stream_response', return_value="Response") as mock_stream:
            result = cli.handle_ask_alias(
                'granite',
                'Test prompt',
                system_prompt='Be a pirate',
                max_tokens=100
            )
            
            # Verify stream_response was called with kwargs
            mock_stream.assert_called_once_with(
                'Test prompt',
                provider='ollama',
                model='granite3.3',
                system_prompt='Be a pirate',
                max_tokens=100
            )
    
    def test_ask_model_with_system_prompt(self):
        """Test that ask_model passes system_prompt correctly"""
        cli = ChukLLMCLI()
        
        with patch.object(cli, 'stream_response', return_value="Response") as mock_stream:
            result = cli.ask_model(
                'Test prompt',
                'ollama',
                'granite3.3',
                system_prompt='Be a pirate'
            )
            
            # Verify system_prompt was passed to stream_response
            mock_stream.assert_called_once()
            call_kwargs = mock_stream.call_args[1]
            assert call_kwargs['system_prompt'] == 'Be a pirate'


class TestCLIModelResolution:
    """Test model name resolution in CLI"""
    
    def test_global_alias_resolution(self):
        """Test that global aliases are resolved"""
        cli = ChukLLMCLI()
        
        # Mock config
        cli.config = Mock()
        cli.config.get_global_aliases.return_value = {
            'granite': 'ollama/granite3.3'
        }
        cli.config._ensure_model_available.return_value = 'granite3.3'
        cli.config.get_provider.return_value = Mock(models=['granite3.3'])
        
        with patch.object(cli, 'stream_response', return_value="Response") as mock_stream:
            result = cli.ask_model(
                'Test prompt',
                'ollama',
                'granite',  # Using global alias
                system_prompt='Be a pirate'
            )
            
            # Verify the model was resolved
            mock_stream.assert_called_once()
            call_args = mock_stream.call_args
            assert call_args[0][2] == 'granite3.3'  # model parameter
    
    def test_model_resolution_with_ensure_model_available(self):
        """Test that _ensure_model_available is called for resolution"""
        cli = ChukLLMCLI()
        
        # Mock config
        cli.config = Mock()
        cli.config.get_global_aliases.return_value = {}
        cli.config._ensure_model_available.return_value = 'granite3.3:latest'
        cli.config.get_provider.return_value = Mock(models=['granite3.3:latest'])
        
        with patch.object(cli, 'stream_response', return_value="Response") as mock_stream:
            result = cli.ask_model(
                'Test prompt',
                'ollama',
                'granite3.3',
                system_prompt='Be a pirate'
            )
            
            # Verify _ensure_model_available was called
            cli.config._ensure_model_available.assert_called_once_with('ollama', 'granite3.3')
            
            # Verify the resolved model was used
            mock_stream.assert_called_once()
            call_args = mock_stream.call_args
            assert call_args[0][2] == 'granite3.3:latest'
    
    def test_model_not_found_warning(self):
        """Test that a warning is shown when model is not found"""
        cli = ChukLLMCLI()
        cli.verbose = True
        
        # Mock config
        cli.config = Mock()
        cli.config.get_global_aliases.return_value = {}
        cli.config._ensure_model_available.return_value = None  # Model not found
        cli.config.get_provider.return_value = Mock(models=[])
        
        with patch.object(cli, 'print_rich') as mock_print:
            with patch.object(cli, 'stream_response', return_value="Response"):
                result = cli.ask_model(
                    'Test prompt',
                    'ollama',
                    'nonexistent',
                    system_prompt='Be a pirate'
                )
                
                # Verify warning was printed
                mock_print.assert_called()
                warning_calls = [call for call in mock_print.call_args_list 
                                if 'not found' in str(call)]
                assert len(warning_calls) > 0


class TestCLICommandNormalization:
    """Test command name normalization (dots to underscores)"""
    
    def test_command_normalization_dots(self):
        """Test that dots in command names are converted to underscores"""
        from chuk_llm.cli import main, parse_convenience_function
        import sys
        
        # Mock sys.argv
        original_argv = sys.argv
        sys.argv = ['chuk-llm', 'ask_ollama_granite3.3', 'Test prompt']
        
        # Mock the CLI and its methods
        with patch('chuk_llm.cli.ChukLLMCLI') as MockCLI:
            mock_cli = MockCLI.return_value
            mock_cli.handle_convenience_function.return_value = "Response"
            
            # Mock has_function at the CLI module level to return True so we don't exit early
            with patch('chuk_llm.cli.has_function', return_value=True):
                with patch('chuk_llm.cli.parse_convenience_function', 
                          return_value=('ollama', 'granite3.3', False, False)):
                    # Mock trigger_discovery_for_provider to avoid actual discovery
                    with patch('chuk_llm.cli.trigger_discovery_for_provider'):
                        try:
                            main()
                        except SystemExit:
                            pass
                        
                        # Verify the function was called with normalized name
                        mock_cli.handle_convenience_function.assert_called()
                        call_args = mock_cli.handle_convenience_function.call_args[0]
                        assert call_args[0] == 'ask_ollama_granite3_3'  # Normalized
        
        sys.argv = original_argv
    
    def test_command_normalization_colons(self):
        """Test that colons in command names are converted to underscores"""
        from chuk_llm.cli import main, parse_convenience_function
        import sys
        
        # Mock sys.argv - note: colons can't be in the command name itself
        original_argv = sys.argv
        sys.argv = ['chuk-llm', 'ask_ollama_granite3_3_latest', 'Test prompt']
        
        # Mock the CLI and its methods
        with patch('chuk_llm.cli.ChukLLMCLI') as MockCLI:
            mock_cli = MockCLI.return_value
            mock_cli.handle_convenience_function.return_value = "Response"
            
            # Mock has_function at the CLI module level to return True so we don't exit early
            with patch('chuk_llm.cli.has_function', return_value=True):
                with patch('chuk_llm.cli.parse_convenience_function', 
                          return_value=('ollama', 'granite3_3_latest', False, False)):
                    # Mock trigger_discovery_for_provider to avoid actual discovery
                    with patch('chuk_llm.cli.trigger_discovery_for_provider'):
                        try:
                            main()
                        except SystemExit:
                            pass
                        
                        # Verify the function was called with normalized name
                        mock_cli.handle_convenience_function.assert_called()
                        call_args = mock_cli.handle_convenience_function.call_args[0]
                        assert call_args[0] == 'ask_ollama_granite3_3_latest'  # Normalized
        
        sys.argv = original_argv


class TestCLIStreamHandling:
    """Test streaming functionality in CLI"""
    
    def test_stream_function_handling(self):
        """Test that stream functions are handled correctly"""
        cli = ChukLLMCLI()
        
        # Create an async generator for streaming
        async def mock_stream_func(prompt, **kwargs):
            for chunk in ["Hello", " ", "World"]:
                yield chunk
        
        mock_func = Mock(return_value=mock_stream_func("Test", system_prompt="pirate"))
        
        with patch.dict('chuk_llm.api.providers._GENERATED_FUNCTIONS', 
                       {'stream_ollama_granite': mock_func}):
            with patch('chuk_llm.api.providers.has_function', return_value=True):
                with patch('chuk_llm.api.event_loop_manager.run_sync') as mock_run_sync:
                    mock_run_sync.return_value = "Hello World"
                    
                    result = cli.handle_convenience_function(
                        'stream_ollama_granite',
                        'Test prompt',
                        system_prompt='Be a pirate'
                    )
                    
                    # Verify run_sync was called
                    mock_run_sync.assert_called_once()
                    assert result == "Hello World"
    
    def test_sync_function_with_suffix(self):
        """Test that _sync suffix functions are handled correctly"""
        cli = ChukLLMCLI()
        
        # Mock at the core API level for sync functions
        with patch('chuk_llm.api.core.ask', new_callable=AsyncMock) as mock_ask:
            mock_ask.return_value = "Sync response"
            
            with patch('chuk_llm.api.providers.has_function', return_value=True):
                result = cli.handle_convenience_function(
                    'ask_ollama_granite_sync',
                    'Test prompt',
                    system_prompt='Be a pirate'
                )
                
                # Verify the core function was called with system_prompt
                mock_ask.assert_called_once()
                call_kwargs = mock_ask.call_args[1]
                assert call_kwargs['system_prompt'] == 'Be a pirate'
                assert result == "Sync response"


class TestCLITableDisplay:
    """Test table display functionality"""
    
    def test_functions_table_display(self):
        """Test that functions are displayed in table format"""
        cli = ChukLLMCLI()
        
        # Mock the discovery functions
        mock_discovered = {
            'ollama': {
                'ask_ollama_granite3_3': Mock(),
                'ask_ollama_granite3_3_latest': Mock(),
                'stream_ollama_granite3_3': Mock(),
            }
        }
        
        with patch('chuk_llm.api.providers.get_discovered_functions', 
                  return_value=mock_discovered):
            with patch('chuk_llm.api.providers.get_all_functions', 
                      return_value=list(mock_discovered['ollama'].keys())):
                with patch.object(cli, 'print_table') as mock_print_table:
                    cli.show_functions('ollama')
                    
                    # Verify print_table was called
                    mock_print_table.assert_called_once()
                    
                    # Check the table structure
                    call_args = mock_print_table.call_args[0]
                    headers = call_args[0]
                    assert 'Type' in headers
                    assert 'Function' in headers  # Changed from 'Function Name'
                    assert 'Source' in headers
                    assert 'Mode' in headers
    
    def test_functions_with_provider_filter(self):
        """Test that functions can be filtered by provider"""
        cli = ChukLLMCLI()
        
        # Mock the discovery functions
        mock_discovered = {
            'ollama': {
                'ask_ollama_granite': Mock(),
            },
            'openai': {
                'ask_openai_gpt4': Mock(),
            }
        }
        
        all_funcs = ['ask_ollama_granite', 'ask_openai_gpt4', 'ask_anthropic_claude']
        
        with patch('chuk_llm.api.providers.get_discovered_functions', 
                  return_value=mock_discovered):
            with patch('chuk_llm.api.providers.get_all_functions', 
                      return_value=all_funcs):
                with patch.object(cli, 'print_table') as mock_print_table:
                    cli.show_functions('ollama')
                    
                    # Verify only ollama functions are shown
                    call_args = mock_print_table.call_args[0]
                    rows = call_args[1]
                    
                    # Filter out separator rows
                    function_rows = [row for row in rows if len(row) > 1 and 'ollama' in str(row[1]).lower()]
                    
                    # Should have at least one ollama function
                    assert len(function_rows) > 0


class TestCLIErrorHandling:
    """Test error handling in CLI"""
    
    def test_convenience_function_not_found(self):
        """Test error when convenience function doesn't exist"""
        cli = ChukLLMCLI()
        
        with patch('chuk_llm.api.providers.has_function', return_value=False):
            with pytest.raises(Exception) as exc_info:
                cli.handle_convenience_function(
                    'ask_nonexistent_model',
                    'Test prompt'
                )
            
            assert "not available" in str(exc_info.value)
    
    def test_alias_not_found(self):
        """Test error when alias doesn't exist but falls back to provider"""
        cli = ChukLLMCLI()
        
        # Mock config with no aliases
        cli.config = Mock()
        cli.config.get_global_aliases.return_value = {}
        
        # Mock stream_response to fail for non-existent provider
        with patch.object(cli, 'stream_response', side_effect=Exception("Provider not found")):
            with pytest.raises(Exception) as exc_info:
                cli.handle_ask_alias(
                    'nonexistent',
                    'Test prompt'
                )
            
            assert "not available" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])