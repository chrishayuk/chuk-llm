# tests/test_system_prompt_generator.py
import json
import pytest
from unittest.mock import Mock, patch
from chuk_llm.llm.system_prompt_generator import SystemPromptGenerator, PromptTemplate, generate_system_prompt


class TestSystemPromptGenerator:
    """Test SystemPromptGenerator functionality"""
    
    def test_initialization_without_provider(self):
        """Test initialization without provider specification"""
        generator = SystemPromptGenerator()
        
        assert generator.provider is None
        assert generator.model is None
        assert "default" in generator.templates
        assert "anthropic_optimized" in generator.templates
        assert "openai_optimized" in generator.templates
    
    def test_initialization_with_provider(self):
        """Test initialization with provider specification"""
        with patch('chuk_llm.llm.system_prompt_generator.get_config') as mock_get_config:
            mock_config = Mock()
            mock_provider = Mock()
            mock_provider.default_model = "gpt-4"
            mock_config.get_provider.return_value = mock_provider
            mock_get_config.return_value = mock_config
            
            generator = SystemPromptGenerator(provider="openai")
            
            assert generator.provider == "openai"
            assert generator.model == "gpt-4"
    
    def test_initialization_with_provider_and_model(self):
        """Test initialization with both provider and model"""
        generator = SystemPromptGenerator(provider="openai", model="gpt-3.5-turbo")
        
        assert generator.provider == "openai"
        assert generator.model == "gpt-3.5-turbo"
    
    def test_generate_prompt_minimal(self):
        """Test generating prompt with minimal parameters"""
        generator = SystemPromptGenerator()
        
        prompt = generator.generate_prompt()
        
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        # Should use minimal template when no tools provided
        assert "intelligent AI assistant" in prompt or "helpful" in prompt
    
    def test_generate_prompt_with_tools_dict(self):
        """Test generating prompt with tools as dictionary"""
        generator = SystemPromptGenerator()
        tools = {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"}
                    },
                    "required": ["location"]
                }
            }
        }
        
        prompt = generator.generate_prompt(tools=tools)
        
        assert isinstance(prompt, str)
        assert "get_weather" in prompt
        assert "weather information" in prompt
        assert "location" in prompt
    
    def test_generate_prompt_with_tools_list(self):
        """Test generating prompt with tools as list"""
        generator = SystemPromptGenerator()
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "Perform calculations",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {"type": "string"}
                        }
                    }
                }
            }
        ]
        
        prompt = generator.generate_prompt(tools=tools)
        
        assert "calculator" in prompt
        assert "calculations" in prompt
    
    def test_generate_prompt_with_custom_user_prompt(self):
        """Test generating prompt with custom user prompt"""
        generator = SystemPromptGenerator()
        custom_prompt = "You are a specialized math assistant."
        
        prompt = generator.generate_prompt(user_system_prompt=custom_prompt)
        
        assert custom_prompt in prompt
    
    def test_generate_prompt_with_tool_config(self):
        """Test generating prompt with tool configuration"""
        generator = SystemPromptGenerator()
        tool_config = "Use tools sparingly and only when necessary."
        
        prompt = generator.generate_prompt(tool_config=tool_config)
        
        assert tool_config in prompt
    
    def test_generate_prompt_json_mode(self):
        """Test generating prompt for JSON mode"""
        generator = SystemPromptGenerator()
        
        prompt = generator.generate_prompt(json_mode=True)
        
        assert "JSON" in prompt
        assert "json" in prompt.lower()
    
    def test_generate_prompt_with_specific_template(self):
        """Test generating prompt with specific template"""
        generator = SystemPromptGenerator()
        
        prompt = generator.generate_prompt(template_name="minimal")
        
        assert isinstance(prompt, str)
        # Minimal template should be shorter and simpler
        assert len(prompt) < 1000  # Reasonable assumption for minimal template
    
    def test_provider_specific_template_selection(self):
        """Test that provider-specific templates are selected automatically"""
        # Test OpenAI
        generator_openai = SystemPromptGenerator(provider="openai")
        prompt_openai = generator_openai.generate_prompt(tools=[{"function": {"name": "test"}}])
        
        # Test Anthropic
        generator_anthropic = SystemPromptGenerator(provider="anthropic")
        prompt_anthropic = generator_anthropic.generate_prompt(tools=[{"function": {"name": "test"}}])
        
        # Test Groq
        generator_groq = SystemPromptGenerator(provider="groq")
        prompt_groq = generator_groq.generate_prompt(tools=[{"function": {"name": "test"}}])
        
        # Should have different content based on provider
        assert prompt_openai != prompt_anthropic
        assert prompt_openai != prompt_groq
        assert prompt_anthropic != prompt_groq
    
    def test_anthropic_tool_formatting(self):
        """Test Anthropic-specific tool formatting"""
        generator = SystemPromptGenerator(provider="anthropic")
        tools = [
            {
                "function": {
                    "name": "search",
                    "description": "Search for information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"}
                        },
                        "required": ["query"]
                    }
                }
            }
        ]
        
        prompt = generator.generate_prompt(tools=tools)
        
        # Anthropic formatting should be human-readable
        assert "**search**" in prompt or "1. search" in prompt
        assert "Search for information" in prompt
        assert "(required)" in prompt or "required" in prompt
    
    def test_openai_tool_formatting(self):
        """Test OpenAI-specific tool formatting"""
        generator = SystemPromptGenerator(provider="openai")
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "Perform calculations"
                }
            }
        ]
        
        prompt = generator.generate_prompt(tools=tools)
        
        # OpenAI formatting should include JSON
        assert "```json" in prompt or "json" in prompt.lower()
        assert "calculator" in prompt
    
    def test_feature_detection_integration(self):
        """Test integration with feature detection"""
        with patch.object(SystemPromptGenerator, '_check_feature') as mock_check:
            # Mock different feature support
            mock_check.side_effect = lambda feature: feature.value in ["tools", "vision"]
            
            generator = SystemPromptGenerator(provider="test")
            prompt = generator.generate_prompt(tools=[{"function": {"name": "test"}}])
            
            assert isinstance(prompt, str)
            # Should have called feature checks
            assert mock_check.call_count > 0
    
    def test_template_compatibility_warnings(self):
        """Test warnings for template compatibility issues"""
        generator = SystemPromptGenerator()
        
        with patch('chuk_llm.llm.system_prompt_generator.logger') as mock_logger:
            # Use JSON mode template with tools (should warn)
            generator.generate_prompt(
                tools=[{"function": {"name": "test"}}],
                template_name="json_mode"
            )
            
            # Should have logged a warning
            mock_logger.warning.assert_called()
    
    def test_clean_prompt_functionality(self):
        """Test prompt cleaning removes empty placeholders"""
        generator = SystemPromptGenerator()
        
        # Create a template with empty placeholders
        test_template = PromptTemplate(
            name="test",
            template="Start\n{{ EMPTY_PLACEHOLDER }}\nMiddle\n{{ ANOTHER_EMPTY }}\nEnd"
        )
        generator.add_template("test", test_template)
        
        prompt = generator.generate_prompt(template_name="test")
        
        # Should not contain placeholder syntax
        assert "{{" not in prompt
        assert "}}" not in prompt
        # Should contain actual content
        assert "Start" in prompt
        assert "Middle" in prompt
        assert "End" in prompt
    
    def test_add_custom_template(self):
        """Test adding custom template"""
        generator = SystemPromptGenerator()
        
        custom_template = PromptTemplate(
            name="custom",
            template="Custom template: {{ USER_SYSTEM_PROMPT }}",
            supports_tools=False
        )
        
        generator.add_template("custom", custom_template)
        
        assert "custom" in generator.templates
        prompt = generator.generate_prompt(
            template_name="custom",
            user_system_prompt="Test prompt"
        )
        assert "Custom template: Test prompt" in prompt
    
    def test_get_available_templates(self):
        """Test getting available template names"""
        generator = SystemPromptGenerator()
        
        templates = generator.get_available_templates()
        
        assert isinstance(templates, list)
        assert "default" in templates
        assert "anthropic_optimized" in templates
        assert "openai_optimized" in templates
        assert "json_mode" in templates
    
    def test_get_template_info(self):
        """Test getting template information"""
        generator = SystemPromptGenerator()
        
        info = generator.get_template_info("default")
        
        assert info is not None
        assert info["name"] == "default"
        assert "supports_tools" in info
        assert "supports_json_mode" in info
        assert "provider_specific" in info
    
    def test_get_template_info_nonexistent(self):
        """Test getting info for nonexistent template"""
        generator = SystemPromptGenerator()
        
        info = generator.get_template_info("nonexistent")
        
        assert info is None
    
    def test_complex_tools_formatting(self):
        """Test formatting of complex nested tool definitions"""
        generator = SystemPromptGenerator()
        complex_tools = [
            {
                "type": "function",
                "function": {
                    "name": "complex_search",
                    "description": "Advanced search with filters",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "filters": {
                                "type": "object",
                                "properties": {
                                    "date_range": {
                                        "type": "object",
                                        "properties": {
                                            "start": {"type": "string"},
                                            "end": {"type": "string"}
                                        }
                                    },
                                    "category": {"type": "string"}
                                }
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        ]
        
        prompt = generator.generate_prompt(tools=complex_tools)
        
        assert "complex_search" in prompt
        assert "Advanced search" in prompt
        assert "filters" in prompt
    
    def test_empty_tools_handling(self):
        """Test handling of empty tools"""
        generator = SystemPromptGenerator()
        
        # Test empty list
        prompt1 = generator.generate_prompt(tools=[])
        assert isinstance(prompt1, str)
        
        # Test empty dict
        prompt2 = generator.generate_prompt(tools={})
        assert isinstance(prompt2, str)
        
        # Test None tools
        prompt3 = generator.generate_prompt(tools=None)
        assert isinstance(prompt3, str)
        
        # Should not contain specific tool definitions
        assert "get_weather" not in prompt1
        assert "calculator" not in prompt1
        
        # But may contain general function/tool guidance
        # This is acceptable behavior for the implementation


class TestPromptTemplate:
    """Test PromptTemplate dataclass"""
    
    def test_prompt_template_creation(self):
        """Test creating PromptTemplate"""
        template = PromptTemplate(
            name="test",
            template="Test template",
            supports_tools=True,
            supports_json_mode=False,
            provider_specific="openai"
        )
        
        assert template.name == "test"
        assert template.template == "Test template"
        assert template.supports_tools is True
        assert template.supports_json_mode is False
        assert template.provider_specific == "openai"
    
    def test_prompt_template_defaults(self):
        """Test PromptTemplate default values"""
        template = PromptTemplate(
            name="minimal",
            template="Minimal template"
        )
        
        assert template.supports_tools is True
        assert template.supports_json_mode is False
        assert template.provider_specific is None
        assert template.min_context_length is None


class TestConvenienceFunction:
    """Test convenience function"""
    
    def test_generate_system_prompt_basic(self):
        """Test basic convenience function usage"""
        prompt = generate_system_prompt()
        
        assert isinstance(prompt, str)
        assert len(prompt) > 0
    
    def test_generate_system_prompt_with_tools(self):
        """Test convenience function with tools"""
        tools = [{"function": {"name": "test", "description": "Test tool"}}]
        
        prompt = generate_system_prompt(tools=tools)
        
        assert "test" in prompt
        assert "Test tool" in prompt
    
    def test_generate_system_prompt_with_provider(self):
        """Test convenience function with provider optimization"""
        with patch('chuk_llm.llm.system_prompt_generator.get_config'):
            prompt = generate_system_prompt(provider="anthropic")
            
            assert isinstance(prompt, str)
    
    def test_generate_system_prompt_error_handling(self):
        """Test convenience function error handling"""
        with patch('chuk_llm.llm.system_prompt_generator.SystemPromptGenerator') as mock_generator:
            mock_generator.side_effect = Exception("Test error")
            
            prompt = generate_system_prompt(user_prompt="Test")
            
            # Should return fallback prompt
            assert "Test" in prompt
    
    def test_generate_system_prompt_fallback_with_tools(self):
        """Test convenience function fallback when tools provided"""
        with patch('chuk_llm.llm.system_prompt_generator.SystemPromptGenerator') as mock_generator:
            mock_generator.side_effect = Exception("Test error")
            
            tools = [{"function": {"name": "test1"}}, {"function": {"name": "test2"}}]
            prompt = generate_system_prompt(tools=tools)
            
            # Should mention number of tools in fallback
            assert "2 tools" in prompt


class TestFeatureIntegration:
    """Test integration with configuration features"""
    
    def test_feature_check_with_config(self):
        """Test feature checking with configuration"""
        with patch('chuk_llm.llm.system_prompt_generator.get_config') as mock_get_config:
            mock_config = Mock()
            mock_config.supports_feature.return_value = True
            mock_get_config.return_value = mock_config
            
            generator = SystemPromptGenerator(provider="test", model="test-model")
            
            # Test feature checking
            supports_vision = generator._supports_vision()
            assert supports_vision is True
            
            mock_config.supports_feature.assert_called()
    
    def test_feature_check_without_config(self):
        """Test feature checking without configuration"""
        generator = SystemPromptGenerator()
        
        # Should handle missing config gracefully
        supports_vision = generator._supports_vision()
        assert supports_vision is False
    
    def test_feature_check_with_error(self):
        """Test feature checking with configuration error"""
        with patch('chuk_llm.llm.system_prompt_generator.get_config') as mock_get_config:
            mock_config = Mock()
            mock_config.supports_feature.side_effect = Exception("Config error")
            mock_get_config.return_value = mock_config
            
            generator = SystemPromptGenerator(provider="test")
            
            # Should handle errors gracefully
            supports_vision = generator._supports_vision()
            assert supports_vision is False


class TestTemplateSelection:
    """Test automatic template selection logic"""
    
    def test_json_mode_template_priority(self):
        """Test that JSON mode takes priority in template selection"""
        generator = SystemPromptGenerator(provider="openai")
        
        # JSON mode should override provider-specific template
        selected = generator._select_template(tools=None, json_mode=True)
        assert selected == "json_mode"
    
    def test_provider_specific_template_selection(self):
        """Test provider-specific template selection"""
        # Test each provider
        for provider in ["openai", "anthropic", "groq"]:
            generator = SystemPromptGenerator(provider=provider)
            selected = generator._select_template(tools=[{"function": {"name": "test"}}], json_mode=False)
            assert selected == f"{provider}_optimized"
    
    def test_reasoning_template_selection(self):
        """Test reasoning template selection"""
        with patch.object(SystemPromptGenerator, '_supports_reasoning', return_value=True):
            generator = SystemPromptGenerator(provider="unknown")  # No provider-specific template
            
            selected = generator._select_template(tools=[{"function": {"name": "test"}}], json_mode=False)
            assert selected == "reasoning"
    
    def test_default_template_with_tools(self):
        """Test default template selection with tools"""
        generator = SystemPromptGenerator(provider="unknown")
        
        with patch.object(generator, '_supports_reasoning', return_value=False):
            selected = generator._select_template(tools=[{"function": {"name": "test"}}], json_mode=False)
            assert selected == "default"
    
    def test_minimal_template_without_tools(self):
        """Test minimal template selection without tools"""
        generator = SystemPromptGenerator(provider="unknown")
        
        selected = generator._select_template(tools=None, json_mode=False)
        assert selected == "minimal"


class TestErrorHandling:
    """Test error handling scenarios"""
    
    def test_invalid_tools_format(self):
        """Test handling of invalid tools format"""
        generator = SystemPromptGenerator()
        
        # Should handle gracefully without crashing
        prompt = generator.generate_prompt(tools="invalid")
        assert isinstance(prompt, str)
    
    def test_missing_template(self):
        """Test handling of missing template"""
        generator = SystemPromptGenerator()
        
        # Should fall back to default template
        prompt = generator.generate_prompt(template_name="nonexistent")
        assert isinstance(prompt, str)
    
    def test_config_loading_failure(self):
        """Test handling configuration loading failure"""
        with patch('chuk_llm.llm.system_prompt_generator.get_config', side_effect=Exception("Config error")):
            # Should not crash
            generator = SystemPromptGenerator(provider="openai")
            assert generator.provider == "openai"
            
            prompt = generator.generate_prompt()
            assert isinstance(prompt, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])