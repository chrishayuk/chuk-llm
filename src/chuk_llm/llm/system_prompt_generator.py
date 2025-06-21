# chuk_llm/llm/system_prompt_generator.py
"""
Advanced System Prompt Generator for ChukLLM
===========================================

Generates dynamic system prompts based on tools JSON, user inputs, and provider-specific optimizations.
Integrates with ChukLLM's unified configuration system for provider-aware prompt generation.
"""

import json
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path

from ..configuration import get_config, Feature

logger = logging.getLogger(__name__)


@dataclass
class PromptTemplate:
    """Template configuration for system prompts"""
    name: str
    template: str
    supports_tools: bool = True
    supports_json_mode: bool = False
    provider_specific: Optional[str] = None
    min_context_length: Optional[int] = None


class SystemPromptGenerator:
    """
    Advanced system prompt generator that adapts to different providers and capabilities.
    """
    
    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize the SystemPromptGenerator.
        
        Args:
            provider: Provider name for optimization (e.g., 'openai', 'anthropic')
            model: Model name for specific optimizations
        """
        self.provider = provider
        self.model = model
        self._config_manager = None
        
        # Load provider configuration if available
        try:
            self._config_manager = get_config()
            if provider and not model:
                try:
                    provider_config = self._config_manager.get_provider(provider)
                    self.model = provider_config.default_model
                except Exception as e:
                    logger.debug(f"Could not get default model for {provider}: {e}")
        except Exception as e:
            logger.debug(f"Could not load provider config: {e}")
        
        # Built-in templates
        self.templates = {
            "default": PromptTemplate(
                name="default",
                template="""You are an intelligent assistant with access to tools that can help you answer user questions effectively.

{{ FORMATTING_INSTRUCTIONS }}

{{ TOOL_DEFINITIONS }}

{{ USER_SYSTEM_PROMPT }}

{{ PROVIDER_SPECIFIC_INSTRUCTIONS }}

{{ TOOL_CONFIGURATION }}""",
                supports_tools=True
            ),
            
            "anthropic_optimized": PromptTemplate(
                name="anthropic_optimized",
                template="""You are Claude, an AI assistant created by Anthropic. You have access to tools that can help you provide accurate and helpful responses.

{{ TOOL_DEFINITIONS }}

When using tools:
- Call functions when they would be helpful to answer the user's question
- Use proper JSON format for all function arguments
- Ensure all required parameters are provided
- You can call multiple functions if needed

{{ USER_SYSTEM_PROMPT }}

{{ TOOL_CONFIGURATION }}""",
                supports_tools=True,
                provider_specific="anthropic"
            ),
            
            "openai_optimized": PromptTemplate(
                name="openai_optimized", 
                template="""You are a helpful assistant with access to function calling capabilities.

{{ TOOL_DEFINITIONS }}

Function calling guidelines:
- Use functions when they would help answer the user's question
- Provide all required parameters in the correct JSON format
- You can call multiple functions in parallel if beneficial
- Always use the exact parameter names specified in the function schemas

{{ USER_SYSTEM_PROMPT }}

{{ TOOL_CONFIGURATION }}""",
                supports_tools=True,
                provider_specific="openai"
            ),
            
            "groq_optimized": PromptTemplate(
                name="groq_optimized",
                template="""You are an intelligent assistant with function calling capabilities. You have access to the following tools:

{{ TOOL_DEFINITIONS }}

Important function calling instructions:
- Call functions when they would be helpful to answer the user's question
- Use exact parameter names as specified in the schemas
- Provide arguments in valid JSON format
- Ensure all required parameters are included
- Be precise with argument types and formats

{{ USER_SYSTEM_PROMPT }}

{{ TOOL_CONFIGURATION }}""",
                supports_tools=True,
                provider_specific="groq"
            ),
            
            "json_mode": PromptTemplate(
                name="json_mode",
                template="""You are an assistant that responds in JSON format only.

{{ USER_SYSTEM_PROMPT }}

You must respond with valid JSON only. Do not include any text outside the JSON structure. Do not use markdown code blocks.

{{ TOOL_CONFIGURATION }}""",
                supports_tools=False,
                supports_json_mode=True
            ),
            
            "reasoning": PromptTemplate(
                name="reasoning",
                template="""You are an advanced reasoning AI assistant. Think step by step and show your reasoning process.

{{ TOOL_DEFINITIONS }}

When solving problems:
- Break down complex questions into smaller parts
- Show your reasoning process clearly
- Use tools when they would provide helpful information
- Consider multiple approaches when appropriate

{{ USER_SYSTEM_PROMPT }}

{{ TOOL_CONFIGURATION }}""",
                supports_tools=True
            ),
            
            "minimal": PromptTemplate(
                name="minimal",
                template="""{{ USER_SYSTEM_PROMPT }}

{{ TOOL_DEFINITIONS }}

{{ TOOL_CONFIGURATION }}""",
                supports_tools=True
            )
        }
    
    def generate_prompt(
        self,
        tools: Optional[Union[Dict, List[Dict]]] = None,
        user_system_prompt: Optional[str] = None,
        tool_config: Optional[str] = None,
        template_name: Optional[str] = None,
        json_mode: bool = False,
        **kwargs
    ) -> str:
        """
        Generate a system prompt based on the provided parameters.
        
        Args:
            tools: Tools JSON schema (dict or list of tool dicts)
            user_system_prompt: User-provided system instructions
            tool_config: Additional tool configuration information
            template_name: Specific template to use (overrides auto-selection)
            json_mode: Whether to optimize for JSON output mode
            **kwargs: Additional template variables
        
        Returns:
            Generated system prompt string
        """
        # Auto-select template if not specified
        if not template_name:
            template_name = self._select_template(tools, json_mode, **kwargs)
        
        template = self.templates.get(template_name, self.templates["default"])
        
        # Validate template compatibility
        if tools and not template.supports_tools:
            logger.warning(f"Template '{template_name}' doesn't support tools, but tools were provided")
        
        if json_mode and not template.supports_json_mode and template_name != "json_mode":
            logger.warning(f"Template '{template_name}' may not be optimized for JSON mode")
        
        # Prepare template variables
        variables = self._prepare_template_variables(
            tools=tools,
            user_system_prompt=user_system_prompt,
            tool_config=tool_config,
            template=template,
            **kwargs
        )
        
        # Generate prompt
        prompt = template.template
        for key, value in variables.items():
            placeholder = f"{{{{ {key} }}}}"
            prompt = prompt.replace(placeholder, str(value))
        
        # Clean up any remaining placeholders
        prompt = self._clean_prompt(prompt)
        
        logger.debug(f"Generated system prompt using template '{template_name}' "
                    f"(provider: {self.provider}, tools: {bool(tools)})")
        
        return prompt
    
    def _select_template(self, tools: Optional[Any], json_mode: bool, **kwargs) -> str:
        """Auto-select the best template based on context"""
        # JSON mode takes priority
        if json_mode:
            return "json_mode"
        
        # Provider-specific optimization
        if self.provider:
            provider_template = f"{self.provider}_optimized"
            if provider_template in self.templates:
                return provider_template
        
        # Capability-based selection
        if self._supports_reasoning():
            return "reasoning"
        
        # Default for tools
        if tools:
            return "default"
        
        # Minimal for no tools
        return "minimal"
    
    def _prepare_template_variables(
        self,
        tools: Optional[Any],
        user_system_prompt: Optional[str],
        tool_config: Optional[str],
        template: PromptTemplate,
        **kwargs
    ) -> Dict[str, str]:
        """Prepare all template variables"""
        variables = {}
        
        # Tool definitions
        variables["TOOL_DEFINITIONS"] = self._format_tool_definitions(tools) if tools else ""
        
        # Formatting instructions
        variables["FORMATTING_INSTRUCTIONS"] = self._get_formatting_instructions(tools)
        
        # User system prompt
        variables["USER_SYSTEM_PROMPT"] = user_system_prompt or self._get_default_user_prompt()
        
        # Tool configuration
        variables["TOOL_CONFIGURATION"] = tool_config or self._get_default_tool_config()
        
        # Provider-specific instructions
        variables["PROVIDER_SPECIFIC_INSTRUCTIONS"] = self._get_provider_instructions()
        
        # Add any additional kwargs
        for key, value in kwargs.items():
            if isinstance(value, str):
                variables[key.upper()] = value
        
        return variables
    
    def _format_tool_definitions(self, tools: Any) -> str:
        """Format tools into a clear definition block"""
        if not tools:
            return ""
        
        # Normalize tools to list format
        if isinstance(tools, dict):
            if "functions" in tools:
                tool_list = tools["functions"]
            elif "tools" in tools:
                tool_list = tools["tools"]
            else:
                # Assume it's a single tool
                tool_list = [tools]
        elif isinstance(tools, list):
            tool_list = tools
        else:
            logger.warning(f"Unexpected tools format: {type(tools)}")
            tool_list = []
        
        if not tool_list:
            return ""
        
        # Format based on provider preferences
        if self.provider == "anthropic":
            return self._format_tools_anthropic(tool_list)
        elif self.provider in ["openai", "groq", "deepseek"]:
            return self._format_tools_openai(tool_list)
        else:
            return self._format_tools_generic(tool_list)
    
    def _format_tools_anthropic(self, tools: List[Dict]) -> str:
        """Format tools for Anthropic/Claude"""
        formatted = "Here are the tools available to you:\n\n"
        for i, tool in enumerate(tools, 1):
            func = tool.get("function", tool)
            name = func.get("name", f"tool_{i}")
            description = func.get("description", "No description provided")
            parameters = func.get("parameters", {})
            
            formatted += f"{i}. **{name}**\n"
            formatted += f"   Description: {description}\n"
            
            if parameters and parameters.get("properties"):
                formatted += f"   Parameters:\n"
                for param_name, param_info in parameters["properties"].items():
                    param_type = param_info.get("type", "any")
                    param_desc = param_info.get("description", "")
                    required = param_name in parameters.get("required", [])
                    req_marker = " (required)" if required else " (optional)"
                    formatted += f"   - {param_name} ({param_type}){req_marker}: {param_desc}\n"
            
            formatted += "\n"
        
        return formatted.strip()
    
    def _format_tools_openai(self, tools: List[Dict]) -> str:
        """Format tools for OpenAI-compatible providers"""
        formatted = "You have access to the following functions:\n\n"
        formatted += "```json\n"
        formatted += json.dumps(tools, indent=2)
        formatted += "\n```\n"
        return formatted
    
    def _format_tools_generic(self, tools: List[Dict]) -> str:
        """Generic tool formatting"""
        formatted = "Available tools:\n\n"
        formatted += json.dumps(tools, indent=2)
        return formatted
    
    def _get_formatting_instructions(self, tools: Optional[Any]) -> str:
        """Get provider-specific formatting instructions"""
        if not tools:
            return ""
        
        if self.provider == "anthropic":
            return """When calling functions, use the following format:
- Call functions when they would be helpful
- Provide all arguments in the correct format
- Use proper JSON for complex parameters"""
        
        elif self.provider in ["openai", "groq", "deepseek"]:
            return """Function calling format:
String and scalar parameters should be specified as is, while lists and objects should use JSON format.
Ensure all required parameters are provided with correct types."""
        
        else:
            return """When using tools:
- Follow the parameter specifications exactly
- Use appropriate data types for each parameter
- Include all required parameters"""
    
    def _get_default_user_prompt(self) -> str:
        """Get default user system prompt based on provider"""
        if self.provider == "anthropic":
            return "You are Claude, an AI assistant created by Anthropic. Be helpful, harmless, and honest."
        elif self.provider == "openai":
            return "You are a helpful assistant."
        elif self.provider == "groq":
            return "You are a helpful AI assistant with fast inference capabilities."
        else:
            return "You are an intelligent AI assistant."
    
    def _get_default_tool_config(self) -> str:
        """Get default tool configuration"""
        if self._supports_parallel_tools():
            return "You can call multiple functions simultaneously when beneficial."
        else:
            return "Call functions one at a time as needed."
    
    def _get_provider_instructions(self) -> str:
        """Get provider-specific instructions"""
        if not self.provider:
            return ""
        
        instructions = []
        
        # Provider-specific capabilities
        if self.provider == "groq":
            instructions.append("Take advantage of ultra-fast inference for quick responses.")
        
        elif self.provider == "anthropic":
            instructions.append("Use your reasoning capabilities to provide thoughtful responses.")
        
        elif self.provider == "gemini":
            instructions.append("Leverage your multimodal capabilities when relevant.")
        
        # Feature-specific instructions
        if self._supports_vision():
            instructions.append("You can analyze images when provided.")
        
        if self._supports_json_mode():
            instructions.append("You can provide structured JSON responses when requested.")
        
        return "\n".join(instructions) if instructions else ""
    
    def _supports_reasoning(self) -> bool:
        """Check if provider/model supports reasoning"""
        return self._check_feature(Feature.REASONING)
    
    def _supports_parallel_tools(self) -> bool:
        """Check if provider/model supports parallel function calls"""
        return self._check_feature(Feature.PARALLEL_CALLS)
    
    def _supports_vision(self) -> bool:
        """Check if provider/model supports vision"""
        return self._check_feature(Feature.VISION)
    
    def _supports_json_mode(self) -> bool:
        """Check if provider/model supports JSON mode"""
        return self._check_feature(Feature.JSON_MODE)
    
    def _check_feature(self, feature: Feature) -> bool:
        """Check if a feature is supported"""
        if not self._config_manager or not self.provider:
            return False
        
        try:
            return self._config_manager.supports_feature(self.provider, feature, self.model)
        except Exception:
            return False
    
    def _clean_prompt(self, prompt: str) -> str:
        """Clean up the generated prompt"""
        # Remove empty placeholder lines
        lines = prompt.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Skip lines that are just empty placeholders
            if line.strip().startswith('{{') and line.strip().endswith('}}'):
                continue
            cleaned_lines.append(line)
        
        # Remove excessive blank lines
        final_lines = []
        prev_blank = False
        
        for line in cleaned_lines:
            is_blank = not line.strip()
            if is_blank and prev_blank:
                continue
            final_lines.append(line)
            prev_blank = is_blank
        
        # Remove trailing blank lines
        while final_lines and not final_lines[-1].strip():
            final_lines.pop()
        
        return '\n'.join(final_lines)
    
    def add_template(self, name: str, template: PromptTemplate) -> None:
        """Add a custom template"""
        self.templates[name] = template
        logger.debug(f"Added custom template: {name}")
    
    def get_available_templates(self) -> List[str]:
        """Get list of available template names"""
        return list(self.templates.keys())
    
    def get_template_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific template"""
        template = self.templates.get(name)
        if not template:
            return None
        
        return {
            "name": template.name,
            "supports_tools": template.supports_tools,
            "supports_json_mode": template.supports_json_mode,
            "provider_specific": template.provider_specific,
            "min_context_length": template.min_context_length
        }
    
    @classmethod
    def load_templates_from_file(cls, file_path: Union[str, Path]) -> Dict[str, PromptTemplate]:
        """Load templates from a JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            templates = {}
            for name, template_data in data.items():
                templates[name] = PromptTemplate(**template_data)
            
            logger.info(f"Loaded {len(templates)} templates from {file_path}")
            return templates
            
        except Exception as e:
            logger.error(f"Failed to load templates from {file_path}: {e}")
            return {}
    
    def save_templates_to_file(self, file_path: Union[str, Path]) -> bool:
        """Save current templates to a JSON file"""
        try:
            template_data = {}
            for name, template in self.templates.items():
                template_data[name] = {
                    "name": template.name,
                    "template": template.template,
                    "supports_tools": template.supports_tools,
                    "supports_json_mode": template.supports_json_mode,
                    "provider_specific": template.provider_specific,
                    "min_context_length": template.min_context_length
                }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(template_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(self.templates)} templates to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save templates to {file_path}: {e}")
            return False


# Convenience functions for easy usage
def generate_system_prompt(
    tools: Optional[Any] = None,
    user_prompt: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs
) -> str:
    """
    Convenience function to generate a system prompt.
    
    Args:
        tools: Tools JSON schema
        user_prompt: User system prompt
        provider: Provider name for optimization
        model: Model name for optimization
        **kwargs: Additional options (json_mode, template_name, etc.)
    
    Returns:
        Generated system prompt
    """
    try:
        generator = SystemPromptGenerator(provider=provider, model=model)
        return generator.generate_prompt(
            tools=tools,
            user_system_prompt=user_prompt,
            **kwargs
        )
    except Exception as e:
        logger.error(f"System prompt generation failed: {e}")
        # Fallback to simple prompt
        prompt_parts = []
        if user_prompt:
            prompt_parts.append(user_prompt)
        if tools:
            prompt_parts.append(f"You have access to {len(tools)} tools.")
        return "\n\n".join(prompt_parts) if prompt_parts else "You are a helpful assistant."


# Example usage and testing
if __name__ == "__main__":
    # Example tools schema
    example_tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather information for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA"
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "Temperature unit"
                        }
                    },
                    "required": ["location"]
                }
            }
        }
    ]
    
    # Test different providers
    providers = ["openai", "anthropic", "groq"]
    
    for provider in providers:
        print(f"\n{'='*60}")
        print(f"Testing {provider.upper()} System Prompt")
        print('='*60)
        
        generator = SystemPromptGenerator(provider=provider)
        prompt = generator.generate_prompt(
            tools=example_tools,
            user_system_prompt=f"You are a helpful weather assistant using {provider}."
        )
        
        print(prompt)
        print("\n")