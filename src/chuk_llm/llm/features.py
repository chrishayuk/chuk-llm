# chuk_llm/llm/features.py
from typing import Dict, Any, List, Optional
import logging
import asyncio

from chuk_llm.configuration import get_config, Feature

logger = logging.getLogger(__name__)


class ProviderAdapter:
    """Adapts provider-specific features to common interface using unified config"""
    
    @staticmethod
    def supports_feature(provider: str, feature: Feature, model: Optional[str] = None) -> bool:
        """Check if provider/model supports a feature using unified config"""
        try:
            config = get_config()
            return config.supports_feature(provider, feature, model)
        except Exception as e:
            logger.warning(f"Could not check feature support for {provider}: {e}")
            return False
    
    @staticmethod
    def validate_text_capability(provider: str, model: Optional[str] = None) -> bool:
        """Validate that provider/model supports basic text completion"""
        # TEXT is fundamental - all models should support it
        if not ProviderAdapter.supports_feature(provider, Feature.TEXT, model):
            logger.error(f"{provider}/{model} doesn't support basic text completion")
            return False
        return True
    
    @staticmethod
    def enable_json_mode(provider: str, model: Optional[str], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Enable JSON mode across different providers with feature validation"""
        kwargs = kwargs.copy()
        
        # Check if provider/model supports JSON mode
        if not ProviderAdapter.supports_feature(provider, Feature.JSON_MODE, model):
            logger.warning(f"{provider}/{model} doesn't support JSON mode, using instruction fallback")
            # Use instruction-based fallback for unsupported providers
            kwargs["_json_mode_instruction"] = (
                "You must respond with valid JSON only. "
                "Do not include any text outside the JSON structure."
            )
            return kwargs
        
        # Apply provider-specific JSON mode
        if provider == "openai":
            kwargs["response_format"] = {"type": "json_object"}
        elif provider == "anthropic":
            # Anthropic uses instruction-based approach
            kwargs["_json_mode_instruction"] = (
                "Please respond with valid JSON only. "
                "Do not include any text outside the JSON structure."
            )
        elif provider == "gemini":
            if "generation_config" not in kwargs:
                kwargs["generation_config"] = {}
            kwargs["generation_config"]["response_mime_type"] = "application/json"
        elif provider == "groq":
            kwargs["response_format"] = {"type": "json_object"}
        elif provider == "mistral":
            # Check if specific model supports native JSON mode
            if model and "large" in model.lower():
                kwargs["response_format"] = {"type": "json_object"}
            else:
                kwargs["_json_mode_instruction"] = (
                    "You must respond with valid JSON only. "
                    "Do not include any text outside the JSON structure."
                )
        
        return kwargs
    
    @staticmethod
    def enable_streaming(provider: str, model: Optional[str], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Enable streaming with feature validation"""
        kwargs = kwargs.copy()
        
        if not ProviderAdapter.supports_feature(provider, Feature.STREAMING, model):
            logger.warning(f"{provider}/{model} doesn't support streaming")
            kwargs["stream"] = False
            return kwargs
        
        kwargs["stream"] = True
        return kwargs
    
    @staticmethod
    def prepare_tools(provider: str, model: Optional[str], tools: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
        """Prepare tools for provider with feature validation"""
        if not tools:
            return None
        
        if not ProviderAdapter.supports_feature(provider, Feature.TOOLS, model):
            logger.warning(f"{provider}/{model} doesn't support function calling")
            return None
        
        # Provider-specific tool preparation
        if provider == "openai":
            return tools  # OpenAI format is the standard
        elif provider == "anthropic":
            # Convert to Anthropic format if needed
            return tools
        elif provider == "gemini":
            # Convert to Gemini format if needed
            return tools
        elif provider == "groq":
            return tools  # Uses OpenAI-compatible format
        elif provider == "mistral":
            return tools  # Uses OpenAI-compatible format
        
        return tools
    
    @staticmethod
    def set_temperature(provider: str, temperature: float, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Set temperature across providers"""
        kwargs = kwargs.copy()
        
        # Most providers use 'temperature' directly
        if provider in ["openai", "anthropic", "groq", "mistral"]:
            kwargs["temperature"] = temperature
        elif provider == "gemini":
            if "generation_config" not in kwargs:
                kwargs["generation_config"] = {}
            kwargs["generation_config"]["temperature"] = temperature
        elif provider == "ollama":
            if "options" not in kwargs:
                kwargs["options"] = {}
            kwargs["options"]["temperature"] = temperature
        
        return kwargs
    
    @staticmethod
    def set_max_tokens(provider: str, model: Optional[str], max_tokens: int, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Set max tokens across providers with model-specific limits"""
        kwargs = kwargs.copy()
        
        # Check model-specific limits from unified config
        try:
            config = get_config()
            provider_config = config.get_provider(provider)
            model_caps = provider_config.get_model_capabilities(model)
            
            if model_caps.max_output_tokens and max_tokens > model_caps.max_output_tokens:
                logger.warning(f"Requested {max_tokens} tokens exceeds {provider}/{model} limit of {model_caps.max_output_tokens}")
                max_tokens = model_caps.max_output_tokens
        except Exception as e:
            logger.debug(f"Could not check token limits: {e}")
        
        # Apply provider-specific parameter names
        if provider in ["openai", "anthropic", "groq", "mistral"]:
            kwargs["max_tokens"] = max_tokens
        elif provider == "gemini":
            if "generation_config" not in kwargs:
                kwargs["generation_config"] = {}
            kwargs["generation_config"]["max_output_tokens"] = max_tokens
        elif provider == "ollama":
            if "options" not in kwargs:
                kwargs["options"] = {}
            kwargs["options"]["num_predict"] = max_tokens
        
        return kwargs
    
    @staticmethod
    def add_system_message(
        provider: str, 
        model: Optional[str],
        messages: List[Dict[str, Any]], 
        system_content: str
    ) -> List[Dict[str, Any]]:
        """Add system message in provider-appropriate way with feature validation"""
        
        # Check if provider/model supports system messages
        if not ProviderAdapter.supports_feature(provider, Feature.SYSTEM_MESSAGES, model):
            logger.warning(f"{provider}/{model} doesn't support system messages, prepending as user message")
            # Fallback: prepend as user message
            return [{"role": "user", "content": f"System: {system_content}\n\nUser: {messages[0].get('content', '')}"}] + messages[1:]
        
        # For providers that support system messages directly
        if provider in ["openai", "groq", "gemini", "ollama", "mistral"]:
            return [{"role": "system", "content": system_content}] + messages
        
        # Anthropic handles system messages specially in the API call
        elif provider == "anthropic":
            return messages  # System message handled in client
        
        return messages
    
    @staticmethod
    def check_vision_support(provider: str, model: Optional[str], messages: List[Dict[str, Any]]) -> bool:
        """Check if messages contain vision content and validate support"""
        # Check for vision content
        has_vision = False
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") in ["image", "image_url"]:
                        has_vision = True
                        break
        
        if has_vision and not ProviderAdapter.supports_feature(provider, Feature.VISION, model):
            logger.error(f"{provider}/{model} doesn't support vision/image inputs")
            return False
        
        return True


class UnifiedLLMInterface:
    """High-level interface that abstracts provider differences using unified config"""
    
    def __init__(self, provider: str, model: Optional[str] = None, **config_kwargs):
        from chuk_llm.llm.client import get_client
        
        self.provider = provider
        
        # Get provider config to determine model
        try:
            config = get_config()
            provider_config = config.get_provider(provider)
            self.model = model or provider_config.default_model
        except Exception as e:
            logger.error(f"Could not get provider config: {e}")
            self.model = model or "default"
        
        # Validate basic text capability
        if not ProviderAdapter.validate_text_capability(self.provider, self.model):
            raise ValueError(f"Provider {self.provider} with model {self.model} doesn't support basic text completion")
        
        self.client = get_client(
            provider=provider,
            model=self.model,
            **config_kwargs
        )
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get provider/model capabilities"""
        try:
            config = get_config()
            provider_config = config.get_provider(self.provider)
            model_caps = provider_config.get_model_capabilities(self.model)
            
            return {
                "provider": self.provider,
                "model": self.model,
                "features": [f.value for f in model_caps.features],
                "max_context_length": model_caps.max_context_length,
                "max_output_tokens": model_caps.max_output_tokens,
                "supports": {
                    "text": Feature.TEXT in model_caps.features,
                    "streaming": Feature.STREAMING in model_caps.features,
                    "tools": Feature.TOOLS in model_caps.features,
                    "vision": Feature.VISION in model_caps.features,
                    "json_mode": Feature.JSON_MODE in model_caps.features,
                    "system_messages": Feature.SYSTEM_MESSAGES in model_caps.features,
                }
            }
        except Exception as e:
            return {"error": str(e)}
    
    async def chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
        stream: bool = False,
        system_message: Optional[str] = None,
        **kwargs
    ):
        """Unified chat interface across all providers with automatic feature validation"""
        
        # Validate basic text capability first
        if not ProviderAdapter.validate_text_capability(self.provider, self.model):
            raise ValueError(f"Provider {self.provider}/{self.model} doesn't support basic text completion")
        
        # Validate vision content
        if not ProviderAdapter.check_vision_support(self.provider, self.model, messages):
            raise ValueError(f"{self.provider}/{self.model} doesn't support vision content in messages")
        
        # Process messages
        processed_messages = messages.copy()
        
        # Add system message if provided
        if system_message:
            processed_messages = ProviderAdapter.add_system_message(
                self.provider, self.model, processed_messages, system_message
            )
        
        # Apply provider-specific settings with feature validation
        if temperature is not None:
            kwargs = ProviderAdapter.set_temperature(self.provider, temperature, kwargs)
        
        if max_tokens is not None:
            kwargs = ProviderAdapter.set_max_tokens(self.provider, self.model, max_tokens, kwargs)
        
        if json_mode:
            kwargs = ProviderAdapter.enable_json_mode(self.provider, self.model, kwargs)
        
        if stream:
            kwargs = ProviderAdapter.enable_streaming(self.provider, self.model, kwargs)
        
        # Prepare tools
        processed_tools = ProviderAdapter.prepare_tools(self.provider, self.model, tools or [])
        
        # Make the request
        return await self.client.create_completion(
            processed_messages,
            tools=processed_tools,
            **kwargs
        )
    
    async def simple_chat(self, message: str, **kwargs) -> str:
        """Simple text-in, text-out interface"""
        messages = [{"role": "user", "content": message}]
        response = await self.chat(messages, **kwargs)
        
        if hasattr(response, '__aiter__'):
            # Handle streaming
            full_response = ""
            async for chunk in response:
                if chunk.get("response"):
                    full_response += chunk["response"]
            return full_response
        else:
            return response.get("response", "")
    
    async def chat_with_tools(
        self,
        message: str,
        tools: List[Dict[str, Any]],
        **kwargs
    ) -> Dict[str, Any]:
        """Chat with function calling"""
        messages = [{"role": "user", "content": message}]
        return await self.chat(messages, tools=tools, **kwargs)


# Enhanced convenience functions
async def quick_chat(provider: str, model: Optional[str] = None, message: str = "", **kwargs) -> str:
    """Quick one-shot chat with automatic model selection"""
    interface = UnifiedLLMInterface(provider, model)
    return await interface.simple_chat(message, **kwargs)


async def multi_provider_chat(
    message: str,
    providers: List[str],
    model_map: Optional[Dict[str, str]] = None,
    **chat_kwargs
) -> Dict[str, Any]:
    """Get responses from multiple providers with capability info"""
    model_map = model_map or {}
    results = {}
    
    async def get_provider_response(provider: str) -> Dict[str, Any]:
        try:
            model = model_map.get(provider)
            interface = UnifiedLLMInterface(provider, model)
            
            # Get capabilities
            capabilities = interface.get_capabilities()
            
            # Get response
            response = await interface.simple_chat(message, **chat_kwargs)
            
            return {
                "response": response,
                "model": interface.model,
                "capabilities": capabilities.get("supports", {}),
                "success": True
            }
        except Exception as e:
            return {
                "error": str(e),
                "success": False
            }
    
    # Create tasks for all providers
    tasks = {provider: get_provider_response(provider) for provider in providers}
    
    # Execute all tasks
    responses = await asyncio.gather(*tasks.values(), return_exceptions=True)
    
    # Combine results
    for provider, response in zip(providers, responses):
        if isinstance(response, Exception):
            results[provider] = {"error": str(response), "success": False}
        else:
            results[provider] = response
    
    return results


async def find_best_provider_for_task(
    message: str,
    required_features: Optional[List[str]] = None,
    exclude_providers: Optional[List[str]] = None
) -> Optional[Dict[str, Any]]:
    """Find and use the best provider for a specific task"""
    from chuk_llm.configuration import CapabilityChecker
    
    required_features = required_features or []
    exclude_providers = exclude_providers or []
    
    # Convert string features to Feature enum
    feature_set = set()
    for feat in required_features:
        try:
            feature_set.add(Feature.from_string(feat))
        except ValueError:
            logger.warning(f"Unknown feature: {feat}")
            continue
    
    # Always require TEXT feature
    feature_set.add(Feature.TEXT)
    
    # Find best provider
    best_provider = CapabilityChecker.get_best_provider_for_features(
        feature_set,
        exclude=set(exclude_providers)
    )
    
    if not best_provider:
        return None
    
    try:
        interface = UnifiedLLMInterface(best_provider)
        response = await interface.simple_chat(message)
        
        return {
            "provider": best_provider,
            "model": interface.model,
            "response": response,
            "capabilities": interface.get_capabilities()
        }
    except Exception as e:
        logger.error(f"Failed to use best provider {best_provider}: {e}")
        return None


# Text capability validation helper
def validate_text_support(provider: str, model: Optional[str] = None) -> bool:
    """Validate that a provider/model supports basic text completion"""
    return ProviderAdapter.validate_text_capability(provider, model)