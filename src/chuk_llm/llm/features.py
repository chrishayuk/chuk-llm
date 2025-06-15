# chuk_llm/llm/features.py - Updated with Mistral support
from typing import Dict, Any, List, Optional
class ProviderAdapter:
    """Adapts provider-specific features to common interface"""
    
    @staticmethod
    def enable_json_mode(provider: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Enable JSON mode across different providers"""
        kwargs = kwargs.copy()
        
        if provider == "openai":
            kwargs["response_format"] = {"type": "json_object"}
        elif provider == "anthropic":
            # Anthropic doesn't have native JSON mode, use system message
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
            # Mistral doesn't have native JSON mode, use system message
            kwargs["_json_mode_instruction"] = (
                "You must respond with valid JSON only. "
                "Do not include any text outside the JSON structure."
            )
        
        return kwargs
    
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
    def set_max_tokens(provider: str, max_tokens: int, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Set max tokens across providers"""
        kwargs = kwargs.copy()
        
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
        messages: List[Dict[str, Any]], 
        system_content: str
    ) -> List[Dict[str, Any]]:
        """Add system message in provider-appropriate way"""
        
        # For providers that support system messages directly
        if provider in ["openai", "groq", "gemini", "ollama", "mistral"]:
            return [{"role": "system", "content": system_content}] + messages
        
        # Anthropic handles system messages specially in the API call
        elif provider == "anthropic":
            return messages
        
        return messages

class UnifiedLLMInterface:
    """High-level interface that abstracts provider differences"""
    
    def __init__(self, provider: str, model: str, **config_kwargs):
        from chuk_llm.llm.client import get_enhanced_client
        
        self.provider = provider
        self.model = model
        self.client = get_enhanced_client(
            provider=provider,
            model=model,
            **config_kwargs
        )
    
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
        """Unified chat interface across all providers"""
        
        # Process messages
        processed_messages = messages.copy()
        
        # Add system message if provided
        if system_message:
            processed_messages = ProviderAdapter.add_system_message(
                self.provider, processed_messages, system_message
            )
        
        # Apply provider-specific settings
        if temperature is not None:
            kwargs = ProviderAdapter.set_temperature(self.provider, temperature, kwargs)
        
        if max_tokens is not None:
            kwargs = ProviderAdapter.set_max_tokens(self.provider, max_tokens, kwargs)
        
        if json_mode:
            kwargs = ProviderAdapter.enable_json_mode(self.provider, kwargs)
        
        # Make the request
        return await self.client.create_completion(
            processed_messages,
            tools=tools,
            stream=stream,
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

# Convenience functions
async def quick_chat(provider: str, model: str, message: str, **kwargs) -> str:
    """Quick one-shot chat"""
    interface = UnifiedLLMInterface(provider, model)
    return await interface.simple_chat(message, **kwargs)

async def multi_provider_chat(
    message: str,
    providers: List[str],
    model_map: Optional[Dict[str, str]] = None
) -> Dict[str, str]:
    """Get responses from multiple providers"""
    model_map = model_map or {}
    results = {}
    
    tasks = []
    for provider in providers:
        model = model_map.get(provider, "default")
        task = quick_chat(provider, model, message)
        tasks.append((provider, task))
    
    import asyncio
    responses = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
    
    for (provider, _), response in zip(tasks, responses):
        if isinstance(response, Exception):
            results[provider] = f"Error: {response}"
        else:
            results[provider] = response
    
    return results