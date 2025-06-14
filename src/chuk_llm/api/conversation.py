# chuk_llm/api/conversation.py
"""Conversation management with memory and context."""

from typing import List, Dict, Any, Optional, AsyncIterator
from contextlib import asynccontextmanager
from chuk_llm.llm.system_prompt_generator import SystemPromptGenerator
from .config import get_current_config, get_client_for_config
from .provider_utils import get_provider_default_model

class ConversationContext:
    """Manages conversation state and history.
    
    This class maintains the conversation history and provides methods
    to interact with the LLM while preserving context across messages.
    """
    
    def __init__(self, config: Dict[str, Any], client):
        self.config = config
        self.client = client
        self.messages = []
        
        # Add initial system message
        if config.get("system_prompt"):
            self.messages.append({
                "role": "system", 
                "content": config["system_prompt"]
            })
        else:
            # Use your existing system prompt generator
            system_generator = SystemPromptGenerator()
            system_content = system_generator.generate_prompt({})
            self.messages.append({
                "role": "system",
                "content": system_content
            })
    
    async def say(self, prompt: str, **kwargs) -> str:
        """Send a message in the conversation and get a response.
        
        Args:
            prompt: The message to send
            **kwargs: Additional arguments for the completion (temperature, etc.)
            
        Returns:
            The LLM's response
            
        Examples:
            response = await chat.say("Hello!")
            response = await chat.say("What's the weather?", temperature=0.8)
        """
        # Add user message to history
        self.messages.append({"role": "user", "content": prompt})
        
        # Prepare completion arguments
        completion_args = {"messages": self.messages.copy()}
        completion_args.update(kwargs)
        
        try:
            # Get response using enhanced client
            response = await self.client.create_completion(**completion_args)
            
            if isinstance(response, dict):
                if response.get("error"):
                    error_msg = f"Error: {response.get('error_message', 'Unknown error')}"
                    # Add error to history so conversation context is preserved
                    self.messages.append({"role": "assistant", "content": error_msg})
                    return error_msg
                
                response_text = response.get("response", "")
            else:
                response_text = str(response)
            
            # Add assistant response to history
            self.messages.append({"role": "assistant", "content": response_text})
            return response_text
            
        except Exception as e:
            error_msg = f"Conversation error: {str(e)}"
            self.messages.append({"role": "assistant", "content": error_msg})
            return error_msg
    
    async def stream_say(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        """Send a message and stream the response.
        
        Args:
            prompt: The message to send
            **kwargs: Additional arguments for the completion
            
        Yields:
            str: Chunks of the response as they arrive
        """
        # Add user message to history
        self.messages.append({"role": "user", "content": prompt})
        
        # Prepare streaming arguments
        completion_args = {
            "messages": self.messages.copy(),
            "stream": True,
        }
        completion_args.update(kwargs)
        
        full_response = ""
        
        try:
            response_stream = await self.client.create_completion(**completion_args)
            
            async for chunk in response_stream:
                if isinstance(chunk, dict):
                    if chunk.get("error"):
                        error_msg = f"[Error: {chunk.get('error_message', 'Unknown error')}]"
                        yield error_msg
                        full_response += error_msg
                        break
                    
                    content = chunk.get("response", "")
                    if content:
                        full_response += content
                        yield content
            
            # Add complete response to history
            self.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            error_msg = f"[Streaming error: {str(e)}]"
            yield error_msg
            full_response += error_msg
            self.messages.append({"role": "assistant", "content": full_response})
    
    def clear(self):
        """Clear conversation history but keep system message.
        
        This preserves the initial system prompt while removing all
        user and assistant messages.
        """
        system_msgs = [msg for msg in self.messages if msg["role"] == "system"]
        self.messages = system_msgs
    
    def get_history(self) -> List[Dict[str, str]]:
        """Get conversation history.
        
        Returns:
            List of message dictionaries with 'role' and 'content' keys
        """
        return self.messages.copy()
    
    def pop_last(self):
        """Remove the last user-assistant exchange.
        
        This removes the most recent user message and assistant response,
        effectively "undoing" the last interaction.
        """
        # Remove from the end, keeping system messages
        removed_count = 0
        while self.messages and self.messages[-1]["role"] != "system" and removed_count < 2:
            self.messages.pop()
            removed_count += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get conversation statistics.
        
        Returns:
            Dictionary with conversation stats like message count, token estimates, etc.
        """
        user_messages = [msg for msg in self.messages if msg["role"] == "user"]
        assistant_messages = [msg for msg in self.messages if msg["role"] == "assistant"]
        
        return {
            "total_messages": len(self.messages),
            "user_messages": len(user_messages),
            "assistant_messages": len(assistant_messages),
            "has_system_prompt": any(msg["role"] == "system" for msg in self.messages),
            "estimated_tokens": sum(len(msg["content"].split()) * 1.3 for msg in self.messages),  # Rough estimate
        }
    
    def set_system_prompt(self, system_prompt: str):
        """Update the system prompt for this conversation.
        
        Args:
            system_prompt: New system prompt to use
            
        Note:
            This replaces any existing system message and clears the conversation history
            since changing the system prompt mid-conversation can cause inconsistencies.
        """
        # Clear all messages and add new system prompt
        self.messages = [{"role": "system", "content": system_prompt}]

@asynccontextmanager
async def conversation(
    provider: str = None,
    model: str = None,
    system_prompt: str = None,
    **kwargs
):
    """Create a conversation context manager.
    
    This provides a clean way to manage conversations with memory,
    automatically handling setup and cleanup.
    
    Args:
        provider: LLM provider to use
        model: Model to use
        system_prompt: System prompt for the conversation
        **kwargs: Additional configuration options
        
    Yields:
        ConversationContext: Context manager for the conversation
        
    Examples:
        # Basic conversation
        async with conversation() as chat:
            response1 = await chat.say("Hi, I'm working on a Python project")
            response2 = await chat.say("Can you help me optimize it?")
            
        # With specific provider and model
        async with conversation(provider="anthropic", model="claude-3-opus") as chat:
            response = await chat.say("Tell me about AI")
            
        # With custom system prompt
        async with conversation(system_prompt="You are a helpful coding assistant") as chat:
            response = await chat.say("How do I optimize this code?")
    """
    # Setup configuration
    final_config = get_current_config().copy()
    
    # Apply overrides
    if provider:
        final_config["provider"] = provider
    if model:
        final_config["model"] = model
    if system_prompt:
        final_config["system_prompt"] = system_prompt
    
    # Update with any additional kwargs
    final_config.update(kwargs)
    
    # Apply smart model resolution using centralized utility
    if final_config.get("model") is None or (provider is not None and model is None):
        provider_default = get_provider_default_model(final_config["provider"])
        if provider_default:
            final_config["model"] = provider_default
        else:
            # Fallback to trying the existing provider config system
            try:
                from chuk_llm.llm.configuration.provider_config import ProviderConfig
                config_mgr = ProviderConfig()
                provider_config = config_mgr.get_provider_config(final_config["provider"])
                fallback_model = provider_config.get("default_model")
                if fallback_model:
                    final_config["model"] = fallback_model
            except Exception:
                # Keep the existing model from global config as last resort
                pass
    
    # Get client
    client = get_client_for_config(final_config)
    
    try:
        # Yield the conversation context
        yield ConversationContext(final_config, client)
    finally:
        # Cleanup is handled by your connection pool and resource manager
        # No explicit cleanup needed here
        pass