# chuk_llm/api/conversation.py
"""Conversation management with memory and context - updated for clean config."""

from typing import List, Dict, Any, Optional, AsyncIterator
from contextlib import asynccontextmanager
from chuk_llm.llm.system_prompt_generator import SystemPromptGenerator
from chuk_llm.configuration.config import get_config
from chuk_llm.llm.client import get_client


class ConversationContext:
    """Manages conversation state and history."""
    
    def __init__(self, provider: str, model: str = None, system_prompt: str = None, **kwargs):
        self.provider = provider
        self.model = model
        self.kwargs = kwargs
        self.messages = []
        
        # Get client
        self.client = get_client(
            provider=provider,
            model=model,
            **kwargs
        )
        
        # Add initial system message
        if system_prompt:
            self.messages.append({
                "role": "system", 
                "content": system_prompt
            })
        else:
            # Use system prompt generator
            system_generator = SystemPromptGenerator()
            system_content = system_generator.generate_prompt({})
            self.messages.append({
                "role": "system",
                "content": system_content
            })
    
    async def say(self, prompt: str, **kwargs) -> str:
        """Send a message in the conversation and get a response."""
        # Add user message to history
        self.messages.append({"role": "user", "content": prompt})
        
        # Prepare completion arguments
        completion_args = {"messages": self.messages.copy()}
        completion_args.update(kwargs)
        
        try:
            # Get response using client
            response = await self.client.create_completion(**completion_args)
            
            if isinstance(response, dict):
                if response.get("error"):
                    error_msg = f"Error: {response.get('error_message', 'Unknown error')}"
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
        """Send a message and stream the response."""
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
        """Clear conversation history but keep system message."""
        system_msgs = [msg for msg in self.messages if msg["role"] == "system"]
        self.messages = system_msgs
    
    def get_history(self) -> List[Dict[str, str]]:
        """Get conversation history."""
        return self.messages.copy()
    
    def pop_last(self):
        """Remove the last user-assistant exchange."""
        removed_count = 0
        while self.messages and self.messages[-1]["role"] != "system" and removed_count < 2:
            self.messages.pop()
            removed_count += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get conversation statistics."""
        user_messages = [msg for msg in self.messages if msg["role"] == "user"]
        assistant_messages = [msg for msg in self.messages if msg["role"] == "assistant"]
        
        return {
            "total_messages": len(self.messages),
            "user_messages": len(user_messages),
            "assistant_messages": len(assistant_messages),
            "has_system_prompt": any(msg["role"] == "system" for msg in self.messages),
            "estimated_tokens": sum(len(msg["content"].split()) * 1.3 for msg in self.messages),
        }
    
    def set_system_prompt(self, system_prompt: str):
        """Update the system prompt for this conversation."""
        self.messages = [{"role": "system", "content": system_prompt}]


@asynccontextmanager
async def conversation(
    provider: str = None,
    model: str = None,
    system_prompt: str = None,
    **kwargs
):
    """
    Create a conversation context manager.
    
    Args:
        provider: LLM provider to use
        model: Model to use
        system_prompt: System prompt for the conversation
        **kwargs: Additional configuration options
        
    Yields:
        ConversationContext: Context manager for the conversation
    """
    # Get defaults from config if not specified
    if not provider:
        config_manager = get_config()
        global_settings = config_manager.get_global_settings()
        provider = global_settings.get("active_provider", "openai")
    
    if not model:
        config_manager = get_config()
        try:
            provider_config = config_manager.get_provider(provider)
            model = provider_config.default_model
        except ValueError:
            model = "gpt-4o-mini"  # Fallback
    
    try:
        # Create and yield conversation context
        yield ConversationContext(
            provider=provider,
            model=model,
            system_prompt=system_prompt,
            **kwargs
        )
    finally:
        # No explicit cleanup needed - clients handle their own cleanup
        pass