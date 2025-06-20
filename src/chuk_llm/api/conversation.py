# chuk_llm/api/conversation.py
"""
Conversation management with memory, context, and automatic session tracking
===========================================================================

Conversations automatically track sessions when available.
"""

from typing import List, Dict, Any, Optional, AsyncIterator
from contextlib import asynccontextmanager
from chuk_llm.llm.system_prompt_generator import SystemPromptGenerator
from chuk_llm.configuration.unified_config import get_config
from chuk_llm.llm.client import get_client

# Import session tracking components
try:
    from chuk_ai_session_manager import SessionManager
    _SESSION_AVAILABLE = True
except ImportError:
    _SESSION_AVAILABLE = False
    SessionManager = None

import os
import logging

logger = logging.getLogger(__name__)

# Check if sessions should be disabled
_SESSIONS_ENABLED = _SESSION_AVAILABLE and os.getenv("CHUK_LLM_DISABLE_SESSIONS", "").lower() not in ("true", "1", "yes")


class ConversationContext:
    """Manages conversation state and history with automatic session tracking."""
    
    def __init__(
        self, 
        provider: str, 
        model: str = None, 
        system_prompt: str = None,
        session_id: Optional[str] = None,
        infinite_context: bool = True,
        token_threshold: int = 4000,
        **kwargs
    ):
        self.provider = provider
        self.model = model
        self.kwargs = kwargs
        self.messages = []
        
        # Initialize session tracking automatically if available
        if _SESSIONS_ENABLED:
            try:
                self.session_manager = SessionManager(
                    session_id=session_id,
                    system_prompt=system_prompt,
                    infinite_context=infinite_context,
                    token_threshold=token_threshold
                )
            except Exception as e:
                logger.debug(f"Could not initialize session manager: {e}")
                self.session_manager = None
        else:
            self.session_manager = None
        
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
    
    @property
    def session_id(self) -> Optional[str]:
        """Get current session ID if tracking is enabled."""
        return self.session_manager.session_id if self.session_manager else None
    
    @property
    def has_session(self) -> bool:
        """Check if session tracking is active."""
        return self.session_manager is not None
    
    async def say(self, prompt: str, **kwargs) -> str:
        """Send a message in the conversation and get a response."""
        # Track user message automatically
        if self.session_manager:
            try:
                await self.session_manager.user_says(prompt)
            except Exception as e:
                logger.debug(f"Session tracking error: {e}")
        
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
            
            # Track AI response automatically
            if self.session_manager:
                try:
                    await self.session_manager.ai_responds(
                        response_text,
                        model=self.model or "unknown",
                        provider=self.provider
                    )
                except Exception as e:
                    logger.debug(f"Session tracking error: {e}")
            
            return response_text
            
        except Exception as e:
            error_msg = f"Conversation error: {str(e)}"
            self.messages.append({"role": "assistant", "content": error_msg})
            
            # Track error automatically
            if self.session_manager:
                try:
                    await self.session_manager.ai_responds(
                        error_msg,
                        model=self.model or "unknown",
                        provider=self.provider
                    )
                except Exception as e:
                    logger.debug(f"Session tracking error: {e}")
            
            return error_msg
    
    async def stream_say(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        """Send a message and stream the response."""
        # Track user message automatically
        if self.session_manager:
            try:
                await self.session_manager.user_says(prompt)
            except Exception as e:
                logger.debug(f"Session tracking error: {e}")
        
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
            
            # Track complete response automatically
            if self.session_manager:
                try:
                    await self.session_manager.ai_responds(
                        full_response,
                        model=self.model or "unknown",
                        provider=self.provider
                    )
                except Exception as e:
                    logger.debug(f"Session tracking error: {e}")
            
        except Exception as e:
            error_msg = f"[Streaming error: {str(e)}]"
            yield error_msg
            full_response += error_msg
            self.messages.append({"role": "assistant", "content": full_response})
            
            # Track error automatically
            if self.session_manager:
                try:
                    await self.session_manager.ai_responds(
                        full_response,
                        model=self.model or "unknown",
                        provider=self.provider
                    )
                except Exception as e:
                    logger.debug(f"Session tracking error: {e}")
    
    def clear(self):
        """Clear conversation history but keep system message."""
        system_msgs = [msg for msg in self.messages if msg["role"] == "system"]
        self.messages = system_msgs
        
        # Note: We don't clear the session manager, allowing tracking to continue
    
    def get_history(self) -> List[Dict[str, str]]:
        """Get conversation history."""
        return self.messages.copy()
    
    async def get_session_history(self) -> List[Dict[str, Any]]:
        """Get session history if available."""
        if self.session_manager:
            try:
                return await self.session_manager.get_conversation()
            except Exception as e:
                logger.debug(f"Could not get session history: {e}")
        return self.get_history()
    
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
        
        stats = {
            "total_messages": len(self.messages),
            "user_messages": len(user_messages),
            "assistant_messages": len(assistant_messages),
            "has_system_prompt": any(msg["role"] == "system" for msg in self.messages),
            "estimated_tokens": sum(len(msg["content"].split()) * 1.3 for msg in self.messages),
            "has_session": self.has_session,
        }
        
        # Add session ID if available
        if self.session_manager:
            stats["session_id"] = self.session_manager.session_id
        
        return stats
    
    async def get_session_stats(self) -> Dict[str, Any]:
        """Get comprehensive stats including session tracking."""
        basic_stats = self.get_stats()
        
        if self.session_manager:
            try:
                session_stats = await self.session_manager.get_stats()
                # Merge stats
                basic_stats.update({
                    "total_tokens": session_stats.get("total_tokens", 0),
                    "estimated_cost": session_stats.get("estimated_cost", 0),
                    "session_segments": session_stats.get("session_segments", 1),
                    "session_duration": session_stats.get("session_duration", "unknown"),
                })
            except Exception as e:
                logger.debug(f"Could not get session stats: {e}")
        
        return basic_stats
    
    def set_system_prompt(self, system_prompt: str):
        """Update the system prompt for this conversation."""
        self.messages = [{"role": "system", "content": system_prompt}]
        
        # Update session manager system prompt if available
        if self.session_manager:
            import asyncio
            asyncio.create_task(self.session_manager.update_system_prompt(system_prompt))


@asynccontextmanager
async def conversation(
    provider: str = None,
    model: str = None,
    system_prompt: str = None,
    session_id: Optional[str] = None,
    infinite_context: bool = True,
    token_threshold: int = 4000,
    **kwargs
):
    """
    Create a conversation context manager with automatic session tracking.
    
    Session tracking is automatic when chuk-ai-session-manager is installed.
    Set CHUK_LLM_DISABLE_SESSIONS=true to disable session tracking.
    
    Args:
        provider: LLM provider to use
        model: Model to use
        system_prompt: System prompt for the conversation
        session_id: Optional existing session ID to continue
        infinite_context: Enable infinite context support (default: True)
        token_threshold: Token limit for infinite context segmentation
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
    
    ctx = None
    try:
        # Create and yield conversation context
        ctx = ConversationContext(
            provider=provider,
            model=model,
            system_prompt=system_prompt,
            session_id=session_id,
            infinite_context=infinite_context,
            token_threshold=token_threshold,
            **kwargs
        )
        yield ctx
    finally:
        # Log final stats if session was available
        if ctx and ctx.session_manager:
            try:
                stats = await ctx.get_session_stats()
                logger.debug(
                    f"Conversation ended - Session: {stats.get('session_id', 'N/A')}, "
                    f"Tokens: {stats.get('total_tokens', 0)}, "
                    f"Cost: ${stats.get('estimated_cost', 0):.6f}"
                )
            except Exception as e:
                logger.debug(f"Could not log final stats: {e}")