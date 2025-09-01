"""
Synchronous conversation API wrapper
"""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from .conversation import ConversationContext
from .conversation import conversation as async_conversation
from .event_loop_manager import EventLoopThread


class ConversationContextSync:
    """Synchronous wrapper for ConversationContext"""

    def __init__(
        self, async_context: ConversationContext, loop_thread: EventLoopThread
    ):
        self._async_context = async_context
        self._loop_thread = loop_thread

    def ask(self, prompt: str, image: str | bytes | None = None, **kwargs: Any) -> str:
        """Synchronous version of ask"""
        return self._loop_thread.run_coro(
            self._async_context.ask(prompt, image=image, **kwargs)
        )

    def stream(
        self, prompt: str, image: str | bytes | None = None, **kwargs: Any
    ) -> Iterator[str]:
        """Synchronous version of stream"""

        async def _collect_stream():
            chunks = []
            async for chunk in self._async_context.stream(
                prompt, image=image, **kwargs
            ):
                chunks.append(chunk)
            return chunks

        chunks = self._loop_thread.run_coro(_collect_stream())
        yield from chunks

    @contextmanager
    def branch(self):
        """Synchronous version of branch"""

        # Create async branch in the event loop
        async def _create_branch():
            return self._async_context.branch()

        async_branch_cm = self._loop_thread.run_coro(_create_branch())

        # Enter the async context manager
        async def _enter_branch():
            return await async_branch_cm.__aenter__()

        async_branch = self._loop_thread.run_coro(_enter_branch())

        # Wrap in sync context
        sync_branch = ConversationContextSync(async_branch, self._loop_thread)

        try:
            yield sync_branch
        finally:
            # Exit the async context manager
            async def _exit_branch():
                await async_branch_cm.__aexit__(None, None, None)

            self._loop_thread.run_coro(_exit_branch())

    def save(self) -> str:
        """Synchronous version of save"""
        return self._loop_thread.run_coro(self._async_context.save())

    def load(self, conversation_id: str) -> "ConversationContextSync":
        """Synchronous version of load"""

        async def _load():
            return await ConversationContext.load(conversation_id)

        async_ctx = self._loop_thread.run_coro(_load())
        return ConversationContextSync(async_ctx, self._loop_thread)

    def summarize(self, max_length: int = 500) -> str:
        """Synchronous version of summarize"""
        return self._loop_thread.run_coro(self._async_context.summarize(max_length))

    def extract_key_points(self) -> list[str]:
        """Synchronous version of extract_key_points"""
        return self._loop_thread.run_coro(self._async_context.extract_key_points())

    def clear(self) -> None:
        """Clear conversation history but keep system message"""
        self._async_context.clear()

    def get_history(self) -> list[dict[str, Any]]:
        """Get conversation history"""
        return self._async_context.get_history()

    def pop_last(self) -> None:
        """Remove the last user-assistant exchange"""
        self._async_context.pop_last()

    def get_stats(self) -> dict[str, Any]:
        """Get conversation statistics"""
        return self._async_context.get_stats()

    def get_session_stats(self) -> dict[str, Any]:
        """Get comprehensive stats including session tracking"""
        return self._loop_thread.run_coro(self._async_context.get_session_stats())

    def set_system_prompt(self, system_prompt: str) -> None:
        """Update the system prompt"""
        self._async_context.set_system_prompt(system_prompt)

    @property
    def messages(self) -> list[dict[str, Any]]:
        """Access to messages for compatibility"""
        return self._async_context.messages

    @property
    def conversation_id(self) -> str:
        """Get unique conversation ID"""
        return self._async_context.conversation_id

    @property
    def session_id(self) -> str | None:
        """Get current session ID if tracking is enabled"""
        return self._async_context.session_id

    @property
    def has_session(self) -> bool:
        """Check if session tracking is active"""
        return self._async_context.has_session


@contextmanager
def conversation_sync(
    provider: str | None = None,
    model: str | None = None,
    system_prompt: str | None = None,
    session_id: str | None = None,
    infinite_context: bool = True,
    token_threshold: int = 4000,
    resume_from: str | None = None,
    **kwargs: Any,
) -> Iterator[ConversationContextSync]:
    """
    Synchronous conversation context manager.

    This provides the same functionality as the async conversation API
    but in a synchronous interface for simpler scripts.

    Example:
        with conversation_sync(provider="openai") as chat:
            response = chat.ask("Hello!")
            print(response)

            # Stream responses
            for chunk in chat.stream("Tell me a story"):
                print(chunk, end="", flush=True)

            # Create branches
            with chat.branch() as branch:
                branch.ask("What if...")
    """
    loop_thread = EventLoopThread()

    # Create async context in the event loop
    async def _create_context():
        return async_conversation(
            provider=provider,
            model=model,
            system_prompt=system_prompt,
            session_id=session_id,
            infinite_context=infinite_context,
            token_threshold=token_threshold,
            resume_from=resume_from,
            **kwargs,
        )

    async_context_cm = loop_thread.run_coro(_create_context())

    # Enter the async context
    async def _enter_context():
        return await async_context_cm.__aenter__()

    async_context = loop_thread.run_coro(_enter_context())

    # Wrap in sync context
    sync_context = ConversationContextSync(async_context, loop_thread)

    try:
        yield sync_context
    finally:
        # Exit the async context
        async def _exit_context():
            await async_context_cm.__aexit__(None, None, None)

        loop_thread.run_coro(_exit_context())
        loop_thread.stop()
