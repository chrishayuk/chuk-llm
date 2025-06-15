# chuk_llm/api/event_loop_manager.py
"""
Event loop manager for synchronous operations
=============================================

Manages a dedicated thread with its own event loop for running async operations
synchronously without event loop conflicts.
"""

import asyncio
import threading
import atexit
import warnings
import sys
from typing import Any, Coroutine, TypeVar
from concurrent.futures import ThreadPoolExecutor
import weakref

# Suppress event loop warnings
warnings.filterwarnings("ignore", message=".*Event loop is closed.*", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*Task exception was never retrieved.*", category=RuntimeWarning)

T = TypeVar('T')


class EventLoopThread:
    """A thread that runs its own event loop"""
    
    def __init__(self):
        self._loop: asyncio.AbstractEventLoop = None
        self._thread: threading.Thread = None
        self._ready = threading.Event()
        self._executor = ThreadPoolExecutor(max_workers=1)
        
    def start(self):
        """Start the event loop thread"""
        if self._thread is not None and self._thread.is_alive():
            return
            
        self._ready.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._ready.wait()  # Wait for loop to be ready
        
    def _run_loop(self):
        """Run the event loop in this thread"""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._ready.set()
        
        # Keep the loop running
        self._loop.run_forever()
        
    def run_coro(self, coro: Coroutine[Any, Any, T]) -> T:
        """Run a coroutine in this thread's event loop"""
        self.start()  # Ensure thread is running
        
        # Submit coroutine to the loop
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()
        
    def stop(self):
        """Stop the event loop and thread"""
        if self._loop and self._thread and self._thread.is_alive():
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=1.0)
            

# Global thread for running async operations
_loop_thread = None
_lock = threading.Lock()


def _get_loop_thread() -> EventLoopThread:
    """Get or create the global event loop thread"""
    global _loop_thread
    
    if _loop_thread is None:
        with _lock:
            if _loop_thread is None:
                _loop_thread = EventLoopThread()
                # Register cleanup
                weakref.finalize(_loop_thread, _cleanup_thread)
    
    return _loop_thread


def _cleanup_thread():
    """Clean up the event loop thread"""
    global _loop_thread
    if _loop_thread:
        _loop_thread.stop()
        _loop_thread = None


# Register cleanup on exit
atexit.register(_cleanup_thread)


def run_sync(coro: Coroutine[Any, Any, T]) -> T:
    """
    Run an async coroutine synchronously.
    
    This uses a dedicated thread with a persistent event loop to avoid conflicts
    with httpx connection pooling and other async resources.
    
    Args:
        coro: The coroutine to run
        
    Returns:
        The result of the coroutine
        
    Raises:
        RuntimeError: If called from an async context
    """
    # Check if we're already in an async context
    try:
        loop = asyncio.get_running_loop()
        raise RuntimeError(
            "Cannot call sync functions from async context. "
            "Use the async version instead."
        )
    except RuntimeError as e:
        if "Cannot call sync functions" in str(e):
            raise e
        # No running loop, we can proceed
    
    # Get the loop thread and run the coroutine
    loop_thread = _get_loop_thread()
    return loop_thread.run_coro(coro)