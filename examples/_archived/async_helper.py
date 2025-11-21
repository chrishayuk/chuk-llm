#!/usr/bin/env python3
"""Helper module for clean async execution without event loop warnings"""

import asyncio
import sys
import os
import warnings
import logging
import atexit

# Suppress all async cleanup warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*Event loop is closed.*")
warnings.filterwarnings("ignore", message=".*coroutine.*was never awaited.*")
warnings.filterwarnings("ignore", message=".*Task exception was never retrieved.*")
logging.getLogger("asyncio").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)

# Monkey-patch asyncio to suppress task exception messages
_original_call_exception_handler = asyncio.base_events.BaseEventLoop.call_exception_handler

def _patched_call_exception_handler(self, context):
    # Ignore "Event loop is closed" errors
    exception = context.get('exception')
    if exception and "Event loop is closed" in str(exception):
        return
    message = context.get('message', '')
    if message and "Task exception was never retrieved" in message:
        return
    _original_call_exception_handler(self, context)

asyncio.base_events.BaseEventLoop.call_exception_handler = _patched_call_exception_handler

def run_async_clean(coro):
    """Run async code cleanly without event loop warnings"""
    # Save original stderr
    original_stderr = sys.stderr
    
    try:
        # Create new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Run the coroutine
        result = loop.run_until_complete(coro)
        
        # Give tasks time to cleanup
        loop.run_until_complete(asyncio.sleep(0.1))
        
        return result
        
    finally:
        # Suppress all stderr during cleanup
        sys.stderr = open(os.devnull, 'w')
        
        try:
            # Cancel all remaining tasks
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            
            # Wait for cancellation with suppressed output
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            
            # Final cleanup
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.run_until_complete(asyncio.sleep(0))
            
        except:
            pass
            
        finally:
            # Close the loop
            try:
                loop.close()
            except:
                pass
            
            # Restore stderr
            sys.stderr.close()
            sys.stderr = original_stderr