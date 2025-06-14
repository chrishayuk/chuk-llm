# src/chuk_llm/api/__init__.py
"""Simple API for ChukLLM - fully dynamic."""

# Core functions
from .core import ask, stream
from .config import configure, get_config, reset_config

# Import the providers module to trigger dynamic generation
from . import providers

# Import all dynamically generated functions using star import
# This works because providers.__all__ is properly set
from .providers import *

# Build the complete __all__ list
__all__ = [
    "ask", "stream", 
    "configure", "get_config", "reset_config",
] + providers.__all__