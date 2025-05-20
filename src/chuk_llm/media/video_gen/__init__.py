# chuk_llm/video_generator/__init__.py
"""
Video generation capabilities for chuk_llm.

This package provides a standardized interface for generating videos
through various AI providers.
"""

from chuk_llm.media.video_gen.base import BaseVideoGenerator
from chuk_llm.media.video_gen.providers.gemini import GeminiVideoGenerator

# Default implementation - can be changed to other providers in the future
VideoGenerator = GeminiVideoGenerator