# chuk_llm/image_generator/__init__.py
"""
Image generation capabilities for chuk_llm.

This package provides a standardized interface for generating images
through various AI providers.
"""

from chuk_llm.media.image_gen.base import BaseImageGenerator
from chuk_llm.media.image_gen.providers.gemini import GeminiImageGenerator

# Default implementation - can be changed to other providers in the future
ImageGenerator = GeminiImageGenerator