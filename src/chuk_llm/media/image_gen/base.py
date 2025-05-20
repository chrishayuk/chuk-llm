# chuk_llm/media/image_gen/base.py
"""
Base classes for image generation capabilities.

This module defines abstract base classes for image generation,
providing a clean separation from text-based LLM interfaces.
"""

from __future__ import annotations

import abc
import asyncio
import logging
import os
from typing import Any, Dict, List, Optional, Union

log = logging.getLogger(__name__)

class BaseImageGenerator(abc.ABC):
    """
    Abstract base class for image generation.
    
    Providers should implement the abstract methods to support their specific API.
    """
    
    @staticmethod
    async def _call_blocking(fn, *args, **kwargs):
        """Run a blocking function in a background thread."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))
    
    async def generate_images(
        self,
        prompt: str,
        output_dir: str = ".",
        number_of_images: int = 1,
        aspect_ratio: str = "1:1",
        person_generation: str = "allow",
        negative_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate images based on a text prompt.
        
        Args:
            prompt: Text description of the image to generate
            output_dir: Directory to save generated images
            number_of_images: Number of images to generate
            aspect_ratio: Aspect ratio of generated images
            person_generation: Policy for generating people
            negative_prompt: Description of what to avoid in generation
            **kwargs: Additional provider-specific arguments
            
        Returns:
            Dictionary with status and file paths
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Call the provider-specific implementation
            response = await self._call_blocking(
                self._generate_images_sync,
                prompt=prompt,
                number_of_images=number_of_images,
                aspect_ratio=aspect_ratio,
                person_generation=person_generation,
                negative_prompt=negative_prompt,
                **kwargs
            )
            
            # Save the generated images
            image_files = await self._save_images(response, output_dir)
            
            return {
                "status": "complete",
                "image_files": image_files,
                "message": f"Generated {len(image_files)} images"
            }
            
        except Exception as e:
            log.error(f"Error in image generation: {e}")
            raise
    
    @abc.abstractmethod
    def _generate_images_sync(
        self,
        prompt: str,
        number_of_images: int,
        aspect_ratio: str,
        person_generation: str,
        negative_prompt: Optional[str],
        **kwargs
    ) -> Any:
        """
        Provider-specific implementation for image generation.
        
        This method must be implemented by provider classes.
        """
        pass
    
    @abc.abstractmethod
    async def _save_images(self, response: Any, output_dir: str) -> List[str]:
        """
        Save generated images to disk.
        
        This method must be implemented by provider classes.
        """
        pass