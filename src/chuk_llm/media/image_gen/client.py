# chuk_llm/image_generator/client.py
"""
image generation client adapter.

This client implements a standardized interface for image generation
following the same patterns as other chuk_llm providers.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv
from google import genai
from google.genai import types as gtypes
from PIL import Image

# base
from chuk_llm.llm.core.base import BaseLLMClient

# mixin
from .mixin import ImageGeneratorMixin

log = logging.getLogger(__name__)

# Honour LOGLEVEL env-var for quick local tweaks
if "LOGLEVEL" in os.environ:
    log.setLevel(os.environ["LOGLEVEL"].upper())

class ImageGeneratorClient(ImageGeneratorMixin, BaseLLMClient):
    """Client for generating images using Google's Imagen model via the Gemini API."""

    def __init__(
        self,
        model: str = "imagen-3.0-generate-002",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,  # Not used for Gemini but kept for consistency
    ) -> None:
        """
        Initialize the image generator client.
        
        Args:
            model: Name of the image generation model to use
            api_key: Optional API key (will use GEMINI_API_KEY environment variable if not provided)
            api_base: Not used for Gemini, kept for interface consistency
        """
        load_dotenv()
        self.model = model
        api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY / GEMINI_API_KEY environment variable not set")
        
        self.client = genai.Client(api_key=api_key)
        log.info("ImageGeneratorClient initialized with model '%s'", model)

    async def create_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        *,
        stream: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Implement the BaseLLMClient interface but just return information about how to use.
        
        This method exists to satisfy the BaseLLMClient interface but isn't the primary way
        to use the image generator. Use generate_images() method instead.
        """
        return {
            "response": (
                "Please use the generate_images() method to create images. "
                "The create_completion method is implemented only for BaseLLMClient compatibility."
            ),
            "tool_calls": []
        }

    def _generate_images_sync(
        self,
        prompt: str,
        number_of_images: int = 1,
        aspect_ratio: str = "1:1",
        person_generation: str = "allow_adult",
        negative_prompt: Optional[str] = None,
        **kwargs
    ) -> Any:
        """Provider-specific implementation of image generation."""
        # Set up generation config
        config = gtypes.GenerateImagesConfig(
            number_of_images=number_of_images,
            aspect_ratio=aspect_ratio,
            person_generation=person_generation,
        )
        
        # Add negative prompt if specified
        if negative_prompt:
            config.negative_prompt = negative_prompt
        
        # Generate images
        log.info(f"Generating {number_of_images} images with prompt: {prompt}")
        return self.client.models.generate_images(
            model=self.model,
            prompt=prompt,
            config=config
        )

    async def _save_images(self, response: Any, output_dir: str) -> List[str]:
        """Save generated images to disk."""
        image_files = []
        
        try:
            for i, gen_image in enumerate(response.generated_images):
                # Get the image data
                image_data = gen_image.image.image_bytes
                
                # Create a unique filename
                filename = f"{Path(output_dir) / f'image_{i+1}_{int(time.time())}.png'}"
                
                # Save the image
                image = Image.open(BytesIO(image_data))
                await self._call_blocking(image.save, filename)
                
                image_files.append(filename)
                log.info(f"Saved image to: {filename}")
                
            return image_files
        except Exception as e:
            log.error(f"Error saving images: {e}")
            raise

    @classmethod
    async def generate_image_for_video(
        cls,
        prompt: str,
        output_dir: str = ".",
        model: str = "imagen-3.0-generate-002", 
        aspect_ratio: str = "16:9",
        api_key: Optional[str] = None,
    ) -> Optional[str]:
        """
        Convenience method to generate an image that can be used as input for video generation.
        
        Args:
            prompt: Text description for the image
            output_dir: Directory to save the image
            model: Name of the image generation model to use
            aspect_ratio: Aspect ratio matching the intended video
            api_key: Optional API key
            
        Returns:
            Path to the generated image if successful, None otherwise
        """
        client = cls(model=model, api_key=api_key)
        result = await client.generate_images(
            prompt=prompt,
            output_dir=output_dir,
            number_of_images=1,
            aspect_ratio=aspect_ratio,
            person_generation="allow_adult"  # Match video defaults
        )
        
        if result["status"] == "complete" and result["image_files"]:
            return result["image_files"][0]
        return None