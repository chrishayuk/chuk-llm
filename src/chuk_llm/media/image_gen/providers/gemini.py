# chuk_llm/media/image_gen/providers/gemini.py
"""
Gemini Imagen image generation client implementation.

This module implements the BaseImageGenerator interface for Google's Imagen model.
"""

from __future__ import annotations

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

from chuk_llm.media.image_gen.base import BaseImageGenerator

log = logging.getLogger(__name__)

# Honour LOGLEVEL env-var for quick local tweaks
if "LOGLEVEL" in os.environ:
    log.setLevel(os.environ["LOGLEVEL"].upper())

class GeminiImageGenerator(BaseImageGenerator):
    """Implementation of image generator for Google's Imagen model via Gemini API."""

    def __init__(
        self,
        model: str = "imagen-3.0-generate-002",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,  # Not used but kept for consistency with other providers
    ) -> None:
        """
        Initialize the Gemini image generator.
        
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
        log.info("Gemini Image Generator initialized with model '%s'", model)

    def _generate_images_sync(
        self,
        prompt: str,
        number_of_images: int = 1,
        aspect_ratio: str = "1:1",
        person_generation: str = "allow",
        negative_prompt: Optional[str] = None,
        **kwargs
    ) -> Any:
        """Generate images synchronously using Gemini's Imagen model."""
        # Map generic person generation option to Gemini-specific values
        gemini_person_generation = "allow_adult"
        if person_generation.lower() in ("deny", "dont_allow", "disallow"):
            gemini_person_generation = "dont_allow"
        
        # Set up generation config
        config = gtypes.GenerateImagesConfig(
            number_of_images=number_of_images,
            aspect_ratio=aspect_ratio,
            person_generation=gemini_person_generation,
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
            person_generation="allow"  # Match video defaults
        )
        
        if result["status"] == "complete" and result["image_files"]:
            return result["image_files"][0]
        return None