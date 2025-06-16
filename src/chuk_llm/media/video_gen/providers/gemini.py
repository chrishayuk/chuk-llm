# chuk_llm/media/video_gen/gemini.py
"""
Gemini Veo video generation client implementation.

This module implements the BaseVideoGenerator interface for Google's Veo model.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv
from google import genai
from google.genai import types as gtypes
from PIL import Image

from chuk_llm.media.video_gen.base import BaseVideoGenerator

log = logging.getLogger(__name__)

# Honour LOGLEVEL env-var for quick local tweaks
if "LOGLEVEL" in os.environ:
    log.setLevel(os.environ["LOGLEVEL"].upper())

class GeminiVideoGenerator(BaseVideoGenerator):
    """Implementation of video generator for Google's Veo model via Gemini API."""

    def __init__(
        self,
        model: str = "veo-2.0-generate-001",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,  # Not used but kept for consistency with other providers
    ) -> None:
        """
        Initialize the Gemini video generator.
        
        Args:
            model: Name of the video generation model to use
            api_key: Optional API key (will use GEMINI_API_KEY environment variable if not provided)
            api_base: Not used for Gemini, kept for interface consistency
        """
        load_dotenv()
        self.model = model
        api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY / GEMINI_API_KEY environment variable not set")
        
        self.client = genai.Client(api_key=api_key)
        log.info("Gemini Video Generator initialized with model '%s'", model)

    def _generate_video_sync(
        self,
        prompt: str,
        output_dir: str,
        aspect_ratio: str,
        person_generation: str,
        negative_prompt: Optional[str],
        duration_seconds: int,
        number_of_videos: int,
        image_path: Optional[str],
        **kwargs
    ) -> Any:
        """Generate videos synchronously using Gemini's Veo model."""
        # Map generic person generation option to Gemini-specific values
        gemini_person_generation = "allow_adult"
        if person_generation.lower() in ("deny", "dont_allow", "disallow"):
            gemini_person_generation = "dont_allow"
        
        # Set up generation config
        config = gtypes.GenerateVideosConfig(
            person_generation=gemini_person_generation,
            aspect_ratio=aspect_ratio,
            number_of_videos=number_of_videos,
            duration_seconds=duration_seconds,
            enhance_prompt=kwargs.get("enhance_prompt", True)
        )
        
        # Add negative prompt if specified
        if negative_prompt:
            config.negative_prompt = negative_prompt
        
        # Read image if specified
        image = None
        if image_path:
            try:
                log.info(f"Loading image from: {image_path}")
                image = Image.open(image_path)
            except Exception as e:
                log.error(f"Failed to load image from {image_path}: {e}")
                raise ValueError(f"Cannot load image from {image_path}: {e}")
        
        # Start video generation operation
        log.info(f"Starting video generation with prompt: {prompt}")
        if image:
            return self.client.models.generate_videos(
                model=self.model,
                prompt=prompt,
                image=image,
                config=config
            )
        else:
            return self.client.models.generate_videos(
                model=self.model,
                prompt=prompt,
                config=config
            )

    def _check_operation_sync(self, operation_id: str) -> Any:
        """Check operation status synchronously."""
        return self.client.operations.get(operation_id)

    def _get_operation_id(self, operation: Any) -> str:
        """Extract operation ID from Gemini operation object."""
        return getattr(operation, "name", str(id(operation)))

    def _is_operation_complete(self, operation: Any) -> bool:
        """Check if Gemini operation is complete."""
        return getattr(operation, "done", False)

    async def _download_videos(self, operation: Any, output_dir: str) -> Dict[str, Any]:
        """Download videos from a completed Gemini operation."""
        video_files = []
        
        try:
            for i, video in enumerate(operation.response.generated_videos):
                filename = f"{Path(output_dir) / f'video_{i+1}_{int(time.time())}.mp4'}"
                await self._call_blocking(self.client.files.download, file=video.video)
                await self._call_blocking(video.video.save, filename)
                video_files.append(filename)
                log.info(f"Saved video to: {filename}")
            
            return {
                "operation_id": self._get_operation_id(operation),
                "status": "complete",
                "video_files": video_files,
                "message": f"Generated {len(video_files)} videos"
            }
        except Exception as e:
            log.error(f"Error downloading videos: {e}")
            return {
                "operation_id": self._get_operation_id(operation),
                "status": "error",
                "message": f"Error downloading videos: {e}"
            }

    @classmethod
    async def generate_from_image(
        cls,
        prompt: str,
        image_path: str,
        output_dir: str = ".",
        model: str = "veo-2.0-generate-001",
        aspect_ratio: str = "16:9",
        person_generation: str = "allow",
        api_key: Optional[str] = None,
        wait_for_completion: bool = True
    ) -> Dict[str, Any]:
        """
        Convenience method to generate a video from an image with minimal configuration.
        
        Args:
            prompt: Text description for the video
            image_path: Path to the input image
            output_dir: Directory to save the video
            model: Name of the video generation model to use
            aspect_ratio: Aspect ratio of the video
            person_generation: Person generation policy
            api_key: Optional API key
            wait_for_completion: Whether to wait for generation to complete
            
        Returns:
            Dictionary with operation information
        """
        client = cls(model=model, api_key=api_key)
        return await client.generate_video(
            prompt=prompt,
            image_path=image_path,
            output_dir=output_dir,
            aspect_ratio=aspect_ratio,
            person_generation=person_generation,
            wait_for_completion=wait_for_completion
        )