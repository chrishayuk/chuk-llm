# chuk_llm/media/video_gen/base.py
"""
Base classes for video generation capabilities.

This module defines abstract base classes for video generation,
providing a clean separation from text-based LLM interfaces.
"""

from __future__ import annotations

import abc
import asyncio
import logging
import os
from typing import Any, Dict, List, Optional, Union

log = logging.getLogger(__name__)

class BaseVideoGenerator(abc.ABC):
    """
    Abstract base class for video generation.
    
    Providers should implement the abstract methods to support their specific API.
    """
    
    @staticmethod
    async def _call_blocking(fn, *args, **kwargs):
        """Run a blocking function in a background thread."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))
    
    async def generate_video(
        self,
        prompt: str,
        output_dir: str = ".",
        aspect_ratio: str = "16:9",
        person_generation: str = "allow",
        negative_prompt: Optional[str] = None,
        duration_seconds: int = 5,
        number_of_videos: int = 1,
        image_path: Optional[str] = None,
        wait_for_completion: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate video based on prompt and optional input image.
        
        Args:
            prompt: Text description of video to generate
            output_dir: Directory to save generated videos
            aspect_ratio: Aspect ratio of the video
            person_generation: Policy for generating people
            negative_prompt: Description of what to avoid in generation
            duration_seconds: Video length in seconds
            number_of_videos: Number of videos to generate
            image_path: Optional path to image to use as first frame
            wait_for_completion: Whether to wait for generation to complete
            **kwargs: Additional provider-specific arguments
            
        Returns:
            Dictionary with operation info and file paths if wait_for_completion
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate the operation
        try:
            # Call the provider-specific implementation
            operation = await self._call_blocking(
                self._generate_video_sync,
                prompt=prompt,
                output_dir=output_dir,
                aspect_ratio=aspect_ratio,
                person_generation=person_generation,
                negative_prompt=negative_prompt,
                duration_seconds=duration_seconds,
                number_of_videos=number_of_videos,
                image_path=image_path,
                **kwargs
            )
            
            operation_id = self._get_operation_id(operation)
            log.info(f"Video generation operation started: {operation_id}")
            
            if not wait_for_completion:
                return {
                    "operation_id": operation_id,
                    "status": "pending",
                    "message": "Video generation in progress. Call check_operation() to monitor."
                }
            
            # Wait for completion
            return await self._wait_for_videos(operation, operation_id, output_dir)
            
        except Exception as e:
            log.error(f"Error in video generation: {e}")
            raise
    
    async def check_operation(self, operation_id: str) -> Dict[str, Any]:
        """
        Check status of a video generation operation.
        
        Args:
            operation_id: Operation ID from generate_video
            
        Returns:
            Dictionary with operation status
        """
        try:
            operation = await self._call_blocking(self._check_operation_sync, operation_id)
            
            if not self._is_operation_complete(operation):
                return {
                    "operation_id": operation_id,
                    "status": "pending",
                    "message": "Video generation is still in progress."
                }
            
            return {
                "operation_id": operation_id,
                "status": "complete",
                "message": "Video generation is complete. Call download_videos() to retrieve."
            }
        except Exception as e:
            log.error(f"Error checking operation {operation_id}: {e}")
            return {
                "operation_id": operation_id,
                "status": "error",
                "message": f"Error checking operation: {e}"
            }
    
    async def download_videos(self, operation_id: str, output_dir: str = ".") -> Dict[str, Any]:
        """
        Download videos from a completed operation.
        
        Args:
            operation_id: Operation ID from generate_video
            output_dir: Directory to save videos
            
        Returns:
            Dictionary with download status and file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            operation = await self._call_blocking(self._check_operation_sync, operation_id)
            
            if not self._is_operation_complete(operation):
                return {
                    "operation_id": operation_id,
                    "status": "pending",
                    "message": "Video generation is still in progress. Cannot download yet."
                }
            
            return await self._download_videos(operation, output_dir)
        except Exception as e:
            log.error(f"Error downloading videos for operation {operation_id}: {e}")
            return {
                "operation_id": operation_id,
                "status": "error",
                "message": f"Error downloading videos: {e}"
            }
    
    async def _wait_for_videos(
        self, 
        operation: Any, 
        operation_id: str,
        output_dir: str
    ) -> Dict[str, Any]:
        """Wait for video generation to complete and download results."""
        log.info("Waiting for video generation to complete...")
        
        # Poll for completion
        while not self._is_operation_complete(operation):
            await asyncio.sleep(20)  # Check every 20 seconds
            operation = await self._call_blocking(self._check_operation_sync, operation_id)
            log.info(f"Operation status: {'DONE' if self._is_operation_complete(operation) else 'PENDING'}")
        
        # Download videos
        return await self._download_videos(operation, output_dir)
    
    @abc.abstractmethod
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
        """
        Provider-specific implementation for video generation.
        
        This method must be implemented by provider classes.
        """
        pass
    
    @abc.abstractmethod
    def _check_operation_sync(self, operation_id: str) -> Any:
        """
        Provider-specific implementation for checking operation status.
        
        This method must be implemented by provider classes.
        """
        pass
    
    @abc.abstractmethod
    def _get_operation_id(self, operation: Any) -> str:
        """
        Extract operation ID from provider-specific operation object.
        
        This method must be implemented by provider classes.
        """
        pass
    
    @abc.abstractmethod
    def _is_operation_complete(self, operation: Any) -> bool:
        """
        Check if operation is complete from provider-specific operation object.
        
        This method must be implemented by provider classes.
        """
        pass
    
    @abc.abstractmethod
    async def _download_videos(self, operation: Any, output_dir: str) -> Dict[str, Any]:
        """
        Download videos from a completed operation.
        
        This method must be implemented by provider classes.
        """
        pass