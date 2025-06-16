#!/usr/bin/env python
"""
Integrated command-line tool for generating images and videos.

This tool provides a unified interface for image and video generation using
the chuk_llm image and video generators.

Example usage:
    # Generate an image
    python media_gen.py image --prompt "A cat playing with a ball of yarn"
    
    # Generate a video
    python media_gen.py video --prompt "A cat jumping in slow motion"
    
    # Generate an image and use it as input for video
    python media_gen.py image-to-video --prompt "A cat sitting on a windowsill"
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
log = logging.getLogger("media_gen_cli")

# Load environment variables from .env file
load_dotenv()

# Check for API key
if not os.getenv("GEMINI_API_KEY") and not os.getenv("GEMINI_API_KEY"):
    log.error("API key not found. Set the GEMINI_API_KEY or GEMINI_API_KEY environment variable.")
    sys.exit(1)

# Import the generator clients
try:
    from chuk_llm.image_generator import ImageGenerator
    from chuk_llm.video_generator import VideoGenerator
except ImportError:
    print("Could not import the required generator clients. Make sure chuk_llm is installed.")
    sys.exit(1)

async def generate_image(args):
    """Handle image generation command."""
    os.makedirs(args.output, exist_ok=True)
    
    try:
        # Create the client
        client = ImageGenerator(model=args.model, api_key=args.api_key)
        
        # Generate images
        print(f"Generating {args.count} image(s) with prompt: {args.prompt}")
        
        result = await client.generate_images(
            prompt=args.prompt,
            output_dir=args.output,
            number_of_images=args.count,
            aspect_ratio=args.aspect_ratio,
            person_generation=args.person_generation,
            negative_prompt=args.negative_prompt
        )
        
        print(f"Status: {result['status']}")
        print(f"Message: {result['message']}")
        
        if result['status'] == 'complete' and 'image_files' in result:
            print(f"Generated images: {', '.join(result['image_files'])}")
            return result['image_files']
        
        return []
    
    except Exception as e:
        log.error(f"Error generating images: {e}")
        sys.exit(1)

async def generate_video(args, input_image=None):
    """Handle video generation command."""
    os.makedirs(args.output, exist_ok=True)
    
    try:
        # Create the client
        client = VideoGenerator(model=args.model, api_key=args.api_key)
        
        # Generate video
        print(f"Generating video with prompt: {args.prompt}")
        if input_image:
            print(f"Using image: {input_image}")
        
        result = await client.generate_video(
            prompt=args.prompt,
            output_dir=args.output,
            aspect_ratio=args.aspect_ratio,
            person_generation=args.person_generation,
            negative_prompt=args.negative_prompt,
            duration_seconds=args.duration,
            number_of_videos=args.count,
            image_path=input_image or args.image,
            wait_for_completion=not args.no_wait
        )
        
        print(f"Status: {result['status']}")
        print(f"Message: {result['message']}")
        
        if result['status'] == 'pending':
            print(f"Operation ID: {result['operation_id']}")
            print("To check status later, run:")
            print(f"  python media_gen.py check-operation {result['operation_id']}")
            print("To download the videos later, run:")
            print(f"  python media_gen.py download-videos {result['operation_id']} --output {args.output}")
        elif result['status'] == 'complete' and 'video_files' in result:
            print(f"Generated videos: {', '.join(result['video_files'])}")
        
        return result
    
    except Exception as e:
        log.error(f"Error generating video: {e}")
        sys.exit(1)

async def image_to_video_workflow(args):
    """Generate an image and then use it to generate a video."""
    # First generate the image
    print("=== Step 1: Generating Image ===")
    image_files = await generate_image(args)
    
    if not image_files:
        log.error("No images were generated. Cannot proceed to video generation.")
        sys.exit(1)
    
    # Then generate the video using the first image
    print("\n=== Step 2: Generating Video ===")
    await generate_video(args, input_image=image_files[0])

async def check_operation(args):
    """Check the status of an operation."""
    try:
        client = VideoGenerator(api_key=args.api_key)
        result = await client.check_operation(args.operation_id)
        
        print(f"Operation ID: {result['operation_id']}")
        print(f"Status: {result['status']}")
        print(f"Message: {result['message']}")
        
    except Exception as e:
        log.error(f"Error checking operation: {e}")
        sys.exit(1)

async def download_videos(args):
    """Download videos from a completed operation."""
    try:
        os.makedirs(args.output, exist_ok=True)
        
        client = VideoGenerator(api_key=args.api_key)
        result = await client.download_videos(args.operation_id, args.output)
        
        print(f"Status: {result['status']}")
        print(f"Message: {result['message']}")
        
        if result['status'] == 'complete' and 'video_files' in result:
            print(f"Downloaded videos: {', '.join(result['video_files'])}")
        
    except Exception as e:
        log.error(f"Error downloading videos: {e}")
        sys.exit(1)

def add_image_args(parser):
    """Add image-specific arguments to a parser."""
    parser.add_argument("--prompt", type=str, required=True, help="Text description for the image")
    parser.add_argument("--output", type=str, default="media", help="Output directory")
    parser.add_argument("--model", type=str, default="imagen-3.0-generate-002", help="Image model")
    parser.add_argument("--aspect-ratio", type=str, default="1:1", 
                        choices=["1:1", "16:9", "9:16", "4:3", "3:4"], help="Aspect ratio")
    parser.add_argument("--person-generation", type=str, default="allow", 
                        choices=["allow", "dont_allow"], help="Person generation policy")
    parser.add_argument("--negative-prompt", type=str, help="Description of what to avoid")
    parser.add_argument("--count", type=int, default=1, choices=[1, 2, 3, 4], 
                        help="Number of images to generate (1-4)")
    parser.add_argument("--api-key", type=str, help="API key (overrides environment variable)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

def add_video_args(parser):
    """Add video-specific arguments to a parser."""
    parser.add_argument("--prompt", type=str, required=True, help="Text description for the video")
    parser.add_argument("--output", type=str, default="media", help="Output directory")
    parser.add_argument("--model", type=str, default="veo-2.0-generate-001", help="Video model")
    parser.add_argument("--aspect-ratio", type=str, default="16:9", 
                        choices=["16:9", "9:16"], help="Aspect ratio")
    parser.add_argument("--person-generation", type=str, default="dont_allow", 
                        choices=["allow", "dont_allow"], help="Person generation policy")
    parser.add_argument("--negative-prompt", type=str, help="Description of what to avoid")
    parser.add_argument("--duration", type=int, default=5, choices=range(5, 9), 
                        help="Video duration in seconds (5-8)")
    parser.add_argument("--count", type=int, default=1, choices=[1, 2], 
                        help="Number of videos to generate (1-2)")
    parser.add_argument("--image", type=str, help="Path to input image (optional)")
    parser.add_argument("--no-wait", action="store_true", help="Don't wait for completion")
    parser.add_argument("--api-key", type=str, help="API key (overrides environment variable)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

async def main():
    parser = argparse.ArgumentParser(description="Media Generation Tool")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Image generation command
    image_parser = subparsers.add_parser("image", help="Generate images")
    add_image_args(image_parser)
    
    # Video generation command
    video_parser = subparsers.add_parser("video", help="Generate videos")
    add_video_args(video_parser)
    
    # Image-to-video workflow command
    i2v_parser = subparsers.add_parser("image-to-video", help="Generate an image then use it for video")
    # Need both image and video args, but using video aspect ratio for both
    i2v_parser.add_argument("--prompt", type=str, required=True, help="Text description")
    i2v_parser.add_argument("--output", type=str, default="media", help="Output directory")
    i2v_parser.add_argument("--image-model", type=str, default="imagen-3.0-generate-002", 
                           help="Image generation model")
    i2v_parser.add_argument("--video-model", type=str, default="veo-2.0-generate-001", 
                           help="Video generation model")
    i2v_parser.add_argument("--aspect-ratio", type=str, default="16:9", 
                            choices=["16:9", "9:16"], help="Aspect ratio")
    i2v_parser.add_argument("--person-generation", type=str, default="dont_allow", 
                            choices=["allow", "dont_allow"], help="Person generation policy")
    i2v_parser.add_argument("--negative-prompt", type=str, help="Description of what to avoid")
    i2v_parser.add_argument("--duration", type=int, default=5, choices=range(5, 9), 
                            help="Video duration in seconds (5-8)")
    i2v_parser.add_argument("--count", type=int, default=1, choices=[1, 2], 
                            help="Number of videos to generate (1-2)")
    i2v_parser.add_argument("--no-wait", action="store_true", help="Don't wait for completion")
    i2v_parser.add_argument("--api-key", type=str, help="API key (overrides environment variable)")
    i2v_parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    # Check operation command
    check_parser = subparsers.add_parser("check-operation", help="Check video operation status")
    check_parser.add_argument("operation_id", type=str, help="Operation ID to check")
    check_parser.add_argument("--api-key", type=str, help="API key (overrides environment variable)")
    check_parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    # Download videos command
    download_parser = subparsers.add_parser("download-videos", help="Download videos from operation")
    download_parser.add_argument("operation_id", type=str, help="Operation ID to download")
    download_parser.add_argument("--output", type=str, default="media", help="Output directory")
    download_parser.add_argument("--api-key", type=str, help="API key (overrides environment variable)")
    download_parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set log level based on verbosity
    if getattr(args, 'verbose', False):
        log.setLevel(logging.DEBUG)
        logging.getLogger("chuk_llm").setLevel(logging.DEBUG)
    
    # Route to the appropriate handler based on command
    if args.command == "image":
        await generate_image(args)
    elif args.command == "video":
        await generate_video(args)
    elif args.command == "image-to-video":
        # Set up args for the image generation part
        args.model = args.image_model
        await image_to_video_workflow(args)
    elif args.command == "check-operation":
        await check_operation(args)
    elif args.command == "download-videos":
        await download_videos(args)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())