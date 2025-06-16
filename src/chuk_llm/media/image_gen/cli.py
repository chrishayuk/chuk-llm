#!/usr/bin/env python
# chuk_llm/image_generator/cli.py
"""
Command-line tool for generating images.

Example usage:
    python image_gen.py --prompt "A cat playing with a ball of yarn" --output images
    python image_gen.py --prompt "A stunning landscape" --aspect-ratio 16:9 --count 4
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# image generator
from .client import ImageGeneratorClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
log = logging.getLogger("image_gen_cli")

# Load environment variables from .env file
load_dotenv()

# Check for API key
if not os.getenv("GEMINI_API_KEY") and not os.getenv("GEMINI_API_KEY"):
    log.error("API key not found. Set the GEMINI_API_KEY or GEMINI_API_KEY environment variable.")
    sys.exit(1)

async def main():
    # setup the parser
    parser = argparse.ArgumentParser(description="Generate images using Gemini Imagen")

    # setup arguments
    parser.add_argument("--prompt", type=str, required=True, help="Text description for the image")
    parser.add_argument("--output", type=str, default="images", help="Output directory for images")
    parser.add_argument("--model", type=str, default="imagen-3.0-generate-002", help="Image generation model")
    parser.add_argument("--aspect-ratio", type=str, default="1:1", 
                        choices=["1:1", "16:9", "9:16", "4:3", "3:4"], help="Aspect ratio")
    parser.add_argument("--person-generation", type=str, default="allow_adult", 
                        choices=["dont_allow", "allow_adult"], help="Person generation policy")
    parser.add_argument("--negative-prompt", type=str, help="Description of what to avoid in generation")
    parser.add_argument("--count", type=int, default=1, choices=[1, 2, 3, 4], 
                        help="Number of images to generate (1-4)")
    parser.add_argument("--api-key", type=str, help="API key (overrides environment variable)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--for-video", action="store_true", 
                        help="Optimize image for use with video generation")
    
    # parse arguments
    args = parser.parse_args()
    
    # Set log level based on verbosity
    if args.verbose:
        log.setLevel(logging.DEBUG)
        logging.getLogger("chuk_llm").setLevel(logging.DEBUG)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    try:
        # Create the client
        client = ImageGeneratorClient(model=args.model, api_key=args.api_key)
        
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
            
            # If generating for video, print command to use the image
            if args.for_video and result['image_files']:
                image_path = result['image_files'][0]
                video_aspect_ratio = args.aspect_ratio.replace(':', '-')
                print("\nTo use this image with video generation, run:")
                print(f"python video_gen.py --prompt \"{args.prompt}\" --image \"{image_path}\" --aspect-ratio {args.aspect_ratio}")
    
    except Exception as e:
        log.error(f"Error generating images: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())