#!/usr/bin/env python
"""
Example script demonstrating how to use the image and video generators from Python code.

This script shows examples of:
1. Generating images
2. Generating videos
3. Creating an image-to-video workflow
"""

import asyncio
import os

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the generators
from chuk_llm.media.image_gen import ImageGenerator
from chuk_llm.media.video_gen import VideoGenerator

# Create output directory
os.makedirs("media_output", exist_ok=True)


async def example_generate_images():
    """Example of generating images."""
    print("\n=== Generating Images ===")

    # Create the image generator
    image_gen = ImageGenerator()

    # Generate images with a prompt
    result = await image_gen.generate_images(
        prompt="A futuristic city with flying cars and tall skyscrapers",
        output_dir="media_output",
        number_of_images=2,
        aspect_ratio="16:9",
    )

    # Print the results
    print(f"Status: {result['status']}")
    if result["status"] == "complete":
        print(f"Generated {len(result['image_files'])} images:")
        for image_file in result["image_files"]:
            print(f"  - {image_file}")

    return result


async def example_generate_video():
    """Example of generating a video."""
    print("\n=== Generating Video ===")

    # Create the video generator
    video_gen = VideoGenerator()

    # Generate a video with a prompt
    result = await video_gen.generate_video(
        prompt="A timelapse of a flower blooming in a garden",
        output_dir="media_output",
        aspect_ratio="16:9",
        duration_seconds=5,
    )

    # Print the results
    print(f"Status: {result['status']}")
    if result["status"] == "complete":
        print(f"Generated {len(result['video_files'])} videos:")
        for video_file in result["video_files"]:
            print(f"  - {video_file}")

    return result


async def example_image_to_video_workflow():
    """Example of an image-to-video workflow."""
    print("\n=== Image-to-Video Workflow ===")

    # First, generate an image
    print("Step 1: Generating an image...")
    image_gen = ImageGenerator()
    img_result = await image_gen.generate_images(
        prompt="A serene lake surrounded by mountains at sunset",
        output_dir="media_output",
        number_of_images=1,
        aspect_ratio="16:9",
    )

    if img_result["status"] != "complete" or not img_result["image_files"]:
        print("Failed to generate image. Cannot proceed to video generation.")
        return

    image_path = img_result["image_files"][0]
    print(f"Generated image: {image_path}")

    # Then, use the image to generate a video
    print("\nStep 2: Generating a video from the image...")
    video_gen = VideoGenerator()
    video_result = await video_gen.generate_video(
        prompt="Camera slowly pans across the lake scene, with gentle water ripples",
        image_path=image_path,
        output_dir="media_output",
        wait_for_completion=True,
    )

    # Print the results
    print(f"Status: {video_result['status']}")
    if video_result["status"] == "complete":
        print(f"Generated {len(video_result['video_files'])} videos:")
        for video_file in video_result["video_files"]:
            print(f"  - {video_file}")

    return video_result


async def example_non_blocking_video_generation():
    """Example of non-blocking video generation with status checking."""
    print("\n=== Non-blocking Video Generation ===")

    # Create the video generator
    video_gen = VideoGenerator()

    # Start video generation without waiting
    print("Starting video generation (non-blocking)...")
    result = await video_gen.generate_video(
        prompt="A drone flying over a cityscape at night with sparkling lights",
        output_dir="media_output",
        wait_for_completion=False,
    )

    if result["status"] != "pending":
        print("Failed to start video generation.")
        return

    operation_id = result["operation_id"]
    print(f"Operation started with ID: {operation_id}")

    # Check status a few times
    for i in range(3):
        print(f"\nChecking status (attempt {i + 1})...")
        await asyncio.sleep(10)  # Wait 10 seconds between checks

        status_result = await video_gen.check_operation(operation_id)
        print(f"Status: {status_result['status']}")
        print(f"Message: {status_result['message']}")

        if status_result["status"] == "complete":
            # Download the videos
            print("\nDownloading videos...")
            download_result = await video_gen.download_videos(
                operation_id, "media_output"
            )

            if download_result["status"] == "complete":
                print(f"Downloaded {len(download_result['video_files'])} videos:")
                for video_file in download_result["video_files"]:
                    print(f"  - {video_file}")
            return download_result

    print(
        "\nOperation still in progress. In a real application, you would poll until completion."
    )
    return result


async def main():
    """Run all examples."""
    # Choose which example to run
    example = input("""Select an example to run:
1. Generate images
2. Generate video
3. Image-to-video workflow
4. Non-blocking video generation
5. Run all examples
Enter number (1-5): """).strip()

    if example == "1":
        await example_generate_images()
    elif example == "2":
        await example_generate_video()
    elif example == "3":
        await example_image_to_video_workflow()
    elif example == "4":
        await example_non_blocking_video_generation()
    elif example == "5":
        await example_generate_images()
        await example_generate_video()
        await example_image_to_video_workflow()
        await example_non_blocking_video_generation()
    else:
        print("Invalid choice. Please run again and select a number from 1-5.")


if __name__ == "__main__":
    asyncio.run(main())
