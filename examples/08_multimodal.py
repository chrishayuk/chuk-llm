#!/usr/bin/env python3
"""
Multimodal Inputs - Vision and Audio
=====================================

Demonstrates using images and audio with LLMs.
Supports image URLs, base64 data, and local files.
"""

import asyncio
import base64
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from chuk_llm import ask
from chuk_llm.core.models import Message, ImageUrlContent, TextContent

async def image_from_url():
    """Analyze image from URL."""
    print("=== Image from URL ===\n")

    # Create message with image URL
    message = Message(
        role="user",
        content=[
            TextContent(text="What do you see in this image? Describe it in detail."),
            ImageUrlContent(
                image_url="https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"
            )
        ]
    )

    response = await ask(
        messages=[message],
        provider="openai",
        model="gpt-4o-mini"
    )

    print(f"AI: {response}\n")

async def multiple_images():
    """Analyze multiple images together."""
    print("=== Multiple Images ===\n")

    message = Message(
        role="user",
        content=[
            TextContent(text="Compare these two images. What are the differences?"),
            ImageUrlContent(
                image_url="https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg"
            ),
            ImageUrlContent(
                image_url="https://upload.wikimedia.org/wikipedia/commons/4/4d/Cat_November_2010-1a.jpg"
            )
        ]
    )

    response = await ask(
        messages=[message],
        provider="openai",
        model="gpt-4o-mini"
    )

    print(f"AI: {response}\n")

async def image_from_base64():
    """Send image as base64 data."""
    print("=== Image from Base64 ===\n")

    # In a real app, you would read an actual image file
    # This is a simple example
    from chuk_llm.core.models import ImageDataContent

    # Simulated base64 image data (in reality, load from file)
    image_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

    message = Message(
        role="user",
        content=[
            TextContent(text="What color is this image?"),
            ImageDataContent(
                mime_type="image/png",
                data=image_data
            )
        ]
    )

    try:
        response = await ask(
            messages=[message],
            provider="openai",
            model="gpt-4o-mini"
        )
        print(f"AI: {response}\n")
    except Exception as e:
        print(f"Note: Base64 example - {e}\n")

async def image_with_specific_questions():
    """Ask specific questions about images."""
    print("=== Specific Questions About Images ===\n")

    image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"

    questions = [
        "What animal is in this image?",
        "What color is the animal?",
        "What is the animal doing?",
        "Where might this photo have been taken?"
    ]

    for question in questions:
        message = Message(
            role="user",
            content=[
                TextContent(text=question),
                ImageUrlContent(image_url=image_url)
            ]
        )

        response = await ask(
            messages=[message],
            provider="openai",
            model="gpt-4o-mini"
        )

        print(f"Q: {question}")
        print(f"A: {response}\n")

async def vision_with_different_providers():
    """Vision support across different providers."""
    print("=== Vision Across Providers ===\n")

    image_url = "https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg"

    message = Message(
        role="user",
        content=[
            TextContent(text="What do you see? Be brief."),
            ImageUrlContent(image_url=image_url)
        ]
    )

    providers = [
        ("openai", "gpt-4o-mini"),
        ("anthropic", "claude-3-5-sonnet-20241022"),
        ("gemini", "gemini-2.0-flash-exp"),
    ]

    for provider, model in providers:
        try:
            response = await ask(
                messages=[message],
                provider=provider,
                model=model
            )
            print(f"{provider}: {response[:100]}...\n")
        except Exception as e:
            print(f"{provider}: Not available - {e}\n")

async def code_in_image():
    """Analyze code or text in images (OCR)."""
    print("=== Code/Text in Images ===\n")

    # Use an image with code
    message = Message(
        role="user",
        content=[
            TextContent(
                text="Extract and explain the code shown in this image."
            ),
            ImageUrlContent(
                image_url="https://example.com/code-screenshot.png"
            )
        ]
    )

    try:
        response = await ask(
            messages=[message],
            provider="openai",
            model="gpt-4o-mini"
        )
        print(f"AI: {response}\n")
    except Exception as e:
        print(f"Note: This example requires a real code image - {e}\n")

async def image_understanding_with_context():
    """Provide context with images."""
    print("=== Image Understanding with Context ===\n")

    message = Message(
        role="user",
        content=[
            TextContent(
                text="""I'm creating a website for pet adoption.
                Look at this image and suggest:
                1. A good title for the listing
                2. Key features to highlight
                3. What type of home would be best"""
            ),
            ImageUrlContent(
                image_url="https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg"
            )
        ]
    )

    response = await ask(
        messages=[message],
        provider="openai",
        model="gpt-4o-mini"
    )

    print(f"AI: {response}\n")

async def chart_analysis():
    """Analyze charts and graphs."""
    print("=== Chart Analysis ===\n")

    # Use an image with a chart/graph
    message = Message(
        role="user",
        content=[
            TextContent(
                text="Analyze this chart. What are the key trends and insights?"
            ),
            ImageUrlContent(
                image_url="https://example.com/sales-chart.png"
            )
        ]
    )

    try:
        response = await ask(
            messages=[message],
            provider="gemini",
            model="gemini-2.0-flash-exp"
        )
        print(f"AI: {response}\n")
    except Exception as e:
        print(f"Note: This example requires a real chart image - {e}\n")

if __name__ == "__main__":
    asyncio.run(image_from_url())
    asyncio.run(multiple_images())
    asyncio.run(image_from_base64())
    asyncio.run(image_with_specific_questions())
    asyncio.run(vision_with_different_providers())
    asyncio.run(code_in_image())
    asyncio.run(image_understanding_with_context())
    asyncio.run(chart_analysis())

    print("="*50)
    print("✅ Multimodal support makes AI truly versatile!")
    print("="*50)
