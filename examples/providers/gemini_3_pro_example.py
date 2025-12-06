#!/usr/bin/env python3
"""
Google Gemini 3 Pro Example
============================

Simple example demonstrating Gemini 3 Pro Preview model usage.

Gemini 3 Pro features:
- Advanced reasoning capabilities
- 1 million token context window
- Adaptive thinking mode
- Multimodal understanding

Requirements:
- pip install chuk-llm
- Set GOOGLE_API_KEY environment variable

Usage:
    python gemini_3_pro_example.py
"""

import asyncio
import os
import sys

from dotenv import load_dotenv

load_dotenv()

# Check environment
if not os.getenv("GOOGLE_API_KEY"):
    print("❌ Please set GOOGLE_API_KEY environment variable")
    print("   export GOOGLE_API_KEY='your_api_key_here'")
    sys.exit(1)

try:
    from chuk_llm.llm.client import get_client
    from chuk_llm.core.models import Message
    from chuk_llm.core.enums import MessageRole
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("   Install with: pip install chuk-llm")
    sys.exit(1)


async def main():
    """Run Gemini 3 Pro example"""
    print("\n" + "=" * 70)
    print("🚀 Google Gemini 3 Pro Preview Example")
    print("=" * 70)
    print(f"API Key: {os.getenv('GOOGLE_API_KEY')[:20]}...")
    print("=" * 70)

    # Create client with Gemini 3 Pro
    client = get_client("gemini", model="gemini-3-pro-preview")

    # Example 1: Advanced reasoning
    print("\n" + "=" * 70)
    print("Example 1: Advanced Reasoning")
    print("=" * 70)

    messages = [
        Message(
            role=MessageRole.USER,
            content="Explain the difference between consciousness and intelligence in AI systems. Be concise but thorough."
        )
    ]

    print("\nPrompt: Explain the difference between consciousness and intelligence in AI systems.")
    print("\n⏳ Gemini 3 Pro is thinking...\n")

    response = await client.create_completion(messages)
    print(f"Response:\n{response['response']}")

    # Example 2: Complex problem solving
    print("\n" + "=" * 70)
    print("Example 2: Complex Problem Solving")
    print("=" * 70)

    messages = [
        Message(
            role=MessageRole.USER,
            content="You have a 3-liter jug and a 5-liter jug. How can you measure exactly 4 liters? Explain step by step."
        )
    ]

    print("\nPrompt: Water jug problem (3L and 5L jugs, measure 4L)")
    print("\n⏳ Gemini 3 Pro is solving...\n")

    response = await client.create_completion(messages)
    print(f"Response:\n{response['response']}")

    # Example 3: Creative writing
    print("\n" + "=" * 70)
    print("Example 3: Creative Writing")
    print("=" * 70)

    messages = [
        Message(
            role=MessageRole.USER,
            content="Write a short poem about the future of artificial intelligence and humanity working together. Make it hopeful and inspiring."
        )
    ]

    print("\nPrompt: Write a poem about AI and humanity's future together")
    print("\n⏳ Gemini 3 Pro is creating...\n")

    response = await client.create_completion(messages)
    print(f"Response:\n{response['response']}")

    print("\n" + "=" * 70)
    print("✅ All Gemini 3 Pro examples completed!")
    print("=" * 70)
    print("\nℹ️  Key Features of Gemini 3 Pro:")
    print("  - Advanced reasoning and problem-solving")
    print("  - 1M token context window")
    print("  - Adaptive thinking capabilities")
    print("  - Multimodal understanding")
    print("  - Enhanced creative abilities")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
