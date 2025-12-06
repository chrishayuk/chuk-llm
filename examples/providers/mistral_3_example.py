#!/usr/bin/env python3
"""
Mistral 3 Series Example
========================

Simple example demonstrating Mistral 3 series models (Large 3, Ministral 3).

Mistral 3 features:
- Mistral Large 3: 675B total params, 41B active (MoE architecture)
- Ministral 3: Available in 3B, 8B, and 14B variants
- All models support multimodal understanding (vision)
- Apache 2.0 license (open-weight, commercial use)

Requirements:
- pip install chuk-llm
- Set MISTRAL_API_KEY environment variable

Usage:
    python mistral_3_example.py
    python mistral_3_example.py --model mistral-large-2512
    python mistral_3_example.py --model ministral-8b-2512
"""

import argparse
import asyncio
import os
import sys

from dotenv import load_dotenv

load_dotenv()

# Check environment
if not os.getenv("MISTRAL_API_KEY"):
    print("❌ Please set MISTRAL_API_KEY environment variable")
    print("   export MISTRAL_API_KEY='your_api_key_here'")
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
    """Run Mistral 3 examples"""
    parser = argparse.ArgumentParser(description="Mistral 3 Series Examples")
    parser.add_argument(
        "--model",
        default="mistral-large-2512",
        choices=["mistral-large-2512", "mistral-large-latest",
                 "ministral-3b-2512", "ministral-8b-2512",
                 "ministral-14b-2512", "ministral-14b-latest"],
        help="Model to use (default: mistral-large-2512)",
    )
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("🚀 Mistral 3 Series Example")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"API Key: {os.getenv('MISTRAL_API_KEY')[:20]}...")
    print("=" * 70)

    # Create client
    client = get_client("mistral", model=args.model)

    # Example 1: Advanced reasoning
    print("\n" + "=" * 70)
    print("Example 1: Reasoning and Problem Solving")
    print("=" * 70)

    messages = [
        Message(
            role=MessageRole.USER,
            content="If you have 3 apples and you give away 2, then someone gives you 5 more, how many do you have? Explain your reasoning."
        )
    ]

    print("\nPrompt: Simple math word problem with reasoning")
    print(f"\n⏳ {args.model} is thinking...\n")

    response = await client.create_completion(messages)
    print(f"Response:\n{response['response']}")

    # Example 2: Code generation
    print("\n" + "=" * 70)
    print("Example 2: Code Generation")
    print("=" * 70)

    messages = [
        Message(
            role=MessageRole.USER,
            content="Write a Python function to check if a string is a palindrome. Include docstring and example usage."
        )
    ]

    print("\nPrompt: Generate Python palindrome checker")
    print(f"\n⏳ {args.model} is coding...\n")

    response = await client.create_completion(messages)
    print(f"Response:\n{response['response']}")

    # Example 3: Creative writing
    print("\n" + "=" * 70)
    print("Example 3: Creative Writing")
    print("=" * 70)

    messages = [
        Message(
            role=MessageRole.USER,
            content="Write a haiku about open-source AI empowering developers."
        )
    ]

    print("\nPrompt: Write a haiku about open-source AI")
    print(f"\n⏳ {args.model} is creating...\n")

    response = await client.create_completion(messages)
    print(f"Response:\n{response['response']}")

    # Example 4: Comparison across Mistral 3 family
    print("\n" + "=" * 70)
    print("Example 4: Mistral 3 Family Comparison")
    print("=" * 70)

    models_to_compare = ["ministral-8b-2512", "ministral-14b-2512", "mistral-large-2512"]
    prompt = "Explain machine learning in exactly 15 words."

    print(f"\nPrompt: {prompt}\n")

    for model_name in models_to_compare:
        try:
            test_client = get_client("mistral", model=model_name)
            messages = [Message(role=MessageRole.USER, content=prompt)]

            import time
            start = time.time()
            response = await test_client.create_completion(messages)
            duration = time.time() - start

            print(f"{model_name} ({duration:.2f}s):")
            print(f"  {response['response']}\n")
        except Exception as e:
            print(f"{model_name}: ❌ {e}\n")

    print("=" * 70)
    print("✅ All Mistral 3 examples completed!")
    print("=" * 70)
    print("\nℹ️  Mistral 3 Series Overview:")
    print("  • Mistral Large 3: 675B total, 41B active params (MoE)")
    print("  • Ministral 3B: Efficient edge deployment")
    print("  • Ministral 8B: Balanced performance/efficiency")
    print("  • Ministral 14B: Enhanced capabilities")
    print("  • All models: Multimodal, Apache 2.0 license")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
