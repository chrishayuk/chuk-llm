#!/usr/bin/env python3
"""
DeepSeek V3.2 Example
=====================

Simple example demonstrating DeepSeek V3.2 models (chat and reasoner modes).

DeepSeek V3.2 features:
- deepseek-chat: Fast, non-thinking mode for quick responses
- deepseek-reasoner: Thinking mode with chain-of-thought reasoning
- 671B parameters with MoE architecture
- Extremely cost-effective ($0.27/M input, $1.10/M output tokens)
- Strong coding and math capabilities

Requirements:
- pip install chuk-llm
- Set DEEPSEEK_API_KEY environment variable

Usage:
    python deepseek_v3_example.py
    python deepseek_v3_example.py --model deepseek-chat
    python deepseek_v3_example.py --model deepseek-reasoner
"""

import argparse
import asyncio
import os
import sys

from dotenv import load_dotenv

load_dotenv()

# Check environment
if not os.getenv("DEEPSEEK_API_KEY"):
    print("❌ Please set DEEPSEEK_API_KEY environment variable")
    print("   export DEEPSEEK_API_KEY='your_api_key_here'")
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
    """Run DeepSeek V3.2 examples"""
    parser = argparse.ArgumentParser(description="DeepSeek V3.2 Examples")
    parser.add_argument(
        "--model",
        default="deepseek-chat",
        choices=["deepseek-chat", "deepseek-reasoner"],
        help="Model to use (default: deepseek-chat)",
    )
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("🚀 DeepSeek V3.2 Example")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Mode: {'Thinking (Chain-of-Thought)' if 'reasoner' in args.model else 'Fast (Non-thinking)'}")
    print(f"API Key: {os.getenv('DEEPSEEK_API_KEY')[:20]}...")
    print("=" * 70)

    # Create client
    client = get_client("deepseek", model=args.model)

    # Example 1: Code generation
    print("\n" + "=" * 70)
    print("Example 1: Code Generation (DeepSeek's Strength)")
    print("=" * 70)

    messages = [
        Message(
            role=MessageRole.USER,
            content="Write a Python function to find the longest common subsequence of two strings using dynamic programming. Include comments explaining the algorithm."
        )
    ]

    print("\nPrompt: Longest common subsequence with DP")
    print(f"\n⏳ {args.model} is coding...\n")

    response = await client.create_completion(messages)
    print(f"Response:\n{response['response']}")

    # Example 2: Mathematical reasoning
    print("\n" + "=" * 70)
    print("Example 2: Mathematical Problem Solving")
    print("=" * 70)

    messages = [
        Message(
            role=MessageRole.USER,
            content="If a train travels at 60 mph for 2 hours, then 80 mph for 3 hours, what is its average speed for the entire journey? Show your work step by step."
        )
    ]

    print("\nPrompt: Average speed calculation")
    print(f"\n⏳ {args.model} is calculating...\n")

    response = await client.create_completion(messages)
    print(f"Response:\n{response['response']}")

    # Example 3: Logic puzzle (better with reasoner)
    print("\n" + "=" * 70)
    print("Example 3: Logic Puzzle")
    print("=" * 70)

    messages = [
        Message(
            role=MessageRole.USER,
            content="Three friends - Alice, Bob, and Carol - are standing in a line. Alice is not at the front. Bob is not at the back. Carol is not in the middle. What is the order from front to back? Explain your reasoning."
        )
    ]

    print("\nPrompt: Logic puzzle (order of friends)")
    print(f"\n⏳ {args.model} is reasoning...\n")

    response = await client.create_completion(messages)
    print(f"Response:\n{response['response']}")

    # Example 4: Comparison between chat and reasoner modes
    print("\n" + "=" * 70)
    print("Example 4: Chat vs Reasoner Comparison")
    print("=" * 70)

    prompt = "What is 15% of 240? Show your calculation."
    print(f"\nPrompt: {prompt}\n")

    for model_name in ["deepseek-chat", "deepseek-reasoner"]:
        try:
            test_client = get_client("deepseek", model=model_name)
            messages = [Message(role=MessageRole.USER, content=prompt)]

            import time
            start = time.time()
            response = await test_client.create_completion(messages)
            duration = time.time() - start

            mode = "Thinking" if "reasoner" in model_name else "Fast"
            print(f"{model_name} [{mode}] ({duration:.2f}s):")
            print(f"  {response['response']}\n")
        except Exception as e:
            print(f"{model_name}: ❌ {e}\n")

    print("=" * 70)
    print("✅ All DeepSeek V3.2 examples completed!")
    print("=" * 70)
    print("\nℹ️  DeepSeek V3.2 Overview:")
    print("  • deepseek-chat: Fast mode for quick responses")
    print("  • deepseek-reasoner: Thinking mode with chain-of-thought")
    print("  • 671B parameters (MoE architecture)")
    print("  • Extremely cost-effective pricing")
    print("  • Excels at: coding, math, logical reasoning")
    print("=" * 70)
    print("\n💡 When to use each mode:")
    print("  • deepseek-chat: General tasks, fast responses needed")
    print("  • deepseek-reasoner: Complex logic, math, detailed reasoning")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
