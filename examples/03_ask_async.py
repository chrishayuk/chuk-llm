#!/usr/bin/env python3
"""
Async model-specific function demonstration
"""

import asyncio
from dotenv import load_dotenv
load_dotenv()

from chuk_llm import (
    # OpenAI model-specific functions (async versions - no _sync suffix)
    ask_openai_gpt4o_mini,
    ask_anthropic_claude_sonnet4_20250514,
    ask_anthropic_sonnet,  # Alias
)

async def main():
    question = "What's 2+2? Explain your reasoning."
    
    print("=== Async Model-Specific Demo ===\n")
    
    # OpenAI specific models
    print("🤖 OpenAI Models:")
    print("-" * 40)
    
    print("GPT-4o Mini:")
    response = await ask_openai_gpt4o_mini(question)
    print(response)
    print()
    
    # Anthropic specific models
    print("\n🧠 Anthropic Models:")
    print("-" * 40)
    
    print("Claude Sonnet 4:")
    response = await ask_anthropic_claude_sonnet4_20250514(question)
    print(response)
    print()
    
    # Using aliases for convenience
    print("\n✨ Using Convenient Aliases:")
    print("-" * 40)
    
    print("Anthropic 'sonnet' alias:")
    response = await ask_anthropic_sonnet(question)
    print(response)
    print()
    
    # Parallel requests example
    print("\n⚡ Parallel Requests Example:")
    print("-" * 40)
    print("Asking both models simultaneously...")
    
    # Create tasks for parallel execution
    tasks = [
        ask_openai_gpt4o_mini("What's 3+3?"),
        ask_anthropic_sonnet("What's 4+4?")
    ]
    
    # Wait for both to complete
    results = await asyncio.gather(*tasks)
    
    print(f"OpenAI: {results[0]}")
    print(f"Anthropic: {results[1]}")

# Run the async main function
if __name__ == "__main__":
    asyncio.run(main())