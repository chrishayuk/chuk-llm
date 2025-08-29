#!/usr/bin/env python3
"""
Model-specific function demonstration - using the correct function names
"""

from dotenv import load_dotenv

load_dotenv()

from chuk_llm import (
    ask_anthropic_claude_sonnet_4_20250514_sync,
    ask_anthropic_sonnet_sync,  # Alias
    # OpenAI model-specific functions
    ask_openai_gpt_4o_mini_sync,
)

question = "What's 2+2? Explain your reasoning."

print("=== Model-Specific Demo ===\n")

# OpenAI specific models
print("🤖 OpenAI Models:")
print("-" * 40)

print("GPT-4o Mini:")
print(ask_openai_gpt_4o_mini_sync(question))
print()

# Anthropic specific models
print("\n🧠 Anthropic Models:")
print("-" * 40)

print("Claude Sonnet 4:")
print(ask_anthropic_claude_sonnet_4_20250514_sync(question))
print()

# Global aliases
print("\n🌍 Global Aliases:")
print("-" * 40)

# Using aliases for convenience
print("\n✨ Using Convenient Aliases:")
print("-" * 40)

print("Anthropic 'sonnet' alias:")
print(ask_anthropic_sonnet_sync(question))
print()
