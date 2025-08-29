#!/usr/bin/env python3
"""
Parameters example - customize LLM behavior
"""

from dotenv import load_dotenv

load_dotenv()

from chuk_llm import ask_anthropic_sync, ask_openai_sync

# Temperature: Controls creativity (0.0 = predictable, 2.0 = very creative)
print("=== Temperature ===")
print("Low (0.2):", ask_openai_sync("Name a color", temperature=0.2))
print("High (1.5):", ask_openai_sync("Name a color", temperature=1.5))
print()

# Max tokens: Limit response length
print("=== Max Tokens ===")
print("Limited to 50 tokens:")
print(ask_anthropic_sync("Explain the universe", max_tokens=50))
print()
