#!/usr/bin/env python3
from chuk_llm import (
    ask_openai_gpt_4o_mini_sync,
    ask_anthropic_claude_3_5_sonnet_20241022_sync,
)

question = "What's 2+2? Explain your reasoning."

print("=== Model-Specific Demo ===\n")

print("OpenAI GPT-4o Mini:")
print(ask_openai_gpt_4o_mini_sync(question))
print()

print("Anthropic Claude-3.5 Sonnet:")
print(ask_anthropic_claude_3_5_sonnet_20241022_sync(question))
print()