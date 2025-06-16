#!/usr/bin/env python3
"""
Parameters example - customize LLM behavior
"""

from dotenv import load_dotenv
load_dotenv()

from chuk_llm import ask_openai_sync, ask_anthropic_sync, ask_sync

# System prompt example
print("=== System Prompt Example ===\n")

print("As a pirate:")
print(ask_sync(
    "How do I cook pasta?",
    system_prompt="You are a helpful pirate. Always speak like a pirate."
))
print()

print("As a chef:")
print(ask_sync(
    "How do I cook pasta?", 
    system_prompt="You are a professional chef. Be precise and technical."
))
print()

# Multiple parameters
print("=== Combined Parameters ===\n")

print("Creative story with limits:")
print(ask_openai_sync(
    "Write a story about a robot",
    temperature=1.2,
    max_tokens=100,
    system_prompt="You are a creative writer who loves plot twists."
))