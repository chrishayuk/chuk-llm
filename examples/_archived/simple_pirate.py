#!/usr/bin/env python3
"""Simple pirate system prompt example"""

from chuk_llm import ask_ollama_granite3_3_latest

# Set a pirate system prompt
pirate_prompt = """You are a pirate captain. Always speak like a pirate with 'arr', 
'matey', 'ahoy', and other pirate expressions. Be dramatic and mention the sea."""

# Ask questions with the pirate personality
response = ask_ollama_granite3_3_latest(
    "What is Python programming?",
    system_prompt=pirate_prompt
)

print("🏴‍☠️ Pirate says:")
print(response)