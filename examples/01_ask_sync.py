#!/usr/bin/env python3
"""
Simple example demonstrating ChukLLM sync functions.
Clean developer experience - just import and use!
"""

from dotenv import load_dotenv

load_dotenv()

from chuk_llm import (
    ask_anthropic_sync,
    ask_deepseek_sync,
    ask_granite_sync,
    ask_groq_sync,
    ask_mistral_sync,
    ask_openai_sync,
    ask_sync,
)

# Simple question
question = "What's 2+2? Answer briefly."

# Default provider
print("Default:")
print(ask_sync(question))
print()

# OpenAI
print("OpenAI:")
print(ask_openai_sync(question))
print()

# Anthropic
print("Anthropic:")
print(ask_anthropic_sync(question))
print()

# Groq (super fast)
print("Groq:")
print(ask_groq_sync(question))
print()

# Mistral
print("Mistral:")
print(ask_mistral_sync(question))
print()

# DeepSeek
print("DeepSeek:")
print(ask_deepseek_sync(question))
print()

# Ollama (local)
print("Granite:")
print(ask_granite_sync(question))
print()
