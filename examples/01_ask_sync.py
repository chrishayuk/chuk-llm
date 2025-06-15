#!/usr/bin/env python3
"""
Simple example demonstrating ChukLLM sync functions.
Clean developer experience - just import and use!
"""

from dotenv import load_dotenv
load_dotenv()

from chuk_llm import (
    ask_sync, 
    ask_openai_sync, 
    ask_anthropic_sync, 
    ask_groq_sync,
    ask_gemini_sync, 
    ask_mistral_sync, 
    ask_ollama_sync, 
    ask_deepseek_sync,
    ask_watsonx_sync,
    ask_perplexity_sync,
    ask_togetherai_sync
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

# Google Gemini
print("Gemini:")
print(ask_gemini_sync(question))
print()

# Mistral
print("Mistral:")
print(ask_mistral_sync(question))
print()

# DeepSeek
print("DeepSeek:")
print(ask_deepseek_sync(question))
print()

# IBM watsonx (Granite models)
print("IBM watsonx:")
print(ask_watsonx_sync(question))
print()

# Perplexity
print("Perplexity:")
print(ask_perplexity_sync(question))
print()

# Together AI
print("Together AI:")
print(ask_togetherai_sync(question))
print()

# Ollama (local)
print("Ollama:")
print(ask_ollama_sync(question))
print()