#!/usr/bin/env python3
from chuk_llm import (
    ask_sync, 
    ask_gemini_sync, 
    ask_openai_sync, 
    ask_mistral_sync, 
    ask_ollama_sync, 
    ask_anthropic_sync, 
    ask_mistral_sync,
    ask_deepseek_sync,
    ask_granite_sync)

# ask the default provider and model
#print(ask_gemini_sync("What's 2+2?"))

print("openai")
print(ask_openai_sync("What's 2+2?"))
print()

print("mistral")
print(ask_mistral_sync("What's 2+2?"))
print()

print("ollama")
print(ask_ollama_sync("What's 2+2?"))
print()

print("anthropic")
print(ask_anthropic_sync("What's 2+2?"))
print()

print("mistral")
print(ask_mistral_sync("What's 2+2?"))
print()

print("deepseek")
print(ask_deepseek_sync("What's 2+2?"))
print()

print("granite")
print(ask_granite_sync("What's 2+2?"))
print()