#!/usr/bin/env python3
"""Test streaming function behavior in sync context"""
from chuk_llm import stream_ollama_granite

# Test 1: Try using stream in sync context
print("Testing stream in sync context:")
result = stream_ollama_granite("Tell me a 3 word story")
print(f"Result type: {type(result)}")
print(f"Result: {result}")

# Try to iterate
try:
    for chunk in result:
        print(chunk, end="")
except Exception as e:
    print(f"\nError iterating: {e}")