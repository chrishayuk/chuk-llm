#!/usr/bin/env python3
"""
Quick Start - Simplest Possible Example
========================================

The fastest way to get started with chuk-llm.
Just one line to ask a question!
"""

from dotenv import load_dotenv
load_dotenv()

from chuk_llm import quick_question

# The simplest way to use chuk-llm
# Auto-detects available providers and uses the best one
answer = quick_question("What is 2+2? Answer briefly.")
print(answer)

# That's it! No configuration needed.
# chuk-llm automatically:
# 1. Detects available API keys in your environment
# 2. Selects the best available provider
# 3. Makes the request and returns the answer
