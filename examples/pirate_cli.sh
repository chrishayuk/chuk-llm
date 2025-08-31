#!/bin/bash
# Pirate system prompt examples using the CLI

echo "🏴‍☠️ Pirate CLI Examples 🏴‍☠️"
echo "================================"
echo ""

# Example 1: Simple pirate response
echo "1. Basic pirate response:"
uv run chuk-llm ask "What is machine learning?" \
  --provider ollama \
  --model granite3.3:latest \
  --system-prompt "You are a pirate. Speak in pirate dialect with 'arr' and 'matey'."

echo ""
echo "--------------------------------"
echo ""

# Example 2: Using convenience function with system prompt
echo "2. Using convenience function:"
uv run chuk-llm ask_ollama_granite3_3_latest "Tell me about databases" \
  --system-prompt "You are Captain Jack Sparrow. Be witty and use pirate speak."

echo ""
echo "--------------------------------"
echo ""

# Example 3: Pirate code explanation
echo "3. Pirate explains code:"
uv run chuk-llm ask "Explain what a for loop does" \
  --provider ollama \
  --model granite3.3:latest \
  --system-prompt "You are a pirate programmer. Explain coding concepts using nautical metaphors."

echo ""
echo "🏴‍☠️ That be all, matey! 🏴‍☠️"