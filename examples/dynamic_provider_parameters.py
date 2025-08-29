#!/usr/bin/env python3
"""
Dynamic Provider with Advanced Parameters Example
=================================================

This example demonstrates using various parameters with dynamic providers:
- Temperature control (creativity)
- Max tokens (response length)
- Top-p (nucleus sampling)
- Frequency and presence penalties
- Stop sequences
- System messages
- JSON mode
"""

import json
import os

from chuk_llm import (
    ask_json,
    ask_sync,
    register_openai_compatible,
    stream_sync_iter,
    unregister_provider,
)


def demo_temperature_control(provider_name: str):
    """Demonstrate temperature effects on creativity."""
    print("\n🌡️  TEMPERATURE CONTROL (Creativity)")
    print("-" * 40)

    prompt = "Write a one-sentence description of a rainbow"

    for temp in [0.0, 0.7, 1.5]:
        response = ask_sync(
            prompt,
            provider=provider_name,
            temperature=temp,
            max_tokens=50,
            seed=42,  # For more consistent comparisons
        )
        print(f"Temperature {temp}: {response}")

    print("\n💡 Higher temperature = more creative/random responses")


def demo_max_tokens(provider_name: str):
    """Demonstrate controlling response length."""
    print("\n📏 MAX TOKENS (Response Length)")
    print("-" * 40)

    prompt = "List the planets in our solar system"

    for max_tok in [10, 30, 100]:
        response = ask_sync(
            prompt,
            provider=provider_name,
            temperature=0,  # Consistent output
            max_tokens=max_tok,
        )
        word_count = len(response.split())
        print(f"\nMax tokens {max_tok} (~{word_count} words):")
        print(f"  {response}")

    print("\n💡 Max tokens limits the response length")


def demo_top_p(provider_name: str):
    """Demonstrate nucleus sampling."""
    print("\n🎯 TOP-P (Nucleus Sampling)")
    print("-" * 40)

    prompt = "Suggest a creative name for a coffee shop"

    for top_p in [0.1, 0.5, 0.95]:
        response = ask_sync(
            prompt, provider=provider_name, temperature=0.8, top_p=top_p, max_tokens=20
        )
        print(f"Top-p {top_p}: {response}")

    print("\n💡 Lower top-p = more focused, higher = more diverse")


def demo_frequency_penalty(provider_name: str):
    """Demonstrate frequency penalty for reducing repetition."""
    print("\n🔄 FREQUENCY PENALTY (Reduce Repetition)")
    print("-" * 40)

    prompt = "Write about the importance of water. Mention water multiple times."

    for freq_penalty in [0.0, 1.0, 2.0]:
        response = ask_sync(
            prompt,
            provider=provider_name,
            temperature=0.7,
            frequency_penalty=freq_penalty,
            max_tokens=60,
        )
        # Count occurrences of "water"
        water_count = response.lower().count("water")
        print(f"\nFreq penalty {freq_penalty} ('water' appears {water_count} times):")
        print(f"  {response}")

    print("\n💡 Higher frequency penalty discourages word repetition")


def demo_presence_penalty(provider_name: str):
    """Demonstrate presence penalty for topic diversity."""
    print("\n🎭 PRESENCE PENALTY (Topic Diversity)")
    print("-" * 40)

    prompt = "List 5 things you can do at a beach"

    for pres_penalty in [0.0, 1.0, 2.0]:
        response = ask_sync(
            prompt,
            provider=provider_name,
            temperature=0.7,
            presence_penalty=pres_penalty,
            max_tokens=80,
        )
        print(f"\nPresence penalty {pres_penalty}:")
        print(f"  {response}")

    print("\n💡 Higher presence penalty encourages exploring new topics")


def demo_stop_sequences(provider_name: str):
    """Demonstrate using stop sequences."""
    print("\n🛑 STOP SEQUENCES")
    print("-" * 40)

    # Example 1: Stop at a number
    response = ask_sync(
        "Count from 1 to 10: 1, 2, 3,",
        provider=provider_name,
        temperature=0,
        stop=["7", "7,"],  # Stop when reaching "7"
        max_tokens=50,
    )
    print(f"Counting (stop at '7'): {response}")

    # Example 2: Stop at punctuation
    response = ask_sync(
        "List fruits: apple, banana,",
        provider=provider_name,
        temperature=0,
        stop=[".", "!", "?"],  # Stop at sentence end
        max_tokens=50,
    )
    print(f"List fruits (stop at punctuation): {response}")

    print("\n💡 Stop sequences terminate generation at specific strings")


def demo_system_messages(provider_name: str):
    """Demonstrate system message for setting behavior."""
    print("\n💬 SYSTEM MESSAGES (Behavior Control)")
    print("-" * 40)

    user_prompt = "Explain what Python is"

    # Different personalities via system message
    personalities = [
        "You are a helpful assistant. Be concise.",
        "You are a pirate. Speak like a pirate in all responses.",
        "You are a 5-year-old. Explain things simply.",
    ]

    for system_msg in personalities:
        # Using messages format
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_prompt},
        ]

        response = ask_sync(
            messages, provider=provider_name, temperature=0.7, max_tokens=50
        )
        print(f"\nSystem: {system_msg[:40]}...")
        print(f"Response: {response}")

    print("\n💡 System messages set the AI's behavior and personality")


def demo_json_mode(provider_name: str):
    """Demonstrate JSON mode for structured output."""
    print("\n📊 JSON MODE (Structured Output)")
    print("-" * 40)

    try:
        response = ask_json(
            "Generate a JSON object for a book with title, author, year, and genres (array)",
            provider=provider_name,
            temperature=0.7,
        )

        print("Response type:", type(response))
        print("JSON output:")
        print(json.dumps(response, indent=2))

        # Verify it's valid JSON
        if isinstance(response, dict):
            print(f"\n✅ Valid JSON with keys: {list(response.keys())}")

    except Exception as e:
        print(f"⚠️  JSON mode may not be supported: {e}")

    print("\n💡 JSON mode ensures structured, parseable output")


def demo_streaming_with_params(provider_name: str):
    """Demonstrate streaming with parameters."""
    print("\n🌊 STREAMING WITH PARAMETERS")
    print("-" * 40)

    print("Creative (high temp): ", end="", flush=True)
    for chunk in stream_sync_iter(
        "Write a creative tagline for a tech company",
        provider=provider_name,
        temperature=1.2,
        max_tokens=15,
        top_p=0.95,
    ):
        content = chunk if isinstance(chunk, str) else chunk.get("response", "")
        print(content, end="", flush=True)

    print("\n\nFocused (low temp):   ", end="", flush=True)
    for chunk in stream_sync_iter(
        "Write a creative tagline for a tech company",
        provider=provider_name,
        temperature=0.2,
        max_tokens=15,
        top_p=0.1,
    ):
        content = chunk if isinstance(chunk, str) else chunk.get("response", "")
        print(content, end="", flush=True)

    print("\n\n💡 Parameters work with streaming too")


def main():
    print("=" * 60)
    print("DYNAMIC PROVIDER - ADVANCED PARAMETERS")
    print("=" * 60)

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  OPENAI_API_KEY not set!")
        print("Please set: export OPENAI_API_KEY='your-key'")
        return

    # Register provider
    print("\n📝 Registering dynamic provider...")
    provider_name = "param_demo"

    register_openai_compatible(
        name=provider_name,
        api_base="https://api.openai.com/v1",
        api_key=os.getenv("OPENAI_API_KEY"),
        models=["gpt-3.5-turbo", "gpt-4"],
        default_model="gpt-3.5-turbo",
    )
    print(f"✅ Provider '{provider_name}' registered")

    # Run all demos
    try:
        demo_temperature_control(provider_name)
        demo_max_tokens(provider_name)
        demo_top_p(provider_name)
        demo_frequency_penalty(provider_name)
        demo_presence_penalty(provider_name)
        demo_stop_sequences(provider_name)
        demo_system_messages(provider_name)
        demo_json_mode(provider_name)
        demo_streaming_with_params(provider_name)

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
    finally:
        # Clean up
        print("\n\n🧹 Cleaning up...")
        unregister_provider(provider_name)
        print(f"✅ Provider '{provider_name}' unregistered")

    # Summary
    print("\n" + "=" * 60)
    print("PARAMETER REFERENCE")
    print("=" * 60)
    print("""
Temperature (0.0-2.0): Controls randomness/creativity
Max Tokens: Limits response length
Top-p (0.0-1.0): Nucleus sampling for diversity
Frequency Penalty (-2.0-2.0): Reduces repetition
Presence Penalty (-2.0-2.0): Encourages topic variety
Stop Sequences: Early termination strings
System Messages: Set AI behavior/personality
JSON Mode: Structured output format
Streaming: Real-time response generation

All parameters work with dynamically registered providers!
""")


if __name__ == "__main__":
    main()
