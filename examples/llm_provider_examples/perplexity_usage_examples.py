#!/usr/bin/env python3
"""
Perplexity Provider Example Usage Script
======================================

A drop-in rewrite of *deepseek_usage_examples.py* that exercises the **Perplexity**
backend through the **chuk-llm** abstraction layer.

Perplexity speaks the OpenAI API dialect, so every call made here goes through
`OpenAILLMClient` with a different `api_base` (``https://api.perplexity.ai``)
and API key.

Prerequisites
-------------
1.  ``pip install openai chuk-llm python-dotenv``
2.  Export your key **and** leave the OpenAI one unset (to avoid accidentally
    sending traffic to OpenAI):

        export PERPLEXITY_API_KEY="sk-…"     # required
        unset OPENAI_API_KEY                  # optional but recommended

Usage
-----
    python perplexity_usage_examples.py
    python perplexity_usage_examples.py --model sonar-pro
    python perplexity_usage_examples.py --skip-functions

The script showcases the following:
  • Basic text completion
  • Streaming output
  • (Optional) Function calling*
  • JSON-only responses
  • Model comparison
  • Simple multi-turn chat
  • Temperature sweep

*Perplexity's models support the OpenAI function-calling interface, but you
 must be on a plan that enables ``tool_calls``.
"""

import asyncio
import argparse
import os
import sys
import time
from typing import Any, Dict, List

from dotenv import load_dotenv

load_dotenv()

# ────────────────────────────────────────────────────────────────────────────
# Environment sanity check
# ────────────────────────────────────────────────────────────────────────────
if not os.getenv("PERPLEXITY_API_KEY"):
    print("❌ Please set PERPLEXITY_API_KEY environment variable")
    print("   export PERPLEXITY_API_KEY='your_api_key_here'")
    sys.exit(1)

try:
    from chuk_llm.llm.llm_client import get_llm_client
    from chuk_llm.llm.configuration.capabilities import CapabilityChecker
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("   Make sure you installed chuk-llm and are running from the repo root")
    sys.exit(1)

PROVIDER = "perplexity"            # << changed from deepseek
DEFAULT_MODEL = "sonar-pro"        # versatile 128K-context model

# =============================================================================
# Example 1: Basic Text Completion
# =============================================================================
async def basic_text_example(model: str = DEFAULT_MODEL):
    print(f"\n🤖 Basic Text Completion with {model}")
    print("=" * 60)

    client = get_llm_client(PROVIDER, model=model)

    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Explain transformers in simple terms (2-3 sentences)."},
    ]

    start = time.time()
    response = await client.create_completion(messages)
    duration = time.time() - start

    print(f"✅ Response ({duration:.2f}s):\n   {response['response']}")
    return response

# =============================================================================
# Example 2: Streaming Response
# =============================================================================
async def streaming_example(model: str = DEFAULT_MODEL):
    print(f"\n⚡ Streaming Example with {model}")
    print("=" * 60)

    client = get_llm_client(PROVIDER, model=model)
    messages = [{"role": "user", "content": "Write a short haiku about artificial intelligence."}]

    print("🌊 Streaming response:\n   ", end="", flush=True)
    start = time.time()
    full = ""

    async for chunk in client.create_completion(messages, stream=True):
        if chunk.get("response"):
            print(chunk["response"], end="", flush=True)
            full += chunk["response"]

    print(f"\n✅ Streaming completed ({time.time() - start:.2f}s)")
    return full

# =============================================================================
# Example 3: Function Calling
# =============================================================================
async def function_calling_example(model: str = DEFAULT_MODEL):
    print(f"\n🔧 Function Calling with {model}")
    print("=" * 60)

    ok, issues = CapabilityChecker.can_handle_request(PROVIDER, model, has_tools=True)
    if not ok:
        print(f"⚠️  Skipping function calling: {', '.join(issues)}")
        return None

    client = get_llm_client(PROVIDER, model=model)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_web",
                "description": "Search the web for information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "max_results": {"type": "integer", "description": "Max results"},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "calculate_math",
                "description": "Evaluate a math expression",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string"},
                        "precision": {"type": "integer"},
                    },
                    "required": ["expression"],
                },
            },
        },
    ]

    messages = [
        {
            "role": "user",
            "content": "Search for 'LLM eval benchmarks 2025' and calculate 3.14159 * 42 with 2 decimal places.",
        }
    ]

    print("🔄 Making function-calling request…")
    response = await client.create_completion(messages, tools=tools)

    if not response.get("tool_calls"):
        print("ℹ️  No tool calls were made\n   ", response["response"])
        return response

    print(f"✅ Tool calls requested: {len(response['tool_calls'])}")
    for i, call in enumerate(response["tool_calls"], 1):
        print(f"   {i}. {call['function']['name']}({call['function']['arguments']})")

    # --- Simulate tool execution (stub) ----------------------------------
    messages.append({"role": "assistant", "content": "", "tool_calls": response["tool_calls"]})
    for call in response["tool_calls"]:
        name = call["function"]["name"]
        if name == "search_web":
            result = '{"results": ["Benchmark X", "Benchmark Y", "Benchmark Z"]}'
        elif name == "calculate_math":
            result = '{"result": 131.95, "expression": "3.14159 * 42", "precision": 2}'
        else:
            result = '{"status": "ok"}'

        messages.append({"role": "tool", "tool_call_id": call["id"], "name": name, "content": result})

    print("🔄 Getting final response…")
    final = await client.create_completion(messages)
    print("✅ Final response:\n   ", final["response"])
    return final

# =============================================================================
# Example 5: Model Comparison
# =============================================================================
async def model_comparison_example():
    print("\n📊 Model Comparison")
    print("=" * 60)

    models = ["sonar-pro", "llama-3.1-sonar-small-128k-online"]
    prompt = "What is machine learning? (one sentence)"
    results: Dict[str, Any] = {}

    for m in models:
        try:
            print(f"🔄 Testing {m}…")
            client = get_llm_client(PROVIDER, model=m)
            start = time.time()
            response = await client.create_completion([{"role": "user", "content": prompt}])
            duration = time.time() - start
            results[m] = {"time": duration, "response": response["response"], "success": True}
        except Exception as e:
            results[m] = {"time": 0, "response": str(e), "success": False}

    for m, res in results.items():
        status = "✅" if res["success"] else "❌"
        snippet = res["response"][:80].replace("\n", " ")
        print(f"   {status} {m} – {res['time']:.2f}s – {snippet}…")
    return results

# =============================================================================
# Example 6: Simple Chat
# =============================================================================
async def simple_chat_example(model: str = DEFAULT_MODEL):
    print("\n💬 Simple Chat Interface")
    print("=" * 60)

    client = get_llm_client(PROVIDER, model=model)
    conversation = [
        "Hello! What's the weather like?",
        "What's the most exciting development in AI recently?",
        "Can you help me write a JavaScript function to sort an array?",
    ]

    messages: List[Dict[str, Any]] = []
    for user_msg in conversation:
        print(f"👤 {user_msg}")
        messages.append({"role": "user", "content": user_msg})
        reply = await client.create_completion(messages)
        print(f"🤖 {reply['response']}\n")
        messages.append({"role": "assistant", "content": reply["response"]})
    return messages

# =============================================================================
# Example 7: Temperature Sweep
# =============================================================================
async def parameters_example(model: str = DEFAULT_MODEL):
    print("\n🎛️  Temperature Sweep")
    print("=" * 60)

    client = get_llm_client(PROVIDER, model=model)
    prompt = "Write a creative opening line for a science-fiction story."
    for temp in (0.1, 0.7, 1.2):
        print(f"\n🌡️  Temperature {temp}:")
        out = await client.create_completion([{"role": "user", "content": prompt}], temperature=temp, max_tokens=50)
        print("   ", out["response"])
    return True

# =============================================================================
# Main driver
# =============================================================================
async def main():
    parser = argparse.ArgumentParser(description="Perplexity Provider Example Script")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model to use")
    parser.add_argument("--skip-functions", action="store_true", help="Skip function-calling example")
    parser.add_argument("--quick", action="store_true", help="Run only basic examples")
    args = parser.parse_args()

    print("🚀 Perplexity Provider Examples")
    print("=" * 60)
    print(f"Using model: {args.model}")
    print(f"API Key: {'✅ Set' if os.getenv('PERPLEXITY_API_KEY') else '❌ Missing'}")

    examples = [
        ("Basic Text", lambda: basic_text_example(args.model)),
        ("Streaming", lambda: streaming_example(args.model)),
        ("JSON Mode", lambda: json_mode_example(args.model)),
        ("Model Comparison", model_comparison_example),
        ("Simple Chat", lambda: simple_chat_example(args.model)),
        ("Parameters Test", lambda: parameters_example(args.model)),
    ]

    if not args.skip_functions:
        examples.insert(2, ("Function Calling", lambda: function_calling_example(args.model)))

    if args.quick:
        examples = examples[:1]

    results: Dict[str, Dict[str, Any]] = {}
    for name, coro in examples:
        print("\n" + "=" * 60)
        start = time.time()
        try:
            result = await coro()
            results[name] = {"success": True, "time": time.time() - start}
            print(f"✅ {name} completed in {results[name]['time']:.2f}s")
        except Exception as e:
            results[name] = {"success": False, "time": 0, "error": str(e)}
            print(f"❌ {name} failed: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    ok = sum(r["success"] for r in results.values())
    total = len(results)
    total_time = sum(r["time"] for r in results.values())
    print(f"✅ Successful: {ok}/{total}")
    print(f"⏱️  Total time: {total_time:.2f}s")

    for n, r in results.items():
        status = "✅" if r["success"] else "❌"
        print(f"   {status} {n}: {r['time']:.2f}s" if r["success"] else f"   {status} {n}: failed")

    if ok == total:
        print("\n🎉 All examples completed successfully!")
    else:
        print("\n⚠️  Some examples failed. Check your API key, model name or plan.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Cancelled by user")
    except Exception as exc:
        print(f"\n💥 Unexpected error: {exc}")
        sys.exit(1)
