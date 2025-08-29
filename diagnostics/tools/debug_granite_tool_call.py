#!/usr/bin/env python3
"""
Deep Debug Script for Granite Tool Call Issue
==============================================

This script analyzes exactly what's happening at each level:
1. Raw Ollama API - what Granite actually sends
2. OllamaLLMClient - what the client extracts
3. Stream function - what the high-level API returns
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set up detailed logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


async def debug_raw_ollama():
    """Step 1: See exactly what raw Ollama sends"""
    print("\n" + "=" * 60)
    print("STEP 1: RAW OLLAMA API - What Granite Actually Sends")
    print("=" * 60)

    import ollama

    model_name = "granite3.3:latest"
    messages = [{"role": "user", "content": "What's the weather in San Francisco?"}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            },
        }
    ]

    client = ollama.AsyncClient()
    stream = await client.chat(
        model=model_name,
        messages=messages,
        tools=tools,
        stream=True,
        options={"temperature": 0.1, "num_predict": 200},
    )

    chunk_num = 0
    async for chunk in stream:
        chunk_num += 1
        print(f"\n📦 Chunk {chunk_num}:")

        # Show chunk type and structure
        print(f"  Type: {type(chunk)}")
        print(f"  Has 'message': {hasattr(chunk, 'message')}")

        if hasattr(chunk, "message") and chunk.message:
            msg = chunk.message
            print(f"  Message type: {type(msg)}")

            # Check all attributes
            print(f"  Message attributes: {dir(msg)}")

            # Check content
            content = getattr(msg, "content", None)
            print(f"  Content: '{content}'")

            # CRITICAL: Check tool_calls
            tool_calls = getattr(msg, "tool_calls", None)
            print(f"  tool_calls attribute exists: {hasattr(msg, 'tool_calls')}")
            print(f"  tool_calls value: {tool_calls}")
            print(f"  tool_calls type: {type(tool_calls)}")

            if tool_calls:
                print("  🎯 TOOL CALLS FOUND!")
                print(f"  Number of tool calls: {len(tool_calls)}")
                for i, tc in enumerate(tool_calls):
                    print(f"    Tool call {i}:")
                    print(f"      Type: {type(tc)}")
                    print(f"      Has 'function': {hasattr(tc, 'function')}")
                    if hasattr(tc, "function"):
                        func = tc.function
                        print(f"      Function type: {type(func)}")
                        print(
                            f"      Function.name: {getattr(func, 'name', 'NO NAME')}"
                        )
                        print(
                            f"      Function.arguments: {getattr(func, 'arguments', 'NO ARGS')}"
                        )

        # Check done status
        if hasattr(chunk, "done"):
            print(f"  Done: {chunk.done}")


async def debug_ollama_client():
    """Step 2: See what OllamaLLMClient extracts"""
    print("\n" + "=" * 60)
    print("STEP 2: OllamaLLMClient - What the Client Extracts")
    print("=" * 60)

    from chuk_llm.llm.providers.ollama_client import OllamaLLMClient

    # Monkey-patch to add debug logging
    original_stream = OllamaLLMClient._stream_completion_async

    async def debug_stream(self, messages, tools=None, **kwargs):
        """Wrapped version with debug output"""
        print("\n🔍 Inside _stream_completion_async")
        print(f"  Model: {self.model}")
        print(f"  Tools provided: {bool(tools)}")

        # Call original
        async for chunk in original_stream(self, messages, tools, **kwargs):
            print("\n  📤 Yielding chunk:")
            print(
                f"    Response: '{chunk.get('response', '')[:50]}...' ({len(chunk.get('response', ''))} chars)"
            )
            print(f"    Tool calls: {chunk.get('tool_calls', [])}")
            yield chunk

    # Apply monkey patch
    OllamaLLMClient._stream_completion_async = debug_stream

    model_name = "granite3.3:latest"
    messages = [{"role": "user", "content": "What's the weather in San Francisco?"}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            },
        }
    ]

    client = OllamaLLMClient(model_name)

    chunk_count = 0
    tool_calls_found = []

    stream = client.create_completion(
        messages=messages, tools=tools, stream=True, temperature=0.1, max_tokens=200
    )

    async for chunk in stream:
        chunk_count += 1
        if chunk.get("tool_calls"):
            tool_calls_found.extend(chunk["tool_calls"])

    print("\n📊 Summary:")
    print(f"  Total chunks: {chunk_count}")
    print(f"  Tool calls found: {len(tool_calls_found)}")
    if tool_calls_found:
        print(f"  Functions: {[tc['function']['name'] for tc in tool_calls_found]}")


async def debug_stream_function():
    """Step 3: See what the high-level stream function returns"""
    print("\n" + "=" * 60)
    print("STEP 3: High-Level Stream Function - What User Gets")
    print("=" * 60)

    from chuk_llm import stream

    prompt = "What's the weather in San Francisco?"
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            },
        }
    ]

    print("Calling stream() with return_tool_calls=True")

    chunk_count = 0
    tool_calls_found = []

    async for chunk in stream(
        prompt,
        provider="ollama",
        model="granite3.3:latest",
        tools=tools,
        return_tool_calls=True,
        max_tokens=200,
        temperature=0.1,
    ):
        chunk_count += 1
        print(f"\n📦 Chunk {chunk_count}:")
        print(f"  Type: {type(chunk)}")

        if isinstance(chunk, dict):
            print(f"  Keys: {list(chunk.keys())}")
            print(f"  Response: '{chunk.get('response', '')[:50]}...'")
            print(f"  Tool calls: {chunk.get('tool_calls', [])}")

            if chunk.get("tool_calls"):
                tool_calls_found.extend(chunk["tool_calls"])
        else:
            print(f"  String chunk: '{str(chunk)[:50]}...'")

    print("\n📊 Summary:")
    print(f"  Total chunks: {chunk_count}")
    print(f"  Tool calls found: {len(tool_calls_found)}")
    if tool_calls_found:
        print(f"  Functions: {[tc['function']['name'] for tc in tool_calls_found]}")


async def debug_extraction_logic():
    """Step 4: Test the extraction logic directly"""
    print("\n" + "=" * 60)
    print("STEP 4: Direct Extraction Logic Test")
    print("=" * 60)

    import ollama

    from chuk_llm.llm.providers.ollama_client import OllamaLLMClient

    # Create a client to test extraction methods
    OllamaLLMClient("granite3.3:latest")

    # Get a real chunk from Ollama
    model_name = "granite3.3:latest"
    messages = [{"role": "user", "content": "What's the weather in San Francisco?"}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            },
        }
    ]

    ollama_client = ollama.AsyncClient()
    stream = await ollama_client.chat(
        model=model_name,
        messages=messages,
        tools=tools,
        stream=True,
        options={"temperature": 0.1, "num_predict": 200},
    )

    # Get first chunk (which should have tool calls)
    first_chunk = None
    async for chunk in stream:
        first_chunk = chunk
        break  # Just get the first one

    if first_chunk:
        print("Testing extraction on first chunk:")
        print(f"  Chunk type: {type(first_chunk)}")

        if hasattr(first_chunk, "message") and first_chunk.message:
            msg = first_chunk.message

            # Test different extraction methods
            print("\n  Method 1: Direct attribute access")
            tool_calls = getattr(msg, "tool_calls", None)
            print(f"    tool_calls: {tool_calls}")

            print("\n  Method 2: Check with hasattr first")
            if hasattr(msg, "tool_calls"):
                print("    Has tool_calls: True")
                print(f"    Value: {msg.tool_calls}")
            else:
                print("    Has tool_calls: False")

            print("\n  Method 3: Try to iterate if exists")
            if tool_calls:
                for i, tc in enumerate(tool_calls):
                    print(f"    Tool call {i}: {tc}")
                    if hasattr(tc, "function"):
                        print(f"      Function: {tc.function}")


async def main():
    """Run all debug steps"""
    print("🔍 DEEP DEBUG: GRANITE TOOL CALL ISSUE")
    print("=" * 60)
    print("Finding where tool calls are lost in the pipeline")

    try:
        # Step 1: Raw Ollama
        await debug_raw_ollama()

        # Step 2: OllamaLLMClient
        await debug_ollama_client()

        # Step 3: High-level stream
        await debug_stream_function()

        # Step 4: Extraction logic
        await debug_extraction_logic()

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 60)
    print("🔍 DEBUG COMPLETE")
    print("=" * 60)
    print("\nKey Questions:")
    print("1. Does raw Ollama show tool_calls in chunk 1? (Should be YES)")
    print("2. Does OllamaLLMClient yield tool_calls? (Currently NO)")
    print("3. Does the stream function pass them through? (Depends on #2)")
    print("4. Is the extraction logic working? (Test directly)")


if __name__ == "__main__":
    asyncio.run(main())
