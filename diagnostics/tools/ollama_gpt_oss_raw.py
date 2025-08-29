#!/usr/bin/env python3
"""
GPT-OSS Raw Response Debug
==========================

Debug what GPT-OSS actually returns for streaming and non-streaming.
"""

import asyncio
import json
import time

import httpx


async def debug_gpt_oss_responses():
    """Debug GPT-OSS raw responses"""

    print("🔍 GPT-OSS RAW RESPONSE DEBUG")
    print("=" * 50)

    model_name = "gpt-oss:latest"

    tools = [
        {
            "type": "function",
            "function": {
                "name": "execute_sql",
                "description": "Execute a SQL query",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "SQL query to execute",
                        }
                    },
                    "required": ["query"],
                },
            },
        }
    ]

    prompt = "Execute this SQL: SELECT name, email FROM users LIMIT 5"

    # Test 1: Streaming Response
    print("\n🧪 TEST 1: STREAMING RESPONSE")
    print("-" * 30)

    payload_streaming = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "tools": tools,
        "stream": True,
        "options": {"num_predict": 100, "temperature": 0.1},
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            print("📡 Making streaming request...")
            start_time = time.time()

            async with client.stream(
                "POST", "http://localhost:11434/api/chat", json=payload_streaming
            ) as response:
                response.raise_for_status()

                chunk_count = 0
                all_chunks = []

                print("📥 Receiving streaming chunks:")

                async for line in response.aiter_lines():
                    if line.strip():
                        chunk_count += 1
                        try:
                            chunk_data = json.loads(line)
                            all_chunks.append(chunk_data)

                            print(f"  Chunk {chunk_count}:")
                            print(f"    Raw: {json.dumps(chunk_data, indent=2)}")

                            # Extract message info
                            if "message" in chunk_data:
                                message = chunk_data["message"]
                                content = message.get("content", "")
                                tool_calls = message.get("tool_calls", [])

                                if content:
                                    print(f"    Content: '{content}'")
                                if tool_calls:
                                    print(f"    Tool Calls: {len(tool_calls)} detected")
                                    for tc in tool_calls:
                                        func_name = tc.get("function", {}).get(
                                            "name", "unknown"
                                        )
                                        print(f"      - {func_name}")

                            print()

                        except json.JSONDecodeError as e:
                            print(f"  Chunk {chunk_count}: JSON decode error: {e}")
                            print(f"    Raw line: {line}")

                end_time = time.time()
                print("📊 Streaming Summary:")
                print(f"    Total chunks: {chunk_count}")
                print(f"    Response time: {end_time - start_time:.2f}s")
                print(
                    f"    Average chunk time: {(end_time - start_time) / max(chunk_count, 1):.3f}s"
                )

    except Exception as e:
        print(f"❌ Streaming test failed: {e}")

    # Test 2: Non-Streaming Response
    print("\n🧪 TEST 2: NON-STREAMING RESPONSE")
    print("-" * 35)

    payload_non_streaming = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "tools": tools,
        "stream": False,
        "options": {"num_predict": 100, "temperature": 0.1},
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            print("📡 Making non-streaming request...")
            start_time = time.time()

            response = await client.post(
                "http://localhost:11434/api/chat", json=payload_non_streaming
            )
            response.raise_for_status()

            end_time = time.time()
            data = response.json()

            print("📥 Non-streaming response:")
            print(f"    Raw: {json.dumps(data, indent=2)}")

            if "message" in data:
                message = data["message"]
                content = message.get("content", "")
                tool_calls = message.get("tool_calls", [])

                print("📊 Non-streaming Summary:")
                print(f"    Content: '{content}'")
                print(f"    Tool Calls: {len(tool_calls)} detected")

                if tool_calls:
                    for i, tc in enumerate(tool_calls):
                        func_name = tc.get("function", {}).get("name", "unknown")
                        args = tc.get("function", {}).get("arguments", {})
                        print(f"      Tool {i + 1}: {func_name}")
                        print(f"        Args: {args}")

                print(f"    Response time: {end_time - start_time:.2f}s")

    except Exception as e:
        print(f"❌ Non-streaming test failed: {e}")

    # Test 3: Check if model is actually running
    print("\n🧪 TEST 3: MODEL STATUS CHECK")
    print("-" * 30)

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Check running models
            ps_response = await client.get("http://localhost:11434/api/ps")
            ps_data = ps_response.json()

            print("🏃 Currently running models:")
            for model in ps_data.get("models", []):
                model_name_running = model.get("name", "unknown")
                size_vram = model.get("size_vram", 0)
                print(f"    {model_name_running} - VRAM: {size_vram / (1024**3):.1f}GB")

            # Check all models
            tags_response = await client.get("http://localhost:11434/api/tags")
            tags_data = tags_response.json()

            print(f"\n📚 Available models: {len(tags_data.get('models', []))}")
            for model in tags_data.get("models", []):
                if "gpt-oss" in model.get("name", ""):
                    print(
                        f"    Found: {model.get('name')} - Size: {model.get('size', 0) / (1024**3):.1f}GB"
                    )

    except Exception as e:
        print(f"❌ Model status check failed: {e}")

    print("\n🎯 DEBUGGING CONCLUSIONS:")
    print("1. Compare streaming vs non-streaming responses")
    print("2. Check if GPT-OSS uses different tool call formats")
    print("3. Verify if streaming actually works or if it's pseudo-streaming")
    print("4. Identify why ChukLLM gets different results than raw API")


async def main():
    await debug_gpt_oss_responses()


if __name__ == "__main__":
    asyncio.run(main())
