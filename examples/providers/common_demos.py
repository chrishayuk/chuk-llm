#!/usr/bin/env python3
"""
Common Demo Functions for All Providers
========================================

Standardized demo functions that work across all LLM providers.
Each demo checks provider capabilities before running.
"""

import json
import time
from typing import Any

from chuk_llm.configuration import Feature, get_config
from chuk_llm.core.models import Message, Tool, ToolFunction, TextContent, ImageUrlContent
from chuk_llm.core.enums import MessageRole, ContentType, ToolType


# =============================================================================
# Demo 1: Basic Completion
# =============================================================================

async def demo_basic_completion(client, provider: str, model: str):
    """Basic text completion - works with all providers"""
    print(f"\n{'='*70}")
    print("Demo 1: Basic Completion")
    print(f"{'='*70}")

    messages = [
        Message(
            role=MessageRole.SYSTEM,
            content="You are a helpful AI assistant."
        ),
        Message(
            role=MessageRole.USER,
            content="Explain quantum computing in 2 sentences."
        ),
    ]

    print(f"Provider: {provider}")
    print(f"Model: {model}")
    print(f"User: {messages[1].content}")
    print("\n⏳ Generating response...")

    start_time = time.time()
    response = await client.create_completion(messages, max_tokens=150)
    duration = time.time() - start_time

    print(f"\n✅ Response ({duration:.2f}s):")
    print(f"{response['response']}")

    return response


# =============================================================================
# Demo 2: Streaming
# =============================================================================

async def demo_streaming(client, provider: str, model: str):
    """Streaming response - works with all providers"""
    print(f"\n{'='*70}")
    print("Demo 2: Streaming")
    print(f"{'='*70}")


    messages = [
        Message(
            role=MessageRole.USER,
            content="Write a haiku about artificial intelligence."
        ),
    ]

    print(f"Provider: {provider}")
    print(f"Model: {model}")
    print(f"User: {messages[0].content}")
    print("\n⏳ Streaming response:")
    print("-" * 70)

    full_response = ""
    chunk_count = 0
    start_time = time.time()

    async for chunk in client.create_completion(messages, stream=True, max_tokens=100):
        if chunk.get("response"):
            content = chunk["response"]
            print(content, end="", flush=True)
            full_response += content
            chunk_count += 1

    duration = time.time() - start_time

    print("\n" + "-" * 70)
    print(f"✅ Streamed {chunk_count} chunks in {duration:.2f}s")

    return full_response


# =============================================================================
# Demo 3: Function Calling
# =============================================================================

async def demo_function_calling(client, provider: str, model: str):
    """Function calling with tools - works with providers that support it"""
    print(f"\n{'='*70}")
    print("Demo 3: Function Calling")
    print(f"{'='*70}")

    tools = [
        Tool(
            type=ToolType.FUNCTION,
            function=ToolFunction(
                name="get_weather",
                description="Get current weather for a location",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City and state, e.g. San Francisco, CA",
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                        },
                    },
                    "required": ["location"],
                },
            ),
        ),
        Tool(
            type=ToolType.FUNCTION,
            function=ToolFunction(
                name="calculate",
                description="Perform mathematical calculations",
                parameters={
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Math expression like '2 + 2'",
                        }
                    },
                    "required": ["expression"],
                },
            ),
        ),
    ]

    messages = [
        Message(
            role=MessageRole.USER,
            content="What's the weather in Tokyo and what is 25 * 8?"
        ),
    ]

    print(f"Provider: {provider}")
    print(f"Model: {model}")
    print(f"Tools: {len(tools)}")
    for tool in tools:
        print(f"  - {tool.function.name}")
    print(f"\nUser: {messages[0].content}")
    print("\n⏳ Processing with tools...")

    # Increase max_tokens for tool calling to avoid truncated tool calls
    # WatsonX needs higher limit when using modern chat endpoint
    response = await client.create_completion(messages, tools=tools, max_tokens=2000)

    # Handle multiple rounds of tool calling (some models call tools sequentially)
    max_rounds = 5
    round_num = 0

    while response.get("tool_calls") and round_num < max_rounds:
        round_num += 1
        tool_calls = response["tool_calls"]

        if round_num == 1:
            print(f"\n🔧 Model called {len(tool_calls)} function(s):")
        else:
            print(f"\n🔧 Round {round_num}: Model called {len(tool_calls)} more function(s):")

        tool_results = []
        for tool_call in tool_calls:
            func_name = tool_call["function"]["name"]
            func_args = json.loads(tool_call["function"]["arguments"])

            print(f"\n  📞 {func_name}({json.dumps(func_args)})")

            # Simulate function execution
            if func_name == "get_weather":
                result = json.dumps({"temperature": 22, "conditions": "partly cloudy"})
            elif func_name == "calculate":
                try:
                    result = json.dumps({"result": eval(func_args.get("expression", "0"))})
                except:
                    result = json.dumps({"error": "Invalid expression"})
            else:
                result = json.dumps({"error": "Unknown function"})

            print(f"     Result: {result}")

            tool_results.append({
                "tool_call_id": tool_call["id"],
                "name": func_name,
                "result": result
            })

        # Continue conversation with results
        messages.append(
            Message(
                role=MessageRole.ASSISTANT,
                content=response.get("response") or "",
                tool_calls=response.get("tool_calls")
            )
        )

        for tool_result in tool_results:
            messages.append(
                Message(
                    role=MessageRole.TOOL,
                    content=tool_result["result"],
                    tool_call_id=tool_result["tool_call_id"],
                    name=tool_result["name"]
                )
            )

        print(f"\n⏳ Getting {'final ' if round_num > 0 else ''}response...")
        response = await client.create_completion(messages, tools=tools)

    # Now we have the final response (no more tool calls)
    if round_num > 0:
        print(f"\n✅ Final Response:")
        print(f"{response['response']}")
        return response
    else:
        print(f"\n✅ Response (no function calls):")
        print(f"{response['response']}")
        return response


# =============================================================================
# Demo 4: Vision
# =============================================================================

async def demo_vision(client, provider: str, model: str):
    """Vision/multimodal - works with providers that support it"""
    print(f"\n{'='*70}")
    print("Demo 4: Vision/Multimodal")
    print(f"{'='*70}")

    # Simple 16x16 red square image (properly encoded PNG)
    red_square_base64 = "iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAAIElEQVR4nGP8z0AaYCJRPcOoBmIAE1GqkMCoBmIAyRoAQC4BH1m1rqAAAAAASUVORK5CYII="

    messages = [
        Message(
            role=MessageRole.USER,
            content=[
                TextContent(
                    type=ContentType.TEXT,
                    text="What color is this image?"
                ),
                ImageUrlContent(
                    type=ContentType.IMAGE_URL,
                    image_url={"url": f"data:image/png;base64,{red_square_base64}"}
                ),
            ],
        )
    ]

    print(f"Provider: {provider}")
    print(f"Model: {model}")
    print(f"User: What color is this image?")
    print(f"Image: [16x16 red square]")
    print("\n⏳ Analyzing image...")

    response = await client.create_completion(messages, max_tokens=100)

    print(f"\n✅ Response:")
    print(f"{response['response']}")

    return response


# =============================================================================
# Demo 4b: Vision with URL
# =============================================================================

async def demo_vision_url(client, provider: str, model: str):
    """Vision with public URL - providers handle URL fetching automatically"""
    print(f"\n{'='*70}")
    print("Demo 4b: Vision with Public URL")
    print(f"{'='*70}")


    # Use a stable, public domain image - Unsplash logo (public API)
    # This is a direct image link that allows programmatic access
    image_url = "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=400&h=300&fit=crop"

    print(f"Provider: {provider}")
    print(f"Model: {model}")
    print(f"User: What is in this image? Describe it briefly.")
    print(f"Image URL: {image_url}")
    print("\n⏳ Analyzing image from URL...")

    messages = [
        Message(
            role=MessageRole.USER,
            content=[
                TextContent(
                    type=ContentType.TEXT,
                    text="What is in this image? Describe it briefly."
                ),
                ImageUrlContent(
                    type=ContentType.IMAGE_URL,
                    image_url={"url": image_url}
                ),
            ],
        )
    ]

    try:
        response = await client.create_completion(messages, max_tokens=150)

        print(f"\n✅ Response:")
        print(f"{response['response']}")

        return response

    except Exception as e:
        print(f"\n❌ Error processing image URL: {e}")
        import traceback
        traceback.print_exc()
        return None


# =============================================================================
# Demo 5: JSON Mode
# =============================================================================

async def demo_json_mode(client, provider: str, model: str):
    """JSON mode - works with providers that support it"""
    print(f"\n{'='*70}")
    print("Demo 5: JSON Mode")
    print(f"{'='*70}")


    messages = [
        Message(
            role=MessageRole.SYSTEM,
            content="You are a helpful assistant that outputs JSON."
        ),
        Message(
            role=MessageRole.USER,
            content="Create a JSON for a person named Alice, age 30, hobbies: reading, coding."
        ),
    ]

    print(f"Provider: {provider}")
    print(f"Model: {model}")
    print(f"User: Create a JSON for Alice...")
    print("\n⏳ Generating JSON...")

    response = await client.create_completion(
        messages,
        response_format={"type": "json_object"},
        max_tokens=200
    )

    print(f"\n✅ JSON Response:")
    try:
        parsed = json.loads(response['response'])
        print(json.dumps(parsed, indent=2))
    except:
        print(response['response'])

    return response


# =============================================================================
# Demo 6: Reasoning with Thinking Visibility
# =============================================================================

async def demo_reasoning(client, provider: str, model: str):
    """Reasoning with extended thinking - shows thinking process"""
    print(f"\n{'='*70}")
    print("Demo 6: Reasoning with Thinking Visibility")
    print(f"{'='*70}")

    # Check if this is a reasoning model
    model_lower = model.lower()
    is_reasoning = any(pattern in model_lower for pattern in [
        "o1", "o3", "o4", "reasoning", "r1", "deepseek-r", "gpt-oss", "gpt-5", "thinking"
    ])

    if not is_reasoning:
        print(f"ℹ️  Model {model} is not a known reasoning model")
        print(f"   Running basic reasoning test anyway...")

    messages = [
        Message(
            role=MessageRole.USER,
            content="A farmer has 17 sheep. All but 9 die. How many are left? Think step by step."
        ),
    ]

    print(f"Provider: {provider}")
    print(f"Model: {model}")
    print(f"Is reasoning model: {'Yes' if is_reasoning else 'No'}")
    print(f"\nUser: {messages[0].content}")
    print("\n⏳ Processing (reasoning models think before responding)...")

    start_time = time.time()
    response = await client.create_completion(messages, max_tokens=500)
    duration = time.time() - start_time

    print(f"\n✅ Response ({duration:.2f}s):")
    print(f"{response['response']}")

    # Check for reasoning metadata
    if response.get("reasoning") or response.get("thinking"):
        print(f"\n🧠 Reasoning Metadata Found:")

        reasoning_info = response.get("reasoning", {})
        thinking = response.get("thinking") or reasoning_info.get("thinking")

        if thinking:
            print(f"\n💭 Extended Thinking Process:")
            print("-" * 70)
            # Show first 500 chars of thinking
            thinking_preview = thinking[:500]
            print(thinking_preview)
            if len(thinking) > 500:
                print(f"... ({len(thinking) - 500} more characters)")
            print("-" * 70)

        if reasoning_info.get("model_type"):
            print(f"\n   Model type: {reasoning_info['model_type']}")

        # Check for thinking tokens (different providers use different field names)
        thinking_tokens = reasoning_info.get("tokens_used") or reasoning_info.get("thinking_tokens")
        if thinking_tokens:
            print(f"   Thinking tokens: {thinking_tokens}")

        # Also show usage info if available
        if response.get("usage"):
            usage = response["usage"]
            if usage.get("reasoning_tokens"):
                print(f"   Reasoning tokens (from usage): {usage['reasoning_tokens']}")
            print(f"   Total tokens: {usage.get('total_tokens', 'N/A')}")
    else:
        if is_reasoning:
            print(f"\nℹ️  No reasoning metadata in response")
            print(f"   (Some providers don't expose thinking process)")

    return response


# =============================================================================
# Demo 7: Structured Outputs (JSON Schema)
# =============================================================================

async def demo_structured_outputs(client, provider: str, model: str):
    """Structured outputs with JSON Schema - ensures guaranteed format"""
    print(f"\n{'='*70}")
    print("Demo 7: Structured Outputs (JSON Schema)")
    print(f"{'='*70}")


    # Define a JSON schema for structured output
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "email": {"type": "string"},
            "skills": {
                "type": "array",
                "items": {"type": "string"}
            },
            "experience_years": {"type": "integer"}
        },
        "required": ["name", "age", "email", "skills"],
        "additionalProperties": False
    }

    messages = [
        Message(
            role=MessageRole.SYSTEM,
            content="You are a helpful assistant that outputs JSON. IMPORTANT: Only output the exact fields requested. Do not add any additional fields."
        ),
        Message(
            role=MessageRole.USER,
            content="""Generate a JSON object with ONLY these 5 fields:
- name: "Sarah"
- age: 28
- email: "sarah@example.com"
- skills: ["Python", "JavaScript"]
- experience_years: 5

Output ONLY these fields, no additional properties."""
        ),
    ]

    print(f"Provider: {provider}")
    print(f"Model: {model}")
    print(f"User: Generate engineer profile...")
    print(f"\n📋 Required Schema:")
    print(json.dumps(schema, indent=2))
    print("\n⏳ Generating structured output...")

    # Use JSON mode
    response = await client.create_completion(
        messages,
        response_format={"type": "json_object"},
        max_tokens=150
    )

    print(f"\n✅ Structured JSON Response:")
    try:
        # Extract JSON from response, handling markdown code fences
        response_text = response['response'].strip()

        # Remove markdown code fences if present
        if response_text.startswith("```"):
            # Find the start and end of the JSON content
            lines = response_text.split('\n')
            # Remove first line (```json or ```)
            lines = lines[1:]
            # Remove last line if it's just ```
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            response_text = '\n'.join(lines)

        parsed = json.loads(response_text)
        print(json.dumps(parsed, indent=2))

        # Validate against schema (basic check)
        required_fields = schema.get("required", [])
        missing_fields = [f for f in required_fields if f not in parsed]
        if missing_fields:
            print(f"\n⚠️  Missing required fields: {missing_fields}")
        else:
            print(f"\n✅ All required fields present")

    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON: {e}")
        print(response['response'])

    return response


# =============================================================================
# Demo 8: Multi-turn Conversation
# =============================================================================

async def demo_conversation(client, provider: str, model: str):
    """Multi-turn conversation with context memory"""
    print(f"\n{'='*70}")
    print("Demo 8: Multi-turn Conversation")
    print(f"{'='*70}")

    conversation = [
        Message(
            role=MessageRole.SYSTEM,
            content="You are a helpful assistant with good memory."
        ),
    ]

    # Turn 1
    conversation.append(Message(role=MessageRole.USER, content="My name is Bob."))
    response1 = await client.create_completion(conversation)

    print(f"Provider: {provider}")
    print(f"Model: {model}")
    print(f"\nTurn 1:")
    print(f"  User: My name is Bob.")
    print(f"  AI: {response1['response']}")

    # Turn 2
    conversation.append(Message(role=MessageRole.ASSISTANT, content=response1['response']))
    conversation.append(Message(role=MessageRole.USER, content="What's my name?"))
    response2 = await client.create_completion(conversation)

    print(f"\nTurn 2:")
    print(f"  User: What's my name?")
    print(f"  AI: {response2['response']}")

    print(f"\n✅ Context maintained across turns")

    return response2


# =============================================================================
# Demo 9: Model Discovery
# =============================================================================

async def demo_model_discovery(client, provider: str, model: str):
    """Discover available models from the provider"""
    print(f"\n{'='*70}")
    print("Demo 9: Model Discovery")
    print(f"{'='*70}")

    try:
        from chuk_llm.api.discovery import discover_models

        print(f"⏳ Discovering models from {provider}...")

        models = await discover_models(provider, force_refresh=True)

        print(f"\n✅ Found {len(models)} models\n")

        # Show first 10 models
        print("Available models:")
        for model_info in models[:10]:
            model_id = model_info.get("name", model_info.get("id", "unknown"))
            print(f"  • {model_id}")

        if len(models) > 10:
            print(f"  ... and {len(models) - 10} more")

        return models
    except Exception as e:
        print(f"❌ Error discovering models: {e}")
        print(f"   This is normal if the provider doesn't support discovery")
        return None


# =============================================================================
# Demo 10: Audio Input (Multimodal)
# =============================================================================

async def demo_audio_input(client, provider: str, model: str):
    """Audio input - works with providers that support it"""
    print(f"\n{'='*70}")
    print("Demo 10: Audio Input (Multimodal)")
    print(f"{'='*70}")


    print(f"Provider: {provider}")
    print(f"Model: {model}")
    print(f"User: Listen to this audio and describe what you hear")

    # Generate a minimal WAV file with synthesized audio (440Hz sine wave = musical note A)
    import wave
    import struct
    import math
    import tempfile
    import os
    import base64

    # Generate 1 second of 440Hz sine wave (musical note A)
    sample_rate = 16000  # 16kHz
    duration = 1.0
    frequency = 440.0

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        wav_file = f.name

    with wave.open(wav_file, 'w') as wav:
        wav.setnchannels(1)  # Mono
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(sample_rate)

        for i in range(int(sample_rate * duration)):
            value = int(32767 * math.sin(2 * math.pi * frequency * i / sample_rate))
            wav.writeframes(struct.pack('<h', value))

    # Read and encode the audio
    with open(wav_file, 'rb') as audio_file:
        audio_data = base64.b64encode(audio_file.read()).decode('utf-8')

    # Clean up temp file
    os.unlink(wav_file)

    try:
        from chuk_llm.core.models import InputAudioContent

        messages = [
            Message(
                role=MessageRole.USER,
                content=[
                    TextContent(type=ContentType.TEXT, text="What do you hear in this audio? It's a musical note."),
                    InputAudioContent(
                        type=ContentType.INPUT_AUDIO,
                        input_audio={
                            "data": audio_data,
                            "format": "wav"
                        }
                    )
                ]
            )
        ]

        print("Audio: [1 second of 440Hz sine wave - musical note A]")
        print("\n⏳ Analyzing audio...")

        start = time.time()
        # Only pass model parameter if it differs from client's model (avoid duplicate parameter)
        if hasattr(client, 'model') and client.model == model:
            response = await client.create_completion(messages, stream=False)
        else:
            response = await client.create_completion(messages, stream=False, model=model)
        elapsed = time.time() - start

        print(f"\n✅ Response ({elapsed:.2f}s):")
        print(response['response'])

        return response

    except ImportError:
        print("⚠️  InputAudioContent not available in this version")
        return None
    except Exception as e:
        print(f"⚠️  Audio demo failed: {e}")
        print("    Audio models may require special access or different API format")
        return None


# =============================================================================
# Demo 11: Temperature and Sampling Parameters
# =============================================================================

async def demo_parameters(client, provider: str, model: str):
    """Temperature and sampling parameters demonstration"""
    print(f"\n{'='*70}")
    print("Demo 11: Temperature and Sampling Parameters")
    print(f"{'='*70}")

    print(f"Provider: {provider}")
    print(f"Model: {model}")
    print(f"Prompt: Write the first sentence of a story about a robot.\n")

    temperatures = [0.0, 0.5, 1.0, 1.5]

    for temp in temperatures:
        messages = [
            Message(
                role=MessageRole.USER,
                content="Write the first sentence of a story about a robot."
            ),
        ]

        try:
            response = await client.create_completion(
                messages,
                temperature=temp,
                max_tokens=100
            )

            print(f"Temperature {temp}:")
            print(f"  {response['response']}\n")

        except Exception as e:
            print(f"Temperature {temp}: ⚠️  Error - {e}\n")

    print("✅ Lower temperature = more deterministic")
    print("   Higher temperature = more creative/random")

    return None


# =============================================================================
# Demo 12: Model Comparison
# =============================================================================

async def demo_model_comparison(provider: str, models: list[str]):
    """Compare multiple models side by side"""
    print(f"\n{'='*70}")
    print("Demo 12: Model Comparison")
    print(f"{'='*70}")

    from chuk_llm.llm.client import get_client

    prompt = "Explain quantum entanglement in one sentence."

    print(f"Provider: {provider}")
    print(f"Comparing {len(models)} models")
    print(f"Prompt: {prompt}\n")

    for model in models:
        try:
            client = get_client(provider, model=model)
            messages = [
                Message(
                    role=MessageRole.USER,
                    content=prompt
                )
            ]

            start_time = time.time()
            response = await client.create_completion(messages, max_tokens=100)
            duration = time.time() - start_time

            print(f"{model} ({duration:.2f}s):")
            print(f"  {response['response']}\n")

        except Exception as e:
            print(f"{model}: ❌ Error - {e}\n")

    return None


# =============================================================================
# Demo 13: Dynamic Model Call (Discovery + Usage)
# =============================================================================

async def demo_dynamic_model_call(provider: str):
    """Discover models and call one of them dynamically"""
    print(f"\n{'='*70}")
    print("Demo 13: Call Dynamically Discovered Model")
    print(f"{'='*70}")

    try:
        from chuk_llm.api.discovery import discover_models
        from chuk_llm.llm.client import get_client

        print(f"⏳ Step 1: Discovering available models from {provider}...")

        models = await discover_models(provider, force_refresh=True)

        if not models:
            print(f"❌ No models found for provider {provider}")
            return None

        # Find a suitable chat model (skip TTS/audio/image models)
        target_model = None
        skip_patterns = ["tts", "whisper", "audio", "realtime", "dall-e", "sora", "embedding", "moderation"]

        for model in models:
            model_id = model.get("name", model.get("id", ""))
            if any(x in model_id.lower() for x in skip_patterns):
                continue
            target_model = model_id
            break

        if not target_model:
            print(f"❌ No suitable chat model found")
            return None

        print(f"✅ Found {len(models)} models")
        print(f"🎯 Selected model: {target_model}")

        print(f"\n⏳ Step 2: Creating client for discovered model...")

        client = get_client(provider, model=target_model)

        print(f"✅ Client created")

        print(f"\n⏳ Step 3: Making request to dynamically discovered model...")

        messages = [
            Message(
                role=MessageRole.USER,
                content="In one sentence, what makes you special?"
            )
        ]

        response = await client.create_completion(messages, max_tokens=100)

        print(f"\n✅ Response from {target_model}:")
        print(f"{response['response']}")

        print(f"\n💡 Dynamic Model Flow:")
        print(f"   1. Discovered {len(models)} models from API")
        print(f"   2. Selected '{target_model}' programmatically")
        print(f"   3. Created client and made successful request")
        print(f"   ✅ No hardcoded model names needed!")

        return response
    except Exception as e:
        print(f"❌ Error in dynamic model call: {e}")
        import traceback
        traceback.print_exc()
        return None


# =============================================================================
# Demo 14: Error Handling
# =============================================================================

async def demo_error_handling(client, provider: str, model: str):
    """Demonstrate error handling"""
    print(f"\n{'='*70}")
    print("Demo 14: Error Handling")
    print(f"{'='*70}")

    # Test 1: Invalid max_tokens
    print("\nTest 1: Handling invalid parameters")
    try:
        messages = [Message(role=MessageRole.USER, content="Hello")]
        response = await client.create_completion(messages, max_tokens=-1)
        print("  ✅ No exception raised (provider may have handled it)")
    except Exception as e:
        print(f"  ✅ Exception raised and caught: {type(e).__name__}")

    # Test 2: Empty message
    print("\nTest 2: Handling empty messages")
    try:
        messages = []
        response = await client.create_completion(messages)
        print("  ✅ No exception raised (provider may have handled it)")
    except Exception as e:
        print(f"  ✅ Exception raised and caught: {type(e).__name__}")

    # Test 3: Very long input (might hit context limit)
    print("\nTest 3: Handling context limits")
    try:
        long_text = "Hello " * 50000  # Very long text
        messages = [Message(role=MessageRole.USER, content=long_text)]
        response = await client.create_completion(messages, max_tokens=10)
        print("  ✅ Provider handled very long input")
    except Exception as e:
        print(f"  ✅ Exception raised and caught: {type(e).__name__}")

    print("\n✅ Error handling demonstration complete")
    return None


# =============================================================================
# Demo Runner Helper
# =============================================================================

async def run_all_demos(client, provider: str, model: str, skip_tools=False, skip_vision=False, skip_discovery=False, skip_audio=False, skip_parameters=False, skip_comparison=False, comparison_models=None, audio_model=None, vision_client=None, vision_model=None):
    """Run all demos for a provider"""
    demos = [
        ("Basic Completion", demo_basic_completion(client, provider, model)),
        ("Streaming", demo_streaming(client, provider, model)),
    ]

    if not skip_tools:
        demos.append(("Function Calling", demo_function_calling(client, provider, model)))

    if not skip_vision:
        # Use custom vision client if provided (e.g., granite-vision for WatsonX)
        vision_client_to_use = vision_client if vision_client else client
        vision_model_to_use = vision_model if vision_model else model
        demos.append(("Vision", demo_vision(vision_client_to_use, provider, vision_model_to_use)))
        demos.append(("Vision with URL", demo_vision_url(vision_client_to_use, provider, vision_model_to_use)))

    demos.extend([
        ("JSON Mode", demo_json_mode(client, provider, model)),
        ("Reasoning", demo_reasoning(client, provider, model)),
        ("Structured Outputs", demo_structured_outputs(client, provider, model)),
        ("Conversation", demo_conversation(client, provider, model)),
    ])

    if not skip_discovery:
        demos.append(("Model Discovery", demo_model_discovery(client, provider, model)))

    # Audio input demo (may not be supported by all providers/models)
    if not skip_audio:
        # Use custom audio model if provided (e.g., voxtral for Mistral), same client
        audio_model_to_use = audio_model if audio_model else model
        demos.append(("Audio Input", demo_audio_input(client, provider, audio_model_to_use)))

    # Temperature/parameters demo
    if not skip_parameters:
        demos.append(("Parameters", demo_parameters(client, provider, model)))

    # Model comparison (if models provided)
    if not skip_comparison and comparison_models:
        demos.append(("Model Comparison", demo_model_comparison(provider, comparison_models)))

    # Dynamic model call
    if not skip_discovery:
        demos.append(("Dynamic Model Call", demo_dynamic_model_call(provider)))

    demos.append(("Error Handling", demo_error_handling(client, provider, model)))

    for name, demo_coro in demos:
        try:
            await demo_coro
        except Exception as e:
            print(f"\n❌ Error in {name}: {e}")
            import traceback
            traceback.print_exc()

    return True
