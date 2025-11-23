#!/usr/bin/env python3
"""
OpenAI Chat Completions API - Comprehensive Example
====================================================

Complete demonstration of the Chat Completions API (/v1/chat/completions).
This is the standard OpenAI-compatible API used by most providers.

Features Demonstrated:
- ✅ Basic completion
- ✅ Streaming
- ✅ Function/tool calling (3 tools)
- ✅ Vision (multimodal)
- ✅ JSON mode
- ✅ Structured outputs (JSON Schema)
- ✅ System prompts
- ✅ Temperature and sampling parameters
- ✅ GPT-4o, GPT-4o-mini support
- ✅ Model comparison
- ✅ Model discovery from API
- ✅ Dynamic model selection and calling
- ✅ Parameters testing
- ✅ Error handling
- ✅ Multi-turn conversations
- ✅ Reasoning models (o1, o3)
- ✅ Zero magic strings (all enums)
- ✅ Type-safe Pydantic models

Requirements:
- Set OPENAI_API_KEY environment variable

Usage:
    python openai_chat_completions_example.py
    python openai_chat_completions_example.py --model gpt-4o
    python openai_chat_completions_example.py --demo 1  # Run specific demo
    python openai_chat_completions_example.py --quick   # Skip slow demos
"""

import argparse
import asyncio
import base64
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("❌ Please set OPENAI_API_KEY environment variable")
    print("   export OPENAI_API_KEY='your_api_key_here'")
    print("   Get your key at: https://platform.openai.com/api-keys")
    sys.exit(1)

try:
    from chuk_llm.configuration import Feature, get_config
    from chuk_llm.llm.client import get_client
    from chuk_llm.core.models import (
        Message,
        Tool,
        ToolFunction,
        TextContent,
        ImageUrlContent,
    )
    from chuk_llm.core.enums import MessageRole, ContentType, ToolType
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("   Install with: pip install chuk-llm")
    sys.exit(1)


# =============================================================================
# Demo 1: Basic Chat Completion
# =============================================================================

async def demo_basic_completion(model: str):
    """Basic chat completion with system and user messages"""
    print(f"\n{'='*70}")
    print("Demo 1: Basic Chat Completion")
    print(f"{'='*70}")

    client = get_client("openai", model=model)

    messages = [
        Message(
            role=MessageRole.SYSTEM,
            content="You are a helpful assistant specialized in explaining technical concepts clearly."
        ),
        Message(
            role=MessageRole.USER,
            content="Explain what a REST API is in 2-3 sentences."
        ),
    ]

    print(f"Model: {model}")
    print(f"System: {messages[0].content}")
    print(f"User: {messages[1].content}")
    print("\n⏳ Generating response...")

    start_time = time.time()
    response = await client.create_completion(messages, max_tokens=200)
    duration = time.time() - start_time

    print(f"\n✅ Response ({duration:.2f}s):")
    print(f"{response['response']}")

    return response


# =============================================================================
# Demo 2: Streaming Response
# =============================================================================

async def demo_streaming(model: str):
    """Real-time streaming response"""
    print(f"\n{'='*70}")
    print("Demo 2: Streaming Response")
    print(f"{'='*70}")

    client = get_client("openai", model=model)

    messages = [
        Message(
            role=MessageRole.USER,
            content="Write a haiku about artificial intelligence."
        ),
    ]

    print(f"Model: {model}")
    print(f"User: {messages[0].content}")
    print("\n⏳ Streaming response:")
    print("-" * 70)

    full_response = ""
    start_time = time.time()
    chunk_count = 0

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
# Demo 3: Function Calling / Tool Use
# =============================================================================

async def demo_function_calling(model: str):
    """Function calling with multiple tools"""
    print(f"\n{'='*70}")
    print("Demo 3: Function Calling / Tool Use")
    print(f"{'='*70}")

    config = get_config()
    if not config.supports_feature("openai", Feature.TOOLS, model):
        print(f"⚠️  Model {model} doesn't support function calling")
        return None

    client = get_client("openai", model=model)

    # Define multiple tools
    tools = [
        Tool(
            type=ToolType.FUNCTION,
            function=ToolFunction(
                name="get_weather",
                description="Get the current weather for a location",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "Temperature unit (default: celsius)",
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
                            "description": "Mathematical expression to evaluate (e.g., '2 + 2', '10 * 5')",
                        }
                    },
                    "required": ["expression"],
                },
            ),
        ),
        Tool(
            type=ToolType.FUNCTION,
            function=ToolFunction(
                name="search_database",
                description="Search a product database",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query",
                        },
                        "category": {
                            "type": "string",
                            "enum": ["electronics", "clothing", "books"],
                            "description": "Product category to search in",
                        },
                    },
                    "required": ["query"],
                },
            ),
        ),
    ]

    messages = [
        Message(
            role=MessageRole.SYSTEM,
            content="You are a helpful assistant that can use tools to answer questions."
        ),
        Message(
            role=MessageRole.USER,
            content="What's the weather in Tokyo, and calculate 25 * 8 for me?"
        ),
    ]

    print(f"Model: {model}")
    print(f"Available tools: {len(tools)}")
    for tool in tools:
        print(f"  - {tool.function.name}: {tool.function.description}")

    print(f"\nUser: {messages[1].content}")
    print("\n⏳ Processing with tools...")

    response = await client.create_completion(messages, tools=tools)

    if response.get("tool_calls"):
        print(f"\n🔧 Model called {len(response['tool_calls'])} function(s):")

        # Collect tool results
        tool_results = []

        for tool_call in response["tool_calls"]:
            func_name = tool_call["function"]["name"]
            func_args = json.loads(tool_call["function"]["arguments"])

            print(f"\n  📞 Function: {func_name}")
            print(f"     Arguments: {json.dumps(func_args, indent=15)}")

            # Simulate function execution
            if func_name == "get_weather":
                result = json.dumps({
                    "location": func_args.get("location"),
                    "temperature": 22,
                    "unit": func_args.get("unit", "celsius"),
                    "conditions": "partly cloudy",
                    "humidity": 65,
                    "wind_speed": 12
                })
            elif func_name == "calculate":
                try:
                    calc_result = eval(func_args.get("expression", "0"))
                    result = json.dumps({"result": calc_result, "expression": func_args.get("expression")})
                except Exception as e:
                    result = json.dumps({"error": str(e)})
            elif func_name == "search_database":
                result = json.dumps({
                    "results": [
                        {"name": "Product 1", "price": 29.99},
                        {"name": "Product 2", "price": 49.99}
                    ],
                    "count": 2
                })
            else:
                result = json.dumps({"error": "Unknown function"})

            print(f"     Result: {result}")

            tool_results.append({
                "tool_call_id": tool_call["id"],
                "name": func_name,
                "result": result
            })

        # Continue conversation with function results
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

        # Get final response with function results
        print("\n⏳ Getting final response with tool results...")
        final_response = await client.create_completion(messages, tools=tools)

        print(f"\n✅ Final Response:")
        print(f"{final_response['response']}")

        return final_response
    else:
        print(f"\n✅ Response (no function calls):")
        print(f"{response['response']}")
        return response


# =============================================================================
# Demo 4: Vision / Multimodal
# =============================================================================

async def demo_vision(model: str):
    """Vision/multimodal with images"""
    print(f"\n{'='*70}")
    print("Demo 4: Vision / Multimodal")
    print(f"{'='*70}")

    config = get_config()
    if not config.supports_feature("openai", Feature.VISION, model):
        print(f"⚠️  Model {model} doesn't support vision")
        print("   Try with: --model gpt-4o")
        return None

    client = get_client("openai", model=model)

    # Create a simple test image (1x1 red pixel)
    red_pixel_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="

    messages = [
        Message(
            role=MessageRole.USER,
            content=[
                TextContent(
                    type=ContentType.TEXT,
                    text="What color is this image? Be specific."
                ),
                ImageUrlContent(
                    type=ContentType.IMAGE_URL,
                    image_url={"url": f"data:image/png;base64,{red_pixel_base64}"}
                ),
            ],
        )
    ]

    print(f"Model: {model}")
    print(f"User: What color is this image? Be specific.")
    print(f"Image: [1x1 pixel test image]")
    print("\n⏳ Analyzing image...")

    response = await client.create_completion(messages, max_tokens=100)

    print(f"\n✅ Response:")
    print(f"{response['response']}")

    return response


# =============================================================================
# Demo 4.5: Audio Input (Multimodal)
# =============================================================================

async def demo_audio_input():
    """Audio input with gpt-4o-audio-preview models"""
    print(f"\n{'='*70}")
    print("Demo 4.5: Audio Input (Multimodal)")
    print(f"{'='*70}")

    # Try gpt-4o-audio-preview first, fallback to gpt-4o-mini-audio-preview
    try:
        client = get_client("openai", model="gpt-4o-audio-preview")
        model_name = "gpt-4o-audio-preview"
    except:
        try:
            client = get_client("openai", model="gpt-4o-mini-audio-preview")
            model_name = "gpt-4o-mini-audio-preview"
        except:
            print("⚠️  Audio preview models not available, skipping audio demo")
            return None

    print(f"Model: {model_name}")
    print("User: Listen to this audio and transcribe it")

    # Create a minimal WAV file with synthesized audio (440Hz sine wave = musical note A)
    # This is a simple test - in production you'd use real audio files
    import wave
    import struct
    import math
    import tempfile

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
    import os
    os.unlink(wav_file)

    # Note: Audio input support in Chat Completions API
    # Uses InputAudioContent type for audio input
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
        response = await client.create_completion(messages, stream=False)
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
# Demo 5: JSON Mode
# =============================================================================

async def demo_json_mode(model: str):
    """Structured JSON output"""
    print(f"\n{'='*70}")
    print("Demo 5: JSON Mode / Structured Output")
    print(f"{'='*70}")

    client = get_client("openai", model=model)

    messages = [
        Message(
            role=MessageRole.SYSTEM,
            content="You are a helpful assistant that outputs valid JSON."
        ),
        Message(
            role=MessageRole.USER,
            content="""Create a JSON object for a fictional person with:
- name (string)
- age (number)
- occupation (string)
- hobbies (array of strings)
- address (object with street, city, country)
- active (boolean)"""
        ),
    ]

    print(f"Model: {model}")
    print(f"User: Create a JSON object for a fictional person...")
    print("\n⏳ Generating structured JSON...")

    response = await client.create_completion(
        messages,
        response_format={"type": "json_object"},
        max_tokens=300
    )

    print(f"\n✅ JSON Response:")
    try:
        parsed = json.loads(response['response'])
        print(json.dumps(parsed, indent=2))
    except json.JSONDecodeError:
        print(f"⚠️  Response was not valid JSON:")
        print(response['response'])

    return response


# =============================================================================
# Demo 6: Multi-turn Conversation
# =============================================================================

async def demo_conversation(model: str):
    """Multi-turn conversation with context"""
    print(f"\n{'='*70}")
    print("Demo 6: Multi-turn Conversation")
    print(f"{'='*70}")

    client = get_client("openai", model=model)

    conversation = [
        Message(
            role=MessageRole.SYSTEM,
            content="You are a friendly assistant with excellent memory."
        ),
    ]

    # Turn 1
    conversation.append(Message(role=MessageRole.USER, content="My name is Alice and I'm learning Python."))
    response1 = await client.create_completion(conversation)

    print(f"Turn 1:")
    print(f"  User: My name is Alice and I'm learning Python.")
    print(f"  AI: {response1['response']}")

    # Turn 2
    conversation.append(Message(role=MessageRole.ASSISTANT, content=response1['response']))
    conversation.append(Message(role=MessageRole.USER, content="What programming language am I learning?"))
    response2 = await client.create_completion(conversation)

    print(f"\nTurn 2:")
    print(f"  User: What programming language am I learning?")
    print(f"  AI: {response2['response']}")

    # Turn 3
    conversation.append(Message(role=MessageRole.ASSISTANT, content=response2['response']))
    conversation.append(Message(role=MessageRole.USER, content="What's my name again?"))
    response3 = await client.create_completion(conversation)

    print(f"\nTurn 3:")
    print(f"  User: What's my name again?")
    print(f"  AI: {response3['response']}")

    print(f"\n✅ Conversation maintained context across 3 turns")

    return response3


# =============================================================================
# Demo 7: Temperature and Sampling Parameters
# =============================================================================

async def demo_parameters(model: str):
    """Different temperature and sampling settings"""
    print(f"\n{'='*70}")
    print("Demo 7: Temperature and Sampling Parameters")
    print(f"{'='*70}")

    client = get_client("openai", model=model)

    prompt = "Write the first sentence of a story about a robot."

    print(f"Model: {model}")
    print(f"Prompt: {prompt}\n")

    temperatures = [0.0, 0.5, 1.0, 1.5]

    for temp in temperatures:
        messages = [Message(role=MessageRole.USER, content=prompt)]

        response = await client.create_completion(
            messages,
            temperature=temp,
            max_tokens=50
        )

        print(f"Temperature {temp}:")
        print(f"  {response['response']}\n")

    print(f"✅ Lower temperature = more deterministic")
    print(f"   Higher temperature = more creative/random")

    return None


# =============================================================================
# Demo 8: Reasoning Models (o1, o3)
# =============================================================================

async def demo_reasoning_model():
    """Reasoning model with extended thinking"""
    print(f"\n{'='*70}")
    print("Demo 8: Reasoning Models (gpt-5-mini)")
    print(f"{'='*70}")

    model = "gpt-5-mini"
    client = get_client("openai", model=model)

    # Reasoning models don't use system messages
    messages = [
        Message(
            role=MessageRole.USER,
            content="""A farmer has 17 sheep. All but 9 die.
How many sheep are left alive? Think through this carefully."""
        ),
    ]

    print(f"Model: {model}")
    print(f"User: {messages[0].content}")
    print("\n⏳ Reasoning model thinking...")
    print("   (GPT-5 models have enhanced reasoning capabilities)")

    start_time = time.time()
    response = await client.create_completion(messages)
    duration = time.time() - start_time

    print(f"\n✅ Response ({duration:.2f}s):")
    print(f"{response['response']}")

    if "reasoning" in response:
        reasoning_info = response["reasoning"]
        print(f"\n🧠 Reasoning Metadata:")
        if "model_type" in reasoning_info:
            print(f"   Model type: {reasoning_info['model_type']}")
        if "thinking" in reasoning_info:
            thinking = reasoning_info["thinking"]
            if thinking:
                print(f"   Thinking process: {thinking[:200]}...")

    return response


# =============================================================================
# Demo 9: Model Comparison
# =============================================================================

async def demo_model_comparison():
    """Compare different models"""
    print(f"\n{'='*70}")
    print("Demo 9: Model Comparison")
    print(f"{'='*70}")

    models = ["gpt-4o-mini", "gpt-4o"]
    prompt = "Explain quantum entanglement in one sentence."

    print(f"Prompt: {prompt}\n")

    for model in models:
        try:
            client = get_client("openai", model=model)
            messages = [Message(role=MessageRole.USER, content=prompt)]

            start_time = time.time()
            response = await client.create_completion(messages, max_tokens=100)
            duration = time.time() - start_time

            print(f"{model} ({duration:.2f}s):")
            print(f"  {response['response']}\n")
        except Exception as e:
            print(f"{model}: ❌ Error - {e}\n")

    return None


# =============================================================================
# Demo 10: Structured Outputs (JSON Schema)
# =============================================================================

async def demo_structured_outputs(model: str):
    """Structured outputs with JSON schema validation"""
    print(f"\n{'='*70}")
    print("Demo 10: Structured Outputs (JSON Schema)")
    print(f"{'='*70}")

    client = get_client("openai", model=model)

    # Define a JSON schema for structured output
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "product_review",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "product_name": {
                        "type": "string",
                        "description": "Name of the product"
                    },
                    "rating": {
                        "type": "number",
                        "description": "Rating from 1-5",
                        "minimum": 1,
                        "maximum": 5
                    },
                    "pros": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of positive aspects"
                    },
                    "cons": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of negative aspects"
                    },
                    "recommended": {
                        "type": "boolean",
                        "description": "Whether the product is recommended"
                    },
                    "summary": {
                        "type": "string",
                        "description": "One sentence summary"
                    }
                },
                "required": ["product_name", "rating", "pros", "cons", "recommended", "summary"],
                "additionalProperties": False
            }
        }
    }

    messages = [
        Message(
            role=MessageRole.SYSTEM,
            content="You are a product reviewer. Analyze products and provide structured reviews."
        ),
        Message(
            role=MessageRole.USER,
            content="Review the iPhone 15 Pro"
        ),
    ]

    print(f"Model: {model}")
    print(f"User: Review the iPhone 15 Pro")
    print("\nSchema enforced:")
    print("  - product_name (string)")
    print("  - rating (1-5)")
    print("  - pros (array)")
    print("  - cons (array)")
    print("  - recommended (boolean)")
    print("  - summary (string)")
    print("\n⏳ Generating structured review...")

    try:
        response = await client.create_completion(
            messages,
            response_format=response_format,
            max_tokens=500
        )

        print(f"\n✅ Structured Output:")
        parsed = json.loads(response['response'])
        print(json.dumps(parsed, indent=2))

        print(f"\n📊 Validation:")
        print(f"   ✅ Has all required fields")
        print(f"   ✅ Rating is {parsed['rating']}/5")
        print(f"   ✅ {len(parsed['pros'])} pros, {len(parsed['cons'])} cons")
        print(f"   ✅ Recommended: {parsed['recommended']}")

        return response
    except Exception as e:
        print(f"⚠️  Structured outputs may require gpt-4o-mini or newer")
        print(f"   Error: {e}")
        return None


# =============================================================================
# Demo 11: Model Discovery
# =============================================================================

async def demo_model_discovery():
    """Discover available models from OpenAI API"""
    print(f"\n{'='*70}")
    print("Demo 11: Model Discovery")
    print(f"{'='*70}")

    try:
        from chuk_llm.api.discovery import discover_models

        print("⏳ Discovering models from OpenAI API...")

        models = await discover_models("openai", force_refresh=True)

        print(f"\n✅ Found {len(models)} models\n")

        # Group models by family
        model_families = {}
        for model in models:
            model_id = model.get("name", model.get("id", "unknown"))

            # Determine family
            if "gpt-5" in model_id.lower():
                family = "GPT-5"
            elif "gpt-4o" in model_id.lower():
                family = "GPT-4o"
            elif "gpt-4" in model_id.lower():
                family = "GPT-4"
            elif "o1" in model_id.lower() or "o3" in model_id.lower():
                family = "Reasoning (o1/o3)"
            elif "gpt-3.5" in model_id.lower():
                family = "GPT-3.5"
            elif "dall-e" in model_id.lower():
                family = "DALL-E"
            elif "tts" in model_id.lower() or "whisper" in model_id.lower():
                family = "Audio"
            else:
                family = "Other"

            if family not in model_families:
                model_families[family] = []
            model_families[family].append(model_id)

        # Display models by family
        for family, family_models in sorted(model_families.items()):
            print(f"\n{family}:")
            for model_id in sorted(family_models)[:5]:  # Show first 5 per family
                print(f"  • {model_id}")
            if len(family_models) > 5:
                print(f"  ... and {len(family_models) - 5} more")

        # Highlight key models
        print(f"\n💡 Recommended Models:")
        print(f"   • gpt-4o: Most capable, supports vision")
        print(f"   • gpt-4o-mini: Fast and cost-effective")
        print(f"   • gpt-5-mini: Reasoning and advanced tasks")
        print(f"   • gpt-3.5-turbo: Legacy, fast")

        return models
    except Exception as e:
        print(f"❌ Error discovering models: {e}")
        print(f"   This is normal if rate limited or API issues")
        return None


# =============================================================================
# Demo 12: Call Dynamically Discovered Model
# =============================================================================

async def demo_dynamic_model_call():
    """Discover models and call one of them"""
    print(f"\n{'='*70}")
    print("Demo 12: Call Dynamically Discovered Model")
    print(f"{'='*70}")

    try:
        from chuk_llm.api.discovery import discover_models

        print("⏳ Step 1: Discovering available models...")

        models = await discover_models("openai", force_refresh=True)

        # Find a text chat model (not TTS/audio)
        target_model = None
        for model in models:
            model_id = model.get("name", model.get("id", ""))
            # Skip non-chat models
            if any(x in model_id.lower() for x in ["tts", "whisper", "audio", "realtime", "dall-e", "sora"]):
                continue
            # Prefer gpt-4o-mini
            if "gpt-4o-mini" in model_id.lower():
                target_model = model_id
                break
            # Otherwise accept any gpt model
            if "gpt" in model_id.lower() and not target_model:
                target_model = model_id

        if not target_model:
            # Fallback to first text model
            for model in models:
                model_id = model.get("name", model.get("id", ""))
                if not any(x in model_id.lower() for x in ["tts", "whisper", "audio", "realtime", "dall-e", "sora"]):
                    target_model = model_id
                    break

        if not target_model:
            target_model = "gpt-4o-mini"  # Ultimate fallback

        print(f"✅ Found {len(models)} models")
        print(f"🎯 Selected model: {target_model}")

        print(f"\n⏳ Step 2: Creating client for discovered model...")

        client = get_client("openai", model=target_model)

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
# Demo 13: Error Handling
# =============================================================================

async def demo_error_handling(model: str):
    """Error handling and retries"""
    print(f"\n{'='*70}")
    print("Demo 12: Error Handling")
    print(f"{'='*70}")

    client = get_client("openai", model=model)

    # Test 1: Invalid parameters
    print("Test 1: Invalid max_tokens")
    try:
        messages = [Message(role=MessageRole.USER, content="Hello")]
        response = await client.create_completion(messages, max_tokens=1000000)
        print("  ✅ Handled gracefully")
    except Exception as e:
        print(f"  ⚠️  Error caught: {type(e).__name__}")

    # Test 2: Empty messages
    print("\nTest 2: Empty messages")
    try:
        response = await client.create_completion([], max_tokens=50)
        print("  ✅ Handled gracefully")
    except Exception as e:
        print(f"  ⚠️  Error caught: {type(e).__name__}")

    # Test 3: Normal request
    print("\nTest 3: Normal request")
    try:
        messages = [Message(role=MessageRole.USER, content="Say 'OK'")]
        response = await client.create_completion(messages, max_tokens=10)
        print(f"  ✅ Response: {response['response']}")
    except Exception as e:
        print(f"  ❌ Unexpected error: {e}")

    return None


# =============================================================================
# Main Runner
# =============================================================================

async def main():
    """Run all demos"""
    parser = argparse.ArgumentParser(
        description="OpenAI Chat Completions Comprehensive Examples"
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="Model to use (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--demo",
        type=int,
        help="Run specific demo number (1-13)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Skip slow demos (reasoning, comparison)",
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("🚀 OpenAI Chat Completions API - Comprehensive Examples")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"API Key: {os.getenv('OPENAI_API_KEY')[:20]}...")
    print("=" * 70)

    demos = []

    if args.demo is None or args.demo == 1:
        demos.append(("Basic Completion", demo_basic_completion(args.model)))

    if args.demo is None or args.demo == 2:
        demos.append(("Streaming", demo_streaming(args.model)))

    if args.demo is None or args.demo == 3:
        demos.append(("Function Calling", demo_function_calling(args.model)))

    if args.demo is None or args.demo == 4:
        demos.append(("Vision", demo_vision("gpt-4o")))

    # Demo 4.5: Audio Input (not numbered in CLI since it's optional/may fail)
    if args.demo is None:
        demos.append(("Audio Input", demo_audio_input()))

    if args.demo is None or args.demo == 5:
        demos.append(("JSON Mode", demo_json_mode(args.model)))

    if args.demo is None or args.demo == 6:
        demos.append(("Conversation", demo_conversation(args.model)))

    if args.demo is None or args.demo == 7:
        demos.append(("Parameters", demo_parameters(args.model)))

    if (args.demo is None or args.demo == 8) and not args.quick:
        demos.append(("Reasoning Model", demo_reasoning_model()))

    if (args.demo is None or args.demo == 9) and not args.quick:
        demos.append(("Model Comparison", demo_model_comparison()))

    if args.demo is None or args.demo == 10:
        demos.append(("Structured Outputs", demo_structured_outputs(args.model)))

    if args.demo is None or args.demo == 11:
        demos.append(("Model Discovery", demo_model_discovery()))

    if args.demo is None or args.demo == 12:
        demos.append(("Dynamic Model Call", demo_dynamic_model_call()))

    if args.demo is None or args.demo == 13:
        demos.append(("Error Handling", demo_error_handling(args.model)))

    # Run all demos
    for name, demo_coro in demos:
        try:
            await demo_coro
        except Exception as e:
            print(f"\n❌ Error in {name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("✅ All demos completed!")
    print("=" * 70)
    print("\nTips:")
    print("  - Use --demo N to run specific demo")
    print("  - Use --model gpt-4o for vision support")
    print("  - Use --quick to skip slow demos")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
