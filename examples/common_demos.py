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

    config = get_config()
    if not config.supports_feature(provider, Feature.STREAMING, model):
        print(f"⚠️  Model {model} doesn't support streaming")
        return None

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

    config = get_config()
    if not config.supports_feature(provider, Feature.TOOLS, model):
        print(f"⚠️  Model {model} doesn't support function calling")
        return None

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

    response = await client.create_completion(messages, tools=tools)

    if response.get("tool_calls"):
        print(f"\n🔧 Model called {len(response['tool_calls'])} function(s):")

        tool_results = []
        for tool_call in response["tool_calls"]:
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

        print("\n⏳ Getting final response...")
        final_response = await client.create_completion(messages, tools=tools)

        print(f"\n✅ Final Response:")
        print(f"{final_response['response']}")

        return final_response
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

    config = get_config()
    if not config.supports_feature(provider, Feature.VISION, model):
        print(f"⚠️  Model {model} doesn't support vision")
        return None

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

    config = get_config()
    if not config.supports_feature(provider, Feature.VISION, model):
        print(f"⚠️  Model {model} doesn't support vision")
        return None

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

    config = get_config()
    if not config.supports_feature(provider, Feature.JSON_MODE, model):
        print(f"⚠️  Model {model} doesn't support JSON mode")
        return None

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

    config = get_config()
    if not config.supports_feature(provider, Feature.JSON_MODE, model):
        print(f"⚠️  Model {model} doesn't support structured outputs/JSON mode")
        return None

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
    print("Demo 7: Multi-turn Conversation")
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
        # Try to import the appropriate discoverer
        discoverer = None

        if provider == "openai":
            from chuk_llm.llm.discovery.openai_discoverer import OpenAIModelDiscoverer
            import os
            discoverer = OpenAIModelDiscoverer(
                provider_name="openai",
                api_key=os.getenv("OPENAI_API_KEY")
            )
        elif provider == "anthropic":
            from chuk_llm.llm.discovery.anthropic_discoverer import AnthropicModelDiscoverer
            import os
            discoverer = AnthropicModelDiscoverer(
                provider_name="anthropic",
                api_key=os.getenv("ANTHROPIC_API_KEY")
            )
        elif provider == "ollama":
            from chuk_llm.llm.discovery.ollama_discoverer import OllamaModelDiscoverer
            discoverer = OllamaModelDiscoverer(provider_name="ollama")
        elif provider == "azure_openai":
            from chuk_llm.llm.discovery.azure_openai_discoverer import AzureOpenAIModelDiscoverer
            import os
            discoverer = AzureOpenAIModelDiscoverer(
                provider_name="azure_openai",
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
            )
        elif provider == "mistral":
            from chuk_llm.llm.discovery.mistral_discoverer import MistralModelDiscoverer
            import os
            discoverer = MistralModelDiscoverer(
                provider_name="mistral",
                api_key=os.getenv("MISTRAL_API_KEY")
            )
        elif provider == "watsonx":
            from chuk_llm.llm.discovery.watsonx_discoverer import WatsonXModelDiscoverer
            import os
            discoverer = WatsonXModelDiscoverer(
                provider_name="watsonx",
                api_key=os.getenv("WATSONX_API_KEY"),
                project_id=os.getenv("WATSONX_PROJECT_ID")
            )
        elif provider == "gemini":
            from chuk_llm.llm.discovery.gemini_discoverer import GeminiModelDiscoverer
            import os
            discoverer = GeminiModelDiscoverer(
                provider_name="gemini",
                api_key=os.getenv("GOOGLE_API_KEY")
            )
        elif provider in ["deepseek", "groq", "openrouter", "perplexity"]:
            # OpenAI-compatible providers can use OpenAI discoverer
            from chuk_llm.llm.discovery.openai_discoverer import OpenAIModelDiscoverer
            import os

            api_key_map = {
                "deepseek": "DEEPSEEK_API_KEY",
                "groq": "GROQ_API_KEY",
                "openrouter": "OPENROUTER_API_KEY",
                "perplexity": "PERPLEXITY_API_KEY",
            }

            api_base_map = {
                "deepseek": "https://api.deepseek.com",
                "groq": "https://api.groq.com/openai",
                "openrouter": "https://openrouter.ai/api",
                "perplexity": "https://api.perplexity.ai",
            }

            discoverer = OpenAIModelDiscoverer(
                provider_name=provider,
                api_key=os.getenv(api_key_map[provider]),
                api_base=api_base_map[provider]
            )
        else:
            print(f"⚠️  Model discovery not available for {provider}")
            print(f"   Supported providers: openai, anthropic, ollama, azure_openai, mistral, watsonx, gemini, deepseek, groq, openrouter, perplexity")
            return None

        print(f"⏳ Discovering models from {provider}...")

        models = await discoverer.discover_models()

        print(f"\n✅ Found {len(models)} models\n")

        # Show first 10 models
        print("Available models:")
        for model_info in models[:10]:
            model_id = model_info.get("id", model_info.get("name", "unknown"))
            print(f"  • {model_id}")

        if len(models) > 10:
            print(f"  ... and {len(models) - 10} more")

        return models
    except Exception as e:
        print(f"❌ Error discovering models: {e}")
        print(f"   This is normal if the provider doesn't support discovery")
        return None


# =============================================================================
# Demo 10: Error Handling
# =============================================================================

async def demo_error_handling(client, provider: str, model: str):
    """Demonstrate error handling"""
    print(f"\n{'='*70}")
    print("Demo 10: Error Handling")
    print(f"{'='*70}")

    # Test 1: Invalid max_tokens
    print("\nTest 1: Handling invalid parameters")
    try:
        messages = [Message(role=MessageRole.USER, content="Hello")]
        response = await client.create_completion(messages, max_tokens=-1)
        print("  ✅ Request succeeded (provider accepted -1)")
    except Exception as e:
        print(f"  ✅ Caught error: {type(e).__name__}")
        print(f"     {str(e)[:100]}")

    # Test 2: Empty message
    print("\nTest 2: Handling empty messages")
    try:
        messages = []
        response = await client.create_completion(messages)
        print("  ✅ Request succeeded with empty messages")
    except Exception as e:
        print(f"  ✅ Caught error: {type(e).__name__}")
        print(f"     {str(e)[:100]}")

    # Test 3: Very long input (might hit context limit)
    print("\nTest 3: Handling context limits")
    try:
        long_text = "Hello " * 50000  # Very long text
        messages = [Message(role=MessageRole.USER, content=long_text)]
        response = await client.create_completion(messages, max_tokens=10)
        print("  ✅ Request succeeded (provider handled long input)")
    except Exception as e:
        print(f"  ✅ Caught error: {type(e).__name__}")
        print(f"     Error handled gracefully")

    print("\n✅ Error handling demonstration complete")
    return None


# =============================================================================
# Demo Runner Helper
# =============================================================================

async def run_all_demos(client, provider: str, model: str, skip_tools=False, skip_vision=False, skip_discovery=False):
    """Run all demos for a provider"""
    demos = [
        ("Basic Completion", demo_basic_completion(client, provider, model)),
        ("Streaming", demo_streaming(client, provider, model)),
    ]

    if not skip_tools:
        demos.append(("Function Calling", demo_function_calling(client, provider, model)))

    if not skip_vision:
        demos.append(("Vision", demo_vision(client, provider, model)))
        demos.append(("Vision with URL", demo_vision_url(client, provider, model)))

    demos.extend([
        ("JSON Mode", demo_json_mode(client, provider, model)),
        ("Reasoning", demo_reasoning(client, provider, model)),
        ("Structured Outputs", demo_structured_outputs(client, provider, model)),
        ("Conversation", demo_conversation(client, provider, model)),
    ])

    if not skip_discovery:
        demos.append(("Model Discovery", demo_model_discovery(client, provider, model)))

    demos.append(("Error Handling", demo_error_handling(client, provider, model)))

    for name, demo_coro in demos:
        try:
            await demo_coro
        except Exception as e:
            print(f"\n❌ Error in {name}: {e}")
            import traceback
            traceback.print_exc()

    return True
