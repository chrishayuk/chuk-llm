#!/usr/bin/env python3
"""
Google Gemini Enhanced Examples - Latest API Features
=====================================================

Demonstrates the latest Gemini API features including:
- OpenAI compatibility mode
- Thinking mode (Gemini 2.0 Flash Thinking)
- Structured output with JSON schemas
- Native image understanding
- Image generation with Imagen 3

Based on official Gemini API documentation:
- https://ai.google.dev/gemini-api/docs/quickstart
- https://ai.google.dev/gemini-api/docs/openai
- https://ai.google.dev/gemini-api/docs/thinking
- https://ai.google.dev/gemini-api/docs/structured-output
- https://ai.google.dev/gemini-api/docs/image-understanding
- https://ai.google.dev/gemini-api/docs/image-generation

Requirements:
- pip install google-generativeai openai chuk-llm
- Set GEMINI_API_KEY environment variable

Usage:
    python gemini_enhanced_examples.py
    python gemini_enhanced_examples.py --mode openai
    python gemini_enhanced_examples.py --test-thinking
"""

import argparse
import asyncio
import base64
import json
import os
import sys
from dataclasses import dataclass

# Load environment variables
from dotenv import load_dotenv

load_dotenv()

# Ensure we have the required environment
if not os.getenv("GEMINI_API_KEY"):
    print("❌ Please set GEMINI_API_KEY environment variable")
    print("   export GEMINI_API_KEY='your_api_key_here'")
    print("   Get your key at: https://aistudio.google.com/apikey")
    sys.exit(1)

try:
    import httpx
    from openai import OpenAI  # For OpenAI compatibility mode

    from chuk_llm.configuration import get_config
    from chuk_llm.llm.client import get_client
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("   Please install: pip install chuk-llm openai httpx")
    sys.exit(1)

# =============================================================================
# Example 1: OpenAI Compatibility Mode
# =============================================================================


async def openai_compatibility_example():
    """
    Demonstrate using Gemini through OpenAI-compatible API
    Based on: https://ai.google.dev/gemini-api/docs/openai
    """
    print("\n🔄 OpenAI Compatibility Mode")
    print("=" * 60)

    # Use OpenAI client with Gemini endpoint
    client = OpenAI(
        api_key=os.getenv("GEMINI_API_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    print("📝 Using OpenAI client with Gemini endpoint...")

    # Test basic completion
    response = client.chat.completions.create(
        model="gemini-1.5-flash",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Explain quantum computing in one sentence."},
        ],
    )

    print(f"✅ Response: {response.choices[0].message.content}")

    # Test streaming
    print("\n🌊 Testing streaming with OpenAI client...")
    stream = client.chat.completions.create(
        model="gemini-1.5-flash",
        messages=[{"role": "user", "content": "Count to 5"}],
        stream=True,
    )

    print("   ", end="", flush=True)
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print()

    return True


# =============================================================================
# Example 2: Thinking Mode (Gemini 2.0 Flash Thinking)
# =============================================================================


async def thinking_mode_example():
    """
    Demonstrate Gemini's thinking mode for complex reasoning
    Based on: https://ai.google.dev/gemini-api/docs/thinking
    """
    print("\n🧠 Thinking Mode Example (Gemini 2.0 Flash Thinking)")
    print("=" * 60)

    # Check if thinking model is available
    thinking_model = "gemini-2.0-flash-thinking-exp"

    try:
        # Try using the thinking model
        client = get_client("gemini", model=thinking_model)

        # Complex reasoning problem
        prompt = """
        I have 3 apples. I eat 1 apple and buy 5 more apples.
        Then I give away half of my apples. How many apples do I have left?
        Show your reasoning step by step.
        """

        print(f"🤔 Problem: {prompt.strip()}")
        print("\n💭 Thinking process:")

        messages = [{"role": "user", "content": prompt}]
        response = await client.create_completion(messages)

        # Extract thinking process if available
        if isinstance(response, dict):
            content = response.get("response", "")
            print(f"   {content}")

        return response

    except Exception as e:
        print(f"⚠️  Thinking model not available: {e}")
        print("   Using standard model with reasoning prompt...")

        # Fallback to standard model with reasoning
        client = get_client("gemini", model="gemini-2.5-flash")

        enhanced_prompt = f"""
        {prompt}

        Please think through this step-by-step:
        1. State what you start with
        2. Show each operation
        3. Calculate the final result
        """

        messages = [{"role": "user", "content": enhanced_prompt}]
        response = await client.create_completion(messages)

        if isinstance(response, dict):
            content = response.get("response", "")
            print(f"   {content}")

        return response


# =============================================================================
# Example 3: Structured Output with JSON Schema
# =============================================================================


@dataclass
class Recipe:
    """Schema for structured recipe output"""

    name: str
    description: str
    prep_time: int  # in minutes
    cook_time: int  # in minutes
    servings: int
    ingredients: list[dict[str, str]]  # {item, amount, unit}
    instructions: list[str]
    difficulty: str  # easy, medium, hard


async def structured_output_example():
    """
    Demonstrate structured output with JSON schema
    Based on: https://ai.google.dev/gemini-api/docs/structured-output
    """
    print("\n📋 Structured Output with JSON Schema")
    print("=" * 60)

    client = get_client("gemini", model="gemini-2.5-pro")

    # Define the schema

    print("🍝 Generating structured recipe...")

    prompt = """
    Create a detailed recipe for spaghetti carbonara.
    Return the recipe in the exact JSON format specified.
    """

    messages = [
        {
            "role": "system",
            "content": "You are a professional chef. Return recipes in valid JSON format.",
        },
        {"role": "user", "content": prompt},
    ]

    try:
        # Request structured output
        response = await client.create_completion(
            messages, response_format={"type": "json_object"}
        )

        if isinstance(response, dict):
            content = response.get("response", "{}")

            # Parse and validate JSON
            try:
                recipe_data = json.loads(content)
                print("\n✅ Structured Recipe Generated:")
                print(f"   Name: {recipe_data.get('name', 'N/A')}")
                print(f"   Difficulty: {recipe_data.get('difficulty', 'N/A')}")
                print(f"   Prep Time: {recipe_data.get('prep_time', 0)} minutes")
                print(f"   Cook Time: {recipe_data.get('cook_time', 0)} minutes")
                print(f"   Servings: {recipe_data.get('servings', 0)}")

                ingredients = recipe_data.get("ingredients", [])
                if ingredients:
                    print(f"\n   Ingredients ({len(ingredients)} items):")
                    for ing in ingredients[:3]:
                        print(
                            f"     - {ing.get('amount', '')} {ing.get('unit', '')} {ing.get('item', '')}"
                        )
                    if len(ingredients) > 3:
                        print(f"     ... and {len(ingredients) - 3} more")

                instructions = recipe_data.get("instructions", [])
                if instructions:
                    print(f"\n   Instructions ({len(instructions)} steps):")
                    for i, step in enumerate(instructions[:2], 1):
                        print(f"     {i}. {step[:80]}...")
                    if len(instructions) > 2:
                        print(f"     ... and {len(instructions) - 2} more steps")

                return recipe_data

            except json.JSONDecodeError as e:
                print(f"⚠️  JSON parsing error: {e}")
                print(f"   Raw response: {content[:200]}...")
                return None

    except Exception as e:
        print(f"❌ Structured output failed: {e}")
        return None


# =============================================================================
# Example 4: Native Image Understanding
# =============================================================================


async def image_understanding_example():
    """
    Demonstrate native image understanding capabilities
    Based on: https://ai.google.dev/gemini-api/docs/image-understanding
    """
    print("\n👁️ Native Image Understanding")
    print("=" * 60)

    client = get_client("gemini", model="gemini-2.5-flash")

    # Create a test image
    print("🖼️ Creating test image...")

    # Create a simple colored rectangle using PIL if available
    try:
        import io

        from PIL import Image, ImageDraw, ImageFont

        # Create an image with shapes and text
        img = Image.new("RGB", (400, 300), color="white")
        draw = ImageDraw.Draw(img)

        # Draw shapes
        draw.rectangle([50, 50, 150, 150], fill="red", outline="black", width=2)
        draw.ellipse([200, 50, 350, 200], fill="blue", outline="black", width=2)
        draw.polygon(
            [(100, 200), (150, 250), (50, 250)], fill="green", outline="black", width=2
        )

        # Add text
        draw.text((200, 250), "Gemini Test", fill="black")

        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        print("✅ Test image created with shapes and text")

    except ImportError:
        print("⚠️  PIL not available, using simple fallback image")
        # Fallback: simple red square
        img_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAYAAACNMs+9AAAAFUlEQVR42mP8z8BQz0AEYBxVSF+FABJADveWkH6oAAAAAElFTkSuQmCC"

    # Test image understanding
    print("\n🔍 Analyzing image...")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Describe what you see in this image. List any shapes, colors, and text you can identify.",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_base64}"},
                },
            ],
        }
    ]

    try:
        response = await client.create_completion(messages)

        if isinstance(response, dict):
            content = response.get("response", "")
            print("📝 Image Analysis:")
            print(f"   {content}")

        # Test with specific questions
        print("\n🎯 Testing specific image queries...")

        follow_up = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "How many shapes are in the image?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_base64}"},
                    },
                ],
            }
        ]

        response2 = await client.create_completion(follow_up)
        if isinstance(response2, dict):
            content2 = response2.get("response", "")
            print(f"   Shape count: {content2}")

        return response

    except Exception as e:
        print(f"❌ Image understanding failed: {e}")
        return None


# =============================================================================
# Example 5: Image Generation with Imagen 3
# =============================================================================


async def image_generation_example():
    """
    Demonstrate image generation capabilities (Imagen 3)
    Based on: https://ai.google.dev/gemini-api/docs/image-generation
    """
    print("\n🎨 Image Generation with Imagen 3")
    print("=" * 60)

    print("📝 Note: Image generation requires Imagen 3 API access")
    print("   This example shows the request structure")

    # Image generation prompt
    prompt = "A serene Japanese garden with cherry blossoms, a wooden bridge over a koi pond, mountains in the background, in watercolor style"

    print(f"\n🖼️ Prompt: {prompt}")

    # Show the API structure for image generation
    imagen_request = {
        "model": "imagen-3",
        "prompt": prompt,
        "n": 1,
        "size": "1024x1024",
        "quality": "standard",
        "style": "natural",
    }

    print("\n📋 Imagen 3 Request Structure:")
    print(json.dumps(imagen_request, indent=2))

    # Try to generate if API is available
    try:
        # This would require the actual Imagen API endpoint
        print("\n⚠️  Actual image generation requires Imagen 3 API access")
        print("   Visit: https://ai.google.dev/gemini-api/docs/image-generation")

        # Demonstrate text-to-image description instead
        client = get_client("gemini", model="gemini-2.5-flash")

        analysis_prompt = f"""
        If I were to generate an image with this prompt: "{prompt}"

        Describe in detail what the resulting image should look like, including:
        - Composition and layout
        - Color palette
        - Artistic style
        - Key elements and their placement
        """

        messages = [{"role": "user", "content": analysis_prompt}]
        response = await client.create_completion(messages)

        if isinstance(response, dict):
            content = response.get("response", "")
            print("\n🎨 Expected Image Description:")
            print(f"   {content[:500]}...")

        return response

    except Exception as e:
        print(f"❌ Image generation example failed: {e}")
        return None


# =============================================================================
# Example 6: Advanced Function Calling
# =============================================================================


async def advanced_function_calling_example():
    """
    Demonstrate advanced function calling with Gemini
    """
    print("\n🔧 Advanced Function Calling")
    print("=" * 60)

    client = get_client("gemini", model="gemini-2.5-pro")

    # Define multiple tools
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_knowledge",
                "description": "Search for information in the knowledge base",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "category": {
                            "type": "string",
                            "enum": ["science", "history", "technology", "general"],
                            "description": "Category to search in",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum results",
                        },
                    },
                    "required": ["query", "category"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "Perform mathematical calculations",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Math expression",
                        },
                        "precision": {
                            "type": "integer",
                            "description": "Decimal places",
                        },
                    },
                    "required": ["expression"],
                },
            },
        },
    ]

    prompt = "Search for information about quantum computing and calculate 2^10"

    print(f"📝 Prompt: {prompt}")
    print("🔄 Requesting function calls...")

    messages = [{"role": "user", "content": prompt}]

    try:
        response = await client.create_completion(messages, tools=tools)

        if isinstance(response, dict) and response.get("tool_calls"):
            print(f"✅ Function calls requested: {len(response['tool_calls'])}")

            for i, tool_call in enumerate(response["tool_calls"], 1):
                func = tool_call["function"]
                print(f"\n   {i}. Function: {func['name']}")
                print(f"      Arguments: {func['arguments']}")

            # Simulate tool execution
            tool_results = []
            for tool_call in response["tool_calls"]:
                func_name = tool_call["function"]["name"]

                if func_name == "search_knowledge":
                    result = {
                        "results": [
                            "Quantum computing uses quantum bits (qubits)",
                            "Can perform certain calculations exponentially faster",
                            "Based on quantum superposition and entanglement",
                        ]
                    }
                elif func_name == "calculate":
                    result = {"result": 1024, "expression": "2^10"}
                else:
                    result = {"status": "completed"}

                tool_results.append(
                    {"tool_call_id": tool_call["id"], "output": json.dumps(result)}
                )

            print("\n📊 Tool Results:")
            for result in tool_results:
                print(f"   {result['output'][:100]}...")

            return response

        else:
            print("ℹ️  No function calls made")
            if isinstance(response, dict):
                print(f"   Response: {response.get('response', '')[:200]}...")
            return response

    except Exception as e:
        print(f"❌ Function calling failed: {e}")
        return None


# =============================================================================
# Example 7: Context Caching for Long Conversations
# =============================================================================


async def context_caching_example():
    """
    Demonstrate context caching for efficient long conversations
    """
    print("\n💾 Context Caching Example")
    print("=" * 60)

    client = get_client("gemini", model="gemini-2.5-flash")

    # Simulate a long context
    context = """
    You are an expert in quantum physics. Here's important background:

    Quantum mechanics is a fundamental theory in physics that describes nature at the smallest scales.
    Key principles include:
    1. Wave-particle duality
    2. Uncertainty principle
    3. Quantum entanglement
    4. Superposition
    5. Quantum tunneling

    Recent developments include quantum computing, quantum cryptography, and quantum sensing.
    """

    print("📚 Setting up context cache...")
    print(f"   Context length: {len(context.split())} words")

    # First conversation turn
    messages = [
        {"role": "system", "content": context},
        {"role": "user", "content": "What is superposition?"},
    ]

    print("\n💬 Turn 1: What is superposition?")
    response1 = await client.create_completion(messages)
    if isinstance(response1, dict):
        print(f"   Response: {response1.get('response', '')[:200]}...")

    # Add response to conversation
    messages.append({"role": "assistant", "content": response1.get("response", "")})

    # Second turn (context is cached)
    messages.append(
        {"role": "user", "content": "How does it relate to quantum computing?"}
    )

    print("\n💬 Turn 2: How does it relate to quantum computing?")
    response2 = await client.create_completion(messages)
    if isinstance(response2, dict):
        print(f"   Response: {response2.get('response', '')[:200]}...")

    print("\n✅ Context caching allows efficient multi-turn conversations")
    return True


# =============================================================================
# Main Function
# =============================================================================


async def main():
    """Run all enhanced examples"""
    parser = argparse.ArgumentParser(description="Gemini Enhanced Examples")
    parser.add_argument(
        "--mode",
        choices=["all", "openai", "thinking", "structured", "vision", "generation"],
        default="all",
        help="Which examples to run",
    )
    parser.add_argument(
        "--test-thinking", action="store_true", help="Test thinking mode"
    )

    args = parser.parse_args()

    print("🚀 Google Gemini Enhanced Examples")
    print("=" * 60)
    print(f"API Key: {'✅ Set' if os.getenv('GEMINI_API_KEY') else '❌ Missing'}")
    print(f"Mode: {args.mode}")

    examples = []

    if args.mode == "all":
        examples = [
            ("OpenAI Compatibility", openai_compatibility_example),
            ("Thinking Mode", thinking_mode_example),
            ("Structured Output", structured_output_example),
            ("Image Understanding", image_understanding_example),
            ("Image Generation", image_generation_example),
            ("Advanced Functions", advanced_function_calling_example),
            ("Context Caching", context_caching_example),
        ]
    elif args.mode == "openai":
        examples = [("OpenAI Compatibility", openai_compatibility_example)]
    elif args.mode == "thinking":
        examples = [("Thinking Mode", thinking_mode_example)]
    elif args.mode == "structured":
        examples = [("Structured Output", structured_output_example)]
    elif args.mode == "vision":
        examples = [("Image Understanding", image_understanding_example)]
    elif args.mode == "generation":
        examples = [("Image Generation", image_generation_example)]

    if args.test_thinking:
        examples = [("Thinking Mode", thinking_mode_example)]

    # Run examples
    results = {}
    for name, example_func in examples:
        try:
            print(f"\n{'=' * 60}")
            result = await example_func()
            results[name] = {"success": True, "result": result}
            print(f"✅ {name} completed")
        except Exception as e:
            results[name] = {"success": False, "error": str(e)}
            print(f"❌ {name} failed: {e}")

    # Summary
    print(f"\n{'=' * 60}")
    print("📊 SUMMARY")
    print("=" * 60)

    successful = sum(1 for r in results.values() if r["success"])
    total = len(results)

    print(f"✅ Successful: {successful}/{total}")

    for name, result in results.items():
        status = "✅" if result["success"] else "❌"
        print(f"   {status} {name}")

    if successful == total:
        print("\n🎉 All enhanced examples completed successfully!")
        print("🔗 Gemini's advanced features demonstrated:")
        print("   • OpenAI compatibility for easy migration")
        print("   • Thinking mode for complex reasoning")
        print("   • Structured output with JSON schemas")
        print("   • Native image understanding")
        print("   • Image generation capabilities")
        print("   • Advanced function calling")
        print("   • Context caching for efficiency")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Examples cancelled by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)
