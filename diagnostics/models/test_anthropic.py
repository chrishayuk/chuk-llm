#!/usr/bin/env python3
"""
Fixed Anthropic Vision Test
===========================

Test vision with working URLs and proper base64 images.
"""

import asyncio
import base64
import os

from dotenv import load_dotenv

load_dotenv()


def create_colored_square_base64(color="red", size=10):
    """Create a simple colored square as base64 PNG"""
    try:
        import io

        from PIL import Image

        # Create a colored square
        img = Image.new("RGB", (size, size), color)

        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        img_data = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return img_data
    except ImportError:
        # Fallback: minimal 1x1 red pixel PNG (valid)
        # This is a properly encoded 1x1 red pixel
        return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="


async def test_working_url_vision():
    """Test with known working image URLs"""
    from chuk_llm.llm.client import get_client

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ Need ANTHROPIC_API_KEY")
        return False

    print("🌐 Testing Vision with Working URLs")
    print("=" * 50)

    # Known working image URLs (these should be stable)
    test_urls = [
        "https://httpbin.org/image/png",  # Returns a simple PNG
        "https://via.placeholder.com/150/FF0000/FFFFFF?text=RED",  # Red square with text
        "https://picsum.photos/100/100",  # Random 100x100 image
    ]

    client = get_client("anthropic", model="claude-sonnet-4-20250514")

    for i, image_url in enumerate(test_urls, 1):
        print(f"\n🔄 Test {i}: {image_url}")

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe what you see in this image in one sentence.",
                    },
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ]

        try:
            response = await client.create_completion(messages, max_tokens=100)

            if (
                response.get("response")
                and "could not load" not in response.get("response", "").lower()
            ):
                print(f"✅ Success: {response['response'][:100]}...")
                return True
            else:
                print(f"❌ Failed: {response.get('response', 'No response')[:100]}...")

        except Exception as e:
            print(f"❌ Error: {e}")

    return False


async def test_proper_base64_vision():
    """Test with properly created base64 images"""
    from chuk_llm.llm.client import get_client

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ Need ANTHROPIC_API_KEY")
        return False

    print("\n📄 Testing Vision with Proper Base64 Images")
    print("=" * 50)

    # Test different colored squares
    colors = ["red", "blue", "green"]

    client = get_client("anthropic", model="claude-sonnet-4-20250514")

    for color in colors:
        print(f"\n🎨 Testing {color} square:")

        # Create a colored square
        base64_image = create_colored_square_base64(color, 20)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What color is this square? Answer with just the color name.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    },
                ],
            }
        ]

        try:
            response = await client.create_completion(messages, max_tokens=20)

            if response.get("response"):
                detected_color = response["response"].strip().lower()
                is_correct = color.lower() in detected_color
                status = "✅" if is_correct else "⚠️"
                print(f"   {status} Expected: {color}, Got: {response['response']}")

                if is_correct:
                    return True
            else:
                print("   ❌ No response")

        except Exception as e:
            print(f"   ❌ Error: {e}")

    return False


async def test_multimodal_conversation():
    """Test a more complex multimodal conversation"""
    from chuk_llm.llm.client import get_client

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ Need ANTHROPIC_API_KEY")
        return False

    print("\n💬 Testing Multimodal Conversation")
    print("=" * 50)

    client = get_client("anthropic", model="claude-sonnet-4-20250514")

    # Create two different colored squares
    red_square = create_colored_square_base64("red", 30)
    blue_square = create_colored_square_base64("blue", 30)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "I'm going to show you two colored squares. Please identify the colors.",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{red_square}"},
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{blue_square}"},
                },
            ],
        }
    ]

    try:
        print("🔄 Sending two images in one message...")
        response = await client.create_completion(messages, max_tokens=100)

        if response.get("response"):
            result = response["response"]
            has_red = "red" in result.lower()
            has_blue = "blue" in result.lower()

            if has_red and has_blue:
                print("✅ Successfully identified both colors!")
                print(f"   Response: {result}")
                return True
            else:
                print("⚠️  Partial success - missing colors")
                print(f"   Response: {result}")
                return False
        else:
            print("❌ No response")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def test_vision_with_tools():
    """Test vision combined with function calling"""
    from chuk_llm.llm.client import get_client

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ Need ANTHROPIC_API_KEY")
        return False

    print("\n🔧 Testing Vision + Function Calling")
    print("=" * 50)

    client = get_client("anthropic", model="claude-sonnet-4-20250514")

    # Define a tool for color analysis
    tools = [
        {
            "type": "function",
            "function": {
                "name": "analyze_color",
                "description": "Analyze and store color information from an image",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "primary_color": {
                            "type": "string",
                            "description": "The main color detected in the image",
                        },
                        "confidence": {
                            "type": "number",
                            "description": "Confidence level from 0-1",
                        },
                        "description": {
                            "type": "string",
                            "description": "Brief description of what was seen",
                        },
                    },
                    "required": ["primary_color", "confidence", "description"],
                },
            },
        }
    ]

    # Create a green square
    green_square = create_colored_square_base64("green", 25)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Please analyze this image and use the analyze_color function to store your findings.",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{green_square}"},
                },
            ],
        }
    ]

    try:
        print("🔄 Testing vision + tools...")
        response = await client.create_completion(messages, tools=tools)

        if response.get("tool_calls"):
            print("✅ Function called successfully!")
            for tool_call in response["tool_calls"]:
                func_name = tool_call["function"]["name"]
                func_args = tool_call["function"]["arguments"]
                print(f"   Function: {func_name}")
                print(f"   Arguments: {func_args}")
            return True
        else:
            print("❌ No function calls made")
            print(f"   Response: {response.get('response', 'No response')}")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def main():
    """Run all vision tests"""
    print("🚀 Anthropic Vision Test Suite")
    print("=" * 60)
    print(f"API Key: {'✅ Set' if os.getenv('ANTHROPIC_API_KEY') else '❌ Missing'}")

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("\n❌ Please set ANTHROPIC_API_KEY environment variable")
        return

    tests = [
        ("Working URL Vision", test_working_url_vision),
        ("Proper Base64 Vision", test_proper_base64_vision),
        ("Multimodal Conversation", test_multimodal_conversation),
        ("Vision + Function Calling", test_vision_with_tools),
    ]

    results = {}

    for test_name, test_func in tests:
        try:
            print(f"\n{'=' * 60}")
            success = await test_func()
            results[test_name] = success
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"\n{status}: {test_name}")
        except Exception as e:
            results[test_name] = False
            print(f"\n❌ ERROR in {test_name}: {e}")

    # Summary
    print(f"\n{'=' * 60}")
    print("📊 TEST SUMMARY")
    print("=" * 60)

    passed = sum(results.values())
    total = len(results)

    for test_name, success in results.items():
        status = "✅" if success else "❌"
        print(f"   {status} {test_name}")

    print(f"\n📈 Results: {passed}/{total} tests passed")

    if passed == total:
        print(
            "🎉 All vision tests passed! Your Anthropic vision integration is working perfectly."
        )
    elif passed > 0:
        print(
            "⚠️  Some tests passed - vision is partially working. Check failed tests above."
        )
    else:
        print(
            "❌ All tests failed - there may be an issue with your vision implementation."
        )

    # Recommendations
    if passed < total:
        print("\n💡 Troubleshooting Tips:")
        print(
            "   1. Ensure you're using a vision-capable model (Claude 4.x or Claude 3.7.x)"
        )
        print("   2. Check that your base64 images are valid PNG format")
        print("   3. Verify your ANTHROPIC_API_KEY has vision access")
        print("   4. Test with smaller images (< 5MB)")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Tests cancelled by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback

        traceback.print_exc()
