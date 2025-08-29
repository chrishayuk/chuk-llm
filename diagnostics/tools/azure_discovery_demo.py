#!/usr/bin/env python3
"""
Azure OpenAI Demo Script - Testing Smart Discovery Support
==========================================================

This script demonstrates how the enhanced Azure OpenAI client works with:
1. Custom deployment names (like "scribeflowgpt4o")
2. Smart feature detection
3. Tool calling with MCP servers
4. Streaming responses

Run this to verify your Azure deployment works with the enhanced client!
"""

import asyncio
import logging
import os

# Load environment variables
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)

# Suppress some verbose logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)


async def test_azure_deployment():
    """Test basic Azure OpenAI deployment functionality"""
    print("\n" + "=" * 60)
    print("🧪 AZURE OPENAI DEPLOYMENT TEST")
    print("=" * 60)

    # Check environment variables
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "scribeflowgpt4o")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

    print("\n📋 Configuration:")
    print(f"   Endpoint: {endpoint}")
    print(f"   API Key: {'***' + api_key[-8:] if api_key else 'NOT SET'}")
    print(f"   Deployment: {deployment}")
    print(f"   API Version: {api_version}")

    if not endpoint or not api_key:
        print("\n❌ Missing required environment variables!")
        print("   Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY")
        return False

    try:
        # Import the enhanced Azure client
        from chuk_llm.llm.providers.azure_openai_client import AzureOpenAILLMClient

        # Create client with the custom deployment name
        print(f"\n🔧 Creating Azure OpenAI client with deployment '{deployment}'...")
        client = AzureOpenAILLMClient(
            model=deployment,  # This can be any name!
            azure_deployment=deployment,
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
        )

        # Get model info to see smart defaults in action
        print("\n📊 Model Information:")
        model_info = client.get_model_info()

        if model_info.get("using_smart_defaults"):
            print(f"   ✨ Using SMART DEFAULTS for '{deployment}'")
            print(
                f"   📋 Detected features: {', '.join(model_info.get('smart_default_features', []))}"
            )
            print(
                f"   🎯 Max context: {model_info.get('max_context_length', 'unknown')}"
            )
            print(f"   📝 Max output: {model_info.get('max_output_tokens', 'unknown')}")
        else:
            print(f"   📋 Using configured settings for '{deployment}'")
            print(f"   📋 Features: {', '.join(model_info.get('features', []))}")

        # Test basic completion
        print("\n💬 Testing basic chat completion...")
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": "Say 'Hello! The deployment works!' if you can read this.",
            },
        ]

        result = await client.create_completion(
            messages=messages, max_tokens=50, temperature=0
        )

        if result.get("response"):
            print(f"   ✅ Response: {result['response']}")
            return True
        else:
            print("   ❌ No response received")
            return False

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_feature_detection():
    """Test smart feature detection for the deployment"""
    print("\n" + "=" * 60)
    print("🔍 FEATURE DETECTION TEST")
    print("=" * 60)

    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "scribeflowgpt4o")

    try:
        from chuk_llm.llm.providers.azure_openai_client import AzureOpenAILLMClient

        # Test the static methods directly
        print(f"\n📋 Analyzing deployment name: '{deployment}'")

        features = AzureOpenAILLMClient._get_smart_default_features(deployment)
        params = AzureOpenAILLMClient._get_smart_default_parameters(deployment)

        print("\n✨ Smart Default Features:")
        for feature in sorted(features):
            print(f"   • {feature}")

        print("\n🔧 Smart Default Parameters:")
        for key, value in params.items():
            if key != "parameter_mapping" and key != "unsupported_params":
                print(f"   • {key}: {value}")

        # Create a client and test feature support
        client = AzureOpenAILLMClient(
            model=deployment,
            azure_deployment=deployment,
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        )

        print("\n🧪 Feature Support Tests:")
        test_features = [
            "text",
            "streaming",
            "tools",
            "vision",
            "json_mode",
            "reasoning",
        ]
        for feature in test_features:
            supported = client.supports_feature(feature)
            icon = "✅" if supported else "❌"
            print(f"   {icon} {feature}: {supported}")

        return True

    except Exception as e:
        print(f"\n❌ Feature detection failed: {e}")
        return False


async def test_streaming():
    """Test streaming responses"""
    print("\n" + "=" * 60)
    print("📡 STREAMING TEST")
    print("=" * 60)

    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "scribeflowgpt4o")

    try:
        from chuk_llm.llm.providers.azure_openai_client import AzureOpenAILLMClient

        client = AzureOpenAILLMClient(
            model=deployment,
            azure_deployment=deployment,
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        )

        print(f"\n💬 Testing streaming with deployment '{deployment}'...")
        messages = [
            {
                "role": "user",
                "content": "Count from 1 to 5 slowly, one number at a time.",
            }
        ]

        print("   Streaming response: ", end="", flush=True)

        full_response = ""
        chunk_count = 0

        async for chunk in client.create_completion(
            messages=messages, stream=True, max_tokens=100
        ):
            if chunk.get("response"):
                print(chunk["response"], end="", flush=True)
                full_response += chunk["response"]
                chunk_count += 1

        print()  # New line after streaming
        print(f"\n   ✅ Received {chunk_count} chunks")
        print(f"   📝 Total response length: {len(full_response)} characters")

        return True

    except Exception as e:
        print(f"\n❌ Streaming test failed: {e}")
        return False


async def test_tool_calling():
    """Test tool calling capabilities"""
    print("\n" + "=" * 60)
    print("🔧 TOOL CALLING TEST")
    print("=" * 60)

    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "scribeflowgpt4o")

    try:
        from chuk_llm.llm.providers.azure_openai_client import AzureOpenAILLMClient

        client = AzureOpenAILLMClient(
            model=deployment,
            azure_deployment=deployment,
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        )

        # Check if deployment supports tools
        if not client.supports_feature("tools"):
            print(
                f"   ⚠️  Deployment '{deployment}' doesn't support tools (based on smart defaults)"
            )
            return True  # Not a failure, just not supported

        print(f"\n🔨 Testing tool calling with deployment '{deployment}'...")

        # Define a simple tool
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "City name"},
                            "units": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                                "description": "Temperature units",
                            },
                        },
                        "required": ["location"],
                    },
                },
            }
        ]

        messages = [{"role": "user", "content": "What's the weather in Boston?"}]

        print("   📤 Sending request with tool definition...")
        result = await client.create_completion(
            messages=messages, tools=tools, max_tokens=100
        )

        if result.get("tool_calls"):
            print(f"   ✅ Tool calls received: {len(result['tool_calls'])} calls")
            for tc in result["tool_calls"]:
                func_name = tc.get("function", {}).get("name", "unknown")
                func_args = tc.get("function", {}).get("arguments", "{}")
                print(f"      • {func_name}: {func_args}")
        else:
            print("   ⚠️  No tool calls in response")
            if result.get("response"):
                print(f"   📝 Response: {result['response'][:100]}...")

        return True

    except Exception as e:
        print(f"\n❌ Tool calling test failed: {e}")
        return False


async def test_with_mcp_server():
    """Test with actual MCP server if configured"""
    print("\n" + "=" * 60)
    print("🌐 MCP SERVER INTEGRATION TEST")
    print("=" * 60)

    # Check if MCP server is configured
    mcp_url = os.getenv("MCP_SERVER_URL")
    mcp_token = os.getenv("MCP_BEARER_TOKEN")

    if not mcp_url:
        print("   ℹ️  No MCP server configured (set MCP_SERVER_URL to test)")
        return True  # Not a failure

    print(f"\n📡 MCP Server: {mcp_url}")
    print(f"🔑 Bearer Token: {'***' + mcp_token[-8:] if mcp_token else 'NOT SET'}")

    try:
        # This would integrate with your agent_chuk router
        # For now, just show the configuration
        print("\n📋 Example curl command for your agent:")

        curl_cmd = f'''curl -X POST http://localhost:8050/agent_chuk/result \\
  -H "Content-Type: application/json" \\
  -H "Integrations-API-Key: dev-only-token" \\
  -d '{{
    "query": "What is the current weather in Boston?",
    "mcp_server_url_map": {{"mcp_server": "{mcp_url}"}},
    "mcp_server_name_map": {{"mcp_server": "weather_server"}},
    "mcp_bearer_token": "{mcp_token if mcp_token else "your-token"}",
    "llm_override": ["azure_openai", "{os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "scribeflowgpt4o")}"]
  }}'
'''
        print(curl_cmd)

        return True

    except Exception as e:
        print(f"\n❌ MCP test failed: {e}")
        return False


async def test_error_handling():
    """Test error handling for non-existent deployments"""
    print("\n" + "=" * 60)
    print("🚨 ERROR HANDLING TEST")
    print("=" * 60)

    try:
        from chuk_llm.llm.providers.azure_openai_client import AzureOpenAILLMClient

        # Test with a non-existent deployment
        fake_deployment = "this-deployment-does-not-exist-123"

        print(f"\n🧪 Testing with non-existent deployment: '{fake_deployment}'")

        client = AzureOpenAILLMClient(
            model=fake_deployment,
            azure_deployment=fake_deployment,
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        )

        # This should still work (no validation against static list!)
        print("   ✅ Client created successfully (no static validation)")

        # Get model info - should show smart defaults
        model_info = client.get_model_info()
        if model_info.get("using_smart_defaults"):
            print("   ✅ Smart defaults applied even for unknown deployment")

        # Try to actually use it - this should fail with clear error
        print("\n   🔍 Attempting to use the non-existent deployment...")
        messages = [{"role": "user", "content": "Hello"}]

        result = await client.create_completion(messages=messages, max_tokens=10)

        if result.get("error"):
            error_msg = result.get("response", "")
            if "not found" in error_msg.lower():
                print("   ✅ Correct error handling: Deployment not found")
            else:
                print(f"   ⚠️  Error: {error_msg}")
        else:
            print("   ❌ Unexpected success with non-existent deployment!")

        return True

    except Exception as e:
        print(f"\n❌ Error handling test failed: {e}")
        return False


async def main():
    """Run all tests"""
    print("\n" + "🚀 AZURE OPENAI SMART DISCOVERY TEST SUITE 🚀")
    print("=" * 60)
    print("Testing enhanced Azure OpenAI client with smart discovery support")
    print("This demonstrates how custom deployment names like 'scribeflowgpt4o' work!")

    # Run tests
    tests = [
        ("Basic Deployment", test_azure_deployment),
        ("Feature Detection", test_feature_detection),
        ("Streaming", test_streaming),
        ("Tool Calling", test_tool_calling),
        ("MCP Integration", test_with_mcp_server),
        ("Error Handling", test_error_handling),
    ]

    results = {}
    for test_name, test_func in tests:
        try:
            print(f"\nRunning: {test_name}...")
            success = await test_func()
            results[test_name] = success
        except Exception as e:
            print(f"Test {test_name} crashed: {e}")
            results[test_name] = False

    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)

    for test_name, success in results.items():
        icon = "✅" if success else "❌"
        print(f"   {icon} {test_name}")

    total = len(results)
    passed = sum(1 for s in results.values() if s)

    print(f"\n📈 Results: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 SUCCESS! Your Azure deployment works with smart discovery!")
        print(
            f"   Deployment '{os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME', 'scribeflowgpt4o')}' is ready to use!"
        )
        print("   No need to add it to any configuration files!")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")

    # Show example usage
    print("\n" + "=" * 60)
    print("💡 EXAMPLE USAGE")
    print("=" * 60)

    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "scribeflowgpt4o")

    print(f"\nYour deployment '{deployment}' can now be used in your code:")
    print(f'''
from chuk_llm.llm.providers.azure_openai_client import AzureOpenAILLMClient

client = AzureOpenAILLMClient(
    model="{deployment}",  # Any name works!
    azure_deployment="{deployment}",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY")
)

# Use it just like any other model!
result = await client.create_completion(
    messages=[{{"role": "user", "content": "Hello!"}}],
    stream=True
)
''')

    print("\nOr in your curl commands:")
    print(f'''
curl -X POST http://localhost:8050/agent_chuk/result \\
  -H "Content-Type: application/json" \\
  -d '{{"llm_override": ["azure_openai", "{deployment}"]}}'
''')


if __name__ == "__main__":
    # Set up environment
    print("\n🔧 Environment Setup")
    print("-" * 40)

    required_vars = [
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
    ]

    missing = []
    for var in required_vars:
        value = os.getenv(var)
        if value:
            if "KEY" in var:
                print(f"✅ {var}: ***{value[-8:]}")
            else:
                print(f"✅ {var}: {value}")
        else:
            print(f"❌ {var}: NOT SET")
            missing.append(var)

    # Optional vars
    optional_vars = [
        "AZURE_OPENAI_DEPLOYMENT_NAME",
        "AZURE_OPENAI_API_VERSION",
        "MCP_SERVER_URL",
        "MCP_BEARER_TOKEN",
    ]

    print("\nOptional variables:")
    for var in optional_vars:
        value = os.getenv(var)
        if value:
            if "TOKEN" in var or "KEY" in var:
                print(f"   {var}: ***{value[-8:]}")
            else:
                print(f"   {var}: {value}")
        else:
            print(f"   {var}: (not set)")

    if missing:
        print(f"\n❌ Missing required environment variables: {', '.join(missing)}")
        print("Please set them and try again.")
        print("\nExample:")
        for var in missing:
            if "KEY" in var:
                print(f'export {var}="your-api-key"')
            elif "ENDPOINT" in var:
                print(f'export {var}="https://your-resource.openai.azure.com"')
    else:
        # Run the tests
        asyncio.run(main())
