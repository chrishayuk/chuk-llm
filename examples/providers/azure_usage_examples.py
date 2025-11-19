#!/usr/bin/env python3
# examples/azure_openai_usage_examples.py
"""
Azure OpenAI Provider Example Usage Script
==========================================

Demonstrates all the features of the Azure OpenAI provider in the chuk-llm library.
Run this script to see Azure-hosted GPT models in action with various capabilities.

Requirements:
- pip install openai chuk-llm
- Set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT environment variables
- Configure your Azure OpenAI deployments

Usage:
    python azure_openai_example.py
    python azure_openai_example.py --deployment my-gpt4-deployment
    python azure_openai_example.py --endpoint https://myresource.openai.azure.com
"""

import argparse
import asyncio
import base64
import os
import sys
import time

# dotenv
from dotenv import load_dotenv

try:
    import httpx
except ImportError:
    httpx = None

# load environment variables
load_dotenv()

# Ensure we have the required environment
if not os.getenv("AZURE_OPENAI_API_KEY"):
    print("❌ Please set AZURE_OPENAI_API_KEY environment variable")
    print("   export AZURE_OPENAI_API_KEY='your_azure_api_key_here'")
    sys.exit(1)

if not os.getenv("AZURE_OPENAI_ENDPOINT"):
    print("❌ Please set AZURE_OPENAI_ENDPOINT environment variable")
    print("   export AZURE_OPENAI_ENDPOINT='https://your-resource.openai.azure.com'")
    sys.exit(1)

try:
    from chuk_llm.configuration import Feature, get_config
    from chuk_llm.llm.client import get_client, get_provider_info
    from chuk_llm.core.models import Message, Tool, ToolFunction, TextContent, ImageUrlContent, ToolCall, FunctionCall
    from chuk_llm.core.enums import MessageRole, ContentType, ToolType
    from chuk_llm.llm.discovery.azure_openai_discoverer import AzureOpenAIModelDiscoverer
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("   Please make sure you're running from the chuk-llm directory")
    sys.exit(1)


def create_test_image(color: str = "red", size: int = 15) -> str:
    """Create a test image as base64 - tries PIL first, fallback to hardcoded"""
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
        print("⚠️  PIL not available, using fallback image")
        # Fallback: 15x15 red square (valid PNG)
        return "iVBORw0KGgoAAAANSUhEUgAAAA8AAAAPCAYAAAA71pVKAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAAdgAAAHYBTnsmCAAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAABYSURBVCiRY2RgYGBkYGBgZGBgYGRgYGBkYGBgZGBgYGRgYGBkYGBgZGBgYGRgYGBkYGBgZGBgYGRgYGBkYGBgZGBgYGRgYGBkYGBgZGBgYGRgYGBgZGBgYGAAAgAANgAOAUUe1wAAAABJRU5ErkJggg=="


async def get_available_deployments():
    """Get available Azure OpenAI deployments using the discovery system"""
    config = get_config()
    configured_deployments = []
    discovered_deployments = []

    # Get configured deployments
    if "azure_openai" in config.providers:
        provider = config.providers["azure_openai"]
        if hasattr(provider, "models"):
            configured_deployments = list(provider.models)

    # Use discovery system to find deployments from Azure API
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")

    if endpoint and api_key:
        try:
            discoverer = AzureOpenAIModelDiscoverer(
                provider_name="azure_openai",
                api_key=api_key,
                azure_endpoint=endpoint,
                api_version="2024-02-01",
            )
            models_data = await discoverer.discover_models()
            discovered_deployments = [m.get("name") for m in models_data if m.get("name")]
        except Exception as e:
            print(f"⚠️  Could not fetch deployments from Azure API: {e}")

    # Combine deployments (configured first, then discovered)
    all_deployments = list(configured_deployments)
    for deployment in discovered_deployments:
        if deployment not in all_deployments:
            all_deployments.append(deployment)

    return {
        "configured": configured_deployments,
        "discovered": discovered_deployments,
        "all": all_deployments,
    }


async def find_working_deployments():
    """Find deployments that actually work by testing them"""
    print("   Testing common deployment names...")

    # Common deployment names to try
    candidates = [
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-4",
        "gpt-4-turbo",
        "gpt-35-turbo",
        "gpt-4.1",
        "gpt-4.1-mini",
    ]

    working_deployments = []

    for deployment in candidates:
        try:
            client = get_client(
                "azure_openai",
                model=deployment,
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            )
            messages = [Message(role=MessageRole.USER, content="test")]
            response = await client.create_completion(messages, max_tokens=1)

            # If we get here without error, deployment works
            if "Error" not in response.get("response", ""):
                working_deployments.append(deployment)
                print(f"   ✅ {deployment}")
        except Exception as e:
            # Skip deployments that don't exist
            pass

    return working_deployments


# =============================================================================
# Example 1: Basic Azure OpenAI Setup
# =============================================================================


async def azure_setup_example(deployment: str = "gpt-4o-mini"):
    """Azure OpenAI setup and basic configuration test"""
    print(f"\n🔧 Azure OpenAI Setup Test with deployment: {deployment}")
    print("=" * 60)

    try:
        # Test basic client creation with Azure-specific parameters
        client = get_client(
            "azure_openai",
            model=deployment,
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2024-02-01",
        )

        # Get client info
        info = client.get_model_info()

        print("✅ Azure OpenAI Client Created Successfully!")
        print(f"   Endpoint: {info.get('azure_specific', {}).get('endpoint', 'N/A')}")
        print(
            f"   Deployment: {info.get('azure_specific', {}).get('deployment', 'N/A')}"
        )
        print(
            f"   API Version: {info.get('azure_specific', {}).get('api_version', 'N/A')}"
        )
        print(
            f"   Auth Type: {info.get('azure_specific', {}).get('authentication_type', 'N/A')}"
        )

        return client

    except Exception as e:
        print(f"❌ Setup failed: {e}")
        return None


# =============================================================================
# Example 2: Basic Text Completion with Azure
# =============================================================================


async def azure_text_example(deployment: str = "gpt-4o-mini"):
    """Basic text completion with Azure OpenAI"""
    print(f"\n🤖 Azure Text Completion with {deployment}")
    print("=" * 60)

    client = get_client(
        "azure_openai",
        model=deployment,
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    )

    messages = [
        Message(
            role=MessageRole.SYSTEM,
            content="You are a helpful AI assistant running on Azure OpenAI.",
        ),
        Message(
            role=MessageRole.USER,
            content="Explain the benefits of using Azure OpenAI vs regular OpenAI (2-3 sentences).",
        ),
    ]

    start_time = time.time()
    response = await client.create_completion(messages)
    duration = time.time() - start_time

    print(f"✅ Azure Response ({duration:.2f}s):")
    print(f"   {response['response']}")

    return response


# =============================================================================
# Example 3: Azure Streaming
# =============================================================================


async def azure_streaming_example(deployment: str = "gpt-4o-mini"):
    """Real-time streaming with Azure OpenAI"""
    print(f"\n⚡ Azure Streaming Example with {deployment}")
    print("=" * 60)

    # Check streaming support
    config = get_config()
    if not config.supports_feature("azure_openai", Feature.STREAMING, deployment):
        print(f"⚠️  Deployment {deployment} doesn't support streaming")
        return None

    client = get_client(
        "azure_openai",
        model=deployment,
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    )

    messages = [
        Message(
            role=MessageRole.USER,
            content="Write a short poem about cloud computing and Azure.",
        )
    ]

    print("🌊 Streaming response from Azure:")
    print("   ", end="", flush=True)

    start_time = time.time()
    full_response = ""

    async for chunk in client.create_completion(messages, stream=True):
        if chunk.get("response"):
            content = chunk["response"]
            print(content, end="", flush=True)
            full_response += content

    duration = time.time() - start_time
    print(f"\n✅ Azure streaming completed ({duration:.2f}s)")

    return full_response


# =============================================================================
# Example 4: Azure Function Calling
# =============================================================================


async def azure_function_calling_example(deployment: str = "gpt-4o-mini"):
    """Function calling with Azure OpenAI"""
    print(f"\n🔧 Azure Function Calling with {deployment}")
    print("=" * 60)

    # Check if deployment supports tools
    config = get_config()
    if not config.supports_feature("azure_openai", Feature.TOOLS, deployment):
        print(
            f"⚠️  Skipping function calling: Deployment {deployment} doesn't support tools"
        )
        return None

    client = get_client(
        "azure_openai",
        model=deployment,
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    )

    # Define Azure-specific tools
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_azure_service_info",
                "description": "Get information about Azure services",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "service_name": {
                            "type": "string",
                            "description": "Name of the Azure service",
                        },
                        "include_pricing": {
                            "type": "boolean",
                            "description": "Whether to include pricing information",
                        },
                    },
                    "required": ["service_name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "calculate_azure_costs",
                "description": "Calculate estimated Azure costs",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "service": {
                            "type": "string",
                            "description": "Azure service name",
                        },
                        "usage_hours": {
                            "type": "number",
                            "description": "Hours of usage per month",
                        },
                        "region": {"type": "string", "description": "Azure region"},
                    },
                    "required": ["service", "usage_hours"],
                },
            },
        },
    ]

    messages = [
        Message(
            role=MessageRole.USER,
            content="Tell me about Azure OpenAI service and calculate costs for 100 hours of usage in East US region.",
        )
    ]

    print("🔄 Making Azure function calling request...")
    response = await client.create_completion(messages, tools=tools)

    if response.get("tool_calls"):
        print(f"✅ Tool calls requested: {len(response['tool_calls'])}")
        for i, tool_call in enumerate(response["tool_calls"], 1):
            func_name = tool_call["function"]["name"]
            func_args = tool_call["function"]["arguments"]
            print(f"   {i}. {func_name}({func_args})")

        # Simulate tool execution with Azure-specific responses
        tool_calls_list = [
            ToolCall(
                id=tc["id"],
                type=ToolType.FUNCTION,
                function=FunctionCall(
                    name=tc["function"]["name"],
                    arguments=tc["function"]["arguments"]
                )
            )
            for tc in response["tool_calls"]
        ]
        messages.append(
            Message(role=MessageRole.ASSISTANT, content="", tool_calls=tool_calls_list)
        )

        # Add mock tool results
        for tool_call in response["tool_calls"]:
            func_name = tool_call["function"]["name"]

            if func_name == "get_azure_service_info":
                result = '{"service": "Azure OpenAI", "description": "Fully managed OpenAI models", "features": ["GPT models", "DALL-E", "Whisper", "Enterprise security"], "regions": ["East US", "West Europe", "Southeast Asia"]}'
            elif func_name == "calculate_azure_costs":
                result = '{"estimated_cost": "$45.50", "breakdown": {"OpenAI API calls": "$40.00", "Data processing": "$5.50"}, "region": "East US", "currency": "USD"}'
            else:
                result = '{"status": "success", "message": "Azure operation completed"}'

            messages.append(
                Message(
                    role=MessageRole.TOOL,
                    tool_call_id=tool_call["id"],
                    name=func_name,
                    content=result,
                )
            )

        # Get final response
        print("🔄 Getting final Azure response...")
        final_response = await client.create_completion(messages)
        print("✅ Final Azure response:")
        print(f"   {final_response['response']}")

        return final_response
    else:
        print("ℹ️  No tool calls were made")
        print(f"   Response: {response['response']}")
        return response


# =============================================================================
# Example 5: Azure Vision with Custom Deployment
# =============================================================================


async def azure_vision_example(deployment: str = "gpt-4o"):
    """Vision capabilities with Azure OpenAI vision models"""
    print(f"\n👁️  Azure Vision Example with {deployment}")
    print("=" * 60)

    # Check if deployment supports vision
    config = get_config()
    if not config.supports_feature("azure_openai", Feature.VISION, deployment):
        print(f"⚠️  Skipping vision: Deployment {deployment} doesn't support vision")
        print("💡 Try a vision-capable deployment like: gpt-4o, gpt-4-turbo")
        return None

    client = get_client(
        "azure_openai",
        model=deployment,
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    )

    # Create a test image
    print("🖼️  Creating test image...")
    test_image = create_test_image("azure", 20)  # Azure blue

    messages = [
        Message(
            role=MessageRole.USER,
            content=[
                TextContent(
                    type=ContentType.TEXT,
                    text="What color is this square? Also mention that this is being processed by Azure OpenAI.",
                ),
                ImageUrlContent(
                    type=ContentType.IMAGE_URL,
                    image_url={"url": f"data:image/png;base64,{test_image}"},
                ),
            ],
        )
    ]

    print("👀 Analyzing image with Azure OpenAI...")
    response = await client.create_completion(messages, max_tokens=100)

    print("✅ Azure vision response:")
    print(f"   {response['response']}")

    return response


# =============================================================================
# Example 6: Azure JSON Mode
# =============================================================================


async def azure_json_mode_example(deployment: str = "gpt-4o-mini"):
    """JSON mode example with Azure OpenAI"""
    print(f"\n📋 Azure JSON Mode Example with {deployment}")
    print("=" * 60)

    # Check JSON mode support
    config = get_config()
    if not config.supports_feature("azure_openai", Feature.JSON_MODE, deployment):
        print(f"⚠️  Deployment {deployment} doesn't support JSON mode")
        return None

    client = get_client(
        "azure_openai",
        model=deployment,
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    )

    messages = [
        Message(
            role=MessageRole.SYSTEM,
            content="You are a helpful assistant designed to output JSON. Generate information about Azure services.",
        ),
        Message(
            role=MessageRole.USER,
            content="Tell me about Azure OpenAI service in JSON format with fields: name, description, key_features (array), pricing_model, and azure_regions (array).",
        ),
    ]

    print("📝 Requesting JSON output from Azure...")

    try:
        response = await client.create_completion(
            messages, response_format={"type": "json_object"}, temperature=0.7
        )

        print("✅ Azure JSON response:")
        print(f"   {response['response']}")

        # Try to parse as JSON to verify
        import json

        try:
            parsed = json.loads(response["response"])
            print(f"✅ Valid JSON structure with keys: {list(parsed.keys())}")
        except json.JSONDecodeError:
            print("⚠️  Response is not valid JSON")

    except Exception as e:
        print(f"❌ Azure JSON mode failed: {e}")
        # Fallback to regular request
        response = await client.create_completion(messages)
        print(f"📝 Fallback response: {response['response'][:200]}...")

    return response


# =============================================================================
# Example 7: Azure Deployment Comparison
# =============================================================================


async def azure_deployment_comparison():
    """Compare different Azure OpenAI deployments"""
    print("\n📊 Azure Deployment Comparison")
    print("=" * 60)

    # Common deployments (customize based on your Azure setup)
    deployments = ["gpt-4.1-mini", "gpt-4.1", "gpt-4o-mini", "gpt-4o", "gpt-4-turbo"]

    prompt = "What is Azure OpenAI? (One sentence)"
    results = {}

    for deployment in deployments:
        try:
            print(f"🔄 Testing Azure deployment {deployment}...")
            client = get_client(
                "azure_openai",
                model=deployment,
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            )
            messages = [Message(role=MessageRole.USER, content=prompt)]

            start_time = time.time()
            response = await client.create_completion(messages)
            duration = time.time() - start_time

            results[deployment] = {
                "response": response.get("response", ""),
                "time": duration,
                "length": len(response.get("response", "")),
                "success": True,
            }

        except Exception as e:
            results[deployment] = {
                "response": f"Error: {str(e)}",
                "time": 0,
                "length": 0,
                "success": False,
            }

    print("\n📈 Azure Deployment Results:")
    for deployment, result in results.items():
        status = "✅" if result["success"] else "❌"
        print(f"   {status} {deployment}:")
        print(f"      Time: {result['time']:.2f}s")
        print(f"      Length: {result['length']} chars")
        print(f"      Response: {result['response'][:80]}...")
        print()

    return results


# =============================================================================
# Example 8: Azure Authentication Methods
# =============================================================================


async def azure_auth_methods_example(deployment: str = "gpt-4o-mini"):
    """Demonstrate different Azure authentication methods"""
    print("\n🔐 Azure Authentication Methods")
    print("=" * 60)

    auth_methods = []

    # Method 1: API Key (most common)
    if os.getenv("AZURE_OPENAI_API_KEY"):
        try:
            client = get_client(
                "azure_openai",
                model=deployment,
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            )

            messages = [Message(role=MessageRole.USER, content="Hello from API key auth!")]
            response = await client.create_completion(messages)

            auth_methods.append(
                {
                    "method": "API Key",
                    "status": "✅ Success",
                    "response_length": len(response.get("response", "")),
                }
            )
        except Exception as e:
            auth_methods.append(
                {"method": "API Key", "status": f"❌ Failed: {e}", "response_length": 0}
            )

    # Method 2: Azure AD Token (if available)
    azure_ad_token = os.getenv("AZURE_AD_TOKEN")
    if azure_ad_token:
        try:
            client = get_client(
                "azure_openai",
                model=deployment,
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                azure_ad_token=azure_ad_token,
            )

            messages = [Message(role=MessageRole.USER, content="Hello from Azure AD token auth!")]
            response = await client.create_completion(messages)

            auth_methods.append(
                {
                    "method": "Azure AD Token",
                    "status": "✅ Success",
                    "response_length": len(response.get("response", "")),
                }
            )
        except Exception as e:
            auth_methods.append(
                {
                    "method": "Azure AD Token",
                    "status": f"❌ Failed: {e}",
                    "response_length": 0,
                }
            )

    print("🔐 Authentication Results:")
    for auth in auth_methods:
        print(f"   {auth['method']}: {auth['status']}")
        if auth["response_length"] > 0:
            print(f"      Response length: {auth['response_length']} chars")

    if not auth_methods:
        print("   ⚠️  No authentication methods available")
        print("   💡 Set AZURE_OPENAI_API_KEY or AZURE_AD_TOKEN")

    return auth_methods


# =============================================================================
# Example 9: Deployment Discovery
# =============================================================================


async def deployment_discovery_example():
    """Discover available Azure OpenAI deployments using discovery system"""
    print("\n🔍 Azure Deployment Discovery")
    print("=" * 60)

    deployment_info = await get_available_deployments()

    print(f"📦 Configured deployments ({len(deployment_info['configured'])}):")
    for deployment in deployment_info["configured"][:10]:  # Show first 10
        # Identify model families from deployment names
        if "gpt-4o" in deployment.lower():
            print(f"   • {deployment} [🚀 GPT-4o - multimodal flagship]")
        elif "gpt-4" in deployment.lower() and "turbo" in deployment.lower():
            print(f"   • {deployment} [⚡ GPT-4 Turbo - fast & capable]")
        elif "gpt-4.1" in deployment.lower():
            print(f"   • {deployment} [🔄 GPT-4.1 - latest generation]")
        elif "gpt-4" in deployment.lower():
            print(f"   • {deployment} [🧠 GPT-4 - powerful reasoning]")
        elif "gpt-3.5" in deployment.lower():
            print(f"   • {deployment} [💨 GPT-3.5 - fast & efficient]")
        else:
            print(f"   • {deployment}")

    if len(deployment_info["discovered"]) > 0:
        print(f"\n🌐 Discovered from Azure API ({len(deployment_info['discovered'])}):")
        # Show deployments that are not in config
        new_deployments = [
            d
            for d in deployment_info["discovered"]
            if d not in deployment_info["configured"]
        ]
        if new_deployments:
            print("   New deployments not in config:")
            for deployment in new_deployments[:5]:  # Show first 5
                print(f"   ✨ {deployment}")
        else:
            print("   All Azure deployments are already configured")

    print(f"\n📊 Total available: {len(deployment_info['all'])} deployments")

    # Highlight Azure-specific benefits
    print("\n☁️  Azure OpenAI Benefits:")
    print("   • Enterprise-grade security and compliance")
    print("   • Private network connectivity with VNet")
    print("   • Data residency and regional availability")
    print("   • Integration with Azure ecosystem")

    # Test a discovered deployment if available
    if deployment_info["all"] and len(deployment_info["all"]) > 0:
        test_deployment = deployment_info["all"][0]
        print(f"\n🧪 Testing deployment: {test_deployment}")
        try:
            client = get_client(
                "azure_openai",
                model=test_deployment,
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            )
            messages = [
                Message(role=MessageRole.USER, content="Say hello from Azure")
            ]
            response = await client.create_completion(messages, max_tokens=20)
            print(f"   ✅ Deployment works: {response['response'][:50]}...")
        except Exception as e:
            print(f"   ⚠️ Deployment test failed: {e}")

    return deployment_info


# =============================================================================
# Example 10: Model Comparison
# =============================================================================


async def model_comparison_example(working_deps=None):
    """Compare different Azure OpenAI deployments side-by-side"""
    print("\n⚖️  Azure Model Comparison")
    print("=" * 60)

    # Use working deployments if provided, otherwise discover
    if working_deps:
        available = working_deps[:3]  # Use up to 3 working deployments
        print(f"Comparing {len(available)} working deployments")
    else:
        try:
            available = await find_working_deployments()
            available = available[:3]

            if not available:
                print("⚠️  No working deployments found")
                return []

            print(f"Comparing {len(available)} working deployments")
        except Exception as e:
            print(f"⚠️  Could not find deployments: {e}")
            return []

    prompt = "What is quantum computing? (2 sentences)"
    results = []

    for model in available:
        try:
            print(f"🔄 Testing {model}...")
            client = get_client(
                "azure_openai",
                model=model,
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            )
            messages = [Message(role=MessageRole.USER, content=prompt)]

            start_time = time.time()
            response = await client.create_completion(messages, max_tokens=100)
            duration = time.time() - start_time

            results.append({
                "model": model,
                "response": response.get("response", ""),
                "time": duration,
                "length": len(response.get("response", "")),
            })
            print(f"   ✅ {model}: {duration:.2f}s")

        except Exception as e:
            print(f"   ⚠️ {model} failed: {str(e)[:100]}")
            results.append({
                "model": model,
                "response": f"Error: {str(e)[:100]}",
                "time": 0,
                "length": 0,
            })

    print("\n📊 Comparison Results:")
    for result in results:
        print(f"\n🤖 {result['model']}:")
        print(f"   ⏱️  Time: {result['time']:.2f}s")
        print(f"   📏 Length: {result['length']} chars")
        if result['length'] > 0:
            print(f"   💬 Response: {result['response'][:100]}...")

    return results


# =============================================================================
# Example 11: Context Window Test
# =============================================================================


async def context_window_test(deployment: str = "gpt-4o-mini"):
    """Test Azure's large context window handling"""
    print(f"\n🪟 Azure Context Window Test with {deployment}")
    print("=" * 60)

    client = get_client(
        "azure_openai",
        model=deployment,
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    )

    # Generate a large context (~4500 words)
    long_text = "The quick brown fox jumps over the lazy dog. " * 500

    messages = [
        Message(
            role=MessageRole.SYSTEM,
            content=f"Here is a document to analyze:\n\n{long_text}\n\nThis document contains a repeated phrase.",
        ),
        Message(
            role=MessageRole.USER,
            content="How many times does the word 'fox' appear in the document? Just give me the number.",
        ),
    ]

    print(f"📄 Testing with ~{len(long_text.split())} words of context...")

    start_time = time.time()
    response = await client.create_completion(messages, max_tokens=150)
    duration = time.time() - start_time

    print(f"✅ Azure processed large context in {duration:.2f}s")
    print(f"📝 Response: {response['response']}")

    return response


# =============================================================================
# Example 12: Dynamic Model Test
# =============================================================================


async def dynamic_model_test(working_deps=None):
    """Test working deployments to prove library flexibility"""
    print("\n🔄 Dynamic Model Test")
    print("=" * 60)

    # Use working deployments
    if working_deps and len(working_deps) > 0:
        # Use the first working deployment
        dynamic_model = working_deps[0]
        print(f"Testing working deployment: {dynamic_model}")
    else:
        print("⚠️  No working deployments available")
        return None

    try:
        client = get_client("azure_openai", model=dynamic_model)
        messages = [
            Message(
                role=MessageRole.USER,
                content="Say hello in exactly one creative word"
            )
        ]

        response = await client.create_completion(messages, max_tokens=10)
        print(f"   ✅ Deployment works: {response['response']}")
        print(f"   Model: {dynamic_model}")

        return response

    except Exception as e:
        print(f"   ⚠️ Test failed: {str(e)[:100]}")
        return None


# =============================================================================
# Example 13: Parallel Processing Test
# =============================================================================


async def parallel_processing_test(deployment: str = "gpt-4o-mini"):
    """Test parallel request processing with Azure"""
    print(f"\n⚡ Azure Parallel Processing Test with {deployment}")
    print("=" * 60)

    client = get_client(
        "azure_openai",
        model=deployment,
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    )

    prompts = [
        "What is machine learning? (1 sentence)",
        "What is deep learning? (1 sentence)",
        "What is neural network? (1 sentence)",
    ]

    # Sequential processing
    print("🔄 Testing sequential processing...")
    sequential_start = time.time()
    sequential_results = []
    for prompt in prompts:
        messages = [Message(role=MessageRole.USER, content=prompt)]
        response = await client.create_completion(messages, max_tokens=50)
        sequential_results.append(response)
    sequential_time = time.time() - sequential_start

    # Parallel processing
    print("⚡ Testing parallel processing...")
    parallel_start = time.time()

    async def process_prompt(prompt):
        messages = [Message(role=MessageRole.USER, content=prompt)]
        return await client.create_completion(messages, max_tokens=50)

    parallel_results = await asyncio.gather(*[process_prompt(p) for p in prompts])
    parallel_time = time.time() - parallel_start

    # Calculate speedup
    speedup = sequential_time / parallel_time if parallel_time > 0 else 0

    print(f"\n📊 Azure Processing Results:")
    print(f"   Sequential: {sequential_time:.2f}s")
    print(f"   Parallel: {parallel_time:.2f}s")
    print(f"   Speedup: {speedup:.2f}x")

    return {
        "sequential_time": sequential_time,
        "parallel_time": parallel_time,
        "speedup": speedup,
        "results": parallel_results,
    }


# =============================================================================
# Main Function
# =============================================================================


async def main():
    """Run all Azure OpenAI examples"""
    parser = argparse.ArgumentParser(description="Azure OpenAI Provider Example Script")
    parser.add_argument(
        "--deployment",
        default="gpt-4o-mini",
        help="Azure deployment name (default: gpt-4o-mini)",
    )
    parser.add_argument("--endpoint", help="Azure OpenAI endpoint (overrides env var)")
    parser.add_argument("--api-version", default="2024-02-01", help="Azure API version")
    parser.add_argument(
        "--skip-vision", action="store_true", help="Skip vision examples"
    )
    parser.add_argument(
        "--skip-functions", action="store_true", help="Skip function calling"
    )
    parser.add_argument("--quick", action="store_true", help="Run only basic examples")

    args = parser.parse_args()

    if args.endpoint:
        os.environ["AZURE_OPENAI_ENDPOINT"] = args.endpoint

    print("🚀 Azure OpenAI Provider Examples")
    print("=" * 60)
    print(f"Using deployment: {args.deployment}")
    print(f"Endpoint: {os.getenv('AZURE_OPENAI_ENDPOINT', 'Not set')}")
    print(f"API Version: {args.api_version}")
    print(f"API Key: {'✅ Set' if os.getenv('AZURE_OPENAI_API_KEY') else '❌ Missing'}")

    # Show deployment capabilities
    try:
        config = get_config()
        supports_tools = config.supports_feature(
            "azure_openai", Feature.TOOLS, args.deployment
        )
        supports_vision = config.supports_feature(
            "azure_openai", Feature.VISION, args.deployment
        )
        supports_streaming = config.supports_feature(
            "azure_openai", Feature.STREAMING, args.deployment
        )
        supports_json = config.supports_feature(
            "azure_openai", Feature.JSON_MODE, args.deployment
        )

        print("Deployment capabilities:")
        print(f"  Tools: {'✅' if supports_tools else '❌'}")
        print(f"  Vision: {'✅' if supports_vision else '❌'}")
        print(f"  Streaming: {'✅' if supports_streaming else '❌'}")
        print(f"  JSON Mode: {'✅' if supports_json else '❌'}")

    except Exception as e:
        print(f"⚠️  Could not check capabilities: {e}")

    # IMPORTANT: Find actual working deployments
    print("\n🔍 Finding working Azure deployments...")
    try:
        working_deployments = await find_working_deployments()

        if not working_deployments:
            print(f"\n❌ NO WORKING DEPLOYMENTS FOUND!")
            print(f"")
            print(f"📌 To use Azure OpenAI, you need to create deployments:")
            print(f"   1. Go to Azure Portal (portal.azure.com)")
            print(f"   2. Navigate to your Azure OpenAI resource")
            print(f"   3. Go to 'Model deployments' section")
            print(f"   4. Click 'Create new deployment'")
            print(f"   5. Choose a model (e.g., gpt-4o-mini) and give it a name")
            print(f"")
            print(f"💡 Tested deployment names: gpt-4o-mini, gpt-4o, gpt-4, gpt-4-turbo, gpt-35-turbo")
            print(f"")
            sys.exit(1)

        print(f"\n✅ Found {len(working_deployments)} working deployment(s)!")
        actual_deployment = working_deployments[0]

        # Get vision deployment if available
        vision_deployments = [d for d in working_deployments if "gpt-4o" in d.lower() or "turbo" in d.lower()]
        vision_deployment = vision_deployments[0] if vision_deployments else actual_deployment

        has_deployment = True
    except Exception as e:
        print(f"⚠️  Deployment check failed: {e}")
        print(f"\nℹ️  Using default deployment '{args.deployment}' (may not work)")
        actual_deployment = args.deployment
        vision_deployment = "gpt-4o"
        has_deployment = False
        working_deployments = []

    examples = [
        ("Azure Setup", lambda: azure_setup_example(actual_deployment)),
        ("Deployment Discovery", deployment_discovery_example),
        ("Azure Text", lambda: azure_text_example(actual_deployment)),
        ("Azure Streaming", lambda: azure_streaming_example(actual_deployment)),
        ("Azure JSON Mode", lambda: azure_json_mode_example(actual_deployment)),
        ("Azure Auth Methods", lambda: azure_auth_methods_example(actual_deployment)),
    ]

    if not args.quick:
        if not args.skip_functions:
            examples.append(
                (
                    "Azure Function Calling",
                    lambda: azure_function_calling_example(actual_deployment),
                )
            )

        if not args.skip_vision:
            examples.append(("Azure Vision", lambda: azure_vision_example(vision_deployment)))

        examples.extend([
            ("Model Comparison", lambda: model_comparison_example(working_deployments)),
            ("Context Window Test", lambda: context_window_test(actual_deployment)),
            ("Parallel Processing", lambda: parallel_processing_test(actual_deployment)),
            ("Dynamic Model Test", lambda: dynamic_model_test(working_deployments)),
        ])

    # Run examples
    results = {}
    for name, example_func in examples:
        try:
            print("\n" + "=" * 60)
            start_time = time.time()
            result = await example_func()
            duration = time.time() - start_time
            results[name] = {"success": True, "result": result, "time": duration}
            print(f"✅ {name} completed in {duration:.2f}s")
        except Exception as e:
            results[name] = {"success": False, "error": str(e), "time": 0}
            print(f"❌ {name} failed: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("📊 AZURE OPENAI SUMMARY")
    print("=" * 60)

    successful = sum(1 for r in results.values() if r["success"])
    total = len(results)
    total_time = sum(r["time"] for r in results.values())

    print(f"✅ Successful: {successful}/{total}")
    print(f"⏱️  Total time: {total_time:.2f}s")
    print(f"🌐 Endpoint: {os.getenv('AZURE_OPENAI_ENDPOINT', 'Not configured')}")
    print(f"🎯 Primary deployment: {actual_deployment}")

    if has_deployment and working_deployments:
        print(f"📦 Working deployments: {', '.join(working_deployments)}")
    elif not has_deployment:
        print(f"\n⚠️  WARNING: No actual deployments found in Azure resource!")
        print(f"   Create a deployment in Azure Portal to use Azure OpenAI.")

    for name, result in results.items():
        status = "✅" if result["success"] else "❌"
        time_str = f"{result['time']:.2f}s" if result["success"] else "failed"
        print(f"   {status} {name}: {time_str}")

    if successful == total:
        print("\n🎉 All Azure OpenAI examples completed successfully!")
        print("🔗 Azure OpenAI provider is working perfectly with chuk-llm!")
        print(f"✨ Features tested: {actual_deployment} capabilities on Azure")
    else:
        print("\n⚠️  Some examples failed. Check your Azure configuration.")

        # Show Azure-specific recommendations
        print("\n💡 Azure Setup Recommendations:")
        print("   • Endpoint: Set AZURE_OPENAI_ENDPOINT to your resource URL")
        print("   • API Key: Set AZURE_OPENAI_API_KEY from Azure portal")
        print("   • Deployments: Ensure your models are deployed in Azure")
        print("   • Regions: Use supported regions for better performance")
        print("   • Quotas: Check your Azure OpenAI quota limits")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Azure OpenAI examples cancelled by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)
