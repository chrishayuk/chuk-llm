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

import asyncio
import argparse
import os
import sys
import time
import base64
from typing import Dict, Any, List

# dotenv
from dotenv import load_dotenv

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
    from chuk_llm.llm.client import get_client, get_provider_info
    from chuk_llm.configuration import get_config, Feature
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("   Please make sure you're running from the chuk-llm directory")
    sys.exit(1)

def create_test_image(color: str = "red", size: int = 15) -> str:
    """Create a test image as base64 - tries PIL first, fallback to hardcoded"""
    try:
        from PIL import Image
        import io
        
        # Create a colored square
        img = Image.new('RGB', (size, size), color)
        
        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return img_data
    except ImportError:
        print("⚠️  PIL not available, using fallback image")
        # Fallback: 15x15 red square (valid PNG)
        return "iVBORw0KGgoAAAANSUhEUgAAAA8AAAAPCAYAAAA71pVKAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAAdgAAAHYBTnsmCAAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAABYSURBVCiRY2RgYGBkYGBgZGBgYGRgYGBkYGBgZGBgYGRgYGBkYGBgZGBgYGRgYGBkYGBgZGBgYGRgYGBkYGBgZGBgYGRgYGBkYGBgZGBgYGRgYGBgZGBgYGAAAgAANgAOAUUe1wAAAABJRU5ErkJggg=="

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
            api_version="2024-02-01"
        )
        
        # Get client info
        info = client.get_model_info()
        
        print("✅ Azure OpenAI Client Created Successfully!")
        print(f"   Endpoint: {info.get('azure_specific', {}).get('endpoint', 'N/A')}")
        print(f"   Deployment: {info.get('azure_specific', {}).get('deployment', 'N/A')}")
        print(f"   API Version: {info.get('azure_specific', {}).get('api_version', 'N/A')}")
        print(f"   Auth Type: {info.get('azure_specific', {}).get('authentication_type', 'N/A')}")
        
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
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant running on Azure OpenAI."},
        {"role": "user", "content": "Explain the benefits of using Azure OpenAI vs regular OpenAI (2-3 sentences)."}
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
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    
    messages = [
        {"role": "user", "content": "Write a short poem about cloud computing and Azure."}
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
        print(f"⚠️  Skipping function calling: Deployment {deployment} doesn't support tools")
        return None
    
    client = get_client(
        "azure_openai",
        model=deployment,
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
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
                            "description": "Name of the Azure service"
                        },
                        "include_pricing": {
                            "type": "boolean",
                            "description": "Whether to include pricing information"
                        }
                    },
                    "required": ["service_name"]
                }
            }
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
                            "description": "Azure service name"
                        },
                        "usage_hours": {
                            "type": "number",
                            "description": "Hours of usage per month"
                        },
                        "region": {
                            "type": "string",
                            "description": "Azure region"
                        }
                    },
                    "required": ["service", "usage_hours"]
                }
            }
        }
    ]
    
    messages = [
        {"role": "user", "content": "Tell me about Azure OpenAI service and calculate costs for 100 hours of usage in East US region."}
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
        messages.append({
            "role": "assistant",
            "content": "",
            "tool_calls": response["tool_calls"]
        })
        
        # Add mock tool results
        for tool_call in response["tool_calls"]:
            func_name = tool_call["function"]["name"]
            
            if func_name == "get_azure_service_info":
                result = '{"service": "Azure OpenAI", "description": "Fully managed OpenAI models", "features": ["GPT models", "DALL-E", "Whisper", "Enterprise security"], "regions": ["East US", "West Europe", "Southeast Asia"]}'
            elif func_name == "calculate_azure_costs":
                result = '{"estimated_cost": "$45.50", "breakdown": {"OpenAI API calls": "$40.00", "Data processing": "$5.50"}, "region": "East US", "currency": "USD"}'
            else:
                result = '{"status": "success", "message": "Azure operation completed"}'
            
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "name": func_name,
                "content": result
            })
        
        # Get final response
        print("🔄 Getting final Azure response...")
        final_response = await client.create_completion(messages)
        print(f"✅ Final Azure response:")
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
        print(f"💡 Try a vision-capable deployment like: gpt-4o, gpt-4-turbo")
        return None
    
    client = get_client(
        "azure_openai",
        model=deployment,
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    
    # Create a test image
    print("🖼️  Creating test image...")
    test_image = create_test_image("azure", 20)  # Azure blue
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What color is this square? Also mention that this is being processed by Azure OpenAI."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{test_image}"
                    }
                }
            ]
        }
    ]
    
    print("👀 Analyzing image with Azure OpenAI...")
    response = await client.create_completion(messages, max_tokens=100)
    
    print(f"✅ Azure vision response:")
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
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    
    messages = [
        {
            "role": "system", 
            "content": "You are a helpful assistant designed to output JSON. Generate information about Azure services."
        },
        {
            "role": "user", 
            "content": "Tell me about Azure OpenAI service in JSON format with fields: name, description, key_features (array), pricing_model, and azure_regions (array)."
        }
    ]
    
    print("📝 Requesting JSON output from Azure...")
    
    try:
        response = await client.create_completion(
            messages,
            response_format={"type": "json_object"},
            temperature=0.7
        )
        
        print(f"✅ Azure JSON response:")
        print(f"   {response['response']}")
        
        # Try to parse as JSON to verify
        import json
        try:
            parsed = json.loads(response['response'])
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
    print(f"\n📊 Azure Deployment Comparison")
    print("=" * 60)
    
    # Common deployments (customize based on your Azure setup)
    deployments = [
        "gpt-4.1-mini",
        "gpt-4.1",
        "gpt-4o-mini",
        "gpt-4o", 
        "gpt-4-turbo"
    ]
    
    prompt = "What is Azure OpenAI? (One sentence)"
    results = {}
    
    for deployment in deployments:
        try:
            print(f"🔄 Testing Azure deployment {deployment}...")
            client = get_client(
                "azure_openai",
                model=deployment,
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
            )
            messages = [{"role": "user", "content": prompt}]
            
            start_time = time.time()
            response = await client.create_completion(messages)
            duration = time.time() - start_time
            
            results[deployment] = {
                "response": response.get("response", ""),
                "time": duration,
                "length": len(response.get("response", "")),
                "success": True
            }
            
        except Exception as e:
            results[deployment] = {
                "response": f"Error: {str(e)}",
                "time": 0,
                "length": 0,
                "success": False
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
    print(f"\n🔐 Azure Authentication Methods")
    print("=" * 60)
    
    auth_methods = []
    
    # Method 1: API Key (most common)
    if os.getenv("AZURE_OPENAI_API_KEY"):
        try:
            client = get_client(
                "azure_openai",
                model=deployment,
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY")
            )
            
            messages = [{"role": "user", "content": "Hello from API key auth!"}]
            response = await client.create_completion(messages)
            
            auth_methods.append({
                "method": "API Key",
                "status": "✅ Success",
                "response_length": len(response.get("response", ""))
            })
        except Exception as e:
            auth_methods.append({
                "method": "API Key", 
                "status": f"❌ Failed: {e}",
                "response_length": 0
            })
    
    # Method 2: Azure AD Token (if available)
    azure_ad_token = os.getenv("AZURE_AD_TOKEN")
    if azure_ad_token:
        try:
            client = get_client(
                "azure_openai",
                model=deployment,
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                azure_ad_token=azure_ad_token
            )
            
            messages = [{"role": "user", "content": "Hello from Azure AD token auth!"}]
            response = await client.create_completion(messages)
            
            auth_methods.append({
                "method": "Azure AD Token",
                "status": "✅ Success",
                "response_length": len(response.get("response", ""))
            })
        except Exception as e:
            auth_methods.append({
                "method": "Azure AD Token",
                "status": f"❌ Failed: {e}",
                "response_length": 0
            })
    
    print("🔐 Authentication Results:")
    for auth in auth_methods:
        print(f"   {auth['method']}: {auth['status']}")
        if auth['response_length'] > 0:
            print(f"      Response length: {auth['response_length']} chars")
    
    if not auth_methods:
        print("   ⚠️  No authentication methods available")
        print("   💡 Set AZURE_OPENAI_API_KEY or AZURE_AD_TOKEN")
    
    return auth_methods

# =============================================================================
# Main Function
# =============================================================================

async def main():
    """Run all Azure OpenAI examples"""
    parser = argparse.ArgumentParser(description="Azure OpenAI Provider Example Script")
    parser.add_argument("--deployment", default="gpt-4o-mini", help="Azure deployment name (default: gpt-4o-mini)")
    parser.add_argument("--endpoint", help="Azure OpenAI endpoint (overrides env var)")
    parser.add_argument("--api-version", default="2024-02-01", help="Azure API version")
    parser.add_argument("--skip-vision", action="store_true", help="Skip vision examples")
    parser.add_argument("--skip-functions", action="store_true", help="Skip function calling")
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
        supports_tools = config.supports_feature("azure_openai", Feature.TOOLS, args.deployment)
        supports_vision = config.supports_feature("azure_openai", Feature.VISION, args.deployment)
        supports_streaming = config.supports_feature("azure_openai", Feature.STREAMING, args.deployment)
        supports_json = config.supports_feature("azure_openai", Feature.JSON_MODE, args.deployment)
        
        print(f"Deployment capabilities:")
        print(f"  Tools: {'✅' if supports_tools else '❌'}")
        print(f"  Vision: {'✅' if supports_vision else '❌'}")
        print(f"  Streaming: {'✅' if supports_streaming else '❌'}")
        print(f"  JSON Mode: {'✅' if supports_json else '❌'}")
        
    except Exception as e:
        print(f"⚠️  Could not check capabilities: {e}")

    examples = [
        ("Azure Setup", lambda: azure_setup_example(args.deployment)),
        ("Azure Text", lambda: azure_text_example(args.deployment)),
        ("Azure Streaming", lambda: azure_streaming_example(args.deployment)),
        ("Azure JSON Mode", lambda: azure_json_mode_example(args.deployment)),
        ("Azure Auth Methods", lambda: azure_auth_methods_example(args.deployment)),
    ]
    
    if not args.quick:
        if not args.skip_functions:
            examples.append(("Azure Function Calling", lambda: azure_function_calling_example(args.deployment)))
        
        if not args.skip_vision:
            examples.append(("Azure Vision", lambda: azure_vision_example("gpt-4o")))
        
        examples.append(("Azure Deployment Comparison", azure_deployment_comparison))
    
    # Run examples
    results = {}
    for name, example_func in examples:
        try:
            print(f"\n" + "="*60)
            start_time = time.time()
            result = await example_func()
            duration = time.time() - start_time
            results[name] = {"success": True, "result": result, "time": duration}
            print(f"✅ {name} completed in {duration:.2f}s")
        except Exception as e:
            results[name] = {"success": False, "error": str(e), "time": 0}
            print(f"❌ {name} failed: {e}")
    
    # Summary
    print(f"\n" + "="*60)
    print("📊 AZURE OPENAI SUMMARY")
    print("="*60)
    
    successful = sum(1 for r in results.values() if r["success"])
    total = len(results)
    total_time = sum(r["time"] for r in results.values())
    
    print(f"✅ Successful: {successful}/{total}")
    print(f"⏱️  Total time: {total_time:.2f}s")
    print(f"🌐 Endpoint: {os.getenv('AZURE_OPENAI_ENDPOINT', 'Not configured')}")
    print(f"🎯 Deployment: {args.deployment}")
    
    for name, result in results.items():
        status = "✅" if result["success"] else "❌"
        time_str = f"{result['time']:.2f}s" if result["success"] else "failed"
        print(f"   {status} {name}: {time_str}")
    
    if successful == total:
        print(f"\n🎉 All Azure OpenAI examples completed successfully!")
        print(f"🔗 Azure OpenAI provider is working perfectly with chuk-llm!")
        print(f"✨ Features tested: {args.deployment} capabilities on Azure")
    else:
        print(f"\n⚠️  Some examples failed. Check your Azure configuration.")
        
        # Show Azure-specific recommendations
        print(f"\n💡 Azure Setup Recommendations:")
        print(f"   • Endpoint: Set AZURE_OPENAI_ENDPOINT to your resource URL")
        print(f"   • API Key: Set AZURE_OPENAI_API_KEY from Azure portal")
        print(f"   • Deployments: Ensure your models are deployed in Azure")
        print(f"   • Regions: Use supported regions for better performance")
        print(f"   • Quotas: Check your Azure OpenAI quota limits")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Azure OpenAI examples cancelled by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)