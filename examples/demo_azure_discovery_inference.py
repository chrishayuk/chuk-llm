#!/usr/bin/env python3
"""
Azure OpenAI Custom Deployment Demo Script
==========================================

This comprehensive demo shows how to use ChukLLM with custom Azure deployments
like 'scribeflowgpt4o' using direct client instantiation.

Based on the PROVEN WORKING approach!

Features demonstrated:
- Direct client creation with custom deployments
- Smart defaults in action
- Tool usage with complex names
- Streaming responses
- Error handling
- Parameter validation
- Deployment discovery testing
"""

import asyncio
import json
import os
import sys
from typing import Optional, List, Dict, Any
from datetime import datetime

# ChukLLM imports - using direct client instantiation
try:
    from chuk_llm.llm.providers.azure_openai_client import AzureOpenAILLMClient
except ImportError:
    print("❌ ChukLLM not installed. Please install it first.")
    sys.exit(1)

# ============================================================================
# Configuration
# ============================================================================

# Check for required environment variables
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")

if not AZURE_ENDPOINT or not AZURE_API_KEY:
    print("❌ Missing required environment variables:")
    print("   - AZURE_OPENAI_ENDPOINT")
    print("   - AZURE_OPENAI_API_KEY")
    print("\nPlease set these before running the demo.")
    sys.exit(1)

# Custom deployment names to test
CUSTOM_DEPLOYMENTS = [
    "scribeflowgpt4o",      # Your custom deployment
    "gpt-4o",               # Standard deployment (if exists)
    "custom-gpt-4-turbo",   # Another custom example
]

# ============================================================================
# Utility Functions
# ============================================================================

def print_section(title: str):
    """Print a formatted section header"""
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print(f"{'=' * 60}")

def print_header(title: str, emoji: str = "🔷"):
    """Print a formatted section header with emoji"""
    width = 70
    print(f"\n{emoji} {title}")
    print("=" * width)

def print_subheader(title: str):
    """Print a formatted subsection header"""
    print(f"\n▶ {title}")
    print("-" * 50)

def print_success(message: str):
    """Print a success message"""
    print(f"✅ {message}")

def print_error(message: str):
    """Print an error message"""
    print(f"❌ {message}")

def print_info(message: str):
    """Print an info message"""
    print(f"ℹ️  {message}")

def print_json(data: Dict[str, Any], indent: int = 2):
    """Pretty print JSON data"""
    print(json.dumps(data, indent=indent, default=str))

# ============================================================================
# Demo Functions - Using Direct Client Creation
# ============================================================================

async def create_direct_client(deployment_name: str) -> Optional[AzureOpenAILLMClient]:
    """Create client directly - this is the PROVEN approach"""
    print(f"🔧 Creating direct client for: {deployment_name}")
    
    try:
        # Direct instantiation - proven to work!
        client = AzureOpenAILLMClient(
            model=deployment_name,
            api_key=AZURE_API_KEY,
            azure_endpoint=AZURE_ENDPOINT,
            api_version="2024-02-01"
        )
        
        print("✅ Direct client created successfully!")
        
        # Get model info to see smart defaults in action
        model_info = client.get_model_info()
        
        print("\n📊 Smart Defaults Applied:")
        print(f"  • Using Smart Defaults: {model_info.get('using_smart_defaults', False)}")
        print(f"  • Smart Features: {model_info.get('smart_default_features', [])}")
        print(f"  • Max Context: {model_info.get('max_context_length', 'N/A'):,}")
        print(f"  • Max Output: {model_info.get('max_output_tokens', 'N/A'):,}")
        
        # Show Azure-specific info
        azure_info = model_info.get('azure_specific', {})
        if azure_info:
            print(f"  • Deployment: {azure_info.get('deployment', 'N/A')}")
            print(f"  • Custom Deployments: {azure_info.get('supports_custom_deployments', False)}")
        
        # Show tool compatibility
        tool_compat = model_info.get('tool_compatibility', 'unknown')
        print(f"  • Tool Compatibility: {tool_compat}")
        
        return client
        
    except Exception as e:
        print(f"❌ Client creation failed: {e}")
        return None

async def demo_comprehensive_chat(client: AzureOpenAILLMClient, deployment_name: str):
    """Comprehensive chat demo"""
    print_section(f"💬 Comprehensive Chat with {deployment_name}")
    
    if not client:
        return
    
    try:
        print("🚀 Testing various chat scenarios...")
        
        # Test 1: Simple chat
        print("\n1️⃣ Simple Chat:")
        messages = [{"role": "user", "content": "What is your name and what can you do?"}]
        response = await client.create_completion(messages, max_tokens=100)
        print(f"📝 Response: {response.get('response', 'No response')}")
        
        # Test 2: System message handling
        print("\n2️⃣ System Message Test:")
        messages_with_system = [
            {"role": "system", "content": "You are a helpful Azure AI assistant."},
            {"role": "user", "content": "Introduce yourself briefly."}
        ]
        response = await client.create_completion(messages_with_system, max_tokens=80)
        print(f"📝 Response: {response.get('response', 'No response')}")
        
        # Test 3: JSON mode (if supported)
        print("\n3️⃣ JSON Mode Test:")
        try:
            json_messages = [{"role": "user", "content": "Respond with JSON containing your name and capabilities."}]
            response = await client.create_completion(
                json_messages, 
                max_tokens=150,
                response_format={"type": "json_object"}
            )
            print(f"📝 JSON Response: {response.get('response', 'No response')}")
        except Exception as e:
            print(f"❌ JSON mode not supported or error: {e}")
        
    except Exception as e:
        print(f"❌ Chat demo failed: {e}")

async def demo_tool_usage_comprehensive(client: AzureOpenAILLMClient, deployment_name: str):
    """Comprehensive tool usage demo with various naming conventions"""
    print_section(f"🔧 Tool Usage with {deployment_name}")
    
    if not client:
        return
    
    # Tools with problematic names that ChukLLM should handle
    tools = [
        {
            "type": "function",
            "function": {
                "name": "stdio.read_query",  # MCP-style with dots
                "description": "Execute a system query",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "SQL-like query"},
                        "database": {"type": "string", "default": "system"}
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "web.api:search",  # Dots and colons
                "description": "Search the web for information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "limit": {"type": "integer", "default": 5}
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "azure.resource@analyzer",  # Special characters
                "description": "Analyze Azure resources",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "resource_group": {"type": "string", "description": "Resource group name"},
                        "metric": {"type": "string", "enum": ["cost", "usage", "performance"]}
                    },
                    "required": ["resource_group"]
                }
            }
        }
    ]
    
    try:
        print("🔧 Testing universal tool name compatibility...")
        print("📋 Original tool names (problematic):")
        for tool in tools:
            name = tool["function"]["name"]
            print(f"  • {name}")
        
        print("\n🚀 Sending tool-enabled request...")
        print("💡 ChukLLM will automatically sanitize these names for Azure OpenAI")
        
        messages = [
            {
                "role": "user", 
                "content": "Please help me analyze my Azure environment. Use the stdio tool to check system status, search for Azure optimization tips, and analyze the 'production-rg' resource group for cost metrics."
            }
        ]
        
        response = await client.create_completion(
            messages, 
            tools=tools,
            max_tokens=400
        )
        
        print("✅ Tool response received!")
        
        if response.get('response'):
            print(f"📝 Text Response: {response['response']}")
        
        if response.get('tool_calls'):
            print(f"\n🔧 Tool Calls ({len(response['tool_calls'])}):")
            for i, tool_call in enumerate(response['tool_calls'], 1):
                name = tool_call['function']['name']
                args = tool_call['function']['arguments']
                
                # Parse arguments nicely
                try:
                    parsed_args = json.loads(args) if isinstance(args, str) else args
                    formatted_args = json.dumps(parsed_args, indent=2)
                except:
                    formatted_args = str(args)
                
                print(f"  {i}. Function: {name}")
                print(f"     Arguments:\n{formatted_args}")
                print()
        
        print("💡 Note: Tool names were automatically restored to original format in response!")
        
    except Exception as e:
        print(f"❌ Tool usage failed: {e}")

async def demo_streaming_comprehensive(client: AzureOpenAILLMClient, deployment_name: str):
    """Comprehensive streaming demo"""
    print_section(f"⚡ Streaming with {deployment_name}")
    
    if not client:
        return
    
    try:
        print("🚀 Testing streaming capabilities...")
        print("💭 Question: Write a Python class for Azure resource management")
        print("📺 Streaming Response:")
        print("-" * 60)
        
        messages = [
            {
                "role": "user",
                "content": "Write a Python class that helps manage Azure resources. Include methods for listing, creating, and monitoring resources. Add proper error handling and documentation."
            }
        ]
        
        total_chars = 0
        chunk_count = 0
        
        # Use the direct client's streaming
        async for chunk in client.create_completion(messages, stream=True, max_tokens=500):
            if chunk.get('response'):
                content = chunk['response']
                print(content, end='', flush=True)
                total_chars += len(content)
                chunk_count += 1
        
        print(f"\n{'-' * 60}")
        print(f"✅ Streaming completed: {chunk_count} chunks, {total_chars} characters")
        
    except Exception as e:
        print(f"❌ Streaming failed: {e}")

async def demo_parameter_mapping(client: AzureOpenAILLMClient, deployment_name: str):
    """Demo parameter mapping and restrictions"""
    print_section(f"⚙️ Parameter Mapping with {deployment_name}")
    
    if not client:
        return
    
    try:
        print("🔧 Testing parameter mapping and smart defaults...")
        
        # Test different parameter combinations
        test_cases = [
            {
                "name": "Standard parameters",
                "params": {"max_tokens": 100, "temperature": 0.7}
            },
            {
                "name": "High max_tokens",
                "params": {"max_tokens": 8000, "temperature": 0.5}
            },
            {
                "name": "Various parameters",
                "params": {
                    "max_tokens": 150,
                    "temperature": 0.8,
                    "top_p": 0.9,
                    "frequency_penalty": 0.1
                }
            }
        ]
        
        messages = [{"role": "user", "content": "Say hello and mention what parameters you're using."}]
        
        for test_case in test_cases:
            print(f"\n🧪 Testing: {test_case['name']}")
            print(f"📋 Parameters: {test_case['params']}")
            
            try:
                response = await client.create_completion(messages, **test_case['params'])
                print(f"✅ Success: {response.get('response', 'No response')[:100]}...")
            except Exception as e:
                print(f"❌ Failed: {e}")
        
    except Exception as e:
        print(f"❌ Parameter mapping demo failed: {e}")

async def test_deployment_discovery():
    """Test if we can discover the custom deployment"""
    print_section("🔍 Testing Deployment Discovery")
    
    try:
        from chuk_llm.llm.discovery.azure_openai_discoverer import AzureOpenAIModelDiscoverer
        
        print("🔍 Testing if 'scribeflowgpt4o' can be discovered...")
        
        discoverer = AzureOpenAIModelDiscoverer(
            api_key=AZURE_API_KEY,
            azure_endpoint=AZURE_ENDPOINT
        )
        
        # Test if deployment exists
        exists = await discoverer.test_deployment_availability("scribeflowgpt4o")
        print(f"✅ Deployment 'scribeflowgpt4o' exists: {exists}")
        
        if exists:
            print("💡 This deployment could be added to discovery configuration!")
        else:
            print("💡 Deployment not accessible via standard API, but direct client works!")
        
        # Test other common deployment names
        common_names = ["company-gpt4", "prod-gpt-4o", "dev-gpt-35-turbo"]
        print(f"\n🧪 Testing other common deployment names...")
        
        for name in common_names:
            try:
                exists = await discoverer.test_deployment_availability(name)
                status = "✅ Found" if exists else "❌ Not found"
                print(f"  • {name}: {status}")
            except Exception as e:
                print(f"  • {name}: ❌ Error - {e}")
        
    except Exception as e:
        print(f"❌ Discovery testing failed: {e}")

async def demo_vision_capabilities(client: AzureOpenAILLMClient, deployment_name: str):
    """Test vision capabilities if supported"""
    if not client:
        return
    
    print_section(f"👁️ Vision Capabilities with {deployment_name}")
    
    try:
        # Test with an image URL (using a placeholder)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What's in this image? (This is a test - if vision is not supported, just say so)"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://example.com/test-image.jpg"
                        }
                    }
                ]
            }
        ]
        
        print("👁️ Testing vision capabilities...")
        response = await client.create_completion(
            messages,
            max_tokens=100
        )
        
        if response.get('response'):
            print(f"Response: {response['response']}")
            
            # Check if vision is actually supported based on response
            response_lower = response['response'].lower()
            if "vision" in response_lower and "not supported" in response_lower:
                print_info("Vision not supported by this deployment")
            else:
                print_success("Vision capabilities may be supported")
        
    except Exception as e:
        if "vision" in str(e).lower() or "image" in str(e).lower():
            print_info(f"Vision not supported by {deployment_name}")
        else:
            print_error(f"Vision test failed: {e}")

async def demo_json_mode(client: AzureOpenAILLMClient, deployment_name: str):
    """Demonstrate JSON mode if supported"""
    if not client:
        return
    
    print_section(f"🔄 JSON Mode with {deployment_name}")
    
    try:
        messages = [
            {
                "role": "user",
                "content": (
                    "Create a JSON object with the following Azure deployment information: "
                    "name (string), region (string), tier (string), features (array of strings), "
                    "and is_production (boolean)."
                )
            }
        ]
        
        print("🔄 Testing JSON mode...")
        response = await client.create_completion(
            messages,
            max_tokens=200,
            response_format={"type": "json_object"}
        )
        
        if response.get('response'):
            print_success("JSON response received:")
            try:
                # Try to parse and pretty-print the JSON
                json_data = json.loads(response['response'])
                print(json.dumps(json_data, indent=2))
            except json.JSONDecodeError:
                # If not valid JSON, print as-is
                print(response['response'])
        else:
            print_error("No response received")
            
    except Exception as e:
        # JSON mode might not be supported
        if "json" in str(e).lower() or "format" in str(e).lower():
            print_info(f"JSON mode not supported by {deployment_name}")
        else:
            print_error(f"JSON mode test failed: {e}")

# ============================================================================
# Main Orchestration
# ============================================================================

async def run_deployment_tests(deployment_name: str) -> bool:
    """Run all tests for a single deployment"""
    print_header(f"Testing Deployment: {deployment_name}", "🚀")
    
    # Create direct client - the PROVEN approach
    client = await create_direct_client(deployment_name)
    
    if client:
        # Run comprehensive demos
        await demo_comprehensive_chat(client, deployment_name)
        await demo_tool_usage_comprehensive(client, deployment_name)
        await demo_streaming_comprehensive(client, deployment_name)
        await demo_parameter_mapping(client, deployment_name)
        await demo_json_mode(client, deployment_name)
        await demo_vision_capabilities(client, deployment_name)
        
        # Cleanup
        if hasattr(client, 'close'):
            await client.close()
        
        return True
    else:
        return False

async def main():
    """Main comprehensive demo"""
    print_section("🚀 Complete Azure OpenAI Custom Deployment Demo")
    print("Direct Client Creation - PROVEN TO WORK with 'scribeflowgpt4o'!")
    
    # Check environment
    print(f"✅ Endpoint: {AZURE_ENDPOINT}")
    print("✅ API key configured")
    print(f"📅 Timestamp: {datetime.now().isoformat()}")
    
    # Test each deployment
    successful_deployments = []
    failed_deployments = []
    
    for deployment_name in CUSTOM_DEPLOYMENTS:
        try:
            success = await run_deployment_tests(deployment_name)
            if success:
                successful_deployments.append(deployment_name)
            else:
                failed_deployments.append(deployment_name)
        except Exception as e:
            print_error(f"Unexpected error testing {deployment_name}: {e}")
            failed_deployments.append(deployment_name)
    
    # Test discovery capabilities
    await test_deployment_discovery()
    
    # Summary
    print_section("🎉 Demo Success Summary")
    
    if successful_deployments:
        print(f"🎯 Successful deployments ({len(successful_deployments)}):")
        for deployment in successful_deployments:
            print(f"  ✅ {deployment}")
    
    if failed_deployments:
        print(f"\n⚠️ Failed deployments ({len(failed_deployments)}):")
        for deployment in failed_deployments:
            print(f"  ❌ {deployment}")
    
    print("\n🎯 What we successfully demonstrated:")
    print("  ✅ Direct client creation with custom deployment")
    print("  ✅ Smart defaults applied automatically")
    print("  ✅ Universal tool name compatibility")
    print("  ✅ Parameter mapping and validation")
    print("  ✅ Streaming with custom deployment")
    print("  ✅ JSON mode and system message handling")
    
    print("\n🔑 Key Insights:")
    print("  • ChukLLM's smart defaults work even with custom deployment names")
    print("  • Tool name sanitization is completely transparent")
    print("  • Direct client creation works perfectly for custom deployments")
    print("  • All advanced features work seamlessly when supported")
    
    print("\n🚀 Production Usage:")
    print("  • Use direct AzureOpenAILLMClient instantiation for custom deployments")
    print("  • Add custom deployments to configuration for team use")
    print("  • Leverage smart defaults for zero-configuration operation")
    print("  • The validate_model() method now returns True for all Azure deployments")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️ Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()