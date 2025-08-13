#!/usr/bin/env python3
"""
Azure OpenAI Custom Deployment Test - LLM Focus
================================================

Tests Azure OpenAI custom deployments at the LLM level:
1. Direct client creation (lowest level)
2. Client factory usage (get_client)

Reads configuration from .env file.
"""

import asyncio
import json
import os
import sys
from typing import Optional, Dict, Any
from datetime import datetime
from pathlib import Path

# Load environment variables from .env file
def load_env_file(env_path: str = ".env"):
    """Load environment variables from .env file"""
    env_file = Path(env_path)
    if env_file.exists():
        print(f"📋 Loading environment from: {env_file.absolute()}")
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    if '=' in line:
                        key, value = line.split('=', 1)
                        # Remove quotes if present
                        value = value.strip().strip('"').strip("'")
                        os.environ[key.strip()] = value
        print("✅ Environment loaded successfully\n")
    else:
        print(f"⚠️ No .env file found at {env_file.absolute()}\n")

# Load .env file first
load_env_file()

# ============================================================================
# Configuration (from environment)
# ============================================================================

# Azure OpenAI Configuration
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "scribeflowgpt4o")

# Test multiple deployments if specified
TEST_DEPLOYMENTS = os.getenv("TEST_DEPLOYMENTS", AZURE_DEPLOYMENT).split(",")

# Validate required variables
missing_vars = []
if not AZURE_ENDPOINT:
    missing_vars.append("AZURE_OPENAI_ENDPOINT")
if not AZURE_API_KEY:
    missing_vars.append("AZURE_OPENAI_API_KEY")

if missing_vars:
    print("❌ Missing required environment variables:")
    for var in missing_vars:
        print(f"   - {var}")
    print("\n📝 Please set these in your .env file:")
    print("   AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/")
    print("   AZURE_OPENAI_API_KEY=your-api-key")
    print("   AZURE_OPENAI_DEPLOYMENT=scribeflowgpt4o  # optional")
    print("   TEST_DEPLOYMENTS=deployment1,deployment2  # optional")
    sys.exit(1)

# ============================================================================
# Utility Functions
# ============================================================================

def print_header(title: str, emoji: str = "🔷"):
    """Print a formatted section header"""
    print(f"\n{emoji} {title}")
    print("=" * 70)

def print_success(message: str):
    """Print a success message"""
    print(f"✅ {message}")

def print_error(message: str):
    """Print an error message"""
    print(f"❌ {message}")

def print_info(message: str):
    """Print an info message"""
    print(f"ℹ️  {message}")

# ============================================================================
# Test 1: Direct Client Testing
# ============================================================================

async def test_direct_client(deployment_name: str) -> bool:
    """Test direct client creation - lowest level"""
    print_header(f"Test 1: Direct Client - {deployment_name}", "1️⃣")
    
    try:
        from chuk_llm.llm.providers.azure_openai_client import AzureOpenAILLMClient
        
        print(f"Creating direct client for deployment: {deployment_name}")
        
        # Create client directly
        client = AzureOpenAILLMClient(
            model=deployment_name,
            api_key=AZURE_API_KEY,
            azure_endpoint=AZURE_ENDPOINT,
            api_version=AZURE_API_VERSION
        )
        
        # Test basic completion
        messages = [{"role": "user", "content": "Say 'Hello from Azure deployment!'"}]
        response = await client.create_completion(messages, max_tokens=50)
        
        if response.get('response'):
            print_success(f"Response: {response['response']}")
            
            # Get model info to see smart defaults
            model_info = client.get_model_info()
            print("\n📊 Model Info:")
            print(f"  • Provider: {model_info.get('provider')}")
            print(f"  • Model: {model_info.get('model')}")
            print(f"  • Using Smart Defaults: {model_info.get('using_smart_defaults', False)}")
            print(f"  • Features: {model_info.get('smart_default_features', [])}")
            print(f"  • Max Context: {model_info.get('max_context_length', 'N/A'):,}")
            print(f"  • Max Output: {model_info.get('max_output_tokens', 'N/A'):,}")
            
            # Test with tools
            print("\n🔧 Testing tool support...")
            tools = [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"}
                        }
                    }
                }
            }]
            
            tool_messages = [{"role": "user", "content": "What's the weather in Seattle?"}]
            tool_response = await client.create_completion(tool_messages, tools=tools, max_tokens=100)
            
            if tool_response.get('tool_calls'):
                print_success(f"Tool calls supported: {len(tool_response['tool_calls'])} calls")
            else:
                print_info("No tool calls in response (model may not support tools)")
            
            await client.close()
            return True
        else:
            print_error("No response from direct client")
            return False
            
    except ImportError as e:
        print_error(f"ChukLLM not installed: {e}")
        print_info("Install with: pip install chuk-llm")
        return False
    except Exception as e:
        print_error(f"Direct client failed: {e}")
        return False

# ============================================================================
# Test 2: Client Factory Testing
# ============================================================================

async def test_client_factory(deployment_name: str) -> bool:
    """Test client factory - uses get_client"""
    print_header(f"Test 2: Client Factory - {deployment_name}", "2️⃣")
    
    try:
        from chuk_llm.llm.client import get_client
        
        print(f"Creating client via factory for deployment: {deployment_name}")
        
        # Test with factory
        client = get_client(
            provider="azure_openai",
            model=deployment_name,
            api_key=AZURE_API_KEY,
            azure_endpoint=AZURE_ENDPOINT,
            api_version=AZURE_API_VERSION
        )
        
        # Test completion
        messages = [{"role": "user", "content": "Say 'Factory works with custom deployment!'"}]
        response = await client.create_completion(messages, max_tokens=50)
        
        if response.get('response'):
            print_success(f"Response: {response['response']}")
            
            # Test streaming
            print("\n⚡ Testing streaming...")
            stream_messages = [{"role": "user", "content": "Count to 5 slowly"}]
            chunks_received = 0
            
            async for chunk in client.create_completion(stream_messages, stream=True, max_tokens=100):
                if chunk.get('response'):
                    chunks_received += 1
                    print(".", end="", flush=True)
            
            print()
            if chunks_received > 0:
                print_success(f"Streaming works: {chunks_received} chunks received")
            else:
                print_info("No streaming chunks received")
            
            return True
        else:
            print_error("No response from factory client")
            return False
            
    except ValueError as e:
        if "not available for provider" in str(e):
            print_error(f"Validation error: {e}")
            print_info("This means get_client is still validating against a list")
            print_info("Fix needed in chuk_llm/llm/client.py")
        else:
            print_error(f"Factory error: {e}")
        return False
    except ImportError as e:
        print_error(f"ChukLLM not installed: {e}")
        return False
    except Exception as e:
        print_error(f"Factory client failed: {e}")
        return False

# ============================================================================
# Main Test Runner
# ============================================================================

async def test_deployment(deployment_name: str) -> Dict[str, bool]:
    """Run all tests for a deployment"""
    print(f"\n{'=' * 70}")
    print(f" 🚀 Testing Deployment: {deployment_name}")
    print(f"{'=' * 70}")
    
    results = {}
    
    # Test 1: Direct client
    results['direct_client'] = await test_direct_client(deployment_name)
    
    # Test 2: Client factory
    results['client_factory'] = await test_client_factory(deployment_name)
    
    return results

async def main():
    """Main test orchestration"""
    print("=" * 70)
    print(" 🎯 Azure OpenAI Custom Deployment LLM Tests")
    print("=" * 70)
    print(f"Endpoint: {AZURE_ENDPOINT}")
    print(f"API Version: {AZURE_API_VERSION}")
    print(f"Deployments to test: {', '.join(TEST_DEPLOYMENTS)}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    all_results = {}
    
    for deployment_name in TEST_DEPLOYMENTS:
        deployment_name = deployment_name.strip()
        results = await test_deployment(deployment_name)
        all_results[deployment_name] = results
    
    # Summary
    print(f"\n{'=' * 70}")
    print(" 📊 Test Results Summary")
    print(f"{'=' * 70}")
    
    for deployment_name, results in all_results.items():
        print(f"\n🔹 {deployment_name}:")
        
        tests = [
            ("Direct Client", 'direct_client'),
            ("Client Factory", 'client_factory')
        ]
        
        for test_name, key in tests:
            status = "✅ PASS" if results.get(key, False) else "❌ FAIL"
            print(f"  {status} - {test_name}")
    
    # Overall status
    all_passed = all(all(results.values()) for results in all_results.values())
    
    print(f"\n{'=' * 70}")
    if all_passed:
        print_success(" All tests passed! Custom deployments work correctly.")
        print("\n Next steps:")
        print(" • Your Azure custom deployments are fully functional in ChukLLM")
        print(" • You can use them with: get_client('azure_openai', model='your-deployment')")
    else:
        print(" ⚠️ Some tests failed:")
        
        for deployment_name, results in all_results.items():
            if not results.get('direct_client'):
                print(f"\n • Direct client failed for {deployment_name}")
                print("   Check: Azure credentials and endpoint")
            if not results.get('client_factory'):
                print(f"\n • Factory failed for {deployment_name}")
                print("   Fix needed: Update get_client() in chuk_llm/llm/client.py")
                print("   The function should skip validation for azure_openai provider")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()