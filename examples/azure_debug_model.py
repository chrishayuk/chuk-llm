#!/usr/bin/env python3
"""
Enhanced debug version of Azure OpenAI discoverer to test specific deployments
"""

import asyncio
import os
import httpx
import json
from pathlib import Path
from typing import Optional, List, Dict, Any

# Load environment variables
try:
    from dotenv import load_dotenv
    env_file = Path(".env")
    if env_file.exists():
        load_dotenv(env_file)
except ImportError:
    pass

class EnhancedAzureDebugger:
    """Enhanced debug version of Azure OpenAI discoverer"""
    
    def __init__(self):
        self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip('/')
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "scribeflowgpt4o")
        
        # Track what we find
        self.working_deployments = []
        self.available_models = []
    
    async def debug_discover_models(self):
        """Debug the model discovery process step by step"""
        
        print("🐛 Enhanced Azure OpenAI Model Discovery")
        print("=" * 60)
        
        if not self.azure_endpoint or not self.api_key:
            print("❌ Missing credentials")
            print(f"   AZURE_OPENAI_ENDPOINT: {self.azure_endpoint or 'NOT SET'}")
            print(f"   AZURE_OPENAI_API_KEY: {'SET' if self.api_key else 'NOT SET'}")
            return
        
        print(f"🔗 Endpoint: {self.azure_endpoint}")
        print(f"🔑 API Key: {'*' * 20}{self.api_key[-8:] if self.api_key else ''}")
        print(f"📅 API Version: {self.api_version}")
        print(f"🎯 Target Deployment: {self.deployment_name}")
        
        # Test various model listing endpoints
        model_endpoints = [
            f"{self.azure_endpoint}/openai/models",
            f"{self.azure_endpoint}/models",
            f"{self.azure_endpoint}/openai/deployments",
            f"{self.azure_endpoint}/deployments"
        ]
        
        headers = {"api-key": self.api_key}
        params = {"api-version": self.api_version}
        
        print(f"\n📡 Testing Model Listing Endpoints:")
        print("-" * 50)
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            for endpoint in model_endpoints:
                print(f"\n🔍 Testing: {endpoint}")
                try:
                    response = await client.get(endpoint, headers=headers, params=params)
                    print(f"   Status: {response.status_code}")
                    
                    if response.status_code == 200:
                        print(f"   ✅ SUCCESS!")
                        try:
                            data = response.json()
                            items = data.get("data", data.get("value", []))
                            if isinstance(items, list):
                                print(f"   📚 Found {len(items)} items")
                                for item in items[:3]:
                                    if isinstance(item, dict):
                                        item_id = item.get("id", item.get("name", "unknown"))
                                        print(f"      • {item_id}")
                                self.available_models.extend(items)
                            else:
                                print(f"   📄 Response type: {type(items)}")
                        except Exception as e:
                            print(f"   ⚠️ Parse error: {e}")
                    elif response.status_code == 404:
                        print(f"   ❌ Not Found")
                    elif response.status_code == 401:
                        print(f"   🔐 Unauthorized")
                    else:
                        print(f"   ❓ Unexpected: {response.status_code}")
                        
                except Exception as e:
                    print(f"   💥 Error: {str(e)[:50]}...")
    
    async def test_specific_deployment(self, deployment_name: str) -> bool:
        """Test if a specific deployment exists and works"""
        
        print(f"\n🎯 Testing Deployment: {deployment_name}")
        print("-" * 40)
        
        # Test chat completion endpoint
        url = f"{self.azure_endpoint}/openai/deployments/{deployment_name}/chat/completions"
        params = {"api-version": self.api_version}
        headers = {
            "api-key": self.api_key,
            "Content-Type": "application/json"
        }
        
        payload = {
            "messages": [{"role": "user", "content": "Say 'yes' if you work"}],
            "max_tokens": 10,
            "temperature": 0
        }
        
        print(f"📡 URL: {url}")
        print(f"📋 Params: {params}")
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, headers=headers, params=params, json=payload)
                
                print(f"📊 Status: {response.status_code}")
                
                if response.status_code == 200:
                    print(f"✅ DEPLOYMENT WORKS!")
                    try:
                        data = response.json()
                        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                        print(f"🤖 Response: {content[:50]}")
                        self.working_deployments.append(deployment_name)
                        return True
                    except Exception as e:
                        print(f"⚠️ Response parse error: {e}")
                        return True  # Still counts as working
                        
                elif response.status_code == 404:
                    print(f"❌ Deployment not found")
                    error_data = response.json() if response.text else {}
                    if error_data.get("error"):
                        print(f"   Error: {error_data['error'].get('message', 'Unknown')}")
                    return False
                    
                elif response.status_code == 401:
                    print(f"🔐 Unauthorized - check API key")
                    return False
                    
                else:
                    print(f"❓ Unexpected status: {response.status_code}")
                    if response.text:
                        print(f"📄 Response: {response.text[:200]}")
                    return False
                    
        except Exception as e:
            print(f"💥 Request failed: {e}")
            return False
    
    async def discover_deployments_by_pattern(self):
        """Try to discover deployments using common patterns"""
        
        print(f"\n🔍 Discovering Deployments by Pattern Testing")
        print("=" * 50)
        
        # Common deployment patterns to test
        deployment_patterns = [
            self.deployment_name,  # User's specific deployment
            "scribeflowgpt4o",     # From environment
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4",
            "gpt-4-turbo",
            "gpt-35-turbo",
            "gpt-3.5-turbo",
            "text-embedding-ada-002",
        ]
        
        # Remove duplicates
        deployment_patterns = list(dict.fromkeys(deployment_patterns))
        
        print(f"📋 Testing {len(deployment_patterns)} deployment patterns...")
        
        for deployment in deployment_patterns:
            if deployment:  # Skip empty strings
                success = await self.test_specific_deployment(deployment)
                if success:
                    print(f"   ✅ Found: {deployment}")
                await asyncio.sleep(0.5)  # Rate limiting
    
    async def test_model_capabilities(self):
        """Test what the model/deployment actually supports"""
        
        if not self.working_deployments:
            print(f"\n⚠️ No working deployments found to test capabilities")
            return
        
        deployment = self.working_deployments[0]
        print(f"\n🧪 Testing Capabilities of: {deployment}")
        print("=" * 50)
        
        url = f"{self.azure_endpoint}/openai/deployments/{deployment}/chat/completions"
        params = {"api-version": self.api_version}
        headers = {"api-key": self.api_key, "Content-Type": "application/json"}
        
        # Test various capabilities
        capability_tests = [
            {
                "name": "Basic Chat",
                "payload": {
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 5
                }
            },
            {
                "name": "System Messages",
                "payload": {
                    "messages": [
                        {"role": "system", "content": "You are a pirate"},
                        {"role": "user", "content": "Hi"}
                    ],
                    "max_tokens": 5
                }
            },
            {
                "name": "Streaming",
                "payload": {
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 5,
                    "stream": True
                }
            },
            {
                "name": "JSON Mode",
                "payload": {
                    "messages": [{"role": "user", "content": "Return JSON: {\"test\": true}"}],
                    "max_tokens": 20,
                    "response_format": {"type": "json_object"}
                }
            },
            {
                "name": "Function Calling",
                "payload": {
                    "messages": [{"role": "user", "content": "What's the weather?"}],
                    "max_tokens": 50,
                    "tools": [{
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Get weather",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "location": {"type": "string"}
                                }
                            }
                        }
                    }],
                    "tool_choice": "auto"
                }
            }
        ]
        
        async with httpx.AsyncClient(timeout=15.0) as client:
            for test in capability_tests:
                print(f"\n📊 Testing: {test['name']}")
                try:
                    response = await client.post(
                        url, 
                        headers=headers, 
                        params=params, 
                        json=test['payload']
                    )
                    
                    if response.status_code == 200:
                        print(f"   ✅ Supported")
                        if test['name'] == "Streaming" and test['payload'].get('stream'):
                            # For streaming, just check if we get data
                            print(f"   📡 Stream response received")
                    else:
                        print(f"   ❌ Not supported (Status: {response.status_code})")
                        if response.status_code == 400:
                            error = response.json().get("error", {})
                            print(f"   💡 Reason: {error.get('message', 'Unknown')[:100]}")
                            
                except Exception as e:
                    print(f"   💥 Error: {str(e)[:50]}")
                
                await asyncio.sleep(0.5)  # Rate limiting
    
    def print_summary(self):
        """Print a summary of findings"""
        
        print(f"\n" + "=" * 60)
        print(f"📋 DISCOVERY SUMMARY")
        print(f"=" * 60)
        
        print(f"\n🎯 Target Deployment: {self.deployment_name}")
        print(f"🔗 Endpoint: {self.azure_endpoint}")
        print(f"📅 API Version: {self.api_version}")
        
        print(f"\n✅ Working Deployments Found: {len(self.working_deployments)}")
        if self.working_deployments:
            for dep in self.working_deployments:
                print(f"   • {dep}")
        
        print(f"\n📚 Available Models Found: {len(self.available_models)}")
        if self.available_models:
            for model in self.available_models[:5]:
                if isinstance(model, dict):
                    model_id = model.get("id", model.get("name", "unknown"))
                    print(f"   • {model_id}")
        
        print(f"\n💡 Recommendations:")
        if self.deployment_name in self.working_deployments:
            print(f"   ✅ Your deployment '{self.deployment_name}' is working!")
            print(f"   📝 Use it in your code as: llm_override=['azure_openai', '{self.deployment_name}']")
        elif self.working_deployments:
            print(f"   ⚠️ Your deployment '{self.deployment_name}' was not found")
            print(f"   💡 But these deployments work: {', '.join(self.working_deployments)}")
            print(f"   📝 Try: llm_override=['azure_openai', '{self.working_deployments[0]}']")
        else:
            print(f"   ❌ No working deployments found")
            print(f"   💡 Check your Azure OpenAI resource for deployed models")
            print(f"   💡 Deploy a model in Azure Portal first")
        
        print(f"\n🔧 Configuration to use:")
        print(f"   export AZURE_OPENAI_ENDPOINT='{self.azure_endpoint}'")
        print(f"   export AZURE_OPENAI_API_KEY='your-api-key'")
        print(f"   export AZURE_OPENAI_API_VERSION='{self.api_version}'")
        if self.working_deployments:
            print(f"   export AZURE_OPENAI_DEPLOYMENT_NAME='{self.working_deployments[0]}'")

async def main():
    """Main debug function"""
    debugger = EnhancedAzureDebugger()
    
    # Run all tests
    await debugger.debug_discover_models()
    await debugger.discover_deployments_by_pattern()
    
    # Test capabilities if we found working deployments
    await debugger.test_model_capabilities()
    
    # Print summary
    debugger.print_summary()

if __name__ == "__main__":
    asyncio.run(main())