#!/usr/bin/env python3
"""
Debug version of Azure OpenAI discoverer to see exactly what URLs are being called
"""

import asyncio
import os
from pathlib import Path

import httpx

# Load environment variables
try:
    from dotenv import load_dotenv

    env_file = Path(".env")
    if env_file.exists():
        load_dotenv(env_file)
except ImportError:
    pass


class DebugAzureDiscoverer:
    """Debug version of Azure OpenAI discoverer"""

    def __init__(self):
        self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.api_version = "2024-02-01"

    async def debug_discover_models(self):
        """Debug the model discovery process step by step"""

        print("🐛 Debug Azure OpenAI Model Discovery")
        print("=" * 45)

        if not self.azure_endpoint or not self.api_key:
            print("❌ Missing credentials")
            return

        print(f"🔗 Endpoint: {self.azure_endpoint}")
        print(f"🔑 API Key: {'*' * (len(self.api_key) - 8)}{self.api_key[-8:]}")
        print(f"📅 API Version: {self.api_version}")

        # Test the exact URL that should be called
        url = f"{self.azure_endpoint}/openai/models"
        params = {"api-version": self.api_version}
        headers = {"api-key": self.api_key}

        print(f"\n🎯 Testing URL: {url}")
        print(f"📋 Params: {params}")
        print("🔐 Headers: api-key=***")

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                print("\n📡 Making GET request...")
                response = await client.get(url, headers=headers, params=params)

                print(f"📊 Response Status: {response.status_code}")
                print(f"📋 Response Headers: {dict(response.headers)}")

                if response.status_code == 200:
                    print("✅ SUCCESS!")
                    try:
                        data = response.json()
                        models = data.get("data", [])
                        print(f"📚 Found {len(models)} models:")
                        for i, model in enumerate(models[:5]):  # Show first 5
                            model_id = model.get("id", "unknown")
                            owned_by = model.get("owned_by", "unknown")
                            print(f"   {i + 1}. {model_id} (owned by: {owned_by})")
                        if len(models) > 5:
                            print(f"   ... and {len(models) - 5} more models")
                    except Exception as e:
                        print(f"❌ Failed to parse JSON: {e}")
                        print(f"📄 Raw response: {response.text[:200]}...")

                elif response.status_code == 404:
                    print("❌ 404 NOT FOUND")
                    print("💡 This means the /openai/models endpoint doesn't exist")
                    print("🔍 Let's try alternative endpoints...")

                    # Try alternative endpoints
                    alternatives = [
                        f"{self.azure_endpoint}/models",
                        f"{self.azure_endpoint.replace('/openai', '')}/openai/models",
                        f"{self.azure_endpoint.replace('/openai', '')}/models",
                    ]

                    for alt_url in alternatives:
                        print(f"\n🔄 Trying: {alt_url}")
                        try:
                            alt_response = await client.get(
                                alt_url, headers=headers, params=params
                            )
                            print(f"   Status: {alt_response.status_code}")
                            if alt_response.status_code == 200:
                                print("   ✅ WORKS!")
                                try:
                                    alt_data = alt_response.json()
                                    alt_models = alt_data.get("data", [])
                                    print(f"   📚 Found {len(alt_models)} models")
                                except:
                                    print(
                                        f"   📄 Response length: {len(alt_response.text)}"
                                    )
                            else:
                                print(f"   ❌ {alt_response.status_code}")
                        except Exception as e:
                            print(f"   💥 Error: {str(e)[:50]}...")

                elif response.status_code == 401:
                    print("❌ 401 UNAUTHORIZED - Check API key")
                elif response.status_code == 403:
                    print("❌ 403 FORBIDDEN - Check permissions")
                else:
                    print(f"❌ Unexpected status: {response.status_code}")
                    print(f"📄 Response: {response.text[:200]}...")

        except Exception as e:
            print(f"💥 Request failed: {e}")

    async def debug_test_chat_completion(self):
        """Test if chat completions work to verify endpoint format"""

        print("\n💬 Testing Chat Completions (to verify endpoint format)")
        print("=" * 55)

        # Try common deployment names
        deployment_names = ["gpt-4o-mini", "gpt-4o", "gpt-4", "gpt-35-turbo"]

        for deployment in deployment_names:
            url = f"{self.azure_endpoint}/openai/deployments/{deployment}/chat/completions"
            params = {"api-version": self.api_version}
            headers = {"api-key": self.api_key, "Content-Type": "application/json"}

            payload = {"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 5}

            print(f"\n🎯 Testing deployment: {deployment}")
            print(f"📡 URL: {url}")

            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.post(
                        url, headers=headers, params=params, json=payload
                    )

                    if response.status_code == 200:
                        print(f"   ✅ WORKS - {deployment} is deployed!")
                        return deployment  # Found a working deployment
                    elif response.status_code == 404:
                        print(f"   ❌ 404 - {deployment} not deployed")
                    else:
                        print(f"   ❓ {response.status_code} - {deployment}")

            except Exception as e:
                print(f"   💥 Error: {str(e)[:30]}...")

        print("\n❌ No working deployments found")
        return None


async def main():
    """Main debug function"""
    debugger = DebugAzureDiscoverer()

    # Test model discovery
    await debugger.debug_discover_models()

    # Test chat completions
    working_deployment = await debugger.debug_test_chat_completion()

    print("\n📋 Debug Summary:")
    print("   • Azure endpoint format appears to be correct")
    print(
        "   • The /openai/models endpoint may not exist in your Azure OpenAI instance"
    )
    print(f"   • Chat completions work: {'Yes' if working_deployment else 'No'}")
    if working_deployment:
        print(f"   • Working deployment found: {working_deployment}")

    print("\n💡 Possible explanations:")
    print("   1. Azure OpenAI may not support the /openai/models listing endpoint")
    print("   2. Your Azure OpenAI resource may be configured differently")
    print("   3. The API version may not support model listing")
    print("   4. This might be an older Azure OpenAI instance")


if __name__ == "__main__":
    asyncio.run(main())
