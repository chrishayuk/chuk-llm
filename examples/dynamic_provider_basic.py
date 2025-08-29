#!/usr/bin/env python3
"""
Basic Dynamic Provider Registration Example
===========================================

This example demonstrates the fundamental dynamic provider registration features:
- Registering a new OpenAI-compatible provider
- Checking if a provider exists
- Using the provider for inference
- Listing and removing providers
"""

import os
from chuk_llm import (
    register_openai_compatible,
    provider_exists,
    list_dynamic_providers,
    unregister_provider,
    ask_sync,
    stream_sync_iter
)

def main():
    print("=" * 60)
    print("BASIC DYNAMIC PROVIDER EXAMPLE")
    print("=" * 60)
    
    # 1. Register a new OpenAI-compatible provider
    print("\n1. Registering a new provider...")
    print("-" * 40)
    
    provider_config = register_openai_compatible(
        name="my_custom_api",
        api_base="https://api.openai.com/v1",  # Can be any OpenAI-compatible endpoint
        api_key=os.getenv("OPENAI_API_KEY", "your-api-key-here"),
        models=["gpt-3.5-turbo", "gpt-4"],
        default_model="gpt-3.5-turbo"
    )
    
    print(f"✅ Registered provider: {provider_config.name}")
    print(f"   Endpoint: {provider_config.api_base}")
    print(f"   Available models: {provider_config.models}")
    print(f"   Default model: {provider_config.default_model}")
    
    # 2. Check if provider exists
    print("\n2. Checking provider existence...")
    print("-" * 40)
    
    if provider_exists("my_custom_api"):
        print("✅ Provider 'my_custom_api' exists and is ready to use")
    
    if not provider_exists("non_existent"):
        print("✅ Provider 'non_existent' correctly reported as not found")
    
    # 3. List all dynamic providers
    print("\n3. Listing dynamic providers...")
    print("-" * 40)
    
    dynamic_providers = list_dynamic_providers()
    print(f"Current dynamic providers: {dynamic_providers}")
    
    # 4. Use the provider for inference
    print("\n4. Testing inference...")
    print("-" * 40)
    
    if os.getenv("OPENAI_API_KEY"):
        try:
            # Simple completion
            response = ask_sync(
                "What is 2+2? Reply with just the number.",
                provider="my_custom_api",
                temperature=0,  # Deterministic response
                max_tokens=5
            )
            print(f"✅ Basic inference: 2+2 = {response}")
            
            # Streaming example
            print("\n   Streaming response: ", end="")
            for chunk in stream_sync_iter(
                "Count from 1 to 5",
                provider="my_custom_api",
                temperature=0,
                max_tokens=20
            ):
                # Handle both string and dict responses
                if isinstance(chunk, str):
                    print(chunk, end="", flush=True)
                else:
                    print(chunk.get("response", ""), end="", flush=True)
            print()  # New line after streaming
            
        except Exception as e:
            print(f"⚠️  Inference error: {e}")
    else:
        print("⚠️  Set OPENAI_API_KEY environment variable to test inference")
        print("   Example: export OPENAI_API_KEY='sk-...'")
    
    # 5. Clean up - Unregister the provider
    print("\n5. Cleaning up...")
    print("-" * 40)
    
    success = unregister_provider("my_custom_api")
    if success:
        print("✅ Successfully unregistered 'my_custom_api'")
    
    remaining = list_dynamic_providers()
    print(f"Remaining dynamic providers: {remaining}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
This example demonstrated:
✅ Registering an OpenAI-compatible provider dynamically
✅ Checking provider existence
✅ Listing all dynamic providers
✅ Using the provider for both regular and streaming inference
✅ Removing a dynamic provider when done

Dynamic providers allow you to:
- Add new API endpoints without changing configuration files
- Use different API keys for different providers
- Temporarily add providers for testing
- Switch between different OpenAI-compatible services
""")

if __name__ == "__main__":
    main()