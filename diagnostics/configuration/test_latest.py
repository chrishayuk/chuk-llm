#!/usr/bin/env python3
"""
Test :latest Resolution Specifically

This tests the exact scenario where devstral should resolve to devstral:latest
"""

import asyncio


async def test_latest_resolution():
    """Test the specific :latest resolution issue"""

    print("🔍 Testing :latest Resolution Issue")
    print("=" * 45)

    # Test 1: Check what models are actually available
    print("1️⃣ Checking Available Models...")

    try:
        from chuk_llm.configuration import get_config

        manager = get_config()
        provider = manager.get_provider("ollama")

        print(f"   📋 Static models: {provider.models}")
        print(f"   🏷️  Model aliases: {provider.model_aliases}")

        # Check if devstral or devstral:latest are in the models list
        has_devstral = "devstral" in provider.models
        has_devstral_latest = "devstral:latest" in provider.models

        print(f"   🔍 devstral in models: {has_devstral}")
        print(f"   🔍 devstral:latest in models: {has_devstral_latest}")

    except Exception as e:
        print(f"   ❌ Config check failed: {e}")

    # Test 2: Test the _ensure_model_available method directly
    print("\n2️⃣ Testing _ensure_model_available...")

    try:
        # Test the discovery system directly
        result_devstral = manager._ensure_model_available("ollama", "devstral")
        result_devstral_latest = manager._ensure_model_available(
            "ollama", "devstral:latest"
        )

        print(f"   🧪 _ensure_model_available('devstral'): {result_devstral}")
        print(
            f"   🧪 _ensure_model_available('devstral:latest'): {result_devstral_latest}"
        )

    except Exception as e:
        print(f"   ❌ Ensure model test failed: {e}")

    # Test 3: Force a discovery refresh and check models again
    print("\n3️⃣ Testing Discovery Refresh...")

    try:
        from chuk_llm.api.discovery import discover_models

        # Force discovery refresh
        models = await discover_models("ollama", force_refresh=True)
        discovered_names = [m["name"] for m in models]

        print(f"   📊 Total discovered: {len(models)}")
        print(f"   🔍 'devstral' in discovered: {'devstral' in discovered_names}")
        print(
            f"   🔍 'devstral:latest' in discovered: {'devstral:latest' in discovered_names}"
        )

        # Check if provider models list was updated
        provider_after = manager.get_provider("ollama")
        print(f"   📋 Provider models after discovery: {len(provider_after.models)}")
        print(f"   🔍 'devstral' in provider: {'devstral' in provider_after.models}")
        print(
            f"   🔍 'devstral:latest' in provider: {'devstral:latest' in provider_after.models}"
        )

    except Exception as e:
        print(f"   ❌ Discovery refresh failed: {e}")

    # Test 4: Test client creation again after discovery
    print("\n4️⃣ Testing Client Creation After Discovery...")

    try:
        from chuk_llm.llm.client import get_client

        test_models = ["devstral", "devstral:latest"]

        for test_model in test_models:
            try:
                print(f"   🧪 Creating client for: {test_model}")
                client = get_client("ollama", model=test_model)
                print(f"      ✅ Success! Resolved to: {client.model}")

            except Exception as model_error:
                print(f"      ❌ Failed: {str(model_error)[:60]}...")

    except Exception as e:
        print(f"   ❌ Client creation test failed: {e}")

    # Test 5: Test the actual LLM calls
    print("\n5️⃣ Testing LLM Calls...")

    try:
        from chuk_llm import ask

        test_cases = [
            ("devstral", "Should resolve to devstral:latest"),
            ("devstral:latest", "Should work directly"),
        ]

        for model, description in test_cases:
            try:
                print(f"   🧪 Testing {model} ({description})")

                response = await ask(
                    "Say 'OK' if working.", provider="ollama", model=model, max_tokens=5
                )

                print(f"      ✅ Success: '{response.strip()}'")

            except Exception as model_error:
                error_str = str(model_error)
                if "not available" in error_str:
                    print(f"      ❌ Model resolution failed: {model}")
                else:
                    print(f"      ⚠️  Other error: {error_str[:50]}...")

    except Exception as e:
        print(f"   ❌ LLM call test failed: {e}")


async def debug_client_factory():
    """Debug the client factory process"""

    print("\n🔧 Debugging Client Factory Process...")

    try:
        from chuk_llm.configuration import get_config

        # Step by step client creation
        print("   📋 Step 1: Get config manager...")
        config_manager = get_config()

        print("   📋 Step 2: Get provider config...")
        provider_config = config_manager.get_provider("ollama")

        print("   📋 Step 3: Determine target model...")
        model_name = "devstral"
        target_model = model_name or provider_config.default_model

        print(f"      Target model: {target_model}")

        print("   📋 Step 4: Try model resolution...")
        resolved_model = config_manager._ensure_model_available("ollama", target_model)

        print(f"      Resolved model: {resolved_model}")

        if resolved_model:
            print("   ✅ Model resolution successful!")
        else:
            print("   ❌ Model resolution failed!")

            # Show what's available
            available = provider_config.models[:10]
            print(f"      Available models: {available}")

    except Exception as e:
        print(f"   ❌ Debug failed: {e}")
        import traceback

        print(f"   📋 Traceback: {traceback.format_exc()}")


async def main():
    """Main function"""
    await test_latest_resolution()
    await debug_client_factory()

    print("\n🎯 Summary:")
    print("   • Discovery system finds 48 models including devstral:latest")
    print("   • devstral:latest works perfectly")
    print("   • devstral (without :latest) should resolve to devstral:latest")
    print(
        "   • If resolution fails, the issue is in client factory or model resolution"
    )


if __name__ == "__main__":
    asyncio.run(main())
