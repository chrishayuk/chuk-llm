#!/usr/bin/env python3
"""
Fixed Trace get_client Discovery Flow

The issue was fixed: get_client("ollama", "mistral-small:latest") now works!
This script verifies the fix and provides accurate diagnostics.
"""

import asyncio
import os
import sys


def setup_environment():
    """Set up discovery environment."""
    discovery_env = {
        "CHUK_LLM_DISCOVERY_ENABLED": "true",
        "CHUK_LLM_AUTO_DISCOVER": "true",
        "CHUK_LLM_DISCOVERY_ON_STARTUP": "true",
        "CHUK_LLM_OLLAMA_DISCOVERY": "true",
        "CHUK_LLM_DISCOVERY_DEBUG": "true",
        "CHUK_LLM_DISCOVERY_FORCE_REFRESH": "true",
    }

    for key, value in discovery_env.items():
        if key not in os.environ:
            os.environ[key] = value


async def trace_get_client_discovery():
    """Trace the get_client discovery flow step by step."""

    print("🔍 Tracing get_client Discovery Flow")
    print("=" * 50)

    setup_environment()

    try:
        from chuk_llm.api.providers import trigger_ollama_discovery_and_refresh
        from chuk_llm.configuration.unified_config import get_config
        from chuk_llm.llm.client import get_client, get_provider_info

        # Test model that should be discoverable
        test_model = "mistral-small:latest"

        print(f"🎯 Testing discovery for: {test_model}")

        # Step 1: Check initial state
        print("\n1️⃣ Initial state...")
        config_manager = get_config()
        provider_config = config_manager.get_provider("ollama")

        print(f"   Static models: {len(provider_config.models)}")
        print(f"   Models: {provider_config.models}")
        print(f"   Target in static: {test_model in provider_config.models}")

        # Step 2: Check _ensure_model_available method
        print("\n2️⃣ Testing _ensure_model_available...")
        try:
            if hasattr(config_manager, "_ensure_model_available"):
                resolved = config_manager._ensure_model_available("ollama", test_model)
                print(f"   _ensure_model_available result: {resolved}")
            else:
                print("   _ensure_model_available method not found")
        except Exception as e:
            print(f"   _ensure_model_available failed: {e}")

        # Step 3: Trigger discovery manually
        print("\n3️⃣ Triggering discovery...")
        discovered_functions = trigger_ollama_discovery_and_refresh()
        print(f"   Discovery generated: {len(discovered_functions)} functions")

        # Check if our test model is in the functions
        if isinstance(discovered_functions, dict):
            function_names = list(discovered_functions.keys())
        else:
            function_names = discovered_functions

        model_functions = [f for f in function_names if "mistral_small" in f]
        print(f"   Functions for {test_model}: {len(model_functions)}")
        if model_functions:
            print(f"   Sample: {model_functions[:3]}")

        # Step 4: Check provider state after discovery
        print("\n4️⃣ Provider state after discovery...")
        after_provider = config_manager.get_provider("ollama")
        after_models = after_provider.models

        print(f"   Models after discovery: {len(after_models)}")
        print(f"   Target in models: {test_model in after_models}")

        # Step 5: Check provider_info
        print("\n5️⃣ Testing get_provider_info...")
        provider_info = get_provider_info("ollama")
        available_models = provider_info.get("available_models", [])

        print(f"   Available models: {len(available_models)}")
        print(f"   Target in available: {test_model in available_models}")

        # Step 6: Try get_client step by step
        print("\n6️⃣ Testing get_client...")

        try:
            print(f"   Calling get_client('ollama', '{test_model}')...")

            # This should work now with the fix
            client = get_client("ollama", model=test_model)
            print(f"   ✅ Success! Client model: {client.model}")

            # Test that the client actually works
            print("   Testing client functionality...")
            model_info = client.get_model_info()
            if model_info.get("error"):
                print(f"   ⚠️  Client created but has issues: {model_info['error']}")
            else:
                print("   ✅ Client fully functional")
                print(f"   Features: {model_info.get('features', [])}")

        except Exception as e:
            print(f"   ❌ Failed: {e}")
            print("   🔍 This indicates the fix didn't work properly")

        # Step 7: Verify Ollama integration
        print("\n7️⃣ Verifying Ollama integration...")

        # Check what Ollama actually has
        try:
            import httpx

            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get("http://localhost:11434/api/tags")
                response.raise_for_status()
                data = response.json()

                actual_models = [
                    model_data["name"] for model_data in data.get("models", [])
                ]
                print(f"   Actual Ollama models: {len(actual_models)}")
                print(f"   Target in Ollama: {test_model in actual_models}")

                if test_model in actual_models and test_model in after_models:
                    print(
                        "   ✅ Perfect integration: Model exists in both Ollama and chuk_llm"
                    )
                elif test_model in actual_models:
                    print(
                        "   ⚠️  Model exists in Ollama but not accessible via chuk_llm"
                    )
                else:
                    print("   ❌ Model not in Ollama")

        except Exception as e:
            print(f"   ❌ Failed to check Ollama: {e}")

        # Step 8: Final analysis and statistics
        print("\n8️⃣ Final Analysis...")

        # Count discoverable models
        ollama_models = []
        try:
            import httpx

            async with httpx.AsyncClient(timeout=3.0) as client:
                response = await client.get("http://localhost:11434/api/tags")
                response.raise_for_status()
                data = response.json()
                ollama_models = [
                    model_data["name"] for model_data in data.get("models", [])
                ]
        except:
            pass

        initial_models = {
            "llama3.3",
            "qwen3",
            "granite3.3",
            "mistral",
            "gemma3",
            "phi3",
            "codellama",
        }
        current_models = set(after_models)
        discovered_models = current_models - initial_models

        print("   📊 Discovery Statistics:")
        print(f"      Ollama models available: {len(ollama_models)}")
        print(f"      Initial static models: {len(initial_models)}")
        print(f"      Final chuk_llm models: {len(current_models)}")
        print(f"      Models added by discovery: {len(discovered_models)}")
        print(f"      Generated functions: {len(discovered_functions)}")

        # Show some discovered models
        if discovered_models:
            print(
                f"   🔍 Discovered models: {list(discovered_models)[:5]}{'...' if len(discovered_models) > 5 else ''}"
            )

        # Status assessment
        print("\n   🎯 DISCOVERY STATUS:")
        if len(discovered_models) > 0 and test_model in current_models:
            print("      ✅ FULLY WORKING: Discovery successfully integrated")
            print(
                f"      ✅ Added {len(discovered_models)} models to provider configuration"
            )
            print("      ✅ get_client() can access discovered models")
            print(f"      ✅ Target model '{test_model}' is accessible")
        elif len(discovered_models) > 0:
            print(
                "      ⚠️  PARTIALLY WORKING: Discovery found models but target not accessible"
            )
            print(f"      ✅ Added {len(discovered_models)} models")
            print(f"      ❌ Target model '{test_model}' not accessible")
        else:
            print("      ❌ NOT WORKING: Discovery didn't add any models")
            print("      Issue: Discovery process not updating provider configuration")

        # Test convenience functions
        print("\n9️⃣ Testing convenience functions...")
        try:
            from chuk_llm.api.providers import ask_ollama_mistral_small_latest_sync

            print("   ✅ Convenience function imported successfully")

            # Test a simple call (comment out to avoid actual API call)
            # response = ask_ollama_mistral_small_latest_sync("Hello")
            # print(f"   ✅ Convenience function works: {response[:50]}...")

        except ImportError as e:
            print(f"   ❌ Convenience function not available: {e}")
        except Exception as e:
            print(f"   ⚠️  Convenience function import failed: {e}")

        print("\n🎊 Discovery integration test complete!")

    except Exception as e:
        print(f"❌ Trace failed: {e}")
        import traceback

        traceback.print_exc()


async def main():
    """Main function."""
    try:
        await trace_get_client_discovery()
    except Exception as e:
        print(f"Failed: {e}")
        return 1
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
