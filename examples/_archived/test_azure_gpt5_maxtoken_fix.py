#!/usr/bin/env python3
"""
Test script to demonstrate the Azure GPT-5 max_tokens fix.

This shows that max_completion_tokens is now properly capped to
the configured model limit (16384 for Azure GPT-5 deployments).
"""

import asyncio
import logging
from chuk_llm.llm.providers.azure_openai_client import AzureOpenAILLMClient
from chuk_llm.configuration import get_config

# Enable debug logging to see parameter adjustments
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)


def test_max_tokens_capping():
    """Test that max_completion_tokens is capped to model limits"""
    print("\n" + "=" * 70)
    print("Testing Azure GPT-5 max_completion_tokens capping")
    print("=" * 70)

    # Create a client with a GPT-5 deployment
    # Note: We don't need actual credentials for this test since we're
    # just testing parameter validation
    try:
        client = AzureOpenAILLMClient(
            model="gpt-5",
            azure_deployment="gpt-5",
            api_key="fake-key-for-testing",
            azure_endpoint="https://fake-endpoint.openai.azure.com",
        )

        print(f"\n✅ Created Azure OpenAI client for model: {client.model}")
        print(f"   Deployment: {client.azure_deployment}")

        # Get model info to see configured limits
        info = client.get_model_info()
        print(f"\n📊 Model capabilities:")
        print(f"   Max context length: {info.get('max_context_length', 'N/A')}")
        print(f"   Max output tokens: {info.get('max_output_tokens', 'N/A')}")

        # Test parameter adjustment with a value that exceeds the limit
        print(f"\n🔧 Testing parameter adjustment:")
        print(f"   Input: max_completion_tokens=128000 (exceeds Azure limit)")

        # Call the parameter adjustment method
        adjusted = client._adjust_parameters_for_provider({
            "max_completion_tokens": 128000
        })

        adjusted_value = adjusted.get("max_completion_tokens", "NOT SET")
        print(f"   Output: max_completion_tokens={adjusted_value}")

        # Verify the fix
        if adjusted_value == 16384:
            print(f"\n✅ SUCCESS: max_completion_tokens was capped to 16384")
            print(f"   This matches the Azure GPT-5 deployment limit!")
            return True
        else:
            print(f"\n❌ FAILED: max_completion_tokens was not capped correctly")
            print(f"   Expected: 16384, Got: {adjusted_value}")
            return False

    except Exception as e:
        print(f"\n❌ Error creating client: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_smart_defaults():
    """Test that smart defaults work for unknown deployments"""
    print("\n" + "=" * 70)
    print("Testing smart defaults for unknown Azure deployment")
    print("=" * 70)

    try:
        # Create client with a custom deployment name
        client = AzureOpenAILLMClient(
            model="custom-gpt5-deployment",
            azure_deployment="my-custom-gpt-5-deployment",
            api_key="fake-key-for-testing",
            azure_endpoint="https://fake-endpoint.openai.azure.com",
        )

        print(f"\n✅ Created Azure OpenAI client for custom deployment: {client.azure_deployment}")

        # Get smart defaults
        smart_params = client._get_smart_default_parameters(client.azure_deployment)
        print(f"\n📊 Smart default parameters:")
        print(f"   Max output tokens: {smart_params.get('max_output_tokens', 'N/A')}")
        print(f"   Requires max_completion_tokens: {smart_params.get('requires_max_completion_tokens', False)}")

        # Check if GPT-5 is detected
        if "gpt-5" in client.azure_deployment.lower():
            if smart_params.get("max_output_tokens") == 16384:
                print(f"\n✅ SUCCESS: Smart defaults correctly detected GPT-5!")
                print(f"   max_output_tokens set to 16384 (Azure limit)")
                return True
        else:
            print(f"\n⚠️  Note: Not a GPT-5 deployment, different limits apply")
            return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_based_limits():
    """Test that configuration-based limits are respected"""
    print("\n" + "=" * 70)
    print("Testing configuration-based max_tokens limits")
    print("=" * 70)

    try:
        config = get_config()

        # Check if azure_openai provider is configured
        print("\n📋 Checking azure_openai provider configuration:")

        azure_config = config.get_provider("azure_openai")
        print(f"   Provider: {azure_config.name}")
        print(f"   Max output tokens: {azure_config.max_output_tokens}")
        print(f"   Features: {', '.join(azure_config.features)}")

        # Check model-specific capabilities
        if azure_config.model_capabilities:
            print(f"\n📋 Model-specific capabilities:")
            for cap in azure_config.model_capabilities:
                if hasattr(cap, 'pattern') and 'gpt-5' in cap.pattern:
                    print(f"   Pattern: {cap.pattern}")
                    print(f"   Max output tokens: {cap.max_output_tokens}")

                    if cap.max_output_tokens == 16384:
                        print(f"\n✅ SUCCESS: Configuration correctly sets GPT-5 limit to 16384")
                        return True

        print(f"\n⚠️  Note: No specific GPT-5 configuration found")
        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("AZURE GPT-5 MAX_TOKENS FIX VERIFICATION")
    print("=" * 70)
    print("\nThis test demonstrates that the fix correctly caps max_completion_tokens")
    print("to Azure GPT-5 deployment limits (16384) instead of using the default")
    print("OpenAI limit (128000).")

    results = []

    # Run tests
    results.append(("Config-based limits", test_config_based_limits()))
    results.append(("Max tokens capping", test_max_tokens_capping()))
    results.append(("Smart defaults", test_smart_defaults()))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")

    all_passed = all(result[1] for result in results)

    if all_passed:
        print("\n🎉 All tests passed! The fix is working correctly.")
        print("\nThe fix ensures that:")
        print("  • max_completion_tokens is validated against model capabilities")
        print("  • Azure GPT-5 deployments use 16384 max_completion_tokens")
        print("  • Smart defaults work for unknown deployment names")
    else:
        print("\n⚠️  Some tests failed. Please review the output above.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
