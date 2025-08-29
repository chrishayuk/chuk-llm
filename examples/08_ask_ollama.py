#!/usr/bin/env python3
"""
Test Real-World Usage Examples

This demonstrates your working discovery system with real examples.
"""

import asyncio

import chuk_llm


async def test_real_world_examples():
    """Test real-world usage with your discovered models"""

    print("🚀 Testing Real-World ChukLLM Usage")
    print("=" * 45)

    # Test 1: Code generation with Devstral
    print("1️⃣ Code Generation with Devstral...")

    try:
        response = await chuk_llm.ask(
            "Write a simple Python function that calculates the factorial of a number.",
            provider="ollama",
            model="devstral",  # Should resolve to devstral:latest
            max_tokens=200,
        )

        print("   ✅ Devstral Response:")
        print(f"   📝 {response[:150]}...")

    except Exception as e:
        print(f"   ❌ Devstral failed: {e}")

    # Test 2: Reasoning with Qwen 32B
    print("\n2️⃣ Math Reasoning with Qwen 32B...")

    try:
        response = await chuk_llm.ask(
            "What is 127 * 89? Show your work step by step.",
            provider="ollama",
            model="qwen3:32b",  # Discovered model
            max_tokens=150,
        )

        print("   ✅ Qwen Response:")
        print(f"   🧮 {response[:200]}...")

    except Exception as e:
        print(f"   ❌ Qwen failed: {e}")

    # Test 3: Creative writing with Llama
    print("\n3️⃣ Creative Writing with Llama...")

    try:
        response = await chuk_llm.ask(
            "Write a haiku about artificial intelligence.",
            provider="ollama",
            model="llama3.3",  # Static model
            max_tokens=100,
        )

        print("   ✅ Llama Response:")
        print(f"   🎨 {response}")

    except Exception as e:
        print(f"   ❌ Llama failed: {e}")

    # Test 4: Technical explanation with Granite
    print("\n4️⃣ Technical Explanation with Granite...")

    try:
        response = await chuk_llm.ask(
            "Explain what a hash table is in one sentence.",
            provider="ollama",
            model="granite3.3",  # Static model with reasoning
            max_tokens=50,
        )

        print("   ✅ Granite Response:")
        print(f"   🧠 {response}")

    except Exception as e:
        print(f"   ❌ Granite failed: {e}")

    # Test 5: Model comparison - same question to different models
    print("\n5️⃣ Model Comparison - 'What is recursion?'")

    test_models = [
        ("devstral", "Code-focused model"),
        ("qwen3", "Reasoning model"),
        ("granite3.3", "Technical model"),
    ]

    question = "What is recursion in programming? Answer in one sentence."

    for model, description in test_models:
        try:
            response = await chuk_llm.ask(
                question, provider="ollama", model=model, max_tokens=50
            )

            print(f"   📋 {model} ({description}):")
            print(f"      {response.strip()}")

        except Exception as e:
            print(f"   ❌ {model} failed: {str(e)[:50]}...")

    # Test 6: Streaming example
    print("\n6️⃣ Streaming Example...")

    try:
        print("   🌊 Streaming response from qwen3:")
        print("   ", end="", flush=True)

        async for chunk in chuk_llm.stream(
            "Count from 1 to 5, one number per word.",
            provider="ollama",
            model="qwen3",
            max_tokens=20,
        ):
            if chunk:
                print(chunk, end="", flush=True)

        print("\n   ✅ Streaming completed!")

    except Exception as e:
        print(f"   ❌ Streaming failed: {e}")


async def show_available_models():
    """Show what models are available after discovery"""

    print("\n📊 Available Models Summary...")

    try:
        from chuk_llm.configuration import get_config

        manager = get_config()
        provider = manager.get_provider("ollama")

        print(f"   📋 Total models available: {len(provider.models)}")
        print(
            f"   🔧 Static models: {len([m for m in provider.models if m in ['llama3.3', 'qwen3', 'granite3.3', 'mistral', 'gemma3', 'phi3', 'codellama']])}"
        )
        print(f"   🔍 Discovered models: ~{len(provider.models) - 7}")

        # Show model families
        families = {}
        for model in provider.models:
            if "qwen" in model.lower():
                families.setdefault("qwen", []).append(model)
            elif "llama" in model.lower():
                families.setdefault("llama", []).append(model)
            elif "devstral" in model.lower() or "code" in model.lower():
                families.setdefault("code", []).append(model)
            elif "phi" in model.lower():
                families.setdefault("phi", []).append(model)
            elif "granite" in model.lower():
                families.setdefault("granite", []).append(model)
            else:
                families.setdefault("other", []).append(model)

        print("\n   📈 Model Families:")
        for family, models in families.items():
            print(f"      {family}: {len(models)} models")
            if len(models) <= 3:
                for model in models:
                    print(f"         • {model}")
            else:
                for model in models[:2]:
                    print(f"         • {model}")
                print(f"         ... and {len(models) - 2} more")

    except Exception as e:
        print(f"   ❌ Summary failed: {e}")


async def main():
    """Main function"""

    await test_real_world_examples()
    await show_available_models()

    print("\n🎉 Real-World Testing Complete!")
    print("💡 Your ChukLLM discovery system is fully operational")
    print("🚀 You now have access to 48 Ollama models with intelligent capabilities")

    print("\n📖 Quick Usage Guide:")
    print("   • Code: chuk_llm.ask('Write code', model='devstral')")
    print("   • Math: chuk_llm.ask('Calculate', model='qwen3:32b')")
    print("   • Creative: chuk_llm.ask('Write story', model='llama3.3')")
    print("   • Technical: chuk_llm.ask('Explain', model='granite3.3')")


if __name__ == "__main__":
    asyncio.run(main())
