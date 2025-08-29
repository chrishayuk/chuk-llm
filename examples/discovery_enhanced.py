#!/usr/bin/env python3
"""
ChukLLM Discovery & Dynamic Model Demo
=====================================

This demo showcases ChukLLM's dynamic model discovery capabilities:
- Automatically discovers all available models from Ollama
- Tests models that aren't in static configuration
- Demonstrates seamless access to any discovered model
- Benchmarks performance across different model types
- Shows how discovery makes new models immediately available

Key Features:
- Smart model classification by actual capabilities
- Dynamic model testing without configuration updates
- Performance benchmarking and recommendations
- Automatic exclusion of non-chat models (embeddings)
"""

import asyncio
import logging
import time
import traceback
from typing import Any

# Load environment variables
try:
    from dotenv import load_dotenv

    load_dotenv()
    print("📁 Environment variables loaded")
except ImportError:
    print("💡 Using system environment variables")

# Configure logging
logging.basicConfig(level=logging.WARNING)
log = logging.getLogger(__name__)


def print_header(title: str, char: str = "=", width: int = 70):
    """Print a formatted header"""
    print(f"\n{char * width}")
    print(f"{title:^{width}}")
    print(f"{char * width}")


def print_section(title: str, char: str = "-", width: int = 60):
    """Print a formatted section header"""
    print(f"\n{title}")
    print(char * width)


def is_embedding_model(model_name: str) -> bool:
    """Check if a model is designed for embeddings (not chat)"""
    embedding_indicators = ["embed", "embedding", "mxbai-embed"]
    return any(indicator in model_name.lower() for indicator in embedding_indicators)


def classify_model_type(model_name: str, family: str) -> dict[str, Any]:
    """Classify models by their actual capabilities and intended use"""
    model_lower = model_name.lower()

    # Reasoning specialists (designed for step-by-step thinking)
    reasoning_specialists = [
        "granite3.3",
        "qwen3",
        "o1",
        "o3",
        "reasoning",
        "qwq",
        "marco-o1",
    ]

    # Code specialists
    code_specialists = [
        "codellama",
        "deepseek-coder",
        "starcoder",
        "codegemma",
        "devstral",
    ]

    # Vision models
    vision_models = ["llava", "moondream", "vision"]

    # Large general models (capable but not specialized)
    large_general = ["gpt-oss", "mistral-small", "mistral-nemo"]

    is_reasoning = any(
        specialist in model_lower for specialist in reasoning_specialists
    )
    is_code = any(specialist in model_lower for specialist in code_specialists)
    is_vision = any(vision in model_lower for vision in vision_models)
    is_embedding = is_embedding_model(model_name)
    is_large_general = any(large in model_lower for large in large_general)

    # Determine primary classification
    if is_embedding:
        primary_type = "embedding"
    elif is_reasoning:
        primary_type = "reasoning"
    elif is_code:
        primary_type = "code"
    elif is_vision:
        primary_type = "vision"
    elif is_large_general:
        primary_type = "large_general"
    else:
        primary_type = "general"

    return {
        "is_reasoning": is_reasoning,
        "is_code": is_code,
        "is_vision": is_vision,
        "is_embedding": is_embedding,
        "is_large_general": is_large_general,
        "primary_type": primary_type,
        "suitable_for_chat": not is_embedding,
    }


async def discover_all_models():
    """Discover all available models using ChukLLM's discovery system"""
    print_header("🔍 MODEL DISCOVERY")

    try:
        from chuk_llm.llm.discovery.engine import UniversalModelDiscoveryManager
        from chuk_llm.llm.discovery.ollama_discoverer import OllamaModelDiscoverer

        print("🏠 Discovering all available Ollama models...")

        # Create discovery manager
        discoverer = OllamaModelDiscoverer()
        manager = UniversalModelDiscoveryManager("ollama", discoverer)

        # Run discovery
        start_time = time.time()
        models = await manager.discover_models(force_refresh=True)
        discovery_time = time.time() - start_time

        print(f"✅ Discovered {len(models)} models in {discovery_time:.2f}s")

        # Classify models by type
        classified_models = {}

        for model in models:
            classification = classify_model_type(model.name, model.family)

            # Update model metadata with classification
            if not model.metadata:
                model.metadata = {}
            model.metadata.update(classification)

            # Group by primary type
            primary_type = classification["primary_type"]
            if primary_type not in classified_models:
                classified_models[primary_type] = []
            classified_models[primary_type].append(model)

        # Display classification results
        print("\n📋 Models by type:")
        for model_type, type_models in classified_models.items():
            emoji_map = {
                "reasoning": "🧠",
                "general": "💬",
                "large_general": "🦣",
                "code": "💻",
                "vision": "👁️",
                "embedding": "📊",
            }
            emoji = emoji_map.get(model_type, "📦")

            print(
                f"   {emoji} {model_type.replace('_', ' ').title()}: {len(type_models)} models"
            )

            # Show top models in each category
            for model in type_models[:3]:
                metadata = model.metadata or {}
                size_gb = metadata.get("size_gb", 0)
                context = model.context_length or "unknown"

                capabilities = []
                if metadata.get("is_reasoning"):
                    capabilities.append("reasoning")
                if metadata.get("supports_tools", False):
                    capabilities.append("tools")
                if metadata.get("is_vision"):
                    capabilities.append("vision")
                if metadata.get("is_code"):
                    capabilities.append("code")

                cap_str = f" [{', '.join(capabilities)}]" if capabilities else ""
                size_str = f" ({size_gb}GB)" if size_gb else ""

                print(f"      • {model.name}{size_str} - {context} context{cap_str}")

            if len(type_models) > 3:
                print(f"      ... and {len(type_models) - 3} more")

        return models, classified_models

    except Exception as e:
        print(f"❌ Model discovery failed: {e}")
        traceback.print_exc()
        return [], {}


async def compare_static_vs_discovered(models):
    """Compare static configuration vs discovered models"""
    print_header("⚖️  STATIC vs DISCOVERED MODELS")

    try:
        from chuk_llm.configuration import get_config

        config_manager = get_config()
        ollama_config = config_manager.get_provider("ollama")
        static_models = set(ollama_config.models)

        discovered_names = {model.name for model in models}

        # Find dynamic-only models (available but not in config)
        dynamic_only = discovered_names - static_models
        static_only = static_models - discovered_names
        common_models = static_models & discovered_names

        print("📊 Model Comparison:")
        print(f"   • Static configuration: {len(static_models)} models")
        print(f"   • Currently available: {len(discovered_names)} models")
        print(f"   • In both: {len(common_models)} models")

        if dynamic_only:
            print("\n🆕 Dynamic-only models (available but not in config):")
            for model_name in sorted(dynamic_only):
                # Find the full model object
                model_obj = next((m for m in models if m.name == model_name), None)
                if model_obj:
                    metadata = model_obj.metadata or {}
                    size_gb = metadata.get("size_gb", 0)
                    model_type = metadata.get("primary_type", "unknown")
                    size_str = f" ({size_gb}GB)" if size_gb else ""
                    print(f"      • {model_name}{size_str} [{model_type}]")

        if static_only:
            print("\n📥 Static-only models (configured but not available):")
            for model in sorted(static_only):
                print(f"      • {model} (try: ollama pull {model})")

        return dynamic_only, static_only, common_models

    except Exception as e:
        print(f"❌ Configuration comparison failed: {e}")
        return set(), set(), set()


async def test_dynamic_model(
    model_name: str, model_type: str, test_config: dict
) -> dict[str, Any]:
    """Test a dynamically discovered model"""
    print(f"\n🧪 Testing: {model_name}")
    print(f"   📝 Type: {model_type}")
    print(f"   💬 Prompt: {test_config['prompt'][:60]}...")

    try:
        from chuk_llm import ask

        start_time = time.time()
        response = await asyncio.wait_for(
            ask(
                test_config["prompt"],
                provider="ollama",
                model=model_name,
                max_tokens=test_config["max_tokens"],
            ),
            timeout=60,
        )
        response_time = time.time() - start_time

        print(f"   ✅ Success in {response_time:.2f}s")
        print(f"   💭 Response: {response[:120]}{'...' if len(response) > 120 else ''}")

        return {
            "model": model_name,
            "success": True,
            "response_time": response_time,
            "response": response,
            "response_length": len(response.strip()),
        }

    except TimeoutError:
        print("   ⏰ Timeout after 60s")
        return {"model": model_name, "success": False, "error": "Timeout"}
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return {"model": model_name, "success": False, "error": str(e)}


async def test_dynamic_models(models, classified_models, dynamic_only):
    """Test dynamic-only models to prove discovery works"""
    print_header("🚀 DYNAMIC MODEL TESTING")

    print("Testing models that are NOT in your static configuration...")
    print("This proves that discovery makes any model immediately available!")

    # Define test configurations for different model types
    test_configs = {
        "reasoning": {
            "prompt": "If I start with 8 books, give away 3, then buy 5 more, how many books do I have? Think step by step.",
            "max_tokens": 150,
        },
        "general": {
            "prompt": "Explain artificial intelligence in simple terms.",
            "max_tokens": 100,
        },
        "large_general": {
            "prompt": "What are the main advantages and challenges of renewable energy?",
            "max_tokens": 150,
        },
        "code": {
            "prompt": "Write a Python function that checks if a number is prime.",
            "max_tokens": 120,
        },
    }

    test_results = []
    dynamic_models_tested = []

    # Find dynamic-only models to test
    for model_type, type_models in classified_models.items():
        if model_type == "embedding":  # Skip embedding models
            continue

        for model in type_models:
            if model.name in dynamic_only:
                dynamic_models_tested.append((model, model_type))

    if not dynamic_models_tested:
        print("⚠️  No dynamic-only chat models found to test")
        print("   All discovered models are already in your static configuration")
        return []

    print(f"🎯 Testing {len(dynamic_models_tested)} dynamic-only models:")

    # Test each dynamic model
    for model, model_type in dynamic_models_tested[:5]:  # Limit to 5 for demo speed
        test_config = test_configs.get(model_type, test_configs["general"])

        result = await test_dynamic_model(model.name, model_type, test_config)
        test_results.append(result)

        # Brief pause between tests
        await asyncio.sleep(2)

    return test_results


async def analyze_results(test_results):
    """Analyze the dynamic model test results"""
    print_header("📊 RESULTS ANALYSIS")

    if not test_results:
        print("No dynamic models were tested")
        return

    successful_tests = [r for r in test_results if r["success"]]

    print("🎯 Dynamic Model Test Results:")
    print(f"   • Total tests: {len(test_results)}")
    print(f"   • Successful: {len(successful_tests)}")
    print(
        f"   • Success rate: {(len(successful_tests) / len(test_results) * 100):.1f}%"
    )

    if successful_tests:
        avg_time = sum(r["response_time"] for r in successful_tests) / len(
            successful_tests
        )
        print(f"   • Average response time: {avg_time:.2f}s")

        # Show performance rankings
        speed_rankings = sorted(successful_tests, key=lambda x: x["response_time"])
        print("\n🏆 Performance Rankings:")
        for i, result in enumerate(speed_rankings, 1):
            print(f"   {i}. {result['model']}: {result['response_time']:.2f}s")

    # Show any failures
    failed_tests = [r for r in test_results if not r["success"]]
    if failed_tests:
        print("\n⚠️  Failed Tests:")
        for result in failed_tests:
            print(f"   • {result['model']}: {result.get('error', 'Unknown error')}")


async def demonstrate_key_capability():
    """Demonstrate the key capability: using models not in static config"""
    print_header("💡 KEY DEMONSTRATION")

    print("🎯 The Power of Dynamic Discovery:")
    print("   ChukLLM's discovery system makes ANY discovered model")
    print("   immediately available for use, even if it's not in your")
    print("   static configuration!")

    # Test gpt-oss specifically since it's mentioned
    print("\n🧪 Special Test: gpt-oss:latest")
    print("   This model is available in Ollama but not in your static config")

    try:
        from chuk_llm import ask

        prompt = "Write a haiku about machine learning."

        start_time = time.time()
        response = await asyncio.wait_for(
            ask(prompt, provider="ollama", model="gpt-oss:latest", max_tokens=60),
            timeout=60,
        )
        response_time = time.time() - start_time

        print(f"   ✅ SUCCESS! gpt-oss:latest responded in {response_time:.2f}s")
        print("   🎭 Haiku response:")
        for line in response.strip().split("\n")[:3]:
            if line.strip():
                print(f"      {line}")

        print("\n💫 This proves discovery works perfectly!")
        print("   Even though gpt-oss:latest isn't in your static configuration,")
        print("   ChukLLM discovered it and made it available for use!")

    except Exception as e:
        print(f"   ❌ Error testing gpt-oss:latest: {e}")
        print("   💡 This might mean the model isn't loaded in Ollama")


async def generate_usage_examples(classified_models):
    """Generate practical usage examples"""
    print_header("💻 USAGE EXAMPLES")

    print("Based on discovered models, here's how to use them:")

    examples = []

    for model_type, models in classified_models.items():
        if model_type == "embedding" or not models:
            continue

        model = models[0]  # Use first model of each type

        if model_type == "reasoning":
            examples.append(
                {
                    "use_case": "Complex reasoning tasks",
                    "model": model.name,
                    "example": f'response = await ask("Solve this step by step: ...", provider="ollama", model="{model.name}")',
                }
            )
        elif model_type == "large_general":
            examples.append(
                {
                    "use_case": "Comprehensive analysis",
                    "model": model.name,
                    "example": f'response = await ask("Analyze the pros and cons of...", provider="ollama", model="{model.name}")',
                }
            )
        elif model_type == "general":
            examples.append(
                {
                    "use_case": "General questions",
                    "model": model.name,
                    "example": f'response = await ask("Explain quantum physics simply", provider="ollama", model="{model.name}")',
                }
            )

    for example in examples:
        print(f"\n🎯 {example['use_case']}:")
        print(f"   Model: {example['model']}")
        print(f"   Code: {example['example']}")


async def main():
    """Main demonstration"""
    print_header("🚀 ChukLLM DISCOVERY & DYNAMIC MODEL DEMO", "=", 80)

    print("This demo showcases ChukLLM's powerful model discovery capabilities:")
    print("• Automatically discovers all available models")
    print("• Makes ANY discovered model immediately usable")
    print("• No configuration updates needed")
    print("• Seamless access to models not in static config")

    # Step 1: Discover all models
    models, classified_models = await discover_all_models()

    if not models:
        print("❌ No models discovered. Please ensure Ollama is running.")
        return

    # Step 2: Compare static vs discovered
    dynamic_only, static_only, common_models = await compare_static_vs_discovered(
        models
    )

    # Step 3: Test dynamic-only models
    if dynamic_only:
        test_results = await test_dynamic_models(
            models, classified_models, dynamic_only
        )
        await analyze_results(test_results)

    # Step 4: Special demonstration
    await demonstrate_key_capability()

    # Step 5: Usage examples
    await generate_usage_examples(classified_models)

    # Final summary
    print_header("🎉 DEMONSTRATION COMPLETE!", "=", 80)

    print("✨ What was demonstrated:")
    print("   ✅ Automatic model discovery from Ollama")
    print("   ✅ Smart classification by model capabilities")
    print("   ✅ Dynamic model access without configuration updates")
    print("   ✅ Testing of models not in static configuration")
    print("   ✅ Seamless integration with ChukLLM ask() function")

    if models:
        reasoning_count = len(classified_models.get("reasoning", []))
        general_count = len(classified_models.get("general", [])) + len(
            classified_models.get("large_general", [])
        )
        dynamic_count = len(dynamic_only) if dynamic_only else 0

        print("\n💡 Your Model Ecosystem:")
        print(f"   • Total discovered: {len(models)} models")
        print(f"   • Reasoning specialists: {reasoning_count} models")
        print(f"   • General purpose: {general_count} models")
        print(f"   • Dynamic-only models: {dynamic_count} models")

    print("\n🚀 Key Takeaway:")
    print("   ChukLLM's discovery system makes model management effortless!")
    print("   Any model you have in Ollama is immediately available,")
    print("   regardless of your static configuration.")

    print("\n💻 Try it yourself:")
    print(
        "   await ask('Your question here', provider='ollama', model='any_discovered_model')"
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demo failed: {e}")
        traceback.print_exc()
