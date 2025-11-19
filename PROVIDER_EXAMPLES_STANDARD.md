# Provider Examples - Standardized Structure

## Overview

All provider example scripts should follow this standardized structure to ensure consistency and completeness across the chuk-llm library.

## Standard Example List (in order)

1. **Feature Detection** - Display model capabilities
2. **Model Discovery** - Discover available models from API ✅ (ADDED TO ALL)
3. **Basic Text** - Simple completion
4. **Streaming** - Real-time streaming response
5. **Function Calling** - Tool/function calling (if supported)
6. **JSON Mode** - Structured JSON output
7. **Model Comparison** - Compare 2-3 models side-by-side
8. **Context Window Test** - Test with long context (~4500 words)
9. **Parallel Processing** - Sequential vs parallel request comparison
10. **Simple Chat** - Multi-turn conversation
11. **Parameters Test** - Temperature and other parameter testing

## Status by Provider

### Model Comparison
- ✅ ALL PROVIDERS COMPLETE
  - groq, deepseek, openrouter, perplexity, mistral, anthropic, gemini, watsonx, azure

### Context Window Test
- ✅ ALL PROVIDERS COMPLETE
  - groq, deepseek, openrouter, perplexity, mistral, anthropic, gemini, watsonx, azure

### Parallel Processing
- ✅ ALL PROVIDERS COMPLETE
  - groq, deepseek, openrouter, perplexity, mistral, anthropic, gemini, watsonx, azure

## Code Templates

### 1. Context Window Test Template

```python
# =============================================================================
# Example N: Context Window Test
# =============================================================================

async def context_window_test(model: str = "default-model"):
    """Test provider's large context window"""
    print(f"\n📏 Context Window Test with {model}")
    print("=" * 60)

    client = get_client("provider_name", model=model)

    # Create a long context (~4500 words)
    long_text = "The quick brown fox jumps over the lazy dog. " * 500

    messages = [
        Message(
            role=MessageRole.SYSTEM,
            content=f"You have been given a long text. Here it is:\n\n{long_text}\n\nPlease analyze this text.",
        ),
        Message(
            role=MessageRole.USER,
            content="How many times does the word 'fox' appear in the text? Also tell me the total word count.",
        ),
    ]

    print(f"📝 Testing with ~{len(long_text.split())} words of context...")

    start_time = time.time()
    response = await client.create_completion(messages, max_tokens=150)
    duration = time.time() - start_time

    print(f"✅ Response ({duration:.2f}s):")
    print(f"   {response.get('response', '')}")

    return response
```

### 2. Parallel Processing Test Template

```python
# =============================================================================
# Example N: Parallel Processing Test
# =============================================================================

async def parallel_processing_test(model: str = "default-model"):
    """Test parallel request processing"""
    print("\n🔀 Parallel Processing Test")
    print("=" * 60)

    prompts = [
        "What is artificial intelligence?",
        "Explain quantum computing.",
        "What is machine learning?",
        "Define neural networks.",
        "What is deep learning?",
    ]

    print(f"📊 Testing {len(prompts)} parallel requests with {model}...")

    # Sequential processing
    print("\n📝 Sequential processing:")
    sequential_start = time.time()

    for prompt in prompts:
        client = get_client("provider_name", model=model)
        await client.create_completion(
            [Message(role=MessageRole.USER, content=prompt)], max_tokens=50
        )

    sequential_time = time.time() - sequential_start
    print(f"   ✅ Completed in {sequential_time:.2f}s")

    # Parallel processing
    print("\n⚡ Parallel processing:")
    parallel_start = time.time()

    async def process_prompt(prompt):
        client = get_client("provider_name", model=model)
        response = await client.create_completion(
            [Message(role=MessageRole.USER, content=prompt)], max_tokens=50
        )
        return response.get("response", "")[:50]

    await asyncio.gather(*[process_prompt(p) for p in prompts])
    parallel_time = time.time() - parallel_start
    print(f"   ✅ Completed in {parallel_time:.2f}s")

    # Results
    speedup = sequential_time / parallel_time if parallel_time > 0 else 0
    print("\n📈 Results:")
    print(f"   Sequential: {sequential_time:.2f}s")
    print(f"   Parallel: {parallel_time:.2f}s")
    print(f"   Speedup: {speedup:.1f}x")

    return {
        "sequential_time": sequential_time,
        "parallel_time": parallel_time,
        "speedup": speedup,
    }
```

### 3. Model Comparison Template (for providers missing it)

```python
# =============================================================================
# Example N: Model Comparison
# =============================================================================

async def model_comparison_example():
    """Compare different models from this provider"""
    print("\n📊 Model Comparison")
    print("=" * 60)

    # List 2-3 models to compare
    models = [
        "model-1",
        "model-2",
        "model-3",
    ]

    prompt = "What is the future of AI? (One sentence)"
    results = {}

    for model in models:
        try:
            print(f"🔄 Testing {model}...")
            client = get_client("provider_name", model=model)
            messages = [Message(role=MessageRole.USER, content=prompt)]

            start_time = time.time()
            response = await client.create_completion(messages, max_tokens=100)
            duration = time.time() - start_time

            results[model] = {
                "response": response.get("response", ""),
                "time": duration,
                "length": len(response.get("response", "")),
                "success": True,
            }

        except Exception as e:
            results[model] = {
                "response": f"Error: {str(e)}",
                "time": 0,
                "length": 0,
                "success": False,
            }

    print("\n📈 Results:")
    for model, result in results.items():
        status = "✅" if result["success"] else "❌"
        print(f"   {status} {model}:")
        print(f"      Time: {result['time']:.2f}s")
        print(f"      Length: {result['length']} chars")
        print(f"      Response: {result['response'][:80]}...")
        print()

    return results
```

## Implementation Checklist

For each provider file:

### Step 1: Add Missing Examples
- [ ] Add `context_window_test()` function
- [ ] Add `parallel_processing_test()` function
- [ ] Add `model_comparison_example()` (if missing)

### Step 2: Update Examples List in main()
Add to the examples list (usually in `if not args.quick:` section):
```python
examples.extend([
    ("Context Window Test", lambda: context_window_test(args.model)),
    ("Parallel Processing", lambda: parallel_processing_test(args.model)),
])
```

### Step 3: Verify Pydantic Models
Ensure all dict messages are converted to Pydantic:
```python
# ❌ OLD:
messages = [{"role": "user", "content": "Hello"}]

# ✅ NEW:
messages = [Message(role=MessageRole.USER, content="Hello")]
```

### Step 4: Verify Discovery
Ensure provider has model discovery example that:
- Uses appropriate discoverer (OpenAICompatibleDiscoverer, provider-specific discoverer)
- Shows discovered models with capability badges
- Tests a dynamically discovered model

## Standard Output Format

All examples should follow this output format:

```
🚀 Provider Name Examples
============================================================
Using model: model-name
API Key: ✅ Set
Model capabilities:
  Feature 1: ✅
  Feature 2: ✅

============================================================

📋 Example 1: Example Name
============================================================
... example output ...
✅ Example Name completed in X.XXs

============================================================

📊 SUMMARY
============================================================
✅ Successful: X/X
⏱️  Total time: XX.XXs
   ✅ Example 1: X.XXs
   ✅ Example 2: X.XXs
   ...

🎉 All examples completed successfully!
```

## Files to Update

All provider example files have been successfully updated! ✅

1. ✅ `/examples/providers/groq_usage_examples.py` - Complete ✓ (gold standard)
2. ✅ `/examples/providers/deepseek_usage_examples.py` - Complete ✓
3. ✅ `/examples/providers/openrouter_usage_examples.py` - Complete ✓
4. ✅ `/examples/providers/perplexity_usage_examples.py` - Complete ✓
5. ✅ `/examples/providers/mistral_usage_examples.py` - Complete ✓
6. ✅ `/examples/providers/anthropic_usage_examples.py` - Complete ✓
7. ✅ `/examples/providers/gemini_usage_examples.py` - Complete ✓
8. ✅ `/examples/providers/watsonx_usage_examples.py` - Complete ✓
9. ✅ `/examples/providers/azure_usage_examples.py` - Complete ✓

**Standardization completed!** All providers now have:
- Model comparison examples
- Context window tests
- Parallel processing tests
- Consistent structure and output format

## Benefits of Standardization

1. **Consistency** - Users get the same experience across all providers
2. **Completeness** - All providers demonstrate full capabilities
3. **Comparability** - Easy to compare providers using same examples
4. **Maintainability** - Standard structure makes updates easier
5. **Documentation** - Examples serve as comprehensive provider docs
