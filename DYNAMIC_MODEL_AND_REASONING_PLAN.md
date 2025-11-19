# Dynamic Model + Reasoning Examples - Implementation Plan

## Requirements

All provider examples need:

### 1. Dynamic Model Test
- Use a model NOT in chuk_llm.yaml config
- Prove the library works with any model from the provider
- Test the model and show it works

### 2. Reasoning/Thoughts Example (where supported)
- Show the model's internal reasoning/thinking process
- Display reasoning tokens separately from output tokens
- Only for providers/models that support it

## Provider-Specific Details

### Providers with Reasoning Support

| Provider | Reasoning Models | Reasoning Field | Token Field |
|----------|-----------------|-----------------|-------------|
| OpenAI (Responses) | gpt-5-mini, gpt-5 | `thinking` in output | `thinking_tokens` in usage |
| Anthropic | claude-sonnet-4-5-20250514 | `thinking` blocks in content | N/A (included in output) |
| DeepSeek | deepseek-reasoner, r1-* | `reasoning_content` | `reasoning_tokens` |
| Groq | deepseek-r1-distill-llama-70b | `reasoning_content` | `reasoning_tokens` |
| Mistral | mistral-large-* (some) | `reasoning` in response | `reasoning_tokens` |

### Providers without Reasoning (Dynamic Model Only)

- Gemini
- Azure
- Watsonx
- Perplexity
- OpenRouter

## Implementation Checklist

### High Priority (Reasoning Support)
- [x] OpenAI Responses - Already complete
- [ ] Anthropic - Add both examples
- [ ] DeepSeek - Add both examples
- [ ] Groq - Add both examples
- [ ] Mistral - Add both examples

### Medium Priority (Dynamic Model Only)
- [ ] Gemini - Add dynamic model test
- [ ] Azure - Add dynamic model test
- [ ] Watsonx - Add dynamic model test
- [ ] Perplexity - Add dynamic model test
- [ ] OpenRouter - Add dynamic model test

## Standard Example Templates

### Template: Dynamic Model Test

```python
async def dynamic_model_test():
    """Test a non-configured model to prove library flexibility"""
    print("\n🔄 Dynamic Model Test")
    print("=" * 60)
    print("Testing a model NOT in chuk_llm.yaml config")

    # Use a model that's likely not in config
    dynamic_model = "provider-specific-unconfigured-model"

    print(f"\n🧪 Testing dynamic model: {dynamic_model}")
    print("   This model is NOT in the config file")

    try:
        client = get_client("provider_name", model=dynamic_model)
        messages = [Message(role=MessageRole.USER, content="Say hello in one word")]

        response = await client.create_completion(messages, max_tokens=10)
        print(f"   ✅ Dynamic model works: {response['response']}")

    except Exception as e:
        print(f"   ⚠️ Test failed: {str(e)[:100]}")
```

### Template: Reasoning/Thoughts Example

```python
async def reasoning_example(model: str = "reasoning-model"):
    """Demonstrate reasoning/thinking capabilities"""
    print(f"\n🧠 Reasoning/Thinking Example with {model}")
    print("=" * 60)

    client = get_client("provider_name", model=model)

    # Use a problem that requires reasoning
    messages = [
        Message(
            role=MessageRole.USER,
            content="I have a 3-gallon jug and a 5-gallon jug. How can I measure exactly 4 gallons?"
        )
    ]

    print("🔄 Requesting response with reasoning...")
    response = await client.create_completion(messages)

    # Provider-specific reasoning extraction
    if response.get("reasoning_content"):  # DeepSeek/Groq style
        print(f"\n🧠 Reasoning Process:")
        print(f"   {response['reasoning_content'][:300]}...")
        print(f"\n📝 Final Answer:")
        print(f"   {response['response']}")

    elif response.get("thinking"):  # Anthropic style
        print(f"\n🧠 Thinking Process:")
        print(f"   {response['thinking'][:300]}...")
        print(f"\n📝 Final Answer:")
        print(f"   {response['response']}")

    # Token breakdown
    if response.get("usage"):
        usage = response["usage"]
        print(f"\n📊 Token Usage:")
        print(f"   Input: {usage.get('input_tokens', 0)} tokens")

        if usage.get("reasoning_tokens"):
            print(f"   🧠 Reasoning: {usage['reasoning_tokens']} tokens")
            output_only = usage.get("output_tokens", 0) - usage["reasoning_tokens"]
            print(f"   📝 Output: {output_only} tokens")
        else:
            print(f"   Output: {usage.get('output_tokens', 0)} tokens")

        print(f"   Total: {usage.get('total_tokens', 0)} tokens")

    return response
```

## Notes

- OpenAI Responses API already has both examples implemented
- Some providers may have reasoning in beta/preview
- Dynamic models should be real models from the provider API
- Test examples should fail gracefully if model doesn't exist
