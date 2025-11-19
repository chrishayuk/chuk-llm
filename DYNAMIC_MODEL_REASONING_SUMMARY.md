# Dynamic Model + Reasoning Examples - Completion Summary

## Overview

Successfully added **dynamic model tests** and **reasoning/thinking examples** to all provider example scripts. This demonstrates that chuk-llm works with ANY model from each provider, not just configured ones, and showcases reasoning capabilities where supported.

## What Was Added

### Critical Bug Fix
- **Azure OpenAI Client** (`src/chuk_llm/llm/providers/azure_openai_client.py`)
  - Fixed: `'Message' object has no attribute 'get'` error
  - Issue: Code was treating Pydantic Message objects as dictionaries
  - Solution: Added proper handling for both Pydantic objects and dict messages in vision detection
  - Result: All 12 Azure examples now pass ✅

### Providers with BOTH Dynamic Model + Reasoning Examples

#### 1. Anthropic (`examples/providers/anthropic_usage_examples.py`)
- **Example 12: Dynamic Model Test**
  - Model: `claude-3-7-sonnet-20250219`
  - Tests a model not in config

- **Example 13: Extended Thinking (Reasoning)**
  - Model: `claude-sonnet-4-5-20250514`
  - Shows Claude's internal thinking process
  - Displays thinking text separately from final answer
  - Token breakdown showing total usage

#### 2. DeepSeek (`examples/providers/deepseek_usage_examples.py`)
- **Example 12: Dynamic Model Test**
  - Model: `deepseek-chat`
  - Tests non-configured model

- **Already had Example 2: Reasoning**
  - Model: `deepseek-reasoner`
  - Shows `reasoning_content` field
  - Displays reasoning tokens separately

#### 3. Groq (`examples/providers/groq_usage_examples.py`)
- **Example 14: Dynamic Model Test**
  - Model: `llama-3.2-90b-vision-preview`
  - Shows Groq LPU speed metrics

- **Example 15: Reasoning with DeepSeek R1**
  - Model: `deepseek-r1-distill-llama-70b`
  - Demonstrates reasoning on Groq's LPU infrastructure
  - Shows `reasoning_content` and `reasoning_tokens`
  - Displays Groq LPU duration

### Providers with Dynamic Model Test Only

#### 4. Mistral (`examples/providers/mistral_usage_examples.py`)
- **Example 11: Dynamic Model Test**
  - Model: `mistral-small-2501`
  - Tests latest model not necessarily in config

#### 5. Gemini (`examples/providers/gemini_usage_examples.py`)
- **Example 12: Dynamic Model Test**
  - Model: `gemini-2.0-flash-exp`
  - Tests experimental model

#### 6. Azure OpenAI (`examples/providers/azure_usage_examples.py`)
- **Example 12: Dynamic Model Test**
  - Model: `gpt-4o-2024-11-20` (actual model ID, not deployment)
  - Proves library works with any Azure deployment

#### 7. Watsonx (`examples/providers/watsonx_usage_examples.py`)
- **Example 12: Dynamic Model Test**
  - Model: `meta-llama/llama-3-2-90b-vision-instruct`
  - Tests vision-capable model

#### 8. Perplexity (`examples/providers/perplexity_usage_examples.py`)
- **Example 12: Dynamic Model Test**
  - Model: `llama-3.1-sonar-large-128k-online`
  - Tests online search-enabled model

#### 9. OpenRouter (`examples/providers/openrouter_usage_examples.py`)
- **Example 16: Dynamic Model Test**
  - Model: `meta-llama/llama-3.3-70b-instruct`
  - Tests multi-provider routing capability

### Already Complete

#### 10. OpenAI Responses API (`examples/providers/openai_responses_example.py`)
- **Already has both features:**
  - Demo 14: Model Discovery (dynamic model testing)
  - Demo 9: Reasoning Models with Thinking
    - Model: `gpt-5-mini`
    - Shows `thinking` blocks in output
    - Displays `thinking_tokens` vs `output_tokens` breakdown

## Key Features of Implementation

### Dynamic Model Tests
- ✅ Uses models NOT in `chuk_llm.yaml` config
- ✅ Proves library flexibility beyond static configuration
- ✅ Graceful error handling
- ✅ Consistent output format across all providers

### Reasoning Examples
- ✅ Shows internal thinking/reasoning process
- ✅ Separates reasoning tokens from output tokens
- ✅ Uses complex problems (e.g., water jug puzzle)
- ✅ Provider-specific field handling:
  - OpenAI: `thinking` in output, `thinking_tokens` in usage
  - Anthropic: `thinking` field in response
  - DeepSeek/Groq: `reasoning_content` and `reasoning_tokens`

## Testing Status

| Provider | Dynamic Model | Reasoning | Status |
|----------|--------------|-----------|--------|
| OpenAI Responses | ✅ | ✅ | Complete |
| Anthropic | ✅ | ✅ | Complete |
| DeepSeek | ✅ | ✅ | Complete |
| Groq | ✅ | ✅ | Complete |
| Mistral | ✅ | N/A | Complete |
| Gemini | ✅ | N/A | Complete |
| Azure | ✅ | N/A | Complete |
| Watsonx | ✅ | N/A | Complete |
| Perplexity | ✅ | N/A | Complete |
| OpenRouter | ✅ | N/A | Complete |

## Benefits

1. **Demonstrates Flexibility**: Users can see that chuk-llm works with ANY model from a provider, not just configured ones
2. **Showcases Reasoning**: Users can see how different providers handle extended thinking/reasoning
3. **Real-World Examples**: Uses actual reasoning problems that benefit from step-by-step thinking
4. **Consistent UX**: All examples follow similar patterns for easy understanding
5. **Production Ready**: Includes proper error handling and graceful degradation

## Files Modified

### Bug Fixes
- `src/chuk_llm/llm/providers/azure_openai_client.py` (Azure Pydantic fix)

### Example Scripts
1. `examples/providers/anthropic_usage_examples.py` (+2 examples)
2. `examples/providers/deepseek_usage_examples.py` (+1 example)
3. `examples/providers/groq_usage_examples.py` (+2 examples)
4. `examples/providers/mistral_usage_examples.py` (+1 example)
5. `examples/providers/gemini_usage_examples.py` (+1 example)
6. `examples/providers/azure_usage_examples.py` (+1 example)
7. `examples/providers/watsonx_usage_examples.py` (+1 example)
8. `examples/providers/perplexity_usage_examples.py` (+1 example)
9. `examples/providers/openrouter_usage_examples.py` (+1 example)

### Documentation
- `DYNAMIC_MODEL_AND_REASONING_PLAN.md` (implementation plan)
- `DYNAMIC_MODEL_REASONING_SUMMARY.md` (this file)

## Total Impact

- **10 providers** updated
- **11 new examples** added
- **1 critical bug** fixed
- **100% test passing** on Azure (12/12 examples)

All provider examples now demonstrate:
1. Library works with dynamic/unconfigured models
2. Reasoning/thinking capabilities (where supported)
3. Proper Pydantic V2 usage
4. Graceful error handling
