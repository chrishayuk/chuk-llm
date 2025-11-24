# Debug Tools for chuk-llm

This directory contains debugging and diagnostic tools for developing and testing LLM provider clients.

## debug_openai_compatible_function_calling.py

A comprehensive diagnostic tool that tests OpenAI-compatible APIs to determine what function calling methods they support.

### What it tests

1. **Native OpenAI tools parameter** - Standard OpenAI function calling (current format)
2. **Legacy functions parameter** - Pre-2023 OpenAI function calling format
3. **Tools with system prompt** - Whether aggressive system prompts trigger function calls
4. **JSON mode** - Whether the model outputs function calls as JSON when instructed
5. **Tool result formats** - Which message roles work for tool results (`tool`, `user`, `function`)
6. **Available models** - Lists models available from the API

### Usage

```bash
# Test any OpenAI-compatible API
python debug_openai_compatible_function_calling.py \
    --provider <provider_name> \
    --model <model_name>

# Examples
python debug_openai_compatible_function_calling.py \
    --provider advantage \
    --model global/gpt-5-chat

python debug_openai_compatible_function_calling.py \
    --provider deepseek \
    --model deepseek-chat

python debug_openai_compatible_function_calling.py \
    --provider groq \
    --model llama-3.3-70b-versatile
```

### Environment Variables

The script looks for API credentials in environment variables:
- `<PROVIDER>_API_KEY` - API key for the provider
- `<PROVIDER>_API_BASE` - API base URL (optional, can be passed as `--api-base`)

Example:
```bash
export ADVANTAGE_API_KEY="your-key-here"
export ADVANTAGE_API_BASE="your-api-base-url"
python debug_openai_compatible_function_calling.py --provider advantage --model global/gpt-5-chat
```

### Output and Recommendations

The script provides clear recommendations based on test results:

#### ✅ Native OpenAI tools support
```
📝 RECOMMENDATION: Use native OpenAI tools
   ├─ This API fully supports OpenAI-style function calling
   ├─ Client can extend OpenAILLMClient directly
   └─ No custom implementation needed (like Moonshot client)
```
→ Client should extend `OpenAILLMClient` directly

#### ✅ JSON mode required
```
📝 RECOMMENDATION: Use JSON mode (like current Advantage implementation)
   ├─ Inject system prompt to guide JSON function calling
   ├─ Parse JSON from response content field
   ├─ Convert to standard tool_calls format
   ├─ Tool result formats that work: user_role
   └─ ⚠️  Convert 'tool' role messages to 'user' role (API doesn't support tool role)
```
→ Client should extend `OpenAICompatibleWithJSONFallback`

#### ✅ Legacy functions support
```
📝 RECOMMENDATION: Use legacy functions parameter
   ├─ This API uses pre-2023 OpenAI function calling
   ├─ Convert tools to functions format in client
   └─ Parse function_call instead of tool_calls
```
→ Client needs custom conversion logic

### Using Results to Build Clients

**If native tools work:**
```python
from .openai_client import OpenAILLMClient

class MyProviderClient(OpenAILLMClient):
    """Simple wrapper - no custom function calling needed"""
    pass
```

**If JSON mode required:**
```python
from .openai_client import OpenAILLMClient

class MyProviderClient(OpenAILLMClient):
    """Uses JSON fallback for function calling"""
    ENABLE_JSON_FUNCTION_FALLBACK = True
    SUPPORTS_TOOL_ROLE = False  # Set based on test results
    SUPPORTS_FUNCTION_ROLE = False
```

### Example Output

```
======================================================================
OPENAI-COMPATIBLE API FUNCTION CALLING DEBUG
======================================================================
Provider: advantage
API Base: <your-api-base-url>
Model: global/gpt-5-chat
API Key: <your-api-key>
======================================================================

======================================================================
TEST 1: Native OpenAI tools parameter
======================================================================
✓ API call successful: True
✓ Has tool_calls: False
✓ Content: I don't have access to real-time data...

======================================================================
TEST 4: JSON mode for function calls
======================================================================
✓ API call successful: True
✓ Content: {"name": "get_weather", "arguments": {"location": "Tokyo"}}
✅ WORKS: Model returned JSON function call!

======================================================================
TEST 5: Tool result message formats
======================================================================
✓ Got function call: {"name": "get_weather", "arguments": {"location": "Tokyo"}}

  Testing 'tool_role':
    ⚠️  API accepted but response doesn't use tool result

  Testing 'user_role':
    ✅ Works! Response: The current weather in Tokyo is sunny...

✅ Working formats: user_role

======================================================================
SUMMARY & RECOMMENDATIONS
======================================================================
✅ 1 approach(es) work!

📝 RECOMMENDATION: Use JSON mode (like current Advantage implementation)
   ├─ Inject system prompt to guide JSON function calling
   ├─ Parse JSON from response content field
   ├─ Convert to standard tool_calls format
   ├─ Tool result formats that work: user_role
   └─ ⚠️  Convert 'tool' role messages to 'user' role (API doesn't support tool role)
```

### Adding New Tests

To add a new test to the script:

1. Create an async function following the naming pattern `test_N_descriptive_name()`
2. Add it to the `results` dict in `main()`
3. Add interpretation logic in the summary section

See existing tests as examples.
