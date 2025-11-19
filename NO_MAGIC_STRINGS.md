# No Magic Strings Policy ✨

## Overview

This codebase follows a **zero magic strings** policy. All constant values are defined as enums or typed constants.

## What We Did

### Before (Magic Strings Everywhere)
```python
# BAD - Magic strings all over
if provider == "openai":  # Typo risk!
    response = requests.post(
        "/chat/completions",  # What if this changes?
        headers={"Content-Type": "application/json"},  # Repeated everywhere
    )

    if response.status_code == 401:  # What does 401 mean?
        error_type = "authentication_error"  # Another magic string!
```

### After (Type-Safe Constants)
```python
# GOOD - Type-safe enums and constants
if provider == Provider.OPENAI:  # IDE autocomplete!
    response = requests.post(
        OpenAIEndpoint.CHAT_COMPLETIONS.value,  # Centralized
        headers={
            HttpHeader.CONTENT_TYPE.value: ContentTypeValue.JSON.value
        },
    )

    if response.status_code == HttpStatus.UNAUTHORIZED:  # Self-documenting
        error_type = ErrorType.AUTHENTICATION_ERROR  # Type-safe
```

## Available Constants

### Enums (`core/enums.py`)

```python
from chuk_llm.core import (
    Provider,           # LLM providers
    Feature,            # Model capabilities
    MessageRole,        # Chat roles
    FinishReason,       # Completion reasons
    ContentType,        # Multimodal content
    ToolType,           # Function call types
    ReasoningGeneration # Reasoning model generations
)
```

### Constants (`core/constants.py`)

```python
from chuk_llm.core import (
    # HTTP
    HttpMethod,         # GET, POST, etc.
    HttpHeader,         # Authorization, Content-Type, etc.
    HttpStatus,         # Status code numbers
    ContentTypeValue,   # MIME types

    # API Endpoints
    OpenAIEndpoint,     # OpenAI API paths
    AnthropicEndpoint,  # Anthropic API paths

    # Errors
    ErrorType,          # Error categories

    # Response Keys
    ResponseKey,        # API response field names
    RequestParam,       # API request parameter names

    # Configuration
    ConfigKey,          # Config file keys
    EnvVar,             # Environment variable names

    # SSE (Server-Sent Events)
    SSEPrefix,          # Event stream prefixes
    SSEEvent,           # Event types

    # Patterns
    ModelPattern,       # Regex patterns for detection

    # Defaults
    Default,            # Default configuration values
)
```

## Usage Examples

### HTTP Requests
```python
# Before
response = client.post("/chat/completions", headers={"Authorization": f"Bearer {key}"})

# After
response = client.post(
    OpenAIEndpoint.CHAT_COMPLETIONS.value,
    headers={HttpHeader.AUTHORIZATION.value: f"Bearer {key}"}
)
```

### Error Handling
```python
# Before
if status == 429:
    return {"error": "rate_limit_error", "retry_after": 60}

# After
if status == HttpStatus.RATE_LIMIT:
    return LLMError(
        error_type=ErrorType.RATE_LIMIT_ERROR.value,
        retry_after=60
    )
```

### Response Parsing
```python
# Before
content = response["choices"][0]["message"]["content"]
tool_calls = response.get("tool_calls", [])

# After
content = response[ResponseKey.CHOICES.value][0][ResponseKey.MESSAGE.value][ResponseKey.CONTENT.value]
tool_calls = response.get(ResponseKey.TOOL_CALLS.value, [])
```

### Environment Variables
```python
# Before
api_key = os.getenv("OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

# After
api_key = os.getenv(EnvVar.OPENAI_API_KEY.value)
endpoint = os.getenv(EnvVar.AZURE_OPENAI_ENDPOINT.value)
```

### Configuration
```python
# Before
config = {"default_provider": "openai", "timeout": 60.0}

# After
config = ProviderConfigModel(
    name=Provider.OPENAI,
    timeout=Default.TIMEOUT,
    # Pydantic validates everything!
)
```

## Benefits

### 1. **Type Safety**
- IDE autocomplete works everywhere
- Typos caught at development time
- mypy/pyright catch errors before running

### 2. **Refactoring Safety**
- Change a value once, updates everywhere
- Find all usages with "Find References"
- No missed string replacements

### 3. **Self-Documenting**
```python
# Which is clearer?
if status == 401:  # What's 401?

# vs
if status == HttpStatus.UNAUTHORIZED:  # Ah, unauthorized!
```

### 4. **Centralized Management**
- All constants in one place
- Easy to see what values exist
- No duplication across files

## Rules

1. **Never use string literals for:**
   - Provider names → Use `Provider` enum
   - HTTP methods → Use `HttpMethod` enum
   - HTTP headers → Use `HttpHeader` enum
   - Status codes → Use `HttpStatus` constants
   - Error types → Use `ErrorType` enum
   - API endpoints → Use endpoint constants
   - Response keys → Use `ResponseKey` enum
   - Environment vars → Use `EnvVar` enum

2. **Exceptions (when magic strings are OK):**
   - User-provided content (prompts, responses)
   - Dynamic data from APIs (model names from discovery)
   - Log messages
   - Error messages for humans

3. **Adding New Constants:**
   ```python
   # Add to appropriate enum/constant class
   class NewEndpoint(str, Enum):
       MY_ENDPOINT = "/my/endpoint"

   # Export from __init__.py
   from .constants import NewEndpoint

   __all__ = [..., "NewEndpoint"]
   ```

## Migration Checklist

When updating code:
- [ ] Replace `"openai"` → `Provider.OPENAI`
- [ ] Replace `"POST"` → `HttpMethod.POST`
- [ ] Replace `"/chat/completions"` → `OpenAIEndpoint.CHAT_COMPLETIONS`
- [ ] Replace `"choices"` → `ResponseKey.CHOICES`
- [ ] Replace `401` → `HttpStatus.UNAUTHORIZED`
- [ ] Replace `"rate_limit_error"` → `ErrorType.RATE_LIMIT_ERROR`
- [ ] Replace `"OPENAI_API_KEY"` → `EnvVar.OPENAI_API_KEY`

## Impact

**Code Quality:**
- ✅ 100% type-safe constants
- ✅ Zero tolerance for magic strings
- ✅ IDE support everywhere

**Developer Experience:**
- ✅ Autocomplete for all constants
- ✅ Compile-time error detection
- ✅ Easier to understand code

**Maintainability:**
- ✅ Single source of truth
- ✅ Easy refactoring
- ✅ No missed updates
