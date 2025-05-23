# tests/test_openai_client.py
import sys
import types
import pytest

# -----------------------------------------------------------------------------
# Provide a stub "openai" module *before* importing the client implementation so
# the real SDK is never required (and no network calls are made).
# -----------------------------------------------------------------------------

dummy_openai = types.ModuleType("openai")


class _DummyCompletions:
    # Placeholder attribute – tests will monkey‑patch the actual callable
    create = lambda *a, **k: None  # noqa: E731


class _DummyChat:
    completions = _DummyCompletions()


class DummyOpenAI:
    """Mimics ``openai.OpenAI`` enough for the client to instantiate."""

    def __init__(self, *args, **kwargs):  # accept arbitrary kwargs
        self.chat = _DummyChat()


dummy_openai.OpenAI = DummyOpenAI
sys.modules["openai"] = dummy_openai

# Now the import is safe – it will pick up our stub instead of the real SDK.
from chuk_llm.llm.providers.openai_client import OpenAILLMClient  # noqa: E402  pylint: disable=wrong-import-position


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


@pytest.fixture
def client():
    """Return a *fresh* client instance for each test."""
    return OpenAILLMClient(model="test-model", api_key="sk-test")


# -----------------------------------------------------------------------------
# Tests – non‑streaming completion
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_completion_one_shot(monkeypatch, client):
    messages = [{"role": "user", "content": "Hi!"}]
    tools_in = [{"name": "demo"}]

    # Make tool sanitisation a no‑op so we can assert exactly what is forwarded.
    monkeypatch.setattr(client, "_sanitize_tool_names", lambda t: t)

    # Mock the _regular_completion method instead of _call_blocking
    async def fake_regular_completion(msgs, tls, **kwargs):
        assert msgs == messages
        assert tls == tools_in
        return {"response": "Hello", "tool_calls": []}

    monkeypatch.setattr(client, "_regular_completion", fake_regular_completion)

    # Use new interface - get awaitable and then await it
    result_awaitable = client.create_completion(messages, tools=tools_in, stream=False)
    result = await result_awaitable
    assert result == {"response": "Hello", "tool_calls": []}

@pytest.mark.asyncio
async def test_create_completion_stream(monkeypatch, client):
    messages = [{"role": "user", "content": "Hello again"}]

    # Mock the _stream_completion_async method
    async def fake_stream_completion_async(msgs, tls, **kwargs):
        assert msgs == messages
        assert tls == []  # No tools provided
        yield {"response": "Hello", "tool_calls": []}
        yield {"response": " World", "tool_calls": []}

    monkeypatch.setattr(client, "_stream_completion_async", fake_stream_completion_async)

    # Use new interface - get async generator directly
    async_iter = client.create_completion(messages, tools=None, stream=True)
    assert hasattr(async_iter, "__aiter__")
    
    pieces = [chunk async for chunk in async_iter]
    assert pieces == [
        {"response": "Hello", "tool_calls": []},
        {"response": " World", "tool_calls": []}
    ]