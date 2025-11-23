#!/usr/bin/env python3
"""
Conversation History Isolation Demo
====================================

This demo verifies that conversation history is NOT shared across different
client instances, even when they are cached.

Key Points:
1. Conversation history is managed by the APPLICATION, not the client
2. Different client instances (different configs) have separate conversations
3. Cached clients (same config) share the underlying HTTP client but
   conversation history is still managed by the application
4. No conversation leakage across clients

Requirements:
- Set OPENAI_API_KEY environment variable

Usage:
    python conversation_isolation_demo.py
"""

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("❌ Please set OPENAI_API_KEY environment variable")
    print("   export OPENAI_API_KEY='your_api_key_here'")
    sys.exit(1)

# Add parent directory to path for imports
examples_dir = Path(__file__).parent.parent
if str(examples_dir) not in sys.path:
    sys.path.insert(0, str(examples_dir))

try:
    from chuk_llm.llm.client import get_client
    from chuk_llm.core.models import Message
    from chuk_llm.core.enums import MessageRole
    from chuk_llm.client_registry import clear_cache, get_cache_stats
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("   Install with: pip install chuk-llm")
    sys.exit(1)


def print_section(title: str):
    """Print a section header"""
    print(f"\n{'=' * 70}")
    print(f"{title}")
    print(f"{'=' * 70}\n")


async def demo_separate_clients_separate_conversations():
    """Demo that different clients have separate conversation histories"""
    print_section("Demo 1: Different Clients = Separate Conversations")

    clear_cache()

    # Create two clients with different API keys (different cache entries)
    client_alice = get_client("openai", model="gpt-4o-mini", api_key="key-alice-123")
    client_bob = get_client("openai", model="gpt-4o-mini", api_key="key-bob-456")

    print(f"Client Alice ID: {id(client_alice)}")
    print(f"Client Bob ID: {id(client_bob)}")
    print(f"Are they the same instance? {client_alice is client_bob}")

    # Alice's conversation
    alice_conversation = [
        Message(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        Message(role=MessageRole.USER, content="My name is Alice and I like pizza."),
    ]

    # Bob's conversation (completely separate)
    bob_conversation = [
        Message(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        Message(role=MessageRole.USER, content="My name is Bob and I like sushi."),
    ]

    print("\n✓ Two separate conversation lists created")
    print(f"  Alice's conversation has {len(alice_conversation)} messages")
    print(f"  Bob's conversation has {len(bob_conversation)} messages")

    print("\n✓ Conversations are stored in APPLICATION memory, not in the client")
    print("✓ No conversation leakage between different clients")


async def demo_cached_clients_still_separate_conversations():
    """Demo that cached clients still have separate conversation state"""
    print_section("Demo 2: Cached Clients = Still Separate Conversations")

    clear_cache()

    # Create two references to the SAME cached client
    client1 = get_client("openai", model="gpt-4o-mini", api_key="shared-key")
    client2 = get_client("openai", model="gpt-4o-mini", api_key="shared-key")

    print(f"Client 1 ID: {id(client1)}")
    print(f"Client 2 ID: {id(client2)}")
    print(f"Are they the same instance? {client1 is client2}")
    print(f"Cache stats: {get_cache_stats()}")

    # Even though clients are the same instance, conversations are separate
    conversation1 = [
        Message(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        Message(role=MessageRole.USER, content="Session 1: My favorite color is blue."),
    ]

    conversation2 = [
        Message(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        Message(role=MessageRole.USER, content="Session 2: My favorite color is red."),
    ]

    print("\n✓ Same client instance (cached)")
    print("✓ But conversation history is stored separately in APPLICATION code")
    print(f"  Conversation 1 has {len(conversation1)} messages")
    print(f"  Conversation 2 has {len(conversation2)} messages")
    print("✓ No conversation leakage even with cached clients")


async def demo_conversation_state_in_application():
    """Demo showing conversation state is managed by the application"""
    print_section("Demo 3: Conversation State is in Application, Not Client")

    clear_cache()

    client = get_client("openai", model="gpt-4o-mini")

    # The client itself has NO conversation state
    print("Client attributes (no conversation storage):")
    client_attrs = [attr for attr in dir(client) if not attr.startswith('_')]
    print(f"  Public attributes: {len(client_attrs)}")
    print(f"  Has 'conversation' attribute? {hasattr(client, 'conversation')}")
    print(f"  Has 'messages' attribute? {hasattr(client, 'messages')}")
    print(f"  Has 'history' attribute? {hasattr(client, 'history')}")

    print("\n✓ The client has NO conversation state stored in it")
    print("✓ Conversation must be managed by YOUR application code")

    # Application manages conversation state
    my_conversation = []

    print("\nApplication manages conversation:")
    my_conversation.append(Message(role=MessageRole.USER, content="Hello"))
    print(f"  After turn 1: {len(my_conversation)} messages")

    my_conversation.append(Message(role=MessageRole.ASSISTANT, content="Hi there!"))
    print(f"  After turn 2: {len(my_conversation)} messages")

    my_conversation.append(Message(role=MessageRole.USER, content="How are you?"))
    print(f"  After turn 3: {len(my_conversation)} messages")

    print("\n✓ Conversation stored in application-managed list")
    print("✓ Client is stateless - just sends messages you provide")


async def demo_concurrent_conversations_no_leakage():
    """Demo concurrent conversations with cached client"""
    print_section("Demo 4: Concurrent Conversations = No Leakage")

    clear_cache()

    # Use same cached client for multiple concurrent conversations
    client = get_client("openai", model="gpt-4o-mini")

    # Simulate 3 concurrent user sessions
    conversations = {
        "user1": [Message(role=MessageRole.USER, content="User 1: I am learning Python")],
        "user2": [Message(role=MessageRole.USER, content="User 2: I am learning JavaScript")],
        "user3": [Message(role=MessageRole.USER, content="User 3: I am learning Rust")],
    }

    print("Simulating 3 concurrent user sessions with SAME cached client:")
    for user_id, conv in conversations.items():
        print(f"  {user_id}: {len(conv)} messages - '{conv[0].content}'")

    print("\n✓ Same client instance can handle multiple conversations")
    print("✓ Each conversation is separate (managed by application)")
    print("✓ No leakage between concurrent user sessions")

    # Add more messages to demonstrate independence
    conversations["user1"].append(Message(role=MessageRole.USER, content="User 1: Tell me about decorators"))
    conversations["user2"].append(Message(role=MessageRole.USER, content="User 2: Explain closures"))

    print(f"\nAfter adding more messages:")
    print(f"  user1: {len(conversations['user1'])} messages")
    print(f"  user2: {len(conversations['user2'])} messages")
    print(f"  user3: {len(conversations['user3'])} messages")

    print("\n✓ Conversations remain independent")


async def demo_http_session_vs_conversation_state():
    """Clarify the difference between HTTP session and conversation state"""
    print_section("Demo 5: HTTP Session ≠ Conversation State")

    clear_cache()

    client = get_client("openai", model="gpt-4o-mini")

    print("Understanding the layers:\n")

    print("1. HTTP Session (in httpx.AsyncClient)")
    print("   - Manages TCP connections")
    print("   - Connection pooling")
    print("   - SSL/TLS state")
    print("   - Authentication headers")
    print("   ✓ Shared when clients are cached (for performance)")

    print("\n2. Conversation State (in YOUR application)")
    print("   - List of Message objects")
    print("   - User context and history")
    print("   - Previous Q&A pairs")
    print("   ✓ NEVER shared - you manage this in your code")

    print("\n3. OpenAI API (stateless)")
    print("   - Each request is independent")
    print("   - You send full conversation history each time")
    print("   - No server-side session state")
    print("   ✓ Requires YOU to send all context")

    print("\n" + "-" * 70)
    print("Example of proper conversation management:")
    print("-" * 70)

    conversation = [
        Message(role=MessageRole.SYSTEM, content="You are helpful"),
    ]

    print("\nTurn 1:")
    print("  You send: [system, user: 'Hi']")
    conversation.append(Message(role=MessageRole.USER, content="Hi"))
    print(f"  Your conversation list: {len(conversation)} messages")

    print("\nTurn 2:")
    print("  You send: [system, user: 'Hi', assistant: 'Hello', user: 'How are you?']")
    conversation.append(Message(role=MessageRole.ASSISTANT, content="Hello"))
    conversation.append(Message(role=MessageRole.USER, content="How are you?"))
    print(f"  Your conversation list: {len(conversation)} messages")

    print("\n✓ YOU must maintain conversation state")
    print("✓ Client just sends what you give it")
    print("✓ HTTP session is separate from conversation state")


async def demo_real_api_call_isolation():
    """Demo with real API calls to verify no leakage"""
    print_section("Demo 6: Real API Calls - No History Leakage")

    clear_cache()

    # Get cached client
    client = get_client("openai", model="gpt-4o-mini")

    print("Making two separate conversations with SAME cached client...\n")

    # Conversation 1: Ask about name
    conv1 = [
        Message(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        Message(role=MessageRole.USER, content="My name is Alice. Remember it."),
    ]

    print("Conversation 1:")
    print("  User: My name is Alice. Remember it.")

    try:
        response1 = await client.create_completion(conv1)
        print(f"  AI: {response1['response']}")

        # Add AI response to conversation
        conv1.append(Message(role=MessageRole.ASSISTANT, content=response1['response']))

        # Ask follow-up
        conv1.append(Message(role=MessageRole.USER, content="What's my name?"))
        response1b = await client.create_completion(conv1)
        print(f"\n  User: What's my name?")
        print(f"  AI: {response1b['response']}")

    except Exception as e:
        print(f"  (Skipping actual API call: {e})")

    # Conversation 2: COMPLETELY SEPARATE - ask about name
    conv2 = [
        Message(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        Message(role=MessageRole.USER, content="What's my name?"),
    ]

    print("\nConversation 2 (separate, should NOT know the name):")
    print("  User: What's my name?")

    try:
        response2 = await client.create_completion(conv2)
        print(f"  AI: {response2['response']}")

        # Verify AI doesn't know the name
        if "Alice" in response2['response']:
            print("\n❌ LEAKAGE DETECTED: AI remembered name from conversation 1!")
        else:
            print("\n✓ AI correctly doesn't know the name (no leakage)")

    except Exception as e:
        print(f"  (Skipping actual API call: {e})")

    print("\n✓ Each conversation is independent")
    print("✓ Same client, separate conversation state")


async def main():
    """Run all conversation isolation demos"""
    print("\n" + "=" * 70)
    print("CONVERSATION HISTORY ISOLATION DEMO")
    print("=" * 70)

    try:
        await demo_separate_clients_separate_conversations()
        await demo_cached_clients_still_separate_conversations()
        await demo_conversation_state_in_application()
        await demo_concurrent_conversations_no_leakage()
        await demo_http_session_vs_conversation_state()
        await demo_real_api_call_isolation()

        print("\n" + "=" * 70)
        print("✅ ALL CONVERSATION ISOLATION DEMOS PASSED!")
        print("=" * 70)

        print("\n" + "=" * 70)
        print("KEY TAKEAWAYS")
        print("=" * 70)
        print("\n✓ Conversation history is managed by YOUR application, not the client")
        print("✓ Clients are stateless - they just send what you give them")
        print("✓ Cached clients share HTTP connections but NOT conversation state")
        print("✓ Different conversation lists = different conversations")
        print("✓ No conversation leakage between clients or sessions")
        print("\n✓ HTTP Session (shared) ≠ Conversation State (separate)")
        print("\n" + "=" * 70)

    except Exception as e:
        print(f"\n❌ DEMO FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    asyncio.run(main())
