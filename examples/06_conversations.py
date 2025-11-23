#!/usr/bin/env python3
"""
Conversations with Memory
==========================

Demonstrates stateful conversations where the AI
remembers previous messages in the conversation.
"""

import asyncio
from dotenv import load_dotenv
load_dotenv()

from chuk_llm import conversation

async def basic_conversation():
    """Basic conversation with memory."""
    print("=== Basic Conversation ===\n")

    async with conversation() as chat:
        # First message
        response = await chat.ask("My name is Alice")
        print(f"AI: {response}\n")

        # Second message - AI remembers the name
        response = await chat.ask("What's my name?")
        print(f"AI: {response}\n")

        # Third message - continues the conversation
        response = await chat.ask("What did we just talk about?")
        print(f"AI: {response}\n")

async def conversation_with_provider():
    """Conversation with specific provider."""
    print("=== Conversation with Specific Provider ===\n")

    async with conversation(provider="anthropic", model="claude-3-5-haiku-20241022") as chat:
        response = await chat.ask("I love Python programming")
        print(f"Claude: {response}\n")

        response = await chat.ask("What language do I love?")
        print(f"Claude: {response}\n")

async def branching_conversations():
    """Conversation branching - explore different paths."""
    print("=== Branching Conversations ===\n")

    async with conversation() as chat:
        await chat.ask("I'm planning a vacation")
        print("Main conversation: Planning a vacation\n")

        # Branch 1: Explore Japan
        async with chat.branch() as japan_branch:
            response = await japan_branch.ask("Tell me about visiting Japan")
            print(f"Japan branch: {response[:100]}...\n")

        # Branch 2: Explore Italy
        async with chat.branch() as italy_branch:
            response = await italy_branch.ask("Tell me about visiting Italy")
            print(f"Italy branch: {response[:100]}...\n")

        # Main conversation unaffected by branches
        response = await chat.ask("I've decided to go with Japan!")
        print(f"Main conversation: {response[:100]}...\n")

async def save_and_resume():
    """Save conversation and resume later."""
    print("=== Save and Resume ===\n")

    conversation_id = None

    # Start conversation
    async with conversation() as chat:
        await chat.ask("My favorite color is blue")
        print("Conversation started: told AI about favorite color\n")

        # Save the conversation
        conversation_id = await chat.save()
        print(f"Conversation saved with ID: {conversation_id}\n")

    # Resume conversation later (even in another session)
    async with conversation(resume_from=conversation_id) as chat:
        response = await chat.ask("What's my favorite color?")
        print(f"Resumed conversation - AI: {response}\n")

async def conversation_streaming():
    """Streaming responses in conversations."""
    print("=== Conversation with Streaming ===\n")

    async with conversation() as chat:
        # Regular message
        await chat.ask("I'm learning about Python")

        # Stream response
        print("AI: ", end="", flush=True)
        async for chunk in chat.stream("Tell me something cool about Python"):
            print(chunk, end="", flush=True)
        print("\n")

async def conversation_history():
    """Access conversation history."""
    print("=== Conversation History ===\n")

    async with conversation() as chat:
        await chat.ask("Hello, I'm Bob")
        await chat.ask("I like pizza")
        await chat.ask("What do you know about me?")

        # Get conversation history
        history = chat.get_history()
        print(f"Total messages in conversation: {len(history)}\n")

        for i, msg in enumerate(history, 1):
            role = msg.role.value if hasattr(msg.role, 'value') else msg.role
            content = str(msg.content)[:50] + "..." if len(str(msg.content)) > 50 else msg.content
            print(f"{i}. {role}: {content}")

        print()

async def conversation_with_system_prompt():
    """Conversation with custom system prompt."""
    print("=== Conversation with System Prompt ===\n")

    system_prompt = """You are a helpful pirate assistant.
    Always respond in pirate speak with 'Arrr!' and nautical terms."""

    async with conversation(system_prompt=system_prompt) as chat:
        response = await chat.ask("What's the weather like?")
        print(f"Pirate AI: {response}\n")

async def multi_turn_problem_solving():
    """Use conversation for multi-turn problem solving."""
    print("=== Multi-Turn Problem Solving ===\n")

    async with conversation() as chat:
        # Step 1
        response = await chat.ask(
            "I need to build a web API. What technology should I use?"
        )
        print(f"AI: {response[:100]}...\n")

        # Step 2 - follows up on previous answer
        response = await chat.ask("Which framework is best for beginners?")
        print(f"AI: {response[:100]}...\n")

        # Step 3 - continues the thread
        response = await chat.ask("Show me a simple example")
        print(f"AI: {response[:100]}...\n")

if __name__ == "__main__":
    asyncio.run(basic_conversation())
    asyncio.run(conversation_with_provider())
    asyncio.run(branching_conversations())
    asyncio.run(save_and_resume())
    asyncio.run(conversation_streaming())
    asyncio.run(conversation_history())
    asyncio.run(conversation_with_system_prompt())
    asyncio.run(multi_turn_problem_solving())

    print("="*50)
    print("✅ Conversations make building chatbots easy!")
    print("="*50)
