#!/usr/bin/env python3
# examples/enhanced_conversation_demo.py
"""
Demonstration of enhanced conversation features
"""

import asyncio
from pathlib import Path

from chuk_llm import ask, conversation


async def demo_1_branching():
    """Demo 1: Conversation Branching"""
    print("\n=== Demo 1: Conversation Branching ===")

    async with conversation(provider="openai") as chat:
        # Main conversation thread
        print("\nMain thread:")
        print("User: Tell me about Python")
        response = await chat.ask("Tell me about Python")
        print(f"AI: {response[:200]}...")

        # Create a branch to explore a tangent
        async with chat.branch() as side_chat:
            print("\n[Branch] User: What about Ruby instead?")
            response = await side_chat.ask("What about Ruby instead?")
            print(f"[Branch] AI: {response[:200]}...")

            print("\n[Branch] User: Which is better for web development?")
            response = await side_chat.ask("Which is better for web development?")
            print(f"[Branch] AI: {response[:200]}...")

        # Back to main thread - it doesn't know about Ruby
        print("\nMain thread (continuing):")
        print("User: What were we discussing?")
        response = await chat.ask("What were we discussing?")
        print(f"AI: {response}")


async def demo_2_persistence():
    """Demo 2: Conversation Persistence"""
    print("\n=== Demo 2: Conversation Persistence ===")

    # Start a conversation and save it
    conversation_id = None

    async with conversation(provider="openai") as chat:
        print("\nStarting conversation...")
        print("User: I'm planning a trip to Japan")
        await chat.ask("I'm planning a trip to Japan")

        print("User: I'm interested in temples and technology")
        await chat.ask("I'm interested in temples and technology")

        # Save the conversation
        conversation_id = await chat.save()
        print(f"\nConversation saved with ID: {conversation_id}")

    # Resume the conversation later
    print("\n--- Resuming conversation later ---")
    async with conversation(resume_from=conversation_id) as chat:
        print("User: Based on my interests, what cities should I visit?")
        response = await chat.ask("Based on my interests, what cities should I visit?")
        print(f"AI: {response[:300]}...")


async def demo_3_multimodal():
    """Demo 3: Multi-Modal Conversations"""
    print("\n=== Demo 3: Multi-Modal Conversations ===")

    # Note: This requires a provider that supports vision (e.g., GPT-4V)
    # Using a placeholder image path - replace with actual image
    image_path = "example_diagram.png"

    if not Path(image_path).exists():
        print("Note: Image file not found, using text-only example")
        async with conversation(provider="openai", model="gpt-4o") as chat:
            print("User: Describe what you would see in a network diagram")
            response = await chat.ask(
                "Describe what you would see in a network diagram"
            )
            print(f"AI: {response[:300]}...")
    else:
        async with conversation(provider="openai", model="gpt-4o") as chat:
            print(f"User: What do you see in this image? [Image: {image_path}]")
            response = await chat.ask(
                "What do you see in this image?", image=image_path
            )
            print(f"AI: {response[:300]}...")

            # Continue with context
            print("\nUser: Can you explain the connections?")
            response = await chat.ask("Can you explain the connections?")
            print(f"AI: {response[:300]}...")


async def demo_4_utilities():
    """Demo 4: Conversation Utilities"""
    print("\n=== Demo 4: Conversation Utilities ===")

    async with conversation(provider="openai") as chat:
        # Have a conversation
        await chat.ask("Let's discuss the history of computing")
        await chat.ask("Tell me about Charles Babbage")
        await chat.ask("What was the Analytical Engine?")
        await chat.ask("How did it influence modern computers?")

        # Get conversation summary
        print("\nGenerating summary...")
        summary = await chat.summarize(max_length=200)
        print(f"Summary: {summary}")

        # Extract key points
        print("\nExtracting key points...")
        key_points = await chat.extract_key_points()
        print("Key points:")
        for point in key_points:
            print(f"  - {point}")

        # Get statistics
        print("\nConversation statistics:")
        stats = await chat.get_session_stats()
        print(f"  Total messages: {stats['total_messages']}")
        print(f"  Estimated tokens: {stats['estimated_tokens']}")
        print(f"  Conversation ID: {stats['conversation_id']}")


async def demo_5_stateless_context():
    """Demo 5: Stateless Calls with Context"""
    print("\n=== Demo 5: Stateless Calls with Context ===")

    # Quick contextual question without full conversation
    print("\nStateless with context string:")
    response = await ask(
        "What's the capital?", context="We're discussing France and its major cities"
    )
    print("User: What's the capital?")
    print("Context: We're discussing France and its major cities")
    print(f"AI: {response}")

    # With message history
    print("\n\nStateless with message history:")
    previous_messages = [
        {"role": "user", "content": "Tell me about the solar system"},
        {
            "role": "assistant",
            "content": "The solar system consists of the Sun and all objects that orbit it, including eight planets...",
        },
        {"role": "user", "content": "Focus on the gas giants"},
        {
            "role": "assistant",
            "content": "The gas giants are Jupiter, Saturn, Uranus, and Neptune...",
        },
    ]

    response = await ask(
        "Which one is the largest?", previous_messages=previous_messages
    )
    print("User: Which one is the largest?")
    print(f"AI: {response}")


async def demo_6_custom_system_prompt():
    """Demo 6: Custom System Prompts"""
    print("\n=== Demo 6: Custom System Prompts ===")

    # Create conversation with custom system prompt
    custom_prompt = """You are an expert Python tutor.
    - Explain concepts clearly with examples
    - Use simple language suitable for beginners
    - Provide code examples when helpful
    - Encourage questions and learning"""

    async with conversation(provider="openai", system_prompt=custom_prompt) as chat:
        print("Using custom system prompt for Python tutor")
        print("\nUser: What is a list comprehension?")
        response = await chat.ask("What is a list comprehension?")
        print(f"AI: {response[:400]}...")


async def main():
    """Run all demos"""
    print("Enhanced Conversation Features Demo")
    print("=" * 50)

    # Run each demo
    demos = [
        demo_1_branching,
        demo_2_persistence,
        demo_3_multimodal,
        demo_4_utilities,
        demo_5_stateless_context,
        demo_6_custom_system_prompt,
    ]

    for demo in demos:
        try:
            await demo()
            print("\n" + "=" * 50)
        except Exception as e:
            print(f"\nError in {demo.__name__}: {e}")
            print("=" * 50)

        # Small delay between demos
        await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(main())
