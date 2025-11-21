#!/usr/bin/env python3
import asyncio

from chuk_llm import conversation


async def main():
    async with conversation(provider="openai") as chat:
        # First exchange
        print("\nUser: My name is Alice and I'm learning Python")
        response = await chat.ask("My name is Alice and I'm learning Python")
        print(f"AI: {response}")

        # Test memory - name
        print("\nUser: What's my name?")
        response = await chat.ask("What's my name?")
        print(f"AI: {response}")

        # Test memory - topic
        print("\nUser: What am I learning?")
        response = await chat.ask("What am I learning?")
        print(f"AI: {response}")

        # Build on context
        print("\nUser: Can you suggest a beginner project?")
        response = await chat.ask("Can you suggest a beginner project?")
        print(f"AI: {response}")


if __name__ == "__main__":
    asyncio.run(main())
