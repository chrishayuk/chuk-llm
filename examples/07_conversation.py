#!/usr/bin/env python3
"""
Conversation example - maintain context across messages
"""

import asyncio
from dotenv import load_dotenv
load_dotenv()

from chuk_llm.api.conversation import conversation

async def chat(conv, user_message):
    """Helper to display and send message"""
    print(f"User: {user_message}")
    response = await conv.say(user_message)
    print(f"AI: {response}\n")
    return response

async def main():
    print("=== Conversation Example ===\n")
    
    async with conversation(provider="openai") as conv:
        # Have a conversation - the AI remembers everything
        await chat(conv, "My name is Alice")
        await chat(conv, "What's my name?")
        await chat(conv, "Nice to meet you too!")

if __name__ == "__main__":
    asyncio.run(main())