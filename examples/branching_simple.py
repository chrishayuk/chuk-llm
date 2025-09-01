#!/usr/bin/env python3
"""
Simplified branching demo that actually works with real API calls
"""

import asyncio
from dotenv import load_dotenv
from chuk_llm import conversation

# Load environment variables
load_dotenv()


async def simple_branching_demo():
    """
    Demonstrate branching with fewer API calls
    """
    print("🌳 === Simple Branching Demo ===\n")
    
    async with conversation(provider="openai", model="gpt-4o-mini") as chat:
        # Main conversation
        print("📖 Main conversation:")
        print("User: What's 2+2?")
        response = await chat.ask("What's 2+2?")
        print(f"AI: {response}\n")
        
        # Branch 1: Explore math further
        print("🔀 Branch 1: Going deeper into math...")
        async with chat.branch() as math_branch:
            print("  [Branch] User: What about 10*10?")
            response = await math_branch.ask("What about 10*10?")
            print(f"  [Branch] AI: {response}\n")
        
        # Branch 2: Different topic
        print("🔀 Branch 2: Switching to history...")
        async with chat.branch() as history_branch:
            print("  [Branch] User: Who was the first president?")
            response = await history_branch.ask("Who was the first president of the United States?")
            print(f"  [Branch] AI: {response}\n")
        
        # Back to main - should only remember 2+2
        print("📖 Back to main conversation:")
        print("User: What was my original question?")
        response = await chat.ask("What was my original question?")
        print(f"AI: {response}\n")
        
        # Get conversation stats
        stats = chat.get_stats()
        print(f"📊 Stats: {stats['total_messages']} messages in main conversation")
        print(f"   Branch count: {stats['branch_count']}")


async def main():
    """Run the simple branching demo"""
    print("This demo shows conversation branching with real API calls.\n")
    
    try:
        await simple_branching_demo()
        print("\n✅ Demo completed successfully!")
        print("Notice how the main conversation doesn't know about the branch questions!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Make sure your OPENAI_API_KEY is set in the .env file")


if __name__ == "__main__":
    asyncio.run(main())