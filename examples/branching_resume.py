#!/usr/bin/env python3
import asyncio
from chuk_llm import conversation

async def main():
    # Start conversation and save it
    conversation_id = None
    async with conversation(provider="openai") as chat:
        print("Main: I like cats")
        await chat.say("I like cats")
        
        async with chat.branch() as branch:
            print("Branch: Actually, I prefer dogs")
            await branch.say("Actually, I prefer dogs")
            
            conversation_id = await branch.save()  # Save the branch!
        
        print("Main: What do I like?")
        response = await chat.say("What do I like?")
        print(f"AI: {response}")  # Should say "cats"
    
    # Resume from the dogs branch
    print("\n--- Resuming from dogs branch ---")
    async with conversation(resume_from=conversation_id) as chat:
        print("Resumed: What breeds are good for apartments?")
        response = await chat.say("What breeds are good for apartments?")
        print(f"AI: {response}")

if __name__ == "__main__":
    asyncio.run(main())