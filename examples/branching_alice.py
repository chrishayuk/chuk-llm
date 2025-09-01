#!/usr/bin/env python3
"""
Alice in Wonderland inspired conversation branching demo
"""

import asyncio
from dotenv import load_dotenv

from chuk_llm import conversation

# Load environment variables
load_dotenv()


async def alice_branching_demo():
    """
    Follow Alice down different rabbit holes while keeping the main story intact
    """
    print("🐰 === Alice's Branching Adventure ===\n")

    async with conversation(provider="openai", model="gpt-4o-mini") as alice_chat:
        # Alice's main journey begins
        print("📖 Main Story:")
        print(
            "Alice: I've just fallen down a rabbit hole and landed in a strange place!"
        )
        response = await alice_chat.ask(
            "I've just fallen down a rabbit hole and landed in a strange place!"
        )
        print(f"Narrator: {response[:200]}...\n")

        print(
            "Alice: I see a table with a bottle labeled 'DRINK ME'. What should I do?"
        )
        response = await alice_chat.ask(
            "I see a table with a bottle labeled 'DRINK ME'. What should I do?"
        )
        print(f"Narrator: {response[:200]}...\n")

        # Branch 1: What if Alice drinks the bottle?
        print("🍾 Branch 1: Alice drinks the mysterious bottle...")
        async with alice_chat.branch() as drink_branch:
            print("\n[DRINK BRANCH] Alice: I drank it! I'm shrinking rapidly!")
            response = await drink_branch.ask("I drank it! I'm shrinking rapidly!")
            print(f"[DRINK BRANCH] Narrator: {response[:200]}...\n")

            print(
                "[DRINK BRANCH] Alice: Now I'm tiny! I can fit through that little door!"
            )
            response = await drink_branch.ask(
                "Now I'm tiny! I can fit through that little door!"
            )
            print(f"[DRINK BRANCH] Narrator: {response[:200]}...\n")

            print("[DRINK BRANCH] Alice: Oh no, I left the key on the table above!")
            response = await drink_branch.ask(
                "Oh no, I left the key on the table above!"
            )
            print(f"[DRINK BRANCH] Narrator: {response[:200]}...\n")

        # Branch 2: What if Alice explores the room instead?
        print("🔍 Branch 2: Alice explores the room first...")
        async with alice_chat.branch() as explore_branch:
            print(
                "\n[EXPLORE BRANCH] Alice: Let me look around this room more carefully first."
            )
            response = await explore_branch.ask(
                "Let me look around this room more carefully first."
            )
            print(f"[EXPLORE BRANCH] Narrator: {response[:200]}...\n")

            print("[EXPLORE BRANCH] Alice: I found a cake that says 'EAT ME'!")
            response = await explore_branch.ask("I found a cake that says 'EAT ME'!")
            print(f"[EXPLORE BRANCH] Narrator: {response[:200]}...\n")

            print("[EXPLORE BRANCH] Alice: Should I eat the cake or drink the bottle?")
            response = await explore_branch.ask(
                "Should I eat the cake or drink the bottle?"
            )
            print(f"[EXPLORE BRANCH] Narrator: {response[:200]}...\n")

        # Branch 3: Meeting the Cheshire Cat
        print("😸 Branch 3: Alice meets the Cheshire Cat...")
        async with alice_chat.branch() as cat_branch:
            print("\n[CAT BRANCH] Alice: A grinning cat just appeared in the air!")
            response = await cat_branch.ask("A grinning cat just appeared in the air!")
            print(f"[CAT BRANCH] Narrator: {response[:200]}...\n")

            print("[CAT BRANCH] Cheshire Cat: 'We're all mad here. Are you mad?'")
            response = await cat_branch.ask(
                "The cat says 'We're all mad here. Are you mad?' How should I respond?"
            )
            print(f"[CAT BRANCH] Narrator: {response[:200]}...\n")

        # Back to the main story - Alice is still standing by the table
        print("📖 Back to Main Story:")
        print(
            "Alice: I'm still standing here by the table, thinking about my options..."
        )
        response = await alice_chat.ask(
            "I'm still standing here by the table, thinking about my options..."
        )
        print(f"Narrator: {response[:200]}...\n")

        print("Alice: What was I just considering? Remind me of my situation.")
        response = await alice_chat.ask(
            "What was I just considering? Remind me of my situation."
        )
        print(f"Narrator: {response}")


async def main():
    """Run Alice's branching adventure"""
    print("Welcome to Alice's Wonderland - where every choice creates a new story!\n")
    print("This demo shows how Alice can explore different possibilities")
    print("without affecting her main journey through Wonderland.\n")

    await alice_branching_demo()

    print("\n" + "=" * 60)
    print("🎭 Notice how in each branch, Alice experienced different adventures,")
    print("but when she returned to the main story, she was back to her")
    print("original situation - standing by the table with the bottle!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
