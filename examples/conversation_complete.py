#!/usr/bin/env python3
"""
Complete Conversation API Examples
===================================

This file demonstrates all conversation features in both async and sync modes:
- Basic ask/response
- Streaming (async only)
- Branching
- Memory persistence
- Saving and resuming
"""

import asyncio
from dotenv import load_dotenv
from chuk_llm import conversation, conversation_sync

# Load environment variables
load_dotenv()


# ============================================================================
# ASYNC EXAMPLES (Recommended for production)
# ============================================================================

async def async_basic_conversation():
    """Basic async conversation with memory"""
    print("\n📝 Async Basic Conversation")
    print("-" * 40)
    
    async with conversation(provider="openai", model="gpt-4o-mini") as chat:
        # First message
        await chat.ask("My name is Alice and I love hiking")
        
        # Memory test
        response = await chat.ask("What's my name and hobby?")
        print(f"Memory Test: {response}")
        
        # Get history
        history = chat.get_history()
        print(f"Total messages: {len(history)}")


async def async_streaming():
    """Streaming responses in real-time (async only)"""
    print("\n🌊 Async Streaming")
    print("-" * 40)
    
    async with conversation(provider="openai", model="gpt-4o-mini") as chat:
        print("Streaming response: ", end="")
        async for chunk in chat.stream("Count from 1 to 5 with one word per number"):
            print(chunk, end="", flush=True)
        print()  # New line after streaming


async def async_branching():
    """Branching to explore alternatives"""
    print("\n🌳 Async Branching")
    print("-" * 40)
    
    async with conversation(provider="openai", model="gpt-4o-mini") as chat:
        # Main conversation
        await chat.ask("I'm learning Python for data science")
        
        # Branch 1: Explore web development
        async with chat.branch() as web_branch:
            await web_branch.ask("Actually, what about web development?")
            web_response = await web_branch.ask("What framework should I start with?")
            print(f"Web Branch: {web_response[:100]}...")
        
        # Branch 2: Explore machine learning
        async with chat.branch() as ml_branch:
            await ml_branch.ask("What about machine learning?")
            ml_response = await ml_branch.ask("What libraries do I need?")
            print(f"ML Branch: {ml_response[:100]}...")
        
        # Main conversation continues with original context
        response = await chat.ask("What Python libraries are best for data science?")
        print(f"Main (Data Science): {response[:100]}...")


async def async_save_and_resume():
    """Save conversation and resume later"""
    print("\n💾 Async Save & Resume")
    print("-" * 40)
    
    # Start and save conversation
    conversation_id = None
    async with conversation(provider="openai", model="gpt-4o-mini") as chat:
        await chat.ask("I'm planning a trip to Japan")
        await chat.ask("I want to visit Tokyo and Kyoto")
        conversation_id = await chat.save()
        print(f"Saved conversation: {conversation_id[:8]}...")
    
    # Resume conversation
    async with conversation(resume_from=conversation_id) as chat:
        response = await chat.ask("What cities did I mention?")
        print(f"Resumed memory: {response}")


async def async_advanced_features():
    """Advanced features like summarization"""
    print("\n✨ Async Advanced Features")
    print("-" * 40)
    
    async with conversation(provider="openai", model="gpt-4o-mini") as chat:
        # Have a conversation
        await chat.ask("Let's discuss climate change")
        await chat.ask("What are the main causes?")
        await chat.ask("What can individuals do to help?")
        
        # Get summary
        summary = await chat.summarize(max_length=100)
        print(f"Summary: {summary}")
        
        # Extract key points
        key_points = await chat.extract_key_points()
        print(f"Key points: {', '.join(key_points[:3])}")
        
        # Get statistics
        stats = chat.get_stats()
        print(f"Stats: {stats['total_messages']} messages, {stats['branch_count']} branches")


# ============================================================================
# SYNC EXAMPLES (Simpler for scripts)
# ============================================================================

def sync_basic_conversation():
    """Basic sync conversation"""
    print("\n📝 Sync Basic Conversation")
    print("-" * 40)
    
    with conversation_sync(provider="openai", model="gpt-4o-mini") as chat:
        # First message
        chat.ask("My name is Bob and I enjoy cooking")
        
        # Memory test
        response = chat.ask("What's my name and what do I enjoy?")
        print(f"Memory Test: {response}")
        
        # Get history
        history = chat.get_history()
        print(f"Total messages: {len(history)}")


def sync_branching():
    """Branching in sync mode"""
    print("\n🌳 Sync Branching")
    print("-" * 40)
    
    with conversation_sync(provider="openai", model="gpt-4o-mini") as chat:
        # Main conversation
        chat.ask("I want to learn a new programming language")
        
        # Branch to explore Python
        with chat.branch() as python_branch:
            python_branch.ask("Tell me about Python")
            py_response = python_branch.ask("What's Python best for?")
            print(f"Python Branch: {py_response[:100]}...")
        
        # Branch to explore JavaScript
        with chat.branch() as js_branch:
            js_branch.ask("Tell me about JavaScript")
            js_response = js_branch.ask("What's JavaScript best for?")
            print(f"JS Branch: {js_response[:100]}...")
        
        # Main conversation continues
        response = chat.ask("Which language should I choose as a beginner?")
        print(f"Main recommendation: {response[:100]}...")


def sync_save_and_resume():
    """Save and resume in sync mode"""
    print("\n💾 Sync Save & Resume")
    print("-" * 40)
    
    # Start and save
    with conversation_sync(provider="openai", model="gpt-4o-mini") as chat:
        chat.ask("I'm learning to play guitar")
        chat.ask("I know C, G, and D chords")
        conversation_id = chat.save()
        print(f"Saved: {conversation_id[:8]}...")
    
    # Resume
    with conversation_sync(resume_from=conversation_id) as chat:
        response = chat.ask("What chords do I know?")
        print(f"Resumed memory: {response}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

async def run_async_examples():
    """Run all async examples"""
    print("\n" + "=" * 50)
    print("ASYNC EXAMPLES (Full Features)")
    print("=" * 50)
    
    await async_basic_conversation()
    await async_streaming()
    await async_branching()
    await async_save_and_resume()
    await async_advanced_features()


def run_sync_examples():
    """Run all sync examples"""
    print("\n" + "=" * 50)
    print("SYNC EXAMPLES (Simple Interface)")
    print("=" * 50)
    
    sync_basic_conversation()
    sync_branching()
    sync_save_and_resume()
    
    print("\n📌 Note: For real-time streaming, use async API")


async def main():
    """Run all examples"""
    print("🎯 ChukLLM Conversation API - Complete Examples")
    print("This demonstrates all conversation features\n")
    
    # Run async examples
    await run_async_examples()
    
    # Run sync examples
    run_sync_examples()
    
    print("\n✅ All examples completed!")
    print("See the code for implementation details")


if __name__ == "__main__":
    # For async examples only:
    # asyncio.run(run_async_examples())
    
    # For sync examples only:
    # run_sync_examples()
    
    # Run everything:
    asyncio.run(main())