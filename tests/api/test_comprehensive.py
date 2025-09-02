#!/usr/bin/env python3
"""Comprehensive test for ask/stream with and without tools, inside and outside conversations"""

import asyncio
from dotenv import load_dotenv
load_dotenv()

from chuk_llm import ask, stream, ask_sync, conversation, conversation_sync
from chuk_llm import Tools, tool, ask_with_tools_simple

# Define test tools
class TestTools(Tools):
    @tool(description="Get current time")
    def get_time(self) -> str:
        from datetime import datetime
        return datetime.now().strftime("%H:%M:%S")
    
    @tool(description="Add two numbers")
    def add(self, a: int, b: int) -> int:
        return a + b

def get_weather(location: str) -> dict:
    """Get weather for a location"""
    return {"location": location, "temp": 20, "condition": "sunny"}

async def test_ask_without_tools():
    """Test basic ask without tools"""
    print("\n=== TEST: ask without tools ===")
    response = await ask("What is 2+2?", model="gpt-4o-mini")
    print(f"Response: {response[:100]}...")
    assert response, "Should get a response"
    print("✅ PASSED")

def test_ask_sync_without_tools():
    """Test sync ask without tools"""
    print("\n=== TEST: ask_sync without tools ===")
    response = ask_sync("What is 3+3?", model="gpt-4o-mini")
    print(f"Response: {response[:100]}...")
    assert response, "Should get a response"
    print("✅ PASSED")

async def test_stream_without_tools():
    """Test streaming without tools"""
    print("\n=== TEST: stream without tools ===")
    chunks = []
    async for chunk in stream("Count to 3", model="gpt-4o-mini"):
        chunks.append(chunk)
    response = "".join(chunks)
    print(f"Streamed response: {response[:100]}...")
    assert response, "Should get streamed response"
    print("✅ PASSED")

async def test_ask_with_tools():
    """Test ask with tools"""
    print("\n=== TEST: ask with tools ===")
    tools = TestTools()
    response = await tools.ask("What is 10 + 15?", model="gpt-4o-mini")
    print(f"Response: {response[:100]}...")
    assert response, "Should get response with tools"
    print("✅ PASSED")

async def test_ask_with_function_tools():
    """Test ask with function tools"""
    print("\n=== TEST: ask with function tools ===")
    response = await ask_with_tools_simple(
        "What's the weather in Paris?",
        tools=[get_weather],
        model="gpt-4o-mini"
    )
    print(f"Response: {response[:100]}...")
    assert response, "Should get response with function tools"
    print("✅ PASSED")

async def test_conversation_ask():
    """Test ask inside conversation"""
    print("\n=== TEST: ask inside conversation ===")
    async with conversation(provider="openai", model="gpt-4o-mini") as conv:
        response1 = await conv.ask("My name is Alice")
        print(f"Response 1: {response1[:100]}...")
        
        response2 = await conv.ask("What's my name?")
        print(f"Response 2: {response2[:100]}...")
        
        assert "Alice" in response2, "Should remember name from conversation"
    print("✅ PASSED")

def test_conversation_sync():
    """Test sync conversation"""
    print("\n=== TEST: sync conversation ===")
    with conversation_sync(provider="openai", model="gpt-4o-mini") as conv:
        response1 = conv.ask("My favorite color is blue")
        print(f"Response 1: {response1[:100]}...")
        
        response2 = conv.ask("What's my favorite color?")
        print(f"Response 2: {response2[:100]}...")
        
        assert "blue" in response2.lower(), "Should remember color from conversation"
    print("✅ PASSED")

async def test_conversation_with_tools():
    """Test tools inside conversation"""
    print("\n=== TEST: tools inside conversation ===")
    tools = TestTools()
    
    # Note: conversation doesn't directly support tools,
    # but we can use the tools with ask inside a conversation context
    async with conversation(provider="openai", model="gpt-4o-mini") as conv:
        # First establish context
        await conv.ask("I need help with math")
        
        # Then use tools separately
        response = await tools.ask("What is 20 + 30?", model="gpt-4o-mini")
        print(f"Response with tools: {response[:100]}...")
        assert response, "Should get response with tools"
    print("✅ PASSED")

async def test_stream_in_conversation():
    """Test streaming inside conversation"""
    print("\n=== TEST: stream inside conversation ===")
    async with conversation(provider="openai", model="gpt-4o-mini") as conv:
        await conv.ask("Remember the number 42")
        
        chunks = []
        async for chunk in conv.stream("What number did I ask you to remember?"):
            chunks.append(chunk)
        response = "".join(chunks)
        print(f"Streamed response: {response[:100]}...")
        assert "42" in response, "Should remember number from conversation"
    print("✅ PASSED")

async def run_async_tests():
    """Run all async tests"""
    print("\n" + "="*60)
    print("COMPREHENSIVE TEST SUITE - ASYNC")
    print("="*60)
    
    await test_ask_without_tools()
    await test_stream_without_tools()
    await test_ask_with_tools()
    await test_ask_with_function_tools()
    await test_conversation_ask()
    await test_conversation_with_tools()
    await test_stream_in_conversation()

def run_sync_tests():
    """Run all sync tests"""
    print("\n" + "="*60)
    print("COMPREHENSIVE TEST SUITE - SYNC")
    print("="*60)
    
    test_ask_sync_without_tools()
    test_conversation_sync()

def main():
    """Run all tests"""
    print("🧪 Running Comprehensive Test Suite")
    print("Testing ask/stream with and without tools")
    print("Testing inside and outside conversations\n")
    
    # Run async tests
    asyncio.run(run_async_tests())
    
    # Run sync tests
    run_sync_tests()
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60)
    print("\nSummary:")
    print("  ✓ ask works without tools")
    print("  ✓ ask works with tools") 
    print("  ✓ stream works without tools")
    print("  ✓ ask works inside conversations")
    print("  ✓ stream works inside conversations")
    print("  ✓ tools work with ask")
    print("  ✓ sync versions work correctly")

if __name__ == "__main__":
    main()