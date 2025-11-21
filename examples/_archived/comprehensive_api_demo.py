#!/usr/bin/env python3
"""Comprehensive demo showing ask/stream with and without tools, inside and outside conversations"""

import asyncio
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
from async_helper import run_async_clean
from dotenv import load_dotenv
load_dotenv()

from chuk_llm import ask, stream, ask_sync, conversation, conversation_sync
from chuk_llm import Tools, tool

# Define demo tools
class DemoTools(Tools):
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

async def demo_ask_without_tools():
    """Demo basic ask without tools"""
    print("\n=== DEMO: ask without tools ===")
    try:
        response = await ask("What is 2+2?", model="gpt-4o-mini")
        print(f"Response: {response[:100]}...")
        print("✅ Success")
    except Exception as e:
        print(f"❌ Error: {e}")

def demo_ask_sync_without_tools():
    """Demo sync ask without tools"""
    print("\n=== DEMO: ask_sync without tools ===")
    try:
        response = ask_sync("What is 3+3?", model="gpt-4o-mini")
        print(f"Response: {response[:100]}...")
        print("✅ Success")
    except Exception as e:
        print(f"❌ Error: {e}")

async def demo_stream_without_tools():
    """Demo streaming without tools"""
    print("\n=== DEMO: stream without tools ===")
    try:
        chunks = []
        async for chunk in stream("Count to 3", model="gpt-4o-mini"):
            chunks.append(chunk)
        response = "".join(chunks)
        print(f"Streamed response: {response[:100]}...")
        print("✅ Success")
    except Exception as e:
        print(f"❌ Error: {e}")

async def demo_ask_with_tools():
    """Demo ask with tools"""
    print("\n=== DEMO: ask with tools ===")
    try:
        tools = DemoTools()
        response = await tools.ask("What is 10 + 15?", model="gpt-4o-mini")
        print(f"Response: {response[:100]}...")
        print("✅ Success")
    except Exception as e:
        print(f"❌ Error: {e}")

async def demo_ask_with_function_tools():
    """Demo ask with function tools"""
    print("\n=== DEMO: ask with function tools ===")
    # Create tool definition in OpenAI format
    weather_tool = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "The location to get weather for"}
                },
                "required": ["location"]
            }
        }
    }
    
    try:
        response = await ask(
            "What's the weather in Paris?",
            tools=[weather_tool],
            model="gpt-4o-mini"
        )
        print(f"Response type: {type(response)}")
        # Response should be a dict when tools are provided
        if response is None:
            print("Response: None (tool call expected but no auto-execution)")
            print("✅ Demonstration complete - tools require execution")
        elif isinstance(response, dict):
            print(f"Response text: {response.get('response', '')[:100] if response.get('response') else 'None'}...")
            print(f"Tool calls: {response.get('tool_calls', [])}")
            print("✅ Demonstration complete")
        else:
            print(f"Response: {str(response)[:100]}...")
            print("✅ Demonstration complete")
    except Exception as e:
        print(f"❌ Error: {e}")

async def demo_conversation_ask():
    """Demo ask inside conversation"""
    print("\n=== DEMO: ask inside conversation ===")
    try:
        async with conversation(provider="openai", model="gpt-4o-mini") as conv:
            response1 = await conv.ask("My name is Alice")
            print(f"Response 1: {response1[:100]}...")
            
            response2 = await conv.ask("What's my name?")
            print(f"Response 2: {response2[:100]}...")
            
            if "Alice" in response2:
                print("✅ Conversation memory working!")
            else:
                print("⚠️ Model may not have retained context")
    except Exception as e:
        print(f"❌ Error: {e}")

def demo_conversation_sync():
    """Demo sync conversation"""
    print("\n=== DEMO: sync conversation ===")
    try:
        with conversation_sync(provider="openai", model="gpt-4o-mini") as conv:
            response1 = conv.ask("My favorite color is blue")
            print(f"Response 1: {response1[:100]}...")
            
            response2 = conv.ask("What's my favorite color?")
            print(f"Response 2: {response2[:100]}...")
            
            if "blue" in response2.lower():
                print("✅ Sync conversation memory working!")
            else:
                print("⚠️ Model may not have retained context")
    except Exception as e:
        print(f"❌ Error: {e}")

async def demo_conversation_with_tools():
    """Demo tools inside conversation"""
    print("\n=== DEMO: tools inside conversation ===")
    try:
        tools = DemoTools()
        
        # First call with tools
        response = await tools.ask("What is 20 + 30?", model="gpt-4o-mini")
        print(f"Tool response: {response[:100]}...")
        
        # Then use in conversation
        async with conversation(provider="openai", model="gpt-4o-mini") as conv:
            response = await conv.ask("I need help with math. What's 15 + 25?")
            print(f"Conversation response: {response[:100]}...")
        
        print("✅ Tools and conversation both working")
    except Exception as e:
        print(f"❌ Error: {e}")

async def demo_stream_in_conversation():
    """Demo streaming inside conversation"""
    print("\n=== DEMO: stream inside conversation ===")
    try:
        async with conversation(provider="openai", model="gpt-4o-mini") as conv:
            await conv.ask("Remember the number 42")
            
            chunks = []
            async for chunk in conv.stream("What number did I ask you to remember?"):
                chunks.append(chunk)
            response = "".join(chunks)
            print(f"Streamed response: {response[:100]}...")
            
            if "42" in response:
                print("✅ Streaming in conversation working!")
            else:
                print("⚠️ Model may not have retained context")
    except Exception as e:
        print(f"❌ Error: {e}")

async def run_async_demos():
    """Run all async demos"""
    print("\n" + "="*60)
    print("COMPREHENSIVE API DEMO - ASYNC")
    print("="*60)
    
    await demo_ask_without_tools()
    await demo_stream_without_tools()
    await demo_ask_with_tools()
    await demo_ask_with_function_tools()
    await demo_conversation_ask()
    await demo_conversation_with_tools()
    await demo_stream_in_conversation()
    
    # Small delay to allow cleanup
    await asyncio.sleep(0.1)

def run_sync_demos():
    """Run all sync demos"""
    print("\n" + "="*60)
    print("COMPREHENSIVE API DEMO - SYNC")
    print("="*60)
    
    demo_ask_sync_without_tools()
    demo_conversation_sync()

def main():
    """Run all demos"""
    print("🎭 Running Comprehensive API Demonstrations")
    print("Demonstrating ask/stream with and without tools")
    print("Testing inside and outside conversations\n")
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not found in environment")
        print("This demo requires an OpenAI API key to run")
        print("Set it with: export OPENAI_API_KEY='your-key-here'")
        return
    
    # Run async demos with clean helper
    run_async_clean(run_async_demos())
    
    # Run sync demos  
    run_sync_demos()
    
    print("\n" + "="*60)
    print("✅ All demonstrations complete!")
    print("="*60)

if __name__ == "__main__":
    main()