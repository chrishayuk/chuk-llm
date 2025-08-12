#!/usr/bin/env python3
"""
Granite 3.3 Debug Analysis
==========================
Comprehensive analysis of Granite's tool calling behavior,
especially focusing on system prompts and repetitive behavior.
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


async def test_without_system_prompt():
    """Test Granite WITHOUT any system prompt"""
    
    import ollama
    
    print("="*60)
    print("TEST 1: NO SYSTEM PROMPT")
    print("="*60)
    
    messages = [
        {"role": "user", "content": "Get the weather for Tokyo, then stop."}
    ]
    
    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"}
                },
                "required": ["location"]
            }
        }
    }]
    
    client = ollama.AsyncClient()
    
    for round_num in range(1, 4):  # Max 3 rounds
        print(f"\n📍 Round {round_num}:")
        print(f"  Sending: {messages[-1]['content'][:100]}")
        
        response = await client.chat(
            model="granite3.3:latest",
            messages=messages,
            tools=tools,
            options={"temperature": 0.1}
        )
        
        if hasattr(response, 'message'):
            msg = response.message
            
            # Check content
            content = getattr(msg, 'content', '')
            if content:
                print(f"  Response: {content[:100]}")
                messages.append({"role": "assistant", "content": content})
            
            # Check tool calls
            tool_calls = getattr(msg, 'tool_calls', None)
            if tool_calls:
                for tc in tool_calls:
                    if hasattr(tc, 'function'):
                        func = tc.function
                        name = getattr(func, 'name', '')
                        args = getattr(func, 'arguments', {})
                        print(f"  🔧 Tool call: {name}({json.dumps(args)})")
                
                # Provide result and ask to stop
                messages.append({
                    "role": "user", 
                    "content": "Tool result: Weather in Tokyo: 18°C, Sunny. Task complete, please stop."
                })
            else:
                if round_num == 1:
                    print("  ⚠️ No tool calls made")
                break


async def test_with_minimal_system_prompt():
    """Test with a minimal system prompt"""
    
    import ollama
    
    print("\n" + "="*60)
    print("TEST 2: MINIMAL SYSTEM PROMPT")
    print("="*60)
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use tools when asked, then stop after completing the task."},
        {"role": "user", "content": "Get the weather for Tokyo, then stop."}
    ]
    
    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"}
                },
                "required": ["location"]
            }
        }
    }]
    
    client = ollama.AsyncClient()
    
    for round_num in range(1, 4):
        print(f"\n📍 Round {round_num}:")
        print(f"  Sending: {messages[-1]['content'][:100]}")
        
        response = await client.chat(
            model="granite3.3:latest",
            messages=messages,
            tools=tools,
            options={"temperature": 0.1}
        )
        
        if hasattr(response, 'message'):
            msg = response.message
            
            content = getattr(msg, 'content', '')
            if content:
                print(f"  Response: {content[:100]}")
                messages.append({"role": "assistant", "content": content})
            
            tool_calls = getattr(msg, 'tool_calls', None)
            if tool_calls:
                for tc in tool_calls:
                    if hasattr(tc, 'function'):
                        func = tc.function
                        name = getattr(func, 'name', '')
                        args = getattr(func, 'arguments', {})
                        print(f"  🔧 Tool call: {name}({json.dumps(args)})")
                
                messages.append({
                    "role": "user", 
                    "content": "Tool result: Weather in Tokyo: 18°C, Sunny. Task complete."
                })
            else:
                if round_num == 1:
                    print("  ⚠️ No tool calls made")
                break


async def test_with_explicit_stop_system_prompt():
    """Test with explicit instructions to stop after task completion"""
    
    import ollama
    
    print("\n" + "="*60)
    print("TEST 3: EXPLICIT STOP INSTRUCTIONS")
    print("="*60)
    
    messages = [
        {"role": "system", "content": """You are an assistant that uses tools efficiently.
IMPORTANT RULES:
1. Call each tool only ONCE per unique request
2. After receiving tool results, provide a summary
3. Do NOT repeat tool calls you've already made
4. Stop after completing the requested task"""},
        {"role": "user", "content": "Get the weather for Tokyo, then stop."}
    ]
    
    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"}
                },
                "required": ["location"]
            }
        }
    }]
    
    client = ollama.AsyncClient()
    
    for round_num in range(1, 4):
        print(f"\n📍 Round {round_num}:")
        print(f"  Sending: {messages[-1]['content'][:100]}")
        
        response = await client.chat(
            model="granite3.3:latest",
            messages=messages,
            tools=tools,
            options={"temperature": 0.1}
        )
        
        if hasattr(response, 'message'):
            msg = response.message
            
            content = getattr(msg, 'content', '')
            if content:
                print(f"  Response: {content[:100]}")
                messages.append({"role": "assistant", "content": content})
            
            tool_calls = getattr(msg, 'tool_calls', None)
            if tool_calls:
                for tc in tool_calls:
                    if hasattr(tc, 'function'):
                        func = tc.function
                        name = getattr(func, 'name', '')
                        args = getattr(func, 'arguments', {})
                        print(f"  🔧 Tool call: {name}({json.dumps(args)})")
                
                messages.append({
                    "role": "user", 
                    "content": "Tool result: Weather in Tokyo: 18°C, Sunny. Remember: task is complete, do not call tools again."
                })
            else:
                if round_num == 1:
                    print("  ⚠️ No tool calls made")
                break


async def test_multi_city_behavior():
    """Test how Granite handles multiple cities"""
    
    import ollama
    
    print("\n" + "="*60)
    print("TEST 4: MULTI-CITY BEHAVIOR")
    print("="*60)
    
    messages = [
        {"role": "user", "content": "Get weather for Tokyo, New York, and Paris. Call get_weather once for each city."}
    ]
    
    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"}
                },
                "required": ["location"]
            }
        }
    }]
    
    client = ollama.AsyncClient()
    tool_call_log = []
    
    print("\nTracking tool calls across rounds:")
    
    for round_num in range(1, 5):
        print(f"\n📍 Round {round_num}:")
        
        response = await client.chat(
            model="granite3.3:latest",
            messages=messages,
            tools=tools,
            options={"temperature": 0.1}
        )
        
        if hasattr(response, 'message'):
            msg = response.message
            
            content = getattr(msg, 'content', '')
            if content:
                messages.append({"role": "assistant", "content": content})
            
            tool_calls = getattr(msg, 'tool_calls', None)
            round_calls = []
            
            if tool_calls:
                for tc in tool_calls:
                    if hasattr(tc, 'function'):
                        func = tc.function
                        name = getattr(func, 'name', '')
                        args = getattr(func, 'arguments', {})
                        location = args.get('location', '?') if isinstance(args, dict) else '?'
                        round_calls.append(location)
                        tool_call_log.append({"round": round_num, "location": location})
                        print(f"  🔧 {name}({location})")
                
                # Provide results
                results = []
                for loc in round_calls:
                    results.append(f"Weather in {loc}: 18°C, Sunny")
                
                # Different continuation strategies
                if round_num == 1:
                    # First round: just give results
                    messages.append({
                        "role": "user",
                        "content": f"Tool results:\n" + "\n".join(results) + "\n\nContinue if needed."
                    })
                elif round_num == 2:
                    # Second round: hint about what's been done
                    all_locations = [tc["location"] for tc in tool_call_log]
                    unique = set(all_locations)
                    messages.append({
                        "role": "user",
                        "content": f"Results received. You've now called: {', '.join(unique)}. Continue only if cities are missing."
                    })
                else:
                    # Later rounds: be explicit
                    messages.append({
                        "role": "user",
                        "content": "All cities have been checked. Please summarize and stop."
                    })
            else:
                print("  No more tool calls")
                break
    
    # Analysis
    print(f"\n📊 Tool Call Analysis:")
    print(f"  Total calls: {len(tool_call_log)}")
    locations = [tc["location"] for tc in tool_call_log]
    unique = set(locations)
    print(f"  Unique cities: {unique}")
    print(f"  Call distribution:")
    for city in unique:
        count = locations.count(city)
        print(f"    {city}: {count} time(s)")


async def test_with_chuk_llm():
    """Test using ChukLLM to see if system prompt is added"""
    
    from chuk_llm import stream
    
    print("\n" + "="*60)
    print("TEST 5: CHUKLLM DEFAULT BEHAVIOR")
    print("="*60)
    print("Testing if ChukLLM adds a system prompt that causes issues\n")
    
    prompt = "Get the weather for Tokyo, then stop."
    
    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"}
                },
                "required": ["location"]
            }
        }
    }]
    
    print("First call:")
    call_count = 0
    
    async for chunk in stream(
        prompt,
        provider="ollama",
        model="granite3.3:latest",
        tools=tools,
        return_tool_calls=True,
        temperature=0.1,
        max_tokens=200
    ):
        if isinstance(chunk, dict) and chunk.get("tool_calls"):
            for tc in chunk["tool_calls"]:
                call_count += 1
                func = tc.get("function", {})
                name = func.get("name", "")
                args = json.loads(func.get("arguments", "{}"))
                print(f"  🔧 Tool call {call_count}: {name}({args.get('location', '?')})")
    
    print(f"\nTotal tool calls in first response: {call_count}")
    
    if call_count > 1:
        print("⚠️ Multiple calls for single city - likely a system prompt issue!")
    elif call_count == 1:
        print("✅ Single call as expected")
    else:
        print("❌ No tool calls made")


async def compare_models():
    """Quick comparison between Granite and GPT-OSS"""
    
    import ollama
    
    print("\n" + "="*60)
    print("MODEL COMPARISON: Granite vs GPT-OSS")
    print("="*60)
    
    prompt = "Get weather for Tokyo and then stop."
    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }
        }
    }]
    
    client = ollama.AsyncClient()
    
    for model in ["granite3.3:latest", "gpt-oss:latest"]:
        print(f"\n📊 Testing {model}:")
        
        messages = [{"role": "user", "content": prompt}]
        
        response = await client.chat(
            model=model,
            messages=messages,
            tools=tools,
            options={"temperature": 0.1}
        )
        
        if hasattr(response, 'message'):
            msg = response.message
            
            content = getattr(msg, 'content', '')
            if content:
                print(f"  Response: {content[:100]}")
            
            tool_calls = getattr(msg, 'tool_calls', None)
            if tool_calls:
                print(f"  Tool calls: {len(tool_calls)}")
                for tc in tool_calls:
                    if hasattr(tc, 'function'):
                        func = tc.function
                        name = getattr(func, 'name', '')
                        args = getattr(func, 'arguments', {})
                        print(f"    - {name}({json.dumps(args)})")
            else:
                print(f"  No tool calls")


async def main():
    """Run all debug tests"""
    
    print("🔍 GRANITE 3.3 TOOL CALLING DEBUG ANALYSIS")
    print("="*60)
    print("Investigating repetitive behavior and system prompt issues\n")
    
    # Test 1: No system prompt
    await test_without_system_prompt()
    
    # Test 2: Minimal system prompt
    await test_with_minimal_system_prompt()
    
    # Test 3: Explicit stop instructions
    await test_with_explicit_stop_system_prompt()
    
    # Test 4: Multi-city behavior
    await test_multi_city_behavior()
    
    # Test 5: ChukLLM default
    await test_with_chuk_llm()
    
    # Test 6: Model comparison
    await compare_models()
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("\nKey Questions:")
    print("1. Does system prompt affect repetition?")
    print("2. Does Granite respect 'stop' instructions?")
    print("3. Is ChukLLM adding a problematic system prompt?")
    print("4. How does Granite compare to GPT-OSS?")


if __name__ == "__main__":
    asyncio.run(main())