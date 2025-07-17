#!/usr/bin/env python3
"""
Enhanced Anthropic Streaming Diagnostic

Tests both accumulation strategies and event handling to identify streaming issues.
Compares Raw Anthropic behavior with chuk-llm implementation.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# Load environment
try:
    from dotenv import load_dotenv
    project_root = Path(__file__).parent.parent
    env_file = project_root / ".env"
    if env_file.exists():
        load_dotenv(env_file)
        print(f"✅ Loaded .env")
    else:
        load_dotenv()
except ImportError:
    print("⚠️ No dotenv")


async def test_anthropic_streaming_strategies():
    """Test different Anthropic streaming accumulation strategies."""
    
    print("🔍 ANTHROPIC STREAMING STRATEGY ANALYSIS")
    print("=" * 55)
    
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ No ANTHROPIC_API_KEY")
        return False
    
    # Test case that should trigger multiple streaming events
    tools = [{
        "type": "function",
        "function": {
            "name": "execute_complex_query",
            "description": "Execute a complex database query with multiple parameters",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "SQL query to execute"},
                    "database": {"type": "string", "description": "Database name"},
                    "timeout": {"type": "integer", "description": "Query timeout in seconds"},
                    "options": {
                        "type": "object",
                        "properties": {
                            "explain": {"type": "boolean", "description": "Include execution plan"},
                            "format": {"type": "string", "enum": ["json", "csv", "table"]}
                        }
                    }
                },
                "required": ["query", "database"]
            }
        }
    }]
    
    messages = [{
        "role": "user", 
        "content": "Execute SQL query 'SELECT u.id, u.name, p.title FROM users u JOIN posts p ON u.id = p.user_id WHERE u.status = \"active\" AND p.published = true ORDER BY p.created_at DESC LIMIT 25' on database 'production_db' with timeout 30 seconds, include execution plan and format as json"
    }]
    
    print("🎯 Test: Complex SQL with multiple nested parameters")
    print("Expected: Should generate multiple input_json_delta events")
    
    # Test raw Anthropic with proper accumulation
    print("\n🔥 RAW ANTHROPIC ANALYSIS:")
    raw_proper = await test_raw_anthropic_proper_accumulation(api_key, messages, tools)
    
    # Test raw Anthropic with broken accumulation (for comparison)
    print("\n🚨 RAW ANTHROPIC (BROKEN LOGIC):")
    raw_broken = await test_raw_anthropic_broken_accumulation(api_key, messages, tools)
    
    # Test chuk-llm current behavior
    print("\n🔧 CHUK-LLM CURRENT:")
    chuk_result = await test_chuk_llm_anthropic(messages, tools)
    
    # Detailed comparison
    print("\n📊 DETAILED COMPARISON:")
    print(f"Raw (proper):     '{raw_proper}'")
    print(f"Raw (broken):     '{raw_broken}'")
    print(f"Chuk-LLM current: '{chuk_result}'")
    
    # Analyze results
    return analyze_anthropic_results({
        "raw_proper": raw_proper,
        "raw_broken": raw_broken,
        "chuk_current": chuk_result
    })


async def test_raw_anthropic_proper_accumulation(api_key, messages, tools):
    """Test raw Anthropic with PROPER streaming accumulation."""
    try:
        import anthropic
        
        client = anthropic.AsyncAnthropic(api_key=api_key)
        
        # Convert OpenAI-style tools to Anthropic format
        anthropic_tools = []
        for tool in tools:
            fn = tool["function"]
            anthropic_tools.append({
                "name": fn["name"],
                "description": fn["description"],
                "input_schema": fn["parameters"]
            })
        
        # PROPER: Track tool calls and accumulate JSON parts correctly
        tool_calls = {}  # {tool_id: {name, input_json_parts, initial_input}}
        event_count = 0
        tool_events = 0
        input_json_events = 0
        
        print(f"  Using model: claude-3-5-sonnet-20241022")
        
        async with client.messages.stream(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2048,
            messages=messages,
            tools=anthropic_tools,
            tool_choice={"type": "auto"}
        ) as stream:
            
            async for event in stream:
                event_count += 1
                
                if hasattr(event, 'type'):
                    if event.type == 'content_block_start':
                        if hasattr(event, 'content_block') and event.content_block.type == 'tool_use':
                            tool_events += 1
                            tool_id = event.content_block.id
                            tool_name = event.content_block.name
                            initial_input = getattr(event.content_block, 'input', {})
                            
                            print(f"    Tool start: {tool_name} (id: {tool_id})")
                            print(f"    Initial input: {initial_input}")
                            
                            tool_calls[tool_id] = {
                                "name": tool_name,
                                "initial_input": initial_input,
                                "input_json_parts": [],
                                "content_index": getattr(event, 'index', 0)
                            }
                    
                    elif event.type == 'content_block_delta':
                        if hasattr(event, 'delta') and hasattr(event.delta, 'type'):
                            if event.delta.type == 'input_json_delta':
                                input_json_events += 1
                                content_index = getattr(event, 'index', 0)
                                
                                print(f"    JSON delta #{input_json_events}: '{event.delta.partial_json}'")
                                
                                # PROPER: Find tool by content index and accumulate
                                for tool_id, tool_data in tool_calls.items():
                                    if tool_data.get('content_index') == content_index:
                                        tool_data['input_json_parts'].append(event.delta.partial_json)
                                        break
                    
                    elif event.type == 'content_block_stop':
                        content_index = getattr(event, 'index', 0)
                        
                        # PROPER: Finalize by reconstructing complete JSON
                        for tool_id, tool_data in tool_calls.items():
                            if tool_data.get('content_index') == content_index:
                                final_input = tool_data["initial_input"]
                                
                                if tool_data["input_json_parts"]:
                                    try:
                                        # PROPER: Concatenate all parts and parse
                                        complete_json = "".join(tool_data["input_json_parts"])
                                        print(f"    Complete JSON: '{complete_json}'")
                                        final_input = json.loads(complete_json)
                                    except json.JSONDecodeError as e:
                                        print(f"    JSON parse error: {e}")
                                        final_input = tool_data["initial_input"]
                                
                                tool_data["final_input"] = final_input
                                break
        
        print(f"  Strategy: PROPER ACCUMULATION")
        print(f"  Events: {event_count}, Tool events: {tool_events}, JSON events: {input_json_events}")
        
        if tool_calls:
            for tool_id, tc in tool_calls.items():
                final_input = tc.get("final_input", tc.get("initial_input", {}))
                print(f"  Final tool: {tc['name']}({json.dumps(final_input)})")
            
            # Return first tool's final input as JSON string
            first_tool = list(tool_calls.values())[0]
            final_input = first_tool.get("final_input", first_tool.get("initial_input", {}))
            return json.dumps(final_input)
        else:
            print("  No tool calls found")
            return ""
        
    except Exception as e:
        print(f"  ❌ Raw proper error: {e}")
        import traceback
        traceback.print_exc()
        return None


async def test_raw_anthropic_broken_accumulation(api_key, messages, tools):
    """Test raw Anthropic with BROKEN streaming accumulation (last part only)."""
    try:
        import anthropic
        
        client = anthropic.AsyncAnthropic(api_key=api_key)
        
        # Convert OpenAI-style tools to Anthropic format
        anthropic_tools = []
        for tool in tools:
            fn = tool["function"]
            anthropic_tools.append({
                "name": fn["name"],
                "description": fn["description"],
                "input_schema": fn["parameters"]
            })
        
        # BROKEN: Only keep last JSON part (replacement instead of accumulation)
        tool_calls = {}  # {tool_id: {name, last_json_part, initial_input}}
        event_count = 0
        tool_events = 0
        input_json_events = 0
        
        async with client.messages.stream(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2048,
            messages=messages,
            tools=anthropic_tools,
            tool_choice={"type": "auto"}
        ) as stream:
            
            async for event in stream:
                event_count += 1
                
                if hasattr(event, 'type'):
                    if event.type == 'content_block_start':
                        if hasattr(event, 'content_block') and event.content_block.type == 'tool_use':
                            tool_events += 1
                            tool_id = event.content_block.id
                            tool_name = event.content_block.name
                            initial_input = getattr(event.content_block, 'input', {})
                            
                            tool_calls[tool_id] = {
                                "name": tool_name,
                                "initial_input": initial_input,
                                "last_json_part": "",
                                "content_index": getattr(event, 'index', 0)
                            }
                    
                    elif event.type == 'content_block_delta':
                        if hasattr(event, 'delta') and hasattr(event.delta, 'type'):
                            if event.delta.type == 'input_json_delta':
                                input_json_events += 1
                                content_index = getattr(event, 'index', 0)
                                
                                # BROKEN: Replace instead of accumulate
                                for tool_id, tool_data in tool_calls.items():
                                    if tool_data.get('content_index') == content_index:
                                        tool_data['last_json_part'] = event.delta.partial_json  # BROKEN!
                                        break
                    
                    elif event.type == 'content_block_stop':
                        content_index = getattr(event, 'index', 0)
                        
                        # BROKEN: Use only last part
                        for tool_id, tool_data in tool_calls.items():
                            if tool_data.get('content_index') == content_index:
                                final_input = tool_data["initial_input"]
                                
                                if tool_data["last_json_part"]:
                                    try:
                                        # BROKEN: Parse only last part
                                        final_input = json.loads(tool_data["last_json_part"])
                                    except json.JSONDecodeError:
                                        final_input = tool_data["initial_input"]
                                
                                tool_data["final_input"] = final_input
                                break
        
        print(f"  Strategy: BROKEN (last part only)")
        print(f"  Events: {event_count}, Tool events: {tool_events}, JSON events: {input_json_events}")
        
        if tool_calls:
            for tool_id, tc in tool_calls.items():
                final_input = tc.get("final_input", tc.get("initial_input", {}))
                print(f"  Final tool: {tc['name']}({json.dumps(final_input)})")
            
            # Return first tool's final input as JSON string
            first_tool = list(tool_calls.values())[0]
            final_input = first_tool.get("final_input", first_tool.get("initial_input", {}))
            return json.dumps(final_input)
        else:
            print("  No tool calls found")
            return ""
        
    except Exception as e:
        print(f"  ❌ Raw broken error: {e}")
        return None


async def test_chuk_llm_anthropic(messages, tools):
    """Test chuk-llm Anthropic streaming."""
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from chuk_llm.llm.client import get_client
        
        client = get_client(provider="anthropic", model="claude-3-5-sonnet-20241022")
        
        # Stream with chuk-llm
        chunk_count = 0
        final_tool_calls = []
        all_chunks = []
        
        async for chunk in client.create_completion(
            messages=messages,
            tools=tools,
            stream=True
        ):
            chunk_count += 1
            all_chunks.append(chunk)
            print(f"    Chuk chunk {chunk_count}: {json.dumps(chunk, indent=2)}")
            
            if chunk.get("tool_calls"):
                # Check for duplication
                for tc in chunk["tool_calls"]:
                    # Avoid duplicates by checking if this exact tool call already exists
                    tc_signature = f"{tc['function']['name']}({tc['function']['arguments']})"
                    existing_signatures = [
                        f"{existing['function']['name']}({existing['function']['arguments']})"
                        for existing in final_tool_calls
                    ]
                    
                    if tc_signature not in existing_signatures:
                        final_tool_calls.append(tc)
        
        print(f"  Chunks: {chunk_count}")
        print(f"  Tool call chunks: {len([c for c in all_chunks if c.get('tool_calls')])}")
        print(f"  Total unique tools: {len(final_tool_calls)}")
        
        if final_tool_calls:
            for i, tc in enumerate(final_tool_calls):
                args = tc.get("function", {}).get("arguments", "")
                name = tc.get("function", {}).get("name", "")
                print(f"  Tool {i+1}: {name}({args})")
            return final_tool_calls[0].get("function", {}).get("arguments", "")
        else:
            print("  No tool calls found")
            return ""
        
    except Exception as e:
        print(f"  ❌ Chuk error: {e}")
        import traceback
        traceback.print_exc()
        return None


def analyze_anthropic_results(results):
    """Analyze the different Anthropic streaming strategies."""
    print("\n🔬 ANALYSIS:")
    
    raw_proper = results["raw_proper"]
    raw_broken = results["raw_broken"] 
    chuk_current = results["chuk_current"]
    
    try:
        # Parse JSON arguments for comparison
        proper_parsed = json.loads(raw_proper) if raw_proper else {}
        broken_parsed = json.loads(raw_broken) if raw_broken else {}
        chuk_parsed = json.loads(chuk_current) if chuk_current else {}
        
        print(f"Proper accumulation parameters: {len(proper_parsed)} keys")
        print(f"Broken accumulation parameters: {len(broken_parsed)} keys")
        print(f"Chuk-LLM parameters: {len(chuk_parsed)} keys")
        
        # Check if proper accumulation gives more complete results
        if len(proper_parsed) > len(broken_parsed):
            print("✅ PROPER ACCUMULATION GIVES MORE COMPLETE RESULTS")
            print("   Anthropic streaming requires accumulating input_json_delta events")
            
            # Check if chuk matches the proper result
            if chuk_parsed == proper_parsed:
                print("✅ CHUK-LLM USES PROPER ACCUMULATION")
                print("   Tool argument streaming works correctly")
                return True
            elif chuk_parsed == broken_parsed:
                print("❌ CHUK-LLM USES BROKEN ACCUMULATION")
                print("   Only processing last JSON delta, losing data")
                print("\n🔧 FIX NEEDED:")
                print("   File: chuk_llm/llm/providers/anthropic_client.py")
                print("   Problem: Not accumulating input_json_delta events properly")
                print("   Solution: Concatenate all partial_json parts before parsing")
                return False
            else:
                print("❓ CHUK-LLM HAS DIFFERENT BEHAVIOR")
                print(f"   Expected (proper): {proper_parsed}")
                print(f"   Got (chuk):        {chuk_parsed}")
                return False
        else:
            print("✅ BOTH ACCUMULATION STRATEGIES WORK")
            print("   May need more complex test case to see differences")
            return chuk_parsed == proper_parsed
            
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing error: {e}")
        print(f"Raw proper: '{raw_proper}'")
        print(f"Raw broken: '{raw_broken}'") 
        print(f"Chuk current: '{chuk_current}'")
        return False


async def test_duplication_specifically_anthropic():
    """Test specifically for Anthropic tool call duplication bug."""
    print("\n🔍 ANTHROPIC DUPLICATION TEST")
    print("=" * 40)
    
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return False
    
    tools = [{
        "type": "function",
        "function": {
            "name": "test_anthropic_tool",
            "description": "Test tool for Anthropic",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "Test message"},
                    "count": {"type": "integer", "description": "Test count"}
                },
                "required": ["message"]
            }
        }
    }]
    
    messages = [{"role": "user", "content": "Call test_anthropic_tool with message 'hello world' and count 42"}]
    
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from chuk_llm.llm.client import get_client
        
        client = get_client(provider="anthropic", model="claude-3-5-sonnet-20241022")
        
        all_tool_calls = []
        chunk_count = 0
        
        async for chunk in client.create_completion(
            messages=messages,
            tools=tools,
            stream=True
        ):
            chunk_count += 1
            if chunk.get("tool_calls"):
                all_tool_calls.extend(chunk["tool_calls"])
        
        print(f"Total chunks: {chunk_count}")
        print(f"Total tool calls collected: {len(all_tool_calls)}")
        
        # Check for duplication
        unique_tool_calls = []
        for tc in all_tool_calls:
            tc_signature = f"{tc['function']['name']}({tc['function']['arguments']})"
            if tc_signature not in [f"{utc['function']['name']}({utc['function']['arguments']})" for utc in unique_tool_calls]:
                unique_tool_calls.append(tc)
        
        print(f"Unique tool calls: {len(unique_tool_calls)}")
        
        if len(all_tool_calls) == len(unique_tool_calls) == 1:
            print("✅ NO DUPLICATION - Perfect!")
            return True
        elif len(unique_tool_calls) == 1 and len(all_tool_calls) > 1:
            print(f"❌ DUPLICATION DETECTED - {len(all_tool_calls)} copies of same tool call")
            return False
        else:
            print("❓ UNEXPECTED TOOL CALL PATTERN")
            for i, tc in enumerate(all_tool_calls):
                print(f"  Tool {i+1}: {tc['function']['name']}({tc['function']['arguments']})")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def main():
    """Run enhanced Anthropic streaming diagnostic."""
    print("🚀 ENHANCED ANTHROPIC STREAMING DIAGNOSTIC")
    print("Testing accumulation strategies and event handling")
    
    # Test 1: Check for duplication specifically
    duplication_ok = await test_duplication_specifically_anthropic()
    
    # Test 2: Verify streaming accumulation strategy
    accumulation_ok = await test_anthropic_streaming_strategies()
    
    print("\n" + "=" * 70)
    print("🎯 ANTHROPIC DIAGNOSTIC SUMMARY:")
    print(f"Duplication test:  {'✅ PASS' if duplication_ok else '❌ FAIL'}")
    print(f"Accumulation test: {'✅ PASS' if accumulation_ok else '❌ FAIL'}")
    
    if duplication_ok and accumulation_ok:
        print("\n✅ ANTHROPIC STREAMING WORKS PERFECTLY!")
        print("   No tool call duplication detected")
        print("   Proper input_json_delta accumulation in use")
    elif duplication_ok and not accumulation_ok:
        print("\n⚠️  ACCUMULATION ISSUE DETECTED")
        print("   No duplication, but streaming accumulation may be incomplete")
        print("   Check input_json_delta event handling")
    elif not duplication_ok and accumulation_ok:
        print("\n⚠️  DUPLICATION ISSUE DETECTED") 
        print("   Accumulation works, but tool calls are being duplicated")
        print("   Check chunk processing logic")
    else:
        print("\n❌ MULTIPLE ISSUES DETECTED")
        print("   Both duplication and accumulation problems found")


if __name__ == "__main__":
    asyncio.run(main())