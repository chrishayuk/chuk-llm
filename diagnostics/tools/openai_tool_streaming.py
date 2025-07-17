#!/usr/bin/env python3
"""
Enhanced Chuk-LLM Streaming Diagnostic

Tests both concatenation vs replacement strategies to identify the exact issue.
Compares Raw OpenAI behavior with chuk-llm implementation.
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


async def test_streaming_strategies():
    """Test different streaming accumulation strategies."""
    
    print("🔍 STREAMING STRATEGY ANALYSIS")
    print("=" * 50)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ No OPENAI_API_KEY")
        return False
    
    # Test case that should trigger multiple chunks
    tools = [{
        "type": "function",
        "function": {
            "name": "execute_sql",
            "description": "Execute a SQL query",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "SQL query to execute"},
                    "database": {"type": "string", "description": "Database name"}
                },
                "required": ["query"]
            }
        }
    }]
    
    messages = [{"role": "user", "content": "Execute SQL query 'SELECT * FROM users WHERE status = \"active\" LIMIT 10' on database 'production'"}]
    
    print("🎯 Test: Complex SQL with multiple parameters")
    print("Expected: Should generate chunks for function name and arguments")
    
    # Test raw OpenAI with different accumulation strategies
    print("\n🔥 RAW OPENAI ANALYSIS:")
    raw_concatenation = await test_raw_openai_concatenation(api_key, messages, tools)
    raw_replacement = await test_raw_openai_replacement(api_key, messages, tools)
    
    # Test chuk-llm current behavior
    print("\n🔧 CHUK-LLM CURRENT:")
    chuk_result = await test_chuk_llm(messages, tools)
    
    # Detailed comparison
    print("\n📊 DETAILED COMPARISON:")
    print(f"Raw (concatenation): '{raw_concatenation}'")
    print(f"Raw (replacement):   '{raw_replacement}'")
    print(f"Chuk-LLM current:    '{chuk_result}'")
    
    # Analyze results
    results = {
        "raw_concat": raw_concatenation,
        "raw_replace": raw_replacement, 
        "chuk_current": chuk_result
    }
    
    return analyze_results(results)


async def test_raw_openai_concatenation(api_key, messages, tools):
    """Test raw OpenAI with concatenation strategy (potentially buggy)."""
    try:
        import openai
        client = openai.AsyncOpenAI(api_key=api_key)
        
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools,
            stream=True
        )
        
        # Concatenation strategy (like original chuk-llm bug)
        tool_calls = {}
        chunk_count = 0
        tool_call_chunks = 0
        raw_chunks = []
        
        async for chunk in response:
            chunk_count += 1
            raw_chunks.append(str(chunk))
            
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.tool_calls:
                tool_call_chunks += 1
                print(f"    Chunk {tool_call_chunks}: {chunk.choices[0].delta.tool_calls}")
                
                for tc in chunk.choices[0].delta.tool_calls:
                    idx = tc.index or 0
                    
                    if idx not in tool_calls:
                        tool_calls[idx] = {"name": "", "arguments": "", "id": None}
                    
                    if tc.id:
                        tool_calls[idx]["id"] = tc.id
                    
                    if tc.function:
                        # CONCATENATION STRATEGY (potentially wrong)
                        if tc.function.name:
                            tool_calls[idx]["name"] += tc.function.name
                            print(f"      Name after concat: '{tool_calls[idx]['name']}'")
                        if tc.function.arguments:
                            tool_calls[idx]["arguments"] += tc.function.arguments
                            print(f"      Args after concat: '{tool_calls[idx]['arguments']}'")
        
        print(f"  Strategy: CONCATENATION")
        print(f"  Chunks: {chunk_count}, Tool chunks: {tool_call_chunks}")
        
        if tool_calls:
            for idx, tc in tool_calls.items():
                print(f"  Final tool: {tc['name']}({tc['arguments']})")
            return list(tool_calls.values())[0]['arguments']
        return ""
        
    except Exception as e:
        print(f"  ❌ Raw concatenation error: {e}")
        return None


async def test_raw_openai_replacement(api_key, messages, tools):
    """Test raw OpenAI with replacement strategy (correct)."""
    try:
        import openai
        client = openai.AsyncOpenAI(api_key=api_key)
        
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools,
            stream=True
        )
        
        # Replacement strategy (correct approach)
        tool_calls = {}
        chunk_count = 0
        tool_call_chunks = 0
        
        async for chunk in response:
            chunk_count += 1
            
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.tool_calls:
                tool_call_chunks += 1
                print(f"    Chunk {tool_call_chunks}: {chunk.choices[0].delta.tool_calls}")
                
                for tc in chunk.choices[0].delta.tool_calls:
                    idx = tc.index or 0
                    
                    if idx not in tool_calls:
                        tool_calls[idx] = {"name": "", "arguments": "", "id": None}
                    
                    if tc.id:
                        tool_calls[idx]["id"] = tc.id
                    
                    if tc.function:
                        # REPLACEMENT STRATEGY (correct)
                        if tc.function.name is not None:
                            tool_calls[idx]["name"] = tc.function.name
                            print(f"      Name after replace: '{tool_calls[idx]['name']}'")
                        if tc.function.arguments is not None:
                            tool_calls[idx]["arguments"] = tc.function.arguments
                            print(f"      Args after replace: '{tool_calls[idx]['arguments']}'")
        
        print(f"  Strategy: REPLACEMENT")
        print(f"  Chunks: {chunk_count}, Tool chunks: {tool_call_chunks}")
        
        if tool_calls:
            for idx, tc in tool_calls.items():
                print(f"  Final tool: {tc['name']}({tc['arguments']})")
            return list(tool_calls.values())[0]['arguments']
        return ""
        
    except Exception as e:
        print(f"  ❌ Raw replacement error: {e}")
        return None


async def test_chuk_llm(messages, tools):
    """Test chuk-llm current implementation."""
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from chuk_llm.llm.client import get_client
        
        client = get_client(provider="openai", model="gpt-4o-mini")
        
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
                # Note: chuk-llm might be extending rather than replacing
                final_tool_calls.extend(chunk["tool_calls"])
        
        print(f"  Chunks: {chunk_count}")
        print(f"  Tool call chunks: {len([c for c in all_chunks if c.get('tool_calls')])}")
        print(f"  Total accumulated tools: {len(final_tool_calls)}")
        
        if final_tool_calls:
            for i, tc in enumerate(final_tool_calls):
                args = tc.get("function", {}).get("arguments", "")
                name = tc.get("function", {}).get("name", "")
                print(f"  Tool {i+1}: {name}({args})")
            return final_tool_calls[0].get("function", {}).get("arguments", "")
        return ""
        
    except Exception as e:
        print(f"  ❌ Chuk error: {e}")
        import traceback
        traceback.print_exc()
        return None


def analyze_results(results):
    """CORRECTED: Analyze the different streaming strategies."""
    print("\n🔬 ANALYSIS:")
    
    raw_concat = results["raw_concat"]
    raw_replace = results["raw_replace"] 
    chuk_current = results["chuk_current"]
    
    # CORRECTED LOGIC: For OpenAI streaming, concatenation is the CORRECT approach
    # because OpenAI sends incremental deltas that must be accumulated
    
    if raw_concat and raw_replace:
        print(f"Raw concatenation result: {len(raw_concat)} chars")
        print(f"Raw replacement result: {len(raw_replace)} chars")
        
        # The concatenation should give the complete, valid JSON
        # The replacement should give only the last fragment
        if len(raw_concat) > len(raw_replace) and raw_concat.startswith('{"'):
            print("✅ CONCATENATION IS CORRECT for OpenAI streaming")
            print("   OpenAI sends incremental deltas that need accumulation")
            
            # Check if chuk matches the correct (concatenation) result
            if chuk_current == raw_concat:
                print("✅ CHUK-LLM WORKS CORRECTLY")
                print("   Tool calls are properly accumulated without duplication")
                return True
            else:
                print("❌ CHUK-LLM DOESN'T MATCH CORRECT RESULT")
                print(f"   Expected: '{raw_concat}'")
                print(f"   Got:      '{chuk_current}'")
                return False
        else:
            print("❓ UNEXPECTED: Replacement gave better result than concatenation")
            return False
    else:
        print("❓ UNCLEAR RESULTS - missing data")
        return False


async def test_duplication_specifically():
    """Test specifically for tool call duplication bug."""
    print("\n🔍 DUPLICATION-SPECIFIC TEST")
    print("=" * 35)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return False
    
    tools = [{
        "type": "function",
        "function": {
            "name": "test_tool",
            "description": "Test tool",
            "parameters": {
                "type": "object",
                "properties": {
                    "param": {"type": "string", "description": "Test parameter"}
                },
                "required": ["param"]
            }
        }
    }]
    
    messages = [{"role": "user", "content": "Call test_tool with param 'hello'"}]
    
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from chuk_llm.llm.client import get_client
        
        client = get_client(provider="openai", model="gpt-4o-mini")
        
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
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def main():
    """CORRECTED: Run enhanced streaming diagnostic."""
    print("🚀 ENHANCED STREAMING DIAGNOSTIC")
    print("Testing for tool call duplication (the real bug)")
    
    # Test 1: Check for duplication specifically
    duplication_ok = await test_duplication_specifically()
    
    # Test 2: Verify streaming strategy works
    strategy_ok = await test_streaming_strategies()
    
    print("\n" + "=" * 60)
    print("🎯 CORRECTED DIAGNOSTIC SUMMARY:")
    print(f"Duplication test: {'✅ PASS' if duplication_ok else '❌ FAIL'}")
    print(f"Strategy test:    {'✅ PASS' if strategy_ok else '❌ FAIL'}")
    
    if duplication_ok and strategy_ok:
        print("\n✅ STREAMING WORKS PERFECTLY!")
        print("   No tool call duplication detected")
        print("   Proper accumulation strategy in use")
    elif duplication_ok and not strategy_ok:
        print("\n✅ DUPLICATION FIXED!")
        print("   The main bug (duplication) is resolved")
        print("   Strategy comparison may have false negatives")
    else:
        print("\n❌ ISSUES REMAIN")
        print("   Tool call duplication still occurring")

if __name__ == "__main__":
    asyncio.run(main())