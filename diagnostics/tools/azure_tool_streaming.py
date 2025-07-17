#!/usr/bin/env python3
"""
Enhanced Azure OpenAI Streaming Diagnostic

Comprehensive testing of Azure OpenAI streaming tool call behavior.
Tests both accumulation strategies and duplication detection.
Compares Raw Azure OpenAI behavior with chuk-llm implementation.
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


async def test_azure_streaming_strategies():
    """Test different Azure OpenAI streaming accumulation strategies."""
    
    print("🔍 AZURE OPENAI STREAMING STRATEGY ANALYSIS")
    print("=" * 60)
    
    # Check for required Azure OpenAI environment variables
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT") or "gpt-4o-mini"
    api_version = os.getenv("AZURE_OPENAI_API_VERSION") or "2024-02-01"
    
    if not api_key:
        print("❌ No AZURE_OPENAI_API_KEY")
        return False
    
    if not endpoint:
        print("❌ No AZURE_OPENAI_ENDPOINT")
        return False
    
    print(f"🔧 Azure endpoint: {endpoint}")
    print(f"🔧 Deployment: {deployment}")
    print(f"🔧 API version: {api_version}")
    
    # Test case that should trigger multiple streaming chunks
    tools = [{
        "type": "function",
        "function": {
            "name": "execute_database_operation",
            "description": "Execute a complex database operation with multiple parameters",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {"type": "string", "description": "Type of database operation"},
                    "query": {"type": "string", "description": "SQL query to execute"},
                    "database": {"type": "string", "description": "Target database name"},
                    "connection_params": {
                        "type": "object",
                        "properties": {
                            "host": {"type": "string", "description": "Database host"},
                            "port": {"type": "integer", "description": "Database port"},
                            "timeout": {"type": "integer", "description": "Connection timeout"},
                            "ssl": {"type": "boolean", "description": "Use SSL connection"}
                        }
                    },
                    "options": {
                        "type": "object",
                        "properties": {
                            "explain_plan": {"type": "boolean", "description": "Include execution plan"},
                            "format": {"type": "string", "enum": ["json", "csv", "table"]},
                            "limit": {"type": "integer", "description": "Result limit"}
                        }
                    }
                },
                "required": ["operation", "query", "database"]
            }
        }
    }]
    
    messages = [{
        "role": "user", 
        "content": "Execute a SELECT operation with query 'SELECT u.user_id, u.username, u.email, p.profile_data, s.session_count FROM users u LEFT JOIN profiles p ON u.user_id = p.user_id LEFT JOIN session_stats s ON u.user_id = s.user_id WHERE u.active = 1 AND u.created_date >= \"2024-01-01\" ORDER BY s.session_count DESC LIMIT 50' on database 'production_analytics' with connection to host 'db.company.com' port 5432 timeout 30 seconds using SSL, include execution plan and format as json with limit 50"
    }]
    
    print("🎯 Test: Complex database operation with nested parameters")
    print("Expected: Should generate multiple chunks for function arguments")
    
    # Test raw Azure OpenAI with different accumulation strategies
    print("\n🔥 RAW AZURE OPENAI ANALYSIS:")
    raw_concatenation = await test_raw_azure_concatenation(api_key, endpoint, deployment, api_version, messages, tools)
    raw_replacement = await test_raw_azure_replacement(api_key, endpoint, deployment, api_version, messages, tools)
    
    # Test chuk-llm current behavior
    print("\n🔧 CHUK-LLM CURRENT:")
    chuk_result = await test_chuk_llm_azure(messages, tools, deployment)
    
    # Detailed comparison
    print("\n📊 COMPARISON:")
    print(f"Raw (concatenation): {len(raw_concatenation)} chars")
    print(f"Raw (replacement):   {len(raw_replacement)} chars") 
    print(f"Chuk-LLM current:    {len(chuk_result)} chars")
    
    # Analyze results
    return analyze_azure_results({
        "raw_concat": raw_concatenation,
        "raw_replace": raw_replacement,
        "chuk_current": chuk_result
    })


async def test_raw_azure_concatenation(api_key, endpoint, deployment, api_version, messages, tools):
    """Test raw Azure OpenAI with concatenation strategy."""
    try:
        import openai
        
        client = openai.AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version
        )
        
        # Concatenation strategy
        tool_calls = {}
        chunk_count = 0
        tool_call_chunks = 0
        
        response = await client.chat.completions.create(
            model=deployment,  # Azure uses deployment name
            messages=messages,
            tools=tools,
            stream=True
        )
        
        async for chunk in response:
            chunk_count += 1
            
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.tool_calls:
                tool_call_chunks += 1
                
                for tc in chunk.choices[0].delta.tool_calls:
                    idx = tc.index or 0
                    
                    if idx not in tool_calls:
                        tool_calls[idx] = {"name": "", "arguments": "", "id": None}
                    
                    if tc.id:
                        tool_calls[idx]["id"] = tc.id
                    
                    if tc.function:
                        # CONCATENATION STRATEGY
                        if tc.function.name:
                            tool_calls[idx]["name"] += tc.function.name
                        if tc.function.arguments:
                            tool_calls[idx]["arguments"] += tc.function.arguments
        
        print(f"  Strategy: CONCATENATION")
        print(f"  Chunks: {chunk_count}, Tool chunks: {tool_call_chunks}")
        
        if tool_calls:
            for idx, tc in tool_calls.items():
                print(f"  Tool: {tc['name']}({len(tc['arguments'])} chars)")
            return list(tool_calls.values())[0]['arguments']
        return ""
        
    except Exception as e:
        print(f"  ❌ Raw Azure concatenation error: {e}")
        import traceback
        traceback.print_exc()
        return None


async def test_raw_azure_replacement(api_key, endpoint, deployment, api_version, messages, tools):
    """Test raw Azure OpenAI with replacement strategy."""
    try:
        import openai
        
        client = openai.AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version
        )
        
        # Replacement strategy
        tool_calls = {}
        chunk_count = 0
        tool_call_chunks = 0
        
        response = await client.chat.completions.create(
            model=deployment,
            messages=messages,
            tools=tools,
            stream=True
        )
        
        async for chunk in response:
            chunk_count += 1
            
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.tool_calls:
                tool_call_chunks += 1
                
                for tc in chunk.choices[0].delta.tool_calls:
                    idx = tc.index or 0
                    
                    if idx not in tool_calls:
                        tool_calls[idx] = {"name": "", "arguments": "", "id": None}
                    
                    if tc.id:
                        tool_calls[idx]["id"] = tc.id
                    
                    if tc.function:
                        # REPLACEMENT STRATEGY
                        if tc.function.name is not None:
                            tool_calls[idx]["name"] = tc.function.name
                        if tc.function.arguments is not None:
                            tool_calls[idx]["arguments"] = tc.function.arguments
        
        print(f"  Strategy: REPLACEMENT")
        print(f"  Chunks: {chunk_count}, Tool chunks: {tool_call_chunks}")
        
        if tool_calls:
            for idx, tc in tool_calls.items():
                print(f"  Tool: {tc['name']}({len(tc['arguments'])} chars)")
            return list(tool_calls.values())[0]['arguments']
        return ""
        
    except Exception as e:
        print(f"  ❌ Raw Azure replacement error: {e}")
        return None


async def test_chuk_llm_azure(messages, tools, deployment):
    """Test chuk-llm Azure OpenAI streaming."""
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from chuk_llm.llm.client import get_client
        
        # Get Azure OpenAI client
        client = get_client(
            provider="azure_openai", 
            model=deployment
        )
        
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
            
            if chunk.get("tool_calls"):
                # Check for duplication
                for tc in chunk["tool_calls"]:
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
                print(f"  Tool {i+1}: {name}({len(args)} chars)")
            return final_tool_calls[0].get("function", {}).get("arguments", "")
        else:
            print("  No tool calls found")
            return ""
        
    except Exception as e:
        print(f"  ❌ Chuk Azure error: {e}")
        import traceback
        traceback.print_exc()
        return None


def analyze_azure_results(results):
    """Analyze the different Azure OpenAI streaming strategies."""
    print("\n🔬 ANALYSIS:")
    
    raw_concat = results["raw_concat"]
    raw_replace = results["raw_replace"] 
    chuk_current = results["chuk_current"]
    
    # FIXED: Handle empty strings and parsing errors gracefully
    def safe_parse_json(json_str):
        if not json_str or json_str.strip() == "":
            return {}
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return {}
    
    try:
        # Parse JSON arguments for comparison
        concat_parsed = safe_parse_json(raw_concat)
        replace_parsed = safe_parse_json(raw_replace)
        chuk_parsed = safe_parse_json(chuk_current)
        
        print(f"Concatenation result: {len(concat_parsed)} parameters")
        print(f"Replacement result: {len(replace_parsed)} parameters")
        print(f"Chuk-LLM result: {len(chuk_parsed)} parameters")
        
        # Azure OpenAI should behave like regular OpenAI (concatenation correct)
        if len(concat_parsed) > len(replace_parsed) and concat_parsed:
            print("✅ CONCATENATION IS CORRECT for Azure OpenAI")
            print("   Azure OpenAI sends incremental deltas like regular OpenAI")
            
            # Check if chuk matches the correct (concatenation) result
            if chuk_parsed == concat_parsed:
                print("✅ CHUK-LLM HANDLES AZURE CORRECTLY")
                print("   Tool call streaming works properly")
                return True
            elif chuk_parsed == replace_parsed:
                print("❌ CHUK-LLM USES BROKEN LOGIC FOR AZURE")
                print("   Not accumulating deltas properly")
                print("\n🔧 FIX NEEDED:")
                print("   File: chuk_llm/llm/providers/azure_openai_client.py")
                print("   Problem: Not using concatenation for Azure deltas")
                print("   Solution: Ensure Azure follows same logic as regular OpenAI")
                return False
            else:
                print("❓ CHUK-LLM HAS DIFFERENT AZURE BEHAVIOR")
                print(f"   Expected (concat): {concat_parsed}")
                print(f"   Got (chuk):        {chuk_parsed}")
                return False
        else:
            print("✅ BOTH STRATEGIES WORK OR SIMPLE CASE")
            print("   May need more complex test to differentiate")
            return chuk_parsed == concat_parsed
            
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing error: {e}")
        print(f"Concat: '{raw_concat}'")
        print(f"Replace: '{raw_replace}'")
        print(f"Chuk: '{chuk_current}'")
        return False


async def test_duplication_specifically_azure():
    """Test specifically for Azure OpenAI tool call duplication bug."""
    print("\n🔍 AZURE OPENAI DUPLICATION TEST")
    print("=" * 45)
    
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT") or "gpt-4o-mini"
    
    if not api_key or not endpoint:
        print("❌ Missing Azure OpenAI credentials")
        return False
    
    tools = [{
        "type": "function",
        "function": {
            "name": "test_azure_tool",
            "description": "Test tool for Azure OpenAI",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "Test message"},
                    "count": {"type": "integer", "description": "Test count"},
                    "options": {
                        "type": "object",
                        "properties": {
                            "verbose": {"type": "boolean", "description": "Verbose output"}
                        }
                    }
                },
                "required": ["message"]
            }
        }
    }]
    
    messages = [{"role": "user", "content": "Call test_azure_tool with message 'hello azure' count 123 and verbose true"}]
    
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from chuk_llm.llm.client import get_client
        
        client = get_client(provider="azure_openai", model=deployment)
        
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
    """Run enhanced Azure OpenAI streaming diagnostic."""
    print("🚀 ENHANCED AZURE OPENAI STREAMING DIAGNOSTIC")
    print("Testing accumulation strategies and duplication detection")
    
    # Test 1: Check for duplication specifically
    duplication_ok = await test_duplication_specifically_azure()
    
    # Test 2: Verify streaming accumulation strategy
    accumulation_ok = await test_azure_streaming_strategies()
    
    print("\n" + "=" * 75)
    print("🎯 AZURE OPENAI DIAGNOSTIC SUMMARY:")
    print(f"Duplication test:  {'✅ PASS' if duplication_ok else '❌ FAIL'}")
    print(f"Accumulation test: {'✅ PASS' if accumulation_ok else '❌ FAIL'}")
    
    if duplication_ok and accumulation_ok:
        print("\n✅ AZURE OPENAI STREAMING WORKS PERFECTLY!")
        print("   No tool call duplication detected")
        print("   Proper delta accumulation in use")
        print("   Azure-specific handling working correctly")
    elif duplication_ok and not accumulation_ok:
        print("\n⚠️  ACCUMULATION ISSUE DETECTED")
        print("   No duplication, but streaming accumulation may be incomplete")
        print("   Check Azure-specific delta handling")
    elif not duplication_ok and accumulation_ok:
        print("\n⚠️  DUPLICATION ISSUE DETECTED") 
        print("   Accumulation works, but tool calls are being duplicated")
        print("   Check Azure chunk processing logic")
    else:
        print("\n❌ MULTIPLE ISSUES DETECTED")
        print("   Both duplication and accumulation problems found")
        print("   Azure OpenAI streaming needs attention")
    
    print("\n💡 AZURE SETUP REMINDER:")
    print("Required environment variables:")
    print("- AZURE_OPENAI_API_KEY")
    print("- AZURE_OPENAI_ENDPOINT") 
    print("- AZURE_OPENAI_DEPLOYMENT (optional, defaults to gpt-4o-mini)")
    print("- AZURE_OPENAI_API_VERSION (optional, defaults to 2024-02-01)")


if __name__ == "__main__":
    asyncio.run(main())