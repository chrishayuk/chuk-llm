#!/usr/bin/env python3
"""
Granite 3.3 Tool Streaming Test
================================

Clean test script for validating tool calling with granite3.3:latest model.
Compares raw Ollama API with ChukLLM implementation.
"""

import asyncio
import json
import time
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


async def test_granite_tool_streaming():
    """Main test for Granite 3.3 tool capabilities"""
    
    print("🪨 GRANITE 3.3 TOOL STREAMING TEST")
    print("=" * 60)
    
    model_name = "granite3.3:latest"
    
    # Check if model is available
    if not await check_model_available(model_name):
        print(f"❌ {model_name} not found. Please install with:")
        print(f"   ollama pull {model_name}")
        return
    
    print(f"📦 Testing model: {model_name}\n")
    
    # Define test cases
    test_cases = [
        {
            "name": "Basic SQL Tool",
            "prompt": "Execute this SQL query: SELECT name, email FROM users WHERE active = true LIMIT 10",
            "tools": [{
                "type": "function",
                "function": {
                    "name": "execute_sql",
                    "description": "Execute a SQL query against the database",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "SQL query to execute"},
                            "database": {"type": "string", "description": "Database name", "default": "main"}
                        },
                        "required": ["query"]
                    }
                }
            }],
            "expected_function": "execute_sql"
        },
        {
            "name": "Weather Query",
            "prompt": "What's the weather like in San Francisco today?",
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "City name"},
                            "units": {"type": "string", "enum": ["celsius", "fahrenheit"], "default": "fahrenheit"}
                        },
                        "required": ["location"]
                    }
                }
            }],
            "expected_function": "get_weather"
        },
        {
            "name": "Multiple Tools",
            "prompt": "Get the user with ID 12345 and then check their account balance",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_user",
                        "description": "Retrieve user information by ID",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "user_id": {"type": "string", "description": "User ID to look up"}
                            },
                            "required": ["user_id"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "get_balance",
                        "description": "Get account balance for a user",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "user_id": {"type": "string", "description": "User ID"},
                                "account_type": {"type": "string", "default": "checking"}
                            },
                            "required": ["user_id"]
                        }
                    }
                }
            ],
            "expected_function": "get_user"
        }
    ]
    
    # Run tests
    overall_results = {}
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🧪 Test {i}/{len(test_cases)}: {test_case['name']}")
        print("-" * 40)
        
        # Test with ChukLLM
        print("  📊 Testing with ChukLLM...")
        chuk_result = await test_chukllm(model_name, test_case["prompt"], test_case["tools"])
        
        # Test with raw Ollama API
        print("  📊 Testing with Raw Ollama API...")
        raw_result = await test_raw_ollama(model_name, test_case["prompt"], test_case["tools"])
        
        # Analyze results
        analyze_results(chuk_result, raw_result, test_case["name"], test_case["expected_function"])
        
        overall_results[test_case["name"]] = {
            "chuk": chuk_result,
            "raw": raw_result,
            "expected_function": test_case["expected_function"]
        }
    
    # Final summary
    print_summary(overall_results, model_name)


async def check_model_available(model_name: str) -> bool:
    """Check if a model is available locally"""
    try:
        import httpx
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                data = response.json()
                models = [m.get("name", "") for m in data.get("models", [])]
                return model_name in models
    except Exception as e:
        print(f"⚠️  Could not check models: {e}")
    return False


async def test_chukllm(model_name: str, prompt: str, tools: List[Dict]) -> Dict[str, Any]:
    """Test using ChukLLM's stream function"""
    try:
        from chuk_llm import stream
        
        result = {
            "success": False,
            "chunks": 0,
            "tool_calls": [],
            "response_text": "",
            "error": None,
            "time": 0
        }
        
        start_time = time.time()
        chunk_count = 0
        response_parts = []
        all_tool_calls = []
        
        # Stream the response
        async for chunk in stream(
            prompt,
            provider="ollama",
            model=model_name,
            tools=tools,
            max_tokens=400,
            temperature=0.1,
            return_tool_calls=True  # Request dict format with tool calls
        ):
            chunk_count += 1
            
            # Handle dict chunks (with the fixed stream function)
            if isinstance(chunk, dict):
                # Collect text
                if chunk.get("response"):
                    response_parts.append(chunk["response"])
                
                # Collect tool calls
                if chunk.get("tool_calls"):
                    all_tool_calls.extend(chunk["tool_calls"])
            else:
                # Handle string chunks (old behavior)
                response_parts.append(str(chunk))
        
        result["chunks"] = chunk_count
        result["response_text"] = "".join(response_parts)
        result["tool_calls"] = all_tool_calls
        result["success"] = True
        result["time"] = time.time() - start_time
        
    except Exception as e:
        result = {
            "success": False,
            "error": str(e),
            "chunks": 0,
            "tool_calls": [],
            "response_text": "",
            "time": 0
        }
    
    return result


async def test_raw_ollama(model_name: str, prompt: str, tools: List[Dict]) -> Dict[str, Any]:
    """Test using raw Ollama API"""
    try:
        import httpx
        
        result = {
            "success": False,
            "chunks": 0,
            "tool_calls": [],
            "response_text": "",
            "error": None,
            "time": 0
        }
        
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "tools": tools,
            "stream": True,
            "options": {
                "num_predict": 400,
                "temperature": 0.1
            }
        }
        
        start_time = time.time()
        chunk_count = 0
        response_parts = []
        all_tool_calls = []
        
        async with httpx.AsyncClient(timeout=90.0) as client:
            async with client.stream("POST", "http://localhost:11434/api/chat", json=payload) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if line.strip():
                        try:
                            data = json.loads(line)
                            
                            if "message" in data:
                                msg = data["message"]
                                
                                # Collect text
                                if msg.get("content"):
                                    chunk_count += 1
                                    response_parts.append(msg["content"])
                                
                                # Collect tool calls
                                if msg.get("tool_calls"):
                                    for tc in msg["tool_calls"]:
                                        if "function" in tc:
                                            all_tool_calls.append({
                                                "function": {
                                                    "name": tc["function"].get("name", ""),
                                                    "arguments": tc["function"].get("arguments", "")
                                                }
                                            })
                        except json.JSONDecodeError:
                            continue
        
        result["chunks"] = chunk_count
        result["response_text"] = "".join(response_parts)
        result["tool_calls"] = all_tool_calls
        result["success"] = True
        result["time"] = time.time() - start_time
        
    except Exception as e:
        result = {
            "success": False,
            "error": str(e),
            "chunks": 0,
            "tool_calls": [],
            "response_text": "",
            "time": 0
        }
    
    return result


def analyze_results(chuk_result: Dict, raw_result: Dict, test_name: str, expected_function: str):
    """Analyze and compare test results"""
    print(f"\n    📋 {test_name} Results:")
    
    # ChukLLM results
    if chuk_result["success"]:
        chuk_tools = len(chuk_result["tool_calls"])
        chuk_time = chuk_result["time"]
        print(f"      ChukLLM: {chuk_result['chunks']} chunks, {chuk_tools} tools, {chuk_time:.2f}s")
        
        if chuk_tools > 0:
            functions = [tc["function"]["name"] for tc in chuk_result["tool_calls"]]
            print(f"        Functions: {', '.join(functions)}")
            if expected_function in functions:
                print(f"        ✅ Found expected: {expected_function}")
    else:
        print(f"      ChukLLM: ❌ Failed - {chuk_result.get('error', 'Unknown error')}")
    
    # Raw Ollama results
    if raw_result["success"]:
        raw_tools = len(raw_result["tool_calls"])
        raw_time = raw_result["time"]
        print(f"      Raw API: {raw_result['chunks']} chunks, {raw_tools} tools, {raw_time:.2f}s")
        
        if raw_tools > 0:
            functions = [tc["function"]["name"] for tc in raw_result["tool_calls"]]
            print(f"        Functions: {', '.join(functions)}")
            if expected_function in functions:
                print(f"        ✅ Found expected: {expected_function}")
    else:
        print(f"      Raw API: ❌ Failed - {raw_result.get('error', 'Unknown error')}")


def print_summary(results: Dict, model_name: str):
    """Print overall summary"""
    print(f"\n{'=' * 60}")
    print(f"📊 SUMMARY for {model_name}")
    print(f"{'=' * 60}")
    
    # Count successes
    chuk_tool_detection = sum(
        1 for r in results.values() 
        if r["chuk"]["success"] and len(r["chuk"]["tool_calls"]) > 0
    )
    raw_tool_detection = sum(
        1 for r in results.values() 
        if r["raw"]["success"] and len(r["raw"]["tool_calls"]) > 0
    )
    
    total_tests = len(results)
    
    print(f"\n🔍 Tool Detection:")
    print(f"  ChukLLM:  {chuk_tool_detection}/{total_tests} tests with tool calls")
    print(f"  Raw API:  {raw_tool_detection}/{total_tests} tests with tool calls")
    
    # Performance comparison
    if chuk_tool_detection == raw_tool_detection:
        print(f"\n✅ PERFECT MATCH: ChukLLM matches Raw API performance!")
    elif chuk_tool_detection > 0:
        print(f"\n⚠️  PARTIAL SUCCESS: ChukLLM detected some but not all tool calls")
    else:
        print(f"\n❌ ISSUE: ChukLLM not detecting tool calls that Raw API finds")
    
    # Detailed breakdown
    print(f"\n📝 Test Details:")
    for test_name, data in results.items():
        chuk_tools = len(data["chuk"]["tool_calls"]) if data["chuk"]["success"] else 0
        raw_tools = len(data["raw"]["tool_calls"]) if data["raw"]["success"] else 0
        
        if chuk_tools == raw_tools and chuk_tools > 0:
            status = "✅"
        elif chuk_tools > 0:
            status = "⚠️"
        else:
            status = "❌"
        
        print(f"  {status} {test_name}: ChukLLM={chuk_tools} tools, Raw={raw_tools} tools")
    
    print(f"\n{'=' * 60}")
    print("🔧 Test complete!")


async def main():
    """Run the test"""
    print("🚀 GRANITE 3.3 TOOL STREAMING TEST")
    print("Testing tool calling capabilities\n")
    
    await test_granite_tool_streaming()


if __name__ == "__main__":
    asyncio.run(main())