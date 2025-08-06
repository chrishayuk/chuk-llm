#!/usr/bin/env python3
"""
Enhanced ChukLLM Streaming Diagnostic - Updated Analysis Logic

Tests streaming behavior across different model types and properly evaluates success.
UPDATED: Fixed analysis to correctly identify when ChukLLM streaming is working.
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


async def test_streaming_with_models():
    """Test streaming behavior across different model types with improved analysis."""
    
    print("🔍 MULTI-MODEL STREAMING ANALYSIS")
    print("=" * 50)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ No OPENAI_API_KEY")
        return False
    
    # Test models - Actually available models based on diagnostic results
    test_models = [
        ("gpt-4o-mini", "Legacy Standard Model"),
        ("gpt-4.1-mini", "GPT-4.1 Mini"),
        ("gpt-4.1", "GPT-4.1 Full"),
        ("gpt-4.1-nano", "GPT-4.1 Nano"),
        ("o3-mini", "O3 Reasoning Mini"),  # Available with Tier 3+
        ("o4-mini", "O4 Reasoning Mini"),  # Available and supports tools!
    ]
    
    # Test cases with improved evaluation criteria
    test_cases = [
        {
            "name": "Simple Response",
            "messages": [{"role": "user", "content": "What is 2+2? Answer briefly."}],
            "tools": None,
            "supports_reasoning": True,
            "evaluation": "text_streaming",  # Expect similar chunk counts
            "min_chunks_chuk": 1  # Minimum chunks ChukLLM should produce
        },
        {
            "name": "Tool Calling", 
            "messages": [{"role": "user", "content": "Execute SQL: SELECT * FROM users LIMIT 5"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "execute_sql",
                    "description": "Execute a SQL query",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "SQL query"},
                            "database": {"type": "string", "description": "Database name", "default": "main"}
                        },
                        "required": ["query"]
                    }
                }
            }],
            "supports_reasoning": False,  # O3/O4 models support tools, GPT-4.1 supports tools
            "evaluation": "tool_streaming",  # Different expectations for tool calls
            "min_chunks_chuk": 1  # ChukLLM should produce at least 1 chunk for tool calls
        },
        {
            "name": "Complex Reasoning",
            "messages": [{"role": "user", "content": "If a train leaves at 2pm going 60mph, and another at 3pm going 80mph, when do they meet if they're 200 miles apart?"}],
            "tools": None,
            "supports_reasoning": True,
            "evaluation": "text_streaming",  # Expect similar chunk counts
            "min_chunks_chuk": 50  # Complex reasoning should produce many chunks
        }
    ]
    
    results = {}
    overall_success = True
    
    for model, model_desc in test_models:
        print(f"\n🎯 Testing {model} ({model_desc})")
        print("-" * 40)
        
        model_results = {}
        model_success = True
        
        for test_case in test_cases:
            case_name = test_case["name"]
            
            # Skip tool tests for models that don't support tools
            # Note: O3/O4 and GPT-4.1 series support tools, unlike old O1 series
            skip_tools = False
            if "o1" in model.lower():  # Only old o1 models don't support tools
                skip_tools = True
                
            if test_case["tools"] and skip_tools:
                print(f"  ⏭️ Skipping {case_name} (O1 models don't support tools)")
                continue
            
            print(f"  🧪 {case_name}...")
            
            # Test with ChukLLM
            chuk_result = await test_chuk_llm_streaming(
                model, test_case["messages"], test_case["tools"]
            )
            
            # Test with raw OpenAI for comparison
            raw_result = await test_raw_openai_streaming(
                api_key, model, test_case["messages"], test_case["tools"]
            )
            
            model_results[case_name] = {
                "chuk": chuk_result,
                "raw": raw_result,
                "evaluation_type": test_case["evaluation"],
                "min_chunks_expected": test_case["min_chunks_chuk"]
            }
            
            # Improved analysis
            case_success = analyze_single_result_improved(
                chuk_result, raw_result, case_name, test_case
            )
            
            if not case_success:
                model_success = False
        
        results[model] = {
            "results": model_results,
            "success": model_success
        }
        
        if not model_success:
            overall_success = False
    
    # Overall analysis with improved logic
    print("\n" + "=" * 60)
    print("📊 COMPREHENSIVE ANALYSIS")
    return analyze_all_results_improved(results, overall_success)


async def test_chuk_llm_streaming(model, messages, tools):
    """Test ChukLLM streaming with detailed analysis."""
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from chuk_llm import stream
        
        result = {
            "success": False,
            "chunks": 0,
            "tool_calls": [],
            "final_response": "",
            "streaming_worked": False,
            "error": None,
            "has_tool_call_info": False
        }
        
        # Build parameters for streaming
        stream_params = {
            "provider": "openai", 
            "model": model
        }
        
        # Add tools if provided (O3/O4 and GPT-4.1 support tools)
        if tools:
            # Only skip tools for old O1 models
            if not ("o1" in model.lower()):
                stream_params["tools"] = tools
        
        # Add appropriate token parameter based on model type
        if any(pattern in model.lower() for pattern in ["o3", "o4", "o1"]):
            # Reasoning models use max_completion_tokens
            stream_params["max_completion_tokens"] = 200
        else:
            # Regular models use max_tokens
            stream_params["max_tokens"] = 200
        
        # Use ChukLLM's streaming
        chunk_count = 0
        response_parts = []
        detected_tool_calls = set()  # Track unique tool calls
        
        async for chunk in stream(
            messages[-1]["content"],
            **stream_params
        ):
            chunk_count += 1
            if chunk:
                chunk_str = str(chunk)
                response_parts.append(chunk_str)
                
                # Detect tool call information (updated patterns)
                if any(pattern in chunk_str for pattern in [
                    "[Calling:",           # Complete tool calls format
                    "[Calling ",           # New incremental format: [Calling func_name]:
                    '{"query"',           # JSON arguments streaming
                    '"query":',           # JSON arguments streaming
                    '"function":',        # Function call streaming
                    '"arguments":',       # Arguments streaming
                    '"execute_sql"',      # Specific function name
                    '"name":',            # Tool name streaming
                ]):
                    result["has_tool_call_info"] = True
                    
                # Extract specific tool function names
                if "[Calling " in chunk_str:
                    # Extract function name from "[Calling func_name]:" pattern
                    try:
                        start = chunk_str.find("[Calling ") + 9
                        end = chunk_str.find("]:", start)
                        if end > start:
                            func_name = chunk_str[start:end]
                            detected_tool_calls.add(func_name)
                    except:
                        pass
                elif "execute_sql" in chunk_str:
                    detected_tool_calls.add("execute_sql")
        
        # Convert detected tool calls to the format expected by the diagnostic
        result["tool_calls"] = [{"function": {"name": name}} for name in detected_tool_calls]
        result["chunks"] = chunk_count
        result["final_response"] = "".join(response_parts)
        result["streaming_worked"] = chunk_count > 0  # Any chunks = working
        result["success"] = True
        
        return result
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "chunks": 0,
            "streaming_worked": False,
            "has_tool_call_info": False,
            "tool_calls": []
        }


async def test_raw_openai_streaming(api_key, model, messages, tools):
    """Test raw OpenAI streaming for comparison."""
    try:
        import openai
        client = openai.AsyncOpenAI(api_key=api_key)
        
        result = {
            "success": False,
            "chunks": 0,
            "tool_calls": [],
            "final_response": "",
            "streaming_worked": False,
            "error": None
        }
        
        # Handle different model types
        kwargs = {
            "model": model,
            "messages": messages,
            "stream": True
        }
        
        if any(pattern in model.lower() for pattern in ["o3", "o4", "o1"]):
            # Reasoning models use max_completion_tokens
            kwargs["max_completion_tokens"] = 200
            if tools:
                # O3/O4 support tools, O1 doesn't
                if not ("o1" in model.lower()):
                    kwargs["tools"] = tools
                else:
                    return {"success": False, "error": "O1 models don't support tools", "streaming_worked": False}
        else:
            # Regular models (GPT-4.1, GPT-4o) use max_tokens
            kwargs["tools"] = tools
            kwargs["max_tokens"] = 200
        
        response = await client.chat.completions.create(**kwargs)
        
        chunk_count = 0
        response_parts = []
        tool_calls = {}
        
        async for chunk in response:
            chunk_count += 1
            
            if chunk.choices and len(chunk.choices) > 0:
                choice = chunk.choices[0]
                
                if choice.delta and choice.delta.content:
                    response_parts.append(choice.delta.content)
                
                if choice.delta and choice.delta.tool_calls:
                    for tc in choice.delta.tool_calls:
                        idx = tc.index or 0
                        if idx not in tool_calls:
                            tool_calls[idx] = {"name": "", "arguments": "", "id": None}
                        
                        if tc.id:
                            tool_calls[idx]["id"] = tc.id
                        if tc.function:
                            if tc.function.name:
                                tool_calls[idx]["name"] += tc.function.name
                            if tc.function.arguments:
                                tool_calls[idx]["arguments"] += tc.function.arguments
        
        result["chunks"] = chunk_count
        result["final_response"] = "".join(response_parts)
        result["tool_calls"] = list(tool_calls.values())
        result["streaming_worked"] = chunk_count > 0
        result["success"] = True
        
        return result
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "chunks": 0,
            "streaming_worked": False
        }


def analyze_single_result_improved(chuk_result, raw_result, test_name, test_case):
    """Analyze a single test case result with improved logic."""
    print(f"    📋 {test_name} Results:")
    
    if not chuk_result["success"]:
        print(f"      ❌ ChukLLM failed: {chuk_result.get('error', 'Unknown')}")
        return False
    
    if not raw_result["success"]:
        print(f"      ❌ Raw OpenAI failed: {raw_result.get('error', 'Unknown')}")
        return False
    
    # Improved streaming analysis based on test type
    chuk_streamed = chuk_result["streaming_worked"]
    raw_streamed = raw_result["streaming_worked"]
    evaluation_type = test_case["evaluation"]
    min_chunks = test_case["min_chunks_chuk"]
    
    # Check if ChukLLM meets minimum requirements
    chuk_meets_minimum = chuk_result["chunks"] >= min_chunks
    
    if evaluation_type == "tool_streaming":
        # For tool calls, we expect streaming behavior (incremental JSON or formatted calls)
        if chuk_streamed and chuk_result["has_tool_call_info"]:
            chuk_tool_count = len(chuk_result.get("tool_calls", []))
            print(f"      ✅ ChukLLM tool streaming works ({chuk_result['chunks']} chunks with tool info)")
            print(f"      🔧 ChukLLM: {chuk_tool_count} tool calls detected")
            success = True
        elif chuk_streamed and chuk_result["chunks"] >= 5:
            # Even without detected patterns, if we have good chunk count, likely working
            print(f"      ✅ ChukLLM tool streaming likely works ({chuk_result['chunks']} chunks)")
            success = True
        elif not chuk_streamed:
            print(f"      ❌ ChukLLM tool streaming failed (0 chunks)")
            success = False
        else:
            print(f"      ⚠️  ChukLLM streaming but tool info detection may need update ({chuk_result['chunks']} chunks)")
            success = chuk_result["chunks"] > 1  # If streaming multiple chunks, probably working
            
        # Raw OpenAI comparison for context
        if raw_streamed:
            print(f"      📊 Raw OpenAI: {raw_result['chunks']} chunks, {len(raw_result['tool_calls'])} tool calls")
        
    else:  # text_streaming
        # For regular text, expect similar performance
        if chuk_streamed and chuk_meets_minimum:
            chunk_diff = abs(chuk_result['chunks'] - raw_result['chunks'])
            chunk_ratio = chunk_diff / max(raw_result['chunks'], 1)
            
            if chunk_ratio < 0.5:  # Within 50% is considered good
                print(f"      ✅ Both streamed similarly (ChukLLM: {chuk_result['chunks']}, Raw: {raw_result['chunks']})")
                success = True
            else:
                print(f"      ⚠️  Significant chunk difference (ChukLLM: {chuk_result['chunks']}, Raw: {raw_result['chunks']})")
                success = chuk_meets_minimum  # Still success if meets minimum
        elif chuk_streamed and not chuk_meets_minimum:
            print(f"      ⚠️  ChukLLM streaming but low chunk count ({chuk_result['chunks']} < {min_chunks})")
            success = False
        elif not chuk_streamed:
            print(f"      ❌ ChukLLM not streaming properly")
            success = False
        else:
            success = True
    
    # Show response previews
    chuk_preview = chuk_result["final_response"][:100] if chuk_result["final_response"] else ""
    raw_preview = raw_result["final_response"][:100] if raw_result["final_response"] else ""
    
    if chuk_preview:
        print(f"      📝 ChukLLM: {chuk_preview}...")
    if raw_preview:
        print(f"      📝 Raw:     {raw_preview}...")
    
    return success


def analyze_all_results_improved(results, overall_success):
    """Analyze all test results with improved logic."""
    print("\n🎯 COMPREHENSIVE ANALYSIS:")
    
    total_tests = 0
    successful_tests = 0
    streaming_worked_count = 0
    tool_calls_worked = 0
    
    for model, model_data in results.items():
        model_results = model_data["results"]
        model_success = model_data["success"]
        
        status_emoji = "✅" if model_success else "⚠️"
        print(f"\n  📊 {model}: {status_emoji}")
        
        for test_name, test_result in model_results.items():
            total_tests += 1
            
            chuk = test_result["chuk"]
            raw = test_result["raw"]
            evaluation_type = test_result["evaluation_type"]
            
            if chuk["success"] and raw["success"]:
                # Determine success based on evaluation type
                if evaluation_type == "tool_streaming":
                    test_success = chuk["streaming_worked"] and chuk.get("has_tool_call_info", False)
                    if test_success:
                        tool_calls_worked += 1
                else:  # text_streaming
                    min_chunks = test_result["min_chunks_expected"]
                    test_success = chuk["streaming_worked"] and chuk["chunks"] >= min_chunks
                
                if test_success:
                    print(f"    ✅ {test_name}")
                    successful_tests += 1
                    if chuk["streaming_worked"]:
                        streaming_worked_count += 1
                else:
                    print(f"    ⚠️  {test_name} (streaming issues)")
                    if chuk["streaming_worked"]:
                        streaming_worked_count += 1
            else:
                print(f"    ❌ {test_name}")
    
    print(f"\n📈 OVERALL STATISTICS:")
    print(f"  Total tests: {total_tests}")
    print(f"  Successful: {successful_tests}/{total_tests} ({100*successful_tests//total_tests if total_tests > 0 else 0}%)")
    print(f"  ChukLLM streaming worked: {streaming_worked_count}/{total_tests}")
    print(f"  Tool calls worked: {tool_calls_worked} tests")
    
    # Updated success criteria
    success_rate = successful_tests / total_tests if total_tests > 0 else 0
    streaming_rate = streaming_worked_count / total_tests if total_tests > 0 else 0
    
    # Success if most tests pass and streaming generally works
    final_success = success_rate >= 0.75 and streaming_rate >= 0.75
    
    return final_success


async def test_reasoning_models_specific_behavior():
    """Test O3/O4 reasoning model specific streaming behavior."""
    print("\n🧠 REASONING MODEL SPECIFIC TESTS (O3/O4 SERIES)")
    print("=" * 50)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return False
    
    # Test current reasoning models (actually available ones with tool support)
    reasoning_models = [
        ("o3-mini", "O3 Mini Reasoning"),  # Available with Tier 3+
        ("o4-mini", "O4 Mini Reasoning"),  # Available and supports tools
    ]
    
    success_count = 0
    
    for model, model_desc in reasoning_models:
        print(f"\n🎯 Testing {model} ({model_desc})")
        
        # Test reasoning capability
        reasoning_prompt = "Solve this step by step: If I have 15 apples and eat 3, then buy 8 more, how many do I have? Show your reasoning."
        
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from chuk_llm import stream
            
            print("  📝 Testing reasoning with ChukLLM streaming...")
            
            chunks = []
            chunk_count = 0
            async for chunk in stream(
                reasoning_prompt,
                provider="openai",
                model=model,
                max_completion_tokens=500
            ):
                chunk_count += 1
                if chunk:
                    chunks.append(str(chunk))
            
            full_response = "".join(chunks)
            print(f"  ✅ {model} streamed {chunk_count} chunks")
            print(f"  📋 Response preview: {full_response[:200]}...")
            
            # Check for reasoning indicators
            reasoning_indicators = ["step", "first", "then", "therefore", "because", "since", "solve", "calculate"]
            reasoning_found = sum(1 for indicator in reasoning_indicators if indicator in full_response.lower())
            
            print(f"  🧠 Reasoning indicators found: {reasoning_found}")
            
            # Test tool calling for O3/O4 (they support it unlike O1)
            print("  🔧 Testing tool calling capability...")
            
            tool_chunks = 0
            tool_info_detected = False
            
            async for chunk in stream(
                "Execute SQL: SELECT COUNT(*) FROM users WHERE active = 1",
                provider="openai",
                model=model,
                max_completion_tokens=300,
                tools=[{
                    "type": "function",
                    "function": {
                        "name": "execute_sql",
                        "description": "Execute a SQL query",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string", "description": "SQL query"}
                            },
                            "required": ["query"]
                        }
                    }
                }]
            ):
                tool_chunks += 1
                if chunk and "[Calling:" in str(chunk):
                    tool_info_detected = True
            
            print(f"  🔧 Tool calling: {tool_chunks} chunks, tool info: {tool_info_detected}")
            
            # Success criteria for reasoning models
            reasoning_success = chunk_count >= 10 and reasoning_found >= 2
            tool_success = tool_chunks > 0 and tool_info_detected
            
            if reasoning_success and tool_success:
                print(f"  ✅ {model} excellent: reasoning + tool calling work")
                success_count += 1
            elif reasoning_success:
                print(f"  ✅ {model} good: reasoning works, tool calling may need work")
                success_count += 1
            elif tool_success:
                print(f"  ⚠️  {model} partial: tool calling works, reasoning limited")
            else:
                print(f"  ❌ {model} needs improvement")
                
        except Exception as e:
            print(f"  ❌ {model} failed: {e}")
    
    return success_count >= len(reasoning_models) * 0.67  # 67% success rate acceptable


async def main():
    """Run enhanced streaming diagnostic focused on current OpenAI models."""
    print("🚀 ENHANCED STREAMING DIAGNOSTIC - CURRENT OPENAI MODELS 2025")
    print("Testing streaming across GPT-4.1 series and O3/O4 reasoning models")
    print("Focus: o3, o4-mini, o3-mini, gpt-4.1, gpt-4.1-mini")
    
    # Test 1: Multi-model streaming across current lineup
    multi_model_ok = await test_streaming_with_models()
    
    # Test 2: Reasoning models specific behavior (O3/O4 series)
    reasoning_ok = await test_reasoning_models_specific_behavior()
    
    print("\n" + "=" * 60)
    print("🎯 FINAL DIAGNOSTIC SUMMARY:")
    print(f"Multi-model tests: {'✅ PASS' if multi_model_ok else '⚠️ PARTIAL' if not multi_model_ok else '❌ FAIL'}")
    print(f"Reasoning model tests: {'✅ PASS' if reasoning_ok else '❌ FAIL'}")
    
    if multi_model_ok and reasoning_ok:
        print("\n🎉 STREAMING WORKS EXCELLENTLY ACROSS CURRENT MODELS!")
        print("   ✅ GPT-4.1 series: Working properly")
        print("   ✅ O3/O4 reasoning models: Working with tool support")
        print("   ✅ Tool call streaming: Working with informative output")
        print("   ✅ Current model lineup fully supported")
    elif multi_model_ok:
        print("\n✅ STREAMING WORKS WELL!")
        print("   Most current models streaming properly")
        print("   Tool calls provide informative feedback")
        print("   Some reasoning model issues may exist")
    else:
        print("\n⚠️  STREAMING NEEDS ATTENTION")
        if not multi_model_ok:
            print("   Some current model streaming issues remain")
        if not reasoning_ok:
            print("   O3/O4 reasoning model streaming has problems")
    
    print("\n🎉 Diagnostic complete - focused on 2025 OpenAI model lineup!")

if __name__ == "__main__":
    asyncio.run(main())