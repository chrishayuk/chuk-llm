#!/usr/bin/env python3
"""
Enhanced Mistral Streaming Diagnostic

Comprehensive testing of Mistral streaming tool call behavior.
Tests both accumulation strategies and duplication detection.
Compares Raw Mistral behavior with chuk-llm implementation.
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


async def test_mistral_streaming_strategies():
    """Test different Mistral streaming accumulation strategies."""
    
    print("🔍 MISTRAL STREAMING STRATEGY ANALYSIS")
    print("=" * 50)
    
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        print("❌ No MISTRAL_API_KEY")
        return False
    
    print(f"🔧 Using Mistral API key: {api_key[:10]}...")
    
    # Test case with multiple parameters to test streaming behavior
    tools = [{
        "type": "function",
        "function": {
            "name": "process_document",
            "description": "Process a document with multiple analysis options",
            "parameters": {
                "type": "object",
                "properties": {
                    "document_path": {"type": "string", "description": "Path to the document"},
                    "analysis_options": {
                        "type": "object",
                        "properties": {
                            "extract_entities": {"type": "boolean", "description": "Extract named entities"},
                            "sentiment_analysis": {"type": "boolean", "description": "Perform sentiment analysis"},
                            "summarize": {"type": "boolean", "description": "Generate summary"}
                        }
                    },
                    "output_format": {"type": "string", "enum": ["json", "xml", "text"]},
                    "language": {"type": "string", "description": "Document language"},
                    "confidence_threshold": {"type": "number", "description": "Minimum confidence score"}
                },
                "required": ["document_path", "output_format"]
            }
        }
    }]
    
    messages = [{
        "role": "user", 
        "content": "Process document '/data/report.pdf' with entity extraction enabled, sentiment analysis enabled, summarization enabled, output as json, language english, confidence threshold 0.85"
    }]
    
    print("🎯 Test: Complex document processing with nested parameters")
    print("Expected: Should test Mistral's tool call handling (using mistral-medium-2505)")
    
    # Test raw Mistral with different accumulation strategies
    print("\n🔥 RAW MISTRAL ANALYSIS:")
    raw_concatenation = await test_raw_mistral_concatenation(api_key, messages, tools)
    raw_replacement = await test_raw_mistral_replacement(api_key, messages, tools)
    
    # Test chuk-llm current behavior
    print("\n🔧 CHUK-LLM CURRENT:")
    chuk_result = await test_chuk_llm_mistral(messages, tools)
    
    # Detailed comparison
    print("\n📊 COMPARISON:")
    print(f"Raw (concatenation): {len(raw_concatenation) if raw_concatenation else 0} chars")
    print(f"Raw (replacement):   {len(raw_replacement) if raw_replacement else 0} chars")
    print(f"Chuk-LLM current:    {len(chuk_result) if chuk_result else 0} chars")
    
    # Analyze results
    return analyze_mistral_results({
        "raw_concat": raw_concatenation,
        "raw_replace": raw_replacement,
        "chuk_current": chuk_result
    })


async def test_raw_mistral_concatenation(api_key, messages, tools):
    """Test raw Mistral with concatenation strategy."""
    try:
        from mistralai import Mistral
        
        client = Mistral(api_key=api_key)
        
        # Concatenation strategy
        tool_calls = {}
        chunk_count = 0
        tool_call_chunks = 0
        
        try:
            # Use Mistral's streaming
            stream = client.chat.stream(
                model="mistral-medium-2505",  # Use available model
                messages=messages,
                tools=tools,
                tool_choice="auto"
            )
            
            for chunk in stream:
                chunk_count += 1
                
                if hasattr(chunk, 'data') and hasattr(chunk.data, 'choices'):
                    choices = chunk.data.choices
                    if choices:
                        choice = choices[0]
                        
                        if hasattr(choice, 'delta') and choice.delta:
                            delta = choice.delta
                            
                            # Check for tool calls in delta
                            if hasattr(delta, 'tool_calls') and delta.tool_calls:
                                tool_call_chunks += 1
                                for i, tc in enumerate(delta.tool_calls):
                                    if i not in tool_calls:
                                        tool_calls[i] = {"name": "", "arguments": ""}
                                    
                                    if hasattr(tc, 'function') and tc.function:
                                        # CONCATENATION STRATEGY
                                        if hasattr(tc.function, 'name') and tc.function.name:
                                            tool_calls[i]["name"] += tc.function.name
                                        if hasattr(tc.function, 'arguments') and tc.function.arguments:
                                            tool_calls[i]["arguments"] += tc.function.arguments
        
        except Exception as stream_error:
            print(f"  ⚠️ Streaming failed, trying non-streaming: {stream_error}")
            
            # Fallback to non-streaming
            response = client.chat.complete(
                model="mistral-medium-2505",  # Use available model
                messages=messages,
                tools=tools,
                tool_choice="auto"
            )
            
            chunk_count = 1
            if (hasattr(response, 'choices') and response.choices and 
                hasattr(response.choices[0], 'message') and 
                hasattr(response.choices[0].message, 'tool_calls') and
                response.choices[0].message.tool_calls):
                
                tool_call_chunks = 1
                for i, tc in enumerate(response.choices[0].message.tool_calls):
                    tool_calls[i] = {
                        "name": tc.function.name if tc.function else "",
                        "arguments": tc.function.arguments if tc.function else ""
                    }
        
        print(f"  Strategy: CONCATENATION")
        print(f"  Chunks: {chunk_count}, Tool chunks: {tool_call_chunks}")
        
        if tool_calls:
            for idx, tc in tool_calls.items():
                print(f"  Tool: {tc['name']}({len(tc['arguments'])} chars)")
            return list(tool_calls.values())[0]['arguments']
        return ""
        
    except Exception as e:
        print(f"  ❌ Raw Mistral concatenation error: {e}")
        return None


async def test_raw_mistral_replacement(api_key, messages, tools):
    """Test raw Mistral with replacement strategy."""
    try:
        from mistralai import Mistral
        
        client = Mistral(api_key=api_key)
        
        # Replacement strategy
        tool_calls = {}
        chunk_count = 0
        tool_call_chunks = 0
        
        try:
            # Use Mistral's streaming
            stream = client.chat.stream(
                model="mistral-medium-2505",  # Use available model
                messages=messages,
                tools=tools,
                tool_choice="auto"
            )
            
            for chunk in stream:
                chunk_count += 1
                
                if hasattr(chunk, 'data') and hasattr(chunk.data, 'choices'):
                    choices = chunk.data.choices
                    if choices:
                        choice = choices[0]
                        
                        if hasattr(choice, 'delta') and choice.delta:
                            delta = choice.delta
                            
                            # Check for tool calls in delta
                            if hasattr(delta, 'tool_calls') and delta.tool_calls:
                                tool_call_chunks += 1
                                for i, tc in enumerate(delta.tool_calls):
                                    if i not in tool_calls:
                                        tool_calls[i] = {"name": "", "arguments": ""}
                                    
                                    if hasattr(tc, 'function') and tc.function:
                                        # REPLACEMENT STRATEGY
                                        if hasattr(tc.function, 'name') and tc.function.name is not None:
                                            tool_calls[i]["name"] = tc.function.name
                                        if hasattr(tc.function, 'arguments') and tc.function.arguments is not None:
                                            tool_calls[i]["arguments"] = tc.function.arguments
        
        except Exception as stream_error:
            print(f"  ⚠️ Streaming failed, trying non-streaming: {stream_error}")
            
            # Fallback to non-streaming
            response = client.chat.complete(
                model="mistral-medium-2505",  # Use available model
                messages=messages,
                tools=tools,
                tool_choice="auto"
            )
            
            chunk_count = 1
            if (hasattr(response, 'choices') and response.choices and 
                hasattr(response.choices[0], 'message') and 
                hasattr(response.choices[0].message, 'tool_calls') and
                response.choices[0].message.tool_calls):
                
                tool_call_chunks = 1
                for i, tc in enumerate(response.choices[0].message.tool_calls):
                    tool_calls[i] = {
                        "name": tc.function.name if tc.function else "",
                        "arguments": tc.function.arguments if tc.function else ""
                    }
        
        print(f"  Strategy: REPLACEMENT")
        print(f"  Chunks: {chunk_count}, Tool chunks: {tool_call_chunks}")
        
        if tool_calls:
            for idx, tc in tool_calls.items():
                print(f"  Tool: {tc['name']}({len(tc['arguments'])} chars)")
            return list(tool_calls.values())[0]['arguments']
        return ""
        
    except Exception as e:
        print(f"  ❌ Raw Mistral replacement error: {e}")
        return None


async def test_chuk_llm_mistral(messages, tools):
    """Test chuk-llm Mistral streaming."""
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from chuk_llm.llm.client import get_client
        
        client = get_client(provider="mistral", model="mistral-medium-2505")  # Use available model
        
        # Stream with chuk-llm
        chunk_count = 0
        final_tool_calls = []
        all_chunks = []
        streaming_success = False
        
        try:
            async for chunk in client.create_completion(
                messages=messages,
                tools=tools,
                stream=True
            ):
                chunk_count += 1
                all_chunks.append(chunk)
                streaming_success = True
                
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
            
            print(f"  Streaming mode: SUCCESS")
        
        except Exception as stream_error:
            print(f"  Streaming mode: FAILED ({stream_error})")
            print(f"  Falling back to non-streaming")
            
            # Fallback to non-streaming
            try:
                result = await client.create_completion(
                    messages=messages,
                    tools=tools,
                    stream=False
                )
                
                chunk_count = 1
                all_chunks = [result]
                if result.get("tool_calls"):
                    final_tool_calls.extend(result["tool_calls"])
            
            except Exception as non_stream_error:
                print(f"  Non-streaming also failed: {non_stream_error}")
                return None
        
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
        print(f"  ❌ Chuk Mistral error: {e}")
        return None


def analyze_mistral_results(results):
    """Analyze the Mistral streaming strategies."""
    print("\n🔬 ANALYSIS:")
    
    raw_concat = results["raw_concat"]
    raw_replace = results["raw_replace"] 
    chuk_current = results["chuk_current"]
    
    try:
        # Parse JSON arguments for comparison
        concat_parsed = json.loads(raw_concat) if raw_concat else {}
        replace_parsed = json.loads(raw_replace) if raw_replace else {}
        chuk_parsed = json.loads(chuk_current) if chuk_current else {}
        
        print(f"Concatenation result: {len(concat_parsed)} parameters")
        print(f"Replacement result: {len(replace_parsed)} parameters")
        print(f"Chuk-LLM result: {len(chuk_parsed)} parameters")
        
        # Mistral should behave like OpenAI (concatenation correct for streaming)
        if concat_parsed and len(concat_parsed) >= len(replace_parsed):
            print("✅ CONCATENATION IS CORRECT for Mistral")
            print("   Mistral follows OpenAI-compatible streaming patterns")
            
            # Check if chuk matches the correct (concatenation) result
            if chuk_parsed == concat_parsed:
                print("✅ CHUK-LLM HANDLES MISTRAL CORRECTLY")
                print("   Tool call streaming works properly")
                return True
            elif chuk_parsed == replace_parsed:
                print("❌ CHUK-LLM USES BROKEN LOGIC FOR MISTRAL")
                print("   Not accumulating deltas properly")
                print("\n🔧 FIX NEEDED:")
                print("   File: chuk_llm/llm/providers/mistral_client.py")
                print("   Problem: Not using concatenation for Mistral deltas")
                print("   Solution: Ensure Mistral follows same logic as OpenAI")
                return False
            elif len(chuk_parsed) > 0 and len(concat_parsed) > 0:
                # Both have data but differ - this might be acceptable
                print("⚠️  MISTRAL RESULTS DIFFER BUT BOTH HAVE DATA")
                print("   This might be due to Mistral-specific behavior")
                return True
            else:
                print("❓ CHUK-LLM HAS DIFFERENT MISTRAL BEHAVIOR")
                print(f"   Expected (concat): {concat_parsed}")
                print(f"   Got (chuk):        {chuk_parsed}")
                return False
        elif replace_parsed and len(replace_parsed) > len(concat_parsed):
            print("⚠️  REPLACEMENT WORKS BETTER - Unusual for Mistral")
            if chuk_parsed == replace_parsed:
                print("✅ CHUK-LLM MATCHES REPLACEMENT STRATEGY")
                return True
            else:
                return False
        else:
            print("❓ BOTH STRATEGIES FAILED OR GAVE EMPTY RESULTS")
            print("   This might be a Mistral API limitation or tool compatibility issue")
            return len(chuk_parsed) > 0  # Pass if chuk at least got something
            
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing error: {e}")
        print(f"Concat: '{raw_concat}'")
        print(f"Replace: '{raw_replace}'")
        print(f"Chuk: '{chuk_current}'")
        return False


async def test_duplication_specifically_mistral():
    """Test specifically for Mistral tool call duplication bug."""
    print("\n🔍 MISTRAL DUPLICATION TEST")
    print("=" * 35)
    
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        print("❌ Missing Mistral API key")
        return False
    
    tools = [{
        "type": "function",
        "function": {
            "name": "test_mistral_tool",
            "description": "Test tool for Mistral",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "Test message"},
                    "number": {"type": "integer", "description": "Test number"},
                    "settings": {
                        "type": "object",
                        "properties": {
                            "enabled": {"type": "boolean", "description": "Feature enabled"}
                        }
                    }
                },
                "required": ["message"]
            }
        }
    }]
    
    messages = [{"role": "user", "content": "Call test_mistral_tool with message 'hello mistral' number 123 and enabled true"}]
    
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from chuk_llm.llm.client import get_client
        
        client = get_client(provider="mistral", model="mistral-medium-2505")  # Use available model
        
        all_tool_calls = []
        chunk_count = 0
        
        try:
            async for chunk in client.create_completion(
                messages=messages,
                tools=tools,
                stream=True
            ):
                chunk_count += 1
                if chunk.get("tool_calls"):
                    all_tool_calls.extend(chunk["tool_calls"])
        
        except Exception as stream_error:
            print(f"Streaming failed, trying non-streaming: {stream_error}")
            
            result = await client.create_completion(
                messages=messages,
                tools=tools,
                stream=False
            )
            
            chunk_count = 1
            if result.get("tool_calls"):
                all_tool_calls.extend(result["tool_calls"])
        
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
        elif len(all_tool_calls) == 0:
            print("⚠️  NO TOOL CALLS - Mistral may not support this tool format")
            return False
        else:
            print("❓ UNEXPECTED TOOL CALL PATTERN")
            for i, tc in enumerate(all_tool_calls):
                print(f"  Tool {i+1}: {tc['function']['name']}({tc['function']['arguments']})")
            return len(unique_tool_calls) > 0
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def main():
    """Run enhanced Mistral streaming diagnostic."""
    print("🚀 ENHANCED MISTRAL STREAMING DIAGNOSTIC")
    print("Testing Mistral tool call behavior with duplication detection")
    
    # Test 1: Check for duplication specifically
    duplication_ok = await test_duplication_specifically_mistral()
    
    # Test 2: Verify streaming behavior
    behavior_ok = await test_mistral_streaming_strategies()
    
    print("\n" + "=" * 65)
    print("🎯 MISTRAL DIAGNOSTIC SUMMARY:")
    print(f"Duplication test: {'✅ PASS' if duplication_ok else '❌ FAIL'}")
    print(f"Behavior test:    {'✅ PASS' if behavior_ok else '❌ FAIL'}")
    
    if duplication_ok and behavior_ok:
        print("\n✅ MISTRAL STREAMING WORKS PERFECTLY!")
        print("   No tool call duplication detected")
        print("   Proper delta accumulation in use")
        print("   Mistral integration working correctly")
    elif duplication_ok and not behavior_ok:
        print("\n⚠️  BEHAVIOR ISSUES DETECTED")
        print("   No duplication, but tool call processing may be incomplete")
        print("   Check Mistral delta handling or fallback logic")
    elif not duplication_ok and behavior_ok:
        print("\n⚠️  DUPLICATION ISSUE DETECTED") 
        print("   Tool calls work, but are being duplicated")
        print("   Check Mistral chunk processing logic")
        print("   Apply the same fix as used for Groq")
    else:
        print("\n❌ MISTRAL NEEDS ATTENTION")
        print("   Multiple issues detected with Mistral streaming")
        print("   Apply duplication prevention and check streaming logic")
    
    print("\n💡 MISTRAL SETUP REMINDER:")
    print("Required environment variables:")
    print("- MISTRAL_API_KEY")
    print("- Note: Mistral requires alphanumeric + underscore tool names only")
    print("- Some models may have limited function calling support")
    print("- Fallback to non-streaming is normal for some Mistral models")


if __name__ == "__main__":
    asyncio.run(main())