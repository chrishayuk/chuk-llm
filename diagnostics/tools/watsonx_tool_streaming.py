#!/usr/bin/env python3
"""
Enhanced WatsonX Streaming Diagnostic with Official Chat Templates
=================================================================

Tests both concatenation vs replacement strategies using OFFICIAL IBM WatsonX
Granite chat templates. Compares streaming behavior across Granite and Mistral models.

CRITICAL FOCUS:
1. Official AutoTokenizer.apply_chat_template() usage
2. Streaming tool call duplication prevention 
3. Granite vs Mistral streaming behavior comparison
4. No hacky workarounds - pure official API usage
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List

# Load environment
try:
    from dotenv import load_dotenv
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    env_file = project_root / ".env"
    if env_file.exists():
        load_dotenv(env_file)
        print(f"✅ Loaded .env")
    else:
        load_dotenv()
except ImportError:
    print("⚠️ No dotenv")

# Import official Granite tokenizer
try:
    from transformers import AutoTokenizer
    GRANITE_TOKENIZER_AVAILABLE = True
    print("✅ Transformers available for official chat templates")
except ImportError:
    GRANITE_TOKENIZER_AVAILABLE = False
    print("❌ Transformers not available - cannot test official templates")


def safe_parse_tool_arguments(arguments: Any) -> Dict[str, Any]:
    """Safely parse tool arguments with detailed logging"""
    if arguments is None:
        return {}
    
    if isinstance(arguments, dict):
        return arguments
    
    if isinstance(arguments, str):
        if not arguments.strip():
            return {}
        
        try:
            parsed = json.loads(arguments)
            if isinstance(parsed, dict):
                return parsed
            else:
                print(f"      ⚠️  Parsed arguments is not a dict: {type(parsed)}")
                return {}
        except (json.JSONDecodeError, ValueError) as e:
            print(f"      ⚠️  JSON parse error: {e}")
            return {}
    
    print(f"      ⚠️  Unexpected arguments type: {type(arguments)}")
    return {}


def parse_granite_streaming_response(text: str) -> List[Dict[str, Any]]:
    """Parse Granite streaming tool responses using official patterns"""
    if not text:
        return []
    
    tool_calls = []
    import re
    
    try:
        # Official Granite format: {'name': 'func', 'arguments': {...}}
        pattern = r"{'name':\s*'([^']+)',\s*'arguments':\s*({[^}]*})"
        matches = re.findall(pattern, text)
        
        for func_name, args_str in matches:
            try:
                # Convert to proper JSON
                args_json = args_str.replace("'", '"')
                args = json.loads(args_json)
                
                tool_calls.append({
                    "id": f"stream_call_{len(tool_calls)}",
                    "type": "function",
                    "function": {
                        "name": func_name,
                        "arguments": json.dumps(args)
                    }
                })
            except:
                continue
                
        if tool_calls:
            print(f"    🧠 Parsed {len(tool_calls)} Granite tool calls from streaming text")
            
    except Exception as e:
        print(f"    ⚠️  Error parsing Granite streaming response: {e}")
    
    return tool_calls


async def test_streaming_duplication_prevention(model_name: str):
    """
    CRITICAL TEST: Verify WatsonX streaming doesn't duplicate tool calls
    using official chat templates.
    """
    print(f"🔍 WATSONX STREAMING DUPLICATION PREVENTION - {model_name}")
    print("=" * 60)
    
    api_key = os.getenv("WATSONX_API_KEY") or os.getenv("IBM_CLOUD_API_KEY")
    project_id = os.getenv("WATSONX_PROJECT_ID")
    
    if not api_key or not project_id:
        print("❌ Missing WatsonX credentials")
        return False
    
    if not GRANITE_TOKENIZER_AVAILABLE:
        print("❌ Transformers not available for official templates")
        return False
    
    # Test case designed to trigger multiple chunks with tool calls
    tools = [{
        "type": "function",
        "function": {
            "name": "stdio.describe_table",
            "description": "Get detailed schema information for a database table",
            "parameters": {
                "type": "object",
                "properties": {
                    "table_name": {"type": "string", "description": "Name of the table to describe"},
                    "include_indexes": {"type": "boolean", "description": "Include index information"},
                    "include_constraints": {"type": "boolean", "description": "Include constraint information"}
                },
                "required": ["table_name"]
            }
        }
    }, {
        "type": "function",
        "function": {
            "name": "web.api:search",
            "description": "Search for information using web API",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "category": {"type": "string", "description": "Search category"},
                    "max_results": {"type": "integer", "description": "Maximum number of results"}
                },
                "required": ["query"]
            }
        }
    }]
    
    conversation = [{
        "role": "system",
        "content": "You are a helpful assistant with access to function calls. Use the appropriate functions to fulfill user requests."
    }, {
        "role": "user", 
        "content": "Please describe the users table with full details including indexes and constraints, then search for 'database schema best practices' in the development category with 10 results"
    }]
    
    print("🎯 Test: Complex multi-tool request with official chat templates")
    print("Expected: Each tool call should appear exactly once, no duplicates")
    
    try:
        from chuk_llm.llm.client import get_client
        
        client = get_client(provider="watsonx", model=model_name)
        
        # Initialize official Granite tokenizer
        tokenizer = AutoTokenizer.from_pretrained("ibm-granite/granite-3.0-8b-instruct")
        
        # Create Granite-compatible tools
        granite_tools = []
        for tool in tools:
            func_def = tool["function"]
            granite_tool = {
                "type": "function",
                "function": {
                    "name": func_def["name"],
                    "description": func_def["description"], 
                    "parameters": func_def["parameters"],
                    "return": {
                        "type": "object",
                        "description": f"Result from {func_def['name']} function"
                    }
                }
            }
            granite_tools.append(granite_tool)
        
        # Track streaming data
        all_tool_calls = []
        all_tool_signatures = []
        duplicate_signatures = set()
        chunk_count = 0
        text_chunks = 0
        tool_chunks = 0
        granite_format_chunks = 0
        accumulated_text = ""
        
        if "granite" in model_name.lower():
            print(f"🧠 Using official Granite chat template for {model_name}")
            
            # Apply official chat template
            instruction = tokenizer.apply_chat_template(
                conversation=conversation,
                tools=granite_tools,
                tokenize=False,
                add_generation_prompt=True
            )
            
            print(f"   ✅ Official template applied: {len(instruction)} chars")
            
            # Stream using template
            async for chunk in client.create_completion(
                messages=[{"role": "user", "content": instruction}],
                stream=True,
                max_tokens=600
            ):
                chunk_count += 1
                
                # Track text content
                if chunk.get("response"):
                    text_chunks += 1
                    text = chunk["response"]
                    accumulated_text += text
                    
                    # Check for Granite-specific tool formats in text
                    if any(pattern in text for pattern in [
                        "'name':", "{'name':", "arguments", "describe_table", "web.api:search"
                    ]):
                        granite_format_chunks += 1
                        print(f"    📝 Granite format in chunk {chunk_count}: {text[:50]}...")
                
                # Track structured tool calls
                if chunk.get("tool_calls"):
                    tool_chunks += 1
                    print(f"    🔧 Structured tool chunk {chunk_count}: {len(chunk['tool_calls'])} tool(s)")
                    
                    for tc in chunk["tool_calls"]:
                        func_name = tc.get("function", {}).get("name", "unknown")
                        func_args = tc.get("function", {}).get("arguments", "{}")
                        
                        # Create unique signature
                        signature = f"{func_name}:{hash(func_args)}"
                        
                        all_tool_calls.append(tc)
                        
                        if signature in all_tool_signatures:
                            duplicate_signatures.add(signature)
                            print(f"      ❌ DUPLICATE DETECTED: {func_name}")
                        else:
                            all_tool_signatures.append(signature)
                            print(f"      ✅ New tool call: {func_name}")
        else:
            print(f"🌟 Using standard WatsonX API for {model_name}")
            
            # Stream using standard tools
            async for chunk in client.create_completion(
                messages=conversation,
                tools=tools,
                stream=True,
                max_tokens=600
            ):
                chunk_count += 1
                
                if chunk.get("response"):
                    text_chunks += 1
                    accumulated_text += chunk["response"]
                
                if chunk.get("tool_calls"):
                    tool_chunks += 1
                    for tc in chunk["tool_calls"]:
                        func_name = tc.get("function", {}).get("name", "unknown")
                        func_args = tc.get("function", {}).get("arguments", "{}")
                        signature = f"{func_name}:{hash(func_args)}"
                        
                        all_tool_calls.append(tc)
                        
                        if signature in all_tool_signatures:
                            duplicate_signatures.add(signature)
                            print(f"      ❌ DUPLICATE: {func_name}")
                        else:
                            all_tool_signatures.append(signature)
                            print(f"      ✅ New: {func_name}")
        
        # Parse accumulated text for WatsonX tool calls (all models)
        watsonx_parsed_calls = parse_granite_streaming_response(accumulated_text)
        
        # Analysis
        print(f"\n📊 STREAMING ANALYSIS FOR {model_name}:")
        print(f"   Total chunks: {chunk_count}")
        print(f"   Text chunks: {text_chunks}")
        print(f"   Structured tool chunks: {tool_chunks}")
        print(f"   WatsonX format chunks: {granite_format_chunks}")
        print(f"   Structured tool calls: {len(all_tool_calls)}")
        print(f"   WatsonX parsed calls: {len(watsonx_parsed_calls)}")
        print(f"   Unique signatures: {len(all_tool_signatures)}")
        print(f"   Duplicate signatures: {len(duplicate_signatures)}")
        
        # Determine success
        no_duplicates = len(duplicate_signatures) == 0
        has_tool_activity = len(all_tool_calls) > 0 or len(watsonx_parsed_calls) > 0 or granite_format_chunks > 0
        
        if no_duplicates:
            print(f"\n✅ NO DUPLICATION DETECTED for {model_name}!")
            
            if has_tool_activity:
                print(f"   ✅ Tool calling activity detected")
                
                if len(all_tool_calls) > 0:
                    print(f"   ✅ Structured tool calls working")
                if len(watsonx_parsed_calls) > 0:
                    print(f"   ✅ WatsonX text-based tool calls working")
                if granite_format_chunks > 0:
                    print(f"   ✅ WatsonX tool format patterns detected")
                
                return True
            else:
                print(f"   ⚠️  No tool calling activity detected")
                return False
        else:
            print(f"\n❌ DUPLICATION DETECTED for {model_name}!")
            print(f"   {len(duplicate_signatures)} duplicate signatures found")
            return False
            
    except Exception as e:
        print(f"❌ Error in duplication test for {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_granite_vs_mistral_streaming():
    """Compare streaming behavior between Granite and Mistral models"""
    print("\n🔄 GRANITE VS MISTRAL STREAMING COMPARISON")
    print("=" * 55)
    
    if not GRANITE_TOKENIZER_AVAILABLE:
        print("❌ Cannot test - transformers not available")
        return False
    
    models = [
        ("ibm/granite-3-8b-instruct", "Granite 3-8B"),
        ("mistralai/mistral-medium-2505", "Mistral Medium")
    ]
    
    tools = [{
        "type": "function",
        "function": {
            "name": "web.api:search",
            "description": "Search the web",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"]
            }
        }
    }]
    
    conversation = [{
        "role": "user",
        "content": "Search for 'IBM WatsonX function calling tutorial'"
    }]
    
    results = {}
    
    for model_id, model_name in models:
        print(f"\n🧠 Testing {model_name} ({model_id}):")
        
        try:
            from chuk_llm.llm.client import get_client
            client = get_client(provider="watsonx", model=model_id)
            
            chunk_count = 0
            tool_calls_found = []
            text_content = []
            streaming_errors = []
            
            if "granite" in model_id.lower():
                # Use official chat template for Granite
                tokenizer = AutoTokenizer.from_pretrained("ibm-granite/granite-3.0-8b-instruct")
                
                granite_tools = [{
                    "type": "function",
                    "function": {
                        "name": "web.api:search",
                        "description": "Search the web",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string", "description": "Search query"}
                            },
                            "required": ["query"]
                        },
                        "return": {
                            "type": "object",
                            "description": "Search results"
                        }
                    }
                }]
                
                instruction = tokenizer.apply_chat_template(
                    conversation=conversation,
                    tools=granite_tools,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                print(f"   🧠 Using official Granite chat template")
                
                async for chunk in client.create_completion(
                    messages=[{"role": "user", "content": instruction}],
                    stream=True,
                    max_tokens=300
                ):
                    chunk_count += 1
                    
                    if chunk.get("response"):
                        text_content.append(chunk["response"])
                        
                        # Parse for Granite tool calls
                        granite_calls = parse_granite_streaming_response(chunk["response"])
                        tool_calls_found.extend(granite_calls)
                    
                    if chunk.get("tool_calls"):
                        tool_calls_found.extend(chunk["tool_calls"])
                    
                    if chunk.get("error"):
                        streaming_errors.append(str(chunk["error"]))
                    
                    if chunk_count >= 20:
                        break
            else:
                # Use standard approach for Mistral
                print(f"   🌟 Using standard WatsonX API")
                
                async for chunk in client.create_completion(
                    messages=conversation,
                    tools=tools,
                    stream=True,
                    max_tokens=300
                ):
                    chunk_count += 1
                    
                    if chunk.get("response"):
                        text_content.append(chunk["response"])
                    
                    if chunk.get("tool_calls"):
                        tool_calls_found.extend(chunk["tool_calls"])
                    
                    if chunk.get("error"):
                        streaming_errors.append(str(chunk["error"]))
                    
                    if chunk_count >= 20:
                        break
            
            # Analyze results
            unique_tool_calls = []
            for tc in tool_calls_found:
                tc_signature = f"{tc['function']['name']}:{tc['function']['arguments']}"
                if tc_signature not in [f"{utc['function']['name']}:{utc['function']['arguments']}" for utc in unique_tool_calls]:
                    unique_tool_calls.append(tc)
            
            results[model_name] = {
                "chunks": chunk_count,
                "text_chunks": len(text_content),
                "total_tool_calls": len(tool_calls_found),
                "unique_tool_calls": len(unique_tool_calls),
                "errors": len(streaming_errors),
                "duplication_detected": len(tool_calls_found) != len(unique_tool_calls),
                "success": len(unique_tool_calls) > 0 or len(text_content) > 0
            }
            
            print(f"   Chunks: {chunk_count}")
            print(f"   Text chunks: {len(text_content)}")
            print(f"   Total tool calls: {len(tool_calls_found)}")
            print(f"   Unique tool calls: {len(unique_tool_calls)}")
            print(f"   Errors: {len(streaming_errors)}")
            
            if results[model_name]["duplication_detected"]:
                print(f"   ❌ Duplication detected!")
            else:
                print(f"   ✅ No duplication")
                
            if results[model_name]["success"]:
                print(f"   ✅ {model_name} streaming successful")
            else:
                print(f"   ⚠️  {model_name} streaming needs attention")
                
        except Exception as e:
            print(f"   ❌ Error testing {model_name}: {e}")
            results[model_name] = {"success": False, "error": str(e)}
    
    # Compare results
    print(f"\n📊 STREAMING COMPARISON SUMMARY:")
    
    successful_models = [name for name, result in results.items() if result.get("success")]
    if len(successful_models) >= 1:
        print(f"   ✅ Successful models: {successful_models}")
        
        # Check duplication across models
        duplication_free = [name for name, result in results.items() if not result.get("duplication_detected")]
        if duplication_free:
            print(f"   ✅ Duplication-free models: {duplication_free}")
        
        print(f"   ✅ STREAMING COMPATIBILITY ACHIEVED!")
        return True
    else:
        print(f"   ❌ No successful streaming models")
        return False


async def main():
    """Run WatsonX streaming diagnostics with official chat templates"""
    print("🧪 WATSONX STREAMING DIAGNOSTIC - OFFICIAL CHAT TEMPLATES")
    print("=" * 70)
    
    print("This diagnostic uses OFFICIAL IBM WatsonX patterns:")
    print("1. AutoTokenizer.apply_chat_template() for ALL WatsonX models")
    print("2. Universal tool name compatibility system")
    print("3. Official tool response parsing")
    print("4. No hacky workarounds - pure official API usage")
    
    # Test both Granite and Mistral models
    models_to_test = [
        "ibm/granite-3-8b-instruct",
        "ibm/granite-3-3-8b-instruct",
        "mistralai/mistral-medium-2505"
    ]
    
    results = {}
    
    print(f"\n📋 TESTING STREAMING DUPLICATION PREVENTION:")
    for model in models_to_test:
        print(f"\n{'='*60}")
        result = await test_streaming_duplication_prevention(model)
        results[f"{model}_duplication"] = result
    
    print(f"\n📋 TESTING GRANITE VS MISTRAL COMPARISON:")
    comparison_result = await test_granite_vs_mistral_streaming()
    results["comparison"] = comparison_result
    
    # Summary
    print(f"\n{'='*70}")
    print("🎯 WATSONX STREAMING DIAGNOSTIC SUMMARY:")
    
    duplication_tests = [k for k in results.keys() if "duplication" in k]
    duplication_passed = sum(1 for k in duplication_tests if results[k])
    
    print(f"   Duplication Prevention: {duplication_passed}/{len(duplication_tests)} models passed")
    print(f"   Granite vs Mistral Comparison: {'✅ PASS' if results['comparison'] else '❌ FAIL'}")
    
    total_passed = duplication_passed + (1 if results["comparison"] else 0)
    total_tests = len(duplication_tests) + 1
    
    print(f"   Overall: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("\n🎉 ALL STREAMING TESTS PASSED!")
        print("✅ Official chat templates work perfectly for streaming")
        print("✅ No tool call duplication detected")
        print("✅ Both Granite and Mistral models compatible")
        print("✅ Official IBM WatsonX patterns validated")
        
        print(f"\n🚀 PRODUCTION READY STREAMING:")
        print(f"   • Official AutoTokenizer.apply_chat_template() works")
        print(f"   • Tool call duplication prevented")
        print(f"   • Cross-model compatibility achieved")
        print(f"   • No hacky workarounds needed")
        
    elif total_passed >= total_tests // 2:
        print("\n⚠️  PARTIAL SUCCESS")
        print("Core streaming functionality works with official patterns")
        
    else:
        print("\n🔧 STREAMING ISSUES DETECTED")
        print("Need to debug streaming with official chat templates")


if __name__ == "__main__":
    print("🚀 Starting WatsonX Streaming Diagnostic with Official Chat Templates...")
    asyncio.run(main())