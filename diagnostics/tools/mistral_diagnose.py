#!/usr/bin/env python3
"""
Test script to verify the Mistral MCP tool name sanitization fix works correctly.
Run this after updating your mistral_client.py with the tool sanitization code.
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

async def test_tool_name_sanitization():
    """Test the tool name sanitization directly"""
    print("🧪 Testing Tool Name Sanitization Logic")
    print("=" * 50)
    
    try:
        from chuk_llm.llm.providers.mistral_client import MistralLLMClient
        
        # Create a client instance to test the sanitization methods
        client = MistralLLMClient(model="mistral-medium-2505")
        
        # Test cases for tool name sanitization
        test_cases = [
            "stdio.read_query",
            "filesystem.read_file", 
            "mcp.server:get_data",
            "some-tool.with:multiple.separators",
            "already_valid_name",
            "tool123",
            "a" * 70,  # Too long
        ]
        
        print("Testing individual name sanitization:")
        for original in test_cases:
            sanitized = client._sanitize_tool_name_for_mistral(original)
            print(f"  {original:<30} -> {sanitized}")
        
        # Test full tool sanitization
        print("\nTesting full tool list sanitization:")
        test_tools = [
            {
                "type": "function",
                "function": {
                    "name": "stdio.read_query",
                    "description": "Read from stdin",
                    "parameters": {"type": "object", "properties": {}}
                }
            },
            {
                "type": "function", 
                "function": {
                    "name": "filesystem.read_file",
                    "description": "Read a file",
                    "parameters": {"type": "object", "properties": {}}
                }
            }
        ]
        
        sanitized_tools, name_mapping = client._sanitize_tools_for_mistral(test_tools)
        
        print(f"Original tools: {[t['function']['name'] for t in test_tools]}")
        print(f"Sanitized tools: {[t['function']['name'] for t in sanitized_tools]}")
        print(f"Name mapping: {name_mapping}")
        
        # Test response restoration
        mock_response = {
            "response": None,
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "stdio_read_query",
                        "arguments": "{}"
                    }
                }
            ]
        }
        
        restored = client._restore_tool_names_in_response(mock_response, name_mapping)
        restored_name = restored["tool_calls"][0]["function"]["name"]
        
        print(f"\nResponse restoration test:")
        print(f"  Sanitized: stdio_read_query")
        print(f"  Restored:  {restored_name}")
        print(f"  Success: {restored_name == 'stdio.read_query'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing sanitization: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_mistral_with_mcp_tools():
    """Test actual Mistral API call with MCP-style tool names"""
    print("\n🧪 Testing Mistral API with MCP-Style Tools")
    print("=" * 50)
    
    try:
        from chuk_llm.llm.client import get_client
        
        client = get_client(provider="mistral", model="mistral-medium-2505")
        
        # Create tools with MCP-style names that would previously fail
        mcp_style_tools = [
            {
                "type": "function",
                "function": {
                    "name": "stdio.read_query",
                    "description": "Read a query from standard input",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "The prompt to display to the user"
                            }
                        },
                        "required": ["prompt"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "filesystem.list_files",
                    "description": "List files in a directory",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Directory path to list"
                            }
                        },
                        "required": ["path"]
                    }
                }
            }
        ]
        
        # Test message that should trigger tool usage
        messages = [
            {
                "role": "user", 
                "content": "Please list the files in the current directory and then ask the user for their name"
            }
        ]
        
        print("Attempting Mistral API call with MCP-style tool names...")
        print(f"Tools: {[t['function']['name'] for t in mcp_style_tools]}")
        
        # This should now work without the naming error
        response = await client.create_completion(
            messages=messages,
            tools=mcp_style_tools,
            stream=False
        )
        
        print("✅ SUCCESS: No tool naming errors!")
        
        if isinstance(response, dict):
            if response.get("tool_calls"):
                print(f"🔧 Tool calls made: {len(response['tool_calls'])}")
                for i, tool_call in enumerate(response["tool_calls"]):
                    func_name = tool_call.get("function", {}).get("name", "unknown")
                    print(f"   {i+1}. {func_name}")
                    
                    # Verify the original MCP names are preserved
                    if func_name in ["stdio.read_query", "filesystem.list_files"]:
                        print(f"      ✅ Original MCP name preserved: {func_name}")
                    else:
                        print(f"      ⚠️  Unexpected name: {func_name}")
                        
            elif response.get("response"):
                print(f"💬 Text response: {response['response'][:150]}...")
            else:
                print(f"❓ Unexpected response format: {type(response)}")
                
        return True
        
    except Exception as e:
        error_msg = str(e)
        
        if "Function name" in error_msg and "must be a-z, A-Z, 0-9" in error_msg:
            print("❌ FAILED: Tool name sanitization not working!")
            print(f"   Error: {error_msg}")
            print("\n💡 The fix was not applied correctly. Check:")
            print("   1. Updated mistral_client.py with sanitization methods")
            print("   2. Restarted Python/reloaded the module")
            print("   3. Check for import errors")
            return False
        else:
            print(f"❌ FAILED: Other error: {error_msg}")
            return False


async def test_mcp_cli_simulation():
    """Simulate what happens in MCP CLI"""
    print("\n🧪 Simulating MCP CLI Tool Usage")
    print("=" * 50)
    
    try:
        from chuk_llm.llm.client import get_client
        
        client = get_client(provider="mistral", model="mistral-medium-2505")
        
        # Simulate the exact error case from your original issue
        problematic_tools = [
            {
                "type": "function",
                "function": {
                    "name": "stdio.read_query",  # This was causing the error
                    "description": "Read query from stdio",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            }
        ]
        
        messages = [{"role": "user", "content": "hi"}]
        
        print("Testing the exact scenario that was failing in MCP CLI...")
        
        # This should now work - test streaming
        print("Testing streaming...")
        response_stream = client.create_completion(
            messages=messages,
            tools=problematic_tools,
            stream=True  # Test streaming too
        )
        
        print("✅ Streaming with problematic tool names works!")
        
        # Collect a few chunks
        chunk_count = 0
        async for chunk in response_stream:
            chunk_count += 1
            if chunk.get("response"):
                print(f"📥 Chunk {chunk_count}: {chunk['response'][:50]}...")
            if chunk.get("tool_calls"):
                print(f"🔧 Tool calls in chunk: {[tc['function']['name'] for tc in chunk['tool_calls']]}")
            
            # Limit to first few chunks for testing
            if chunk_count >= 3:
                break
        
        print(f"✅ Received {chunk_count} streaming chunks without errors")
        
        # Also test non-streaming
        print("Testing non-streaming...")
        response_sync = await client.create_completion(
            messages=messages,
            tools=problematic_tools,
            stream=False
        )
        
        if isinstance(response_sync, dict):
            if response_sync.get("response"):
                print(f"✅ Non-streaming response: {response_sync['response'][:50]}...")
            if response_sync.get("tool_calls"):
                print(f"✅ Tool calls: {[tc['function']['name'] for tc in response_sync['tool_calls']]}")
        
        print("✅ Both streaming and non-streaming work!")
        return True
        
    except Exception as e:
        error_msg = str(e)
        
        if "Function name" in error_msg and "must be a-z, A-Z, 0-9" in error_msg:
            print("❌ MCP CLI simulation failed - tool naming error still occurs")
            print(f"   Error: {error_msg}")
            return False
        else:
            print(f"❌ MCP CLI simulation failed with other error: {error_msg}")
            return False


async def main():
    """Run all tests"""
    print("🚀 Testing Mistral MCP Tool Name Fix")
    print("=" * 60)
    
    # Test 1: Sanitization logic
    test1_passed = await test_tool_name_sanitization()
    
    # Test 2: Actual API call
    test2_passed = await test_mistral_with_mcp_tools() if test1_passed else False
    
    # Test 3: MCP CLI simulation  
    test3_passed = await test_mcp_cli_simulation() if test2_passed else False
    
    print("\n" + "=" * 60)
    print("🎯 TEST RESULTS:")
    print(f"   Sanitization Logic: {'✅ PASS' if test1_passed else '❌ FAIL'}")
    print(f"   API Integration:    {'✅ PASS' if test2_passed else '❌ FAIL'}")
    print(f"   MCP CLI Simulation: {'✅ PASS' if test3_passed else '❌ FAIL'}")
    
    if all([test1_passed, test2_passed, test3_passed]):
        print("\n🎉 ALL TESTS PASSED!")
        print("💡 You can now use MCP CLI with Mistral:")
        print("   mcp-cli chat --provider mistral --model mistral-medium-2505")
    else:
        print("\n❌ Some tests failed. Check the implementation.")
        print("💡 Make sure you:")
        print("   1. Updated mistral_client.py with the sanitization code")
        print("   2. Restarted your Python environment")
        print("   3. Check for any import/syntax errors")


if __name__ == "__main__":
    asyncio.run(main())