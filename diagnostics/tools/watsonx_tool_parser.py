#!/usr/bin/env python3
"""
Test script to verify the enhanced Granite parser works with your exact outputs
"""
import json
import re
import uuid
import ast

def _parse_watsonx_tool_formats(text: str):
    """Enhanced parser for Granite tool formats"""
    if not text or not isinstance(text, str):
        return []
    
    tool_calls = []
    
    try:
        # Format 1: Granite <tool_call>[...] format (YOUR ISSUE)
        tool_call_pattern = r'<tool_call>\s*(\[.*?\])\s*(?:</tool_call>|$)'
        tool_call_matches = re.findall(tool_call_pattern, text, re.DOTALL)
        
        for match in tool_call_matches:
            try:
                parsed_array = json.loads(match)
                if isinstance(parsed_array, list):
                    for item in parsed_array:
                        if isinstance(item, dict) and 'name' in item and 'arguments' in item:
                            func_name = item['name']
                            func_args = item['arguments']
                            
                            if isinstance(func_args, dict):
                                args_json = json.dumps(func_args)
                            elif isinstance(func_args, str):
                                args_json = func_args
                            else:
                                args_json = json.dumps(func_args)
                            
                            tool_calls.append({
                                "id": f"call_{uuid.uuid4().hex[:8]}",
                                "type": "function",
                                "function": {
                                    "name": func_name,
                                    "arguments": args_json
                                }
                            })
                            print(f"      ✅ Parsed <tool_call> format: {func_name}")
            except (json.JSONDecodeError, ValueError) as e:
                print(f"      ⚠️  Failed to parse <tool_call> format: {e}")
                continue
        
        # Format 2: Granite JSON function format (YOUR OTHER ISSUE)
        json_function_pattern = r'\{\s*"function"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{.*?\})\s*\}'
        json_function_matches = re.findall(json_function_pattern, text, re.DOTALL)
        
        for func_name, args_str in json_function_matches:
            try:
                args = json.loads(args_str)
                tool_calls.append({
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": func_name,
                        "arguments": json.dumps(args)
                    }
                })
                print(f"      ✅ Parsed JSON function format: {func_name}")
            except json.JSONDecodeError:
                continue
        
        # Format 3: Standard patterns (existing functionality)
        granite_pattern = r"'name':\s*'([^']+)',\s*'arguments':\s*(\{[^}]*\})"
        granite_matches = re.findall(granite_pattern, text)
        for func_name, args_str in granite_matches:
            try:
                args_json = args_str.replace("'", '"')
                args = json.loads(args_json)
                tool_calls.append({
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": func_name,
                        "arguments": json.dumps(args)
                    }
                })
            except (json.JSONDecodeError, ValueError):
                try:
                    args = ast.literal_eval(args_str)
                    tool_calls.append({
                        "id": f"call_{uuid.uuid4().hex[:8]}",
                        "type": "function",
                        "function": {
                            "name": func_name,
                            "arguments": json.dumps(args)
                        }
                    })
                except:
                    continue
    
    except Exception as e:
        print(f"      ⚠️  Error parsing: {e}")
    
    return tool_calls


def test_granite_parser():
    """Test with your actual Granite outputs"""
    print("🧪 TESTING ENHANCED GRANITE PARSER WITH YOUR ACTUAL OUTPUTS")
    print("=" * 65)
    
    # Your exact outputs from the test
    test_cases = [
        {
            "name": "Granite <tool_call> format (from your test output)",
            "text": '<tool_call>[{"arguments": {"table_name": "users"}, "name": "stdio_describe_table"}]',
            "expected_tool": "stdio_describe_table",
            "expected_param": "users"
        },
        {
            "name": "Granite JSON function format (from your test output)",  
            "text": '{\n    "function": "stdio_describe_table",\n    "arguments": {\n        "table_name": "products"\n    }\n}',
            "expected_tool": "stdio_describe_table", 
            "expected_param": "products"
        },
        {
            "name": "WatsonX capabilities search (from your test output)",
            "text": '<tool_call>[{"arguments": {"query": "WatsonX capabilities", "category": "technology"}, "name": "search"}]',
            "expected_tool": "search",
            "expected_param": "WatsonX capabilities"
        },
        {
            "name": "Mixed format with surrounding text",
            "text": 'I need to call the database. <tool_call>[{"arguments": {"table_name": "orders"}, "name": "stdio_describe_table"}]</tool_call> This will help.',
            "expected_tool": "stdio_describe_table",
            "expected_param": "orders"
        }
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"\n🔍 Test {i+1}: {test_case['name']}")
        print(f"   Text: {test_case['text'][:80]}...")
        
        result = _parse_watsonx_tool_formats(test_case['text'])
        
        if result:
            print(f"   ✅ SUCCESS: Found {len(result)} tool calls")
            
            for j, tc in enumerate(result):
                func_name = tc['function']['name']
                func_args = tc['function']['arguments']
                
                print(f"      Tool {j+1}: {func_name}")
                print(f"      Arguments: {func_args}")
                
                # Validate expected results
                if func_name == test_case['expected_tool']:
                    print(f"         ✅ Tool name matches expected: {func_name}")
                else:
                    print(f"         ⚠️  Tool name mismatch: got {func_name}, expected {test_case['expected_tool']}")
                
                # Check parameter
                try:
                    parsed_args = json.loads(func_args)
                    found_expected = False
                    for key, value in parsed_args.items():
                        if test_case['expected_param'] in str(value):
                            print(f"         ✅ Expected parameter found: {key}='{value}'")
                            found_expected = True
                            break
                    
                    if not found_expected:
                        print(f"         ⚠️  Expected parameter '{test_case['expected_param']}' not found")
                        
                except json.JSONDecodeError:
                    print(f"         ⚠️  Could not parse arguments as JSON")
        else:
            print(f"   ❌ FAILED: No tool calls found")
    
    print(f"\n" + "=" * 65)
    print("🎯 SUMMARY:")
    print("If all tests show ✅ SUCCESS, then the enhanced parser will fix your Granite issues!")
    print("Apply this fix to chuk_llm/llm/providers/watsonx_client.py")


if __name__ == "__main__":
    test_granite_parser()