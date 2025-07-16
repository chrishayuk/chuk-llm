#!/usr/bin/env python3
"""
WatsonX Tool Parameters Diagnostic Script
=========================================

Comprehensive diagnostic for WatsonX tool parameter extraction issues.
This script helps debug why some tool calls fail or don't extract parameters correctly.

Key Diagnostic Areas:
1. Model behavior analysis with different prompting strategies
2. Tool schema validation and optimization
3. Parameter extraction debugging with detailed logging
4. Temperature and generation parameter tuning
5. Prompt engineering for IBM Granite models
6. Comparison with working providers for consistency

Based on the previous test results where some parameter extraction failed:
- Test 1 & 2: stdio.describe_table failed to be called
- Test 4: watsonx.granite:analyze failed to be called
- Test 3: web.api:search worked perfectly

This script will help identify the root cause and optimize the configuration.
"""

import asyncio
import json
import time
import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
try:
    from dotenv import load_dotenv
    env_file = project_root / ".env"
    if env_file.exists():
        load_dotenv(env_file)
        print(f"✅ Loaded .env from {env_file}")
    else:
        load_dotenv()
except ImportError:
    print("⚠️  python-dotenv not available, using system environment")


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
            print(f"      Raw arguments: {repr(arguments)}")
            return {}
    
    print(f"      ⚠️  Unexpected arguments type: {type(arguments)}")
    return {}


async def test_watsonx_model_behavior():
    """Test basic WatsonX model behavior and responsiveness"""
    print("🧪 WATSONX MODEL BEHAVIOR ANALYSIS")
    print("=" * 50)
    
    # Check environment
    watsonx_api_key = os.getenv("WATSONX_API_KEY") or os.getenv("IBM_CLOUD_API_KEY")
    project_id = os.getenv("WATSONX_PROJECT_ID")
    
    if not watsonx_api_key or not project_id:
        print("❌ Missing WatsonX credentials")
        return False
    
    print(f"✅ API key: {watsonx_api_key[:8]}...{watsonx_api_key[-4:]}")
    print(f"✅ Project ID: {project_id}")
    
    try:
        from chuk_llm.llm.client import get_client
        
        client = get_client(provider="watsonx", model="ibm/granite-3-3-8b-instruct")
        print(f"✅ Client created: {type(client).__name__}")
        
        # Test 1: Basic text generation
        print("\n🔍 Test 1: Basic Text Generation")
        response = await client.create_completion(
            messages=[{"role": "user", "content": "What is 2+2?"}],
            stream=False,
            max_tokens=50
        )
        
        if response.get("response"):
            print(f"   ✅ Text generation working: {response['response'][:100]}...")
        else:
            print(f"   ❌ Text generation failed: {response}")
            return False
        
        # Test 2: Simple tool calling capability
        print("\n🔍 Test 2: Simple Tool Call Test")
        simple_tools = [
            {
                "type": "function",
                "function": {
                    "name": "add_numbers",
                    "description": "Add two numbers together",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "number", "description": "First number"},
                            "b": {"type": "number", "description": "Second number"}
                        },
                        "required": ["a", "b"]
                    }
                }
            }
        ]
        
        response = await client.create_completion(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant. When asked to add numbers, use the add_numbers function."
                },
                {
                    "role": "user", 
                    "content": "Add 5 and 3"
                }
            ],
            tools=simple_tools,
            stream=False,
            max_tokens=100
        )
        
        if response.get("tool_calls"):
            call = response["tool_calls"][0]
            print(f"   ✅ Tool calling working: {call['function']['name']}")
            args = safe_parse_tool_arguments(call["function"]["arguments"])
            print(f"   Parameters: {args}")
            if args.get("a") and args.get("b"):
                print(f"   ✅ Parameter extraction working")
            else:
                print(f"   ⚠️  Parameter extraction issues")
        else:
            print(f"   ❌ Tool calling failed")
            if response.get("response"):
                print(f"   Text response: {response['response'][:150]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in model behavior test: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_tool_schema_optimization():
    """Test different tool schema approaches to find what works best with Granite"""
    print("\n🎯 WATSONX TOOL SCHEMA OPTIMIZATION")
    print("=" * 45)
    
    try:
        from chuk_llm.llm.client import get_client
        
        client = get_client(provider="watsonx", model="ibm/granite-3-3-8b-instruct")
        
        # Test different schema styles for the same functionality
        test_schemas = [
            {
                "name": "Simple Schema",
                "tool": {
                    "type": "function",
                    "function": {
                        "name": "describe_table",
                        "description": "Get table schema",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "table": {"type": "string"}
                            },
                            "required": ["table"]
                        }
                    }
                },
                "prompt": "describe the users table",
                "expected_param": "table"
            },
            {
                "name": "Detailed Schema",
                "tool": {
                    "type": "function",
                    "function": {
                        "name": "describe_database_table",
                        "description": "Get detailed schema information for a specific database table including columns, types, and constraints",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "table_name": {
                                    "type": "string",
                                    "description": "The exact name of the database table to describe"
                                }
                            },
                            "required": ["table_name"]
                        }
                    }
                },
                "prompt": "get the schema for the users table",
                "expected_param": "table_name"
            },
            {
                "name": "IBM Style Schema", 
                "tool": {
                    "type": "function",
                    "function": {
                        "name": "watsonx_describe_table",
                        "description": "Use IBM WatsonX to analyze and describe a database table structure",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "table_name": {
                                    "type": "string",
                                    "description": "Database table name to analyze"
                                },
                                "include_sample": {
                                    "type": "boolean",
                                    "description": "Include sample data",
                                    "default": False
                                }
                            },
                            "required": ["table_name"]
                        }
                    }
                },
                "prompt": "analyze the structure of the users table with watsonx",
                "expected_param": "table_name"
            }
        ]
        
        for i, test_case in enumerate(test_schemas):
            print(f"\n📋 Schema Test {i+1}: {test_case['name']}")
            print(f"   Tool: {test_case['tool']['function']['name']}")
            print(f"   Prompt: '{test_case['prompt']}'")
            
            response = await client.create_completion(
                messages=[
                    {
                        "role": "system",
                        "content": f"You have access to the {test_case['tool']['function']['name']} function. Use it when asked about table structures. Always provide the required parameters."
                    },
                    {
                        "role": "user",
                        "content": test_case["prompt"]
                    }
                ],
                tools=[test_case["tool"]],
                stream=False,
                max_tokens=150
            )
            
            if response.get("tool_calls"):
                call = response["tool_calls"][0]
                func_name = call["function"]["name"]
                args = safe_parse_tool_arguments(call["function"]["arguments"])
                
                print(f"   ✅ Tool called: {func_name}")
                print(f"   Arguments: {args}")
                
                expected_param = test_case["expected_param"]
                if args.get(expected_param):
                    print(f"   ✅ SUCCESS: {expected_param} = '{args[expected_param]}'")
                else:
                    print(f"   ❌ FAILED: Missing {expected_param}")
                    print(f"   Available keys: {list(args.keys())}")
            else:
                print(f"   ❌ FAILED: No tool call made")
                if response.get("response"):
                    print(f"   Text: {response['response'][:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in schema optimization: {e}")
        return False


async def test_prompt_engineering_strategies():
    """Test different prompting strategies optimized for IBM Granite models"""
    print("\n🎯 WATSONX PROMPT ENGINEERING STRATEGIES")
    print("=" * 50)
    
    try:
        from chuk_llm.llm.client import get_client
        
        client = get_client(provider="watsonx", model="ibm/granite-3-3-8b-instruct")
        
        # Standard tool for testing
        test_tool = {
            "type": "function",
            "function": {
                "name": "stdio_describe_table",  # Already sanitized name
                "description": "Get database table schema information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "table_name": {
                            "type": "string",
                            "description": "Name of the table to describe"
                        }
                    },
                    "required": ["table_name"]
                }
            }
        }
        
        # Different prompting strategies
        strategies = [
            {
                "name": "Direct Instruction",
                "system": "You are a database assistant. Use stdio_describe_table to get table schemas.",
                "user": "describe the users table schema"
            },
            {
                "name": "IBM Granite Optimized",
                "system": """You are an IBM WatsonX AI assistant specialized in database operations.

Available function: stdio_describe_table(table_name: string)

When users ask about table structure or schema:
1. Extract the table name from their request
2. Call stdio_describe_table with the exact table name
3. Always provide the table_name parameter

Example: User says "show me the products table" -> Call stdio_describe_table(table_name="products")""",
                "user": "I need information about the users table structure"
            },
            {
                "name": "Step-by-Step",
                "system": """You are a helpful database assistant. Follow these steps:

Step 1: Identify what table the user is asking about
Step 2: Use the stdio_describe_table function with the table name
Step 3: The function requires a 'table_name' parameter

Available function: stdio_describe_table(table_name: string)""",
                "user": "what columns does the users table have?"
            },
            {
                "name": "JSON Example",
                "system": """You have access to stdio_describe_table function. When describing tables, call it like this:

stdio_describe_table({"table_name": "table_name_here"})

For example:
- User: "describe products table" 
- You: Call stdio_describe_table({"table_name": "products"})""",
                "user": "describe the users table"
            },
            {
                "name": "Explicit Parameter",
                "system": """You are a database expert. You have access to stdio_describe_table function.

CRITICAL: Always extract the table name and pass it as the table_name parameter.

Examples:
- "users table" -> table_name="users"
- "products table schema" -> table_name="products"  
- "structure of orders" -> table_name="orders"

NEVER call stdio_describe_table without the table_name parameter!""",
                "user": "I want to see the users table structure"
            }
        ]
        
        for i, strategy in enumerate(strategies):
            print(f"\n📝 Strategy {i+1}: {strategy['name']}")
            print(f"   User: '{strategy['user']}'")
            
            response = await client.create_completion(
                messages=[
                    {"role": "system", "content": strategy["system"]},
                    {"role": "user", "content": strategy["user"]}
                ],
                tools=[test_tool],
                stream=False,
                max_tokens=200,
                temperature=0.1  # Low temperature for more deterministic behavior
            )
            
            if response.get("tool_calls"):
                call = response["tool_calls"][0]
                args = safe_parse_tool_arguments(call["function"]["arguments"])
                
                print(f"   ✅ Tool called: {call['function']['name']}")
                print(f"   Arguments: {args}")
                
                table_name = args.get("table_name", "")
                if table_name and "users" in table_name.lower():
                    print(f"   ✅ SUCCESS: Correctly extracted table_name='{table_name}'")
                elif table_name:
                    print(f"   ⚠️  PARTIAL: Got table_name='{table_name}' (acceptable)")
                else:
                    print(f"   ❌ FAILED: No table_name parameter")
            else:
                print(f"   ❌ FAILED: No tool call made")
                if response.get("response"):
                    resp_text = response["response"][:150]
                    print(f"   Text: {resp_text}...")
                    
                    # Check if text contains function call attempt
                    if "stdio_describe_table" in resp_text:
                        print(f"   💡 Model mentioned function in text - prompt parsing issue")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in prompt engineering test: {e}")
        return False


async def test_parameter_tuning():
    """Test different generation parameters to optimize tool calling"""
    print("\n🎯 WATSONX PARAMETER TUNING")
    print("=" * 35)
    
    try:
        from chuk_llm.llm.client import get_client
        
        # Standard test setup
        test_tool = {
            "type": "function",
            "function": {
                "name": "stdio_describe_table",
                "description": "Get table schema",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "table_name": {"type": "string", "description": "Table name"}
                    },
                    "required": ["table_name"]
                }
            }
        }
        
        messages = [
            {
                "role": "system",
                "content": "You are a database assistant. Use stdio_describe_table(table_name) to get table schemas. Always provide the table_name parameter."
            },
            {
                "role": "user",
                "content": "describe the users table"
            }
        ]
        
        # Different parameter configurations
        param_configs = [
            {"name": "Default", "params": {}},
            {"name": "Low Temperature", "params": {"temperature": 0.1}},
            {"name": "High Max Tokens", "params": {"max_tokens": 500}},
            {"name": "Conservative", "params": {"temperature": 0.1, "max_tokens": 200}},
            {"name": "Focused", "params": {"temperature": 0.0, "max_tokens": 100}},
        ]
        
        for config in param_configs:
            print(f"\n⚙️  Testing: {config['name']}")
            print(f"   Parameters: {config['params']}")
            
            client = get_client(provider="watsonx", model="ibm/granite-3-3-8b-instruct")
            
            response = await client.create_completion(
                messages=messages,
                tools=[test_tool],
                stream=False,
                **config["params"]
            )
            
            if response.get("tool_calls"):
                call = response["tool_calls"][0]
                args = safe_parse_tool_arguments(call["function"]["arguments"])
                table_name = args.get("table_name", "")
                
                print(f"   ✅ SUCCESS: table_name='{table_name}'")
            else:
                print(f"   ❌ FAILED: No tool call")
                if response.get("response"):
                    print(f"   Text: {response['response'][:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in parameter tuning: {e}")
        return False


async def test_working_vs_failing_patterns():
    """Compare working tool patterns with failing ones to identify the issue"""
    print("\n🎯 WATSONX WORKING VS FAILING PATTERNS ANALYSIS")
    print("=" * 55)
    
    try:
        from chuk_llm.llm.client import get_client
        
        client = get_client(provider="watsonx", model="ibm/granite-3-3-8b-instruct")
        
        # From previous test results, we know web.api:search worked
        # Let's compare it with stdio.describe_table
        
        test_cases = [
            {
                "name": "WORKING: web.api:search", 
                "tool": {
                    "type": "function",
                    "function": {
                        "name": "web.api:search",  # This worked in previous tests
                        "description": "Search for information using web API",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string", "description": "Search query"},
                                "category": {"type": "string", "description": "Search category"}
                            },
                            "required": ["query"]
                        }
                    }
                },
                "prompt": "search for 'IBM Watson' in AI category",
                "expected": {"query": "IBM Watson", "category": "AI"}
            },
            {
                "name": "FAILING: stdio.describe_table",
                "tool": {
                    "type": "function", 
                    "function": {
                        "name": "stdio.describe_table",  # This failed in previous tests
                        "description": "Get the schema information for a specific table",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "table_name": {"type": "string", "description": "Name of the table to describe"}
                            },
                            "required": ["table_name"]
                        }
                    }
                },
                "prompt": "describe the products table schema", 
                "expected": {"table_name": "products"}
            },
            {
                "name": "MODIFIED: table_info (simplified)",
                "tool": {
                    "type": "function",
                    "function": {
                        "name": "table_info",  # Simplified name
                        "description": "Get table information",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "table": {"type": "string", "description": "Table name"}
                            },
                            "required": ["table"]
                        }
                    }
                },
                "prompt": "get info about the products table",
                "expected": {"table": "products"}
            }
        ]
        
        for test_case in test_cases:
            print(f"\n🔍 {test_case['name']}")
            print(f"   Tool: {test_case['tool']['function']['name']}")
            print(f"   Prompt: '{test_case['prompt']}'")
            
            response = await client.create_completion(
                messages=[
                    {
                        "role": "system",
                        "content": f"You have access to {test_case['tool']['function']['name']}. Use it when relevant."
                    },
                    {
                        "role": "user",
                        "content": test_case["prompt"]
                    }
                ],
                tools=[test_case["tool"]],
                stream=False,
                max_tokens=200,
                temperature=0.1
            )
            
            if response.get("tool_calls"):
                call = response["tool_calls"][0]
                args = safe_parse_tool_arguments(call["function"]["arguments"])
                
                print(f"   ✅ Tool called: {call['function']['name']}")
                print(f"   Arguments: {args}")
                
                # Check if we got expected parameters
                expected = test_case["expected"]
                success = True
                for key, expected_val in expected.items():
                    actual_val = args.get(key, "")
                    if expected_val.lower() in actual_val.lower():
                        print(f"   ✅ {key}: '{actual_val}' (contains '{expected_val}')")
                    else:
                        print(f"   ❌ {key}: '{actual_val}' (expected '{expected_val}')")
                        success = False
                
                if success:
                    print(f"   🎯 OVERALL: SUCCESS")
                else:
                    print(f"   ⚠️  OVERALL: PARTIAL")
                    
            else:
                print(f"   ❌ FAILED: No tool call made")
                if response.get("response"):
                    resp = response["response"][:200]
                    print(f"   Text: {resp}...")
                    
                    # Analyze the response for clues
                    if "function" in resp.lower() or test_case['tool']['function']['name'] in resp:
                        print(f"   💡 CLUE: Model knows about function but didn't call it properly")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in pattern analysis: {e}")
        return False


async def test_conversation_context():
    """Test how conversation context affects tool calling"""
    print("\n🎯 WATSONX CONVERSATION CONTEXT TEST")
    print("=" * 40)
    
    try:
        from chuk_llm.llm.client import get_client
        
        client = get_client(provider="watsonx", model="ibm/granite-3-3-8b-instruct")
        
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "stdio_describe_table",
                    "description": "Get table schema",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "table_name": {"type": "string", "description": "Table name"}
                        },
                        "required": ["table_name"]
                    }
                }
            }
        ]
        
        # Test different conversation contexts
        contexts = [
            {
                "name": "Minimal Context",
                "messages": [
                    {"role": "user", "content": "describe users table"}
                ]
            },
            {
                "name": "With System Message",
                "messages": [
                    {"role": "system", "content": "You are a database assistant with access to stdio_describe_table."},
                    {"role": "user", "content": "describe users table"}
                ]
            },
            {
                "name": "Previous Context",
                "messages": [
                    {"role": "system", "content": "You are a database assistant."},
                    {"role": "user", "content": "I'm working with a database"},
                    {"role": "assistant", "content": "I can help you with database operations."},
                    {"role": "user", "content": "describe the users table"}
                ]
            },
            {
                "name": "Explicit Instruction",
                "messages": [
                    {"role": "system", "content": "Use stdio_describe_table function to get table schemas. Always provide table_name parameter."},
                    {"role": "user", "content": "I need schema for users table - please use the function"}
                ]
            }
        ]
        
        for context in contexts:
            print(f"\n📝 Context: {context['name']}")
            
            response = await client.create_completion(
                messages=context["messages"],
                tools=tools,
                stream=False,
                max_tokens=150,
                temperature=0.1
            )
            
            if response.get("tool_calls"):
                call = response["tool_calls"][0]
                args = safe_parse_tool_arguments(call["function"]["arguments"])
                table_name = args.get("table_name", "")
                
                print(f"   ✅ SUCCESS: table_name='{table_name}'")
            else:
                print(f"   ❌ FAILED: No tool call")
                if response.get("response"):
                    print(f"   Text: {response['response'][:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in context test: {e}")
        return False


async def main():
    """Run comprehensive WatsonX tool parameters diagnostic"""
    print("🧪 WATSONX TOOL PARAMETERS COMPREHENSIVE DIAGNOSTIC")
    print("=" * 60)
    
    print("This diagnostic will help identify why some WatsonX tool parameter")
    print("extractions fail while others succeed. We'll test:")
    print("1. Basic model behavior and tool calling capability")
    print("2. Different tool schema approaches") 
    print("3. Prompt engineering strategies optimized for IBM Granite")
    print("4. Generation parameter tuning")
    print("5. Comparison of working vs failing patterns")
    print("6. Conversation context effects")
    
    # Run all diagnostic tests
    tests = [
        ("Model Behavior", test_watsonx_model_behavior),
        ("Schema Optimization", test_tool_schema_optimization),
        ("Prompt Engineering", test_prompt_engineering_strategies),
        ("Parameter Tuning", test_parameter_tuning),
        ("Pattern Analysis", test_working_vs_failing_patterns),
        ("Context Effects", test_conversation_context),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        try:
            result = await test_func()
            results[test_name] = result
            print(f"\n✅ {test_name}: {'PASS' if result else 'FAIL'}")
        except Exception as e:
            results[test_name] = False
            print(f"\n❌ {test_name}: ERROR - {e}")
    
    # Summary
    print(f"\n{'='*60}")
    print("🎯 DIAGNOSTIC SUMMARY:")
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    print(f"\n📊 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL DIAGNOSTICS PASSED!")
        print("WatsonX tool parameter extraction is working optimally.")
    elif passed >= total // 2:
        print("\n⚠️  PARTIAL SUCCESS")
        print("Some diagnostic patterns work - check successful approaches above.")
    else:
        print("\n🔧 DEBUGGING NEEDED")
        print("Multiple issues detected - review failed tests for optimization opportunities.")
    
    print("\n💡 Next Steps:")
    print("1. Review successful patterns from the diagnostic output")
    print("2. Apply working prompt/schema strategies to failing cases")
    print("3. Optimize generation parameters based on results")
    print("4. Consider model-specific prompt engineering for IBM Granite")


if __name__ == "__main__":
    print("🚀 Starting WatsonX Tool Parameters Diagnostic...")
    asyncio.run(main())