#!/usr/bin/env python3
"""
Ollama Tool Streaming Diagnostic
===============================

Tests tool calling and streaming behavior across different Ollama model types.
Equivalent to the OpenAI tool streaming diagnostic but for local Ollama models.
"""

import asyncio
import json
import sys
import time
from pathlib import Path

# Load environment
try:
    from dotenv import load_dotenv

    project_root = Path(__file__).parent.parent
    env_file = project_root / ".env"
    if env_file.exists():
        load_dotenv(env_file)
        print("✅ Loaded .env")
    else:
        load_dotenv()
except ImportError:
    print("⚠️ No dotenv")


def classify_ollama_tool_support(model_name):
    """Classify Ollama models by tool calling capabilities"""
    model_lower = model_name.lower()

    # Models known to support tools well
    excellent_tool_models = ["qwen3", "granite3.3", "gpt-oss", "mistral"]
    good_tool_models = ["llama3.1", "llama3.2", "llama3"]

    # Models that don't support tools
    no_tool_models = ["embed", "embedding", "vision", "llava"]

    if any(pattern in model_lower for pattern in no_tool_models):
        return {
            "supports_tools": False,
            "tool_quality": "none",
            "reason": "embedding/vision model",
        }
    elif any(pattern in model_lower for pattern in excellent_tool_models):
        return {
            "supports_tools": True,
            "tool_quality": "excellent",
            "reason": "known excellent tool model",
        }
    elif any(pattern in model_lower for pattern in good_tool_models):
        return {
            "supports_tools": True,
            "tool_quality": "good",
            "reason": "capable tool model",
        }
    else:
        return {
            "supports_tools": True,  # Assume most models support tools
            "tool_quality": "unknown",
            "reason": "general capability assumed",
        }


async def test_ollama_tool_streaming():
    """Test tool streaming behavior across different Ollama model types."""

    print("🔍 OLLAMA TOOL STREAMING ANALYSIS")
    print("=" * 50)

    # Discover available models
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from chuk_llm.llm.discovery.ollama_discoverer import OllamaModelDiscoverer

        discoverer = OllamaModelDiscoverer()
        models = await discoverer.discover_models()

        if not models:
            print("❌ No Ollama models found")
            return False

        print(f"✅ Found {len(models)} Ollama models")

        # Filter for tool-capable models
        tool_capable_models = []
        for model_data in models:
            model_name = model_data.get("name", "")
            tool_info = classify_ollama_tool_support(model_name)

            if tool_info["supports_tools"]:
                tool_capable_models.append((model_name, tool_info))

        print(f"🔧 Selected {len(tool_capable_models)} tool-capable models for testing")

        # Select representative models for testing
        test_models = []
        quality_groups = {"excellent": [], "good": [], "unknown": []}

        for model_name, tool_info in tool_capable_models:
            quality = tool_info["tool_quality"]
            quality_groups[quality].append((model_name, tool_info))

        # Take up to 2 from each quality group
        for quality, models in quality_groups.items():
            test_models.extend(models[:2])

        # Limit total tests
        test_models = test_models[:6]

        print(f"\n🎯 Testing {len(test_models)} representative tool-capable models:")
        for model_name, tool_info in test_models:
            print(
                f"   • {model_name} [{tool_info['tool_quality']}] - {tool_info['reason']}"
            )

    except Exception as e:
        print(f"❌ Model discovery failed: {e}")
        # Fallback to known tool-capable models
        test_models = [
            ("qwen3:latest", classify_ollama_tool_support("qwen3:latest")),
            ("granite3.3:latest", classify_ollama_tool_support("granite3.3:latest")),
            ("gpt-oss:latest", classify_ollama_tool_support("gpt-oss:latest")),
            ("llama3.1:latest", classify_ollama_tool_support("llama3.1:latest")),
            ("mistral:latest", classify_ollama_tool_support("mistral:latest")),
        ]
        print(f"🔄 Using fallback model list: {len(test_models)} models")

    # Define tool test cases
    tool_test_cases = [
        {
            "name": "Simple Tool Call",
            "prompt": "Execute SQL: SELECT * FROM users LIMIT 5",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "execute_sql",
                        "description": "Execute a SQL query",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "SQL query to execute",
                                },
                                "database": {
                                    "type": "string",
                                    "description": "Database name",
                                    "default": "main",
                                },
                            },
                            "required": ["query"],
                        },
                    },
                }
            ],
            "expected_function": "execute_sql",
            "min_chunks": 1,
        },
        {
            "name": "Math Tool Call",
            "prompt": "Calculate the square root of 144 using a math function",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "calculate_math",
                        "description": "Perform mathematical calculations",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "operation": {
                                    "type": "string",
                                    "description": "Math operation to perform",
                                },
                                "numbers": {
                                    "type": "array",
                                    "items": {"type": "number"},
                                    "description": "Numbers to operate on",
                                },
                            },
                            "required": ["operation", "numbers"],
                        },
                    },
                }
            ],
            "expected_function": "calculate_math",
            "min_chunks": 1,
        },
        {
            "name": "File Tool Call",
            "prompt": "Read the contents of config.json file",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "description": "Read contents of a file",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "filename": {
                                    "type": "string",
                                    "description": "Name of file to read",
                                },
                                "encoding": {
                                    "type": "string",
                                    "description": "File encoding",
                                    "default": "utf-8",
                                },
                            },
                            "required": ["filename"],
                        },
                    },
                }
            ],
            "expected_function": "read_file",
            "min_chunks": 1,
        },
        {
            "name": "Complex Multi-Tool Scenario",
            "prompt": "First read data.csv, then calculate the average of the numbers, and save result to output.txt",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "description": "Read contents of a file",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "filename": {
                                    "type": "string",
                                    "description": "Name of file to read",
                                }
                            },
                            "required": ["filename"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "calculate_average",
                        "description": "Calculate average of numbers",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "numbers": {
                                    "type": "array",
                                    "items": {"type": "number"},
                                }
                            },
                            "required": ["numbers"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "write_file",
                        "description": "Write content to a file",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "filename": {"type": "string"},
                                "content": {"type": "string"},
                            },
                            "required": ["filename", "content"],
                        },
                    },
                },
            ],
            "expected_function": "read_file",  # First expected function
            "min_chunks": 5,
        },
    ]

    results = {}
    overall_success = True

    for model_name, tool_info in test_models:
        print(f"\n🎯 Testing {model_name} (Tool Quality: {tool_info['tool_quality']})")
        print("-" * 60)

        model_results = {}
        model_success = True

        for test_case in tool_test_cases:
            case_name = test_case["name"]

            # Skip complex tests for unknown quality models to save time
            if tool_info["tool_quality"] == "unknown" and "Complex" in case_name:
                print(f"  ⏭️ Skipping {case_name} (unknown tool quality)")
                continue

            print(f"  🧪 {case_name}...")

            # Test with ChukLLM
            chuk_result = await test_chuk_llm_ollama_tool_streaming(
                model_name, test_case["prompt"], test_case["tools"]
            )

            # Test with raw Ollama API for comparison
            raw_result = await test_raw_ollama_tool_streaming(
                model_name, test_case["prompt"], test_case["tools"]
            )

            model_results[case_name] = {
                "chuk": chuk_result,
                "raw": raw_result,
                "expected_function": test_case["expected_function"],
                "min_chunks_expected": test_case["min_chunks"],
            }

            # Analyze results
            case_success = analyze_ollama_tool_result(
                chuk_result, raw_result, case_name, test_case, tool_info
            )

            if not case_success:
                model_success = False

        results[model_name] = {
            "results": model_results,
            "success": model_success,
            "tool_info": tool_info,
        }

        if not model_success:
            overall_success = False

    # Overall analysis
    print("\n" + "=" * 60)
    print("📊 COMPREHENSIVE OLLAMA TOOL STREAMING ANALYSIS")
    return analyze_all_ollama_tool_results(results, overall_success)


async def test_chuk_llm_ollama_tool_streaming(model_name, prompt, tools):
    """Test ChukLLM tool streaming with Ollama models"""
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
            "has_tool_call_info": False,
            "tool_call_formats": [],
            "response_time": 0,
        }

        # Use ChukLLM's streaming with tools
        chunk_count = 0
        response_parts = []
        detected_tool_calls = set()

        start_time = time.time()

        async for chunk in stream(
            prompt, provider="ollama", model=model_name, tools=tools, max_tokens=300
        ):
            chunk_count += 1
            if chunk:
                chunk_str = str(chunk)
                response_parts.append(chunk_str)

                # Enhanced tool call detection patterns for Ollama
                tool_patterns = {
                    "bracket_calling": "[Calling:",  # [Calling: function_name]
                    "bracket_tool": "[Tool:",  # [Tool: function_name]
                    "using_tool": "Using tool:",  # Using tool: function_name
                    "function_call": "Function call:",  # Function call: function_name
                    "executing": "Executing:",  # Executing: function_name
                    "json_name": '"name":',  # JSON with "name" field
                    "json_function": '"function":',  # JSON with "function" field
                    "json_arguments": '"arguments":',  # JSON with "arguments" field
                    "tool_use": "tool_use",  # General tool use indicator
                    "calling_function": "calling function",  # Calling function text
                    "invoke": "invoke",  # Invoke function
                }

                detected_patterns = []
                for pattern_name, pattern in tool_patterns.items():
                    if pattern.lower() in chunk_str.lower():
                        detected_patterns.append(pattern_name)
                        result["has_tool_call_info"] = True

                if detected_patterns:
                    result["tool_call_formats"].extend(detected_patterns)

                # Extract function names from various patterns
                chunk_lower = chunk_str.lower()

                # Method 1: Extract from bracket patterns
                for prefix in ["[calling:", "[tool:", "using tool:", "executing:"]:
                    if prefix in chunk_lower:
                        try:
                            start = chunk_lower.find(prefix) + len(prefix)
                            end = (
                                chunk_str.find("]", start)
                                if "]" in chunk_str[start:]
                                else start + 20
                            )
                            func_name = chunk_str[start:end].strip().replace("]", "")
                            if func_name:
                                detected_tool_calls.add(func_name)
                        except:
                            pass

                # Method 2: Look for specific function names from tools
                for tool in tools:
                    func_name = tool["function"]["name"]
                    if func_name.lower() in chunk_lower:
                        detected_tool_calls.add(func_name)

                # Method 3: JSON parsing attempt
                try:
                    if '{"' in chunk_str and '"name"' in chunk_str:
                        import re

                        name_match = re.search(r'"name":\s*"([^"]+)"', chunk_str)
                        if name_match:
                            detected_tool_calls.add(name_match.group(1))
                except:
                    pass

        end_time = time.time()

        # Convert detected tool calls to expected format
        result["tool_calls"] = [
            {"function": {"name": name}} for name in detected_tool_calls
        ]
        result["chunks"] = chunk_count
        result["final_response"] = "".join(response_parts)
        result["streaming_worked"] = chunk_count > 0
        result["success"] = True
        result["response_time"] = end_time - start_time

        return result

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "chunks": 0,
            "streaming_worked": False,
            "has_tool_call_info": False,
            "tool_calls": [],
            "tool_call_formats": [],
            "response_time": 0,
        }


async def test_raw_ollama_tool_streaming(model_name, prompt, tools):
    """Test raw Ollama API tool streaming for comparison"""
    try:
        import httpx

        result = {
            "success": False,
            "chunks": 0,
            "tool_calls": [],
            "final_response": "",
            "streaming_worked": False,
            "error": None,
            "response_time": 0,
        }

        # Prepare Ollama API request with tools
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "tools": tools,
            "stream": True,
            "options": {"num_predict": 300},
        }

        chunk_count = 0
        response_parts = []
        tool_calls_detected = []

        start_time = time.time()

        async with (
            httpx.AsyncClient(timeout=60.0) as client,
            client.stream(
                "POST", "http://localhost:11434/api/chat", json=payload
            ) as response,
        ):
            response.raise_for_status()

            async for line in response.aiter_lines():
                if line.strip():
                    try:
                        chunk_data = json.loads(line)

                        # Handle regular message content
                        if "message" in chunk_data:
                            message = chunk_data["message"]

                            if "content" in message and message["content"]:
                                chunk_count += 1
                                response_parts.append(message["content"])

                            # Handle tool calls in message
                            if "tool_calls" in message and message["tool_calls"]:
                                for tool_call in message["tool_calls"]:
                                    if "function" in tool_call:
                                        tool_calls_detected.append(
                                            {
                                                "function": {
                                                    "name": tool_call["function"].get(
                                                        "name", ""
                                                    ),
                                                    "arguments": tool_call[
                                                        "function"
                                                    ].get("arguments", ""),
                                                }
                                            }
                                        )
                    except json.JSONDecodeError:
                        continue

        end_time = time.time()

        result["chunks"] = chunk_count
        result["final_response"] = "".join(response_parts)
        result["tool_calls"] = tool_calls_detected
        result["streaming_worked"] = chunk_count > 0
        result["success"] = True
        result["response_time"] = end_time - start_time

        return result

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "chunks": 0,
            "streaming_worked": False,
            "response_time": 0,
        }


def analyze_ollama_tool_result(
    chuk_result, raw_result, test_name, test_case, tool_info
):
    """Analyze a single Ollama tool test case result"""
    print(f"    📋 {test_name} Results:")

    if not chuk_result["success"]:
        print(f"      ❌ ChukLLM failed: {chuk_result.get('error', 'Unknown')}")
        return False

    if not raw_result["success"]:
        print(f"      ❌ Raw Ollama failed: {raw_result.get('error', 'Unknown')}")
        return False

    # Analyze tool calling performance
    chuk_streamed = chuk_result["streaming_worked"]
    chuk_tool_count = len(chuk_result.get("tool_calls", []))
    raw_tool_count = len(raw_result.get("tool_calls", []))
    has_tool_info = chuk_result.get("has_tool_call_info", False)
    tool_formats = chuk_result.get("tool_call_formats", [])
    expected_function = test_case["expected_function"]
    min_chunks = test_case["min_chunks"]

    # Check if expected function was detected
    chuk_functions = [
        tc["function"]["name"] for tc in chuk_result.get("tool_calls", [])
    ]
    raw_functions = [tc["function"]["name"] for tc in raw_result.get("tool_calls", [])]
    expected_detected_chuk = expected_function in chuk_functions
    expected_detected_raw = expected_function in raw_functions

    # Determine success based on tool quality and results
    tool_quality = tool_info["tool_quality"]

    if tool_quality == "excellent":
        # High expectations for excellent models
        if (
            chuk_streamed
            and (chuk_tool_count > 0 or has_tool_info)
            and chuk_result["chunks"] >= min_chunks
        ):
            print(f"      ✅ Excellent tool streaming ({chuk_result['chunks']} chunks)")
            print(
                f"      🔧 ChukLLM: {chuk_tool_count} tool calls, patterns: {list(set(tool_formats))}"
            )
            if expected_detected_chuk:
                print(f"      ✨ Expected function '{expected_function}' detected!")
            success = True
        else:
            print("      ⚠️  Expected better performance from excellent model")
            success = False

    elif tool_quality == "good":
        # Moderate expectations for good models
        if chuk_streamed and (
            chuk_tool_count > 0 or has_tool_info or chuk_result["chunks"] >= min_chunks
        ):
            print(f"      ✅ Good tool streaming ({chuk_result['chunks']} chunks)")
            print(f"      🔧 ChukLLM: {chuk_tool_count} tool calls detected")
            success = True
        else:
            print("      ⚠️  Tool streaming below expectations")
            success = False

    else:  # unknown quality
        # Lower expectations for unknown models
        if chuk_streamed and chuk_result["chunks"] >= 1:
            print(f"      ✅ Basic streaming works ({chuk_result['chunks']} chunks)")
            if has_tool_info or chuk_tool_count > 0:
                print("      🎉 Bonus: Tool patterns detected!")
            success = True
        else:
            print("      ❌ No streaming detected")
            success = False

    # Show comparison with raw Ollama
    print(
        f"      📊 Raw Ollama: {raw_result['chunks']} chunks, {raw_tool_count} tool calls"
    )
    if expected_detected_raw:
        print(f"      📊 Raw Ollama detected '{expected_function}'")

    # Show response times
    chuk_time = chuk_result.get("response_time", 0)
    raw_time = raw_result.get("response_time", 0)
    print(f"      ⏱️  Response times - ChukLLM: {chuk_time:.2f}s, Raw: {raw_time:.2f}s")

    # Show response preview
    chuk_preview = (
        chuk_result["final_response"][:150] if chuk_result["final_response"] else ""
    )
    if chuk_preview:
        print(f"      📝 Response: {chuk_preview}...")

    return success


def analyze_all_ollama_tool_results(results, overall_success):
    """Analyze all Ollama tool test results"""
    print("\n🎯 COMPREHENSIVE OLLAMA TOOL ANALYSIS:")

    total_tests = 0
    successful_tests = 0
    streaming_worked_count = 0
    tool_calls_detected = 0

    # Results by tool quality
    quality_performance = {}

    for model_name, model_data in results.items():
        model_results = model_data["results"]
        model_success = model_data["success"]
        tool_info = model_data["tool_info"]
        tool_quality = tool_info["tool_quality"]

        if tool_quality not in quality_performance:
            quality_performance[tool_quality] = {
                "total": 0,
                "successful": 0,
                "streaming": 0,
                "tools": 0,
            }

        status_emoji = "✅" if model_success else "⚠️"
        print(f"\n  📊 {model_name} ({tool_quality}): {status_emoji}")

        for test_name, test_result in model_results.items():
            total_tests += 1
            quality_performance[tool_quality]["total"] += 1

            chuk = test_result["chuk"]

            if chuk["success"]:
                has_tool_info = chuk.get("has_tool_call_info", False)
                has_tool_calls = len(chuk.get("tool_calls", [])) > 0
                min_chunks = test_result["min_chunks_expected"]

                # Success criteria based on streaming and tool detection
                test_success = (
                    chuk["streaming_worked"]
                    and chuk["chunks"] >= min_chunks
                    and (has_tool_info or has_tool_calls)
                )

                if test_success:
                    print(f"    ✅ {test_name}")
                    successful_tests += 1
                    quality_performance[tool_quality]["successful"] += 1
                else:
                    print(f"    ⚠️  {test_name}")

                if chuk["streaming_worked"]:
                    streaming_worked_count += 1
                    quality_performance[tool_quality]["streaming"] += 1

                if has_tool_info or has_tool_calls:
                    tool_calls_detected += 1
                    quality_performance[tool_quality]["tools"] += 1
            else:
                print(f"    ❌ {test_name}")

    print("\n📈 OVERALL OLLAMA TOOL STATISTICS:")
    print(f"  Total tests: {total_tests}")
    print(
        f"  Successful: {successful_tests}/{total_tests} ({100 * successful_tests // total_tests if total_tests > 0 else 0}%)"
    )
    print(f"  ChukLLM streaming worked: {streaming_worked_count}/{total_tests}")
    print(f"  Tool calls detected: {tool_calls_detected}/{total_tests}")

    print("\n📊 Performance by Tool Quality:")
    for quality, perf in quality_performance.items():
        if perf["total"] > 0:
            success_rate = (perf["successful"] / perf["total"]) * 100
            streaming_rate = (perf["streaming"] / perf["total"]) * 100
            tool_rate = (perf["tools"] / perf["total"]) * 100
            print(
                f"  {quality.title()}: {success_rate:.0f}% success, {streaming_rate:.0f}% streaming, {tool_rate:.0f}% tools"
            )

    # Success criteria
    success_rate = successful_tests / total_tests if total_tests > 0 else 0
    streaming_rate = streaming_worked_count / total_tests if total_tests > 0 else 0
    tool_rate = tool_calls_detected / total_tests if total_tests > 0 else 0

    # Success if most tests pass, streaming works, and tools are detected
    final_success = success_rate >= 0.6 and streaming_rate >= 0.8 and tool_rate >= 0.4

    return final_success


async def test_specific_ollama_tool_models():
    """Test specific high-quality Ollama models for tool capabilities"""
    print("\n🏆 SPECIFIC HIGH-QUALITY MODEL TESTS")
    print("=" * 50)

    # Test your best tool models specifically
    specific_models = [
        ("qwen3:latest", "Reasoning + Tools Champion"),
        ("granite3.3:latest", "Enterprise Tool Model"),
        ("gpt-oss:latest", "Efficient Tool Model"),
    ]

    success_count = 0

    for model_name, description in specific_models:
        print(f"\n🎯 Testing {model_name} - {description}")

        # Complex tool scenario
        prompt = "I need to process user data: first read users.csv, calculate the average age, then save statistics to report.txt"
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "read_csv",
                    "description": "Read CSV file data",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filename": {
                                "type": "string",
                                "description": "CSV file to read",
                            },
                            "delimiter": {
                                "type": "string",
                                "description": "CSV delimiter",
                                "default": ",",
                            },
                        },
                        "required": ["filename"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "calculate_statistics",
                    "description": "Calculate statistics on data",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "data": {"type": "array", "description": "Data to analyze"},
                            "stat_type": {
                                "type": "string",
                                "description": "Type of statistic",
                            },
                        },
                        "required": ["data", "stat_type"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "write_report",
                    "description": "Write report to file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filename": {
                                "type": "string",
                                "description": "Output filename",
                            },
                            "content": {
                                "type": "string",
                                "description": "Report content",
                            },
                        },
                        "required": ["filename", "content"],
                    },
                },
            },
        ]

        try:
            result = await test_chuk_llm_ollama_tool_streaming(
                model_name, prompt, tools
            )

            if result["success"]:
                chunk_count = result["chunks"]
                tool_count = len(result["tool_calls"])
                has_tool_info = result["has_tool_call_info"]
                response_time = result["response_time"]

                print(
                    f"  ✅ {model_name} streamed {chunk_count} chunks in {response_time:.2f}s"
                )
                print(f"  🔧 Tool calls detected: {tool_count}")
                print(f"  📋 Tool patterns found: {has_tool_info}")

                if tool_count > 0:
                    functions = [tc["function"]["name"] for tc in result["tool_calls"]]
                    print(f"  🎯 Functions called: {functions}")

                response_preview = result["final_response"][:200]
                print(f"  📝 Response: {response_preview}...")

                # Quality assessment
                if chunk_count >= 10 and (tool_count > 0 or has_tool_info):
                    print("  🏆 Excellent tool performance!")
                    success_count += 1
                elif chunk_count >= 5 and has_tool_info:
                    print("  ✅ Good tool performance!")
                    success_count += 0.8
                else:
                    print("  ⚠️  Basic performance")
                    success_count += 0.4

            else:
                print(
                    f"  ❌ {model_name} failed: {result.get('error', 'Unknown error')}"
                )

        except Exception as e:
            print(f"  ❌ {model_name} test failed: {e}")

    return success_count >= 2  # At least 2 models working well


async def main():
    """Run comprehensive Ollama tool streaming diagnostic"""
    print("🚀 OLLAMA TOOL STREAMING DIAGNOSTIC - COMPREHENSIVE ANALYSIS")
    print("Testing tool calling and streaming across your local Ollama models")
    print("Equivalent to OpenAI tool diagnostic but for local model ecosystem")

    # Test 1: Multi-model tool streaming across available models
    multi_model_ok = await test_ollama_tool_streaming()

    # Test 2: Specific high-quality model tests
    specific_models_ok = await test_specific_ollama_tool_models()

    print("\n" + "=" * 60)
    print("🎯 FINAL OLLAMA TOOL DIAGNOSTIC SUMMARY:")
    print(f"Multi-model tool tests: {'✅ PASS' if multi_model_ok else '⚠️ PARTIAL'}")
    print(f"Specific model tests: {'✅ PASS' if specific_models_ok else '⚠️ PARTIAL'}")

    if multi_model_ok and specific_models_ok:
        print("\n🎉 OLLAMA TOOL STREAMING WORKS EXCELLENTLY!")
        print("   ✅ Tool-capable models streaming properly")
        print("   ✅ Tool call detection working across models")
        print("   ✅ Complex tool scenarios handled well")
        print("   ✅ ChukLLM tool integration seamless")
        print("   ✅ Your local models support advanced tool workflows")
    elif multi_model_ok or specific_models_ok:
        print("\n✅ OLLAMA TOOL STREAMING WORKS WELL OVERALL!")
        print("   Most tool-capable models working properly")
        print("   Tool call detection functional across model types")
        if not multi_model_ok:
            print("   Some model-specific tool patterns to optimize")
        if not specific_models_ok:
            print("   Some advanced tool scenarios to enhance")
    else:
        print("\n⚠️  OLLAMA TOOL DIAGNOSTIC NEEDS ATTENTION")
        print("   Tool capabilities may be limited in current setup")
        print("   Consider testing with known tool-capable models")
        print("   Some models may not support advanced tool calling")

    print("\n🔧 Ollama tool streaming diagnostic complete!")
    print("Your local model ecosystem is evaluated for tool calling capabilities!")


if __name__ == "__main__":
    asyncio.run(main())
