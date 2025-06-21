#!/usr/bin/env python3
"""
Watson X Provider Example Usage Script - Universal Config Version
================================================================

Demonstrates all the features of the Watson X provider using the unified config system.
Tests universal format support, JSON mode, function calling, and system parameter handling.

Requirements:
- pip install ibm-watsonx-ai chuk-llm
- Set WATSONX_API_KEY (or IBM_CLOUD_API_KEY) environment variable
- Set WATSONX_PROJECT_ID environment variable
- Set WATSONX_AI_URL environment variable (optional, defaults to us-south)

Usage:
    python watsonx_example.py
    python watsonx_example.py --model ibm/granite-3-8b-instruct
    python watsonx_example.py --skip-tools
    python watsonx_example.py --test-benchmark
"""

import asyncio
import argparse
import os
import sys
import time
import json
import base64
from typing import Dict, Any, List

# dotenv
from dotenv import load_dotenv

# load environment variables
load_dotenv() 

# Ensure we have the required environment
if not (os.getenv("WATSONX_API_KEY") or os.getenv("IBM_CLOUD_API_KEY")):
    print("❌ Please set WATSONX_API_KEY or IBM_CLOUD_API_KEY environment variable")
    print("   export WATSONX_API_KEY='your_api_key_here'")
    print("   # or")
    print("   export IBM_CLOUD_API_KEY='your_ibm_cloud_api_key_here'")
    sys.exit(1)

if not os.getenv("WATSONX_PROJECT_ID"):
    print("❌ Please set WATSONX_PROJECT_ID environment variable")
    print("   export WATSONX_PROJECT_ID='your_project_id_here'")
    sys.exit(1)

try:
    from chuk_llm.llm.client import get_client, get_provider_info, validate_provider_setup
    from chuk_llm.configuration import get_config, CapabilityChecker, Feature
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("   Please make sure you're running from the chuk-llm directory")
    sys.exit(1)

def create_test_image(color="red", size=20):
    """Create a test image as base64 - tries PIL first, fallback to hardcoded"""
    try:
        from PIL import Image
        import io
        
        # Create a colored square
        img = Image.new('RGB', (size, size), color)
        
        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return img_data
    except ImportError:
        print("⚠️  PIL not available, using fallback image")
        # Fallback: 20x20 red square (valid PNG)
        return "iVBORw0KGgoAAAANSUhEUgAAABQAAAAUCAYAAACNiR0NAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAAdgAAAHYBTnsmCAAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAABYSURBVDiN7dMxDQAhDAVQPlYgASuwAhuwAiuwAiuwAiuwAiuwAiuwgv8FJpBMJnfJfc0TDaVLkiRJkiRJkmQpY621zjl775xzSimllFJKKaWUUkoppZRSSimllFJKKe8AK0wGkZ6oONkAAAAASUVORK5CYII="

# =============================================================================
# Example 1: Basic Text Completion
# =============================================================================

async def basic_text_example(model: str = "ibm/granite-3-8b-instruct"):
    """Basic text completion example"""
    print(f"\n🤖 Basic Text Completion with {model}")
    print("=" * 60)
    
    client = get_client("watsonx", model=model)
    
    messages = [
        {"role": "user", "content": "Explain the concept of recursion in programming in simple terms (2-3 sentences)."}
    ]
    
    start_time = time.time()
    response = await client.create_completion(messages)
    duration = time.time() - start_time
    
    print(f"✅ Response ({duration:.2f}s):")
    print(f"   {response['response']}")
    
    return response

# =============================================================================
# Example 2: Streaming Response
# =============================================================================

async def streaming_example(model: str = "ibm/granite-3-8b-instruct"):
    """Real-time streaming example"""
    print(f"\n⚡ Streaming Example with {model}")
    print("=" * 60)
    
    # Check streaming support using unified config
    config = get_config()
    if not config.supports_feature("watsonx", Feature.STREAMING, model):
        print(f"⚠️  Model {model} doesn't support streaming")
        return None
    
    client = get_client("watsonx", model=model)
    
    messages = [
        {"role": "user", "content": "Write a short poem about the beauty of code."}
    ]
    
    print("🌊 Streaming response:")
    print("   ", end="", flush=True)
    
    start_time = time.time()
    full_response = ""
    
    async for chunk in client.create_completion(messages, stream=True):
        if chunk.get("response"):
            content = chunk["response"]
            print(content, end="", flush=True)
            full_response += content
    
    duration = time.time() - start_time
    print(f"\n✅ Streaming completed ({duration:.2f}s)")
    
    return full_response

# =============================================================================
# Example 3: Function Calling
# =============================================================================

async def function_calling_example(model: str = "ibm/granite-3-8b-instruct"):
    """Function calling with tools"""
    print(f"\n🔧 Function Calling with {model}")
    print("=" * 60)
    
    # Check if model supports tools using unified config
    config = get_config()
    if not config.supports_feature("watsonx", Feature.TOOLS, model):
        print(f"⚠️  Skipping function calling: Model {model} doesn't support tools")
        return None
    
    client = get_client("watsonx", model=model)
    
    # Define tools
    tools = [
        {
            "type": "function",
            "function": {
                "name": "add_numbers",
                "description": "Adds two numbers together",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {
                            "type": "number",
                            "description": "First number"
                        },
                        "b": {
                            "type": "number", 
                            "description": "Second number"
                        }
                    },
                    "required": ["a", "b"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "analyze_sentiment",
                "description": "Analyze the sentiment of a given text",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "The text to analyze"
                        },
                        "include_confidence": {
                            "type": "boolean", 
                            "description": "Whether to include confidence score"
                        }
                    },
                    "required": ["text"]
                }
            }
        }
    ]
    
    messages = [
        {"role": "user", "content": "Please add 15 and 27, and also analyze the sentiment of this text: 'I absolutely love working with Watson X!'"}
    ]
    
    print("🔄 Making function calling request...")
    response = await client.create_completion(messages, tools=tools)
    
    if response.get("tool_calls"):
        print(f"✅ Tool calls requested: {len(response['tool_calls'])}")
        for i, tool_call in enumerate(response["tool_calls"], 1):
            func_name = tool_call["function"]["name"]
            func_args = tool_call["function"]["arguments"]
            print(f"   {i}. {func_name}({func_args})")
        
        # Simulate tool execution
        messages.append({
            "role": "assistant",
            "content": "",
            "tool_calls": response["tool_calls"]
        })
        
        # Add mock tool results
        for tool_call in response["tool_calls"]:
            func_name = tool_call["function"]["name"]
            
            if func_name == "add_numbers":
                result = '{"sum": 42}'
            elif func_name == "analyze_sentiment":
                result = '{"sentiment": "positive", "confidence": 0.98, "score": 0.95}'
            else:
                result = '{"status": "success"}'
            
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "name": func_name,
                "content": result
            })
        
        # Get final response
        print("🔄 Getting final response...")
        final_response = await client.create_completion(messages)
        print(f"✅ Final response:")
        print(f"   {final_response['response']}")
        
        return final_response
    else:
        print("ℹ️  No tool calls were made")
        print(f"   Response: {response['response']}")
        return response

# =============================================================================
# Example 4: Universal Vision Format
# =============================================================================

async def universal_vision_example(model: str = "meta-llama/llama-3-2-90b-vision-instruct"):
    """Vision capabilities using universal image format"""
    print(f"\n👁️  Universal Vision Format Example with {model}")
    print("=" * 60)
    
    # Check if model supports vision using unified config
    config = get_config()
    if not config.supports_feature("watsonx", Feature.VISION, model):
        print(f"⚠️  Skipping vision: Model {model} doesn't support vision")
        print(f"💡 Vision-capable models: meta-llama/llama-3-2-90b-vision-instruct")
        return None
    
    client = get_client("watsonx", model=model)
    
    # Create a proper test image
    print("🖼️  Creating test image...")
    test_image_b64 = create_test_image("blue", 30)
    
    # Test universal image_url format (this should work with all providers)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What color is this square? Please describe it in one sentence."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{test_image_b64}"
                    }
                }
            ]
        }
    ]
    
    print("👀 Analyzing image using universal format...")
    response = await client.create_completion(messages, max_tokens=100)
    
    print(f"✅ Vision response:")
    print(f"   {response['response']}")
    
    return response

# =============================================================================
# Example 5: System Parameter Support
# =============================================================================

async def system_parameter_example(model: str = "ibm/granite-3-8b-instruct"):
    """System parameter example with different personas"""
    print(f"\n🎭 System Parameter Example with {model}")
    print("=" * 60)
    
    # Check system message support
    config = get_config()
    if not config.supports_feature("watsonx", Feature.SYSTEM_MESSAGES, model):
        print(f"⚠️  Model {model} doesn't support system messages")
        return None
    
    client = get_client("watsonx", model=model)
    
    # Test different personas using the system parameter
    personas = [
        {
            "name": "Helpful Assistant",
            "system": "You are a helpful, harmless, and honest AI assistant.",
            "query": "How do I bake a cake?"
        },
        {
            "name": "Pirate Captain",
            "system": "You are a friendly pirate captain. Speak like a pirate and use nautical terms.",
            "query": "How do I bake a cake?"
        },
        {
            "name": "Technical Expert",
            "system": "You are a senior software engineer with expertise in Python and system design.",
            "query": "How do I optimize a slow database query?"
        }
    ]
    
    for persona in personas:
        print(f"\n🎭 Testing {persona['name']} persona:")
        
        messages = [
            {"role": "user", "content": persona["query"]}
        ]
        
        # Use the system parameter properly
        response = await client.create_completion(
            messages, 
            system=persona["system"],
            max_tokens=150
        )
        print(f"   {response['response'][:200]}...")
    
    return True

# =============================================================================
# Example 6: JSON Mode Support
# =============================================================================

async def json_mode_example(model: str = "ibm/granite-3-8b-instruct"):
    """JSON mode example using response_format"""
    print(f"\n📋 JSON Mode Example with {model}")
    print("=" * 60)
    
    # Check JSON mode support using unified config
    config = get_config()
    if not config.supports_feature("watsonx", Feature.JSON_MODE, model):
        print(f"⚠️  Model {model} doesn't support JSON mode")
        return None
    
    client = get_client("watsonx", model=model)
    
    # Test JSON mode with different requests - more specific prompts
    json_tasks = [
        {
            "name": "Employee Profile",
            "prompt": """Create a JSON employee profile with exactly these fields: name, department, skills (array), years_experience. 
            Employee: A data scientist named Sarah who works in Analytics with Python, SQL, and machine learning skills for 5 years.""",
            "expected_keys": ["name", "department", "skills", "years_experience"]
        },
        {
            "name": "Model Information", 
            "prompt": """Generate JSON with exactly these fields: model_name, provider, capabilities (array), use_cases (array).
            Model: IBM Granite which is an enterprise AI model for coding, reasoning, and enterprise applications.""",
            "expected_keys": ["model_name", "provider", "capabilities", "use_cases"]
        },
        {
            "name": "System Status",
            "prompt": """Generate a system status JSON with exactly these fields: status, cpu_usage, memory_usage, services (array).
            System: Healthy server running at 45% CPU, 67% memory, with services: web, database, cache.""",
            "expected_keys": ["status", "cpu_usage", "memory_usage", "services"]
        }
    ]
    
    for task in json_tasks:
        print(f"\n📋 {task['name']} JSON Generation:")
        
        messages = [
            {"role": "user", "content": task["prompt"]}
        ]
        
        # Test using response_format with explicit system instruction
        response = await client.create_completion(
            messages,
            response_format={"type": "json_object"},
            system="You must respond with valid JSON only. No markdown, no code blocks, no explanations. Just pure JSON.",
            max_tokens=300
        )
        
        if response.get("response"):
            try:
                # Clean the response - remove any markdown formatting
                clean_response = response["response"].strip()
                if clean_response.startswith("```json"):
                    clean_response = clean_response.replace("```json", "").replace("```", "").strip()
                
                json_data = json.loads(clean_response)
                print(f"   ✅ Valid JSON with keys: {list(json_data.keys())}")
                
                # Check if expected keys are present
                found_keys = set(json_data.keys())
                expected_keys = set(task["expected_keys"])
                missing_keys = expected_keys - found_keys
                
                if missing_keys:
                    print(f"   ⚠️  Missing expected keys: {missing_keys}")
                else:
                    print(f"   ✅ All expected keys found")
                
                # Pretty print a sample
                sample_json = json.dumps(json_data, indent=2)
                if len(sample_json) > 200:
                    sample_json = sample_json[:200] + "..."
                print(f"   📄 Sample: {sample_json}")
                
            except json.JSONDecodeError as e:
                print(f"   ❌ Invalid JSON: {e}")
                print(f"   📄 Raw response: {response['response'][:200]}...")
        else:
            print(f"   ❌ No response received")
    
    return True

# =============================================================================
# Example 7: Model Comparison using Unified Config
# =============================================================================

async def model_comparison_example():
    """Compare different Watson X models using unified config"""
    print(f"\n📊 Model Comparison")
    print("=" * 60)
    
    # Get all Watson X models from unified config
    config = get_config()
    provider_config = config.get_provider("watsonx")
    models = [
        "ibm/granite-3-8b-instruct",               # IBM Granite - very reliable
        "meta-llama/llama-3-2-1b-instruct",        # Llama 3.2 1B - fast
        "meta-llama/llama-3-2-3b-instruct",        # Llama 3.2 3B - good balance  
        "meta-llama/llama-3-3-70b-instruct",       # Llama 3.3 70B - most capable
        "mistralai/mistral-large",                 # Mistral Large - enterprise
    ][:4]  # Test top 4 models
    
    prompt = "What is artificial intelligence? (One sentence)"
    results = {}
    
    for model in models:
        try:
            print(f"🔄 Testing {model}...")
            
            # Get model capabilities
            model_caps = provider_config.get_model_capabilities(model)
            features = [f.value for f in model_caps.features] if model_caps else []
            
            client = get_client("watsonx", model=model)
            messages = [{"role": "user", "content": prompt}]
            
            start_time = time.time()
            response = await client.create_completion(messages)
            duration = time.time() - start_time
            
            results[model] = {
                "response": response.get("response", ""),
                "time": duration,
                "length": len(response.get("response", "")),
                "features": features,
                "success": True
            }
            
        except Exception as e:
            results[model] = {
                "response": f"Error: {str(e)}",
                "time": 0,
                "length": 0,
                "features": [],
                "success": False
            }
    
    print("\n📈 Results:")
    for model, result in results.items():
        status = "✅" if result["success"] else "❌"
        model_short = model.replace("meta-llama/", "").replace("ibm/", "").replace("mistralai/", "")
        print(f"   {status} {model_short}:")
        print(f"      Time: {result['time']:.2f}s")
        print(f"      Length: {result['length']} chars")
        print(f"      Features: {', '.join(result['features'][:3])}...")
        print(f"      Response: {result['response'][:80]}...")
        print()
    
    return results

# =============================================================================
# Example 8: Feature Detection with Universal Config
# =============================================================================

async def feature_detection_example(model: str = "ibm/granite-3-8b-instruct"):
    """Detect and display model features using unified config"""
    print(f"\n🔬 Feature Detection for {model}")
    print("=" * 60)
    
    # Get model info
    model_info = get_provider_info("watsonx", model)
    
    print("📋 Model Information:")
    print(f"   Provider: {model_info['provider']}")
    print(f"   Model: {model_info['model']}")
    print(f"   Max Context: {model_info['max_context_length']:,} tokens")
    print(f"   Max Output: {model_info['max_output_tokens']:,} tokens")
    
    print("\n🎯 Supported Features:")
    for feature, supported in model_info['supports'].items():
        status = "✅" if supported else "❌"
        print(f"   {status} {feature}")
    
    print("\n📊 Rate Limits:")
    for tier, limit in model_info['rate_limits'].items():
        print(f"   {tier}: {limit} requests/min")
    
    # Test actual client info
    client = get_client("watsonx", model=model)
    client_info = client.get_model_info()
    print(f"\n🔧 Client Features:")
    print(f"   Streaming: {'✅' if client_info.get('supports_streaming') else '❌'}")
    print(f"   JSON Mode: {'✅' if client_info.get('supports_json_mode') else '❌'}")
    print(f"   System Messages: {'✅' if client_info.get('supports_system_messages') else '❌'}")
    print(f"   Function Calling: {'✅' if client_info.get('supports_function_calling') else '❌'}")
    
    return model_info

# =============================================================================
# Example 9: Comprehensive Feature Test
# =============================================================================

async def comprehensive_feature_test(model: str = "ibm/granite-3-8b-instruct"):
    """Test all features in one comprehensive example"""
    print(f"\n🚀 Comprehensive Feature Test with {model}")
    print("=" * 60)
    
    # Check if model supports vision first
    config = get_config()
    if not config.supports_feature("watsonx", Feature.VISION, model):
        print(f"⚠️  Model {model} doesn't support vision - using text-only comprehensive test")
        return await comprehensive_text_only_test(model)
    
    client = get_client("watsonx", model=model)
    
    # Create a test image
    print("🖼️  Creating test image...")
    test_image_b64 = create_test_image("green", 25)
    
    # Test: System message + Vision + Tools + JSON mode
    tools = [
        {
            "type": "function",
            "function": {
                "name": "image_analysis_result",
                "description": "Store the structured result of image analysis",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_type": {"type": "string"},
                        "dominant_colors": {"type": "array", "items": {"type": "string"}},
                        "dimensions": {"type": "string"},
                        "description": {"type": "string"}
                    },
                    "required": ["image_type", "dominant_colors", "description"]
                }
            }
        }
    ]
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Please analyze this image and use the image_analysis_result function to store your findings in a structured format."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{test_image_b64}"
                    }
                }
            ]
        }
    ]
    
    print("🔄 Testing: System + Vision + Tools...")
    
    # Test with all features combined
    response = await client.create_completion(
        messages,
        tools=tools,
        system="You are an expert image analyst. Always use the provided function to structure your results.",
        max_tokens=300
    )
    
    if response.get("tool_calls"):
        print(f"✅ Tool calls generated: {len(response['tool_calls'])}")
        for tc in response["tool_calls"]:
            print(f"   🔧 {tc['function']['name']}: {tc['function']['arguments'][:100]}...")
        
        # Simulate tool execution
        messages.append({
            "role": "assistant",
            "tool_calls": response["tool_calls"]
        })
        
        # Add tool result
        for tc in response["tool_calls"]:
            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "name": tc["function"]["name"],
                "content": '{"status": "stored", "analysis_id": "test_123"}'
            })
        
        # Get final response
        final_response = await client.create_completion(messages)
        print(f"✅ Final analysis: {final_response['response'][:150]}...")
        
    else:
        print(f"ℹ️  Direct response: {response['response'][:150]}...")
    
    print("✅ Comprehensive test completed!")
    return response

async def comprehensive_text_only_test(model: str):
    """Comprehensive test without vision for non-vision models"""
    client = get_client("watsonx", model=model)
    
    # Test: System message + Tools + JSON mode
    tools = [
        {
            "type": "function",
            "function": {
                "name": "text_analysis_result",
                "description": "Store the structured result of text analysis",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sentiment": {"type": "string"},
                        "key_topics": {"type": "array", "items": {"type": "string"}},
                        "word_count": {"type": "number"},
                        "summary": {"type": "string"}
                    },
                    "required": ["sentiment", "key_topics", "summary"]
                }
            }
        }
    ]
    
    messages = [
        {
            "role": "user",
            "content": "Please analyze this text and use the text_analysis_result function: 'I absolutely love working with Watson X and IBM's enterprise AI capabilities. The platform is fantastic!'"
        }
    ]
    
    print("🔄 Testing: System + Tools + JSON (text-only)...")
    
    response = await client.create_completion(
        messages,
        tools=tools,
        system="You are an expert text analyst. Always use the provided function to structure your results.",
        max_tokens=300
    )
    
    if response.get("tool_calls"):
        print(f"✅ Tool calls generated: {len(response['tool_calls'])}")
        for tc in response["tool_calls"]:
            print(f"   🔧 {tc['function']['name']}: {tc['function']['arguments'][:100]}...")
    else:
        print(f"ℹ️  Direct response: {response['response'][:150]}...")
    
    return response

# =============================================================================
# Example 10: Comprehensive Model Benchmark
# =============================================================================

async def comprehensive_benchmark():
    """Comprehensive benchmark across all Watson X models"""
    print(f"\n🏁 Comprehensive Model Benchmark")
    print("=" * 60)
    
    # All major Watson X models (using non-deprecated, reliable versions)
    models = [
        "ibm/granite-3-8b-instruct",               # IBM Granite - very reliable
        "meta-llama/llama-3-2-1b-instruct",        # Llama 3.2 1B - fast and reliable
        "meta-llama/llama-3-2-3b-instruct",        # Llama 3.2 3B - good performance
        "meta-llama/llama-3-3-70b-instruct",       # Llama 3.3 70B - latest, powerful
        "mistralai/mistral-large",                 # Mistral Large - reliable
    ]
    
    # Quick benchmark tasks
    tasks = [
        {
            "name": "Reasoning",
            "prompt": "If a train travels 120 miles in 2 hours, and then 180 miles in 3 hours, what is its average speed for the entire journey?"
        },
        {
            "name": "Creative Writing", 
            "prompt": "Write a creative two-sentence story about a robot discovering emotions."
        },
        {
            "name": "Code Generation",
            "prompt": "Write a Python function that finds the second largest number in a list."
        }
    ]
    
    results = {}
    
    print("🔄 Running benchmark across all models...")
    print(f"📋 Models: {len(models)}, Tasks: {len(tasks)}")
    
    for model in models:
        print(f"\n🤖 Testing {model}...")
        model_results = {}
        
        try:
            client = get_client("watsonx", model=model)
            
            for task in tasks:
                print(f"   📝 {task['name']}...", end="", flush=True)
                
                messages = [{"role": "user", "content": task["prompt"]}]
                
                start_time = time.time()
                response = await client.create_completion(messages, max_tokens=200)
                duration = time.time() - start_time
                
                model_results[task["name"]] = {
                    "success": True,
                    "time": duration,
                    "length": len(response.get("response", "")),
                    "response": response.get("response", "")[:100] + "..."
                }
                
                print(f" ✅ ({duration:.2f}s)")
            
            results[model] = model_results
            
        except Exception as e:
            print(f"   ❌ Failed: {str(e)}")
            results[model] = {"error": str(e)}
    
    # Display comprehensive results
    print(f"\n📊 BENCHMARK RESULTS")
    print("=" * 80)
    
    # Performance summary table
    print(f"{'Model':<35} {'Reasoning':<12} {'Creative':<12} {'Coding':<12} {'Avg Time':<10}")
    print("-" * 85)
    
    for model, model_results in results.items():
        if "error" in model_results:
            print(f"{model:<35} {'❌ Failed':<40}")
            continue
            
        times = []
        status_icons = []
        
        for task in tasks:
            task_result = model_results.get(task["name"], {})
            if task_result.get("success"):
                times.append(task_result["time"])
                status_icons.append(f"{task_result['time']:.2f}s")
            else:
                status_icons.append("❌")
        
        avg_time = sum(times) / len(times) if times else 0
        
        print(f"{model:<35} {status_icons[0]:<12} {status_icons[1]:<12} {status_icons[2]:<12} {avg_time:.2f}s")
    
    # Speed ranking
    speed_ranking = []
    for model, model_results in results.items():
        if "error" not in model_results:
            times = [r["time"] for r in model_results.values() if r.get("success")]
            if times:
                avg_time = sum(times) / len(times)
                speed_ranking.append((model, avg_time))
    
    speed_ranking.sort(key=lambda x: x[1])
    
    print(f"\n🏃 Speed Ranking (Average Response Time):")
    for i, (model, avg_time) in enumerate(speed_ranking, 1):
        model_short = model.replace("meta-llama/", "").replace("ibm/", "").replace("mistralai/", "")
        print(f"   {i}. {model_short:<25} {avg_time:.2f}s")
    
    return results

# =============================================================================
# Main Function
# =============================================================================

async def main():
    """Run all examples"""
    parser = argparse.ArgumentParser(description="Watson X Provider Example Script - Universal Config")
    parser.add_argument("--model", default="ibm/granite-3-8b-instruct", help="Model to use (default: ibm/granite-3-8b-instruct)")
    parser.add_argument("--skip-vision", action="store_true", help="Skip vision examples")
    parser.add_argument("--skip-tools", action="store_true", help="Skip function calling")
    parser.add_argument("--test-benchmark", action="store_true", help="Run comprehensive benchmark")
    parser.add_argument("--quick", action="store_true", help="Run only basic examples")
    parser.add_argument("--comprehensive", action="store_true", help="Run comprehensive feature test")
    
    args = parser.parse_args()
    
    print("🚀 Watson X Provider Examples (Universal Config v3)")
    print("=" * 60)
    print(f"Using model: {args.model}")
    print(f"API Key: {'✅ Set' if (os.getenv('WATSONX_API_KEY') or os.getenv('IBM_CLOUD_API_KEY')) else '❌ Missing'}")
    print(f"Project ID: {'✅ Set' if os.getenv('WATSONX_PROJECT_ID') else '❌ Missing'}")
    print(f"Watson X URL: {os.getenv('WATSONX_AI_URL', 'https://us-south.ml.cloud.ibm.com (default)')}")
    
    # Show config info
    try:
        config = get_config()
        provider_config = config.get_provider("watsonx")
        print(f"Available models: {len(provider_config.models)}")
        print(f"Baseline features: {', '.join(f.value for f in provider_config.features)}")
        
        # Check if the selected model supports vision
        if config.supports_feature("watsonx", Feature.VISION, args.model):
            print(f"✅ Model {args.model} supports vision")
        else:
            print(f"⚠️  Model {args.model} doesn't support vision - vision tests will be skipped")
            if not args.skip_vision:
                vision_models = ["meta-llama/llama-3-2-90b-vision-instruct"]
                print(f"💡 For vision tests, try: {', '.join(vision_models)}")
                
    except Exception as e:
        print(f"⚠️  Config warning: {e}")
    
    # Run comprehensive test if requested
    if args.comprehensive:
        await comprehensive_feature_test(args.model)
        return
    
    # Run benchmark if requested
    if args.test_benchmark:
        await comprehensive_benchmark()
        return
    
    examples = [
        ("Feature Detection", lambda: feature_detection_example(args.model)),
        ("Basic Text", lambda: basic_text_example(args.model)),
        ("Streaming", lambda: streaming_example(args.model)),
        ("System Parameter", lambda: system_parameter_example(args.model)),
        ("JSON Mode", lambda: json_mode_example(args.model)),
    ]
    
    if not args.quick:
        if not args.skip_tools:
            examples.append(("Function Calling", lambda: function_calling_example(args.model)))
        
        if not args.skip_vision:
            examples.append(("Universal Vision", lambda: universal_vision_example("meta-llama/llama-3-2-90b-vision-instruct")))
        
        examples.extend([
            ("Model Comparison", model_comparison_example),
            ("Comprehensive Test", lambda: comprehensive_feature_test(args.model)),
            ("Comprehensive Benchmark", comprehensive_benchmark),
        ])
    
    # Run examples
    results = {}
    for name, example_func in examples:
        try:
            print(f"\n" + "="*60)
            start_time = time.time()
            result = await example_func()
            duration = time.time() - start_time
            results[name] = {"success": True, "result": result, "time": duration}
            print(f"✅ {name} completed in {duration:.2f}s")
        except Exception as e:
            results[name] = {"success": False, "error": str(e), "time": 0}
            print(f"❌ {name} failed: {e}")
    
    # Summary
    print(f"\n" + "="*60)
    print("📊 SUMMARY")
    print("="*60)
    
    successful = sum(1 for r in results.values() if r["success"])
    total = len(results)
    total_time = sum(r["time"] for r in results.values())
    
    print(f"✅ Successful: {successful}/{total}")
    print(f"⏱️  Total time: {total_time:.2f}s")
    
    for name, result in results.items():
        status = "✅" if result["success"] else "❌"
        time_str = f"{result['time']:.2f}s" if result["success"] else "failed"
        print(f"   {status} {name}: {time_str}")
    
    if successful == total:
        print(f"\n🎉 All examples completed successfully!")
        print(f"🔗 Watson X provider is working perfectly with universal config!")
        print(f"✨ Features tested: System params, JSON mode, Universal vision, Tools, Streaming")
    else:
        print(f"\n⚠️  Some examples failed. Check your API key, project ID, and model access.")
        
        # Show model recommendations
        print(f"\n💡 Model Recommendations:")
        print(f"   • For tools + reliability: ibm/granite-3-8b-instruct")
        print(f"   • For speed: meta-llama/llama-3-2-1b-instruct")
        print(f"   • For balance: meta-llama/llama-3-2-3b-instruct")
        print(f"   • For capability: meta-llama/llama-3-3-70b-instruct")
        print(f"   • For enterprise: mistralai/mistral-large")
        print(f"   • For vision: meta-llama/llama-3-2-90b-vision-instruct")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Examples cancelled by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)