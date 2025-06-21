#!/usr/bin/env python3
"""
Google Gemini Provider Example Usage Script - Universal Config Version
======================================================================

Demonstrates all the features of the Gemini provider using the unified config system.
Tests universal vision format, JSON mode support, function calling, and system parameter handling.

Requirements:
- pip install google-genai chuk-llm
- Set GEMINI_API_KEY environment variable

Usage:
    python gemini_example.py
    python gemini_example.py --model gemini-2.0-flash-exp
    python gemini_example.py --skip-vision
    python gemini_example.py --test-multimodal
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
if not os.getenv("GEMINI_API_KEY"):
    print("❌ Please set GEMINI_API_KEY environment variable")
    print("   export GEMINI_API_KEY='your_api_key_here'")
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
        return "iVBORw0KGgoAAAANSUhEUgAAABQAAAAUCAYAAACNiR0NAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAAdgAAAHYBTnsmCAAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAABYSURBVDiN7dMxDQAhDAVQPlYgASuwAhuwAiuwAiuwAiuwAiuwAiuwgv8FJpBMJnfJfc0TDaVLkiRJkiRJkmQpY621zjn775xzSimllFJKKaWUUkoppZRSSimllFJKKe8AK0wGkZ6oONkAAAAASUVORK5CYII="

# =============================================================================
# Example 1: Basic Text Completion
# =============================================================================

async def basic_text_example(model: str = "gemini-2.0-flash-exp"):
    """Basic text completion example"""
    print(f"\n🤖 Basic Text Completion with {model}")
    print("=" * 60)
    
    client = get_client("gemini", model=model)
    
    messages = [
        {"role": "user", "content": "Explain large language models in simple terms (2-3 sentences)."}
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

async def streaming_example(model: str = "gemini-2.0-flash-exp"):
    """Real-time streaming example"""
    print(f"\n⚡ Streaming Example with {model}")
    print("=" * 60)
    
    # Check streaming support using unified config
    config = get_config()
    if not config.supports_feature("gemini", Feature.STREAMING, model):
        print(f"⚠️  Model {model} doesn't support streaming")
        return None
    
    client = get_client("gemini", model=model)
    
    messages = [
        {"role": "user", "content": "Write a short poem about the future of technology."}
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

async def function_calling_example(model: str = "gemini-2.0-flash-exp"):
    """Function calling with tools"""
    print(f"\n🔧 Function Calling with {model}")
    print("=" * 60)
    
    # Check if model supports tools using unified config
    config = get_config()
    if not config.supports_feature("gemini", Feature.TOOLS, model):
        print(f"⚠️  Skipping function calling: Model {model} doesn't support tools")
        return None
    
    client = get_client("gemini", model=model)
    
    # Define tools
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_location_info",
                "description": "Get information about a location including coordinates and timezone",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The location name (city, country)"
                        },
                        "include_weather": {
                            "type": "boolean", 
                            "description": "Whether to include weather information"
                        }
                    },
                    "required": ["location"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "unit_converter",
                "description": "Convert between different units of measurement",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "value": {
                            "type": "number",
                            "description": "The value to convert"
                        },
                        "from_unit": {
                            "type": "string",
                            "description": "The unit to convert from"
                        },
                        "to_unit": {
                            "type": "string",
                            "description": "The unit to convert to"
                        }
                    },
                    "required": ["value", "from_unit", "to_unit"]
                }
            }
        }
    ]
    
    messages = [
        {"role": "user", "content": "Get information about Tokyo, Japan and convert 100 kilometers to miles."}
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
            
            if func_name == "get_location_info":
                result = '{"location": "Tokyo, Japan", "coordinates": {"lat": 35.6762, "lng": 139.6503}, "timezone": "Asia/Tokyo", "population": 37400068}'
            elif func_name == "unit_converter":
                result = '{"original_value": 100, "from_unit": "kilometers", "to_unit": "miles", "converted_value": 62.137, "formula": "km * 0.621371"}'
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

async def universal_vision_example(model: str = "gemini-2.0-flash-exp"):
    """Vision capabilities using universal image_url format"""
    print(f"\n👁️  Universal Vision Format Example with {model}")
    print("=" * 60)
    
    # Check if model supports vision using unified config
    config = get_config()
    if not config.supports_feature("gemini", Feature.VISION, model):
        print(f"⚠️  Skipping vision: Model {model} doesn't support vision")
        print(f"💡 Vision-capable models: gemini-2.0-flash-exp, gemini-1.5-pro, gemini-1.5-flash")
        
        # Suggest a vision-capable model
        vision_models = ["gemini-2.0-flash-exp", "gemini-1.5-pro", "gemini-1.5-flash"]
        for suggested_model in vision_models:
            if config.supports_feature("gemini", Feature.VISION, suggested_model):
                print(f"💡 Try: --model {suggested_model}")
                break
        
        return None
    
    client = get_client("gemini", model=model)
    
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
    
    try:
        # Add timeout to prevent hanging on vision requests
        response = await asyncio.wait_for(
            client.create_completion(messages, max_tokens=100),
            timeout=30.0  # 30 second timeout
        )
        
        print(f"✅ Vision response:")
        print(f"   {response['response']}")
        
        return response
        
    except asyncio.TimeoutError:
        print("❌ Vision request timed out after 30 seconds")
        return {"response": "Timeout error", "tool_calls": [], "error": True}
    except Exception as e:
        print(f"❌ Vision request failed: {e}")
        return {"response": f"Error: {str(e)}", "tool_calls": [], "error": True}

# =============================================================================
# Example 5: System Parameter Support
# =============================================================================

async def system_parameter_example(model: str = "gemini-2.0-flash-exp"):
    """System parameter example with different personas"""
    print(f"\n🎭 System Parameter Example with {model}")
    print("=" * 60)
    
    # Check system message support
    config = get_config()
    if not config.supports_feature("gemini", Feature.SYSTEM_MESSAGES, model):
        print(f"⚠️  Model {model} doesn't support system messages")
        return None
    
    client = get_client("gemini", model=model)
    
    # Test different personas using the system parameter
    personas = [
        {
            "name": "Creative Writer",
            "system": "You are a creative writer who loves to tell engaging stories with vivid descriptions.",
            "query": "Describe a sunset."
        },
        {
            "name": "Technical Expert",
            "system": "You are a technical expert who explains complex concepts clearly and precisely.",
            "query": "Explain how neural networks work."
        },
        {
            "name": "Friendly Teacher",
            "system": "You are a patient and encouraging teacher who makes learning fun for students.",
            "query": "Teach me about photosynthesis."
        }
    ]
    
    for persona in personas:
        print(f"\n🎭 Testing {persona['name']} persona:")
        
        messages = [
            {"role": "user", "content": persona["query"]}
        ]
        
        try:
            # Use the system parameter properly
            response = await client.create_completion(
                messages, 
                system=persona["system"],
                max_tokens=150
            )
            print(f"   {response['response'][:200]}...")
        except Exception as e:
            # Fallback: add system instruction to user message
            system_messages = [
                {"role": "user", "content": f"System: {persona['system']}\n\nUser: {persona['query']}"}
            ]
            response = await client.create_completion(system_messages, max_tokens=150)
            print(f"   {response['response'][:200]}...")
    
    return True

# =============================================================================
# Example 6: JSON Mode Support
# =============================================================================

async def json_mode_example(model: str = "gemini-2.0-flash-exp"):
    """JSON mode example using response_format"""
    print(f"\n📋 JSON Mode Example with {model}")
    print("=" * 60)
    
    # Check JSON mode support using unified config
    config = get_config()
    if not config.supports_feature("gemini", Feature.JSON_MODE, model):
        print(f"⚠️  Model {model} doesn't support JSON mode")
        return None
    
    client = get_client("gemini", model=model)
    
    # Test JSON mode with different requests - more specific prompts
    json_tasks = [
        {
            "name": "Technology Profile",
            "prompt": """Create a JSON technology profile with exactly these fields: name, category, year_invented, inventors (array), applications. 
            Technology: JavaScript programming language invented in 1995 by Brendan Eich for web development, server-side programming, and mobile apps.""",
            "expected_keys": ["name", "category", "year_invented", "inventors", "applications"]
        },
        {
            "name": "AI Model Analysis", 
            "prompt": """Generate JSON with exactly these fields: model_family, capabilities (array), strengths (array), use_cases (array).
            Model: Google Gemini which is a multimodal AI with vision, reasoning, coding, and language capabilities.""",
            "expected_keys": ["model_family", "capabilities", "strengths", "use_cases"]
        },
        {
            "name": "Research Paper",
            "prompt": """Generate a research paper JSON with exactly these fields: title, authors (array), year, abstract, keywords (array).
            Paper: About transformer architecture for natural language processing published in 2017.""",
            "expected_keys": ["title", "authors", "year", "abstract", "keywords"]
        }
    ]
    
    for task in json_tasks:
        print(f"\n📋 {task['name']} JSON Generation:")
        
        messages = [
            {"role": "user", "content": task["prompt"]}
        ]
        
        try:
            # Test using Gemini-specific JSON mode
            response = await client.create_completion(
                messages,
                generation_config={
                    "response_mime_type": "application/json"
                },
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
        
        except Exception as e:
            print(f"   ❌ JSON mode failed: {e}")
            # Fallback to regular request with JSON instruction
            fallback_messages = [
                {"role": "user", "content": task["prompt"] + " Please respond only with valid JSON."}
            ]
            response = await client.create_completion(fallback_messages)
            print(f"   📝 Fallback response: {response['response'][:200]}...")
    
    return True

# =============================================================================
# Example 7: Model Comparison using Unified Config
# =============================================================================

async def model_comparison_example():
    """Compare different Gemini models using unified config"""
    print(f"\n📊 Model Comparison")
    print("=" * 60)
    
    # Get all Gemini models from unified config
    config = get_config()
    provider_config = config.get_provider("gemini")
    models = [
        "gemini-2.0-flash-exp",                 # Gemini 2.0 Flash Experimental
        "gemini-2.0-flash-thinking-exp",        # Gemini 2.0 Flash Thinking
        "gemini-1.5-pro",                       # Gemini 1.5 Pro
        "gemini-1.5-flash",                     # Gemini 1.5 Flash
        "gemini-1.5-flash-8b",                  # Gemini 1.5 Flash 8B
    ][:4]  # Test top 4 models
    
    prompt = "What is artificial intelligence? (One sentence)"
    results = {}
    
    for model in models:
        try:
            print(f"🔄 Testing {model}...")
            
            # Get model capabilities
            model_caps = provider_config.get_model_capabilities(model)
            features = [f.value for f in model_caps.features] if model_caps else []
            
            client = get_client("gemini", model=model)
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
        model_short = model.replace("gemini-", "").replace("-exp", "")
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

async def feature_detection_example(model: str = "gemini-2.0-flash-exp"):
    """Detect and display model features using unified config"""
    print(f"\n🔬 Feature Detection for {model}")
    print("=" * 60)
    
    # Get model info
    model_info = get_provider_info("gemini", model)
    
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
    client = get_client("gemini", model=model)
    client_info = client.get_model_info()
    print(f"\n🔧 Client Features:")
    print(f"   Streaming: {'✅' if client_info.get('supports_streaming') else '❌'}")
    print(f"   Vision: {'✅' if client_info.get('supports_vision') else '❌'}")
    print(f"   Function Calling: {'✅' if client_info.get('supports_function_calling') else '❌'}")
    print(f"   JSON Mode: {'✅' if client_info.get('supports_json_mode') else '❌'}")
    print(f"   System Messages: {'✅' if client_info.get('supports_system_messages') else '❌'}")
    
    return model_info

# =============================================================================
# Example 9: Comprehensive Feature Test
# =============================================================================

async def comprehensive_feature_test(model: str = "gemini-2.0-flash-exp"):
    """Test all features in one comprehensive example"""
    print(f"\n🚀 Comprehensive Feature Test with {model}")
    print("=" * 60)
    
    # Check if model supports vision first
    config = get_config()
    if not config.supports_feature("gemini", Feature.VISION, model):
        print(f"⚠️  Model {model} doesn't support vision - using text-only comprehensive test")
        return await comprehensive_text_only_test(model)
    
    client = get_client("gemini", model=model)
    
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
    
    try:
        # Test with all features combined, with timeout
        response = await asyncio.wait_for(
            client.create_completion(
                messages,
                tools=tools,
                system="You are an expert image analyst. Always use the provided function to structure your results.",
                max_tokens=300
            ),
            timeout=30.0
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
    
    except asyncio.TimeoutError:
        print("❌ Comprehensive test timed out after 30 seconds")
        return {"response": "Timeout error", "error": True}
    except Exception as e:
        print(f"❌ Comprehensive test failed: {e}")
        return {"response": f"Error: {str(e)}", "error": True}
    
    print("✅ Comprehensive test completed!")
    return response

async def comprehensive_text_only_test(model: str):
    """Comprehensive test without vision for non-vision models"""
    client = get_client("gemini", model=model)
    
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
            "content": "Please analyze this text and use the text_analysis_result function: 'I absolutely love working with Google Gemini! The multimodal capabilities are fantastic!'"
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
# Example 10: Multimodal Capabilities Test
# =============================================================================

async def multimodal_example(model: str = "gemini-2.0-flash-exp"):
    """Test multimodal capabilities with multiple content types"""
    print(f"\n🎭 Multimodal Capabilities Test with {model}")
    print("=" * 60)
    
    # Check if model supports vision
    config = get_config()
    if not config.supports_feature("gemini", Feature.VISION, model):
        print(f"⚠️  Model {model} doesn't support vision - skipping multimodal test")
        return None
    
    client = get_client("gemini", model=model)
    
    # Create multiple test images
    print("🖼️  Creating test images...")
    red_image = create_test_image("red", 20)
    blue_image = create_test_image("blue", 20)
    
    # Test with multiple images and text
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "I'm showing you two colored squares. Please compare them and tell me which is which color."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{red_image}"
                    }
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{blue_image}"
                    }
                },
                {
                    "type": "text",
                    "text": "Which image is red and which is blue?"
                }
            ]
        }
    ]
    
    print("👀 Analyzing multiple images...")
    
    try:
        response = await asyncio.wait_for(
            client.create_completion(messages, max_tokens=200),
            timeout=30.0
        )
        
        print(f"✅ Multimodal response:")
        print(f"   {response['response']}")
        
        return response
        
    except asyncio.TimeoutError:
        print("❌ Multimodal test timed out after 30 seconds")
        return {"response": "Timeout error", "error": True}
    except Exception as e:
        print(f"❌ Multimodal test failed: {e}")
        return {"response": f"Error: {str(e)}", "error": True}

# =============================================================================
# Main Function
# =============================================================================

async def main():
    """Run all examples"""
    parser = argparse.ArgumentParser(description="Google Gemini Provider Example Script - Universal Config")
    parser.add_argument("--model", default="gemini-2.0-flash-exp", help="Model to use (default: gemini-2.0-flash-exp)")
    parser.add_argument("--skip-vision", action="store_true", help="Skip vision examples")
    parser.add_argument("--skip-functions", action="store_true", help="Skip function calling")
    parser.add_argument("--test-multimodal", action="store_true", help="Focus on multimodal capabilities")
    parser.add_argument("--quick", action="store_true", help="Run only basic examples")
    parser.add_argument("--comprehensive", action="store_true", help="Run comprehensive feature test")
    
    args = parser.parse_args()
    
    print("🚀 Google Gemini Provider Examples (Universal Config v3)")
    print("=" * 60)
    print(f"Using model: {args.model}")
    print(f"API Key: {'✅ Set' if os.getenv('GEMINI_API_KEY') else '❌ Missing'}")
    
    # Show config info
    try:
        config = get_config()
        provider_config = config.get_provider("gemini")
        print(f"Available models: {len(provider_config.models)}")
        print(f"Baseline features: {', '.join(f.value for f in provider_config.features)}")
        
        # Check if the selected model supports vision
        if config.supports_feature("gemini", Feature.VISION, args.model):
            print(f"✅ Model {args.model} supports vision")
        else:
            print(f"⚠️  Model {args.model} doesn't support vision - vision tests will be skipped")
            if not args.skip_vision:
                vision_models = ["gemini-2.0-flash-exp", "gemini-1.5-pro", "gemini-1.5-flash"]
                print(f"💡 For vision tests, try: {', '.join(vision_models)}")
                
    except Exception as e:
        print(f"⚠️  Config warning: {e}")
    
    # Run comprehensive test if requested
    if args.comprehensive:
        await comprehensive_feature_test(args.model)
        return
    
    # Focus on multimodal if requested
    if args.test_multimodal:
        await multimodal_example(args.model)
        return
    
    examples = [
        ("Feature Detection", lambda: feature_detection_example(args.model)),
        ("Basic Text", lambda: basic_text_example(args.model)),
        ("Streaming", lambda: streaming_example(args.model)),
        ("System Parameter", lambda: system_parameter_example(args.model)),
        ("JSON Mode", lambda: json_mode_example(args.model)),
    ]
    
    if not args.quick:
        if not args.skip_functions:
            examples.append(("Function Calling", lambda: function_calling_example(args.model)))
        
        if not args.skip_vision:
            examples.append(("Universal Vision", lambda: universal_vision_example(args.model)))
            examples.append(("Multimodal", lambda: multimodal_example(args.model)))
        
        examples.extend([
            ("Model Comparison", model_comparison_example),
            ("Comprehensive Test", lambda: comprehensive_feature_test(args.model)),
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
        print(f"🔗 Google Gemini provider is working perfectly with universal config!")
        print(f"✨ Features tested: System params, JSON mode, Universal vision, Tools, Streaming")
    else:
        print(f"\n⚠️  Some examples failed. Check your API key and model access.")
        
        # Show model recommendations
        print(f"\n💡 Model Recommendations:")
        print(f"   • For latest features: gemini-2.0-flash-exp")
        print(f"   • For reasoning: gemini-2.0-flash-thinking-exp")
        print(f"   • For production: gemini-1.5-pro, gemini-1.5-flash")
        print(f"   • For efficiency: gemini-1.5-flash-8b")
        print(f"   • For vision: gemini-2.0-flash-exp, gemini-1.5-pro")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Examples cancelled by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)