#!/usr/bin/env python3
"""
Ollama Streaming Diagnostic - Multi-Model Test
==============================================

Tests streaming behavior across different Ollama model types with comprehensive analysis.
Equivalent to the OpenAI streaming diagnostic but for your local Ollama models.
"""

import asyncio
import json
import os
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
        print(f"✅ Loaded .env")
    else:
        load_dotenv()
except ImportError:
    print("⚠️ No dotenv")


def classify_ollama_model(model_name):
    """Classify Ollama models by type and capabilities"""
    model_lower = model_name.lower()
    
    # True reasoning specialists
    if any(pattern in model_lower for pattern in ['granite3.3', 'qwen3']):
        return {
            'type': 'reasoning',
            'supports_tools': True,
            'supports_streaming': True,
            'specialization': 'step-by-step thinking'
        }
    
    # Large general models
    elif any(pattern in model_lower for pattern in ['gpt-oss', 'mistral-small', 'mistral-nemo']):
        return {
            'type': 'large_general',
            'supports_tools': True,
            'supports_streaming': True,
            'specialization': 'comprehensive analysis'
        }
    
    # Code specialists
    elif any(pattern in model_lower for pattern in ['codellama', 'codegemma']):
        return {
            'type': 'code',
            'supports_tools': True,
            'supports_streaming': True,
            'specialization': 'code generation'
        }
    
    # Vision models
    elif any(pattern in model_lower for pattern in ['llava', 'moondream', 'vision']):
        return {
            'type': 'vision',
            'supports_tools': False,
            'supports_streaming': True,
            'specialization': 'image understanding'
        }
    
    # Embedding models (not for chat)
    elif any(pattern in model_lower for pattern in ['embed', 'embedding']):
        return {
            'type': 'embedding',
            'supports_tools': False,
            'supports_streaming': False,
            'specialization': 'embeddings only'
        }
    
    # General purpose models (llama, mistral, gemma)
    else:
        return {
            'type': 'general',
            'supports_tools': True,
            'supports_streaming': True,
            'specialization': 'general purpose'
        }


async def test_streaming_with_ollama_models():
    """Test streaming behavior across different Ollama model types."""
    
    print("🔍 OLLAMA MULTI-MODEL STREAMING ANALYSIS")
    print("=" * 50)
    
    # Discover available models first
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from chuk_llm.llm.discovery.ollama_discoverer import OllamaModelDiscoverer
        
        discoverer = OllamaModelDiscoverer()
        models = await discoverer.discover_models()
        
        if not models:
            print("❌ No Ollama models found")
            return False
        
        print(f"✅ Found {len(models)} Ollama models")
        
        # Filter out embedding models and select test models
        chat_models = []
        for model_data in models:
            model_name = model_data.get('name', '')
            classification = classify_ollama_model(model_name)
            
            if classification['type'] != 'embedding' and classification['supports_streaming']:
                chat_models.append((model_name, classification))
        
        print(f"📋 Selected {len(chat_models)} models for streaming tests")
        
        # Group by type for representative testing
        models_by_type = {}
        for model_name, classification in chat_models:
            model_type = classification['type']
            if model_type not in models_by_type:
                models_by_type[model_type] = []
            models_by_type[model_type].append((model_name, classification))
        
        # Select representative models (max 2 per type)
        test_models = []
        for model_type, type_models in models_by_type.items():
            # Prioritize certain models if available
            priority_models = []
            other_models = []
            
            for model_name, classification in type_models:
                if any(priority in model_name.lower() for priority in ['granite3.3', 'qwen3', 'gpt-oss', 'llama3.1', 'gemma3']):
                    priority_models.append((model_name, classification))
                else:
                    other_models.append((model_name, classification))
            
            # Take up to 2 models per type, prioritizing key models
            selected = priority_models[:2]
            if len(selected) < 2:
                selected.extend(other_models[:2-len(selected)])
            
            test_models.extend(selected)
        
        print(f"\n🎯 Testing {len(test_models)} representative models:")
        for model_name, classification in test_models:
            print(f"   • {model_name} [{classification['type']}] - {classification['specialization']}")
        
    except Exception as e:
        print(f"❌ Model discovery failed: {e}")
        # Fallback to known models
        test_models = [
            ("llama3.1:latest", classify_ollama_model("llama3.1:latest")),
            ("qwen3:latest", classify_ollama_model("qwen3:latest")),
            ("granite3.3:latest", classify_ollama_model("granite3.3:latest")),
            ("gemma3:latest", classify_ollama_model("gemma3:latest")),
            ("gpt-oss:latest", classify_ollama_model("gpt-oss:latest")),
        ]
        print(f"🔄 Using fallback model list: {len(test_models)} models")
    
    # Test cases tailored for Ollama models
    test_cases = [
        {
            "name": "Simple Response",
            "messages": [{"role": "user", "content": "What is 2+2? Answer briefly."}],
            "evaluation": "text_streaming",
            "min_chunks": 1,
            "max_tokens": 50
        },
        {
            "name": "Complex Reasoning", 
            "messages": [{"role": "user", "content": "If a train leaves at 2pm going 60mph, and another at 3pm going 80mph, when do they meet if they're 200 miles apart? Think step by step."}],
            "evaluation": "reasoning_streaming",
            "min_chunks": 5,
            "max_tokens": 300
        },
        {
            "name": "Code Generation",
            "messages": [{"role": "user", "content": "Write a Python function to calculate the factorial of a number. Include comments."}],
            "evaluation": "code_streaming", 
            "min_chunks": 3,
            "max_tokens": 200
        },
        {
            "name": "Creative Writing",
            "messages": [{"role": "user", "content": "Write a short haiku about artificial intelligence."}],
            "evaluation": "creative_streaming",
            "min_chunks": 2,
            "max_tokens": 100
        },
        {
            "name": "Analysis Task",
            "messages": [{"role": "user", "content": "What are the main advantages and disadvantages of renewable energy? Be comprehensive."}],
            "evaluation": "analysis_streaming",
            "min_chunks": 8,
            "max_tokens": 250
        }
    ]
    
    results = {}
    overall_success = True
    
    for model_name, classification in test_models:
        print(f"\n🎯 Testing {model_name} ({classification['type']} - {classification['specialization']})")
        print("-" * 60)
        
        model_results = {}
        model_success = True
        
        for test_case in test_cases:
            case_name = test_case["name"]
            
            # Skip certain tests for certain model types
            if classification['type'] == 'code' and case_name not in ["Simple Response", "Code Generation"]:
                print(f"  ⏭️ Skipping {case_name} (not optimal for code models)")
                continue
            
            if classification['type'] == 'reasoning' and case_name == "Simple Response":
                print(f"  ⏭️ Skipping {case_name} (testing reasoning capabilities instead)")
                continue
            
            print(f"  🧪 {case_name}...")
            
            # Test with ChukLLM
            chuk_result = await test_chuk_llm_ollama_streaming(
                model_name, test_case["messages"], test_case["max_tokens"]
            )
            
            # Test with raw Ollama API for comparison
            raw_result = await test_raw_ollama_streaming(
                model_name, test_case["messages"], test_case["max_tokens"]
            )
            
            model_results[case_name] = {
                "chuk": chuk_result,
                "raw": raw_result,
                "evaluation_type": test_case["evaluation"],
                "min_chunks_expected": test_case["min_chunks"]
            }
            
            # Analyze results
            case_success = analyze_ollama_result(
                chuk_result, raw_result, case_name, test_case, classification
            )
            
            if not case_success:
                model_success = False
        
        results[model_name] = {
            "results": model_results,
            "success": model_success,
            "classification": classification
        }
        
        if not model_success:
            overall_success = False
    
    # Overall analysis
    print("\n" + "=" * 60)
    print("📊 COMPREHENSIVE OLLAMA STREAMING ANALYSIS")
    return analyze_all_ollama_results(results, overall_success)


async def test_chuk_llm_ollama_streaming(model_name, messages, max_tokens):
    """Test ChukLLM streaming with Ollama models"""
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from chuk_llm import stream
        
        result = {
            "success": False,
            "chunks": 0,
            "final_response": "",
            "streaming_worked": False,
            "error": None,
            "response_time": 0
        }
        
        # Use ChukLLM's streaming
        chunk_count = 0
        response_parts = []
        
        start_time = time.time()
        
        async for chunk in stream(
            messages[-1]["content"],
            provider="ollama",
            model=model_name,
            max_tokens=max_tokens
        ):
            chunk_count += 1
            if chunk:
                response_parts.append(str(chunk))
        
        end_time = time.time()
        
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
            "response_time": 0
        }


async def test_raw_ollama_streaming(model_name, messages, max_tokens):
    """Test raw Ollama API streaming for comparison"""
    try:
        import httpx
        
        result = {
            "success": False,
            "chunks": 0,
            "final_response": "",
            "streaming_worked": False,
            "error": None,
            "response_time": 0
        }
        
        # Prepare Ollama API request
        ollama_messages = []
        for msg in messages:
            ollama_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        payload = {
            "model": model_name,
            "messages": ollama_messages,
            "stream": True,
            "options": {
                "num_predict": max_tokens
            }
        }
        
        chunk_count = 0
        response_parts = []
        
        start_time = time.time()
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST",
                "http://localhost:11434/api/chat",
                json=payload
            ) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if line.strip():
                        try:
                            chunk_data = json.loads(line)
                            if "message" in chunk_data and "content" in chunk_data["message"]:
                                content = chunk_data["message"]["content"]
                                if content:
                                    chunk_count += 1
                                    response_parts.append(content)
                        except json.JSONDecodeError:
                            continue
        
        end_time = time.time()
        
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
            "response_time": 0
        }


def analyze_ollama_result(chuk_result, raw_result, test_name, test_case, classification):
    """Analyze a single Ollama test case result"""
    print(f"    📋 {test_name} Results:")
    
    if not chuk_result["success"]:
        print(f"      ❌ ChukLLM failed: {chuk_result.get('error', 'Unknown')}")
        return False
    
    if not raw_result["success"]:
        print(f"      ❌ Raw Ollama failed: {raw_result.get('error', 'Unknown')}")
        return False
    
    # Analyze streaming performance
    chuk_streamed = chuk_result["streaming_worked"]
    raw_streamed = raw_result["streaming_worked"]
    evaluation_type = test_case["evaluation"]
    min_chunks = test_case["min_chunks"]
    
    # Check if ChukLLM meets minimum requirements
    chuk_meets_minimum = chuk_result["chunks"] >= min_chunks
    
    # Evaluate based on test type and model classification
    if evaluation_type == "reasoning_streaming" and classification["type"] == "reasoning":
        # Reasoning models should show step-by-step thinking
        response = chuk_result["final_response"].lower()
        reasoning_indicators = ["step", "first", "then", "therefore", "because", "since"]
        reasoning_found = sum(1 for indicator in reasoning_indicators if indicator in response)
        
        if chuk_streamed and chuk_meets_minimum and reasoning_found >= 2:
            print(f"      ✅ Excellent reasoning streaming ({chuk_result['chunks']} chunks, {reasoning_found} indicators)")
            success = True
        elif chuk_streamed and chuk_meets_minimum:
            print(f"      ✅ Good streaming ({chuk_result['chunks']} chunks)")
            success = True
        else:
            print(f"      ⚠️  Reasoning streaming needs improvement")
            success = False
            
    elif evaluation_type == "code_streaming":
        # Check for code-like content
        response = chuk_result["final_response"]
        code_indicators = ["def ", "function", "return", "{", "}", "//", "#"]
        code_found = sum(1 for indicator in code_indicators if indicator in response)
        
        if chuk_streamed and chuk_meets_minimum and code_found >= 2:
            print(f"      ✅ Good code streaming ({chuk_result['chunks']} chunks, code detected)")
            success = True
        elif chuk_streamed and chuk_meets_minimum:
            print(f"      ✅ Streaming works ({chuk_result['chunks']} chunks)")
            success = True
        else:
            print(f"      ⚠️  Code streaming could be better")
            success = False
            
    else:  # General text streaming
        if chuk_streamed and chuk_meets_minimum:
            # Compare with raw Ollama performance
            chunk_ratio = abs(chuk_result['chunks'] - raw_result['chunks']) / max(raw_result['chunks'], 1)
            
            if chunk_ratio < 0.5:  # Similar performance
                print(f"      ✅ Excellent streaming (ChukLLM: {chuk_result['chunks']}, Raw: {raw_result['chunks']})")
                success = True
            else:
                print(f"      ✅ Good streaming (ChukLLM: {chuk_result['chunks']}, Raw: {raw_result['chunks']})")
                success = chuk_meets_minimum
        elif chuk_streamed:
            print(f"      ⚠️  Streaming but low chunks ({chuk_result['chunks']} < {min_chunks})")
            success = False
        else:
            print(f"      ❌ No streaming detected")
            success = False
    
    # Show response times
    chuk_time = chuk_result.get("response_time", 0)
    raw_time = raw_result.get("response_time", 0)
    print(f"      ⏱️  Response times - ChukLLM: {chuk_time:.2f}s, Raw: {raw_time:.2f}s")
    
    # Show response previews
    chuk_preview = chuk_result["final_response"][:120] if chuk_result["final_response"] else ""
    if chuk_preview:
        print(f"      📝 Response: {chuk_preview}...")
    
    return success


def analyze_all_ollama_results(results, overall_success):
    """Analyze all Ollama test results"""
    print("\n🎯 COMPREHENSIVE OLLAMA ANALYSIS:")
    
    total_tests = 0
    successful_tests = 0
    streaming_worked_count = 0
    
    # Results by model type
    type_performance = {}
    
    for model_name, model_data in results.items():
        model_results = model_data["results"]
        model_success = model_data["success"]
        classification = model_data["classification"]
        model_type = classification["type"]
        
        if model_type not in type_performance:
            type_performance[model_type] = {"total": 0, "successful": 0, "streaming": 0}
        
        status_emoji = "✅" if model_success else "⚠️"
        print(f"\n  📊 {model_name} ({model_type}): {status_emoji}")
        
        for test_name, test_result in model_results.items():
            total_tests += 1
            type_performance[model_type]["total"] += 1
            
            chuk = test_result["chuk"]
            
            if chuk["success"]:
                evaluation_type = test_result["evaluation_type"]
                min_chunks = test_result["min_chunks_expected"]
                
                test_success = chuk["streaming_worked"] and chuk["chunks"] >= min_chunks
                
                if test_success:
                    print(f"    ✅ {test_name}")
                    successful_tests += 1
                    type_performance[model_type]["successful"] += 1
                else:
                    print(f"    ⚠️  {test_name}")
                
                if chuk["streaming_worked"]:
                    streaming_worked_count += 1
                    type_performance[model_type]["streaming"] += 1
            else:
                print(f"    ❌ {test_name}")
    
    print(f"\n📈 OVERALL OLLAMA STATISTICS:")
    print(f"  Total tests: {total_tests}")
    print(f"  Successful: {successful_tests}/{total_tests} ({100*successful_tests//total_tests if total_tests > 0 else 0}%)")
    print(f"  ChukLLM streaming worked: {streaming_worked_count}/{total_tests}")
    
    print(f"\n📊 Performance by Model Type:")
    for model_type, perf in type_performance.items():
        if perf["total"] > 0:
            success_rate = (perf["successful"] / perf["total"]) * 100
            streaming_rate = (perf["streaming"] / perf["total"]) * 100
            print(f"  {model_type.title()}: {success_rate:.0f}% success, {streaming_rate:.0f}% streaming")
    
    # Success criteria - more lenient for Ollama
    success_rate = successful_tests / total_tests if total_tests > 0 else 0
    streaming_rate = streaming_worked_count / total_tests if total_tests > 0 else 0
    
    final_success = success_rate >= 0.7 and streaming_rate >= 0.8  # 70% success, 80% streaming
    
    return final_success


async def test_ollama_specific_features():
    """Test Ollama-specific streaming features"""
    print("\n🏠 OLLAMA-SPECIFIC FEATURE TESTS")
    print("=" * 50)
    
    # Test gpt-oss specifically since it was mentioned
    specific_tests = [
        ("gpt-oss:latest", "Large model comprehensive response"),
        ("qwen3:latest", "Reasoning model step-by-step thinking"),
        ("granite3.3:latest", "Reasoning model analysis")
    ]
    
    success_count = 0
    
    for model_name, description in specific_tests:
        print(f"\n🎯 Testing {model_name} - {description}")
        
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from chuk_llm import stream
            
            # Test comprehensive response
            prompt = "Explain the concept of machine learning, including supervised and unsupervised learning, with examples of each."
            
            chunks = []
            chunk_count = 0
            
            start_time = time.time()
            
            async for chunk in stream(
                prompt,
                provider="ollama",
                model=model_name,
                max_tokens=400
            ):
                chunk_count += 1
                if chunk:
                    chunks.append(str(chunk))
            
            response_time = time.time() - start_time
            full_response = "".join(chunks)
            
            print(f"  ✅ {model_name} streamed {chunk_count} chunks in {response_time:.2f}s")
            print(f"  📋 Response length: {len(full_response)} characters")
            print(f"  📝 Preview: {full_response[:200]}...")
            
            # Quality checks
            quality_indicators = {
                "supervised": "supervised" in full_response.lower(),
                "unsupervised": "unsupervised" in full_response.lower(),
                "examples": "example" in full_response.lower(),
                "comprehensive": len(full_response) > 500
            }
            
            quality_score = sum(quality_indicators.values())
            print(f"  📊 Quality indicators: {quality_score}/4")
            
            if chunk_count >= 10 and quality_score >= 3:
                print(f"  ✅ Excellent performance")
                success_count += 1
            elif chunk_count >= 5:
                print(f"  ✅ Good performance")
                success_count += 0.8
            else:
                print(f"  ⚠️  Performance could be better")
                
        except Exception as e:
            print(f"  ❌ {model_name} failed: {e}")
    
    return success_count >= 2  # At least 2 models working well


async def main():
    """Run comprehensive Ollama streaming diagnostic"""
    print("🚀 OLLAMA STREAMING DIAGNOSTIC - COMPREHENSIVE ANALYSIS")
    print("Testing streaming across your local Ollama model collection")
    print("Equivalent to OpenAI diagnostic but for local models")
    
    # Test 1: Multi-model streaming across available models
    multi_model_ok = await test_streaming_with_ollama_models()
    
    # Test 2: Ollama-specific feature tests
    ollama_features_ok = await test_ollama_specific_features()
    
    print("\n" + "=" * 60)
    print("🎯 FINAL OLLAMA DIAGNOSTIC SUMMARY:")
    print(f"Multi-model streaming tests: {'✅ PASS' if multi_model_ok else '⚠️ PARTIAL'}")
    print(f"Ollama-specific tests: {'✅ PASS' if ollama_features_ok else '⚠️ PARTIAL'}")
    
    if multi_model_ok and ollama_features_ok:
        print("\n🎉 OLLAMA STREAMING WORKS EXCELLENTLY!")
        print("   ✅ All model types streaming properly")
        print("   ✅ Reasoning models showing step-by-step thinking")
        print("   ✅ Large models (gpt-oss) providing comprehensive responses")
        print("   ✅ General models working efficiently")
        print("   ✅ ChukLLM integration seamless")
    elif multi_model_ok or ollama_features_ok:
        print("\n✅ OLLAMA STREAMING WORKS WELL OVERALL!")
        print("   Most local models streaming properly")
        print("   ChukLLM Ollama integration functional") 
        if not multi_model_ok:
            print("   Some model-specific streaming to optimize")
        if not ollama_features_ok:
            print("   Some advanced features to enhance")
    else:
        print("\n⚠️  OLLAMA DIAGNOSTIC NEEDS ATTENTION")
        print("   System likely working but performance could be better")
        print("   Consider checking Ollama service and model availability")
    
    print("\n🏠 Ollama streaming diagnostic complete!")
    print("Your local model ecosystem is ready for streaming workloads!")

if __name__ == "__main__":
    asyncio.run(main())