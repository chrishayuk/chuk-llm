#!/usr/bin/env python3
"""
Enhanced ChukLLM Dynamic Model Discovery & Inference Demo
========================================================

This comprehensive demo showcases the full capabilities of the ChukLLM discovery system:

✨ Key Features:
- Universal multi-provider model discovery (Ollama, OpenAI, HuggingFace, etc.)
- Enhanced model categorization with reasoning, vision, and code specializations
- Dynamic inference with automatic parameter handling
- Cross-provider model comparison and recommendations
- Health monitoring and configuration generation
- Advanced filtering and ranking algorithms

🎯 What's Demonstrated:
- Discovery across all configured providers
- Model capability inference and categorization
- Dynamic-only model identification and testing
- Reasoning model parameter handling (o1, o3 series)
- Vision model capabilities
- Code-specialized model testing
- Performance tier assessment
- Model recommendations by use case
- Configuration file generation
- Health checks and diagnostics
"""

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import asdict
import traceback

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print(f"📁 Loaded environment variables from .env")
except ImportError:
    print("💡 Using system environment variables")

# Configure logging
logging.basicConfig(level=logging.WARNING)  # Reduce noise
log = logging.getLogger(__name__)

# Demo configuration
DEMO_CONFIG = {
    "max_models_per_provider": 8,  # Test more models since you have many
    "test_timeout": 45,  # Longer timeout for larger models
    "enable_advanced_tests": True,
    "generate_configs": True,
    "run_health_checks": True,
    "test_known_models": True  # Test your specific model collection
}

def print_header(title: str, char: str = "=", width: int = 70):
    """Print a formatted header"""
    print(f"\n{char * width}")
    print(f"{title:^{width}}")
    print(f"{char * width}")

def print_section(title: str, char: str = "-", width: int = 60):
    """Print a formatted section header"""
    print(f"\n{title}")
    print(char * len(title))

def print_model_info(model, prefix: str = "   •"):
    """Print formatted model information"""
    metadata = model.metadata or {}
    
    # Basic info
    family = model.family
    size_info = ""
    
    # Size information
    if metadata.get('size_gb'):
        size_info = f" ({metadata['size_gb']}GB)"
    elif model.parameters:
        size_info = f" ({model.parameters})"
    
    # Special capabilities
    capabilities = []
    if metadata.get('is_reasoning', False) or metadata.get('reasoning_capable', False):
        capabilities.append("🧠 reasoning")
    if metadata.get('is_vision', False) or metadata.get('supports_vision', False):
        capabilities.append("👁️ vision")
    if metadata.get('specialization') == 'code':
        capabilities.append("💻 code")
    if metadata.get('supports_tools', False):
        capabilities.append("🔧 tools")
    
    cap_str = f" [{', '.join(capabilities)}]" if capabilities else ""
    
    print(f"{prefix} {model.name} [{family}]{size_info}{cap_str}")
    
    # Additional details
    if metadata.get('performance_tier'):
        tier = metadata['performance_tier']
        print(f"      └─ Performance: {tier} | Context: {model.context_length or 'unknown'}")

async def discover_all_providers():
    """Discover models from all available providers"""
    print_header("🔍 UNIVERSAL MODEL DISCOVERY")
    
    try:
        # Setup discovery environment
        os.environ['CHUK_LLM_DISCOVERY_ENABLED'] = 'true'
        os.environ['CHUK_LLM_AUTO_DISCOVER'] = 'true'
        print("🔧 Discovery environment enabled")
        
        # Import discovery system
        from chuk_llm.llm.discovery import UniversalDiscoveryManager
        from chuk_llm.configuration import get_config
        
        # Initialize with configuration
        config_manager = get_config()
        discovery_manager = UniversalDiscoveryManager(config_manager)
        
        # Check available providers
        providers = discovery_manager.get_available_providers()
        print(f"🎯 Discovery enabled for {len(providers)} providers: {', '.join(providers)}")
        
        if not providers:
            # Try manual discovery setup for known providers
            print("⚠️  No providers auto-configured, trying manual setup...")
            await setup_manual_discovery()
            return None, None
        
        # Run discovery
        print(f"\n🚀 Starting discovery across all providers...")
        start_time = time.time()
        
        results = await discovery_manager.discover_all_models(force_refresh=True)
        
        discovery_time = time.time() - start_time
        
        # Display results
        print(f"\n✅ Discovery completed in {discovery_time:.2f}s")
        print(f"📊 Total models discovered: {results.total_models}")
        
        # Provider breakdown
        print(f"\n📋 Models by provider:")
        for provider, models in results.models_by_provider.items():
            status = "✅" if provider not in results.errors else "❌"
            print(f"   {status} {provider}: {len(models)} models")
            
            # Show top models for each provider
            if models:
                sorted_models = sorted(models, key=lambda m: _get_model_priority(m), reverse=True)
                
                # Show different categories
                reasoning_models = [m for m in models if m.metadata.get('reasoning_capable', False)]
                vision_models = [m for m in models if m.metadata.get('supports_vision', False)]
                code_models = [m for m in models if m.metadata.get('specialization') == 'code']
                
                if reasoning_models:
                    print(f"      🧠 Reasoning: {reasoning_models[0].name}")
                if vision_models:
                    print(f"      👁️  Vision: {vision_models[0].name}")
                if code_models:
                    print(f"      💻 Code: {code_models[0].name}")
                
                # Show top 3 overall
                for i, model in enumerate(sorted_models[:3]):
                    if i == 0:
                        print(f"      🏆 Top models:")
                    print_model_info(model, "         •")
                    
                if len(models) > 3:
                    print(f"         ... and {len(models) - 3} more")
        
        # Show errors if any
        if results.errors:
            print(f"\n⚠️  Provider errors:")
            for provider, error in results.errors.items():
                print(f"   ❌ {provider}: {error}")
        
        # Enhanced summary stats
        summary = results.summary
        print(f"\n📈 Discovery Summary:")
        print(f"   • Success rate: {summary['success_rate']}%")
        print(f"   • Total discovery time: {discovery_time:.2f}s")
        print(f"   • Reasoning models: {summary['special_model_counts']['reasoning_models']}")
        print(f"   • Vision models: {summary['special_model_counts']['vision_models']}")
        print(f"   • Code models: {summary['special_model_counts']['code_models']}")
        
        # Show top families
        if 'top_families' in summary:
            families = summary['top_families'][:5]
            family_str = ", ".join([f"{name} ({count})" for name, count in families])
            print(f"   • Top families: {family_str}")
        
        return discovery_manager, results
        
    except Exception as e:
        print(f"❌ Discovery failed: {e}")
        traceback.print_exc()
        return None, None

async def setup_manual_discovery():
    """Setup manual discovery for known providers when auto-config fails"""
    print("🔧 Setting up manual discovery...")
    
    # Test Ollama discovery directly
    try:
        from chuk_llm.llm.discovery.ollama_discoverer import OllamaModelDiscoverer
        from chuk_llm.llm.discovery.engine import UniversalModelDiscoveryManager
        
        print("🏠 Testing Ollama direct discovery:")
        ollama_discoverer = OllamaModelDiscoverer()
        ollama_manager = UniversalModelDiscoveryManager("ollama", ollama_discoverer)
        
        models = await ollama_manager.discover_models(force_refresh=True)
        print(f"   ✅ Found {len(models)} Ollama models directly")
        
        # Show interesting models from your collection
        expected_models = [
            'gpt-oss:latest', 'qwen3:latest', 'gemma3:latest', 'granite3.3:latest',
            'mistral-small:latest', 'llama3.2:latest', 'mistral-nemo:latest', 
            'llama3.1:latest', 'mistral:latest', 'llama3:latest'
        ]
        
        found_models = {m.name for m in models}
        print(f"   📋 Known models found:")
        for expected in expected_models:
            if expected in found_models:
                print(f"      ✅ {expected}")
            else:
                print(f"      ❌ {expected} (not found)")
        
        # Show model details
        if models:
            print(f"   🎯 Top discovered models:")
            sorted_models = sorted(models, key=_get_model_priority, reverse=True)
            for model in sorted_models[:5]:
                print_model_info(model, "      •")
        
        return models
        
    except Exception as e:
        print(f"   ❌ Direct Ollama discovery failed: {e}")
        return []

def _get_model_priority(model) -> float:
    """Calculate model priority for sorting"""
    score = 0
    metadata = model.metadata or {}
    
    # Reasoning models get highest priority
    if metadata.get('is_reasoning', False) or metadata.get('reasoning_capable', False):
        score += 100
    
    # Performance tier scoring
    tier_scores = {
        'reasoning': 95, 'advanced-reasoning': 90, 'high': 80, 'large': 70,
        'medium': 50, 'fast': 60, 'small': 30, 'nano': 10
    }
    tier = metadata.get('performance_tier', 'medium')
    score += tier_scores.get(tier, 40)
    
    # Capability bonuses
    if metadata.get('supports_tools', False):
        score += 10
    if metadata.get('is_vision', False) or metadata.get('supports_vision', False):
        score += 8
    if metadata.get('specialization') == 'code':
        score += 6
    
    return score

async def identify_dynamic_models(discovery_manager, results):
    """Identify models that exist only in discovery (not in static config)"""
    print_header("🎯 DYNAMIC-ONLY MODEL IDENTIFICATION")
    
    try:
        # Get static configuration for comparison
        from chuk_llm.configuration import get_config
        config_manager = get_config()
        
        static_models = {}
        
        # Collect static models from each provider
        for provider_name in results.models_by_provider.keys():
            try:
                provider_config = config_manager.get_provider(provider_name)
                static_models[provider_name] = set(provider_config.models)
                print(f"   📋 {provider_name}: {len(static_models[provider_name])} static models")
            except Exception as e:
                static_models[provider_name] = set()
                print(f"   📋 {provider_name}: No static configuration")
        
        # Find dynamic-only models
        dynamic_models = {}
        total_dynamic = 0
        
        for provider_name, models in results.models_by_provider.items():
            dynamic_models[provider_name] = []
            static_set = static_models[provider_name]
            
            for model in models:
                if model.name not in static_set:
                    dynamic_models[provider_name].append(model)
                    total_dynamic += 1
        
        print(f"\n🔍 Dynamic-only models found: {total_dynamic}")
        
        # Display dynamic models by provider
        for provider_name, models in dynamic_models.items():
            if models:
                print(f"\n   🎯 {provider_name.upper()}: {len(models)} dynamic-only models")
                
                # Sort by priority and show top models
                sorted_models = sorted(models, key=_get_model_priority, reverse=True)
                for model in sorted_models[:DEMO_CONFIG['max_models_per_provider']]:
                    print_model_info(model, "      •")
                
                if len(models) > DEMO_CONFIG['max_models_per_provider']:
                    remaining = len(models) - DEMO_CONFIG['max_models_per_provider']
                    print(f"      ... and {remaining} more models")
        
        if total_dynamic == 0:
            print("   ✅ All discovered models are already in static configuration!")
            print("   💡 This means your configuration is comprehensive.")
        
        return dynamic_models
        
    except Exception as e:
        print(f"❌ Dynamic model identification failed: {e}")
        return {}

async def run_inference_tests(dynamic_models):
    """Run comprehensive inference tests with your specific model collection"""
    print_header("🧪 COMPREHENSIVE MODEL INFERENCE TESTS")
    
    test_results = []
    
    # Your actual model collection (from ollama list)
    known_ollama_models = [
        'gpt-oss:latest', 'qwen3:latest', 'gemma3:latest', 'granite3.3:latest',
        'mistral-small:latest', 'llama3.2:latest', 'mistral-nemo:latest', 
        'llama3.1:latest', 'mistral:latest', 'llama3:latest'
    ]
    
    # Test different categories of models
    test_categories = {
        'reasoning': {
            'models': ['granite3.3:latest', 'qwen3:latest', 'llama3.1:latest'],
            'prompt': 'If I have 3 apples and buy 2 more, then give 1 to my friend, how many apples do I have left? Think step by step.',
            'max_tokens': 200
        },
        'general': {
            'models': ['gemma3:latest', 'mistral:latest', 'llama3:latest'],
            'prompt': 'What is the capital of France? Answer in one sentence.',
            'max_tokens': 50
        },
        'large': {
            'models': ['mistral-small:latest', 'gpt-oss:latest'],
            'prompt': 'Explain machine learning in simple terms. Be concise.',
            'max_tokens': 150
        },
        'specialized': {
            'models': ['mistral-nemo:latest', 'llama3.2:latest'],
            'prompt': 'Write a haiku about programming.',
            'max_tokens': 100
        }
    }
    
    for category, test_info in test_categories.items():
        print_section(f"🎯 Testing {category.title()} Models")
        
        for model_name in test_info['models']:
            # Check if this model exists in dynamic models or run anyway
            model_found = False
            model_obj = None
            
            # Look for the model in discovered models
            for provider_name, models in dynamic_models.items():
                for model in models:
                    if model.name == model_name or model_name in model.name:
                        model_found = True
                        model_obj = model
                        break
            
            # Test the model
            description = f"{category} model"
            if model_obj and model_obj.metadata:
                metadata = model_obj.metadata
                size_gb = metadata.get('size_gb', 0)
                if size_gb:
                    description += f" ({size_gb}GB)"
            
            success, response, response_time = await test_model_inference_advanced(
                'ollama', model_name, test_info['prompt'], test_info['max_tokens'], description
            )
            
            test_results.append({
                'provider': 'ollama',
                'model': model_name,
                'category': category,
                'success': success,
                'response_time': response_time,
                'error': response if not success else None,
                'metadata': model_obj.metadata if model_obj else {}
            })
            
            # Brief pause between tests
            await asyncio.sleep(1.5)
    
    # Also test any OpenAI models if available
    if 'openai' in dynamic_models and dynamic_models['openai']:
        print_section(f"🎯 Testing OpenAI Models")
        
        openai_models = dynamic_models['openai'][:3]  # Test first 3
        for model in openai_models:
            metadata = model.metadata or {}
            
            # Choose prompt based on model type
            if metadata.get('is_reasoning', False):
                prompt = 'What is 2+2? Show your reasoning.'
                max_tokens = 100
                params = {'max_completion_tokens': max_tokens}  # For reasoning models
            else:
                prompt = 'What is the capital of Japan?'
                max_tokens = 30
                params = {'max_tokens': max_tokens}
            
            success, response, response_time = await test_openai_model(
                model.name, prompt, params, metadata
            )
            
            test_results.append({
                'provider': 'openai',
                'model': model.name,
                'category': 'reasoning' if metadata.get('is_reasoning') else 'general',
                'success': success,
                'response_time': response_time,
                'error': response if not success else None,
                'metadata': metadata
            })
            
            await asyncio.sleep(1)
    
    return test_results

async def test_model_inference_advanced(provider: str, model_name: str, prompt: str, max_tokens: int, description: str = ""):
    """Advanced model inference testing with better error handling"""
    print(f"\n🧪 Testing: {provider}/{model_name}")
    if description:
        print(f"   📝 {description}")
    print(f"   💬 Prompt: {prompt[:60]}{'...' if len(prompt) > 60 else ''}")
    
    try:
        from chuk_llm import ask
        
        # Test with basic parameters first
        params = {'max_tokens': max_tokens}
        
        print(f"   ⚙️  Parameters: {params}")
        
        # Execute test with timeout
        start_time = time.time()
        
        response = await asyncio.wait_for(
            ask(prompt, provider=provider, model=model_name, **params),
            timeout=DEMO_CONFIG['test_timeout']
        )
        
        response_time = time.time() - start_time
        
        # Display results
        print(f"   ✅ Success in {response_time:.2f}s")
        
        # Show response preview
        response_preview = response.strip()[:120]
        if len(response) > 120:
            response_preview += "..."
        print(f"   💬 Response: {response_preview}")
        
        # Check response quality
        if len(response.strip()) < 5:
            print(f"   ⚠️  Warning: Very short response")
        elif prompt.lower() in response.lower():
            print(f"   ✨ Good: Response addresses the prompt")
        
        return True, response, response_time
        
    except asyncio.TimeoutError:
        print(f"   ⏰ Timeout after {DEMO_CONFIG['test_timeout']}s")
        return False, "Timeout", None
    except Exception as e:
        error_msg = str(e)
        print(f"   ❌ Error: {error_msg}")
        
        # Enhanced error analysis
        if "not found" in error_msg.lower() or "does not exist" in error_msg.lower():
            print("   💡 Model may not be loaded in Ollama. Try: ollama pull " + model_name.replace(':latest', ''))
        elif "connection" in error_msg.lower():
            print("   💡 Check if Ollama is running: ollama serve")
        elif "rate limit" in error_msg.lower():
            print("   💡 Rate limit hit - consider adding delays between requests")
        elif "max_tokens" in error_msg.lower():
            print("   💡 Token parameter issue - model might need different parameters")
        else:
            print(f"   🔍 Error type: {type(e).__name__}")
        
        return False, error_msg, None

async def test_openai_model(model_name: str, prompt: str, params: dict, metadata: dict):
    """Test OpenAI model with proper parameter handling"""
    print(f"\n🧪 Testing: openai/{model_name}")
    
    model_type = "reasoning" if metadata.get('is_reasoning') else "standard"
    generation = metadata.get('generation', 'unknown')
    print(f"   📝 {model_type} model [{generation}]")
    
    try:
        from chuk_llm import ask
        
        print(f"   ⚙️  Parameters: {params}")
        
        start_time = time.time()
        response = await asyncio.wait_for(
            ask(prompt, provider='openai', model=model_name, **params),
            timeout=DEMO_CONFIG['test_timeout']
        )
        response_time = time.time() - start_time
        
        print(f"   ✅ Success in {response_time:.2f}s")
        print(f"   💬 Response: {response[:100]}{'...' if len(response) > 100 else ''}")
        
        return True, response, response_time
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False, str(e), None

async def analyze_test_results(test_results):
    """Analyze and summarize test results with enhanced insights"""
    print_header("📊 COMPREHENSIVE TEST RESULTS ANALYSIS")
    
    total_tests = len(test_results)
    successful_tests = sum(1 for r in test_results if r['success'])
    
    print(f"🎯 Overall Results:")
    print(f"   • Total tests: {total_tests}")
    print(f"   • Successful: {successful_tests}")
    print(f"   • Success rate: {(successful_tests/total_tests*100):.1f}%" if total_tests > 0 else "   • Success rate: N/A")
    
    # Results by provider
    providers = set(r['provider'] for r in test_results)
    print(f"\n📋 Results by Provider:")
    
    for provider in providers:
        provider_results = [r for r in test_results if r['provider'] == provider]
        provider_successful = sum(1 for r in provider_results if r['success'])
        provider_total = len(provider_results)
        
        success_rate = (provider_successful/provider_total*100) if provider_total > 0 else 0
        print(f"   {provider}: {provider_successful}/{provider_total} ({success_rate:.1f}%)")
        
        # Show successful models with details
        successful_models = [r for r in provider_results if r['success']]
        if successful_models:
            print(f"      ✅ Working models:")
            for result in successful_models:
                model_name = result['model']
                resp_time = result.get('response_time', 0)
                category = result.get('category', 'unknown')
                print(f"         • {model_name} ({category}) - {resp_time:.2f}s")
        
        # Show failed models with error analysis
        failed_models = [r for r in provider_results if not r['success']]
        if failed_models:
            print(f"      ❌ Issues found:")
            
            # Group by error type for better analysis
            error_groups = {}
            for result in failed_models:
                error = result['error'] or "Unknown error"
                error_type = "Other"
                
                if "timeout" in error.lower():
                    error_type = "Timeout"
                elif "not found" in error.lower() or "does not exist" in error.lower():
                    error_type = "Model Not Available"
                elif "connection" in error.lower():
                    error_type = "Connection Error"
                elif "rate limit" in error.lower():
                    error_type = "Rate Limited"
                elif "max_tokens" in error.lower() or "parameter" in error.lower():
                    error_type = "Parameter Error"
                
                if error_type not in error_groups:
                    error_groups[error_type] = []
                error_groups[error_type].append(result['model'])
            
            for error_type, models in error_groups.items():
                models_str = ", ".join(models[:3])
                if len(models) > 3:
                    models_str += f" (+{len(models)-3} more)"
                print(f"         • {error_type}: {models_str}")
    
    # Category analysis (for your Ollama models)
    categories = set(r.get('category', 'unknown') for r in test_results)
    if len(categories) > 1:
        print(f"\n🏷️  Results by Model Category:")
        
        for category in sorted(categories):
            if category == 'unknown':
                continue
            category_results = [r for r in test_results if r.get('category') == category]
            category_successful = sum(1 for r in category_results if r['success'])
            category_total = len(category_results)
            
            if category_total > 0:
                success_rate = (category_successful/category_total*100)
                print(f"   {category.title()}: {category_successful}/{category_total} ({success_rate:.1f}%)")
    
    # Performance analysis
    successful_with_time = [r for r in test_results if r['success'] and r['response_time']]
    if successful_with_time:
        times = [r['response_time'] for r in successful_with_time]
        avg_time = sum(times) / len(times)
        
        print(f"\n⏱️  Performance Analysis:")
        print(f"   • Average response time: {avg_time:.2f}s")
        print(f"   • Fastest response: {min(times):.2f}s")
        print(f"   • Slowest response: {max(times):.2f}s")
        
        # Performance by provider
        for provider in providers:
            provider_times = [r['response_time'] for r in test_results 
                            if r['provider'] == provider and r['success'] and r['response_time']]
            if provider_times:
                avg_provider_time = sum(provider_times) / len(provider_times)
                fastest = min(provider_times)
                slowest = max(provider_times)
                print(f"   • {provider}: {avg_provider_time:.2f}s avg ({fastest:.2f}s - {slowest:.2f}s range)")
        
        # Speed rankings
        speed_rankings = sorted(successful_with_time, key=lambda x: x['response_time'])
        print(f"\n🏆 Speed Champions (fastest responses):")
        for i, result in enumerate(speed_rankings[:5], 1):
            print(f"   {i}. {result['provider']}/{result['model']}: {result['response_time']:.2f}s")
    
    # Model size vs performance (for Ollama models with size info)
    size_performance = []
    for result in successful_with_time:
        metadata = result.get('metadata', {})
        size_gb = metadata.get('size_gb', 0)
        if size_gb > 0:
            size_performance.append((result['model'], size_gb, result['response_time']))
    
    if size_performance:
        print(f"\n📏 Size vs Performance (Ollama models):")
        size_performance.sort(key=lambda x: x[1])  # Sort by size
        for model, size_gb, resp_time in size_performance:
            efficiency = size_gb / resp_time if resp_time > 0 else 0
            print(f"   • {model}: {size_gb}GB, {resp_time:.2f}s (efficiency: {efficiency:.1f})")
    
    # Recommendations based on results
    print(f"\n💡 Recommendations:")
    
    if successful_tests == total_tests:
        print(f"   🎉 Perfect! All models are working correctly.")
    elif successful_tests > total_tests * 0.8:
        print(f"   👍 Great! Most models are working well.")
    elif successful_tests > total_tests * 0.5:
        print(f"   ⚠️  Mixed results. Some models need attention.")
    else:
        print(f"   🔧 Many issues found. Check your setup.")
    
    # Specific recommendations
    failed_models = [r for r in test_results if not r['success']]
    connection_errors = [r for r in failed_models if 'connection' in (r.get('error', '') or '').lower()]
    not_found_errors = [r for r in failed_models if 'not found' in (r.get('error', '') or '').lower()]
    
    if connection_errors:
        print(f"   🔌 Connection issues detected. Ensure Ollama is running: ollama serve")
    
    if not_found_errors:
        print(f"   📥 Some models not found. Try pulling missing models:")
        for result in not_found_errors[:3]:
            model_name = result['model'].replace(':latest', '')
            print(f"      ollama pull {model_name}")
    
    if successful_with_time:
        fastest_provider = min(providers, key=lambda p: 
            sum(r['response_time'] for r in test_results if r['provider'] == p and r['success'] and r['response_time']) /
            max(1, len([r for r in test_results if r['provider'] == p and r['success'] and r['response_time']]))
        )
        print(f"   🏃‍♂️ For speed, use: {fastest_provider} (fastest average response)")
    
    return {
        'total_tests': total_tests,
        'successful_tests': successful_tests,
        'success_rate': (successful_tests/total_tests*100) if total_tests > 0 else 0,
        'average_response_time': avg_time if successful_with_time else None,
        'providers_tested': len(providers),
        'categories_tested': len(categories) if len(categories) > 1 else 0
    }

async def generate_model_recommendations(discovery_manager):
    """Generate model recommendations by use case"""
    print_header("💡 MODEL RECOMMENDATIONS")
    
    try:
        use_cases = ["general", "reasoning", "vision", "code"]
        
        for use_case in use_cases:
            print_section(f"🎯 Best Models for {use_case.title()}")
            
            recommendations = discovery_manager.get_model_recommendations(use_case)
            
            if recommendations:
                for i, rec in enumerate(recommendations[:5], 1):
                    capabilities = []
                    if rec.get('reasoning'):
                        capabilities.append("🧠")
                    if rec.get('vision'):
                        capabilities.append("👁️")
                    if rec.get('tools'):
                        capabilities.append("🔧")
                    
                    cap_str = "".join(capabilities)
                    context = rec.get('context_length', 'unknown')
                    tier = rec.get('performance_tier', 'medium')
                    
                    print(f"   {i}. {rec['provider']}/{rec['model']} {cap_str}")
                    print(f"      └─ Score: {rec['score']:.0f} | {tier} performance | {context} context")
            else:
                print(f"   No specific recommendations for {use_case}")
    
    except Exception as e:
        print(f"❌ Recommendation generation failed: {e}")

async def run_health_checks(discovery_manager):
    """Run system health checks"""
    print_header("🏥 SYSTEM HEALTH CHECKS")
    
    try:
        health_status = await discovery_manager.health_check()
        
        overall_status = health_status['overall_status']
        status_icons = {
            'healthy': '✅',
            'degraded': '⚠️',
            'critical': '❌'
        }
        
        print(f"🎯 Overall Status: {status_icons.get(overall_status, '❓')} {overall_status.upper()}")
        print(f"   • Providers: {health_status['healthy_providers']}/{health_status['total_providers']} healthy")
        
        print(f"\n📋 Provider Health:")
        for provider, status in health_status['providers'].items():
            status_icon = "✅" if status['status'] == 'healthy' else "❌"
            response_time = status.get('response_time')
            model_count = status.get('model_count', 0)
            
            print(f"   {status_icon} {provider}: {model_count} models", end="")
            if response_time:
                print(f" ({response_time}s)")
            else:
                print(f" - {status.get('error', 'Unknown error')}")
    
    except Exception as e:
        print(f"❌ Health check failed: {e}")

async def generate_configuration_files(discovery_manager):
    """Generate configuration files based on discovered models"""
    print_header("⚙️  CONFIGURATION GENERATION")
    
    try:
        config_updates = discovery_manager.generate_config_updates()
        
        if config_updates:
            print("📝 Generated configuration updates:")
            
            for provider, yaml_config in config_updates.items():
                print(f"\n🔧 {provider.upper()} Configuration:")
                print("   " + "\n   ".join(yaml_config.split("\n")[:10]))  # Show first 10 lines
                if len(yaml_config.split("\n")) > 10:
                    print("   ... (truncated)")
                
                # Optionally save to file
                if DEMO_CONFIG['generate_configs']:
                    config_file = Path(f"discovered_{provider}_models.yaml")
                    config_file.write_text(yaml_config)
                    print(f"   💾 Saved to: {config_file}")
        else:
            print("📝 No configuration updates generated")
            print("   💡 This could mean no new models were discovered")
    
    except Exception as e:
        print(f"❌ Configuration generation failed: {e}")

async def main():
    """Main demo orchestrator"""
    print_header("🚀 ENHANCED ChukLLM DISCOVERY & INFERENCE DEMO", "=", 80)
    
    print("🎯 This demo showcases the full capabilities of ChukLLM's discovery system:")
    print("   • Multi-provider model discovery (Ollama, OpenAI, etc.)")
    print("   • Enhanced model categorization and inference")
    print("   • Dynamic model testing and evaluation")
    print("   • Model recommendations and health monitoring")
    print("   • Configuration generation and optimization")
    
    print(f"\n📋 Based on your system, we'll test these Ollama models:")
    known_models = [
        'gpt-oss:latest (13GB)', 'qwen3:latest (5.2GB)', 'gemma3:latest (3.3GB)',
        'granite3.3:latest (4.9GB)', 'mistral-small:latest (14GB)', 'llama3.2:latest (2.0GB)',
        'mistral-nemo:latest (7.1GB)', 'llama3.1:latest (4.7GB)', 'mistral:latest (4.1GB)'
    ]
    for model in known_models:
        print(f"   • {model}")
    
    # Step 1: Universal Discovery
    discovery_manager, results = await discover_all_providers()
    if not discovery_manager and not results:
        # If universal discovery fails, try manual approach
        print("\n🔧 Trying direct model testing approach...")
        await test_known_models_directly()
        return
    
    # Step 2: Identify Dynamic Models
    dynamic_models = await identify_dynamic_models(discovery_manager, results)
    
    # Step 3: Run Comprehensive Inference Tests
    if any(models for models in dynamic_models.values()) or DEMO_CONFIG['test_known_models']:
        test_results = await run_inference_tests(dynamic_models)
        analysis_summary = await analyze_test_results(test_results)
        
        # Additional analysis for your specific setup
        await analyze_model_collection(test_results)
    else:
        print("\n💡 No dynamic-only models found for testing.")
        print("   All discovered models are already in static configuration!")
    
    # Step 4: Generate Recommendations
    if DEMO_CONFIG['enable_advanced_tests'] and discovery_manager:
        await generate_model_recommendations(discovery_manager)
    
    # Step 5: Health Checks
    if DEMO_CONFIG['run_health_checks'] and discovery_manager:
        await run_health_checks(discovery_manager)
    
    # Step 6: Configuration Generation
    if DEMO_CONFIG['generate_configs'] and discovery_manager:
        await generate_configuration_files(discovery_manager)
    
    # Final Summary
    print_header("🎉 DEMO COMPLETE!", "=", 80)
    
    print("✨ What was demonstrated:")
    print("   ✅ Universal multi-provider model discovery")
    print("   ✅ Enhanced model categorization and capability inference")
    print("   ✅ Dynamic model identification and testing")
    print("   ✅ Comprehensive inference testing with your model collection")
    print("   ✅ Performance analysis and size-vs-speed insights")
    print("   ✅ Error analysis and troubleshooting recommendations")
    print("   ✅ Health monitoring and diagnostics")
    print("   ✅ Automatic configuration generation")
    
    print(f"\n💡 Discovery Results:")
    if results:
        print(f"   • Total models discovered: {results.total_models}")
        print(f"   • Providers: {len(results.models_by_provider)}")
        print(f"   • Success rate: {results.summary['success_rate']}%")
        print(f"   • Reasoning models: {results.summary['special_model_counts']['reasoning_models']}")
        print(f"   • Vision models: {results.summary['special_model_counts']['vision_models']}")
        print(f"   • Code models: {results.summary['special_model_counts']['code_models']}")
    
    print(f"\n🏆 Your Model Collection Highlights:")
    print(f"   • Large models (>10GB): mistral-small, gpt-oss")
    print(f"   • Reasoning models: granite3.3, qwen3, llama3.1")
    print(f"   • Efficient models (<5GB): gemma3, llama3.2, mistral")
    print(f"   • Specialized: mistral-nemo (instruction-tuned)")
    
    print(f"\n🚀 Next Steps:")
    print(f"   • Use discovered models in your applications")
    print(f"   • Update configuration files with generated YAML")
    print(f"   • Monitor model performance over time")
    print(f"   • Consider pulling newer model versions as they become available")
    
    print(f"\n💬 The discovery system makes ANY discovered model available for use!")
    print(f"   Just use: await ask(prompt, provider='ollama', model='qwen3:latest')")

async def test_known_models_directly():
    """Direct testing approach when universal discovery fails"""
    print_header("🔧 DIRECT MODEL TESTING")
    
    test_models = [
        ('qwen3:latest', 'reasoning'),
        ('gemma3:latest', 'general'),
        ('granite3.3:latest', 'reasoning'),
        ('llama3.1:latest', 'general'),
        ('mistral:latest', 'general')
    ]
    
    print("Testing your known Ollama models directly...")
    
    results = []
    for model_name, category in test_models:
        prompt = "What is 2+2? Answer briefly." if category == 'general' else "If I have 3 apples and buy 2 more, how many do I have? Think step by step."
        max_tokens = 50 if category == 'general' else 150
        
        success, response, response_time = await test_model_inference_advanced(
            'ollama', model_name, prompt, max_tokens, f"{category} model"
        )
        
        results.append({
            'model': model_name,
            'category': category,
            'success': success,
            'response_time': response_time,
            'error': response if not success else None
        })
        
        await asyncio.sleep(2)  # Longer pause for direct testing
    
    # Simple analysis
    successful = sum(1 for r in results if r['success'])
    print(f"\n📊 Direct Testing Results: {successful}/{len(results)} models working")
    
    for result in results:
        status = "✅" if result['success'] else "❌"
        time_str = f" ({result['response_time']:.2f}s)" if result['response_time'] else ""
        print(f"   {status} {result['model']}{time_str}")

async def analyze_model_collection(test_results):
    """Special analysis for your specific model collection"""
    print_header("🏆 YOUR MODEL COLLECTION ANALYSIS")
    
    # Group by size categories
    size_categories = {
        'Large (>10GB)': [],
        'Medium (5-10GB)': [],
        'Small (<5GB)': []
    }
    
    # Known sizes from your ollama list
    model_sizes = {
        'gpt-oss:latest': 13.0,
        'mistral-small:latest': 14.0,
        'mistral-nemo:latest': 7.1,
        'qwen3:latest': 5.2,
        'granite3.3:latest': 4.9,
        'llama3.1:latest': 4.7,
        'mistral:latest': 4.1,
        'gemma3:latest': 3.3,
        'llama3.2:latest': 2.0
    }
    
    # Categorize results by size
    for result in test_results:
        if result['provider'] == 'ollama':
            model_name = result['model']
            size_gb = model_sizes.get(model_name, 0)
            
            if size_gb > 10:
                size_categories['Large (>10GB)'].append((result, size_gb))
            elif size_gb >= 5:
                size_categories['Medium (5-10GB)'].append((result, size_gb))
            else:
                size_categories['Small (<5GB)'].append((result, size_gb))
    
    # Analysis by size category
    for category, models in size_categories.items():
        if models:
            successful = sum(1 for result, _ in models if result['success'])
            total = len(models)
            avg_time = sum(result['response_time'] for result, _ in models 
                          if result['success'] and result['response_time']) / max(1, successful)
            
            print(f"\n📏 {category} Models: {successful}/{total} working")
            if successful > 0:
                print(f"   ⏱️  Average response time: {avg_time:.2f}s")
                
                # Best performer in category
                best_model = min([r for r, _ in models if r['success'] and r['response_time']], 
                               key=lambda x: x['response_time'], default=None)
                if best_model:
                    size = model_sizes.get(best_model['model'], 0)
                    print(f"   🏆 Fastest: {best_model['model']} ({best_model['response_time']:.2f}s, {size}GB)")
    
    # Efficiency analysis (response time per GB)
    print(f"\n⚡ Efficiency Rankings (speed per GB):")
    efficiency_scores = []
    
    for result in test_results:
        if result['provider'] == 'ollama' and result['success'] and result['response_time']:
            model_name = result['model']
            size_gb = model_sizes.get(model_name, 1)  # Avoid division by zero
            efficiency = size_gb / result['response_time']  # Higher = less efficient
            efficiency_scores.append((model_name, efficiency, result['response_time'], size_gb))
    
    # Sort by efficiency (lower score = more efficient)
    efficiency_scores.sort(key=lambda x: x[1])
    
    for i, (model_name, efficiency, resp_time, size_gb) in enumerate(efficiency_scores[:5], 1):
        print(f"   {i}. {model_name}: {resp_time:.2f}s for {size_gb}GB (efficiency: {efficiency:.2f})")
    
    # Recommendations for your collection
    print(f"\n💡 Recommendations for Your Collection:")
    
    if efficiency_scores:
        most_efficient = efficiency_scores[0][0]
        print(f"   🏃‍♂️ Most efficient: {most_efficient}")
    
    large_working = [r for r, s in size_categories['Large (>10GB)'] if r['success']]
    if large_working:
        print(f"   🦣 Large models working: {len(large_working)} available for complex tasks")
    
    small_working = [r for r, s in size_categories['Small (<5GB)'] if r['success']]
    if small_working:
        print(f"   🐰 Small models working: {len(small_working)} available for quick tasks")
    
    reasoning_models = ['granite3.3:latest', 'qwen3:latest', 'llama3.1:latest']
    reasoning_working = [r for r in test_results if r['model'] in reasoning_models and r['success']]
    if reasoning_working:
        print(f"   🧠 Reasoning capability: {len(reasoning_working)} models ready for complex thinking")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demo failed: {e}")
        traceback.print_exc()