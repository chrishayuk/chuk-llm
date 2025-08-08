#!/usr/bin/env python3
"""
Working ChukLLM Discovery Demo
==============================

This script demonstrates the discovery system working correctly with direct
discoverer instantiation, bypassing configuration issues.

Features:
- Direct Ollama model discovery with enhanced categorization
- OpenAI model discovery (when API key available)
- Universal inference engine for capability detection
- Model analysis and recommendations
- Configuration generation from discovered models
- Proper .env file loading with python-dotenv

This version works around configuration parsing issues by using the discoverers directly.
"""

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    
    # Look for .env file in current directory and parent directories
    env_file = None
    current_dir = Path.cwd()
    
    for path in [current_dir, *current_dir.parents]:
        potential_env = path / ".env"
        if potential_env.exists():
            env_file = potential_env
            break
    
    if env_file:
        load_dotenv(env_file)
        print(f"📁 Loaded environment from: {env_file}")
    else:
        load_dotenv()  # Try default locations
        print("📁 Loaded environment variables (default locations)")
        
except ImportError:
    print("💡 python-dotenv not available, using system environment variables")
    print("   Install with: pip install python-dotenv")
except Exception as e:
    print(f"⚠️  Error loading .env file: {e}")
    print("   Continuing with system environment variables...")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)

async def check_services():
    """Check available services"""
    print("🔍 Checking Available Services...")
    
    services = {"ollama": False, "openai": False}
    
    # Check Ollama
    try:
        import httpx
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get("http://localhost:11434/api/tags")
            response.raise_for_status()
            data = response.json()
            model_count = len(data.get("models", []))
            services["ollama"] = True
            print(f"✅ Ollama running with {model_count} models")
    except Exception as e:
        print(f"❌ Ollama not available: {e}")
        print("   💡 Start with: ollama serve")
    
    # Check OpenAI API key
    if os.getenv('OPENAI_API_KEY'):
        services["openai"] = True
        print("✅ OpenAI API key found")
    else:
        print("❌ OPENAI_API_KEY not set")
        print("   💡 Set with: export OPENAI_API_KEY=your_api_key")
    
    return services

async def demo_environment_info():
    """Show environment information and API key status"""
    print("\n" + "="*60)
    print("🌍 Environment Information")
    print("="*60)
    
    # Check environment variables
    env_vars = {
        'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
        'ANTHROPIC_API_KEY': os.getenv('ANTHROPIC_API_KEY'),
        'GROQ_API_KEY': os.getenv('GROQ_API_KEY'),
        'DEEPSEEK_API_KEY': os.getenv('DEEPSEEK_API_KEY'),
        'HUGGINGFACE_API_KEY': os.getenv('HUGGINGFACE_API_KEY'),
    }
    
    print("🔑 API Keys Status:")
    for var_name, value in env_vars.items():
        if value:
            # Show first 3 and last 4 characters for security
            masked_key = value[:3] + "*" * (len(value) - 7) + value[-4:] if len(value) > 10 else "*" * len(value)
            print(f"   ✅ {var_name}: {masked_key}")
        else:
            print(f"   ❌ {var_name}: Not set")
    
    # Show Python environment
    print(f"\n🐍 Python Environment:")
    print(f"   Python: {sys.version.split()[0]}")
    print(f"   Working directory: {Path.cwd()}")
    
    # Check for .env file
    env_files = []
    current_dir = Path.cwd()
    for path in [current_dir, *current_dir.parents]:
        env_file = path / ".env"
        if env_file.exists():
            env_files.append(env_file)
    
    if env_files:
        print(f"\n📁 Found .env files:")
        for env_file in env_files:
            print(f"   📄 {env_file}")
    else:
        print(f"\n📁 No .env files found")
        print(f"   💡 Create .env file with: echo 'OPENAI_API_KEY=your_key' > .env")

async def demo_ollama_direct_discovery():
    """Demonstrate direct Ollama discovery without config manager"""
    print("\n" + "="*60)
    print("🏠 Direct Ollama Discovery Demo")
    print("="*60)
    
    try:
        # Import the discoverer directly
        from chuk_llm.llm.discovery.ollama_discoverer import OllamaModelDiscoverer
        from chuk_llm.llm.discovery.engine import UniversalModelDiscoveryManager
        
        print("🔍 Initializing Ollama discoverer...")
        
        # Create discoverer directly
        ollama_discoverer = OllamaModelDiscoverer(
            provider_name="ollama",
            api_base="http://localhost:11434"
        )
        
        # Create universal manager with the discoverer
        discovery_manager = UniversalModelDiscoveryManager(
            provider_name="ollama",
            discoverer=ollama_discoverer
        )
        
        print("🔍 Discovering Ollama models...")
        start_time = time.time()
        
        discovered_models = await discovery_manager.discover_models(force_refresh=True)
        discovery_time = time.time() - start_time
        
        print(f"✅ Discovered {len(discovered_models)} models in {discovery_time:.2f}s")
        
        if not discovered_models:
            print("   💡 No models found. Try: ollama pull llama3.3")
            return []
        
        # Analyze discovered models with enhanced family detection
        def get_model_family(model):
            """Enhanced family detection with fallback logic"""
            family = model.family or model.metadata.get('model_family', 'unknown')
            if family == 'unknown':
                model_name_lower = model.name.lower()
                if 'llama' in model_name_lower:
                    return 'llama'
                elif 'qwen' in model_name_lower:
                    return 'qwen'
                elif 'granite' in model_name_lower:
                    return 'granite'
                elif 'mistral' in model_name_lower:
                    return 'mistral'
                elif 'gemma' in model_name_lower:
                    return 'gemma'
                elif 'phi' in model_name_lower:
                    return 'phi'
            return family
        
        families = {}
        specializations = {}
        reasoning_models = []
        vision_models = []
        code_models = []
        total_size_gb = 0
        
        for model in discovered_models:
            # Enhanced family analysis
            family = get_model_family(model)
            families[family] = families.get(family, 0) + 1
            
            # Specialization analysis
            specialization = model.metadata.get('specialization', 'general')
            specializations[specialization] = specializations.get(specialization, 0) + 1
            
            # Special capabilities
            if model.metadata.get('reasoning_capable', False):
                reasoning_models.append(model)
            
            if model.metadata.get('supports_vision', False):
                vision_models.append(model)
            
            if model.metadata.get('specialization') == 'code':
                code_models.append(model)
            
            # Size tracking
            size_gb = model.metadata.get('size_gb', 0)
            if size_gb:
                total_size_gb += size_gb
        
        print(f"\n📊 Discovery Analysis:")
        print(f"   🏷️  Model Families: {dict(sorted(families.items(), key=lambda x: x[1], reverse=True))}")
        print(f"   🎯 Specializations: {dict(sorted(specializations.items(), key=lambda x: x[1], reverse=True))}")
        print(f"   💾 Total Size: {total_size_gb:.1f} GB")
        print(f"   🧠 Reasoning Capable: {len(reasoning_models)}")
        print(f"   👁️  Vision Capable: {len(vision_models)}")
        print(f"   💻 Code Specialized: {len(code_models)}")
        
        # Show top models by capabilities
        print(f"\n🏆 Top Discovered Models:")
        
        # Sort by reasoning capability, then size, then family preference
        family_priority = {'llama': 0, 'qwen': 1, 'granite': 2, 'mistral': 3, 'gemma': 4}
        
        sorted_models = sorted(discovered_models, key=lambda m: (
            0 if m.metadata.get('reasoning_capable', False) else 1,  # Reasoning first
            -(m.metadata.get('size_gb', 0)),  # Larger models next
            family_priority.get(get_model_family(m), 10)  # Family preference
        ))
        
        for i, model in enumerate(sorted_models[:8], 1):
            family = get_model_family(model)
            specialization = model.metadata.get('specialization', 'general')
            size_gb = model.metadata.get('size_gb', 0)
            params = model.metadata.get('estimated_parameters', 'unknown')
            performance = model.metadata.get('performance_tier', 'unknown')
            
            # Icons for capabilities
            icons = []
            if model.metadata.get('reasoning_capable', False):
                icons.append("🧠")
            if model.metadata.get('supports_vision', False):
                icons.append("👁️")
            if model.metadata.get('supports_tools', False):
                icons.append("🔧")
            if model.metadata.get('supports_streaming', False):
                icons.append("⚡")
            
            icon_str = "".join(icons) if icons else "💬"
            
            print(f"   {i:2d}. {icon_str} {model.name}")
            print(f"       ├─ Family: {family} | Specialization: {specialization}")
            print(f"       ├─ Size: {size_gb:.1f}GB | Parameters: {params}")
            print(f"       ├─ Performance: {performance}")
            
            # Show inferred capabilities
            capabilities = model.capabilities
            if capabilities:
                cap_names = [f.value if hasattr(f, 'value') else str(f) for f in capabilities]
                cap_str = ", ".join(cap_names[:4])
                if len(cap_names) > 4:
                    cap_str += f" (+{len(cap_names)-4} more)"
                print(f"       └─ Capabilities: {cap_str}")
        
        # Show capability inference stats
        print(f"\n🔧 Capability Inference Analysis:")
        capability_counts = {}
        for model in discovered_models:
            for feature in model.capabilities:
                feature_name = feature.value if hasattr(feature, 'value') else str(feature)
                capability_counts[feature_name] = capability_counts.get(feature_name, 0) + 1
        
        for capability, count in sorted(capability_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(discovered_models)) * 100
            print(f"   📈 {capability}: {count}/{len(discovered_models)} models ({percentage:.0f}%)")
        
        return discovered_models
        
    except ImportError as e:
        print(f"❌ Ollama discoverer not available: {e}")
        return []
    except Exception as e:
        print(f"❌ Ollama discovery failed: {e}")
        return []

async def demo_enhanced_model_analysis(models: List):
    """Enhanced model analysis with better family detection and insights"""
    if not models:
        return
    
    print("\n" + "="*60)
    print("🔬 Enhanced Model Analysis")
    print("="*60)
    
    def get_enhanced_family(model):
        """Get family with enhanced detection"""
        family = model.family or model.metadata.get('model_family', 'unknown')
        if family == 'unknown':
            name_lower = model.name.lower()
            if 'llama' in name_lower:
                return 'llama'
            elif 'qwen' in name_lower:
                return 'qwen'  
            elif 'granite' in name_lower:
                return 'granite'
            elif 'mistral' in name_lower:
                return 'mistral'
            elif 'gemma' in name_lower:
                return 'gemma'
            elif 'phi' in name_lower:
                return 'phi'
        return family
    
    # Advanced analysis
    analysis = {
        'by_family': {},
        'by_size_tier': {'small': [], 'medium': [], 'large': [], 'xl': []},
        'by_capability': {},
        'recommendations': {}
    }
    
    for model in models:
        # Enhanced family detection
        family = get_enhanced_family(model)
        
        # Family grouping
        if family not in analysis['by_family']:
            analysis['by_family'][family] = []
        analysis['by_family'][family].append(model)
        
        # Size tier analysis
        size_gb = model.metadata.get('size_gb', 0)
        if size_gb < 2:
            analysis['by_size_tier']['small'].append(model)
        elif size_gb < 8:
            analysis['by_size_tier']['medium'].append(model)
        elif size_gb < 15:
            analysis['by_size_tier']['large'].append(model)
        else:
            analysis['by_size_tier']['xl'].append(model)
        
        # Capability analysis
        capabilities = model.capabilities or []
        for cap in capabilities:
            cap_name = cap.value if hasattr(cap, 'value') else str(cap)
            if cap_name not in analysis['by_capability']:
                analysis['by_capability'][cap_name] = []
            analysis['by_capability'][cap_name].append(model)
    
    # Display analysis
    print("🏷️  Family Distribution:")
    for family, family_models in sorted(analysis['by_family'].items(), key=lambda x: len(x[1]), reverse=True):
        if family == 'unknown':
            continue
        
        total_size = sum(m.metadata.get('size_gb', 0) for m in family_models)
        reasoning_count = sum(1 for m in family_models if m.metadata.get('reasoning_capable', False))
        
        print(f"   📦 {family}: {len(family_models)} models ({total_size:.1f}GB total)")
        if reasoning_count > 0:
            print(f"      └─ {reasoning_count} reasoning-capable")
    
    print(f"\n📏 Size Distribution:")
    for tier, tier_models in analysis['by_size_tier'].items():
        if tier_models:
            avg_size = sum(m.metadata.get('size_gb', 0) for m in tier_models) / len(tier_models)
            print(f"   📊 {tier.upper()}: {len(tier_models)} models (avg {avg_size:.1f}GB)")
    
    # Model recommendations
    print(f"\n🎯 Smart Recommendations:")
    
    # Best reasoning model
    reasoning_models = [m for m in models if m.metadata.get('reasoning_capable', False)]
    if reasoning_models:
        best_reasoning = max(reasoning_models, key=lambda m: m.metadata.get('size_gb', 0))
        print(f"   🧠 Best for reasoning: {best_reasoning.name} ({best_reasoning.metadata.get('size_gb', 0):.1f}GB)")
    
    # Most efficient model
    efficient_models = [m for m in models if 2 < m.metadata.get('size_gb', 0) < 8]
    if efficient_models:
        most_efficient = max(efficient_models, key=lambda m: len(m.capabilities or []))
        print(f"   ⚡ Most efficient: {most_efficient.name} ({most_efficient.metadata.get('size_gb', 0):.1f}GB)")
    
    # Largest model
    if models:
        largest = max(models, key=lambda m: m.metadata.get('size_gb', 0))
        print(f"   🏆 Most powerful: {largest.name} ({largest.metadata.get('size_gb', 0):.1f}GB)")
    
    # Family recommendations
    family_counts = {}
    for family, family_models in analysis['by_family'].items():
        if family != 'unknown':
            family_counts[family] = len(family_models)
    
    if family_counts:
        dominant_family = max(family_counts.items(), key=lambda x: x[1])
        print(f"   👑 Dominant family: {dominant_family[0]} ({dominant_family[1]} models)")

async def demo_openai_direct_discovery():
    """Demonstrate direct OpenAI discovery without config manager"""
    print("\n" + "="*60)
    print("🤖 Direct OpenAI Discovery Demo")
    print("="*60)
    
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ OPENAI_API_KEY not set - using fallback model list")
        await demo_openai_fallback()
        return []
    
    try:
        # Import the discoverer directly
        from chuk_llm.llm.discovery.openai_discoverer import OpenAIModelDiscoverer
        from chuk_llm.llm.discovery.engine import UniversalModelDiscoveryManager
        
        print("🔍 Initializing OpenAI discoverer...")
        
        # Create discoverer directly
        openai_discoverer = OpenAIModelDiscoverer(
            provider_name="openai",
            api_key=os.getenv('OPENAI_API_KEY'),
            api_base="https://api.openai.com/v1"
        )
        
        # Create universal manager
        discovery_manager = UniversalModelDiscoveryManager(
            provider_name="openai",
            discoverer=openai_discoverer
        )
        
        print("🔍 Discovering OpenAI models...")
        start_time = time.time()
        
        discovered_models = await discovery_manager.discover_models(force_refresh=True)
        discovery_time = time.time() - start_time
        
        print(f"✅ Discovered {len(discovered_models)} OpenAI models in {discovery_time:.2f}s")
        
        if not discovered_models:
            print("   💡 No models discovered - trying fallback list")
            await demo_openai_fallback()
            return []
        
        # Categorize models
        reasoning_models = [m for m in discovered_models if m.metadata.get('is_reasoning', False)]
        vision_models = [m for m in discovered_models if m.metadata.get('is_vision', False)]
        standard_models = [m for m in discovered_models if not m.metadata.get('is_reasoning', False) and not m.metadata.get('is_vision', False)]
        
        print(f"   🧠 Reasoning models: {len(reasoning_models)}")
        print(f"   👁️  Vision models: {len(vision_models)}")
        print(f"   💬 Standard models: {len(standard_models)}")
        
        # Show reasoning models (highest priority)
        if reasoning_models:
            print(f"\n🧠 Advanced Reasoning Models:")
            for model in reasoning_models[:5]:  # Top 5
                generation = model.metadata.get('generation', 'unknown')
                reasoning_type = model.metadata.get('reasoning_type', 'standard')
                performance_tier = model.metadata.get('performance_tier', 'medium')
                context_length = model.metadata.get('estimated_context_length', 0)
                max_output = model.metadata.get('estimated_max_output', 0)
                
                print(f"   🎯 {model.name}")
                print(f"      ├─ Generation: {generation}")
                print(f"      ├─ Reasoning: {reasoning_type}")
                print(f"      ├─ Performance: {performance_tier}")
                if context_length:
                    print(f"      ├─ Context: {context_length:,} tokens")
                if max_output:
                    print(f"      └─ Max Output: {max_output:,} tokens")
                
                # Show special parameter requirements
                param_reqs = model.metadata.get('parameter_requirements', {})
                if param_reqs:
                    requirements = []
                    if param_reqs.get('use_max_completion_tokens'):
                        requirements.append('max_completion_tokens')
                    if param_reqs.get('no_system_messages'):
                        requirements.append('no system messages')
                    if param_reqs.get('no_streaming'):
                        requirements.append('no streaming')
                    
                    if requirements:
                        print(f"         ⚠️  Requirements: {', '.join(requirements)}")
        
        # Show vision models
        if vision_models:
            print(f"\n👁️  Vision-Capable Models:")
            for model in vision_models[:3]:
                performance_tier = model.metadata.get('performance_tier', 'medium')
                context_length = model.metadata.get('estimated_context_length', 0)
                
                print(f"   🎯 {model.name}")
                print(f"      ├─ Performance: {performance_tier}")
                print(f"      └─ Context: {context_length:,} tokens" if context_length else "      └─ Vision: ✅ Supported")
        
        return discovered_models
        
    except ImportError as e:
        print(f"❌ OpenAI discoverer not available: {e}")
        await demo_openai_fallback()
        return []
    except Exception as e:
        print(f"❌ OpenAI discovery failed: {e}")
        print("   Falling back to known model list...")
        await demo_openai_fallback()
        return []

async def demo_openai_fallback():
    """Show OpenAI fallback models when API discovery fails"""
    print("\n📋 OpenAI Known Models (Fallback):")
    
    fallback_models = [
        {"name": "o1-mini", "type": "🧠 Reasoning", "features": "Chain-of-thought reasoning, no streaming"},
        {"name": "o1-preview", "type": "🧠 Reasoning", "features": "Advanced reasoning, 128K context"},
        {"name": "gpt-4o", "type": "👁️ Vision", "features": "Vision + text, 128K context"},
        {"name": "gpt-4o-mini", "type": "👁️ Vision", "features": "Efficient vision model, 128K context"},
        {"name": "gpt-4-turbo", "type": "💬 Standard", "features": "High performance, 128K context"},
        {"name": "gpt-3.5-turbo", "type": "💬 Standard", "features": "Fast and efficient, 16K context"}
    ]
    
    for i, model in enumerate(fallback_models, 1):
        print(f"   {i}. {model['type']} {model['name']}")
        print(f"      └─ {model['features']}")

async def demo_inference_engine():
    """Demonstrate the universal inference engine capabilities"""
    print("\n" + "="*60)
    print("🔧 Universal Inference Engine Demo")
    print("="*60)
    
    try:
        from chuk_llm.llm.discovery.engine import ConfigDrivenInferenceEngine, DiscoveredModel
        from chuk_llm.configuration import Feature
        
        print("🔍 Testing capability inference on sample models...")
        
        # Create a sample inference config
        sample_config = {
            "default_features": ["text", "streaming"],
            "default_context_length": 8192,
            "default_max_output_tokens": 4096,
            
            "family_rules": {
                "llama": {
                    "patterns": [r"llama"],
                    "features": ["text", "streaming", "tools", "reasoning"],
                    "base_context_length": 8192,
                    "context_rules": {
                        r"llama.*3\.1": 128000,
                        r"llama.*3\.3": 128000,
                    }
                },
                "reasoning": {
                    "patterns": [r"o1", r"o3", r"reasoning"],
                    "features": ["text", "reasoning"],
                    "base_context_length": 128000,
                    "restrictions": {
                        "no_streaming": True,
                        "no_system_messages": True
                    },
                    "special_params": {
                        "use_max_completion_tokens": True
                    }
                }
            }
        }
        
        # Create inference engine
        engine = ConfigDrivenInferenceEngine("test_provider", sample_config)
        
        # Test models
        test_models = [
            ("llama3.3:70b", {}),
            ("o1-mini", {}),
            ("qwen3:32b", {}),
            ("granite3.3:8b", {}),
            ("phi3:vision", {})
        ]
        
        print(f"\n🧪 Inference Test Results:")
        for model_name, metadata in test_models:
            # Create sample discovered model
            model = DiscoveredModel(
                name=model_name,
                provider="test_provider",
                metadata=metadata
            )
            
            # Apply inference
            inferred_model = engine.infer_capabilities(model)
            
            print(f"\n   🎯 {model_name}")
            print(f"      ├─ Family: {inferred_model.family}")
            
            # Show inferred capabilities
            caps = [f.value if hasattr(f, 'value') else str(f) for f in inferred_model.capabilities]
            print(f"      ├─ Capabilities: {', '.join(caps)}")
            print(f"      ├─ Context: {inferred_model.context_length:,} tokens")
            print(f"      └─ Max Output: {inferred_model.max_output_tokens:,} tokens")
            
            # Show special metadata if any
            if inferred_model.metadata.get('special_parameters'):
                special = inferred_model.metadata['special_parameters']
                print(f"         ⚠️  Special: {special}")
        
        print(f"\n✅ Inference engine successfully categorized all test models")
        
    except ImportError as e:
        print(f"❌ Inference engine not available: {e}")
    except Exception as e:
        print(f"❌ Inference engine demo failed: {e}")

async def demo_model_comparison(ollama_models: List = None, openai_models: List = None):
    """Compare discovered models across providers"""
    print("\n" + "="*60)
    print("⚖️  Cross-Provider Model Comparison")
    print("="*60)
    
    all_models = []
    if ollama_models:
        all_models.extend([("ollama", m) for m in ollama_models])
    if openai_models:
        all_models.extend([("openai", m) for m in openai_models])
    
    if not all_models:
        print("❌ No models available for comparison")
        return
    
    print(f"🔍 Comparing {len(all_models)} models across providers...")
    
    # Categorize by capability
    categories = {
        "reasoning": [],
        "vision": [],
        "code": [],
        "general": []
    }
    
    def get_enhanced_family(model):
        """Get family with enhanced detection"""
        family = model.family or model.metadata.get('model_family', 'unknown')
        if family == 'unknown':
            name_lower = model.name.lower()
            if 'llama' in name_lower:
                return 'llama'
            elif 'qwen' in name_lower:
                return 'qwen'
            elif 'granite' in name_lower:
                return 'granite'
            elif 'mistral' in name_lower:
                return 'mistral'
            elif 'gemma' in name_lower:
                return 'gemma'
        return family
    
    for provider, model in all_models:
        # Check capabilities
        is_reasoning = (
            model.metadata.get('reasoning_capable', False) or 
            model.metadata.get('is_reasoning', False) or
            'reasoning' in (model.family or '')
        )
        
        is_vision = (
            model.metadata.get('supports_vision', False) or
            model.metadata.get('is_vision', False) or
            'vision' in (model.family or '')
        )
        
        is_code = (
            model.metadata.get('specialization') == 'code' or
            model.metadata.get('is_code', False) or
            'code' in (model.family or '')
        )
        
        model_info = {
            'provider': provider,
            'model': model,
            'name': model.name,
            'family': get_enhanced_family(model)
        }
        
        if is_reasoning:
            categories["reasoning"].append(model_info)
        elif is_vision:
            categories["vision"].append(model_info)
        elif is_code:
            categories["code"].append(model_info)
        else:
            categories["general"].append(model_info)
    
    # Show comparison by category
    for category, models in categories.items():
        if not models:
            continue
            
        icon_map = {
            "reasoning": "🧠",
            "vision": "👁️",
            "code": "💻", 
            "general": "💬"
        }
        
        print(f"\n{icon_map[category]} {category.upper()} Models ({len(models)}):")
        
        # Sort by provider for easy comparison
        models.sort(key=lambda x: (x['provider'], x['family']))
        
        for model_info in models:
            provider = model_info['provider']
            model = model_info['model']
            name = model_info['name']
            family = model_info['family']
            
            # Provider-specific details
            if provider == "ollama":
                size = model.metadata.get('size_gb', 0)
                params = model.metadata.get('estimated_parameters', 'unknown')
                print(f"   🏠 {name} ({family})")
                print(f"      └─ Size: {size:.1f}GB | Params: {params}")
            
            elif provider == "openai":
                generation = model.metadata.get('generation', 'unknown')
                performance = model.metadata.get('performance_tier', 'unknown')
                context = model.metadata.get('estimated_context_length', 0)
                print(f"   ☁️  {name} ({generation})")
                print(f"      └─ Performance: {performance} | Context: {context:,}" + (" tokens" if context else ""))
    
    # Summary comparison
    print(f"\n📊 Provider Summary:")
    ollama_count = len([m for p, m in all_models if p == "ollama"])
    openai_count = len([m for p, m in all_models if p == "openai"])
    
    if ollama_count > 0:
        print(f"   🏠 Ollama: {ollama_count} local models (flexible, private, no API costs)")
    if openai_count > 0:
        print(f"   ☁️  OpenAI: {openai_count} cloud models (cutting-edge, scalable, API required)")

async def demo_practical_usage():
    """Demonstrate practical usage patterns"""
    print("\n" + "="*60)
    print("💡 Practical Usage Examples")
    print("="*60)
    
    print("🔧 How to use discovery in your code:")
    print()
    
    code_examples = [
        {
            "title": "Basic Discovery",
            "code": '''# Discover all Ollama models
from chuk_llm.llm.discovery.ollama_discoverer import OllamaModelDiscoverer
from chuk_llm.llm.discovery.engine import UniversalModelDiscoveryManager

discoverer = OllamaModelDiscoverer()
manager = UniversalModelDiscoveryManager("ollama", discoverer)
models = await manager.discover_models()'''
        },
        {
            "title": "Find Reasoning Models",
            "code": '''# Find models good for reasoning tasks
reasoning_models = [
    m for m in models 
    if m.metadata.get('reasoning_capable', False)
]

best_reasoning = max(reasoning_models, 
    key=lambda m: m.metadata.get('size_gb', 0))
print(f"Best reasoning model: {best_reasoning.name}")'''
        },
        {
            "title": "Integration with ChukLLM",
            "code": '''# Use discovered model with ChukLLM
from chuk_llm import ask

# The model name from discovery works directly
response = await ask(
    "Explain quantum computing",
    provider="ollama", 
    model=best_reasoning.name,
    max_tokens=500
)'''
        }
    ]
    
    for i, example in enumerate(code_examples, 1):
        print(f"{i}. {example['title']}:")
        print("```python")
        print(example['code'])
        print("```")
        print()
    
    print("💡 Key Benefits:")
    print("   • Automatic model categorization and capability detection")
    print("   • Seamless integration with existing ChukLLM APIs")
    print("   • Cross-provider compatibility and standardization")
    print("   • Dynamic model availability without manual configuration")

async def main():
    """Main demo function"""
    print("🚀 ChukLLM Discovery System - Working Demo")
    print("=" * 60)
    print("Direct discoverer demonstration without configuration dependencies")
    print()
    
    # Check what services are available
    services = await check_services()
    
    if not any(services.values()):
        print("\n❌ No services available - please set up Ollama or OpenAI API key")
        return
    
    # Show environment information
    await demo_environment_info()
    
    print("\n🎬 Starting Discovery Demos\n")
    
    ollama_models = []
    openai_models = []
    
    # Demo Ollama if available
    if services["ollama"]:
        ollama_models = await demo_ollama_direct_discovery()
        
        # Enhanced analysis for Ollama models
        if ollama_models:
            await demo_enhanced_model_analysis(ollama_models)
    
    # Demo OpenAI if available  
    if services["openai"]:
        openai_models = await demo_openai_direct_discovery()
    
    # Show inference engine capabilities
    await demo_inference_engine()
    
    # Compare models across providers
    await demo_model_comparison(ollama_models, openai_models)
    
    # Show practical usage
    await demo_practical_usage()
    
    # Summary
    print("\n" + "="*60)
    print("🎉 Working Discovery Demo Complete!")
    print("="*60)
    print()
    total_models = len(ollama_models) + len(openai_models)
    print(f"✅ Successfully discovered {total_models} models")
    if ollama_models:
        print(f"   🏠 Ollama: {len(ollama_models)} local models")
    if openai_models:
        print(f"   ☁️  OpenAI: {len(openai_models)} cloud models")
    
    print()
    print("🔧 The discovery system provides:")
    print("   • Automatic model detection and categorization")
    print("   • Universal capability inference across providers")  
    print("   • Reasoning model detection with special parameter handling")
    print("   • Cross-provider compatibility and comparison")
    print("   • Direct integration with ChukLLM APIs")
    print()
    print("💡 This demo bypasses configuration issues by using discoverers directly.")
    print("💡 In production, enable discovery through your chuk_llm.yaml configuration.")

if __name__ == "__main__":
    asyncio.run(main())