#!/usr/bin/env python3
"""
Quick fix script for diagnostic issues.
Run this to address the main problems identified in your diagnostic output.
"""

import os
import sys
from pathlib import Path

def main():
    print("🔧 Fixing ChukLLM diagnostic issues...")
    
    # Load .env file if it exists
    try:
        from dotenv import load_dotenv
        if Path(".env").exists():
            load_dotenv()
            print("✅ Loaded .env file")
        else:
            print("⚠️  No .env file found")
    except ImportError:
        print("⚠️  python-dotenv not installed, skipping .env loading")
    
    # 1. Check for missing API keys (check multiple possible names)
    missing_keys = []
    required_keys = {
        # Check both common environment variable names
        ("GEMINI_API_KEY"): "gemini",
        ("MISTRAL_API_KEY",): "mistral", 
        ("ANTHROPIC_API_KEY",): "anthropic",
        ("OPENAI_API_KEY",): "openai",
        ("GROQ_API_KEY",): "groq",
        ("DEEPSEEK_API_KEY",): "deepseek",
        ("PERPLEXITY_API_KEY",): "perplexity",
        ("WATSONX_API_KEY", "IBM_CLOUD_API_KEY"): "watsonx"
    }
    
    for possible_keys, provider in required_keys.items():
        # Check if any of the possible keys exist
        found_key = False
        for key in possible_keys:
            if os.getenv(key):
                found_key = True
                break
        
        if not found_key:
            missing_keys.append((possible_keys[0], provider))
    
    if missing_keys:
        print("\n⚠️  Missing API keys:")
        for key, provider in missing_keys:
            print(f"   - {key} (for {provider})")
        print("\nTo fix: Set these environment variables or add to .env file")
        print("Example: export GROQ_API_KEY=your_key_here")
        
        # Show .env file example
        print("\n📝 Example .env file content:")
        print("# Add these to your .env file:")
        for key, provider in missing_keys:
            print(f"{key}=your-{provider}-key-here")
    else:
        print("✅ All API keys found!")
    
    # 2. Check model capability configuration
    print("\n🔍 Checking model capabilities...")
    
    try:
        from chuk_llm.configuration.unified_config import get_config
        config = get_config()
        
        # Check Mistral vision configuration
        mistral_config = config.get_provider("mistral")
        magistral_has_vision = mistral_config.supports_feature("vision", "magistral-medium-2506")
        
        if magistral_has_vision:
            print("❌ Issue: magistral-medium-2506 incorrectly marked as having vision")
            print("   Fix: Update your chuk_llm.yaml model_capabilities section")
            print("   The Magistral models are reasoning-only, not multimodal")
        else:
            print("✅ Magistral vision capability correctly configured")
            
        # Check for vision models
        vision_models = []
        for model in mistral_config.models:
            if mistral_config.supports_feature("vision", model):
                vision_models.append(model)
        
        print(f"✅ Mistral vision models: {vision_models}")
        
    except Exception as e:
        print(f"❌ Config check failed: {e}")
    
    # 3. Check Ollama status (if applicable)
    print("\n🐋 Checking Ollama status...")
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            models = response.json().get("models", [])
            print(f"✅ Ollama running with {len(models)} models")
        else:
            print("⚠️  Ollama API responded with error")
    except Exception:
        print("⚠️  Ollama not running or not accessible")
        print("   To fix: Start Ollama with 'ollama serve'")
    
    # 4. Provide diagnostic command recommendations
    print("\n💡 Recommended diagnostic commands:")
    print("   # Test without problematic providers:")
    print("   uv run diagnostics/capabilities/llm_diagnostics.py --providers openai anthropic groq")
    print("   ")
    print("   # Test with specific model overrides:")
    print("   uv run diagnostics/capabilities/llm_diagnostics.py --model 'mistral:vision=pixtral-large-2411'")
    print("   ")
    print("   # Quick test without vision:")
    print("   uv run diagnostics/capabilities/llm_diagnostics.py --skip-image")
    print("   ")
    print("   # Test just the working providers:")
    print("   uv run diagnostics/capabilities/llm_diagnostics.py --providers openai anthropic groq watsonx")
    
    # 5. Check file structure
    print("\n📁 Checking file structure...")
    expected_files = [
        "diagnostics/capabilities/utils/result_models.py",
        "diagnostics/capabilities/utils/test_runners.py", 
        "diagnostics/capabilities/utils/provider_configs.py",
        "diagnostics/capabilities/utils/display_utils.py"
    ]
    
    for file_path in expected_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ Missing: {file_path}")
    
    print("\n🎯 Summary of main issues from your output:")
    print("1. Gemini fails due to missing GOOGLE_API_KEY")
    print("2. Mistral magistral-medium-2506 incorrectly configured with vision")
    print("3. Perplexity has limited streaming/tools support")
    print("4. DeepSeek streaming+tools combination has issues")
    
    print("\n✅ Apply the fixed files above to resolve most issues!")

if __name__ == "__main__":
    main()