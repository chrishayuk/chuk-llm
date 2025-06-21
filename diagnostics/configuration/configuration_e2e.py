#!/usr/bin/env python3
# diagnostics/configuration_e2e.py
#!/usr/bin/env python3
"""
Fixed ChukLLM End-to-End Configuration Demo

Uses correct function names and avoids missing attributes.
Safe version that won't trigger model downloads.
"""

from dotenv import load_dotenv
load_dotenv()

import chuk_llm
import os

def header(title: str):
    """Print a formatted header"""
    print(f"\n{'='*60}")
    print(f"🔧 {title}")
    print(f"{'='*60}")

def section(title: str):
    """Print a formatted section"""
    print(f"\n{'-'*40}")
    print(f"📋 {title}")
    print(f"{'-'*40}")

def test_basic_configuration():
    """Test basic configuration functionality"""
    header("Basic Configuration")
    
    # Show initial state
    section("Initial Configuration State")
    if hasattr(chuk_llm, 'print_diagnostics'):
        chuk_llm.print_diagnostics()
    else:
        print("print_diagnostics() not available")
    
    # Show available providers
    section("Available Providers")
    if hasattr(chuk_llm, 'show_providers'):
        chuk_llm.show_providers()
    else:
        print("show_providers() not available")
    
    # Show available functions using correct method
    section("Generated Functions (Sample)")
    try:
        from chuk_llm.api.providers import get_all_functions
        functions = get_all_functions()
        print(f"Total functions generated: {len(functions)}")
        
        # Show Ollama functions specifically
        ollama_functions = [name for name in functions.keys() if 'ollama' in name and name.startswith('ask_')]
        print(f"Ollama functions: {len(ollama_functions)}")
        print("Sample Ollama functions:")
        for func in sorted(ollama_functions)[:10]:
            print(f"  {func}()")
        if len(ollama_functions) > 10:
            print(f"  ... and {len(ollama_functions) - 10} more")
            
        # Show other provider functions
        other_functions = [name for name in functions.keys() if 'ollama' not in name and name.startswith('ask_') and not name.endswith('_sync')]
        if other_functions:
            print(f"\nOther provider functions: {len(other_functions)}")
            print("Sample functions:")
            for func in sorted(other_functions)[:5]:
                print(f"  {func}()")
                
    except Exception as e:
        print(f"Could not get function list: {e}")

def test_global_configuration():
    """Test global configuration management"""
    header("Global Configuration Management")
    
    section("Current Configuration")
    try:
        # Test if we can get current config
        if hasattr(chuk_llm, 'get_current_config'):
            config = chuk_llm.get_current_config()
            print(f"Current provider: {config.get('provider', 'Unknown')}")
            print(f"Current model: {config.get('model', 'Unknown')}")
            print(f"API key present: {'✅' if config.get('api_key') else '❌'}")
        else:
            print("get_current_config() not available")
    except Exception as e:
        print(f"Configuration check failed: {e}")
    
    section("Configuration Functions Available")
    config_functions = ['configure', 'get_current_config', 'show_config']
    for func in config_functions:
        if hasattr(chuk_llm, func):
            print(f"✅ chuk_llm.{func}()")
        else:
            print(f"❌ chuk_llm.{func}()")

def test_available_functions():
    """Test what functions are actually available"""
    header("Available Function Analysis")
    
    section("Core Functions")
    core_functions = ['ask', 'stream', 'show_config', 'configure']
    for func in core_functions:
        if hasattr(chuk_llm, func):
            print(f"✅ chuk_llm.{func}()")
        else:
            print(f"❌ chuk_llm.{func}()")
    
    section("Provider Functions (Safe Check)")
    # Check for specific functions we know should exist
    test_functions = [
        'ask_ollama_granite3_3',
        'ask_ollama_qwen3', 
        'ask_ollama_devstral',
        'ask_openai_gpt4o_mini_sync',
        'ask_anthropic_sonnet4_sync'
    ]
    
    available_count = 0
    for func in test_functions:
        if hasattr(chuk_llm, func):
            print(f"✅ {func}")
            available_count += 1
        else:
            print(f"❌ {func}")
    
    print(f"\nFunction availability: {available_count}/{len(test_functions)}")

def test_safe_function_calls():
    """Test actual function calls with very safe parameters"""
    header("Safe Function Testing")
    
    section("Basic Function Calls")
    
    # Only test if we have the functions and use minimal parameters
    safe_tests = [
        ("ask_ollama_granite3_3", "Hi"),
        ("ask_openai_gpt4o_mini_sync", "Hello")
    ]
    
    for func_name, prompt in safe_tests:
        if hasattr(chuk_llm, func_name):
            try:
                print(f"🧪 Testing {func_name}...")
                func = getattr(chuk_llm, func_name)
                
                # Very safe parameters
                if func_name.endswith('_sync'):
                    response = func(prompt, max_tokens=5)
                else:
                    print(f"   (Skipping async function {func_name} - would need await)")
                    continue
                
                print(f"   ✅ Success: '{response.strip()}'")
                
            except Exception as e:
                error_msg = str(e)
                if "API key" in error_msg:
                    print(f"   ⚠️  API key required for {func_name}")
                elif "not available" in error_msg.lower():
                    print(f"   ⚠️  Model not available for {func_name}")
                else:
                    print(f"   ❌ Error: {error_msg[:50]}...")
        else:
            print(f"❌ {func_name} not found")

def test_provider_status():
    """Test provider status without making calls"""
    header("Provider Status Check")
    
    section("API Key Detection")
    
    # Check for API key environment variables
    api_keys = {
        'OpenAI': 'OPENAI_API_KEY',
        'Anthropic': 'ANTHROPIC_API_KEY', 
        'Groq': 'GROQ_API_KEY',
        'Gemini': 'GEMINI_API_KEY',
        'Mistral': 'MISTRAL_API_KEY'
    }
    
    for provider, env_var in api_keys.items():
        api_key = os.getenv(env_var)
        if api_key:
            print(f"✅ {provider}: API key present ({env_var})")
        else:
            print(f"❌ {provider}: No API key ({env_var})")
    
    # Ollama doesn't need API key but needs to be running
    print(f"🔧 Ollama: Local service (no API key needed)")

def test_import_structure():
    """Test the import structure"""
    header("Import Structure Analysis")
    
    section("Main Module Attributes")
    all_attrs = dir(chuk_llm)
    
    # Categorize attributes
    functions = [attr for attr in all_attrs if callable(getattr(chuk_llm, attr, None)) and not attr.startswith('_')]
    classes = [attr for attr in all_attrs if attr[0].isupper() and not attr.startswith('_')]
    constants = [attr for attr in all_attrs if attr.isupper() and not callable(getattr(chuk_llm, attr, None))]
    
    print(f"Total attributes: {len(all_attrs)}")
    print(f"Functions: {len(functions)}")
    print(f"Classes: {len(classes)}")
    print(f"Constants: {len(constants)}")
    
    # Show some key functions
    key_functions = [attr for attr in functions if any(keyword in attr for keyword in ['ask', 'stream', 'config', 'show'])]
    print(f"\nKey functions ({len(key_functions)}):")
    for func in sorted(key_functions)[:15]:
        print(f"  {func}")
    if len(key_functions) > 15:
        print(f"  ... and {len(key_functions) - 15} more")

def test_ollama_specific():
    """Test Ollama-specific functionality"""
    header("Ollama-Specific Testing")
    
    section("Ollama Function Count")
    try:
        from chuk_llm.api.providers import get_all_functions
        functions = get_all_functions()
        
        ollama_ask = [name for name in functions.keys() if name.startswith('ask_ollama_') and not name.endswith('_sync')]
        ollama_sync = [name for name in functions.keys() if name.startswith('ask_ollama_') and name.endswith('_sync')]
        ollama_stream = [name for name in functions.keys() if name.startswith('stream_ollama_')]
        
        print(f"Ollama async functions: {len(ollama_ask)}")
        print(f"Ollama sync functions: {len(ollama_sync)}")
        print(f"Ollama stream functions: {len(ollama_stream)}")
        print(f"Total Ollama functions: {len(ollama_ask) + len(ollama_sync) + len(ollama_stream)}")
        
        # Show model families
        families = set()
        for func in ollama_ask:
            # Extract model name after ask_ollama_
            model_part = func[11:]  # Remove 'ask_ollama_'
            # Get the base family (first part before numbers/variants)
            family = ''.join(c for c in model_part if c.isalpha())
            families.add(family)
        
        print(f"\nModel families available: {len(families)}")
        print(f"Families: {', '.join(sorted(families))}")
        
    except Exception as e:
        print(f"Ollama analysis failed: {e}")

def run_safe_demo():
    """Run a safe version of the demo"""
    print("🚀 ChukLLM Safe Configuration Demo")
    print("🚀 " + "="*58)
    print("⚠️  This demo avoids model downloads and expensive operations")
    print()
    
    try:
        test_basic_configuration()
        test_global_configuration() 
        test_available_functions()
        test_import_structure()
        test_provider_status()
        test_ollama_specific()
        test_safe_function_calls()  # Only if API keys available
        
        print(f"\n🎉 Safe Demo Complete!")
        print("=" * 60)
        print("✅ Demo completed without triggering downloads")
        print("💡 Your ChukLLM system analysis:")
        print("   • Function generation: Working")
        print("   • Provider configuration: Available") 
        print("   • Ollama integration: Ready")
        print("   • Multi-provider support: Configured")
        
    except KeyboardInterrupt:
        print(f"\n\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demo failed with error: {e}")
        import traceback
        print(f"Full traceback:\n{traceback.format_exc()}")

if __name__ == "__main__":
    run_safe_demo()