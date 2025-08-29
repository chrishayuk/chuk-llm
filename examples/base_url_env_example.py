#!/usr/bin/env python3
"""
Base URL from Environment Variables Example
===========================================

This example demonstrates how to configure provider base URLs using environment variables.
This is particularly useful for:
- Using proxy servers (e.g., corporate proxies)
- Switching between different API endpoints (dev/staging/prod)
- Using OpenAI-compatible providers (e.g., Perplexity, Together AI, Anyscale)
- Local development with mock servers
"""

import os
from chuk_llm import (
    ask_sync,
    register_provider,
    register_openai_compatible,
    get_config,
    unregister_provider
)

def demo_standard_env_patterns():
    """
    Demonstrate standard environment variable patterns.
    
    For a provider named 'openai', ChukLLM automatically checks:
    - OPENAI_API_BASE
    - OPENAI_BASE_URL
    - OPENAI_API_URL
    - OPENAI_ENDPOINT
    """
    print("\n📋 STANDARD ENVIRONMENT PATTERNS")
    print("-" * 40)
    
    # Set example environment variable
    original_base = os.getenv("OPENAI_API_BASE")
    os.environ["OPENAI_API_BASE"] = "https://proxy.company.com/openai/v1"
    
    print("Environment variable set:")
    print(f"  OPENAI_API_BASE = {os.environ['OPENAI_API_BASE']}")
    
    # The base URL will be automatically picked up
    config = get_config()
    base_url = config.get_api_base("openai")
    print(f"\nResolved base URL for 'openai': {base_url}")
    
    # Restore original
    if original_base:
        os.environ["OPENAI_API_BASE"] = original_base
    else:
        del os.environ["OPENAI_API_BASE"]
    
    print("\n💡 ChukLLM automatically checks these patterns:")
    print("   - {PROVIDER}_API_BASE")
    print("   - {PROVIDER}_BASE_URL")
    print("   - {PROVIDER}_API_URL")
    print("   - {PROVIDER}_ENDPOINT")

def demo_custom_env_variable():
    """
    Demonstrate using a custom environment variable name.
    """
    print("\n🔧 CUSTOM ENVIRONMENT VARIABLE")
    print("-" * 40)
    
    # Set a custom environment variable
    os.environ["COMPANY_LLM_ENDPOINT"] = "https://llm.company.internal/v1"
    os.environ["COMPANY_LLM_KEY"] = "company-key-123"
    
    print("Custom environment variables:")
    print(f"  COMPANY_LLM_ENDPOINT = {os.environ['COMPANY_LLM_ENDPOINT']}")
    print(f"  COMPANY_LLM_KEY = {os.environ['COMPANY_LLM_KEY']}")
    
    # Register provider with custom env variable names
    provider = register_openai_compatible(
        name="company_llm",
        api_base_env="COMPANY_LLM_ENDPOINT",  # Will check this env var
        api_key_env="COMPANY_LLM_KEY",         # Will check this env var
        models=["gpt-3.5-turbo", "gpt-4"],
        default_model="gpt-3.5-turbo"
    )
    
    print(f"\n✅ Registered provider with custom env vars")
    
    # Verify it picks up the environment values
    config = get_config()
    base_url = config.get_api_base("company_llm")
    api_key = config.get_api_key("company_llm")
    
    print(f"Resolved base URL: {base_url}")
    print(f"API key found: {'Yes' if api_key else 'No'}")
    
    # Clean up
    unregister_provider("company_llm")
    del os.environ["COMPANY_LLM_ENDPOINT"]
    del os.environ["COMPANY_LLM_KEY"]

def demo_proxy_server():
    """
    Demonstrate using a proxy server for OpenAI.
    """
    print("\n🔐 PROXY SERVER CONFIGURATION")
    print("-" * 40)
    
    # Example: Corporate proxy that adds auth headers
    proxy_url = "https://api-proxy.company.com/openai/v1"
    
    print(f"Setting up proxy: {proxy_url}")
    os.environ["OPENAI_API_BASE"] = proxy_url
    
    # The standard OpenAI provider will now use the proxy
    if os.getenv("OPENAI_API_KEY"):
        try:
            response = ask_sync(
                "Say 'Proxy works!' in 5 words or less",
                provider="openai",
                temperature=0,
                max_tokens=10
            )
            print(f"✅ Response via proxy: {response}")
        except Exception as e:
            print(f"⚠️  Proxy test failed: {e}")
    else:
        print("⚠️  Set OPENAI_API_KEY to test proxy")
    
    # Clean up
    if "OPENAI_API_BASE" in os.environ:
        del os.environ["OPENAI_API_BASE"]

def demo_openai_compatible_providers():
    """
    Demonstrate using OpenAI-compatible providers via base URL.
    """
    print("\n🔄 OPENAI-COMPATIBLE PROVIDERS")
    print("-" * 40)
    
    # Examples of OpenAI-compatible services
    compatible_services = [
        {
            "name": "Perplexity AI",
            "base_url": "https://api.perplexity.ai",
            "env_var": "PERPLEXITY_API_BASE",
            "models": ["pplx-7b-online", "pplx-70b-online"]
        },
        {
            "name": "Together AI",
            "base_url": "https://api.together.xyz/v1",
            "env_var": "TOGETHER_API_BASE",
            "models": ["mistralai/Mixtral-8x7B-Instruct-v0.1"]
        },
        {
            "name": "Anyscale",
            "base_url": "https://api.endpoints.anyscale.com/v1",
            "env_var": "ANYSCALE_API_BASE",
            "models": ["meta-llama/Llama-2-70b-chat-hf"]
        },
        {
            "name": "Local LM Studio",
            "base_url": "http://localhost:1234/v1",
            "env_var": "LM_STUDIO_API_BASE",
            "models": ["local-model"]
        }
    ]
    
    for service in compatible_services:
        print(f"\n{service['name']}:")
        print(f"  Base URL: {service['base_url']}")
        print(f"  Env var: {service['env_var']}")
        print(f"  Models: {', '.join(service['models'][:2])}")

def demo_environment_override():
    """
    Demonstrate environment variable override priority.
    """
    print("\n🎯 ENVIRONMENT OVERRIDE PRIORITY")
    print("-" * 40)
    
    # Register a provider with a configured base URL
    provider = register_openai_compatible(
        name="test_provider",
        api_base="https://configured.api.com/v1",
        api_base_env="TEST_PROVIDER_OVERRIDE",
        models=["model-1"],
        default_model="model-1"
    )
    
    config = get_config()
    
    # Without environment variable
    base_url = config.get_api_base("test_provider")
    print(f"Without env override: {base_url}")
    
    # With environment variable (takes priority)
    os.environ["TEST_PROVIDER_OVERRIDE"] = "https://override.api.com/v1"
    base_url = config.get_api_base("test_provider")
    print(f"With env override: {base_url}")
    
    # Standard pattern also works
    os.environ["TEST_PROVIDER_API_BASE"] = "https://standard.api.com/v1"
    del os.environ["TEST_PROVIDER_OVERRIDE"]
    base_url = config.get_api_base("test_provider")
    print(f"With standard pattern: {base_url}")
    
    print("\n💡 Priority order:")
    print("   1. Custom env var (api_base_env)")
    print("   2. Standard patterns ({PROVIDER}_API_BASE, etc.)")
    print("   3. Configured api_base")
    
    # Clean up
    unregister_provider("test_provider")
    if "TEST_PROVIDER_API_BASE" in os.environ:
        del os.environ["TEST_PROVIDER_API_BASE"]

def demo_multi_environment_setup():
    """
    Demonstrate multi-environment setup (dev/staging/prod).
    """
    print("\n🚀 MULTI-ENVIRONMENT SETUP")
    print("-" * 40)
    
    # Determine environment
    env = os.getenv("APP_ENV", "development")
    
    print(f"Current environment: {env}")
    
    # Environment-specific configurations
    env_configs = {
        "development": {
            "base_url": "http://localhost:8000/v1",
            "api_key": "dev-key"
        },
        "staging": {
            "base_url": "https://staging-api.company.com/v1",
            "api_key": "staging-key"
        },
        "production": {
            "base_url": "https://api.company.com/v1",
            "api_key": "prod-key"
        }
    }
    
    # Set environment variables based on current environment
    config = env_configs.get(env, env_configs["development"])
    os.environ["APP_LLM_BASE"] = config["base_url"]
    os.environ["APP_LLM_KEY"] = config["api_key"]
    
    print(f"Configuration for {env}:")
    print(f"  Base URL: {config['base_url']}")
    print(f"  API Key: {'***' + config['api_key'][-4:]}")
    
    # Register provider using environment
    provider = register_openai_compatible(
        name="app_llm",
        api_base_env="APP_LLM_BASE",
        api_key_env="APP_LLM_KEY",
        models=["gpt-3.5-turbo"],
        default_model="gpt-3.5-turbo"
    )
    
    print(f"\n✅ Provider configured for {env} environment")
    
    # Clean up
    unregister_provider("app_llm")
    del os.environ["APP_LLM_BASE"]
    del os.environ["APP_LLM_KEY"]

def main():
    print("=" * 60)
    print("BASE URL FROM ENVIRONMENT VARIABLES")
    print("=" * 60)
    
    # Run demonstrations
    demo_standard_env_patterns()
    demo_custom_env_variable()
    demo_proxy_server()
    demo_openai_compatible_providers()
    demo_environment_override()
    demo_multi_environment_setup()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
Base URL environment variable support enables:

✅ Automatic Detection
   - Standard patterns like {PROVIDER}_API_BASE
   - No code changes needed

✅ Custom Variables
   - Specify custom env var names with api_base_env
   - Full control over naming

✅ Proxy Support
   - Route through corporate proxies
   - Add authentication layers

✅ OpenAI-Compatible Services
   - Use Perplexity, Together AI, Anyscale, etc.
   - Local LM Studio or Ollama

✅ Multi-Environment
   - Different endpoints for dev/staging/prod
   - Environment-specific configurations

✅ Override Priority
   1. Runtime registration (highest)
   2. Custom env variable
   3. Standard env patterns
   4. Configuration file (lowest)

Best Practices:
- Use environment variables for sensitive data
- Document required environment variables
- Provide sensible defaults
- Use consistent naming patterns
- Validate environment on startup
""")

if __name__ == "__main__":
    main()