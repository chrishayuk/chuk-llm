#!/usr/bin/env python3
"""
Environment-Based Dynamic Provider Configuration
================================================

This example shows how to configure dynamic providers using environment variables.
Perfect for containerized deployments, CI/CD pipelines, and multi-environment setups.
"""

import os
import sys
from typing import Optional, Dict, List
from chuk_llm import (
    register_openai_compatible,
    register_provider,
    unregister_provider,
    list_dynamic_providers,
    provider_exists,
    ask_sync,
    stream_sync_iter
)

def load_provider_from_env(
    prefix: str = "LLM",
    provider_name: Optional[str] = None
) -> Optional[Dict]:
    """
    Load provider configuration from environment variables.
    
    Environment variables pattern:
    - {PREFIX}_PROVIDER_NAME: Name for the provider
    - {PREFIX}_API_BASE: API endpoint URL
    - {PREFIX}_API_KEY: API key
    - {PREFIX}_MODELS: Comma-separated list of models
    - {PREFIX}_DEFAULT_MODEL: Default model to use
    - {PREFIX}_CLIENT_CLASS: (Optional) Client class to use
    
    Examples:
    - LLM_PROVIDER_NAME=company_openai
    - LLM_API_BASE=https://api.openai.com/v1
    - LLM_API_KEY=sk-...
    - LLM_MODELS=gpt-3.5-turbo,gpt-4
    - LLM_DEFAULT_MODEL=gpt-3.5-turbo
    """
    # Get configuration from environment
    name = provider_name or os.getenv(f"{prefix}_PROVIDER_NAME")
    api_base = os.getenv(f"{prefix}_API_BASE")
    api_key = os.getenv(f"{prefix}_API_KEY")
    models_str = os.getenv(f"{prefix}_MODELS")
    default_model = os.getenv(f"{prefix}_DEFAULT_MODEL")
    client_class = os.getenv(f"{prefix}_CLIENT_CLASS")
    
    # Check required fields
    if not name:
        return None
    if not api_base and not client_class:
        return None
    
    # Parse models list
    models = None
    if models_str:
        models = [m.strip() for m in models_str.split(",") if m.strip()]
    
    return {
        "name": name,
        "api_base": api_base,
        "api_key": api_key,
        "models": models,
        "default_model": default_model,
        "client_class": client_class
    }

def register_from_openai_env() -> Optional[str]:
    """
    Register a provider using standard OpenAI environment variables.
    
    Uses:
    - OPENAI_API_KEY
    - OPENAI_API_BASE (optional, defaults to OpenAI)
    - OPENAI_MODEL (optional, for default model)
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    
    # Use custom base URL if provided, otherwise OpenAI
    api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    
    # Determine provider name from base URL
    if "openai.com" in api_base:
        provider_name = "env_openai"
    elif "azure" in api_base.lower():
        provider_name = "env_azure"
    else:
        # Extract domain for name
        import urllib.parse
        parsed = urllib.parse.urlparse(api_base)
        provider_name = f"env_{parsed.hostname.split('.')[0]}"
    
    # Get model preference
    default_model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    
    # Common OpenAI-compatible models
    models = [
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-16k", 
        "gpt-4",
        "gpt-4-turbo",
        "gpt-4o",
        "gpt-4o-mini"
    ]
    
    # Register the provider
    config = register_openai_compatible(
        name=provider_name,
        api_base=api_base,
        api_key=api_key,
        models=models,
        default_model=default_model
    )
    
    print(f"✅ Registered '{provider_name}' from OpenAI environment")
    print(f"   Endpoint: {api_base}")
    print(f"   Default model: {default_model}")
    
    return provider_name

def load_multiple_providers():
    """
    Load multiple providers from different environment prefixes.
    Useful for multi-provider setups.
    """
    prefixes = ["PRIMARY", "BACKUP", "TEST"]
    registered = []
    
    for prefix in prefixes:
        config = load_provider_from_env(prefix)
        if config:
            if config.get("client_class"):
                # Use generic registration
                provider = register_provider(
                    name=config["name"],
                    client_class=config["client_class"],
                    api_base=config.get("api_base"),
                    api_key=config.get("api_key"),
                    models=config.get("models"),
                    default_model=config.get("default_model")
                )
            else:
                # Use OpenAI-compatible registration
                provider = register_openai_compatible(
                    name=config["name"],
                    api_base=config["api_base"],
                    api_key=config.get("api_key"),
                    models=config.get("models"),
                    default_model=config.get("default_model")
                )
            
            registered.append(config["name"])
            print(f"✅ Loaded '{config['name']}' from {prefix}_* environment")
    
    return registered

def demo_api_base_env():
    """Demonstrate using api_base_env without hardcoding URLs."""
    print("\n🔗 API BASE FROM ENVIRONMENT (NO HARDCODED URL)")
    print("-" * 40)
    
    # Set environment variables
    os.environ["CUSTOM_LLM_ENDPOINT"] = "https://llm.company.com/v1"
    os.environ["CUSTOM_LLM_KEY"] = "test-key-123"
    
    print("Environment variables set:")
    print(f"  CUSTOM_LLM_ENDPOINT = {os.environ['CUSTOM_LLM_ENDPOINT']}")
    print(f"  CUSTOM_LLM_KEY = {os.environ['CUSTOM_LLM_KEY']}")
    
    # Register provider WITHOUT hardcoding the URL
    # The URL will be read from the environment variable
    provider = register_openai_compatible(
        name="env_based_provider",
        api_base_env="CUSTOM_LLM_ENDPOINT",  # NO api_base parameter needed!
        api_key_env="CUSTOM_LLM_KEY",
        models=["gpt-3.5-turbo", "gpt-4"],
        default_model="gpt-3.5-turbo"
    )
    
    print("\n✅ Provider registered without hardcoding URL!")
    print("   The base URL is dynamically read from CUSTOM_LLM_ENDPOINT")
    
    # Verify it works
    config = get_config()
    resolved_base = config.get_api_base("env_based_provider")
    print(f"\nResolved base URL: {resolved_base}")
    
    # Clean up
    unregister_provider("env_based_provider")
    del os.environ["CUSTOM_LLM_ENDPOINT"]
    del os.environ["CUSTOM_LLM_KEY"]

def demo_environment_based_config():
    """Demonstrate environment-based configuration."""
    print("\n🌍 ENVIRONMENT-BASED CONFIGURATION")
    print("-" * 40)
    
    # Try to load from OpenAI environment
    provider_name = register_from_openai_env()
    
    if provider_name:
        # Test the provider
        try:
            response = ask_sync(
                "Say 'Environment config works!' in 5 words or less",
                provider=provider_name,
                temperature=0,
                max_tokens=10
            )
            print(f"\n✅ Test response: {response}")
        except Exception as e:
            print(f"\n⚠️  Test failed: {e}")
    else:
        print("\n⚠️  No OpenAI environment variables found")
        print("   Set these environment variables:")
        print("   - OPENAI_API_KEY=your-api-key")
        print("   - OPENAI_API_BASE=https://api.openai.com/v1 (optional)")
        print("   - OPENAI_MODEL=gpt-3.5-turbo (optional)")

def demo_multi_environment():
    """Demonstrate loading multiple providers from environment."""
    print("\n🔄 MULTI-PROVIDER ENVIRONMENT SETUP")
    print("-" * 40)
    
    # Example: Set some test environment variables
    # In production, these would be set externally
    if not os.getenv("PRIMARY_PROVIDER_NAME"):
        print("Setting example environment variables...")
        os.environ["PRIMARY_PROVIDER_NAME"] = "primary_api"
        os.environ["PRIMARY_API_BASE"] = "https://api.openai.com/v1"
        os.environ["PRIMARY_API_KEY"] = os.getenv("OPENAI_API_KEY", "test-key")
        os.environ["PRIMARY_MODELS"] = "gpt-3.5-turbo,gpt-4"
        os.environ["PRIMARY_DEFAULT_MODEL"] = "gpt-3.5-turbo"
        
        os.environ["BACKUP_PROVIDER_NAME"] = "backup_api"
        os.environ["BACKUP_API_BASE"] = "https://backup.api.com/v1"
        os.environ["BACKUP_API_KEY"] = "backup-key"
        os.environ["BACKUP_MODELS"] = "model-1,model-2"
        os.environ["BACKUP_DEFAULT_MODEL"] = "model-1"
    
    # Load all configured providers
    providers = load_multiple_providers()
    
    if providers:
        print(f"\n✅ Loaded {len(providers)} providers: {providers}")
        
        # List all dynamic providers
        all_dynamic = list_dynamic_providers()
        print(f"All dynamic providers: {all_dynamic}")
    else:
        print("\n⚠️  No providers configured in environment")

def demo_docker_compose_style():
    """
    Show how this would work in a Docker Compose setup.
    """
    print("\n🐳 DOCKER COMPOSE CONFIGURATION EXAMPLE")
    print("-" * 40)
    
    docker_compose_example = """
Example docker-compose.yml:
```yaml
version: '3.8'

services:
  app:
    image: your-app:latest
    environment:
      # Primary OpenAI Provider
      LLM_PROVIDER_NAME: openai_prod
      LLM_API_BASE: https://api.openai.com/v1
      LLM_API_KEY: ${OPENAI_API_KEY}
      LLM_MODELS: gpt-3.5-turbo,gpt-4,gpt-4-turbo
      LLM_DEFAULT_MODEL: gpt-3.5-turbo
      
      # Backup Anthropic Provider
      BACKUP_PROVIDER_NAME: anthropic_backup
      BACKUP_CLIENT_CLASS: AnthropicLLMClient
      BACKUP_API_KEY: ${ANTHROPIC_API_KEY}
      BACKUP_MODELS: claude-3-opus,claude-3-sonnet
      BACKUP_DEFAULT_MODEL: claude-3-sonnet
      
      # Development Ollama Provider
      DEV_PROVIDER_NAME: local_ollama
      DEV_API_BASE: http://ollama:11434
      DEV_MODELS: llama3,mistral
      DEV_DEFAULT_MODEL: llama3
```

Then in your application:
```python
from chuk_llm import ask_sync

# Providers are auto-registered from environment
response = ask_sync("Hello!", provider="openai_prod")
```
"""
    print(docker_compose_example)

def demo_kubernetes_style():
    """
    Show how this would work in a Kubernetes setup.
    """
    print("\n☸️  KUBERNETES CONFIGURATION EXAMPLE")
    print("-" * 40)
    
    k8s_example = """
Example Kubernetes ConfigMap and Deployment:
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: llm-config
data:
  LLM_PROVIDER_NAME: "openai_k8s"
  LLM_API_BASE: "https://api.openai.com/v1"
  LLM_MODELS: "gpt-3.5-turbo,gpt-4"
  LLM_DEFAULT_MODEL: "gpt-3.5-turbo"
---
apiVersion: v1
kind: Secret
metadata:
  name: llm-secrets
stringData:
  LLM_API_KEY: "your-openai-key"
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app
spec:
  template:
    spec:
      containers:
      - name: app
        envFrom:
        - configMapRef:
            name: llm-config
        - secretRef:
            name: llm-secrets
```
"""
    print(k8s_example)

def demo_cloud_function_style():
    """
    Show how this would work in cloud functions/lambdas.
    """
    print("\n⚡ CLOUD FUNCTION CONFIGURATION EXAMPLE")
    print("-" * 40)
    
    cloud_example = """
Example for AWS Lambda, Google Cloud Functions, etc:

```python
import os
from chuk_llm import register_from_openai_env, ask_sync

def lambda_handler(event, context):
    # Provider auto-registered from Lambda environment variables
    provider = register_from_openai_env()
    
    if not provider:
        # Fallback to configured provider
        provider = "openai"
    
    response = ask_sync(
        event["prompt"],
        provider=provider,
        temperature=event.get("temperature", 0.7)
    )
    
    return {
        "statusCode": 200,
        "body": {"response": response}
    }
```

Configure via Lambda Environment Variables:
- OPENAI_API_KEY: your-key
- OPENAI_API_BASE: https://proxy.company.com/openai/v1
- OPENAI_MODEL: gpt-3.5-turbo
"""
    print(cloud_example)

def main():
    print("=" * 60)
    print("ENVIRONMENT-BASED PROVIDER CONFIGURATION")
    print("=" * 60)
    
    # Run demonstrations
    demo_api_base_env()
    demo_environment_based_config()
    demo_multi_environment()
    demo_docker_compose_style()
    demo_kubernetes_style()
    demo_cloud_function_style()
    
    # Clean up
    print("\n🧹 Cleaning up dynamic providers...")
    for provider in list_dynamic_providers():
        unregister_provider(provider)
        print(f"   Removed: {provider}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
Environment-based configuration enables:
✅ Zero-code provider configuration
✅ Easy multi-environment deployments
✅ Secure API key management via secrets
✅ Container and serverless compatibility
✅ Dynamic provider switching
✅ CI/CD pipeline integration

Best Practices:
1. Never hardcode API keys - use environment variables
2. Use secret management tools (Vault, K8s Secrets, etc.)
3. Separate configuration from code
4. Use prefixes for multiple providers
5. Document required environment variables
6. Provide sensible defaults where appropriate
""")

if __name__ == "__main__":
    main()