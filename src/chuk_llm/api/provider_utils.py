# src/chuk_llm/api/provider_utils.py
"""Provider configuration utilities for the ChukLLM API."""

import os
from typing import Optional, Dict, Any

def get_provider_default_model(provider: str) -> Optional[str]:
    """Get the default model for a provider from YAML configuration.
    
    Args:
        provider: The provider name (e.g., 'anthropic', 'openai', 'groq')
        
    Returns:
        The default model name for the provider, or None if not found
        
    Examples:
        model = get_provider_default_model('anthropic')
        # Returns: 'claude-3-7-sonnet-20250219'
        
        model = get_provider_default_model('groq')
        # Returns: 'llama-3.3-70b-versatile'
    """
    try:
        import yaml
        
        # Find providers.yaml using comprehensive path resolution
        module_dir = os.path.dirname(__file__)
        possible_paths = [
            # Same directory as this module
            os.path.join(module_dir, "providers.yaml"),
            
            # Parent directories (going up the tree)
            os.path.join(module_dir, "..", "providers.yaml"),
            os.path.join(module_dir, "..", "..", "providers.yaml"),
            os.path.join(module_dir, "..", "..", "..", "providers.yaml"),
            
            # Current working directory and common config locations
            os.path.join(os.getcwd(), "providers.yaml"),
            os.path.join(os.getcwd(), "config", "providers.yaml"),
            os.path.join(os.getcwd(), "src", "providers.yaml"),
            
            # Environment variable override
            os.environ.get('CHUK_LLM_PROVIDERS_YAML', ''),
            os.environ.get('PROVIDERS_YAML', ''),
            
            # Home directory config
            os.path.expanduser("~/.chuk_llm/providers.yaml"),
            os.path.expanduser("~/.config/chuk_llm/providers.yaml"),
        ]
        
        # Filter out empty paths (from empty env vars)
        possible_paths = [p for p in possible_paths if p]
        
        # Try each path in order
        for config_path in possible_paths:
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                
                if config and provider in config:
                    provider_config = config[provider]
                    
                    # Handle inheritance
                    if 'inherits' in provider_config:
                        parent = provider_config['inherits']
                        if parent in config:
                            # Start with parent config, then override
                            merged_config = config[parent].copy()
                            merged_config.update(provider_config)
                            provider_config = merged_config
                    
                    return provider_config.get('default_model')
                break
                
    except ImportError:
        # PyYAML not available
        pass
    except Exception as e:
        # Log warning but don't crash
        print(f"Warning: Could not read provider config for {provider}: {e}")
    
    return None

def get_provider_config(provider: str) -> Dict[str, Any]:
    """Get the full configuration for a provider from YAML.
    
    Args:
        provider: The provider name
        
    Returns:
        Dictionary with the provider's configuration, or empty dict if not found
        
    Examples:
        config = get_provider_config('anthropic')
        # Returns: {'api_key_env': 'ANTHROPIC_API_KEY', 'default_model': 'claude-3-7-sonnet-20250219'}
    """
    try:
        import yaml
        
        # Find providers.yaml using same logic as get_provider_default_model
        module_dir = os.path.dirname(__file__)
        possible_paths = [
            os.path.join(module_dir, "providers.yaml"),
            os.path.join(module_dir, "..", "providers.yaml"),
            os.path.join(module_dir, "..", "..", "providers.yaml"),
            os.path.join(module_dir, "..", "..", "..", "providers.yaml"),
            os.path.join(os.getcwd(), "providers.yaml"),
            os.path.join(os.getcwd(), "config", "providers.yaml"),
            os.path.join(os.getcwd(), "src", "providers.yaml"),
            os.environ.get('CHUK_LLM_PROVIDERS_YAML', ''),
            os.environ.get('PROVIDERS_YAML', ''),
            os.path.expanduser("~/.chuk_llm/providers.yaml"),
            os.path.expanduser("~/.config/chuk_llm/providers.yaml"),
        ]
        
        possible_paths = [p for p in possible_paths if p]
        
        for config_path in possible_paths:
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                
                if config and provider in config:
                    provider_config = config[provider].copy()
                    
                    # Handle inheritance
                    if 'inherits' in provider_config:
                        parent = provider_config['inherits']
                        if parent in config:
                            # Start with parent config, then override
                            merged_config = config[parent].copy()
                            merged_config.update(provider_config)
                            provider_config = merged_config
                    
                    return provider_config
                break
                
    except Exception as e:
        print(f"Warning: Could not read provider config for {provider}: {e}")
    
    return {}

def get_all_providers() -> list:
    """Get a list of all available providers from YAML configuration.
    
    Returns:
        List of provider names
        
    Examples:
        providers = get_all_providers()
        # Returns: ['openai', 'anthropic', 'groq', 'deepseek', ...]
    """
    try:
        import yaml
        
        # Find providers.yaml using same logic
        module_dir = os.path.dirname(__file__)
        possible_paths = [
            os.path.join(module_dir, "providers.yaml"),
            os.path.join(module_dir, "..", "providers.yaml"),
            os.path.join(module_dir, "..", "..", "providers.yaml"),
            os.path.join(module_dir, "..", "..", "..", "providers.yaml"),
            os.path.join(os.getcwd(), "providers.yaml"),
            os.path.join(os.getcwd(), "config", "providers.yaml"),
            os.path.join(os.getcwd(), "src", "providers.yaml"),
            os.environ.get('CHUK_LLM_PROVIDERS_YAML', ''),
            os.environ.get('PROVIDERS_YAML', ''),
            os.path.expanduser("~/.chuk_llm/providers.yaml"),
            os.path.expanduser("~/.config/chuk_llm/providers.yaml"),
        ]
        
        possible_paths = [p for p in possible_paths if p]
        
        for config_path in possible_paths:
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                
                if config:
                    # Filter out special keys (starting with '__')
                    providers = [k for k in config.keys() if not k.startswith('__')]
                    return providers
                break
                
    except Exception as e:
        print(f"Warning: Could not read providers list: {e}")
    
    # Fallback to common providers
    return ["openai", "anthropic", "gemini", "groq", "mistral", "ollama", "deepseek", "perplexity", "watsonx"]

def find_providers_yaml_path() -> Optional[str]:
    """Find the path to the providers.yaml file.
    
    Returns:
        Path to providers.yaml if found, None otherwise
        
    Examples:
        path = find_providers_yaml_path()
        # Returns: '/path/to/your/project/src/chuk_llm/providers.yaml'
    """
    module_dir = os.path.dirname(__file__)
    possible_paths = [
        os.path.join(module_dir, "providers.yaml"),
        os.path.join(module_dir, "..", "providers.yaml"),
        os.path.join(module_dir, "..", "..", "providers.yaml"),
        os.path.join(module_dir, "..", "..", "..", "providers.yaml"),
        os.path.join(os.getcwd(), "providers.yaml"),
        os.path.join(os.getcwd(), "config", "providers.yaml"),
        os.path.join(os.getcwd(), "src", "providers.yaml"),
        os.environ.get('CHUK_LLM_PROVIDERS_YAML', ''),
        os.environ.get('PROVIDERS_YAML', ''),
        os.path.expanduser("~/.chuk_llm/providers.yaml"),
        os.path.expanduser("~/.config/chuk_llm/providers.yaml"),
    ]
    
    possible_paths = [p for p in possible_paths if p]
    
    for config_path in possible_paths:
        if os.path.exists(config_path):
            return os.path.abspath(config_path)
    
    return None