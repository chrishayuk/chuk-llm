#!/usr/bin/env python3
"""
ChukLLM Demo Script
==================

This script demonstrates the key features of ChukLLM:
1. Configuration inheritance system
2. CLI functionality
3. Dynamic provider functions
4. Global aliases
5. Model discovery

Run with: python demo.py
"""

import os
import shutil
import subprocess
from textwrap import dedent


def print_header(title: str):
    """Print a fancy header"""
    print(f"\n{'=' * 60}")
    print(f"  🚀 {title}")
    print(f"{'=' * 60}")


def print_step(step: str):
    """Print a step"""
    print(f"\n📋 {step}")
    print("-" * 50)


def print_info(info: str):
    """Print info message"""
    print(f"ℹ️  {info}")


def print_success(msg: str):
    """Print success message"""
    print(f"✅ {msg}")


def print_error(msg: str):
    """Print error message"""
    print(f"❌ {msg}")


def run_command(cmd: str, description: str = None):
    """Run a command and show output"""
    if description:
        print(f"\n🔧 {description}")

    print(f"$ {cmd}")

    try:
        # Handle shell commands with proper argument splitting
        if cmd.startswith("uv run chuk-llm ask "):
            # For ask commands, use shlex to properly parse arguments
            import shlex

            args = shlex.split(cmd)
        else:
            args = cmd.split()

        result = subprocess.run(args, capture_output=True, text=True, timeout=30)

        if result.stdout:
            print(result.stdout)

        if result.stderr and result.returncode == 0:
            print(f"ℹ️  {result.stderr}")
        elif result.stderr and result.returncode != 0:
            print(f"❌ Error: {result.stderr}")

        return result.returncode == 0

    except subprocess.TimeoutExpired:
        print("❌ Command timed out")
        return False
    except Exception as e:
        print(f"❌ Command failed: {e}")
        return False


def create_demo_configs():
    """Create demo configuration files"""

    print_step("Creating demo configuration files")

    # Create a providers.yaml that extends the package config
    providers_yaml = dedent("""
    # Demo providers.yaml - Extends package configuration
    # This demonstrates the inheritance system

    __global__:
      # Override the active provider
      active_provider: ollama
      demo_mode: true

    __global_aliases__:
      # Add custom aliases
      demo_gpt: openai/gpt-4o-mini
      demo_claude: anthropic/claude-sonnet-4
      demo_local: ollama/llama3.3
      quick: ollama/granite3.3

    # Extend ollama configuration
    ollama:
      # Add custom models (these will be added to package models)
      models:
        - demo-model-1
        - demo-model-2

      # Add custom aliases (these will be merged with package aliases)
      model_aliases:
        demo: demo-model-1
        custom: demo-model-2
        super_fast: llama3.3

      # Override default model
      default_model: granite3.3

      # Add custom configuration
      extra:
        demo_setting: true
        custom_timeout: 60
        dynamic_discovery:
          demo_discovery: true

    # Add a completely new provider
    demo_provider:
      api_key_env: DEMO_API_KEY
      client_class: demo.client.DemoClient
      default_model: demo-1
      models:
        - demo-1
        - demo-2
      features:
        - text
        - streaming
      model_aliases:
        fast: demo-1
        smart: demo-2
    """)

    with open("providers.yaml", "w") as f:
        f.write(providers_yaml)

    print_success("Created providers.yaml (inheritance mode)")

    # Also create a complete replacement example
    chuk_llm_yaml = dedent("""
    # Demo chuk_llm.yaml - Complete replacement
    # This would completely replace the package configuration

    __global__:
      active_provider: openai
      active_model: gpt-4o-mini
      replacement_mode: true

    __global_aliases__:
      gpt: openai/gpt-4o-mini
      claude: anthropic/claude-sonnet-4

    openai:
      api_key_env: OPENAI_API_KEY
      client_class: chuk_llm.llm.providers.openai_client.OpenAILLMClient
      default_model: gpt-4o-mini
      models:
        - gpt-4o-mini
        - gpt-4o
      features:
        - text
        - streaming
        - json_mode

    anthropic:
      api_key_env: ANTHROPIC_API_KEY
      client_class: chuk_llm.llm.providers.anthropic_client.AnthropicLLMClient
      default_model: claude-sonnet-4
      models:
        - claude-sonnet-4
        - claude-opus-4
      features:
        - text
        - streaming
        - reasoning
    """)

    with open("chuk_llm_replacement_example.yaml", "w") as f:
        f.write(chuk_llm_yaml)

    print_success("Created chuk_llm_replacement_example.yaml (replacement mode)")


def demo_cli_basic():
    """Demo basic CLI functionality"""

    print_header("CLI Basic Functionality")

    # Show help
    run_command("uv run chuk-llm help", "Show CLI help")

    # List providers
    run_command("uv run chuk-llm providers", "List all available providers")

    # Show configuration
    run_command("uv run chuk-llm config", "Show current configuration")

    # Show aliases
    run_command("uv run chuk-llm aliases", "Show available global aliases")


def demo_cli_models():
    """Demo model listing and discovery"""

    print_header("Model Management")

    # List models for different providers
    run_command("uv run chuk-llm models ollama", "List Ollama models")
    run_command("uv run chuk-llm models openai", "List OpenAI models")

    # Test provider connection
    run_command("uv run chuk-llm test ollama", "Test Ollama provider")

    # Show dynamic functions
    run_command("uv run chuk-llm functions", "Show available dynamic functions")


def demo_cli_queries():
    """Demo asking questions using different methods"""

    print_header("AI Queries Demo")

    print_info(
        "Note: These demos will only work if you have the required API keys set up"
    )
    print_info("Ollama demos require Ollama to be running locally")

    # Demo global aliases
    print_step("Using Global Aliases")

    # Simple math question
    run_command('uv run chuk-llm ask_granite "What is 2+2?"', "Ask granite (via alias)")

    # More complex question with verbose mode
    run_command(
        'uv run chuk-llm ask_quick "Explain Python in one sentence" --verbose',
        "Ask with verbose output",
    )

    # Demo direct provider calls
    print_step("Direct Provider Calls")

    run_command(
        'uv run chuk-llm ask "What is machine learning?" --provider ollama --model granite3.3',
        "Direct provider and model specification",
    )

    # Demo JSON mode
    print_step("JSON Responses")

    run_command(
        'uv run chuk-llm ask "List 3 programming languages" --provider ollama --json',
        "Request JSON response",
    )


def demo_discovery():
    """Demo model discovery"""

    print_header("Dynamic Model Discovery")

    print_info("This demonstrates how ChukLLM can discover new models")

    # Discover Ollama models
    run_command("uv run chuk-llm discover ollama", "Discover Ollama models")

    # Show discovered functions
    run_command(
        "uv run chuk-llm discovered ollama", "Show discovered functions for Ollama"
    )


def demo_config_inheritance():
    """Demo configuration inheritance"""

    print_header("Configuration Inheritance Demo")

    print_step("Current Configuration (with providers.yaml)")

    print_info("We're currently using providers.yaml which extends the package config")

    # Show current config
    run_command("uv run chuk-llm config", "Show current merged configuration")

    # Show how aliases work
    run_command("uv run chuk-llm aliases", "Show merged aliases")

    print_step("Switching to Complete Replacement Mode")

    print_info("Let's temporarily switch to complete replacement mode")

    # Backup current config
    if os.path.exists("providers.yaml"):
        shutil.move("providers.yaml", "providers.yaml.backup")

    # Copy replacement config
    if os.path.exists("chuk_llm_replacement_example.yaml"):
        shutil.copy("chuk_llm_replacement_example.yaml", "chuk_llm.yaml")

        print_success("Switched to chuk_llm.yaml (replacement mode)")

        # Show the difference
        run_command("uv run chuk-llm config", "Show replacement configuration")
        run_command("uv run chuk-llm aliases", "Show replacement aliases")

        # Clean up
        os.remove("chuk_llm.yaml")
        if os.path.exists("providers.yaml.backup"):
            shutil.move("providers.yaml.backup", "providers.yaml")

        print_success("Restored original configuration")


def demo_advanced_features():
    """Demo advanced features"""

    print_header("Advanced Features")

    print_step("Verbose Mode")
    run_command("uv run chuk-llm providers --verbose", "Verbose provider listing")

    print_step("Quiet Mode")
    run_command("uv run chuk-llm providers --quiet", "Quiet provider listing")

    print_step("Testing Provider Capabilities")
    run_command("uv run chuk-llm test ollama --verbose", "Detailed provider test")


def demo_programming_interface():
    """Demo the Python programming interface"""

    print_header("Python Programming Interface")

    print_step("Creating a Python demo script")

    python_demo = dedent("""
    # Demo of ChukLLM Python API

    try:
        import chuk_llm

        print("🐍 ChukLLM Python API Demo")
        print("=" * 40)

        # Show configuration
        config = chuk_llm.get_config()
        providers = config.get_all_providers()
        print(f"Available providers: {', '.join(providers)}")

        # Show global aliases
        aliases = config.get_global_aliases()
        print(f"Global aliases: {', '.join(aliases.keys())}")

        # Try a simple sync call (if Ollama is available)
        print("\\n🤖 Testing sync API...")
        try:
            response = chuk_llm.ask_sync("What is 1+1?", provider="ollama", model="granite3.3")
            print(f"Response: {response[:100]}...")
        except Exception as e:
            print(f"Sync call failed (this is normal if Ollama isn't running): {e}")

        # Show available provider functions
        print("\\n🔧 Available dynamic functions:")
        try:
            from chuk_llm.api.providers import list_provider_functions
            functions = list_provider_functions()
            print(f"Total functions: {len(functions)}")

            # Show a few examples
            examples = [f for f in functions if 'granite' in f or 'claude' in f][:5]
            for example in examples:
                print(f"  - {example}()")

        except Exception as e:
            print(f"Function listing failed: {e}")

        print("\\n✅ Python API demo complete!")

    except ImportError as e:
        print(f"❌ Could not import chuk_llm: {e}")
        print("Make sure chuk-llm is installed: uv add chuk-llm")
    """)

    with open("python_demo.py", "w") as f:
        f.write(python_demo)

    print_success("Created python_demo.py")

    # Run the Python demo
    run_command("uv run python python_demo.py", "Run Python API demo")


def cleanup():
    """Clean up demo files"""
    print_step("Cleaning up demo files")

    files_to_remove = [
        "providers.yaml",
        "chuk_llm_replacement_example.yaml",
        "python_demo.py",
        "providers.yaml.backup",
    ]

    for file in files_to_remove:
        if os.path.exists(file):
            os.remove(file)
            print(f"Removed {file}")

    print_success("Cleanup complete")


def main():
    """Main demo function"""

    print_header("ChukLLM Feature Demonstration")

    print_info("This demo will showcase ChukLLM's key features:")
    print_info("• Configuration inheritance system")
    print_info("• CLI functionality")
    print_info("• Dynamic provider functions")
    print_info("• Model discovery")
    print_info("• Python API")

    print_info("\nNote: Some features require API keys and running services")
    print_info("The demo will continue even if some commands fail")

    input("\nPress Enter to start the demo...")

    try:
        # Create demo configuration files
        create_demo_configs()

        # Demo CLI functionality
        demo_cli_basic()
        demo_cli_models()
        demo_cli_queries()
        demo_discovery()
        demo_config_inheritance()
        demo_advanced_features()

        # Demo Python interface
        demo_programming_interface()

        print_header("Demo Complete!")

        print_success("ChukLLM demo finished successfully!")
        print_info("Key takeaways:")
        print_info("• Use providers.yaml to extend package config")
        print_info("• Use chuk_llm.yaml to completely replace config")
        print_info("• CLI supports streaming, aliases, and discovery")
        print_info("• Dynamic functions are generated for all models")
        print_info("• Both CLI and Python APIs are available")

        print_info("\nNext steps:")
        print_info("• Set up your API keys")
        print_info("• Create your own providers.yaml")
        print_info("• Try the global aliases: ask_granite, ask_claude, etc.")
        print_info("• Explore the Python API in your code")

    except KeyboardInterrupt:
        print_error("\nDemo interrupted by user")
    except Exception as e:
        print_error(f"Demo failed: {e}")
    finally:
        cleanup()


if __name__ == "__main__":
    main()
