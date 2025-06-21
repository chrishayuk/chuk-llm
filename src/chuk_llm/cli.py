#!/usr/bin/env python3
# chuk_llm/cli.py
"""
ChukLLM CLI - Quick access to AI models from command line

Usage:
    chuk-llm ask_granite "What is Python?"
    chuk-llm ask_claude "Explain quantum computing"
    chuk-llm ask "What is AI?" --provider openai --model gpt-4o-mini
    chuk-llm providers
    chuk-llm models ollama
    chuk-llm test ollama
    chuk-llm config
    chuk-llm discover ollama
    chuk-llm functions
"""

import argparse
import sys
from typing import Optional, List

try:
    from rich.console import Console
    from rich.table import Table
    from rich.markdown import Markdown
    _rich_available = True
    console = Console()
except ImportError:
    _rich_available = False
    console = None

try:
    from .configuration import get_config, CapabilityChecker
    from .api.sync import ask_sync, quick_question, stream_sync_iter
    from .api.providers import (
        refresh_provider_functions, 
        trigger_ollama_discovery_and_refresh,
        list_provider_functions,
        get_discovered_functions,
        show_config as show_provider_config
    )
except ImportError as e:
    print(f"Error importing chuk_llm: {e}")
    print("Make sure chuk-llm is properly installed")
    sys.exit(1)


class ChukLLMCLI:
    """ChukLLM Command Line Interface"""
    
    def __init__(self, verbose: bool = False):
        self.config = get_config()
        self.verbose = verbose
    
    def print_rich(self, content: str, style: str = ""):
        """Print with rich formatting if available"""
        if _rich_available and console:
            if style == "error":
                console.print(content, style="bold red")
            elif style == "success":
                console.print(content, style="bold green")
            elif style == "info":
                console.print(content, style="bold blue")
            elif style == "markdown":
                console.print(Markdown(content))
            else:
                console.print(content)
        else:
            print(content)
    
    def print_table(self, headers: List[str], rows: List[List[str]], title: str = ""):
        """Print table with rich formatting if available"""
        if _rich_available and console:
            table = Table(title=title, show_header=True, header_style="bold magenta")
            for header in headers:
                table.add_column(header)
            for row in rows:
                table.add_row(*row)
            console.print(table)
        else:
            print(f"\n{title}")
            print(" | ".join(headers))
            print("-" * (len(" | ".join(headers))))
            for row in rows:
                print(" | ".join(row))
            print()

    def ask_model(self, prompt: str, provider: str, model: Optional[str] = None, 
                 json_mode: bool = False, stream: bool = True) -> str:
        """Ask a question to a specific model using sync API with optional streaming"""
        try:
            if stream and not json_mode:
                return self.stream_response(prompt, provider, model, json_mode=json_mode)
            else:
                # Non-streaming for JSON mode or when explicitly disabled
                if model:
                    response = ask_sync(prompt, provider=provider, model=model, json_mode=json_mode)
                else:
                    response = ask_sync(prompt, provider=provider, json_mode=json_mode)
                return response
        except Exception as e:
            raise Exception(f"Failed to get response from {provider}: {e}")

    def handle_ask_alias(self, alias: str, prompt: str, **kwargs) -> str:
        """Handle ask_alias commands (like ask_granite) with streaming"""
        try:
            # Check if this is a global alias
            global_aliases = self.config.get_global_aliases()
            
            if alias in global_aliases:
                alias_target = global_aliases[alias]
                if "/" in alias_target:
                    provider, model = alias_target.split("/", 1)
                    if self.verbose:
                        self.print_rich(f"Using {provider}/{model} via alias '{alias}'", "info")
                    return self.stream_response(prompt, provider=provider, model=model, **kwargs)
                else:
                    if self.verbose:
                        self.print_rich(f"Using provider '{alias_target}' via alias '{alias}'", "info")
                    return self.stream_response(prompt, provider=alias_target, **kwargs)
            else:
                # Try as direct provider
                if self.verbose:
                    self.print_rich(f"Using provider '{alias}' directly", "info")
                return self.stream_response(prompt, provider=alias, **kwargs)
                
        except Exception as e:
            raise Exception(f"Alias or provider '{alias}' not available: {e}")

    def stream_response(self, prompt: str, provider: str = None, model: str = None, **kwargs) -> str:
        """Stream a response and display it in real-time"""
        try:
            from .api.sync import stream_sync_iter
            
            full_response = ""
            
            # Show what we're calling (only in verbose mode for provider info)
            if self.verbose:
                if provider and model:
                    self.print_rich(f"🤖 {provider}/{model}:", "info")
                elif provider:
                    self.print_rich(f"🤖 {provider}:", "info")
                else:
                    self.print_rich("🤖 Response:", "info")
                print("")  # Add a blank line
            
            # Stream the response
            for chunk in stream_sync_iter(prompt, provider=provider, model=model, **kwargs):
                print(chunk, end="", flush=True)
                full_response += chunk
            
            print("\n")  # Add final newline
            return full_response
            
        except Exception as e:
            # Fallback to non-streaming
            if self.verbose:
                self.print_rich(f"⚠ Streaming failed, using non-streaming: {e}", "error")
            if provider and model:
                return ask_sync(prompt, provider=provider, model=model, **kwargs)
            elif provider:
                return ask_sync(prompt, provider=provider, **kwargs)
            else:
                return ask_sync(prompt, **kwargs)

    def show_providers(self) -> None:
        """List all available providers"""
        try:
            providers = self.config.get_all_providers()
            
            rows = []
            for provider_name in sorted(providers):
                try:
                    provider = self.config.get_provider(provider_name)
                    model_count = len(provider.models) if provider.models else 0
                    features = ", ".join([f.value for f in list(provider.features)[:3]])
                    if len(provider.features) > 3:
                        features += "..."
                    
                    rows.append([
                        provider_name,
                        provider.default_model or "N/A",
                        str(model_count),
                        features,
                        "✓" if self.config.get_api_key(provider_name) else "✗"
                    ])
                except Exception as e:
                    rows.append([provider_name, "ERROR", "0", str(e), "✗"])
            
            self.print_table(
                ["Provider", "Default Model", "Models", "Features", "API Key"],
                rows,
                "Available Providers"
            )
        except Exception as e:
            self.print_rich(f"Error listing providers: {e}", "error")

    def show_models(self, provider: str) -> None:
        """List models for a specific provider"""
        try:
            provider_config = self.config.get_provider(provider)
            
            if not provider_config.models:
                self.print_rich(f"No models configured for provider '{provider}'", "error")
                return
            
            rows = []
            for model in provider_config.models:
                try:
                    caps = provider_config.get_model_capabilities(model)
                    features = ", ".join([f.value for f in list(caps.features)[:3]])
                    if len(caps.features) > 3:
                        features += "..."
                    
                    rows.append([
                        model,
                        str(caps.max_context_length) if caps.max_context_length else "N/A",
                        str(caps.max_output_tokens) if caps.max_output_tokens else "N/A",
                        features
                    ])
                except Exception:
                    rows.append([model, "N/A", "N/A", "Unknown"])
            
            # Add aliases
            if provider_config.model_aliases:
                rows.append(["--- ALIASES ---", "", "", ""])
                for alias, target in provider_config.model_aliases.items():
                    rows.append([f"{alias} → {target}", "", "", "Alias"])
            
            self.print_table(
                ["Model", "Context", "Output", "Features"],
                rows,
                f"Models for {provider}"
            )
        except Exception as e:
            self.print_rich(f"Error listing models for '{provider}': {e}", "error")

    def test_provider(self, provider: str) -> None:
        """Test if a provider is working"""
        try:
            provider_config = self.config.get_provider(provider)
            api_key = self.config.get_api_key(provider)
            
            self.print_rich(f"Testing provider: {provider}", "info")
            
            # Check configuration
            self.print_rich(f"✓ Provider configuration found", "success")
            self.print_rich(f"  - Client: {provider_config.client_class}")
            self.print_rich(f"  - Default model: {provider_config.default_model}")
            self.print_rich(f"  - Models available: {len(provider_config.models) if provider_config.models else 0}")
            
            # Check API key
            if provider.lower() == "ollama":
                self.print_rich(f"✓ Ollama doesn't require an API key", "success")
            elif api_key:
                self.print_rich(f"✓ API key found", "success")
            else:
                self.print_rich(f"✗ No API key found", "error")
                if provider_config.api_key_env:
                    self.print_rich(f"  Set environment variable: {provider_config.api_key_env}")
                return
            
            # Test basic capabilities
            try:
                can_handle, problems = CapabilityChecker.can_handle_request(provider)
                if can_handle:
                    self.print_rich(f"✓ Basic capabilities check passed", "success")
                else:
                    self.print_rich(f"⚠ Capability issues: {', '.join(problems)}", "error")
            except Exception as e:
                self.print_rich(f"⚠ Could not check capabilities: {e}", "error")
            
            # Try a simple request
            self.print_rich("Testing with simple request...", "info")
            
            try:
                response = self.ask_model(
                    "Say 'Hello from ChukLLM CLI!' and nothing else.", 
                    provider
                )
                self.print_rich(f"✓ Test request successful", "success")
                self.print_rich(f"Response: {response}")
            except Exception as e:
                self.print_rich(f"✗ Test request failed: {e}", "error")
        
        except Exception as e:
            self.print_rich(f"Error testing provider '{provider}': {e}", "error")

    def discover_models(self, provider: str) -> None:
        """Discover new models for a provider"""
        try:
            self.print_rich(f"Discovering models for {provider}...", "info")
            
            if provider == "ollama":
                new_functions = trigger_ollama_discovery_and_refresh()
                self.print_rich(f"✓ Discovered {len(new_functions)} new Ollama functions", "success")
            else:
                new_functions = refresh_provider_functions(provider)
                self.print_rich(f"✓ Refreshed {len(new_functions)} functions for {provider}", "success")
            
            # Show some discovered functions
            if new_functions:
                provider_funcs = [name for name in new_functions.keys() 
                                if name.startswith(f'ask_{provider}_') and not name.endswith('_sync')][:5]
                if provider_funcs:
                    self.print_rich("Example functions:", "info")
                    for func_name in provider_funcs:
                        self.print_rich(f"  - {func_name}()")
                        
        except Exception as e:
            self.print_rich(f"Error discovering models for {provider}: {e}", "error")

    def show_functions(self) -> None:
        """List all available provider functions"""
        try:
            all_functions = list_provider_functions()
            
            # Group by type
            ask_funcs = [f for f in all_functions if f.startswith('ask_') and not f.endswith('_sync')]
            stream_funcs = [f for f in all_functions if f.startswith('stream_')]
            sync_funcs = [f for f in all_functions if f.endswith('_sync')]
            
            self.print_rich(f"Total provider functions: {len(all_functions)}", "info")
            
            if ask_funcs:
                self.print_rich(f"\nAsync functions ({len(ask_funcs)}):", "info")
                for func in ask_funcs[:10]:  # Show first 10
                    self.print_rich(f"  - {func}()")
                if len(ask_funcs) > 10:
                    self.print_rich(f"  ... and {len(ask_funcs) - 10} more")
            
            if sync_funcs:
                self.print_rich(f"\nSync functions ({len(sync_funcs)}):", "info")
                for func in sync_funcs[:10]:  # Show first 10
                    self.print_rich(f"  - {func}()")
                if len(sync_funcs) > 10:
                    self.print_rich(f"  ... and {len(sync_funcs) - 10} more")
                    
        except Exception as e:
            self.print_rich(f"Error listing functions: {e}", "error")

    def show_discovered_functions(self, provider: str = None) -> None:
        """Show functions that were discovered dynamically"""
        try:
            discovered = get_discovered_functions(provider)
            
            if not discovered:
                self.print_rich("No discovered functions found", "info")
                return
            
            for provider_name, functions in discovered.items():
                if functions:
                    self.print_rich(f"\nDiscovered functions for {provider_name}:", "info")
                    for func_name in sorted(functions.keys())[:10]:
                        self.print_rich(f"  - {func_name}()")
                    if len(functions) > 10:
                        self.print_rich(f"  ... and {len(functions) - 10} more")
                        
        except Exception as e:
            self.print_rich(f"Error showing discovered functions: {e}", "error")

    def show_config(self) -> None:
        """Show configuration information"""
        try:
            # Use the provider's show_config function which is more comprehensive
            show_provider_config()
            
        except Exception as e:
            # Fallback to basic config info
            self.print_rich(f"Error with enhanced config display: {e}", "error")
            
            global_settings = self.config.get_global_settings()
            global_aliases = self.config.get_global_aliases()
            
            self.print_rich("ChukLLM Configuration", "info")
            
            # Global settings
            if global_settings:
                self.print_rich("\nGlobal Settings:", "info")
                for key, value in global_settings.items():
                    self.print_rich(f"  {key}: {value}")
            
            # Global aliases
            if global_aliases:
                rows = []
                for alias, target in global_aliases.items():
                    rows.append([alias, target])
                
                self.print_table(
                    ["Alias", "Target"],
                    rows,
                    "Global Aliases"
                )
            
            # Provider summary
            providers = self.config.get_all_providers()
            self.print_rich(f"\nTotal providers configured: {len(providers)}", "info")

    def show_aliases(self) -> None:
        """Show available global aliases"""
        try:
            global_aliases = self.config.get_global_aliases()
            
            if not global_aliases:
                self.print_rich("No global aliases configured", "info")
                return
            
            rows = []
            for alias, target in global_aliases.items():
                rows.append([f"ask_{alias}", target])
            
            self.print_table(
                ["CLI Command", "Target (Provider/Model)"],
                rows,
                "Available Global Aliases"
            )
            
            self.print_rich(f"\nExample usage:", "info")
            example_alias = list(global_aliases.keys())[0]
            self.print_rich(f"  chuk-llm ask_{example_alias} \"Your question here\"")
            
        except Exception as e:
            self.print_rich(f"Error showing aliases: {e}", "error")

    def show_help(self) -> None:
        """Show help information"""
        help_text = """
# ChukLLM CLI Help

## Quick Ask Commands (Global Aliases)
```bash
chuk-llm ask_granite "What is Python?"
chuk-llm ask_claude "Explain quantum computing"
chuk-llm ask_gpt "Write a haiku about code"
chuk-llm ask_llama "What is machine learning?"
```

## General Ask Command
```bash
chuk-llm ask "Question" --provider openai --model gpt-4o-mini
chuk-llm ask "Question" --json  # Request JSON response
```

## Simple Commands
```bash
chuk-llm providers             # Show all providers
chuk-llm models ollama         # Show models for provider
chuk-llm test anthropic        # Test provider connection
chuk-llm config                # Show configuration
chuk-llm aliases               # Show global aliases
chuk-llm discover ollama       # Discover new models
chuk-llm functions             # Show all dynamic functions
```

## Examples
```bash
# Quick questions using global aliases
chuk-llm ask_claude "What's the weather API for Python?"
chuk-llm ask_granite "Explain Python decorators"

# JSON responses
chuk-llm ask "List 3 Python libraries" --json --provider openai

# Discovery and testing
chuk-llm discover ollama       # Find new models
chuk-llm test ollama           # Test with actual LLM call
chuk-llm functions             # See all available functions
```

## Using with uvx
```bash
uvx chuk-llm ask_granite "What is Python?"
uvx chuk-llm providers
uvx chuk-llm discover ollama
```
"""
        self.print_rich(help_text, "markdown")


def main():
    """Main CLI entry point"""
    if len(sys.argv) < 2:
        print("ChukLLM CLI - Quick access to AI models")
        print("")
        print("Usage:")
        print("  chuk-llm ask_granite \"What is Python?\"")
        print("  chuk-llm providers")
        print("  chuk-llm models ollama")
        print("  chuk-llm test anthropic")
        print("  chuk-llm help")
        print("")
        print("Options:")
        print("  --verbose, -v    Show detailed provider/model information")
        print("  --quiet, -q      Minimal output")
        print("")
        print("Run 'chuk-llm help' for detailed help")
        return
    
    # Check for global flags
    verbose = False
    quiet = False
    args = sys.argv[1:]
    
    if "--verbose" in args or "-v" in args:
        verbose = True
        args = [arg for arg in args if arg not in ["--verbose", "-v"]]
    
    if "--quiet" in args or "-q" in args:
        quiet = True
        verbose = False  # quiet overrides verbose
        args = [arg for arg in args if arg not in ["--quiet", "-q"]]
    
    if not args:
        print("No command specified")
        return
    
    command = args[0]
    # Update sys.argv to reflect the filtered args for subcommand parsing
    sys.argv = ["chuk-llm"] + args
    
    cli = ChukLLMCLI(verbose=verbose)
    
    try:
        # Handle ask_alias commands (ask_granite, ask_claude, etc.)
        if command.startswith("ask_"):
            if len(args) < 2:
                print(f"Usage: chuk-llm {command} \"your question here\"")
                sys.exit(1)
            
            alias = command[4:]  # Remove 'ask_' prefix
            prompt = args[1]
            
            # Note: We don't need to print the response since streaming handles it
            cli.handle_ask_alias(alias, prompt)
        
        # Handle general ask command
        elif command == "ask":
            parser = argparse.ArgumentParser(description="Ask a question")
            parser.add_argument("prompt", help="The question to ask")
            parser.add_argument("--provider", "-p", required=True, help="Provider to use")
            parser.add_argument("--model", "-m", help="Specific model to use")
            parser.add_argument("--json", action="store_true", help="Request JSON response")
            parser.add_argument("--no-stream", action="store_true", help="Disable streaming")
            
            try:
                args = parser.parse_args(args[1:])  # Use filtered args
                
                # Note: We don't need to print the response since streaming handles it
                cli.ask_model(
                    args.prompt, 
                    args.provider, 
                    args.model,
                    json_mode=args.json,
                    stream=not args.no_stream
                )
            except SystemExit:
                # Handle argument parsing errors gracefully
                print("Error: Invalid arguments. Use quotes around your question.")
                print('Example: chuk-llm ask "What is machine learning?" --provider ollama')
                sys.exit(1)
        
        # Simple commands
        elif command == "providers":
            cli.show_providers()
        
        elif command == "models":
            if len(args) < 2:
                print("Usage: chuk-llm models <provider>")
                sys.exit(1)
            cli.show_models(args[1])
        
        elif command == "test":
            if len(args) < 2:
                print("Usage: chuk-llm test <provider>")
                sys.exit(1)
            cli.test_provider(args[1])
        
        elif command == "discover":
            if len(args) < 2:
                print("Usage: chuk-llm discover <provider>")
                sys.exit(1)
            cli.discover_models(args[1])
        
        elif command == "functions":
            cli.show_functions()
        
        elif command == "discovered":
            provider = args[1] if len(args) > 1 else None
            cli.show_discovered_functions(provider)
        
        elif command == "config":
            cli.show_config()
        
        elif command == "aliases":
            cli.show_aliases()
        
        elif command == "help":
            cli.show_help()
        
        else:
            print(f"Unknown command: {command}")
            print("Run 'chuk-llm help' for available commands")
            sys.exit(1)
    
    except Exception as e:
        cli.print_rich(f"Error: {e}", "error")
        sys.exit(1)


if __name__ == "__main__":
    main()