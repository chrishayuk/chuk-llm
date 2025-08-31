#!/usr/bin/env python3
# chuk_llm/cli.py
"""
ChukLLM CLI - Quick access to AI models from command line

Usage:
    chuk-llm ask_granite "What is Python?"
    chuk-llm ask_claude "Explain quantum computing"
    chuk-llm ask_ollama_gemma3 "Hello"                    # NEW: Convenience functions
    chuk-llm ask_ollama_mistral_small_latest "Tell joke"  # NEW: Discovered models
    chuk-llm ask "What is AI?" --provider openai --model gpt-4o-mini
    chuk-llm providers
    chuk-llm models ollama
    chuk-llm test ollama
    chuk-llm config
    chuk-llm discover ollama
    chuk-llm functions
"""

import argparse
import re
import sys
from typing import Any

try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.table import Table

    _rich_available = True
    console = Console()
except ImportError:
    _rich_available = False
    console = None

try:
    from .api.providers import (
        _GENERATED_FUNCTIONS,
        get_discovered_functions,
        has_function,
        list_provider_functions,
        refresh_provider_functions,
        trigger_ollama_discovery_and_refresh,
    )
    from .api.providers import (
        show_config as show_provider_config,  # noqa: F401
    )
    from .api.sync import ask_sync, quick_question, stream_sync_iter  # noqa: F401
    from .configuration import CapabilityChecker, get_config
except ImportError as e:
    print(f"Error importing chuk_llm: {e}")
    print("Make sure chuk-llm is properly installed")
    sys.exit(1)


def parse_convenience_function(command: str) -> tuple[str, str, bool, bool] | None:
    """
    Parse convenience function commands like ask_ollama_gemma3.

    Returns:
        Tuple of (provider, model, is_sync, is_stream) or None if not a convenience function
    """
    # Pattern for convenience functions:
    # ask_{provider}_{model}[_sync]
    # stream_{provider}_{model}[_sync]

    # Check for streaming functions
    stream_match = re.match(
        r"^stream_([a-z0-9_]+?)_([a-z0-9_:.-]+?)(?:_sync)?$", command
    )
    if stream_match:
        provider, model = stream_match.groups()
        is_sync = command.endswith("_sync")
        return provider, model, is_sync, True

    # Check for ask functions
    ask_match = re.match(r"^ask_([a-z0-9_]+?)_([a-z0-9_:.-]+?)(?:_sync)?$", command)
    if ask_match:
        provider, model = ask_match.groups()
        is_sync = command.endswith("_sync")
        return provider, model, is_sync, False

    return None


def trigger_discovery_for_provider(provider: str, quiet: bool = True) -> bool:
    """
    Trigger discovery for a specific provider.

    Returns:
        True if discovery succeeded, False otherwise
    """
    try:
        config_manager = get_config()
        provider_config = config_manager.get_provider(provider)

        # Check if provider has discovery enabled in its configuration
        if provider_config.extra and "dynamic_discovery" in provider_config.extra:
            discovery_config = provider_config.extra["dynamic_discovery"]
            if not discovery_config.get("enabled", False):
                if not quiet:
                    import logging

                    logging.getLogger(__name__).info(
                        f"Discovery disabled for {provider} in configuration"
                    )
                return False

        # For Ollama, use its specific discovery
        if provider == "ollama":
            new_functions = trigger_ollama_discovery_and_refresh()
            return bool(new_functions)

        # For OpenAI-compatible providers, trigger discovery through config manager
        if provider_config.client_class and "OpenAI" in provider_config.client_class:
            try:
                # Force discovery refresh through config manager
                import asyncio

                async def discover() -> None:
                    # Get discovery config from provider
                    discovery_data = provider_config.extra.get("dynamic_discovery", {})
                    if not discovery_data.get("enabled", False):
                        return False

                    from chuk_llm.configuration.models import DiscoveryConfig

                    discovery_config = DiscoveryConfig(
                        enabled=True,
                        discoverer_type=discovery_data.get("discoverer_type", "openai"),
                        cache_timeout=0
                        if not quiet
                        else discovery_data.get(
                            "cache_timeout", 300
                        ),  # Force refresh in CLI
                        inference_config=discovery_data.get("inference_config", {}),
                        discoverer_config=discovery_data.get("discoverer_config", {}),
                    )

                    # Use config manager's discovery system
                    success = await config_manager._refresh_provider_models(
                        provider, discovery_config
                    )
                    return success

                # Run discovery
                success = asyncio.run(discover())

                if success and not quiet:
                    discovered = config_manager.get_discovered_models(provider)
                    import logging

                    logging.getLogger(__name__).info(
                        f"Discovered {len(discovered)} models for {provider}"
                    )

                return success

            except Exception as e:
                if not quiet:
                    import logging

                    logging.getLogger(__name__).debug(
                        f"Discovery failed for {provider}: {e}"
                    )
                return False

        # For other providers, try to refresh their functions
        new_functions = refresh_provider_functions(provider)
        return bool(new_functions)

    except Exception as e:
        if not quiet:
            import logging

            logging.getLogger(__name__).debug(f"Discovery error for {provider}: {e}")
        return False


class ChukLLMCLI:
    """ChukLLM Command Line Interface"""

    def __init__(self, verbose: bool = False) -> None:
        self.config = get_config()
        self.verbose = verbose

    def print_rich(self, content: str, style: str = "") -> None:
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

    def print_table(
        self, headers: list[str], rows: list[list[str]], title: str = ""
    ) -> None:
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

    def ask_model(
        self,
        prompt: str,
        provider: str,
        model: str | None = None,
        json_mode: bool = False,
        stream: bool = True,
        base_url: str | None = None,
        api_key: str | None = None,
        system_prompt: str | None = None,
    ) -> str:
        """Ask a question to a specific model using sync API with optional streaming"""
        try:
            # Try to resolve the model name
            if model:
                # Check if the model is a global alias that points to a different provider
                global_aliases = self.config.get_global_aliases()
                if model in global_aliases:
                    alias_target = global_aliases[model]
                    if "/" in alias_target:
                        # It's a provider/model alias
                        alias_provider, alias_model = alias_target.split("/", 1)
                        if alias_provider == provider:
                            # Same provider, just use the aliased model
                            model = alias_model
                            if self.verbose:
                                self.print_rich(
                                    f"📝 Using aliased model '{model}' for global alias",
                                    "info",
                                )
                        # If different provider, we keep the original model and let it fail
                        # (user explicitly specified a provider)

                # Now try resolution with the config (handles provider-specific aliases and discovery)
                resolved_model = self.config._ensure_model_available(provider, model)

                if resolved_model and resolved_model != model:
                    if self.verbose:
                        self.print_rich(
                            f"📝 Using model '{resolved_model}' for '{model}'", "info"
                        )
                    model = resolved_model
                elif not resolved_model:
                    # Model couldn't be resolved
                    # Let it pass through - the API will give a proper error message
                    if self.verbose:
                        self.print_rich(
                            f"⚠️ Model '{model}' not found in config, trying anyway",
                            "warning",
                        )

            # Apply dynamic configuration if provided
            extra_kwargs = {}
            if base_url:
                extra_kwargs["base_url"] = base_url
            if api_key:
                extra_kwargs["api_key"] = api_key
            if system_prompt:
                extra_kwargs["system_prompt"] = system_prompt

            if stream and not json_mode:
                # Use streaming (this prints as it goes)
                return self.stream_response(
                    prompt, provider, model, json_mode=json_mode, **extra_kwargs
                )
            else:
                # Non-streaming for JSON mode or when explicitly disabled
                if model:
                    response = ask_sync(
                        prompt,
                        provider=provider,
                        model=model,
                        json_mode=json_mode,
                        **extra_kwargs,
                    )
                else:
                    response = ask_sync(
                        prompt, provider=provider, json_mode=json_mode, **extra_kwargs
                    )

                # Print the response for non-streaming
                print(response)
                return response
        except Exception as e:
            error_msg = f"Failed to get response from {provider}: {e}"
            print(error_msg)
            raise Exception(error_msg) from e

    def handle_ask_alias(self, alias: str, prompt: str, **kwargs: Any) -> str:
        """Handle ask_alias commands (like ask_granite) with streaming"""
        try:
            # Check if this is a global alias
            global_aliases = self.config.get_global_aliases()

            if alias in global_aliases:
                alias_target = global_aliases[alias]
                if "/" in alias_target:
                    provider, model = alias_target.split("/", 1)
                    if self.verbose:
                        self.print_rich(
                            f"Using {provider}/{model} via alias '{alias}'", "info"
                        )
                    return self.stream_response(
                        prompt, provider=provider, model=model, **kwargs
                    )
                else:
                    if self.verbose:
                        self.print_rich(
                            f"Using provider '{alias_target}' via alias '{alias}'",
                            "info",
                        )
                    return self.stream_response(prompt, provider=alias_target, **kwargs)
            else:
                # Try as direct provider
                if self.verbose:
                    self.print_rich(f"Using provider '{alias}' directly", "info")
                return self.stream_response(prompt, provider=alias, **kwargs)

        except Exception as e:
            raise Exception(f"Alias or provider '{alias}' not available: {e}") from e

    def handle_convenience_function(
        self, function_name: str, prompt: str, **kwargs
    ) -> str:
        """Handle convenience function calls like ask_ollama_gemma3"""
        try:
            # Check if the function exists
            if not has_function(function_name):
                raise Exception(f"Function '{function_name}' not available")

            # Get the actual function
            func = _GENERATED_FUNCTIONS[function_name]

            # Show what we're calling in verbose mode
            if self.verbose:
                parsed = parse_convenience_function(function_name)
                if parsed:
                    provider, model, is_sync, is_stream = parsed
                    self.print_rich(
                        f"🤖 {provider}/{model} (via {function_name}):", "info"
                    )
                    print("")

            # Handle different function types
            if function_name.endswith("_sync"):
                # Sync function - call directly
                response = func(prompt, **kwargs)
                print(response)
                return response
            elif function_name.startswith("stream_"):
                # Stream function - handle async streaming
                from .api.event_loop_manager import run_sync

                async def stream_and_print() -> None:
                    full_response = ""
                    async for chunk in func(prompt, **kwargs):
                        if isinstance(chunk, dict):
                            content = chunk.get("response", "")
                        else:
                            content = str(chunk)
                        print(content, end="", flush=True)
                        full_response += content
                    print()  # Final newline
                    return full_response

                return run_sync(stream_and_print())
            else:
                # Regular function - might be sync or async due to auto-detection
                result = func(prompt, **kwargs)

                # Check if it's a coroutine (async) or already a result (sync)
                import inspect

                if inspect.iscoroutine(result):
                    # It's async, need to run it
                    from .api.event_loop_manager import run_sync

                    response = run_sync(result)
                else:
                    # It's already a result (auto-detected sync context)
                    response = result

                print(response)
                return response

        except Exception as e:
            raise Exception(
                f"Error calling convenience function '{function_name}': {e}"
            ) from e

    def stream_response(
        self, prompt: str, provider: str = None, model: str = None, **kwargs
    ) -> str:
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

                # Show dynamic configuration if provided
                if "base_url" in kwargs:
                    self.print_rich(f"  Base URL: {kwargs['base_url']}", "info")
                if "api_key" in kwargs:
                    self.print_rich(
                        f"  API Key: ***{kwargs['api_key'][-4:] if len(kwargs['api_key']) > 4 else '****'}",
                        "info",
                    )
                print("")  # Add a blank line

            # Stream the response
            for chunk in stream_sync_iter(
                prompt, provider=provider, model=model, **kwargs
            ):
                print(chunk, end="", flush=True)
                full_response += chunk

            print()  # Add final newline after streaming
            return full_response

        except Exception as e:
            # Fallback to non-streaming
            if self.verbose:
                self.print_rich(
                    f"⚠ Streaming failed ({e}), using non-streaming", "error"
                )

            # Try non-streaming fallback
            try:
                if provider and model:
                    response = ask_sync(
                        prompt, provider=provider, model=model, **kwargs
                    )
                elif provider:
                    response = ask_sync(prompt, provider=provider, **kwargs)
                else:
                    response = ask_sync(prompt, **kwargs)

                print(response)  # Print the fallback response
                return response
            except Exception as fallback_error:
                error_msg = (
                    f"Both streaming and non-streaming failed: {e}, {fallback_error}"
                )
                print(error_msg)
                raise Exception(error_msg) from fallback_error

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

                    rows.append(
                        [
                            provider_name,
                            provider.default_model or "N/A",
                            str(model_count),
                            features,
                            "✓" if self.config.get_api_key(provider_name) else "✗",
                        ]
                    )
                except Exception as e:
                    rows.append([provider_name, "ERROR", "0", str(e), "✗"])

            self.print_table(
                ["Provider", "Default Model", "Models", "Features", "API Key"],
                rows,
                "Available Providers",
            )
        except Exception as e:
            self.print_rich(f"Error listing providers: {e}", "error")

    def show_models(self, provider: str) -> None:
        """List models for a specific provider with discovery"""
        try:
            provider_config = self.config.get_provider(provider)

            # Trigger discovery based on provider type
            self.print_rich(
                f"🔍 Fetching available {provider} models from API...", "info"
            )

            # Use the discovery system properly
            discovered_models = set()
            discovery_succeeded = False

            if provider == "ollama":
                try:
                    from chuk_llm.api.providers import (
                        trigger_ollama_discovery_and_refresh,
                    )

                    trigger_ollama_discovery_and_refresh()
                    # Get discovered models from config manager
                    discovered_models = self.config.get_discovered_models(provider)
                    discovery_succeeded = bool(discovered_models)
                except Exception as e:
                    self.print_rich(f"⚠️  Ollama discovery unavailable: {e}", "error")
            else:
                # Use the config manager's discovery instead of making direct API calls
                try:
                    # Trigger discovery through config manager
                    success = trigger_discovery_for_provider(provider, quiet=False)
                    if success:
                        # Get discovered models from config manager
                        discovered_models = self.config.get_discovered_models(provider)
                        discovery_succeeded = bool(discovered_models)
                        if discovered_models:
                            self.print_rich(
                                f"✅ Found {len(discovered_models)} available models",
                                "success",
                            )
                except Exception as e:
                    self.print_rich(f"⚠️  Discovery failed for {provider}: {e}", "error")

            # Get all models (static + discovered) from config manager
            all_models = self.config.get_all_available_models(provider)
            static_models = set(provider_config.models)

            if not all_models:
                self.print_rich(f"No models found for provider '{provider}'", "error")
                return

            # Display models with proper source indication
            rows = []
            for model in sorted(all_models):
                # Determine source and status correctly
                in_static = model in static_models
                in_discovered = model in discovered_models

                if in_static and in_discovered:
                    source = "✅ Configured + Available"
                    status = "🟢"
                elif in_static and not in_discovered and discovery_succeeded:
                    # We did discovery but didn't find this model
                    source = "📋 Configured (Not Found)"
                    status = "🔴"
                elif in_static:
                    # No discovery was done or discovery failed
                    source = "📋 Configured"
                    status = "⚪"
                elif in_discovered:
                    source = "🔍 Discovered Only"
                    status = "🟡"
                else:
                    source = "❓ Unknown"
                    status = "❓"

                # Get capabilities from provider config
                try:
                    caps = provider_config.get_model_capabilities(model)
                    features = ", ".join([f.value for f in list(caps.features)[:3]])
                    if len(caps.features) > 3:
                        features += "..."

                    context = (
                        str(caps.max_context_length)
                        if caps.max_context_length
                        else "N/A"
                    )
                    output = (
                        str(caps.max_output_tokens) if caps.max_output_tokens else "N/A"
                    )
                except Exception:
                    features = "system_messages, tools, streaming..."
                    context = "131072"  # Groq default
                    output = "32768"  # Groq default

                rows.append([status, model, source, context, output, features])

            # Add aliases section
            if provider_config.model_aliases:
                rows.append(["", "--- ALIASES ---", "", "", "", ""])
                for alias, target in provider_config.model_aliases.items():
                    # Check if target is available
                    target_status = "\U0001f7e2" if target in all_models else "🔴"
                    rows.append(
                        [target_status, f"{alias} → {target}", "Alias", "", "", ""]
                    )

            # Calculate statistics
            static_count = len(static_models)
            discovered_count = len(discovered_models)
            total_count = len(all_models)
            new_count = len([m for m in discovered_models if m not in static_models])

            self.print_table(
                ["", "Model", "Source", "Context", "Output", "Features"],
                rows,
                f"Models for {provider} ({total_count} total, {static_count} configured, {discovered_count} discovered, {new_count} new)",
            )

            # Show helpful hints
            if provider == "ollama" and not discovered_models:
                self.print_rich(
                    "\n💡 Tip: Make sure Ollama is running to see available models",
                    "info",
                )
            elif new_count > 0:
                self.print_rich(
                    f"\n✨ Found {new_count} additional models not in configuration!",
                    "success",
                )
                self.print_rich(
                    "These models are immediately available for use.", "info"
                )
            elif discovered_models and new_count == 0:
                self.print_rich(
                    f"\n✅ All {discovered_count} discovered models are already configured",
                    "info",
                )

        except Exception as e:
            self.print_rich(f"Error listing models for '{provider}': {e}", "error")
            if self.verbose:
                import traceback

                self.print_rich(traceback.format_exc(), "error")

    def test_provider(self, provider: str) -> None:
        """Test if a provider is working"""
        try:
            provider_config = self.config.get_provider(provider)
            api_key = self.config.get_api_key(provider)

            self.print_rich(f"Testing provider: {provider}", "info")

            # Check configuration
            self.print_rich("✓ Provider configuration found", "success")
            self.print_rich(f"  - Client: {provider_config.client_class}")
            self.print_rich(f"  - Default model: {provider_config.default_model}")
            self.print_rich(
                f"  - Models available: {len(provider_config.models) if provider_config.models else 0}"
            )

            # Check API key
            if provider.lower() == "ollama":
                self.print_rich("✓ Ollama doesn't require an API key", "success")
            elif api_key:
                self.print_rich("✓ API key found", "success")
            else:
                self.print_rich("✗ No API key found", "error")
                if provider_config.api_key_env:
                    self.print_rich(
                        f"  Set environment variable: {provider_config.api_key_env}"
                    )
                return

            # Test basic capabilities
            try:
                can_handle, problems = CapabilityChecker.can_handle_request(provider)
                if can_handle:
                    self.print_rich("✓ Basic capabilities check passed", "success")
                else:
                    self.print_rich(
                        f"⚠ Capability issues: {', '.join(problems)}", "error"
                    )
            except Exception as e:
                self.print_rich(f"⚠ Could not check capabilities: {e}", "error")

            # Try a simple request
            self.print_rich("Testing with simple request...", "info")

            try:
                response = self.ask_model(
                    "Say 'Hello from ChukLLM CLI!' and nothing else.", provider
                )
                self.print_rich("✓ Test request successful", "success")
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
                self.print_rich(
                    f"✓ Discovered {len(new_functions)} new Ollama functions", "success"
                )
            else:
                new_functions = refresh_provider_functions(provider)
                self.print_rich(
                    f"✓ Refreshed {len(new_functions)} functions for {provider}",
                    "success",
                )

            # Show some discovered functions
            if new_functions:
                provider_funcs = [
                    name
                    for name in new_functions
                    if name.startswith(f"ask_{provider}_")
                    and not name.endswith("_sync")
                ][:5]
                if provider_funcs:
                    self.print_rich("Example functions:", "info")
                    for func_name in provider_funcs:
                        self.print_rich(f'  - chuk-llm {func_name} "your question"')

        except Exception as e:
            self.print_rich(f"Error discovering models for {provider}: {e}", "error")

    def show_functions(self, provider: str = None) -> None:
        """List all available provider functions"""
        try:
            all_functions = list_provider_functions()

            # Filter by provider if specified
            if provider:
                # Trigger discovery for the provider first
                self.print_rich(
                    f"🔍 Fetching available {provider} functions from API...", "info"
                )

                discovered_count = 0
                if provider == "ollama":
                    try:
                        from chuk_llm.api.providers import (
                            trigger_ollama_discovery_and_refresh,
                        )

                        new_functions = trigger_ollama_discovery_and_refresh()
                        discovered_count = len(new_functions)
                    except Exception as e:
                        self.print_rich(
                            f"⚠️  Ollama discovery unavailable: {e}", "error"
                        )
                else:
                    try:
                        success = trigger_discovery_for_provider(provider, quiet=False)
                        if success:
                            discovered = get_discovered_functions(provider)
                            if discovered and provider in discovered:
                                discovered_count = len(discovered[provider])
                    except Exception as e:
                        self.print_rich(
                            f"⚠️  Discovery failed for {provider}: {e}", "error"
                        )

                # Filter functions for this provider
                provider_prefix = f"ask_{provider}_"
                stream_prefix = f"stream_{provider}_"
                all_functions = [
                    f
                    for f in all_functions
                    if f.startswith(provider_prefix) or f.startswith(stream_prefix)
                ]

            if not all_functions:
                if provider:
                    self.print_rich(
                        f"No functions found for provider '{provider}'", "error"
                    )
                else:
                    self.print_rich("No functions found", "error")
                return

            # Group by type
            ask_funcs = [
                f
                for f in all_functions
                if f.startswith("ask_") and not f.endswith("_sync")
            ]
            stream_funcs = [
                f
                for f in all_functions
                if f.startswith("stream_") and not f.endswith("_sync")
            ]
            ask_sync_funcs = [
                f for f in all_functions if f.startswith("ask_") and f.endswith("_sync")
            ]
            stream_sync_funcs = [
                f
                for f in all_functions
                if f.startswith("stream_") and f.endswith("_sync")
            ]

            # Calculate counts
            async_count = len(ask_funcs) + len(stream_funcs)
            sync_count = len(ask_sync_funcs) + len(stream_sync_funcs)

            # Get discovered functions to mark source
            discovered_all = get_discovered_functions(provider)
            discovered_funcs = set()
            if discovered_all:
                for prov, funcs in discovered_all.items():
                    if not provider or prov == provider:
                        discovered_funcs.update(funcs.keys())

            # Build the table
            rows = []

            # Add async functions
            for func in sorted(ask_funcs):
                source = (
                    "🔍 Discovered" if func in discovered_funcs else "📋 Configured"
                )
                rows.append(["ask", func, source, "async"])

            for func in sorted(stream_funcs):
                source = (
                    "🔍 Discovered" if func in discovered_funcs else "📋 Configured"
                )
                rows.append(["stream", func, source, "async"])

            # Add separator
            if ask_funcs or stream_funcs:
                rows.append(["", "--- SYNC FUNCTIONS ---", "", ""])

            # Add sync functions
            for func in sorted(ask_sync_funcs):
                source = (
                    "🔍 Discovered" if func in discovered_funcs else "📋 Configured"
                )
                rows.append(["ask", func, source, "sync"])

            for func in sorted(stream_sync_funcs):
                source = (
                    "🔍 Discovered" if func in discovered_funcs else "📋 Configured"
                )
                rows.append(["stream", func, source, "sync"])

            # Create title
            if provider:
                configured_count = len(
                    [f for f in all_functions if f not in discovered_funcs]
                )
                title = f"Functions for {provider} ({len(all_functions)} total, {configured_count} configured, {discovered_count} discovered)"
            else:
                title = f"All Provider Functions ({len(all_functions)} total, {async_count} async, {sync_count} sync)"

            # Print the table
            self.print_table(["Type", "Function", "Source", "Mode"], rows, title)

            # Add usage examples below the table
            if rows:
                self.print_rich("\n📖 Usage examples:", "info")
                if ask_funcs or ask_sync_funcs:
                    example_func = ask_funcs[0] if ask_funcs else ask_sync_funcs[0]
                    self.print_rich(f'  chuk-llm {example_func} "your question"')
                if stream_funcs or stream_sync_funcs:
                    example_func = (
                        stream_funcs[0] if stream_funcs else stream_sync_funcs[0]
                    )
                    self.print_rich(f'  chuk-llm {example_func} "tell me a story"')

            if provider == "ollama":
                self.print_rich(
                    "\n💡 Tip: Make sure Ollama is running to see available functions",
                    "info",
                )

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
                    self.print_rich(
                        f"\nDiscovered functions for {provider_name}:", "info"
                    )
                    for func_name in sorted(functions.keys())[:10]:
                        self.print_rich(f'  - chuk-llm {func_name} "your question"')
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

                self.print_table(["Alias", "Target"], rows, "Global Aliases")

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
                "Available Global Aliases",
            )

            self.print_rich("\nExample usage:", "info")
            example_alias = list(global_aliases.keys())[0]
            self.print_rich(f'  chuk-llm ask_{example_alias} "Your question here"')

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

## Convenience Functions (Generated by Discovery)
```bash
chuk-llm ask_ollama_gemma3 "Hello world"
chuk-llm ask_ollama_mistral_small_latest "Tell me a joke"
chuk-llm stream_ollama_qwen3 "Explain Python in detail"
```

## General Ask Command
```bash
chuk-llm ask "Question" --provider openai --model gpt-4o-mini
chuk-llm ask "Question" --json  # Request JSON response

# Dynamic provider configuration
chuk-llm ask "Question" --provider openai --base-url https://api.custom.com/v1 --api-key sk-custom-key
chuk-llm ask "Question" --provider ollama --base-url http://remote-server:11434
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

# Using discovered convenience functions
chuk-llm ask_ollama_gemma3 "What is machine learning?"
chuk-llm ask_ollama_qwen3_latest "Write a Python function"

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
uvx chuk-llm ask_ollama_gemma3 "Hello"
uvx chuk-llm providers
uvx chuk-llm discover ollama
```
"""
        self.print_rich(help_text, "markdown")


def main() -> None:
    """Main CLI entry point with automatic discovery for convenience functions"""
    if len(sys.argv) < 2:
        print("ChukLLM CLI - Quick access to AI models")
        print("")
        print("Usage:")
        print('  chuk-llm ask_granite "What is Python?"')
        print(
            '  chuk-llm ask_ollama_gemma3 "Hello world"          # NEW: Convenience functions'
        )
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

    # Normalize command name: convert dots to underscores for function names
    # This allows users to type ask_ollama_granite3.3 instead of ask_ollama_granite3_3
    normalized_command = command.replace(".", "_")

    # Update sys.argv to reflect the filtered args for subcommand parsing
    sys.argv = ["chuk-llm"] + args

    cli = ChukLLMCLI(verbose=verbose)

    # 🎯 AUTO-DISCOVERY: Check if this is a convenience function and trigger discovery if needed
    parsed_convenience = parse_convenience_function(normalized_command)
    if parsed_convenience:
        provider, model, is_sync, is_stream = parsed_convenience

        # Check if function exists, if not trigger discovery silently
        if not has_function(normalized_command):
            trigger_discovery_for_provider(provider, quiet=True)

            # Check again after discovery
            if not has_function(normalized_command):
                # Only show error if discovery didn't help
                print(f"❌ Function '{command}' not available")
                print(
                    f'💡 Try: chuk-llm ask --provider {provider} --model {model} "your question"'
                )
                sys.exit(1)

    try:
        # 🎯 Handle convenience functions first (ask_provider_model pattern)
        if parsed_convenience:
            # Parse arguments for convenience functions
            parser = argparse.ArgumentParser(description=f"Use {command} function")
            parser.add_argument("prompt", help="The question to ask")
            parser.add_argument(
                "--system-prompt",
                "-s",
                help="System prompt to set the AI's behavior/personality",
            )
            parser.add_argument(
                "--max-tokens", type=int, help="Maximum tokens in response"
            )
            parser.add_argument(
                "--temperature", type=float, help="Temperature for sampling"
            )

            try:
                parsed_args = parser.parse_args(args[1:])

                # Build kwargs from parsed args
                kwargs = {}
                if parsed_args.system_prompt:
                    kwargs["system_prompt"] = parsed_args.system_prompt
                if parsed_args.max_tokens:
                    kwargs["max_tokens"] = parsed_args.max_tokens
                if parsed_args.temperature:
                    kwargs["temperature"] = parsed_args.temperature

                cli.handle_convenience_function(
                    normalized_command, parsed_args.prompt, **kwargs
                )
                return
            except SystemExit:
                # argparse calls sys.exit on error, catch and show usage
                print(
                    f'Usage: chuk-llm {command} "your question here" [--system-prompt "..."]'
                )
                sys.exit(1)

        # Handle ask_alias commands (ask_granite, ask_claude, etc.)
        elif normalized_command.startswith("ask_"):
            # Parse arguments for ask_alias commands
            parser = argparse.ArgumentParser(description=f"Use {command} alias")
            parser.add_argument("prompt", help="The question to ask")
            parser.add_argument(
                "--system-prompt",
                "-s",
                help="System prompt to set the AI's behavior/personality",
            )
            parser.add_argument(
                "--max-tokens", type=int, help="Maximum tokens in response"
            )
            parser.add_argument(
                "--temperature", type=float, help="Temperature for sampling"
            )

            try:
                parsed_args = parser.parse_args(args[1:])

                alias = normalized_command[4:]  # Remove 'ask_' prefix

                # Build kwargs from parsed args
                kwargs = {}
                if parsed_args.system_prompt:
                    kwargs["system_prompt"] = parsed_args.system_prompt
                if parsed_args.max_tokens:
                    kwargs["max_tokens"] = parsed_args.max_tokens
                if parsed_args.temperature:
                    kwargs["temperature"] = parsed_args.temperature

                # Note: We don't need to print the response since streaming handles it
                cli.handle_ask_alias(alias, parsed_args.prompt, **kwargs)
            except SystemExit:
                # argparse calls sys.exit on error, catch and show usage
                print(
                    f'Usage: chuk-llm {command} "your question here" [--system-prompt "..."]'
                )
                sys.exit(1)

        # Handle general ask command
        elif command == "ask":
            parser = argparse.ArgumentParser(description="Ask a question")
            parser.add_argument("prompt", help="The question to ask")
            parser.add_argument(
                "--provider", "-p", required=True, help="Provider to use"
            )
            parser.add_argument("--model", "-m", help="Specific model to use")
            parser.add_argument(
                "--base-url", "-b", help="Override base URL for the provider"
            )
            parser.add_argument(
                "--api-key", "-k", help="Override API key for the provider"
            )
            parser.add_argument(
                "--json", action="store_true", help="Request JSON response"
            )
            parser.add_argument(
                "--no-stream", action="store_true", help="Disable streaming"
            )
            parser.add_argument(
                "--system-prompt",
                "-s",
                help="System prompt to set the AI's behavior/personality",
            )

            try:
                parsed_args = parser.parse_args(args[1:])  # Use filtered args

                # 🎯 AUTO-DISCOVERY: If model is specified, ensure it's available
                if parsed_args.model and parsed_args.provider:
                    # Check if the model is available for this provider
                    try:
                        provider_config = cli.config.get_provider(parsed_args.provider)
                        if parsed_args.model not in provider_config.models:
                            # Model not found, try discovery
                            trigger_discovery_for_provider(
                                parsed_args.provider, quiet=True
                            )
                    except Exception:
                        # If provider doesn't exist, let the normal error handling deal with it
                        pass

                # Call ask_model and ensure response is handled
                response = cli.ask_model(
                    parsed_args.prompt,
                    parsed_args.provider,
                    parsed_args.model,
                    json_mode=parsed_args.json,
                    stream=not parsed_args.no_stream,
                    base_url=parsed_args.base_url,
                    api_key=parsed_args.api_key,
                    system_prompt=parsed_args.system_prompt,
                )

                # The streaming methods should already print, but ensure we have output
                if not response and not parsed_args.no_stream:
                    # If streaming didn't work, try non-streaming fallback
                    if not quiet:
                        print("⚠ Streaming may have failed, trying non-streaming...")
                    response = cli.ask_model(
                        parsed_args.prompt,
                        parsed_args.provider,
                        parsed_args.model,
                        json_mode=parsed_args.json,
                        stream=False,
                        base_url=parsed_args.base_url,
                        api_key=parsed_args.api_key,
                        system_prompt=parsed_args.system_prompt,
                    )
                    if response:
                        print(response)

            except SystemExit:
                # Handle argument parsing errors gracefully
                print("Error: Invalid arguments. Use quotes around your question.")
                print(
                    'Example: chuk-llm ask "What is machine learning?" --provider ollama'
                )
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
            # Support optional provider filter like models command
            provider = args[1] if len(args) > 1 else None
            cli.show_functions(provider)

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
