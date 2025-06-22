# diagnostics/capabilities/utils/display_utils.py
"""
Enhanced display utilities for diagnostic results.
Updated to work with the new result models and provider system.
"""
from __future__ import annotations

from typing import List, Optional
from .result_models import ProviderResult

# Optional Rich imports
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    console = Console()
    HAS_RICH = True
except ImportError:
    console = None
    HAS_RICH = False

class DiagnosticDisplay:
    """Handles formatting and display of diagnostic results"""
    
    @staticmethod
    def print_summary(results: List[ProviderResult]):
        """Print a comprehensive summary of all results"""
        if not HAS_RICH:
            DiagnosticDisplay._print_plain_summary(results)
            return
        
        # Performance overview
        DiagnosticDisplay._print_performance_summary(results)
        
        # Timing table
        DiagnosticDisplay._print_timing_table(results)
        
        # Error analysis
        DiagnosticDisplay._print_error_analysis(results)
        
        # Main results table
        DiagnosticDisplay._print_results_table(results)
    
    @staticmethod
    def _print_performance_summary(results: List[ProviderResult]):
        """Print a high-level performance overview"""
        if not results:
            return
        
        # Calculate aggregate stats
        total_providers = len(results)
        providers_with_all_features = len([r for r in results if len(r.feature_set) >= 4])
        providers_with_errors = len([r for r in results if r.has_errors()])
        
        avg_success_rate = sum(r.success_rate for r in results) / len(results)
        
        # Create performance panel
        perf_text = Text()
        perf_text.append(f"📊 {total_providers} providers tested\n")
        perf_text.append(f"🎯 {providers_with_all_features} full-featured providers\n")
        perf_text.append(f"⚠️  {providers_with_errors} providers with errors\n")
        perf_text.append(f"📈 {avg_success_rate:.1%} average success rate")
        
        console.print(Panel(perf_text, title="🚀 Performance Overview", border_style="blue"))
    
    @staticmethod
    def _print_timing_table(results: List[ProviderResult]):
        """Print detailed timing information"""
        timing_table = Table(title="⏱️ Execution Time (seconds)")
        timing_table.add_column("Provider", style="cyan", min_width=12)
        
        capabilities = [("Text", "text"), ("Stream", "stream"), ("Tools", "tools"), 
                       ("Str.Tools", "stream_tools"), ("Vision", "vision")]
        
        for display_name, _ in capabilities:
            timing_table.add_column(display_name, justify="right", min_width=8)
        
        timing_table.add_column("Avg", justify="right", style="bold", min_width=6)
        
        for result in results:
            # Calculate average timing
            times = [t for t in result.timings.values() if t > 0]
            avg_time = sum(times) / len(times) if times else 0
            
            row_data = [result.provider]
            
            for _, timing_key in capabilities:
                if timing_key in result.timings:
                    time_val = result.timings[timing_key]
                    # Color code timing: green < 2s, yellow < 5s, red >= 5s
                    if time_val < 2.0:
                        style = "green"
                    elif time_val < 5.0:
                        style = "yellow"
                    else:
                        style = "red"
                    row_data.append(f"[{style}]{time_val:.2f}[/{style}]")
                else:
                    row_data.append("—")
            
            # Add average
            if avg_time > 0:
                row_data.append(f"{avg_time:.2f}")
            else:
                row_data.append("—")
            
            timing_table.add_row(*row_data)
        
        console.print(timing_table)
    
    @staticmethod
    def _print_error_analysis(results: List[ProviderResult]):
        """Print detailed error analysis"""
        errors = [
            (result.provider, stage, msg.splitlines()[0][:100])
            for result in results 
            for stage, msg in result.errors.items()
        ]
        
        if not errors:
            return
        
        error_table = Table(title="❌ Error Analysis")
        error_table.add_column("Provider", style="red", min_width=12)
        error_table.add_column("Capability", style="yellow", min_width=12)
        error_table.add_column("Error Message", style="dim")
        
        for provider, stage, message in errors:
            # Categorize error types
            if "rate limit" in message.lower():
                stage_style = "orange3"
            elif "not support" in message.lower():
                stage_style = "blue"
            elif "authentication" in message.lower() or "api key" in message.lower():
                stage_style = "red"
            else:
                stage_style = "yellow"
            
            error_table.add_row(
                provider, 
                f"[{stage_style}]{stage}[/{stage_style}]", 
                message
            )
        
        console.print(error_table)
    
    @staticmethod
    def _print_results_table(results: List[ProviderResult]):
        """Print the main results table"""
        results_table = Table(title="🏆 LLM Provider Diagnostics - Final Results")
        results_table.add_column("Provider", style="cyan", min_width=12)
        results_table.add_column("Model(s)", style="blue", min_width=20)
        
        capabilities = [
            ("Text", "text_completion"), 
            ("Stream", "streaming_text"), 
            ("Tools", "function_call"), 
            ("Str.Tools", "streaming_function_call"), 
            ("Vision", "vision")
        ]
        
        for display_name, _ in capabilities:
            results_table.add_column(display_name, justify="center", min_width=8)
        
        results_table.add_column("Success Rate", justify="center", style="bold", min_width=12)
        results_table.add_column("Features", style="magenta", min_width=15)
        
        for result in results:
            # Truncate long model summaries
            model_summary = ", ".join(f"{cap}:{model}" for cap, model in result.models.items())
            if len(model_summary) > 35:
                model_summary = model_summary[:32] + "..."
            
            row_data = [result.provider, model_summary]
            
            # Add capability results
            for _, attr_name in capabilities:
                capability_value = getattr(result, attr_name)
                row_data.append(DiagnosticDisplay._format_bool(capability_value))
            
            # Add success rate with color coding
            success_rate = result.success_rate
            if success_rate >= 0.8:
                rate_style = "green"
            elif success_rate >= 0.5:
                rate_style = "yellow"
            else:
                rate_style = "red"
            
            row_data.append(f"[{rate_style}]{success_rate:.1%}[/{rate_style}]")
            row_data.append(", ".join(sorted(result.feature_set)) or "—")
            
            results_table.add_row(*row_data)
        
        console.print(results_table)
    
    @staticmethod
    def _format_bool(value: Optional[bool]) -> str:
        """Format boolean values for display with colors"""
        if value is True:
            return "[green]✅[/green]"
        elif value is False:
            return "[red]❌[/red]"
        else:
            return "[dim]—[/dim]"
    
    @staticmethod
    def _print_plain_summary(results: List[ProviderResult]):
        """Plain text fallback when Rich is not available"""
        print("\n" + "="*80)
        print("DIAGNOSTIC RESULTS SUMMARY")
        print("="*80)
        
        for result in results:
            features = ", ".join(sorted(result.feature_set)) or "—"
            models_str = ", ".join(f"{k}:{v}" for k, v in result.models.items())
            
            print(f"\n{result.provider.upper()}")
            print(f"  Features: {features}")
            print(f"  Models: {models_str}")
            print(f"  Success Rate: {result.success_rate:.1%}")
            print(f"  Capabilities: {result.successful_capabilities}/{result.total_capabilities}")
            
            if result.has_errors():
                print(f"  Errors: {len(result.errors)} issues")
                for stage, error in result.errors.items():
                    print(f"    {stage}: {error.splitlines()[0][:80]}")
            
            # Show timing summary
            if result.timings:
                avg_time = sum(result.timings.values()) / len(result.timings)
                fastest = result.get_fastest_capability()
                slowest = result.get_slowest_capability()
                print(f"  Timing: avg={avg_time:.2f}s, fastest={fastest}, slowest={slowest}")
        
        print("\n" + "="*80)
    
    @staticmethod
    def print_live_tick(provider: str, capability: str, success: Optional[bool], timing: float = None):
        """Print a live progress tick during testing"""
        status_icon = DiagnosticDisplay._format_bool(success)
        timing_str = f" ({timing:.2f}s)" if timing else ""
        
        if HAS_RICH:
            console.print(f"[cyan][{provider}][/cyan] {capability:<12} {status_icon}{timing_str}")
        else:
            plain_icon = {True: "✅", False: "❌", None: "—"}[success]
            print(f"[{provider}] {capability:<12} {plain_icon}{timing_str}")
    
    @staticmethod
    def print_provider_header(provider: str):
        """Print a header for a provider being tested"""
        if HAS_RICH:
            header_text = Text(f"Testing {provider.upper()}", style="bold cyan")
            console.print(Panel(header_text, expand=False))
        else:
            print(f"\n=== {provider.upper()} ===")
    
    @staticmethod
    def print_quick_summary(results: List[ProviderResult]):
        """Print a quick one-line summary for each provider"""
        if HAS_RICH:
            table = Table(title="Quick Summary")
            table.add_column("Provider", style="cyan")
            table.add_column("Score", justify="center")
            table.add_column("Features", style="magenta")
            
            for result in results:
                score = f"{result.successful_capabilities}/{result.total_capabilities}"
                features = ", ".join(sorted(result.feature_set)) or "—"
                table.add_row(result.provider, score, features)
            
            console.print(table)
        else:
            print("\nQUICK SUMMARY:")
            for result in results:
                score = f"{result.successful_capabilities}/{result.total_capabilities}"
                features = ", ".join(sorted(result.feature_set)) or "—"
                print(f"  {result.provider:<12} {score:<5} {features}")
    
    @staticmethod
    def print_capabilities_matrix(results: List[ProviderResult]):
        """Print a matrix showing which providers support which capabilities"""
        if not results:
            return
        
        if HAS_RICH:
            matrix_table = Table(title="🔍 Capabilities Matrix")
            matrix_table.add_column("Provider", style="cyan")
            
            capabilities = ["Text", "Stream", "Tools", "Str+Tools", "Vision"]
            for cap in capabilities:
                matrix_table.add_column(cap, justify="center")
            
            for result in results:
                row = [result.provider]
                row.append(DiagnosticDisplay._format_bool(result.text_completion))
                row.append(DiagnosticDisplay._format_bool(result.streaming_text))
                row.append(DiagnosticDisplay._format_bool(result.function_call))
                row.append(DiagnosticDisplay._format_bool(result.streaming_function_call))
                row.append(DiagnosticDisplay._format_bool(result.vision))
                
                matrix_table.add_row(*row)
            
            console.print(matrix_table)
        else:
            print("\nCAPABILITIES MATRIX:")
            header = f"{'Provider':<12} {'Text':<6} {'Stream':<8} {'Tools':<7} {'S+T':<5} {'Vision':<8}"
            print(header)
            print("-" * len(header))
            
            for result in results:
                text_icon = {True: "✅", False: "❌", None: "—"}[result.text_completion]
                stream_icon = {True: "✅", False: "❌", None: "—"}[result.streaming_text]
                tools_icon = {True: "✅", False: "❌", None: "—"}[result.function_call]
                stream_tools_icon = {True: "✅", False: "❌", None: "—"}[result.streaming_function_call]
                vision_icon = {True: "✅", False: "❌", None: "—"}[result.vision]
                
                print(f"{result.provider:<12} {text_icon:<6} {stream_icon:<8} {tools_icon:<7} {stream_tools_icon:<5} {vision_icon:<8}")