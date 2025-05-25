# benchmarks/llm_benchmark.py
"""
Comprehensive LLM Provider and Model Benchmarking System
"""
import asyncio
import json
import time
import statistics
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import logging
from datetime import datetime
import traceback

# chuk_llm imports
from chuk_llm.llm.llm_client import get_llm_client
from chuk_llm.llm.core.base import BaseLLMClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark tests"""
    name: str
    description: str
    messages: List[Dict[str, Any]]
    tools: Optional[List[Dict[str, Any]]] = None
    expected_features: Optional[List[str]] = None  # ["streaming", "tools", "json"]
    timeout: float = 60.0
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None

@dataclass
class PerformanceMetrics:
    """Performance metrics for a single test"""
    provider: str
    model: str
    test_name: str
    
    # Timing metrics
    first_token_latency: Optional[float] = None
    total_response_time: float = 0.0
    tokens_per_second: Optional[float] = None
    
    # Streaming metrics
    chunk_count: int = 0
    avg_chunk_interval: Optional[float] = None
    streaming_duration: Optional[float] = None
    
    # Quality metrics
    response_length: int = 0
    tool_calls_count: int = 0
    success: bool = False
    error_message: Optional[str] = None
    
    # Raw data
    full_response: Optional[str] = None
    tool_calls: Optional[List[Dict]] = None
    
    @property
    def summary(self) -> str:
        """Human readable summary"""
        if not self.success:
            return f"❌ FAILED: {self.error_message}"
        
        parts = [
            f"✅ {self.total_response_time:.2f}s total"
        ]
        
        if self.first_token_latency:
            parts.append(f"🚀 {self.first_token_latency:.2f}s first token")
        
        if self.tokens_per_second:
            parts.append(f"⚡ {self.tokens_per_second:.1f} tok/s")
        
        if self.chunk_count > 0:
            parts.append(f"📦 {self.chunk_count} chunks")
        
        if self.tool_calls_count > 0:
            parts.append(f"🔧 {self.tool_calls_count} tools")
        
        return " | ".join(parts)

class LLMBenchmark:
    """Main benchmarking system"""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Built-in test configurations
        self.test_configs = self._create_default_tests()
        
        # Results storage
        self.results: List[PerformanceMetrics] = []
    
    def _create_default_tests(self) -> List[BenchmarkConfig]:
        """Create default benchmark test configurations"""
        
        # Simple function for testing tools
        simple_tool = {
            "type": "function",
            "function": {
                "name": "get_current_time",
                "description": "Get the current time",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "timezone": {
                            "type": "string",
                            "description": "Timezone (e.g., 'UTC', 'EST')"
                        }
                    }
                }
            }
        }
        
        return [
            BenchmarkConfig(
                name="simple_response",
                description="Basic response generation",
                messages=[
                    {"role": "user", "content": "Hello! How are you today?"}
                ],
                expected_features=["basic"]
            ),
            
            BenchmarkConfig(
                name="creative_writing",
                description="Creative text generation",
                messages=[
                    {"role": "user", "content": "Write a short haiku about technology."}
                ],
                expected_features=["creative"]
            ),
            
            BenchmarkConfig(
                name="reasoning_task",
                description="Logical reasoning",
                messages=[
                    {"role": "user", "content": "If a train travels 60 mph for 2 hours, then 80 mph for 1.5 hours, what's the total distance?"}
                ],
                expected_features=["reasoning"]
            ),
            
            BenchmarkConfig(
                name="long_response",
                description="Extended response generation",
                messages=[
                    {"role": "user", "content": "Explain the history of the internet in detail."}
                ],
                max_tokens=500,
                expected_features=["long_form"]
            ),
            
            BenchmarkConfig(
                name="function_calling",
                description="Tool/function calling capability",
                messages=[
                    {"role": "user", "content": "What time is it in UTC?"}
                ],
                tools=[simple_tool],
                expected_features=["tools"]
            ),
            
            BenchmarkConfig(
                name="json_response",
                description="Structured JSON output",
                messages=[
                    {"role": "user", "content": "Return a JSON object with three programming languages and their primary use cases."}
                ],
                expected_features=["json"],
                temperature=0.1
            ),
            
            BenchmarkConfig(
                name="conversation_context",
                description="Multi-turn conversation handling",
                messages=[
                    {"role": "user", "content": "I'm thinking of a number between 1 and 10."},
                    {"role": "assistant", "content": "That sounds fun! I'd love to guess your number. Is it 7?"},
                    {"role": "user", "content": "No, it's higher than 7. Try again."}
                ],
                expected_features=["context"]
            ),
            
            BenchmarkConfig(
                name="streaming_test",
                description="Streaming response performance",
                messages=[
                    {"role": "user", "content": "Count from 1 to 20, putting each number on a new line."}
                ],
                expected_features=["streaming"]
            )
        ]
    
    def add_custom_test(self, config: BenchmarkConfig):
        """Add a custom test configuration"""
        self.test_configs.append(config)
    
    async def benchmark_provider_model(
        self,
        provider: str,
        model: str,
        test_configs: Optional[List[BenchmarkConfig]] = None,
        test_streaming: bool = True,
        verbose: bool = True
    ) -> List[PerformanceMetrics]:
        """Benchmark a specific provider and model combination"""
        
        if test_configs is None:
            test_configs = self.test_configs
        
        if verbose:
            print(f"\n🔬 Benchmarking {provider} - {model}")
            print("=" * 60)
        
        try:
            client = get_llm_client(provider, model=model)
        except Exception as e:
            error_msg = f"Failed to initialize {provider} client: {str(e)}"
            logger.error(error_msg)
            if verbose:
                print(f"❌ {error_msg}")
            return []
        
        provider_results = []
        
        for i, config in enumerate(test_configs, 1):
            if verbose:
                print(f"\n📝 Test {i}/{len(test_configs)}: {config.name}")
                print(f"   {config.description}")
            
            # Test non-streaming
            result = await self._run_single_test(
                client, provider, model, config, stream=False
            )
            provider_results.append(result)
            
            if verbose:
                print(f"   Non-streaming: {result.summary}")
            
            # Test streaming if supported and requested
            if test_streaming and "streaming" not in (config.expected_features or []):
                # Add small delay between tests
                await asyncio.sleep(0.5)
                
                streaming_result = await self._run_single_test(
                    client, provider, model, config, stream=True
                )
                # Mark as streaming test
                streaming_result.test_name = f"{config.name}_streaming"
                provider_results.append(streaming_result)
                
                if verbose:
                    print(f"   Streaming:     {streaming_result.summary}")
        
        self.results.extend(provider_results)
        return provider_results
    
    async def _run_single_test(
        self,
        client: BaseLLMClient,
        provider: str,
        model: str,
        config: BenchmarkConfig,
        stream: bool = False
    ) -> PerformanceMetrics:
        """Run a single benchmark test"""
        
        metrics = PerformanceMetrics(
            provider=provider,
            model=model,
            test_name=config.name + ("_streaming" if stream else "")
        )
        
        try:
            start_time = time.time()
            first_token_time = None
            chunks = []
            full_response = ""
            tool_calls = []
            
            # Prepare request parameters
            kwargs = {}
            if config.max_tokens:
                kwargs["max_tokens"] = config.max_tokens
            if config.temperature is not None:
                kwargs["temperature"] = config.temperature
            
            if stream:
                # Streaming test
                chunk_times = []
                
                async for chunk in client.create_completion(
                    config.messages,
                    tools=config.tools,
                    stream=True,
                    **kwargs
                ):
                    current_time = time.time()
                    chunk_times.append(current_time)
                    
                    # Mark first meaningful content
                    if first_token_time is None and chunk.get("response"):
                        first_token_time = current_time
                    
                    # Collect response data
                    if chunk.get("response"):
                        full_response += chunk["response"]
                        chunks.append(chunk)
                    
                    if chunk.get("tool_calls"):
                        tool_calls.extend(chunk["tool_calls"])
                
                end_time = time.time()
                
                # Calculate streaming metrics
                metrics.chunk_count = len(chunks)
                if first_token_time:
                    metrics.first_token_latency = first_token_time - start_time
                    metrics.streaming_duration = end_time - first_token_time
                
                if len(chunk_times) > 1:
                    intervals = [chunk_times[i] - chunk_times[i-1] 
                               for i in range(1, len(chunk_times))]
                    metrics.avg_chunk_interval = statistics.mean(intervals)
            
            else:
                # Non-streaming test
                result = await asyncio.wait_for(
                    client.create_completion(
                        config.messages,
                        tools=config.tools,
                        stream=False,
                        **kwargs
                    ),
                    timeout=config.timeout
                )
                
                end_time = time.time()
                
                full_response = result.get("response", "")
                tool_calls = result.get("tool_calls", [])
            
            # Calculate final metrics
            metrics.total_response_time = end_time - start_time
            metrics.response_length = len(full_response) if full_response else 0
            metrics.tool_calls_count = len(tool_calls) if tool_calls else 0
            metrics.full_response = full_response
            metrics.tool_calls = tool_calls
            metrics.success = True
            
            # Calculate tokens per second (rough estimate)
            if metrics.response_length > 0 and metrics.total_response_time > 0:
                # Rough estimate: 4 characters per token
                estimated_tokens = metrics.response_length / 4
                metrics.tokens_per_second = estimated_tokens / metrics.total_response_time
        
        except asyncio.TimeoutError:
            metrics.error_message = f"Timeout after {config.timeout}s"
            metrics.total_response_time = config.timeout
        except Exception as e:
            metrics.error_message = str(e)
            metrics.total_response_time = time.time() - start_time
            logger.error(f"Test failed for {provider}-{model}: {e}")
        
        return metrics
    
    async def benchmark_multiple(
        self,
        provider_models: List[Tuple[str, str]],
        test_configs: Optional[List[BenchmarkConfig]] = None,
        test_streaming: bool = True,
        verbose: bool = True
    ) -> Dict[str, List[PerformanceMetrics]]:
        """Benchmark multiple provider-model combinations"""
        
        results = {}
        
        for provider, model in provider_models:
            key = f"{provider}-{model}"
            try:
                results[key] = await self.benchmark_provider_model(
                    provider, model, test_configs, test_streaming, verbose
                )
                
                # Small delay between providers to be nice to APIs
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Failed to benchmark {key}: {e}")
                results[key] = []
        
        return results
    
    def _serialize_metrics(self, metrics: List[PerformanceMetrics]) -> List[Dict]:
        """Convert metrics to JSON-serializable format"""
        import json
        
        def make_serializable(obj):
            """Recursively convert objects to JSON-serializable format"""
            if obj is None:
                return None
            elif isinstance(obj, (str, int, float, bool)):
                return obj
            elif isinstance(obj, list):
                return [make_serializable(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: make_serializable(value) for key, value in obj.items()}
            elif hasattr(obj, '__dict__'):
                # Convert object to dict recursively
                return make_serializable(obj.__dict__)
            else:
                # Fallback to string representation
                return str(obj)
        
        serializable_metrics = []
        
        for metric in metrics:
            # Convert dataclass to dict
            metric_dict = asdict(metric)
            
            # Make everything serializable
            safe_metric = make_serializable(metric_dict)
            
            serializable_metrics.append(safe_metric)
        
        return serializable_metrics

    def generate_report(
        self,
        results: Optional[Dict[str, List[PerformanceMetrics]]] = None,
        save_to_file: bool = True
    ) -> str:
        """Generate a comprehensive benchmark report"""
        
        if results is None:
            # Group existing results by provider-model
            results = {}
            for metric in self.results:
                key = f"{metric.provider}-{metric.model}"
                if key not in results:
                    results[key] = []
                results[key].append(metric)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate report
        report_lines = [
            "# LLM Provider & Model Benchmark Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Executive Summary",
        ]
        
        # Calculate summary statistics
        all_successful = [m for metrics in results.values() for m in metrics if m.success]
        
        if all_successful:
            avg_response_time = statistics.mean([m.total_response_time for m in all_successful])
            avg_first_token = statistics.mean([m.first_token_latency for m in all_successful if m.first_token_latency])
            
            report_lines.extend([
                f"- **Total Tests**: {sum(len(metrics) for metrics in results.values())}",
                f"- **Successful Tests**: {len(all_successful)}",
                f"- **Average Response Time**: {avg_response_time:.2f}s",
                f"- **Average First Token Latency**: {avg_first_token:.2f}s",
                ""
            ])
        
        # Provider comparison table
        report_lines.extend([
            "## Provider Comparison",
            "",
            "| Provider-Model | Success Rate | Avg Response Time | Avg First Token | Avg Tokens/sec |",
            "|----------------|--------------|-------------------|-----------------|----------------|"
        ])
        
        for key, metrics in results.items():
            successful = [m for m in metrics if m.success]
            total = len(metrics)
            success_rate = len(successful) / total if total > 0 else 0
            
            if successful:
                avg_time = statistics.mean([m.total_response_time for m in successful])
                avg_first_token = statistics.mean([m.first_token_latency for m in successful if m.first_token_latency]) or 0
                avg_tokens_sec = statistics.mean([m.tokens_per_second for m in successful if m.tokens_per_second]) or 0
                
                report_lines.append(
                    f"| {key} | {success_rate:.1%} | {avg_time:.2f}s | {avg_first_token:.2f}s | {avg_tokens_sec:.1f} |"
                )
            else:
                report_lines.append(f"| {key} | 0.0% | - | - | - |")
        
        report_lines.append("")
        
        # Detailed results by test
        report_lines.extend([
            "## Detailed Results by Test",
            ""
        ])
        
        # Group by test name
        by_test = {}
        for metrics in results.values():
            for metric in metrics:
                test_name = metric.test_name
                if test_name not in by_test:
                    by_test[test_name] = []
                by_test[test_name].append(metric)
        
        for test_name, test_metrics in by_test.items():
            report_lines.extend([
                f"### {test_name}",
                "",
                "| Provider-Model | Status | Response Time | First Token | Response Length |",
                "|----------------|--------|---------------|-------------|-----------------|"
            ])
            
            for metric in test_metrics:
                status = "✅" if metric.success else "❌"
                first_token = f"{metric.first_token_latency:.2f}s" if metric.first_token_latency else "-"
                
                report_lines.append(
                    f"| {metric.provider}-{metric.model} | {status} | "
                    f"{metric.total_response_time:.2f}s | {first_token} | {metric.response_length} chars |"
                )
            
            report_lines.append("")
        
        # Performance insights
        report_lines.extend([
            "## Performance Insights",
            ""
        ])
        
        if all_successful:
            # Find fastest provider
            fastest = min(all_successful, key=lambda m: m.total_response_time)
            report_lines.append(f"- **Fastest Overall**: {fastest.provider}-{fastest.model} ({fastest.total_response_time:.2f}s)")
            
            # Find best first token latency
            with_first_token = [m for m in all_successful if m.first_token_latency]
            if with_first_token:
                fastest_first_token = min(with_first_token, key=lambda m: m.first_token_latency)
                report_lines.append(f"- **Fastest First Token**: {fastest_first_token.provider}-{fastest_first_token.model} ({fastest_first_token.first_token_latency:.2f}s)")
            
            # Find highest throughput
            with_throughput = [m for m in all_successful if m.tokens_per_second]
            if with_throughput:
                highest_throughput = max(with_throughput, key=lambda m: m.tokens_per_second)
                report_lines.append(f"- **Highest Throughput**: {highest_throughput.provider}-{highest_throughput.model} ({highest_throughput.tokens_per_second:.1f} tokens/sec)")
        
        report_content = "\n".join(report_lines)
        
        if save_to_file:
            report_file = self.output_dir / f"benchmark_report_{timestamp}.md"
            with open(report_file, 'w') as f:
                f.write(report_content)
            
            # Also save raw data as JSON
            json_file = self.output_dir / f"benchmark_data_{timestamp}.json"
            with open(json_file, 'w') as f:
                # Use the serialization helper
                serializable_data = {
                    key: self._serialize_metrics(metrics)
                    for key, metrics in results.items()
                }
                json.dump(serializable_data, f, indent=2)
            
            print(f"📊 Report saved to: {report_file}")
            print(f"📄 Raw data saved to: {json_file}")
        
        return report_content

# Convenience functions for common benchmarking scenarios

async def quick_benchmark(providers: List[str], models: Dict[str, str] = None):
    """Quick benchmark of common providers"""
    benchmark = LLMBenchmark()
    
    # Default models if not specified
    default_models = {
        "openai": "gpt-4o-mini",
        "anthropic": "claude-3-5-sonnet-20241022",
        "groq": "llama-3.3-70b-versatile",
        "ollama": "qwen3",
        "gemini": "gemini-2.0-flash"
    }
    
    if models:
        default_models.update(models)
    
    provider_models = [(p, default_models.get(p, "default")) for p in providers]
    
    results = await benchmark.benchmark_multiple(provider_models)
    report = benchmark.generate_report(results)
    
    return results, report

async def streaming_benchmark(provider_models: List[Tuple[str, str]]):
    """Benchmark focused on streaming performance"""
    benchmark = LLMBenchmark()
    
    # Streaming-focused tests
    streaming_tests = [
        BenchmarkConfig(
            name="fast_count",
            description="Quick counting task",
            messages=[{"role": "user", "content": "Count from 1 to 10"}]
        ),
        BenchmarkConfig(
            name="story_generation",
            description="Short story generation",
            messages=[{"role": "user", "content": "Write a 100-word story about a robot"}],
            max_tokens=150
        ),
        BenchmarkConfig(
            name="explanation_task",
            description="Technical explanation",
            messages=[{"role": "user", "content": "Explain how neural networks work in simple terms"}],
            max_tokens=200
        )
    ]
    
    results = await benchmark.benchmark_multiple(
        provider_models,
        test_configs=streaming_tests,
        test_streaming=True
    )
    
    return results, benchmark.generate_report(results)

if __name__ == "__main__":
    import sys
    
    async def main():
        # Example usage
        if len(sys.argv) > 1:
            # Run specific providers from command line
            providers = sys.argv[1:]
            results, report = await quick_benchmark(providers)
        else:
            # Default benchmark
            provider_models = [
                ("openai", "gpt-4o-mini"),
                ("groq", "llama-3.3-70b-versatile"),
                # Add more as available
            ]
            
            benchmark = LLMBenchmark()
            results = await benchmark.benchmark_multiple(provider_models)
            report = benchmark.generate_report(results)
        
        print("\n" + "="*60)
        print("BENCHMARK COMPLETE")
        print("="*60)
    
    asyncio.run(main()) 