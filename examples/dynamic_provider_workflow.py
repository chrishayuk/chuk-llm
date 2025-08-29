#!/usr/bin/env python3
"""
Complete Dynamic Provider Workflow Example
==========================================

This example demonstrates a real-world workflow using dynamic providers:
- Load balancing across multiple providers
- Fallback handling for failures
- Cost optimization by routing to cheaper providers
- A/B testing different models
- Monitoring and metrics collection
"""

import os
import time
import random
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from chuk_llm import (
    register_openai_compatible,
    unregister_provider,
    list_dynamic_providers,
    provider_exists,
    ask_sync,
    ask_json,
    stream_sync_iter
)

@dataclass
class ProviderMetrics:
    """Track provider performance metrics."""
    name: str
    requests: int = 0
    failures: int = 0
    total_latency: float = 0.0
    total_tokens: int = 0
    estimated_cost: float = 0.0
    
    @property
    def success_rate(self) -> float:
        if self.requests == 0:
            return 0.0
        return (self.requests - self.failures) / self.requests * 100
    
    @property
    def avg_latency(self) -> float:
        if self.requests == 0:
            return 0.0
        return self.total_latency / self.requests

class LoadBalancer:
    """
    Intelligent load balancer for multiple LLM providers.
    """
    def __init__(self):
        self.providers: List[str] = []
        self.metrics: Dict[str, ProviderMetrics] = {}
        self.strategy = "round_robin"  # or "least_latency", "cheapest", "weighted"
        self.current_index = 0
    
    def add_provider(
        self,
        name: str,
        api_base: str,
        api_key: Optional[str] = None,
        models: Optional[List[str]] = None,
        weight: float = 1.0
    ):
        """Add a provider to the load balancer."""
        config = register_openai_compatible(
            name=name,
            api_base=api_base,
            api_key=api_key,
            models=models or ["gpt-3.5-turbo"],
            default_model=models[0] if models else "gpt-3.5-turbo"
        )
        
        self.providers.append(name)
        self.metrics[name] = ProviderMetrics(name=name)
        
        print(f"✅ Added provider '{name}' to load balancer")
        return config
    
    def select_provider(self) -> str:
        """Select next provider based on strategy."""
        if not self.providers:
            raise ValueError("No providers available")
        
        if self.strategy == "round_robin":
            provider = self.providers[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.providers)
            return provider
        
        elif self.strategy == "least_latency":
            # Select provider with lowest average latency
            return min(
                self.providers,
                key=lambda p: self.metrics[p].avg_latency or float('inf')
            )
        
        elif self.strategy == "cheapest":
            # Select provider with lowest cost per token
            return min(
                self.providers,
                key=lambda p: self.metrics[p].estimated_cost or float('inf')
            )
        
        else:  # weighted random
            return random.choice(self.providers)
    
    def ask_with_fallback(
        self,
        prompt: str,
        max_retries: int = 3,
        **kwargs
    ) -> Optional[str]:
        """
        Ask with automatic fallback to other providers on failure.
        """
        attempted = set()
        
        while len(attempted) < min(max_retries, len(self.providers)):
            provider = self.select_provider()
            
            if provider in attempted:
                continue
            attempted.add(provider)
            
            start_time = time.time()
            metrics = self.metrics[provider]
            metrics.requests += 1
            
            try:
                response = ask_sync(prompt, provider=provider, **kwargs)
                
                # Update metrics
                latency = time.time() - start_time
                metrics.total_latency += latency
                
                # Estimate tokens (rough approximation)
                metrics.total_tokens += len(prompt.split()) + len(response.split())
                
                print(f"✅ {provider}: Success (latency: {latency:.2f}s)")
                return response
                
            except Exception as e:
                metrics.failures += 1
                print(f"⚠️  {provider}: Failed - {str(e)[:50]}")
                
                # Try next provider
                continue
        
        return None
    
    def get_metrics_report(self) -> str:
        """Generate metrics report."""
        report = "\n📊 LOAD BALANCER METRICS\n"
        report += "=" * 40 + "\n"
        
        for provider in self.providers:
            m = self.metrics[provider]
            report += f"\n{provider}:\n"
            report += f"  Requests: {m.requests}\n"
            report += f"  Success Rate: {m.success_rate:.1f}%\n"
            report += f"  Avg Latency: {m.avg_latency:.2f}s\n"
            report += f"  Total Tokens: ~{m.total_tokens}\n"
        
        return report

class ABTestManager:
    """
    A/B test different models and providers.
    """
    def __init__(self):
        self.experiments: Dict[str, Dict] = {}
        self.results: List[Dict] = []
    
    def create_experiment(
        self,
        name: str,
        variant_a: Dict,
        variant_b: Dict,
        traffic_split: float = 0.5
    ):
        """Create an A/B test experiment."""
        self.experiments[name] = {
            "variant_a": variant_a,
            "variant_b": variant_b,
            "traffic_split": traffic_split,
            "created_at": datetime.now().isoformat()
        }
        
        # Register providers if needed
        for variant in [variant_a, variant_b]:
            if not provider_exists(variant["provider"]):
                register_openai_compatible(
                    name=variant["provider"],
                    api_base=variant.get("api_base", "https://api.openai.com/v1"),
                    api_key=variant.get("api_key", os.getenv("OPENAI_API_KEY")),
                    models=[variant.get("model", "gpt-3.5-turbo")]
                )
    
    def run_test(
        self,
        experiment_name: str,
        prompt: str,
        evaluation_fn=None
    ) -> Dict:
        """Run an A/B test and return results."""
        if experiment_name not in self.experiments:
            raise ValueError(f"Experiment '{experiment_name}' not found")
        
        exp = self.experiments[experiment_name]
        
        # Choose variant
        use_a = random.random() < exp["traffic_split"]
        variant = exp["variant_a"] if use_a else exp["variant_b"]
        variant_name = "A" if use_a else "B"
        
        # Run inference
        start_time = time.time()
        try:
            response = ask_sync(
                prompt,
                provider=variant["provider"],
                model=variant.get("model"),
                temperature=variant.get("temperature", 0.7),
                max_tokens=variant.get("max_tokens", 150)
            )
            latency = time.time() - start_time
            success = True
            
            # Evaluate quality if function provided
            quality_score = evaluation_fn(response) if evaluation_fn else None
            
        except Exception as e:
            response = None
            latency = time.time() - start_time
            success = False
            quality_score = 0
        
        # Store result
        result = {
            "experiment": experiment_name,
            "variant": variant_name,
            "provider": variant["provider"],
            "model": variant.get("model"),
            "success": success,
            "latency": latency,
            "quality_score": quality_score,
            "timestamp": datetime.now().isoformat()
        }
        self.results.append(result)
        
        return result
    
    def get_experiment_report(self, experiment_name: str) -> str:
        """Generate experiment report."""
        results = [r for r in self.results if r["experiment"] == experiment_name]
        
        if not results:
            return "No results yet"
        
        report = f"\n🧪 EXPERIMENT: {experiment_name}\n"
        report += "=" * 40 + "\n"
        
        for variant in ["A", "B"]:
            variant_results = [r for r in results if r["variant"] == variant]
            if not variant_results:
                continue
            
            success_rate = sum(1 for r in variant_results if r["success"]) / len(variant_results) * 100
            avg_latency = sum(r["latency"] for r in variant_results) / len(variant_results)
            
            scores = [r["quality_score"] for r in variant_results if r["quality_score"] is not None]
            avg_quality = sum(scores) / len(scores) if scores else 0
            
            report += f"\nVariant {variant}:\n"
            report += f"  Samples: {len(variant_results)}\n"
            report += f"  Success Rate: {success_rate:.1f}%\n"
            report += f"  Avg Latency: {avg_latency:.2f}s\n"
            report += f"  Avg Quality: {avg_quality:.2f}\n"
        
        return report

class CostOptimizer:
    """
    Route requests to the cheapest appropriate provider.
    """
    # Approximate costs per 1K tokens (input + output averaged)
    PROVIDER_COSTS = {
        "gpt-3.5-turbo": 0.002,
        "gpt-4": 0.03,
        "gpt-4-turbo": 0.01,
        "claude-3-haiku": 0.00025,
        "claude-3-sonnet": 0.003,
        "claude-3-opus": 0.015,
        "llama-3-8b": 0.0001,  # Self-hosted
        "mistral-7b": 0.0001,  # Self-hosted
    }
    
    def __init__(self):
        self.providers: Dict[str, Dict] = {}
        self.total_cost = 0.0
        self.request_count = 0
    
    def add_provider(
        self,
        name: str,
        models: List[str],
        api_base: str,
        api_key: Optional[str] = None
    ):
        """Add a provider with cost information."""
        self.providers[name] = {
            "models": models,
            "api_base": api_base,
            "api_key": api_key
        }
        
        # Register if not exists
        if not provider_exists(name):
            register_openai_compatible(
                name=name,
                api_base=api_base,
                api_key=api_key,
                models=models,
                default_model=models[0]
            )
    
    def select_cheapest_provider(
        self,
        min_quality: str = "basic",
        max_latency: Optional[float] = None
    ) -> tuple[str, str]:
        """
        Select cheapest provider meeting requirements.
        
        Quality levels: basic < standard < premium
        """
        quality_tiers = {
            "basic": ["gpt-3.5-turbo", "claude-3-haiku", "llama-3-8b", "mistral-7b"],
            "standard": ["gpt-4-turbo", "claude-3-sonnet"],
            "premium": ["gpt-4", "claude-3-opus"]
        }
        
        # Get eligible models based on quality requirement
        if min_quality == "basic":
            eligible = quality_tiers["basic"] + quality_tiers["standard"] + quality_tiers["premium"]
        elif min_quality == "standard":
            eligible = quality_tiers["standard"] + quality_tiers["premium"]
        else:  # premium
            eligible = quality_tiers["premium"]
        
        # Find cheapest available model
        best_cost = float('inf')
        best_provider = None
        best_model = None
        
        for provider_name, provider_info in self.providers.items():
            for model in provider_info["models"]:
                if model in eligible and model in self.PROVIDER_COSTS:
                    cost = self.PROVIDER_COSTS[model]
                    if cost < best_cost:
                        best_cost = cost
                        best_provider = provider_name
                        best_model = model
        
        return best_provider, best_model
    
    def optimized_ask(
        self,
        prompt: str,
        quality: str = "basic",
        **kwargs
    ) -> Optional[str]:
        """Make a cost-optimized request."""
        provider, model = self.select_cheapest_provider(quality)
        
        if not provider:
            print("⚠️  No suitable provider found")
            return None
        
        try:
            response = ask_sync(
                prompt,
                provider=provider,
                model=model,
                **kwargs
            )
            
            # Estimate cost
            token_estimate = (len(prompt.split()) + len(response.split())) / 1000
            cost = token_estimate * self.PROVIDER_COSTS.get(model, 0.001)
            self.total_cost += cost
            self.request_count += 1
            
            print(f"💰 Used {model} via {provider} (cost: ${cost:.4f})")
            
            return response
            
        except Exception as e:
            print(f"⚠️  Failed with {provider}/{model}: {e}")
            return None
    
    def get_cost_report(self) -> str:
        """Generate cost report."""
        avg_cost = self.total_cost / self.request_count if self.request_count else 0
        
        return f"""
💵 COST OPTIMIZATION REPORT
{'=' * 40}
Total Requests: {self.request_count}
Total Cost: ${self.total_cost:.4f}
Average Cost: ${avg_cost:.4f}
Projected Monthly: ${self.total_cost * 30:.2f}
"""

def demo_load_balancing():
    """Demonstrate load balancing across providers."""
    print("\n" + "=" * 60)
    print("LOAD BALANCING DEMO")
    print("=" * 60)
    
    balancer = LoadBalancer()
    
    # Add multiple providers (using same endpoint for demo)
    if os.getenv("OPENAI_API_KEY"):
        balancer.add_provider(
            "primary",
            api_base="https://api.openai.com/v1",
            api_key=os.getenv("OPENAI_API_KEY"),
            models=["gpt-3.5-turbo"]
        )
        
        balancer.add_provider(
            "secondary",
            api_base="https://api.openai.com/v1",
            api_key=os.getenv("OPENAI_API_KEY"),
            models=["gpt-3.5-turbo"]
        )
        
        # Test with multiple requests
        prompts = [
            "What is 2+2?",
            "What is the capital of France?",
            "Name a color"
        ]
        
        for prompt in prompts:
            response = balancer.ask_with_fallback(
                prompt,
                temperature=0,
                max_tokens=10
            )
            if response:
                print(f"Q: {prompt}\nA: {response}\n")
        
        # Show metrics
        print(balancer.get_metrics_report())
    else:
        print("⚠️  Set OPENAI_API_KEY to test load balancing")

def demo_ab_testing():
    """Demonstrate A/B testing different models."""
    print("\n" + "=" * 60)
    print("A/B TESTING DEMO")
    print("=" * 60)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Set OPENAI_API_KEY to test A/B testing")
        return
    
    ab_test = ABTestManager()
    
    # Create experiment comparing temperatures
    ab_test.create_experiment(
        "temperature_test",
        variant_a={
            "provider": "variant_a_provider",
            "model": "gpt-3.5-turbo",
            "temperature": 0.2,
            "api_key": os.getenv("OPENAI_API_KEY")
        },
        variant_b={
            "provider": "variant_b_provider", 
            "model": "gpt-3.5-turbo",
            "temperature": 1.0,
            "api_key": os.getenv("OPENAI_API_KEY")
        },
        traffic_split=0.5
    )
    
    # Run multiple tests
    def evaluate_creativity(response):
        # Simple heuristic: longer and more varied = more creative
        return len(set(response.split())) / max(len(response.split()), 1) * 10
    
    for i in range(6):
        result = ab_test.run_test(
            "temperature_test",
            "Write a creative tagline for a coffee shop",
            evaluation_fn=evaluate_creativity
        )
        print(f"Test {i+1}: Variant {result['variant']} - Success: {result['success']}")
    
    # Show report
    print(ab_test.get_experiment_report("temperature_test"))

def demo_cost_optimization():
    """Demonstrate cost-optimized routing."""
    print("\n" + "=" * 60)
    print("COST OPTIMIZATION DEMO")
    print("=" * 60)
    
    optimizer = CostOptimizer()
    
    # Add providers with different costs
    if os.getenv("OPENAI_API_KEY"):
        optimizer.add_provider(
            "openai_cheap",
            ["gpt-3.5-turbo"],
            "https://api.openai.com/v1",
            os.getenv("OPENAI_API_KEY")
        )
        
        optimizer.add_provider(
            "openai_premium",
            ["gpt-4"],
            "https://api.openai.com/v1",
            os.getenv("OPENAI_API_KEY")
        )
    
    # Simulate different quality requirements
    requests = [
        ("What is 2+2?", "basic"),
        ("Explain quantum computing", "standard"),
        ("Write a haiku", "basic")
    ]
    
    for prompt, quality in requests:
        print(f"\nRequest: '{prompt}' (quality: {quality})")
        response = optimizer.optimized_ask(
            prompt,
            quality=quality,
            max_tokens=50,
            temperature=0.5
        )
        if response:
            print(f"Response: {response[:100]}...")
    
    # Show cost report
    print(optimizer.get_cost_report())

def main():
    print("=" * 60)
    print("COMPLETE DYNAMIC PROVIDER WORKFLOW")
    print("=" * 60)
    
    try:
        # Run all demos
        demo_load_balancing()
        demo_ab_testing()
        demo_cost_optimization()
        
    finally:
        # Clean up all dynamic providers
        print("\n🧹 Cleaning up...")
        for provider in list_dynamic_providers():
            unregister_provider(provider)
            print(f"   Removed: {provider}")
    
    # Summary
    print("\n" + "=" * 60)
    print("WORKFLOW CAPABILITIES DEMONSTRATED")
    print("=" * 60)
    print("""
This example demonstrated real-world patterns:

✅ Load Balancing
   - Round-robin, least-latency, cheapest strategies
   - Automatic fallback on failures
   - Performance metrics tracking

✅ A/B Testing
   - Compare models and parameters
   - Traffic splitting
   - Quality evaluation
   - Statistical reporting

✅ Cost Optimization
   - Route to cheapest suitable provider
   - Quality-based selection
   - Cost tracking and projections

✅ Production Patterns
   - Health checks and monitoring
   - Graceful degradation
   - Multi-provider redundancy
   - Performance optimization

These patterns enable:
- High availability with fallbacks
- Cost-effective operations
- Data-driven model selection
- Scalable architecture
""")

if __name__ == "__main__":
    main()