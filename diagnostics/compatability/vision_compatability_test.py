#!/usr/bin/env python3
"""
Vision Compatibility Test - Test vision capabilities across providers
"""
import asyncio
import base64
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Any, Optional

# Suppress warnings
os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from chuk_llm.llm.client import get_client
from chuk_llm.configuration import get_config, Feature


def create_test_image_base64(color: str = "red", size: int = 10) -> str:
    """Create a simple colored square as base64 PNG"""
    try:
        from PIL import Image
        import io
        
        # Create a colored square
        img = Image.new('RGB', (size, size), color)
        
        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return img_data
    except ImportError:
        # Fallback images if PIL not available
        fallback_images = {
            "red": "iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAYAAACNMs+9AAAAGElEQVR42mP8z4AKGKHoaIciRsYBEK7o4RAAgqwD2vD/qYAAAAAASUVORK5CYII=",
            "blue": "iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAYAAACNMs+9AAAAGElEQVR42mP8//8/AQYGBuCCo4dDBsZRAACkwgMNvOGrfgAAAABJRU5ErkJggg==",
            "green": "iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAYAAACNMs+9AAAAGElEQVR42mNgYGBgAAAABQABBLAAo4dDBgMAJsADAR8LPTIAAAAASUVORK5CYII="
        }
        return fallback_images.get(color, fallback_images["red"])


class VisionCompatibilityTester:
    """Test vision capabilities with consistent interface across providers"""
    
    def __init__(self):
        self.vision_providers = ["openai", "anthropic", "gemini", "groq", "ollama", "mistral"]
        
        # Create test images
        self.test_images = {
            "red_square": create_test_image_base64("red", 15),
            "blue_square": create_test_image_base64("blue", 15),
            "green_square": create_test_image_base64("green", 15)
        }
        
        # Provider-specific models that support vision
        self.vision_models = {
            "openai": "gpt-4o",
            "anthropic": "claude-sonnet-4-20250514", 
            "gemini": "gemini-2.0-flash",  # Updated to latest model
            "groq": "llama-3.2-11b-vision-preview",
            "ollama": "llava",
            "mistral": "mistral-medium-2505"  # Updated to tools+vision capable model
        }
    
    def get_universal_image_format(self, image_b64: str) -> Dict[str, Any]:
        """Get universal image format that should work across all providers"""
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{image_b64}"
            }
        }
    
    def get_anthropic_native_format(self, image_b64: str) -> Dict[str, Any]:
        """Get Anthropic's native format for comparison"""
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": image_b64
            }
        }
    
    async def _test_provider_vision(self, provider: str) -> Dict[str, Any]:
        """Test vision capabilities for a specific provider"""
        
        results = {
            "provider": provider,
            "supports_vision": False,
            "model_used": None,
            "single_image_test": None,
            "multiple_images_test": None,
            "format_tests": {},
            "error_details": []
        }
        
        try:
            # Check if provider supports vision in config
            config = get_config()
            
            # Get vision-capable model for this provider
            model = self.vision_models.get(provider)
            if not model:
                results["error_details"].append("No vision model configured")
                return results
            
            # Check if this model supports vision
            try:
                supports_vision = config.supports_feature(provider, Feature.VISION, model)
                results["supports_vision"] = supports_vision
                results["model_used"] = model
                
                if not supports_vision:
                    results["error_details"].append("Model doesn't support vision according to config")
                    return results
                    
            except Exception as e:
                results["error_details"].append(f"Config check failed: {e}")
                # Continue anyway for providers not in config - they might still work
                results["supports_vision"] = True
                results["model_used"] = model
                log.warning(f"Config check failed for {provider}/{model}, continuing anyway: {e}")
            
            # Test if we can get a client
            try:
                client = get_client(provider, model=model)
                log.debug(f"Successfully created {provider} client with model {model}")
            except Exception as e:
                results["error_details"].append(f"Failed to create client: {e}")
                return results
            
            # Test 1: Single image with universal format
            try:
                await self._test_single_image_universal(client, results)
            except Exception as e:
                results["single_image_test"] = f"❌ Error: {e}"
                results["error_details"].append(f"Single image test: {e}")
            
            # Test 2: Multiple images (if single image worked)
            if results["single_image_test"] and "✅" in str(results["single_image_test"]):
                try:
                    await self._test_multiple_images(client, results)
                except Exception as e:
                    results["multiple_images_test"] = f"❌ Error: {e}"
                    results["error_details"].append(f"Multiple images test: {e}")
            
            # Test 3: Format compatibility
            try:
                await self._test_image_formats(client, results)
            except Exception as e:
                results["error_details"].append(f"Format tests: {e}")
                
        except Exception as e:
            results["error_details"].append(f"Provider test failed: {e}")
        
        return results
    
    async def _test_single_image_universal(self, client, results: Dict[str, Any]):
        """Test single image with universal format"""
        
        # Use universal image_url format
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What color is this square? Answer with just the color name."
                    },
                    self.get_universal_image_format(self.test_images["red_square"])
                ]
            }
        ]
        
        try:
            response = await client.create_completion(messages, max_tokens=50)
            
            if response and response.get("response") and not response.get("error"):
                response_text = response["response"].lower()
                if "red" in response_text:
                    results["single_image_test"] = "✅ Correctly identified red square"
                elif any(color in response_text for color in ["blue", "green", "yellow", "purple"]):
                    results["single_image_test"] = f"⚠️ Identified wrong color: {response['response']}"
                else:
                    results["single_image_test"] = f"⚠️ No color identified: {response['response']}"
            else:
                results["single_image_test"] = f"❌ No valid response: {response.get('response', 'No response')}"
        except Exception as e:
            results["single_image_test"] = f"❌ Error: {str(e)}"
            results["error_details"].append(f"Single image test error: {e}")
    
    async def _test_multiple_images(self, client, results: Dict[str, Any]):
        """Test multiple images in one message"""
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "I'm showing you two squares. What colors are they? List both colors."
                    },
                    self.get_universal_image_format(self.test_images["red_square"]),
                    self.get_universal_image_format(self.test_images["blue_square"])
                ]
            }
        ]
        
        try:
            response = await client.create_completion(messages, max_tokens=100)
            
            if response and response.get("response") and not response.get("error"):
                response_text = response["response"].lower()
                has_red = "red" in response_text
                has_blue = "blue" in response_text
                
                if has_red and has_blue:
                    results["multiple_images_test"] = "✅ Identified both red and blue squares"
                elif has_red or has_blue:
                    results["multiple_images_test"] = f"⚠️ Partially correct: {response['response']}"
                else:
                    results["multiple_images_test"] = f"❌ No colors identified: {response['response']}"
            else:
                results["multiple_images_test"] = f"❌ No valid response: {response.get('response', 'No response')}"
        except Exception as e:
            results["multiple_images_test"] = f"❌ Error: {str(e)}"
            results["error_details"].append(f"Multiple images test error: {e}")
    
    async def _test_image_formats(self, client, results: Dict[str, Any]):
        """Test different image format variations"""
        
        results["format_tests"] = {}
        
        # Test 1: Universal image_url format (should work for all)
        try:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "What color is this?"},
                    self.get_universal_image_format(self.test_images["green_square"])
                ]
            }]
            
            response = await client.create_completion(messages, max_tokens=20)
            if response and response.get("response") and "green" in response["response"].lower():
                results["format_tests"]["universal_image_url"] = "✅ Works correctly"
            elif response and response.get("response"):
                results["format_tests"]["universal_image_url"] = f"⚠️ Response but wrong: {response['response']}"
            else:
                results["format_tests"]["universal_image_url"] = "❌ No response"
                
        except Exception as e:
            if "format" in str(e).lower() or "image" in str(e).lower():
                results["format_tests"]["universal_image_url"] = "❌ Format rejected"
            else:
                results["format_tests"]["universal_image_url"] = f"❌ Error: {e}"
        
        # Test 2: Anthropic native format (for comparison)
        try:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "What color is this?"},
                    self.get_anthropic_native_format(self.test_images["green_square"])
                ]
            }]
            
            response = await client.create_completion(messages, max_tokens=20)
            if response and response.get("response") and "green" in response["response"].lower():
                results["format_tests"]["anthropic_native"] = "✅ Works correctly"
            elif response and response.get("response"):
                results["format_tests"]["anthropic_native"] = f"⚠️ Response but wrong: {response['response']}"
            else:
                results["format_tests"]["anthropic_native"] = "❌ No response"
                
        except Exception as e:
            if "format" in str(e).lower() or "image" in str(e).lower():
                results["format_tests"]["anthropic_native"] = "❌ Format rejected"
            else:
                results["format_tests"]["anthropic_native"] = f"❌ Error: {e}"
    
    async def run_vision_compatibility_test(self, providers: List[str] = None) -> Dict[str, Any]:
        """Run comprehensive vision compatibility testing"""
        
        test_providers = providers or self.vision_providers
        
        print("👁️  Vision Compatibility Testing")
        print("=" * 80)
        print("Testing vision capabilities across providers with unified interface")
        print(f"Testing providers: {', '.join(test_providers)}")
        
        results = {
            "metadata": {
                "test_version": "2.0",
                "providers_tested": test_providers,
                "models_used": {p: self.vision_models.get(p, "default") for p in test_providers}
            },
            "provider_results": {},
            "summary": {}
        }
        
        # Test each provider
        for provider in test_providers:
            print(f"\n🔍 Testing {provider} with model {self.vision_models.get(provider, 'default')}...")
            
            # Check if we have required env vars
            required_env_vars = {
                "openai": "OPENAI_API_KEY",
                "anthropic": "ANTHROPIC_API_KEY", 
                "gemini": "GEMINI_API_KEY",
                "groq": "GROQ_API_KEY",
                "ollama": None,  # Usually local
                "mistral": "MISTRAL_API_KEY"
            }
            
            env_var = required_env_vars.get(provider)
            if env_var and not os.getenv(env_var):
                print(f"   ⚠️ Skipping {provider} - {env_var} not set")
                results["provider_results"][provider] = {
                    "provider": provider,
                    "supports_vision": False,
                    "error_details": [f"Missing {env_var}"],
                    "skipped": True
                }
                continue
            
            try:
                results["provider_results"][provider] = await self._test_provider_vision(provider)
                
                # Print immediate results with more detail
                result = results["provider_results"][provider]
                if result.get("skipped"):
                    print(f"   ⚠️ Skipped")
                elif not result["supports_vision"]:
                    error_summary = ", ".join(result.get("error_details", ["Unknown"])[:2])
                    print(f"   ❌ No vision support: {error_summary}")
                elif result["single_image_test"] and "✅" in str(result["single_image_test"]):
                    print(f"   ✅ Vision working")
                else:
                    error_summary = result.get("single_image_test", "Unknown issue")
                    print(f"   ⚠️ Vision issues: {error_summary}")
                    
            except Exception as e:
                print(f"   ❌ Test failed: {e}")
                results["provider_results"][provider] = {
                    "provider": provider,
                    "supports_vision": False,
                    "error_details": [f"Test failed: {e}"],
                    "failed": True
                }
        
        # Generate summary
        results["summary"] = self._generate_summary(results["provider_results"])
        
        return results
    
    def _generate_summary(self, provider_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of vision compatibility"""
        
        summary = {
            "providers_with_working_vision": [],
            "providers_without_vision": [],
            "providers_skipped": [],
            "universal_format_compatibility": {},
            "recommendations": []
        }
        
        # Categorize providers
        for provider, results in provider_results.items():
            if results.get("skipped"):
                summary["providers_skipped"].append(f"{provider} (missing API key)")
            elif results.get("failed"):
                summary["providers_without_vision"].append(f"{provider} (test failed)")
            elif results["supports_vision"] and results.get("single_image_test") and "✅" in str(results["single_image_test"]):
                summary["providers_with_working_vision"].append(provider)
            else:
                summary["providers_without_vision"].append(provider)
        
        # Check universal format compatibility
        universal_format_works = 0
        total_tested = 0
        
        for provider, results in provider_results.items():
            if results.get("skipped") or results.get("failed"):
                continue
            total_tested += 1
            
            if results.get("format_tests", {}).get("universal_image_url") == "✅ Works correctly":
                universal_format_works += 1
        
        if total_tested > 0:
            summary["universal_format_compatibility"] = f"{universal_format_works}/{total_tested} providers"
        else:
            summary["universal_format_compatibility"] = "0/0 providers (none tested)"
        
        # Generate recommendations
        working_count = len(summary["providers_with_working_vision"])
        total_count = len([p for p in provider_results.keys() if not provider_results[p].get("skipped")])
        
        if working_count == 0:
            summary["recommendations"].append("❌ No providers successfully handled vision inputs")
        elif working_count == total_count:
            summary["recommendations"].append("✅ All tested providers support vision with universal format")
        else:
            summary["recommendations"].append(
                f"⚠️ Vision support varies: {working_count}/{total_count} providers work"
            )
        
        if universal_format_works == total_tested and total_tested > 0:
            summary["recommendations"].append("✅ Universal image_url format works across all vision providers")
        else:
            summary["recommendations"].append("🔧 Some providers may need format adjustments")
        
        return summary
    
    def print_results(self, results: Dict[str, Any]):
        """Print vision compatibility results"""
        
        print("\n" + "="*80)
        print("👁️  VISION COMPATIBILITY REPORT")
        print("="*80)
        
        summary = results["summary"]
        
        # Working providers
        if summary["providers_with_working_vision"]:
            print("\n✅ Providers with working vision support:")
            for provider in summary["providers_with_working_vision"]:
                provider_results = results["provider_results"][provider]
                model = provider_results.get("model_used", "unknown")
                print(f"   • {provider} ({model}):")
                print(f"     - Single image: {provider_results.get('single_image_test', 'Not tested')}")
                if provider_results.get('multiple_images_test'):
                    print(f"     - Multiple images: {provider_results['multiple_images_test']}")
                
                # Format tests
                if provider_results.get("format_tests"):
                    print("     - Format compatibility:")
                    for fmt, result in provider_results["format_tests"].items():
                        print(f"       • {fmt}: {result}")
        
        # Non-working providers
        if summary["providers_without_vision"]:
            print("\n❌ Providers without working vision:")
            for provider in summary["providers_without_vision"]:
                provider_results = results["provider_results"][provider]
                errors = provider_results.get("error_details", [])
                print(f"   • {provider}: {', '.join(errors[:2])}")  # Show first 2 errors
        
        # Skipped providers
        if summary["providers_skipped"]:
            print("\n⚠️ Providers skipped:")
            for provider in summary["providers_skipped"]:
                print(f"   • {provider}")
        
        # Format compatibility summary
        print(f"\n📋 Universal Format Compatibility: {summary['universal_format_compatibility']}")
        
        # Recommendations
        print("\n🎯 Recommendations:")
        for rec in summary["recommendations"]:
            print(f"   {rec}")
        
        # Best practices
        print("\n💡 Best Practice for Universal Vision:")
        print("   Use the image_url format that works across all providers:")
        print("   {")
        print('     "type": "image_url",')
        print('     "image_url": {')
        print('       "url": "data:image/png;base64,<base64_string>"')
        print("     }")
        print("   }")
        print("   Each provider should handle conversion to their native format internally.")
        
        print("="*80)


async def main():
    """Run vision compatibility test"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test vision compatibility across providers")
    parser.add_argument("--providers", nargs="*", 
                       help="Providers to test (default: all available)", 
                       choices=["openai", "anthropic", "gemini", "groq", "ollama", "mistral"])
    parser.add_argument("--quick", action="store_true", help="Test only providers with API keys set")
    
    args = parser.parse_args()
    
    # Default providers to test
    all_providers = ["openai", "anthropic", "gemini", "groq", "ollama", "mistral"]
    
    if args.quick:
        # Only test providers with API keys
        env_vars = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY", 
            "gemini": "GEMINI_API_KEY",
            "groq": "GROQ_API_KEY",
            "ollama": None,
            "mistral": "MISTRAL_API_KEY"
        }
        providers = [p for p in all_providers if env_vars[p] is None or os.getenv(env_vars[p])]
        print(f"Quick mode: Testing providers with API keys: {providers}")
    else:
        providers = args.providers or all_providers
    
    tester = VisionCompatibilityTester()
    results = await tester.run_vision_compatibility_test(providers)
    
    # Print results
    tester.print_results(results)
    
    # Return summary for potential use
    return results


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Vision compatibility test cancelled")
    except Exception as e:
        print(f"\n💥 Test failed: {e}")
        import traceback
        traceback.print_exc()