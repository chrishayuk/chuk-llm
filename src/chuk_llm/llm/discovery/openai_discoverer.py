# chuk_llm/llm/discovery/openai_discoverer.py
"""
OpenAI-specific model discoverer with enhanced reasoning model support - FIXED VERSION
"""

import logging
import os
import re
from typing import Dict, List, Optional, Any
import httpx

from .base import BaseModelDiscoverer, DiscoveredModel, DiscovererFactory

log = logging.getLogger(__name__)


class OpenAIModelDiscoverer(BaseModelDiscoverer):
    """Enhanced OpenAI model discoverer with reasoning model support"""
    
    def __init__(self, provider_name: str = "openai", api_key: Optional[str] = None, 
                 api_base: str = "https://api.openai.com/v1", **config):
        super().__init__(provider_name, **config)
        # FIXED: Proper API key handling - only fall back to env if api_key is None
        if api_key is None:
            self.api_key = os.getenv("OPENAI_API_KEY")
        else:
            self.api_key = api_key
        self.api_base = api_base.rstrip('/')
        
        # Enhanced model categorization patterns
        self.reasoning_patterns = [r'o1', r'o3', r'o4', r'o5', r'o6', r'gpt-5', r'reasoning', r'think']
        self.vision_patterns = [r'gpt-4.*vision', r'gpt-4o', r'gpt-5.*vision', r'vision']
        self.code_patterns = [r'code', r'davinci-code']
        
        # Model family definitions
        self.model_families = {
            'o1': {
                'generation': 'o1',
                'reasoning_type': 'chain-of-thought',
                'supports_streaming': False,
                'supports_system_messages': False,
                'context_length': 128000,
                'max_output': 32768,
                'parameter_requirements': {
                    'use_max_completion_tokens': True,
                    'no_system_messages': True,
                    'no_streaming': True
                }
            },
            'o3': {
                'generation': 'o3',
                'reasoning_type': 'advanced-reasoning',
                'supports_streaming': True,
                'supports_system_messages': True,
                'context_length': 200000,
                'max_output': 64000,
                'parameter_requirements': {
                    'use_max_completion_tokens': True,
                    'no_system_messages': False,
                    'no_streaming': False
                }
            },
            'o4': {
                'generation': 'o4', 
                'reasoning_type': 'next-gen-reasoning',
                'supports_streaming': True,
                'supports_system_messages': True,
                'context_length': 300000,
                'max_output': 100000,
                'parameter_requirements': {
                    'use_max_completion_tokens': True,
                    'no_system_messages': False,
                    'no_streaming': False
                }
            },
            'gpt-5': {
                'generation': 'gpt5',
                'reasoning_type': 'advanced-reasoning',
                'supports_streaming': True,
                'supports_system_messages': True,
                'context_length': 200000,
                'max_output': 64000,
                'parameter_requirements': {
                    'use_max_completion_tokens': True,
                    'no_system_messages': False,
                    'no_streaming': False
                }
            },
            'gpt-4': {
                'generation': 'gpt4',
                'reasoning_type': 'standard',
                'supports_streaming': True,
                'supports_system_messages': True,
                'context_length': 128000,
                'max_output': 8192,
                'parameter_requirements': {}
            },
            'gpt-3.5': {
                'generation': 'gpt35',
                'reasoning_type': 'standard',
                'supports_streaming': True,
                'supports_system_messages': True,
                'context_length': 16384,
                'max_output': 4096,
                'parameter_requirements': {}
            }
        }
    
    async def discover_models(self) -> List[Dict[str, Any]]:
        """Discover OpenAI models via API with enhanced categorization"""
        try:
            if not self.api_key:
                log.warning("No OpenAI API key available for discovery")
                return self._get_fallback_models()
            
            headers = {"Authorization": f"Bearer {self.api_key}"}
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(f"{self.api_base}/models", headers=headers)
                response.raise_for_status()
                data = response.json()
                
                models = []
                discovered_count = 0
                
                for model_data in data.get("data", []):
                    model_id = model_data["id"]
                    
                    # Skip fine-tuned and organization-specific models for general discovery
                    if ":" in model_id or model_id.startswith("ft-"):
                        continue
                    
                    # Skip deprecated or internal models
                    if any(skip in model_id.lower() for skip in ['deprecated', 'internal', 'test']):
                        continue
                    
                    discovered_count += 1
                    
                    # Enhanced model categorization
                    model_info = self._categorize_model(model_id, model_data)
                    models.append(model_info)
                
                log.info(f"Discovered {len(models)} usable models from {discovered_count} total OpenAI API models")
                
                # Sort models by importance (reasoning models first, then by generation)
                models.sort(key=self._model_sort_key)
                return models
                
        except Exception as e:
            log.error(f"Failed to discover OpenAI models: {e}")
            return self._get_fallback_models()
    
    def _categorize_model(self, model_id: str, model_data: Dict[str, Any]) -> Dict[str, Any]:
        """Categorize a model with enhanced metadata"""
        # Determine model family and characteristics
        family_info = self._get_family_info(model_id)
        
        model_info = {
            "name": model_id,
            "created_at": model_data.get("created"),
            "owned_by": model_data.get("owned_by", "openai"),
            "object": model_data.get("object", "model"),
            "source": "openai_api",
            
            # Enhanced categorization
            "is_reasoning": self._is_reasoning_model(model_id),
            "is_vision": self._is_vision_model(model_id),
            "is_code": self._is_code_model(model_id),
            "generation": family_info.get("generation", "unknown"),
            "variant": self._extract_variant(model_id),
            "model_family": self._determine_family(model_id),
            "reasoning_type": family_info.get("reasoning_type", "standard"),
            
            # Technical capabilities from family info
            "supports_tools": True,  # Most OpenAI models support tools
            "supports_streaming": family_info.get("supports_streaming", True),
            "supports_system_messages": family_info.get("supports_system_messages", True),
            "estimated_context_length": family_info.get("context_length", 8192),
            "estimated_max_output": family_info.get("max_output", 4096),
            "parameter_requirements": family_info.get("parameter_requirements", {}),
            
            # Pricing tier (estimated)
            "pricing_tier": self._estimate_pricing_tier(model_id),
            "performance_tier": self._estimate_performance_tier(model_id)
        }
        
        return model_info
    
    def _get_family_info(self, model_id: str) -> Dict[str, Any]:
        """Get family information for a model"""
        model_lower = model_id.lower()
        
        # Check each family pattern
        for family_key, family_info in self.model_families.items():
            if family_key in model_lower or any(family_key in part for part in model_lower.split('-')):
                return family_info
        
        # Default for unknown models
        return {
            'generation': 'unknown',
            'reasoning_type': 'standard',
            'supports_streaming': True,
            'supports_system_messages': True,
            'context_length': 8192,
            'max_output': 4096,
            'parameter_requirements': {}
        }
    
    def _get_fallback_models(self) -> List[Dict[str, Any]]:
        """Fallback model list with known OpenAI models including reasoning"""
        fallback_models = [
            # O1 Reasoning series (highest priority)
            {
                "name": "o1-mini", 
                "family": "o1", 
                "is_reasoning": True,
                "supports_streaming": False,
                "supports_system_messages": False,
                "context": 128000,
                "max_output": 32768,
                "parameter_requirements": {
                    "use_max_completion_tokens": True,
                    "no_system_messages": True,
                    "no_streaming": True
                },
                "pricing_tier": "premium",
                "performance_tier": "reasoning"
            },
            {
                "name": "o1-preview", 
                "family": "o1", 
                "is_reasoning": True,
                "supports_streaming": False,
                "supports_system_messages": False,
                "context": 128000,
                "max_output": 32768,
                "parameter_requirements": {
                    "use_max_completion_tokens": True,
                    "no_system_messages": True,
                    "no_streaming": True
                },
                "pricing_tier": "premium",
                "performance_tier": "reasoning"
            },
            
            # O3 Reasoning series
            {
                "name": "o3-mini", 
                "family": "o3", 
                "is_reasoning": True,
                "supports_streaming": True,
                "supports_system_messages": True,
                "context": 200000,
                "max_output": 64000,
                "parameter_requirements": {"use_max_completion_tokens": True},
                "pricing_tier": "premium",
                "performance_tier": "advanced-reasoning"
            },
            
            # GPT-4.1 series (newest standard)
            {
                "name": "gpt-4.1", 
                "family": "gpt4", 
                "is_reasoning": False,
                "supports_streaming": True,
                "supports_system_messages": True,
                "context": 128000,
                "max_output": 8192,
                "is_vision": False,
                "pricing_tier": "standard",
                "performance_tier": "high"
            },
            {
                "name": "gpt-4.1-mini", 
                "family": "gpt4", 
                "is_reasoning": False,
                "supports_streaming": True,
                "supports_system_messages": True,
                "context": 128000,
                "max_output": 16384,
                "is_vision": False,
                "pricing_tier": "economy",
                "performance_tier": "high"
            },
            
            # GPT-4o series (multimodal)
            {
                "name": "gpt-4o", 
                "family": "gpt4", 
                "is_reasoning": False,
                "supports_streaming": True,
                "supports_system_messages": True,
                "context": 128000,
                "max_output": 8192,
                "is_vision": True,
                "pricing_tier": "standard",
                "performance_tier": "high"
            },
            {
                "name": "gpt-4o-mini", 
                "family": "gpt4", 
                "is_reasoning": False,
                "supports_streaming": True,
                "supports_system_messages": True,
                "context": 128000,
                "max_output": 16384,
                "is_vision": True,
                "pricing_tier": "economy",
                "performance_tier": "medium"
            },
            
            # GPT-4 Turbo
            {
                "name": "gpt-4-turbo", 
                "family": "gpt4", 
                "is_reasoning": False,
                "supports_streaming": True,
                "supports_system_messages": True,
                "context": 128000,
                "max_output": 8192,
                "is_vision": True,
                "pricing_tier": "standard",
                "performance_tier": "high"
            },
            
            # GPT-3.5 Turbo (legacy but stable)
            {
                "name": "gpt-3.5-turbo", 
                "family": "gpt35", 
                "is_reasoning": False,
                "supports_streaming": True,
                "supports_system_messages": True,
                "context": 16384,
                "max_output": 4096,
                "is_vision": False,
                "pricing_tier": "budget",
                "performance_tier": "medium"
            }
        ]
        
        # Convert to discovery format
        models = []
        for model_data in fallback_models:
            models.append({
                "name": model_data["name"],
                "source": "openai_fallback",
                "model_family": model_data.get("family", "unknown"),
                "is_reasoning": model_data.get("is_reasoning", False),
                "is_vision": model_data.get("is_vision", False),
                "supports_tools": True,
                "supports_streaming": model_data.get("supports_streaming", True),
                "supports_system_messages": model_data.get("supports_system_messages", True),
                "estimated_context_length": model_data.get("context", 8192),
                "estimated_max_output": model_data.get("max_output", 4096),
                "parameter_requirements": model_data.get("parameter_requirements", {}),
                "pricing_tier": model_data.get("pricing_tier", "standard"),
                "performance_tier": model_data.get("performance_tier", "medium"),
                "owned_by": "openai",
                "object": "model"
            })
        
        log.info(f"Using fallback list: {len(models)} OpenAI models")
        return models
    
    def _is_reasoning_model(self, model_name: str) -> bool:
        """Check if model is a reasoning model"""
        model_lower = model_name.lower()
        return any(re.search(pattern, model_lower) for pattern in self.reasoning_patterns)
    
    def _is_vision_model(self, model_name: str) -> bool:
        """Check if model supports vision"""
        model_lower = model_name.lower()
        return any(re.search(pattern, model_lower) for pattern in self.vision_patterns)
    
    def _is_code_model(self, model_name: str) -> bool:
        """Check if model is specialized for code"""
        model_lower = model_name.lower()
        return any(re.search(pattern, model_lower) for pattern in self.code_patterns)
    
    def _extract_variant(self, model_name: str) -> str:
        """Extract model variant (mini, turbo, preview, etc.)"""
        model_lower = model_name.lower()
        variants = ["mini", "turbo", "preview", "nano", "max", "pro"]
        
        for variant in variants:
            if variant in model_lower:
                return variant
        
        return "standard"
    
    def _determine_family(self, model_name: str) -> str:
        """Determine model family for configuration"""
        model_lower = model_name.lower()
        
        if any(pattern in model_lower for pattern in ["o1", "o3", "o4", "o5", "o6"]):
            return "reasoning"
        elif "gpt-5" in model_lower:
            return "reasoning"  # GPT-5 is a reasoning model
        elif "gpt-4" in model_lower:
            return "gpt4"
        elif "gpt-3.5" in model_lower:
            return "gpt35"
        else:
            return "unknown"
    
    def _estimate_pricing_tier(self, model_name: str) -> str:
        """Estimate pricing tier based on model name"""
        model_lower = model_name.lower()
        
        if any(pattern in model_lower for pattern in ["o1", "o3", "o4", "o5", "o6", "gpt-5"]):
            return "premium"  # Reasoning models and GPT-5 are premium
        elif "mini" in model_lower or "nano" in model_lower:
            return "economy"
        elif "gpt-3.5" in model_lower:
            return "budget"
        else:
            return "standard"
    
    def _estimate_performance_tier(self, model_name: str) -> str:
        """Estimate performance tier based on model name"""
        model_lower = model_name.lower()
        
        if "o1" in model_lower:
            return "reasoning"
        elif any(pattern in model_lower for pattern in ["o3", "o4", "o5", "o6", "gpt-5"]):
            return "advanced-reasoning"
        elif "gpt-4" in model_lower:
            return "high"
        elif "gpt-3.5" in model_lower:
            return "medium"
        else:
            return "standard"
    
    def _model_sort_key(self, model: Dict[str, Any]) -> tuple:
        """Sort key for models (reasoning first, then by generation)"""
        priority_map = {
            "reasoning": 0,
            "advanced-reasoning": 1,
            "high": 2,
            "medium": 3,
            "standard": 4
        }
        
        performance_tier = model.get("performance_tier", "standard")
        is_reasoning = model.get("is_reasoning", False)
        generation = model.get("generation", "unknown")
        
        # Sort by: reasoning models first, then performance tier, then generation
        return (
            0 if is_reasoning else 1,
            priority_map.get(performance_tier, 5),
            generation
        )
    
    def normalize_model_data(self, raw_model: Dict[str, Any]) -> DiscoveredModel:
        """Convert OpenAI model data to DiscoveredModel with enhanced metadata"""
        return DiscoveredModel(
            name=raw_model.get("name", "unknown"),
            provider=self.provider_name,
            created_at=raw_model.get("created_at"),
            family=raw_model.get("model_family", "unknown"),
            metadata={
                "owned_by": raw_model.get("owned_by"),
                "object": raw_model.get("object"),
                "source": raw_model.get("source", "openai_api"),
                
                # Enhanced capabilities metadata
                "is_reasoning": raw_model.get("is_reasoning", False),
                "is_vision": raw_model.get("is_vision", False),
                "is_code": raw_model.get("is_code", False),
                "generation": raw_model.get("generation", "unknown"),
                "variant": raw_model.get("variant", "standard"),
                "reasoning_type": raw_model.get("reasoning_type", "standard"),
                
                # Technical capabilities
                "supports_tools": raw_model.get("supports_tools", False),
                "supports_streaming": raw_model.get("supports_streaming", True),
                "supports_system_messages": raw_model.get("supports_system_messages", True),
                "estimated_context_length": raw_model.get("estimated_context_length", 8192),
                "estimated_max_output": raw_model.get("estimated_max_output", 4096),
                "parameter_requirements": raw_model.get("parameter_requirements", {}),
                
                # Business metadata
                "pricing_tier": raw_model.get("pricing_tier", "standard"),
                "performance_tier": raw_model.get("performance_tier", "medium"),
            }
        )
    
    async def test_model_availability(self, model_name: str) -> bool:
        """Test if a specific model is available - FIXED VERSION"""
        # FIXED: Comprehensive API key validation
        if not self.api_key:
            log.debug(f"API key is None - cannot test model availability: {model_name}")
            return False
            
        if isinstance(self.api_key, str) and not self.api_key.strip():
            log.debug(f"API key is empty or whitespace - cannot test model availability: {model_name}")
            return False
        
        try:
            import openai
            
            # Additional validation - ensure we can create the client
            if not self.api_key:
                return False
                
            client = openai.AsyncOpenAI(api_key=self.api_key, base_url=self.api_base)
            
            # Get family info for proper parameters
            family_info = self._get_family_info(model_name)
            
            # Test with minimal request using proper parameters
            test_params = {
                "model": model_name,
                "messages": [{"role": "user", "content": "Test"}],
            }
            
            # Use appropriate token parameter
            if family_info.get("parameter_requirements", {}).get("use_max_completion_tokens", False):
                test_params["max_completion_tokens"] = 5
            else:
                test_params["max_tokens"] = 5
            
            response = await client.chat.completions.create(**test_params)
            await client.close()
            return True
            
        except ImportError as e:
            log.debug(f"OpenAI package not available for testing model availability: {e}")
            return False
        except Exception as e:
            log.debug(f"Model {model_name} availability test failed: {e}")
            return False


# Register the discoverer
DiscovererFactory.register_discoverer("openai", OpenAIModelDiscoverer)