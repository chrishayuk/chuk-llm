# chuk_llm/llm/discovery/ollama_discoverer.py
"""
Ollama-specific model discoverer with enhanced local model support
"""

import logging
import re
from typing import Dict, List, Optional, Any
import httpx
import asyncio

from .base import BaseModelDiscoverer, DiscoveredModel, DiscovererFactory

log = logging.getLogger(__name__)


class OllamaModelDiscoverer(BaseModelDiscoverer):
    """Ollama-specific model discoverer with enhanced capabilities"""
    
    def __init__(self, provider_name: str = "ollama", api_base: str = "http://localhost:11434", **config):
        super().__init__(provider_name, **config)
        self.api_base = api_base.rstrip('/')
        self.timeout = config.get('timeout', 10.0)
        
        # Model family patterns for Ollama
        self.family_patterns = {
            'llama': {
                'patterns': [r'llama', r'yi-coder'],
                'base_context': 8192,
                'context_rules': {
                    r'llama-?2': 4096,
                    r'llama-?3\.1': 128000,
                    r'llama-?3\.[23]': 128000,
                },
                'capabilities': ['text', 'streaming', 'tools', 'system_messages'],
                'reasoning_capable': True
            },
            'qwen': {
                'patterns': [r'qwen', r'codeqwen'],
                'base_context': 32768,
                'context_rules': {
                    r'qwen2': 32768,
                    r'qwen2\.5': 32768,
                    r'qwen3': 32768,
                },
                'capabilities': ['text', 'streaming', 'tools', 'reasoning', 'system_messages'],
                'reasoning_capable': True
            },
            'granite': {
                'patterns': [r'granite'],
                'base_context': 8192,
                'capabilities': ['text', 'streaming', 'tools', 'reasoning', 'system_messages'],
                'reasoning_capable': True
            },
            'mistral': {
                'patterns': [r'mistral', r'mixtral'],
                'base_context': 32768,
                'capabilities': ['text', 'streaming', 'tools', 'system_messages'],
                'reasoning_capable': False
            },
            'gemma': {
                'patterns': [r'gemma', r'codegemma'],
                'base_context': 8192,
                'capabilities': ['text', 'streaming', 'tools', 'system_messages'],
                'reasoning_capable': False
            },
            'phi': {
                'patterns': [r'phi'],
                'base_context': 4096,
                'context_rules': {
                    r'phi-?3': 128000,
                    r'phi4': 128000,
                },
                'capabilities': ['text', 'streaming', 'system_messages', 'reasoning'],
                'reasoning_capable': True
            },
            'code': {
                'patterns': [r'codellama', r'starcoder', r'deepseek-coder', r'codegemma', r'devstral'],
                'base_context': 16384,
                'capabilities': ['text', 'streaming', 'tools', 'system_messages'],
                'specialization': 'code'
            },
            'vision': {
                'patterns': [r'.*vision.*', r'.*llava.*', r'.*moondream.*'],
                'base_context': 8192,
                'capabilities': ['text', 'streaming', 'vision', 'multimodal', 'system_messages'],
                'specialization': 'vision'
            },
            'reasoning': {
                'patterns': [r'.*reasoning.*', r'.*phi4.*', r'.*qwq.*', r'.*marco-o1.*'],
                'base_context': 32768,
                'capabilities': ['text', 'streaming', 'reasoning', 'system_messages'],
                'specialization': 'reasoning'
            }
        }
        
        # Size-based capability rules
        self.size_rules = {
            'large_model': {
                'min_size_gb': 10,
                'add_capabilities': ['reasoning']
            },
            'very_large_model': {
                'min_size_gb': 50,
                'add_capabilities': ['parallel_calls', 'reasoning']
            }
        }
    
    async def discover_models(self) -> List[Dict[str, Any]]:
        """Discover Ollama models via API with enhanced categorization"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.api_base}/api/tags")
                response.raise_for_status()
                data = response.json()
                
                models = []
                for model_data in data.get("models", []):
                    enhanced_model = await self._enhance_model_data(model_data)
                    models.append(enhanced_model)
                
                log.info(f"Discovered {len(models)} Ollama models with enhanced metadata")
                
                # Sort models by usefulness (reasoning models first, then by size)
                models.sort(key=self._model_sort_key, reverse=True)
                return models
                
        except httpx.ConnectError:
            log.warning(f"Could not connect to Ollama at {self.api_base}")
            return []
        except Exception as e:
            log.error(f"Failed to discover Ollama models: {e}")
            return []
    
    async def _enhance_model_data(self, model_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance model data with capabilities and metadata"""
        model_name = model_data.get("name", "unknown")
        model_size = model_data.get("size", 0)
        
        # Get detailed model info if possible
        detailed_info = await self.get_model_metadata(model_name)
        
        # Determine model family and capabilities
        family_info = self._determine_model_family(model_name)
        capabilities = self._determine_capabilities(model_name, model_size)
        context_length = self._determine_context_length(model_name, family_info)
        
        enhanced_model = {
            "name": model_name,
            "size": model_size,
            "size_gb": round(model_size / (1024**3), 1) if model_size else 0,
            "modified_at": model_data.get("modified_at"),
            "digest": model_data.get("digest"),
            "source": "ollama_api",
            
            # Enhanced metadata
            "model_family": family_info.get("family", "unknown"),
            "specialization": family_info.get("specialization", "general"),
            "reasoning_capable": family_info.get("reasoning_capable", False),
            "estimated_parameters": self._estimate_parameters(model_name, model_size),
            "performance_tier": self._estimate_performance_tier(model_size),
            
            # Technical capabilities
            "capabilities": capabilities,
            "supports_tools": "tools" in capabilities,
            "supports_streaming": "streaming" in capabilities,
            "supports_vision": "vision" in capabilities,
            "supports_reasoning": "reasoning" in capabilities,
            "estimated_context_length": context_length,
            "estimated_max_output": min(context_length // 4, 8192) if context_length else 4096,
            
            # Detailed info from show command if available
            "detailed_info": detailed_info
        }
        
        return enhanced_model
    
    async def get_model_metadata(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed Ollama model information"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.api_base}/api/show",
                    json={"name": model_name}
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            log.debug(f"Failed to get Ollama model metadata for {model_name}: {e}")
            return None
    
    def _determine_model_family(self, model_name: str) -> Dict[str, Any]:
        """Determine model family based on name patterns"""
        model_lower = model_name.lower()
        
        for family, family_info in self.family_patterns.items():
            for pattern in family_info['patterns']:
                if re.search(pattern, model_lower):
                    return {
                        'family': family,
                        'specialization': family_info.get('specialization', 'general'),
                        'reasoning_capable': family_info.get('reasoning_capable', False),
                        'base_context': family_info.get('base_context', 8192),
                        'context_rules': family_info.get('context_rules', {}),
                        'base_capabilities': family_info.get('capabilities', ['text', 'streaming'])
                    }
        
        return {
            'family': 'unknown',
            'specialization': 'general',
            'reasoning_capable': False,
            'base_context': 8192,
            'context_rules': {},
            'base_capabilities': ['text', 'streaming']
        }
    
    def _determine_capabilities(self, model_name: str, model_size: int) -> List[str]:
        """Determine model capabilities based on name and size"""
        model_lower = model_name.lower()
        family_info = self._determine_model_family(model_name)
        capabilities = set(family_info['base_capabilities'])
        
        # Add capabilities based on patterns
        if any(pattern in model_lower for pattern in ['instruct', 'chat']):
            capabilities.add('system_messages')
        
        if any(pattern in model_lower for pattern in ['vision', 'llava', 'moondream']):
            capabilities.update(['vision', 'multimodal'])
        
        if any(pattern in model_lower for pattern in ['qwen', 'granite', 'llama3.1', 'phi4', 'reasoning']):
            capabilities.add('reasoning')
        
        if any(pattern in model_lower for pattern in ['gemma', 'mistral', 'granite', 'qwen']):
            capabilities.add('tools')
        
        # Add capabilities based on size
        if model_size:
            size_gb = model_size / (1024**3)
            if size_gb >= 10:
                capabilities.add('reasoning')
            if size_gb >= 50:
                capabilities.add('parallel_calls')
        
        return sorted(list(capabilities))
    
    def _determine_context_length(self, model_name: str, family_info: Dict[str, Any]) -> int:
        """Determine context length based on model name and family"""
        model_lower = model_name.lower()
        base_context = family_info.get('base_context', 8192)
        
        # Check family-specific context rules
        context_rules = family_info.get('context_rules', {})
        for pattern, context_length in context_rules.items():
            if re.search(pattern, model_lower):
                return context_length
        
        # Check for known large context patterns
        large_context_patterns = [
            (r'phi4', 128000),
            (r'llama.*3\.[23]', 128000),
            (r'qwen.*3', 32768),
            (r'granite.*3', 8192)
        ]
        
        for pattern, context_length in large_context_patterns:
            if re.search(pattern, model_lower):
                return context_length
        
        return base_context
    
    def _estimate_parameters(self, model_name: str, model_size: int) -> Optional[str]:
        """Estimate parameter count from model name and size"""
        model_lower = model_name.lower()
        
        # Extract from name patterns
        param_patterns = [
            (r'(\d+(?:\.\d+)?)b', lambda x: f"{x}B"),
            (r'(\d+)b', lambda x: f"{x}B"),
            (r'(\d+)billion', lambda x: f"{x}B")
        ]
        
        for pattern, formatter in param_patterns:
            match = re.search(pattern, model_lower)
            if match:
                return formatter(match.group(1))
        
        # Estimate from size (rough approximation)
        if model_size:
            size_gb = model_size / (1024**3)
            if size_gb < 1:
                return "< 1B"
            elif size_gb < 4:
                return "1-3B"
            elif size_gb < 8:
                return "3-7B"
            elif size_gb < 15:
                return "7-13B"
            elif size_gb < 30:
                return "13-30B"
            elif size_gb < 70:
                return "30-70B"
            else:
                return "70B+"
        
        return None
    
    def _estimate_performance_tier(self, model_size: int) -> str:
        """Estimate performance tier based on size"""
        if not model_size:
            return "unknown"
        
        size_gb = model_size / (1024**3)
        
        if size_gb < 1:
            return "nano"
        elif size_gb < 4:
            return "small"
        elif size_gb < 8:
            return "medium"
        elif size_gb < 15:
            return "large"
        elif size_gb < 30:
            return "extra-large"
        else:
            return "massive"
    
    def _model_sort_key(self, model: Dict[str, Any]) -> float:
        """Sort key for models (higher = better)"""
        score = 0
        
        # Reasoning models get priority
        if model.get("reasoning_capable", False):
            score += 100
        
        # Size-based scoring (larger models generally better, but not always)
        size_gb = model.get("size_gb", 0)
        if 1 <= size_gb <= 30:  # Sweet spot for most use cases
            score += size_gb * 2
        elif size_gb > 30:
            score += 30 * 2 + (size_gb - 30) * 0.5  # Diminishing returns for very large models
        
        # Capability bonuses
        capabilities = model.get("capabilities", [])
        capability_bonuses = {
            "reasoning": 20,
            "tools": 15,
            "vision": 10,
            "multimodal": 10,
            "streaming": 5
        }
        
        for capability in capabilities:
            score += capability_bonuses.get(capability, 0)
        
        # Family bonuses for popular/well-supported families
        family_bonuses = {
            "llama": 10,
            "qwen": 8,
            "granite": 6,
            "mistral": 5,
            "gemma": 4
        }
        
        family = model.get("model_family", "unknown")
        score += family_bonuses.get(family, 0)
        
        return score
    
    def normalize_model_data(self, raw_model: Dict[str, Any]) -> DiscoveredModel:
        """Convert Ollama model data to DiscoveredModel"""
        return DiscoveredModel(
            name=raw_model.get("name", "unknown"),
            provider=self.provider_name,
            size_bytes=raw_model.get("size"),
            modified_at=raw_model.get("modified_at"),
            family=raw_model.get("model_family", "unknown"),
            parameters=raw_model.get("estimated_parameters"),
            metadata={
                "digest": raw_model.get("digest"),
                "source": raw_model.get("source", "ollama_api"),
                
                # Enhanced Ollama-specific metadata
                "size_gb": raw_model.get("size_gb", 0),
                "specialization": raw_model.get("specialization", "general"),
                "reasoning_capable": raw_model.get("reasoning_capable", False),
                "performance_tier": raw_model.get("performance_tier", "unknown"),
                
                # Technical capabilities
                "capabilities": raw_model.get("capabilities", []),
                "supports_tools": raw_model.get("supports_tools", False),
                "supports_streaming": raw_model.get("supports_streaming", True),
                "supports_vision": raw_model.get("supports_vision", False),
                "supports_reasoning": raw_model.get("supports_reasoning", False),
                "estimated_context_length": raw_model.get("estimated_context_length", 8192),
                "estimated_max_output": raw_model.get("estimated_max_output", 4096),
                
                # Detailed model info if available
                "detailed_info": raw_model.get("detailed_info")
            }
        )
    
    async def pull_model(self, model_name: str) -> bool:
        """Pull/download a model to Ollama"""
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:  # Longer timeout for downloads
                response = await client.post(
                    f"{self.api_base}/api/pull",
                    json={"name": model_name}
                )
                response.raise_for_status()
                return True
        except Exception as e:
            log.error(f"Failed to pull Ollama model {model_name}: {e}")
            return False
    
    async def delete_model(self, model_name: str) -> bool:
        """Delete a model from Ollama"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.delete(
                    f"{self.api_base}/api/delete",
                    json={"name": model_name}
                )
                response.raise_for_status()
                return True
        except Exception as e:
            log.error(f"Failed to delete Ollama model {model_name}: {e}")
            return False
    
    def get_popular_models(self) -> List[Dict[str, str]]:
        """Get list of popular models that can be pulled"""
        return [
            {"name": "llama3.3", "description": "Latest Llama model with excellent reasoning"},
            {"name": "qwen3", "description": "High-performance multilingual model with tools"},
            {"name": "granite3.3", "description": "IBM's open-source model with strong capabilities"},
            {"name": "mistral", "description": "Efficient and fast model for general use"},
            {"name": "gemma3", "description": "Google's efficient model family"},
            {"name": "phi3", "description": "Microsoft's compact but capable model"},
            {"name": "codellama", "description": "Specialized for code generation"},
            {"name": "llava", "description": "Vision-capable model for image analysis"}
        ]


# Register the discoverer
DiscovererFactory.register_discoverer("ollama", OllamaModelDiscoverer)