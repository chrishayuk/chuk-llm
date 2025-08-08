# chuk_llm/llm/discovery/general_discoverers.py
"""
General provider discoverers for other LLM providers
"""

import logging
from typing import Dict, List, Optional, Any
import httpx
from pathlib import Path
import time

from .base import BaseModelDiscoverer, DiscoveredModel, DiscovererFactory

log = logging.getLogger(__name__)


class OpenAICompatibleDiscoverer(BaseModelDiscoverer):
    """Generic discoverer for OpenAI-compatible APIs (Groq, Deepseek, etc.)"""

    def __init__(self, provider_name: str, api_key: str, api_base: str, **config):
        super().__init__(provider_name, **config)
        self.api_key = api_key
        self.api_base = api_base.rstrip('/')

        # Provider-specific model filtering
        self.model_filters = {
            'groq': ['ft-', ':'],  # Skip fine-tuned and org models
            'deepseek': ['ft-'],
            'perplexity': ['ft-'],
            'together': ['ft-']
        }

    async def discover_models(self) -> List[Dict[str, Any]]:
        """Discover models via OpenAI-compatible API"""
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(f"{self.api_base}/models", headers=headers)
                response.raise_for_status()
                data = response.json()

                models = []
                filters = self.model_filters.get(self.provider_name, [])

                for model_data in data.get("data", []):
                    model_id = model_data["id"]

                    # Apply provider-specific filtering
                    if any(filter_str in model_id for filter_str in filters):
                        continue

                    models.append({
                        "name": model_id,
                        "created_at": model_data.get("created"),
                        "owned_by": model_data.get("owned_by"),
                        "object": model_data.get("object"),
                        "source": f"{self.provider_name}_api",
                        "provider_specific": self._get_provider_specifics(model_id)
                    })

                return models

        except Exception as e:
            log.error(f"Failed to discover {self.provider_name} models: {e}")
            return []

    def _get_provider_specifics(self, model_id: str) -> Dict[str, Any]:
        """Get provider-specific model characteristics"""
        if self.provider_name == 'groq':
            return self._get_groq_specifics(model_id)
        elif self.provider_name == 'deepseek':
            return self._get_deepseek_specifics(model_id)
        elif self.provider_name == 'perplexity':
            return self._get_perplexity_specifics(model_id)
        else:
            return {}

    def _get_groq_specifics(self, model_id: str) -> Dict[str, Any]:
        """Groq-specific model characteristics"""
        model_lower = model_id.lower()

        characteristics = {
            "speed_tier": "ultra-fast",
            "supports_streaming": True,
            "supports_tools": True,
            "estimated_context_length": 32768
        }

        if "llama" in model_lower:
            characteristics.update({
                "model_family": "llama",
                "reasoning_capable": True if "70b" in model_lower else False,
                "estimated_context_length": 32768 if "3.1" in model_lower else 8192
            })
        elif "mixtral" in model_lower:
            characteristics.update({
                "model_family": "mixtral",
                "reasoning_capable": True,
                "estimated_context_length": 32768
            })
        elif "gemma" in model_lower:
            characteristics.update({
                "model_family": "gemma",
                "reasoning_capable": False,
                "estimated_context_length": 8192
            })

        return characteristics

    def _get_deepseek_specifics(self, model_id: str) -> Dict[str, Any]:
        """Deepseek-specific model characteristics"""
        model_lower = model_id.lower()

        characteristics = {
            "speed_tier": "fast",
            "supports_streaming": True,
            "supports_tools": True,
            "estimated_context_length": 65536
        }

        if "chat" in model_lower:
            characteristics.update({
                "model_family": "deepseek_chat",
                "reasoning_capable": False,
                "specialization": "chat"
            })
        elif "reasoner" in model_lower or "reasoning" in model_lower:
            characteristics.update({
                "model_family": "deepseek_reasoning",
                "reasoning_capable": True,
                "specialization": "reasoning",
                "parameter_requirements": {"use_max_completion_tokens": True}
            })
        elif "coder" in model_lower:
            characteristics.update({
                "model_family": "deepseek_coder",
                "reasoning_capable": True,
                "specialization": "code"
            })

        return characteristics

    def _get_perplexity_specifics(self, model_id: str) -> Dict[str, Any]:
        """Perplexity-specific model characteristics"""
        model_lower = model_id.lower()

        characteristics = {
            "speed_tier": "medium",
            "supports_streaming": True,
            "supports_tools": False,
            "estimated_context_length": 127072,
            "has_web_search": True
        }

        if "sonar" in model_lower:
            characteristics.update({
                "model_family": "sonar",
                "has_web_search": True,
                "supports_vision": True if "pro" in model_lower else False
            })
        elif "research" in model_lower:
            characteristics.update({
                "model_family": "research",
                "reasoning_capable": True,
                "specialization": "research"
            })

        return characteristics

    def normalize_model_data(self, raw_model: Dict[str, Any]) -> DiscoveredModel:
        """Convert model data to DiscoveredModel"""
        provider_specifics = raw_model.get("provider_specific", {})

        return DiscoveredModel(
            name=raw_model.get("name", "unknown"),
            provider=self.provider_name,
            created_at=raw_model.get("created_at"),
            family=provider_specifics.get("model_family", "unknown"),
            metadata={
                "owned_by": raw_model.get("owned_by"),
                "object": raw_model.get("object"),
                "source": raw_model.get("source"),
                **provider_specifics
            }
        )


class HuggingFaceModelDiscoverer(BaseModelDiscoverer):
    """Hugging Face model discoverer"""

    def __init__(self, provider_name: str = "huggingface", api_key: Optional[str] = None, **config):
        super().__init__(provider_name, **config)
        self.api_key = api_key
        self.search_query = config.get("search_query", "text-generation")
        self.limit = config.get("limit", 50)
        self.sort = config.get("sort", "downloads")

    async def discover_models(self) -> List[Dict[str, Any]]:
        """Discover Hugging Face models via API"""
        try:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            params = {
                "search": self.search_query,
                "limit": self.limit,
                "sort": self.sort,
                "direction": "desc",
                "filter": "text-generation"
            }

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    "https://huggingface.co/api/models",
                    headers=headers,
                    params=params
                )
                response.raise_for_status()
                data = response.json()

                models = []
                for model_data in data:
                    # Skip models that aren't suitable for inference
                    if not self._is_suitable_for_inference(model_data):
                        continue

                    models.append({
                        "name": model_data["id"],
                        "downloads": model_data.get("downloads", 0),
                        "likes": model_data.get("likes", 0),
                        "created_at": model_data.get("createdAt"),
                        "modified_at": model_data.get("lastModified"),
                        "tags": model_data.get("tags", []),
                        "library_name": model_data.get("library_name"),
                        "source": "huggingface_api",
                        "model_characteristics": self._analyze_hf_model(model_data)
                    })

                return models

        except Exception as e:
            log.error(f"Failed to discover Hugging Face models: {e}")
            return []

    def _is_suitable_for_inference(self, model_data: Dict[str, Any]) -> bool:
        """Check if HF model is suitable for inference"""
        tags = model_data.get("tags", [])
        library = model_data.get("library_name", "")

        # Must be a text generation model
        if "text-generation" not in tags:
            return False

        # Skip if it's not a supported library
        supported_libs = ["transformers", "text-generation-inference"]
        if library and library not in supported_libs:
            return False

        # Skip models that are too large for typical use
        downloads = model_data.get("downloads", 0)
        if downloads < 100:  # Skip very unpopular models
            return False

        return True

    def _analyze_hf_model(self, model_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze HF model characteristics"""
        model_id = model_data["id"].lower()
        tags = model_data.get("tags", [])

        characteristics = {
            "model_family": self._determine_hf_family(model_id, tags),
            "estimated_size": self._estimate_hf_size(model_id, tags),
            "specialization": self._determine_hf_specialization(model_id, tags),
            "reasoning_capable": self._has_reasoning_capability(model_id, tags),
            "supports_tools": self._supports_function_calling(model_id, tags),
            "popularity_score": self._calculate_popularity_score(model_data)
        }

        return characteristics

    def _determine_hf_family(self, model_id: str, tags: List[str]) -> str:
        """Determine HF model family"""
        if "llama" in model_id:
            return "llama"
        elif "mistral" in model_id or "mixtral" in model_id:
            return "mistral"
        elif "qwen" in model_id:
            return "qwen"
        elif "gemma" in model_id:
            return "gemma"
        elif "phi" in model_id:
            return "phi"
        else:
            return "unknown"

    def _estimate_hf_size(self, model_id: str, tags: List[str]) -> str:
        """Estimate model size from ID and tags"""
        import re

        # Look for size indicators
        size_patterns = [
            (r'(\d+)b', lambda x: f"{x}B"),
            (r'(\d+)billion', lambda x: f"{x}B"),
            (r'(\d+\.?\d*)b', lambda x: f"{x}B")
        ]

        for pattern, formatter in size_patterns:
            match = re.search(pattern, model_id.lower())
            if match:
                return formatter(match.group(1))

        # Check tags for size info
        for tag in tags:
            if any(size in tag.lower() for size in ['7b', '13b', '30b', '70b']):
                return tag

        return "unknown"

    def _determine_hf_specialization(self, model_id: str, tags: List[str]) -> str:
        """Determine model specialization"""
        if any(spec in model_id.lower() for spec in ['code', 'coder']):
            return "code"
        elif any(spec in model_id.lower() for spec in ['chat', 'instruct']):
            return "chat"
        elif any(spec in model_id.lower() for spec in ['math', 'reasoning']):
            return "reasoning"
        else:
            return "general"

    def _has_reasoning_capability(self, model_id: str, tags: List[str]) -> bool:
        """Check if model has reasoning capabilities"""
        reasoning_indicators = ['reasoning', 'math', 'logic', '70b', '405b']
        return any(indicator in model_id.lower() for indicator in reasoning_indicators)

    def _supports_function_calling(self, model_id: str, tags: List[str]) -> bool:
        """Check if model supports function calling"""
        # Most recent instruct models support function calling
        return any(indicator in model_id.lower() for indicator in ['instruct', 'chat', 'tool'])

    def _calculate_popularity_score(self, model_data: Dict[str, Any]) -> float:
        """Calculate popularity score for ranking"""
        downloads = model_data.get("downloads", 0)
        likes = model_data.get("likes", 0)

        # Weighted score
        score = downloads * 0.7 + likes * 0.3
        return round(score, 2)
