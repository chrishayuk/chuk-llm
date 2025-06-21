# chuk_llm/llm/discovery/providers.py
"""
Provider-specific model discoverers using the universal discovery system
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
import httpx

from .engine import BaseModelDiscoverer, DiscoveredModel

log = logging.getLogger(__name__)


class OllamaModelDiscoverer(BaseModelDiscoverer):
    """Ollama-specific model discoverer"""
    
    def __init__(self, api_base: str = "http://localhost:11434", **config):
        super().__init__("ollama", **config)
        self.api_base = api_base.rstrip('/')
    
    async def discover_models(self) -> List[Dict[str, Any]]:
        """Discover Ollama models via API"""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.api_base}/api/tags")
                response.raise_for_status()
                data = response.json()
                
                models = []
                for model_data in data.get("models", []):
                    models.append({
                        "name": model_data["name"],
                        "size": model_data["size"],
                        "modified_at": model_data["modified_at"],
                        "digest": model_data["digest"],
                        "source": "ollama_api"
                    })
                
                return models
                
        except Exception as e:
            log.error(f"Failed to discover Ollama models: {e}")
            return []
    
    async def get_model_metadata(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed Ollama model information"""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{self.api_base}/api/show",
                    json={"name": model_name}
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            log.debug(f"Failed to get Ollama model metadata for {model_name}: {e}")
            return None
    
    def normalize_model_data(self, raw_model: Dict[str, Any]) -> DiscoveredModel:
        """Convert Ollama model data to DiscoveredModel"""
        return DiscoveredModel(
            name=raw_model.get("name", "unknown"),
            provider=self.provider_name,
            size_bytes=raw_model.get("size"),
            modified_at=raw_model.get("modified_at"),
            metadata={
                "digest": raw_model.get("digest"),
                "source": raw_model.get("source", "ollama_api")
            }
        )


class OpenAIModelDiscoverer(BaseModelDiscoverer):
    """OpenAI-compatible model discoverer (OpenAI, Groq, etc.)"""
    
    def __init__(self, api_key: str, api_base: str = "https://api.openai.com/v1", **config):
        super().__init__(config.get("provider_name", "openai"), **config)
        self.api_key = api_key
        self.api_base = api_base.rstrip('/')
    
    async def discover_models(self) -> List[Dict[str, Any]]:
        """Discover models via OpenAI-compatible API"""
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(f"{self.api_base}/models", headers=headers)
                response.raise_for_status()
                data = response.json()
                
                models = []
                for model_data in data.get("data", []):
                    models.append({
                        "name": model_data["id"],
                        "created_at": model_data.get("created"),
                        "owned_by": model_data.get("owned_by"),
                        "object": model_data.get("object"),
                        "source": f"{self.provider_name}_api"
                    })
                
                return models
                
        except Exception as e:
            log.error(f"Failed to discover {self.provider_name} models: {e}")
            return []
    
    def normalize_model_data(self, raw_model: Dict[str, Any]) -> DiscoveredModel:
        """Convert OpenAI-style model data to DiscoveredModel"""
        return DiscoveredModel(
            name=raw_model.get("name", "unknown"),
            provider=self.provider_name,
            created_at=raw_model.get("created_at"),
            metadata={
                "owned_by": raw_model.get("owned_by"),
                "object": raw_model.get("object"),
                "source": raw_model.get("source", f"{self.provider_name}_api")
            }
        )


class HuggingFaceModelDiscoverer(BaseModelDiscoverer):
    """Hugging Face model discoverer"""
    
    def __init__(self, api_key: Optional[str] = None, **config):
        super().__init__("huggingface", **config)
        self.api_key = api_key
        self.search_query = config.get("search_query", "text-generation")
        self.limit = config.get("limit", 50)
    
    async def discover_models(self) -> List[Dict[str, Any]]:
        """Discover Hugging Face models via API"""
        try:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            params = {
                "search": self.search_query,
                "limit": self.limit,
                "sort": "downloads",
                "direction": "desc"
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
                    models.append({
                        "name": model_data["id"],
                        "downloads": model_data.get("downloads", 0),
                        "likes": model_data.get("likes", 0),
                        "created_at": model_data.get("createdAt"),
                        "modified_at": model_data.get("lastModified"),
                        "tags": model_data.get("tags", []),
                        "library_name": model_data.get("library_name"),
                        "source": "huggingface_api"
                    })
                
                return models
                
        except Exception as e:
            log.error(f"Failed to discover Hugging Face models: {e}")
            return []
    
    def normalize_model_data(self, raw_model: Dict[str, Any]) -> DiscoveredModel:
        """Convert Hugging Face model data to DiscoveredModel"""
        return DiscoveredModel(
            name=raw_model.get("name", "unknown"),
            provider=self.provider_name,
            created_at=raw_model.get("created_at"),
            modified_at=raw_model.get("modified_at"),
            metadata={
                "downloads": raw_model.get("downloads", 0),
                "likes": raw_model.get("likes", 0),
                "tags": raw_model.get("tags", []),
                "library_name": raw_model.get("library_name"),
                "source": raw_model.get("source", "huggingface_api")
            }
        )


class AnthropicModelDiscoverer(BaseModelDiscoverer):
    """Anthropic model discoverer (using known model list since no public API)"""
    
    def __init__(self, **config):
        super().__init__("anthropic", **config)
        # Anthropic doesn't have a public models API, so we use known models
        self.known_models = config.get("known_models", [
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229", 
            "claude-3-haiku-20240307",
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
        ])
    
    async def discover_models(self) -> List[Dict[str, Any]]:
        """Return known Anthropic models"""
        models = []
        for model_name in self.known_models:
            models.append({
                "name": model_name,
                "source": "anthropic_known",
                "available": True
            })
        return models
    
    def normalize_model_data(self, raw_model: Dict[str, Any]) -> DiscoveredModel:
        """Convert known Anthropic model to DiscoveredModel"""
        return DiscoveredModel(
            name=raw_model.get("name", "unknown"),
            provider=self.provider_name,
            metadata={
                "source": raw_model.get("source", "anthropic_known"),
                "available": raw_model.get("available", True)
            }
        )


class LocalModelDiscoverer(BaseModelDiscoverer):
    """Discoverer for local model files"""
    
    def __init__(self, model_paths: List[str], **config):
        super().__init__("local", **config)
        self.model_paths = model_paths
        self.file_extensions = config.get("file_extensions", [".gguf", ".bin", ".safetensors"])
    
    async def discover_models(self) -> List[Dict[str, Any]]:
        """Discover local model files"""
        models = []
        
        for model_path in self.model_paths:
            try:
                from pathlib import Path
                path = Path(model_path)
                
                if path.is_file():
                    # Single file
                    if any(path.name.endswith(ext) for ext in self.file_extensions):
                        models.append(self._file_to_model_data(path))
                elif path.is_dir():
                    # Directory with model files
                    for ext in self.file_extensions:
                        for model_file in path.rglob(f"*{ext}"):
                            models.append(self._file_to_model_data(model_file))
                            
            except Exception as e:
                log.warning(f"Failed to scan model path {model_path}: {e}")
        
        return models
    
    def _file_to_model_data(self, file_path) -> Dict[str, Any]:
        """Convert file path to model data"""
        from pathlib import Path
        import time
        
        path = Path(file_path)
        stat = path.stat()
        
        return {
            "name": path.stem,  # Filename without extension
            "path": str(path),
            "size": stat.st_size,
            "modified_at": time.ctime(stat.st_mtime),
            "extension": path.suffix,
            "source": "local_file"
        }
    
    def normalize_model_data(self, raw_model: Dict[str, Any]) -> DiscoveredModel:
        """Convert local file model to DiscoveredModel"""
        return DiscoveredModel(
            name=raw_model.get("name", "unknown"),
            provider=self.provider_name,
            size_bytes=raw_model.get("size"),
            modified_at=raw_model.get("modified_at"),
            metadata={
                "path": raw_model.get("path"),
                "extension": raw_model.get("extension"),
                "source": raw_model.get("source", "local_file")
            }
        )


# Factory for creating provider-specific discoverers
class DiscovererFactory:
    """Factory for creating provider-specific discoverers"""
    
    DISCOVERERS = {
        "ollama": OllamaModelDiscoverer,
        "openai": OpenAIModelDiscoverer,
        "groq": OpenAIModelDiscoverer,
        "deepseek": OpenAIModelDiscoverer,
        "perplexity": OpenAIModelDiscoverer,
        "huggingface": HuggingFaceModelDiscoverer,
        "anthropic": AnthropicModelDiscoverer,
        "local": LocalModelDiscoverer,
    }
    
    @classmethod
    def create_discoverer(cls, provider_name: str, **config) -> BaseModelDiscoverer:
        """Create a discoverer for the given provider"""
        discoverer_class = cls.DISCOVERERS.get(provider_name)
        
        if not discoverer_class:
            raise ValueError(f"No discoverer available for provider: {provider_name}")
        
        # Add provider_name to config for OpenAI-compatible providers
        if discoverer_class == OpenAIModelDiscoverer:
            config["provider_name"] = provider_name
        
        return discoverer_class(**config)
    
    @classmethod
    def register_discoverer(cls, provider_name: str, discoverer_class: type):
        """Register a custom discoverer for a provider"""
        cls.DISCOVERERS[provider_name] = discoverer_class
    
    @classmethod
    def list_supported_providers(cls) -> List[str]:
        """List providers that support discovery"""
        return list(cls.DISCOVERERS.keys())