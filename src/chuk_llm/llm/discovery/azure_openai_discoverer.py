# chuk_llm/llm/discovery/azure_openai_discoverer.py
"""
Azure OpenAI-specific model discoverer with deployment and available model discovery
"""

import logging
import os
import re
from typing import Dict, List, Optional, Any
import httpx
import asyncio

from .base import BaseModelDiscoverer, DiscoveredModel, DiscovererFactory
from .openai_discoverer import OpenAIModelDiscoverer

log = logging.getLogger(__name__)


class AzureOpenAIModelDiscoverer(BaseModelDiscoverer):
    """
    Azure OpenAI model discoverer that can discover:
    1. Deployed models (via deployments API)
    2. Available models for deployment (via models API)
    
    Key differences from OpenAI:
    - Uses Azure authentication and endpoints
    - Discovers deployments rather than direct models
    - Provides deployment name mapping
    - Supports both AD token and API key authentication
    """
    
    def __init__(self, 
                 provider_name: str = "azure_openai", 
                 api_key: Optional[str] = None,
                 azure_endpoint: Optional[str] = None,
                 api_version: str = "2024-02-01",
                 azure_ad_token: Optional[str] = None,
                 **config):
        super().__init__(provider_name, **config)
        
        # Azure-specific configuration
        self.api_key = api_key
        self.azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_version = api_version
        self.azure_ad_token = azure_ad_token
        
        # Validation
        if not self.azure_endpoint:
            log.warning("No Azure OpenAI endpoint provided - discovery will be limited")
        
        if not self.api_key and not self.azure_ad_token:
            log.warning("No Azure OpenAI credentials provided - using fallback models")
        
        # Initialize OpenAI discoverer for model family information
        self.openai_discoverer = OpenAIModelDiscoverer(
            provider_name="openai",
            api_key=None  # Don't use API key for family info
        )
        
        # Azure-specific deployment patterns
        self.deployment_patterns = {
            'reasoning_deployments': [
                r'.*o1.*', r'.*o3.*', r'.*o4.*', r'.*o5.*', r'.*gpt-5.*', r'.*reasoning.*'
            ],
            'vision_deployments': [
                r'.*gpt-4.*vision.*', r'.*gpt-4o.*', r'.*gpt-5.*vision.*', r'.*vision.*'
            ],
            'chat_deployments': [
                r'.*gpt-4.*', r'.*gpt-3.*', r'.*gpt-5.*', r'.*chat.*'
            ],
            'embedding_deployments': [
                r'.*embed.*', r'.*ada.*'
            ]
        }
    
    async def discover_models(self) -> List[Dict[str, Any]]:
        """
        Discover Azure OpenAI models via available models API only
        
        Returns only actually discovered models, no fallbacks.
        """
        if not self.azure_endpoint or not (self.api_key or self.azure_ad_token):
            log.warning("No Azure OpenAI credentials - cannot discover models")
            return []
        
        discovered_models = []
        
        # Try to discover available models (this should work)
        try:
            available_models = await self._discover_available_models()
            discovered_models.extend(available_models)
            log.info(f"Discovered {len(available_models)} available Azure OpenAI models")
        except Exception as e:
            log.error(f"Failed to discover available Azure models: {e}")
            # No fallback - return empty list
            return []
        
        # Note: Deployment discovery requires Azure Management API since 2024
        # We can't easily list active deployments with just the data plane API
        log.debug("Azure deployment discovery requires Azure Management API (not implemented)")
        
        # Sort models by usefulness
        discovered_models.sort(key=self._azure_model_sort_key)
        
        return discovered_models
    
    async def _discover_deployments(self) -> List[Dict[str, Any]]:
        """
        DEPRECATED: Azure OpenAI removed simple deployment listing in 2024
        
        The /openai/deployments endpoint was deprecated in favor of the Azure Management API:
        https://management.azure.com/subscriptions/{}/resourceGroups/{}/providers/Microsoft.CognitiveServices/accounts/{}/deployments
        
        This method is kept for potential future restoration of the API.
        """
        log.warning("Azure OpenAI deployment discovery via data plane API is no longer supported as of 2024")
        log.info("To discover deployments, use the Azure Management API with subscription/resource group info")
        return []
    
    async def _discover_available_models(self) -> List[Dict[str, Any]]:
        """Discover models available for deployment in Azure OpenAI"""
        if not self.azure_endpoint:
            return []
        
        headers = self._get_auth_headers()
        if not headers:
            return []
        
        # Use the correct Azure OpenAI models API endpoint
        url = f"{self.azure_endpoint}/openai/models"
        params = {"api-version": self.api_version}
        
        log.debug(f"Requesting Azure models from: {url}")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            models = []
            for model_data in data.get("data", []):
                # Use OpenAI discoverer's categorization for available models
                enhanced_model = self.openai_discoverer._categorize_model(
                    model_data["id"], model_data
                )
                
                # Add Azure-specific metadata
                enhanced_model.update({
                    "provider": "azure_openai",
                    "deployment_status": "available_for_deployment",
                    "azure_model_id": model_data["id"],
                    "source": "azure_models_api"
                })
                
                models.append(enhanced_model)
            
            return models
    
    def _get_auth_headers(self) -> Optional[Dict[str, str]]:
        """Get authentication headers for Azure OpenAI API"""
        if self.azure_ad_token:
            return {"Authorization": f"Bearer {self.azure_ad_token}"}
        elif self.api_key:
            return {"api-key": self.api_key}
        else:
            return None
    
    def _enhance_deployment_data(self, deployment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance Azure deployment data with model categorization"""
        deployment_id = deployment_data.get("id", "unknown")
        model_name = deployment_data.get("model", "unknown")
        
        # Get base model information from OpenAI discoverer
        base_model_info = self.openai_discoverer._categorize_model(model_name, {"id": model_name})
        
        # Azure-specific enhancement
        enhanced_data = {
            "name": deployment_id,  # In Azure, the deployment name is what we use
            "underlying_model": model_name,  # The actual OpenAI model
            "deployment_id": deployment_id,
            "deployment_status": "deployed",
            
            # Deployment metadata
            "created_at": deployment_data.get("created_at"),
            "updated_at": deployment_data.get("updated_at"),
            "status": deployment_data.get("status", "unknown"),
            "scale_type": deployment_data.get("scale_settings", {}).get("scale_type"),
            "capacity": self._extract_capacity(deployment_data),
            
            # Source info
            "source": "azure_deployments_api",
            "provider": "azure_openai",
            
            # Inherit model characteristics from base model
            **{k: v for k, v in base_model_info.items() if k not in ["name", "source"]},
            
            # Azure-specific capabilities
            "azure_specific": {
                "deployment_name": deployment_id,
                "base_model": model_name,
                "endpoint_type": "azure_openai",
                "supports_content_filtering": True,
                "supports_managed_identity": True,
            }
        }
        
        # Determine deployment category
        enhanced_data["deployment_category"] = self._categorize_deployment(deployment_id)
        
        return enhanced_data
    
    def _extract_capacity(self, deployment_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract capacity information from deployment data"""
        scale_settings = deployment_data.get("scale_settings", {})
        
        capacity_info = {}
        
        if "capacity" in scale_settings:
            capacity_info["capacity"] = scale_settings["capacity"]
        
        if "scale_type" in scale_settings:
            capacity_info["scale_type"] = scale_settings["scale_type"]
        
        return capacity_info if capacity_info else None
    
    def _categorize_deployment(self, deployment_name: str) -> str:
        """Categorize Azure deployment based on naming patterns"""
        deployment_lower = deployment_name.lower()
        
        for category, patterns in self.deployment_patterns.items():
            for pattern in patterns:
                if re.search(pattern, deployment_lower):
                    return category.replace('_deployments', '')
        
        return 'general'
    
    def _get_azure_fallback_models(self) -> List[Dict[str, Any]]:
        """Fallback models for Azure OpenAI when API discovery fails"""
        # Get OpenAI fallback models and convert to Azure deployments
        openai_fallback = self.openai_discoverer._get_fallback_models()
        
        azure_models = []
        for model in openai_fallback:
            # Convert to Azure deployment format
            azure_model = {
                **model,
                "name": f"{model['name']}-deployment",  # Common Azure naming pattern
                "underlying_model": model["name"],
                "deployment_id": f"{model['name']}-deployment",
                "deployment_status": "assumed_available",
                "provider": "azure_openai",
                "source": "azure_fallback",
                "azure_specific": {
                    "deployment_name": f"{model['name']}-deployment",
                    "base_model": model["name"],
                    "endpoint_type": "azure_openai",
                    "supports_content_filtering": True,
                    "supports_managed_identity": True,
                }
            }
            azure_models.append(azure_model)
        
        log.info(f"Using Azure fallback models: {len(azure_models)} models")
        return azure_models
    
    def _azure_model_sort_key(self, model: Dict[str, Any]) -> tuple:
        """Sort key for Azure models (deployed first, then by capability)"""
        deployment_status = model.get("deployment_status", "unknown")
        
        # Priority order: deployed > available > fallback
        status_priority = {
            "deployed": 0,
            "available_for_deployment": 1,
            "available_not_deployed": 2,
            "assumed_available": 3,
            "unknown": 4
        }
        
        # Use OpenAI model sort key for capability sorting
        capability_priority = self.openai_discoverer._model_sort_key(model)
        
        return (
            status_priority.get(deployment_status, 5),
            capability_priority
        )
    
    def normalize_model_data(self, raw_model: Dict[str, Any]) -> DiscoveredModel:
        """Convert Azure OpenAI model data to DiscoveredModel"""
        return DiscoveredModel(
            name=raw_model.get("name", "unknown"),
            provider=self.provider_name,
            created_at=raw_model.get("created_at"),
            family=raw_model.get("model_family", "unknown"),
            metadata={
                # Azure-specific metadata
                "deployment_id": raw_model.get("deployment_id"),
                "underlying_model": raw_model.get("underlying_model"),
                "deployment_status": raw_model.get("deployment_status"),
                "deployment_category": raw_model.get("deployment_category"),
                "azure_specific": raw_model.get("azure_specific", {}),
                
                # Capacity and scaling info
                "capacity": raw_model.get("capacity"),
                "scale_type": raw_model.get("scale_type"),
                "status": raw_model.get("status"),
                
                # Inherit all OpenAI model capabilities
                "owned_by": raw_model.get("owned_by"),
                "object": raw_model.get("object"),
                "source": raw_model.get("source"),
                
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
    
    async def test_deployment_availability(self, deployment_name: str) -> bool:
        """Test if a specific Azure OpenAI deployment is available"""
        if not self.azure_endpoint or not (self.api_key or self.azure_ad_token):
            log.debug(f"No credentials - cannot test deployment availability: {deployment_name}")
            return False
        
        try:
            import openai
            
            # Create Azure OpenAI client
            client_kwargs = {
                "api_version": self.api_version,
                "azure_endpoint": self.azure_endpoint,
            }
            
            if self.azure_ad_token:
                client_kwargs["azure_ad_token"] = self.azure_ad_token
            else:
                client_kwargs["api_key"] = self.api_key
            
            client = openai.AsyncAzureOpenAI(**client_kwargs)
            
            # Test with minimal request
            test_params = {
                "model": deployment_name,  # In Azure, model parameter is the deployment name
                "messages": [{"role": "user", "content": "Test"}],
                "max_tokens": 5
            }
            
            response = await client.chat.completions.create(**test_params)
            await client.close()
            return True
            
        except ImportError as e:
            log.debug(f"OpenAI package not available for testing deployment availability: {e}")
            return False
        except Exception as e:
            log.debug(f"Deployment {deployment_name} availability test failed: {e}")
            return False
    
    async def create_deployment(self, model_name: str, deployment_name: str, 
                               capacity: int = 10) -> bool:
        """Create a new deployment in Azure OpenAI (if supported)"""
        # This would require management API access and specific permissions
        log.warning("Deployment creation not implemented - requires Azure management API access")
        return False
    
    async def discover_deployments_by_testing(self, 
                                          common_deployment_names: List[str] = None) -> List[Dict[str, Any]]:
        """
        Alternative deployment discovery by testing common deployment names
        
        Since Azure OpenAI no longer provides a simple API to list deployments,
        this method tests common deployment names to see which ones exist.
        """
        if not self.azure_endpoint or not (self.api_key or self.azure_ad_token):
            log.warning("No Azure credentials - cannot test deployments")
            return []
        
        if common_deployment_names is None:
            # Common Azure OpenAI deployment naming patterns
            common_deployment_names = [
                "gpt-5", "gpt-5-mini", "gpt-5-turbo",  # GPT-5 variants
                "gpt-4o", "gpt-4o-mini", "gpt-4", "gpt-4-turbo",
                "gpt-35-turbo", "gpt-3.5-turbo", 
                "o1-mini", "o1-preview",
                "o3-mini", "o4-mini",  # Future reasoning models
                "text-embedding-ada-002", "text-embedding-3-large"
            ]
        
        discovered_deployments = []
        log.info(f"Testing {len(common_deployment_names)} common deployment names...")
        
        for deployment_name in common_deployment_names:
            try:
                is_available = await self.test_deployment_availability(deployment_name)
                if is_available:
                    log.info(f"✅ Found active deployment: {deployment_name}")
                    
                    # Create deployment info based on name
                    deployment_info = {
                        "name": deployment_name,
                        "deployment_id": deployment_name,
                        "deployment_status": "active_discovered",
                        "underlying_model": deployment_name,  # Assume deployment name matches model
                        "source": "deployment_testing",
                        "discovery_method": "availability_test"
                    }
                    
                    # Enhance with model family info from OpenAI discoverer
                    enhanced_info = self.openai_discoverer._categorize_model(deployment_name, {
                        "id": deployment_name,
                        "owned_by": "openai"
                    })
                    
                    # Merge the info
                    deployment_info.update(enhanced_info)
                    deployment_info["provider"] = "azure_openai"
                    
                    discovered_deployments.append(deployment_info)
                else:
                    log.debug(f"❌ Deployment not found: {deployment_name}")
                    
            except Exception as e:
                log.debug(f"Error testing deployment {deployment_name}: {e}")
                continue
        
        log.info(f"Discovered {len(discovered_deployments)} active deployments via testing")
        return discovered_deployments

# Register the Azure OpenAI discoverer
DiscovererFactory.register_discoverer("azure_openai", AzureOpenAIModelDiscoverer)