# chuk_llm/llm/discovery/__init__.py
"""
Model discovery system - Clean modular version with Azure OpenAI support
"""

# Import base classes and factory
from .base import BaseModelDiscoverer, DiscoveredModel, DiscovererFactory

# Import engine components
from .engine import (
    ConfigDrivenInferenceEngine, 
    UniversalModelDiscoveryManager
)

# Import manager
from .manager import UniversalDiscoveryManager, DiscoveryResults

# Clean exports - no legacy imports
__all__ = [
    # Base classes
    "BaseModelDiscoverer",
    "DiscoveredModel", 
    "DiscovererFactory",
    
    # Engine components
    "ConfigDrivenInferenceEngine",
    "UniversalModelDiscoveryManager", 
    
    # Manager
    "UniversalDiscoveryManager",
    "DiscoveryResults",
]

# Initialize factory (triggers auto-import of discoverers)
def _initialize_factory():
    """Initialize the discoverer factory"""
    try:
        DiscovererFactory._auto_import_discoverers()
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Failed to initialize discovery factory: {e}")

# Auto-initialize on import
_initialize_factory()