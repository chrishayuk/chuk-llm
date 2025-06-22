# chuk_llm/api/session_utils.py
"""
Session utilities for chuk_llm with graceful Redis handling.
"""

import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def check_session_backend_availability() -> Dict[str, Any]:
    """
    Check availability of session storage backends.
    
    Returns:
        Dictionary with availability information and recommendations
    """
    result = {
        "memory_available": True,
        "redis_available": False,
        "current_provider": os.getenv("SESSION_PROVIDER", "memory"),
        "recommendations": [],
        "errors": []
    }
    
    # Check Redis availability
    try:
        import redis
        result["redis_available"] = True
        result["redis_version"] = redis.__version__
    except ImportError:
        result["errors"].append("Redis package not installed")
        result["recommendations"].append(
            "For persistent sessions, install with: pip install chuk_llm[redis]"
        )
    
    # Check current configuration
    provider = result["current_provider"]
    if provider == "redis" and not result["redis_available"]:
        result["errors"].append(
            f"SESSION_PROVIDER=redis but Redis not available"
        )
        result["recommendations"].append(
            "Either install Redis support or switch to memory: export SESSION_PROVIDER=memory"
        )
    
    # Check session manager availability
    try:
        import chuk_ai_session_manager
        result["session_manager_available"] = True
        result["session_manager_version"] = chuk_ai_session_manager.__version__
        
        # Get storage info if available
        try:
            storage_info = chuk_ai_session_manager.get_storage_info()
            result["storage_info"] = storage_info
        except Exception as e:
            result["storage_errors"] = [str(e)]
            
    except ImportError:
        # This should not happen since it's a core dependency
        result["session_manager_available"] = False
        result["errors"].append("chuk-ai-session-manager missing (this is unexpected)")
        result["recommendations"].append(
            "Reinstall chuk_llm: pip install --force-reinstall chuk_llm"
        )
    
    return result


def validate_session_configuration() -> bool:
    """
    Validate the current session configuration.
    
    Returns:
        True if configuration is valid, False otherwise
    """
    info = check_session_backend_availability()
    
    if info["errors"]:
        for error in info["errors"]:
            logger.warning(f"Session configuration issue: {error}")
        return False
    
    return True


def get_session_recommendations() -> list[str]:
    """
    Get recommendations for improving session configuration.
    
    Returns:
        List of recommendation strings
    """
    info = check_session_backend_availability()
    return info.get("recommendations", [])


def auto_configure_sessions() -> bool:
    """
    Automatically configure sessions with the best available backend.
    
    Returns:
        True if successfully configured, False otherwise
    """
    try:
        # Try to configure with the best available option
        import chuk_ai_session_manager
        
        # Check what's available
        info = check_session_backend_availability()
        
        if info["redis_available"] and not info["errors"]:
            # Redis is available and working
            success = chuk_ai_session_manager.configure_storage()
            if success:
                logger.info("Sessions configured with Redis backend")
                return True
        
        # Fall back to memory
        os.environ["SESSION_PROVIDER"] = "memory"
        success = chuk_ai_session_manager.configure_storage()
        if success:
            logger.info("Sessions configured with memory backend")
            return True
            
    except Exception as e:
        logger.debug(f"Auto-configuration failed: {e}")
    
    return False


def print_session_diagnostics():
    """Print comprehensive session diagnostics."""
    info = check_session_backend_availability()
    
    print("🔍 ChukLLM Session Diagnostics")
    print("=" * 40)
    
    # Current status
    print(f"Session Manager: {'✅' if info.get('session_manager_available') else '❌'}")
    print(f"Memory Storage: {'✅' if info['memory_available'] else '❌'}")
    print(f"Redis Storage: {'✅' if info['redis_available'] else '❌'}")
    print(f"Current Provider: {info['current_provider']}")
    
    # Storage info
    if "storage_info" in info:
        storage = info["storage_info"]
        print(f"Backend: {storage.get('backend', 'unknown')}")
        print(f"Sandbox ID: {storage.get('sandbox_id', 'unknown')}")
        
    # Errors
    if info["errors"]:
        print("\n⚠️  Issues:")
        for error in info["errors"]:
            print(f"   • {error}")
    
    # Recommendations  
    if info["recommendations"]:
        print("\n💡 Recommendations:")
        for rec in info["recommendations"]:
            print(f"   • {rec}")
    
    if not info["errors"]:
        print("\n✅ Session configuration looks good!")
    
    print()  # Empty line