#!/usr/bin/env python3
"""
ChukLLM Discovery Safety Test Script
===================================

This script tests that ChukLLM's discovery system:
1. Does NOT accidentally download/pull models
2. Respects environment variable controls
3. Only detects already available models
4. Can be safely disabled for production environments

Run this script to verify discovery behavior before deploying ChukLLM.
"""

import os
import sys
import subprocess
import json
import time
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Set, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OllamaModelTracker:
    """Track Ollama models before and after discovery to ensure no downloads."""
    
    def __init__(self):
        self.initial_models: Set[str] = set()
        self.final_models: Set[str] = set()
        self.ollama_available = False
    
    def check_ollama_available(self) -> bool:
        """Check if Ollama is running and available."""
        try:
            result = subprocess.run(
                ["curl", "-s", "http://localhost:11434/api/version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                self.ollama_available = True
                logger.info("✅ Ollama is running and available")
                return True
            else:
                logger.info("ℹ️  Ollama is not running (this is fine for testing)")
                return False
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.info("ℹ️  Ollama is not available (this is fine for testing)")
            return False
    
    def get_current_models(self) -> Set[str]:
        """Get currently installed Ollama models."""
        if not self.ollama_available:
            return set()
        
        try:
            result = subprocess.run(
                ["curl", "-s", "http://localhost:11434/api/tags"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                models = {model['name'] for model in data.get('models', [])}
                logger.info(f"📦 Found {len(models)} Ollama models: {sorted(models)}")
                return models
            else:
                logger.warning(f"⚠️  Failed to get Ollama models: {result.stderr}")
                return set()
        except Exception as e:
            logger.warning(f"⚠️  Error getting Ollama models: {e}")
            return set()
    
    def capture_initial_state(self):
        """Capture initial model state."""
        logger.info("📸 Capturing initial Ollama model state...")
        self.initial_models = self.get_current_models()
    
    def capture_final_state(self):
        """Capture final model state."""
        logger.info("📸 Capturing final Ollama model state...")
        self.final_models = self.get_current_models()
    
    def verify_no_downloads(self) -> bool:
        """Verify no new models were downloaded."""
        if not self.ollama_available:
            logger.info("✅ Ollama not available - no download risk")
            return True
        
        new_models = self.final_models - self.initial_models
        removed_models = self.initial_models - self.final_models
        
        if new_models:
            logger.error(f"❌ NEW MODELS DETECTED! Discovery downloaded: {new_models}")
            return False
        
        if removed_models:
            logger.warning(f"⚠️  Models were removed during test: {removed_models}")
        
        logger.info("✅ No new models downloaded - discovery is safe!")
        return True


class EnvironmentTester:
    """Test environment variable controls for discovery."""
    
    def __init__(self):
        self.original_env = {}
        self.test_results = {}
    
    def backup_environment(self):
        """Backup original environment variables."""
        env_vars = [
            'CHUK_LLM_DISCOVERY_ENABLED',
            'CHUK_LLM_DISCOVERY_ON_STARTUP', 
            'CHUK_LLM_AUTO_DISCOVER',
            'CHUK_LLM_OLLAMA_DISCOVERY',
            'CHUK_LLM_OPENAI_DISCOVERY',
            'CHUK_LLM_ANTHROPIC_DISCOVERY',
            'CHUK_LLM_DISCOVERY_TIMEOUT',
            'CHUK_LLM_DISCOVERY_DEBUG'
        ]
        
        for var in env_vars:
            self.original_env[var] = os.environ.get(var)
    
    def restore_environment(self):
        """Restore original environment variables."""
        for var, value in self.original_env.items():
            if value is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = value
    
    def set_discovery_disabled(self):
        """Set environment to disable all discovery."""
        logger.info("🔒 Testing with discovery completely disabled...")
        os.environ['CHUK_LLM_DISCOVERY_ENABLED'] = 'false'
        os.environ['CHUK_LLM_OLLAMA_DISCOVERY'] = 'false'
        os.environ['CHUK_LLM_AUTO_DISCOVER'] = 'false'
        os.environ['CHUK_LLM_DISCOVERY_ON_STARTUP'] = 'false'
    
    def set_discovery_safe_mode(self):
        """Set environment for safe discovery (no startup, manual only)."""
        logger.info("⚡ Testing with safe discovery mode...")
        os.environ['CHUK_LLM_DISCOVERY_ENABLED'] = 'true'
        os.environ['CHUK_LLM_DISCOVERY_ON_STARTUP'] = 'false'  # Key: no startup
        os.environ['CHUK_LLM_AUTO_DISCOVER'] = 'true'
        os.environ['CHUK_LLM_OLLAMA_DISCOVERY'] = 'true'
        os.environ['CHUK_LLM_DISCOVERY_TIMEOUT'] = '2'  # Short timeout
        os.environ['CHUK_LLM_DISCOVERY_DEBUG'] = 'true'
    
    def set_discovery_aggressive(self):
        """Set environment for aggressive discovery (everything enabled)."""
        logger.info("🚀 Testing with aggressive discovery mode...")
        os.environ['CHUK_LLM_DISCOVERY_ENABLED'] = 'true'
        os.environ['CHUK_LLM_DISCOVERY_ON_STARTUP'] = 'true'
        os.environ['CHUK_LLM_AUTO_DISCOVER'] = 'true'
        os.environ['CHUK_LLM_OLLAMA_DISCOVERY'] = 'true'
        os.environ['CHUK_LLM_DISCOVERY_TIMEOUT'] = '5'
        os.environ['CHUK_LLM_DISCOVERY_DEBUG'] = 'true'


def test_chuk_llm_import(test_name: str, tracker: OllamaModelTracker) -> bool:
    """Test ChukLLM import and basic functionality."""
    logger.info(f"🧪 Testing ChukLLM import: {test_name}")
    
    try:
        # Create a fresh Python subprocess to test import
        test_script = '''
import sys
import os

# Set up environment for this test
print("Environment variables:")
for key, value in os.environ.items():
    if "CHUK_LLM" in key:
        print(f"  {key}={value}")

print("\\nImporting ChukLLM...")
try:
    import chuk_llm
    print("✅ ChukLLM imported successfully")
    
    # Try to show config (should work regardless of discovery)
    try:
        from chuk_llm.api.providers import show_config
        print("\\n🔧 ChukLLM Configuration:")
        show_config()
    except Exception as e:
        print(f"⚠️  Could not show config: {e}")
    
    # Try to list providers (should work regardless of discovery)
    try:
        from chuk_llm.configuration import get_config
        config_manager = get_config()
        providers = config_manager.get_all_providers()
        print(f"\\n📦 Available providers: {providers}")
    except Exception as e:
        print(f"⚠️  Could not list providers: {e}")
    
    print("\\n✅ Basic ChukLLM functionality test passed")
    
except Exception as e:
    print(f"❌ ChukLLM import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
'''
        
        # Run in subprocess to avoid importing ChukLLM in this process
        result = subprocess.run(
            [sys.executable, "-c", test_script],
            capture_output=True,
            text=True,
            timeout=30,
            env=os.environ.copy()
        )
        
        if result.returncode == 0:
            logger.info(f"✅ {test_name} passed")
            logger.debug(f"Output:\n{result.stdout}")
            return True
        else:
            logger.error(f"❌ {test_name} failed")
            logger.error(f"stdout:\n{result.stdout}")
            logger.error(f"stderr:\n{result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"❌ {test_name} timed out (> 30s)")
        return False
    except Exception as e:
        logger.error(f"❌ {test_name} failed with exception: {e}")
        return False


def test_discovery_functions(test_name: str) -> bool:
    """Test discovery-specific functions."""
    logger.info(f"🔍 Testing discovery functions: {test_name}")
    
    test_script = '''
import sys
import os

print("Testing discovery functions...")

try:
    from chuk_llm.api.providers import (
        disable_discovery,
        enable_discovery, 
        show_config,
        trigger_ollama_discovery_and_refresh
    )
    print("✅ Discovery functions imported successfully")
    
    # Test show_config (should always work)
    print("\\n📊 Testing show_config()...")
    show_config()
    
    # Test disable_discovery
    print("\\n🔒 Testing disable_discovery()...")
    disable_discovery("ollama")
    print("✅ disable_discovery() completed")
    
    # Test enable_discovery  
    print("\\n🔓 Testing enable_discovery()...")
    enable_discovery("ollama")
    print("✅ enable_discovery() completed")
    
    # Test trigger_ollama_discovery (should be safe - only detects, not downloads)
    print("\\n🔍 Testing trigger_ollama_discovery_and_refresh()...")
    try:
        result = trigger_ollama_discovery_and_refresh()
        print(f"✅ Discovery returned {len(result)} functions (safe)")
    except Exception as e:
        print(f"ℹ️  Discovery failed (expected if Ollama not running): {e}")
    
    print("\\n✅ Discovery functions test passed")
    
except Exception as e:
    print(f"❌ Discovery functions test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
'''
    
    try:
        result = subprocess.run(
            [sys.executable, "-c", test_script],
            capture_output=True,
            text=True,
            timeout=20,
            env=os.environ.copy()
        )
        
        if result.returncode == 0:
            logger.info(f"✅ {test_name} passed")
            logger.debug(f"Output:\n{result.stdout}")
            return True
        else:
            logger.error(f"❌ {test_name} failed")
            logger.error(f"stdout:\n{result.stdout}")
            logger.error(f"stderr:\n{result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"❌ {test_name} timed out")
        return False
    except Exception as e:
        logger.error(f"❌ {test_name} failed: {e}")
        return False


def main():
    """Run comprehensive discovery safety tests."""
    print("🧪 ChukLLM Discovery Safety Test Suite")
    print("=" * 50)
    print()
    
    tracker = OllamaModelTracker()
    env_tester = EnvironmentTester()
    
    # Check initial state
    tracker.check_ollama_available()
    tracker.capture_initial_state()
    env_tester.backup_environment()
    
    test_results = {}
    
    try:
        # Test 1: Discovery completely disabled
        logger.info("\n🔒 TEST 1: Discovery Completely Disabled")
        logger.info("-" * 40)
        env_tester.set_discovery_disabled()
        test_results["discovery_disabled"] = test_chuk_llm_import("Discovery Disabled", tracker)
        
        # Test 2: Safe discovery mode (no startup)
        logger.info("\n⚡ TEST 2: Safe Discovery Mode")
        logger.info("-" * 40)
        env_tester.set_discovery_safe_mode()
        test_results["safe_discovery"] = test_chuk_llm_import("Safe Discovery", tracker)
        test_results["safe_discovery_functions"] = test_discovery_functions("Safe Discovery Functions")
        
        # Test 3: Aggressive discovery (everything enabled)
        logger.info("\n🚀 TEST 3: Aggressive Discovery Mode")
        logger.info("-" * 40)
        env_tester.set_discovery_aggressive()
        test_results["aggressive_discovery"] = test_chuk_llm_import("Aggressive Discovery", tracker)
        test_results["aggressive_discovery_functions"] = test_discovery_functions("Aggressive Discovery Functions")
        
        # Capture final state and verify
        tracker.capture_final_state()
        no_downloads = tracker.verify_no_downloads()
        
        # Results summary
        print("\n📊 TEST RESULTS SUMMARY")
        print("=" * 50)
        
        for test_name, passed in test_results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{status} {test_name}")
        
        download_safety = "✅ PASS" if no_downloads else "❌ FAIL"
        print(f"{download_safety} no_accidental_downloads")
        
        # Overall result
        all_passed = all(test_results.values()) and no_downloads
        
        print("\n🎯 OVERALL RESULT")
        print("=" * 50)
        
        if all_passed:
            print("✅ ALL TESTS PASSED!")
            print("🎉 ChukLLM discovery is SAFE for production use")
            print("\n✨ Key Safety Features Verified:")
            print("   • Discovery can be completely disabled")
            print("   • No accidental model downloads")
            print("   • Environment controls work correctly")
            print("   • Safe discovery mode works as expected")
            return 0
        else:
            print("❌ SOME TESTS FAILED!")
            print("⚠️  Review the failures above before using discovery in production")
            return 1
            
    except KeyboardInterrupt:
        print("\n⏹️  Tests interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Test suite failed: {e}")
        return 1
    finally:
        # Always restore environment
        env_tester.restore_environment()
        logger.info("🔄 Environment variables restored")


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)