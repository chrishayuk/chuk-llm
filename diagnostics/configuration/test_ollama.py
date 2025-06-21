#!/usr/bin/env python3
"""
Simple test for dynamic discovery without hanging
"""

import sys
import os

def test_basic_discovery():
    """Test basic discovery functionality without async"""
    
    print("🔍 Basic Discovery Test")
    print("=" * 40)
    
    try:
        # Test 1: Import and check basic functionality
        print("📦 Step 1: Testing imports...")
        import chuk_llm
        print("   ✅ chuk_llm imported successfully")
        
        # Test 2: Check if providers module is accessible
        print("\n🔧 Step 2: Testing providers module...")
        try:
            from chuk_llm.api import providers
            print("   ✅ providers module imported")
            
            # Check if refresh function exists
            if hasattr(providers, 'refresh_provider_functions'):
                print("   ✅ refresh_provider_functions available")
            else:
                print("   ❌ refresh_provider_functions not found")
                
            if hasattr(providers, 'trigger_ollama_discovery_and_refresh'):
                print("   ✅ trigger_ollama_discovery_and_refresh available")
            else:
                print("   ❌ trigger_ollama_discovery_and_refresh not found")
                
        except Exception as e:
            print(f"   ❌ providers module error: {e}")
            return False
        
        # Test 3: Check current functions
        print("\n📋 Step 3: Checking current functions...")
        try:
            if hasattr(providers, 'list_provider_functions'):
                current_functions = providers.list_provider_functions()
                ollama_functions = [f for f in current_functions if 'ollama' in f]
                print(f"   Current Ollama functions: {len(ollama_functions)}")
                
                # Show some examples
                examples = [f for f in ollama_functions if f.startswith('ask_ollama_')][:5]
                for example in examples:
                    print(f"     • {example}")
                    
            else:
                print("   ❌ list_provider_functions not available")
                
        except Exception as e:
            print(f"   ❌ function listing error: {e}")
        
        # Test 4: Check if specific functions exist
        print("\n🎯 Step 4: Checking for specific functions...")
        
        test_functions = [
            'ask_ollama_devstral',
            'ask_ollama_qwen3_32b', 
            'ask_ollama_phi4_reasoning'
        ]
        
        found_functions = []
        
        for func_name in test_functions:
            # Check in main chuk_llm module
            if hasattr(chuk_llm, func_name):
                print(f"   ✅ {func_name} - found in chuk_llm")
                found_functions.append(func_name)
            # Check in providers module
            elif hasattr(providers, func_name):
                print(f"   🔍 {func_name} - found in providers")
                found_functions.append(func_name)
            else:
                print(f"   ❌ {func_name} - not found")
        
        return len(found_functions) > 0
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_manual_refresh():
    """Test manual refresh without hanging"""
    
    print("\n🔄 Manual Refresh Test")
    print("=" * 30)
    
    try:
        from chuk_llm.api import providers
        
        # Get initial count
        if hasattr(providers, 'list_provider_functions'):
            initial_functions = providers.list_provider_functions()
            initial_count = len([f for f in initial_functions if 'ollama' in f])
            print(f"   Initial Ollama functions: {initial_count}")
        else:
            print("   ❌ Cannot get initial function count")
            return False
        
        # Try to trigger refresh (with timeout protection)
        print("   Attempting manual refresh...")
        
        try:
            import signal
            import time
            
            def timeout_handler(signum, frame):
                raise TimeoutError("Refresh took too long")
            
            # Set timeout for 10 seconds
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(10)
            
            try:
                # Try the refresh
                if hasattr(providers, 'refresh_provider_functions'):
                    result = providers.refresh_provider_functions('ollama')
                    print(f"   ✅ Refresh completed: {len(result) if result else 0} new functions")
                else:
                    print("   ❌ refresh_provider_functions not available")
                
            finally:
                signal.alarm(0)  # Cancel timeout
                
        except TimeoutError:
            print("   ⚠️  Refresh timed out (likely discovery issue)")
            return False
        except Exception as e:
            print(f"   ⚠️  Refresh failed: {e}")
            return False
        
        # Check new count
        if hasattr(providers, 'list_provider_functions'):
            new_functions = providers.list_provider_functions()
            new_count = len([f for f in new_functions if 'ollama' in f])
            print(f"   Final Ollama functions: {new_count}")
            
            if new_count > initial_count:
                print(f"   🎉 Added {new_count - initial_count} new functions!")
                return True
            else:
                print("   📝 No new functions added (may be normal)")
                return True
        
    except Exception as e:
        print(f"   ❌ Manual refresh test failed: {e}")
        return False


def test_getattr_access():
    """Test __getattr__ access without hanging"""
    
    print("\n🔧 Testing Dynamic Access")
    print("=" * 30)
    
    try:
        from chuk_llm.api import providers
        
        # Test accessing a function that might trigger discovery
        test_function = 'ask_ollama_devstral'
        
        print(f"   Trying to access {test_function}...")
        
        try:
            # Use timeout here too
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("Access took too long")
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(5)  # 5 second timeout
            
            try:
                func = getattr(providers, test_function)
                print(f"   ✅ {test_function} accessed successfully")
                
                # Check if it's callable
                if callable(func):
                    print(f"   ✅ Function is callable")
                    
                    # Check signature without calling
                    import inspect
                    try:
                        sig = inspect.signature(func)
                        print(f"   📋 Signature: {sig}")
                    except:
                        print("   📋 Signature: <unavailable>")
                        
                    return True
                else:
                    print(f"   ❌ Not callable")
                    return False
                    
            finally:
                signal.alarm(0)
                
        except AttributeError:
            print(f"   ❌ {test_function} not found via __getattr__")
            return False
        except TimeoutError:
            print(f"   ⚠️  Access timed out (likely discovery hanging)")
            return False
        except Exception as e:
            print(f"   ❌ Access error: {e}")
            return False
            
    except Exception as e:
        print(f"   ❌ Dynamic access test failed: {e}")
        return False


def test_config_discovery_status():
    """Test if discovery is properly configured"""
    
    print("\n⚙️  Testing Discovery Configuration")
    print("=" * 40)
    
    try:
        from chuk_llm.configuration import get_config
        
        config = get_config()
        
        # Check Ollama provider
        try:
            ollama_provider = config.get_provider('ollama')
            print("   ✅ Ollama provider found")
            
            # Check discovery configuration
            discovery_config = ollama_provider.extra.get('dynamic_discovery')
            if discovery_config:
                enabled = discovery_config.get('enabled', False)
                discoverer_type = discovery_config.get('discoverer_type', 'unknown')
                
                print(f"   Discovery enabled: {enabled}")
                print(f"   Discoverer type: {discoverer_type}")
                
                if enabled:
                    print("   ✅ Discovery is properly configured")
                    return True
                else:
                    print("   ⚠️  Discovery is disabled")
                    return False
            else:
                print("   ❌ No discovery configuration found")
                return False
                
        except Exception as e:
            print(f"   ❌ Error checking Ollama provider: {e}")
            return False
            
    except Exception as e:
        print(f"   ❌ Config test failed: {e}")
        return False


def main():
    """Main test function"""
    
    print("🧪 ChukLLM Dynamic Discovery Test")
    print("=" * 50)
    print("This test checks dynamic function generation without hanging")
    print()
    
    results = []
    
    # Run tests
    results.append(("Basic Discovery", test_basic_discovery()))
    results.append(("Config Check", test_config_discovery_status()))
    results.append(("Manual Refresh", test_manual_refresh()))
    results.append(("Dynamic Access", test_getattr_access()))
    
    # Show results
    print("\n📊 Test Results")
    print("=" * 30)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name:<20} {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Summary: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! Dynamic discovery should be working.")
        print("\n💡 Try this:")
        print("   import chuk_llm")
        print("   response = await chuk_llm.ask_ollama_devstral('Hello!')")
    elif passed > 0:
        print("⚠️  Some tests passed. Discovery may work partially.")
    else:
        print("❌ No tests passed. Discovery needs debugging.")
    
    return passed == len(results)


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)