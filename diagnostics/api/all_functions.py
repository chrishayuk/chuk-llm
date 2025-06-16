#!/usr/bin/env python3
"""
Script to list ALL available ChukLLM functions in detail
"""

def list_all_functions():
    """List every single available function with categorization"""
    
    print("📋 ChukLLM - Complete Function Inventory")
    print("=" * 60)
    
    try:
        from chuk_llm.api.providers import list_provider_functions
        from chuk_llm.configuration import get_config
        
        all_functions = list_provider_functions()
        config = get_config()
        providers = config.get_all_providers()
        
        print(f"Total Functions: {len(all_functions)}")
        print(f"Providers: {len(providers)}")
        print()
        
        # Group functions by category
        sync_functions = [f for f in all_functions if f.endswith('_sync')]
        async_functions = [f for f in all_functions if f.startswith('ask_') and not f.endswith('_sync')]
        stream_functions = [f for f in all_functions if f.startswith('stream_')]
        utility_functions = [f for f in all_functions if f in ['quick_question', 'compare_providers', 'show_config']]
        
        print("🎯 SUMMARY BY TYPE")
        print("-" * 30)
        print(f"Sync Functions:    {len(sync_functions)}")
        print(f"Async Functions:   {len(async_functions)}")
        print(f"Stream Functions:  {len(stream_functions)}")
        print(f"Utility Functions: {len(utility_functions)}")
        print()
        
        # 1. UTILITY FUNCTIONS
        print("🔧 UTILITY FUNCTIONS")
        print("-" * 30)
        for func in sorted(utility_functions):
            print(f"   {func}")
        print()
        
        # 2. GLOBAL ALIAS FUNCTIONS (sync only)
        global_aliases = []
        for func in sync_functions:
            # Check if it's a global alias (doesn't start with ask_<provider>_)
            is_provider_function = any(func.startswith(f'ask_{provider}_') for provider in providers)
            if not is_provider_function and func != 'ask_sync':
                global_aliases.append(func)
        
        print(f"🌍 GLOBAL ALIAS FUNCTIONS ({len(global_aliases)})")
        print("-" * 30)
        for func in sorted(global_aliases):
            print(f"   {func}")
        print()
        
        # 3. PROVIDER-SPECIFIC FUNCTIONS
        print("🏢 PROVIDER-SPECIFIC FUNCTIONS")
        print("-" * 30)
        
        for provider in sorted(providers):
            # Get all functions for this provider
            provider_sync = [f for f in sync_functions if f.startswith(f'ask_{provider}_')]
            provider_async = [f for f in async_functions if f.startswith(f'ask_{provider}_')]
            provider_stream = [f for f in stream_functions if f.startswith(f'stream_{provider}_')]
            
            total_provider_funcs = len(provider_sync) + len(provider_async) + len(provider_stream)
            
            print(f"\n📦 {provider.upper()} ({total_provider_funcs} functions)")
            print(f"    Sync: {len(provider_sync)}, Async: {len(provider_async)}, Stream: {len(provider_stream)}")
            
            # Show base function
            base_sync = f"ask_{provider}_sync"
            if base_sync in provider_sync:
                print(f"    Base: {base_sync}")
            
            # Show model/alias functions (just the first few)
            other_sync = [f for f in provider_sync if f != base_sync]
            if other_sync:
                print(f"    Model/Alias functions ({len(other_sync)}):")
                for func in sorted(other_sync)[:5]:
                    print(f"      {func}")
                if len(other_sync) > 5:
                    print(f"      ... and {len(other_sync) - 5} more")
        
        print("\n" + "=" * 60)
        print("📊 COMPLETE FUNCTION BREAKDOWN")
        print("=" * 60)
        
        # Show ALL functions by type
        print(f"\n🔄 ALL SYNC FUNCTIONS ({len(sync_functions)}):")
        for i, func in enumerate(sorted(sync_functions), 1):
            print(f"{i:3d}. {func}")
        
        print(f"\n⚡ ALL ASYNC FUNCTIONS ({len(async_functions)}):")
        for i, func in enumerate(sorted(async_functions), 1):
            print(f"{i:3d}. {func}")
        
        print(f"\n🌊 ALL STREAM FUNCTIONS ({len(stream_functions)}):")
        for i, func in enumerate(sorted(stream_functions), 1):
            print(f"{i:3d}. {func}")
            
    except Exception as e:
        print(f"❌ Error listing functions: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    list_all_functions()