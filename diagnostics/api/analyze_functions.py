#!/usr/bin/env python3
"""
Analyze why we're generating so many functions
"""

def analyze_function_generation():
    """Break down the function generation to see where all these functions come from"""
    
    print("=== Function Generation Analysis ===\n")
    
    try:
        from chuk_llm.configuration import get_config
        from chuk_llm.api.providers import list_provider_functions
        
        config = get_config()
        all_functions = list_provider_functions()
        
        print(f"Total functions: {len(all_functions)}")
        print()
        
        # Break down by provider
        provider_breakdown = {}
        function_type_breakdown = {
            'ask_async': 0,
            'ask_sync': 0, 
            'stream_async': 0,
            'utility': 0,
            'global_alias': 0
        }
        
        for func in all_functions:
            # Determine function type
            if func.startswith('ask_') and func.endswith('_sync'):
                function_type_breakdown['ask_sync'] += 1
            elif func.startswith('ask_'):
                function_type_breakdown['ask_async'] += 1
            elif func.startswith('stream_'):
                function_type_breakdown['stream_async'] += 1
            elif func in ['quick_question', 'compare_providers', 'show_config']:
                function_type_breakdown['utility'] += 1
            else:
                function_type_breakdown['global_alias'] += 1
            
            # Determine provider (for non-utility functions)
            if '_' in func and not func in ['quick_question', 'compare_providers', 'show_config']:
                parts = func.split('_')
                if len(parts) >= 2:
                    provider = parts[1]
                    if provider not in provider_breakdown:
                        provider_breakdown[provider] = {
                            'total': 0,
                            'base': 0,  # ask_openai_sync
                            'models': 0,  # ask_openai_gpt4o_sync
                            'aliases': 0  # ask_openai_gpt4o_mini_sync
                        }
                    provider_breakdown[provider]['total'] += 1
                    
                    # Categorize further
                    remaining = '_'.join(parts[2:]) if len(parts) > 2 else ''
                    if not remaining or remaining in ['sync']:
                        provider_breakdown[provider]['base'] += 1
                    else:
                        # Check if it's a model or alias
                        try:
                            provider_config = config.get_provider(provider)
                            # This is a rough heuristic - could be either model or alias
                            provider_breakdown[provider]['models'] += 1
                        except:
                            provider_breakdown[provider]['models'] += 1
        
        print("Function Type Breakdown:")
        for func_type, count in function_type_breakdown.items():
            print(f"  {func_type}: {count}")
        print()
        
        print("Provider Breakdown:")
        for provider, breakdown in sorted(provider_breakdown.items()):
            print(f"  {provider}: {breakdown['total']} total")
            print(f"    Base functions: {breakdown['base']}")
            print(f"    Model/alias functions: {breakdown['models']}")
        print()
        
        # Show detailed breakdown for OpenAI
        print("OpenAI Functions Analysis:")
        openai_functions = [f for f in all_functions if f.startswith('ask_openai_') or f.startswith('stream_openai_')]
        
        # Group by model/alias
        openai_models = {}
        for func in openai_functions:
            parts = func.split('_')
            if len(parts) >= 3:
                model_part = '_'.join(parts[2:]).replace('_sync', '')
                if model_part not in openai_models:
                    openai_models[model_part] = []
                openai_models[model_part].append(func)
        
        print(f"  OpenAI generates functions for {len(openai_models)} different models/aliases:")
        for model, funcs in sorted(openai_models.items())[:10]:  # Show first 10
            print(f"    {model}: {len(funcs)} functions - {funcs}")
        if len(openai_models) > 10:
            print(f"    ... and {len(openai_models) - 10} more models")
        print()
        
        # Calculate what we'd expect
        providers = config.get_all_providers()
        expected_calculation = 0
        
        print("Expected Function Count Calculation:")
        for provider_name in providers:
            try:
                provider_config = config.get_provider(provider_name)
                
                # Base functions: ask_X, ask_X_sync, stream_X (3 per provider)
                base_funcs = 3
                
                # Model functions: ask_X_model, ask_X_model_sync, stream_X_model (3 per model)
                model_funcs = len(provider_config.models) * 3
                
                # Alias functions: ask_X_alias, ask_X_alias_sync, stream_X_alias (3 per alias)
                alias_funcs = len(provider_config.model_aliases) * 3
                
                provider_total = base_funcs + model_funcs + alias_funcs
                expected_calculation += provider_total
                
                print(f"  {provider_name}: {provider_total}")
                print(f"    Base: {base_funcs}, Models: {model_funcs} ({len(provider_config.models)} models), Aliases: {alias_funcs} ({len(provider_config.model_aliases)} aliases)")
                
            except Exception as e:
                print(f"  {provider_name}: Error - {e}")
        
        # Add global aliases
        global_aliases = config.get_global_aliases()
        global_alias_funcs = len(global_aliases) * 3  # ask_X, ask_X_sync, stream_X
        expected_calculation += global_alias_funcs
        
        # Add utility functions
        utility_funcs = 3
        expected_calculation += utility_funcs
        
        print(f"\nGlobal aliases: {global_alias_funcs} ({len(global_aliases)} aliases)")
        print(f"Utility functions: {utility_funcs}")
        print(f"Expected total: {expected_calculation}")
        print(f"Actual total: {len(all_functions)}")
        print(f"Difference: {len(all_functions) - expected_calculation}")
        
    except Exception as e:
        print(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_function_generation()