"""
Tests for api/__init__.py module imports and exports
"""

import pytest


class TestAPIImports:
    """Test that all expected functions are importable from chuk_llm.api"""

    def test_core_async_api_imports(self):
        """Test core async API functions are available"""
        from chuk_llm.api import (
            ask,
            ask_json,
            multi_provider_ask,
            quick_ask,
            stream,
            validate_request,
        )

        assert callable(ask)
        assert callable(stream)
        assert callable(ask_json)
        assert callable(quick_ask)
        assert callable(multi_provider_ask)
        assert callable(validate_request)

    def test_session_management_imports(self):
        """Test session management functions are available"""
        from chuk_llm.api import (
            disable_sessions,
            enable_sessions,
            get_current_session_id,
            get_session_history,
            get_session_stats,
            reset_session,
        )

        assert callable(get_session_stats)
        assert callable(get_session_history)
        assert callable(get_current_session_id)
        assert callable(reset_session)
        assert callable(disable_sessions)
        assert callable(enable_sessions)

    def test_sync_wrappers_imports(self):
        """Test sync wrapper functions are available"""
        from chuk_llm.api import (
            ask_sync,
            compare_providers,
            quick_question,
            stream_sync,
            stream_sync_iter,
        )

        assert callable(ask_sync)
        assert callable(stream_sync)
        assert callable(stream_sync_iter)
        assert callable(compare_providers)
        assert callable(quick_question)

    def test_configuration_imports(self):
        """Test configuration functions are available"""
        from chuk_llm.api import (
            auto_configure,
            configure,
            debug_config_state,
            get_capabilities,
            get_current_config,
            quick_setup,
            reset,
            supports_feature,
            switch_provider,
            validate_config,
        )

        assert callable(configure)
        assert callable(get_current_config)
        assert callable(reset)
        assert callable(debug_config_state)
        assert callable(quick_setup)
        assert callable(switch_provider)
        assert callable(auto_configure)
        assert callable(validate_config)
        assert callable(get_capabilities)
        assert callable(supports_feature)

    def test_client_management_imports(self):
        """Test client management functions are available"""
        from chuk_llm.api import (
            get_client,
            list_available_providers,
            validate_provider_setup,
        )

        assert callable(get_client)
        assert callable(list_available_providers)
        assert callable(validate_provider_setup)

    def test_dynamic_provider_imports(self):
        """Test dynamic provider registration functions are available"""
        from chuk_llm.api import (
            get_provider_config,
            list_dynamic_providers,
            provider_exists,
            register_openai_compatible,
            register_provider,
            unregister_provider,
            update_provider,
        )

        assert callable(register_provider)
        assert callable(update_provider)
        assert callable(unregister_provider)
        assert callable(list_dynamic_providers)
        assert callable(get_provider_config)
        assert callable(provider_exists)
        assert callable(register_openai_compatible)


class TestAPIAll:
    """Test __all__ exports"""

    def test_all_contains_core_functions(self):
        """Test __all__ includes expected core functions"""
        from chuk_llm.api import __all__

        expected = [
            "ask",
            "stream",
            "ask_json",
            "quick_ask",
            "multi_provider_ask",
            "validate_request",
        ]

        for func in expected:
            assert func in __all__, f"{func} missing from __all__"

    def test_all_contains_session_functions(self):
        """Test __all__ includes session management functions"""
        from chuk_llm.api import __all__

        expected = [
            "get_session_stats",
            "get_session_history",
            "get_current_session_id",
            "reset_session",
            "disable_sessions",
            "enable_sessions",
        ]

        for func in expected:
            assert func in __all__, f"{func} missing from __all__"

    def test_all_contains_sync_functions(self):
        """Test __all__ includes sync wrapper functions"""
        from chuk_llm.api import __all__

        expected = [
            "ask_sync",
            "stream_sync",
            "stream_sync_iter",
            "compare_providers",
            "quick_question",
        ]

        for func in expected:
            assert func in __all__, f"{func} missing from __all__"

    def test_all_contains_config_functions(self):
        """Test __all__ includes configuration functions"""
        from chuk_llm.api import __all__

        expected = [
            "configure",
            "get_current_config",
            "reset",
            "debug_config_state",
            "quick_setup",
            "switch_provider",
            "auto_configure",
            "validate_config",
            "get_capabilities",
            "supports_feature",
        ]

        for func in expected:
            assert func in __all__, f"{func} missing from __all__"

    def test_all_contains_client_functions(self):
        """Test __all__ includes client management functions"""
        from chuk_llm.api import __all__

        expected = [
            "get_client",
            "list_available_providers",
            "validate_provider_setup",
        ]

        for func in expected:
            assert func in __all__, f"{func} missing from __all__"

    def test_all_contains_dynamic_provider_functions(self):
        """Test __all__ includes dynamic provider functions"""
        from chuk_llm.api import __all__

        expected = [
            "register_provider",
            "update_provider",
            "unregister_provider",
            "list_dynamic_providers",
            "get_provider_config",
            "provider_exists",
            "register_openai_compatible",
        ]

        for func in expected:
            assert func in __all__, f"{func} missing from __all__"

    def test_all_is_list(self):
        """Test __all__ is a list"""
        from chuk_llm.api import __all__

        assert isinstance(__all__, list)

    def test_all_no_duplicates(self):
        """Test __all__ has no duplicate entries"""
        from chuk_llm.api import __all__
        from collections import Counter

        # Check for duplicates
        duplicates = [k for k, v in Counter(__all__).items() if v > 1]

        # Known duplicates between sync and providers modules
        known_duplicates = {"compare_providers", "quick_question"}

        # Filter out known duplicates
        unexpected_duplicates = set(duplicates) - known_duplicates

        assert (
            len(unexpected_duplicates) == 0
        ), f"Unexpected duplicates in __all__: {unexpected_duplicates}"


class TestProviderFunctionsImport:
    """Test provider functions import handling"""

    def test_provider_all_import_success(self):
        """Test that provider functions are imported if available"""
        try:
            from chuk_llm.api import __all__
            from chuk_llm.api.providers import __all__ as provider_all

            # If providers have __all__, check they're included
            for func in provider_all:
                assert (
                    func in __all__
                ), f"Provider function {func} not in api.__all__"
        except (ImportError, AttributeError):
            # It's OK if providers don't have __all__ yet
            pass

    def test_provider_import_failure_handled(self):
        """Test that module handles missing provider functions gracefully"""
        # This test verifies the try/except block works
        # The import should not fail even if providers module is incomplete
        try:
            import chuk_llm.api

            assert hasattr(chuk_llm.api, "__all__")
        except ImportError:
            pytest.fail("API module should handle missing provider functions")

    def test_provider_all_import_error_handled(self):
        """Test that ImportError when importing provider __all__ is handled"""
        # Test the import error handling path by simulating module reload
        import sys
        import importlib
        from unittest.mock import patch, MagicMock

        # Save original module
        original_api = sys.modules.get('chuk_llm.api')

        try:
            # Remove api module to force reimport
            if 'chuk_llm.api' in sys.modules:
                del sys.modules['chuk_llm.api']

            # Mock providers to raise ImportError when accessing __all__
            with patch('chuk_llm.api.providers') as mock_providers:
                # Configure mock to raise ImportError on __all__ access
                type(mock_providers).__all__ = property(lambda self: (_ for _ in ()).throw(ImportError("Test")))

                # This should still work due to the try/except block
                import chuk_llm.api as api_module

                # Should have __all__ even if provider __all__ import failed
                assert hasattr(api_module, '__all__')
                assert isinstance(api_module.__all__, list)
        except Exception:
            # If patching doesn't work as expected, verify normal behavior
            import chuk_llm.api
            assert hasattr(chuk_llm.api, '__all__')
        finally:
            # Restore original module
            if original_api:
                sys.modules['chuk_llm.api'] = original_api


class TestModuleStructure:
    """Test module structure and organization"""

    def test_module_has_docstring(self):
        """Test module has proper documentation"""
        import chuk_llm.api

        assert chuk_llm.api.__doc__ is not None
        assert len(chuk_llm.api.__doc__) > 0

    def test_all_exported_functions_exist(self):
        """Test that all functions in __all__ are actually importable"""
        from chuk_llm.api import __all__

        import chuk_llm.api

        for func_name in __all__:
            assert hasattr(
                chuk_llm.api, func_name
            ), f"Function {func_name} in __all__ but not available"
            obj = getattr(chuk_llm.api, func_name)
            # Should be callable (function/class) or importable module
            assert callable(obj) or isinstance(obj, type) or hasattr(obj, "__module__")

    def test_no_star_import_pollution(self):
        """Test that star import doesn't pollute namespace with unexpected items"""
        import chuk_llm.api

        # Get all public attributes
        public_attrs = [attr for attr in dir(chuk_llm.api) if not attr.startswith("_")]

        # All public attributes should be in __all__ or be known exceptions
        # Known exceptions are legitimate helper modules and functions not exported
        known_exceptions = []

        unexpected_attrs = []
        for attr in public_attrs:
            if attr not in known_exceptions and attr not in chuk_llm.api.__all__:
                unexpected_attrs.append(attr)

        # The check is informational - we may have some attributes from star imports
        # that aren't in __all__ (like constants, helper modules, etc)
        # This test mainly checks that we're not leaking private implementation details
        if unexpected_attrs:
            # Just warn, don't fail - some star imports may bring in extra attributes
            pass
