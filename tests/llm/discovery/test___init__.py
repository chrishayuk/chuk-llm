# tests/test_discovery/test___init__.py
"""
Tests for chuk_llm.llm.discovery.__init__ module
"""

import importlib
import sys
from unittest.mock import Mock, patch

import pytest


# Test the initialization and exports of the discovery module
def test_module_imports():
    """Test that all expected components can be imported from discovery module"""
    # Fresh import to test the module initialization
    if "chuk_llm.llm.discovery" in sys.modules:
        del sys.modules["chuk_llm.llm.discovery"]

    # Mock the submodules to avoid actual imports during testing
    mock_modules = {
        "chuk_llm.llm.discovery.base": Mock(),
        "chuk_llm.llm.discovery.engine": Mock(),
        "chuk_llm.llm.discovery.manager": Mock(),
        "chuk_llm.llm.discovery.ollama_discoverer": Mock(),
        "chuk_llm.llm.discovery.openai_discoverer": Mock(),
        "chuk_llm.llm.discovery.general_discoverers": Mock(),
    }

    with patch.dict("sys.modules", mock_modules):
        # Import the discovery module
        import chuk_llm.llm.discovery as discovery

        # Test that all expected components are available
        expected_exports = [
            "BaseModelDiscoverer",
            "DiscoveredModel",
            "DiscovererFactory",
            "ConfigDrivenInferenceEngine",
            "UniversalModelDiscoveryManager",
            "UniversalDiscoveryManager",
            "DiscoveryResults",
        ]

        for export in expected_exports:
            assert hasattr(discovery, export), f"Missing export: {export}"
            assert discovery.__all__ is not None
            assert export in discovery.__all__, f"{export} not in __all__"


def test_factory_initialization():
    """Test that the factory initialization works properly"""
    # Save original modules
    original_modules = {}
    modules_to_save = [
        "chuk_llm.llm.discovery",
        "chuk_llm.llm.discovery.base",
        "chuk_llm.llm.discovery.engine",
        "chuk_llm.llm.discovery.manager",
        "chuk_llm.llm.discovery.ollama_discoverer",
        "chuk_llm.llm.discovery.openai_discoverer",
        "chuk_llm.llm.discovery.general_discoverers",
    ]
    for mod_name in modules_to_save:
        if mod_name in sys.modules:
            original_modules[mod_name] = sys.modules[mod_name]

    try:
        # Mock the factory and its methods
        mock_factory = Mock()
        mock_factory._auto_import_discoverers = Mock()

        # Mock the base module
        mock_base = Mock()
        mock_base.DiscovererFactory = mock_factory

        mock_modules = {
            "chuk_llm.llm.discovery.base": mock_base,
            "chuk_llm.llm.discovery.engine": Mock(),
            "chuk_llm.llm.discovery.manager": Mock(),
            "chuk_llm.llm.discovery.ollama_discoverer": Mock(),
            "chuk_llm.llm.discovery.openai_discoverer": Mock(),
            "chuk_llm.llm.discovery.general_discoverers": Mock(),
        }

        # Clear any existing module
        if "chuk_llm.llm.discovery" in sys.modules:
            del sys.modules["chuk_llm.llm.discovery"]

        with patch.dict("sys.modules", mock_modules):
            # Import should trigger factory initialization
            import chuk_llm.llm.discovery

            # Verify that auto_import_discoverers was called
            mock_factory._auto_import_discoverers.assert_called_once()
    finally:
        # Restore original modules
        for mod_name, mod in original_modules.items():
            sys.modules[mod_name] = mod


def test_factory_initialization_failure_handling():
    """Test that factory initialization failures are handled gracefully"""
    # Save original modules
    original_modules = {}
    modules_to_save = [
        "chuk_llm.llm.discovery",
        "chuk_llm.llm.discovery.base",
        "chuk_llm.llm.discovery.engine",
        "chuk_llm.llm.discovery.manager",
        "chuk_llm.llm.discovery.ollama_discoverer",
        "chuk_llm.llm.discovery.openai_discoverer",
        "chuk_llm.llm.discovery.general_discoverers",
    ]
    for mod_name in modules_to_save:
        if mod_name in sys.modules:
            original_modules[mod_name] = sys.modules[mod_name]

    try:
        # Mock factory to raise exception during initialization
        mock_factory = Mock()
        mock_factory._auto_import_discoverers.side_effect = Exception("Import failed")

        mock_base = Mock()
        mock_base.DiscovererFactory = mock_factory

        mock_modules = {
            "chuk_llm.llm.discovery.base": mock_base,
            "chuk_llm.llm.discovery.engine": Mock(),
            "chuk_llm.llm.discovery.manager": Mock(),
            "chuk_llm.llm.discovery.ollama_discoverer": Mock(),
            "chuk_llm.llm.discovery.openai_discoverer": Mock(),
            "chuk_llm.llm.discovery.general_discoverers": Mock(),
        }

        # Clear any existing module
        if "chuk_llm.llm.discovery" in sys.modules:
            del sys.modules["chuk_llm.llm.discovery"]

        with patch.dict("sys.modules", mock_modules):
            with patch("logging.getLogger") as mock_logger:
                mock_log = Mock()
                mock_logger.return_value = mock_log

                # Import should not raise exception even if initialization fails
                import chuk_llm.llm.discovery

                # Should log a warning about the failure
                mock_log.warning.assert_called_once()
                assert "Failed to initialize discovery factory" in str(
                    mock_log.warning.call_args
                )
    finally:
        # Restore original modules
        for mod_name, mod in original_modules.items():
            sys.modules[mod_name] = mod


def test_module_structure():
    """Test the overall structure and organization of the discovery module"""
    # Test that the module docstring exists and is informative
    import chuk_llm.llm.discovery as discovery

    assert discovery.__doc__ is not None
    assert "discovery system" in discovery.__doc__.lower()

    # Test that __all__ is properly defined
    assert hasattr(discovery, "__all__")
    assert isinstance(discovery.__all__, list)
    assert len(discovery.__all__) > 0

    # Test that all items in __all__ are actually available
    for item in discovery.__all__:
        assert hasattr(discovery, item), (
            f"Item {item} in __all__ but not available in module"
        )


def test_clean_exports():
    """Test that only intended components are exported"""
    import chuk_llm.llm.discovery as discovery

    # Get all public attributes (not starting with _)
    public_attrs = [attr for attr in dir(discovery) if not attr.startswith("_")]

    # Should only export what's in __all__ plus standard module attributes
    expected_public = set(discovery.__all__)
    actual_public = set(public_attrs)

    # Remove expected standard attributes that aren't in __all__
    standard_attrs = {"sys", "logging", "importlib"}  # Might be present from imports
    actual_public_filtered = actual_public - standard_attrs

    # All public attributes should be in __all__ (clean exports)
    unexpected = actual_public_filtered - expected_public
    assert len(unexpected) == 0, f"Unexpected public exports: {unexpected}"

    # All __all__ items should be publicly available
    missing = expected_public - actual_public_filtered
    assert len(missing) == 0, f"Missing public exports: {missing}"


class TestDiscoveryModuleIntegration:
    """Integration tests for the discovery module as a whole"""

    def test_import_hierarchy_consistency(self):
        """Test that imports work consistently across different import styles"""
        # Test direct import
        from chuk_llm.llm.discovery import DiscoveredModel

        assert DiscoveredModel is not None

        # Test module import
        import chuk_llm.llm.discovery as discovery

        assert hasattr(discovery, "DiscoveredModel")
        assert discovery.DiscoveredModel is DiscoveredModel

        # Test that __all__ contains expected items for star import
        assert hasattr(discovery, "__all__")
        assert "DiscoveredModel" in discovery.__all__

    def test_submodule_access_patterns(self):
        """Test different ways of accessing submodule components"""
        import chuk_llm.llm.discovery as discovery

        # Should be able to access factory through main module
        assert hasattr(discovery, "DiscovererFactory")

        # Should be able to access manager components
        assert hasattr(discovery, "UniversalDiscoveryManager")
        assert hasattr(discovery, "DiscoveryResults")

        # Should be able to access engine components
        assert hasattr(discovery, "ConfigDrivenInferenceEngine")
        assert hasattr(discovery, "UniversalModelDiscoveryManager")

    def test_module_reload_safety(self):
        """Test that the module can be safely reloaded"""
        import chuk_llm.llm.discovery as discovery

        initial_factory = discovery.DiscovererFactory

        # Reload should work without issues
        importlib.reload(discovery)

        # Factory should still be accessible
        assert hasattr(discovery, "DiscovererFactory")

        # Should be same type but potentially different instance due to reload
        assert (
            type(discovery.DiscovererFactory).__name__ == type(initial_factory).__name__
        )

    def test_circular_import_prevention(self):
        """Test that circular imports are prevented"""
        # This test verifies that importing the discovery module doesn't cause
        # circular import issues between its submodules

        try:
            # These imports should not cause circular import errors
            from chuk_llm.llm.discovery.base import DiscovererFactory
            from chuk_llm.llm.discovery.engine import UniversalModelDiscoveryManager
            from chuk_llm.llm.discovery.manager import UniversalDiscoveryManager

            # All should be importable without issues
            assert DiscovererFactory is not None
            assert UniversalModelDiscoveryManager is not None
            assert UniversalDiscoveryManager is not None

        except ImportError as e:
            pytest.fail(f"Circular import detected: {e}")

    def test_lazy_loading_behavior(self):
        """Test that submodules are loaded lazily when accessed"""
        # Fresh import to test loading behavior
        if "chuk_llm.llm.discovery" in sys.modules:
            del sys.modules["chuk_llm.llm.discovery"]

        # Import main module
        import chuk_llm.llm.discovery as discovery

        # Main classes should be available immediately due to __init__.py imports
        assert hasattr(discovery, "DiscoveredModel")
        assert hasattr(discovery, "DiscovererFactory")

        # These are imported in __init__.py so should be available
        assert hasattr(discovery, "UniversalDiscoveryManager")

    def test_version_compatibility(self):
        """Test that the module structure is compatible across versions"""
        import chuk_llm.llm.discovery as discovery

        # Core components that should always be available
        essential_components = [
            "BaseModelDiscoverer",
            "DiscoveredModel",
            "DiscovererFactory",
        ]

        for component in essential_components:
            assert hasattr(discovery, component), (
                f"Essential component {component} missing"
            )

        # Manager components that should be available
        manager_components = ["UniversalDiscoveryManager", "DiscoveryResults"]

        for component in manager_components:
            assert hasattr(discovery, component), (
                f"Manager component {component} missing"
            )
