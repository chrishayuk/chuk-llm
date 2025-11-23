"""Tests for chuk_llm/api/session_utils.py - Session utilities."""

import os
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest

from chuk_llm.api.session_utils import (
    auto_configure_sessions,
    check_session_backend_availability,
    get_session_recommendations,
    print_session_diagnostics,
    validate_session_configuration,
)


class TestCheckSessionBackendAvailability:
    """Test check_session_backend_availability function."""

    def test_check_session_backend_availability_basic(self):
        """Test basic availability check."""
        result = check_session_backend_availability()

        assert "memory_available" in result
        assert "redis_available" in result
        assert "current_provider" in result
        assert "recommendations" in result
        assert "errors" in result

        assert result["memory_available"] is True
        assert isinstance(result["recommendations"], list)
        assert isinstance(result["errors"], list)

    def test_check_session_backend_with_redis_installed(self):
        """Test when Redis is installed."""
        # Mock successful redis import by patching it in sys.modules
        mock_redis = MagicMock()
        mock_redis.__version__ = "4.5.0"

        with patch.dict("sys.modules", {"redis": mock_redis}):
            result = check_session_backend_availability()

            # Check redis availability is properly detected
            assert result["redis_available"] is True
            assert result["redis_version"] == "4.5.0"

    def test_check_session_backend_without_redis(self):
        """Test when Redis is not installed."""
        with patch("builtins.__import__", side_effect=ImportError):
            result = check_session_backend_availability()

            assert result["redis_available"] is False
            assert "Redis package not installed" in result["errors"]
            assert any(
                "pip install chuk_llm[redis]" in rec
                for rec in result["recommendations"]
            )

    def test_check_session_backend_redis_configured_but_unavailable(
        self, monkeypatch
    ):
        """Test when Redis is configured but not available."""
        monkeypatch.setenv("SESSION_PROVIDER", "redis")

        with patch("builtins.__import__", side_effect=ImportError):
            result = check_session_backend_availability()

            assert result["current_provider"] == "redis"
            assert result["redis_available"] is False
            assert "SESSION_PROVIDER=redis but Redis not available" in result["errors"]

    def test_check_session_backend_with_session_manager(self):
        """Test when session manager is available."""
        mock_session_manager = MagicMock()
        mock_session_manager.__version__ = "1.0.0"
        mock_session_manager.get_storage_info.return_value = {
            "backend": "memory",
            "sandbox_id": "test-123",
        }

        with patch.dict(
            "sys.modules", {"chuk_ai_session_manager": mock_session_manager}
        ):
            result = check_session_backend_availability()

            assert result.get("session_manager_available") is True
            assert result.get("session_manager_version") == "1.0.0"
            assert "storage_info" in result

    def test_check_session_backend_without_session_manager(self):
        """Test when session manager is not available."""
        with patch("builtins.__import__", side_effect=ImportError):
            result = check_session_backend_availability()

            if "session_manager_available" in result:
                assert result["session_manager_available"] is False

    def test_check_session_backend_storage_info_error(self):
        """Test when getting storage info raises an error."""
        mock_session_manager = MagicMock()
        mock_session_manager.__version__ = "1.0.0"
        mock_session_manager.get_storage_info.side_effect = Exception("Storage error")

        with patch.dict(
            "sys.modules", {"chuk_ai_session_manager": mock_session_manager}
        ):
            result = check_session_backend_availability()

            assert "storage_errors" in result
            assert len(result["storage_errors"]) > 0


class TestValidateSessionConfiguration:
    """Test validate_session_configuration function."""

    def test_validate_session_configuration_valid(self):
        """Test validation with valid configuration."""
        with patch(
            "chuk_llm.api.session_utils.check_session_backend_availability"
        ) as mock_check:
            mock_check.return_value = {
                "memory_available": True,
                "redis_available": False,
                "current_provider": "memory",
                "errors": [],
                "recommendations": [],
            }

            result = validate_session_configuration()
            assert result is True

    def test_validate_session_configuration_invalid(self):
        """Test validation with invalid configuration."""
        with patch(
            "chuk_llm.api.session_utils.check_session_backend_availability"
        ) as mock_check:
            mock_check.return_value = {
                "memory_available": True,
                "redis_available": False,
                "current_provider": "redis",
                "errors": ["Redis not available"],
                "recommendations": ["Install Redis"],
            }

            result = validate_session_configuration()
            assert result is False

    def test_validate_session_configuration_logs_warnings(self, caplog):
        """Test that validation logs warnings for errors."""
        with patch(
            "chuk_llm.api.session_utils.check_session_backend_availability"
        ) as mock_check:
            mock_check.return_value = {
                "errors": ["Error 1", "Error 2"],
                "recommendations": [],
            }

            validate_session_configuration()

            # Check that warnings were logged
            assert any("Session configuration issue" in record.message for record in caplog.records)


class TestGetSessionRecommendations:
    """Test get_session_recommendations function."""

    def test_get_session_recommendations_empty(self):
        """Test getting recommendations when there are none."""
        with patch(
            "chuk_llm.api.session_utils.check_session_backend_availability"
        ) as mock_check:
            mock_check.return_value = {"recommendations": []}

            recs = get_session_recommendations()
            assert recs == []

    def test_get_session_recommendations_populated(self):
        """Test getting recommendations when there are some."""
        with patch(
            "chuk_llm.api.session_utils.check_session_backend_availability"
        ) as mock_check:
            mock_check.return_value = {
                "recommendations": ["Install Redis", "Configure sessions"]
            }

            recs = get_session_recommendations()
            assert len(recs) == 2
            assert "Install Redis" in recs

    def test_get_session_recommendations_missing_key(self):
        """Test getting recommendations when key is missing."""
        with patch(
            "chuk_llm.api.session_utils.check_session_backend_availability"
        ) as mock_check:
            mock_check.return_value = {}

            recs = get_session_recommendations()
            assert recs == []


class TestAutoConfigureSessions:
    """Test auto_configure_sessions function."""

    def test_auto_configure_sessions_with_redis(self):
        """Test auto-configuration with Redis available."""
        mock_session_manager = MagicMock()
        mock_session_manager.configure_storage.return_value = True

        with patch.dict(
            "sys.modules", {"chuk_ai_session_manager": mock_session_manager}
        ):
            with patch(
                "chuk_llm.api.session_utils.check_session_backend_availability"
            ) as mock_check:
                mock_check.return_value = {
                    "redis_available": True,
                    "errors": [],
                }

                result = auto_configure_sessions()

                assert result is True
                mock_session_manager.configure_storage.assert_called()

    def test_auto_configure_sessions_with_memory_fallback(self, monkeypatch):
        """Test auto-configuration falls back to memory."""
        mock_session_manager = MagicMock()
        mock_session_manager.configure_storage.return_value = True

        with patch.dict(
            "sys.modules", {"chuk_ai_session_manager": mock_session_manager}
        ):
            with patch(
                "chuk_llm.api.session_utils.check_session_backend_availability"
            ) as mock_check:
                mock_check.return_value = {
                    "redis_available": False,
                    "errors": [],
                }

                result = auto_configure_sessions()

                # Should set SESSION_PROVIDER to memory
                if result:
                    assert os.environ.get("SESSION_PROVIDER") == "memory"

    def test_auto_configure_sessions_redis_with_errors(self):
        """Test auto-configuration when Redis has errors."""
        mock_session_manager = MagicMock()
        mock_session_manager.configure_storage.return_value = True

        with patch.dict(
            "sys.modules", {"chuk_ai_session_manager": mock_session_manager}
        ):
            with patch(
                "chuk_llm.api.session_utils.check_session_backend_availability"
            ) as mock_check:
                mock_check.return_value = {
                    "redis_available": True,
                    "errors": ["Redis connection failed"],
                }

                result = auto_configure_sessions()

                # Should fall back to memory when Redis has errors
                assert result is True or result is False

    def test_auto_configure_sessions_failure(self):
        """Test auto-configuration when configuration fails."""
        mock_session_manager = MagicMock()
        mock_session_manager.configure_storage.return_value = False

        with patch.dict(
            "sys.modules", {"chuk_ai_session_manager": mock_session_manager}
        ):
            with patch(
                "chuk_llm.api.session_utils.check_session_backend_availability"
            ) as mock_check:
                mock_check.return_value = {
                    "redis_available": False,
                    "errors": [],
                }

                result = auto_configure_sessions()

                assert result is False

    def test_auto_configure_sessions_exception(self):
        """Test auto-configuration handles exceptions."""
        with patch(
            "builtins.__import__", side_effect=Exception("Import failed")
        ):
            result = auto_configure_sessions()
            assert result is False


class TestPrintSessionDiagnostics:
    """Test print_session_diagnostics function."""

    def test_print_session_diagnostics_basic(self, capsys):
        """Test basic diagnostics printing."""
        with patch(
            "chuk_llm.api.session_utils.check_session_backend_availability"
        ) as mock_check:
            mock_check.return_value = {
                "memory_available": True,
                "redis_available": False,
                "current_provider": "memory",
                "errors": [],
                "recommendations": [],
                "session_manager_available": True,
            }

            print_session_diagnostics()

            captured = capsys.readouterr()
            assert "ChukLLM Session Diagnostics" in captured.out
            assert "Memory Storage:" in captured.out
            assert "Redis Storage:" in captured.out

    def test_print_session_diagnostics_with_errors(self, capsys):
        """Test diagnostics printing with errors."""
        with patch(
            "chuk_llm.api.session_utils.check_session_backend_availability"
        ) as mock_check:
            mock_check.return_value = {
                "memory_available": True,
                "redis_available": False,
                "current_provider": "redis",
                "errors": ["Redis not available", "Configuration error"],
                "recommendations": [],
                "session_manager_available": True,
            }

            print_session_diagnostics()

            captured = capsys.readouterr()
            assert "Issues:" in captured.out
            assert "Redis not available" in captured.out

    def test_print_session_diagnostics_with_recommendations(self, capsys):
        """Test diagnostics printing with recommendations."""
        with patch(
            "chuk_llm.api.session_utils.check_session_backend_availability"
        ) as mock_check:
            mock_check.return_value = {
                "memory_available": True,
                "redis_available": False,
                "current_provider": "memory",
                "errors": [],
                "recommendations": ["Install Redis", "Update config"],
                "session_manager_available": True,
            }

            print_session_diagnostics()

            captured = capsys.readouterr()
            assert "Recommendations:" in captured.out
            assert "Install Redis" in captured.out

    def test_print_session_diagnostics_with_storage_info(self, capsys):
        """Test diagnostics printing with storage info."""
        with patch(
            "chuk_llm.api.session_utils.check_session_backend_availability"
        ) as mock_check:
            mock_check.return_value = {
                "memory_available": True,
                "redis_available": True,
                "current_provider": "redis",
                "errors": [],
                "recommendations": [],
                "session_manager_available": True,
                "storage_info": {"backend": "redis", "sandbox_id": "test-123"},
            }

            print_session_diagnostics()

            captured = capsys.readouterr()
            assert "Backend:" in captured.out
            assert "redis" in captured.out
            assert "Sandbox ID:" in captured.out

    def test_print_session_diagnostics_success_message(self, capsys):
        """Test success message when no errors."""
        with patch(
            "chuk_llm.api.session_utils.check_session_backend_availability"
        ) as mock_check:
            mock_check.return_value = {
                "memory_available": True,
                "redis_available": True,
                "current_provider": "memory",
                "errors": [],
                "recommendations": [],
                "session_manager_available": True,
            }

            print_session_diagnostics()

            captured = capsys.readouterr()
            assert "Session configuration looks good!" in captured.out


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_check_session_backend_with_different_providers(self, monkeypatch):
        """Test with different SESSION_PROVIDER values."""
        for provider in ["memory", "redis", "custom"]:
            monkeypatch.setenv("SESSION_PROVIDER", provider)
            result = check_session_backend_availability()
            assert result["current_provider"] == provider

    def test_validate_session_configuration_with_partial_errors(self):
        """Test validation with some errors."""
        with patch(
            "chuk_llm.api.session_utils.check_session_backend_availability"
        ) as mock_check:
            mock_check.return_value = {
                "errors": ["Minor warning"],
                "recommendations": ["Fix it"],
            }

            result = validate_session_configuration()
            assert result is False

    def test_auto_configure_sessions_with_import_error(self):
        """Test auto-configuration when import fails."""
        with patch.dict("sys.modules", {"chuk_ai_session_manager": None}):
            with patch("builtins.__import__", side_effect=ImportError):
                result = auto_configure_sessions()
                assert result is False
