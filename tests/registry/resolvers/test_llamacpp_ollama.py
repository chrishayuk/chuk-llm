"""
Comprehensive tests for LlamaCpp Ollama Resolver.

Tests cover:
- Model discovery from Ollama storage
- Manifest parsing
- GGUF blob detection
- Model name resolution
- Cross-platform path handling
"""

import json
import platform
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from chuk_llm.registry.resolvers.llamacpp_ollama import (
    OllamaModel,
    OllamaModelRegistry,
    _get_ollama_data_dir,
    discover_ollama_models,
    find_ollama_model,
)


class TestOllamaDataDir:
    """Test platform-specific Ollama data directory resolution."""

    @patch("platform.system", return_value="Darwin")
    def test_get_ollama_data_dir_macos(self, mock_system):
        """Test macOS path resolution."""
        data_dir = _get_ollama_data_dir()
        assert data_dir == Path.home() / ".ollama"

    @patch("platform.system", return_value="Linux")
    def test_get_ollama_data_dir_linux(self, mock_system):
        """Test Linux path resolution."""
        data_dir = _get_ollama_data_dir()
        assert data_dir == Path.home() / ".ollama"

    @patch("platform.system", return_value="Windows")
    @patch("os.getenv", return_value="C:\\Users\\Test\\AppData\\Local")
    def test_get_ollama_data_dir_windows_with_env(self, mock_env, mock_system):
        """Test Windows path resolution with LOCALAPPDATA."""
        data_dir = _get_ollama_data_dir()
        assert data_dir == Path("C:\\Users\\Test\\AppData\\Local") / "Ollama"

    @patch("platform.system", return_value="Windows")
    @patch("os.getenv", return_value=None)
    def test_get_ollama_data_dir_windows_fallback(self, mock_env, mock_system):
        """Test Windows path resolution fallback."""
        data_dir = _get_ollama_data_dir()
        assert data_dir == Path.home() / "AppData" / "Local" / "Ollama"

    @patch("platform.system", return_value="FreeBSD")
    def test_get_ollama_data_dir_unknown_system(self, mock_system):
        """Test unknown system defaults to Unix-style path."""
        data_dir = _get_ollama_data_dir()
        assert data_dir == Path.home() / ".ollama"


class TestOllamaModelRegistry:
    """Test OllamaModelRegistry discovery functionality."""

    def test_init_default(self):
        """Test registry initialization with default path."""
        registry = OllamaModelRegistry()
        assert isinstance(registry.ollama_data_dir, Path)

    def test_init_custom_path(self, tmp_path):
        """Test registry initialization with custom path."""
        custom_dir = tmp_path / "custom_ollama"
        registry = OllamaModelRegistry(ollama_data_dir=custom_dir)
        assert registry.ollama_data_dir == custom_dir

    def test_discover_models_no_directory(self, tmp_path):
        """Test discovery when Ollama directory doesn't exist."""
        nonexistent_dir = tmp_path / "nonexistent"
        registry = OllamaModelRegistry(ollama_data_dir=nonexistent_dir)

        models = registry.discover_models()

        assert models == []

    def test_discover_models_empty_directory(self, tmp_path):
        """Test discovery with empty Ollama directory."""
        ollama_dir = tmp_path / "ollama"
        ollama_dir.mkdir()

        registry = OllamaModelRegistry(ollama_data_dir=ollama_dir)
        models = registry.discover_models()

        assert models == []

    def test_discover_models_no_blobs(self, tmp_path):
        """Test discovery when blobs directory doesn't exist."""
        ollama_dir = tmp_path / "ollama"
        manifests_dir = ollama_dir / "models" / "manifests"
        manifests_dir.mkdir(parents=True)

        registry = OllamaModelRegistry(ollama_data_dir=ollama_dir)
        models = registry.discover_models()

        assert models == []

    def test_discover_models_with_gguf_blobs(self, tmp_path):
        """Test discovery of GGUF model blobs."""
        ollama_dir = tmp_path / "ollama"
        blobs_dir = ollama_dir / "models" / "blobs"
        blobs_dir.mkdir(parents=True)

        # Create a fake GGUF blob (>100MB)
        blob_file = blobs_dir / "sha256-abc123"
        blob_file.write_bytes(b"0" * (101 * 1024 * 1024))  # 101MB

        registry = OllamaModelRegistry(ollama_data_dir=ollama_dir)
        models = registry.discover_models()

        assert len(models) == 1
        assert models[0].name == "sha256-abc123"
        assert models[0].gguf_path == blob_file
        assert models[0].size_bytes > 100 * 1024 * 1024
        assert models[0].digest == "sha256-abc123"

    def test_discover_models_skips_small_files(self, tmp_path):
        """Test that files under 100MB are skipped."""
        ollama_dir = tmp_path / "ollama"
        blobs_dir = ollama_dir / "models" / "blobs"
        blobs_dir.mkdir(parents=True)

        # Create a small file (<100MB)
        small_file = blobs_dir / "sha256-small"
        small_file.write_bytes(b"0" * (50 * 1024 * 1024))  # 50MB

        registry = OllamaModelRegistry(ollama_data_dir=ollama_dir)
        models = registry.discover_models()

        assert models == []

    def test_discover_models_skips_non_sha256_files(self, tmp_path):
        """Test that non-sha256 files are skipped."""
        ollama_dir = tmp_path / "ollama"
        blobs_dir = ollama_dir / "models" / "blobs"
        blobs_dir.mkdir(parents=True)

        # Create a large file without sha256 prefix
        other_file = blobs_dir / "other-file"
        other_file.write_bytes(b"0" * (101 * 1024 * 1024))

        registry = OllamaModelRegistry(ollama_data_dir=ollama_dir)
        models = registry.discover_models()

        assert models == []

    def test_discover_models_with_manifest(self, tmp_path):
        """Test discovery with manifest mapping."""
        ollama_dir = tmp_path / "ollama"

        # Create manifest structure
        manifests_dir = (
            ollama_dir / "models" / "manifests" / "registry.ollama.ai" / "library"
        )
        manifests_dir.mkdir(parents=True)

        # Create manifest file
        manifest_file = manifests_dir / "llama3.2"
        manifest = {
            "layers": [
                {
                    "mediaType": "application/vnd.ollama.image.model",
                    "digest": "sha256-abc123",
                }
            ]
        }
        manifest_file.write_text(json.dumps(manifest))

        # Create corresponding blob
        blobs_dir = ollama_dir / "models" / "blobs"
        blobs_dir.mkdir(parents=True)
        blob_file = blobs_dir / "sha256-abc123"
        blob_file.write_bytes(b"0" * (101 * 1024 * 1024))

        registry = OllamaModelRegistry(ollama_data_dir=ollama_dir)
        models = registry.discover_models()

        assert len(models) == 1
        assert models[0].name == "library:llama3.2"
        assert models[0].gguf_path == blob_file

    def test_discover_models_with_gguf_mediatype(self, tmp_path):
        """Test discovery with GGUF media type in manifest."""
        ollama_dir = tmp_path / "ollama"

        # Create manifest with gguf mediaType
        manifests_dir = ollama_dir / "models" / "manifests" / "registry" / "lib"
        manifests_dir.mkdir(parents=True)

        manifest_file = manifests_dir / "model1"
        manifest = {
            "layers": [{"mediaType": "application/vnd.gguf", "digest": "sha256-def456"}]
        }
        manifest_file.write_text(json.dumps(manifest))

        # Create blob
        blobs_dir = ollama_dir / "models" / "blobs"
        blobs_dir.mkdir(parents=True)
        blob_file = blobs_dir / "sha256-def456"
        blob_file.write_bytes(b"0" * (101 * 1024 * 1024))

        registry = OllamaModelRegistry(ollama_data_dir=ollama_dir)
        models = registry.discover_models()

        assert len(models) == 1
        assert models[0].name == "lib:model1"

    def test_discover_models_ignores_invalid_json(self, tmp_path):
        """Test that invalid manifest JSON is ignored."""
        ollama_dir = tmp_path / "ollama"

        manifests_dir = ollama_dir / "models" / "manifests" / "reg" / "lib"
        manifests_dir.mkdir(parents=True)

        # Create invalid manifest
        manifest_file = manifests_dir / "broken"
        manifest_file.write_text("invalid json {")

        # Create blob
        blobs_dir = ollama_dir / "models" / "blobs"
        blobs_dir.mkdir(parents=True)
        blob_file = blobs_dir / "sha256-xyz789"
        blob_file.write_bytes(b"0" * (101 * 1024 * 1024))

        registry = OllamaModelRegistry(ollama_data_dir=ollama_dir)
        models = registry.discover_models()

        # Should find blob but with digest as name (no manifest match)
        assert len(models) == 1
        assert models[0].name == "sha256-xyz789"

    def test_discover_models_sorted_by_size(self, tmp_path):
        """Test that models are sorted by size."""
        ollama_dir = tmp_path / "ollama"
        blobs_dir = ollama_dir / "models" / "blobs"
        blobs_dir.mkdir(parents=True)

        # Create blobs of different sizes
        blob1 = blobs_dir / "sha256-large"
        blob1.write_bytes(b"0" * (200 * 1024 * 1024))

        blob2 = blobs_dir / "sha256-small"
        blob2.write_bytes(b"0" * (101 * 1024 * 1024))

        blob3 = blobs_dir / "sha256-medium"
        blob3.write_bytes(b"0" * (150 * 1024 * 1024))

        registry = OllamaModelRegistry(ollama_data_dir=ollama_dir)
        models = registry.discover_models()

        assert len(models) == 3
        # Should be sorted by size (ascending)
        assert models[0].size_bytes < models[1].size_bytes < models[2].size_bytes
        assert models[0].name == "sha256-small"
        assert models[2].name == "sha256-large"

    def test_find_model_exact_match(self, tmp_path):
        """Test finding model by exact name match."""
        ollama_dir = tmp_path / "ollama"

        # Create manifest
        manifests_dir = ollama_dir / "models" / "manifests" / "registry" / "lib"
        manifests_dir.mkdir(parents=True)
        manifest_file = manifests_dir / "llama3"
        manifest = {
            "layers": [
                {
                    "mediaType": "application/vnd.ollama.image.model",
                    "digest": "sha256-abc",
                }
            ]
        }
        manifest_file.write_text(json.dumps(manifest))

        # Create blob
        blobs_dir = ollama_dir / "models" / "blobs"
        blobs_dir.mkdir(parents=True)
        blob_file = blobs_dir / "sha256-abc"
        blob_file.write_bytes(b"0" * (101 * 1024 * 1024))

        registry = OllamaModelRegistry(ollama_data_dir=ollama_dir)
        model = registry.find_model("lib:llama3")

        assert model is not None
        assert model.name == "lib:llama3"
        assert model.gguf_path == blob_file

    def test_find_model_partial_match(self, tmp_path):
        """Test finding model by partial name match."""
        ollama_dir = tmp_path / "ollama"

        manifests_dir = ollama_dir / "models" / "manifests" / "registry" / "lib"
        manifests_dir.mkdir(parents=True)
        manifest_file = manifests_dir / "llama3.2-instruct"
        manifest = {
            "layers": [
                {
                    "mediaType": "application/vnd.ollama.image.model",
                    "digest": "sha256-xyz",
                }
            ]
        }
        manifest_file.write_text(json.dumps(manifest))

        blobs_dir = ollama_dir / "models" / "blobs"
        blobs_dir.mkdir(parents=True)
        blob_file = blobs_dir / "sha256-xyz"
        blob_file.write_bytes(b"0" * (101 * 1024 * 1024))

        registry = OllamaModelRegistry(ollama_data_dir=ollama_dir)
        model = registry.find_model("llama3.2")

        assert model is not None
        assert "llama3.2" in model.name

    def test_find_model_not_found(self, tmp_path):
        """Test finding nonexistent model returns None."""
        ollama_dir = tmp_path / "ollama"
        ollama_dir.mkdir()

        registry = OllamaModelRegistry(ollama_data_dir=ollama_dir)
        model = registry.find_model("nonexistent-model")

        assert model is None


class TestConvenienceFunctions:
    """Test top-level convenience functions."""

    @patch("chuk_llm.registry.resolvers.llamacpp_ollama.OllamaModelRegistry")
    def test_discover_ollama_models(self, mock_registry_class):
        """Test discover_ollama_models convenience function."""
        mock_registry = MagicMock()
        mock_models = [
            OllamaModel("model1", Path("/path1"), 1000, "digest1"),
            OllamaModel("model2", Path("/path2"), 2000, "digest2"),
        ]
        mock_registry.discover_models.return_value = mock_models
        mock_registry_class.return_value = mock_registry

        result = discover_ollama_models()

        mock_registry_class.assert_called_once()
        mock_registry.discover_models.assert_called_once()
        assert result == mock_models

    @patch("chuk_llm.registry.resolvers.llamacpp_ollama.OllamaModelRegistry")
    def test_find_ollama_model(self, mock_registry_class):
        """Test find_ollama_model convenience function."""
        mock_registry = MagicMock()
        mock_model = OllamaModel("test-model", Path("/path"), 1000, "digest")
        mock_registry.find_model.return_value = mock_model
        mock_registry_class.return_value = mock_registry

        result = find_ollama_model("test-model")

        mock_registry_class.assert_called_once()
        mock_registry.find_model.assert_called_once_with("test-model")
        assert result == mock_model

    @patch("chuk_llm.registry.resolvers.llamacpp_ollama.OllamaModelRegistry")
    def test_find_ollama_model_not_found(self, mock_registry_class):
        """Test find_ollama_model when model doesn't exist."""
        mock_registry = MagicMock()
        mock_registry.find_model.return_value = None
        mock_registry_class.return_value = mock_registry

        result = find_ollama_model("nonexistent")

        assert result is None


class TestOllamaModel:
    """Test OllamaModel NamedTuple."""

    def test_ollama_model_creation(self):
        """Test creating OllamaModel instance."""
        model = OllamaModel(
            name="test-model",
            gguf_path=Path("/path/to/model.gguf"),
            size_bytes=1024 * 1024 * 1024,
            digest="sha256-abc123",
        )

        assert model.name == "test-model"
        assert model.gguf_path == Path("/path/to/model.gguf")
        assert model.size_bytes == 1024 * 1024 * 1024
        assert model.digest == "sha256-abc123"

    def test_ollama_model_immutable(self):
        """Test that OllamaModel is immutable (NamedTuple)."""
        model = OllamaModel("test", Path("/path"), 1000, "digest")

        with pytest.raises(AttributeError):
            model.name = "new-name"
