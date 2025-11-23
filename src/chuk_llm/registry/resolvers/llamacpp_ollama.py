# chuk_llm/registry/resolvers/llamacpp_ollama.py
"""
Ollama → llama.cpp Resolver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Discovers and resolves Ollama's downloaded GGUF models for use with llama-server.

Ollama stores models as GGUF blobs under ~/.ollama/models/blobs/ with
SHA-256 filenames. This resolver helps you reuse those models with llama.cpp
without re-downloading.

This is a resolver because it:
1. Discovers available models from Ollama's local storage
2. Resolves model names to GGUF file paths
3. Provides model metadata (size, digest, etc.)
"""

from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import NamedTuple

from pydantic import BaseModel, Field


class OllamaModel(NamedTuple):
    """Discovered Ollama model."""

    name: str
    gguf_path: Path
    size_bytes: int
    digest: str


class OllamaModelRegistry(BaseModel):
    """Registry of Ollama models available for llama.cpp."""

    ollama_data_dir: Path = Field(default_factory=lambda: _get_ollama_data_dir())

    @staticmethod
    def _get_ollama_data_dir() -> Path:
        """
        Get Ollama data directory based on platform.

        - macOS/Linux: ~/.ollama
        - Windows: %LOCALAPPDATA%\\Ollama (typically C:\\Users\\<user>\\AppData\\Local\\Ollama)
        """
        system = platform.system()

        if system == "Darwin" or system == "Linux":
            return Path.home() / ".ollama"
        elif system == "Windows":
            # Windows uses %LOCALAPPDATA%\Ollama
            import os

            local_app_data = os.getenv("LOCALAPPDATA")
            if local_app_data:
                return Path(local_app_data) / "Ollama"
            else:
                # Fallback to %USERPROFILE%\AppData\Local\Ollama
                return Path.home() / "AppData" / "Local" / "Ollama"
        else:
            # Default fallback for unknown systems
            return Path.home() / ".ollama"

    def discover_models(self) -> list[OllamaModel]:
        """
        Discover all Ollama models that can be used with llama.cpp.

        Returns:
            List of discovered models with their GGUF paths.
        """
        models: list[OllamaModel] = []

        # Check if Ollama data directory exists
        if not self.ollama_data_dir.exists():
            return models

        # Read manifests to map digests to model names
        manifests_dir = self.ollama_data_dir / "models" / "manifests"
        name_to_digest = {}

        if manifests_dir.exists():
            # Walk through manifest directory structure
            # Handles both: registry/provider/tag AND registry/provider/model/tag
            for registry_dir in manifests_dir.iterdir():
                if not registry_dir.is_dir():
                    continue

                for provider_dir in registry_dir.iterdir():
                    if not provider_dir.is_dir():
                        continue

                    for model_entry in provider_dir.iterdir():
                        # Check if this is a direct manifest file or a model directory
                        manifest_files = []

                        if model_entry.is_file():
                            # Format: registry/provider/tag (e.g., library/llama3.2/latest)
                            manifest_files.append(
                                (model_entry, f"{provider_dir.name}:{model_entry.name}")
                            )
                        elif model_entry.is_dir():
                            # Format: registry/provider/model/tag (e.g., vanilj/Phi-4/latest)
                            for tag_file in model_entry.iterdir():
                                if tag_file.is_file():
                                    manifest_files.append(
                                        (
                                            tag_file,
                                            f"{provider_dir.name}/{model_entry.name}:{tag_file.name}",
                                        )
                                    )

                        # Process each manifest file
                        for manifest_file, model_name in manifest_files:
                            try:
                                manifest = json.loads(manifest_file.read_text())
                                # Extract the model layer digest (the big GGUF blob)
                                for layer in manifest.get("layers", []):
                                    media_type = layer.get("mediaType", "")
                                    if (
                                        "model" in media_type
                                        or "gguf" in media_type.lower()
                                    ):
                                        digest = layer.get("digest", "")
                                        if digest:
                                            # Convert sha256:xxx to sha256-xxx (blob filename format)
                                            blob_digest = digest.replace(":", "-")
                                            name_to_digest[model_name] = blob_digest
                                            break
                            except (json.JSONDecodeError, KeyError):
                                continue

        # Now look for GGUF blobs
        blobs_dir = self.ollama_data_dir / "models" / "blobs"
        if not blobs_dir.exists():
            return models

        for blob_file in blobs_dir.iterdir():
            if not blob_file.is_file():
                continue

            # Ollama uses sha256-{hash} format
            if not blob_file.name.startswith("sha256-"):
                continue

            # Only include files larger than 100MB (likely model files)
            size = blob_file.stat().st_size
            if size < 100 * 1024 * 1024:
                continue

            # Try to find the model name
            digest = blob_file.name  # sha256-xxx
            found_name: str | None = None

            for name, dig in name_to_digest.items():
                if dig == digest:
                    found_name = name
                    break

            # If no manifest match, use the digest as name
            if not found_name:
                found_name = blob_file.name

            models.append(
                OllamaModel(
                    name=found_name,
                    gguf_path=blob_file,
                    size_bytes=size,
                    digest=digest,
                )
            )

        return sorted(models, key=lambda m: m.size_bytes)

    def find_model(self, name: str) -> OllamaModel | None:
        """
        Find a specific Ollama model by name.

        Args:
            name: Model name (e.g., "llama3.2", "registry/library/llama3.2:latest")

        Returns:
            OllamaModel if found, None otherwise.
        """
        models = self.discover_models()

        # Try exact match first
        for model in models:
            if model.name == name:
                return model

        # Try partial match
        for model in models:
            if name in model.name:
                return model

        return None


def _get_ollama_data_dir() -> Path:
    """Get Ollama data directory based on platform."""
    return OllamaModelRegistry._get_ollama_data_dir()


def discover_ollama_models() -> list[OllamaModel]:
    """
    Convenience function to discover Ollama models.

    Returns:
        List of discovered Ollama models.
    """
    registry = OllamaModelRegistry()
    return registry.discover_models()


def find_ollama_model(name: str) -> OllamaModel | None:
    """
    Convenience function to find a specific Ollama model.

    Args:
        name: Model name

    Returns:
        OllamaModel if found, None otherwise.
    """
    registry = OllamaModelRegistry()
    return registry.find_model(name)
