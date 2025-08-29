# benchmarks/utils/config_loader.py
"""
Test configuration loader for benchmark suites.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class TestConfigLoader:
    """Loads and manages test configurations from JSON files"""

    def __init__(self, config_dir: Path | None = None):
        """
        Initialize the config loader.

        Args:
            config_dir: Directory containing test config JSON files.
                       Defaults to benchmarks/test_configs/
        """
        if config_dir is None:
            # Default to test_configs directory relative to this file
            self.config_dir = Path(__file__).parent.parent / "test_configs"
        else:
            self.config_dir = Path(config_dir)

        self._config_cache: dict[str, dict[str, Any]] = {}

    def load_test_suite(self, suite_name: str) -> list[dict[str, Any]]:
        """
        Load test configuration for a given suite.

        Args:
            suite_name: Name of the test suite (e.g., 'quick', 'lightning', 'standard')

        Returns:
            List of test configurations

        Raises:
            FileNotFoundError: If the config file doesn't exist
            ValueError: If the config is invalid
        """
        # Check cache first
        if suite_name in self._config_cache:
            return self._config_cache[suite_name]["tests"]

        # Load from file
        config_file = self.config_dir / f"{suite_name}.json"

        if not config_file.exists():
            # Check for available configs
            available = self.get_available_suites()
            raise FileNotFoundError(
                f"Test suite '{suite_name}' not found. "
                f"Available suites: {', '.join(available)}"
            )

        try:
            with open(config_file, encoding="utf-8") as f:
                config = json.load(f)

            # Validate config structure
            self._validate_config(config, suite_name)

            # Cache and return
            self._config_cache[suite_name] = config
            log.info(
                f"Loaded test suite '{suite_name}' with {len(config['tests'])} tests"
            )

            return config["tests"]

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {config_file}: {e}") from e
        except Exception as e:
            raise ValueError(f"Error loading config {config_file}: {e}") from e

    def get_available_suites(self) -> list[str]:
        """Get list of available test suites"""
        if not self.config_dir.exists():
            return []

        suites = []
        for config_file in self.config_dir.glob("*.json"):
            suites.append(config_file.stem)

        return sorted(suites)

    def get_suite_info(self, suite_name: str) -> dict[str, Any]:
        """Get metadata about a test suite"""
        if suite_name not in self._config_cache:
            # Load it to cache
            self.load_test_suite(suite_name)

        config = self._config_cache[suite_name]
        return {
            "name": config["name"],
            "description": config["description"],
            "test_count": len(config["tests"]),
            "test_names": [test["name"] for test in config["tests"]],
        }

    def create_custom_suite(
        self,
        suite_name: str,
        description: str,
        tests: list[dict[str, Any]],
        save_to_file: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Create a custom test suite.

        Args:
            suite_name: Name for the custom suite
            description: Description of the suite
            tests: List of test configurations
            save_to_file: Whether to save the suite to a JSON file

        Returns:
            List of test configurations
        """
        config = {"name": suite_name, "description": description, "tests": tests}

        # Validate the custom config
        self._validate_config(config, suite_name)

        # Cache it
        self._config_cache[suite_name] = config

        # Optionally save to file
        if save_to_file:
            self.config_dir.mkdir(parents=True, exist_ok=True)
            config_file = self.config_dir / f"{suite_name}.json"

            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)

            log.info(f"Saved custom suite '{suite_name}' to {config_file}")

        return tests

    def _validate_config(self, config: dict[str, Any], suite_name: str) -> None:
        """Validate configuration structure"""
        required_fields = ["name", "description", "tests"]

        for field in required_fields:
            if field not in config:
                raise ValueError(
                    f"Missing required field '{field}' in {suite_name} config"
                )

        if not isinstance(config["tests"], list):
            raise ValueError(f"'tests' must be a list in {suite_name} config")

        if not config["tests"]:
            raise ValueError(f"No tests defined in {suite_name} config")

        # Validate each test
        for i, test in enumerate(config["tests"]):
            self._validate_test_config(test, f"{suite_name}.tests[{i}]")

    def _validate_test_config(self, test: dict[str, Any], context: str) -> None:
        """Validate individual test configuration"""
        required_fields = ["name", "description", "messages", "test_type"]

        for field in required_fields:
            if field not in test:
                raise ValueError(f"Missing required field '{field}' in {context}")

        # Validate messages
        if not isinstance(test["messages"], list) or not test["messages"]:
            raise ValueError(f"'messages' must be a non-empty list in {context}")

        # Validate message structure
        for j, message in enumerate(test["messages"]):
            if not isinstance(message, dict):
                raise ValueError(f"Message {j} must be a dict in {context}")

            if "role" not in message or "content" not in message:
                raise ValueError(
                    f"Message {j} missing 'role' or 'content' in {context}"
                )

        # Validate optional numeric fields
        for field in ["max_tokens", "temperature", "expected_min_tokens"]:
            if field in test and not isinstance(test[field], int | float):
                raise ValueError(f"'{field}' must be numeric in {context}")


# Global loader instance
_loader = TestConfigLoader()


def load_test_suite(suite_name: str) -> list[dict[str, Any]]:
    """Convenience function to load a test suite"""
    return _loader.load_test_suite(suite_name)


def get_available_suites() -> list[str]:
    """Convenience function to get available suites"""
    return _loader.get_available_suites()


def get_suite_info(suite_name: str) -> dict[str, Any]:
    """Convenience function to get suite info"""
    return _loader.get_suite_info(suite_name)


def create_custom_suite(
    suite_name: str,
    description: str,
    tests: list[dict[str, Any]],
    save_to_file: bool = False,
) -> list[dict[str, Any]]:
    """Convenience function to create custom suite"""
    return _loader.create_custom_suite(suite_name, description, tests, save_to_file)
