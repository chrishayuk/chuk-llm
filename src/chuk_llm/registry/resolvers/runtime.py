"""
Runtime testing capability resolver.

Tests model capabilities dynamically when unknown models are discovered.
This allows immediate use of new models without waiting for offline
capability updates.
"""

from __future__ import annotations

import logging
import os

from chuk_llm.registry.models import ModelCapabilities, ModelSpec
from chuk_llm.registry.resolvers.base import BaseCapabilityResolver
from chuk_llm.registry.runtime_tester import RuntimeCapabilityTester

log = logging.getLogger(__name__)


class RuntimeTestingResolver(BaseCapabilityResolver):
    """
    Tests model capabilities at runtime for unknown models.

    This resolver performs actual API calls to test capabilities when
    a model is not found in the YAML cache or other resolvers.

    Configuration:
    - Enable: Set CHUK_LLM_RUNTIME_TESTING=true
    - Disable: Set CHUK_LLM_RUNTIME_TESTING=false (default)

    Note: Runtime testing makes API calls which may incur costs.
    """

    def __init__(self, enabled: bool | None = None):
        """
        Initialize runtime testing resolver.

        Args:
            enabled: Whether to enable runtime testing.
                    If None, reads from CHUK_LLM_RUNTIME_TESTING env var.
                    Defaults to False for safety (no unexpected API costs).
        """
        if enabled is None:
            env_value = os.getenv("CHUK_LLM_RUNTIME_TESTING", "false").lower()
            enabled = env_value in ("true", "1", "yes", "on")

        self.enabled = enabled

        if self.enabled:
            log.info("Runtime capability testing ENABLED - will test unknown models")
        else:
            log.debug(
                "Runtime capability testing DISABLED - set CHUK_LLM_RUNTIME_TESTING=true to enable"
            )

    async def get_capabilities(self, spec: ModelSpec) -> ModelCapabilities:
        """
        Get capabilities by testing the model at runtime.

        Args:
            spec: Model specification

        Returns:
            Tested capabilities or empty capabilities if disabled
        """
        if not self.enabled:
            # Return empty capabilities - don't test
            return ModelCapabilities(
                source="runtime_resolver_disabled",
            )

        # Runtime testing is enabled - test the model
        log.info(f"Runtime testing {spec.provider}/{spec.name} (may incur API costs)")

        try:
            tester = RuntimeCapabilityTester(spec.provider)
            capabilities = await tester.test_model(spec.name)

            log.info(
                f"Runtime test complete for {spec.provider}/{spec.name}: "
                f"tools={capabilities.supports_tools}, "
                f"vision={capabilities.supports_vision}, "
                f"json={capabilities.supports_json_mode}, "
                f"streaming={capabilities.supports_streaming}"
            )

            return capabilities

        except Exception as e:
            log.warning(f"Runtime testing failed for {spec.provider}/{spec.name}: {e}")
            # Return empty capabilities on failure
            return ModelCapabilities(
                source="runtime_test_failed",
            )
