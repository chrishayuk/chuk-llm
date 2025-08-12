# chuk_llm/llm/providers/groq_client.py
"""
Groq chat-completion adapter - OpenAI-compatible with actual Groq models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Since Groq is OpenAI-compatible, this inherits from OpenAILLMClient
and overrides Groq-specific behavior.

PRODUCTION MODELS (as of January 2025):
- llama-3.1-8b-instant: 131k context, 131k output
- llama-3.3-70b-versatile: 131k context, 32k output  
- meta-llama/llama-guard-4-12b: 131k context, 1k output
- whisper-large-v3: Audio transcription
- whisper-large-v3-turbo: Audio transcription

PREVIEW MODELS:
- deepseek-r1-distill-llama-70b: 131k context, 131k output
- meta-llama/llama-4-maverick-17b-128e-instruct: 131k context, 8k output
- meta-llama/llama-4-scout-17b-16e-instruct: 131k context, 8k output
- moonshotai/kimi-k2-instruct: 131k context, 16k output
- openai/gpt-oss-120b: 131k context, 32k output
- openai/gpt-oss-20b: 131k context, 32k output
- qwen/qwen3-32b: 131k context, 40k output
- playai-tts: Text-to-speech models
- compound-beta/compound-beta-mini: Groq systems

CRITICAL FEATURES:
1. Inherits from OpenAILLMClient for code reuse
2. Smart discovery for all model types
3. Tool calls always return a list (never None)
4. Groq-specific error handling and retry logic
5. Ultra-fast inference optimizations
6. Accurate context/token limits based on actual Groq models
"""
from __future__ import annotations

import asyncio
import logging
import uuid
import json
from typing import Any, AsyncIterator, Dict, List, Optional, Union, Tuple, Set

# Import the OpenAI client as base
from chuk_llm.llm.providers.openai_client import OpenAILLMClient

log = logging.getLogger(__name__)


class GroqAILLMClient(OpenAILLMClient):
    """
    Groq-specific adapter that inherits from OpenAI client.
    
    Since Groq is OpenAI-compatible, we inherit most functionality
    and only override Groq-specific behavior.
    
    Key features:
    - Ultra-fast inference for all models
    - Comprehensive model support (Llama, DeepSeek, GPT-OSS, Qwen, etc.)
    - Smart defaults for all models
    - Groq-specific error handling
    - Huge context windows (131k tokens for most models)
    """
    
    # Known Groq production models with their actual limits
    GROQ_PRODUCTION_MODELS = {
        "llama-3.1-8b-instant": {
            "context": 131072,
            "max_output": 131072,
            "family": "llama",
            "features": {"text", "streaming", "tools", "system_messages", "json_mode"}
        },
        "llama-3.3-70b-versatile": {
            "context": 131072,
            "max_output": 32768,
            "family": "llama",
            "features": {"text", "streaming", "tools", "system_messages", "json_mode", "parallel_calls"}
        },
        "meta-llama/llama-guard-4-12b": {
            "context": 131072,
            "max_output": 1024,
            "family": "llama-guard",
            "features": {"text", "streaming", "system_messages"}  # Safety model
        },
        "whisper-large-v3": {
            "context": None,
            "max_output": None,
            "family": "whisper",
            "features": {"audio", "transcription"}
        },
        "whisper-large-v3-turbo": {
            "context": None,
            "max_output": None,
            "family": "whisper",
            "features": {"audio", "transcription"}
        }
    }
    
    # Known Groq preview models with their actual limits
    GROQ_PREVIEW_MODELS = {
        "deepseek-r1-distill-llama-70b": {
            "context": 131072,
            "max_output": 131072,
            "family": "deepseek",
            "features": {"text", "streaming", "tools", "reasoning", "system_messages", "json_mode"}
        },
        "meta-llama/llama-4-maverick-17b-128e-instruct": {
            "context": 131072,
            "max_output": 8192,
            "family": "llama4",
            "features": {"text", "streaming", "tools", "system_messages", "json_mode"}
        },
        "meta-llama/llama-4-scout-17b-16e-instruct": {
            "context": 131072,
            "max_output": 8192,
            "family": "llama4",
            "features": {"text", "streaming", "tools", "system_messages", "json_mode"}
        },
        "meta-llama/llama-prompt-guard-2-22m": {
            "context": 512,
            "max_output": 512,
            "family": "llama-guard",
            "features": {"text", "streaming"}  # Safety model
        },
        "meta-llama/llama-prompt-guard-2-86m": {
            "context": 512,
            "max_output": 512,
            "family": "llama-guard",
            "features": {"text", "streaming"}  # Safety model
        },
        "moonshotai/kimi-k2-instruct": {
            "context": 131072,
            "max_output": 16384,
            "family": "kimi",
            "features": {"text", "streaming", "tools", "system_messages", "json_mode"}
        },
        "openai/gpt-oss-120b": {
            "context": 131072,
            "max_output": 32766,
            "family": "gpt-oss",
            "features": {"text", "streaming", "tools", "reasoning", "system_messages", "json_mode"}
        },
        "openai/gpt-oss-20b": {
            "context": 131072,
            "max_output": 32768,
            "family": "gpt-oss",
            "features": {"text", "streaming", "tools", "reasoning", "system_messages", "json_mode"}
        },
        "playai-tts": {
            "context": 8192,
            "max_output": 8192,
            "family": "tts",
            "features": {"text_to_speech", "audio"}
        },
        "playai-tts-arabic": {
            "context": 8192,
            "max_output": 8192,
            "family": "tts",
            "features": {"text_to_speech", "audio"}
        },
        "qwen/qwen3-32b": {
            "context": 131072,
            "max_output": 40960,
            "family": "qwen",
            "features": {"text", "streaming", "tools", "reasoning", "system_messages", "json_mode"}
        },
        "compound-beta": {
            "context": 131072,
            "max_output": 8192,
            "family": "compound",
            "features": {"text", "streaming", "tools", "system_messages", "reasoning"}  # System model
        },
        "compound-beta-mini": {
            "context": 131072,
            "max_output": 8192,
            "family": "compound",
            "features": {"text", "streaming", "tools", "system_messages", "reasoning"}  # System model
        }
    }

    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> None:
        # Use Groq's API endpoint
        groq_base = api_base or "https://api.groq.com/openai/v1"
        
        # Initialize parent with Groq settings
        super().__init__(
            model=model,
            api_key=api_key,
            api_base=groq_base
        )
        
        # Override provider detection to always be "groq"
        self.detected_provider = "groq"
        self.provider_name = "groq"
        
        log.debug(f"Groq client initialized: model={self.model}, base={groq_base}")

    # ================================================================
    # GROQ-SPECIFIC SMART DEFAULTS
    # ================================================================
    
    def _get_known_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Check if this is a known Groq model and return its info"""
        # Check production models
        if model_name in self.GROQ_PRODUCTION_MODELS:
            return self.GROQ_PRODUCTION_MODELS[model_name]
        
        # Check preview models
        if model_name in self.GROQ_PREVIEW_MODELS:
            return self.GROQ_PREVIEW_MODELS[model_name]
        
        return None
    
    @staticmethod
    def _get_smart_default_features(model_name: str) -> Set[str]:
        """
        Get smart default features for models hosted on Groq.
        
        First checks known models, then uses pattern matching for unknown models.
        """
        # First check if it's a known model
        client = GroqAILLMClient.__new__(GroqAILLMClient)
        known_info = client._get_known_model_info(model_name)
        if known_info:
            return known_info["features"]
        
        model_lower = model_name.lower()
        
        # Base features that ALL text models on Groq should have
        base_features = {"text", "streaming", "system_messages"}
        
        # === WHISPER/AUDIO MODELS ===
        if 'whisper' in model_lower:
            return {"audio", "transcription"}
        
        # === TTS MODELS ===
        if 'tts' in model_lower or 'playai' in model_lower:
            return {"text_to_speech", "audio"}
        
        # === GUARD/SAFETY MODELS ===
        if 'guard' in model_lower:
            return {"text", "streaming"}  # Safety models don't usually support tools
        
        # === GPT-OSS MODELS (open source reasoning) ===
        elif 'gpt-oss' in model_lower:
            features = base_features | {"tools", "reasoning", "json_mode"}
            log.info(f"[groq] Detected GPT-OSS model '{model_name}' - enabling tools and reasoning")
            return features
        
        # === DEEPSEEK MODELS ===
        elif 'deepseek' in model_lower:
            features = base_features | {"tools", "json_mode"}
            if 'r1' in model_lower or 'reason' in model_lower:
                features.add("reasoning")
            return features
        
        # === LLAMA MODELS ===
        elif 'llama' in model_lower:
            # Guard models are safety-focused
            if 'guard' in model_lower:
                return {"text", "streaming"}
            
            features = base_features | {"tools", "json_mode"}
            
            # Llama 4 models
            if 'llama-4' in model_lower or 'llama4' in model_lower:
                features.add("parallel_calls")
            
            # Larger models support more features
            if any(size in model_lower for size in ['70b', '405b']):
                features.add("parallel_calls")
            
            return features
        
        # === QWEN MODELS ===
        elif 'qwen' in model_lower:
            features = base_features | {"tools", "reasoning", "json_mode"}
            return features
        
        # === KIMI MODELS ===
        elif 'kimi' in model_lower:
            features = base_features | {"tools", "json_mode"}
            return features
        
        # === COMPOUND SYSTEMS ===
        elif 'compound' in model_lower:
            features = base_features | {"tools", "reasoning"}
            return features
        
        # === REASONING MODELS ===
        elif any(pattern in model_lower for pattern in ['reasoning', 'r1', 'distill']):
            features = base_features | {"reasoning", "tools", "json_mode"}
            log.info(f"[groq] Detected reasoning model '{model_name}' - enabling reasoning and tools")
            return features
        
        # === UNKNOWN MODEL - Be optimistic ===
        else:
            log.info(f"[groq] Unknown model '{model_name}' - assuming modern capabilities (tools support)")
            # Be optimistic - most modern models on Groq support tools
            # Default to 131k context which is standard for Groq
            return base_features | {"tools", "json_mode"}
    
    @staticmethod
    def _get_smart_default_parameters(model_name: str) -> Dict[str, Any]:
        """
        Get smart default parameters for models hosted on Groq.
        
        Uses actual Groq limits when known, smart defaults otherwise.
        """
        # First check if it's a known model
        client = GroqAILLMClient.__new__(GroqAILLMClient)
        known_info = client._get_known_model_info(model_name)
        if known_info:
            return {
                "max_context_length": known_info.get("context", 131072),
                "max_output_tokens": known_info.get("max_output", 8192),
                "supports_tools": "tools" in known_info["features"],
                "supports_reasoning": "reasoning" in known_info["features"],
                "ultra_fast_inference": True,
                "model_family": known_info["family"]
            }
        
        model_lower = model_name.lower()
        
        # === WHISPER/AUDIO MODELS ===
        if 'whisper' in model_lower:
            return {
                "supports_audio": True,
                "supports_transcription": True,
                "ultra_fast_inference": True,
                "model_type": "audio"
            }
        
        # === TTS MODELS ===
        if 'tts' in model_lower or 'playai' in model_lower:
            return {
                "max_context_length": 8192,
                "max_output_tokens": 8192,
                "supports_tts": True,
                "ultra_fast_inference": True,
                "model_type": "tts"
            }
        
        # === GPT-OSS MODELS ===
        elif 'gpt-oss' in model_lower:
            # These are actually on Groq with 131k context!
            if '120b' in model_lower:
                return {
                    "max_context_length": 131072,
                    "max_output_tokens": 32766,
                    "supports_tools": True,
                    "supports_reasoning": True,
                    "ultra_fast_inference": True,
                    "open_source": True
                }
            else:  # 20b
                return {
                    "max_context_length": 131072,
                    "max_output_tokens": 32768,
                    "supports_tools": True,
                    "supports_reasoning": True,
                    "ultra_fast_inference": True,
                    "open_source": True
                }
        
        # === DEEPSEEK MODELS ===
        elif 'deepseek' in model_lower:
            return {
                "max_context_length": 131072,  # Groq standard
                "max_output_tokens": 131072 if 'r1' in model_lower else 32768,
                "supports_tools": True,
                "supports_reasoning": 'r1' in model_lower or 'reason' in model_lower,
                "ultra_fast_inference": True
            }
        
        # === LLAMA MODELS ===
        elif 'llama' in model_lower:
            # Guard models have tiny limits
            if 'guard' in model_lower:
                if '22m' in model_lower or '86m' in model_lower:
                    return {
                        "max_context_length": 512,
                        "max_output_tokens": 512,
                        "supports_tools": False,
                        "ultra_fast_inference": True,
                        "model_type": "safety"
                    }
                else:
                    return {
                        "max_context_length": 131072,
                        "max_output_tokens": 1024,
                        "supports_tools": False,
                        "ultra_fast_inference": True,
                        "model_type": "safety"
                    }
            
            # Llama 4 models
            if 'llama-4' in model_lower or 'llama4' in model_lower:
                return {
                    "max_context_length": 131072,
                    "max_output_tokens": 8192,
                    "supports_tools": True,
                    "ultra_fast_inference": True
                }
            
            # Standard Llama models on Groq
            if '70b' in model_lower:
                return {
                    "max_context_length": 131072,
                    "max_output_tokens": 32768,
                    "supports_tools": True,
                    "ultra_fast_inference": True
                }
            elif '8b' in model_lower:
                return {
                    "max_context_length": 131072,
                    "max_output_tokens": 131072,  # 8b model has huge output!
                    "supports_tools": True,
                    "ultra_fast_inference": True
                }
            else:
                return {
                    "max_context_length": 131072,  # Groq default
                    "max_output_tokens": 32768,
                    "supports_tools": True,
                    "ultra_fast_inference": True
                }
        
        # === QWEN MODELS ===
        elif 'qwen' in model_lower:
            return {
                "max_context_length": 131072,
                "max_output_tokens": 40960,  # Qwen has 40k output on Groq
                "supports_tools": True,
                "supports_reasoning": True,
                "ultra_fast_inference": True
            }
        
        # === KIMI MODELS ===
        elif 'kimi' in model_lower:
            return {
                "max_context_length": 131072,
                "max_output_tokens": 16384,
                "supports_tools": True,
                "ultra_fast_inference": True
            }
        
        # === COMPOUND SYSTEMS ===
        elif 'compound' in model_lower:
            return {
                "max_context_length": 131072,
                "max_output_tokens": 8192,
                "supports_tools": True,
                "supports_reasoning": True,
                "ultra_fast_inference": True,
                "model_type": "system"
            }
        
        # === DEFAULT FOR UNKNOWN MODELS ===
        else:
            # Groq typically offers 131k context for most models
            # Be generous since Groq hosts powerful models
            return {
                "max_context_length": 131072,  # Groq standard
                "max_output_tokens": 32768,     # Conservative but generous
                "supports_tools": True,
                "ultra_fast_inference": True
            }
    
    def supports_feature(self, feature_name: str) -> bool:
        """
        Enhanced feature support with Groq-specific smart defaults.
        """
        try:
            # First try the configuration system (inherited)
            config_supports = super().supports_feature(feature_name)
            
            # If configuration gives a definitive answer, trust it
            if config_supports is not None:
                return config_supports
            
            # Configuration returned None - use Groq smart defaults
            smart_features = self._get_smart_default_features(self.model)
            supports_smart = feature_name in smart_features
            
            if supports_smart:
                log.info(f"[groq] No config for {self.model} - using smart default: supports {feature_name}")
            else:
                log.debug(f"[groq] No config for {self.model} - smart default: doesn't support {feature_name}")
            
            return supports_smart
            
        except Exception as e:
            log.warning(f"Feature support check failed for {feature_name}: {e}")
            # For Groq, be optimistic about basic features
            return feature_name in {"text", "streaming", "tools", "system_messages"}

    # ================================================================
    # GROQ-SPECIFIC MODEL INFO
    # ================================================================

    def get_model_info(self) -> Dict[str, Any]:
        """Get model info with Groq-specific enhancements"""
        # Get base info from parent
        info = super().get_model_info()
        
        # Check if this is a known Groq model
        known_info = self._get_known_model_info(self.model)
        is_production = self.model in self.GROQ_PRODUCTION_MODELS
        is_preview = self.model in self.GROQ_PREVIEW_MODELS
        
        # Override/add Groq-specific metadata
        info.update({
            "provider": "groq",
            "detected_provider": "groq",
            "model_status": "production" if is_production else "preview" if is_preview else "unknown",
            "groq_specific": {
                "ultra_fast_inference": True,
                "openai_compatible": True,
                "function_calling_notes": "May require retry fallbacks for complex tool schemas",
                "model_family": self._detect_groq_model_family(),
                "duplication_fix": "enabled",
                "optimized_for": self._get_optimization_profile(),
                "huge_context": "131k tokens standard" if not known_info or known_info.get("context", 0) > 100000 else "Limited context",
            },
            # Groq doesn't support these OpenAI parameters
            "unsupported_parameters": [
                "frequency_penalty", "presence_penalty", "logit_bias",
                "user", "n", "best_of", "top_k", "seed"
            ]
        })
        
        # Use known model info or smart defaults
        if known_info:
            info.update({
                "using_known_model": True,
                "max_context_length": known_info.get("context", 131072),
                "max_output_tokens": known_info.get("max_output", 32768),
                "model_family": known_info["family"],
                "features": list(known_info["features"])
            })
        elif not self._has_explicit_model_config(self.model):
            smart_params = self._get_smart_default_parameters(self.model)
            info.update({
                "using_smart_defaults": True,
                "smart_default_features": list(self._get_smart_default_features(self.model)),
                **smart_params
            })
        
        return info

    def _detect_groq_model_family(self) -> str:
        """Detect model family for optimizations"""
        # Check known models first
        known_info = self._get_known_model_info(self.model)
        if known_info:
            return known_info["family"]
        
        model_lower = self.model.lower()
        
        # Audio models
        if "whisper" in model_lower:
            return "whisper"
        elif "tts" in model_lower or "playai" in model_lower:
            return "tts"
        # Safety models
        elif "guard" in model_lower:
            return "llama-guard"
        # GPT-OSS models
        elif "gpt-oss" in model_lower:
            return "gpt-oss"
        # DeepSeek family
        elif "deepseek" in model_lower:
            return "deepseek"
        # Llama family
        elif "llama" in model_lower:
            if "llama-4" in model_lower or "llama4" in model_lower:
                return "llama4"
            return "llama"
        # Qwen family
        elif "qwen" in model_lower:
            return "qwen"
        # Kimi family
        elif "kimi" in model_lower:
            return "kimi"
        # Compound systems
        elif "compound" in model_lower:
            return "compound"
        # Reasoning models
        elif any(pattern in model_lower for pattern in ['reasoning', 'r1', 'distill']):
            return "reasoning"
        else:
            return "unknown"
    
    def _get_optimization_profile(self) -> str:
        """Get optimization profile for the model"""
        model_lower = self.model.lower()
        
        # Audio/TTS models
        if any(pattern in model_lower for pattern in ['whisper', 'tts', 'playai']):
            return "audio_processing"
        
        # Safety models
        if 'guard' in model_lower:
            return "safety_filtering"
        
        # Large models
        if any(pattern in model_lower for pattern in ['70b', '120b', 'large']):
            return "high_throughput"
        
        # Small/fast models
        elif any(pattern in model_lower for pattern in ['8b', '7b', '20b', 'instant', 'mini']):
            return "low_latency"
        
        # Reasoning models
        elif any(pattern in model_lower for pattern in ['reasoning', 'r1', 'distill']):
            return "deep_reasoning"
        
        # System models
        elif 'compound' in model_lower:
            return "multi_model_system"
        
        else:
            return "balanced"

    # ================================================================
    # GROQ-SPECIFIC NORMALISATION
    # ================================================================

    def _normalise_message(self, msg) -> Dict[str, Any]:
        """
        Override to ensure tool_calls is always a list, never None.
        This fixes the test failures.
        """
        result = super()._normalise_message(msg)
        
        # CRITICAL FIX: Ensure tool_calls is always a list
        if result.get("tool_calls") is None:
            result["tool_calls"] = []
        
        return result

    # ================================================================
    # GROQ-SPECIFIC ERROR HANDLING
    # ================================================================

    async def _stream_completion_async(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        name_mapping: Dict[str, str] = None,
        **kwargs: Any,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Enhanced streaming with Groq-specific error handling.
        """
        try:
            # Try normal streaming first
            async for chunk in super()._stream_completion_async(
                messages, tools, name_mapping, **kwargs
            ):
                # Ensure tool_calls is always a list in chunks
                if chunk.get("tool_calls") is None:
                    chunk["tool_calls"] = []
                yield chunk
                
        except Exception as e:
            error_str = str(e)
            
            # Handle Groq-specific function calling errors
            if "Failed to call a function" in error_str and tools:
                log.warning(f"Groq function calling failed, retrying without tools")
                
                # Retry without tools
                try:
                    async for chunk in super()._stream_completion_async(
                        messages, tools=None, name_mapping=None, **kwargs
                    ):
                        # Ensure tool_calls is empty list
                        chunk["tool_calls"] = []
                        yield chunk
                    
                    # Add note about disabled tools
                    yield {
                        "response": "\n\n[Note: Function calling disabled due to Groq limitation]",
                        "tool_calls": [],
                    }
                    
                except Exception as retry_error:
                    log.error(f"Groq streaming retry failed: {retry_error}")
                    yield {
                        "response": f"Streaming error: {str(retry_error)}",
                        "tool_calls": [],
                        "error": True
                    }
            else:
                # Re-raise for parent to handle
                raise

    async def _regular_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        name_mapping: Dict[str, str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Enhanced non-streaming with Groq-specific error handling."""
        try:
            result = await super()._regular_completion(
                messages, tools, name_mapping, **kwargs
            )
            
            # Ensure tool_calls is always a list
            if result.get("tool_calls") is None:
                result["tool_calls"] = []
            
            return result
            
        except Exception as e:
            error_str = str(e)
            
            # Handle Groq-specific function calling errors
            if "Failed to call a function" in error_str and tools:
                log.warning(f"Groq function calling failed, retrying without tools")
                
                try:
                    result = await super()._regular_completion(
                        messages, tools=None, name_mapping=None, **kwargs
                    )
                    
                    # Ensure tool_calls is empty list
                    result["tool_calls"] = []
                    
                    # Add note about disabled tools
                    original_response = result.get("response", "")
                    result["response"] = (original_response + 
                                       "\n\n[Note: Function calling disabled due to Groq limitation]")
                    
                    return result
                    
                except Exception as retry_error:
                    log.error(f"Groq retry failed: {retry_error}")
                    return {
                        "response": f"Error: {str(retry_error)}",
                        "tool_calls": [],
                        "error": True
                    }
            else:
                # Re-raise for parent to handle
                raise

    # ================================================================
    # GROQ-SPECIFIC ENHANCEMENTS
    # ================================================================

    def _enhance_messages_for_groq(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enhance messages with better instructions for Groq function calling.
        Groq models need explicit guidance for proper function calling.
        """
        if not tools or not self.supports_feature("system_messages"):
            return messages
        
        # Don't enhance for models that don't support tools
        if not self.supports_feature("tools"):
            return messages
        
        enhanced_messages = messages.copy()
        
        # Create function calling guidance tailored to model family
        model_family = self._detect_groq_model_family()
        
        if model_family in ["llama", "llama4", "qwen", "deepseek", "gpt-oss"]:
            # These models need clear, structured instructions
            function_names = [tool.get("function", {}).get("name", "unknown") for tool in tools]
            guidance = (
                f"You have access to the following functions: {', '.join(function_names)}.\n"
                "When calling functions:\n"
                "1. Use proper JSON format for arguments\n"
                "2. Ensure all required parameters are provided\n"
                "3. Use exact parameter names as specified\n"
                "4. Call functions when appropriate to help answer the user's question"
            )
        else:
            # Generic guidance for unknown models
            function_names = [tool.get("function", {}).get("name", "unknown") for tool in tools]
            guidance = f"Available functions: {', '.join(function_names)}. Use them when appropriate."
        
        # Add or enhance system message (only if system messages are supported)
        if enhanced_messages and enhanced_messages[0].get("role") == "system":
            enhanced_messages[0]["content"] = enhanced_messages[0]["content"] + "\n\n" + guidance
        else:
            enhanced_messages.insert(0, {
                "role": "system",
                "content": guidance
            })
        
        return enhanced_messages

    def _validate_tool_call_arguments(self, tool_call: Dict[str, Any]) -> bool:
        """
        Validate tool call arguments to prevent Groq function calling errors.
        """
        try:
            if "function" not in tool_call:
                return False
            
            function = tool_call["function"]
            if "arguments" not in function:
                return False
            
            # Try to parse arguments as JSON
            args = function["arguments"]
            if isinstance(args, str):
                json.loads(args)  # This will raise if invalid JSON
            elif not isinstance(args, dict):
                return False
            
            return True
            
        except (json.JSONDecodeError, TypeError, KeyError):
            return False

    async def close(self):
        """Cleanup resources"""
        # Reset name mapping from universal system
        if hasattr(self, '_current_name_mapping'):
            self._current_name_mapping = {}
        
        # Call parent cleanup
        await super().close()