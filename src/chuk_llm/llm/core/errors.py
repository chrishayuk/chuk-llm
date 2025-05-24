# chuk_llm/llm/core/errors.py
from typing import Optional, Type, Dict, Any
import asyncio
import logging
from functools import wraps
from enum import Enum

class ErrorSeverity(Enum):
    RECOVERABLE = "recoverable"      # Can retry
    PERMANENT = "permanent"          # Don't retry
    RATE_LIMITED = "rate_limited"    # Retry with backoff

class LLMError(Exception):
    """Base exception for LLM errors"""
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.PERMANENT, 
                 provider: str = None, model: str = None, **kwargs):
        super().__init__(message)
        self.severity = severity
        self.provider = provider
        self.model = model
        self.metadata = kwargs

class RateLimitError(LLMError):
    """Rate limit exceeded"""
    def __init__(self, message: str, retry_after: Optional[int] = None, **kwargs):
        super().__init__(message, ErrorSeverity.RATE_LIMITED, **kwargs)
        self.retry_after = retry_after

class ModelNotFoundError(LLMError):
    """Model not found or no access"""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, ErrorSeverity.PERMANENT, **kwargs)

class APIError(LLMError):
    """General API error"""
    def __init__(self, message: str, status_code: Optional[int] = None, **kwargs):
        super().__init__(message, ErrorSeverity.RECOVERABLE, **kwargs)
        self.status_code = status_code

class ProviderErrorMapper:
    """Maps provider-specific errors to unified LLM errors"""
    
    @staticmethod
    def map_openai_error(error: Exception, provider: str, model: str) -> LLMError:
        if hasattr(error, 'status_code'):
            if error.status_code == 429:
                return RateLimitError(
                    str(error), 
                    retry_after=getattr(error, 'retry_after', None),
                    provider=provider, 
                    model=model
                )
            elif error.status_code == 404:
                return ModelNotFoundError(str(error), provider=provider, model=model)
            else:
                return APIError(str(error), status_code=error.status_code, provider=provider, model=model)
        return LLMError(str(error), provider=provider, model=model)

def with_retry(
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    max_backoff: float = 60.0,
    retryable_severities: tuple[ErrorSeverity, ...] = (ErrorSeverity.RECOVERABLE, ErrorSeverity.RATE_LIMITED)
):
    """Decorator for automatic retry with exponential backoff"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_error = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except LLMError as e:
                    last_error = e
                    
                    # Don't retry permanent errors
                    if e.severity not in retryable_severities:
                        raise
                    
                    # Don't retry on last attempt
                    if attempt >= max_retries:
                        raise
                    
                    # Calculate backoff time
                    if e.severity == ErrorSeverity.RATE_LIMITED and hasattr(e, 'retry_after'):
                        wait_time = e.retry_after
                    else:
                        wait_time = min(backoff_factor ** attempt, max_backoff)
                    
                    logging.info(f"Retrying after {wait_time}s (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(wait_time)
                    
                except Exception as e:
                    # Convert unknown errors to LLMError
                    mapped_error = LLMError(str(e), ErrorSeverity.RECOVERABLE)
                    if mapped_error.severity in retryable_severities and attempt < max_retries:
                        last_error = mapped_error
                        wait_time = min(backoff_factor ** attempt, max_backoff)
                        await asyncio.sleep(wait_time)
                        continue
                    raise mapped_error
                    
            raise last_error
        return wrapper
    return decorator