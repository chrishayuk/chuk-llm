# chuk_llm/llm/core/types.py
import logging
from typing import TypedDict, List, Optional, Union, Any, Dict
from pydantic import BaseModel, Field, field_validator, ConfigDict
import json

class ToolCallFunction(BaseModel):
    name: str
    arguments: str  # JSON string
    
    @field_validator('arguments')
    @classmethod
    def validate_arguments(cls, v):
        if isinstance(v, dict):
            return json.dumps(v)
        try:
            json.loads(v)  # Validate it's valid JSON
            return v
        except json.JSONDecodeError:
            return "{}"  # Fallback to empty object

class ToolCall(BaseModel):
    model_config = ConfigDict(extra='forbid')
    
    id: str
    type: str = "function"
    function: ToolCallFunction

class LLMResponse(BaseModel):
    model_config = ConfigDict(extra='forbid')
    
    response: Optional[str] = None
    tool_calls: List[ToolCall] = Field(default_factory=list)
    error: bool = False
    error_message: Optional[str] = None
    
    @field_validator('response', 'tool_calls', mode='before')
    @classmethod
    def validate_response_xor_tools(cls, v, info):
        # At least one of response or tool_calls should have content
        values = info.data if hasattr(info, 'data') else {}
        response = values.get('response')
        tool_calls = values.get('tool_calls', [])
        
        if not response and not tool_calls and not values.get('error', False):
            raise ValueError("Response must have either text content or tool calls")
        return v
        
class StreamChunk(LLMResponse):
    """Streaming chunk with metadata"""
    chunk_index: Optional[int] = None
    is_final: bool = False
    timestamp: Optional[float] = None
    
    @field_validator('response', mode='before')
    @classmethod
    def allow_empty_chunks(cls, v):
        # Streaming chunks can be empty
        return v

class ResponseValidator:
    """Validates and normalizes LLM responses"""
    
    @staticmethod
    def validate_response(raw_response: Dict[str, Any], is_streaming: bool = False) -> Union[LLMResponse, StreamChunk]:
        """Validate and convert raw response to typed model"""
        try:
            if is_streaming:
                return StreamChunk(**raw_response)
            else:
                return LLMResponse(**raw_response)
        except Exception as e:
            # Return error response if validation fails
            error_class = StreamChunk if is_streaming else LLMResponse
            return error_class(
                response=None,
                tool_calls=[],
                error=True,
                error_message=f"Response validation failed: {str(e)}"
            )
    
    @staticmethod
    def normalize_tool_calls(raw_tool_calls: List[Any]) -> List[ToolCall]:
        """Normalize tool calls from different providers"""
        normalized = []
        
        for tc in raw_tool_calls:
            try:
                if isinstance(tc, dict):
                    # Handle different provider formats
                    if 'function' in tc:
                        # OpenAI/Anthropic format
                        normalized.append(ToolCall(**tc))
                    elif 'name' in tc:
                        # Alternative format - convert to standard
                        normalized.append(ToolCall(
                            id=tc.get('id', f"call_{len(normalized)}"),
                            type="function",
                            function=ToolCallFunction(
                                name=tc['name'],
                                arguments=tc.get('arguments', '{}')
                            )
                        ))
            except Exception as e:
                logging.warning(f"Failed to normalize tool call: {tc}, error: {e}")
                continue
                
        return normalized