"""
Fast JSON Utilities
===================

Uses the fastest available JSON library:
1. orjson (Rust-based, fastest)
2. ujson (C-based, fast)
3. stdlib json (fallback)
"""

import json as _stdlib_json
from typing import Any

# Try to import fast JSON libraries
_json_lib = "stdlib"

try:
    import orjson

    _json_lib = "orjson"
    _orjson_available = True
except ImportError:
    _orjson_available = False

try:
    import ujson  # type: ignore[import-untyped]

    _ujson_available = True
    if _json_lib == "stdlib":
        _json_lib = "ujson"
except ImportError:
    _ujson_available = False


# Pre-compile the dumps function to eliminate runtime checks
if _orjson_available:

    def dumps(obj: Any, **kwargs: Any) -> str:
        """
        Serialize object to JSON string using orjson (fastest).

        Args:
            obj: Object to serialize
            **kwargs: Options (indent, sort_keys)

        Returns:
            JSON string
        """
        # Fast path: no options
        if not kwargs:
            return orjson.dumps(obj).decode("utf-8")

        # Slow path: with options
        option = 0
        if kwargs.get("indent"):
            option |= orjson.OPT_INDENT_2
        if kwargs.get("sort_keys"):
            option |= orjson.OPT_SORT_KEYS
        return orjson.dumps(obj, option=option).decode("utf-8")

elif _ujson_available:

    def dumps(obj: Any, **kwargs: Any) -> str:
        """
        Serialize object to JSON string using ujson (fast).

        Args:
            obj: Object to serialize
            **kwargs: Library-specific options

        Returns:
            JSON string
        """
        return ujson.dumps(obj, **kwargs)

else:

    def dumps(obj: Any, **kwargs: Any) -> str:
        """
        Serialize object to JSON string using stdlib json (fallback).

        Args:
            obj: Object to serialize
            **kwargs: Library-specific options

        Returns:
            JSON string
        """
        return _stdlib_json.dumps(obj, **kwargs)


# Pre-compile the loads function to eliminate runtime checks
if _orjson_available:

    def loads(s: str | bytes) -> Any:
        """
        Deserialize JSON to Python object using orjson (fastest).

        Args:
            s: JSON string or bytes

        Returns:
            Deserialized Python object
        """
        # Fast path: already bytes
        if isinstance(s, bytes):
            return orjson.loads(s)
        # Slow path: need to encode
        return orjson.loads(s.encode("utf-8"))

elif _ujson_available:

    def loads(s: str | bytes) -> Any:
        """
        Deserialize JSON to Python object using ujson (fast).

        Args:
            s: JSON string or bytes

        Returns:
            Deserialized Python object
        """
        return ujson.loads(s)

else:

    def loads(s: str | bytes) -> Any:
        """
        Deserialize JSON to Python object using stdlib json (fallback).

        Args:
            s: JSON string or bytes

        Returns:
            Deserialized Python object
        """
        if isinstance(s, bytes):
            s = s.decode("utf-8")
        return _stdlib_json.loads(s)


# Pre-compile the dump function to eliminate runtime checks
if _orjson_available:

    def dump(obj: Any, fp: Any, **kwargs: Any) -> None:
        """
        Serialize object to JSON file using orjson (fastest).

        Args:
            obj: Object to serialize
            fp: File-like object
            **kwargs: Options (indent, sort_keys)
        """
        option = 0
        if kwargs.get("indent"):
            option |= orjson.OPT_INDENT_2
        if kwargs.get("sort_keys"):
            option |= orjson.OPT_SORT_KEYS
        data = orjson.dumps(obj, option=option)
        # Handle both text and binary file objects
        try:
            fp.write(data)
        except TypeError:
            fp.write(data.decode("utf-8"))

elif _ujson_available:

    def dump(obj: Any, fp: Any, **kwargs: Any) -> None:
        """
        Serialize object to JSON file using ujson (fast).

        Args:
            obj: Object to serialize
            fp: File-like object
            **kwargs: Library-specific options
        """
        fp.write(ujson.dumps(obj, **kwargs))

else:

    def dump(obj: Any, fp: Any, **kwargs: Any) -> None:
        """
        Serialize object to JSON file using stdlib json (fallback).

        Args:
            obj: Object to serialize
            fp: File-like object
            **kwargs: Library-specific options
        """
        _stdlib_json.dump(obj, fp, **kwargs)


# Pre-compile the load function to eliminate runtime checks
if _orjson_available:

    def load(fp: Any) -> Any:
        """
        Deserialize JSON file to Python object using orjson (fastest).

        Args:
            fp: File-like object

        Returns:
            Deserialized Python object
        """
        content = fp.read()
        if isinstance(content, str):
            content = content.encode("utf-8")
        return orjson.loads(content)

elif _ujson_available:

    def load(fp: Any) -> Any:
        """
        Deserialize JSON file to Python object using ujson (fast).

        Args:
            fp: File-like object

        Returns:
            Deserialized Python object
        """
        return ujson.loads(fp.read())

else:

    def load(fp: Any) -> Any:
        """
        Deserialize JSON file to Python object using stdlib json (fallback).

        Args:
            fp: File-like object

        Returns:
            Deserialized Python object
        """
        content = fp.read()
        if isinstance(content, bytes):
            content = content.decode("utf-8")
        return _stdlib_json.loads(content)


def get_json_library() -> str:
    """
    Get the name of the JSON library being used.

    Returns:
        "orjson", "ujson", or "stdlib"
    """
    return _json_lib


# Export speedup information
def get_performance_info():  # type: ignore[no-untyped-def]
    """
    Get information about JSON performance optimizations.

    Returns:
        Performance info model
    """
    from .api_models import PerformanceInfo

    speedup_map = {
        "orjson": "2-3x faster than stdlib",
        "ujson": "1.5-2x faster than stdlib",
        "stdlib": "baseline performance",
    }

    return PerformanceInfo(
        library=_json_lib,  # type: ignore[arg-type]
        orjson_available=_orjson_available,
        ujson_available=_ujson_available,
        speedup=speedup_map.get(_json_lib, "unknown"),
    )


# Pydantic V2 custom serializer configuration
def get_pydantic_json_config() -> dict[str, Any]:
    """
    Get Pydantic V2 model_config for fast JSON serialization.

    Returns:
        Config dict for Pydantic models
    """
    if _orjson_available:
        # Use orjson for model serialization
        return {
            "json_loads": loads,
            "json_dumps": lambda obj, **kwargs: dumps(obj, **kwargs).encode("utf-8"),
        }
    elif _ujson_available:
        return {
            "json_loads": loads,
            "json_dumps": lambda obj, **kwargs: dumps(obj, **kwargs),
        }
    else:
        # Use stdlib
        return {
            "json_loads": _stdlib_json.loads,
            "json_dumps": _stdlib_json.dumps,
        }


if __name__ == "__main__":
    # Quick benchmark
    print(f"Using JSON library: {_json_lib}")
    print(f"Performance info: {get_performance_info()}")

    # Test serialization
    test_obj = {"test": "data", "number": 42, "nested": {"key": "value"}}
    json_str = dumps(test_obj)
    print(f"\nSerialized: {json_str}")

    # Test deserialization
    loaded = loads(json_str)
    print(f"Deserialized: {loaded}")
