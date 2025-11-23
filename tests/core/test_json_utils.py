"""
Tests for chuk_llm.core.json_utils
==================================

Test fast JSON utilities with all available backends.
"""

import io
import json
import sys
from typing import Any
from unittest.mock import patch

import pytest

# Import the module to ensure coverage
import chuk_llm.core.json_utils as json_utils


# =============================================================================
# Test Data
# =============================================================================

TEST_DATA = {
    "string": "hello",
    "number": 42,
    "float": 3.14,
    "boolean": True,
    "null": None,
    "array": [1, 2, 3],
    "nested": {"key": "value", "count": 10},
}

TEST_JSON_STR = '{"string":"hello","number":42,"float":3.14,"boolean":true,"null":null,"array":[1,2,3],"nested":{"key":"value","count":10}}'


# =============================================================================
# Library Detection Tests
# =============================================================================

def test_get_json_library():
    """Test getting the active JSON library name"""
    library = json_utils.get_json_library()
    assert library in ["orjson", "ujson", "stdlib"]
    assert isinstance(library, str)


def test_get_performance_info():
    """Test getting performance information"""
    info = json_utils.get_performance_info()

    assert hasattr(info, "library")
    assert hasattr(info, "orjson_available")
    assert hasattr(info, "ujson_available")
    assert hasattr(info, "speedup")

    assert info.library in ["orjson", "ujson", "stdlib"]
    assert isinstance(info.orjson_available, bool)
    assert isinstance(info.ujson_available, bool)
    assert isinstance(info.speedup, str)


# =============================================================================
# dumps() Tests
# =============================================================================

def test_dumps_basic():
    """Test basic JSON serialization"""
    result = json_utils.dumps(TEST_DATA)
    assert isinstance(result, str)

    # Verify it's valid JSON
    parsed = json.loads(result)
    assert parsed == TEST_DATA


def test_dumps_with_indent():
    """Test JSON serialization with indentation"""
    result = json_utils.dumps(TEST_DATA, indent=2)
    assert isinstance(result, str)
    assert "\n" in result or "  " in result  # Has formatting

    # Verify it's valid JSON
    parsed = json.loads(result)
    assert parsed == TEST_DATA


def test_dumps_with_sort_keys():
    """Test JSON serialization with sorted keys"""
    data = {"z": 1, "a": 2, "m": 3}
    result = json_utils.dumps(data, sort_keys=True)
    assert isinstance(result, str)

    # Verify it's valid JSON
    parsed = json.loads(result)
    assert parsed == data


def test_dumps_simple_types():
    """Test serialization of simple types"""
    assert json_utils.dumps("string") == '"string"'
    assert json_utils.dumps(42) == "42"
    assert json_utils.dumps(3.14) in ["3.14", "3.140000000000000124"]  # Float precision
    assert json_utils.dumps(True) == "true"
    assert json_utils.dumps(False) == "false"
    assert json_utils.dumps(None) == "null"
    # Different libraries format arrays differently
    result = json_utils.dumps([1, 2, 3])
    assert result in ["[1,2,3]", "[1, 2, 3]"]


# =============================================================================
# loads() Tests
# =============================================================================

def test_loads_from_string():
    """Test JSON deserialization from string"""
    result = json_utils.loads(TEST_JSON_STR)
    assert result == TEST_DATA


def test_loads_from_bytes():
    """Test JSON deserialization from bytes"""
    json_bytes = TEST_JSON_STR.encode("utf-8")
    result = json_utils.loads(json_bytes)
    assert result == TEST_DATA


def test_loads_simple_types():
    """Test deserialization of simple types"""
    assert json_utils.loads('"string"') == "string"
    assert json_utils.loads("42") == 42
    assert json_utils.loads("3.14") == 3.14
    assert json_utils.loads("true") is True
    assert json_utils.loads("false") is False
    assert json_utils.loads("null") is None
    assert json_utils.loads("[1,2,3]") == [1, 2, 3]


def test_loads_invalid_json():
    """Test error handling for invalid JSON"""
    with pytest.raises(Exception):  # JSONDecodeError or similar
        json_utils.loads("{invalid json}")


# =============================================================================
# dump() Tests (file writing)
# =============================================================================

def test_dump_to_file():
    """Test JSON serialization to file"""
    fp = io.StringIO()
    json_utils.dump(TEST_DATA, fp)

    # Read back and verify
    fp.seek(0)
    content = fp.read()
    parsed = json.loads(content)
    assert parsed == TEST_DATA


def test_dump_to_bytes_file():
    """Test JSON serialization to binary file"""
    # Use a wrapper that can handle both strings and bytes
    class BytesFileWrapper:
        def __init__(self):
            self.buffer = io.BytesIO()

        def write(self, data):
            if isinstance(data, str):
                data = data.encode("utf-8")
            self.buffer.write(data)

        def read(self):
            return self.buffer.getvalue()

        def seek(self, pos):
            self.buffer.seek(pos)

    fp = BytesFileWrapper()
    json_utils.dump(TEST_DATA, fp)

    # Read back and verify
    fp.seek(0)
    content = fp.read()
    if isinstance(content, bytes):
        content = content.decode("utf-8")
    parsed = json.loads(content)
    assert parsed == TEST_DATA


def test_dump_with_indent():
    """Test JSON serialization to file with formatting"""
    fp = io.StringIO()
    json_utils.dump(TEST_DATA, fp, indent=2)

    fp.seek(0)
    content = fp.read()
    assert "\n" in content or "  " in content  # Has formatting
    parsed = json.loads(content)
    assert parsed == TEST_DATA


def test_dump_with_sort_keys():
    """Test JSON serialization to file with sorted keys"""
    data = {"z": 1, "a": 2}
    fp = io.StringIO()
    json_utils.dump(data, fp, sort_keys=True)

    fp.seek(0)
    content = fp.read()
    parsed = json.loads(content)
    assert parsed == data


# =============================================================================
# load() Tests (file reading)
# =============================================================================

def test_load_from_string_file():
    """Test JSON deserialization from string file"""
    fp = io.StringIO(TEST_JSON_STR)
    result = json_utils.load(fp)
    assert result == TEST_DATA


def test_load_from_bytes_file():
    """Test JSON deserialization from bytes file"""
    fp = io.BytesIO(TEST_JSON_STR.encode("utf-8"))
    result = json_utils.load(fp)
    assert result == TEST_DATA


def test_load_from_empty_file():
    """Test error handling for empty file"""
    fp = io.StringIO("")
    with pytest.raises(Exception):  # JSONDecodeError or similar
        json_utils.load(fp)


# =============================================================================
# get_pydantic_json_config() Tests
# =============================================================================

def test_get_pydantic_json_config():
    """Test getting Pydantic JSON configuration"""
    config = json_utils.get_pydantic_json_config()

    assert isinstance(config, dict)
    assert "json_loads" in config
    assert "json_dumps" in config
    assert callable(config["json_loads"])
    assert callable(config["json_dumps"])

    # Test that the functions work
    test_str = '{"test": "value"}'
    loaded = config["json_loads"](test_str)
    assert loaded == {"test": "value"}

    # dumps might return str or bytes depending on backend
    dumped = config["json_dumps"]({"test": "value"})
    assert dumped is not None


# =============================================================================
# Round-trip Tests
# =============================================================================

def test_round_trip_dumps_loads():
    """Test round-trip serialization and deserialization"""
    serialized = json_utils.dumps(TEST_DATA)
    deserialized = json_utils.loads(serialized)
    assert deserialized == TEST_DATA


def test_round_trip_dump_load():
    """Test round-trip file serialization and deserialization"""
    # Write
    fp_write = io.StringIO()
    json_utils.dump(TEST_DATA, fp_write)

    # Read
    fp_write.seek(0)
    result = json_utils.load(fp_write)
    assert result == TEST_DATA


def test_round_trip_complex_data():
    """Test round-trip with complex nested data"""
    complex_data = {
        "users": [
            {"id": 1, "name": "Alice", "active": True, "tags": ["admin", "user"]},
            {"id": 2, "name": "Bob", "active": False, "tags": ["user"]},
        ],
        "metadata": {
            "version": "1.0.0",
            "timestamp": 1234567890,
            "config": {"timeout": 30, "retries": 3, "debug": False},
        },
    }

    serialized = json_utils.dumps(complex_data)
    deserialized = json_utils.loads(serialized)
    assert deserialized == complex_data


# =============================================================================
# Edge Cases
# =============================================================================

def test_empty_dict():
    """Test serialization of empty dictionary"""
    result = json_utils.dumps({})
    assert result == "{}"
    assert json_utils.loads(result) == {}


def test_empty_list():
    """Test serialization of empty list"""
    result = json_utils.dumps([])
    assert result == "[]"
    assert json_utils.loads(result) == []


def test_unicode_strings():
    """Test serialization of Unicode strings"""
    data = {"emoji": "😀", "chinese": "你好", "arabic": "مرحبا"}
    serialized = json_utils.dumps(data)
    deserialized = json_utils.loads(serialized)
    assert deserialized == data


def test_large_numbers():
    """Test serialization of large numbers"""
    data = {"small": 1, "large": 9007199254740991, "negative": -1000000}
    serialized = json_utils.dumps(data)
    deserialized = json_utils.loads(serialized)
    assert deserialized == data


def test_deeply_nested():
    """Test serialization of deeply nested structures"""
    data: dict[str, Any] = {"level": 0}
    current = data
    for i in range(1, 10):
        current["nested"] = {"level": i}
        current = current["nested"]

    serialized = json_utils.dumps(data)
    deserialized = json_utils.loads(serialized)
    assert deserialized == data


# =============================================================================
# Library-specific behavior tests
# =============================================================================

def test_orjson_fast_path():
    """Test orjson fast path (no kwargs)"""
    if json_utils.get_json_library() == "orjson":
        result = json_utils.dumps({"test": "data"})
        assert isinstance(result, str)
        assert "test" in result


def test_orjson_slow_path():
    """Test orjson slow path (with kwargs)"""
    if json_utils.get_json_library() == "orjson":
        result = json_utils.dumps({"test": "data"}, indent=2)
        assert isinstance(result, str)
        assert "test" in result


def test_orjson_loads_bytes_fast_path():
    """Test orjson loads with bytes (fast path)"""
    if json_utils.get_json_library() == "orjson":
        json_bytes = b'{"test": "data"}'
        result = json_utils.loads(json_bytes)
        assert result["test"] == "data"


def test_orjson_loads_string_slow_path():
    """Test orjson loads with string (slow path)"""
    if json_utils.get_json_library() == "orjson":
        json_str = '{"test": "data"}'
        result = json_utils.loads(json_str)
        assert result["test"] == "data"


def test_orjson_load_from_string_file():
    """Test orjson load with string content"""
    if json_utils.get_json_library() == "orjson":
        fp = io.StringIO('{"test": "data"}')
        result = json_utils.load(fp)
        assert result["test"] == "data"


def test_orjson_load_from_bytes_file():
    """Test orjson load with bytes content"""
    if json_utils.get_json_library() == "orjson":
        fp = io.BytesIO(b'{"test": "data"}')
        result = json_utils.load(fp)
        assert result["test"] == "data"


def test_orjson_dump_to_binary_file():
    """Test orjson dump to binary file (handles TypeError)"""
    if json_utils.get_json_library() == "orjson":
        fp = io.BytesIO()
        json_utils.dump({"test": "data"}, fp)
        fp.seek(0)
        content = fp.read()
        assert b"test" in content


def test_orjson_dump_to_text_file():
    """Test orjson dump to text file (handles TypeError)"""
    if json_utils.get_json_library() == "orjson":
        fp = io.StringIO()
        json_utils.dump({"test": "data"}, fp)
        fp.seek(0)
        content = fp.read()
        assert "test" in content


def test_ujson_dumps():
    """Test ujson dumps"""
    if json_utils.get_json_library() == "ujson":
        result = json_utils.dumps({"test": "data"})
        assert isinstance(result, str)


def test_ujson_loads():
    """Test ujson loads"""
    if json_utils.get_json_library() == "ujson":
        result = json_utils.loads('{"test": "data"}')
        assert result["test"] == "data"


def test_ujson_dump():
    """Test ujson dump"""
    if json_utils.get_json_library() == "ujson":
        fp = io.StringIO()
        json_utils.dump({"test": "data"}, fp)
        fp.seek(0)
        content = fp.read()
        assert "test" in content


def test_ujson_load():
    """Test ujson load"""
    if json_utils.get_json_library() == "ujson":
        fp = io.StringIO('{"test": "data"}')
        result = json_utils.load(fp)
        assert result["test"] == "data"


def test_stdlib_dumps():
    """Test stdlib dumps"""
    if json_utils.get_json_library() == "stdlib":
        result = json_utils.dumps({"test": "data"})
        assert isinstance(result, str)


def test_stdlib_loads_from_string():
    """Test stdlib loads from string"""
    if json_utils.get_json_library() == "stdlib":
        result = json_utils.loads('{"test": "data"}')
        assert result["test"] == "data"


def test_stdlib_loads_from_bytes():
    """Test stdlib loads from bytes (decodes first)"""
    if json_utils.get_json_library() == "stdlib":
        result = json_utils.loads(b'{"test": "data"}')
        assert result["test"] == "data"


def test_stdlib_dump():
    """Test stdlib dump"""
    if json_utils.get_json_library() == "stdlib":
        fp = io.StringIO()
        json_utils.dump({"test": "data"}, fp)
        fp.seek(0)
        content = fp.read()
        assert "test" in content


def test_stdlib_load_from_string_file():
    """Test stdlib load from string file"""
    if json_utils.get_json_library() == "stdlib":
        fp = io.StringIO('{"test": "data"}')
        result = json_utils.load(fp)
        assert result["test"] == "data"


def test_stdlib_load_from_bytes_file():
    """Test stdlib load from bytes file (decodes first)"""
    if json_utils.get_json_library() == "stdlib":
        fp = io.BytesIO(b'{"test": "data"}')
        result = json_utils.load(fp)
        assert result["test"] == "data"


# =============================================================================
# Pydantic config tests
# =============================================================================

def test_pydantic_config_orjson():
    """Test Pydantic config with orjson"""
    if json_utils.get_json_library() == "orjson":
        config = json_utils.get_pydantic_json_config()

        # Test loads
        loaded = config["json_loads"]('{"test": "data"}')
        assert loaded["test"] == "data"

        # Test dumps (returns bytes for orjson)
        dumped = config["json_dumps"]({"test": "data"})
        assert isinstance(dumped, bytes)
        assert b"test" in dumped


def test_pydantic_config_ujson():
    """Test Pydantic config with ujson"""
    if json_utils.get_json_library() == "ujson":
        config = json_utils.get_pydantic_json_config()

        # Test loads
        loaded = config["json_loads"]('{"test": "data"}')
        assert loaded["test"] == "data"

        # Test dumps
        dumped = config["json_dumps"]({"test": "data"})
        assert isinstance(dumped, str)
        assert "test" in dumped


def test_pydantic_config_stdlib():
    """Test Pydantic config with stdlib"""
    if json_utils.get_json_library() == "stdlib":
        config = json_utils.get_pydantic_json_config()

        # Test loads
        loaded = config["json_loads"]('{"test": "data"}')
        assert loaded["test"] == "data"

        # Test dumps
        dumped = config["json_dumps"]({"test": "data"})
        assert isinstance(dumped, str)
        assert "test" in dumped


# =============================================================================
# Additional edge cases
# =============================================================================

def test_dumps_with_both_indent_and_sort_keys():
    """Test dumps with both indent and sort_keys"""
    data = {"z": 1, "a": 2, "m": 3}
    result = json_utils.dumps(data, indent=2, sort_keys=True)
    assert isinstance(result, str)

    # Verify it's valid JSON
    parsed = json.loads(result)
    assert parsed == data


def test_dump_with_both_indent_and_sort_keys():
    """Test dump with both indent and sort_keys"""
    data = {"z": 1, "a": 2}
    fp = io.StringIO()
    json_utils.dump(data, fp, indent=2, sort_keys=True)

    fp.seek(0)
    content = fp.read()
    parsed = json.loads(content)
    assert parsed == data


def test_special_characters_in_strings():
    """Test handling of special characters"""
    data = {
        "quote": 'He said "hello"',
        "backslash": "path\\to\\file",
        "newline": "line1\nline2",
        "tab": "col1\tcol2"
    }
    serialized = json_utils.dumps(data)
    deserialized = json_utils.loads(serialized)
    assert deserialized == data


def test_mixed_type_array():
    """Test array with mixed types"""
    data = [1, "string", True, None, {"nested": "dict"}, [1, 2]]
    serialized = json_utils.dumps(data)
    deserialized = json_utils.loads(serialized)
    assert deserialized == data


def test_float_precision():
    """Test float precision handling"""
    data = {"pi": 3.141592653589793, "e": 2.718281828459045}
    serialized = json_utils.dumps(data)
    deserialized = json_utils.loads(serialized)

    # Check that values are close (accounting for precision)
    assert abs(deserialized["pi"] - data["pi"]) < 0.000001
    assert abs(deserialized["e"] - data["e"]) < 0.000001


def test_empty_string_value():
    """Test empty string as value"""
    data = {"empty": "", "not_empty": "value"}
    serialized = json_utils.dumps(data)
    deserialized = json_utils.loads(serialized)
    assert deserialized == data


def test_performance_info_fields():
    """Test all fields in performance info"""
    info = json_utils.get_performance_info()

    # Check library field
    assert info.library in ["orjson", "ujson", "stdlib"]

    # Check availability flags
    assert isinstance(info.orjson_available, bool)
    assert isinstance(info.ujson_available, bool)

    # Check speedup description
    assert info.speedup in [
        "2-3x faster than stdlib",
        "1.5-2x faster than stdlib",
        "baseline performance"
    ]


def test_library_availability_consistency():
    """Test that library selection is consistent with availability"""
    lib = json_utils.get_json_library()
    info = json_utils.get_performance_info()

    if lib == "orjson":
        assert info.orjson_available is True
    elif lib == "ujson":
        assert info.ujson_available is True
        # orjson not available (or we'd use it)
        if info.orjson_available:
            # This shouldn't happen
            pytest.fail("ujson selected but orjson is available")
    else:  # stdlib
        # Neither fast library available
        pass  # stdlib is always available


def test_module_level_variables():
    """Test module-level variables are set correctly"""
    assert hasattr(json_utils, '_json_lib')
    assert hasattr(json_utils, '_orjson_available')
    assert hasattr(json_utils, '_ujson_available')

    assert json_utils._json_lib in ["orjson", "ujson", "stdlib"]
    assert isinstance(json_utils._orjson_available, bool)
    assert isinstance(json_utils._ujson_available, bool)
