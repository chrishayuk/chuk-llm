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
