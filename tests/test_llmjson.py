import pytest
from src.util.llmjson import extract_json


def test_extract_json_bare():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_fenced_and_prose():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('好的：{"a": 2} 完') == {"a": 2}


def test_extract_json_no_object_raises():
    with pytest.raises(ValueError):
        extract_json("这里没有 JSON 对象")
