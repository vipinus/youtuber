from src.util.jobs import write_json, read_json, exists


def test_write_read_roundtrip(tmp_path):
    job = tmp_path / "roswell-1947"
    assert exists(job, "topic.json") is False
    write_json(job, "topic.json", {"slug": "roswell-1947", "年": 1947})
    assert exists(job, "topic.json") is True
    assert read_json(job, "topic.json") == {"slug": "roswell-1947", "年": 1947}


def test_write_json_is_utf8_unescaped(tmp_path):
    job = tmp_path / "x"
    p = write_json(job, "a.json", {"t": "飞碟"})
    assert "飞碟" in p.read_text(encoding="utf-8")  # 中文未被转义成 \uXXXX


def test_write_json_atomic_no_temp_left(tmp_path):
    from src.util.jobs import write_json
    job = tmp_path / "x"
    write_json(job, "a.json", {"k": 1})
    names = sorted(p.name for p in job.iterdir())
    assert names == ["a.json"]        # 无残留临时文件
