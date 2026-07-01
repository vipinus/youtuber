from src.config import Config
from src.llm import FakeLLM
from src.context import Ctx
from src.util.jobs import write_json
from src.stages.s5_subs import run, build_lines, ass_time, OUTPUT, INPUT


def test_ass_time():
    assert ass_time(0) == "0:00:00.00"
    assert ass_time(3.25) == "0:00:03.25"
    assert ass_time(3661.5) == "1:01:01.50"


def test_build_lines_breaks_on_punct_and_length():
    timing = {"segments": [{"words": [
        {"text": "今", "start_sec": 0.0, "dur_sec": 0.3},
        {"text": "天", "start_sec": 0.3, "dur_sec": 0.3},
        {"text": "。", "start_sec": 0.6, "dur_sec": 0.3},   # 句读 → 断行
        {"text": "很", "start_sec": 0.9, "dur_sec": 0.3},
        {"text": "好", "start_sec": 1.2, "dur_sec": 0.3},
    ]}]}
    lines = build_lines(timing, max_chars=16)
    assert len(lines) == 2
    assert lines[0][2] == "今天。"
    assert lines[0][0] == 0.0
    assert round(lines[0][1], 1) == 0.9        # 末字(。) start0.6+dur0.3
    assert lines[1][2] == "很好"


def test_build_lines_breaks_on_length():
    chars = "一二三四五六七八九十甲乙丙丁戊己庚辛壬癸"  # 20 字，无标点
    words = [{"text": ch, "start_sec": round(i * 0.3, 3), "dur_sec": 0.3}
             for i, ch in enumerate(chars)]
    lines = build_lines({"segments": [{"words": words}]}, max_chars=16)
    assert len(lines) == 2
    assert lines[0][2] == chars[:16]          # 前 16 字断行
    assert lines[1][2] == chars[16:]          # 余 4 字
    assert round(lines[0][0], 3) == 0.0       # 行首 = 首字 start
    assert round(lines[0][1], 3) == 4.8       # 行尾 = 第16字(idx15) start+dur = 4.5+0.3
    assert round(lines[1][0], 3) == 4.8       # 第2行首 = 第17字(idx16) start


def test_run_writes_ass(tmp_path):
    ctx = Ctx(slug="c", config=Config({}), root=tmp_path, llm=FakeLLM())
    write_json(ctx.job, INPUT, {"segments": [{"words": [
        {"text": "飞", "start_sec": 0.0, "dur_sec": 0.3},
        {"text": "碟", "start_sec": 0.3, "dur_sec": 0.3},
    ]}]})
    run(ctx)
    txt = (ctx.job / OUTPUT).read_text(encoding="utf-8")
    assert "[Script Info]" in txt
    assert "[Events]" in txt
    assert txt.count("Dialogue:") == 1
    assert "飞碟" in txt
