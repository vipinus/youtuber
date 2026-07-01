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


def test_run_uses_script_text_with_punctuation(tmp_path):
    from src.config import Config
    from src.llm import FakeLLM
    from src.context import Ctx
    from src.util.jobs import write_json
    from src.stages.s5_subs import run, OUTPUT
    ctx = Ctx(slug="c", config=Config({}), root=tmp_path, llm=FakeLLM())
    # script 原文有标点；timing 的 words 无标点（模拟真机 edge-tts）
    write_json(ctx.job, "script.json", {"segments": [
        {"id": 1, "text": "今天，很好。"},
    ]})
    write_json(ctx.job, "timing.json", {"segments": [
        {"id": 1, "words": [
            {"text": "今天", "start_sec": 0.0, "dur_sec": 0.6},
            {"text": "很好", "start_sec": 0.6, "dur_sec": 0.6},
        ]},
    ]})
    run(ctx)
    txt = (ctx.job / OUTPUT).read_text(encoding="utf-8")
    assert "，" in txt and "。" in txt        # 标点从 script 原文保留


def test_split_cues_full_sentences_and_trailing_punct():
    """句末标点各成一条完整句；标点留句尾，不以标点开头(用户反馈:标点不分两段/整句表达)。"""
    from src.stages.s5_subs import _split_cues
    text = "然而，军方很快改口。这到底是飞碟，还是气球？"
    cues = _split_cues(text, max_chars=16)
    assert cues == ["然而，军方很快改口。", "这到底是飞碟，还是气球？"]
    assert "".join(cues) == text                       # 不丢字不加字
    for c in cues:
        assert c[0] not in "，。！？、；："              # 不以标点开头
        assert c[-1] in "。！？"                         # 整句以句末标点收尾


def test_split_cues_long_clause_breaks_at_comma_not_midword():
    """超长句在逗号处断、逗号留前句尾；不在词中间硬切(用户反馈:完整的词不分两句)。"""
    from src.stages.s5_subs import _split_cues
    text = "为什么军方最初会如此肯定地宣布发现了飞碟，随后却又改口。"
    cues = _split_cues(text, max_chars=16)
    assert len(cues) == 2
    assert cues[0].endswith("，")                        # 断在逗号，标点跟前句
    assert cues[1] == "随后却又改口。"
    assert "".join(cues) == text
    assert all(c[0] not in "，。！？、；：" for c in cues)


def test_split_cues_short_fragment_joins_next():
    """短引导词(然而，)不单独成条，并入后句避免碎片。"""
    from src.stages.s5_subs import _split_cues
    cues = _split_cues("然而，事情并不简单。", max_chars=16)
    assert cues == ["然而，事情并不简单。"]


def test_seg_lines_one_cue_maps_full_span():
    from src.stages.s5_subs import seg_lines
    words = [{"text": "军方", "start_sec": 0.0, "dur_sec": 0.5},
             {"text": "宣布", "start_sec": 0.5, "dur_sec": 0.5},
             {"text": "发现", "start_sec": 1.0, "dur_sec": 0.5},
             {"text": "飞碟", "start_sec": 1.5, "dur_sec": 0.5}]
    lines = seg_lines("军方宣布发现飞碟。", words, max_chars=16)
    assert len(lines) == 1
    assert lines[0][2] == "军方宣布发现飞碟。"
    assert lines[0][0] == 0.0 and round(lines[0][1], 1) == 2.0
