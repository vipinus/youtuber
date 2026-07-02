"""P7 T17 — F5-TTS 声音克隆配音(scripts/zion_clone_tts.py)的纯函数部分离线单测。

⚠️ 范围声明：真正的克隆合成(main()/tts.infer)要吃 f5_tts/torch，这台开发机
没装（也不该装——那是 zion clone-venv 的依赖），所以只单测零依赖的纯函数
pick_reference()（参考段挑选规则），并用 ast 静态校验模块顶层不 import
f5_tts/torch/网络库（保证本地 `import scripts.zion_clone_tts` 不触发任何重
依赖/网络）。_slice_reference() 用真实 ffmpeg 做一个轻量冒烟测试（ffmpeg 是
纯 CLI 工具，不算"重依赖"，其它测试文件里也大量直接用它）。main() 的
F5TTS()/tts.infer 调用本身需要在 zion 上做真实冒烟跑（GPU，orchestrator 稍后
补跑）。
"""
from __future__ import annotations

import ast
import inspect

import scripts.zion_clone_tts as zct


def _seg(id_, start, end, en="text", zh="", gender=None):
    return {"id": id_, "start": start, "end": end, "en": en, "zh": zh, "gender": gender}


# ---------------------------------------------------------------------------
# pick_reference — 规则 1：本段自己时长落在 [3,15]s 就用它自己
# ---------------------------------------------------------------------------

def test_pick_reference_prefers_self_when_duration_in_range():
    segments = [_seg(0, 0.0, 5.0, en="hello there")]      # dur=5s，在区间内
    assert zct.pick_reference(segments, 0) is segments[0]


def test_pick_reference_self_at_lower_boundary_3s_is_used():
    segments = [_seg(0, 0.0, 3.0)]                        # dur==MIN_REF_SEC(含边界)
    assert zct.pick_reference(segments, 0) is segments[0]


def test_pick_reference_self_at_upper_boundary_15s_is_used():
    segments = [_seg(0, 0.0, 15.0)]                       # dur==MAX_REF_SEC(含边界)
    assert zct.pick_reference(segments, 0) is segments[0]


# ---------------------------------------------------------------------------
# pick_reference — 规则 2：本段太短/太长 -> 挑同 gender 且时长达标、离本段
# start 最近的其它段
# ---------------------------------------------------------------------------

def test_pick_reference_falls_back_to_nearest_same_gender_segment():
    segments = [
        _seg(0, 0.0, 1.0, gender="male"),          # self: 太短(1s)
        _seg(1, 2.0, 10.0, gender="male"),          # dur=8s 达标，同性别，|2-0|=2 最近
        _seg(2, 11.0, 20.0, gender="female"),       # dur=9s 达标，但性别不同
        _seg(3, 21.0, 30.0, gender="male"),         # dur=9s 达标，同性别，但更远(|21-0|=21)
    ]
    assert zct.pick_reference(segments, 0) is segments[1]


def test_pick_reference_ignores_gender_when_self_gender_unknown():
    """自身 gender 是 None 时，"同性别"这条规则不适用(None==None 不算真信号)
    ——直接进入"不管性别、只要时长达标、最近"这条。"""
    segments = [
        _seg(0, 0.0, 1.0, gender=None),             # self: 太短，性别未知
        _seg(1, 2.0, 10.0, gender="male"),          # dur=8s 达标，最近
        _seg(2, 50.0, 60.0, gender="female"),       # dur=10s 达标，但更远
    ]
    assert zct.pick_reference(segments, 0) is segments[1]


def test_pick_reference_falls_back_to_any_gender_when_no_same_gender_eligible():
    segments = [
        _seg(0, 0.0, 1.0, gender="female"),         # self: 太短
        _seg(1, 5.0, 13.0, gender="male"),          # dur=8s 达标，最近(|5-0|=5)
        _seg(2, 50.0, 59.0, gender="male"),         # dur=9s 达标，但更远
        # 全片没有一段 gender=="female" 且时长达标
    ]
    assert zct.pick_reference(segments, 0) is segments[1]


# ---------------------------------------------------------------------------
# pick_reference — 兜底：没有任何达标段(3-15s) -> 挑全片时长最长的一段
# ---------------------------------------------------------------------------

def test_pick_reference_ultimate_fallback_is_longest_segment():
    segments = [
        _seg(0, 0.0, 1.0),        # self: dur=1s(太短)
        _seg(1, 2.0, 2.5),        # dur=0.5s(太短)
        _seg(2, 3.0, 20.0),       # dur=17s(太长)，但全片最长
    ]
    assert zct.pick_reference(segments, 0) is segments[2]


def test_pick_reference_empty_segments_returns_none():
    assert zct.pick_reference([], 0) is None


# ---------------------------------------------------------------------------
# _slice_reference — 真实 ffmpeg 冒烟：切片可解码 + 按 id 缓存复用(不重复切)
# ---------------------------------------------------------------------------

def test_slice_reference_produces_decodable_wav_and_caches_by_id(tmp_path):
    import subprocess

    vocals = tmp_path / "vocals.mp3"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono",
         "-t", "3", str(vocals)],
        check=True, capture_output=True,
    )
    ref_seg = _seg(7, 0.0, 2.0)
    cache: dict[int, object] = {}
    tmp_dir = tmp_path / "_ref_slices"
    tmp_dir.mkdir()

    out1 = zct._slice_reference(vocals, ref_seg, cache, tmp_dir)
    assert out1.exists()
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type",
         "-of", "csv=p=0", str(out1)],
        capture_output=True, text=True, check=True,
    )
    assert "audio" in probe.stdout

    mtime_before = out1.stat().st_mtime_ns
    out2 = zct._slice_reference(vocals, ref_seg, cache, tmp_dir)
    assert out2 == out1
    assert out2.stat().st_mtime_ns == mtime_before          # 缓存命中，没有重新切


# ---------------------------------------------------------------------------
# 离线安全：模块顶层不 import f5_tts/torch/网络库——只在 main() 里惰性 import，
# 保证本地(zion 之外)静态检查/单测这个脚本文件完全不需要装它们。
# ---------------------------------------------------------------------------

def test_module_import_does_no_heavy_or_network_imports_at_top_level():
    banned = {
        "torch", "torchaudio", "f5_tts",
        "requests", "urllib3", "httpx", "paramiko", "fabric",
    }
    tree = ast.parse(inspect.getsource(zct))
    for node in tree.body:
        if isinstance(node, ast.Import):
            assert all(a.name.split(".")[0] not in banned for a in node.names)
        if isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] not in banned


def test_main_usage_error_does_not_import_f5_tts(monkeypatch, capsys):
    """argv 参数不足时应在触发 f5_tts import 之前就打印用法并返回非零——离线
    单测这个分支不该报 ModuleNotFoundError（f5_tts 根本没装）。"""
    rc = zct.main(["zion_clone_tts.py", "only_one_arg"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "用法" in err


def test_normalize_zh_numbers_real_cases():
    """真片实句:数字必须转汉字,否则 F5 被英文参考带偏念英文数字(用户实听发现)。"""
    f = zct.normalize_zh_numbers
    assert f("它的直径有305米。") == "它的直径有三百零五米。"
    assert f("大约1000英尺") == "大约一千英尺"
    assert f("1975年就开始经营了") == "一九七五年就开始经营了"
    assert f("每24小时算一天") == "每二十四小时算一天"
    assert f("以50微米的精确度") == "以五十微米的精确度"
    assert f("每16天才能收到一次数据") == "每十六天才能收到一次数据"
    assert f("速度是3.5倍") == "速度是三点五倍"
    assert f("占50%") == "占百分之五十"
    assert f("在离我们260英里高空") == "在离我们二百六十英里高空"
    assert f("没有数字的句子。") == "没有数字的句子。"


def test_int_to_cn_edges():
    f = zct._int_to_cn
    assert f(0) == "零"
    assert f(10) == "十"
    assert f(12) == "十二"
    assert f(110) == "一百一十"
    assert f(305) == "三百零五"
    assert f(1000) == "一千"
    assert f(10086) == "一万零八十六"
    assert f(120000) == "十二万"
