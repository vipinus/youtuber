"""P7 T9 — 性别匹配配音：zion 端基频检测脚本(scripts/zion_gpu_asr.py)的纯函数
部分离线单测。

⚠️ 范围声明：真正的基频提取(detect_genders/_segment_median_f0)要吃
torch/torchaudio 张量，这台开发机没装 torch（也不该装——那是 zion GPU 机的
依赖），所以只单测零依赖的纯函数 classify_f0（阈值分类），并用 ast 静态校验
模块顶层不 import torch/torchaudio（保证本地 `import scripts.zion_gpu_asr`
不触发任何重依赖/网络）。detect_genders 本身的行为需要在 zion 上做一次真实
冒烟跑（GPU 目前忙，orchestrator 稍后补跑，见任务报告）。
"""
from __future__ import annotations

import ast
import inspect

import scripts.zion_gpu_asr as zga


# ---------------------------------------------------------------------------
# classify_f0 — 纯函数，阈值分类
# ---------------------------------------------------------------------------

def test_classify_f0_below_male_threshold_is_male():
    assert zga.classify_f0(120.0) == "male"
    assert zga.classify_f0(zga.MALE_MAX_HZ - 0.1) == "male"


def test_classify_f0_above_female_threshold_is_female():
    assert zga.classify_f0(220.0) == "female"
    assert zga.classify_f0(zga.FEMALE_MIN_HZ + 0.1) == "female"


def test_classify_f0_ambiguous_band_is_none():
    """[145, 175] 模糊地带（含边界）——宁可不猜，绝不瞎猜。"""
    assert zga.classify_f0(zga.MALE_MAX_HZ) is None      # == 145 -> 不算 male
    assert zga.classify_f0(zga.FEMALE_MIN_HZ) is None    # == 175 -> 不算 female
    assert zga.classify_f0(160.0) is None


def test_classify_f0_none_input_is_none():
    """静音/时长太短/可信帧不足时上游传 None 进来，必须原样透传 None。"""
    assert zga.classify_f0(None) is None


# ---------------------------------------------------------------------------
# 离线安全：模块顶层不 import torch/torchaudio/网络库——只在函数体内惰性 import，
# 保证本地(zion 之外)静态检查/单测这个脚本文件完全不需要装 torch。
# ---------------------------------------------------------------------------

def test_module_import_does_no_heavy_or_network_imports_at_top_level():
    banned = {
        "torch", "torchaudio", "faster_whisper",
        "requests", "urllib3", "httpx", "paramiko", "fabric",
    }
    tree = ast.parse(inspect.getsource(zga))
    for node in tree.body:
        if isinstance(node, ast.Import):
            assert all(a.name.split(".")[0] not in banned for a in node.names)
        if isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] not in banned


def test_separate_returns_vocals_and_no_vocals_paths_shape():
    """静态校验 separate() 的返回值形状约定（vocals, no_vocals）没有回归——
    不真跑 demucs，只检查函数签名/源码里两个路径变量都在。"""
    src = inspect.getsource(zga.separate)
    assert "vocals.mp3" in src
    assert "no_vocals.mp3" in src
    assert "return vocals, no_vocals" in src
