"""zion 端 F5-TTS one-shot 跨语种声音克隆配音脚本（P7 T17）。

**必须**经 `~/clone-py` 调用，不能直接跑 clone-venv 里的裸 python——同
zion_gpu_asr.py 的坑：wrapper 先设好 LD_LIBRARY_PATH 再 exec
`~/clone-venv/bin/python`，直接跑裸 venv python 加载不了 torch 的底层 CUDA
库（详见 memory `zion-gpu-asr-setup`）。

用法（在 zion 上）：
    ~/clone-py zion_clone_tts.py <zh_script.json> <vocals.mp3> <out_dir>

做的事（F5 python API 已在 zion 上端到端实测验通，见 P7 T17 计划）：
  1. 读 zh_script.json 的段列表（id/start/end/en/zh[/gender]，d3_translate 的
     产物）。
  2. `F5TTS()` 只构造一次（模型加载 ~30-45s），后续每段复用同一进程调
     `tts.infer(...)`（约 0.4×RTF/段）。**绝不**逐段 shell out
     `f5-tts_infer-cli`——那样每段都要重新加载模型，真跑验证过慢到不可接受。
  3. 每段的克隆参考音频 = 该纪录片说话人自己的原声切片（demucs 分离出的干净
     人声 vocals.mp3，按参考段的 start/end 用 ffmpeg 切出）。参考段的挑选规则
     见纯函数 pick_reference()（3–15s 时长门槛：太短 F5 抓不住音色，太长拖慢
     推理且容易带进背景噪声）。参考文本(ref_text)必须是参考段自己的英文原文
     ——放错了文本会让 F5 把音频和文本对不上，克隆质量崩。
  4. 逐段合成中文 gen_text -> out_dir/n{id:03d}.wav。单段失败（网络抖动/某段
     参考音频损坏等）只打印 warn 并跳过，不中止整批——下游 d4_tts 发现某段
     没有对应 wav 时会自动回退非克隆引擎补上，不能让一段坏的拖垮整部片子。
  5. 每 20 段打印一次进度，全部跑完打印完成标记 "ZION_CLONE_DONE" 供编排侧
     探测成功与否（同 zion_gpu_asr.py 的 "ZION_DONE" 手法）。

保持依赖精简：只用 stdlib + f5_tts（惰性 import，只在 main() 里真正要用时才
import）——本地(zion 之外)单测/import 这个文件不需要装 f5_tts/torch，只测零
依赖的纯函数 pick_reference()。
"""
from __future__ import annotations

import json
import subprocess
import sys
import traceback
from pathlib import Path

MIN_REF_SEC = 3.0      # 参考音频太短 F5 抓不住音色特征
MAX_REF_SEC = 15.0     # 太长拖慢单段推理，且更容易夹带背景噪声/换气声

# -- 数字转汉字(纯函数,零依赖)——F5 双语模型遇英文参考音色会把阿拉伯数字念成
# 英文(用户实听发现"305米"念成 three hundred five 米)。喂 TTS 前把数字规范化
# 成汉字;字幕不动(屏显阿拉伯数字更清爽,只在这里改 gen_text)。 --------------
import re

_CN_DIGITS = "零一二三四五六七八九"
_CN_UNITS = ["", "十", "百", "千"]
_CN_GROUPS = ["", "万", "亿"]


def _int_to_cn(n: int) -> str:
    """整数→中文数词(到亿级)。10-19 说"十X"不说"一十X"。"""
    if n == 0:
        return "零"
    groups = []
    while n > 0:
        groups.append(n % 10000)
        n //= 10000
    parts = []
    for gi in range(len(groups) - 1, -1, -1):
        g = groups[gi]
        if g == 0:
            if parts and not parts[-1].endswith("零"):
                parts.append("零")
            continue
        s, zero_pending = "", False
        for pos in range(3, -1, -1):
            d = (g // (10 ** pos)) % 10
            if d == 0:
                if s:
                    zero_pending = True
                continue
            if zero_pending:
                s += "零"
                zero_pending = False
            s += _CN_DIGITS[d] + _CN_UNITS[pos]
        if gi == len(groups) - 1 and s.startswith("一十"):
            s = s[1:]                       # 12 -> 十二
        if parts and g < 1000:              # 跨组前导零:10086 -> 一万零八十六
            s = "零" + s
        parts.append(s + _CN_GROUPS[gi])
    out = "".join(parts)
    return out.rstrip("零") or "零"


def _digits_cn(s: str) -> str:
    return "".join(_CN_DIGITS[int(c)] for c in s)


def normalize_zh_numbers(text: str) -> str:
    """中文旁白里的数字→汉字读法:年份逐位(1975年→一九七五年)、百分比、
    小数(3.5→三点五)、普通整数(305→三百零五)。顺序敏感:年份→百分比→小数→整数。"""
    text = re.sub(r"(\d{4})(?=年)", lambda m: _digits_cn(m.group(1)), text)
    text = re.sub(r"(\d+)(?:\.(\d+))?%",
                  lambda m: "百分之" + _int_to_cn(int(m.group(1)))
                  + (("点" + _digits_cn(m.group(2))) if m.group(2) else ""), text)
    text = re.sub(r"(\d+)\.(\d+)",
                  lambda m: _int_to_cn(int(m.group(1))) + "点" + _digits_cn(m.group(2)), text)
    text = re.sub(r"\d+", lambda m: _int_to_cn(int(m.group(0))), text)
    return text


def _dur(seg: dict) -> float:
    return float(seg.get("end", 0.0)) - float(seg.get("start", 0.0))


def pick_reference(segments: list[dict], idx: int) -> dict | None:
    """纯函数、零依赖（不碰音频/IO/torch）——给 segments[idx] 挑克隆参考段。

    规则（已与用户对齐，见 P7 T17 计划）：
      1. 优先本段自己：时长落在 [MIN_REF_SEC, MAX_REF_SEC] 就直接用它自己
         （文本天然对得上：ref_text 用它自己的 en 字段）。
      2. 否则：从其它段里选"时长达标 + gender 与本段相同"里，按 |start| 距离
         本段最近的一段（同说话人的音色最接近，参考文本也现成）。本段
         gender 未知(None)时这条不适用，直接进第 3 条（None 跟 None"凑巧同
         性别"没有实际信号，不该被当成匹配）。
      3. 再否则：不管 gender，只要时长达标，一样按 |start| 距离最近挑。
      4. 兜底：以上都挑不出(全片没有一段落在 3–15s 区间)——退化成整部片子里
         时长最长的一段，不管是否达标，也好过没有参考（宁可克隆音色不够
         精准，也不能完全没有参考音频）。
    segments 为空列表时返回 None（调用方据此判断整段跳过）。
    """
    if not segments:
        return None

    self_seg = segments[idx]
    if MIN_REF_SEC <= _dur(self_seg) <= MAX_REF_SEC:
        return self_seg

    my_start = float(self_seg.get("start", 0.0))
    my_gender = self_seg.get("gender")

    def _eligible(require_gender: bool) -> list[dict]:
        out = []
        for j, seg in enumerate(segments):
            if j == idx:
                continue
            if not (MIN_REF_SEC <= _dur(seg) <= MAX_REF_SEC):
                continue
            if require_gender and seg.get("gender") != my_gender:
                continue
            out.append(seg)
        return out

    pool = _eligible(require_gender=True) if my_gender is not None else []
    if not pool:
        pool = _eligible(require_gender=False)
    if pool:
        return min(pool, key=lambda seg: abs(float(seg.get("start", 0.0)) - my_start))

    return max(segments, key=_dur)          # 兜底：全片最长的一段


def _slice_reference(
    vocals: Path, ref_seg: dict, cache: dict[int, Path], tmp_dir: Path,
) -> Path:
    """按参考段的 start/end 从 vocals 里切一段 mono 24k wav，按 segment id 缓存
    ——同一个参考段被多个目标段复用时只切一次（真跑时一个"话多"的说话人常常
    被选中当几十段的参考，重复切片纯浪费 ffmpeg 调用）。"""
    ref_id = int(ref_seg.get("id", 0))
    cached = cache.get(ref_id)
    if cached is not None:
        return cached
    start = float(ref_seg.get("start", 0.0))
    end = float(ref_seg.get("end", 0.0))
    out_path = tmp_dir / f"ref_{ref_id:03d}.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(vocals), "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
         "-ac", "1", "-ar", "24000", str(out_path)],
        check=True, capture_output=True,
    )
    cache[ref_id] = out_path
    return out_path


def main(argv: list[str]) -> int:
    if len(argv) < 4:
        print("用法: zion_clone_tts.py <zh_script.json> <vocals.mp3> <out_dir>",
              file=sys.stderr)
        return 2

    zh_script_path = Path(argv[1]).resolve()
    vocals_path = Path(argv[2]).resolve()
    out_dir = Path(argv[3]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_tmp_dir = out_dir / "_ref_slices"
    ref_tmp_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(zh_script_path.read_text(encoding="utf-8"))
    segments = data.get("segments", [])
    if not segments:
        print("zh_script 0 段，没有可克隆合成的内容")
        print("ZION_CLONE_DONE")
        return 0

    from f5_tts.api import F5TTS

    tts = F5TTS()                            # 只加载一次模型(~30-45s)，下面逐段复用
    ref_cache: dict[int, Path] = {}

    n_ok = 0
    n_fail = 0
    for idx, seg in enumerate(segments):
        zh = (seg.get("zh") or "").strip()
        if not zh:
            continue
        seg_id = int(seg.get("id", 0))
        try:
            ref_seg = pick_reference(segments, idx)
            if ref_seg is None:
                raise RuntimeError("没有可用的参考段")
            ref_file = _slice_reference(vocals_path, ref_seg, ref_cache, ref_tmp_dir)
            ref_text = ref_seg.get("en", "")
            out_wav = out_dir / f"n{seg_id:03d}.wav"
            tts.infer(
                ref_file=str(ref_file), ref_text=ref_text,
                gen_text=normalize_zh_numbers(zh),   # 数字→汉字,防英文参考带偏数字读音
                file_wave=str(out_wav), remove_silence=False,
            )
            n_ok += 1
        except Exception as e:                       # 单段失败不中止整批(同 d8 SDXL 挂不中止 run 的思路)
            traceback.print_exc()
            print(f"警告: 段 {seg_id} 克隆合成失败，跳过(下游会用其它引擎回退补上): {e}",
                  file=sys.stderr)
            n_fail += 1
        if (idx + 1) % 20 == 0:
            print(f"进度: {idx + 1}/{len(segments)} 段(成功 {n_ok} 失败 {n_fail})")

    print(f"完成: {n_ok}/{len(segments)} 段克隆成功，{n_fail} 段失败")
    print("ZION_CLONE_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
