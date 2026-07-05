"""zion 端 CosyVoice2 跨语种声音克隆配音(2026-07-05 用户 A/B 盲听选定替换 F5-TTS)。

**必须**经 `~/cosy-py` 调用(cosy-venv 的 LD_LIBRARY_PATH wrapper),CLI 契约与
zion_clone_tts.py 完全一致,编排侧(src/remote.py)按 tts.provider 分派脚本名:

    ~/cosy-py zion_cosy_tts.py <zh_script.json> <vocals.mp3> <out_dir>

复用 zion_clone_tts 的三件套(同目录 import,纯函数无 torch 依赖):
  - pick_reference:自己→同性别最近→任意最近→全片最长 的参考选择规则
  - normalize_zh_numbers:数字转中文读法(克隆引擎念阿拉伯数字易走英文)
  - 参考能量门思想:切片 max_volume < -45dB 拒用(纯音乐片静音参考产咔咔声的教训)

环境备忘(装机踩坑记录,见任务#57):
  - numpy==1.26.4 + scipy==1.11.4 锁定(scipy 新版要 numpy2,但部分 C 扩展只认 numpy1)
  - pyworld 需 nix-shell -p gcc + LD_LIBRARY_PATH + --no-build-isolation 编译
  - inference_cross_lingual 的参考要传 **wav 路径**(本版 frontend 内部自行 load_wav)
  - 落盘用 soundfile 直写(torchaudio.save 在这台 NixOS 上要 torchcodec,是坑)
  - 切勿从 /tmp 跑本脚本(/tmp/dis.py 用户实验文件会遮蔽标准库 dis)
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import traceback
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))                       # 同目录复用 zion_clone_tts 纯函数

from zion_clone_tts import normalize_zh_numbers, pick_reference  # noqa: E402

COSY_REPO = os.environ.get("COSY_REPO", "/home/fabric/CosyVoice")
COSY_MODEL_DIR = os.environ.get("COSY_MODEL_DIR", "/home/fabric/cosy-models/CosyVoice2-0.5B")
PROGRESS_EVERY = 20
REF_MAX_SILENCE_DB = -45.0     # 参考能量门:低于此视为静音参考(咔咔声教训)


def _slice_reference_16k(vocals: Path, ref_seg: dict, cache: dict[int, Path],
                         tmp_dir: Path) -> Path:
    """参考切片:16k mono(CosyVoice2 frontend 要求) + 与 F5 生产链同款降噪清洗
    (highpass+afftdn,克隆引擎会连声学环境一起复刻) + 能量门。按段 id 缓存。"""
    ref_id = int(ref_seg.get("id", 0))
    cached = cache.get(ref_id)
    if cached is not None:
        return cached
    start = float(ref_seg.get("start", 0.0))
    end = float(ref_seg.get("end", 0.0))
    out_path = tmp_dir / f"ref16k_{ref_id:03d}.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(vocals), "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
         "-ac", "1", "-ar", "16000",
         "-af", "highpass=f=90,afftdn=nr=12:nf=-32",
         str(out_path)],
        check=True, capture_output=True,
    )
    probe = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", str(out_path), "-af", "volumedetect",
         "-f", "null", "-"], capture_output=True, text=True)
    m = re.search(r"max_volume:\s*(-?[\d.]+) dB", probe.stderr or "")
    if m and float(m.group(1)) < REF_MAX_SILENCE_DB:
        out_path.unlink(missing_ok=True)
        raise RuntimeError(f"参考切片近乎静音(max {m.group(1)}dB),弃用该参考")
    cache[ref_id] = out_path
    return out_path


def main(argv: list[str]) -> int:
    if len(argv) < 4:
        print("用法: zion_cosy_tts.py <zh_script.json> <vocals.mp3> <out_dir>",
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

    sys.path.insert(0, COSY_REPO)
    sys.path.insert(0, str(Path(COSY_REPO) / "third_party" / "Matcha-TTS"))
    import soundfile as sf
    from cosyvoice.cli.cosyvoice import CosyVoice2

    cv = CosyVoice2(COSY_MODEL_DIR, load_jit=False, load_trt=False, fp16=True)
    ref_cache: dict[int, Path] = {}
    done = skipped = 0
    for idx, seg in enumerate(segments):
        zh = (seg.get("zh") or "").strip()
        if not zh:
            continue
        seg_id = int(seg.get("id", 0))
        try:
            ref_seg = pick_reference(segments, idx)
            if ref_seg is None:
                raise RuntimeError("全片选不出任何参考段")
            ref_path = _slice_reference_16k(vocals_path, ref_seg, ref_cache, ref_tmp_dir)
            text = normalize_zh_numbers(zh)
            for out in cv.inference_cross_lingual(str(text), str(ref_path), stream=False):
                sf.write(str(out_dir / f"n{seg_id:03d}.wav"),
                         out["tts_speech"].squeeze(0).cpu().numpy(), cv.sample_rate)
                break
            done += 1
        except Exception:  # noqa: BLE001 — 单段失败跳过,d4 对缺失段回退 edge,绝不拖垮整片
            traceback.print_exc()
            print(f"段 {seg_id} 克隆失败,跳过(d4 将回退 edge)", file=sys.stderr)
            skipped += 1
        if (done + skipped) % PROGRESS_EVERY == 0:
            print(f"进度 {done + skipped}/{len(segments)} (成功 {done}, 跳过 {skipped})")

    print(f"完成: 成功 {done}, 跳过 {skipped}")
    print("ZION_CLONE_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
