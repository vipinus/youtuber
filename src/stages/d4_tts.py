"""P7 T5：阶段④——把 d3_translate 产出的中文配音稿(zh_script.json)逐段合成
中文旁白音频(narration/nNNN.mp3)，记录每段的真实时长(tts_dur)和视频锚点起点
(start)，供下游 d7_assemble 按段起点摆放旁白、与(去人声后的)背景混音。

复用 src/tts.py 的 get_tts（FakeTTS 离线可测 / EdgeTTS 真跑），复用
src/util/ffprobe.duration_sec 量真实音频时长（含引擎尾静音，修时间轴漂移，
与 s4_tts 同一手法）。zh_script 的 zh 字段偶尔会因翻译缺失回退成英文——这里
不区分中英文，只要非空就照常配音；只有空串才跳过。

P7 T9：性别匹配配音——按段 gender 字段（d2_gpu 经 zion 基频估计、d3_translate
透传）挑男声/女声引擎，修用户反馈的"画面是女性说话却配了男声"问题。
gender=="female" 用 tts.voice_female，其余（"male"/None/缺失）落
tts.voice（默认男声旁白，不确定时不瞎猜、走默认更安全）。同一 voice 只建一
个引擎实例(engines 缓存)，避免重复构造。选中的 voice 写进 narration.json
每段的 "voice" 字段，供调试核对配音是否配对。

P7 T17：声音克隆配音——tts.provider=="clone" 时整段流程改道：把 zh_script.json
+ vocals.mp3（d2_gpu 落的 demucs 干声轨）一次性甩给
src/remote.py.get_remote(config).clone_synthesize()，remote 侧（zion 上经
scripts/zion_clone_tts.py 跑 F5-TTS one-shot 跨语种声音克隆）批量把整部片子
的中文旁白都合成好，音色是每个说话人自己的原声。合成完这里再逐段核对：
narration/n{id:03d}.wav 存在就量真实时长记 manifest(voice=="clone")；不存在
（该段 F5 推理失败/参考音频损坏等 remote 侧已跳过的段）就 ctx.warn 并回退
tts.fallback_provider（默认 edge，测试用 fake 保持离线）指定的普通引擎单独
补一份——一段克隆失败绝不能让这段旁白整个消失。
"""
from __future__ import annotations

from pathlib import Path

from src.context import Ctx
from src.remote import get_remote
from src.tts import TTSEngine, get_tts
from src.util.ffprobe import duration_sec
from src.util.jobs import read_json, write_json

OUTPUT = "narration.json"
INPUT = "zh_script.json"
VOCALS_INPUT = "vocals.mp3"
NARRATION_DIR = "narration"


def _voice_for(gender: str | None, ctx: Ctx) -> str:
    if gender == "female":
        return ctx.config.get("tts.voice_female", "zh-CN-XiaoxiaoNeural")
    return ctx.config.get("tts.voice", "zh-CN-YunyangNeural")


def _warn_skip_and_empty(ctx: Ctx, segments: list[dict], skipped: int, out_segments: list[dict]) -> None:
    if skipped:
        ctx.warn(
            f"{ctx.slug}: d4_tts 有 {skipped}/{len(segments)} 段 zh 为空，已跳过配音"
        )
    if not out_segments:
        ctx.warn(f"{ctx.slug}: d4_tts 没有可用配音段（zh_script 为空或全部 zh 缺失）")


def _run_clone(ctx: Ctx, segments: list[dict], narration_dir: Path) -> None:
    zh_script_path = ctx.job / INPUT
    vocals_path = ctx.job / VOCALS_INPUT
    remote = get_remote(ctx.config)
    remote.clone_synthesize(zh_script_path, vocals_path, narration_dir)

    fallback_provider = ctx.config.get("tts.fallback_provider", "edge")
    engines: dict[str, TTSEngine] = {}
    out_segments: list[dict] = []
    skipped = 0
    for seg in segments:
        zh = (seg.get("zh") or "").strip()
        if not zh:
            skipped += 1
            continue
        seg_id = int(seg.get("id", 0))
        clone_name = f"n{seg_id:03d}.wav"
        clone_path = narration_dir / clone_name
        if clone_path.exists():
            dur = round(duration_sec(clone_path), 3)
            out_segments.append({
                "id": seg_id,
                "start": seg.get("start"),
                "file": f"{NARRATION_DIR}/{clone_name}",
                "tts_dur": dur,
                "zh": zh,
                "voice": "clone",
            })
            continue

        # 该段克隆缺失(F5 推理失败/参考音频损坏等) -> 回退普通引擎单独补上，
        # 绝不让一段坏的拖累这段旁白整个消失。
        ctx.warn(
            f"{ctx.slug}: d4_tts 段 {seg_id} 克隆配音缺失(未产出 {clone_name})，"
            f"回退用 {fallback_provider} 引擎补上"
        )
        voice = _voice_for(seg.get("gender"), ctx)
        engine = engines.get(voice)
        if engine is None:
            engine = get_tts(ctx.config, voice=voice, provider=fallback_provider)
            engines[voice] = engine
        audio_bytes, _words = engine.synth(zh)
        name = f"n{seg_id:03d}.mp3"
        path = narration_dir / name
        path.write_bytes(audio_bytes)
        dur = round(duration_sec(path), 3)
        out_segments.append({
            "id": seg_id,
            "start": seg.get("start"),
            "file": f"{NARRATION_DIR}/{name}",
            "tts_dur": dur,
            "zh": zh,
            "voice": voice,
        })

    _warn_skip_and_empty(ctx, segments, skipped, out_segments)
    write_json(ctx.job, OUTPUT, {"segments": out_segments})


def run(ctx: Ctx) -> None:
    data = read_json(ctx.job, INPUT)
    segments = data.get("segments", [])

    narration_dir = ctx.job / NARRATION_DIR
    narration_dir.mkdir(parents=True, exist_ok=True)

    if ctx.config.get("tts.provider", "edge") == "clone":
        _run_clone(ctx, segments, narration_dir)
        return

    engines: dict[str, TTSEngine] = {}          # 按 voice 名缓存引擎，同一 voice 不重复构造
    out_segments: list[dict] = []
    skipped = 0
    for seg in segments:
        zh = (seg.get("zh") or "").strip()
        if not zh:
            skipped += 1
            continue
        seg_id = int(seg.get("id", 0))
        voice = _voice_for(seg.get("gender"), ctx)
        engine = engines.get(voice)
        if engine is None:
            engine = get_tts(ctx.config, voice=voice)
            engines[voice] = engine
        audio_bytes, _words = engine.synth(zh)
        name = f"n{seg_id:03d}.mp3"
        path = narration_dir / name
        path.write_bytes(audio_bytes)
        dur = round(duration_sec(path), 3)      # 真实音频时长（含尾静音），供 d7 定位/避免重叠
        out_segments.append({
            "id": seg_id,
            "start": seg.get("start"),           # 视频起点锚点：d7 按此摆放旁白
            "file": f"{NARRATION_DIR}/{name}",
            "tts_dur": dur,
            "zh": zh,
            "voice": voice,                       # 调试核对：这段实际用了哪个音色
        })

    _warn_skip_and_empty(ctx, segments, skipped, out_segments)
    write_json(ctx.job, OUTPUT, {"segments": out_segments})
