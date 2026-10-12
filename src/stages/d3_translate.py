"""P7 T4：阶段③——把 d2_gpu 产出的英文转录(transcript.json)逐段译成
口语化中文配音稿(zh_script.json)，保留原有段数与时间轴，供下游 d4_tts/d5_subs
按句起点配音、烧字幕。

关键坑（已踩过，别重蹈）：src.util.llmjson.extract_json 只从第一个 `{` 找到
最后一个 `}` 解析——它解析的是一个 JSON **对象**，不是顶层数组。所以绝不能让
LLM 直接回一个 `[...]` 数组，必须让它包一层对象，如 `{"items": [...]}`，
本文件的 SYSTEM/build_prompt 与 _translate_batch 都遵循这一点。
"""
from __future__ import annotations

import json

from src.context import Ctx
from src.util.jobs import read_json, write_json
from src.util.llmjson import extract_json

OUTPUT = "zh_script.json"
INPUT = "transcript.json"
CHUNK_SIZE = 80  # 每批喂给 LLM 的段数上限，避免长纪录片一次性撑爆输出/被截断

SYSTEM = (
    "你是纪录片译配撰稿人，把英文解说逐句译成口语化、适合配音朗读的中文。"
    "默认完整自然地翻译，不要为了简短而删信息；每条有 max_chars 字数上限"
    "(由该句的播放时长折算)——只有完整译文会超过上限时才压缩意译保核心。"
    "自然口语，不要生硬直译，不要加解释或备注，不要合并或拆分句子。"
    "只输出一个 JSON 对象，不要额外文字、不要 markdown 代码块。"
)


# 重写模式(用户定 2026-07-14,p7.rewrite=true):不逐句翻译,基于内容重写成地道中文
# 纪录片解说。仍逐段对应(id/长度对齐画面),但用中文纪录片叙事腔调重新撰稿——增强
# 悬念/画面感/戏剧性,忠于事实不编造。用于把英文纪录片"改写"成中文版而非直译。
REWRITE_SYSTEM = (
    "你是资深中文纪录片编导兼撰稿人。下面给你一部英文纪录片的逐段解说文字。请"
    "**基于内容重写**成一部地道、有张力、悬念感强、适合中文观众的纪录片解说——"
    "用中文纪录片的叙事腔调重新撰稿,可调整措辞、增强画面感与戏剧张力、让语言更"
    "抓人,但**必须忠于事实、不得编造**,且**逐段对应**(每个 id 输出一条改写后的"
    "中文,长度与原句时长相当以便对齐画面,别大幅增删)。绝不逐字直译,要像中文纪录片"
    "重新配的解说。每条有 max_chars 字数上限(按时长折算),超了就精炼。"
    "只输出一个 JSON 对象,不要额外文字、不要 markdown 代码块。"
)


# 填满时长模式(用户定 2026-07-26,p7.fill_gaps=true):字幕常被切得很碎/文字相对时长偏短,
# 逐段直译后配音短、段内留长静音,听起来一顿一顿。此模式让每条写到接近 max_chars(=按时长
# 折算的目标字数),把整段时长填满、配音连续流畅不卡顿;原文偏短就在忠于事实前提下自然扩充
# 措辞/补描述衔接。绝不编造具体数据/事件/人名/地名/结论——只在原意内把话说饱满连贯。
FILL_SYSTEM = (
    "你是纪录片译配撰稿人。把英文解说译成流畅连贯、适合配音朗读的中文,核心目标是"
    "**让配音在整段时间内连续不断、不留长时间静音,一段之内一气呵成、不一顿一顿**。"
    "每条给了 max_chars(按该句播放时长折算的目标字数)——**尽量把译文写到接近 max_chars、"
    "用足时长**。若原句相对时长偏短(直译会很短、放完就静音),就自然地把话说得更充分:"
    "丰富措辞、展开描述、补充与主题和画面相符的衔接性/描述性解说,把这段时长填满、读起来顺。"
    "但**必须忠于原意与事实,绝不编造具体的数据、事件、人名、地名或结论**——只在原意范围内"
    "把话说得更连贯饱满。逐段对应每个 id,自然口语、连贯流畅,不要生硬直译。"
    "只输出一个 JSON 对象,不要额外文字、不要 markdown 代码块。"
)


def build_prompt(items: list[dict], rewrite: bool = False, fill: bool = False) -> str:
    payload = json.dumps({"items": items}, ensure_ascii=False)
    if rewrite:
        verb = "重写成中文纪录片解说稿(纪录片叙事腔,非逐字翻译)"
    elif fill:
        verb = "译成流畅连贯、填满时长的中文配音稿"
    else:
        verb = "改写成中文配音稿"
    if rewrite:
        tail = "忠于事实、逐段对应、长度贴合 max_chars。\n"
    elif fill:
        tail = ("每条尽量写到接近 max_chars 把时长填满,让配音连续流畅、不留长静音;"
                "原文相对时长偏短就在忠于事实前提下自然扩充措辞/补描述,不得编造具体事实。\n")
    else:
        tail = "默认完整翻译；仅当译文超过该条 max_chars 时才压缩意译。\n"
    return (
        f"把下面每条英文解说{verb}，输出一个 JSON 对象（不是数组），"
        '格式必须是 {"items": [{"id": 整数, "zh": "中文"}, ...]}，'
        "id 必须和输入的 id 一一对应，不要遗漏、不要新增、不要打乱顺序。"
        f"{tail}"
        f"输入：{payload}"
    )


# 字数上限折算(字/秒):贴合 CV2 实测吐字速率(2026-07-05 实测 4.34字/秒),取 4.0
# 留 8% 余量——这样译文默认就能在 1.0x 自然语速下装进时间槽(用户定的优先级:
# 装不下先精简翻译,不行再变速)。d7 的提速(≤1.25)退成极端超密段的兜底。
# 仍是"上限而非目标"(精炼太短不强求),只是上限收紧到能 1.0x 装下。
ZH_CHARS_PER_SEC = 4.0


def char_budgets(segments: list[dict], duration: float, cps: float = ZH_CHARS_PER_SEC) -> list[int]:
    """逐段字数预算:到下一段开播为止的窗口(留 0.15s 间隙) × 语速。至少 6 字。"""
    out: list[int] = []
    for i, seg in enumerate(segments):
        start = float(seg.get("start", 0.0))
        nxt = float(segments[i + 1].get("start", duration)) if i + 1 < len(segments) else float(duration or start + 10)
        window = max(nxt - start - 0.15, 1.0)
        out.append(max(int(window * cps), 6))
    return out


# 段落合并(用户定 2026-07-26,p7.coalesce_segments=true):平台字幕常被切成很碎的小段
# (甚至时间重叠),逐段配音就一顿一顿、段间静音。把语流连续(间隙<=max_gap)的相邻小段
# 合并成整段落(累计时长<=max_dur),文字拼接、span 取首段 start 到末段 end。合并后每段是
# 完整语义单元,配 fill_gaps 让 AI 写连贯饱满的中文填满整段→配音一气呵成。gender 取首段
# (仅用于单说话人旁白线,多说话人别开——会把不同说话人并到一起)。
def coalesce_segments(segments: list[dict], max_gap: float = 1.5,
                      max_dur: float = 22.0) -> list[dict]:
    out: list[dict] = []
    cur: dict | None = None
    for s in segments:
        st = float(s.get("start", 0.0) or 0.0)
        en = float(s.get("end", st) or st)
        txt = (s.get("text", "") or "").strip()
        if cur is None:
            cur = {"start": st, "end": en, "text": txt, "gender": s.get("gender")}
            continue
        gap = st - float(cur["end"])
        new_dur = max(en, float(cur["end"])) - float(cur["start"])
        if gap <= max_gap and new_dur <= max_dur:          # 语流连续且不超长→并入本段落
            cur["text"] = (str(cur["text"]).rstrip() + " " + txt).strip()
            cur["end"] = max(float(cur["end"]), en)
        else:                                              # 停顿或超长→另起段落
            out.append(cur)
            cur = {"start": st, "end": en, "text": txt, "gender": s.get("gender")}
    if cur is not None:
        out.append(cur)
    return out


def _translate_batch(ctx: Ctx, batch: list[tuple[int, str, int]]) -> dict[int, str]:
    items = [{"id": i, "en": text, "max_chars": mc} for i, text, mc in batch]
    rewrite = bool(ctx.config.get("p7.rewrite", False))
    fill = bool(ctx.config.get("p7.fill_gaps", False)) and not rewrite
    system = REWRITE_SYSTEM if rewrite else (FILL_SYSTEM if fill else SYSTEM)
    parsed = None
    for attempt in range(3):                    # LLM 偶发吐残缺 JSON(真跑遇到)——重试而非杀死整条 run
        raw = ctx.llm.generate(system, build_prompt(items, rewrite, fill))
        try:
            parsed = extract_json(raw)
            break
        except (json.JSONDecodeError, ValueError) as e:
            ctx.warn(f"{ctx.slug}: d3 批次 JSON 解析失败(第{attempt + 1}次尝试): {e}")
    if parsed is None:                          # 重试尽仍坏→本批次回退(上层会补 zh=en 并告警)
        return {}
    zh_map: dict[int, str] = {}
    for item in parsed.get("items", []):
        try:
            idx = int(item.get("id"))
        except (TypeError, ValueError):
            continue
        zh_map[idx] = item.get("zh", "")
    return zh_map


def run(ctx: Ctx) -> None:
    data = read_json(ctx.job, INPUT)
    segments = data.get("segments", [])
    duration = data.get("duration", 0)

    if not segments:
        write_json(ctx.job, OUTPUT, {"duration": duration, "segments": []})
        ctx.warn(f"{ctx.slug}: d3_translate 转录 0 段，跳过翻译")
        return

    # 段落合并(p7.coalesce_segments):碎小段合并成整段落,配 fill_gaps 让配音连续不卡顿。
    # 仅单说话人旁白线用(会并不同说话人)。合并后重排 id、透传 mode。
    if bool(ctx.config.get("p7.coalesce_segments", False)):
        n0 = len(segments)
        segments = coalesce_segments(
            segments,
            max_gap=float(ctx.config.get("p7.coalesce_max_gap", 1.5)),
            max_dur=float(ctx.config.get("p7.coalesce_max_dur", 22.0)))
        ctx.warn(f"{ctx.slug}: d3 段落合并 {n0}→{len(segments)} 段(填满时长、配音连续)")

    budgets = char_budgets(segments, duration,
                           float(ctx.config.get("p7.zh_chars_per_sec", ZH_CHARS_PER_SEC)))
    from src.util.ytsubs import strip_sound_effects
    indexed = list(enumerate(segments))
    # 批大小可配(p7.chunk_size,默认 CHUNK_SIZE)。rewrite 模式输出更长、易被 LLM 截断
    # 成坏 JSON(2026-07-14 UAP 片踩到:80 段一批连丢导致整批回退英文),故 rewrite 线
    # 用更小的批(如 40)提升 JSON 完整率。
    chunk_size = max(int(ctx.config.get("p7.chunk_size", CHUNK_SIZE)), 1)
    zh_map: dict[int, str] = {}
    for start in range(0, len(indexed), chunk_size):
        chunk = indexed[start : start + chunk_size]
        # 音效词剥一遍再翻译(2026-07-07 用户定:覆盖 whisper 路径,平台字幕已在 ytsubs 剥过)
        batch = [(i, strip_sound_effects(seg.get("text", "")), budgets[i]) for i, seg in chunk]
        zh_map.update(_translate_batch(ctx, batch))

    # 超预算段(>1.3x)定向返工一轮:更严的措辞逼 LLM 压缩;返工仍超就交给 d7 的
    # atempo(上限已降到 1.25)+截尾兜底,不无限重试
    over = [(i, seg.get("text", ""), budgets[i]) for i, seg in indexed
            if zh_map.get(i) and len(zh_map[i]) > budgets[i] * 1.3]
    if over:
        ctx.warn(f"{ctx.slug}: d3 有 {len(over)} 段译文超字数预算1.3x，定向压缩返工一轮")
        for start in range(0, len(over), chunk_size):
            redo = _translate_batch(ctx, over[start : start + chunk_size])
            for i, zh in redo.items():
                if zh and len(zh) < len(zh_map.get(i, zh) or zh):
                    zh_map[i] = zh   # 只在真变短时替换

    out_segments = []
    missing = 0
    for i, seg in indexed:
        en_clean = strip_sound_effects(seg.get("text", ""))   # en 存干净版(引用文本/兜底)
        zh = zh_map.get(i)
        if zh is None:
            zh = en_clean  # 回退：保留英文原文(已剥音效词)，不留空、不错位、不崩
            missing += 1
        else:
            zh = strip_sound_effects(zh)   # LLM 若保留了方括号,再剥一遍
        if not zh.strip():                 # 剥完空了(整段就是音效词)→ 跳过,不发空段
            continue
        out_segments.append(
            {
                "id": i,
                "start": seg.get("start"),
                "end": seg.get("end"),
                "en": en_clean,
                "zh": zh,
                # P7 T9：透传 d2_gpu(zion 侧基频估计)写入的说话人性别，供 d4_tts
                # 挑男/女声。转录段没有该字段(旧数据/检测失败)时落 None——绝不
                # 瞎猜，d4 会按 None 走默认男声旁白。
                "gender": seg.get("gender"),
            }
        )

    if missing:
        ctx.warn(
            f"{ctx.slug}: d3_translate 有 {missing}/{len(segments)} 段缺少翻译"
            "（LLM 返回条数不足），已回退用英文原文顶上"
        )

    out = {"duration": duration, "segments": out_segments}
    if data.get("mode"):
        out["mode"] = data["mode"]   # subs_only(字幕版)等模式贯穿下游
    write_json(ctx.job, OUTPUT, out)
