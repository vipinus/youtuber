"""zion 端 Stable Diffusion XL 缩略图背景生成脚本（P7 T8）。

**必须**经 `~/gpu-py` 调用，不能直接跑 `~/asr-venv/bin/python`——同
zion_gpu_asr.py 的坑：wrapper 先设好 LD_LIBRARY_PATH 再 exec venv python，
直接跑裸 venv python 加载不了 torch 的底层 CUDA 库。

用法（在 zion 上）：
    ~/gpu-py zion_sdxl.py "<prompt>" <out.png> [genWxH] [outWxH] [--seq]
    尺寸默认 1024x576 -> 1280x720(d8 缩略图档,行为不变)。
    --seq 切 enable_sequential_cpu_offload：显存 <2GB 但慢好几倍——zion 同时是
    桌面机(KDE/游戏常驻 ~3GB 显存)，桌面占用大时 model offload 会 OOM，banner
    这类离线任务用 --seq 硬挤进去。

已在 zion 上实测（RTX 3070 Ti / 8GB，diffusers + torch 2.6.0+cu124，模型已
缓存）：加载 ~1s、28 步推理 ~12s、输出 1280x720 PNG。8GB 显存靠
`enable_model_cpu_offload` + `enable_vae_slicing` 撑住（sequential offload
更省显存但明显更慢，cuDNN 修好后不需要）。

prompt 必须在末尾追加 "no text"（negative_prompt 里也排除 text/watermark/
logo）——中文标题和频道水印由本地 d8_meta 用 Pillow 后期叠加，不指望 SDXL
自己出中文字（历史上 SD 系模型画字基本是乱码）。
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # 抗显存碎片

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
NEGATIVE_PROMPT = "text, watermark, logo, blurry, low quality, deformed"
GEN_W, GEN_H = 1024, 576          # 生成分辨率(16:9，SDXL 原生档)，再 resize 到缩略图尺寸
OUT_W, OUT_H = 1280, 720
STEPS = 28
GUIDANCE = 7.0


def generate(prompt: str, out_path: str,
             gen_wh: tuple[int, int] = (GEN_W, GEN_H),
             out_wh: tuple[int, int] = (OUT_W, OUT_H),
             seq: bool = False) -> None:
    import torch
    from diffusers import StableDiffusionXLPipeline

    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, variant="fp16", use_safetensors=True,
    )
    if seq:
        pipe.enable_sequential_cpu_offload()  # <2GB 显存,慢;桌面/游戏占着显存时的兜底档
    else:
        pipe.enable_model_cpu_offload()      # cuDNN 兼容后按需 offload 即可塞进 8GB，比 sequential 快
    pipe.enable_vae_slicing()

    img = pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        height=gen_wh[1], width=gen_wh[0],
        num_inference_steps=STEPS, guidance_scale=GUIDANCE,
    ).images[0].resize(out_wh)
    img.save(out_path)


def _parse_wh(s: str) -> tuple[int, int]:
    w, h = s.lower().split("x")
    return int(w), int(h)


def main(argv: list[str]) -> int:
    args = [a for a in argv[1:] if a != "--seq"]
    seq = "--seq" in argv[1:]
    if len(args) < 2:
        print("用法: zion_sdxl.py <prompt> <out.png> [genWxH] [outWxH] [--seq]", file=sys.stderr)
        return 2

    prompt, out_path = args[0], args[1]
    gen_wh = _parse_wh(args[2]) if len(args) > 2 else (GEN_W, GEN_H)
    out_wh = _parse_wh(args[3]) if len(args) > 3 else (OUT_W, OUT_H)
    generate(prompt, out_path, gen_wh, out_wh, seq=seq)
    print(f"缩略图背景生成完成 -> {out_path} ({out_wh[0]}x{out_wh[1]})")
    print("ZION_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
