"""zion 端 Stable Diffusion XL 缩略图背景生成脚本（P7 T8）。

**必须**经 `~/gpu-py` 调用，不能直接跑 `~/asr-venv/bin/python`——同
zion_gpu_asr.py 的坑：wrapper 先设好 LD_LIBRARY_PATH 再 exec venv python，
直接跑裸 venv python 加载不了 torch 的底层 CUDA 库。

用法（在 zion 上）：
    ~/gpu-py zion_sdxl.py "<prompt>" <out.png>

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


def generate(prompt: str, out_path: str) -> None:
    import torch
    from diffusers import StableDiffusionXLPipeline

    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, variant="fp16", use_safetensors=True,
    )
    pipe.enable_model_cpu_offload()      # cuDNN 兼容后按需 offload 即可塞进 8GB，比 sequential 快
    pipe.enable_vae_slicing()

    img = pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        height=GEN_H, width=GEN_W,
        num_inference_steps=STEPS, guidance_scale=GUIDANCE,
    ).images[0].resize((OUT_W, OUT_H))
    img.save(out_path)


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print("用法: zion_sdxl.py <prompt> <out.png>", file=sys.stderr)
        return 2

    prompt, out_path = argv[1], argv[2]
    generate(prompt, out_path)
    print(f"缩略图背景生成完成 -> {out_path} ({OUT_W}x{OUT_H})")
    print("ZION_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
