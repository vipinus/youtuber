"""列出某作业 s6 选中的每段素材(标题/来源/许可)，供人工过一眼再合成/上传。
用法：.venv/bin/python scripts/review_clips.py <slug>"""
from __future__ import annotations
import json, sys
from pathlib import Path

def main(slug: str) -> None:
    p = Path("jobs") / slug / "credits.json"
    items = json.loads(p.read_text(encoding="utf-8")).get("items", [])
    print(f"=== {slug}: {len(items)} 段素材 ===")
    for it in items:
        flag = "  ⚠️兜底" if it.get("source") == "fallback" else ""
        print(f"seg {it['id']:>2} [{it.get('source','?'):>8}] {it.get('license','?'):<13} "
              f"{it.get('attribution','')}{flag}")
    n_fb = sum(1 for i in items if i.get("source") == "fallback")
    print(f"\n真素材 {len(items)-n_fb} / 兜底 {n_fb}。逐条确认无不当内容后再合成/上传。")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "roswell-1947")
