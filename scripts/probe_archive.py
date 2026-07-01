"""Live probe for src.media.ArchiveVideo — real network, NOT a pytest test.

Runs one real search + fetch against archive.org for a sample query and prints
the chosen clip + license, so we can eyeball the real path once (P6 T1).

Usage:
    .venv/bin/python scripts/probe_archive.py ["query"]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.media import ArchiveVideo  # noqa: E402


def main() -> None:
    query = sys.argv[1] if len(sys.argv) > 1 else "weather balloon"
    src = ArchiveVideo()

    print(f"[probe] searching archive.org mediatype:movies for: {query!r}")
    candidates = src.search(query, limit=5)
    print(f"[probe] {len(candidates)} candidate(s) passed license gate (guardrail#2):")
    for c in candidates:
        print(f"  - {c['manifest']!r:45s} license={c['license']:15s} attribution={c['attribution']}")

    if not candidates:
        print("[probe] NO legal candidates found for this query — nothing to fetch.")
        return

    chosen = candidates[0]
    dest = Path(__file__).resolve().parent / "_probe_out.mp4"
    print(f"\n[probe] fetching {chosen['manifest']!r} -> {dest}")
    src.fetch(chosen, dest)

    size = dest.stat().st_size
    print(f"[probe] downloaded {size} bytes")
    print(f"[probe] chosen clip   : {chosen['manifest']}")
    print(f"[probe] license       : {chosen['license']}")
    print(f"[probe] attribution   : {chosen['attribution']}")
    print(f"[probe] saved to      : {dest}")


if __name__ == "__main__":
    main()
