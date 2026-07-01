from __future__ import annotations


class StageBlocked(Exception):
    """阶段判定该案例不应继续（如证据不足）。编排器捕获后停止本案例。"""
