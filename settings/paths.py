# -*- coding: utf-8 -*-
"""路径常量（单一来源）。

其它模块一律写 `from settings.paths import ROOT_DIR / DATA_DIR`，
不要再到处自己算 `Path(__file__).parent.parent` —— 目录层级一调整就会全部错位。

本文件不读取任何环境变量，纯几何关系，可被任意模块安全导入。
"""
from __future__ import annotations
from pathlib import Path

# backend/ 目录 = settings/ 的上级
ROOT_DIR: Path = Path(__file__).resolve().parent.parent

# 本地 JSON 数据落盘目录（DB_DRIVER=local 时使用）
DATA_DIR: Path = ROOT_DIR / "data"
DOCS_DIR: Path = ROOT_DIR / "docs"
CHROMA_DIR: Path = ROOT_DIR / "chroma"

def resolve_path(raw: str) -> Path:
    """把配置里的相对路径解析为绝对路径。

    相对路径一律以 backend/ 为基准解释，避免进程工作目录不同导致落盘位置漂移
    （Repo 读写用的就是这个返回值）。
    """
    p = Path(raw)
    return p if p.is_absolute() else (ROOT_DIR / p).resolve()
