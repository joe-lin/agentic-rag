# -*- coding: utf-8 -*-
"""运行时配置：环境变量 / .env 的**唯一**读取入口。
load_dotenv() 在本包首次导入时执行一次：
    - 本地开发：读 backend/.env
    - 部署环境（CloudBase Run）：读进程环境变量，无 .env 也不报错

约定：任何模块都不得再裸调 os.getenv()，新增配置项一律加到这里。
"""
from __future__ import annotations
import os
import logging
from dotenv import load_dotenv

logger = logging.getLogger("ai-menu.settings")

from settings.paths import ROOT_DIR, DATA_DIR, resolve_path

load_dotenv()


def _get(name: str, default: str = "") -> str:
    return os.getenv(name, default)


def _bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


def _int(name: str, default: int = 0) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


# ==================== 日志 ====================
LOG_LEVEL = _get("LOG_LEVEL", "INFO").upper()
DATA_FILE = resolve_path(_get("DATA_FILE", ""))


chat_model = _get("CHAT_MODEL", "")
api_key = _get("CHAT_API_KEY", "")
base_url = _get("CHAT_BASE_URL", "")


praison_chat_model = _get("PRAISON_CHAT_MODEL", "")
praison_api_key = _get("PRAISON_API_KEY", "")
praison_base_url = _get("PRAISON_BASE_URL", "")