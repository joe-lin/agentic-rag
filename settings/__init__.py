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
logger = logging.getLogger("agentic-rag.settings")
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



chat_model = _get("CHAT_MODEL", "")
chat_api_key = _get("CHAT_API_KEY", "")
chat_base_url = _get("CHAT_BASE_URL", "")


praison_chat_model = _get("PRAISON_CHAT_MODEL", "")
praison_api_key = _get("PRAISON_API_KEY", "")
praison_base_url = _get("PRAISON_BASE_URL", "")

embedding_model = _get("EMBEDDING_MODEL", "")
embedding_api_key = _get("EMBEDDING_API_KEY", "")
embedding_base_url = _get("EMBEDDING_BASE_URL", "")

# 向量库
collection_name = "agentic-rag"
persist_directory = "data/chroma_db"
k = 6
# 检索后重排：向量召回 candidate_k 条候选，再用交叉编码器 rerank 取前 k 条。
# 基础 bi-encoder 对“问句 vs 密集事实清单”类查询排序偏弱，reranker 可显著纠正。
rerank_model = _get("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
candidate_k = _int("CANDIDATE_K", 20)
# HyDE：检索前先用 LLM 生成“假设性答案”作为检索嵌入，弥补 bi-encoder
# 对“抽象问句 vs 密集事实清单”匹配弱的短板。可设 HYDE_ENABLED=false 关闭。
hyde_enabled = _bool("HYDE_ENABLED", True)
data_path = "data"
md5_hex_store = "rag/md5.txt"
allow_knowledge_file_type = ["txt","pdf"]

chunk_size = 200
chunk_overlap = 20
separators = ["\n\n", "\n", "。", ".","?","？","!", "；"," ",""]

#prompts
rag_summarize_prompt_path = "prompts/rag_summarize.txt"
product_expert_prompt = "prompts/product_expert.txt"