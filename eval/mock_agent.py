# -*- coding: utf-8 -*-
"""离线 mock：不加载 LLM / Chroma，用固定响应端到端验证 runner→judge→聚合→门控。

刻意覆盖了各种分数形态（满分、部分分、硬门控归零、模板污染），
所以 mock 跑出来的分差本身就是判分逻辑的自检：如果所有 case 都同分，说明判分坏了。

用法：python3 -m eval.runner --mock
"""
from __future__ import annotations

from pathlib import Path

EVAL = Path(__file__).resolve().parent

MOCK: dict[str, dict] = {
    "case_001": {
        "answer": "OceanBase 向量检索支持基于图的 HNSW 和基于倒排的 IVF 索引算法；量化算法包括 Flat、SQ8、PQ、BQ；距离算法支持曼哈顿距离、欧式距离、内积和余弦距离。",
        "called": True, "pages": [8], "golden": "full",
    },
    "case_002": {
        "answer": "支持 HNSW 与 IVF 索引，量化有 Flat、SQ8、PQ、BQ，距离包括曼哈顿、欧式、内积、余弦。",
        "called": True, "pages": [8], "golden": "first",
    },
    "case_003": {
        "answer": "该跨境电商案例采用 IVF 索引，向量查询使用 cosine 算法，向量维度 384，分区数为 300。",
        "called": True, "pages": [22], "golden": "full",
    },
    "case_004": {
        "answer": "数据量更大后为了节省内存，货拉拉计划从 HNSW 升级到 IVF-SQ8，当前 HNSW 压缩算法可单分区支撑业务、无须分库分表；快速索引重建仅 25 分钟。",
        "called": False, "pages": [], "golden": "none",
    },
    "case_005": {
        "answer": "你好，有什么可以帮你？",
        "called": True, "pages": [], "golden": "none",
    },
    "case_006": {
        "answer": "OceanBase 是 100% 根自研的一体化分布式数据库，基于单机分布式一体化架构，高度兼容 Oracle 和 MySQL，支持事务处理 TP 和实时分析处理 AP 等多工作负载，同时原生支持向量检索与多模数据混合检索，可助力客户构建面向 GenAI 的一体化数据底座，已广泛应用于金融、运营商、零售、互联网以及公共服务等多个行业场景之中。",
        "called": True, "pages": [6], "golden": "full",
    },
    "case_007": {
        "answer": "现有资料中没有天气数据，无法从知识库回答，当前也没有可用的联网检索工具。",
        "called": False, "pages": [], "golden": "none",
    },
    "case_008": {
        "answer": "【事实】小ai辅食助手的 VIP 家庭包定价 29.9 元，包含 3 个月权益；建议暂不做低价促销，下一步最小行动是核对支付流水。",
        "called": False, "pages": [], "golden": "none",
    },
    "case_009": {
        "answer": "现有资料中未提及 2027 年的定价，无法从知识库给出该数字。",
        "called": False, "pages": [], "golden": "none",
    },
    "case_010": {
        "answer": "【事实】支持带标量过滤的混合检索，标量过滤可以辅助快速而且精确地缩小查询空间。建议暂不做独立向量库的迁移。",
        "called": True, "pages": [8], "golden": "full",
    },
}


def _golden_preview(case_id: str, mode: str) -> str:
    if mode == "none":
        return ""
    path = EVAL / "workspace" / case_id / "golden.txt"
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8")
    return text.split("---")[0] if mode == "first" else text


def run(case: dict) -> dict:
    spec = MOCK.get(case["case_id"])
    if spec is None:
        raise KeyError(f"mock 未覆盖 {case['case_id']}")
    answer = spec["answer"]
    pages = spec["pages"]
    preview = _golden_preview(case["case_id"], spec["golden"])

    calls = []
    if spec["called"]:
        calls.append({
            "name": "rag_summarize",
            "args": {"query": case["user_question"]},
            "duration_ms": 12.3,
            "retrieved": [
                {
                    "chunk_id": "mock-298",
                    "source": "oceanbase向量能力与ai应用案例集（2025).pdf",
                    "page": page,
                    "preview": preview,
                }
                for page in pages
            ],
            "prompt": "(mock)",
            "result_chars": len(answer),
            "error": "",
        })
    return {
        "answer": answer,
        "trace": {
            "question": case["user_question"],
            "answer": answer,
            "tool_calls": calls,
            "error": "",
            "duration_ms": 12.3,
        },
    }
