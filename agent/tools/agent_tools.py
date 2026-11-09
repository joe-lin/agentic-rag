
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from time import perf_counter

from rag.rag_service import RagSummarizeService
from utils.trace import RetrievedChunk, record_tool


rag = RagSummarizeService()


def rag_summarize(query: str) -> str:
    """
    从向量存储（知识库）中检索并总结与用户问题相关的知识。
    当用户询问产品功能、OceanBase/数据库特性、已有文档/资料中的内容，
    或任何需要引用知识库才能准确回答的问题时调用本工具；
    纯闲聊或无需查资料的常识问题不要调用。
    """
    started = perf_counter()
    try:
        result = rag.rag_summarize(query)
    except Exception as exc:
        record_tool(
            "rag_summarize",
            args={"query": query},
            duration_ms=(perf_counter() - started) * 1000,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise

    retrieved = [RetrievedChunk(**item) for item in rag.last_retrieved]
    record_tool(
        "rag_summarize",
        args={"query": query},
        duration_ms=(perf_counter() - started) * 1000,
        retrieved=retrieved,
        prompt=rag.last_prompt,
        result_chars=len(result),
    )
    return result
