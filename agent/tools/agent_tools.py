
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from rag.rag_service import RagSummarizeService


rag = RagSummarizeService()


def rag_summarize(query: str) -> str:
    """
    从向量存储（知识库）中检索并总结与用户问题相关的知识。
    当用户询问产品功能、OceanBase/数据库特性、已有文档/资料中的内容，
    或任何需要引用知识库才能准确回答的问题时调用本工具；
    纯闲聊或无需查资料的常识问题不要调用。
    """
    return rag.rag_summarize(query)

