from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from praisonaiagents.llm.llm import LLM

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from settings import (
    praison_chat_model, praison_api_key, praison_base_url,
    embedding_model, embedding_api_key, embedding_base_url,
    chat_model as _chat_model_name, chat_api_key, chat_base_url,
)
# from langchain_core.embeddings import Embeddings
# import litellm


# class LiteLLMEmbeddings(Embeddings):
#     """用 litellm.embedding 作底层的 LangChain Embeddings 适配器，
#     可直接作为 Chroma 的 embedding_function（需提供 embed_documents / embed_query）。"""
#     def __init__(self, model: str, api_key: str | None = None, base_url: str | None = None):
#         # litellm 需要 provider 前缀；OpenAI 兼容服务统一加 openai/ 前缀
#         self.model = model if "/" in model else f"openai/{model}"
#         self.api_key = api_key
#         self.base_url = base_url

#     def _embed(self, texts):
#         resp = litellm.embedding(
#             model=self.model,
#             input=texts,
#             api_base=self.base_url,
#             api_key=self.api_key,
#         )
#         # litellm 返回的 data 是列表，每项为 {embedding: [...]}（也支持属性访问）
#         return [item["embedding"] for item in resp["data"]]

#     def embed_documents(self, texts):
#         return self._embed(texts)

#     def embed_query(self, text):
#         return self._embed([text])[0]

import sqlite3
import json
import os


# 持久化向量缓存：OpenAIEmbeddings 是 pydantic 模型，类属性名以 _ 开头会被当成
# 私有属性而失效，故缓存连接放在模块级全局变量，避免被 pydantic 拦截。
_EMB_CACHE_DB = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "embed_cache.db"
)
_EMB_CACHE_CONN = None


def _emb_db():
    global _EMB_CACHE_CONN
    if _EMB_CACHE_CONN is None:
        os.makedirs(os.path.dirname(_EMB_CACHE_DB), exist_ok=True)
        _EMB_CACHE_CONN = sqlite3.connect(_EMB_CACHE_DB)
        _EMB_CACHE_CONN.execute(
            "CREATE TABLE IF NOT EXISTS emb (txt TEXT PRIMARY KEY, vec TEXT)"
        )
        _EMB_CACHE_CONN.commit()
    return _EMB_CACHE_CONN


class NormalizedOpenAIEmbeddings(OpenAIEmbeddings):
    """bge-m3 经 SiliconFlow 的 OpenAI 兼容接口返回的是未归一化向量；Chroma 默认
    用 L2 距离，未归一化时向量模长差异会主导排序，语义相关块反而排不进 top-k。
    这里在 embed_documents/embed_query 后做 L2 归一化，使 L2 排序等价于余弦排序。

    另：实测 SiliconFlow 的 embedding 接口对同一文本会返回存在微小抖动的向量
    （约 1/5 概率偏移，单维抖动 ~1e-3），这点抖动足以在某些轮次改变最近邻、令向量
    召回集合翻转，导致同一 query 的 RAG 召回不可复现。为此加一层【持久化向量缓存】：
    同一文本首次调用落盘，之后所有进程（CLI / Web / 评测）都读同一份向量——
    既抹平接口噪声、又跨进程确定，还能省去重复的 embedding API 调用。
    """
    @staticmethod
    def _norm(vec):
        import math
        n = math.sqrt(sum(x * x for x in vec))
        return [round(x / n, 5) for x in vec] if n else list(vec)

    def _embed_cached(self, text, raw_fn):
        try:
            row = _emb_db().execute("SELECT vec FROM emb WHERE txt=?", (text,)).fetchone()
            if row:
                return json.loads(row[0])
        except Exception:
            pass
        vec = self._norm(raw_fn(text))
        try:
            _emb_db().execute(
                "INSERT OR IGNORE INTO emb VALUES (?, ?)", (text, json.dumps(vec))
            )
            _emb_db().commit()
        except Exception:
            pass
        return vec

    def embed_documents(self, texts):
        return [
            self._embed_cached(t, lambda x: super().embed_documents([x])[0])
            for t in texts
        ]

    def embed_query(self, text):
        return self._embed_cached(text, lambda x: super().embed_query(x))

    def embed_documents(self, texts):
        return [self._norm(v) for v in super().embed_documents(texts)]

    def embed_query(self, text):
        return self._norm(super().embed_query(text))


embed_model = NormalizedOpenAIEmbeddings(
    model=embedding_model,
    api_key=embedding_api_key,
    base_url=embedding_base_url
)

chat_model = ChatOpenAI(
    model=_chat_model_name,
    api_key=chat_api_key,
    base_url=chat_base_url
)

# HyDE 专用模型：temperature 必须为 0。
# 默认 ChatOpenAI 的 temperature 为 0.7，会让 _hyde() 每次生成不同的假设文档，
# 进而产生不同的检索嵌入，导致同一查询的召回结果漂移（实测同一 query 连跑两次，
# top 召回块完全不同）。此模型仅供 HyDE 使用，不影响主总结链路的 temperature 设定。
hyde_chat_model = ChatOpenAI(
    model=_chat_model_name,
    api_key=chat_api_key,
    base_url=chat_base_url,
    temperature=0.0,
)

praison_llm = LLM(
     model=praison_chat_model
    ,api_key=praison_api_key
    ,base_url=praison_base_url
    ,timeout=300   # 长文档生成（如 PRD 定稿）单次请求可能偏慢，放大请求超时避免误杀
    )

if __name__ == '__main__':
    # praison_llm.get_response("认识oceanbase")

    print(len(embed_model.embed_query("认识oceanbase")))