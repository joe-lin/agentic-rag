import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))


"""
总结服务类：用户提问，搜索参考资料，将提问和参考资料提交给模型，让模型总结回复
"""
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
import requests


from rag.vector_store import VectorStoreService
from utils.prompt_loader import load_rag_prompts
from utils.logger_handler import logger
from model.factory import chat_model, hyde_chat_model
from settings import embedding_api_key, embedding_base_url, rerank_model, candidate_k, k, hyde_enabled


def print_prompt(prompt):
    # print("="*20)
    # print(prompt.to_string())
    # print("="*20)
    pass
    return prompt


class RagSummarizeService(object):
    def __init__(self):
        self.vector_store = VectorStoreService()
        self.retriever = self.vector_store.get_retriever()
        self.prompt_text = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = chat_model
        self.hyde_model = hyde_chat_model
        self.chain = self._init_chain()
        # 评测用：最近一次调用的检索证据与最终 prompt。
        # 这里刻意不 import utils.trace，避免 rag 层依赖评测设施；由工具层取走后写入 trace。
        # 单例 + 实例属性 => 仅适用于串行调用，评测 runner 已保证串行。
        self.last_retrieved: list[dict] = []
        self.last_prompt: str = ""

    def _init_chain(self):
        chain = self.prompt_template | print_prompt | self.model | StrOutputParser()
        return chain

    def retriever_docs(self, query: str) -> list[Document]:
        # 0) HyDE：先用 LLM 生成一段“假设性答案”，用它做检索嵌入，
        #    弥补 bi-encoder 对“抽象问句 vs 密集事实清单”匹配弱的短板。
        #    注意：部分 chat 端点不保证 temperature=0 的确定性（实测同一 query
        #    多次生成的假设文档长度/内容均不同），因此绝不能只靠 HyDE 召回。
        search_text = self._hyde(query) or query
        # 1) 向量召回候选（多取一些，给 reranker 留选择空间）
        candidates = self.vector_store.get_retriever(candidate_k).invoke(search_text)
        # 1.2) 确定性保底：用【原始 query】再向量召回一次（嵌入与检索均为确定操作），
        #      与 HyDE 召回合并。这样即使 HyDE 输出漂移，稳定实体查询也绝不会漏召回，
        #      整体召回结果对“同一 query”变得可复现。
        candidates = self._merge_docs(
            candidates, self.vector_store.get_retriever(candidate_k).invoke(query)
        )
        # 1.5) 词法召回：bi-encoder 对密集事实清单常漏召回，用关键词重叠兜底命中。
        #      与向量候选合并（按内容去重）后统一交给 reranker 定序。
        lex = self.vector_store.lexical_search(query, top_n=10)
        candidates = self._merge_docs(candidates, lex)
        # 1.8) 关键：向量召回在并列分数项上的返回顺序不稳定（集合确定、次序随机），
        #      而 rerank 在并列分时按“输入顺序”做 tie-break，顺序一抖就放大成 top-k 轮转。
        #      在重排前按【块内容】这个对分块唯一且确定的键排序，锁死输入顺序，
        #      保证 rerank 输出可复现。（内容完全相同的极少数重复块，顺序无所谓。）
        candidates.sort(key=lambda d: d.page_content)
        # 2) 交叉编码器重排（用原始 query 判相关性），取前 k 条；失败回退向量顺序。
        return self._rerank(query, candidates, k)

    @staticmethod
    def _merge_docs(a: list[Document], b: list[Document]) -> list[Document]:
        """按对象身份去重合并两个候选列表（保留先到者顺序）。"""
        seen = {id(d) for d in a}
        merged = list(a)
        for d in b:
            if id(d) not in seen:
                seen.add(id(d))
                merged.append(d)
        return merged

    def _hyde(self, query: str) -> str | None:
        """生成假设性答案（Hypothetical Document Embeddings）。

        让 LLM 仅凭自身知识写一段像知识库文档的回答要点（含关键术语），
        用其嵌入去检索，能显著拉近“抽象问题”与“密集事实清单答案块”的距离。
        任何异常都回退到原始 query，不影响主链路。
        """
        if not hyde_enabled:
            return None
        prompt = (
            "你是 OceanBase 向量数据库技术专家。请仅凭你的知识，针对下面的问题写一段"
            "像知识库文档那样的简明回答要点，列出关键术语（如算法名、指标名），不要解释、"
            "不要说“不知道”。问题：\n" + query
        )
        try:
            resp = self.hyde_model.invoke([HumanMessage(content=prompt)])
            text = getattr(resp, "content", None) or str(resp)
            return text.strip() or None
        except Exception as exc:
            logger.warning(f"[hyde] 生成失败，回退原查询: {exc}")
            return None

    @staticmethod
    def _rerank(query: str, docs: list[Document], top_n: int) -> list[Document]:
        """用 SiliconFlow 的 bge-reranker 交叉编码器对候选文档重排。

        基础 bi-encoder 对“问句 vs 密集事实清单”类查询排序偏弱（曾出现无关页
        余弦 0.80 压过答案块 0.40 的情况），交叉编码器做 query-doc 全注意力匹配
        可显著纠正。返回按相关性降序的前 top_n 条；任何异常都回退到原始顺序。
        """
        if not docs:
            return docs
        url = f"{embedding_base_url.rstrip('/')}/rerank"
        try:
            resp = requests.post(
                url,
                headers={"Authorization": f"Bearer {embedding_api_key}"},
                json={
                    "model": rerank_model,
                    "query": query,
                    "documents": [d.page_content for d in docs],
                    "top_n": top_n,
                    "return_documents": False,
                },
                timeout=30,
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])
            if not results:
                return docs[:top_n]
            # results 已按 relevance_score 降序
            return [docs[r["index"]] for r in results if 0 <= r["index"] < len(docs)][:top_n]
        except Exception as exc:
            logger.warning(f"[rerank] 失败，回退向量排序: {exc}")
            return docs[:top_n]

    def rag_summarize(self, query: str) -> str:

        context_docs = self.retriever_docs(query)

        context = ""
        counter = 0
        for doc in context_docs:
            counter += 1
            context += f"【参考资料{counter}】: 参考资料：{doc.page_content} | 参考元数据：{doc.metadata}\n"

        self.last_retrieved = [
            {
                "chunk_id": str(getattr(doc, "id", "") or ""),
                "source": os.path.basename(str(doc.metadata.get("source", ""))),
                "page": doc.metadata.get("page", ""),
                "preview": doc.page_content,
            }
            for doc in context_docs
        ]
        self.last_prompt = self.prompt_template.format(input=query, context=context)

        return self.chain.invoke(
            {
                "input": query,
                "context": context,
            }
        )


if __name__ == '__main__':
    rag = RagSummarizeService()

    print(rag.rag_summarize("我是OceanBase"))
